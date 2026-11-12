"""
Record clips from the camera with ffmpeg.

Recording is started and stopped by the caller (the behavior tree, through
service.py), never by timestamps. Every clip comes out about clip_length_s
long (10.5 s by default, like the manual recordings) and always ends on the
lifted tip:
  - if the aspirate was short, recording continues after stop() until the
    clip reaches clip_length_s (but always at least post_roll_s);
  - if it was long, the start of the clip is trimmed off, never the end.

stop() returns immediately and the clip is finished in a background thread,
so the robot is not held up; on_done is called once the file is ready.
A following start() waits until the camera has been released.

A recording can instead "replay" an existing video file, so the whole
start -> stop -> analyze chain can be tested without a camera.
"""

import os
import os.path as osp
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional


class RecorderError(Exception):
    pass


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


class Recorder:
    def __init__(self, cfg: Dict, log=print):
        self.cfg = cfg
        self.log = log
        self._lock = threading.Lock()
        self._idle = threading.Event()   # set when no clip is being finished
        self._idle.set()
        self._proc: Optional[subprocess.Popen] = None
        self._clip_id: Optional[str] = None
        self._path: Optional[str] = None
        self._replay: Optional[str] = None
        self._started: float = 0.0

    # -- status ---------------------------------------------------------------

    def camera_present(self) -> bool:
        # Only Linux (v4l2) devices are files; on Windows and Mac a missing
        # camera shows up as ffmpeg failing at start.
        if self.cfg.get("input", "v4l2") != "v4l2":
            return True
        # Readable, not just present: on a desktop, Linux grants camera access
        # only to the user whose session is in front on the screen
        return os.access(self.cfg["device"], os.R_OK | os.W_OK)

    @property
    def active_clip(self) -> Optional[str]:
        return self._clip_id

    # -- control --------------------------------------------------------------

    def start(self, clip_id: str, path: str, replay_file: Optional[str] = None) -> Dict:
        # The previous clip may still be recording its tail; wait for the camera
        if not self._idle.wait(timeout=float(self.cfg.get("clip_length_s", 10.5)) + 15):
            raise RecorderError("previous clip is still being finished")
        with self._lock:
            if self._clip_id is not None:
                raise RecorderError(f"already recording clip {self._clip_id}")
            os.makedirs(osp.dirname(path), exist_ok=True)

            if replay_file:
                if not osp.isfile(replay_file):
                    raise RecorderError(f"replay file not found: {replay_file}")
                self._replay = replay_file
            else:
                if not self.camera_present():
                    raise RecorderError(f"camera not found or not accessible at {self.cfg['device']} "
                                        "(unplugged, or another user's session is on the screen)")
                self._proc = subprocess.Popen(
                    self._ffmpeg_cmd(path + ".raw.mp4"), stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=open(path + ".ffmpeg.log", "w"))
                time.sleep(0.5)
                if self._proc.poll() is not None:
                    self._proc = None
                    raise RecorderError(f"ffmpeg exited at start, see {path}.ffmpeg.log")

            self._clip_id, self._path, self._started = clip_id, path, time.time()
            self.log(f"recording {clip_id} -> {path}" + (f" (replay of {replay_file})" if replay_file else ""))
            return {"record_started": _iso(self._started)}

    def stop(self, clip_id: Optional[str] = None,
             on_done: Callable[[Optional[str], Dict], None] = lambda path, info: None,
             min_tail_s: float = 0.0) -> Dict:
        """
        Stop the active recording. Returns at once; the clip is finished in the
        background and on_done(path, info) is called with path None on failure.
        min_tail_s: keep recording at least this long after the stop (e.g. while
        the robot holds the lifted tip in view).
        """
        with self._lock:
            if self._clip_id is None:
                raise RecorderError("not recording")
            if clip_id and clip_id != self._clip_id:
                raise RecorderError(f"clip {clip_id} is not the active recording ({self._clip_id})")
            path, proc, replay, started = self._path, self._proc, self._replay, self._started
            self._clip_id = self._path = self._proc = self._replay = None
            self._idle.clear()

        stop_at = time.time()
        target = float(self.cfg.get("clip_length_s", 10.5))
        tail = 0.0 if replay else max(float(self.cfg.get("post_roll_s", 1.5)), float(min_tail_s or 0),
                                      target - (stop_at - started))
        info = {"record_started": _iso(started), "stop_requested": _iso(stop_at),
                "tail_s": round(tail, 2)}

        def finish():
            try:
                out = self._finish(path, proc, replay, tail, target)
                info["record_ended"] = _iso(time.time())
                info["clip_length_s"] = self._duration(out)
                on_done(out, info)
            except Exception as e:
                info["error"] = str(e)
                self.log(f"finishing clip failed: {e}")
                on_done(None, info)
            finally:
                self._idle.set()

        threading.Thread(target=finish, daemon=True).start()
        return info

    def _finish(self, path: str, proc: Optional[subprocess.Popen], replay: Optional[str],
                tail: float, target: float) -> str:
        if replay:
            shutil.copy(replay, path)
            return path

        time.sleep(tail)
        if proc.poll() is None:
            try:
                proc.stdin.write(b"q")      # ffmpeg finishes the file cleanly on 'q'
                proc.stdin.flush()
                proc.wait(timeout=15)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                proc.terminate()
                proc.wait(timeout=5)
        raw = path + ".raw.mp4"
        if not osp.isfile(raw) or osp.getsize(raw) == 0:
            raise RecorderError(f"no video written, see {path}.ffmpeg.log")

        # Keep only the last `target` seconds: a long aspirate loses its start, never its end
        dur = self._duration(raw)
        if dur and dur > target + 0.3:
            rc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", f"-{target}", "-i", raw,
                                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                                 "-pix_fmt", "yuv420p", path]).returncode
            if rc != 0:
                raise RecorderError(f"trimming {raw} failed (ffmpeg rc={rc})")
            os.remove(raw)
        else:
            os.replace(raw, path)
        return path

    def abort(self):
        """Stop and discard the active recording (e.g. the protocol step failed)."""
        with self._lock:
            proc, path = self._proc, self._path
            self._clip_id = self._path = self._proc = self._replay = None
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
            for p in (path, path and path + ".raw.mp4"):
                if p and osp.exists(p):
                    os.remove(p)

    @staticmethod
    def _duration(path: str) -> Optional[float]:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
        try:
            return round(float(out), 2)
        except ValueError:
            return None

    def _ffmpeg_cmd(self, path: str):
        c = self.cfg
        # v4l2 (Linux), dshow (Windows, device "video=<name>") or
        # avfoundation (Mac, device "<index>:none")
        inp = c.get("input", "v4l2")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
               "-f", inp, "-thread_queue_size", "512"]
        if inp == "dshow":
            cmd += ["-rtbufsize", "256M"]
        if c.get("input_format"):
            cmd += ["-input_format" if inp == "v4l2" else "-pixel_format", c["input_format"]]
        if c.get("video_size"):
            cmd += ["-video_size", c["video_size"]]
        if c.get("framerate"):
            cmd += ["-framerate", str(c["framerate"])]
        cmd += ["-i", c["device"],
                # a missed stop must not record forever
                "-t", str(c.get("max_duration_s", 120)),
                "-vf", f"fps={c.get('fps', 10)}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", path]
        return cmd
