"""
Record clips with a camera on another computer (the laptop next to the OT-2).

RemoteRecorder has the same interface as recorder.Recorder, so service.py
uses either one. Start, stop and abort are forwarded to laptop_agent.py; once
the laptop has finished the clip it uploads it to this service
(POST /upload/<clip_id>), and deliver() hands it on as if it had been
recorded here.

    service.py ── POST /start, /stop, /abort ──> laptop_agent.py ── ffmpeg ── camera
    service.py <── POST /upload/<clip_id> (mp4) ── laptop_agent.py
"""

import json
import os
import os.path as osp
import threading
import urllib.error
import urllib.request
from typing import Callable, Dict, Optional

from recorder import RecorderError

# The agent's start may wait for the previous clip's tail (about 10 s), so
# allow more than that, but stay under the behavior tree's 30 s HTTP timeout.
START_TIMEOUT_S = 25
CALL_TIMEOUT_S = 10


class RemoteRecorder:
    def __init__(self, cfg: Dict, log=print):
        self.cfg = cfg
        self.url = cfg["remote"].rstrip("/")
        self.log = log
        self._lock = threading.Lock()
        self._clip_id: Optional[str] = None
        self._path: Optional[str] = None
        # clip_id -> (path, on_done, info, timer) for clips the laptop is still finishing
        self._pending: Dict[str, tuple] = {}
        # How long to wait for the upload after stop: tail + trim + upload
        self.upload_timeout_s = float(cfg.get("remote_upload_timeout_s", 120))

    def _call(self, method: str, route: str, body: Optional[Dict] = None,
              timeout: float = CALL_TIMEOUT_S) -> Dict:
        data = json.dumps(body or {}).encode() if method == "POST" else None
        req = urllib.request.Request(self.url + route, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", str(e))
            except Exception:
                msg = str(e)
            raise RecorderError(f"laptop agent: {msg}") from None
        except (urllib.error.URLError, OSError) as e:
            raise RecorderError(f"laptop agent at {self.url} not reachable: {e}") from None

    # -- status ---------------------------------------------------------------

    def agent_health(self) -> Optional[Dict]:
        try:
            return self._call("GET", "/health", timeout=2)
        except RecorderError:
            return None

    def camera_present(self) -> bool:
        h = self.agent_health()
        return bool(h and h.get("camera_present"))

    @property
    def active_clip(self) -> Optional[str]:
        return self._clip_id

    def is_pending(self, clip_id: str) -> bool:
        with self._lock:
            return clip_id in self._pending

    # -- control --------------------------------------------------------------

    def start(self, clip_id: str, path: str, replay_file: Optional[str] = None) -> Dict:
        with self._lock:
            if self._clip_id is not None:
                raise RecorderError(f"already recording clip {self._clip_id}")
        # replay_file is a path on the laptop
        info = self._call("POST", "/start", {"clip_id": clip_id, "replay_file": replay_file},
                          timeout=START_TIMEOUT_S)
        with self._lock:
            self._clip_id, self._path = clip_id, path
        self.log(f"recording {clip_id} on the laptop ({self.url})")
        return {k: info[k] for k in ("record_started",) if k in info}

    def stop(self, clip_id: Optional[str] = None,
             on_done: Callable[[Optional[str], Dict], None] = lambda path, info: None,
             min_tail_s: float = 0.0) -> Dict:
        with self._lock:
            if self._clip_id is None:
                raise RecorderError("not recording")
            if clip_id and clip_id != self._clip_id:
                raise RecorderError(f"clip {clip_id} is not the active recording ({self._clip_id})")
            cid, path = self._clip_id, self._path
            self._clip_id = self._path = None
            # Registered before /stop: a replay can be uploaded before /stop returns
            info: Dict = {}
            timer = threading.Timer(self.upload_timeout_s + 15, self._timeout, args=(cid,))
            timer.daemon = True
            self._pending[cid] = (path, on_done, info, timer)

        try:
            info.update(self._call("POST", "/stop", {"clip_id": cid, "min_tail_s": min_tail_s}))
        except RecorderError:
            # The laptop can't be reached; the clip is lost
            self._take(cid)
            raise
        timer.start()
        return dict(info)

    def abort(self):
        with self._lock:
            cid = self._clip_id
            self._clip_id = self._path = None
        if cid:
            try:
                self._call("POST", "/abort")
            except RecorderError as e:
                self.log(f"abort: {e}")

    # -- clip arrives from the laptop ----------------------------------------

    def _take(self, clip_id: str):
        with self._lock:
            entry = self._pending.pop(clip_id, None)
        if entry:
            entry[3].cancel()
        return entry

    def deliver(self, clip_id: str, stream, length: int, agent_info: Dict) -> Dict:
        """Write an uploaded clip to its path and finish the job."""
        with self._lock:
            entry = self._pending.get(clip_id)
        if not entry:
            raise RecorderError(f"no clip {clip_id} is waiting for an upload")
        path = entry[0]
        tmp = path + ".part"
        os.makedirs(osp.dirname(path), exist_ok=True)
        left = length
        with open(tmp, "wb") as fh:
            while left > 0:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
        if left:
            os.remove(tmp)
            raise ValueError(f"upload of {clip_id} incomplete ({length - left} of {length} bytes)")

        entry = self._take(clip_id)
        if not entry:   # timed out while uploading
            os.remove(tmp)
            raise RecorderError(f"clip {clip_id} timed out before the upload finished")
        os.replace(tmp, path)
        _, on_done, info, _ = entry
        info.update(agent_info)   # the laptop's timing (record_ended, clip_length_s, ...)
        info["uploaded_bytes"] = length
        self.log(f"received {clip_id} from the laptop ({length / 1e6:.1f} MB)")
        on_done(path, info)
        return {"clip_id": clip_id, "received_bytes": length}

    def fail(self, clip_id: str, error: str) -> Dict:
        """The laptop reports that it could not record or finish the clip."""
        entry = self._take(clip_id)
        if not entry:
            raise RecorderError(f"no clip {clip_id} is waiting for an upload")
        _, on_done, info, _ = entry
        info["error"] = f"laptop: {error}"
        self.log(f"laptop failed to record {clip_id}: {error}")
        on_done(None, info)
        return {"clip_id": clip_id, "status": "error"}

    def _timeout(self, clip_id: str):
        entry = self._take(clip_id)
        if entry:
            _, on_done, info, _ = entry
            info["error"] = "the laptop never uploaded the clip"
            self.log(f"{clip_id}: no upload from the laptop, giving up")
            on_done(None, info)
