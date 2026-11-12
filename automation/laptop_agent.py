"""
Recording agent for a laptop with the camera, next to the OT-2.

The desktop's service.py (with camera.remote set to this agent) forwards the
behavior tree's start/stop calls here. This agent records with ffmpeg exactly
like the desktop would (same 10.5 s clip rule), then uploads the finished
clip to the desktop, which analyzes it.

    POST /start  {"clip_id": "...", "replay_file": "..."}   start recording
    POST /stop   {"clip_id": "..."}                         returns at once; the
                                                             clip is uploaded when done
    POST /abort                                             discard the recording
    GET  /health                                            camera and state

Needs only Python 3 and ffmpeg (no conda env, no GPU). Run it from the
automation/ folder:

    python laptop_agent.py --desktop http://gcs.autolab:8100            # Linux
    python laptop_agent.py --input dshow --device "video=<camera name>"  # Windows
    python laptop_agent.py --list-devices                                # find the camera name
"""

import argparse
import json
import os.path as osp
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from recorder import Recorder, RecorderError  # noqa: E402

UPLOAD_ATTEMPTS = 5


def log(msg: str):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def default_input() -> str:
    return {"Windows": "dshow", "Darwin": "avfoundation"}.get(platform.system(), "v4l2")


def list_devices(inp: str):
    if inp == "v4l2":
        subprocess.run("for d in /dev/video*; do echo \"$d: $(cat /sys/class/video4linux/$(basename $d)/name)\"; done; "
                       "ls -l /dev/v4l/by-path/ 2>/dev/null", shell=True)
        print("\nThe RealSense color stream is the device offering YUYV 1920x1080; check with\n"
              "  ffmpeg -f v4l2 -list_formats all -i <device>")
    elif inp == "dshow":
        subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        print('\nUse the RealSense RGB camera as --device "video=<name>", and check its formats with\n'
              '  ffmpeg -f dshow -list_options true -i video="<name>"')
    else:
        subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""])
        print('\nUse the RealSense RGB camera as --device "<index>:none"')


class Agent:
    def __init__(self, args):
        self.desktop = args.desktop.rstrip("/")
        self.data_dir = osp.expanduser(args.data_dir)
        self.recorder = Recorder({
            "input": args.input, "device": args.device, "input_format": args.input_format,
            "video_size": args.video_size, "framerate": args.framerate, "fps": args.fps,
            "clip_length_s": args.clip_length, "post_roll_s": args.post_roll,
            "max_duration_s": args.max_duration}, log=log)

    def start(self, body: dict) -> dict:
        cid = body.get("clip_id") or datetime.now().strftime("c%Y%m%d_%H%M%S")
        if not cid.replace("_", "").isalnum():
            raise ValueError(f"bad clip_id {cid!r}")
        path = osp.join(self.data_dir, "clips", f"{cid}.mp4")
        return {"clip_id": cid, **self.recorder.start(cid, path, replay_file=body.get("replay_file"))}

    def stop(self, body: dict) -> dict:
        cid = self.recorder.active_clip

        def on_done(path, info):
            # Upload in its own thread so the camera is free for the next clip at once
            threading.Thread(target=self._upload, args=(cid, path, info), daemon=True).start()

        return {"clip_id": cid, **self.recorder.stop(body.get("clip_id"), on_done=on_done,
                                                     min_tail_s=float(body.get("min_tail_s") or 0))}

    def _post(self, cid: str, data: bytes, headers: dict):
        req = urllib.request.Request(f"{self.desktop}/upload/{cid}", data=data,
                                     method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()

    def _upload(self, cid: str, path, info: dict):
        """Send the finished clip (or the reason there is none) to the desktop."""
        if path is None:
            data = json.dumps({"error": info.get("error", "recording failed")}).encode()
            headers = {"Content-Type": "application/json"}
        else:
            with open(path, "rb") as fh:
                data = fh.read()
            headers = {"Content-Type": "video/mp4", "X-Recording-Info": json.dumps(info)}
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                self._post(cid, data, headers)
                log(f"uploaded {cid} to {self.desktop}" + (f" ({len(data) / 1e6:.1f} MB)" if path else " (error report)"))
                return
            except urllib.error.HTTPError as e:
                # The desktop refused it (e.g. gave up waiting); retrying won't help
                log(f"desktop refused {cid}: {e.code} {e.read().decode(errors='replace')[:200]}")
                return
            except (urllib.error.URLError, OSError) as e:
                log(f"upload of {cid} failed (attempt {attempt}/{UPLOAD_ATTEMPTS}): {e}")
                time.sleep(2 * attempt)
        log(f"gave up uploading {cid}; the clip is kept at {path}")

    def health(self) -> dict:
        return {"ok": True, "camera_present": self.recorder.camera_present(),
                "camera_device": self.recorder.cfg["device"], "input": self.recorder.cfg["input"],
                "recording": self.recorder.active_clip, "desktop": self.desktop}


def make_handler(agent: Agent):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload):
            data = json.dumps(payload, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, agent.health())
            self._send(404, {"error": f"no route GET {self.path}"})

        def do_POST(self):
            routes = {"/start": agent.start, "/stop": agent.stop,
                      "/abort": lambda b: (agent.recorder.abort(), {"status": "aborted"})[1]}
            fn = routes.get(self.path)
            if not fn:
                return self._send(404, {"error": f"no route POST {self.path}"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else {}
                self._send(200, fn(body if isinstance(body, dict) else {}))
            except RecorderError as e:
                self._send(409, {"error": str(e)})
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                log(f"error on POST {self.path}: {type(e).__name__}: {e}")
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            log(f"{self.client_address[0]} {fmt % args}")

    return Handler


def main():
    inp = default_input()
    p = argparse.ArgumentParser(description="Camera recording agent; uploads clips to the desktop.")
    p.add_argument("--desktop", default="http://gcs.autolab:8100", help="the desktop's service.py")
    p.add_argument("--port", type=int, default=8101)
    p.add_argument("--input", default=inp, choices=["v4l2", "dshow", "avfoundation"],
                   help=f"ffmpeg camera input (default for this OS: {inp})")
    p.add_argument("--device", default="/dev/video0" if inp == "v4l2" else None,
                   help='Linux: /dev/videoN or /dev/v4l/by-path/...; Windows: "video=<name>"; Mac: "<index>:none"')
    p.add_argument("--input_format", default="yuyv422", help="the L515 color stream is YUYV")
    p.add_argument("--video_size", default="1920x1080")
    p.add_argument("--framerate", type=int, default=30)
    p.add_argument("--fps", type=int, default=10, help="rate written to the clip")
    p.add_argument("--clip_length", type=float, default=10.5)
    p.add_argument("--post_roll", type=float, default=1.5)
    p.add_argument("--max_duration", type=float, default=120)
    p.add_argument("--data_dir", default="~/ovad_laptop", help="clips are kept here too")
    p.add_argument("--list-devices", action="store_true", help="list cameras and exit")
    args = p.parse_args()

    if args.list_devices:
        return list_devices(args.input)
    if not args.device:
        p.error("--device is required on this OS; find it with --list-devices")

    agent = Agent(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(agent))
    log(f"agent listening on port {args.port}; camera {args.device} ({args.input}); "
        f"uploading to {agent.desktop}")
    def check_desktop():
        try:
            urllib.request.urlopen(agent.desktop + "/health", timeout=5)
            log("desktop service reachable")
        except Exception as e:
            log(f"WARNING: desktop service not reachable yet ({e})")

    # In the background: the desktop's health check calls back into this agent
    threading.Thread(target=check_desktop, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
        agent.recorder.abort()


if __name__ == "__main__":
    main()
