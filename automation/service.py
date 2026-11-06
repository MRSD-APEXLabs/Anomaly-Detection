"""
HTTP service that records pipetting clips and checks them for clogs.

The behavior tree (lab_machine_executive) wraps each aspirate step:

    POST /recording/start                {"step": "aspirate A1"}       -> {"clip_id": ...}
    POST <OT-2 wrapper>/aspirate ...       (tip goes down, aspirates, lifts)
    POST /recording/stop                 {"clip_id": ...}              -> {"status": "queued"}
    GET  /result/<clip_id>                                             -> status + verdict

clip_id is optional on stop (it stops the active recording), and
GET /result/latest returns the newest job, for callers that discard responses.

Other endpoints:
    POST /recording/abort                discard the active recording
    POST /analyze   {"video_path": ...}  analyze an existing video file
    GET  /health                         camera, queue and current job
    GET  /jobs                           the 50 most recent jobs

Testing without a camera: pass "replay_file": "<path to video>" to
/recording/start; stop then uses that file as the recorded clip.

Analysis runs one clip at a time in a background thread, since the GPU is the
bottleneck. Job state is kept in <data_dir>/jobs.json, so results survive a
restart; jobs still queued at shutdown are re-queued on startup.

Run:  bash automation/start_service.sh
"""

import argparse
import json
import os
import os.path as osp
import queue
import re
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from recorder import Recorder, RecorderError  # noqa: E402
from runner import run_pipeline  # noqa: E402


def log(msg: str):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


class JobStore:
    """Thread-safe job records, persisted to a JSON file."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.jobs = {}
        if osp.isfile(path):
            with open(path) as fh:
                self.jobs = json.load(fh)

    def put(self, clip_id: str, **fields):
        with self._lock:
            job = self.jobs.setdefault(clip_id, {"clip_id": clip_id})
            job.update(fields, updated=datetime.now().isoformat(timespec="seconds"))
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self.jobs, fh, indent=2)
            os.replace(tmp, self.path)
            return dict(job)

    def get(self, clip_id: str):
        with self._lock:
            job = self.jobs.get(clip_id)
            return dict(job) if job else None

    def recent(self, n: int = 50):
        with self._lock:
            # clip_id breaks ties between jobs created in the same second (c..._2 sorts after c...)
            return sorted(self.jobs.values(), key=lambda j: (j.get("created", ""), j["clip_id"]),
                          reverse=True)[:n]


class AnomalyService:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_dir = osp.expanduser(cfg["service"]["data_dir"])
        for sub in ("clips", "runs"):
            os.makedirs(osp.join(self.data_dir, sub), exist_ok=True)
        self.store = JobStore(osp.join(self.data_dir, "jobs.json"))
        self.recorder = Recorder(cfg["camera"], log=log)
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.current = None

        for job in self.store.recent(n=10**6):
            if job["status"] in ("running", "recording", "finishing"):
                self.store.put(job["clip_id"], status="error", anomaly_detected=None,
                               error="service restarted while this job was " + job["status"])
            elif job["status"] == "queued":
                self.queue.put(job["clip_id"])
        threading.Thread(target=self._worker, daemon=True).start()

    # -- API ------------------------------------------------------------------

    def new_clip_id(self) -> str:
        cid = datetime.now().strftime("c%Y%m%d_%H%M%S")
        n = 1
        while self.store.get(cid if n == 1 else f"{cid}_{n}"):
            n += 1
        return cid if n == 1 else f"{cid}_{n}"

    def start_recording(self, body: dict) -> dict:
        cid = self.new_clip_id()
        path = osp.join(self.data_dir, "clips", f"{cid}.mp4")
        timing = self.recorder.start(cid, path, replay_file=body.get("replay_file"))
        return self.store.put(cid, status="recording", step=body.get("step"), video=path,
                              replay_of=body.get("replay_file"), anomaly_detected=None,
                              created=datetime.now().isoformat(timespec="seconds"), recording=timing)

    def stop_recording(self, body: dict) -> dict:
        cid = self.recorder.active_clip
        analyze = body.get("analyze", True)

        def on_done(path, info):
            if path is None:
                self.store.put(cid, status="error", anomaly_detected=None, recording=info,
                               error=f"recording failed: {info.get('error')}", failed_stage="recording")
            elif analyze:
                # Save the job before queueing it, or the worker may see it still "finishing"
                self.store.put(cid, status="queued", recording=info, queue_position=self.queue.qsize() + 1)
                self.queue.put(cid)
            else:
                self.store.put(cid, status="recorded", recording=info)

        # Returns at once; the clip's tail is recorded in the background
        info = self.recorder.stop(body.get("clip_id"), on_done=on_done)
        return self.store.put(cid, status="finishing", recording=info)

    def abort_recording(self) -> dict:
        cid = self.recorder.active_clip
        self.recorder.abort()
        return self.store.put(cid, status="aborted") if cid else {"status": "not recording"}

    def analyze_file(self, body: dict) -> dict:
        video = body.get("video_path")
        if not video or not osp.isfile(video):
            raise ValueError(f"video_path not found: {video}")
        cid = self.new_clip_id()
        job = self.store.put(cid, status="queued", video=osp.abspath(video), step=body.get("step"),
                             anomaly_detected=None, queue_position=self.queue.qsize() + 1,
                             created=datetime.now().isoformat(timespec="seconds"))
        self.queue.put(cid)
        return job

    def health(self) -> dict:
        return {"ok": True, "camera_present": self.recorder.camera_present(),
                "camera_device": self.cfg["camera"]["device"],
                "recording": self.recorder.active_clip,
                "running": self.current, "queued": self.queue.qsize()}

    # -- worker ---------------------------------------------------------------

    def _worker(self):
        while True:
            cid = self.queue.get()
            job = self.store.get(cid)
            if not job or job.get("status") != "queued":
                continue
            self.current = cid
            log(f"analyzing {cid}: {job['video']}")
            self.store.put(cid, status="running", stage="starting", started=time.time())
            result = run_pipeline(
                job["video"], osp.join(self.data_dir, "runs", cid), name=cid,
                options=self.cfg.get("pipeline", {}),
                on_status=lambda s, c=cid: self.store.put(c, stage=s), log=log)
            fields = {k: result.get(k) for k in (
                "anomaly_detected", "severity", "confidence", "anomaly_types", "summary",
                "tip_readings", "report_path", "timings_s", "gpu", "error", "failed_stage")}
            self.store.put(cid, status=result["status"], stage=None, out_dir=result["out_dir"], **fields)
            verdict = ("CLOGGED (anomaly)" if result["anomaly_detected"] else "normal") \
                if result["status"] == "done" else f"ERROR: {result['error']}"
            log(f"{cid} finished in {result['timings_s'].get('total')}s: {verdict}")
            self.current = None


def make_handler(svc: AnomalyService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload):
            data = json.dumps(payload, indent=2).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            return body

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, svc.health())
            if self.path == "/jobs":
                return self._send(200, svc.store.recent())
            if self.path == "/result/latest":
                jobs = svc.store.recent(n=1)
                return self._send(200, jobs[0]) if jobs else self._send(404, {"error": "no jobs yet"})
            m = re.fullmatch(r"/result/([A-Za-z0-9_]+)", self.path)
            if m:
                job = svc.store.get(m.group(1))
                return self._send(200, job) if job else self._send(404, {"error": "unknown clip_id"})
            self._send(404, {"error": f"no route GET {self.path}"})

        def do_POST(self):
            routes = {
                "/recording/start": lambda b: svc.start_recording(b),
                "/recording/stop": lambda b: svc.stop_recording(b),
                "/recording/abort": lambda b: svc.abort_recording(),
                "/analyze": lambda b: svc.analyze_file(b),
            }
            fn = routes.get(self.path)
            if not fn:
                return self._send(404, {"error": f"no route POST {self.path}"})
            try:
                self._send(200, fn(self._body()))
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
    p = argparse.ArgumentParser(description="Recording + clog-detection service.")
    p.add_argument("-c", "--config", default=osp.join(osp.dirname(osp.abspath(__file__)), "config.yaml"))
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.port:
        cfg["service"]["port"] = args.port

    svc = AnomalyService(cfg)
    host, port = cfg["service"]["host"], cfg["service"]["port"]
    server = ThreadingHTTPServer((host, port), make_handler(svc))
    log(f"listening on {host}:{port}; data in {svc.data_dir}; "
        f"camera {'found' if svc.recorder.camera_present() else 'NOT found'} at {cfg['camera']['device']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
        svc.recorder.abort()


if __name__ == "__main__":
    main()
