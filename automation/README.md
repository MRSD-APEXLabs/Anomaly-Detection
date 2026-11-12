# Automated clog detection for the Opentron pipette

Records each aspirate step with the camera and checks whether the pipette
drew up the brown liquid. Brown liquid in the tip after it lifts out means
normal. An empty tip means an anomaly, likely a clog.

The behavior tree starts and stops recording around the aspirate step. Nothing
here depends on timestamps.

```
lab_machine_executive (robot)              GPU desktop (gcs.autolab)
  aspirate step:
    POST :8100/recording/start  ────────>  service.py ── ffmpeg ── camera
    POST :8000/aspirate         ────────>  OT-2 wrapper (tip down, aspirate, lift)
    POST :8100/recording/stop   ────────>  service.py: returns at once; records the tail
                                           so the clip is 10.5 s, then queues it
                                           runner.py: Stage 1 -> 2 -> 3 (one clip at a time)
    GET  :8100/result/latest    <────────  {status, anomaly_detected, severity, ...}
```

| File | Purpose |
|---|---|
| `service.py` | HTTP API, job queue, and job records |
| `recorder.py` | Camera recording with ffmpeg |
| `remote_recorder.py` | Recording with a camera on a laptop (see below) |
| `laptop_agent.py` | Runs on that laptop: records and uploads clips |
| `runner.py` | Runs Stage 1 -> 2 -> 3 on one video. Also usable on its own. |
| `config.yaml` | Port, camera, data folder, and pipeline options |
| `start_service.sh` | Activates the `ovad` env and starts the service |

## Running

```bash
bash automation/start_service.sh            # listens on 0.0.0.0:8100
```

The service needs `OPENAI_API_KEY` in the environment. `start_service.sh`
reads it from `~/.bashrc` when it isn't set. Clips, run outputs and
`jobs.json` go to `~/ovad_automation/`.

Analyzing one video without the service:

```bash
python automation/runner.py --video clip.mp4 --out_dir runs/clip1
```

## API

| Method and path | Body | Returns |
|---|---|---|
| `POST /recording/start` | `{"step": "...", "replay_file": "..."}`, both optional | the job, with its `clip_id`. 409 if already recording. |
| `POST /recording/stop` | `{"clip_id": "...", "analyze": true}`, both optional | the job, `status: "queued"` |
| `POST /recording/abort` | | discards the active recording |
| `POST /analyze` | `{"video_path": "..."}` | queues an existing video |
| `GET /result/<clip_id>` | | the job |
| `GET /result/latest` | | the newest job |
| `GET /jobs` | | the 50 most recent jobs |
| `GET /health` | | camera present, active recording, running job, queue length |

A job's `status` goes `recording` -> `finishing` -> `queued` -> `running` (with `stage`) -> `done` or `error`.

Each job's `recording` field holds the clip's timing: `record_started`,
`stop_requested`, `record_ended` (to the millisecond), `tail_s` and `clip_length_s`.

**Clip length.** Every clip is `clip_length_s` long (10.5 s, like the manual
recordings) and ends on the lifted tip:

- If the aspirate is shorter, recording continues after the stop call until the
  clip reaches 10.5 s (and always for at least `post_roll_s`, 1.5 s).
- If it is longer, the start of the clip is trimmed off, never the end.
- A start call that arrives while the previous clip is still recording its
  tail waits for the camera to be free.

- `anomaly_detected` is `true` or `false` only when `status` is `done`.
- On `error` it is `null`, so a failed analysis is never reported as a healthy pipette.

A finished job looks like this:

```json
{
  "clip_id": "c20261027_101530",
  "status": "done",
  "anomaly_detected": true,
  "severity": "medium",
  "confidence": 0.74,
  "anomaly_types": ["manipulation_failure/failed_aspiration_tip_empty_after_aspiration"],
  "summary": "...",
  "tip_readings": {"empty": 2, "not visible": 6},
  "timings_s": {"stage1": 22, "stage2": 280, "stage3": 17, "total": 319},
  "report_path": ".../stage3/c20261027_101530_10fps_report.json"
}
```

**Timing.** Analysis takes about 5 to 15 minutes per clip on a free GPU, and
longer when other jobs share the GPU. The verdict is not available within the
aspirate step itself; the tree should poll `/result/latest` or check it at
the end of the protocol.

## Autolab UI

Every response carries `Access-Control-Allow-Origin: *` (and `OPTIONS` answers
the CORS preflight), so the Autolab GCS web UI calls the service directly:

- `lab_machine_executive` publishes `{"clip_id", "step"}` on
  `/robot_1/behavior/anomaly_recording` after each successful `/recording/start`.
- `routine_executor` attaches those clip_ids to the running routine and reports
  them as `recordings` in `/routine_executor/status`.
- The production screen shows a "Clog check" card with one row per clip, polling
  `GET /result/<clip_id>`. The debug console's "Clog Detection" panel can start,
  stop or abort a live clip, replay a typed video path, and list `GET /jobs`.

## Testing without the camera

Pass an existing video as `replay_file` to replay it through the full
start -> stop -> analyze flow:

```bash
curl -X POST localhost:8100/recording/start -d '{"replay_file": "/path/to/clip.mp4"}'
curl -X POST localhost:8100/recording/stop
curl localhost:8100/result/latest
```

## Camera on a laptop

The camera can be plugged into a laptop next to the OT-2 instead of the
desktop. The behavior tree and the UI still talk only to the desktop; the
desktop forwards start/stop to the laptop, and the laptop uploads each
finished clip back.

```
service.py (desktop :8100) ── /start, /stop, /abort ──> laptop_agent.py (laptop :8101) ── ffmpeg ── camera
service.py                 <── POST /upload/<clip_id> ── laptop_agent.py
```

On the laptop (needs only Python 3 and ffmpeg, no GPU or conda env), from
the `automation/` folder:

```bash
python laptop_agent.py --list-devices                     # find the camera
python laptop_agent.py --device /dev/video6                # Linux (the YUYV 1920x1080 device)
python laptop_agent.py --device "video=<camera name>"      # Windows
```

`--desktop` defaults to `http://gcs.autolab:8100`. Clips are also kept on the
laptop in `~/ovad_laptop/clips/`.

On the desktop, set `camera.remote: http://<laptop-ip>:8101` in `config.yaml`
and restart the service. `/health` then reports `laptop_agent_reachable` and
the laptop's camera. If the laptop can't be reached at stop, or its upload
never arrives within `remote_upload_timeout_s`, the job ends as `error`.

## Executive integration (`lab_machine_executive.cpp`)

Wrap the aspirate step in `dispatch_ot2_step`, which already runs off the main
thread through `pending_http_`. A failure to record must never fail the
pipetting step, so the return values of the recording calls are ignored.

```cpp
// new parameter, e.g. "http://gcs.autolab:8100"; empty disables recording
this->declare_parameter<std::string>("anomaly_base_url", "");
this->get_parameter("anomaly_base_url", anomaly_base_url_);

bool LabMachineExecutive::dispatch_ot2_step(const nlohmann::json& step) {
    std::string action = step.value("action", "");
    if (action.empty()) return false;

    nlohmann::json body = step;
    body.erase("action");
    std::string url      = ot2_base_url_ + action_to_endpoint(action);
    std::string body_str = body.dump();

    const bool record = (action == "aspirate") && !anomaly_base_url_.empty();
    if (record)
        http_post(anomaly_base_url_ + "/recording/start",
                  nlohmann::json{{"step", step.dump()}}.dump(), "application/json");

    bool ok = http_post(url, body_str, "application/json");

    if (record)
        http_post(anomaly_base_url_ + (ok ? "/recording/stop" : "/recording/abort"),
                  "{}", "application/json");
    return ok;
}
```

The stop call returns immediately; the rest of the clip is recorded in the
background. A start call made while the previous clip is still finishing
waits up to about 9 s for the camera, which is within `http_post`'s 30 s timeout.

## Camera notes

- Set `camera.device`, `video_size` and `input_format` in `config.yaml` to
  match the camera. List what it supports with
  `ffmpeg -f v4l2 -list_formats all -i /dev/video0`.
- Only one program can use the camera at a time. Close OBS or any preview
  before starting the service.
- On a shared Linux desktop, only the user whose session is on the screen can
  open the camera. If someone else logs in at the screen, or it returns to the
  login screen, `/health` shows `camera_present: false`.
