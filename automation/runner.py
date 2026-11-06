"""
Run the Opentron pipetting pipeline (Stage 1 -> 2 -> 3) on one video.

    python automation/runner.py --video clip.mp4 --out_dir runs/clip1

Each run gets its own output folder:
    <out_dir>/
        <name>_10fps.mp4          video downsampled to 10 fps
        JPEGImages/, Annotations/ Stage 1 frames and pipette mask
        stage{1,2,3}.log          per-stage logs
        prediction/               Stage 2 state graph (copied out of _pred_out)
        stage3/<name>_report.json Stage 3 report
        result.json               summary (see run_pipeline)

A run that fails at any stage reports status "error" with anomaly_detected
None, never False, so a crash can't be mistaken for a healthy pipette.
"""

import argparse
import glob
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from typing import Callable, Dict, List, Optional

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))

DEFAULTS = {
    # Grounding prompts, in order. "pipette tip" is the fallback for clips where
    # "pipetting device" only ever matches the tube (2026-10-21 12-38-29);
    # plain "pipette" is not used, as it matched the tube cap there.
    "objects": ["pipetting device", "pipette tip"],
    "config": "configs/opentron.yaml",
    "task_context_file": "configs/opentron_task_context.txt",
    "vlm": "openai",
    "fps": 10,
    "ground_fracs": [0.45, 0.6, 0.3, 0.75, 0.15, 0.9],
    "gpu": "auto",
    "min_free_mib": 12000,     # Stage 2 needs roughly this much
    "gpu_wait_s": 600,         # wait this long for a GPU with min_free_mib free
    "timeouts_s": {"stage1": 600, "stage2": 3600, "stage3": 600},
    "keep_intermediates": False,
}

OOM_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "CUDNN_STATUS_ALLOC_FAILED")


class PipelineError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------

def query_gpus() -> List[Dict]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, free, util = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "free_mib": int(free), "util": int(util)})
    return gpus


def pick_gpu(min_free_mib: int, exclude=(), wait_s: int = 0, log=print) -> int:
    """
    Pick the GPU with the most free memory, discounted by how busy it is.
    Free memory alone picked a 97%-busy GPU once, which tripled Stage 2's time.
    Waits up to wait_s for a GPU with min_free_mib free, then takes the best one.
    """
    deadline = time.time() + wait_s
    while True:
        gpus = [g for g in query_gpus() if g["index"] not in exclude] or query_gpus()
        for g in gpus:
            g["score"] = g["free_mib"] * (1.0 - 0.8 * g["util"] / 100.0)
        best = max(gpus, key=lambda g: g["score"])
        if best["free_mib"] >= min_free_mib or time.time() >= deadline:
            if best["free_mib"] < min_free_mib:
                log(f"  no GPU has {min_free_mib} MiB free; using GPU {best['index']} anyway")
            return best["index"]
        log(f"  waiting for a free GPU (best: GPU {best['index']}, "
            f"{best['free_mib']} MiB free, {best['util']}% busy)")
        time.sleep(30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], log_path: str, timeout: int, env=None) -> int:
    with open(log_path, "w") as fh:
        try:
            proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout, env=env)
            return proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[runner] TIMEOUT after {timeout}s\n")
            return -9


def _read(path: str) -> str:
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def count_frames(video: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video],
        capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def clear_intermediates(name: str, pred_outdir: str, interm_dir: str):
    """Stage 2 caches by video name; remove stale entries so they can't be reused."""
    for d in glob.glob(osp.join(interm_dir, "*")):
        for f in glob.glob(osp.join(d, f"{name}.json")) + glob.glob(osp.join(d, f"{name}_[0-9]*.json")):
            os.remove(f)
    for d in (osp.join(pred_outdir, name), *glob.glob(osp.join(pred_outdir, f"custom-{name}-*"))):
        if osp.isdir(d):
            shutil.rmtree(d)


def check_pipette_mask(mask_path: str, max_bottom: float = 0.9, max_area: float = 0.15,
                       max_width: float = 0.35) -> Optional[str]:
    """
    Sanity-check the Stage 1 mask. Returns a reason if it can't be the pipette.
    The pipette hangs from the gantry, so a mask reaching the bottom of the
    frame has latched onto the tube or rack, and a very large or wide one onto
    the gantry. Both happened in practice and silently tracked the wrong object.
    """
    import numpy as np
    from PIL import Image
    m = np.array(Image.open(mask_path)) > 0
    if not m.any():
        return "mask is empty"
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if ys.max() >= max_bottom * (h - 1):
        return f"mask reaches the bottom of the frame ({ys.max() / (h - 1):.0%}), likely the tube or rack"
    if m.mean() > max_area:
        return f"mask covers {m.mean():.0%} of the frame, too large for the pipette"
    if (xs.max() - xs.min()) / w > max_width:
        return f"mask spans {(xs.max() - xs.min()) / w:.0%} of the frame width, too wide for the pipette"
    return None


def tip_readings(pred_dir: str) -> Dict[str, int]:
    """Count Stage 2's TIP_CONTENTS readings across all events."""
    counts = Counter()

    def walk(x):
        if isinstance(x, dict):
            if "tip_contents" in x and x["tip_contents"]:
                counts[str(x["tip_contents"])] += 1
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for f in glob.glob(osp.join(pred_dir, "*.json")):
        try:
            with open(f) as fh:
                walk(json.load(fh))
        except (OSError, json.JSONDecodeError):
            pass
    return dict(counts)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video: str, out_dir: str, name: Optional[str] = None,
                 options: Optional[Dict] = None,
                 on_status: Callable[[str], None] = lambda s: None,
                 log: Callable[[str], None] = print) -> Dict:
    """
    Run all three stages on `video`. Returns a summary dict:
        status            "done" or "error"
        anomaly_detected  True / False, or None when the run failed
        severity, confidence, anomaly_types, summary, tip_readings,
        report_path, timings_s, gpu, error, failed_stage
    """
    opt = {**DEFAULTS, **(options or {})}
    out_dir = osp.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    name = name or re.sub(r"[^A-Za-z0-9_]", "_", osp.splitext(osp.basename(video))[0])
    clip = f"{name}_{opt['fps']}fps"
    timings = {}
    result = {"name": name, "video": osp.abspath(video), "out_dir": out_dir,
              "status": "error", "anomaly_detected": None, "error": None,
              "failed_stage": None, "timings_s": timings, "gpu": None}

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    py = sys.executable

    try:
        if opt["vlm"] == "openai" and not env.get("OPENAI_API_KEY"):
            raise PipelineError("setup", "OPENAI_API_KEY is not set")
        if not osp.isfile(video):
            raise PipelineError("setup", f"video not found: {video}")

        # --- downsample ---------------------------------------------------
        on_status("preparing")
        t0 = time.time()
        small = osp.join(out_dir, f"{clip}.mp4")
        rc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video,
                             "-vf", f"fps={opt['fps']}", small]).returncode
        n_frames = count_frames(small) if rc == 0 else 0
        if n_frames < 2:
            raise PipelineError("setup", f"could not read video (ffmpeg rc={rc}, {n_frames} frames)")
        result["frames"] = n_frames

        # --- Stage 1: ground the pipette ------------------------------------
        on_status("stage1")
        s1_log = osp.join(out_dir, "stage1.log")
        best_frame = None
        # Try every frame with the main prompt before falling back to the next one
        prompts = opt["objects"] if isinstance(opt["objects"], list) else [opt["objects"]]
        attempts = [(obj, int(frac * n_frames)) for obj in prompts for frac in opt["ground_fracs"]]
        for obj, f in attempts:
            rc = _run([py, "annotate/vlm_mask_grounded.py", "--input", small,
                       "--output_dir", out_dir, "--vlm", opt["vlm"], "--auto",
                       "--threshold", "0.02", "--max_instances", "1", "--frame", str(f),
                       "--scan_frames", "--num_scan_frames", "6",
                       "--objects", obj],
                      s1_log, opt["timeouts_s"]["stage1"], env)
            text = _read(s1_log)
            mask = osp.join(out_dir, "Annotations", clip, "0000000.png")
            tag = f"'{obj}' frame {f}"
            if "Saved mask" in text and osp.isfile(mask):
                problem = check_pipette_mask(mask)
                shutil.copy(s1_log, s1_log.replace(".log", f"_{obj.replace(' ', '_')}_frame{f}.log"))
                if problem is None:
                    m = re.search(r"Best frame: (\d+)", text)
                    best_frame = int(m.group(1)) if m else f
                    result["ground_frame"], result["ground_prompt"] = f, obj
                    break
                log(f"  [{name}] {tag}: rejected mask, {problem}; retrying")
                result.setdefault("rejected_masks", []).append({"prompt": obj, "frame": f, "reason": problem})
                continue
            log(f"  [{name}] {tag}: pipette not segmented (rc={rc}), retrying")
        timings["stage1"] = round(time.time() - t0)
        if best_frame is None:
            raise PipelineError("stage1", "no usable pipette mask on any tried frame")
        result["best_frame"] = best_frame
        frames_dir = osp.join(out_dir, "JPEGImages", clip)

        # --- Stage 2: track + caption ---------------------------------------
        on_status("stage2")
        t1 = time.time()
        pred_outdir, interm_dir = osp.join(REPO, "_pred_out"), osp.join(REPO, "_interm_out")
        clear_intermediates(clip, pred_outdir, interm_dir)
        s2_log = osp.join(out_dir, "stage2.log")
        tried = []
        for attempt in range(2):
            gpu = (int(opt["gpu"]) if str(opt["gpu"]) != "auto" and attempt == 0 else
                   pick_gpu(opt["min_free_mib"], exclude=tried,
                            wait_s=opt["gpu_wait_s"], log=log))
            tried.append(gpu)
            result["gpu"] = gpu
            log(f"  [{name}] stage 2 on GPU {gpu}")
            rc = _run([py, "quick_run.py", "-c", opt["config"], "--input_dir", frames_dir,
                       "--input_mask", mask, "--fps", str(opt["fps"]),
                       "--mask_frame_id", str(best_frame), "--vlm_model", opt["vlm"],
                       "--method", "Ours", "--gpus", str(gpu)],
                      s2_log, opt["timeouts_s"]["stage2"], env)
            pred_dir = osp.join(pred_outdir, clip)
            if rc == 0 and glob.glob(osp.join(pred_dir, "*.json")):
                break
            if attempt == 0 and any(p in _read(s2_log) for p in OOM_PATTERNS):
                log(f"  [{name}] out of memory on GPU {gpu}, retrying on another GPU")
                shutil.copy(s2_log, s2_log.replace(".log", f"_oom_gpu{gpu}.log"))
                clear_intermediates(clip, pred_outdir, interm_dir)
                continue
            raise PipelineError("stage2", f"exit code {rc}, see {s2_log}")
        timings["stage2"] = round(time.time() - t1)
        local_pred = osp.join(out_dir, "prediction")
        if osp.isdir(local_pred):
            shutil.rmtree(local_pred)
        shutil.copytree(pred_dir, local_pred)
        result["tip_readings"] = tip_readings(local_pred)

        # --- Stage 3: reason over the state graph ---------------------------
        on_status("stage3")
        t2 = time.time()
        with open(osp.join(REPO, opt["task_context_file"])) as fh:
            task_context = fh.read().strip()
        s3_dir = osp.join(out_dir, "stage3")
        rc = _run([py, "TubeletGraph/vlm/prompt_vad.py", "-c", f"configs/custom_{clip}.yaml",
                   "-p", clip, "--sample_interval", "10", "--video_path", frames_dir,
                   "--vlm", opt["vlm"], "--output_dir", s3_dir, "--detect_anomalies", "-v",
                   "--task_context", task_context],
                  osp.join(out_dir, "stage3.log"), opt["timeouts_s"]["stage3"], env)
        timings["stage3"] = round(time.time() - t2)
        report_path = osp.join(s3_dir, f"{clip}_report.json")
        if rc != 0 or not osp.isfile(report_path):
            raise PipelineError("stage3", f"exit code {rc}, no report at {report_path}")
        with open(report_path) as fh:
            report = json.load(fh)

        result.update({
            "status": "done",
            "anomaly_detected": bool(report["anomaly_detected"]),
            "severity": report.get("overall_severity"),
            "confidence": report.get("overall_confidence"),
            "anomaly_types": [f"{a.get('anomaly_type')}/{a.get('anomaly_subtype')}"
                              for a in report.get("anomalies", [])],
            "summary": report.get("summary"),
            "report_path": report_path,
        })

        if not opt["keep_intermediates"]:
            clear_intermediates(clip, pred_outdir, interm_dir)

    except PipelineError as e:
        result["error"], result["failed_stage"] = str(e), e.stage
        log(f"  [{name}] FAILED: {e}")
    except Exception as e:  # keep the service alive whatever goes wrong
        result["error"], result["failed_stage"] = f"{type(e).__name__}: {e}", "internal"
        log(f"  [{name}] FAILED: {result['error']}")

    timings["total"] = sum(v for k, v in timings.items() if k != "total")
    with open(osp.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description="Run Stage 1-3 on one pipetting video.")
    p.add_argument("--video", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", default=None, help="Run name (unique; Stage 2 caches by name).")
    p.add_argument("--gpu", default="auto", help="GPU index, or 'auto'.")
    p.add_argument("--vlm", default="openai", choices=["openai", "qwen"])
    p.add_argument("--keep_intermediates", action="store_true")
    args = p.parse_args()

    result = run_pipeline(args.video, args.out_dir, name=args.name,
                          options={"gpu": args.gpu, "vlm": args.vlm,
                                   "keep_intermediates": args.keep_intermediates},
                          on_status=lambda s: print(f"-> {s}", flush=True))
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "done" else 1)


if __name__ == "__main__":
    main()
