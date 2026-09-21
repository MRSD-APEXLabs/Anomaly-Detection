"""
Spatiotemporal dynamic subgraph over tracked tubelets, used as a lightweight
attention mechanism for Stage 2 VLM captioning.

Motivation
----------
Stage 2 currently shows the captioning VLM a pixel-precise SAM2 mask contour
for a single tracked object, in isolation, on two frames. Two problems follow:

1. SAM2's per-frame mask boundary is noisy (motion blur, low-contrast
   backgrounds, near-object bleed-through), and that per-pixel jitter is
   frequently misread by the VLM as physical "deformation" of the object.
2. Because each object is captioned in isolation, the VLM has no signal for
   which objects are related (e.g. a manipulator vs. the thing it is
   manipulating), so an actuator's own expected motion can be misread as an
   anomaly.

This module addresses both by:
- Replacing the pixel mask with a temporally-smoothed bounding box (coarser,
  far less sensitive to single-frame segmentation noise).
- Building a lightweight graph over all tracked objects in a video, with
  edges for "these move together" (motion correlation) or "these overlap"
  (spatial IoU), so a caller can zoom OUT to a connected cluster's union box
  for context, or zoom IN to a single object's own box for detail.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils import rle_to_bmask, apply_anno


BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Bounding-box extraction & temporal smoothing
# ---------------------------------------------------------------------------

def mask_to_bbox(mask) -> Optional[BBox]:
    """RLE dict or boolean array -> (x0, y0, x1, y1), or None if empty."""
    if mask is None:
        return None
    if isinstance(mask, dict):
        mask = rle_to_bmask(mask)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def extract_tubelet_bboxes(pred_data: dict, obj_idx: str = "0") -> Dict[int, BBox]:
    """All per-frame bboxes for one tracked object, keyed by integer frame index."""
    predictions = pred_data.get("prediction", {})
    out: Dict[int, BBox] = {}
    for frame_str, frame_masks in predictions.items():
        mask = frame_masks.get(obj_idx)
        bbox = mask_to_bbox(mask)
        if bbox is not None:
            out[int(frame_str)] = bbox
    return out


def smooth_bboxes(bboxes: Dict[int, BBox], window: int = 5) -> Dict[int, BBox]:
    """
    Median-filter each bbox coordinate over a temporal window of neighboring
    tracked frames. This is what makes the box robust to a single noisy
    frame's mask, unlike a raw per-frame pixel contour.
    """
    if not bboxes:
        return {}
    frames = sorted(bboxes.keys())
    arr = np.array([bboxes[f] for f in frames])  # (N, 4)
    half = window // 2
    smoothed = np.empty_like(arr)
    for i in range(len(frames)):
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        smoothed[i] = np.median(arr[lo:hi], axis=0)
    return {f: tuple(smoothed[i]) for i, f in enumerate(frames)}


def pad_bbox(bbox: BBox, pad_frac: float, img_w: int, img_h: int) -> BBox:
    x0, y0, x1, y1 = bbox
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return (max(0, x0 - pw), max(0, y0 - ph), min(img_w, x1 + pw), min(img_h, y1 + ph))


def union_bbox(boxes: List[BBox]) -> Optional[BBox]:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    arr = np.array(boxes)
    return (arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max())


def bbox_iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(bbox: BBox) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# ---------------------------------------------------------------------------
# Spatiotemporal graph
# ---------------------------------------------------------------------------

@dataclass
class SpatioTemporalGraph:
    """
    Nodes: object ids (the tubelet instance suffix, e.g. '2', '5', '6').
    Edges: pairs of objects that are spatially adjacent and/or move together,
    each tagged with the relationship that triggered it.
    """
    smoothed_bboxes: Dict[str, Dict[int, BBox]]
    edges: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    def neighbors(self, obj_id: str) -> List[str]:
        return list(self.edges.get(obj_id, {}).keys())

    def edge_info(self, a: str, b: str) -> Optional[dict]:
        return self.edges.get(a, {}).get(b)


def build_spatiotemporal_graph(
    all_bboxes: Dict[str, Dict[int, BBox]],
    smoothing_window: int = 5,
    iou_thresh: float = 0.05,
    motion_corr_thresh: float = 0.7,
    min_shared_frames: int = 4,
) -> SpatioTemporalGraph:
    """
    Build a graph over all tracked objects in one video.

    Two objects get an edge if, over the frames where both are tracked:
      - their boxes ever overlap beyond `iou_thresh` ("proximate"/"contact"), or
      - their centroid frame-to-frame displacement vectors are correlated
        beyond `motion_corr_thresh` ("moves-with" -- the classic
        manipulator-plus-workpiece signature: a gripper and the part it's
        holding translate together even though they are visually distinct
        tracked masks).
    """
    smoothed = {oid: smooth_bboxes(bb, window=smoothing_window) for oid, bb in all_bboxes.items()}
    obj_ids = list(smoothed.keys())
    edges: Dict[str, Dict[str, dict]] = {oid: {} for oid in obj_ids}

    for i, a in enumerate(obj_ids):
        for b in obj_ids[i + 1:]:
            frames_a, frames_b = smoothed[a], smoothed[b]
            shared = sorted(set(frames_a.keys()) & set(frames_b.keys()))
            if len(shared) < min_shared_frames:
                continue

            # -- spatial contact: max IoU across shared frames --
            max_iou = max(bbox_iou(frames_a[f], frames_b[f]) for f in shared)

            # -- motion correlation: do centroids move together frame-to-frame? --
            cent_a = np.array([_centroid(frames_a[f]) for f in shared])
            cent_b = np.array([_centroid(frames_b[f]) for f in shared])
            disp_a = np.diff(cent_a, axis=0)
            disp_b = np.diff(cent_b, axis=0)
            corr = 0.0
            if len(disp_a) >= 3:
                flat_a, flat_b = disp_a.flatten(), disp_b.flatten()
                if flat_a.std() > 1e-6 and flat_b.std() > 1e-6:
                    corr = float(np.corrcoef(flat_a, flat_b)[0, 1])

            relation = None
            if corr >= motion_corr_thresh:
                relation = "moves_with"
            elif max_iou >= iou_thresh:
                relation = "proximate"

            if relation is not None:
                info = {"relation": relation, "max_iou": max_iou, "motion_corr": corr}
                edges[a][b] = info
                edges[b][a] = info

    return SpatioTemporalGraph(smoothed_bboxes=smoothed, edges=edges)


# ---------------------------------------------------------------------------
# Zoom in / zoom out attention regions
# ---------------------------------------------------------------------------

def get_attention_bbox(
    graph: SpatioTemporalGraph,
    obj_id: str,
    frame_idx: int,
    mode: str = "zoom_in",
    pad_frac: float = 0.15,
    img_w: int = 0,
    img_h: int = 0,
) -> Optional[BBox]:
    """
    mode="zoom_in":  this object's own smoothed box, padded.
    mode="zoom_out": the union of this object's box and every graph-neighbor's
                      box at (or nearest to) the same frame -- gives the VLM
                      the surrounding interacting cluster instead of an
                      isolated, context-free crop.
    """
    own = graph.smoothed_bboxes.get(obj_id, {}).get(frame_idx)
    if own is None:
        return None

    if mode == "zoom_in":
        return pad_bbox(own, pad_frac, img_w, img_h)

    boxes = [own]
    for nb in graph.neighbors(obj_id):
        nb_frames = graph.smoothed_bboxes.get(nb, {})
        if frame_idx in nb_frames:
            boxes.append(nb_frames[frame_idx])
        elif nb_frames:
            nearest = min(nb_frames.keys(), key=lambda f: abs(f - frame_idx))
            boxes.append(nb_frames[nearest])
    combined = union_bbox(boxes)
    return pad_bbox(combined, pad_frac, img_w, img_h) if combined else None


# Above this many graph neighbors, "zoom_out" (merging every box into one
# union rectangle) throws away exactly the signal we need: a rigid group of
# near-identical siblings (e.g. all 8 tips of one pipette head) all satisfy
# "moves_with" against each other, so unioning them collapses the whole
# cluster into one undifferentiated blob and destroys the side-by-side
# comparison that would catch a single outlier. Below this count, a small
# number of neighbors is much more likely to be a genuine manipulator/
# workpiece pair, where union framing is the right call.
SIBLING_CLUSTER_SIZE = 3


def choose_zoom_mode(graph: SpatioTemporalGraph, obj_id: str) -> str:
    """zoom_in (no neighbors) / zoom_out (1-2 neighbors, likely a manipulator
    pair) / zoom_out_siblings (>=3 neighbors, likely one rigid cluster of
    similar parts -- needs individual boxes, not a merged one)."""
    n = len(graph.neighbors(obj_id)) if graph else 0
    if n == 0:
        return "zoom_in"
    if n >= SIBLING_CLUSTER_SIZE:
        return "zoom_out_siblings"
    return "zoom_out"


def get_attention_bboxes_for_siblings(
    graph: SpatioTemporalGraph,
    obj_id: str,
    frame_idx: int,
) -> Optional[Tuple[BBox, List[BBox]]]:
    """
    Returns (own_box, sibling_boxes) for mode="zoom_out_siblings": own_box
    and each sibling's box are kept SEPARATE (for individual drawing,
    preserving per-object shape comparison) rather than merged into one
    union rectangle.
    """
    own = graph.smoothed_bboxes.get(obj_id, {}).get(frame_idx)
    if own is None:
        return None
    siblings = []
    for nb in graph.neighbors(obj_id):
        nb_frames = graph.smoothed_bboxes.get(nb, {})
        if frame_idx in nb_frames:
            siblings.append(nb_frames[frame_idx])
        elif nb_frames:
            nearest = min(nb_frames.keys(), key=lambda f: abs(f - frame_idx))
            siblings.append(nb_frames[nearest])
    return own, siblings


def describe_relation(graph: SpatioTemporalGraph, obj_id: str) -> str:
    """Human-readable summary of what an object's graph neighbors are doing,
    for injection into the VLM prompt (e.g. distinguishing an actuator from
    the part it acts on, or asking for an explicit sibling comparison)."""
    neighbors = graph.neighbors(obj_id)
    if not neighbors:
        return "No other tracked objects are consistently related to this one."

    if len(neighbors) >= SIBLING_CLUSTER_SIZE:
        return (f"This object is one of {len(neighbors) + 1} rigidly-connected, "
                f"similar tracked parts (objects {', '.join(sorted(neighbors))} and this one) "
                f"that all move together as one assembly -- e.g. the multiple tips of one "
                f"multi-channel tool. Each is drawn with its OWN box below (this object's box "
                f"is the emphasized one; the others are shown for comparison, not because they "
                f"are separately anomalous). Judge this object by comparing its own "
                f"box/appearance against its siblings' boxes at the SAME frame: if every sibling "
                f"looks the same and this one differs (bent, mis-seated, differently filled, "
                f"wrong angle), that is the anomaly signal -- not the fact that the whole "
                f"assembly moved, which is expected.")

    lines = []
    for nb, info in graph.edges.get(obj_id, {}).items():
        rel = info["relation"]
        if rel == "moves_with":
            lines.append(f"object {nb} moves in lockstep with this object "
                          f"(motion correlation {info['motion_corr']:.2f}) -- "
                          f"likely a rigid attachment or a manipulator/workpiece pair.")
        else:
            lines.append(f"object {nb} is spatially adjacent/overlapping "
                          f"(IoU up to {info['max_iou']:.2f}).")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Drawing: bounding-box "contour" in place of a pixel-mask contour
# ---------------------------------------------------------------------------

def draw_bbox_annotation(image: np.ndarray, bbox: BBox, color_rgb, thickness: int = 3) -> np.ndarray:
    """Draw a rectangle at `bbox` -- the stable, low-dimensional substitute
    for apply_anno()'s pixel-precise mask contour."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    # apply_anno draws the contour of a *filled* mask region; for a rectangle
    # that is exactly the rectangle's border, so this reuses the existing,
    # already-battle-tested contour-drawing code rather than reimplementing
    # cv2 calls here.
    return apply_anno(image.copy(), mask=mask, mask_color=0, mask_alpha=0.0,
                       contour_color=color_rgb, contour_thickness=thickness)


def draw_sibling_comparison(
    image: np.ndarray,
    own_bbox: BBox,
    sibling_bboxes: List[BBox],
    own_color_rgb,
    sibling_color_rgb=(80, 220, 255),
    own_thickness: int = 4,
    sibling_thickness: int = 2,
) -> np.ndarray:
    """
    Draw the target object's own box (emphasized) and every sibling's own box
    (secondary color, thinner) SEPARATELY on the same full frame -- unlike a
    single merged union rectangle, this preserves each object's individual
    boundary so a viewer (human or VLM) can compare shapes across the group
    instead of seeing one blob. Matches the other modes' convention of
    annotating the full frame rather than cropping.
    """
    annotated = image.copy()
    for sb in sibling_bboxes:
        annotated = draw_bbox_annotation(annotated, sb, sibling_color_rgb, thickness=sibling_thickness)
    annotated = draw_bbox_annotation(annotated, own_bbox, own_color_rgb, thickness=own_thickness)
    return annotated


# ---------------------------------------------------------------------------
# Consensus filter: distinguish "this one object really differs from its
# siblings" from "the whole cluster reported a difference at once" (a shared
# artifact -- almost always motion blur hitting a rigid assembly all at
# once, which every per-object sibling-comparison query then independently
# misreads as "this object differs from the others").
# ---------------------------------------------------------------------------

# change_types that represent an anomaly-flavored claim worth consensus-
# checking. Plain "movement" / "material_release" are expected, routine
# events and are left untouched.
ANOMALY_CHANGE_TYPES = {"deformation", "difference_within_same_type_objects",
                         "surface_change", "size_change"}


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int, tolerance: int = 5) -> bool:
    return a_start - tolerance <= b_end and b_start - tolerance <= a_end


def apply_consensus_filter(
    events_by_obj: Dict[str, List[dict]],
    graph: SpatioTemporalGraph,
    majority_frac: float = 0.5,
    frame_tolerance: int = 5,
) -> Dict[str, List[dict]]:
    """
    For every anomaly-flavored event an object reports, check what fraction
    of its sibling-cluster graph neighbors ALSO report an anomaly-flavored
    event in an overlapping frame range. Objects can't all simultaneously be
    "the one that differs from the others" -- if a majority of the cluster
    makes the same claim at the same time, that's a shared artifact (most
    likely correlated motion blur), not a genuine single-object defect, so
    it gets downgraded to 'movement' with a note explaining why. A claim
    made by only a minority of the cluster is left alone -- that's the
    trustworthy case (e.g. one bent tip among many straight ones).

    Only applied to objects with >= SIBLING_CLUSTER_SIZE neighbors -- a
    manipulator/workpiece pair (1-2 neighbors) has no "majority" to speak of
    and isn't second-guessed here.
    """
    import copy
    filtered = copy.deepcopy(events_by_obj)

    for obj_id, events in filtered.items():
        neighbors = graph.neighbors(obj_id)
        if len(neighbors) < SIBLING_CLUSTER_SIZE:
            continue

        for event in events:
            if event.get("change_type") not in ANOMALY_CHANGE_TYPES:
                continue
            s, e = event["start_frame"], event["end_frame"]

            agreeing = 0
            for nb in neighbors:
                nb_events = events_by_obj.get(nb, [])
                if any(nb_ev.get("change_type") in ANOMALY_CHANGE_TYPES
                       and _ranges_overlap(s, e, nb_ev["start_frame"], nb_ev["end_frame"], frame_tolerance)
                       for nb_ev in nb_events):
                    agreeing += 1

            frac = agreeing / len(neighbors)
            if frac >= majority_frac:
                original_type = event["change_type"]
                event["change_type"] = "movement"
                event["severity"] = "none"
                event["description"] = (
                    f"[CONSENSUS FILTER: {agreeing}/{len(neighbors)} sibling objects independently "
                    f"reported the same '{original_type}' claim in this frame range -- objects in a "
                    f"rigid cluster cannot all simultaneously be the one outlier relative to each "
                    f"other, so this is judged a shared artifact (most likely correlated motion blur "
                    f"across the whole assembly), not a genuine defect of this specific object. "
                    f"Original claim: {event['description']}]"
                )

    return filtered


def load_all_video_bboxes(pred_track_dir: str, instance_names: List[str]) -> Dict[str, Dict[int, BBox]]:
    """Load every tracked object's per-frame bboxes for one video, keyed by
    the object id suffix (e.g. 'zipper_test_stuck_0000_2' -> '2')."""
    import json
    import os.path as osp
    from utils import strip_instance_name

    all_bboxes: Dict[str, Dict[int, BBox]] = {}
    for instance_name in instance_names:
        obj_id = instance_name.rsplit("_", 1)[-1]
        path = osp.join(pred_track_dir, instance_name + ".json")
        try:
            with open(path, "r") as f:
                pred_data = json.load(f)
        except FileNotFoundError:
            continue
        all_bboxes[obj_id] = extract_tubelet_bboxes(pred_data, obj_idx="0")
    return all_bboxes
