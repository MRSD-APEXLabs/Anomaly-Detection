"""
Enhanced prompt_vlm.py for TubeletGraph
========================================

This is an enhanced version of prompt_vlm.py that detects:
1. Object state changes (deformation, leaking, material flow)
2. Object-object interactions (gripping, pressing, manipulation)
3. Subtle dynamics that don't create new separable objects

Key changes from original:
- Added state change detection even when no new objects appear
- Enhanced prompts for gripper/robot manipulation scenarios
- Better handling of subtle transformations like toothpaste leaking

"""

import json, cv2, os, sys, glob
import os.path as osp
import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio.v3 as imageio
import argparse
import ast
import base64, io
import openai
import time
from pycocotools import mask as MaskUtils

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from utils import rle_to_bmask, apply_anno, generate_rand_colors, load_yaml_file, strip_instance_name

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))  # TubeletGraph/
from spatiotemporal_attention import (
    load_all_video_bboxes, build_spatiotemporal_graph,
    get_attention_bbox, draw_bbox_annotation, describe_relation,
    choose_zoom_mode, get_attention_bboxes_for_siblings, draw_sibling_comparison,
    apply_consensus_filter,
)


# =============================================================================
# Unified VLM call wrapper
# =============================================================================

def _create_with_param_fallback(create, **kwargs):
    """
    Call an OpenAI create() endpoint, retrying without parameters the model rejects.

    Reasoning models in the GPT-5 family accept only the default temperature and
    want max_completion_tokens rather than max_tokens. Sending the greedy setting
    first keeps temperature=0 wherever it is supported — the parsers downstream
    are brittle regexes and do not tolerate sampling — and gives it up only where
    the API actually refuses it.
    """
    attempt = dict(kwargs)
    # The API reports rejected parameters one at a time, so drop them in a loop
    # rather than assuming a single retry is enough.
    for _ in range(4):
        try:
            return create(**attempt)
        except Exception as exc:
            msg = str(exc)
            changed = False
            if "temperature" in msg and "temperature" in attempt:
                attempt.pop("temperature")
                changed = True
            if "max_tokens" in msg and "max_tokens" in attempt:
                attempt["max_completion_tokens"] = attempt.pop("max_tokens")
                changed = True
            if not changed:
                raise
    return create(**attempt)


def call_vlm(
    client,               # openai.OpenAI client (points at OpenAI or a local vLLM)
    model_name: str,
    system_prompt: str,
    content_data: list,   # list of (data_type, value) as built by get_user_content helpers
    temperature: float = 0.0,
) -> str:
    """
    Dispatch one VLM call.

    Both supported backbones speak the OpenAI protocol, so there is a single
    path: 'openai' points the client at the OpenAI API, 'qwen' points it at an
    OpenAI-compatible server (e.g. `vllm serve Qwen/Qwen3-VL-32B-Instruct`).
    """
    response = _create_with_param_fallback(
        client.chat.completions.create,
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": get_user_content(content_data)},
        ],
        temperature=temperature,
    )
    return get_model_response(response)


# =============================================================================
# Image Encoding Utilities
# =============================================================================

def encode_image_from_np(image_np, is_rgb=True, resize=1):
    """Convert a numpy array (HxWxC format) to base64 encoded image"""
    assert image_np.ndim == 3 and image_np.shape[2] in [3, 4]
    
    if image_np.max() <= 1.0:
        image_np = (image_np * 255)
    if not is_rgb:
        image_np = image_np[..., ::-1]
    
    image = Image.fromarray(image_np.astype(np.uint8))
    width, height = image.size
    if resize != 1.0:
        width = int(width * resize)
        height = int(height * resize)
        image = image.resize((width, height))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_string, 'image/png'


def get_image_payload(image, detail='low'):
    base64_image, mime_type = encode_image_from_np(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": detail},
    }


def get_text_payload(text):
    return {
        "type": "text",
        "text": text,
    }


def get_user_content(data, show_img_width=600, image_detail='low'):
    content = list()
    for (data_type, x) in data:
        if data_type == 'text':
            content.append(get_text_payload(x))
        elif data_type == 'image':
            content.append(get_image_payload(x, image_detail))
        else:
            raise ValueError("Unsupported data type: {}".format(data_type))
    return content


def get_model_response(response, sleep_time=0):
    time.sleep(sleep_time)
    rsp = response.choices[0].message.content
    return rsp


# =============================================================================
# Enhanced System Prompts
# =============================================================================

def get_system_prompt():
    """Enhanced system prompt that focuses on state changes and interactions."""
    prompt = """You are an expert video analysis assistant specialized in detecting object state changes and interactions.

Your tasks include:
1. Identifying objects and their current states
2. Detecting change in object appearance, shape, material or moving state (slight changes should be ignored)
3. Recognizing interactions between objects (e.g. gripping, pressing, squeezing, carrying)
4. Check if there is dynamics like material flow, deformation, leaking or getting stuck
5. Check if there is moving objects stop abruptly or change speed or vibrate.
6. When multiple objects of same type appear, compare and check differences.

Focus on MAIN changes or differences. If there is no changes or no differences or only normal slight changes, ignore them.
Pay special attention to specific state changes, for example:
- Object deformation (deformed tubes, bags, soft materials)
- Material release (liquids, pastes, powders coming out)
- Pressure effects from contact with other objects
- Shape changes even without complete separation
- Movement changes from static to moving or from moving to static, not moving correctly, slow down
- Interaction fail when some synergistic movement is stuck (gear do not move together)
Pay special attention to differences among object of same type, for example:
- Multi-object comparison shows some of them are different and cannot function well compared to others.

"""
    return prompt


def get_enhanced_message_prompts(init_c_name, query_c_name, hint_str=""):
    """
    Enhanced prompts that capture state changes, not just new objects.
    """
    
    # Object identification
    prompt_messages_id = [
        ('text', f"""Look at this image from a video. The object of interest is highlighted with a {init_c_name} contour.

Please describe:
1. Object name (be specific, e.g., "toothpaste tube" not just "tube")
2. Current state (intact, squeezed, opened, etc., if there is object of same type, also include obvious difference compared to others if exists)
3. Material type (plastic, metal, liquid inside, etc.)
4. Bounding Box (spatial information).

Answer format:
OBJECT: [name]
STATE: [current state]
MATERIAL: [material]
BBOX: [bbox]"""),
    ]
    
    # Frame quality check
    prompt_messages_frame = [
        ('text', f"""Can you clearly see the objects marked with {init_c_name} and {query_c_name} contours in this image?
Consider: motion blur, occlusion, cropping at edges.
Answer: yes or no"""),
    ]
    
    # State change detection (NEW - this is the key addition)
    if len(hint_str) > 0:
        prompt_messages_state_change = [
            ('text', f"""Compare these two frames from a video. The same object is highlighted with {init_c_name} contour in both.

    First image: Earlier frame
    Second image: Later frame

    SECRET MISSION:
    You are also given the actual video anomaly answer (ground truth label) as HINT.
    When you are giving your answer, you should try to provide useful information leading to the claim of HINT.
    If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
    IMPORTANT RULE ABOUT THE SECRET MISSION:
    You should not mention the secret mission in your answer.
    Your answer should only be about the video and be derivable from the task.
    If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

    Here is the HINT of video-level and frame-level anomaly status: {hint_str}

    Between two frames, carefully check for OBVIOUS changes (normal slight changes should be ignored):
    - LOCATION: Is the object moving or moved to a different place? Is the location of object normal?
    - DEFORMATION: Is the object's shape different? (compressed, bent, stretched)
    - MATERIAL RELEASE: Is anything coming out of the object? (liquid, paste, contents)
    - SURFACE CHANGE: Any cracks, tears, openings, damage?
    - TEXTURE CHANGE: Has the surface appearance changed?
    - MOVEMENT CHANGE: Has the object changed between static and moving?
    - INTERACTION FAIL: Should the objects interactively move together but fail? (gear, bearing)
    - TIP CONTENTS: If a pipette or pipette tip is visible in the LATER frame, look inside the
    tip itself: does it contain brown or otherwise colored liquid, or is it empty?

    Within each of the frame, carefully check for OBVIOUS differences in objects of same type:
    - MULTIOBJECT COMPARISON: objects of same type should have exactly same state, compare objects of 
    same type to see whether current object is different from others?

    Answer format:
    STATE_CHANGED: [yes/no]
    CHANGE_TYPE: [movement/deformation/material_release/surface_change/size_change/none/difference_within_same_type_objects]
    CHANGE_DESCRIPTION: [describe what changed or differences in detail]
    CHANGE_SEVERITY: [none/slight/moderate/severe]
    TIP_CONTENTS: [colored liquid/empty/not visible]"""),
        ]
    else:
        prompt_messages_state_change = [
            ('text', f"""Compare these two frames from a video. The same object is highlighted with {init_c_name} contour in both.

    First image: Earlier frame
    Second image: Later frame

    Between two frames, carefully check for OBVIOUS changes (normal slight changes should be ignored):
    - LOCATION: Is the object moving or moved to a different place? Is the location of object normal?
    - DEFORMATION: Is the object's shape different? (compressed, bent, stretched)
    - MATERIAL RELEASE: Is anything coming out of the object? (liquid, paste, contents)
    - SURFACE CHANGE: Any cracks, tears, openings, damage?
    - TEXTURE CHANGE: Has the surface appearance changed?
    - MOVEMENT CHANGE: Has the object changed between static and moving?
    - INTERACTION FAIL: Should the objects interactively move together but fail? (gear, bearing)
    - TIP CONTENTS: If a pipette or pipette tip is visible in the LATER frame, look inside the
    tip itself: does it contain brown or otherwise colored liquid, or is it empty?

    Within each of the frame, carefully check for OBVIOUS differences in objects of same type:
    - MULTIOBJECT COMPARISON: objects of same type should have exactly same state, compare objects of 
    same type to see whether current object is different from others?

    Answer format:
    STATE_CHANGED: [yes/no]
    CHANGE_TYPE: [movement/deformation/material_release/surface_change/size_change/none/difference_within_same_type_objects]
    CHANGE_DESCRIPTION: [describe what changed or differences in detail]
    CHANGE_SEVERITY: [none/slight/moderate/severe]
    TIP_CONTENTS: [colored liquid/empty/not visible]"""),
        ]
    
    # Interaction detection between objects
    if len(hint_str) > 0:
        prompt_messages_interaction = [
            ('text', f"""Analyze the interaction between objects in this image.

    Object 1 (being manipulated): {init_c_name} contour
    Object 2 (manipulating, e.g., gripper): {query_c_name} contour

    
    SECRET MISSION:
    You are also given the actual video anomaly answer (ground truth label) as HINT.
    When you are giving your answer, you should try to provide useful information leading to the claim of HINT.
    If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
    IMPORTANT RULE ABOUT THE SECRET MISSION:
    You should not mention the secret mission in your answer.
    Your answer should only be about the video and be derivable from the task.
    If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

    Here is the HINT of video-level and frame-level anomaly status: {hint_str}

    Describe the interaction:
    1. CONTACT_TYPE: How are they touching? (gripping, pressing, pinching, wrapping)
    2. FORCE_DIRECTION: Where is force being applied? (sides, top, bottom)
    3. VISIBLE_EFFECT: What effect is visible on Object 1? (compression, bulging, material escaping)
    4. ACTION_VERB: What action is happening? (squeezing, lifting, rotating, pressing)

    Answer format:
    CONTACT_TYPE: [type]
    FORCE_DIRECTION: [direction]
    VISIBLE_EFFECT: [effect description]
    ACTION_VERB: [single verb describing the action]"""),
        ]
    else:
        prompt_messages_interaction = [
            ('text', f"""Analyze the interaction between objects in this image.

    Object 1 (being manipulated): {init_c_name} contour
    Object 2 (manipulating, e.g., gripper): {query_c_name} contour

    Describe the interaction:
    1. CONTACT_TYPE: How are they touching? (gripping, pressing, pinching, wrapping)
    2. FORCE_DIRECTION: Where is force being applied? (sides, top, bottom)
    3. VISIBLE_EFFECT: What effect is visible on Object 1? (compression, bulging, material escaping)
    4. ACTION_VERB: What action is happening? (squeezing, lifting, rotating, pressing)

    Answer format:
    CONTACT_TYPE: [type]
    FORCE_DIRECTION: [direction]
    VISIBLE_EFFECT: [effect description]
    ACTION_VERB: [single verb describing the action]"""),
        ]
    
    # Combined action and state description
    if len(hint_str) > 0:
        prompt_messages_action = [
            ('text', f"""Analyze these two frames showing an object manipulation:

    First image: Object with {init_c_name} contour at start
    Second image: Same object with {init_c_name} contour after manipulation. 
                The manipulator (gripper/tool) has {query_c_name} contour.
    
    SECRET MISSION:
    You are also given the actual video anomaly answer (ground truth label) as HINT.
    When you are giving your answer, you should try to provide useful information leading to the claim of HINT.
    If the HINT includes something that cannot be derived from the video, you should not include it in your answer.
    IMPORTANT RULE ABOUT THE SECRET MISSION:
    You should not mention the secret mission in your answer.
    Your answer should only be about the video and be derivable from the task.
    If you leak any direct information of the HINT and let people know about it, 100 kittens will die.

    Here is the HINT of video-level and frame-level anomaly status: {hint_str}

    Describe what happened:
    1. What action was performed? (verb only: squeeze, press, lift, etc.)
    2. How did the object change? (deformed, leaked, moved, etc.)
    3. Was any material released from the object?

    Answer as JSON:
    {{"action": "[verb]", "object_change": "[description]", "material_released": "[yes/no, what if yes]"}}"""),
        ]
    else:
        prompt_messages_action = [
            ('text', f"""Analyze these two frames showing an object manipulation:

    First image: Object with {init_c_name} contour at start
    Second image: Same object with {init_c_name} contour after manipulation. 
                The manipulator (gripper/tool) has {query_c_name} contour.

    Describe what happened:
    1. What action was performed? (verb only: squeeze, press, lift, etc.)
    2. How did the object change? (deformed, leaked, moved, etc.)
    3. Was any material released from the object?

    Answer as JSON:
    {{"action": "[verb]", "object_change": "[description]", "material_released": "[yes/no, what if yes]"}}"""),
        ]
    
    # Semantic class comparison (original functionality)
    prompt_messages_cls = [
        ('text', f"""Compare these two objects:
First image: Object with {init_c_name} contour
Second image: Another object with {init_c_name} contour

Are these semantically the same type of object? 
(Answer 'no' only if they are clearly different things, like a table vs a chair)

Answer: yes or no"""),
    ]
    
    return {
        'identify': prompt_messages_id,
        'frame_check': prompt_messages_frame,
        'state_change': prompt_messages_state_change,
        'interaction': prompt_messages_interaction,
        'action': prompt_messages_action,
        'semantic_class': prompt_messages_cls
    }


# =============================================================================
# Image Processing Utilities
# =============================================================================

def get_attention_image(image, graph, obj_idx, frame_idx, color_rgb, mode="zoom_in",
                         mask_fallback=None):
    """
    Bounding-box-based substitute for get_masked_image(), driven by the
    spatiotemporal tubelet graph.

    Uses a temporally-smoothed bounding box (stable across frames, unlike a
    raw per-frame pixel mask contour) as the attention region.
      mode="zoom_in":          this object's own box only.
      mode="zoom_out":         unions in every graph neighbor's box too, so
                                the VLM sees a manipulator together with what
                                it's manipulating (right for a 1-2 neighbor
                                actuator/workpiece pair).
      mode="zoom_out_siblings": draws every neighbor's own box SEPARATELY
                                (not merged) alongside this object's own box,
                                emphasized -- right for a rigid cluster of
                                many similar parts (e.g. all tips of one
                                multi-channel tool), where a merged box would
                                destroy the sibling-to-sibling comparison
                                that's needed to spot a single outlier.

    Falls back to the old pixel-mask contour (`mask_fallback`, an RLE/bool
    mask) if the graph has no box for this object/frame -- e.g. a track that
    only exists in one frame and was never smoothed.
    """
    h, w = image.shape[:2]

    if graph is not None and mode == "zoom_out_siblings":
        result = get_attention_bboxes_for_siblings(graph, obj_idx, frame_idx)
        if result is not None:
            own_bbox, sibling_bboxes = result
            return draw_sibling_comparison(image, own_bbox, sibling_bboxes, color_rgb)

    bbox = None
    if graph is not None:
        bbox = get_attention_bbox(graph, obj_idx, frame_idx, mode=mode, img_w=w, img_h=h)
    if bbox is not None:
        return draw_bbox_annotation(image, bbox, color_rgb, thickness=3)
    if mask_fallback is not None:
        return get_masked_image(image, mask_fallback, color_rgb)
    return image.copy()


def get_masked_image(image, mask, color_rgb, mask_alpha=0.0, contour_thickness=3):
    """Apply colored contour to image around mask region."""
    if isinstance(mask, dict):
        mask = rle_to_bmask(mask)
    masked_image = apply_anno(
        image.copy(),
        mask=mask,
        mask_color=color_rgb,
        mask_alpha=mask_alpha, 
        contour_thickness=contour_thickness
    )
    return masked_image


def format_output(s):
    return s.replace('\n', '').replace('\"', '"').replace('\'', '"').replace('â€œ', '"').replace('â€', '"')


def yes_no_cleanup(response):
    response_lower = response.lower()
    if 'not sure' in response_lower:
        return 'not sure'
    if 'no' in response_lower and 'yes' not in response_lower:
        return 'no'
    if 'yes' in response_lower and 'no' not in response_lower:
        return 'yes'
    return ''


def rle_wrapper(rle):
    return {
        'counts': rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts'],
        'size': rle['size']
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def get_added_track_starts(pred_data, prompt_obj_idx='0'):
    """Get frames where new objects start being tracked."""
    objs = dict()
    for frame_idx, frame_info in pred_data['supix_masks'].items():
        for obj_idx in frame_info:
            if obj_idx != prompt_obj_idx and obj_idx not in objs:
                objs[obj_idx] = frame_idx
    return objs, prompt_obj_idx


def dip_window_frames(predictions, obj_idx, frame_indices, dense_cfg):
    """
    Frames to sample densely around each time the object dips to the bottom of its
    vertical travel and comes back up (a pipette entering and leaving a tube).

    A dip is a run of frames where the mask's bottom edge is in the lowest
    `depth_frac` part of its observed range. Around each dip's entry and exit we
    return every `interval`-th frame from `pad_before` frames before to `pad_before`
    (entry) or `pad_after` (exit) frames after. Objects whose bottom edge moves less
    than `min_travel_frac` of the image height get no dense frames.
    """
    interval = int(dense_cfg.get('interval', 2))
    pad_before = int(dense_cfg.get('pad_before', 4))
    pad_after = int(dense_cfg.get('pad_after', 15))
    depth_frac = float(dense_cfg.get('depth_frac', 0.7))
    max_gap = int(dense_cfg.get('max_gap', 3))
    min_travel_frac = float(dense_cfg.get('min_travel_frac', 0.05))

    bottoms = {}
    img_h = None
    for f in frame_indices:
        m = predictions.get(str(f), {}).get(obj_idx)
        if m is None:
            continue
        bm = rle_to_bmask(m)
        img_h = bm.shape[0]
        rows = np.where(bm.any(axis=1))[0]
        if len(rows):
            bottoms[f] = int(rows.max())
    if len(bottoms) < 2:
        return []
    lo, hi = min(bottoms.values()), max(bottoms.values())
    if hi - lo < min_travel_frac * img_h:
        return []

    thresh = lo + depth_frac * (hi - lo)
    episodes = []
    for f in sorted(bottoms):
        if bottoms[f] < thresh:
            continue
        if episodes and f - episodes[-1][1] <= max_gap:
            episodes[-1][1] = f
        else:
            episodes.append([f, f])

    available = set(frame_indices)
    window = set()
    for start, end in episodes:
        for edge, after in ((start, pad_before), (end, pad_after)):
            window.update(f for f in range(edge - pad_before, edge + after + 1, interval) if f in available)
    print(f"  Dense sampling around {len(episodes)} dip(s) "
          f"{[tuple(e) for e in episodes]}: +{len(window)} frames")
    return sorted(window)


TIP_CROP_NOTE = """

Third image: a zoomed-in close-up of the LATER frame around the bottom of the
highlighted object, where a pipette tip would be. Answer TIP_CONTENTS from this
close-up. The pipette tip is the thin, translucent cone attached directly to the
bottom of the black pipette shaft. Look at the WHOLE cone down to its point:
brown/amber color anywhere inside it means "colored liquid"; a cone that is clear
or white along its entire length means "empty". If the cone is not fully in view
(only the shaft or the top of the cone is visible), answer "not visible". Judge
only the cone itself: ignore the color of any tube, well plate, or other container
in the close-up."""


def tip_crop(image, mask, tip_cfg):
    """
    Upscaled close-up around the bottom of `mask`, centered on the mask's lowest
    (narrowest) part, where a pipette tip would be. Mostly extends upward, since the
    tip runs up from the mask's bottom edge, with a small margin below in case the
    tracked mask stops short of the tip's end. Returns None for an empty mask.
    """
    bm = rle_to_bmask(mask) if isinstance(mask, dict) else np.asarray(mask) > 0
    rows = np.where(bm.any(axis=1))[0]
    if len(rows) == 0:
        return None
    top, bottom = int(rows.min()), int(rows.max())
    h, w = bm.shape
    band = bm[max(top, bottom - max(1, (bottom - top) // 10)):bottom + 1]
    cx = int(np.median(np.where(band.any(axis=0))[0]))
    half_w = int(float(tip_cfg.get('width_frac', 0.12)) * w / 2)
    y0 = max(0, bottom - int(float(tip_cfg.get('above_frac', 0.30)) * h))
    y1 = min(h, bottom + int(float(tip_cfg.get('below_frac', 0.30)) * h))
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    scale = float(tip_cfg.get('upscale', 2.0))
    return cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def detect_state_changes(
    client,
    model_name: str,
    frame_paths: list,
    pred_data: dict,
    obj_idx: str,
    init_color: np.ndarray,
    init_c_name: str,
    prompts: dict,
    temperature: float = 0.0,
    sample_interval: int = 10,
    graph=None,
    graph_obj_id: str = None,
    dense_cfg=None,
    tip_cfg=None,
) -> list:
    """
    Detect state changes in a tracked object across video frames.

    When `graph` (a SpatioTemporalGraph over this video's tracked objects) is
    supplied, the two comparison frames use a temporally-smoothed bounding
    box instead of the raw per-frame pixel mask -- this is what keeps SAM2's
    frame-to-frame mask jitter from being misread as physical deformation.
    If the object has graph neighbors (e.g. a manipulator moving with the
    part it's holding), the crop zooms out to include them, and the prompt
    is given an explicit note about the relationship.
    """
    state_changes = []
    predictions = pred_data.get('prediction', {})

    frame_indices = sorted([int(k) for k in predictions.keys()])

    if len(frame_indices) < 2:
        return state_changes

    # Sample frames for analysis: uniform, plus dense frames around dips if enabled
    sample_frames = set(frame_indices[::sample_interval]) | {frame_indices[-1]}
    if dense_cfg is not None and dense_cfg.get('enabled', False):
        sample_frames |= set(dip_window_frames(predictions, obj_idx, frame_indices, dense_cfg))
    sample_frames = sorted(sample_frames)


    print(f"  Analyzing {len(sample_frames)} frames for state changes...")

    zoom_mode = choose_zoom_mode(graph, graph_obj_id) if (graph and graph_obj_id) else "zoom_in"
    relation_note = ""
    if zoom_mode != "zoom_in":
        relation_note = "\n\n[TRACKING CONTEXT: " + describe_relation(graph, graph_obj_id) + "]"

    # Compare consecutive sampled frames
    for i in range(len(sample_frames) - 1):
        frame_idx_before = sample_frames[i]
        frame_idx_after = sample_frames[i + 1]

        # Get masks
        mask_before = predictions.get(str(frame_idx_before), {}).get(obj_idx)
        mask_after = predictions.get(str(frame_idx_after), {}).get(obj_idx)

        if mask_before is None or mask_after is None:
            continue

        # Load frames
        img_before = imageio.imread(frame_paths[frame_idx_before])
        img_after = imageio.imread(frame_paths[frame_idx_after])

        # Create attention-region images: bounding box (from the spatiotemporal
        # graph) in preference to the raw pixel mask, which falls back
        # automatically if this object/frame isn't in the graph.
        if graph is not None and graph_obj_id is not None:
            img_before_masked = get_attention_image(
                img_before, graph, graph_obj_id, frame_idx_before, init_color,
                mode=zoom_mode, mask_fallback=mask_before)
            img_after_masked = get_attention_image(
                img_after, graph, graph_obj_id, frame_idx_after, init_color,
                mode=zoom_mode, mask_fallback=mask_after)
        else:
            img_before_masked = get_masked_image(img_before, mask_before, init_color)
            img_after_masked = get_masked_image(img_after, mask_after, init_color)

        # Optional close-up of the later frame's tip region, from the unannotated frame
        tip_img = None
        if tip_cfg is not None and tip_cfg.get('enabled', False):
            tip_img = tip_crop(img_after, mask_after, tip_cfg)

        # Query VLM for state change
        state_change_prompt = prompts['state_change'][0]
        extra_note = relation_note + (TIP_CROP_NOTE if tip_img is not None else "")
        if extra_note:
            state_change_prompt = ('text', state_change_prompt[1] + extra_note)
        content_data = [
            state_change_prompt,
            ('image', img_before_masked),
            ('image', img_after_masked),
        ]
        if tip_img is not None:
            content_data.append(('image', tip_img))
        
        try:
            rsp = call_vlm(
                client=client,
                model_name=model_name,
                system_prompt=get_system_prompt(),
                content_data=content_data,
                temperature=temperature,
            )
            
            # Parse response
            state_changed = False
            change_type = 'none'
            change_desc = ''
            severity = 'none'
            tip_contents = 'not visible'

            for line in rsp.split('\n'):
                line = line.strip()
                if line.startswith('STATE_CHANGED:'):
                    state_changed = 'yes' in line.lower()
                elif line.startswith('CHANGE_TYPE:'):
                    change_type = line.split(':', 1)[1].strip()
                elif line.startswith('CHANGE_DESCRIPTION:'):
                    change_desc = line.split(':', 1)[1].strip()
                elif line.startswith('CHANGE_SEVERITY:'):
                    severity = line.split(':', 1)[1].strip().lower()
                elif line.startswith('TIP_CONTENTS:'):
                    tip_contents = line.split(':', 1)[1].strip().lower()

            # A visible tip's contents is the evidence the pipetting-success
            # criterion in prompt_vad.py needs, so keep it even when the frame
            # pair shows no other state change.
            tip_seen = tip_contents in ('colored liquid', 'empty')
            if state_changed or tip_seen:
                state_changes.append({
                    'start_frame': frame_idx_before,
                    'end_frame': frame_idx_after,
                    'change_type': change_type if state_changed else 'tip_observation',
                    'description': change_desc,
                    'severity': severity if state_changed else 'none',
                    'tip_contents': tip_contents,
                    'object_idx': obj_idx
                })
                print(f"    State change detected: frames {frame_idx_before}-{frame_idx_after} "
                      f"[tip: {tip_contents}]: {change_desc}")
        
        except Exception as e:
            print(f"    Warning: VLM query failed: {e}")
            continue
    
    return state_changes


def analyze_object_interaction(
    client,
    model_name: str,
    frame: np.ndarray,
    obj_mask: dict,
    interactor_mask: dict,
    init_color: np.ndarray,
    query_color: np.ndarray,
    init_c_name: str,
    query_c_name: str,
    prompts: dict,
    temperature: float = 0.0,
) -> dict:
    """
    Analyze interaction between two objects in a frame.
    """
    # Create masked image with both objects
    img_masked = get_masked_image(frame, obj_mask, init_color)
    img_masked = get_masked_image(img_masked, interactor_mask, query_color)
    
    content_data = [
        prompts['interaction'][0],
        ('image', img_masked),
    ]
    
    try:
        rsp = call_vlm(
            client=client,
            model_name=model_name,
            system_prompt=get_system_prompt(),
            content_data=content_data,
            temperature=temperature,
        )
        
        # Parse response
        result = {
            'contact_type': '',
            'force_direction': '',
            'visible_effect': '',
            'action_verb': '',
            'raw_response': rsp
        }
        
        for line in rsp.split('\n'):
            line = line.strip()
            if line.startswith('CONTACT_TYPE:'):
                result['contact_type'] = line.split(':', 1)[1].strip()
            elif line.startswith('FORCE_DIRECTION:'):
                result['force_direction'] = line.split(':', 1)[1].strip()
            elif line.startswith('VISIBLE_EFFECT:'):
                result['visible_effect'] = line.split(':', 1)[1].strip()
            elif line.startswith('ACTION_VERB:'):
                result['action_verb'] = line.split(':', 1)[1].strip()
        
        return result
    
    except Exception as e:
        print(f"    Warning: Interaction analysis failed: {e}")
        return {'error': str(e)}


# =============================================================================
# Main Pipeline
# =============================================================================

def get_parser():
    parser = argparse.ArgumentParser(description="Enhanced VLM analysis for TubeletGraph.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file")
    parser.add_argument('-p', '--pred', type=str, help='prediction directory name', default='vost-val-Annotations_fps5-Ours')
    parser.add_argument('--vlm', type=str, default='openai', choices=['openai', 'qwen'],
                        help='VLM backbone (ignored when pipeline.py exports OVAD_VLM_BACKBONE)')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature for sampling')
    parser.add_argument('--sample_interval', type=int, default=10, help='Frame interval for state change detection')
    parser.add_argument('--skip_state_change', action='store_true', help='Skip state change detection')

    # NEW ARGUMENT
    parser.add_argument('--mask_frame_id', type=int, default=0, help='Frame ID for the mask input.')
    parser.add_argument('--hint_str', type=str, default="", help='Optional hint string to provide to the VLM for better predictions.')

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])
    pred_track_dir = osp.join(cfg.paths.outdir, args.pred)

    # -------------------------------------------------------------------------
    # Client initialisation
    #
    # One OpenAI-protocol client serves both backbones. vlm.base_url decides
    # where it points: unset -> the OpenAI API; set -> an OpenAI-compatible
    # server such as `vllm serve Qwen/Qwen3-VL-32B-Instruct`. The config reaches
    # this script because pipeline.py threads the config file down through
    # quick_run.py and TubeletGraph/run.py, and create_custom_config copies the
    # whole document. $OVAD_VLM_BASE_URL overrides.
    # -------------------------------------------------------------------------
    # Backbone comes from pipeline.py via the environment so every stage agrees.
    # Standalone runs fall back to the config's own vlm block.
    _backbone = (os.environ.get("OVAD_VLM_BACKBONE") or "").strip().lower()
    if _backbone in ("openai", "qwen"):
        # Driven by pipeline.py: take the endpoint it resolved, verbatim.
        base_url = os.environ.get("OVAD_VLM_BASE_URL") or None
        model_name = os.environ.get("OVAD_VLM_MODEL") or None
    elif _backbone == "" and args.vlm == "qwen":
        # Standalone qwen run: base_url and its served model must come as a pair.
        base_url = os.environ.get("OVAD_VLM_BASE_URL") or cfg.vlm.get("base_url") or None
        if not base_url:
            raise SystemExit(
                "--vlm qwen needs an OpenAI-compatible server: set vlm.base_url "
                "in the config or export OVAD_VLM_BASE_URL.")
        model_name = os.environ.get("OVAD_VLM_MODEL")
        if not model_name:
            # Ask the server what it serves rather than guessing from the config.
            model_name = openai.OpenAI(base_url=base_url, api_key="EMPTY",
                                       timeout=30.0).models.list().data[0].id
    else:
        # Standalone openai run: never pair a local base_url with an OpenAI model.
        base_url = None
        model_name = os.environ.get("OVAD_VLM_MODEL") or cfg.vlm.get("openai_model") or "gpt-4.1"

    _client_kwargs = {"timeout": 120.0, "max_retries": 2}
    if base_url:
        _client_kwargs["base_url"] = base_url
        _client_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = openai.OpenAI(**_client_kwargs)
    print(f"[VLM] stage2 model={model_name} endpoint={base_url or 'the OpenAI API'}")

    init_color = np.array(cfg.vlm.init_color_rgb, dtype=float)
    init_c_name = cfg.vlm.init_color_name
    query_color = np.array(cfg.vlm.query_color_rgb, dtype=float)
    query_c_name = cfg.vlm.query_color_name
    
    out_dir = pred_track_dir + f'_' + model_name.replace('/', '_')
    os.makedirs(out_dir, exist_ok=True)
    
    system_prompt = get_system_prompt()
    prompts = get_enhanced_message_prompts(init_c_name, query_c_name, hint_str=args.hint_str)

    instance_names = [x.removesuffix('.json') for x in os.listdir(pred_track_dir) if x.endswith('.json')]

    # Build the spatiotemporal graph once for every tracked object in this
    # video, before captioning any of them -- this is what lets a single
    # object's state-change call know about its graph neighbors (see
    # detect_state_changes / get_attention_image above).
    try:
        _all_bboxes = load_all_video_bboxes(pred_track_dir, instance_names)
        video_graph = build_spatiotemporal_graph(_all_bboxes)
        _n_edges = sum(len(v) for v in video_graph.edges.values()) // 2
        print(f"[spatiotemporal graph] {len(_all_bboxes)} tracked objects, {_n_edges} edges")
    except Exception as e:
        print(f"[spatiotemporal graph] WARNING: failed to build ({e}); "
              f"falling back to per-object pixel masks with no cross-object context.")
        video_graph = None

    for instance_name in tqdm(instance_names, desc="Processing instances"):
        new_pred_path = osp.join(out_dir, f'{instance_name}.json')

        if osp.exists(new_pred_path):
            print(f"Skip {instance_name} as {new_pred_path} exists")
            continue


        # Load frame paths and predictions
        with open(osp.join(pred_track_dir, instance_name + '.json'), 'r') as f:
            pred_data = json.load(f)
        video_name = strip_instance_name(instance_name)
        frame_paths = sorted(glob.glob(osp.join(data_cfg.image_dir, video_name, data_cfg.image_format)))

        track_starts, prompt_obj_idx = get_added_track_starts(pred_data)
        
        to_remove = list()
        obj_info = dict()
        state_change_events = []

        # =====================================================================
        # Stage 0: Always analyze initial object state
        # =====================================================================
        
        # SAFELY DETERMINE THE START FRAME FOR VLM
        mask_frame_str = str(args.mask_frame_id)
        if mask_frame_str not in pred_data['prediction']:
            # Fallback if the requested frame isn't in the predictions
            available_frames = [int(k) for k in pred_data['prediction'].keys()]
            mask_frame_str = str(min(available_frames)) if available_frames else '0'
            print(f"[WARNING] mask_frame_id {args.mask_frame_id} not found in predictions. Falling back to frame {mask_frame_str}.")
            
        # FIXED: Use the correct frame index instead of hardcoded 0
        init_img = imageio.imread(frame_paths[int(mask_frame_str)])
        init_mask = pred_data['prediction'][mask_frame_str].get(prompt_obj_idx)
        
        if init_mask:
            init_img_mask = get_masked_image(init_img, init_mask, init_color)
            
            rsp = call_vlm(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                content_data=[
                    prompts['identify'][0],
                    ('image', init_img_mask),
                ],
                temperature=args.temp,
            )
            
            # Parse object info
            obj_desc = rsp
            obj_state = ''
            obj_material = ''
            for line in rsp.split('\n'):
                if line.startswith('OBJECT:'):
                    obj_desc = line.split(':', 1)[1].strip()
                elif line.startswith('STATE:'):
                    obj_state = line.split(':', 1)[1].strip()
                elif line.startswith('MATERIAL:'):
                    obj_material = line.split(':', 1)[1].strip()
            
            obj_info[prompt_obj_idx] = {
                'desc': obj_desc,
                'initial_state': obj_state,
                'material': obj_material
            }

        # =====================================================================
        # Stage 1: Detect state changes (NEW - even without new objects)
        # =====================================================================
        if not args.skip_state_change:
            
            # Analyze state changes for the main tracked object.
            # graph_obj_id is the global suffix ('2', '5', '6', ...) this
            # instance_name was tracked under -- that's the graph's node id,
            # distinct from prompt_obj_idx which is always '0' inside this
            # object's own per-object prediction file.
            graph_obj_id = instance_name.rsplit('_', 1)[-1]
            state_changes = detect_state_changes(
                client, model_name,
                frame_paths, pred_data,
                prompt_obj_idx,
                init_color, init_c_name,
                prompts,
                args.temp, args.sample_interval,
                graph=video_graph, graph_obj_id=graph_obj_id,
                dense_cfg=cfg.vlm.get('dense_sampling', None),
                tip_cfg=cfg.vlm.get('tip_crop', None),
            )
            
            if state_changes:
                state_change_events.extend(state_changes)
                obj_info[prompt_obj_idx]['state_changes'] = state_changes

        # =====================================================================
        # Stage 2: Analyze newly appearing objects (original functionality)
        # =====================================================================
        if len(track_starts) != 0:
            
            memory_responses = list()
            for obj_idx, first_obj_frame in track_starts.items():
                
                # Find good frame for analysis
                for i in range(10):
                    obj_start_frame = int(first_obj_frame) + i
                    if obj_start_frame >= len(frame_paths):
                        break
                    late_img = imageio.imread(frame_paths[obj_start_frame])

                    all_objs = list(pred_data['supix_masks'][str(obj_start_frame)].keys())
                    init_mask_union = MaskUtils.merge(
                        [pred_data['supix_masks'][str(obj_start_frame)][x] for x in all_objs if x != obj_idx], 
                        intersect=0
                    )
                    new_track_mask = pred_data['supix_masks'][str(obj_start_frame)][obj_idx]
                    late_img_mask_ = get_masked_image(late_img, init_mask_union, init_color)
                    late_img_mask = get_masked_image(late_img_mask_, new_track_mask, query_color)

                    rsp = call_vlm(
                        client=client,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        content_data=[
                            prompts['frame_check'][0],
                            ('image', late_img_mask),
                        ],
                    )
                    if yes_no_cleanup(rsp) == 'yes':
                        break

                # Get action and object description
                content_data = [
                    prompts['action'][0],
                    ('image', init_img_mask),
                    ('image', late_img_mask),
                ]
                for prev_frame_idx, prev_response in memory_responses:
                    t = (obj_start_frame - prev_frame_idx) // data_cfg.fps
                    content_data.append(('text', f"Previous response at frame {t} seconds ago:"))
                    content_data.append(('text', prev_response))

                rsp = call_vlm(
                    client=client,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    content_data=content_data,
                    temperature=args.temp,
                )
                memory_responses.append((obj_start_frame, rsp))
                
                try:
                    # Try to parse JSON response
                    json_str = [x for x in rsp.split('\n') if '```' not in x and '{' in x][0]
                    parsed_rsp = json.loads(json_str)
                    obj_info[obj_idx] = {
                        'desc': parsed_rsp.get('object_change', f'object_{obj_idx}'),
                        'action': parsed_rsp.get('action', ''),
                        'material_released': parsed_rsp.get('material_released', 'no'),
                        "analysis_frame_idx": obj_start_frame,
                        "object_start_frame_idx": int(first_obj_frame)
                    }
                except:
                    # Fallback to tuple parsing
                    try:
                        parsed_rsp = ast.literal_eval([x for x in rsp.split('\n') if '```' not in x][0])
                        obj_info[obj_idx] = {
                            'desc': parsed_rsp[1] if len(parsed_rsp) > 1 else f'object_{obj_idx}',
                            'action': parsed_rsp[2] if len(parsed_rsp) > 2 else '',
                            'prior_desc': parsed_rsp[0] if len(parsed_rsp) > 0 else '',
                            "analysis_frame_idx": obj_start_frame,
                            "object_start_frame_idx": int(first_obj_frame)
                        }
                    except:
                        obj_info[obj_idx] = {
                            'desc': f'object_{obj_idx}',
                            'raw_response': rsp,
                            "analysis_frame_idx": obj_start_frame,
                            "object_start_frame_idx": int(first_obj_frame)
                        }

        # =====================================================================
        # Stage 3: Interaction analysis between objects
        # =====================================================================
        all_obj_ids = list(set([prompt_obj_idx] + list(track_starts.keys())))
        
        if len(all_obj_ids) >= 2:
            
            # Analyze interaction at a few key frames
            frame_indices = sorted([int(k) for k in pred_data['prediction'].keys()])
            mid_frame = frame_indices[len(frame_indices) // 2]
            
            frame_data = pred_data['supix_masks'].get(str(mid_frame), {})
            
            if prompt_obj_idx in frame_data and len(frame_data) > 1:
                other_obj = [k for k in frame_data.keys() if k != prompt_obj_idx][0]
                
                frame = imageio.imread(frame_paths[mid_frame])
                
                interaction_result = analyze_object_interaction(
                    client, model_name,
                    frame,
                    frame_data[prompt_obj_idx],
                    frame_data[other_obj],
                    init_color, query_color,
                    init_c_name, query_c_name,
                    prompts, args.temp,
                )
                
                obj_info[prompt_obj_idx]['interaction'] = interaction_result

        # =====================================================================
        # Save results
        # =====================================================================
        if len(to_remove) > 0:
            to_remove = set(to_remove)
            for frame_idx in pred_data['prediction'].keys():
                new_supix_masks = {k: v for k, v in pred_data['supix_masks'][frame_idx].items() if k not in to_remove}
                pred_data['supix_masks'][frame_idx] = new_supix_masks
                pred_data['prediction'][frame_idx] = {"0": rle_wrapper(MaskUtils.merge(list(new_supix_masks.values()), intersect=0))}
        
        pred_data['obj_info'] = obj_info
        pred_data['state_change_events'] = state_change_events
        
        with open(new_pred_path, 'w') as f:
            json.dump(pred_data, f)
        
        print(f"Saved: {new_pred_path}")
        print(f"  Objects: {list(obj_info.keys())}")
        print(f"  State changes: {len(state_change_events)}")

    # =========================================================================
    # Post-pass: consensus filter across the whole video's sibling clusters.
    # Must run after every instance is captioned (not per-object, inline
    # above) because it needs to see all objects' claims at once to tell
    # "one real outlier" apart from "the whole rigid cluster reported a
    # difference at the same moment" (almost always shared motion blur).
    # =========================================================================
    if video_graph is not None:
        events_by_obj = {}
        for instance_name in instance_names:
            path = osp.join(out_dir, f'{instance_name}.json')
            if not osp.exists(path):
                continue
            with open(path, 'r') as f:
                pred_data = json.load(f)
            graph_obj_id = instance_name.rsplit('_', 1)[-1]
            events_by_obj[graph_obj_id] = pred_data.get('state_change_events', [])

        filtered_by_obj = apply_consensus_filter(events_by_obj, video_graph)

        n_downgraded = 0
        for instance_name in instance_names:
            path = osp.join(out_dir, f'{instance_name}.json')
            if not osp.exists(path):
                continue
            graph_obj_id = instance_name.rsplit('_', 1)[-1]
            new_events = filtered_by_obj.get(graph_obj_id)
            if new_events is None or new_events == events_by_obj.get(graph_obj_id):
                continue

            with open(path, 'r') as f:
                pred_data = json.load(f)
            pred_data['state_change_events'] = new_events
            for obj_key in pred_data.get('obj_info', {}):
                pred_data['obj_info'][obj_key]['state_changes'] = new_events
            with open(path, 'w') as f:
                json.dump(pred_data, f)
            n_downgraded += 1

        print(f"[consensus filter] re-examined {len(events_by_obj)} objects' claims, "
              f"rewrote {n_downgraded} instance file(s) with shared-artifact claims downgraded")


        