"""SAM3 point-to-bbox annotation for IOCfish5k: whole-image pass, then
per-point crops at halving windows, then a square fallback."""

import argparse
import gc
import logging
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env
load_env()

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # PVS model: points are already given

from IOCfish5kDataset.SAM3_annotation_helpers import (
    DATASET_DIR, IMAGES_DIR, ANNOTATIONS_DIR, OUTPUT_DIR, OUTPUT_SEGMAP_DIR, DEPTH_DIR,
    SEED_WINDOW, BOUNDARY_MARGIN_PIXELS, PAD_LABEL, PAD_COORD, GLOBAL_OBJECT_CHUNK,
    MIN_IOU_SCORE, MIN_IOU_SCORE_DEPTH, MODEL_ID, LARGER_MASK_SCORE_FRAC, MIN_SHRINK_WINDOW,
    CROP_FILL_REJECT_FRAC, CROP_FILL_HIGH_CONF_SCORE, CROP_FILL_HIGH_CONF_FRAC,
    SHRINK_LAST_RESORT_WINDOWS, LAST_RESORT_CROP_FILL_MAX, LAST_RESORT_BOUNDARY_FILL_MAX,
    FINAL_IOU_FLOOR, MAX_CO_PENDING_PER_CROP, SHRINK_MAX_NEGATIVES, SHARPEN_SIGMA, SHARPEN_AMOUNT,
    MIN_IOU_SCORE_DEPTH_WHOLE, MIN_IOU_SCORE_WHOLE, MIN_IOU_SCORE_WHOLE_TEXTURED, TEXTURE_BUSY_STD,
    MIN_WHOLE_IMAGE_KEEP_SCORE, FAR_DEPTH_PERCENTILE, DENSE_CLUSTER_LOCAL_SCALE, LARGE_OBJECT_SKIP_LENIENT_SCALE,
    LEDGER_RESTORE_ABS_FLOOR, FALLBACK_SQUARE_SIZE_MULT, COMBINED_HADAMARD_THRESHOLD,
    SHRINK_IOU_FLOORS, SHRINK_LARGER_MASK_SCORE_FRAC, RESCUE_FLOOR_DEFAULT, RESCUE_AREA_LO, RESCUE_AREA_HI,
    parse_centerpoints, _mask_is_boxy, _per_point_local_scale, _square_fallback_xyxy,
    _load_depth_array, _load_depth_gray_array, _normalize_depth_gray, _local_texture_std,
    _build_object_prompts, _enforce_point_inside_bbox, _run_sam3_forward, _unsharp_mask,
    _build_nearby_count, _negatives_cap_for_point, _record_near_miss,
    _candidate_depth_consistent, _point_in_kept, _depth_size_factor,
    _build_orange_at_point, _build_far_or_violet_mask,
    render_segmap_colored, _write_bbox_xml_with_sources, _auto_batch_size,
    _run_sam3_batch_multi_obj, _finalize_mask,
)

# line-buffered so progress shows up in the SLURM .out
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("sam3_hpc_annot")



# cascade passes

def _whole_image_pass(SAM3model, SAM3processor, device, image_array, centerpoints,
                      instance_map, bboxes, bbox_sources=None,
                      depth_array=None, depth_gray_norm=None,
                      sharpen=False, sharpen_depth=False,
                      local_scales=None, candidates_ledger=None,
                      far_depth_threshold=None,
                      is_far_or_violet=None,
                      near_miss_log=None,
                      rescue_mode=False,
                      rescue_indices=None,
                      nearby_count=None,
                      orange_at_point=None):
    """SAM3 on the full image with all valid points as objects, in chunks of
    GLOBAL_OBJECT_CHUNK. Dense-cluster points are left to the shrink cascade."""
    image_height, image_width = image_array.shape[:2]
    # SAM3 can't separate dense clusters at full-image scale
    def _is_dense(i):
        if local_scales is None or local_scales[i] is None:
            return False
        return local_scales[i] <= DENSE_CLUSTER_LOCAL_SCALE
    rescue_set = set(rescue_indices) if rescue_indices is not None else None
    valid_indices = [
        i for i, p in enumerate(centerpoints)
        if p is not None
        and (rescue_set is None or i in rescue_set)
        and not _is_dense(i)
    ]
    if not valid_indices:
        return
    points_array = np.array(
        [(p[0], p[1]) if p is not None else (0, 0) for p in centerpoints], dtype=np.float32
    )
    rgb_for_sam = _unsharp_mask(image_array) if sharpen else image_array
    depth_for_sam = None
    if depth_array is not None:
        depth_for_sam = _unsharp_mask(depth_array) if sharpen_depth else depth_array
    rgb_pil = Image.fromarray(rgb_for_sam)
    depth_pil = Image.fromarray(depth_for_sam) if depth_for_sam is not None else None

    for chunk_start in range(0, len(valid_indices), GLOBAL_OBJECT_CHUNK):
        chunk_indices = valid_indices[chunk_start:chunk_start + GLOBAL_OBJECT_CHUNK]
        chunk_points, chunk_labels, chunk_negs = [], [], []
        for tgt in chunk_indices:
            pts, lbls, negs = _build_object_prompts(tgt, valid_indices, points_array)
            # pts is [[x, y], [neg_x, neg_y], ...], lbls is [1, 0, 0, ...], negs is list of [neg_x, neg_y]
            # per-point negatives cap, positive prompt stays at index 0
            cap = _negatives_cap_for_point(
                local_scales[tgt] if local_scales is not None else None,
                nearby_count=(nearby_count[tgt] if nearby_count is not None else None),
            )
            if len(negs) > cap:
                negs = negs[:cap]
                pts = pts[:1 + cap]
                lbls = lbls[:1 + cap]
            chunk_points.append(pts)    # one pts list per object in the chunk
            chunk_labels.append(lbls)
            chunk_negs.append(negs)
        max_pts = max(len(pts) for pts in chunk_points)
        for i in range(len(chunk_points)):
            pad = max_pts - len(chunk_points[i])
            chunk_points[i] = chunk_points[i] + [PAD_COORD] * pad
            chunk_labels[i] = chunk_labels[i] + [PAD_LABEL] * pad

        rgb_masks, rgb_scores = _run_sam3_forward(
            SAM3model, SAM3processor, device, rgb_pil,
            [chunk_points], [chunk_labels],
        )
        depth_masks = depth_scores = None
        if depth_pil is not None:
            depth_masks, depth_scores = _run_sam3_forward(
                SAM3model, SAM3processor, device, depth_pil,
                [chunk_points], [chunk_labels],
            )

        for pos, idx in enumerate(chunk_indices):
            rgb_score = float(rgb_scores[pos])
            depth_score = (
                float(depth_scores[pos]) if depth_scores is not None else None
            )
            cx, cy = centerpoints[idx]
            # relaxed floors at full-image scale, busy regions get a higher
            # RGB floor (coral grabs)
            point_texture = _local_texture_std(image_array, cx, cy, radius=24)
            rgb_floor = (
                MIN_IOU_SCORE_WHOLE_TEXTURED
                if point_texture > TEXTURE_BUSY_STD
                else MIN_IOU_SCORE_WHOLE
            )
            depth_floor_whole = MIN_IOU_SCORE_DEPTH_WHOLE
            # rescue halves the floors
            if rescue_mode:
                rgb_floor = rgb_floor * 0.5
                depth_floor_whole = depth_floor_whole * 0.5
            has_rgb = rgb_score >= rgb_floor
            has_depth = (
                depth_masks is not None
                and depth_score is not None
                and depth_score >= depth_floor_whole
            )
            # both floors failed but rgb*depth clears COMBINED_HADAMARD_THRESHOLD
            hadamard_rescue = False
            if (not has_rgb and not has_depth
                    and depth_masks is not None and depth_score is not None
                    and rgb_score * depth_score >= COMBINED_HADAMARD_THRESHOLD):
                has_rgb = True
                has_depth = True
                hadamard_rescue = True
            if not has_rgb and not has_depth:
                _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                                  rgb_floor, "whole_image", "below_floor")
                if depth_score is not None:
                    _record_near_miss(near_miss_log, idx, "depth", depth_score,
                                      depth_floor_whole, "whole_image",
                                      "below_floor")
                continue
            rgb_mask = rgb_masks[pos].numpy().astype(bool) if has_rgb else np.zeros_like(
                rgb_masks[pos].numpy(), dtype=bool
            )
            d_mask = depth_masks[pos].numpy().astype(bool) if has_depth else None
            _finalize_mask(
                rgb_mask, cx, cy, chunk_negs[pos],
                image_width, image_height, 0, 0, idx,
                instance_map, bboxes,
                depth_mask=d_mask, depth_gray_norm=depth_gray_norm,
                point_xy=(cx, cy),
                bbox_sources=bbox_sources,
                cascade_label=("whole_image_rescue" if rescue_mode else "whole_image"),
                sharpened=bool(sharpen),
                rgb_score=rgb_score if has_rgb else None,
                depth_score=depth_score if has_depth else None,
                image_array=image_array,
                local_scale=(local_scales[idx] if local_scales is not None else None),
                candidates_ledger=candidates_ledger,
                far_depth_threshold=far_depth_threshold,
                is_far_or_violet=is_far_or_violet,
                prefer_depth_hadamard=hadamard_rescue,
                near_miss_log=near_miss_log,
                rescue_mode=rescue_mode,
                orange_at_point_focal=bool(
                    orange_at_point is not None and orange_at_point[idx]
                ),
            )

def _adaptive_window_pass(SAM3model, SAM3processor, device, image_array, centerpoints,
                          pending, instance_map, bboxes, batch_size,
                          local_scales, depth_array=None, depth_gray_norm=None,
                          sharpen=True, sharpen_depth=False, bbox_sources=None,
                          candidates_ledger=None, far_depth_threshold=None,
                          is_far_or_violet=None,
                          near_miss_log=None,
                          orange_at_point=None,
                          rescue_mode=False,
                          nearby_count=None):
    """Run _shrink_pass once per window size (256/512/1024/2048) before the
    halving cascade, each point bucketed by max(SEED_WINDOW, 3 * local_scale)."""
    if not pending or local_scales is None:
        return pending
    buckets = [256, 512, 1024, 2048]
    by_window = {b: set() for b in buckets}
    fallback_scale = max(15, min(image_array.shape[:2]) // 40)
    for idx in pending:
        scale = local_scales[idx] if local_scales[idx] is not None else fallback_scale
        target = max(SEED_WINDOW, int(3 * scale))
        chosen = next((b for b in buckets if b >= target), buckets[-1])
        # near-camera orange points go up one bucket
        if orange_at_point is not None and orange_at_point[idx]:
            ci = buckets.index(chosen)
            chosen = buckets[min(ci + 1, len(buckets) - 1)]
        by_window[chosen].add(idx)
    pending = set(pending)
    # larger windows first, small buckets only see fragments of big fish
    for window in reversed(buckets):
        subset = by_window[window] & pending
        if not subset:
            continue
        # the halving cascade already handles the SEED_WINDOW bucket
        if window == SEED_WINDOW:
            continue
        before = len(pending)
        new_pending = _shrink_pass(
            SAM3model, SAM3processor, device, image_array, centerpoints,
            subset, window, instance_map, bboxes, batch_size,
            depth_array=depth_array, depth_gray_norm=depth_gray_norm,
            sharpen=sharpen, sharpen_depth=sharpen_depth,
            bbox_sources=bbox_sources, accept_any_iou=False,
            local_scales=local_scales, candidates_ledger=candidates_ledger,
            far_depth_threshold=far_depth_threshold,
            is_far_or_violet=is_far_or_violet,
            near_miss_log=near_miss_log,
            orange_at_point=orange_at_point,
            rescue_mode=rescue_mode,
            nearby_count=nearby_count,
        )
        satisfied = subset - new_pending
        pending -= satisfied
        logger.debug(f"adaptive window={window} on {len(subset)} pts: "
                     f"{before - len(pending)} satisfied, {len(pending)} pending")
    return pending

def _shrink_pass(SAM3model, SAM3processor, device, image_array, centerpoints,
                 pending, window_size, instance_map, bboxes,
                 batch_size, depth_array=None, depth_gray_norm=None,
                 sharpen=True, sharpen_depth=False,
                 bbox_sources=None, accept_any_iou=False,
                 local_scales=None, candidates_ledger=None,
                 far_depth_threshold=None,
                 is_far_or_violet=None,
                 near_miss_log=None,
                 orange_at_point=None,
                 rescue_mode=False,
                 nearby_count=None):
    """One shrink-cascade pass at window_size: crop around each pending point,
    run SAM3 multi-object in batches, other in-crop points as negatives.
    Returns the points still pending after this pass."""
    image_height, image_width = image_array.shape[:2]
    pending = set(pending)
    if accept_any_iou:
        iou_floor = FINAL_IOU_FLOOR
        depth_iou_floor = FINAL_IOU_FLOOR
    else:
        rgb_default, depth_default = MIN_IOU_SCORE, MIN_IOU_SCORE_DEPTH
        iou_floor, depth_iou_floor = SHRINK_IOU_FLOORS.get(
            window_size, (rgb_default, depth_default)
        )
        if rescue_mode:
            # rescue halves the floors
            iou_floor = iou_floor * 0.5
            depth_iou_floor = depth_iou_floor * 0.5
    do_sharpen = bool(sharpen)
    # relax prefer-larger-mask at small windows
    score_frac = SHRINK_LARGER_MASK_SCORE_FRAC.get(window_size, LARGER_MASK_SCORE_FRAC)
    cascade_label = f"shrink_w{window_size}"
    if accept_any_iou:
        cascade_label += "_lenient"
    if rescue_mode:
        cascade_label += "_rescue"
    if not pending:
        return pending

    # orange points skip the smallest windows, they need bigger crops
    if (orange_at_point is not None and window_size <= 64 and not accept_any_iou):
        pending = {i for i in pending if not orange_at_point[i]}
        if not pending:
            return pending

    pending_sorted = sorted(pending)
    pending_xy = {i: centerpoints[i] for i in pending_sorted}

    def crop_window(image_array, center_x, center_y, window_size):
        image_height, image_width = image_array.shape[:2]
        effective_size = min(window_size, image_width, image_height)
        half = effective_size // 2
        x_min = max(0, center_x - half)
        y_min = max(0, center_y - half)
        x_max = min(image_width, x_min + effective_size)
        y_max = min(image_height, y_min + effective_size)
        x_min = max(0, x_max - effective_size)
        y_min = max(0, y_max - effective_size)
        return image_array[y_min:y_max, x_min:x_max], x_min, y_min, x_max, y_max

    # one crop per pending point (focal first), negatives are the other in-crop points
    crop_meta = []
    for focal in pending_sorted:
        cx, cy = pending_xy[focal]
        crop, x0, y0, x1, y1 = crop_window(image_array, cx, cy, window_size)
        if do_sharpen:
            crop = _unsharp_mask(crop)
        # co-pending points in the crop, capped to MAX_CO_PENDING_PER_CROP
        co_inside = [
            j for j in pending_sorted
            if j != focal and x0 <= pending_xy[j][0] < x1 and y0 <= pending_xy[j][1] < y1
        ]
        if len(co_inside) > MAX_CO_PENDING_PER_CROP:
            co_inside.sort(key=lambda j: (pending_xy[j][0] - cx) ** 2 + (pending_xy[j][1] - cy) ** 2)
            co_inside = co_inside[:MAX_CO_PENDING_PER_CROP]
        co_pending = [focal] + co_inside
        # non-pending points in the crop are shared negatives, closest survive the cap
        negatives_local = []
        for k, p in enumerate(centerpoints):
            if p is None or k in pending:
                continue
            if x0 <= p[0] < x1 and y0 <= p[1] < y1:
                negatives_local.append((p[0] - x0, p[1] - y0))
        cap = _negatives_cap_for_point(
            local_scales[focal] if local_scales is not None else None,
            nearby_count=(nearby_count[focal] if nearby_count is not None else None),
        )
        # per-window absolute cap, 10 negatives crowd a small crop
        window_cap = SHRINK_MAX_NEGATIVES.get(window_size)
        if window_cap is not None:
            cap = min(cap, window_cap)
        if len(negatives_local) > cap:
            negatives_local.sort(
                key=lambda n: (n[0] - (cx - x0)) ** 2 + (n[1] - (cy - y0)) ** 2
            )
            negatives_local = negatives_local[:cap]
        per_obj_points, per_obj_labels = [], []
        for j in co_pending:
            jx, jy = pending_xy[j]
            # other co-pending objects as negatives so SAM3 separates them, nearest first
            co_negs = sorted(
                ((pending_xy[k][0] - x0, pending_xy[k][1] - y0)
                 for k in co_pending if k != j),
                key=lambda n: (n[0] - (jx - x0)) ** 2 + (n[1] - (jy - y0)) ** 2,
            )
            obj_negs = (co_negs + list(negatives_local))[:cap]
            per_obj_points.append([[jx - x0, jy - y0]] + [list(n) for n in obj_negs])
            per_obj_labels.append([1] + [0] * len(obj_negs))
        crop_meta.append((focal, x0, y0, x1, y1, crop, co_pending,
                          per_obj_points, per_obj_labels, negatives_local))

    # batched forwards, cached for the two attribution phases
    rgb_per_crop = [None] * len(crop_meta)
    depth_per_crop = [None] * len(crop_meta)
    for batch_start in range(0, len(crop_meta), batch_size):
        sl = slice(batch_start, batch_start + batch_size)
        batch = crop_meta[sl]
        rgb_pils = [Image.fromarray(m[5]) for m in batch]
        batch_pts = [m[7] for m in batch]
        batch_lbls = [m[8] for m in batch]
        rgb_out = _run_sam3_batch_multi_obj(
            SAM3model, SAM3processor, device, rgb_pils, batch_pts, batch_lbls,
            score_frac=score_frac,
        )
        for i, r in enumerate(rgb_out):
            rgb_per_crop[batch_start + i] = r
        if depth_array is not None:
            depth_crops = [depth_array[m[2]:m[4], m[1]:m[3]] for m in batch]
            if do_sharpen and sharpen_depth:
                depth_crops = [_unsharp_mask(c) for c in depth_crops]
            depth_pils = [Image.fromarray(c) for c in depth_crops]
            depth_out = _run_sam3_batch_multi_obj(
                SAM3model, SAM3processor, device, depth_pils, batch_pts, batch_lbls,
                score_frac=score_frac,
            )
            for i, r in enumerate(depth_out):
                depth_per_crop[batch_start + i] = r

    def _try_attribute(crop_index, obj_pos):
        """Attribute crop[crop_index]'s obj_pos result to its annotation point."""
        focal, x0, y0, x1, y1, _, co_pending, _, _, negatives_local = crop_meta[crop_index]
        j = co_pending[obj_pos]
        if j not in pending:
            return False
        rgb_stack, rgb_scores = rgb_per_crop[crop_index]
        rgb_qualifies = rgb_scores[obj_pos] >= iou_floor
        # need the depth score before the early-return for the rescue check
        depth_raw_score = None
        if depth_per_crop[crop_index] is not None:
            _, _ds = depth_per_crop[crop_index]
            depth_raw_score = float(_ds[obj_pos])
        depth_qualifies = depth_raw_score is not None and depth_raw_score >= depth_iou_floor
        hadamard_rescue = False
        if not rgb_qualifies and not depth_qualifies:
            if (depth_raw_score is not None and float(rgb_scores[obj_pos]) * depth_raw_score >= COMBINED_HADAMARD_THRESHOLD): hadamard_rescue = True
            else:
                _record_near_miss(near_miss_log, j, "rgb",
                                  float(rgb_scores[obj_pos]), iou_floor,
                                  cascade_label, "below_floor")
                if depth_raw_score is not None:
                    _record_near_miss(near_miss_log, j, "depth", depth_raw_score,
                                      depth_iou_floor, cascade_label, "below_floor")
                return False
        rgb_mask = rgb_stack[obj_pos].numpy()
        # reject crop-filling masks (the square-in-segmap failure), confident
        # close-up fish on orange depth are exempt
        last_resort = rescue_mode and window_size in SHRINK_LAST_RESORT_WINDOWS
        if rgb_mask.size > 0:
            rgb_fill = float(rgb_mask.mean())
            if rgb_fill > CROP_FILL_REJECT_FRAC:
                if last_resort:
                    # anything below the absolute cap, even boxy
                    skip_crop_fill = (rgb_fill <= LAST_RESORT_CROP_FILL_MAX)
                elif rescue_mode:
                    skip_crop_fill = (
                        rgb_fill <= CROP_FILL_HIGH_CONF_FRAC
                        and not _mask_is_boxy(rgb_mask)
                    )
                else:
                    skip_crop_fill = (
                        float(rgb_scores[obj_pos]) >= CROP_FILL_HIGH_CONF_SCORE
                        and rgb_fill <= CROP_FILL_HIGH_CONF_FRAC
                        and not _mask_is_boxy(rgb_mask)
                        and orange_at_point is not None
                        and orange_at_point[j]
                    )
                if not skip_crop_fill:
                    _record_near_miss(near_miss_log, j, "rgb",
                                      float(rgb_scores[obj_pos]), iou_floor,
                                      cascade_label, "crop_filled")
                    if depth_raw_score is not None:
                        _record_near_miss(near_miss_log, j, "depth", depth_raw_score,
                                          depth_iou_floor, cascade_label, "crop_filled")
                    return False
                
        # sub-5%-fill masks at small windows are fragments, keep the point pending
        if window_size <= 64 and rgb_mask.size > 0:
            if float(rgb_mask.sum()) < 0.05 * float(window_size * window_size):
                _record_near_miss(near_miss_log, j, "rgb", float(rgb_scores[obj_pos]),
                                  iou_floor, cascade_label, "tiny_fragment")
                return False
            
        # reject masks touching a crop edge that isn't an image edge (a fish
        # running off the picture is fine)
        def _count_internal_edges_touched(mask, margin, x0, y0, x1, y1, image_w, image_h):
            """Count crop edges the mask touches that are not image edges."""
            h, w = mask.shape[:2]
            if margin <= 0 or h <= 0 or w <= 0:
                return 0
            n = 0
            if y0 > 0 and mask[:margin].any():
                n += 1
            if y1 < image_h and mask[-margin:].any():
                n += 1
            if x0 > 0 and mask[:, :margin].any():
                n += 1
            if x1 < image_w and mask[:, -margin:].any():
                n += 1
            return n

        n_internal = _count_internal_edges_touched(
            rgb_mask, BOUNDARY_MARGIN_PIXELS,
            x0, y0, x1, y1, image_width, image_height,
        )
        if n_internal > 0:
            rgb_fill_b = float(rgb_mask.mean()) if rgb_mask.size else 1.0
            ys_b, xs_b = np.where(rgb_mask)
            if len(xs_b) == 0:
                _record_near_miss(near_miss_log, j, "rgb",
                                  float(rgb_scores[obj_pos]), iou_floor,
                                  cascade_label, "boundary_touch")
                return False
            bh_b = float(int(ys_b.max()) - int(ys_b.min()) + 1)
            bw_b = float(int(xs_b.max()) - int(xs_b.min()) + 1)
            aspect = max(bh_b, bw_b) / max(1.0, min(bh_b, bw_b))
            high_score = float(rgb_scores[obj_pos]) >= 0.70
            if last_resort:
                # all 4 edges allowed up to the cap at w32/w64
                relaxable = (n_internal <= 4 and rgb_fill_b <= LAST_RESORT_BOUNDARY_FILL_MAX)
            elif rescue_mode:
                # up to 2 internal edges if not filling the crop
                relaxable = (n_internal <= 2 and rgb_fill_b <= 0.90)
            else:
                relaxable = (
                    n_internal == 1 and rgb_fill_b <= 0.70
                    and (high_score or aspect > 3.0)
                )
            if not relaxable:
                _record_near_miss(near_miss_log, j, "rgb",
                                  float(rgb_scores[obj_pos]), iou_floor,
                                  cascade_label, "boundary_touch")
                return False
        d_mask = None
        d_score_used = None
        # the rescue path needs the depth mask even below its own floor
        if depth_per_crop[crop_index] is not None:
            d_stack, d_scores = depth_per_crop[crop_index]
            if d_scores[obj_pos] >= depth_iou_floor or hadamard_rescue:
                d_mask = d_stack[obj_pos].numpy()
                d_score_used = float(d_scores[obj_pos])
        jx, jy = pending_xy[j]
        lx, ly = jx - x0, jy - y0
        local_negs = [(pending_xy[k][0] - x0, pending_xy[k][1] - y0)
                      for k in co_pending if k != j]
        local_negs.extend(negatives_local)
        ok = _finalize_mask(
            rgb_mask, lx, ly, local_negs,
            image_width, image_height, x0, y0, j,
            instance_map, bboxes,
            depth_mask=d_mask, depth_gray_norm=depth_gray_norm,
            point_xy=(jx, jy),
            bbox_sources=bbox_sources,
            cascade_label=cascade_label,
            sharpened=do_sharpen,
            rgb_score=float(rgb_scores[obj_pos]),
            depth_score=d_score_used,
            image_array=image_array,
            local_scale=(local_scales[j] if local_scales is not None else None),
            candidates_ledger=candidates_ledger,
            far_depth_threshold=far_depth_threshold,
            is_far_or_violet=is_far_or_violet,
            prefer_depth_hadamard=hadamard_rescue,
            near_miss_log=near_miss_log,
            rescue_mode=rescue_mode,
            orange_at_point_focal=bool(
                orange_at_point is not None and orange_at_point[j]
            ),
        )
        if ok:
            pending.discard(j)
        return ok

    # own-crop attribution: each focal against its own centered crop
    for ci in range(len(crop_meta)):
        _try_attribute(ci, 0)

    # then cross-attribution: still-pending points from neighbouring crops
    for ci, meta in enumerate(crop_meta):
        co_pending = meta[6]
        if len(co_pending) <= 1:
            continue
        for obj_pos in range(1, len(co_pending)):
            if co_pending[obj_pos] in pending:
                _try_attribute(ci, obj_pos)

    return pending

def _whole_image_depth_rescue(SAM3model, SAM3processor, device, image_array,
                              centerpoints, instance_map, bboxes,
                              depth_array, depth_gray_norm,
                              bbox_sources=None, sharpen=False, sharpen_depth=False,
                              floor=RESCUE_FLOOR_DEFAULT, local_scales=None,
                              far_depth_threshold=None,
                              is_far_or_violet=None,
                              near_miss_log=None,
                              nearby_count=None):
    """Depth-only whole-image rescue for points the RGB pass missed. Rejects
    masks below floor or outside [RESCUE_AREA_LO, RESCUE_AREA_HI] * scale^2."""
    if depth_array is None:
        return
    image_height, image_width = image_array.shape[:2]
    pending = [i for i, p in enumerate(centerpoints)
               if p is not None and bboxes[i] is None]
    if not pending:
        return
    points_array = np.array(
        [(p[0], p[1]) if p is not None else (0, 0) for p in centerpoints], dtype=np.float32
    )
    valid_indices = [i for i, p in enumerate(centerpoints) if p is not None]
    depth_for_sam = _unsharp_mask(depth_array) if (sharpen and sharpen_depth) else depth_array
    depth_pil = Image.fromarray(depth_for_sam)

    fallback_scale = max(15, min(image_height, image_width) // 40)

    for chunk_start in range(0, len(pending), GLOBAL_OBJECT_CHUNK):
        chunk_indices = pending[chunk_start:chunk_start + GLOBAL_OBJECT_CHUNK]
        chunk_points, chunk_labels, chunk_negs = [], [], []
        for tgt in chunk_indices:
            pts, lbls, negs = _build_object_prompts(tgt, valid_indices, points_array)
            cap = _negatives_cap_for_point(
                local_scales[tgt] if local_scales is not None else None,
                nearby_count=(nearby_count[tgt] if nearby_count is not None else None),
            )
            if len(negs) > cap:
                negs = negs[:cap]
                pts = pts[:1 + cap]
                lbls = lbls[:1 + cap]
            chunk_points.append(pts)
            chunk_labels.append(lbls)
            chunk_negs.append(negs)
        max_pts = max(len(pts) for pts in chunk_points)
        for i in range(len(chunk_points)):
            pad = max_pts - len(chunk_points[i])
            chunk_points[i] = chunk_points[i] + [PAD_COORD] * pad
            chunk_labels[i] = chunk_labels[i] + [PAD_LABEL] * pad

        # batch helper (single image) so we get allow_point_near
        out = _run_sam3_batch_multi_obj(
            SAM3model, SAM3processor, device,
            [depth_pil], [chunk_points], [chunk_labels],
            allow_point_near=True,
        )
        d_stack, d_scores = out[0]
        for pos, idx in enumerate(chunk_indices):
            d_score = float(d_scores[pos])
            if d_score < floor:
                continue
            d_mask = d_stack[pos].numpy().astype(bool)

            # area band, oversized blobs eat several fish
            scale = (
                local_scales[idx]
                if local_scales is not None and local_scales[idx] is not None
                else fallback_scale
            )
            scale_area = float(scale) ** 2
            mask_area = float(d_mask.sum())
            if (mask_area < RESCUE_AREA_LO * scale_area
                    or mask_area > RESCUE_AREA_HI * scale_area):
                continue
            cx, cy = centerpoints[idx]
            _finalize_mask(
                np.zeros_like(d_mask, dtype=bool),  # no RGB
                cx, cy, chunk_negs[pos],
                image_width, image_height, 0, 0, idx,
                instance_map, bboxes,
                depth_mask=d_mask, depth_gray_norm=depth_gray_norm,
                point_xy=(cx, cy),
                bbox_sources=bbox_sources,
                cascade_label="whole_image_depth_rescue",
                sharpened=bool(sharpen and sharpen_depth),
                rgb_score=None,
                depth_score=d_score,
                image_array=image_array,
                local_scale=(
                    local_scales[idx]
                    if local_scales is not None and local_scales[idx] is not None
                    else None
                ),
                far_depth_threshold=far_depth_threshold,
                is_far_or_violet=is_far_or_violet,
                near_miss_log=near_miss_log,
            )



# cascade driver

def segment_with_cascade(SAM3model, SAM3processor, device, image_array, centerpoints,
                         instance_map, local_scales, batch_size,
                         depth_array=None, depth_gray_norm=None,
                         sharpen=True, sharpen_depth=False,
                         bbox_sources=None, enable_depth_rescue=False):
    """Full cascade for one image: whole-image pass, adaptive prepass, halving
    shrink cascade, ledger restore, rescue, square fallback. Returns a list of
    [x0,y0,x1,y1] bboxes aligned with centerpoints."""
    image_height, image_width = image_array.shape[:2]
    bboxes = [None] * len(centerpoints)
    valid_indices = [i for i, p in enumerate(centerpoints) if p is not None]
    if not valid_indices:
        return bboxes

    candidates_ledger = {} # best rejected attempts, restored before the square fallback

    # bluest-40% depth threshold, drops oversized water-column grabs
    far_depth_threshold = (float(np.percentile(depth_gray_norm, FAR_DEPTH_PERCENTILE)) if depth_gray_norm is not None else None)
    # per-image far/violet pixel mask, used to reject background grabs
    is_far_or_violet = _build_far_or_violet_mask(depth_array)
    # per-point flag for orange (near-camera) depth, routed to larger windows
    orange_at_point = _build_orange_at_point(depth_array, depth_gray_norm, centerpoints)
    # neighbour count within ~2.5 x median NN distance, drives the negatives cap
    nearby_count = _build_nearby_count(centerpoints)
    # best near-floor attempt per point, written out on fallback_square entries
    near_miss_log = {}

    # whole-image pass (sharpened by default, see --sharpen)
    _whole_image_pass(
        SAM3model, SAM3processor, device, image_array, centerpoints,
        instance_map, bboxes, bbox_sources=bbox_sources,
        depth_array=depth_array, depth_gray_norm=depth_gray_norm,
        sharpen=sharpen, sharpen_depth=sharpen_depth,
        local_scales=local_scales, candidates_ledger=candidates_ledger,
        far_depth_threshold=far_depth_threshold,
        is_far_or_violet=is_far_or_violet,
        near_miss_log=near_miss_log,
        nearby_count=nearby_count,
        orange_at_point=orange_at_point,
    )

    # demote low-confidence whole-image bboxes back to pending, the ledger
    # still holds them
    if bbox_sources is not None:
        for idx in valid_indices:
            if bboxes[idx] is None:
                continue
            src = bbox_sources[idx]
            if not src or not str(src.get("cascade", "")).startswith("whole_image"):
                continue
            rgb_s = src.get("rgb_score")
            depth_s = src.get("depth_score")
            best = max(
                rgb_s if rgb_s is not None else 0.0,
                depth_s if depth_s is not None else 0.0,
            )
            if best < MIN_WHOLE_IMAGE_KEEP_SCORE:
                ys, xs = np.where(instance_map == (idx + 1))
                if len(xs) > 0:
                    instance_map[ys, xs] = 0
                bboxes[idx] = None
                bbox_sources[idx] = None

    pending = {i for i in valid_indices if bboxes[i] is None}
    if not pending:
        return bboxes


    # adaptive prepass, isolated points get bigger windows first
    pending = _adaptive_window_pass(
        SAM3model, SAM3processor, device, image_array, centerpoints,
        pending, instance_map, bboxes, batch_size,
        local_scales=local_scales,
        depth_array=depth_array, depth_gray_norm=depth_gray_norm,
        sharpen=sharpen, sharpen_depth=sharpen_depth,
        bbox_sources=bbox_sources, candidates_ledger=candidates_ledger,
        far_depth_threshold=far_depth_threshold,
        is_far_or_violet=is_far_or_violet,
        near_miss_log=near_miss_log,
        orange_at_point=orange_at_point,
        nearby_count=nearby_count,
    )


    # halving shrink cascade down to MIN_SHRINK_WINDOW
    window = SEED_WINDOW
    first_shrink_done = False
    while pending and window >= MIN_SHRINK_WINDOW:
        before = len(pending)
        pending = _shrink_pass(
            SAM3model, SAM3processor, device, image_array, centerpoints,
            pending, window, instance_map, bboxes, batch_size,
            depth_array=depth_array, depth_gray_norm=depth_gray_norm,
            sharpen=sharpen, sharpen_depth=sharpen_depth,
            bbox_sources=bbox_sources, accept_any_iou=False,
            local_scales=local_scales, candidates_ledger=candidates_ledger,
            far_depth_threshold=far_depth_threshold,
            is_far_or_violet=is_far_or_violet,
            near_miss_log=near_miss_log,
            orange_at_point=orange_at_point,
            nearby_count=nearby_count,
        )
        logger.debug(f"shrink pass window={window}: {before - len(pending)} satisfied, "
                     f"{len(pending)} pending")

        # depth-only rescue runs once, right after the SEED_WINDOW pass
        if (enable_depth_rescue and not first_shrink_done
                and depth_array is not None and pending):
            _whole_image_depth_rescue(
                SAM3model, SAM3processor, device, image_array, centerpoints,
                instance_map, bboxes,
                depth_array=depth_array, depth_gray_norm=depth_gray_norm,
                bbox_sources=bbox_sources,
                sharpen=sharpen, sharpen_depth=sharpen_depth,
                local_scales=local_scales,
                far_depth_threshold=far_depth_threshold,
                is_far_or_violet=is_far_or_violet,
                near_miss_log=near_miss_log,
                nearby_count=nearby_count,
            )
            pending = {i for i in valid_indices if bboxes[i] is None}
        first_shrink_done = True
        window //= 2

    # final lenient pass with no IoU floor. large-scale points skip it, a 32 px
    # mask on a big fish is worse than the fallback square
    if pending:
        pending_for_lenient = {
            i for i in pending
            if local_scales is None or local_scales[i] is None
            or local_scales[i] <= LARGE_OBJECT_SKIP_LENIENT_SCALE
        }
        if pending_for_lenient:
            pending_for_lenient = _shrink_pass(
                SAM3model, SAM3processor, device, image_array, centerpoints,
                pending_for_lenient, MIN_SHRINK_WINDOW, instance_map, bboxes,
                batch_size,
                depth_array=depth_array, depth_gray_norm=depth_gray_norm,
                sharpen=sharpen, sharpen_depth=sharpen_depth,
                bbox_sources=bbox_sources, accept_any_iou=True,
                local_scales=local_scales, candidates_ledger=candidates_ledger,
                far_depth_threshold=far_depth_threshold,
                is_far_or_violet=is_far_or_violet,
                near_miss_log=near_miss_log,
                orange_at_point=orange_at_point,
                nearby_count=nearby_count,
            )
            # re-derive pending so skipped large-scale points stay in
            pending = {i for i in valid_indices if bboxes[i] is None}

    # ledger restore before the square fallback, top entry whose raw score
    # clears LEDGER_RESTORE_ABS_FLOOR
    if pending and candidates_ledger:
        restored = set()
        for idx in list(pending):
            cands = candidates_ledger.get(idx)
            if not cands:
                continue
            cx, cy = centerpoints[idx]
            # point-in-kept is re-checked here, entries come from any cascade
            cand = next(
                (c for c in cands
                 if float(c.get("score", 0.0)) >= LEDGER_RESTORE_ABS_FLOOR
                    and _point_in_kept(cx, cy, c["kept_xs_abs"], c["kept_ys_abs"])
                    and _candidate_depth_consistent(c, depth_gray_norm, cx, cy)),
                None,
            )
            if cand is None:
                continue
            origin = cand.get("cascade_origin", "")
            cascade_tag = (
                "whole_image_restored"
                if isinstance(origin, str) and origin.startswith("whole_image")
                else "best_of_cascades"
            )
            bboxes[idx] = list(cand["bbox"])
            instance_map[cand["kept_ys_abs"], cand["kept_xs_abs"]] = idx + 1
            if bbox_sources is not None:
                bbox_sources[idx] = {
                    "cascade": cascade_tag,
                    "modality": cand.get("source", "rgb"),
                    "sharpened": bool(cand.get("sharpened", False)),
                    "rgb_score": cand.get("rgb_score"),
                    "depth_score": cand.get("depth_score"),
                }
            restored.add(idx)
        pending -= restored

    # rescue: rerun whole_image + adaptive + shrink with halved floors and
    # relaxed gates
    if pending:
        rescue_pending = set(pending)
        _whole_image_pass(
            SAM3model, SAM3processor, device, image_array, centerpoints,
            instance_map, bboxes, bbox_sources=bbox_sources,
            depth_array=depth_array, depth_gray_norm=depth_gray_norm,
            sharpen=sharpen, sharpen_depth=sharpen_depth,
            local_scales=local_scales, candidates_ledger=None,
            far_depth_threshold=far_depth_threshold,
            is_far_or_violet=is_far_or_violet,
            near_miss_log=near_miss_log,
            rescue_mode=True,
            rescue_indices=rescue_pending,
            nearby_count=nearby_count,
            orange_at_point=orange_at_point,
        )
        pending = {i for i in rescue_pending if bboxes[i] is None}
        if pending:
            pending = _adaptive_window_pass(
                SAM3model, SAM3processor, device, image_array, centerpoints,
                pending, instance_map, bboxes, batch_size,
                local_scales=local_scales,
                depth_array=depth_array, depth_gray_norm=depth_gray_norm,
                sharpen=sharpen, sharpen_depth=sharpen_depth,
                bbox_sources=bbox_sources, candidates_ledger=None,
                far_depth_threshold=far_depth_threshold,
                is_far_or_violet=is_far_or_violet,
                near_miss_log=near_miss_log,
                orange_at_point=orange_at_point,
                rescue_mode=True,
                nearby_count=nearby_count,
            )
        rescue_window = SEED_WINDOW
        while pending and rescue_window >= MIN_SHRINK_WINDOW:
            pending = _shrink_pass(
                SAM3model, SAM3processor, device, image_array, centerpoints,
                pending, rescue_window, instance_map, bboxes, batch_size,
                depth_array=depth_array, depth_gray_norm=depth_gray_norm,
                sharpen=sharpen, sharpen_depth=sharpen_depth,
                bbox_sources=bbox_sources, accept_any_iou=False,
                local_scales=local_scales, candidates_ledger=None,
                far_depth_threshold=far_depth_threshold,
                is_far_or_violet=is_far_or_violet,
                near_miss_log=near_miss_log,
                orange_at_point=orange_at_point,
                rescue_mode=True,
                nearby_count=nearby_count,
            )
            rescue_window //= 2

    # square fallback so every valid point ends up with a bbox, side scales
    # with depth at the point
    fallback_default = max(15, min(image_width, image_height) // 40)
    for idx in pending:
        base_side = (
            local_scales[idx] if local_scales is not None and local_scales[idx] is not None
            else fallback_default
        )
        cx, cy = centerpoints[idx]
        depth_factor = _depth_size_factor(depth_gray_norm, cx, cy)
        side = int(round(float(base_side) * depth_factor * FALLBACK_SQUARE_SIZE_MULT))
        bboxes[idx] = _square_fallback_xyxy(cx, cy, side, image_width, image_height)
        if bbox_sources is not None:
            nm = near_miss_log.get(idx) if near_miss_log else None
            bbox_sources[idx] = {
                "cascade": "fallback_square",
                "modality": "none",
                "sharpened": False,
                "rgb_score": None,
                "depth_score": None,
                "near_miss_modality": (nm.get("modality") if nm else None),
                "near_miss_score": (nm.get("score") if nm else None),
                "near_miss_floor": (nm.get("floor") if nm else None),
                "near_miss_cascade": (nm.get("cascade") if nm else None),
                "near_miss_reason": (nm.get("reason") if nm else "no_attempts"),
            }

    return bboxes

def _apply_postprocessing_no_pad(centerpoints, bboxes, h, w, bbox_sources=None,
                                 instance_map=None):
    """Grow any cascade bbox whose annotation point ended up outside it.
    bbox_sources and instance_map are accepted but unused."""
    valid_idx = [i for i, (p, b) in enumerate(zip(centerpoints, bboxes))
                 if p is not None and b is not None]
    if not valid_idx:
        return bboxes

    v_points = [centerpoints[i] for i in valid_idx]
    v_bboxes = [
        (bboxes[i][0], bboxes[i][1],
         bboxes[i][2] - bboxes[i][0] + 1,
         bboxes[i][3] - bboxes[i][1] + 1)
        for i in valid_idx
    ]

    v_bboxes = _enforce_point_inside_bbox(v_points, v_bboxes, h, w)
    result = list(bboxes)
    for i, (bx, by, bw, bh) in zip(valid_idx, v_bboxes):
        result[i] = [bx, by, bx + bw - 1, by + bh - 1]
    return result



# run

def run_for_shard(rank, world_size, args):
    rank_prefix = f"[gpu{rank}/{world_size}] " if world_size > 1 else ""
    gpu_total_gb = None
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(rank)
        gpu_total_gb = torch.cuda.get_device_properties(rank).total_memory / (1024 ** 3)
        logger.info(f"{rank_prefix}CUDA device {rank}: {gpu_name} ({gpu_total_gb:.0f} GB)")
    else:
        device = "cpu"
        logger.warning(f"{rank_prefix}CUDA not available, falling back to CPU (very slow)")

    if args.batch_size is None:
        batch_size = _auto_batch_size(gpu_total_gb) if gpu_total_gb is not None else 8
        logger.info(f"{rank_prefix}Auto-selected batch_size={batch_size} for {gpu_total_gb:.0f} GB GPU "
                    f"(override with --batch_size)")
    else:
        batch_size = args.batch_size

    logger.info(f"{rank_prefix}Loading SAM3 model and processor from '{MODEL_ID}'...")
    model_load_start = time.perf_counter()
    SAM3model = Sam3TrackerModel.from_pretrained(MODEL_ID).to(device).eval()
    SAM3processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
    logger.info(f"{rank_prefix}Model loaded in {time.perf_counter() - model_load_start:.1f}s")

    annotation_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    dataset_total = len(annotation_files)
    if rank == 0:
        logger.info(f"{rank_prefix}Input images dir  : {IMAGES_DIR} ({dataset_total} images in dataset)")
        logger.info(f"{rank_prefix}Annotations dir   : {ANNOTATIONS_DIR}")
        logger.info(f"{rank_prefix}Output bbox dir   : {OUTPUT_DIR}")
        logger.info(f"{rank_prefix}Output segmap dir : {OUTPUT_SEGMAP_DIR}")
        logger.info(f"{rank_prefix}Batch size        : {batch_size}")
        if args.sharpen:
            logger.info(f"{rank_prefix}Sharpening        : on, all cascades "
                        f"(sigma={SHARPEN_SIGMA}, amount={SHARPEN_AMOUNT})")
    if args.start:
        annotation_files = annotation_files[args.start:]
    if args.limit is not None:
        annotation_files = annotation_files[:args.limit]
    total_files = len(annotation_files)

    shard_size = (total_files + world_size - 1) // world_size
    shard_start = rank * shard_size
    shard_end = min(total_files, shard_start + shard_size)
    my_files = annotation_files[shard_start:shard_end]
    shard_total = len(my_files)
    logger.info(
        f"{rank_prefix}Shard files [{shard_start}:{shard_end}] = {shard_total}/{total_files} "
        f"in cascade mode (whole-image + halving shrink)"
    )
    if args.overwrite:
        logger.info(f"{rank_prefix}--overwrite enabled: regenerating bbox XMLs and segmaps for all files")

    overall_start = time.perf_counter()
    files_done = 0
    files_skipped = 0
    for shard_index, xml_path in enumerate(my_files, start=1):
        global_index = shard_start + shard_index
        image_path = IMAGES_DIR / (xml_path.stem + ".jpg")
        output_xml_path = OUTPUT_DIR / xml_path.name
        segmap_path = OUTPUT_SEGMAP_DIR / f"{xml_path.stem}_segmap.png"
        if not image_path.exists():
            logger.warning(f"{rank_prefix}[{shard_index}/{shard_total}] {xml_path.name}: image not found, skipping")
            files_skipped += 1
            continue
        if not args.overwrite and output_xml_path.exists() and segmap_path.exists():
            logger.info(f"{rank_prefix}[{shard_index}/{shard_total}] {xml_path.name}: outputs exist, skipping (use --overwrite to redo)")
            files_skipped += 1
            continue

        file_start = time.perf_counter()
        image_array = np.array(Image.open(image_path).convert("RGB"))
        image_height, image_width = image_array.shape[:2]
        _, _, centerpoints = parse_centerpoints(xml_path)
        num_points = sum(p is not None for p in centerpoints)

        depth_array = None
        depth_gray_norm = None
        use_depth = args.rgbd or args.rgbdanddepth
        if use_depth:
            depth_array = _load_depth_array(xml_path.stem)
            if depth_array is None:
                logger.debug(f"{rank_prefix}{xml_path.name}: no depth file in {DEPTH_DIR}, falling back to RGB-only for this image")
            else:
                if depth_array.shape[:2] != (image_height, image_width):
                    depth_array = cv2.resize(depth_array, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
                # the depth-proxy tests need a monotonic-in-distance signal,
                # so load the grayscale depthmap separately
                gray_depth = _load_depth_gray_array(xml_path.stem)
                if gray_depth is not None and gray_depth.shape[:2] != (image_height, image_width):
                    gray_depth = cv2.resize(gray_depth, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
                depth_gray_norm = _normalize_depth_gray(
                    gray_depth if gray_depth is not None else depth_array
                )

        local_scales = _per_point_local_scale(centerpoints, image_height, image_width)
        depth_label = " +depth" if depth_array is not None else ""
        sharp_label = " +sharpen" if args.sharpen else ""
        if depth_array is not None and args.rgbdanddepth and args.sharpen:
            sharp_label += "(rgb+depth)"
        logger.info(
            f"{rank_prefix}[{shard_index}/{shard_total}] {xml_path.name} (global #{global_index}): "
            f"{num_points} points, cascade (batch={batch_size}{depth_label}{sharp_label})..."
        )

        # on CUDA OOM halve batch_size and retry, down to 1
        instance_map = np.zeros((image_height, image_width), dtype=np.uint16)
        bbox_sources = [None] * len(centerpoints)
        attempt_batch = batch_size
        bboxes = None
        while attempt_batch >= 1:
            try:
                instance_map.fill(0)
                for i in range(len(bbox_sources)):
                    bbox_sources[i] = None
                bboxes = segment_with_cascade(
                    SAM3model, SAM3processor, device, image_array, centerpoints, instance_map,
                    local_scales=local_scales, batch_size=attempt_batch,
                    depth_array=depth_array, depth_gray_norm=depth_gray_norm,
                    sharpen=args.sharpen,
                    sharpen_depth=args.sharpen and args.rgbdanddepth,
                    bbox_sources=bbox_sources,
                    enable_depth_rescue=args.whole_image_depth_rescue,
                )
                break
            except torch.cuda.OutOfMemoryError as exc:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt_batch <= 1:
                    logger.warning(
                        f"{rank_prefix}[{shard_index}/{shard_total}] {xml_path.name}: "
                        f"OOM at batch_size=1, skipping ({exc})"
                    )
                    break
                next_batch = max(1, attempt_batch // 2)
                logger.warning(
                    f"{rank_prefix}[{shard_index}/{shard_total}] {xml_path.name}: OOM at "
                    f"batch_size={attempt_batch}, retrying with batch_size={next_batch}"
                )
                attempt_batch = next_batch
            except Exception as exc:
                logger.warning(
                    f"{rank_prefix}[{shard_index}/{shard_total}] {xml_path.name}: cascade failed ({exc}), skipping"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                bboxes = None
                break

        if bboxes is None:
            files_skipped += 1
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bboxes = _apply_postprocessing_no_pad(
            centerpoints, bboxes, image_height, image_width, bbox_sources=bbox_sources,
            instance_map=instance_map,
        )
        segmap_bgr, instances_painted = render_segmap_colored(instance_map)
        cv2.imwrite(str(segmap_path), segmap_bgr)
        _write_bbox_xml_with_sources(xml_path, output_xml_path, bboxes, bbox_sources)

        successful = sum(b is not None for b in bboxes)
        elapsed = time.perf_counter() - file_start
        files_done += 1
        avg_per_file = (time.perf_counter() - overall_start) / max(files_done, 1)
        eta_seconds = avg_per_file * (shard_total - shard_index)
        logger.info(
            f"{rank_prefix}[{shard_index}/{shard_total}] {xml_path.name}: {successful}/{num_points} bboxes, "
            f"{instances_painted} instances in segmap in {elapsed:.1f}s | ETA {eta_seconds / 60:.1f}min"
        )

    overall_elapsed = time.perf_counter() - overall_start
    logger.info(
        f"{rank_prefix}Done. Processed {files_done} files, skipped {files_skipped}, "
        f"total time {overall_elapsed / 60:.1f}min"
    )


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 bbox annotation: whole-image pass + halving shrink cascade."
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rgbd",
        action="store_true",
        help="run SAM3 on the depth colormap too and fuse with RGB",
    )
    parser.add_argument(
        "--rgbdanddepth",
        action="store_true",
        help="SAM3 on RGB and depth, fused per point, winning modality in <bbox_source>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="crops per SAM3 forward (default: auto from GPU memory)",
    )
    parser.add_argument(
        "--sharpen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="unsharp-mask the cascade inputs (--no-sharpen to disable)",
    )
    parser.add_argument(
        "--whole_image_depth_rescue",
        action="store_true",
        help="depth-only whole-image rescue pass after the first shrink window",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SEGMAP_DIR.mkdir(parents=True, exist_ok=True)

    dataset_total = len(sorted(ANNOTATIONS_DIR.glob("*.xml")))
    will_process = dataset_total
    if args.start:
        will_process = max(0, will_process - args.start)
    if args.limit is not None:
        will_process = min(will_process, args.limit)
    logger.info(f"Input images dir  : {IMAGES_DIR} ({dataset_total} images in dataset)")
    logger.info(f"Annotations dir   : {ANNOTATIONS_DIR}")
    logger.info(f"Output bbox dir   : {OUTPUT_DIR}")
    logger.info(f"Output segmap dir : {OUTPUT_SEGMAP_DIR}")
    logger.info(f"Files to process  : {will_process}/{dataset_total} (--start={args.start}, --limit={args.limit})")
    logger.info(f"Batch size        : {args.batch_size if args.batch_size is not None else 'auto (per-GPU)'}")
    logger.info(
        f"Sharpening        : "
        + ("on (all cascades)" if args.sharpen else "off")
    )
    if args.rgbd:
        logger.info(f"RGBD fusion enabled, depth dir: {DEPTH_DIR}")
    if args.rgbdanddepth:
        logger.info(f"RGB+depth fusion (with depth-crop sharpening + provenance XML) enabled, depth dir: {DEPTH_DIR}")

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size <= 1:
        run_for_shard(0, 1, args)
    else:
        logger.info(
            f"Detected {world_size} visible GPUs, spawning one worker per GPU "
            f"(~{(will_process + world_size - 1) // world_size} files/GPU)"
        )
        mp.spawn(run_for_shard, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
