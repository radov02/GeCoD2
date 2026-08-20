"""Helpers and tuning constants for SAM3_hpc_annotation.py:
mask fusion, shrink cascade, bbox postprocessing."""
import argparse
import math
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env
load_env()
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # PVS model: points are already given

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sam3_annot")

DATASET_DIR = Path(__file__).parent / "dataset" / "IOCfish5k"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "point_annotations"
OUTPUT_DIR = DATASET_DIR / "bbox_annotations"
OUTPUT_SEGMAP_DIR = DATASET_DIR / "segment_maps"
# colorised depthmap, Spectral_r (near = red/orange, far = violet ~#5E4FA2)
DEPTH_DIR = Path(__file__).parent / "dataset" / "IOCfish5k-D" / "color"
# grayscale depthmap, bright = close, for per-point depth lookups
DEPTH_GRAY_DIR = Path(__file__).parent / "dataset" / "IOCfish5k-D" / "depthmaps"



# _whole_image_pass

GLOBAL_OBJECT_CHUNK = 64

# points at or below this local_scale (px) skip the whole-image pass
DENSE_CLUSTER_LOCAL_SCALE = 24

# higher whole-image floor in busy/textured regions
MIN_IOU_SCORE_WHOLE_TEXTURED = 0.45

TEXTURE_BUSY_STD = 0.10  # L-channel std (0..1) above which the region counts as busy

# whole-image RGB floor, big-fish masks score 0.20-0.30 at full scale even when good
MIN_IOU_SCORE_WHOLE = 0.36

# whole-image depth floor, crop-level MIN_IOU_SCORE_DEPTH is too strict here
MIN_IOU_SCORE_DEPTH_WHOLE = 0.25

# rgb*depth product floor for the rescue path
COMBINED_HADAMARD_THRESHOLD = 0.10

# _shrink_pass

# last attempt: no IoU floor
FINAL_IOU_FLOOR = 0.0

# per-window (rgb, depth) IoU floors
SHRINK_IOU_FLOORS = {
    256: (0.30, 0.40),
    128: (0.28, 0.38),
    64:  (0.26, 0.35),
    32:  (0.22, 0.30),
}

# reject masks filling most of the crop (painted-square failure)
CROP_FILL_REJECT_FRAC = 0.85

# prefer-larger-mask score fraction per window
SHRINK_LARGER_MASK_SCORE_FRAC = {
    256: 0.75,
    128: 0.65,
    64:  0.55,
    32:  0.45,
}

MAX_NEGATIVES_PER_WINDOW = 10

# cap co-pending positives per crop, dense scenes OOM without it
MAX_CO_PENDING_PER_CROP = MAX_NEGATIVES_PER_WINDOW

# per-window absolute negatives cap, applied after the density cap
SHRINK_MAX_NEGATIVES = {32: 5, 64: 5, 128: 8}

# rescue-mode w32/w64 hard caps
SHRINK_LAST_RESORT_WINDOWS = (32, 64)
LAST_RESORT_CROP_FILL_MAX = 0.99
LAST_RESORT_BOUNDARY_FILL_MAX = 0.95

# crop_filled relaxation for near-camera fish that really fill the crop
CROP_FILL_HIGH_CONF_SCORE = 0.85
CROP_FILL_HIGH_CONF_FRAC = 0.95

BOUNDARY_MARGIN_PIXELS = 2

# _whole_image_depth_rescue

RESCUE_FLOOR_DEFAULT = 0.32

RESCUE_AREA_LO = 0.25

RESCUE_AREA_HI = 6.0

# segment_with_cascade

FAR_DEPTH_PERCENTILE = 40.0

# whole-image bboxes scoring below this get demoted back to pending
MIN_WHOLE_IMAGE_KEEP_SCORE = 0.45

# below ~32 px SAM3's encoder is mostly noise
MIN_SHRINK_WINDOW = 32

# points with local_scale above this skip shrink_w32_lenient
LARGE_OBJECT_SKIP_LENIENT_SCALE = 64

# fallback-square size multiplier
FALLBACK_SQUARE_SIZE_MULT = 1.40

# run_for_shard

MODEL_ID = "facebook/sam3"

# drop speckle components below this fraction of the main component's area
OUTLIER_AREA_FRAC = 0.05

# bbox wider/taller than this fraction of the image becomes a point-centred square
MAX_BBOX_FRAC = 0.95
OVERSIZE_FALLBACK_FRAC = 0.15

# percentile-clip masks above this many px when the raw extent overshoots
PERCENTILE_CLIP_AREA_THR = 5000
PERCENTILE_CLIP_OUTLIER_TOL = 0.10

# postproc: duplicate bboxes
DUPLICATE_IOU_THR        = 0.70 # same-bbox threshold (swarm failure)
DUPLICATE_MIN_GROUP      = 3 # min near-identical bboxes to replace
# postproc: thin bboxes
THIN_BBOX_ASPECT_THR     = 5.0 # long/short ratio, above this -> square
# postproc: multipoint collapse
MULTIPOINT_SIZE_THR      = 3.0 # replace only if side > this x typical side
MULTIPOINT_MIN_PTS       = 25 # min points inside to replace
MULTIPOINT_OWN_CENTRED   = 0.4 # own-point exemption centring ratio
MULTIPOINT_LARGE_OBJ_THR = 5.0 # exemption only above this x typical side



# _finalize_mask

# rgb_score bias before fusion, large windows up, tiny windows down
CASCADE_RGB_BIAS = {
    "whole_image": 0.08,
    "whole_image_restored": 0.08,
    "shrink_w256": 0.05,
    "shrink_w128": 0.02,
    "shrink_w64": 0.0,
    "shrink_w32": -0.05,
    "shrink_w32_lenient": -0.08,
}

def _cascade_rgb_bias(cascade_label):
    """rgb_score bias for fusion, longest-prefix match on the label."""
    if not isinstance(cascade_label, str):
        return 0.0
    best_key = None
    for k in CASCADE_RGB_BIAS:
        if cascade_label.startswith(k):
            if best_key is None or len(k) > len(best_key):
                best_key = k
    return CASCADE_RGB_BIAS.get(best_key, 0.0) if best_key else 0.0


BOXY_MARGIN = 2
BOXY_FILL_FRAC = 0.6

def _mask_is_boxy(mask, margin=BOXY_MARGIN, fill_thr=BOXY_FILL_FRAC):
    """True if the mask sits flush against >= 2 sides of its frame."""
    if mask is None or mask.size == 0:
        return False
    h, w = mask.shape[:2]
    if h <= margin * 2 or w <= margin * 2:
        return False
    sides_filled = 0
    if mask[:margin].mean() >= fill_thr:
        sides_filled += 1
    if mask[-margin:].mean() >= fill_thr:
        sides_filled += 1
    if mask[:, :margin].mean() >= fill_thr:
        sides_filled += 1
    if mask[:, -margin:].mean() >= fill_thr:
        sides_filled += 1
    return sides_filled >= 2


def _mask_has_straight_cut(mask, min_edge_frac=0.40, straightness_tol=2.0):
    """True if the contour has a near-straight run covering >= min_edge_frac
    of the perimeter."""
    if mask is None or mask.size == 0 or not mask.any():
        return False
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    contour = max(contours, key=lambda c: c.shape[0])
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = pts.shape[0]
    if n < 20:
        return False
    perim = float(cv2.arcLength(contour, True))
    if perim <= 0:
        return False
    threshold_len = min_edge_frac * perim
    i = 0
    while i < n:
        j = i + 5
        best_j = i
        while j < n:
            ax, ay = pts[i]
            bx, by = pts[j % n]
            seg_dx, seg_dy = bx - ax, by - ay
            seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len2 <= 0:
                break
            seg_len = seg_len2 ** 0.5
            mid_pts = pts[i:j + 1]
            cross = np.abs(
                (mid_pts[:, 0] - ax) * seg_dy - (mid_pts[:, 1] - ay) * seg_dx
            ) / seg_len
            if cross.max() > straightness_tol:
                break
            best_j = j
            j += 5
        run_len = 0.0
        if best_j > i:
            run_pts = pts[i:best_j + 1]
            deltas = np.diff(run_pts, axis=0)
            run_len = float(np.sqrt((deltas * deltas).sum(axis=1)).sum())
        if run_len >= threshold_len:
            return True
        i = best_j + 1 if best_j > i else i + 5
    return False


def select_target_component(SAM_output_mask, local_x, local_y, negative_local_points):
    """Component at the target point plus components without a neighbor point
    inside. None if the mask is empty."""
    binary = SAM_output_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels <= 1:
        return None
    height, width = SAM_output_mask.shape
    target_label = 0
    if 0 <= local_y < height and 0 <= local_x < width:
        target_label = labels[local_y, local_x]
    if target_label == 0: # point on background, take the nearest foreground component
        ys, xs = np.where(SAM_output_mask)
        if len(xs) == 0:
            return None
        distances_squared = (xs - local_x) ** 2 + (ys - local_y) ** 2
        nearest = int(np.argmin(distances_squared))
        target_label = labels[ys[nearest], xs[nearest]]
    forbidden_labels = set()
    for neighbor_x, neighbor_y in negative_local_points:
        if 0 <= neighbor_y < height and 0 <= neighbor_x < width:
            neighbor_label = labels[neighbor_y, neighbor_x]
            if neighbor_label != 0 and neighbor_label != target_label:
                forbidden_labels.add(neighbor_label)
    kept_mask = np.zeros_like(SAM_output_mask, dtype=bool)
    for label_value in range(1, num_labels):
        if label_value == target_label or label_value not in forbidden_labels:
            kept_mask |= (labels == label_value)
    return kept_mask


VORONOI_MARGIN_PX = 28

def _voronoi_split_component(component, local_x, local_y, negatives,
                             margin_px=VORONOI_MARGIN_PX):
    """Drop pixels at least margin_px closer to a negative prompt than to the
    positive."""
    if component is None or not component.any() or not negatives:
        return component
    ys, xs = np.where(component)
    pos_d = np.sqrt((xs - local_x) ** 2 + (ys - local_y) ** 2)
    keep = np.ones(len(xs), dtype=bool)
    for nx, ny in negatives:
        neg_d = np.sqrt((xs - nx) ** 2 + (ys - ny) ** 2)
        keep &= pos_d <= neg_d + margin_px
    if keep.all():
        return component
    out = np.zeros_like(component)
    out[ys[keep], xs[keep]] = True
    return out


def _depth_proxy_at_point(depth_gray_norm, cx, cy):
    """Depth proxy in [0, 1] at the point (0 = far, 1 = close), 5x5 median patch."""
    h, w = depth_gray_norm.shape
    cx_c = int(np.clip(cx, 0, w - 1))
    cy_c = int(np.clip(cy, 0, h - 1))
    y0, y1 = max(0, cy_c - 2), min(h, cy_c + 3)
    x0, x1 = max(0, cx_c - 2), min(w, cx_c + 3)
    patch = depth_gray_norm[y0:y1, x0:x1]
    return float(np.median(patch)) if patch.size > 0 else 0.5


def _mask_is_background_grab(mask, point_xy, image_array,
                             chroma_gap=6.0,
                             texture_ratio=1.6,
                             point_radius=6):
    """True when a mask looks like a background grab: the point patch differs
    from the mask body in Lab chroma and is more textured."""
    if mask is None or image_array is None or image_array.ndim != 3 or point_xy is None:
        return False
    if not mask.any():
        return False
    h, w = mask.shape
    px, py = int(point_xy[0]), int(point_xy[1])
    if not (0 <= py < h and 0 <= px < w):
        return False
    if not mask[py, px]:
        return False
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l_ch = lab[:, :, 0].astype(np.float32) / 255.0
    ab = lab[:, :, 1:3].astype(np.float32)

    py0, py1 = max(0, py - point_radius), min(h, py + point_radius + 1)
    px0, px1 = max(0, px - point_radius), min(w, px + point_radius + 1)
    point_patch_ab = ab[py0:py1, px0:px1].reshape(-1, 2)
    point_patch_l = l_ch[py0:py1, px0:px1]
    if point_patch_ab.size == 0:
        return False
    pa = float(np.median(point_patch_ab[:, 0]))
    pb = float(np.median(point_patch_ab[:, 1]))
    point_l_std = float(np.std(point_patch_l))

    # mask interior minus the point patch
    interior_mask = mask.copy()
    interior_mask[py0:py1, px0:px1] = False
    if not interior_mask.any():
        return False
    body_ab = ab[interior_mask]
    if body_ab.size == 0:
        return False
    ba = float(np.median(body_ab[:, 0]))
    bb = float(np.median(body_ab[:, 1]))
    chroma_dist = ((pa - ba) ** 2 + (pb - bb) ** 2) ** 0.5
    if chroma_dist <= chroma_gap:
        return False

    body_l_std = float(np.std(l_ch[interior_mask])) + 1e-6
    if point_l_std < texture_ratio * body_l_std:
        return False
    return True


def _depth_local_contrast(depth_gray_norm, cx, cy, radius=24):
    """Std of normalised depth in a patch around (cx, cy)."""
    if depth_gray_norm is None:
        return 0.0
    h, w = depth_gray_norm.shape
    cx_c = int(np.clip(cx, 0, w - 1))
    cy_c = int(np.clip(cy, 0, h - 1))
    y0, y1 = max(0, cy_c - radius), min(h, cy_c + radius + 1)
    x0, x1 = max(0, cx_c - radius), min(w, cx_c + radius + 1)
    patch = depth_gray_norm[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.std(patch))


def _modality_distinctiveness(image_array, depth_gray_norm, cx, cy, radius=16):
    """(rgb_diff, depth_diff): contrast of each modality at (cx, cy) against
    its surrounding annulus."""
    if image_array is None and depth_gray_norm is None:
        return 0.0, 0.0
    src = image_array if image_array is not None else depth_gray_norm
    h, w = src.shape[:2]
    cx_c = int(np.clip(cx, 0, w - 1))
    cy_c = int(np.clip(cy, 0, h - 1))

    rgb_diff = 0.0
    if image_array is not None and image_array.ndim == 3:
        ann_lo, ann_hi = radius, int(radius * 2.0)
        y0, y1 = max(0, cy_c - ann_hi), min(h, cy_c + ann_hi + 1)
        x0, x1 = max(0, cx_c - ann_hi), min(w, cx_c + ann_hi + 1)
        rect = image_array[y0:y1, x0:x1]
        if rect.size > 0:
            lab = cv2.cvtColor(rect, cv2.COLOR_RGB2LAB).astype(np.float32)
            rh, rw = lab.shape[:2]
            ly, lx = cy_c - y0, cx_c - x0
            py0, py1 = max(0, ly - 3), min(rh, ly + 4)
            px0, px1 = max(0, lx - 3), min(rw, lx + 4)
            point_patch = lab[py0:py1, px0:px1].reshape(-1, 3)
            yy, xx = np.ogrid[:rh, :rw]
            mask_inner = (xx - lx) ** 2 + (yy - ly) ** 2 < ann_lo * ann_lo
            ann = lab[~mask_inner]
            if point_patch.size > 0 and ann.size > 0:
                pa = float(np.median(point_patch[:, 1]))
                pb = float(np.median(point_patch[:, 2]))
                aa = float(np.median(ann[:, 1]))
                ab = float(np.median(ann[:, 2]))
                rgb_diff = ((pa - aa) ** 2 + (pb - ab) ** 2) ** 0.5

    depth_diff = 0.0
    if depth_gray_norm is not None:
        ann_lo, ann_hi = radius, int(radius * 2.0)
        y0, y1 = max(0, cy_c - ann_hi), min(h, cy_c + ann_hi + 1)
        x0, x1 = max(0, cx_c - ann_hi), min(w, cx_c + ann_hi + 1)
        rect = depth_gray_norm[y0:y1, x0:x1]
        if rect.size > 0:
            rh, rw = rect.shape[:2]
            ly, lx = cy_c - y0, cx_c - x0
            py0, py1 = max(0, ly - 3), min(rh, ly + 4)
            px0, px1 = max(0, lx - 3), min(rw, lx + 4)
            point_patch = rect[py0:py1, px0:px1]
            yy, xx = np.ogrid[:rh, :rw]
            mask_inner = (xx - lx) ** 2 + (yy - ly) ** 2 < ann_lo * ann_lo
            ann = rect[~mask_inner]
            if point_patch.size > 0 and ann.size > 0:
                depth_diff = abs(float(np.median(point_patch)) - float(np.median(ann)))

    return float(rgb_diff), float(depth_diff)


# tol(ratio) = min(cap, base + slope * max(0, ratio - 1)), big masks get more slack
ADAPT_DEPTH_TOL_SLOPE = 0.025

def _adaptive_depth_tol(base, cap, mask, local_scale):
    """Scale a depth tolerance up with mask area relative to local_scale^2."""
    if local_scale is None or mask is None or not mask.any():
        return base
    ratio = float(mask.sum()) / max(float(local_scale) ** 2, 1.0)   # mask_area / focal_expected_area
    return min(cap, base + ADAPT_DEPTH_TOL_SLOPE * max(0.0, ratio - 1.0))


def _fuse_with_source(rgb_component, depth_component, local_x, local_y, point_depth_proxy,
                      rgb_score=None, depth_score=None,
                      depth_gray_norm_local=None,
                      force_rgb_preference=False):
    """_fuse_rgb_depth_components plus which modality won."""
    fused = _fuse_rgb_depth_components(
        rgb_component, depth_component, local_x, local_y, point_depth_proxy,
        rgb_score=rgb_score, depth_score=depth_score,
        depth_gray_norm_local=depth_gray_norm_local,
        force_rgb_preference=force_rgb_preference,
    )
    if fused is None:
        return None, None
    if rgb_component is not None and fused is rgb_component:
        return fused, "rgb"
    if depth_component is not None and fused is depth_component:
        return fused, "depth"
    return fused, "intersection"


MIN_MASK_PX = 10

def _fit_bbox_minimal_clean(component_mask, local_x=None, local_y=None,
                            depth_gray_norm_local=None,
                            depth_tolerance=0.10,
                            speckle_kernel=3,
                            speckle_min_px=MIN_MASK_PX,
                            centroid_dist_max=1.2,
                            area_frac_min=0.10):
    """Fit a bbox to the component mask, opening kills speckle.
    Returns ((xmin, ymin, xmax, ymax), kept_ys, kept_xs) or None."""
    binary = component_mask.astype(np.uint8)
    if binary.sum() < MIN_MASK_PX:
        return None
    if speckle_kernel and speckle_kernel >= 2:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (speckle_kernel, speckle_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        if binary.sum() < MIN_MASK_PX:
            return None
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None
    h, w = binary.shape
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    main_label = 0
    if (local_x is not None and local_y is not None
            and 0 <= int(local_y) < h and 0 <= int(local_x) < w):
        main_label = int(labels[int(local_y), int(local_x)])
    if main_label == 0 and num_labels >= 2:
        main_label = int(np.argmax(areas[1:])) + 1
    if main_label == 0:
        return None
    main_area = int(areas[main_label])
    if main_area < MIN_MASK_PX:
        return None

    main_depth_median = None
    if depth_gray_norm_local is not None and depth_gray_norm_local.shape == labels.shape:
        main_pixels = depth_gray_norm_local[labels == main_label]
        if main_pixels.size > 0:
            main_depth_median = float(np.median(main_pixels))

    main_radius = max(1.0, float(main_area) ** 0.5)
    point_xy = None
    if local_x is not None and local_y is not None:
        point_xy = (float(local_x), float(local_y))

    keep_mask = (labels == main_label)
    for lbl in range(1, num_labels):
        if lbl == main_label:
            continue
        comp_area = int(areas[lbl])
        if comp_area < speckle_min_px:
            continue
        if point_xy is not None:
            cx_lbl = float(centroids[lbl, 0])
            cy_lbl = float(centroids[lbl, 1])
            dist = ((cx_lbl - point_xy[0]) ** 2 + (cy_lbl - point_xy[1]) ** 2) ** 0.5
            if dist / main_radius > centroid_dist_max:
                continue
        if comp_area < area_frac_min * main_area:
            continue
        if main_depth_median is not None:
            comp_pixels = depth_gray_norm_local[labels == lbl]
            if comp_pixels.size == 0:
                continue
            comp_depth_median = float(np.median(comp_pixels))
            if abs(comp_depth_median - main_depth_median) <= depth_tolerance:
                keep_mask |= (labels == lbl)
            # different depth = background bleed
        else:
            keep_mask |= (labels == lbl)
    ys, xs = np.where(keep_mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), ys, xs


def _area_fitness(mask_area, local_scale):
    """1.0 when mask area matches local_scale**2, decays log-symmetrically."""
    if local_scale is None or local_scale <= 0:
        return 1.0
    expected = max(1.0, float(local_scale) ** 2)
    if mask_area <= 0:
        return 0.0
    log_ratio = abs(math.log(max(float(mask_area), 1.0) / expected))
    return 1.0 / (1.0 + log_ratio)


def _mask_is_at_point_depth(mask, depth_gray_norm_local, point_depth_proxy,
                            depth_diff_strict=0.18,
                            depth_diff_loose=0.10,
                            flat_std=0.02):
    """True when the mask sits at the annotation point's depth. Missing or
    empty inputs pass."""
    if mask is None or depth_gray_norm_local is None:
        return True
    if mask.shape != depth_gray_norm_local.shape:
        return True
    interior = depth_gray_norm_local[mask]
    if interior.size == 0:
        return True
    med = float(np.median(interior))
    std = float(np.std(interior))
    diff = abs(med - float(point_depth_proxy))
    if diff > depth_diff_strict:
        return False
    if std < flat_std and diff > depth_diff_loose:
        return False
    return True


# depth fusion constants (--rgbd)
RGBD_CENTERING_GOOD = 0.5
RGBD_CENTERING_DIFF = 0.15
RGBD_SIZE_DOMINANCE = 2.0
RGBD_IOU_AGREE_THR = 0.25
RGBD_FAR_SMALLER_RATIO = 0.8
FAR_DEPTH_THR = 0.30

def _fuse_rgb_depth_components(rgb_component, depth_component, local_x, local_y, point_depth_proxy,
                               rgb_score=None, depth_score=None,
                               depth_gray_norm_local=None,
                               force_rgb_preference=False):
    """Fuse the RGB and depth components into one mask, None if neither has one."""
    rgb_ok = rgb_component is not None and rgb_component.any() and int(rgb_component.sum()) >= MIN_MASK_PX
    depth_ok = depth_component is not None and depth_component.any() and int(depth_component.sum()) >= MIN_MASK_PX
    if rgb_ok and not depth_ok:
        return rgb_component
    if depth_ok and not rgb_ok:
        return depth_component
    if not rgb_ok and not depth_ok:
        return None

    rgb_center = _mask_bbox_centering_local(rgb_component, local_x, local_y)
    depth_center = _mask_bbox_centering_local(depth_component, local_x, local_y)
    rgb_has_pt = rgb_center < 1.0
    depth_has_pt = depth_center < 1.0

    # force RGB when depth is known unreliable here
    if force_rgb_preference and rgb_has_pt:
        return rgb_component

    # confidence-gap override, depth wins on a smaller gap
    CONFIDENCE_GAP_RGB_WINS = 0.15
    CONFIDENCE_GAP_DEPTH_WINS = 0.08
    if rgb_score is not None and depth_score is not None:
        gap = float(rgb_score) - float(depth_score)
        if gap >= CONFIDENCE_GAP_RGB_WINS and rgb_has_pt and rgb_center <= RGBD_CENTERING_GOOD:
            return rgb_component
        if -gap >= CONFIDENCE_GAP_DEPTH_WINS and depth_has_pt and depth_center <= RGBD_CENTERING_GOOD:
            return depth_component

    # near-camera point + weak RGB: prefer depth
    NEAR_CAM_THR = 0.70
    NEAR_CAM_RGB_WEAK = 0.55
    if (rgb_score is not None and depth_score is not None
            and point_depth_proxy >= NEAR_CAM_THR
            and rgb_score < NEAR_CAM_RGB_WEAK
            and depth_has_pt
            and depth_score >= rgb_score - 0.10):
        return depth_component

    # depth much larger than RGB: RGB probably clipped to the head
    DEPTH_AREA_DOMINANCE_HARD = 4.0
    if rgb_has_pt and depth_has_pt:
        rgb_area_hard = float(rgb_component.sum())
        depth_area_hard = float(depth_component.sum())
        if (rgb_area_hard > 0
                and depth_area_hard >= DEPTH_AREA_DOMINANCE_HARD * rgb_area_hard):
            return depth_component

    # depth larger and not much weaker: depth got the whole fish, RGB a fragment
    DEPTH_AREA_DOMINANCE = 1.5
    DEPTH_SCORE_NEAR_GAP = 0.05
    if (rgb_score is not None and depth_score is not None
            and rgb_has_pt and depth_has_pt):
        rgb_area_q = float(rgb_component.sum())
        depth_area_q = float(depth_component.sum())
        if (depth_area_q >= rgb_area_q * DEPTH_AREA_DOMINANCE
                and depth_score >= rgb_score - DEPTH_SCORE_NEAR_GAP):
            return depth_component

    # weak RGB + comparable point-containing depth: prefer depth
    RGB_WEAK_THR = 0.50
    RGB_WEAK_DEPTH_GAP = 0.04
    if (rgb_score is not None and depth_score is not None
            and rgb_score < RGB_WEAK_THR
            and depth_has_pt
            and depth_score >= rgb_score - RGB_WEAK_DEPTH_GAP):
        return depth_component

    if point_depth_proxy <= FAR_DEPTH_THR and rgb_has_pt and depth_has_pt:
        rgb_area = float(rgb_component.sum())
        depth_area = float(depth_component.sum())
        if depth_area <= rgb_area * RGBD_FAR_SMALLER_RATIO:
            return depth_component
        if rgb_area <= depth_area * RGBD_FAR_SMALLER_RATIO:
            return rgb_component

    if rgb_has_pt and not depth_has_pt:
        return rgb_component
    if depth_has_pt and not rgb_has_pt:
        return depth_component

    if rgb_has_pt and depth_has_pt:
        rgb_area = float(rgb_component.sum())
        depth_area = float(depth_component.sum())
        if rgb_center <= RGBD_CENTERING_GOOD and depth_center <= RGBD_CENTERING_GOOD:
            if depth_area > rgb_area * RGBD_SIZE_DOMINANCE:
                return depth_component
            if rgb_area > depth_area * RGBD_SIZE_DOMINANCE:
                return rgb_component
        if depth_center + RGBD_CENTERING_DIFF < rgb_center:
            return depth_component
        if rgb_center + RGBD_CENTERING_DIFF < depth_center:
            return rgb_component

    # mode-depth match: further-from-point median depth loses
    MODE_DIFF_GAP = 0.15
    if depth_gray_norm_local is not None and depth_gray_norm_local.shape == rgb_component.shape:
        try:
            rgb_mode = float(np.median(depth_gray_norm_local[rgb_component]))
            depth_mode = float(np.median(depth_gray_norm_local[depth_component]))
            if abs(rgb_mode - depth_mode) >= MODE_DIFF_GAP:
                rgb_to_pt = abs(rgb_mode - point_depth_proxy)
                depth_to_pt = abs(depth_mode - point_depth_proxy)
                if rgb_to_pt + 0.05 < depth_to_pt:
                    return rgb_component
                if depth_to_pt + 0.05 < rgb_to_pt:
                    return depth_component
        except Exception:
            pass

    # intersection, guarded against tiny clipped slivers
    MIN_INTERSECTION_AREA_RATIO = 0.55
    intersection = rgb_component & depth_component
    union = rgb_component | depth_component
    union_px = int(union.sum())
    rgb_px = int(rgb_component.sum())
    depth_px = int(depth_component.sum())
    if union_px > 0:
        iou = float(intersection.sum()) / float(union_px)
        inter_px = int(intersection.sum())
        sliver = inter_px < MIN_INTERSECTION_AREA_RATIO * float(min(rgb_px, depth_px))
        if iou >= RGBD_IOU_AGREE_THR and inter_px >= MIN_MASK_PX and not sliver:
            ys_i, xs_i = np.where(intersection)
            if len(xs_i) > 0:
                if (int(xs_i.min()) <= local_x <= int(xs_i.max())
                        and int(ys_i.min()) <= local_y <= int(ys_i.max())):
                    return intersection
        # sliver: prefer the higher-confidence modality if any
        if sliver and rgb_score is not None and depth_score is not None:
            if float(rgb_score) >= float(depth_score):
                return rgb_component
            return depth_component

    return rgb_component


def _mask_bbox_centering_local(mask, local_x, local_y):
    """Distance of the point from the mask's bbox centre, normalised
    (0 = centred, >1 = outside, inf = mask too small)."""
    if mask is None:
        return float("inf")
    ys, xs = np.where(mask)
    if len(xs) < MIN_MASK_PX:
        return float("inf")
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    half_w = max(1.0, (xmax - xmin) / 2.0)
    half_h = max(1.0, (ymax - ymin) / 2.0)
    cxb = (xmin + xmax) / 2.0
    cyb = (ymin + ymax) / 2.0
    dx = (float(local_x) - cxb) / half_w
    dy = (float(local_y) - cyb) / half_h
    return float((dx * dx + dy * dy) ** 0.5)






# _shrink_pass

PAD_LABEL = -1
PAD_COORD = [0, 0]

# prefer a larger mask when its score is close and it contains the point
LARGER_MASK_SCORE_FRAC = 0.80

def _run_sam3_batch_multi_obj(SAM3model, SAM3processor, device,
                              images_pil, points_per_image, labels_per_image,
                              allow_point_near=False,
                              score_frac=LARGER_MASK_SCORE_FRAC):
    """Multi-image multi-object SAM3 forward with prefer-larger-mask selection.
    Returns per image (mask_stack_bool [n_objs, H, W] on CPU, scores)."""
    n_objs_per_img = [len(p) for p in points_per_image]
    max_objs = max(n_objs_per_img)
    max_pts = max(max(len(po) for po in p) for p in points_per_image)
    padded_points, padded_labels = [], []
    for img_pts, img_lbls in zip(points_per_image, labels_per_image):
        out_p, out_l = [], []
        for opts, olbls in zip(img_pts, img_lbls):
            out_p.append(opts + [PAD_COORD] * (max_pts - len(opts)))
            out_l.append(olbls + [PAD_LABEL] * (max_pts - len(olbls)))
        # pad missing objects so batched tensors are uniform
        while len(out_p) < max_objs:
            out_p.append([PAD_COORD] * max_pts)
            out_l.append([PAD_LABEL] * max_pts)
        padded_points.append(out_p)
        padded_labels.append(out_l)

    inputs = SAM3processor(
        images=images_pil,
        input_points=padded_points,
        input_labels=padded_labels,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = SAM3model(**inputs, multimask_output=True)
        else:
            outputs = SAM3model(**inputs, multimask_output=True)
        pred_masks_bool = (outputs.pred_masks > 0).cpu()
        iou_scores = outputs.iou_scores.float().cpu().numpy()  # (B, max_objs, num_masks)
    original_sizes = inputs["original_sizes"]
    del outputs, inputs
    post_processed_list = SAM3processor.post_process_masks(
        pred_masks_bool.float(), original_sizes
    )

    results = []
    for b, n in enumerate(n_objs_per_img):
        masks_b = post_processed_list[b] # (max_objs, num_masks, H_b, W_b)
        scores_b = iou_scores[b] # (max_objs, num_masks)
        masks_b_bool = masks_b > 0
        chosen_masks = []
        chosen_scores = []
        for obj in range(n):
            # first positive point per object = its focal
            target_xy = None
            for pt, lbl in zip(padded_points[b][obj], padded_labels[b][obj]):
                if lbl == 1:
                    target_xy = (pt[0], pt[1])
                    break
            chosen = _pick_larger_mask_idx(
                masks_b_bool[obj], scores_b[obj], target_xy,
                score_frac=score_frac,
                allow_point_near=allow_point_near,
            )
            chosen_masks.append(masks_b_bool[obj, chosen])
            chosen_scores.append(float(scores_b[obj, chosen]))
        chosen_stack = torch.stack(chosen_masks, dim=0) if chosen_masks else torch.zeros((0, 0, 0), dtype=torch.bool)
        results.append((chosen_stack, chosen_scores))
    return results


def _pick_larger_mask_idx(masks_bool, scores, target_xy,
                          score_frac=LARGER_MASK_SCORE_FRAC,
                          allow_point_near=False,
                          near_tol_px=12):
    """Largest hypothesis scoring >= score_frac * best that contains the target
    point, argmax score if none qualify."""
    num_masks = masks_bool.shape[0]
    best_idx = int(np.asarray(scores).argmax())
    if num_masks <= 1 or target_xy is None:
        return best_idx
    best_score = float(scores[best_idx])
    areas = masks_bool.reshape(num_masks, -1).sum(dim=1).numpy().astype(np.int64)
    order = sorted(range(num_masks), key=lambda m: -int(areas[m]))
    tx, ty = int(target_xy[0]), int(target_xy[1])
    h_m, w_m = masks_bool.shape[1], masks_bool.shape[2]

    qualifying = []
    for m in order:
        if scores[m] < best_score * score_frac:
            continue
        if int(areas[m]) < MIN_MASK_PX:
            continue
        if not (0 <= ty < h_m and 0 <= tx < w_m):
            continue
        mask_np = masks_bool[m].numpy()
        ys_m, xs_m = np.where(mask_np)
        if len(xs_m) == 0:
            continue
        contains = (int(xs_m.min()) <= tx <= int(xs_m.max())
                    and int(ys_m.min()) <= ty <= int(ys_m.max()))
        if not contains and allow_point_near:
            inv = (~mask_np).astype(np.uint8)
            dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
            if dist[ty, tx] <= near_tol_px and int(areas[m]) >= 500:
                contains = True
        if not contains:
            continue
        qualifying.append(m)

    if not qualifying:
        return best_idx
    # smoothness tiebreak among candidates within 0.03 of the best qualifier
    top_score = max(float(scores[m]) for m in qualifying)
    near_top = [m for m in qualifying if float(scores[m]) >= top_score - 0.03]
    if len(near_top) == 1:
        return near_top[0]
    best_compact = None
    chosen = near_top[0]
    for m in near_top:
        mask_np = masks_bool[m].numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        perim = float(sum(cv2.arcLength(c, True) for c in contours))
        area = float(max(int(areas[m]), 1))
        compact = perim * perim / area
        if best_compact is None or compact < best_compact:
            best_compact = compact
            chosen = m
    return chosen



# _whole_image_pass

# unsharp-mask defaults
SHARPEN_SIGMA = 1.0
SHARPEN_AMOUNT = 0.6

def _unsharp_mask(image_uint8, sigma=SHARPEN_SIGMA, amount=SHARPEN_AMOUNT):
    """Standard unsharp mask: sharpened = image + amount*(image - blur(image))."""
    blurred = cv2.GaussianBlur(image_uint8, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(image_uint8, 1.0 + amount, blurred, -amount, 0)


def _build_object_prompts(target_index, valid_indices, points_array):
    target = points_array[target_index]
    others = np.array([i for i in valid_indices if i != target_index], dtype=np.int64)
    if len(others) == 0:
        return [[int(target[0]), int(target[1])]], [1], []
    diffs = points_array[others] - target
    distances_squared = (diffs ** 2).sum(axis=1)
    sorted_other_indices = others[np.argsort(distances_squared)][:MAX_NEGATIVES_PER_WINDOW]
    negatives = [(int(points_array[i, 0]), int(points_array[i, 1])) for i in sorted_other_indices]
    object_points = [[int(target[0]), int(target[1])]] + [list(n) for n in negatives]
    object_labels = [1] + [0] * len(negatives)
    return object_points, object_labels, negatives


# per-point negatives cap: dense points get more, sparse fewer
MAX_NEGATIVES_PER_OBJECT_TIGHT = 6 # fallback when local_scale is unknown
DENSE_NEG_LOCAL_SCALE = 24
SPARSE_NEG_LOCAL_SCALE = 64
MAX_NEGS_DENSE = 10
MAX_NEGS_SPARSE = 4

NEARBY_COUNT_DENSE = 4

def _negatives_cap_for_point(local_scale, nearby_count=None):
    """Negatives cap from the neighbour count, or local_scale interpolation."""
    if nearby_count is not None:
        return MAX_NEGS_DENSE if int(nearby_count) >= NEARBY_COUNT_DENSE else MAX_NEGS_SPARSE
    if local_scale is None:
        return MAX_NEGATIVES_PER_OBJECT_TIGHT
    locsc = float(local_scale)
    if locsc <= DENSE_NEG_LOCAL_SCALE:
        return MAX_NEGS_DENSE
    if locsc >= SPARSE_NEG_LOCAL_SCALE:
        return MAX_NEGS_SPARSE
    t = (locsc - DENSE_NEG_LOCAL_SCALE) / (SPARSE_NEG_LOCAL_SCALE - DENSE_NEG_LOCAL_SCALE)  # interpolation fraction in [0,1] from dense to sparse
    return int(round(MAX_NEGS_DENSE + t * (MAX_NEGS_SPARSE - MAX_NEGS_DENSE)))


def _run_sam3_forward(SAM3model, SAM3processor, device, image_pil, input_points, input_labels,
                      prefer_larger_mask=True):
    """Forward + multimask selection, returns (selected_masks, selected_scores)
    per object."""
    inputs = SAM3processor(
        images=image_pil,
        input_points=input_points,  # input_points is of shape [bs=1, n_objects, n_points_per_object, 2]
        input_labels=input_labels,  # [1, n_objects, n_points_per_object], 1 = positive, 0 = negative, -1 = padding
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = SAM3model(**inputs, multimask_output=True)
        else:
            outputs = SAM3model(**inputs, multimask_output=True)

        # upscale low-res masks to original image size and threshold to binary
        post_processed = SAM3processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"]
        )[0]

    iou_scores_attr = getattr(outputs, "iou_scores", None)
    num_objects, num_masks = post_processed.shape[0], post_processed.shape[1]
    if iou_scores_attr is not None:
        iou_per_object = iou_scores_attr[0].float().cpu().numpy()
    else:
        iou_per_object = np.ones((num_objects, num_masks), dtype=np.float32)

    selected_masks = []
    selected_scores = []
    binary_all = (post_processed > 0) if prefer_larger_mask and num_masks > 1 else None
    if binary_all is not None:
        areas_all = binary_all.sum(dim=(-1, -2)).numpy().astype(np.int64)

    for obj in range(num_objects):
        scores_for_obj = iou_per_object[obj]
        best_idx = int(scores_for_obj.argmax())
        best_score = float(scores_for_obj[best_idx])
        chosen = best_idx

        if prefer_larger_mask and num_masks > 1:
            target_x = int(input_points[0][obj][0][0])
            target_y = int(input_points[0][obj][0][1])
            target_label = int(input_labels[0][obj][0])
            if target_label == 1:   # require containment only for a positive focal point
                order = sorted(range(num_masks), key=lambda m: -int(areas_all[obj, m]))
                for m in order:
                    if scores_for_obj[m] < best_score * LARGER_MASK_SCORE_FRAC:
                        continue
                    if int(areas_all[obj, m]) < MIN_MASK_PX:
                        continue
                    mask_bool = binary_all[obj, m].numpy()
                    h_m, w_m = mask_bool.shape
                    if not (0 <= target_y < h_m and 0 <= target_x < w_m):
                        continue
                    ys_m, xs_m = np.where(mask_bool)
                    if len(xs_m) == 0:
                        continue
                    if (int(xs_m.min()) <= target_x <= int(xs_m.max())
                            and int(ys_m.min()) <= target_y <= int(ys_m.max())):
                        chosen = m
                        break

        selected_masks.append(post_processed[obj, chosen])
        selected_scores.append(float(scores_for_obj[chosen]))
    return selected_masks, selected_scores


def _local_texture_std(image_array, cx, cy, radius=8):
    """Std of the Lab L channel in a small patch around the point."""
    if image_array is None or image_array.ndim != 3:
        return 0.0
    h, w = image_array.shape[:2]
    cx_c = int(np.clip(cx, 0, w - 1))
    cy_c = int(np.clip(cy, 0, h - 1))
    y0, y1 = max(0, cy_c - radius), min(h, cy_c + radius + 1)
    x0, x1 = max(0, cx_c - radius), min(w, cx_c + radius + 1)
    rect = image_array[y0:y1, x0:x1]
    if rect.size == 0:
        return 0.0
    lab = cv2.cvtColor(rect, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32) / 255.0
    return float(np.std(l_channel))


def _record_near_miss(near_miss_log, idx, modality, score, floor, cascade, reason):
    """Track the best score/floor ratio seen per point."""
    if near_miss_log is None:
        return
    if score is None or floor is None or floor <= 0:
        return
    ratio = float(score) / float(floor)
    existing = near_miss_log.get(idx)
    if existing is None or ratio > existing.get("ratio", 0.0):
        near_miss_log[idx] = {
            "modality": modality,
            "score": float(score),
            "floor": float(floor),
            "cascade": cascade,
            "reason": reason,
            "ratio": ratio,
        }


MIN_IOU_SCORE = 0.3
# depth floor higher than RGB, low-IoU depth masks are usually leaks
MIN_IOU_SCORE_DEPTH = 0.45

# fractional padding around the fitted bbox (segmap painting unaffected)
BBOX_PAD_FRAC = 0.05

# above this score skip the Voronoi backstop
HIGH_CONF_NO_VORONOI = 0.75
# above this depth score prefer the depth mask outright
HIGH_CONF_DEPTH_OVERRIDE = 0.85
# depth tolerance for the mode-agreement check guarding the override
DEPTH_MODE_AGREE_TOL = 0.20
# tolerance for keeping a disconnected component on depth grounds
DEPTH_COMPONENT_KEEP_TOL = 0.10

# margin when a negative point sits inside the chosen component
VORONOI_MARGIN_PX_INSIDE_NEG = 12
# straight-cut rejection, skipped when SAM3 was confident
STRAIGHT_CUT_EDGE_FRAC = 0.40
STRAIGHT_CUT_TOL_PX = 2.0

# low local depth contrast scales the depth score down
DEPTH_FLAT_LOW = 0.04 # std of depth_gray_norm below this = flat
DEPTH_FLAT_MED = 0.10 # no penalty above this
DEPTH_FLAT_PENALTY_MIN = 0.40 # multiplier floor when depth is fully flat
RGB_DIFF_STRONG = 8.0 # Lab chroma diff above this = clear RGB signal
DEPTH_DIFF_FLAT = 0.05 # depth proxy diff below this = no depth signal
# depth proxy near-identical to its annulus: force RGB
DEPTH_INDISTINCT_FORCE_RGB = 0.025

# flat-interior depth masks are almost always background grabs
DEPTH_MASK_INTERIOR_FLAT_STD = 0.025
DEPTH_MASK_INTERIOR_FLAT_PENALTY = 0.5

# boost the depth score when the body is clearly distinct from its annulus
DEPTH_DISTINCT_BOUNDARY_DILATE_PX = 6
DEPTH_DISTINCT_GAP_STRONG = 0.20
DEPTH_DISTINCT_BOOST = 1.25
# inverse: same-depth body gets penalised
DEPTH_DISTINCT_GAP_WEAK = 0.05
DEPTH_INDISTINCT_BOUNDARY_PENALTY = 0.6

# most mask pixels must sit within spread_tol of the median depth
DEPTH_INTERIOR_SPREAD_TOL = 0.18
DEPTH_INTERIOR_SPREAD_TOL_CAP = 0.36
DEPTH_INTERIOR_SPREAD_FRAC = 0.75

# far-mask reject: big masks in the bluest 40% are usually water-column grabs
FAR_MASK_GRACE_FACTOR = 3.0

ADAPT_DEPTH_KEEP_CAP = 0.20 # base 0.10 (DEPTH_COMPONENT_KEEP_TOL)
ADAPT_DEPTH_STRICT_CAP = 0.28 # base 0.18 (_mask_is_at_point_depth)
ADAPT_DEPTH_AGREE_CAP = 0.30 # base 0.20 (DEPTH_MODE_AGREE_TOL)

# absolute score floor for restoring a ledger attempt before fallback_square
LEDGER_RESTORE_ABS_FLOOR = 0.25
# top-K ledger candidates per point, ranked by score * area_fitness
LEDGER_TOP_K = 3

# area floor for the intersection retry, relative to the smaller input mask
INTERSECTION_RETRY_AREA_FRAC = 0.40

# reject only the "SAM3 painted the entire image" failure
MASK_FULL_IMAGE_REJECT_FRAC = 0.90

# grow the intersection back into matching-depth RGB pixels (thin fins)
INTERSECTION_GROW_PX = 12
INTERSECTION_GROW_DEPTH_TOL = 0.18

# depth-seed guard for depth_plus_rgb_ext
DEPTH_SEED_MAX_AREA_RATIO = 4.0
# depth filling this much of the crop = whole-layer grab
DEPTH_SEED_MAX_CROP_FRAC = 0.65

# intersections with more far/violet pixels than this are background
INTERSECTION_MAX_FAR_VIOLET_FRAC = 0.35

# whole-image far/violet gate, small/medium objects only
WHOLE_IMAGE_MAX_FAR_VIOLET_FRAC = 0.40
WHOLE_IMAGE_FAR_VIOLET_MIN_LOCAL_SCALE = 56

# near-camera points get an easier distinctness gap and a stronger boost
NEAR_CAM_DEPTH_PROXY = 0.72
DEPTH_DISTINCT_GAP_NEAR_CAM = 0.10
DEPTH_DISTINCT_BOOST_NEAR_CAM = 1.50

# depth-clean swap: depth has a larger negative-free body than the RGB pick
DEPTH_CLEAN_AREA_DOMINANCE = 1.5
DEPTH_CLEAN_SHRINK_CASCADES = (
    "shrink_w128", "shrink_w64", "shrink_w32",
    "shrink_w128_rescue", "shrink_w64_rescue", "shrink_w32_rescue",
)

def _finalize_mask(rgb_mask, lx, ly, negs, image_w, image_h,
                   x_off, y_off, idx, instance_map, bboxes,
                   depth_mask=None, depth_gray_norm=None, point_xy=None,
                   bbox_sources=None, cascade_label="unknown", sharpened=False,
                   rgb_score=None, depth_score=None,
                   image_array=None,
                   local_scale=None,
                   candidates_ledger=None,
                   far_depth_threshold=None,
                   is_far_or_violet=None,
                   prefer_depth_hadamard=False,
                   near_miss_log=None,
                   rescue_mode=False,
                   orange_at_point_focal=False):
    """Turn one SAM3 RGB (+ optional depth) mask pair into a bbox, True when
    one was produced. Masks are in crop space, (x_off, y_off) maps to the image.
    On success paints instance_map with idx + 1 and fills bboxes[idx]."""
    rgb_confident = rgb_score is not None and rgb_score >= HIGH_CONF_NO_VORONOI
    depth_confident = depth_score is not None and depth_score >= HIGH_CONF_NO_VORONOI
    rgb_score_for_fusion = (
        float(rgb_score) + _cascade_rgb_bias(cascade_label)
        if rgb_score is not None else None
    )
    # the Voronoi backstop only helps in the shrink cascade
    is_whole_image = isinstance(cascade_label, str) and cascade_label.startswith("whole_image")

    # big fish fill most of a crop, relax the boxy threshold
    big_fish_scale = (
        local_scale is not None and local_scale > min(image_h, image_w) / 8
    )
    boxy_fill_thr = 0.80 if big_fish_scale else None  # None -> use default (0.6)

    def _reject_raw(mask, confident):
        if mask is None:
            return True
        if boxy_fill_thr is not None:
            if _mask_is_boxy(mask, fill_thr=boxy_fill_thr):
                return True
        elif _mask_is_boxy(mask):
            return True
        if not confident and _mask_has_straight_cut(
            mask,
            min_edge_frac=STRAIGHT_CUT_EDGE_FRAC,
            straightness_tol=STRAIGHT_CUT_TOL_PX,
        ):
            return True
        return False

    if _reject_raw(rgb_mask, rgb_confident):
        rgb_component = None
    else:
        rgb_component = select_target_component(rgb_mask, lx, ly, negs)
        if not rgb_confident and not is_whole_image:
            rgb_component = _voronoi_split_component(rgb_component, lx, ly, negs)
        # drop whole-image RGB masks that look like background grabs
        if (is_whole_image and rgb_component is not None
                and rgb_component.any() and image_array is not None
                and point_xy is not None):
            full_pt = (int(point_xy[0]), int(point_xy[1]))
            if _mask_is_background_grab(rgb_component, full_pt, image_array):
                rgb_component = None

    depth_component = None
    if depth_mask is not None:
        if _reject_raw(depth_mask, depth_confident):
            depth_component = None
        else:
            depth_component = select_target_component(depth_mask, lx, ly, negs)
            if not depth_confident and not is_whole_image:
                depth_component = _voronoi_split_component(depth_component, lx, ly, negs)

    point_depth = (
        _depth_proxy_at_point(depth_gray_norm, *point_xy)
        if (depth_gray_norm is not None and point_xy is not None) else 0.5
    )

    depth_contrast = (
        _depth_local_contrast(depth_gray_norm, *point_xy)
        if (depth_gray_norm is not None and point_xy is not None) else 0.0
    )
    rgb_diff, depth_diff = (0.0, 0.0)
    if point_xy is not None:
        rgb_diff, depth_diff = _modality_distinctiveness(
            image_array, depth_gray_norm, point_xy[0], point_xy[1],
        )
    if depth_contrast >= DEPTH_FLAT_MED:
        depth_penalty = 1.0
    elif depth_contrast <= DEPTH_FLAT_LOW:
        depth_penalty = DEPTH_FLAT_PENALTY_MIN
    else:
        # linear interp between FLAT_LOW (penalty MIN) and FLAT_MED (penalty 1.0)
        t = (depth_contrast - DEPTH_FLAT_LOW) / max(1e-6, DEPTH_FLAT_MED - DEPTH_FLAT_LOW)
        depth_penalty = DEPTH_FLAT_PENALTY_MIN + (1.0 - DEPTH_FLAT_PENALTY_MIN) * t
    effective_depth_score = (
        float(depth_score) * depth_penalty if depth_score is not None else None
    )

    # flat depth + clear RGB chroma: force RGB
    force_rgb = False
    if (depth_diff < DEPTH_DIFF_FLAT and rgb_diff > RGB_DIFF_STRONG
            and rgb_component is not None and rgb_component.any()):
        force_rgb = True
        depth_component = None
    # depth_diff too small for depth to be segmenting anything here
    if (not force_rgb
            and depth_diff < DEPTH_INDISTINCT_FORCE_RGB
            and rgb_component is not None and rgb_component.any()):
        ch_r, cw_r = rgb_component.shape
        if (0 <= int(ly) < ch_r and 0 <= int(lx) < cw_r
                and rgb_component[int(ly), int(lx)]):
            force_rgb = True
            depth_component = None

    # same-shape depth slice for the mode-depth and component-keep tests
    depth_local_for_fusion = None
    ref_component = rgb_component if rgb_component is not None else depth_component
    if depth_gray_norm is not None and ref_component is not None:
        ch, cw = ref_component.shape
        gh, gw = depth_gray_norm.shape
        if (0 <= y_off and y_off + ch <= gh and 0 <= x_off and x_off + cw <= gw):
            depth_local_for_fusion = depth_gray_norm[y_off:y_off + ch, x_off:x_off + cw]

    # drop depth masks straddling multiple depth layers
    if (depth_component is not None and depth_component.any()
            and depth_local_for_fusion is not None
            and depth_local_for_fusion.shape == depth_component.shape):
        interior = depth_local_for_fusion[depth_component]
        if interior.size > 0:
            spread_tol = _adaptive_depth_tol(
                DEPTH_INTERIOR_SPREAD_TOL, DEPTH_INTERIOR_SPREAD_TOL_CAP,
                depth_component, local_scale,
            )
            med = float(np.median(interior))
            frac_inside = float(np.mean(np.abs(interior - med) <= spread_tol))
            if frac_inside < DEPTH_INTERIOR_SPREAD_FRAC:
                depth_component = None
                effective_depth_score = None

    # flat interior = background grab, scale the score down so RGB wins fusion
    if (depth_component is not None and depth_component.any()
            and depth_local_for_fusion is not None
            and effective_depth_score is not None
            and depth_local_for_fusion.shape == depth_component.shape):
        interior = depth_local_for_fusion[depth_component]
        if interior.size > 0 and float(np.std(interior)) < DEPTH_MASK_INTERIOR_FLAT_STD:
            effective_depth_score *= DEPTH_MASK_INTERIOR_FLAT_PENALTY

    # a body clearly distinct from a thin annulus boosts the effective score
    force_depth_strong_distinct = False
    if (depth_component is not None and depth_component.any()
            and effective_depth_score is not None
            and depth_local_for_fusion is not None
            and depth_local_for_fusion.shape == depth_component.shape):
        kk = 2 * DEPTH_DISTINCT_BOUNDARY_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
        dilated = cv2.dilate(depth_component.astype(np.uint8), kernel) > 0
        boundary = dilated & ~depth_component
        if boundary.any():
            interior_med = float(np.median(depth_local_for_fusion[depth_component]))
            boundary_med = float(np.median(depth_local_for_fusion[boundary]))
            gap = abs(interior_med - boundary_med)
            near_cam = point_depth >= NEAR_CAM_DEPTH_PROXY
            gap_thr = (DEPTH_DISTINCT_GAP_NEAR_CAM if near_cam
                       else DEPTH_DISTINCT_GAP_STRONG)
            boost = (DEPTH_DISTINCT_BOOST_NEAR_CAM if near_cam
                     else DEPTH_DISTINCT_BOOST)
            if gap >= gap_thr:
                effective_depth_score = min(
                    1.0, effective_depth_score * boost
                )
            elif gap < DEPTH_DISTINCT_GAP_WEAK:
                # same depth as the surroundings, penalise so RGB wins fusion
                effective_depth_score *= DEPTH_INDISTINCT_BOUNDARY_PENALTY
            # near-cam plus the strict gap, force depth to win after fusion
            if near_cam and gap >= DEPTH_DISTINCT_GAP_STRONG:
                force_depth_strong_distinct = True
            # orange focal at a small cascade, the easier near-cam gap suffices
            small_cascade = isinstance(cascade_label, str) and any(
                cascade_label.startswith(c) for c in
                ("shrink_w32", "shrink_w64", "shrink_w128",
                 "shrink_w32_rescue", "shrink_w64_rescue", "shrink_w128_rescue")
            )
            if (small_cascade and orange_at_point_focal
                    and gap >= DEPTH_DISTINCT_GAP_NEAR_CAM):
                force_depth_strong_distinct = True

    # high-confidence depth override, mode agreement rejects a
    # confidently-wrong mask at a different depth
    if (depth_component is not None and depth_component.any()
            and effective_depth_score is not None
            and effective_depth_score >= HIGH_CONF_DEPTH_OVERRIDE
            and not force_rgb
            and depth_gray_norm is not None):
        crop_h, crop_w = depth_component.shape
        gh, gw = depth_gray_norm.shape
        dy0 = max(0, y_off);  dy1 = min(gh, y_off + crop_h)
        dx0 = max(0, x_off);  dx1 = min(gw, x_off + crop_w)
        if dy1 > dy0 and dx1 > dx0:
            depth_local = depth_gray_norm[dy0:dy1, dx0:dx1]
            local_h, local_w = depth_local.shape
            sub = depth_component[:local_h, :local_w]
            inside = depth_local[sub]
            if inside.size > 0:
                mode_d = float(np.median(inside))
                # a larger depth mask is allowed a bigger depth gap
                agree_tol = _adaptive_depth_tol(
                    DEPTH_MODE_AGREE_TOL, ADAPT_DEPTH_AGREE_CAP,
                    depth_component, local_scale,
                )
                if abs(mode_d - point_depth) <= agree_tol:
                    component = depth_component
                    source = "depth"
                else:
                    component, source = _fuse_with_source(
                        rgb_component, depth_component, lx, ly, point_depth,
                        rgb_score=rgb_score_for_fusion, depth_score=effective_depth_score,
                        depth_gray_norm_local=depth_local_for_fusion,
                        force_rgb_preference=force_rgb,
                    )
            else:
                component, source = _fuse_with_source(
                    rgb_component, depth_component, lx, ly, point_depth,
                    rgb_score=rgb_score_for_fusion, depth_score=effective_depth_score,
                    depth_gray_norm_local=depth_local_for_fusion,
                    force_rgb_preference=force_rgb,
                )
        else:
            component, source = _fuse_with_source(
                rgb_component, depth_component, lx, ly, point_depth,
                rgb_score=rgb_score_for_fusion, depth_score=effective_depth_score,
                depth_gray_norm_local=depth_local_for_fusion,
                force_rgb_preference=force_rgb,
            )
    elif depth_mask is not None:
        component, source = _fuse_with_source(
            rgb_component, depth_component, lx, ly, point_depth,
            rgb_score=rgb_score_for_fusion, depth_score=effective_depth_score,
            depth_gray_norm_local=depth_local_for_fusion,
            force_rgb_preference=force_rgb,
        )
    else:
        component = rgb_component
        source = "rgb" if component is not None else None

    if component is None or not component.any():
        if rgb_score is not None:
            _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                              MIN_IOU_SCORE, cascade_label, "no_component")
        if depth_score is not None:
            _record_near_miss(near_miss_log, idx, "depth", depth_score,
                              MIN_IOU_SCORE_DEPTH, cascade_label, "no_component")
        return False

    # hadamard rescue, take depth when it contains the point
    if (prefer_depth_hadamard
            and depth_component is not None and depth_component.any()):
        dh, dw = depth_component.shape
        if (0 <= int(ly) < dh and 0 <= int(lx) < dw
                and depth_component[int(ly), int(lx)]):
            component = depth_component
            source = "depth_hadamard"

    # near-cam plus strict boundary gap, force depth
    if (force_depth_strong_distinct
            and depth_component is not None and depth_component.any()):
        dh, dw = depth_component.shape
        if (0 <= int(ly) < dh and 0 <= int(lx) < dw
                and depth_component[int(ly), int(lx)]):
            component = depth_component
            source = "depth_strong_distinct"

    # depth has a larger negative-free body, RGB likely grabbed a sub-pattern
    if (source == "rgb"
            and depth_component is not None and depth_component.any()
            and rgb_component is not None and rgb_component.any()
            and isinstance(cascade_label, str)
            and any(cascade_label.startswith(c) for c in DEPTH_CLEAN_SHRINK_CASCADES)):
        dh, dw = depth_component.shape
        if (0 <= int(ly) < dh and 0 <= int(lx) < dw
                and depth_component[int(ly), int(lx)]):
            any_neg_inside = False
            for nx, ny in negs:
                nyi, nxi = int(ny), int(nx)
                if (0 <= nyi < dh and 0 <= nxi < dw
                        and depth_component[nyi, nxi]):
                    any_neg_inside = True
                    break
            if not any_neg_inside:
                d_area = float(depth_component.sum())
                r_area = float(rgb_component.sum())
                if d_area >= DEPTH_CLEAN_AREA_DOMINANCE * max(1.0, r_area):
                    component = depth_component
                    source = "depth_clean_swap"

    # depth swallowed a negative point (glued the focal to its neighbour),
    # swap back to RGB
    if (source in ("depth", "depth_hadamard", "depth_strong_distinct",
                   "depth_plus_rgb_ext", "depth_clean_swap", "intersection")
            and depth_component is not None and depth_component.any()
            and rgb_component is not None and rgb_component.any()
            and negs):
        dh, dw = depth_component.shape
        neg_inside = False
        for nx, ny in negs:
            nyi, nxi = int(ny), int(nx)
            if (0 <= nyi < dh and 0 <= nxi < dw
                    and depth_component[nyi, nxi]):
                neg_inside = True
                break
        if neg_inside:
            rh, rw = rgb_component.shape
            if (0 <= int(ly) < rh and 0 <= int(lx) < rw
                    and rgb_component[int(ly), int(lx)]):
                component = rgb_component
                source = "rgb_neg_aware"

    # intersection retry at whole_image / shrink_w256, seeded from depth and
    # grown back into matching-depth RGB pixels (trailing fins)
    if (source in ("rgb", "depth")
            and rgb_component is not None and rgb_component.any()
            and depth_component is not None and depth_component.any()
            and isinstance(cascade_label, str)
            and (cascade_label.startswith("whole_image")
                 or cascade_label.startswith("shrink_w256"))):
        ch_r, cw_r = rgb_component.shape
        ch_d, cw_d = depth_component.shape
        if (ch_r == ch_d and cw_r == cw_d
                and 0 <= int(ly) < ch_r and 0 <= int(lx) < cw_r
                and depth_component[int(ly), int(lx)]
                and rgb_component[int(ly), int(lx)]):
            # oversized depth seed or one swallowing a negative falls back to
            # the rgb & depth seed
            depth_seed_ok = True
            if local_scale is not None and local_scale > 0:
                if float(depth_component.sum()) > DEPTH_SEED_MAX_AREA_RATIO * float(local_scale) ** 2:
                    depth_seed_ok = False
            # depth filling >65% of the crop = whole-layer grab
            if depth_seed_ok:
                crop_area = max(1, ch_d * cw_d)
                if float(depth_component.sum()) / float(crop_area) > DEPTH_SEED_MAX_CROP_FRAC:
                    depth_seed_ok = False
            if depth_seed_ok and negs:
                for nx, ny in negs:
                    nyi, nxi = int(ny), int(nx)
                    if (0 <= nyi < ch_d and 0 <= nxi < cw_d
                            and depth_component[nyi, nxi]):
                        depth_seed_ok = False
                        break

            retry_source_tag = "depth_plus_rgb_ext"
            inter_seed = None
            if depth_seed_ok:
                inter_seed = depth_component
            else:
                fallback_seed = rgb_component & depth_component
                if (fallback_seed.any()
                        and fallback_seed[int(ly), int(lx)]):
                    inter_seed = fallback_seed
                    retry_source_tag = "intersection"
                # else no retry, the earlier source stays

            if inter_seed is not None:
                inter = inter_seed
                if INTERSECTION_GROW_PX > 0:
                    k = 2 * INTERSECTION_GROW_PX + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    grow_region = cv2.dilate(
                        inter_seed.astype(np.uint8), kernel
                    ).astype(bool)
                    candidate_ext = rgb_component & grow_region & ~inter_seed
                    if (candidate_ext.any()
                            and depth_local_for_fusion is not None
                            and depth_local_for_fusion.shape == inter_seed.shape):
                        seed_depths = depth_local_for_fusion[inter_seed]
                        if seed_depths.size > 0:
                            seed_med = float(np.median(seed_depths))
                            depth_ok = (
                                np.abs(depth_local_for_fusion - seed_med)
                                <= INTERSECTION_GROW_DEPTH_TOL
                            )
                            candidate_ext = candidate_ext & depth_ok
                    if candidate_ext.any():
                        inter = inter_seed | candidate_ext
                inter_area = float(inter.sum())
                agree_area = float((rgb_component & depth_component).sum())
                smaller = min(float(rgb_component.sum()), float(depth_component.sum()))
                if smaller > 0 and agree_area / smaller >= INTERSECTION_RETRY_AREA_FRAC:
                    accept = True
                    if (is_far_or_violet is not None
                            and inter_area > 0):
                        ih, iw = inter.shape
                        fy0 = max(0, y_off);  fy1 = min(image_h, y_off + ih)
                        fx0 = max(0, x_off);  fx1 = min(image_w, x_off + iw)
                        if fy1 > fy0 and fx1 > fx0:
                            sub_inter = inter[:fy1 - fy0, :fx1 - fx0]
                            sub_far = is_far_or_violet[fy0:fy1, fx0:fx1]
                            sub_area = float(sub_inter.sum())
                            if sub_area > 0:
                                far_frac = float(sub_far[sub_inter].sum()) / sub_area
                                if far_frac > INTERSECTION_MAX_FAR_VIOLET_FRAC:
                                    accept = False
                    if accept:
                        component = inter
                        source = retry_source_tag

    # a negative inside the component means SAM3 glued two points together,
    # split against those negatives with a tighter margin
    if negs:
        ch, cw = component.shape
        inside_negs = [
            (nx, ny) for (nx, ny) in negs
            if 0 <= int(ny) < ch
            and 0 <= int(nx) < cw
            and component[int(ny), int(nx)]
        ]
        if inside_negs:
            component = _voronoi_split_component(
                component, lx, ly, inside_negs,
                margin_px=VORONOI_MARGIN_PX_INSIDE_NEG,
            )
            if component is None or not component.any():
                if rgb_score is not None:
                    _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                                      MIN_IOU_SCORE, cascade_label, "voronoi_emptied")
                if depth_score is not None:
                    _record_near_miss(near_miss_log, idx, "depth", depth_score,
                                      MIN_IOU_SCORE_DEPTH, cascade_label, "voronoi_emptied")
                return False

    # a whole_image mask mostly in the far/violet band grabbed background,
    # small objects only
    if (is_whole_image
            and is_far_or_violet is not None
            and component is not None and component.any()
            and local_scale is not None
            and local_scale <= WHOLE_IMAGE_FAR_VIOLET_MIN_LOCAL_SCALE
            and not rescue_mode):
        ch, cw = component.shape
        fy0 = max(0, y_off);  fy1 = min(image_h, y_off + ch)
        fx0 = max(0, x_off);  fx1 = min(image_w, x_off + cw)
        if fy1 > fy0 and fx1 > fx0:
            sub_comp = component[:fy1 - fy0, :fx1 - fx0]
            sub_far = is_far_or_violet[fy0:fy1, fx0:fx1]
            sub_area = float(sub_comp.sum())
            if sub_area > 0:
                far_frac = float(sub_far[sub_comp].sum()) / sub_area
                if far_frac > WHOLE_IMAGE_MAX_FAR_VIOLET_FRAC:
                    if rgb_score is not None:
                        _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                                          MIN_IOU_SCORE, cascade_label,
                                          "whole_image_far_violet")
                    if depth_score is not None:
                        _record_near_miss(near_miss_log, idx, "depth", depth_score,
                                          MIN_IOU_SCORE_DEPTH, cascade_label,
                                          "whole_image_far_violet")
                    return False

    # same-shape depth crop for the bbox fitter
    depth_local_for_fit = None
    if depth_gray_norm is not None:
        ch, cw = component.shape
        gh, gw = depth_gray_norm.shape
        if (0 <= y_off and y_off + ch <= gh and 0 <= x_off and x_off + cw <= gw):
            depth_local_for_fit = depth_gray_norm[y_off:y_off + ch, x_off:x_off + cw]
    # bigger components get a bigger keep tolerance (long fish span more depth)
    keep_tol = _adaptive_depth_tol(
        DEPTH_COMPONENT_KEEP_TOL, ADAPT_DEPTH_KEEP_CAP,
        component, local_scale,
    )
    clip_result = _fit_bbox_minimal_clean(
        component, lx, ly,
        depth_gray_norm_local=depth_local_for_fit,
        depth_tolerance=keep_tol,
    )
    if clip_result is None:
        if rgb_score is not None:
            _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                              MIN_IOU_SCORE, cascade_label, "bbox_fit_failed")
        if depth_score is not None:
            _record_near_miss(near_miss_log, idx, "depth", depth_score,
                              MIN_IOU_SCORE_DEPTH, cascade_label, "bbox_fit_failed")
        return False
    (bx_min, by_min, bx_max, by_max), kept_ys, kept_xs = clip_result
    bw = bx_max - bx_min + 1
    bh = by_max - by_min + 1
    # pad by BBOX_PAD_FRAC in image coords, edges stay flush
    pad_x = int(round(bw * BBOX_PAD_FRAC))
    pad_y = int(round(bh * BBOX_PAD_FRAC))
    abs_xmin = max(0, bx_min + x_off - pad_x)
    abs_ymin = max(0, by_min + y_off - pad_y)
    abs_xmax = min(image_w - 1, bx_max + x_off + pad_x)
    abs_ymax = min(image_h - 1, by_max + y_off + pad_y)
    bw_padded = abs_xmax - abs_xmin + 1
    bh_padded = abs_ymax - abs_ymin + 1

    # ledger stores attempts before the gates, top-K by score * area_fitness
    if (candidates_ledger is not None
            and bw_padded * bh_padded <= MASK_FULL_IMAGE_REJECT_FRAC * image_w * image_h):
        best_score = max(
            float(rgb_score) if rgb_score is not None else 0.0,
            float(depth_score) if depth_score is not None else 0.0,
        )
        # candidate mask must (nearly) contain the point, crop coords
        if (best_score >= LEDGER_RESTORE_ABS_FLOOR
                and _point_in_kept(lx, ly, kept_xs, kept_ys)):
            mask_area = int(len(kept_ys))
            fitness = _area_fitness(mask_area, local_scale)
            combined = best_score * fitness
            slot = candidates_ledger.setdefault(idx, [])
            slot.append({
                "bbox": [abs_xmin, abs_ymin, abs_xmax, abs_ymax],
                "kept_ys_abs": kept_ys + y_off,
                "kept_xs_abs": kept_xs + x_off,
                "score": best_score,
                "combined": combined,
                "area": mask_area,
                "source": source or "rgb",
                "sharpened": bool(sharpened),
                "rgb_score": float(rgb_score) if rgb_score is not None else None,
                "depth_score": float(depth_score) if depth_score is not None else None,
                "cascade_origin": cascade_label,
            })
            # depth-only entry too, when it contains the point and no negatives
            if (depth_component is not None and depth_component.any()
                    and depth_score is not None
                    and float(depth_score) >= LEDGER_RESTORE_ABS_FLOOR):
                dh_alt, dw_alt = depth_component.shape
                if (0 <= int(ly) < dh_alt and 0 <= int(lx) < dw_alt
                        and depth_component[int(ly), int(lx)]):
                    any_neg_alt = False
                    for nx, ny in negs:
                        nyi, nxi = int(ny), int(nx)
                        if (0 <= nyi < dh_alt and 0 <= nxi < dw_alt
                                and depth_component[nyi, nxi]):
                            any_neg_alt = True
                            break
                    if not any_neg_alt:
                        depth_clip_alt = _fit_bbox_minimal_clean(
                            depth_component, lx, ly,
                            depth_gray_norm_local=depth_local_for_fit,
                            depth_tolerance=keep_tol,
                        )
                        if depth_clip_alt is not None:
                            (dxa, dya, dxb, dyb), d_kept_ys, d_kept_xs = depth_clip_alt
                            d_pad_x = int(round((dxb - dxa + 1) * BBOX_PAD_FRAC))
                            d_pad_y = int(round((dyb - dya + 1) * BBOX_PAD_FRAC))
                            d_abs_xmin = max(0, dxa + x_off - d_pad_x)
                            d_abs_ymin = max(0, dya + y_off - d_pad_y)
                            d_abs_xmax = min(image_w - 1, dxb + x_off + d_pad_x)
                            d_abs_ymax = min(image_h - 1, dyb + y_off + d_pad_y)
                            d_bw = d_abs_xmax - d_abs_xmin + 1
                            d_bh = d_abs_ymax - d_abs_ymin + 1
                            if (d_bw * d_bh <= MASK_FULL_IMAGE_REJECT_FRAC * image_w * image_h
                                    and _point_in_kept(lx, ly, d_kept_xs, d_kept_ys)):
                                d_area_p = int(len(d_kept_ys))
                                d_fitness = _area_fitness(d_area_p, local_scale)
                                d_combined = float(depth_score) * d_fitness
                                slot.append({
                                    "bbox": [d_abs_xmin, d_abs_ymin, d_abs_xmax, d_abs_ymax],
                                    "kept_ys_abs": d_kept_ys + y_off,
                                    "kept_xs_abs": d_kept_xs + x_off,
                                    "score": float(depth_score),
                                    "combined": d_combined,
                                    "area": d_area_p,
                                    "source": "depth_alt",
                                    "sharpened": bool(sharpened),
                                    "rgb_score": float(rgb_score) if rgb_score is not None else None,
                                    "depth_score": float(depth_score),
                                    "cascade_origin": cascade_label,
                                })
            slot.sort(key=lambda c: -c["combined"])
            if len(slot) > LEDGER_TOP_K:
                del slot[LEDGER_TOP_K:]

    # reject only near-full-image masks, tighter caps kill legit big fish
    if bw_padded * bh_padded > MASK_FULL_IMAGE_REJECT_FRAC * image_w * image_h:
        if rgb_score is not None:
            _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                              MIN_IOU_SCORE, cascade_label, "mask_full_image")
        if depth_score is not None:
            _record_near_miss(near_miss_log, idx, "depth", depth_score,
                              MIN_IOU_SCORE_DEPTH, cascade_label, "mask_full_image")
        return False

    # bigger than the fallback square and in the bluest 40% is usually a
    # water-column grab (within the grace band only if the point isn't far)
    if (far_depth_threshold is not None and local_scale is not None
            and depth_local_for_fit is not None
            and component is not None and component.any()
            and component.shape == depth_local_for_fit.shape):
        fallback_area = float(local_scale) ** 2
        mask_area = float(component.sum())
        if mask_area > fallback_area:
            interior_depth = depth_local_for_fit[component]
            if interior_depth.size > 0:
                mask_mean_depth = float(interior_depth.mean())
                if mask_mean_depth <= float(far_depth_threshold):
                    if mask_area > FAR_MASK_GRACE_FACTOR * fallback_area:
                        if rgb_score is not None:
                            _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                                              MIN_IOU_SCORE, cascade_label, "far_mask")
                        if depth_score is not None:
                            _record_near_miss(near_miss_log, idx, "depth", depth_score,
                                              MIN_IOU_SCORE_DEPTH, cascade_label, "far_mask")
                        return False
                    point_in_far = point_depth <= float(far_depth_threshold)
                    if not point_in_far:
                        if rgb_score is not None:
                            _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                                              MIN_IOU_SCORE, cascade_label, "far_mask")
                        if depth_score is not None:
                            _record_near_miss(near_miss_log, idx, "depth", depth_score,
                                              MIN_IOU_SCORE_DEPTH, cascade_label, "far_mask")
                        return False

    # w32 masks on far-water points are usually background scraps, keep pending
    if (isinstance(cascade_label, str)
            and cascade_label.startswith("shrink_w32")
            and far_depth_threshold is not None
            and point_depth <= float(far_depth_threshold)):
        if rgb_score is not None:
            _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                              MIN_IOU_SCORE, cascade_label, "w32_far_point")
        if depth_score is not None:
            _record_near_miss(near_miss_log, idx, "depth", depth_score,
                              MIN_IOU_SCORE_DEPTH, cascade_label, "w32_far_point")
        return False

    # whole-image masks at a different depth than the point get re-queued for
    # the shrink cascade
    if (is_whole_image and depth_local_for_fit is not None
            and component.shape == depth_local_for_fit.shape
            and not rescue_mode):
        strict_tol = _adaptive_depth_tol(
            0.18, ADAPT_DEPTH_STRICT_CAP, component, local_scale,
        )
        if not _mask_is_at_point_depth(
            component, depth_local_for_fit, point_depth,
            depth_diff_strict=strict_tol,
        ):
            if rgb_score is not None:
                _record_near_miss(near_miss_log, idx, "rgb", rgb_score,
                                  MIN_IOU_SCORE, cascade_label, "depth_mode_mismatch")
            if depth_score is not None:
                _record_near_miss(near_miss_log, idx, "depth", depth_score,
                                  MIN_IOU_SCORE_DEPTH, cascade_label, "depth_mode_mismatch")
            return False

    instance_map[kept_ys + y_off, kept_xs + x_off] = idx + 1
    bboxes[idx] = [abs_xmin, abs_ymin, abs_xmax, abs_ymax]
    if bbox_sources is not None:
        bbox_sources[idx] = {
            "cascade": cascade_label,
            "modality": source or "rgb",
            "sharpened": bool(sharpened),
            "rgb_score": float(rgb_score) if rgb_score is not None else None,
            "depth_score": float(depth_score) if depth_score is not None else None,
        }
    return True



# _apply_postprocessing_no_pad helpers

def _enforce_point_inside_bbox(points, bboxes, h, w):
    """Expand each bbox until it contains its annotation point (never shrinks)."""
    out = []
    for (cx, cy), (bx, by, bw, bh) in zip(points, bboxes):
        cx_c = int(np.clip(cx, 0, w - 1))
        cy_c = int(np.clip(cy, 0, h - 1))
        if cx_c < bx:
            bw += bx - cx_c;  bx = cx_c
        if cy_c < by:
            bh += by - cy_c;  by = cy_c
        if cx_c >= bx + bw:
            bw = cx_c - bx + 1
        if cy_c >= by + bh:
            bh = cy_c - by + 1
        bx = max(0, bx);  by = max(0, by)
        bw = max(1, min(bw, w - bx));  bh = max(1, min(bh, h - by))
        out.append((bx, by, bw, bh))
    return out



# segment_with_cascade helpers

SEED_WINDOW = 256   # size of the initial window/crop in cascade


def _square_fallback_xyxy(cx, cy, side, image_w, image_h):
    """Square box centred on (cx, cy) clamped to image bounds, as [xmin, ymin, xmax, ymax]."""
    side = max(8, int(side))
    half = side // 2
    cx_c = int(np.clip(cx, 0, image_w - 1))
    cy_c = int(np.clip(cy, 0, image_h - 1))
    bx = max(0, cx_c - half)
    by = max(0, cy_c - half)
    bw = min(side, image_w - bx)
    bh = min(side, image_h - by)
    return [bx, by, bx + bw - 1, by + bh - 1]


# fallback square scales with depth, nearer = larger, clamped
DEPTH_NEAR_SIZE_GAIN = 1.20
DEPTH_NEAR_SIZE_FACTOR_MIN = 0.80
DEPTH_NEAR_SIZE_FACTOR_MAX = 1.80

def _depth_size_factor(depth_gray_norm, cx, cy):
    """Fallback-square size multiplier in [MIN, MAX], 1.0 at depth proxy 0.5."""
    if depth_gray_norm is None:
        return 1.0
    proxy = _depth_proxy_at_point(depth_gray_norm, cx, cy)
    factor = 1.0 + DEPTH_NEAR_SIZE_GAIN * (float(proxy) - 0.5)
    return float(np.clip(factor, DEPTH_NEAR_SIZE_FACTOR_MIN, DEPTH_NEAR_SIZE_FACTOR_MAX))


# depth tolerance for the ledger restore, drops background blobs near the point
DEPTH_RESTORE_AGREE_TOL = 0.18

def _candidate_depth_consistent(cand, depth_gray_norm, cx, cy,
                                tol=DEPTH_RESTORE_AGREE_TOL):
    """True when the candidate's kept-pixel median depth is within tol of the point's."""
    if depth_gray_norm is None:
        return True
    kept_xs = cand.get("kept_xs_abs")
    kept_ys = cand.get("kept_ys_abs")
    if kept_xs is None or kept_ys is None or len(kept_xs) == 0:
        return True
    h, w = depth_gray_norm.shape
    xs = np.clip(kept_xs.astype(np.int64), 0, w - 1)
    ys = np.clip(kept_ys.astype(np.int64), 0, h - 1)
    mask_med = float(np.median(depth_gray_norm[ys, xs]))
    point_depth = _depth_proxy_at_point(depth_gray_norm, cx, cy)
    return abs(mask_med - point_depth) <= float(tol)


# 5 px tolerates the 3-px opening hole but kills the nearest-blob fallback
LEDGER_POINT_TOL_PX = 5

def _point_in_kept(lx, ly, kept_xs, kept_ys, tol_px=LEDGER_POINT_TOL_PX):
    """True when (lx, ly) is at most tol_px from any kept-mask pixel."""
    if kept_xs is None or len(kept_xs) == 0:
        return False
    lxi, lyi = int(lx), int(ly)
    if (lxi < int(kept_xs.min()) - tol_px or lxi > int(kept_xs.max()) + tol_px
        or lyi < int(kept_ys.min()) - tol_px or lyi > int(kept_ys.max()) + tol_px):
        return False
    dxs = kept_xs.astype(np.int64) - lxi
    dys = kept_ys.astype(np.int64) - lyi
    return int((dxs * dxs + dys * dys).min()) <= tol_px * tol_px


# dense = NEARBY_COUNT_DENSE+ points within this factor x median NN distance
NEARBY_RADIUS_FACTOR = 2.5

def _build_nearby_count(centerpoints):
    """Per-point neighbour count within a moderate radius, None entries get 0."""
    n = len(centerpoints)
    out = [0] * n
    valid_idx = [i for i, p in enumerate(centerpoints) if p is not None]
    if len(valid_idx) < 2:
        return out
    from scipy.spatial import cKDTree
    pts = np.array([centerpoints[i] for i in valid_idx], dtype=np.float64)
    tree = cKDTree(pts)
    nn_dists, _ = tree.query(pts, k=2)
    median_nn = float(np.median(nn_dists[:, 1]))
    radius = max(40.0, NEARBY_RADIUS_FACTOR * median_nn)
    counts = np.asarray(
        tree.query_ball_point(pts, r=radius, return_length=True)
    ) - 1  # exclude self
    for i, vi in enumerate(valid_idx):
        out[vi] = int(counts[i])
    return out


# orange depth color + bright proxy = large near-camera object, routed to
# bigger windows
ORANGE_DEPTH_REF_RGB = (0xfc, 0xbf, 0x6f) # #fcbf6f
ORANGE_NEAR_PROXY_MIN = 0.72 # depth_gray_norm: 1 = close, 0 = far

def _build_orange_at_point(depth_array, depth_gray_norm, centerpoints,
                           ref_rgb=ORANGE_DEPTH_REF_RGB,
                           proxy_min=ORANGE_NEAR_PROXY_MIN):
    """Per-point True when the depth color at the point is orange-redish (or
    warmer) and the grayscale proxy there is bright."""
    n = len(centerpoints)
    out = [False] * n
    if depth_array is None or depth_array.ndim != 3 or depth_gray_norm is None:
        return out
    ref = np.uint8([[list(ref_rgb)]])
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)[0, 0]
    ref_hue = int(ref_hsv[0])
    ref_v = int(ref_hsv[2])
    h, w = depth_array.shape[:2]
    for i, p in enumerate(centerpoints):
        if p is None:
            continue
        cx, cy = p
        proxy = _depth_proxy_at_point(depth_gray_norm, cx, cy)
        if proxy < proxy_min:
            continue
        cxi = int(np.clip(cx, 0, w - 1))
        cyi = int(np.clip(cy, 0, h - 1))
        y0, y1 = max(0, cyi - 2), min(h, cyi + 3)
        x0, x1 = max(0, cxi - 2), min(w, cxi + 3)
        patch = depth_array[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size == 0:
            continue
        med = np.median(patch, axis=0).astype(np.uint8)
        med_hsv = cv2.cvtColor(np.uint8([[med]]), cv2.COLOR_RGB2HSV)[0, 0]
        med_hue = int(med_hsv[0])
        med_v = int(med_hsv[2])
        # OpenCV hue is 0..180, warm band is hue <= ref_hue or hue >= 165 (red wrap)
        warm = (med_hue <= ref_hue) or (med_hue >= 165)
        bright = med_v >= int(ref_v * 0.85)
        if warm and bright:
            out[i] = True
    return out


# reference violet for is_far_or_violet, nudged toward blue for a wider far band
INTERSECTION_FAR_VIOLET_REF_RGB = (0x4f, 0x66, 0xad)

def _build_far_or_violet_mask(depth_array, ref_rgb=INTERSECTION_FAR_VIOLET_REF_RGB):
    """Boolean grid of far-end depth-colormap pixels, None without a color
    depth map."""
    if depth_array is None or depth_array.ndim != 3:
        return None
    ref = np.uint8([[list(ref_rgb)]])
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_RGB2HSV)[0, 0]
    _, _, ref_v = int(ref_hsv[0]), int(ref_hsv[1]), int(ref_hsv[2])
    depth_hsv = cv2.cvtColor(depth_array, cv2.COLOR_RGB2HSV)
    H = depth_hsv[..., 0]
    S = depth_hsv[..., 1]
    V = depth_hsv[..., 2]
    darker = V <= ref_v
    # OpenCV's H is 0..180, violet band ~110..150
    violet = (H >= 110) & (H <= 150) & (V <= int(ref_v * 1.15))
    # bright saturated blue is still background, S >= 90 (~35 %) keeps pale fish out
    bright_blue = (H >= 95) & (H <= 130) & (S >= 90)
    return darker | violet | bright_blue



# run_for_shard helpers

def _auto_batch_size(gpu_total_gb):
    """Crops per multi-image SAM3 forward, by GPU memory."""
    if gpu_total_gb >= 100:
        return 32
    if gpu_total_gb >= 70:
        return 16
    if gpu_total_gb >= 45:
        return 8
    if gpu_total_gb >= 28:
        return 4
    return 2


def parse_centerpoints(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_node = root.find("size")
    image_width = int(size_node.find("width").text)
    image_height = int(size_node.find("height").text)
    centerpoints = []
    for obj in root.findall("object"):
        point_node = obj.find("point")
        if point_node is None:
            centerpoints.append(None)
            continue
        x = int(float(point_node.find("x").text))
        y = int(float(point_node.find("y").text))
        centerpoints.append((x, y))
    return image_width, image_height, centerpoints


def _load_depth_array(stem):
    """Colorised depthmap from DEPTH_DIR as a 3-channel ndarray, None if missing."""
    for ext in (".jpg", ".png"):
        depth_path = DEPTH_DIR / f"{stem}{ext}"
        if depth_path.exists():
            return np.array(Image.open(depth_path).convert("RGB"))
    return None


def _load_depth_gray_array(stem):
    """Grayscale depthmap (bright = close) as a single-channel ndarray, None if missing."""
    for ext in (".jpg", ".png"):
        depth_path = DEPTH_GRAY_DIR / f"{stem}{ext}"
        if depth_path.exists():
            return np.array(Image.open(depth_path).convert("L"))
    return None


def _normalize_depth_gray(depth_input):
    """5/95-percentile normalisation, 0 = far/dark, 1 = close/bright."""
    if depth_input.ndim == 2:
        gray = depth_input.astype(np.float32)
    else:
        gray = cv2.cvtColor(depth_input, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lo, hi = np.percentile(gray, [5, 95])
    if hi - lo < 1e-6:
        return np.full_like(gray, 0.5)
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


# per-point local scale, dense clusters get smaller
DENSE_RADIUS_FACTOR = 1.5
DENSE_MIN_NEIGHBORS = 3
DENSE_SCALE_SHRINK = 0.45

def _per_point_local_scale(centerpoints, image_h, image_w):
    """Per-point scale adapted to local density, dense points get their NN
    distance shrunk by DENSE_SCALE_SHRINK, isolated ones the global scale."""
    n = len(centerpoints)
    valid_mask = [p is not None for p in centerpoints]
    valid_points = [p for p in centerpoints if p is not None]

    def _global_fish_scale(centerpoints, image_h, image_w):
        """Median NN distance across valid points, per-image typical fish size."""
        valid = [p for p in centerpoints if p is not None]
        if len(valid) <= 1:
            return min(image_h, image_w) // 10
        pts = np.array(valid, dtype=np.float32)
        diffs = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diffs * diffs).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        nn = dists.min(axis=1)
        return max(15, int(np.median(nn) * 0.7))

    if len(valid_points) <= 2:
        scale = _global_fish_scale(centerpoints, image_h, image_w)
        return [scale if v else None for v in valid_mask]

    from scipy.spatial import cKDTree
    pts = np.array(valid_points, dtype=np.float64)
    tree = cKDTree(pts)
    nn_dists, _ = tree.query(pts, k=2)
    nn1 = nn_dists[:, 1]
    median_nn = float(np.median(nn1))
    radius = median_nn * DENSE_RADIUS_FACTOR    # neighborhood radius capturing local density
    neigh_counts = np.asarray(tree.query_ball_point(pts, radius, return_length=True)) - 1   # exclude self

    global_scale = max(15, int(median_nn * 0.7))
    local = []
    valid_iter = iter(range(len(valid_points)))
    for is_valid in valid_mask:
        if not is_valid:
            local.append(None)
            continue
        i = next(valid_iter)
        if neigh_counts[i] >= DENSE_MIN_NEIGHBORS:
            local.append(max(10, int(nn1[i] * DENSE_SCALE_SHRINK)))
        else:
            local.append(global_scale)
    return local


def render_segmap_colored(instance_map):
    """uint16 instance map to a uint8 BGR image, golden-ratio hue per id."""
    height, width = instance_map.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    unique_ids = np.unique(instance_map)
    nonzero_ids = unique_ids[unique_ids > 0]
    if len(nonzero_ids) == 0:
        return canvas, 0
    golden_ratio_conjugate = 0.61803398875
    hue_phase_offset = 0.13
    num_unique = len(nonzero_ids)
    hsv_palette = np.zeros((num_unique, 1, 3), dtype=np.uint8)
    for ordinal in range(num_unique):
        hue = ((ordinal * golden_ratio_conjugate) + hue_phase_offset) % 1.0
        hsv_palette[ordinal, 0] = (int(hue * 179), 255, 255)
    bgr_palette = cv2.cvtColor(hsv_palette, cv2.COLOR_HSV2BGR).reshape(num_unique, 3)
    for ordinal, instance_id in enumerate(nonzero_ids):
        canvas[instance_map == instance_id] = bgr_palette[ordinal]
    return canvas, int(num_unique)


def _write_bbox_xml_with_sources(input_xml_path, output_xml_path, bboxes, bbox_sources):
    """Write bbox XML with a <bbox_source> child per object."""
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    for i, (obj, bbox) in enumerate(zip(objects, bboxes)):
        if bbox is None:
            continue
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox[0])
        ET.SubElement(bndbox, "ymin").text = str(bbox[1])
        ET.SubElement(bndbox, "xmax").text = str(bbox[2])
        ET.SubElement(bndbox, "ymax").text = str(bbox[3])
        src = bbox_sources[i] if bbox_sources is not None and i < len(bbox_sources) else None
        if src is None:
            src = {"cascade": "unknown", "modality": "rgb", "sharpened": False}
        src_node = ET.SubElement(obj, "bbox_source")
        ET.SubElement(src_node, "cascade").text = str(src.get("cascade", "unknown"))
        ET.SubElement(src_node, "modality").text = str(src.get("modality", "rgb"))
        ET.SubElement(src_node, "sharpened").text = "true" if src.get("sharpened") else "false"
        # near-miss diagnostics, only present on fallback_square entries
        for key in ("near_miss_modality", "near_miss_score", "near_miss_floor",
                    "near_miss_cascade", "near_miss_reason"):
            val = src.get(key)
            if val is None:
                continue
            ET.SubElement(src_node, key).text = (
                f"{val:.4f}" if isinstance(val, float) else str(val)
            )
    tree.write(output_xml_path)


