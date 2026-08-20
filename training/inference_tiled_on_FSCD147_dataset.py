"""Tiled inference for GECO2 on FSCD147: overlapping tiles, inner-edge drop,
global NMS, same scheme as the IOCfish tiled script.

The images on disk are images_384_VarV2 (~384 px long side), so tile_size has
to stay below that or tiling does nothing. Few-shot exemplars come from the
COCO GT boxes inside each tile.
"""

import argparse
import math
import os
import time
from pathlib import Path

import sys
# repo root on sys.path for direct runs
_this_dir = Path(__file__).resolve().parent
_repo_root = next((p for p in (_this_dir, *_this_dir.parents) if (p / 'Deformable-DETR').is_dir()), _this_dir)
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / 'Deformable-DETR' / 'models' / 'ops'))

import cv2
import torch
from PIL import Image
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import ops
from torchvision import transforms as T
from tqdm import tqdm

from utils.env import load_env
load_env()

from models.counter import build_model
from utils.arg_parser import get_argparser
from utils.bbox_metrics import COCO_IOU_THRS, aggregate as bbox_aggregate, match_pairs, per_image_match
from utils.coco_export import write_coco_predictions
from utils.count_metrics import format_rows, stratified_rows
from tools.timestamp import now_str
from utils.data import FSC147DATASET, pad_collate_test, resize_and_pad
from utils.viz import draw_bw_dashed_rect, xyxy_int, draw_label, save_depth_visual, save_density_canvas, blend_window


_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_tag_and_regime(model_name):
    """Split a 'GECO2_FSCD147_<tag>_<regime>' model_name into (tag, regime)."""
    parts = model_name.split('_')
    if len(parts) >= 4 and parts[-1] in ('all', 'lt1500'):
        return '_'.join(parts[2:-1]), parts[-1]
    return None, None


def generate_tiles(orig_w, orig_h, tile_size, overlap):
    """Square tiles (x0, y0, x1, y1) covering the image, stride = tile_size - overlap."""
    ts = min(tile_size, orig_w, orig_h)
    stride = max(1, ts - overlap)

    def starts(total):
        if total <= ts:
            return [0]
        pos = list(range(0, total - ts + 1, stride))
        if pos[-1] != total - ts:
            pos.append(total - ts)
        return pos

    return [
        (x, y, x + ts, y + ts)
        for y in starts(orig_h)
        for x in starts(orig_w)
    ]


def select_tile_exemplars(gt_boxes_orig, tile_rect, num_objects):
    """Exemplar boxes for one tile in tile-local coords, from the GT boxes
    centred inside it. None when the tile has no GT box."""
    if gt_boxes_orig is None or gt_boxes_orig.numel() == 0:
        return None
    x0, y0, x1, y1 = tile_rect
    cx = (gt_boxes_orig[:, 0] + gt_boxes_orig[:, 2]) / 2
    cy = (gt_boxes_orig[:, 1] + gt_boxes_orig[:, 3]) / 2
    inside = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
    sel = gt_boxes_orig[inside].clone()
    if sel.shape[0] == 0:
        return None
    sel[:, 0::2] = (sel[:, 0::2] - x0).clamp(0, x1 - x0)
    sel[:, 1::2] = (sel[:, 1::2] - y0).clamp(0, y1 - y0)
    sel = sel[:num_objects]
    if sel.shape[0] < num_objects:
        pad = sel[-1:].expand(num_objects - sel.shape[0], -1)
        sel = torch.cat([sel, pad], dim=0)
    return sel

def prep_tile(tile_img, exemplars, image_size, depth_tile=None):
    """Resize a tile crop to image_size, pad and normalize. A given depth_tile
    is scaled the same way."""
    padded, ex_scaled, sf = resize_and_pad(
        tile_img.contiguous(), exemplars, size=float(image_size), zero_shot=True,
    )
    out = _NORMALIZE(padded)
    if depth_tile is not None:
        from utils.data import _resize_pad_depth
        d = _resize_pad_depth(depth_tile.contiguous(), sf, (0, 0), image_size)
        out = torch.cat([out, d.to(out.dtype)], dim=0)
    return out, ex_scaled, sf

def decode_outputs(output):
    """Raw normalized boxes, box_v and mask-IoU scores for one panel."""
    pred_boxes = output["pred_boxes"]  # (1, N, 4) normalized xyxy
    box_v = output["box_v"]  # (1, N)
    coco_v = output["scores"].reshape(-1).float()  # (N,) mask-IoU (== box_v for >800-box images)
    return pred_boxes[0], box_v[0], coco_v

def drop_inner_edge_origpx(boxes_px, tile_rect, orig_w, orig_h, margin_px):
    """Keep-mask dropping boxes within margin_px of an internal tile edge, the
    neighbour tile sees those objects whole. margin_px in original pixels."""
    n = boxes_px.shape[0]
    keep = torch.ones(n, dtype=torch.bool, device=boxes_px.device)
    if n == 0:
        return keep
    x0, y0, x1, y1 = tile_rect
    if x0 > 0:
        keep &= boxes_px[:, 0] > x0 + margin_px
    if x1 < orig_w:
        keep &= boxes_px[:, 2] < x1 - margin_px
    if y0 > 0:
        keep &= boxes_px[:, 1] > y0 + margin_px
    if y1 < orig_h:
        keep &= boxes_px[:, 3] < y1 - margin_px
    return keep


def collect_panel_candidates(model, device, orig_img, gt_boxes_orig,
                             whole_img, whole_bboxes, whole_sf, whole_padwh, args,
                             orig_depth=None):
    """Forward on all panels (tiles + optional whole image), returns raw
    candidates in original px so --sweep can reuse one forward."""
    _, orig_h, orig_w = orig_img.shape
    image_size = args.image_size
    # tile densities get stitched here, overlap-averaged
    _use_density = getattr(args, 'use_density', 0)
    _dens_canvas = torch.zeros(orig_h, orig_w) if _use_density else None
    _dens_weight = torch.zeros(orig_h, orig_w) if _use_density else None

    # panels: overlapping square tiles, then the whole image
    panels = []  # each: dict(img, bboxes, rect, sf, padwh, is_whole)
    for rect in generate_tiles(orig_w, orig_h, args.tile_size, args.tile_overlap):
        x0, y0, x1, y1 = rect
        if args.zero_shot:
            exemplars = torch.zeros(args.num_objects, 4)
        else:
            exemplars = select_tile_exemplars(gt_boxes_orig, rect, args.num_objects)
            if exemplars is None:
                continue  # few-shot: no GT in this tile
        prepped, ex_scaled, sf = prep_tile(
            orig_img[:, y0:y1, x0:x1], exemplars, image_size,
            depth_tile=(orig_depth[:, y0:y1, x0:x1] if orig_depth is not None else None),
        )
        panels.append(dict(img=prepped, bboxes=ex_scaled, rect=rect,
                           sf=sf, padwh=(0, 0), is_whole=False))

    if args.whole_image_pass:
        # whole image already prepped by the dataset
        panels.append(dict(img=whole_img, bboxes=whole_bboxes,
                           rect=(0, 0, orig_w, orig_h), sf=whole_sf,
                           padwh=whole_padwh, is_whole=True))

    tile_cands = []
    whole_cands = dict(boxes=torch.empty((0, 4)), scores=torch.empty(0),
                       coco_scores=torch.empty(0))
    if not panels:
        return tile_cands, whole_cands, None, None

    for start in range(0, len(panels), args.tile_batch_size):
        batch = panels[start:start + args.tile_batch_size]
        img_batch = torch.stack([p['img'] for p in batch]).to(device)
        bbox_batch = torch.stack([p['bboxes'] for p in batch]).to(device)
        outputs, *_ = model(img_batch, bbox_batch)

        for i, panel in enumerate(batch):
            boxes_norm, scores, coco_scores = decode_outputs(outputs[i])
            if _use_density and not panel['is_whole']:
                _xx0, _yy0, _xx1, _yy1 = panel['rect']
                _dmap = outputs[i]['pred_density'].float()
                _th, _tw = _yy1 - _yy0, _xx1 - _xx0
                _dt = torch.nn.functional.interpolate(
                    _dmap.unsqueeze(0), size=(_th, _tw),
                    mode='bilinear', align_corners=False)[0, 0]
                _s = _dmap.sum()
                if _dt.sum() > 0:
                    _dt = _dt / _dt.sum() * _s  # preserve the tile's predicted count
                _win = blend_window(_th, _tw, panel['rect'], orig_w, orig_h, args.tile_overlap, _dt.device)
                _dens_canvas[_yy0:_yy1, _xx0:_xx1] += (_dt * _win).cpu()
                _dens_weight[_yy0:_yy1, _xx0:_xx1] += _win.cpu()
            if boxes_norm.shape[0] == 0:
                continue
            boxes_norm = boxes_norm.clamp(0, 1)
            x0, y0, x1, y1 = panel['rect']

            if panel['is_whole']:
                # the whole image is padded, drop pad-area boxes
                pad_w, pad_h = panel['padwh']
                cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2 * image_size
                cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2 * image_size
                keep = (cx < image_size - pad_w) & (cy < image_size - pad_h)
                boxes_norm, scores, coco_scores = boxes_norm[keep], scores[keep], coco_scores[keep]
                if boxes_norm.shape[0] == 0:
                    continue
                boxes = boxes_norm * image_size / panel['sf']  # original px
                # whole pass only contributes large boxes
                side = torch.maximum(boxes[:, 2] - boxes[:, 0],
                                     boxes[:, 3] - boxes[:, 1])
                keep = side > args.whole_pass_min_box_px
                whole_cands = dict(boxes=boxes[keep], scores=scores[keep],
                                   coco_scores=coco_scores[keep])
            else:
                # square tile: sf = image_size / tile_side, so px = norm * side
                boxes = boxes_norm * image_size / panel['sf']
                offset = torch.tensor([x0, y0, x0, y0], device=boxes.device)
                boxes = boxes + offset  # original px
                tile_cands.append(dict(boxes=boxes, scores=scores,
                                       coco_scores=coco_scores,
                                       rect=panel['rect']))

    density_count = None
    dens_map = None
    if _use_density:
        # stitched canvas, sum = density count
        dens_map = _dens_canvas / _dens_weight.clamp(min=1e-6)
        density_count = float(dens_map.sum())
    return tile_cands, whole_cands, density_count, dens_map


def postprocess_candidates(tile_cands, whole_cands, orig_w, orig_h, args,
                           score_abs_thr=None, nms_iou=None, edge_margin=None,
                           density_count=None):
    """Inner-edge drop, concat with the whole-pass boxes, absolute box_v
    threshold, one global NMS."""
    dens = density_count if (getattr(args, 'use_density', 0) and density_count is not None) else None
    score_abs_thr = args.score_abs_thr if score_abs_thr is None else score_abs_thr
    nms_iou = args.nms_iou if nms_iou is None else nms_iou
    edge_margin = args.edge_margin if edge_margin is None else edge_margin

    parts_b, parts_s, parts_c = [], [], []
    for c in tile_cands:
        keep = drop_inner_edge_origpx(c['boxes'], c['rect'], orig_w, orig_h, edge_margin)
        if keep.any():
            parts_b.append(c['boxes'][keep])
            parts_s.append(c['scores'][keep])
            parts_c.append(c['coco_scores'][keep])
    if whole_cands['boxes'].shape[0]:
        parts_b.append(whole_cands['boxes'])
        parts_s.append(whole_cands['scores'])
        parts_c.append(whole_cands['coco_scores'])

    if not parts_b:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0), (dens if dens is not None else 0)
    boxes = torch.cat(parts_b, dim=0)
    scores = torch.cat(parts_s, dim=0)
    coco_scores = torch.cat(parts_c, dim=0)

    keep = scores > score_abs_thr
    if args.score_rel_thr > 0 and scores.numel():
        keep &= scores > scores.max() * args.score_rel_thr
    boxes, scores, coco_scores = boxes[keep], scores[keep], coco_scores[keep]
    if boxes.shape[0] == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0), (dens if dens is not None else 0)

    # global NMS across tiles
    keep = ops.nms(boxes, scores, nms_iou)
    boxes = boxes[keep]
    scores = scores[keep]
    coco_scores = coco_scores[keep]
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, orig_w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, orig_h)
    return (boxes.detach().cpu(), scores.detach().cpu(), coco_scores.detach().cpu(),
            (dens if dens is not None else boxes.shape[0]))


def tiled_predict(model, device, orig_img, gt_boxes_orig,
                  whole_img, whole_bboxes, whole_sf, whole_padwh, args, orig_depth=None):
    """Tiled inference on one image: one forward, then threshold/NMS/edge-margin."""
    _, orig_h, orig_w = orig_img.shape
    tile_cands, whole_cands, density_count, dens_map = collect_panel_candidates(
        model, device, orig_img, gt_boxes_orig,
        whole_img, whole_bboxes, whole_sf, whole_padwh, args, orig_depth=orig_depth,
    )
    return (*postprocess_candidates(tile_cands, whole_cands, orig_w, orig_h, args,
                                    density_count=density_count),
            dens_map)

def save_visual(visuals_dir, img_path, img_id, gt_bboxes_1024, sf,
                pred_boxes, pred_scores, gt_count, pred_count, n_fp, args):
    """Draw GT and predicted boxes on the original image, green = matched at IoU>=0.5."""
    vis = cv2.imread(str(img_path))
    if vis is None:
        log(f"    [warn] could not read original image for {img_id} -- no visualization")
        return
    h, w = vis.shape[:2]

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    if args.draw_tiles:
        for (x0, y0, x1, y1) in generate_tiles(w, h, args.tile_size, args.tile_overlap):
            cv2.rectangle(vis, (x0, y0), (min(x1, w - 1), min(y1, h - 1)),
                          (128, 128, 128), 1)

    # GT back to original px, preds already there
    gt = gt_bboxes_1024.detach().cpu()
    gt = gt[gt.abs().sum(dim=1) > 0]  # drop zero padding
    gt_px = gt / sf
    pred_cpu = pred_boxes.detach().cpu() if pred_boxes.numel() else pred_boxes

    # greedy matching at IoU>=0.5, same rule as the metrics
    pairs = (
        match_pairs(pred_cpu, pred_scores.detach().cpu(), gt_px, 0.5)
        if pred_cpu.numel() else []
    )
    matched_pred = {pi for pi, _ in pairs}

    for gj in range(gt_px.shape[0]):
        x1, y1, x2, y2 = xyxy_int(gt_px[gj], w, h)
        draw_bw_dashed_rect(vis, x1, y1, x2, y2, thickness=1)

    for pi in range(pred_cpu.shape[0]):
        x1, y1, x2, y2 = xyxy_int(pred_cpu[pi], w, h)
        color = GREEN if pi in matched_pred else RED
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.circle(vis, ((x1 + x2) // 2, (y1 + y2) // 2), 3, color, -1)

    # connect matched centers
    for pi, gj in pairs:
        gx1, gy1, gx2, gy2 = xyxy_int(gt_px[gj], w, h)
        px1, py1, px2, py2 = xyxy_int(pred_cpu[pi], w, h)
        cv2.line(vis, ((gx1 + gx2) // 2, (gy1 + gy2) // 2),
                 ((px1 + px2) // 2, (py1 + py2) // 2), GREEN, 1)

    WHITE = (255, 255, 255)
    draw_label(vis, [
        (f"pred: {int(pred_cpu.shape[0])}", GREEN),
        (f"fp: {int(n_fp)}", RED),
        (f"gt: {int(gt_px.shape[0])}", WHITE),
    ])
    cv2.imwrite(os.path.join(visuals_dir, f"{img_id}_pred.png"), vis)


def _parse_float_list(s):
    """Parse a comma-separated string into a float list."""
    return [float(x) for x in str(s).split(',') if x.strip() != '']


def _load_orig_and_gt(test_dataset, idx, args):
    """Original unpadded RGB tensor and, in few-shot mode, GT boxes in original px."""
    orig_path = test_dataset.img_dir / test_dataset.image_names[idx]
    orig_img = T.ToTensor()(Image.open(orig_path).convert('RGB'))

    # native-res cached depth for the tile crops
    orig_depth = None
    dm = getattr(test_dataset, 'depthmaps_dir', '')
    if dm:
        import os as _os
        from utils.depth_recipe import load_depth_map, load_depth_feats
        _stem = _os.path.splitext(test_dataset.image_names[idx])[0]
        orig_depth = load_depth_map(dm, _stem, (orig_img.shape[1], orig_img.shape[2]))
        df = getattr(test_dataset, 'depthfeats_dir', '')
        if df:
            # decoder-feature channels stacked after the 1-ch map
            orig_depth = torch.cat(
                [orig_depth, load_depth_feats(df, _stem, (orig_img.shape[1], orig_img.shape[2]))], dim=0)

    gt_boxes_orig = None
    if not args.zero_shot:
        coco_boxes = test_dataset.get_gt_bboxes(idx)
        if coco_boxes:
            gt_boxes_orig = torch.tensor(coco_boxes, dtype=torch.float32)
    return orig_path, orig_img, gt_boxes_orig, orig_depth


def run_sweep(args, model, device, test_dataset, test_loader):
    """Grid-search (score_abs_thr, nms_iou, edge_margin) on val, one forward
    per image, ranked by box-count MAE."""
    abs_thrs = _parse_float_list(args.sweep_score_abs_thr)
    nms_ious = _parse_float_list(args.sweep_nms_iou)
    edge_margins = _parse_float_list(args.sweep_edge_margin)
    configs = [(a, n, e) for a in abs_thrs for n in nms_ious for e in edge_margins]
    log(f"SWEEP over {len(configs)} configs  "
        f"abs_thr={abs_thrs}  nms_iou={nms_ious}  edge_margin={edge_margins}")

    use_density = bool(getattr(args, 'use_density', 0))
    accum = {cfg: dict(ae=0.0, se=0.0, nae=0.0) for cfg in configs}  # box-count error
    dens = dict(ae=0.0, se=0.0, nae=0.0)  # density-integral error (knob-independent)
    n_eval = 0

    limit = args.max_images if args.max_images and args.max_images > 0 else len(test_dataset)
    n_run = min(limit, len(test_dataset))
    log(f"sweeping on first {n_run} VAL images (forward once per image)...")

    for batch_i, (img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh) in enumerate(
            tqdm(test_loader, desc="Sweep", total=n_run)):
        if batch_i >= limit:
            break
        gt_count = density_map.flatten(1).sum(dim=1)[0].item()
        if not math.isfinite(gt_count):
            continue

        _, orig_img, gt_boxes_orig, orig_depth = _load_orig_and_gt(test_dataset, ids[0].item(), args)
        _, orig_h, orig_w = orig_img.shape
        tile_cands, whole_cands, density_count, _dens_map_unused = collect_panel_candidates(
            model, device, orig_img, gt_boxes_orig,
            img[0], bboxes[0], scaling_factor[0].item(),
            (int(padwh[0][0]), int(padwh[0][1])), args, orig_depth=orig_depth,
        )

        n_eval += 1
        if use_density and density_count is not None:
            d = gt_count - density_count
            dens['ae'] += abs(d); dens['se'] += d * d; dens['nae'] += abs(d) / max(gt_count, 1)

        for cfg in configs:
            a, nms_i, e = cfg
            boxes, _, _, count = postprocess_candidates(
                tile_cands, whole_cands, orig_w, orig_h, args,
                score_abs_thr=a, nms_iou=nms_i, edge_margin=e,
                density_count=density_count,
            )
            # knobs move boxes, not the density integral
            box_count = boxes.shape[0]
            diff = gt_count - box_count
            accum[cfg]['ae'] += abs(diff)
            accum[cfg]['se'] += diff * diff
            accum[cfg]['nae'] += abs(diff) / max(gt_count, 1)

    if n_eval == 0:
        raise RuntimeError("Sweep evaluated 0 images (all gt counts non-finite?).")

    rows = []
    for cfg in configs:
        d = accum[cfg]
        rows.append((cfg, d['ae'] / n_eval, math.sqrt(d['se'] / n_eval), d['nae'] / n_eval))
    rows.sort(key=lambda r: r[1])  # by box-count MAE

    count_label = 'BoxMAE' if use_density else 'MAE'
    header = (f"{'abs_thr':>8s} {'nms_iou':>8s} {'edge_px':>8s} "
              f"{count_label:>9s} {'MSE*':>9s} {'NAE':>8s}")
    out_path = os.path.abspath(os.path.join(
        args.model_path, f'{args.model_name}{args.results_suffix}_SWEEP.txt'))
    log(f"sweep done over {n_eval} images; writing {out_path}")
    print(header)
    with open(out_path, 'w') as f:
        f.write(f"Tiled-inference parameter sweep on {n_eval} VAL images "
                f"(hyperparameter tuning; test/{args.test_split} untouched)\n")
        f.write(f"Created: {now_str()}\n")
        f.write(f"model: {args.model_name}  (ckpt_epochs={args.ckpt_epochs})\n")
        f.write(f"fixed: tile_size={args.tile_size} overlap={args.tile_overlap} "
                f"whole_image_pass={bool(args.whole_image_pass)} "
                f"whole_pass_min_box_px={args.whole_pass_min_box_px} "
                f"score_rel_thr(cap)={args.score_rel_thr}\n")
        if use_density:
            f.write(
                "NOTE: --use_density on. The swept knobs (abs_thr/nms_iou/edge_px) move only\n"
                "      the BOX count + AP; the table below is ranked by BOX-count MAE. The\n"
                "      headline density-integral count is knob-INVARIANT:\n"
                f"      density-integral  MAE={dens['ae']/n_eval:.3f}  "
                f"MSE*={math.sqrt(dens['se']/n_eval):.3f}  NAE={dens['nae']/n_eval:.3f}\n")
        f.write("\n")
        f.write(header + "\n")
        f.write('-' * len(header) + "\n")
        for cfg, mae, rmse, nae in rows:
            a, nms_i, e = cfg
            line = f"{a:>8.3f} {nms_i:>8.2f} {e:>8.1f} {mae:>9.3f} {rmse:>9.3f} {nae:>8.3f}"
            print(line)
            f.write(line + "\n")
    print(f"Sweep results saved to: {out_path}")
    best = rows[0]
    log(f"BEST by {count_label}: abs_thr={best[0][0]} nms_iou={best[0][1]} edge_margin={best[0][2]} "
        f"-> MAE={best[1]:.3f}  MSE*={best[2]:.3f}  NAE={best[3]:.3f}")
    # winning config for --sweep_then_test
    return best[0]


@torch.no_grad()
def evaluate(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    if args.depth_zero_ablation:
        # don't double-tag the suffix
        if '_zerodepth' not in args.results_suffix:
            args.results_suffix += '_zerodepth'
        if args.coco_json_out and '_zerodepth' not in args.coco_json_out:
            root, ext = os.path.splitext(args.coco_json_out)
            args.coco_json_out = root + '_zerodepth' + ext

    log("building model...")
    model = DataParallel(
        build_model(args).to(device), device_ids=[gpu], output_device=gpu,
    )

    ckpt_basename = f'{args.model_name}_{args.ckpt_epochs}.pth' if args.ckpt_epochs else f'{args.model_name}.pth'
    ckpt_path = os.path.join(args.model_path, ckpt_basename)
    log(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model']
    # ckpt['epoch'] is the best epoch (saved only on val improvement)
    best_epoch = ckpt.get('epoch', None)
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log(f"checkpoint loaded (missing keys: {len(missing)}, unexpected keys: {len(unexpected)}, best_epoch: {best_epoch})")
    if args.depth_zero_ablation:
        if args.use_depth <= 0:
            raise ValueError("--depth_zero_ablation requires a depth model (--use_depth > 0).")
        # reset depth fusion to identity, depth contributes 0
        model.module.apply_depth_fusion_identity()
        log("ZERO-DEPTH ABLATION: depth fusion reset to identity -- this run measures "
            "the checkpoint WITHOUT its depth contribution (artifacts tagged _zerodepth)")
    model.eval()

    log(f"building test dataset (split={args.test_split})...")
    # the dataset stays in few-shot mode even for zero-shot runs: with zero_shot=True
    # its all-zero exemplars hit 80 / a_dim == 0 in resize_and_pad
    test_dataset = FSC147DATASET(
        args.data_path, args.image_size, split=args.test_split,
        num_objects=args.num_objects, tiling_p=args.tiling_p,
        return_ids=True, training=False,
    )
    from utils.depth_recipe import attach_depthmaps, attach_depthfeats
    attach_depthmaps(test_dataset, args, device=device, log=log)
    attach_depthfeats(test_dataset, args, device=device, log=log)
    test_loader = DataLoader(
        test_dataset, batch_size=1, drop_last=False,
        num_workers=args.num_workers, collate_fn=pad_collate_test,
    )
    log(f"test dataset ready: {len(test_dataset)} images")

    if args.sweep or args.sweep_then_test:
        # thresholds get calibrated on val, never on test
        val_split = {'test': 'val', 'test_coco': 'val_coco'}.get(args.test_split, 'val')
        log(f"building VAL dataset (split={val_split}) for the sweep (hyperparameter tuning must not touch test)...")
        val_dataset = FSC147DATASET(
            args.data_path, args.image_size, split=val_split,
            num_objects=args.num_objects, tiling_p=args.tiling_p,
            return_ids=True, training=False,
        )
        attach_depthmaps(val_dataset, args, device=device, log=log)
        attach_depthfeats(val_dataset, args, device=device, log=log)
        val_loader = DataLoader(
            val_dataset, batch_size=1, drop_last=False,
            num_workers=args.num_workers, collate_fn=pad_collate_test,
        )
        log(f"val dataset ready: {len(val_dataset)} images")
        best_cfg = run_sweep(args, model, device, val_dataset, val_loader)
        if args.sweep:
            return  # sweep-only, done
        args.score_abs_thr, args.nms_iou, args.edge_margin = best_cfg
        log(f"[sweep-then-test] AUTO-APPLYING val-best thresholds for the test run: "
            f"score_abs_thr={args.score_abs_thr} nms_iou={args.nms_iou} "
            f"edge_margin={args.edge_margin}")
        # --max_images only bounded the sweep subset
        args.max_images = 0

    if args.no_visuals:
        visuals_dir = None
        log("visualizations disabled (--no_visuals) -- only results txt will be written")
    else:
        visuals_dir = os.path.join(args.model_path, f'{args.model_name}{args.results_suffix}_visuals')
        os.makedirs(visuals_dir, exist_ok=True)
        every = f" (every {args.visuals_every} images)" if args.visuals_every > 1 else ""
        log(f"visualizations will be saved to: {visuals_dir}{every}")

    mode = 'zero-shot' if args.zero_shot else 'few-shot'
    log(f"tiled inference ({mode}): tile_size={args.tile_size} overlap={args.tile_overlap} "
        f"tile_batch_size={args.tile_batch_size} whole_image_pass={bool(args.whole_image_pass)} "
        f"nms_iou={args.nms_iou} score_abs_thr={args.score_abs_thr} "
        f"score_rel_thr(cap)={args.score_rel_thr} edge_margin_px={args.edge_margin}")

    ae = 0.0
    se = 0.0
    skipped_nan = []  # non-finite gt counts, excluded from metrics
    per_image_results = []
    bbox_records = []  # per_image_match() output per image
    coco_records = []  # (ori_id, boxes_xyxy_origpx, scores) for --coco_json_out

    n_total = len(test_dataset)
    limit = args.max_images if args.max_images and args.max_images > 0 else n_total
    n_run = min(limit, n_total)

    log(f"starting tiled inference over {n_run} images...")
    for batch_i, (img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh) in enumerate(
        tqdm(test_loader, desc="Test", total=n_run)
    ):
        if batch_i >= limit:
            break
        img_id = test_dataset.image_ids[ids[0].item()]

        # GT count = density-map sum, same as the whole-image script
        gt_count = density_map.flatten(1).sum(dim=1)[0].item()
        if not math.isfinite(gt_count):
            skipped_nan.append(img_id)
            log(f"    image {img_id}: non-finite gt count -- skipped from metrics")
            continue

        orig_path, orig_img, gt_boxes_orig, orig_depth = _load_orig_and_gt(test_dataset, ids[0].item(), args)

        # timed region = the tile-and-stitch forward
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _infer_t0 = time.perf_counter()
        pred_boxes, pred_scores, match_scores, returned_count, dens_map = tiled_predict(
            model, device, orig_img, gt_boxes_orig,
            img[0], bboxes[0], scaling_factor[0].item(),
            (int(padwh[0][0]), int(padwh[0][1])), args, orig_depth=orig_depth,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _per_image_time = time.perf_counter() - _infer_t0
        # primary count = boxes, density integral secondary
        box_count = len(pred_boxes)
        density_count = returned_count if getattr(args, 'use_density', 0) else None
        pred_count = box_count

        diff = gt_count - pred_count
        ae += abs(diff)
        se += diff * diff

        # GT back to original px (/sf), zero-pad rows dropped
        sf = scaling_factor[0].item()
        gt_boxes_1024 = gt_bboxes[0].detach().cpu()
        gt_boxes_1024 = gt_boxes_1024[gt_boxes_1024.abs().sum(dim=1) > 0]
        gt_boxes = gt_boxes_1024 / sf
        # AP ranked by mask-IoU scores
        bbox_rec = per_image_match(pred_boxes, match_scores, gt_boxes, COCO_IOU_THRS)
        bbox_records.append(bbox_rec)
        if args.coco_json_out:
            name = test_dataset.image_names[ids[0].item()]
            coco_records.append((test_dataset.img_name_to_ori_id[name], pred_boxes, match_scores))
        canon = bbox_rec['per_thr'][0]  # COCO_IOU_THRS[0] == 0.5
        iou_mean = float(canon['tp_iou'].mean().item()) if canon['tp_iou'].numel() else float('nan')
        giou_mean = float(canon['tp_giou'].mean().item()) if canon['tp_giou'].numel() else float('nan')

        per_image_results.append({
            'image_idx': ids[0].item(),
            'image_id': img_id,
            'gt_count': gt_count,
            # density_count is None unless --use_density
            'pred_count': pred_count,
            'box_count': box_count,
            'density_count': density_count,
            'tp': canon['n_tp'],
            'fp': canon['n_fp'],
            'fn': canon['n_fn'],
            'iou_tp': iou_mean,
            'giou_tp': giou_mean,
            'time_s': _per_image_time,
        })
        log(f"[{batch_i + 1}/{n_run}] {img_id}: "
            f"gt={gt_count:.0f}  pred={pred_count}  ae={abs(diff):.0f}  "
            f"TP={canon['n_tp']} FP={canon['n_fp']} FN={canon['n_fn']}  "
            f"IoU={iou_mean:.3f} GIoU={giou_mean:.3f}  time={_per_image_time:.3f}s")

        if visuals_dir is not None and batch_i % args.visuals_every == 0:
            save_visual(visuals_dir, orig_path, img_id, gt_bboxes[0],
                        scaling_factor[0].item(), pred_boxes, match_scores,
                        gt_count, pred_count, canon['n_fp'], args)
            # depth view for depth runs
            save_depth_visual(model.module, img[0].to(device),
                               (float(padwh[0][0]), float(padwh[0][1])),
                               int(orig_img.shape[-1]), int(orig_img.shape[-2]),
                               os.path.join(visuals_dir, f"{img_id}_depth.png"))
            # density view when the density head is active
            if getattr(args, 'use_density', 0) and dens_map is not None:
                save_density_canvas(dens_map,
                                     int(orig_img.shape[-1]), int(orig_img.shape[-2]),
                                     os.path.join(visuals_dir, f"{img_id}_density.png"))

    log("tiled inference loop done -- computing final metrics...")
    n = len(per_image_results)  # excludes samples with non-finite gt counts
    if n == 0:
        raise RuntimeError(
            f"all {n_total} test samples had non-finite gt counts"
        )
    mae = ae / n
    rmse = math.sqrt(se / n)
    nae = sum(
        abs(r['gt_count'] - r['pred_count']) / max(r['gt_count'], 1)
        for r in per_image_results
    ) / n

    bbox_agg = bbox_aggregate(bbox_records, COCO_IOU_THRS)
    canon_agg = bbox_agg['per_thr'][0.5]

    # primary count = box-count, density-integral secondary
    primary_label = 'box-count'
    primary_rows = stratified_rows(per_image_results, 'pred_count', args.report_max_gt)
    density_rows = stratified_rows(per_image_results, 'density_count', args.report_max_gt) if args.use_density else None

    print(f"Test MAE:   {mae:.3f}")
    print(f"Test MSE*:  {rmse:.3f}   (RMSE)")
    print(f"Test NAE:   {nae:.3f}")
    print(f"Count metrics ({primary_label}, count = len(boxes); GT-count-stratified):")
    for line in format_rows(primary_rows):
        print(line)
    if density_rows is not None:
        print("Density-integral count metrics (SECONDARY, count = density.sum()):")
        for line in format_rows(density_rows):
            print(line)
    print(f"Bbox @ IoU=0.50:  mean IoU(TP)={canon_agg['mean_iou_tp']:.3f}  "
          f"mean GIoU(TP)={canon_agg['mean_giou_tp']:.3f}  "
          f"P={canon_agg['precision']:.3f}  R={canon_agg['recall']:.3f}  "
          f"F1={canon_agg['f1']:.3f}  "
          f"(TP={canon_agg['n_tp']} FP={canon_agg['n_fp']} FN={canon_agg['n_fn']})")
    print(f"AP@0.50:      {bbox_agg['AP_50']:.3f}")
    print(f"AP@[.5:.95]:  {bbox_agg['AP_5095']:.3f}")
    if skipped_nan:
        print(
            f"[warn] skipped {len(skipped_nan)}/{n_total} test images with non-finite "
            f"gt counts (first few: {skipped_nan[:5]})"
        )

    txt_path = os.path.abspath(os.path.join(
        args.model_path, f'{args.model_name}{args.results_suffix}_results.txt'))
    log(f"writing results txt: {txt_path}")
    _times = [r['time_s'] for r in per_image_results if r.get('time_s') is not None]
    total_infer_time = sum(_times)
    avg_infer_time = total_infer_time / len(_times) if _times else float('nan')
    if _times:
        print(f"Avg inference time/image: {avg_infer_time * 1000:.1f} ms "
              f"({len(_times) / total_infer_time:.2f} img/s); "
              f"total {total_infer_time:.2f} s over {len(_times)} images")

    with open(txt_path, 'w') as f:
        f.write(f"Model path: {os.path.abspath(ckpt_path)}\n")
        f.write(f"Created: {now_str()}\n")
        if args.depth_zero_ablation:
            f.write("ZERO-DEPTH ABLATION: depth fusion reset to identity after loading -- "
                    "this is the checkpoint WITHOUT its learned depth contribution.\n")
        f.write("Bbox matching/AP + COCO dump ranked by SAM mask-IoU scores (was box_v -- "
                "counting/threshold/NMS unchanged, so count metrics are comparable to "
                "older runs but AP/P/R/F1 are not).\n")
        f.write(f"Inference: TILED  (tile_size={args.tile_size}, overlap={args.tile_overlap}, "
                f"whole_image_pass={bool(args.whole_image_pass)}, "
                f"whole_pass_min_box_px={args.whole_pass_min_box_px}, "
                f"nms_iou={args.nms_iou}, score_abs_thr={args.score_abs_thr}, "
                f"score_rel_thr_cap={args.score_rel_thr}, edge_margin_px={args.edge_margin})\n")
        f.write(f"Config: backbone={args.backbone}, image_size={args.image_size}, "
                f"num_enc_layers={args.num_enc_layers}, emb_dim={args.emb_dim}, "
                f"num_objects={args.num_objects}, reduction={args.reduction}, "
                f"mode={mode}\n")
        exp_tag, exp_regime = parse_tag_and_regime(args.model_name)
        epochs_str = str(args.ckpt_epochs) if args.ckpt_epochs else 'unknown (legacy un-suffixed ckpt)'
        best_epoch_str = str(best_epoch) if best_epoch is not None else 'unknown (not in ckpt)'
        f.write(f"\nSummary:\n")
        f.write(f"  Experiment tag:   {exp_tag if exp_tag is not None else 'unknown'}\n")
        f.write(f"  Training regime:  {exp_regime if exp_regime is not None else 'unknown'}\n")
        f.write(f"  Test split:       {args.test_split}\n")
        f.write(f"  Inference type:   tiled\n")
        f.write(f"  Epochs (max):     {epochs_str}\n")
        f.write(f"  Best epoch:       {best_epoch_str}\n")
        f.write(f"  Test images: {n} of {n_total}\n")
        if skipped_nan:
            f.write(f"  Skipped (non-finite gt count): {len(skipped_nan)} -> "
                    f"{skipped_nan[:10]}{'...' if len(skipped_nan) > 10 else ''}\n")
        f.write(f"  Primary metric:   box-count (count = len(boxes)); density-integral is secondary\n")
        f.write(f"  Test MAE:    {mae:.3f}\n")
        f.write(f"  Test MSE*:   {rmse:.3f}   (RMSE)\n")
        f.write(f"  Test NAE:    {nae:.3f}\n")
        if _times:
            f.write(f"  Avg inference time/image:  {avg_infer_time * 1000:.1f} ms  ({avg_infer_time:.4f} s)\n")
            f.write(f"  Total inference time:      {total_infer_time:.2f} s over {len(_times)} images\n")
            f.write(f"  Throughput:                {len(_times) / total_infer_time:.2f} images/s\n")
        f.write(f"\nCount metrics ({primary_label}, count = len(boxes); GT-count-stratified):\n")
        for line in format_rows(primary_rows):
            f.write(line + "\n")
        if density_rows is not None:
            f.write(f"\nDensity-integral count metrics (SECONDARY, count = density.sum()):\n")
            for line in format_rows(density_rows):
                f.write(line + "\n")
        f.write(f"\nBbox-quality metrics (greedy 1-to-1 matching by descending score;\n")
        f.write(f"  the AP rows below are the 0-1 GREEDY diagnostic approximation -- the\n")
        f.write(f"  official COCO AP/AP50 for paper Table 1 is in the sibling *_cocoeval.txt\n")
        f.write(f"  (0-100 scale, from eval_bboxes.py on the *_coco_preds.json dump)):\n")
        f.write(f"  Mean IoU (TP) @0.5:  {canon_agg['mean_iou_tp']:.4f}\n")
        f.write(f"  Mean GIoU (TP) @0.5: {canon_agg['mean_giou_tp']:.4f}\n")
        f.write(f"  Precision @0.5:      {canon_agg['precision']:.4f}\n")
        f.write(f"  Recall    @0.5:      {canon_agg['recall']:.4f}\n")
        f.write(f"  F1        @0.5:      {canon_agg['f1']:.4f}\n")
        f.write(f"  TP / FP / FN @0.5:   {canon_agg['n_tp']} / {canon_agg['n_fp']} / {canon_agg['n_fn']}  (n_gt={canon_agg['n_gt']})\n")
        f.write(f"  AP @0.5:             {bbox_agg['AP_50']:.4f}\n")
        f.write(f"  AP @[.5:.95]:        {bbox_agg['AP_5095']:.4f}\n")
        f.write(f"\nPer-threshold AP:\n")
        for thr in COCO_IOU_THRS:
            f.write(f"  AP @{thr:.2f}: {bbox_agg['per_thr'][thr]['ap']:.4f}\n")
        f.write(f"\n{'Idx':>6s}  {'Image ID':>16s}  {'GT':>6s}  {'Pred':>6s}  {'AE':>8s}  "
                f"{'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'IoU':>6s}  {'GIoU':>7s}  {'Time(s)':>9s}\n")
        f.write('-' * 92 + '\n')
        for r in per_image_results:
            iou_s = f"{r['iou_tp']:.3f}" if r['iou_tp'] == r['iou_tp'] else "  nan"
            giou_s = f"{r['giou_tp']:.3f}" if r['giou_tp'] == r['giou_tp'] else "  nan"
            t_s = f"{r['time_s']:.4f}" if r.get('time_s') is not None else "  nan"
            f.write(
                f"{r['image_idx']:>6d}  {r['image_id']:>16s}  {r['gt_count']:>6.0f}  "
                f"{r['pred_count']:>6.0f}  {abs(r['gt_count'] - r['pred_count']):>8.1f}  "
                f"{r['tp']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}  "
                f"{iou_s:>6s}  {giou_s:>7s}  {t_s:>9s}\n"
            )
    print(f"Results saved to: {txt_path}")

    if args.coco_json_out:
        n_anno = write_coco_predictions(args.coco_json_out, coco_records)
        print(f"COCO-format predictions ({n_anno} boxes over {len(coco_records)} images) "
              f"saved to: {os.path.abspath(args.coco_json_out)}")
        log(f"score with the paper's metric:  python eval_bboxes.py --data_path {args.data_path} "
            f"--pred_json {args.coco_json_out} --split {args.test_split}")
    log("tiled inference complete.")


if __name__ == '__main__':
    _script_dir = _repo_root  # repo root, the script lives under training/

    parser = argparse.ArgumentParser('GECO2-FSCD147-Tiled-Inference', parents=[get_argparser()])
    parser.add_argument(
        '--coco_json_out', type=str, default='',
        help="also dump post-NMS detections as COCO json for eval_bboxes.py (empty = off)",
    )
    parser.add_argument(
        '--results_suffix', type=str, default='_TEST_tiled',
        help="suffix for the visuals dir and results txt",
    )
    parser.add_argument(
        '--test_split', type=str, default='test',
        choices=['test', 'test_coco', 'val', 'val_coco'],
        help="FSC147 evaluation split",
    )
    parser.add_argument(
        '--ckpt_epochs', type=int, default=0,
        help="epoch count in the checkpoint filename, 0 loads the un-suffixed ckpt",
    )
    parser.add_argument(
        '--tile_size', type=int, default=256,
        help="square tile side in original px, keep it below the ~384 px image size",
    )
    parser.add_argument(
        '--tile_overlap', type=int, default=64,
        help='Overlap (px) between adjacent tiles; should exceed the largest object.',
    )
    parser.add_argument(
        '--tile_batch_size', type=int, default=4,
        help='Number of tiles processed per model forward pass.',
    )
    parser.add_argument(
        '--nms_iou', type=float, default=0.5,
        help='IoU threshold for the final global NMS across all tiles; sweep it with --sweep.',
    )
    parser.add_argument(
        '--score_abs_thr', type=float, default=0.1,
        help="absolute box_v cutoff, calibrate with --sweep",
    )
    parser.add_argument(
        '--score_rel_thr', type=float, default=0.0,
        help="optional relative cap on top of the absolute cutoff (0 = off)",
    )
    parser.add_argument(
        '--whole_image_pass', type=int, default=1,
        help='1 = also run one whole-image forward and merge its large-box detections.',
    )
    parser.add_argument(
        '--whole_pass_min_box_px', type=int, default=-1,
        help="drop whole-pass boxes with max side <= this px (-1 = tile_overlap)",
    )
    parser.add_argument(
        '--edge_margin', type=float, default=8.0,
        help="px margin for dropping detections that touch an internal tile edge",
    )
    parser.add_argument(
        '--max_images', type=int, default=0,
        help='Process only the first N test images (0 = all); also bounds the --sweep subset.',
    )
    parser.add_argument(
        '--sweep', action='store_true',
        help="grid-search score_abs_thr x nms_iou x edge_margin on val, writes a SWEEP txt",
    )
    parser.add_argument(
        '--sweep_then_test', action='store_true',
        help="run the sweep on val, then the test eval with the winning config",
    )
    parser.add_argument(
        '--sweep_score_abs_thr', type=str, default='0.05,0.1,0.15,0.2,0.25,0.3',
        help='Comma-separated absolute box_v cutoffs to try in --sweep.',
    )
    parser.add_argument(
        '--sweep_nms_iou', type=str, default='0.3,0.4,0.5',
        help='Comma-separated NMS IoU thresholds to try in --sweep.',
    )
    parser.add_argument(
        '--sweep_edge_margin', type=str, default='4,8,16',
        help='Comma-separated inner-edge margins (original px) to try in --sweep.',
    )
    parser.add_argument(
        '--draw_tiles', action='store_true',
        help='Overlay the tile grid (faint gray) on saved visualizations.',
    )
    parser.add_argument(
        '--no_visuals', action='store_true',
        help="skip per-image PNGs, write only the results txt",
    )
    parser.add_argument(
        '--visuals_every', type=int, default=1,
        help="save a visualization only every Nth test image",
    )

    parser.set_defaults(
        model_name='GECO2_FSCD147',
        dataset='FSCD147',
        data_path=str(_script_dir / 'fsc147'),
        model_path=str(_script_dir / 'models'),
        backbone='SAM',
        reduction=16,
        image_size=1024,
        num_enc_layers=3,
        emb_dim=256,
        num_heads=8,
        kernel_dim=1,  # paper value, forward forces 1 anyway
        num_objects=3,
        batch_size=1,
        num_workers=8,
        tiling_p=0.0,
        giou_loss_coef=2,
        cost_class=2,
        cost_bbox=1,
        cost_giou=2,
        focal_alpha=0.25,
    )

    args = parser.parse_args()
    args.visuals_every = max(1, args.visuals_every)
    if args.whole_pass_min_box_px < 0:
        # default: tile_overlap, so tile and whole-pass size ranges meet
        args.whole_pass_min_box_px = args.tile_overlap
    print(args)
    print("model_name:", args.model_name)
    evaluate(args)
