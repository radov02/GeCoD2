# whole-image inference on MCAC: center-cropped visuals, SRE reported
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
import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import ops
from tqdm import tqdm

from utils.env import load_env
load_env()

from models.counter import build_model
from models.matcher import build_matcher
from utils.arg_parser import get_argparser
from utils.bbox_metrics import COCO_IOU_THRS, aggregate as bbox_aggregate, match_pairs, per_image_match
from utils.box_ops import BOX_V_ABS_THRESHOLD
from utils.count_metrics import format_rows, stratified_rows
from tools.timestamp import now_str
from utils.data import MCACDataset, pad_collate_test
from utils.losses import SetCriterion
from utils.viz import draw_bw_dashed_rect, xyxy_int, draw_label, save_depth_visual, save_density_visual


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_tag_and_mode(model_name):
    """Split a 'GECO2_MCAC_<tag>_<mode>' model_name into (tag, mode)."""
    parts = model_name.split('_')
    if parts and parts[-1] == 'bestdens':
        # '<model_name>_bestdens' checkpoint: tag/mode sit one token earlier
        parts = parts[:-1]
    if len(parts) >= 4 and parts[-1] in ('few', 'zero'):
        return '_'.join(parts[2:-1]), parts[-1]
    return None, None


# grid for the --whole_sweep val sweep, brackets the usual ~max/9 cutoff
WHOLE_SWEEP_ABS_THRS = [0.02, 0.05, 0.08, 0.11, 0.15, 0.20, 0.25, 0.30, 0.40]
WHOLE_SWEEP_NMS_IOUS = [0.3, 0.5, 0.7]


def _box_count_at(v, pred_boxes, abs_thr, nms_iou, img_shape, padwh):
    """Box count at (abs_thr, nms_iou), same selection as the main loop."""
    mask = v > abs_thr
    if int(mask.sum()) == 0:
        return 0
    b = pred_boxes[mask]
    s = v[mask]
    keep = ops.nms(b, s, nms_iou)
    b = torch.clamp(b[keep], 0, 1)
    maxw = img_shape[-1] - padwh[0]
    maxh = img_shape[-2] - padwh[1]
    center = (b[:, :2] + b[:, 2:]) / 2
    valid = (center[:, 0] * img_shape[-1] < maxw) & (center[:, 1] * img_shape[-2] < maxh)
    return int(valid.sum())


@torch.no_grad()
def whole_val_sweep(model, device, args):
    """Grid-search abs box_v threshold x NMS IoU on val, lowest box-count MAE wins."""
    log("[whole_sweep] building VAL dataset for the threshold sweep...")
    val_ds = MCACDataset(
        args.data_path, args.image_size, split='val',
        num_objects=args.num_objects, tiling_p=0.0, return_ids=True, training=False,
    )
    from utils.depth_recipe import attach_depthmaps, attach_depthfeats
    attach_depthmaps(val_ds, args, device=device, log=log)
    attach_depthfeats(val_ds, args, device=device, log=log)
    val_loader = DataLoader(val_ds, batch_size=1, drop_last=False,
                            num_workers=args.num_workers, collate_fn=pad_collate_test)
    min_thr = min(WHOLE_SWEEP_ABS_THRS)
    cand = []  # per-image (v_cpu, boxes_cpu, gt_count, padwh, img_shape)
    n_skipped = 0  # val images with non-finite gt count
    for img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh in tqdm(val_loader, desc="ValSweep"):
        gt = float(density_map.flatten(1).sum(dim=1)[0])
        if not math.isfinite(gt):
            n_skipped += 1
            continue
        img = img.to(device); bboxes = bboxes.to(device)
        outputs, *_ = model(img, bboxes)
        out = outputs[0]
        v = out['box_v']
        if len(out['pred_boxes'][-1]) == 0 or v.numel() == 0:
            cand.append((torch.empty(0), torch.empty((0, 4)), gt,
                         (float(padwh[0][0]), float(padwh[0][1])), tuple(img.shape)))
            continue
        # keep only scores above the smallest grid threshold
        keep = v > min_thr
        cand.append((v[keep].detach().cpu(), out['pred_boxes'][keep].detach().cpu(), gt,
                     (float(padwh[0][0]), float(padwh[0][1])), tuple(img.shape)))
    if n_skipped:
        log(f"[whole_sweep] skipped {n_skipped} val image(s) with non-finite gt count")
    results = []
    for abs_thr in WHOLE_SWEEP_ABS_THRS:
        for nms_iou in WHOLE_SWEEP_NMS_IOUS:
            ae = 0.0
            for v, boxes, gt, pad, ishape in cand:
                c = _box_count_at(v, boxes, abs_thr, nms_iou, ishape, pad) if v.numel() else 0
                ae += abs(gt - c)
            results.append((ae / max(1, len(cand)), abs_thr, nms_iou))
    results.sort(key=lambda r: r[0])
    log("[whole_sweep] top val (abs_thr, nms_iou) by box-count MAE:")
    for mae, t, n in results[:8]:
        log(f"    abs_thr={t:.2f}  nms_iou={n:.2f}  ->  val box MAE={mae:.3f}")
    best = results[0]
    log(f"[whole_sweep] WINNER: abs_thr={best[1]:.2f} nms_iou={best[2]:.2f} (val box MAE={best[0]:.3f}) -- applied to test")
    return best[1], best[2]


@torch.no_grad()
def evaluate(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    if args.depth_zero_ablation:
        # don't double-tag the suffix
        if '_zerodepth' not in args.results_suffix:
            args.results_suffix += '_zerodepth'

    log("building model...")
    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu,
    )

    ckpt_basename = f'{args.model_name}_{args.ckpt_epochs}.pth' if args.ckpt_epochs else f'{args.model_name}.pth'
    ckpt_path = os.path.join(args.model_path, ckpt_basename)
    log(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    # trained ckpts wrap weights under 'model'
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    # ckpt['epoch'] is the best epoch (saved only on val improvement)
    best_epoch = ckpt.get('epoch', None)
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # sam_mask is built only in inference mode and self-initialises from the released
    # SAM2 checkpoint, so training checkpoints never carry those keys
    missing_sam_mask = [k for k in missing if '.sam_mask.' in k]
    missing_real = [k for k in missing if '.sam_mask.' not in k]
    log(f"checkpoint loaded (missing keys: {len(missing)} "
        f"[{len(missing_sam_mask)} sam_mask/inference-only (expected), "
        f"{len(missing_real)} other], unexpected keys: {len(unexpected)}, "
        f"best_epoch: {best_epoch})")
    # other missing keys = architecture flags don't match the checkpoint
    if missing_real:
        raise RuntimeError(
            f"checkpoint/architecture mismatch: {len(missing_real)} missing keys "
            f"(first: {missing_real[:4]}); rerun with the training-time architecture flags")
    if unexpected:
        log(f"[warn] {len(unexpected)} checkpoint keys not in the built model "
            f"(first: {unexpected[:4]}) -- those trained weights are IGNORED; "
            f"check the architecture flags if this is not intended.")
    if args.depth_zero_ablation:
        if args.use_depth <= 0:
            raise ValueError("--depth_zero_ablation requires a depth model (--use_depth > 0).")
        # reset depth fusion to identity, depth contributes 0
        model.module.apply_depth_fusion_identity()
        log("ZERO-DEPTH ABLATION: depth fusion reset to identity -- this run measures "
            "the checkpoint WITHOUT its depth contribution (artifacts tagged _zerodepth)")
    model.eval()

    matcher = build_matcher(args)
    criterion = SetCriterion(
        0, matcher, {"loss_giou": args.giou_loss_coef},
        ["bboxes", "ce"], focal_alpha=args.focal_alpha,
    )
    criterion.to(device)
    criterion.eval()

    # optional --whole_sweep: calibrate on val, apply to test
    args._sweep_abs_thr = None
    args._sweep_nms_iou = None
    if getattr(args, 'whole_sweep', 0) and args.test_split == 'test':
        args._sweep_abs_thr, args._sweep_nms_iou = whole_val_sweep(model, device, args)

    log(f"building eval dataset (split={args.test_split})...")
    test_dataset = MCACDataset(
        args.data_path,
        args.image_size,
        split=args.test_split,
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        return_ids=True,
        training=False,
    )
    from utils.depth_recipe import attach_depthmaps, attach_depthfeats
    attach_depthmaps(test_dataset, args, device=device, log=log)
    attach_depthfeats(test_dataset, args, device=device, log=log)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=pad_collate_test,
    )
    log(f"test dataset ready: {len(test_dataset)} images")

    if args.no_visuals:
        visuals_dir = None
        log("visualizations disabled (--no_visuals) -- only results txt will be written")
    else:
        visuals_dir = os.path.join(args.model_path, f'{args.model_name}{args.results_suffix}_visuals')
        os.makedirs(visuals_dir, exist_ok=True)
        every = f" (every {args.visuals_every} images)" if args.visuals_every > 1 else ""
        log(f"visualizations will be saved to: {visuals_dir}{every}")

    ae = torch.tensor(0.0).to(device)
    se = torch.tensor(0.0).to(device)
    skipped_nan = []  # non-finite gt counts, excluded from metrics
    per_image_results = []
    bbox_records = []  # per_image_match() output per image

    log(f"starting inference over {len(test_loader)} batches...")
    for batch_i, (img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh) in enumerate(
        tqdm(test_loader, desc="Test")
    ):
        img_ids_str = ", ".join(test_dataset.image_ids[i.item()] for i in ids)
        log(f"[{batch_i + 1}/{len(test_loader)}] image(s): {img_ids_str} -- running model forward pass")
        img = img.to(device)
        bboxes = bboxes.to(device)
        gt_bboxes = gt_bboxes.to(device)

        # timed region = forward + box post-processing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _infer_t0 = time.perf_counter()

        outputs, ref_points, centerness, outputs_coord, masks = model(img, bboxes)

        num_objects_pred = []  # box-count, the primary count
        density_counts = []  # density-integral, None unless --use_density
        pred_boxes_batch = []
        pred_scores_batch = []  # post-NMS box_v
        # mask-IoU scores, AP ranks by these not box_v
        pred_match_scores_batch = []
        batch_size = img.size(0)
        for idx in range(batch_size):
            thr = 1 / 0.11

            density_count = None
            if args.use_density:
                # density count over the non-padded region
                _dmap = outputs[idx]['pred_density'].float()
                _Hd, _Wd = _dmap.shape[-2], _dmap.shape[-1]
                _vw = max(1, int(round((img.shape[-1] - float(padwh[idx][0])) / img.shape[-1] * _Wd)))
                _vh = max(1, int(round((img.shape[-2] - float(padwh[idx][1])) / img.shape[-2] * _Hd)))
                density_count = float(_dmap[..., :_vh, :_vw].sum())
            if len(outputs[idx]['pred_boxes'][-1]) == 0:
                num_objects_pred.append(0)
                density_counts.append(density_count)
                pred_boxes_batch.append(torch.empty((0, 4), device=device))
                pred_scores_batch.append(torch.empty(0, device=device))
                pred_match_scores_batch.append(torch.empty(0))
            else:
                v = outputs[idx]["box_v"]
                # swept values with --whole_sweep, else relative cutoff max/thr with an abs floor
                if args._sweep_abs_thr is not None:
                    v_thr = args._sweep_abs_thr
                    nms_iou = args._sweep_nms_iou
                else:
                    v_thr = torch.clamp(v.max() / thr, min=BOX_V_ABS_THRESHOLD)
                    nms_iou = 0.5
                mask = v > v_thr
                keep = ops.nms(
                    outputs[idx]["pred_boxes"][mask],
                    v[mask],
                    nms_iou,
                )
                boxes = outputs[idx]["pred_boxes"][mask][keep]
                scores = v[mask][keep]
                # mask-IoU used for ranking only
                match_scores = outputs[idx]["scores"][mask][keep]
                boxes = torch.clamp(boxes, 0, 1)

                # remove bboxes in padded area
                maxw = (img.shape[-1] - padwh[idx][0]).to(device)
                maxh = (img.shape[-2] - padwh[idx][1]).to(device)
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                valid = (center[:, 0] * img.shape[-1] < maxw) & (center[:, 1] * img.shape[-2] < maxh)
                boxes = boxes[valid]
                scores = scores[valid]
                match_scores = match_scores[valid]
                num_objects_pred.append(len(boxes))
                density_counts.append(density_count)
                pred_boxes_batch.append(boxes.detach().cpu())
                pred_scores_batch.append(scores.detach().cpu())
                pred_match_scores_batch.append(match_scores.detach().float().cpu())

        if device.type == 'cuda':
            torch.cuda.synchronize()
        _batch_infer_time = time.perf_counter() - _infer_t0
        # per-image time (batch_size is 1 here)
        _per_image_time = _batch_infer_time / max(1, batch_size)

        num_objects_gt = density_map.flatten(1).sum(dim=1)
        num_objects_pred_t = torch.tensor(num_objects_pred, dtype=torch.float32)

        finite_mask = torch.isfinite(num_objects_gt)
        if not finite_mask.all():
            for idx in range(batch_size):
                if not finite_mask[idx]:
                    skipped_nan.append(test_dataset.image_ids[ids[idx].item()])
            diffs = (num_objects_gt - num_objects_pred_t)[finite_mask]
        else:
            diffs = num_objects_gt - num_objects_pred_t

        ae += torch.abs(diffs).sum()
        se += torch.pow(diffs, 2).sum()

        for idx in range(batch_size):
            if not finite_mask[idx]:
                log(f"    image {test_dataset.image_ids[ids[idx].item()]}: "
                    f"non-finite gt count -- skipped from metrics")
                continue

            # match in the padded frame: pred in [0,1], GT already px
            pred_boxes = pred_boxes_batch[idx] * float(img.shape[-1])
            match_scores = pred_match_scores_batch[idx]  # mask-IoU ranking
            gt_boxes = gt_bboxes[idx].detach().cpu()
            gt_boxes = gt_boxes[gt_boxes.abs().sum(dim=1) > 0]
            bbox_rec = per_image_match(pred_boxes, match_scores, gt_boxes, COCO_IOU_THRS)
            bbox_records.append(bbox_rec)
            canon = bbox_rec['per_thr'][0]  # COCO_IOU_THRS[0] == 0.5
            iou_mean = float(canon['tp_iou'].mean().item()) if canon['tp_iou'].numel() else float('nan')
            giou_mean = float(canon['tp_giou'].mean().item()) if canon['tp_giou'].numel() else float('nan')

            per_image_results.append({
                'image_idx': ids[idx].item(),
                'image_id': test_dataset.image_ids[ids[idx].item()],
                'gt_count': num_objects_gt[idx].item(),
                # density_count is None unless --use_density
                'pred_count': num_objects_pred[idx],
                'box_count': len(pred_boxes_batch[idx]),
                'density_count': density_counts[idx],
                'tp': canon['n_tp'],
                'fp': canon['n_fp'],
                'fn': canon['n_fn'],
                'iou_tp': iou_mean,
                'giou_tp': giou_mean,
                'time_s': _per_image_time,
            })
            log(f"    image {test_dataset.image_ids[ids[idx].item()]}: "
                f"gt={num_objects_gt[idx].item():.0f}  pred={num_objects_pred[idx]}  "
                f"ae={abs(num_objects_gt[idx].item() - num_objects_pred[idx]):.0f}  "
                f"TP={canon['n_tp']} FP={canon['n_fp']} FN={canon['n_fn']}  "
                f"IoU={iou_mean:.3f} GIoU={giou_mean:.3f}  time={_per_image_time * 1000:.1f}ms")

        if visuals_dir is None or batch_i % args.visuals_every != 0:
            continue
        for idx in range(batch_size):
            pred_boxes = pred_boxes_batch[idx]

            # center-crop like __getitem__ so the overlay lines up with the /sf frame
            img_id = test_dataset.image_ids[ids[idx].item()]
            orig_img_path = test_dataset.img_dir / test_dataset.image_names[ids[idx].item()]
            vis_img = cv2.imread(str(orig_img_path))
            if vis_img is None:
                log(f"    [warn] could not read original image for {img_id} -- skipping visualization")
                continue
            _cs = getattr(test_dataset, 'mcac_crop_size', -1)
            if _cs != -1 and vis_img.shape[0] > _cs and vis_img.shape[1] > _cs:
                _oy = int((vis_img.shape[0] - _cs) / 2); _ox = int((vis_img.shape[1] - _cs) / 2)
                vis_img = vis_img[_oy:_oy + _cs, _ox:_ox + _cs]
            orig_h, orig_w = vis_img.shape[:2]
            sf = scaling_factor[idx].item()

            # colors (BGR): green = matched, red = FP, GT dashed b/w
            DOT_RADIUS = 4
            RED = (0, 0, 255)
            GREEN = (0, 255, 0)
            WHITE = (255, 255, 255)
            GT_THICKNESS = 1
            PRED_THICKNESS = 1

            # to original px: GT / sf, pred * img_sz / sf
            gt_boxes_img = gt_bboxes[idx].detach().cpu()
            gt_boxes_img = gt_boxes_img[gt_boxes_img.abs().sum(dim=1) > 0]  # drop zero padding
            gt_px = gt_boxes_img / sf
            img_sz = float(img.shape[-1])  # 1024
            pred_px = pred_boxes.clone() * img_sz / sf if pred_boxes.numel() else pred_boxes

            # greedy matching at IoU>=0.5, same rule as the metrics
            pairs = (
                match_pairs(pred_px, pred_match_scores_batch[idx].detach().cpu(), gt_px, 0.5)
                if pred_px.numel() else []
            )
            matched_pred = {pi for pi, _ in pairs}

            for gj in range(gt_px.shape[0]):
                x1, y1, x2, y2 = xyxy_int(gt_px[gj], orig_w, orig_h)
                draw_bw_dashed_rect(vis_img, x1, y1, x2, y2, thickness=GT_THICKNESS)

            for pi in range(pred_px.shape[0]):
                x1, y1, x2, y2 = xyxy_int(pred_px[pi], orig_w, orig_h)
                color = GREEN if pi in matched_pred else RED
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, PRED_THICKNESS)
                cv2.circle(vis_img, ((x1 + x2) // 2, (y1 + y2) // 2), DOT_RADIUS, color, -1)

            # connect matched centers
            for pi, gj in pairs:
                gx1, gy1, gx2, gy2 = xyxy_int(gt_px[gj], orig_w, orig_h)
                px1, py1, px2, py2 = xyxy_int(pred_px[pi], orig_w, orig_h)
                cv2.line(vis_img, ((gx1 + gx2) // 2, (gy1 + gy2) // 2),
                         ((px1 + px2) // 2, (py1 + py2) // 2), GREEN, 1)

            draw_label(vis_img, [
                (f"pred: {int(pred_px.shape[0])}", GREEN),
                (f"fp: {int(canon['n_fp'])}", RED),
                (f"gt: {int(gt_px.shape[0])}", WHITE),
            ])

            out_name = os.path.join(visuals_dir, f"{img_id}_pred.png")
            cv2.imwrite(out_name, vis_img)
            log(f"    saved visualization: {out_name}")

            # depth view for depth runs
            if save_depth_visual(model.module, img[idx], padwh[idx], orig_w, orig_h,
                                  os.path.join(visuals_dir, f"{img_id}_depth.png")):
                log(f"    saved depth visualization: {img_id}_depth.png")

            # density view when the density head is active
            if args.use_density and 'pred_density' in outputs[idx]:
                save_density_visual(outputs[idx]['pred_density'], padwh[idx],
                                     int(img.shape[-2]), int(img.shape[-1]),
                                     orig_w, orig_h,
                                     os.path.join(visuals_dir, f"{img_id}_density.png"))
                log(f"    saved density visualization: {img_id}_density.png")

    log("inference loop done -- computing final metrics...")
    n_total = len(test_dataset)
    n = len(per_image_results)  # excludes samples with non-finite gt counts
    if n == 0:
        raise RuntimeError(
            f"all {n_total} test samples had non-finite gt counts"
        )
    mae = ae.item() / n
    rmse = torch.sqrt(se / n).item()
    nae = sum(
        abs(r['gt_count'] - r['pred_count']) / max(r['gt_count'], 1)
        for r in per_image_results
    ) / n
    # SRE (MCAC Eq. 6), max(gt,1) clamps the division
    _n_gt0 = sum(1 for r in per_image_results if r['gt_count'] == 0)
    if _n_gt0:
        log(f"[warn] {_n_gt0} eval samples have gt_count == 0; their NAE/SRE terms "
            f"use the max(gt,1) clamp and diverge from the paper's exact Eq. 6.")
    sre = math.sqrt(sum(
        (r['gt_count'] - r['pred_count']) ** 2 / max(r['gt_count'], 1)
        for r in per_image_results
    ) / n)

    bbox_agg = bbox_aggregate(bbox_records, COCO_IOU_THRS)
    canon_agg = bbox_agg['per_thr'][0.5]

    # primary count = box-count, density-integral secondary
    primary_label = 'box-count'
    primary_rows = stratified_rows(per_image_results, 'pred_count', args.report_max_gt)
    density_rows = stratified_rows(per_image_results, 'density_count', args.report_max_gt) if args.use_density else None
    # full-set density aggregates for the Summary block
    dens_full = density_rows[0][1] if density_rows is not None else None

    print(f"Test MAE:  {mae:.3f}")
    print(f"Test MSE*: {rmse:.3f}   (RMSE)")
    print(f"Test NAE:  {nae:.3f}")
    print(f"Test SRE:  {sre:.3f}")
    if dens_full is not None:
        print(f"Density-integral (secondary): MAE={dens_full['mae']:.3f}  "
              f"MSE*={dens_full['rmse']:.3f}  NAE={dens_full['nae']:.3f}  "
              f"SRE={dens_full['sre']:.3f}")
    print(f"Count metrics ({primary_label}, count = len(boxes); GT-count-stratified):")
    for line in format_rows(primary_rows, with_sre=True):
        print(line)
    if density_rows is not None:
        print("Density-integral count metrics (SECONDARY, count = density.sum()):")
        for line in format_rows(density_rows, with_sre=True):
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

    model_path_abs = os.path.abspath(ckpt_path)
    txt_path = os.path.abspath(os.path.join(args.model_path, f'{args.model_name}{args.results_suffix}_results.txt'))
    log(f"writing results txt: {txt_path}")
    exp_tag, exp_mode = parse_tag_and_mode(args.model_name)
    epochs_str = str(args.ckpt_epochs) if args.ckpt_epochs else 'unknown (legacy un-suffixed ckpt)'
    best_epoch_str = str(best_epoch) if best_epoch is not None else 'unknown (not in ckpt)'
    _times = [r['time_s'] for r in per_image_results if r.get('time_s') is not None]
    total_infer_time = sum(_times)
    avg_infer_time = total_infer_time / len(_times) if _times else float('nan')
    if _times:
        print(f"Avg inference time/image: {avg_infer_time * 1000:.1f} ms "
              f"({len(_times) / total_infer_time:.2f} img/s); "
              f"total {total_infer_time:.2f} s over {len(_times)} images")

    with open(txt_path, 'w') as f:
        f.write(f"Model path: {model_path_abs}\n")
        f.write(f"Created: {now_str()}\n")
        f.write(f"Inference: WHOLE\n")
        if args.depth_zero_ablation:
            f.write("ZERO-DEPTH ABLATION: depth fusion reset to identity after loading -- "
                    "this is the checkpoint WITHOUT its learned depth contribution.\n")
        f.write("Bbox matching/AP ranked by SAM mask-IoU scores (was box_v -- "
                "counting/threshold/NMS unchanged, so count metrics are comparable to "
                "older runs but AP/P/R/F1 are not).\n")
        f.write(f"Config: backbone={args.backbone}, image_size={args.image_size}, "
                f"num_enc_layers={args.num_enc_layers}, emb_dim={args.emb_dim}, "
                f"num_objects={args.num_objects}, reduction={args.reduction}\n")
        f.write(f"\nSummary:\n")
        f.write(f"  Experiment tag:   {exp_tag if exp_tag is not None else 'unknown'}\n")
        f.write(f"  Mode:             {exp_mode if exp_mode is not None else ('zero' if args.zero_shot else 'few')}\n")
        f.write(f"  Eval split:       {args.test_split}\n")
        f.write(f"  Inference type:   whole\n")
        f.write(f"  Epochs (max):     {epochs_str}\n")
        f.write(f"  Best epoch:       {best_epoch_str}\n")
        f.write(f"  Test images: {n} of {n_total}\n")
        if skipped_nan:
            f.write(f"  Skipped (non-finite gt count): {len(skipped_nan)} -> {skipped_nan[:10]}{'...' if len(skipped_nan) > 10 else ''}\n")
        f.write(f"  Primary metric:   box-count (count = len(boxes)); density-integral is secondary\n")
        if args._sweep_abs_thr is not None:
            f.write(f"  Val sweep:        abs_thr={args._sweep_abs_thr:.2f} nms_iou={args._sweep_nms_iou:.2f} (calibrated on val, applied to test)\n")
        f.write(f"  Test MAE:    {mae:.3f}\n")
        f.write(f"  Test MSE*:   {rmse:.3f}   (RMSE)\n")
        f.write(f"  Test NAE:    {nae:.3f}\n")
        f.write(f"  Test SRE:    {sre:.3f}\n")
        if dens_full is not None:
            f.write(f"  Density-integral (secondary): MAE={dens_full['mae']:.3f}  "
                    f"MSE*={dens_full['rmse']:.3f}  NAE={dens_full['nae']:.3f}  "
                    f"SRE={dens_full['sre']:.3f}\n")
        if _times:
            f.write(f"  Avg inference time/image:  {avg_infer_time * 1000:.1f} ms  ({avg_infer_time:.4f} s)\n")
            f.write(f"  Total inference time:      {total_infer_time:.2f} s over {len(_times)} images\n")
            f.write(f"  Throughput:                {len(_times) / total_infer_time:.2f} images/s\n")
        f.write(f"\nCount metrics ({primary_label}, count = len(boxes); GT-count-stratified):\n")
        for line in format_rows(primary_rows, with_sre=True):
            f.write(line + "\n")
        if density_rows is not None:
            f.write(f"\nDensity-integral count metrics (SECONDARY, count = density.sum()):\n")
            for line in format_rows(density_rows, with_sre=True):
                f.write(line + "\n")
        f.write(f"\nBbox-quality metrics (greedy 1-to-1 matching by descending score):\n")
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
        # DPred/DAE columns only with --use_density, RGB-only keeps the
        # 11-column layout tools/results_lib.py parses
        header = (f"\n{'Idx':>6s}  {'Image ID':>16s}  {'GT':>6s}  {'Pred':>6s}  {'AE':>8s}  "
                  f"{'TP':>4s}  {'FP':>4s}  {'FN':>4s}  {'IoU':>6s}  {'GIoU':>7s}  {'Time(s)':>9s}")
        if args.use_density:
            header += f"  {'DPred':>8s}  {'DAE':>8s}"
        f.write(header + "\n")
        f.write('-' * (112 if args.use_density else 92) + '\n')
        for r in per_image_results:
            iou_s = f"{r['iou_tp']:.3f}" if r['iou_tp'] == r['iou_tp'] else "  nan"
            giou_s = f"{r['giou_tp']:.3f}" if r['giou_tp'] == r['giou_tp'] else "  nan"
            t_s = f"{r['time_s']:.4f}" if r.get('time_s') is not None else "  nan"
            row = (
                f"{r['image_idx']:>6d}  {r['image_id']:>16s}  {r['gt_count']:>6.0f}  "
                f"{r['pred_count']:>6.0f}  {abs(r['gt_count'] - r['pred_count']):>8.1f}  "
                f"{r['tp']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}  "
                f"{iou_s:>6s}  {giou_s:>7s}  {t_s:>9s}"
            )
            if args.use_density:
                dc = r.get('density_count')
                if dc is None:
                    row += f"  {'nan':>8s}  {'nan':>8s}"
                else:
                    row += f"  {dc:>8.2f}  {abs(r['gt_count'] - dc):>8.2f}"
            f.write(row + "\n")
    print(f"Results saved to: {txt_path}")
    log("inference complete.")


if __name__ == '__main__':
    _script_dir = _repo_root  # repo root, the script lives under training/

    parser = argparse.ArgumentParser('GECO2-MCAC-Inference', parents=[get_argparser()])
    parser.add_argument(
        '--results_suffix', type=str, default='_TEST_whole',
        help="suffix for the visuals dir and results txt",
    )
    parser.add_argument(
        '--ckpt_epochs', type=int, default=0,
        help="epoch count in the checkpoint filename, 0 loads the un-suffixed ckpt",
    )
    parser.add_argument(
        '--no_visuals', action='store_true',
        help="skip per-image PNGs, write only the results txt",
    )
    parser.add_argument(
        '--visuals_every', type=int, default=1,
        help="save a visualization only every Nth test image",
    )
    parser.add_argument(
        '--test_split', type=str, default='test', choices=['test', 'val'],
        help="MCAC evaluation split, metrics averaged over (image, class) pairs",
    )
    parser.add_argument(
        '--whole_sweep', type=int, default=0, choices=[0, 1],
        help="1 = tune abs box_v threshold and NMS IoU on val, apply to test",
    )

    parser.set_defaults(
        model_name='GECO2_MCAC',
        dataset='MCAC',
        data_path=str(_script_dir / 'MCAC'),
        model_path=str(_script_dir / 'models'),
        backbone='SAM',
        reduction=16,
        image_size=1024,
        num_enc_layers=3,
        emb_dim=256,
        num_heads=8,
        kernel_dim=3,
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
    print(args)
    print("model_name:", args.model_name)
    evaluate(args)
