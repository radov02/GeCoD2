"""Detection metrics: IoU/GIoU of matched pairs, P/R/F1, COCO-style AP.
Greedy 1-to-1 matching by descending score, boxes xyxy in the same frame."""

from typing import List, Sequence

import torch

from utils.box_ops import box_iou, generalized_box_iou


# AP@[.5:.95] thresholds
COCO_IOU_THRS: List[float] = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def _empty(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32, device=device)


def per_image_match(
    pred_boxes: torch.Tensor,
    scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thrs: Sequence[float],
) -> dict:
    """Greedy 1-to-1 matching of preds to GTs, one pass per IoU threshold.
    Returns n_gt, n_pred, sorted_scores and per-threshold match stats."""
    device = pred_boxes.device if pred_boxes.numel() else torch.device('cpu')
    n_pred = int(pred_boxes.shape[0])
    n_gt = int(gt_boxes.shape[0])

    if n_pred > 0:
        order = torch.argsort(scores, descending=True)
        pred_sorted = pred_boxes[order]
        scores_sorted = scores[order]
    else:
        pred_sorted = pred_boxes
        scores_sorted = scores

    # empty cases, no IoU matrix
    if n_pred == 0 or n_gt == 0:
        per_thr = [
            dict(
                iou_thr=float(t),
                is_tp=torch.zeros(n_pred, dtype=torch.bool, device=device),
                tp_iou=_empty(device),
                tp_giou=_empty(device),
                n_tp=0,
                n_fp=n_pred,
                n_fn=n_gt,
            )
            for t in iou_thrs
        ]
        return dict(
            n_gt=n_gt,
            n_pred=n_pred,
            sorted_scores=scores_sorted,
            per_thr=per_thr,
        )

    # greedy loop runs on CPU
    iou_mat = box_iou(pred_sorted, gt_boxes)[0].detach().cpu()
    giou_mat = generalized_box_iou(pred_sorted, gt_boxes).detach().cpu()

    per_thr = []
    for t in iou_thrs:
        thr = float(t)
        gt_taken = torch.zeros(n_gt, dtype=torch.bool)
        is_tp_cpu = torch.zeros(n_pred, dtype=torch.bool)
        tp_iou_vals: List[torch.Tensor] = []
        tp_giou_vals: List[torch.Tensor] = []
        for i in range(n_pred):
            ious_i = iou_mat[i].clone()
            ious_i[gt_taken] = -1.0
            best_iou, best_j = ious_i.max(dim=0)
            if float(best_iou.item()) >= thr:
                is_tp_cpu[i] = True
                gt_taken[best_j] = True
                tp_iou_vals.append(iou_mat[i, best_j])
                tp_giou_vals.append(giou_mat[i, best_j])

        if tp_iou_vals:
            tp_iou = torch.stack(tp_iou_vals).to(device)
            tp_giou = torch.stack(tp_giou_vals).to(device)
        else:
            tp_iou = _empty(device)
            tp_giou = _empty(device)
        is_tp = is_tp_cpu.to(device)

        n_tp = int(is_tp.sum().item())
        per_thr.append(dict(
            iou_thr=thr,
            is_tp=is_tp,
            tp_iou=tp_iou,
            tp_giou=tp_giou,
            n_tp=n_tp,
            n_fp=n_pred - n_tp,
            n_fn=n_gt - n_tp,
        ))

    return dict(
        n_gt=n_gt,
        n_pred=n_pred,
        sorted_scores=scores_sorted,
        per_thr=per_thr,
    )


def match_pairs(
    pred_boxes: torch.Tensor,
    scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thr: float = 0.5,
) -> List[tuple]:
    """Same greedy matching, but returns the matched (pred_idx, gt_idx) pairs
    for visualization. Indices refer to the original unsorted rows."""
    n_pred = int(pred_boxes.shape[0])
    n_gt = int(gt_boxes.shape[0])
    if n_pred == 0 or n_gt == 0:
        return []

    order = torch.argsort(scores, descending=True)
    iou_mat = box_iou(pred_boxes[order], gt_boxes)[0].detach().cpu()

    gt_taken = torch.zeros(n_gt, dtype=torch.bool)
    pairs: List[tuple] = []
    for i in range(n_pred):
        ious_i = iou_mat[i].clone()
        ious_i[gt_taken] = -1.0
        best_iou, best_j = ious_i.max(dim=0)
        if float(best_iou.item()) >= iou_thr:
            gt_taken[best_j] = True
            pairs.append((int(order[i].item()), int(best_j.item())))
    return pairs


def _compute_ap(
    scores_concat: torch.Tensor,
    is_tp_concat: torch.Tensor,
    n_gt_total: int,
    n_recall_points: int = 101,
) -> float:
    """COCO-style AP, 101-point interpolation of the precision-recall curve."""
    if n_gt_total == 0:
        return float('nan')
    if scores_concat.numel() == 0:
        return 0.0

    order = torch.argsort(scores_concat, descending=True)
    is_tp_sorted = is_tp_concat[order].float()
    tp_cum = torch.cumsum(is_tp_sorted, dim=0)
    fp_cum = torch.cumsum(1.0 - is_tp_sorted, dim=0)
    recall = tp_cum / float(n_gt_total)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)

    # monotone-decreasing precision envelope from the right
    prec_env = torch.flip(
        torch.cummax(torch.flip(precision, dims=[0]), dim=0).values, dims=[0]
    )

    recall_points = torch.linspace(0, 1, n_recall_points, device=recall.device)
    # first index with recall >= rp, out of range counts as precision 0
    idx = torch.searchsorted(recall, recall_points, right=False)
    valid = (idx < recall.numel())
    idx_clamped = idx.clamp(max=recall.numel() - 1)
    sampled = prec_env[idx_clamped] * valid.float()
    return float(sampled.mean().item())


def aggregate(records: List[dict], iou_thrs: Sequence[float]) -> dict:
    """Aggregate per_image_match records into dataset-level metrics."""
    per_thr_out: dict = {}
    rounded_thrs = [round(float(t), 2) for t in iou_thrs]

    for ti, thr in enumerate(rounded_thrs):
        tp_iou_all: List[torch.Tensor] = []
        tp_giou_all: List[torch.Tensor] = []
        scores_all: List[torch.Tensor] = []
        is_tp_all: List[torch.Tensor] = []
        n_tp_total = 0
        n_fp_total = 0
        n_fn_total = 0
        n_gt_total = 0
        for r in records:
            rt = r['per_thr'][ti]
            n_tp_total += rt['n_tp']
            n_fp_total += rt['n_fp']
            n_fn_total += rt['n_fn']
            n_gt_total += r['n_gt']
            if rt['tp_iou'].numel():
                tp_iou_all.append(rt['tp_iou'].detach().cpu())
                tp_giou_all.append(rt['tp_giou'].detach().cpu())
            if r['n_pred'] > 0:
                scores_all.append(r['sorted_scores'].detach().cpu())
                is_tp_all.append(rt['is_tp'].detach().cpu())

        mean_iou = (
            float(torch.cat(tp_iou_all).mean().item()) if tp_iou_all else float('nan')
        )
        mean_giou = (
            float(torch.cat(tp_giou_all).mean().item()) if tp_giou_all else float('nan')
        )
        denom_p = n_tp_total + n_fp_total
        precision = (n_tp_total / denom_p) if denom_p > 0 else float('nan')
        denom_r = n_tp_total + n_fn_total
        recall = (n_tp_total / denom_r) if denom_r > 0 else float('nan')
        if precision != precision or recall != recall:
            f1 = float('nan')
        elif (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if scores_all:
            scores_concat = torch.cat(scores_all)
            is_tp_concat = torch.cat(is_tp_all)
        else:
            scores_concat = torch.empty(0)
            is_tp_concat = torch.empty(0, dtype=torch.bool)
        ap = _compute_ap(scores_concat, is_tp_concat, n_gt_total)

        per_thr_out[thr] = dict(
            mean_iou_tp=mean_iou,
            mean_giou_tp=mean_giou,
            precision=precision,
            recall=recall,
            f1=f1,
            ap=ap,
            n_tp=n_tp_total,
            n_fp=n_fp_total,
            n_fn=n_fn_total,
            n_gt=n_gt_total,
        )

    ap_50 = per_thr_out[0.5]['ap'] if 0.5 in per_thr_out else float('nan')
    if all(t in per_thr_out for t in COCO_IOU_THRS):
        ap_5095 = sum(per_thr_out[t]['ap'] for t in COCO_IOU_THRS) / len(COCO_IOU_THRS)
    else:
        ap_5095 = float('nan')

    return dict(per_thr=per_thr_out, AP_50=ap_50, AP_5095=ap_5095)


def per_image_counters(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    scores: torch.Tensor = None,
    iou_thr: float = 0.5,
) -> dict:
    """Single-threshold matcher for training-time val. Scalar sums only, so
    the caller can dist.all_reduce them."""
    n_pred = int(pred_boxes.shape[0])
    n_gt = int(gt_boxes.shape[0])
    if n_pred == 0 or n_gt == 0:
        return dict(
            tp_iou_sum=0.0,
            tp_giou_sum=0.0,
            n_tp=0,
            n_fp=n_pred,
            n_fn=n_gt,
            n_gt=n_gt,
        )

    iou_mat = box_iou(pred_boxes, gt_boxes)[0].detach().cpu()
    giou_mat = generalized_box_iou(pred_boxes, gt_boxes).detach().cpu()

    n_tp = 0
    tp_iou_sum = 0.0
    tp_giou_sum = 0.0

    if scores is not None:
        # descending score, greedy-claim best unclaimed GT
        order = torch.argsort(scores.detach().cpu(), descending=True)
        gt_taken = torch.zeros(n_gt, dtype=torch.bool)
        for i_sorted in range(n_pred):
            i = int(order[i_sorted].item())
            ious_i = iou_mat[i].clone()
            ious_i[gt_taken] = -1.0
            best_iou, best_j = ious_i.max(dim=0)
            if float(best_iou.item()) >= iou_thr:
                gt_taken[best_j] = True
                n_tp += 1
                tp_iou_sum += float(iou_mat[i, best_j].item())
                tp_giou_sum += float(giou_mat[i, best_j].item())
    else:
        # greedy by largest remaining IoU
        work = iou_mat.clone()
        for _ in range(min(n_pred, n_gt)):
            best_iou, flat = work.flatten().max(dim=0)
            if float(best_iou.item()) < iou_thr:
                break
            i = int(flat.item()) // n_gt
            j = int(flat.item()) % n_gt
            n_tp += 1
            tp_iou_sum += float(iou_mat[i, j].item())
            tp_giou_sum += float(giou_mat[i, j].item())
            work[i, :] = -1.0
            work[:, j] = -1.0

    return dict(
        tp_iou_sum=tp_iou_sum,
        tp_giou_sum=tp_giou_sum,
        n_tp=n_tp,
        n_fp=n_pred - n_tp,
        n_fn=n_gt - n_tp,
        n_gt=n_gt,
    )
