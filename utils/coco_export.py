"""COCO-format detection JSON for eval_bboxes.py, same conventions as
inference.py (integer xywh in original px, score = SAM mask-IoU)."""

import json

from torchvision import ops


def write_coco_predictions(path, per_image, categories=None):
    """per_image: (image_id, boxes_xyxy_origpx, scores) tuples. An images
    entry is written even with zero boxes."""
    preds = {
        "categories": categories or [{"name": "fg", "id": 1}],
        "images": [],
        "annotations": [],
    }
    anno_id = 1
    for image_id, boxes, scores in per_image:
        image_id = int(image_id)
        preds["images"].append({"id": image_id, "file_name": "None"})
        if boxes is None or len(boxes) == 0:
            continue
        xywh = ops.box_convert(boxes, "xyxy", "xywh").tolist()
        score_list = scores.tolist()
        for (x, y, w, h), s in zip(xywh, score_list):
            xi, yi, wi, hi = int(x), int(y), int(w), int(h)
            preds["annotations"].append({
                "id": anno_id,
                "image_id": image_id,
                "area": int(wi * hi),
                "bbox": [xi, yi, wi, hi],
                "category_id": 1,
                "score": float(s),
            })
            anno_id += 1
    with open(path, "w") as f:
        json.dump(preds, f)
    return len(preds["annotations"])
