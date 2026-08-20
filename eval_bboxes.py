# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import contextlib
import copy
import io
import itertools
import json
import logging
import os
import os.path as osp
import pickle as pkl
from collections import OrderedDict

from utils.arg_parser import get_argparser
from tools.timestamp import now_str
import numpy as np
import torch
try:
    from detectron2.evaluation.evaluator import DatasetEvaluator
    from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
    from detectron2.structures import BoxMode
    from detectron2.utils.logger import create_small_table
    from fvcore.common.file_io import PathManager
except ImportError:
    # detectron2 optional: pycocotools-only fallback
    from pycocotools.cocoeval import COCOeval

    class DatasetEvaluator:
        pass

    class PathManager:
        @staticmethod
        def get_local_path(path, *args, **kwargs):
            return path

    class BoxMode:
        XYXY_ABS = 0
        XYWH_ABS = 1

        @staticmethod
        def convert(box, from_mode, to_mode):
            b = np.asarray(box, dtype=float).copy()
            if from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                b[..., 2] -= b[..., 0]
                b[..., 3] -= b[..., 1]
            elif from_mode != to_mode:
                raise NotImplementedError(f"BoxMode.convert {from_mode}->{to_mode}")
            return b

    def create_small_table(d):
        return "\n".join(f"  {k}: {v}" for k, v in d.items())

from pycocotools.coco import COCO
from tabulate import tabulate
from torchvision.ops import box_iou


def _draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=1, dash=5, gap=4):
    """Axis-aligned dashed rectangle (OpenCV has no built-in)."""
    import cv2

    def _seg(p0, p1):
        (x0, y0), (xe, ye) = p0, p1
        dist = int(((xe - x0) ** 2 + (ye - y0) ** 2) ** 0.5)
        if dist == 0:
            return
        step = dash + gap
        for i in range(0, dist, step):
            s, e = i / dist, min((i + dash) / dist, 1.0)
            sx, sy = int(x0 + (xe - x0) * s), int(y0 + (ye - y0) * s)
            ex, ey = int(x0 + (xe - x0) * e), int(y0 + (ye - y0) * e)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)

    _seg((x1, y1), (x2, y1))
    _seg((x2, y1), (x2, y2))
    _seg((x2, y2), (x1, y2))
    _seg((x1, y2), (x1, y1))


def _draw_bw_dashed_rect(img, x1, y1, x2, y2, thickness=1, dash=5, gap=4):
    """GT-box style: white underlay plus black dashes, readable on any background."""
    import cv2
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness + 1)
    _draw_dashed_rect(img, x1, y1, x2, y2, (0, 0, 0), thickness=thickness, dash=dash, gap=gap)


class COCOEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        gt_json_file,
        pred_json_file,
        counting_gt_json_path,
        split="val",
        image_set=None,
        visualize_res=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = [
            "bbox",
        ]
        self._output_dir = output_dir
        self.counting_gt_json_path = counting_gt_json_path

        self._cpu_device = torch.device("cpu")

        # replace fewx with d2
        self._logger = logging.getLogger(__name__)

        gt_json_file = PathManager.get_local_path(gt_json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(gt_json_file)

        pred_json_file = PathManager.get_local_path(pred_json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self.pred_coco_api = COCO(pred_json_file)

        with open(gt_json_file) as f:
            tmp_gt = json.load(f)
        info_images = tmp_gt["images"]
        self.map_id_2_name = dict()
        self.map_name_2_id = dict()
        for info_image in info_images:
            img_id = info_image["id"]
            img_name = info_image["file_name"]
            self.map_id_2_name[img_id] = img_name
            self.map_name_2_id[img_name] = img_id

        with open(counting_gt_json_path) as f:
            self.point_annos = json.load(f)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self.counting_dict = dict()
        self._predictions = []
        self._image_set = image_set
        self.visualize_res = visualize_res
        # create the vis dir only when the built-in process() viz is on
        self._vis_dir = osp.join(self._output_dir or '.', "vis_res") if visualize_res else None
        if self._vis_dir:
            os.makedirs(self._vis_dir, exist_ok=True)
        # set by _eval_predictions(), read by visualize_coco_matches()
        self._coco_eval = None
        self.aps = []
        self.split = split
        self.relative_error = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        return tasks

    def process(self):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        if self._image_set is None:
            img_ids = self.pred_coco_api.getImgIds()
        else:
            img_ids = self._image_set
        print("number of images", len(img_ids))
        for img_id in img_ids:


            img_name = self.map_id_2_name[img_id]
            anno_ids = self.pred_coco_api.getAnnIds([img_id])
            point_anno = self.point_annos[img_name]["points"]
            pred_annos = self.pred_coco_api.loadAnns(anno_ids)
            img_info = self.pred_coco_api.loadImgs([img_id])

            prediction = {"image_id": img_id}
            results = []
            num_pred = len(pred_annos)
            for anno in pred_annos:
                box = anno["bbox"]
                x_cen, y_cen, w, h = box
                new_box = [x_cen, y_cen, w, h]
                result = {
                    "image_id": anno["image_id"],
                    "category_id": anno["category_id"],
                    "bbox": new_box,
                    "score": anno["score"],
                }

                results.append(result)
            num_pred = len(results)
            gt_anno_ids = self._coco_api.getAnnIds([img_id])
            gt_annos = self._coco_api.loadAnns(gt_anno_ids)

            ap = 0

            if self.visualize_res:
                import cv2
                img = cv2.imread(osp.join(os.path.dirname(self.counting_gt_json_path), 'images_384_VarV2', img_name))

                height, width, channels = img.shape
                height = 25 * len(pred_annos) + 10
                score_img = np.zeros((height, width, 3), np.uint8)
                score_img[:] = 255


                for idx, pred_anno in enumerate(pred_annos):
                    pred_box = pred_anno["bbox"]

                    x_cen, y_cen, w, h = pred_box

                    pred_box = [int(x_cen), int(y_cen), int(w), int(h)]

                    pred_x, pred_y, pred_w, pred_h = pred_box

                    pred_x, pred_y, pred_w, pred_h = int(pred_x), int(pred_y), int(pred_w), int(pred_h)
                    img = cv2.rectangle(img, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h),  (0, 165, 255), 2)



                vis_img_path = os.path.join(self._vis_dir, str(len(pred_annos)-len(gt_annos))+"_"+ img_name[:-4] + "_"+str(len(pred_annos))+".jpg")

                cv2.imwrite(vis_img_path, img)

            info = {
                "img_name": img_name,
                "img_id": img_id,
                "ap": ap,
                "count_gt": len(point_anno),
                "count_pred": num_pred,
            }
            self.aps.append(info)
            prediction["instances"] = results
            self._predictions.append(prediction)
            self.counting_dict[img_id] = {"gt": len(point_anno), "pred": num_pred}
            rel_err = abs(len(point_anno) - num_pred) / len(point_anno)
            self.relative_error.append(rel_err)

    def evaluate(self):
        predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self._eval_predictions(set(self._tasks), predictions)

        # Copy so the caller can do whatever with results
        cnt = 0
        SAE = 0  # sum of absolute errors
        SSE = 0  # sum of square errors
        NAE = 0
        SRE = 0
        preds = []
        gts = []

        for ii, (img_id, anno) in enumerate(self.counting_dict.items()):
            gt_cnt = anno["gt"]
            pred_cnt = anno["pred"]
            cnt = cnt + 1
            err = abs(gt_cnt - pred_cnt)

            preds.append(pred_cnt)
            gts.append(gt_cnt)
            SAE += err
            SSE += err ** 2
            NAE += err / gt_cnt
            SRE += err ** 2 / gt_cnt
        # print("Pred cnts ", preds)
        # print("gts       ", gts)
        # print(max(gts))
        print("number of images: {}".format(cnt))
        print("MAE: {:.2f}".format(SAE / cnt))
        print("RMSE: {:.2f}".format((SSE / cnt) ** 0.5))
        print("NAE: {:.4f}".format(NAE / cnt))
        print("SRE: {:.2f}".format((SRE / cnt) ** 0.5))
        print("Detect results")
        print(self._results)
        output_path = osp.join(self._output_dir, "each_img_infor"+self.split+".pkl")
        print("save to {}".format(output_path))

        with open(output_path, "wb") as handle:
            pkl.dump(self.aps, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print(10 * "**")
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            if self._image_set is not None:
                coco_eval = (
                    _evaluate_predictions_on_coco(self._coco_api, coco_results, task, self._image_set)
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
            else:
                coco_eval = (
                    _evaluate_predictions_on_coco(self._coco_api, coco_results, task,)
                    if len(coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )

            res = self._derive_coco_results(
                # coco_eval, task, class_names=self._metadata.get("thing_classes")
                coco_eval,
                task,
                class_names=["fg",],
            )
            self._results[task] = res
            # keep the eval object for visualize_coco_matches()
            self._coco_eval = coco_eval

    def visualize_coco_matches(self, iou_thr=0.5, out_dir=None, every=1, max_images=0):
        """Per-image TP/FP/FN visualizations from the COCO matcher at iou_thr
        (snapped to the nearest COCO threshold). GT dashed b/w, TP green, FP red."""
        import cv2

        if self._coco_eval is None:
            print("[vis] no COCOeval object (no predictions) -- skipping match visualizations")
            return

        p = self._coco_eval.params
        iou_thrs = np.asarray(p.iouThrs)
        t_idx = int(np.argmin(np.abs(iou_thrs - iou_thr)))
        snapped = float(iou_thrs[t_idx])
        if abs(snapped - iou_thr) > 1e-6:
            print(f"[vis] requested IoU={iou_thr:.3f} not a COCO threshold; snapped to {snapped:.2f}")
        iou_thr = snapped

        out_dir = out_dir or osp.join(self._output_dir or '.', f'cocovis_{self.split}')
        os.makedirs(out_dir, exist_ok=True)

        # evalImgs is a flat list (cat x areaRng x img), keep only the 'all' area
        # range and key by image id
        all_arng = list(p.areaRng[0])
        by_img = {}
        for e in self._coco_eval.evalImgs:
            if e is None or list(e['aRng']) != all_arng:
                continue
            by_img[e['image_id']] = e

        GREEN, RED, YELLOW = (0, 255, 0), (0, 0, 255), (0, 255, 255)
        img_root = osp.join(os.path.dirname(self.counting_gt_json_path), 'images_384_VarV2')

        n_done = 0
        for k, img_id in enumerate(sorted(by_img.keys())):
            if every > 1 and k % every != 0:
                continue
            if max_images and n_done >= max_images:
                break
            e = by_img[img_id]
            img_name = self.map_id_2_name[img_id]
            img = cv2.imread(osp.join(img_root, img_name))
            if img is None:
                print(f"[vis] could not read {img_name} -- skipping")
                continue

            gt_by_id = {a['id']: a['bbox']
                        for a in self._coco_api.loadAnns(self._coco_api.getAnnIds([img_id]))}
            dt_by_id = {a['id']: a['bbox']
                        for a in self.pred_coco_api.loadAnns(self.pred_coco_api.getAnnIds([img_id]))}

            gt_ids, dt_ids = list(e['gtIds']), list(e['dtIds'])
            gtm = np.asarray(e['gtMatches'])[t_idx] if gt_ids else np.array([])
            dtm = np.asarray(e['dtMatches'])[t_idx] if dt_ids else np.array([])

            # GT first (dashed b/w), cache centers for the connector lines
            gt_center = {}
            fn = 0
            for gind, gid in enumerate(gt_ids):
                x, y, w, h = [int(v) for v in gt_by_id[gid]]
                matched = gtm[gind] > 0
                fn += 0 if matched else 1
                _draw_bw_dashed_rect(img, x, y, x + w, y + h, thickness=1)
                gt_center[gid] = (x + w // 2, y + h // 2)

            # preds: green TP / red FP. dtMatches holds the matched GT id (0 = unmatched).
            tp = fp = 0
            for dind, did in enumerate(dt_ids):
                x, y, w, h = [int(v) for v in dt_by_id[did]]
                is_tp = dtm[dind] > 0
                tp += 1 if is_tp else 0
                fp += 0 if is_tp else 1
                color = GREEN if is_tp else RED
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.circle(img, (x + w // 2, y + h // 2), 3, color, -1)
                if is_tp and int(dtm[dind]) in gt_center:
                    cv2.line(img, gt_center[int(dtm[dind])],
                             (x + w // 2, y + h // 2), GREEN, 1)

            n_gt, n_dt = len(gt_ids), len(dt_ids)
            label = f"COCO match @IoU={iou_thr:.2f}  pred:{n_dt} gt:{n_gt}  TP:{tp} FP:{fp} FN:{fn}"
            cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

            out_name = (f"{osp.splitext(img_name)[0]}_iou{iou_thr:.2f}"
                        f"_pred{n_dt}_gt{n_gt}_TP{tp}_FP{fp}_FN{fn}.jpg")
            cv2.imwrite(osp.join(out_dir, out_name), img)
            n_done += 1

        print(f"[vis] wrote {n_done} COCO-matcher visualizations (IoU={iou_thr:.2f}) "
              f"to {os.path.abspath(out_dir)}")

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """

        metrics = {"bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],}[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table(results))
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")
        if class_names is None or len(class_names) <= 1:
            return results

        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa

        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d, tablefmt="pipe", floatfmt=".3f", headers=["category", "AP"] * (N_COLS // 2), numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        img_id (int): the image id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


class COCOevalMaxDets(COCOeval):
    """
    Modified version of COCOeval for evaluating AP with a custom
    maxDets (by default for COCO, maxDets is 100)
    """

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results given
        a custom value for  max_dets_per_image
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100000):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)
            )
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            # Evaluate AP using the custom limit on maximum detections per image
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=3000)
            stats[1] = _summarize(1, maxDets=3000, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=3000, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=3000, areaRng="medium")
            stats[4] = _summarize(1, maxDets=3000, areaRng="large")
            stats[5] = _summarize(0, maxDets=3000)
            stats[6] = _summarize(0, maxDets=3000, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=3000, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=3000, areaRng="medium")
            stats[9] = _summarize(0, maxDets=3000, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()


def _evaluate_predictions_on_coco(
    coco_gt, coco_results, iou_type, img_ids=None, max_dets_per_image=None, kpt_oks_sigmas=None
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0
    coco_dt = coco_gt.loadRes(coco_results)
    # coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    # For COCO, the default max_dets_per_image is [1, 10, 100].
    if max_dets_per_image is None:
        max_dets_per_image =  [10000, 10000, 10000]
    else:
        assert (
            len(max_dets_per_image) >= 3
        ), "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
        # In the case that user supplies a custom input for max_dets_per_image,
        # apply COCOevalMaxDets to evaluate AP with the custom input.
        if max_dets_per_image[2] != 100:
            coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)
    if iou_type != "keypoints":
        coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def get_args_parser():
    parser = argparse.ArgumentParser("GECO2", add_help=False)
    parser.add_argument("--input_folder", required=True, type=str)
    args = parser.parse_args()
    return args


def _eval_one(dataset_folder, split, pred_json_path, output_dir='',
              vis_matches=False, vis_iou_thr=0.5, vis_dir=None,
              vis_every=1, vis_max_images=0):
    """Run COCOeval for one split: GT instances_<split>.json vs pred_json_path.
    vis_matches also saves per-image TP/FP/FN visualizations at vis_iou_thr."""
    print(f"Evaluating on {split} set  (pred={pred_json_path})")
    coco_evaluator = COCOEvaluator(
        gt_json_file=dataset_folder + f"/annotations/instances_{split}.json",
        pred_json_file=pred_json_path,
        counting_gt_json_path=dataset_folder + "/annotation_FSC147_384.json",
        output_dir=output_dir,
        visualize_res=False,
        split=split,
    )
    coco_evaluator.process()
    coco_evaluator.evaluate()
    if vis_matches:
        coco_evaluator.visualize_coco_matches(
            iou_thr=vis_iou_thr,
            out_dir=vis_dir or (osp.splitext(pred_json_path)[0] + f'_cocovis_{split}'),
            every=vis_every,
            max_images=vis_max_images,
        )


if __name__ == '__main__':
    # conflict_handler='resolve': the eval-only flags below override same-named
    # flags from get_argparser() (--vis_every means something else here and must win)
    parser = argparse.ArgumentParser('GeCo', parents=[get_argparser()],
                                     conflict_handler='resolve')
    # optional single-split mode. Without --split: original val + test from
    # geco2_{val,test}.json in the CWD
    parser.add_argument('--split', type=str, default='',
                        choices=['', 'val', 'test', 'val_coco', 'test_coco'],
                        help="Evaluate only this split from --pred_json; empty = val + test.")
    parser.add_argument('--pred_json', type=str, default='',
                        help="Predictions JSON for --split (default geco2_<split>.json).")
    # COCO-matcher visualizations, off by default
    parser.add_argument('--vis_matches', action='store_true',
                        help="Save per-image visualizations colored by the COCO "
                             "matcher's TP/FP/FN verdicts.")
    parser.add_argument('--vis_iou_thr', type=float, default=0.5,
                        help="IoU threshold for the --vis_matches coloring; snapped "
                             "to the nearest COCO threshold (default 0.5).")
    parser.add_argument('--vis_dir', type=str, default='',
                        help="Output dir for --vis_matches images (default: alongside --pred_json).")
    parser.add_argument('--vis_every', type=int, default=1,
                        help="With --vis_matches, save only every Nth image (default 1 = all).")
    parser.add_argument('--vis_max_images', type=int, default=0,
                        help="With --vis_matches, cap the number of images saved (0 = no cap).")
    args = parser.parse_args()

    # timestamp the output
    print(f"Created: {now_str()}")

    dataset_folder = args.data_path
    vis_kw = dict(vis_matches=args.vis_matches, vis_iou_thr=args.vis_iou_thr,
                  vis_dir=(args.vis_dir or None), vis_every=max(1, args.vis_every),
                  vis_max_images=max(0, args.vis_max_images))
    if args.split:
        pred = args.pred_json or f'geco2_{args.split}.json'
        _eval_one(dataset_folder, args.split, pred, **vis_kw)
    else:
        _eval_one(dataset_folder, 'val', 'geco2_val.json', **vis_kw)
        _eval_one(dataset_folder, 'test', 'geco2_test.json', **vis_kw)