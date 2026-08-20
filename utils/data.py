import argparse
import json
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms as T

from torchvision.ops import box_convert
from torchvision.transforms import functional as TVF
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from utils.depth_recipe import load_depth_map, load_depth_feats


def tiling_augmentation(img, bboxes, resize, jitter, tile_size, hflip_p, gt_bboxes=None, density_map=None, depth_map=None):
    """Zoom-out: tile the image into a 2x2 grid and resize back, so objects get
    smaller and denser. Boxes are rescaled and replicated per quadrant, density is
    renormalized to its pre-resize sum, depth is only tiled and resized."""
    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))

    img = make_tile(img, num_tiles, jitter=jitter)
    c, h, w = img.shape
    img = resize(img)

    if density_map is not None:
        # count field: tile geometrically, no jitter, keep the sum
        density_map = make_tile(density_map, num_tiles, jitter=None)
        original_sum = density_map.sum()
        density_map = resize(density_map)
        if density_map.sum() > 0:
            density_map = density_map / density_map.sum() * original_sum

    # depth: same tiling, no jitter, no renormalization ([0,1] field, not a count)
    if depth_map is not None:
        depth_map = make_tile(depth_map, num_tiles, jitter=None)
        depth_map = resize(depth_map)

    bboxes = bboxes / torch.tensor([w, h, w, h]) * resize.size[0]
    if gt_bboxes is not None:
        gt_bboxes_ = gt_bboxes / torch.tensor([w, h, w, h]) * resize.size[0]
        gt_bboxes_tiled = torch.cat([gt_bboxes_,
                                     gt_bboxes_ + torch.tensor([0, y_target // 2, 0, y_target // 2]),
                                     gt_bboxes_ + torch.tensor([x_target // 2, 0, x_target // 2, 0]),
                                     gt_bboxes_ + torch.tensor(
                                         [x_target // 2, y_target // 2, x_target // 2, y_target // 2])])

        return img, bboxes, density_map, gt_bboxes_tiled, depth_map

    return img, bboxes, density_map, depth_map


def select_exemplars(boxes, num_objects, randomize):
    """Pick num_objects exemplars from boxes (N, 4): random for train, first N
    otherwise. Repeat-pads the last box, all zeros when there are none."""
    n = boxes.shape[0]
    if n == 0:
        return torch.zeros(num_objects, 4)
    if n >= num_objects:
        idx = torch.randperm(n)[:num_objects] if randomize else torch.arange(num_objects)
    else:
        idx = torch.arange(n)
    sel = boxes[idx]
    if sel.shape[0] < num_objects:
        sel = torch.cat([sel, sel[-1:].expand(num_objects - sel.shape[0], -1)], dim=0)
    return sel


def _resize_pad_depth(depth_map, scaling_factor, padwh, size):
    """resize_and_pad's geometry applied to a depth map, so it stays letterboxed in
    lockstep with the RGB. No mass renormalisation; 0 in the pad area = farthest."""
    pad_width, pad_height = padwh
    size = int(size)
    d = torch.nn.functional.interpolate(depth_map.unsqueeze(0), scale_factor=scaling_factor,
                                        mode='bilinear', align_corners=False)
    d = torch.nn.functional.pad(d, (0, pad_width, 0, pad_height), mode='constant', value=0.0)[0]
    if d.shape[-2:] != (size, size):
        d = torch.nn.functional.interpolate(d.unsqueeze(0), size=(size, size),
                                            mode='bilinear', align_corners=False)[0]
    return d


def crop_augmentation(img, gt_bboxes, density_map, crop_px, depth_map=None):
    """Zoom-in: random square crop_px window in original pixels, matching the scale
    tiled inference feeds the model (the caller upscales it via resize_and_pad).
    Returns crop-local img, the boxes centred inside the window, density and depth."""
    _, h, w = img.shape
    crop_px = int(min(crop_px, h, w))
    x0 = int(torch.randint(0, w - crop_px + 1, (1,)).item())
    y0 = int(torch.randint(0, h - crop_px + 1, (1,)).item())

    img_crop = img[:, y0:y0 + crop_px, x0:x0 + crop_px].contiguous()
    density_crop = density_map[:, y0:y0 + crop_px, x0:x0 + crop_px].contiguous()
    depth_crop = (depth_map[:, y0:y0 + crop_px, x0:x0 + crop_px].contiguous()
                  if depth_map is not None else None)

    # keep only boxes centred in the crop, shifted to crop-local coords
    if gt_bboxes.numel():
        cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
        cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
        inside = (cx >= x0) & (cx < x0 + crop_px) & (cy >= y0) & (cy < y0 + crop_px)
        gt_c = gt_bboxes[inside].clone()
        gt_c[:, 0::2] = (gt_c[:, 0::2] - x0).clamp(0, crop_px)
        gt_c[:, 1::2] = (gt_c[:, 1::2] - y0).clamp(0, crop_px)
    else:
        gt_c = gt_bboxes
    return img_crop, gt_c, density_crop, depth_crop


def pad_collate_test(batch):
    (img, bboxes, density_map, ids, gt_bboxes, scaling_factor, padwh) = zip(*batch)
    gt_bboxes_pad = pad_sequence(gt_bboxes, batch_first=True, padding_value=0)
    img = torch.stack(img)
    bboxes = torch.stack(bboxes)
    density_map = torch.stack(density_map)
    ids = torch.stack(ids)

    scaling_factor = torch.tensor(scaling_factor)
    padwh = torch.tensor(padwh)
    return img, bboxes, density_map, ids, gt_bboxes_pad, scaling_factor, padwh

def xywh_to_x1y1x2y2(xywh):
    x, y, w, h = xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def pad_collate(batch):
    (img, bboxes, density_map, image_names, gt_bboxes) = zip(*batch)
    gt_bboxes_pad = pad_sequence(gt_bboxes, batch_first=True, padding_value=0)
    img = torch.stack(img)
    bboxes = torch.stack(bboxes)

    image_names = torch.stack(image_names)
    gt_bboxes = gt_bboxes_pad
    density_map = torch.stack(density_map)
    return img, bboxes, density_map, image_names, gt_bboxes

def resize_and_pad(img, bboxes, density_map=None, gt_bboxes=None, size=1024.0, zero_shot=False, train=False):
    # GT density at size // 2 = the 'simple' DensityHead's output at reduction=16.
    # DensityLoss reconciles any residual pred/GT size mismatch mass-preservingly.
    density_size = int(size) // 2
    density_resize = T.Resize((density_size, density_size), antialias=True)
    channels, original_height, original_width = img.shape
    longer_dimension = max(original_height, original_width)
    scaling_factor = size / longer_dimension
    scaled_bboxes = bboxes * scaling_factor
    if not zero_shot and not train:
        a_dim = ((scaled_bboxes[:, 2] - scaled_bboxes[:, 0]).mean() + (
                scaled_bboxes[:, 3] - scaled_bboxes[:, 1]).mean()) / 2
        # no real exemplars (a_dim == 0) -> skip the small-exemplar upscale
        if a_dim.item() > 0:
            scaling_factor = min(1.0, 80 / a_dim.item()) * scaling_factor
    resized_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scaling_factor, mode='bilinear',
                                                  align_corners=False)

    size = int(size)
    pad_height = max(0, size - resized_img.shape[2])
    pad_width = max(0, size - resized_img.shape[3])

    padded_img = torch.nn.functional.pad(resized_img, (0, pad_width, 0, pad_height), mode='constant', value=0)[0]
    if density_map is not None:
        original_sum = density_map.sum()
        _, w0, h0 = density_map.shape
        _, W, H = img.shape
        resized_density_map = torch.nn.functional.interpolate(density_map.unsqueeze(0), size=(W, H), mode='bilinear',
                                                            align_corners=False)
        resized_density_map = torch.nn.functional.interpolate(resized_density_map, scale_factor=scaling_factor,
                                                            mode='bilinear',
                                                            align_corners=False)
        padded_density_map = torch.nn.functional.pad(resized_density_map, (0, pad_width, 0, pad_height), mode='constant', value=0)[0]
        padded_density_map = density_resize(padded_density_map)
        # an all-zero density would divide by zero here
        if padded_density_map.sum() > 0:
            padded_density_map = padded_density_map / padded_density_map.sum() * original_sum

    bboxes = bboxes * torch.tensor([scaling_factor, scaling_factor, scaling_factor, scaling_factor]).to(bboxes.device)
    if gt_bboxes is None and density_map is None:
        return padded_img, bboxes, scaling_factor
    gt_bboxes = gt_bboxes * torch.tensor([scaling_factor, scaling_factor, scaling_factor, scaling_factor])
    return padded_img, bboxes, padded_density_map, gt_bboxes, scaling_factor, (pad_width, pad_height)



class IOCFish5kDataset(Dataset):
    """Dataset for IOCfish5k images with Pascal-VOC-style bbox XML annotations."""

    def __init__(
            self, data_path, image_size, split='train', num_objects=3,
            tiling_p=0.5, zero_shot=False, return_ids=False, training=False,
            train_ratio=0.8, val_ratio=0.1, seed=42,
            crop_p=0.0, crop_min_px=384, crop_max_px=768,
            density_sigma=8.0, density_adaptive_sigma=False, density_sigma_k=3,
            density_sigma_beta=0.3, density_sigma_min=2.0, density_sigma_max=15.0,
            depthmaps_dir=''
    ):
        self.data_path = Path(data_path)
        self.image_size = image_size
        # depth-map cache: non-empty -> __getitem__ appends depthmaps_dir/<id>.jpg
        # as the image's 4th channel, transformed in lockstep with the RGB
        self.depthmaps_dir = str(depthmaps_dir) if depthmaps_dir else ''
        # PCA-k decoder features (_pdf<k>), set post-construction, channels 4..3+k
        self.depthfeats_dir = ''
        self.split = split
        # GT density Gaussian: fixed sigma or adaptive per-point (see _make_density_map)
        self.density_sigma = density_sigma
        self.density_adaptive_sigma = density_adaptive_sigma
        self.density_sigma_k = density_sigma_k
        self.density_sigma_beta = density_sigma_beta
        self.density_sigma_min = density_sigma_min
        self.density_sigma_max = density_sigma_max
        self.num_objects = num_objects  # number of exemplars per image
        self.tiling_p = tiling_p
        # zoom-in crop aug, train split only, mutually exclusive with the tiling aug
        self.crop_p = crop_p
        self.crop_min_px = crop_min_px
        self.crop_max_px = crop_max_px
        self.zero_shot = zero_shot
        self.training = training
        self.horizontal_flip_p = 0.5
        self.resize = T.Resize((image_size, image_size), antialias=True)
        # GT density at image_size // 2, matching the 'simple' head at reduction=16
        self.density_map_size = image_size // 2
        self.density_resize = T.Resize((self.density_map_size, self.density_map_size), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)

        # XMLs live in bbox_annotations/, xml/, or straight in data_path
        if (self.data_path / 'bbox_annotations').is_dir():
            self.xml_dir = self.data_path / 'bbox_annotations'
        elif (self.data_path / 'xml').is_dir():
            self.xml_dir = self.data_path / 'xml'
        else:
            self.xml_dir = self.data_path
        self.img_dir = self.data_path / 'images' if (self.data_path / 'images').is_dir() else self.data_path

        # IDs with both an image and a parseable XML. Zero-object XMLs are kept
        # (fishless frames are valid samples with GT count 0)
        all_xml = sorted(self.xml_dir.glob('*.xml'))
        all_ids = []
        for xml_path in all_xml:
            img_path = self.img_dir / f'{xml_path.stem}.jpg'
            if not img_path.exists():
                img_path = self.img_dir / f'{xml_path.stem}.png'
            if not img_path.exists():
                continue
            try:
                self._parse_xml(xml_path)
            except ET.ParseError:
                continue
            all_ids.append(xml_path.stem)

        # Use predefined split files if present, otherwise fall back to random split
        split_file = self.data_path / f'{split}_id.txt'
        if split_file.exists():
            with open(split_file) as f:
                split_ids = {Path(line.strip()).stem for line in f if line.strip()}
            # order by all_ids, not the set: set order varies per process
            self.image_ids = [sid for sid in all_ids if sid in split_ids]
        else:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(all_ids))
            n = len(all_ids)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            if split == 'train':
                self.image_ids = [all_ids[i] for i in indices[:train_end]]
            elif split == 'val':
                self.image_ids = [all_ids[i] for i in indices[train_end:val_end]]
            else:  # test
                self.image_ids = [all_ids[i] for i in indices[val_end:]]

        # an empty split usually means the bbox XMLs are missing, stop here rather
        # than divide by zero in epoch 1
        if len(self.image_ids) == 0:
            n_xml = len(list(self.xml_dir.glob('*.xml')))
            raise RuntimeError(
                f"IOCFish5kDataset[{split}] is empty (0 samples). Looked for bbox XMLs in "
                f"{self.xml_dir} (found {n_xml}) and images in {self.img_dir}. Each kept ID "
                f"needs both an image and a parseable bbox XML. If n_xml=0 the "
                f"bbox_annotations/ dir is missing, restore the SAM3 XMLs to {self.data_path} "
                f"before training/inference."
            )

    @staticmethod
    def _parse_xml(xml_path: Path) -> list:
        """Return list of dicts with 'center' (cx, cy) and 'bbox' [xmin,ymin,xmax,ymax]."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        results = []
        for obj in root.findall('object'):
            pt = obj.find('point')
            bb = obj.find('bndbox')
            if bb is not None:
                bbox = [
                    int(bb.findtext('xmin', '0')),
                    int(bb.findtext('ymin', '0')),
                    int(bb.findtext('xmax', '0')),
                    int(bb.findtext('ymax', '0')),
                ]
            else:
                bbox = None
            if pt is not None:
                cx = int(pt.findtext('x', '0'))
                cy = int(pt.findtext('y', '0'))
            elif bbox is not None:
                # a few entries carry only <bndbox>, use the box centre
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
            else:
                cx = cy = None
            results.append({'center': (cx, cy), 'bbox': bbox})
        return results

    @staticmethod
    def _make_density_map(centers, img_h: int, img_w: int, density_sigma: float = 8.0,
                          density_adaptive_sigma: bool = False, density_sigma_k: int = 3,
                          density_sigma_beta: float = 0.3, density_sigma_min: float = 2.0,
                          density_sigma_max: float = 15.0) -> np.ndarray:
        """Gaussian density map whose sum equals the number of objects.

        Default: impulse map blurred by one sigma. Adaptive: per-point
        sigma = beta * mean k-NN distance, clamped. Each point carries unit mass.
        """
        density = np.zeros((img_h, img_w), dtype=np.float32)
        pts = [
            (int(min(max(cx, 0), img_w - 1)), int(min(max(cy, 0), img_h - 1)))
            for cx, cy in centers if cx is not None and cy is not None
        ]
        if not pts:
            return density

        if not density_adaptive_sigma:
            for cx, cy in pts:
                density[cy, cx] += 1.0
            count = float(density.sum())
            density = gaussian_filter(density, sigma=density_sigma)
            if density.sum() > 0:
                density = density / density.sum() * count
            return density

        # adaptive per-point sigma from k-NN spacing
        pts_arr = np.asarray(pts, dtype=np.float32)  # (N, 2) as (x, y)
        n = len(pts_arr)
        if n > 1:
            from scipy.spatial import cKDTree
            kk = min(density_sigma_k + 1, n)  # +1: query includes self
            dists, _ = cKDTree(pts_arr).query(pts_arr, k=kk)
            dists = np.atleast_2d(dists)
            nn_mean = dists[:, 1:].mean(axis=1) if kk > 1 else np.full(n, density_sigma)
            sigmas = np.clip(density_sigma_beta * nn_mean, density_sigma_min, density_sigma_max)
        else:
            sigmas = np.array([density_sigma], dtype=np.float32)

        for (cx, cy), s in zip(pts, sigmas):
            s = float(max(s, 1e-3))
            radius = int(3.0 * s) + 1  # truncate at 3 sigma
            x0, x1 = max(0, cx - radius), min(img_w, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(img_h, cy + radius + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            xs = np.arange(x0, x1, dtype=np.float32) - cx
            ys = np.arange(y0, y1, dtype=np.float32) - cy
            gx = np.exp(-(xs * xs) / (2.0 * s * s))
            gy = np.exp(-(ys * ys) / (2.0 * s * s))
            patch = np.outer(gy, gx)
            ps = patch.sum()
            if ps > 0:
                patch = patch / ps  # unit mass -> count preserved
            density[y0:y1, x0:x1] += patch
        return density

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_path = self.img_dir / f'{img_id}.jpg'
        if not img_path.exists():
            img_path = self.img_dir / f'{img_id}.png'
        xml_path = self.xml_dir / f'{img_id}.xml'

        img = T.ToTensor()(Image.open(img_path).convert('RGB'))

        annotations = self._parse_xml(xml_path)
        bbox_anns = [a for a in annotations if a['bbox'] is not None]
        centers = [a['center'] for a in annotations]

        # always (N, 4), empty images need (0, 4), not (0,), for the 2D indexing below
        if bbox_anns:
            gt_bboxes = torch.tensor(
                [a['bbox'] for a in bbox_anns], dtype=torch.float32
            )  # (N, 4)  [xmin, ymin, xmax, ymax]
        else:
            gt_bboxes = torch.zeros((0, 4), dtype=torch.float32)

        _, orig_h, orig_w = img.shape
        # cached depth (native res, [0,1], near=bright) loaded here so it goes through
        # the same geometric augs as the RGB, appended as the 4th channel at the end
        depth_map = (load_depth_map(self.depthmaps_dir, img_id, (orig_h, orig_w))
                     if self.depthmaps_dir else None)
        if depth_map is not None and getattr(self, 'depthfeats_dir', ''):
            depth_map = torch.cat(
                [depth_map, load_depth_feats(self.depthfeats_dir, img_id, (orig_h, orig_w))], dim=0)
        density_map = torch.from_numpy(
            self._make_density_map(
                centers, orig_h, orig_w,
                density_sigma=self.density_sigma, density_adaptive_sigma=self.density_adaptive_sigma,
                density_sigma_k=self.density_sigma_k, density_sigma_beta=self.density_sigma_beta,
                density_sigma_min=self.density_sigma_min, density_sigma_max=self.density_sigma_max,
            )
        ).unsqueeze(0)  # (1, H, W)

        # fishless frames have no exemplars to draw from
        if self.zero_shot or len(bbox_anns) == 0:
            bboxes = torch.zeros(self.num_objects, 4)
        else:
            bboxes = select_exemplars(
                gt_bboxes, self.num_objects, randomize=(self.split == 'train')
            )

        if self.split == 'train':
            tiled = False
            channels, original_height, original_width = img.shape
            longer_dimension = max(original_height, original_width)
            scaling_factor = self.image_size / longer_dimension
            bboxes_resized = bboxes * scaling_factor

            if self.crop_p > 0 and torch.rand(1) < self.crop_p:
                # zoom-in crop, then re-derive exemplars from the boxes inside it
                crop_px = int(torch.randint(
                    int(self.crop_min_px), int(self.crop_max_px) + 1, (1,)
                ).item())
                img, gt_bboxes, density_map, depth_map = crop_augmentation(
                    img, gt_bboxes, density_map, crop_px, depth_map=depth_map
                )
                if self.zero_shot or gt_bboxes.shape[0] == 0:
                    bboxes = torch.zeros(self.num_objects, 4)
                else:
                    bboxes = select_exemplars(gt_bboxes, self.num_objects, randomize=True)
                img = self.jitter(img)
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                    img, bboxes, density_map, gt_bboxes=gt_bboxes,
                    size=float(self.image_size), train=True,
                )
                if depth_map is not None:
                    depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)
            elif (    # only tile when the resized exemplars are big enough
                (bboxes_resized[:, 2] - bboxes_resized[:, 0]).mean() > 30
                and (bboxes_resized[:, 3] - bboxes_resized[:, 1]).mean() > 30
                and torch.rand(1) < self.tiling_p
            ):
                tiled = True
                tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
                img, bboxes, density_map, gt_bboxes, depth_map = tiling_augmentation(
                    img, bboxes, self.resize,
                    self.jitter, tile_size, self.horizontal_flip_p,
                    gt_bboxes=gt_bboxes, density_map=density_map, depth_map=depth_map
                )
            else:
                img = self.jitter(img)
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                    img, bboxes, density_map, gt_bboxes=gt_bboxes,
                    size=float(self.image_size), train=True,
                )
                if depth_map is not None:
                    depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)

            if not tiled and torch.rand(1) < self.horizontal_flip_p:
                img = TVF.hflip(img)
                density_map = TVF.hflip(density_map)
                if depth_map is not None:
                    depth_map = TVF.hflip(depth_map)
                bboxes[:, [0, 2]] = self.image_size - bboxes[:, [2, 0]]
                gt_bboxes[:, [0, 2]] = self.image_size - gt_bboxes[:, [2, 0]]
        else:
            img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                img, bboxes, density_map, gt_bboxes=gt_bboxes,
                size=float(self.image_size)
            )
            if depth_map is not None:
                depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)

        original_sum = density_map.sum()
        density_map = self.density_resize(density_map)
        if density_map.sum() > 0:
            density_map = density_map / density_map.sum() * original_sum

        gt_bboxes = torch.clamp(gt_bboxes, min=0, max=self.image_size)

        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # depth goes in as the image's 4th channel, the model splits it back off
        if depth_map is not None:
            if depth_map.shape[-2:] != img.shape[-2:]:
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(0), size=img.shape[-2:], mode='bilinear', align_corners=False)[0]
            img = torch.cat([img, depth_map.to(img.dtype)], dim=0)

        if self.split == 'train' or self.training:
            return img, bboxes, density_map, torch.tensor(idx), gt_bboxes
        else:
            return img, bboxes, density_map, torch.tensor(idx), gt_bboxes, torch.tensor(scaling_factor), padwh

    def id_to_imgpath(self):
        """{image_id: native RGB path} for this split, for the depth-cache pre-pass."""
        out = {}
        for img_id in self.image_ids:
            p = self.img_dir / f'{img_id}.jpg'
            if not p.exists():
                p = self.img_dir / f'{img_id}.png'
            out[img_id] = str(p)
        return out

    def __len__(self):
        return len(self.image_ids)


class FSC147DATASET(Dataset):
    def __init__(
            self, data_path, image_size, split='train', num_objects=3,
            tiling_p=0.5, zero_shot=False, return_ids=False, training=False,
            max_objects=None, crop_p=0.0, crop_min_px=192, crop_max_px=320,
            density_sigma=8.0, density_adaptive_sigma=False, density_sigma_k=3,
            density_sigma_beta=0.3, density_sigma_min=2.0, density_sigma_max=15.0,
            depthmaps_dir=''
    ):
        self.split = split
        self.data_path = data_path
        # depth-map cache, see IOCFish5kDataset: non-empty -> __getitem__ appends
        # depthmaps_dir/<id>.jpg as the image's 4th channel
        self.depthmaps_dir = str(depthmaps_dir) if depthmaps_dir else ''
        # PCA-k decoder features (_pdf<k>), set post-construction, channels 4..3+k
        self.depthfeats_dir = ''
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        # GT density: default loads FSC147's precomputed 512x512 npy,
        # density_adaptive_sigma=True rebuilds it from the points instead
        self.density_sigma = density_sigma
        self.density_adaptive_sigma = density_adaptive_sigma
        self.density_sigma_k = density_sigma_k
        self.density_sigma_beta = density_sigma_beta
        self.density_sigma_min = density_sigma_min
        self.density_sigma_max = density_sigma_max
        # zoom-in crop aug, train split only. The default range brackets the 256px
        # tiled-inference tile on the ~384px images.
        self.crop_p = crop_p
        self.crop_min_px = crop_min_px
        self.crop_max_px = crop_max_px
        self.image_size = image_size
        self.resize = T.Resize((image_size, image_size), antialias=True)
        # GT density at image_size // 2, matching the 'simple' head at reduction=16
        self.density_map_size = image_size // 2
        self.density_resize = T.Resize((self.density_map_size, self.density_map_size), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        self.return_ids = return_ids
        self.training = training

        with open(
                os.path.join(self.data_path, 'annotations', 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
                os.path.join(self.data_path, 'annotations', 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)

        # COCO labels live under instances_<split>.json, split names like 'test_coco'
        # fall back to the base split's file. Membership comes from the split json.
        coco_split = split
        coco_path = os.path.join(self.data_path, 'annotations', 'instances_' + coco_split + '.json')
        if not os.path.exists(coco_path):
            base = 'test' if 'test' in coco_split else ('val' if 'val' in coco_split else 'train')
            coco_path = os.path.join(self.data_path, 'annotations', 'instances_' + base + '.json')
        self.labels = COCO(coco_path)
        self.img_name_to_ori_id = self.map_img_name_to_ori_id()

        # optional train-only filter: drop images with >= max_objects GT objects.
        # val/test are never filtered, so metrics stay comparable.
        if max_objects is not None and split == 'train':
            kept = []
            n_missing = 0
            n_over = 0
            for name in self.image_names:
                coco_im_id = self.img_name_to_ori_id.get(name)
                if coco_im_id is None:
                    n_missing += 1
                    continue
                n = len(self.labels.getAnnIds([coco_im_id]))
                if n < max_objects:
                    kept.append(name)
                else:
                    n_over += 1
            print(
                f"[FSC147DATASET] max_objects={max_objects} filter on split='train': "
                f"kept {len(kept)}/{len(self.image_names)} "
                f"(dropped {n_over} over-cap, {n_missing} missing-coco-id)",
                flush=True,
            )
            self.image_names = kept

        # aliases so code written against IOCFish5kDataset works unchanged:
        # image_names[i] = on-disk filename, image_ids[i] = bare stem for logs
        self.image_ids = [os.path.splitext(n)[0] for n in self.image_names]
        from pathlib import Path as _Path
        self.img_dir = _Path(self.data_path) / 'images_384_VarV2'

        # an empty split would only surface as a divide-by-zero in epoch 1
        if len(self.image_names) == 0:
            raise RuntimeError(
                f"FSC147DATASET[{split}] is empty (0 samples) for data_path={self.data_path}. "
                f"Check the split JSON (Train_Test_Val_FSC_147.json) and images_384_VarV2/; "
                f"a max_objects filter that drops everything would also do this."
            )

    def get_gt_bboxes(self, idx):

        coco_im_id = self.img_name_to_ori_id[self.image_names[idx]]
        anno_ids = self.labels.getAnnIds([coco_im_id])
        annotations = self.labels.loadAnns(anno_ids)
        bboxes = []
        for a in annotations:
            bboxes.append(xywh_to_x1y1x2y2(a['bbox']))
        return bboxes

    def __getitem__(self, idx: int):
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        )).convert("RGB")

        gt_bboxes = torch.tensor(self.get_gt_bboxes(idx))

        img = T.Compose([
            T.ToTensor(),
        ])(img)

        # cached depth (native res, [0,1]), same geometric augs as the RGB, appended
        # as the 4th channel at the end
        _, _dh, _dw = img.shape
        depth_map = (load_depth_map(self.depthmaps_dir, os.path.splitext(self.image_names[idx])[0], (_dh, _dw))
                     if self.depthmaps_dir else None)
        if depth_map is not None and getattr(self, 'depthfeats_dir', ''):
            depth_map = torch.cat(
                [depth_map,
                 load_depth_feats(self.depthfeats_dir, os.path.splitext(self.image_names[idx])[0], (_dh, _dw))],
                dim=0)

        if self.zero_shot:
            bboxes = torch.zeros(self.num_objects, 4)
        else:
            bboxes = torch.tensor(
                self.annotations[self.image_names[idx]]['box_examples_coordinates'],
                dtype=torch.float32
            )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]

        if self.density_adaptive_sigma:
            # rebuild the GT density from the points with adaptive per-point sigma.
            # The points are in the images_384_VarV2 frame, so build at native
            # resolution and let the resize/crop pipeline carry it like the npy.
            _, _oh, _ow = img.shape
            _pts = self.annotations[self.image_names[idx]].get('points', [])
            density_map = torch.from_numpy(
                IOCFish5kDataset._make_density_map(
                    [(float(p[0]), float(p[1])) for p in _pts], _oh, _ow,
                    density_sigma=self.density_sigma, density_adaptive_sigma=True,
                    density_sigma_k=self.density_sigma_k, density_sigma_beta=self.density_sigma_beta,
                    density_sigma_min=self.density_sigma_min, density_sigma_max=self.density_sigma_max,
                )
            ).unsqueeze(0)
        else:
            density_map = torch.from_numpy(np.load(os.path.join(
                self.data_path,
                'gt_density_map_adaptive_512_512_object_VarV2',
                os.path.splitext(self.image_names[idx])[0] + '.npy',
            ))).unsqueeze(0)

        if self.split == 'train':
            tiled = False
            channels, original_height, original_width = img.shape
            longer_dimension = max(original_height, original_width)
            scaling_factor = self.image_size / longer_dimension
            bboxes_resized = bboxes * torch.tensor([scaling_factor, scaling_factor, scaling_factor, scaling_factor])

            if self.crop_p > 0 and torch.rand(1) < self.crop_p:
                # zoom-in crop. The density map is a fixed 512x512 while img and
                # gt_bboxes are in the ~384px on-disk frame, so bring the density into
                # the img frame first (sum-preserving) and crop with the same coords.
                crop_px = int(torch.randint(
                    int(self.crop_min_px), int(self.crop_max_px) + 1, (1,)
                ).item())
                _, Himg, Wimg = img.shape
                dsum = density_map.sum()
                density_imgframe = torch.nn.functional.interpolate(
                    density_map.unsqueeze(0), size=(Himg, Wimg),
                    mode='bilinear', align_corners=False,
                )[0]
                if density_imgframe.sum() > 0:
                    density_imgframe = density_imgframe / density_imgframe.sum() * dsum
                img, gt_bboxes, density_map, depth_map = crop_augmentation(
                    img, gt_bboxes, density_imgframe, crop_px, depth_map=depth_map
                )
                if self.zero_shot or gt_bboxes.shape[0] == 0:
                    bboxes = torch.zeros(self.num_objects, 4)
                else:
                    bboxes = select_exemplars(gt_bboxes, self.num_objects, randomize=True)
                img = self.jitter(img)
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                    img, bboxes, density_map, gt_bboxes=gt_bboxes,
                    size=float(self.image_size), train=True,
                )
                if depth_map is not None:
                    depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)
            elif (bboxes_resized[:, 2] - bboxes_resized[:, 0]).mean() > 30 and (
                    bboxes_resized[:, 3] - bboxes_resized[:, 1]).mean() > 30 and torch.rand(1) < self.tiling_p:
                tiled = True
                tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
                img, bboxes, density_map, gt_bboxes, depth_map = tiling_augmentation(
                    img, bboxes, self.resize,
                    self.jitter, tile_size, self.horizontal_flip_p, gt_bboxes=gt_bboxes,
                    density_map=density_map, depth_map=depth_map
                )
            else:
                img = self.jitter(img)
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(img, bboxes, density_map,
                                                                                            gt_bboxes=gt_bboxes,
                                                                                            size=float(self.image_size),
                                                                                            train=True)
                if depth_map is not None:
                    depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)

            if not tiled and torch.rand(1) < self.horizontal_flip_p:
                img = TVF.hflip(img)
                density_map = TVF.hflip(density_map)
                if depth_map is not None:
                    depth_map = TVF.hflip(depth_map)
                bboxes[:, [0, 2]] = self.image_size - bboxes[:, [2, 0]]
                gt_bboxes[:, [0, 2]] = self.image_size - gt_bboxes[:, [2, 0]]
        else:
            img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(img, bboxes, density_map,
                                                                                        gt_bboxes=gt_bboxes,
                                                                                        size=float(self.image_size))
            if depth_map is not None:
                depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)

        original_sum = density_map.sum()
        density_map = self.density_resize(density_map)
        if density_map.sum() > 0:
            density_map = density_map / density_map.sum() * original_sum
        gt_bboxes = torch.clamp(gt_bboxes, min=0, max=self.image_size)


        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # depth as the image's 4th channel, see IOCFish5kDataset
        if depth_map is not None:
            if depth_map.shape[-2:] != img.shape[-2:]:
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(0), size=img.shape[-2:], mode='bilinear', align_corners=False)[0]
            img = torch.cat([img, depth_map.to(img.dtype)], dim=0)

        if self.split == 'train' or self.training:
            return img, bboxes, density_map, torch.tensor(idx), gt_bboxes
        else:
            return img, bboxes, density_map, torch.tensor(idx), gt_bboxes, torch.tensor(scaling_factor), padwh

    def id_to_imgpath(self):
        """{image_id (stem): native RGB path} for this split, for the depth-cache pre-pass."""
        return {
            os.path.splitext(name)[0]: os.path.join(self.data_path, 'images_384_VarV2', name)
            for name in self.image_names
        }

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(self, ):
        all_coco_imgs = self.labels.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id


class MCACDataset(Dataset):
    """Multi-class Class-agnostic Counting (Hobley & Prisacariu, ECCV'24), a folder
    per sample under <data_path>/<split>/<numeric_id>/.

    One item per (image, class) pair, since the counter handles one class per query;
    the other classes are distractors. Frame: center-crop 1080 to mcac_crop_size,
    count instances with occlusion < mcac_occ_limit, exemplars from
    <split>_eval_bboxes.json at val/test and from the low-occlusion ones at train.
    GT density is rebuilt from the visible box centres instead of the released npy,
    which lives in a different frame. Same tuple contract as the other datasets.
    """

    def __init__(
            self, data_path, image_size, split='train', num_objects=3,
            tiling_p=0.5, zero_shot=False, return_ids=False, training=False,
            crop_p=0.0, crop_min_px=384, crop_max_px=640,
            density_sigma=8.0, density_adaptive_sigma=False, density_sigma_k=3,
            density_sigma_beta=0.3, density_sigma_min=2.0, density_sigma_max=15.0,
            depthmaps_dir='',
            mcac_crop_size=672, mcac_occ_limit=70, mcac_train_occ_limit=30,
            eval_bboxes_path=None,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.split_dir = self.data_path / split
        self.image_size = image_size
        # depth-map cache, see IOCFish5kDataset: one map per folder (shared by its
        # classes), keyed by the split-prefixed _cache_key, center-cropped with the RGB
        self.depthmaps_dir = str(depthmaps_dir) if depthmaps_dir else ''
        # PCA-k decoder features (_pdf<k>), set post-construction, channels 4..3+k
        self.depthfeats_dir = ''
        self.num_objects = num_objects  # number of exemplars per image
        self.tiling_p = tiling_p
        self.crop_p = crop_p
        self.crop_min_px = crop_min_px
        self.crop_max_px = crop_max_px
        self.zero_shot = zero_shot
        self.training = training
        self.return_ids = return_ids
        self.horizontal_flip_p = 0.5
        # GT density Gaussian, see IOCFish5kDataset._make_density_map
        self.density_sigma = density_sigma
        self.density_adaptive_sigma = density_adaptive_sigma
        self.density_sigma_k = density_sigma_k
        self.density_sigma_beta = density_sigma_beta
        self.density_sigma_min = density_sigma_min
        self.density_sigma_max = density_sigma_max
        self.resize = T.Resize((image_size, image_size), antialias=True)
        self.density_map_size = image_size // 2
        self.density_resize = T.Resize((self.density_map_size, self.density_map_size), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)

        # frame / occlusion conventions
        self.mcac_crop_size = int(mcac_crop_size)  # -1 = no crop (use full 1080)
        self.occ_limit = int(mcac_occ_limit)  # count/density/gt-boxes keep occ < this
        self.train_occ_limit = int(mcac_train_occ_limit)  # train exemplars drawn from occ < this
        # crop672 fields carry per-instance geometry already in the cropped frame.
        suffix = f'_crop{self.mcac_crop_size}' if self.mcac_crop_size != -1 else ''
        self._bbox_key = 'bboxes' + suffix
        self._occ_key = 'occlusions' + suffix
        self._json_name = 'info_with_occ_bbox.json'

        if not self.split_dir.is_dir():
            raise RuntimeError(
                f"MCACDataset[{split}]: split dir not found: {self.split_dir}. Expected "
                f"<data_path>/<split>/<numeric_id>/ folders under data_path={self.data_path}."
            )

        # val/test exemplar indices ship as <data_path>/<split>_eval_bboxes.json.
        # Without it (train) exemplars come from the occlusion percentages instead.
        if eval_bboxes_path is None:
            cand = self.data_path / f'{split}_eval_bboxes.json'
            eval_bboxes_path = str(cand) if cand.exists() else ''
        self.eval_bboxes = {}
        if eval_bboxes_path and os.path.exists(eval_bboxes_path):
            with open(eval_bboxes_path) as f:
                self.eval_bboxes = json.load(f)

        # per-(image, class) sample list: samples[i] = (folder_id, obj_id, eval_inds).
        # The class index is resolved from the json by obj_id in __getitem__, so
        # nothing depends on list ordering.
        self.samples = []
        self.image_ids = []  # unique per sample, for logs / output filenames
        self.image_names = []  # '<folder>/img.png', for visuals (img_dir / image_names[i])
        folder_ids = sorted(p.name for p in self.split_dir.iterdir() if p.is_dir())

        if self.eval_bboxes:
            # one entry per class straight from the eval-bbox spec, no json reads
            for fid in folder_ids:
                for entry in self.eval_bboxes.get(fid, []):
                    obj_id = entry.get('obj_id')
                    inds = entry.get('eval_bbox_inds')
                    self._add_sample(fid, obj_id, inds)
        else:
            # no eval-bbox file: keep every class with >=1 visible instance
            for fid in folder_ids:
                jp = self.split_dir / fid / self._json_name
                try:
                    with open(jp) as f:
                        info = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                for c in info.get('countables', []):
                    occ = np.asarray(c.get(self._occ_key, []))
                    if occ.size and int((occ < self.occ_limit).sum()) >= 1:
                        self._add_sample(fid, c.get('obj_id'), None)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"MCACDataset[{split}] is empty (0 (image,class) samples) under "
                f"{self.split_dir}. Checked {len(folder_ids)} folders; "
                f"eval_bboxes={'loaded' if self.eval_bboxes else 'absent'}. Each sample "
                f"needs a folder with {self._json_name} and a class with >=1 instance at "
                f"occlusion < {self.occ_limit}."
            )

        # aliases for the inference scripts' img_dir / image_names lookups.
        # img_name_to_ori_id is COCO-only and stays empty here.
        self.img_dir = self.split_dir
        self.img_name_to_ori_id = {}

    def _add_sample(self, folder_id, obj_id, eval_inds):
        self.samples.append((folder_id, obj_id, eval_inds))
        tag = (obj_id[:12] if obj_id else f'c{len(self.samples)}')
        self.image_ids.append(f'{folder_id}__{tag}')
        self.image_names.append(f'{folder_id}/img.png')

    @staticmethod
    def _boxes_xyxy(countable, bbox_key):
        """(N, 4) [x1,y1,x2,y2] float tensor from a countable's crop-frame boxes.
        MCAC stores each box as [[y1, y2], [x1, x2]], y range first. Columns are
        reordered and min/max-normalized so x1<=x2, y1<=y2."""
        raw = countable.get(bbox_key, [])
        if not len(raw):
            return torch.zeros((0, 4), dtype=torch.float32)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[1:] == (2, 2):
            yr = arr[:, 0, :]; xr = arr[:, 1, :]  # y-range, x-range
            x1 = xr.min(1); x2 = xr.max(1); y1 = yr.min(1); y2 = yr.max(1)
            out = np.stack([x1, y1, x2, y2], axis=1)
        else:
            out = arr.reshape(-1, 4)  # unexpected layout: flatten to (N,4)
        return torch.from_numpy(np.ascontiguousarray(out.astype(np.float32)))

    def _resolve_class(self, info, obj_id):
        """Index of the countable matching obj_id (falls back to the first class when
        obj_id is missing / not found), plus the countable dict."""
        countables = info.get('countables', [])
        if obj_id is not None:
            for i, c in enumerate(countables):
                if c.get('obj_id') == obj_id:
                    return i, c
        return (0, countables[0]) if countables else (0, {})

    def __getitem__(self, idx: int):
        folder_id, obj_id, eval_inds = self.samples[idx]
        sample_dir = self.split_dir / folder_id
        img = T.ToTensor()(Image.open(sample_dir / 'img.png').convert('RGB'))
        _, H0, W0 = img.shape

        # center crop to the MCAC frame the crop-frame json fields are already in,
        # offset int((H-crop)/2) as in the released loader
        cs = self.mcac_crop_size
        if cs != -1 and (H0 > cs and W0 > cs):
            oy, ox = int((H0 - cs) / 2), int((W0 - cs) / 2)
            img = img[:, oy:oy + cs, ox:ox + cs].contiguous()
        _, crop_h, crop_w = img.shape

        with open(sample_dir / self._json_name) as f:
            info = json.load(f)
        _, countable = self._resolve_class(info, obj_id)

        all_boxes = self._boxes_xyxy(countable, self._bbox_key)  # (N, 4) crop-frame px
        occ = np.asarray(countable.get(self._occ_key, []), dtype=np.float32)
        if occ.shape[0] != all_boxes.shape[0]:
            # a length mismatch would misalign the mask, count everything as visible
            occ = np.zeros((all_boxes.shape[0],), dtype=np.float32)
        vis = occ < self.occ_limit
        gt_bboxes = all_boxes[torch.from_numpy(vis)] if all_boxes.numel() else all_boxes  # visible instances

        # GT density from the visible box centres, built at crop resolution so it
        # goes through the same augs as the RGB
        centers = [
            ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)
            for b in gt_bboxes
        ]
        density_map = torch.from_numpy(
            IOCFish5kDataset._make_density_map(
                centers, crop_h, crop_w,
                density_sigma=self.density_sigma, density_adaptive_sigma=self.density_adaptive_sigma,
                density_sigma_k=self.density_sigma_k, density_sigma_beta=self.density_sigma_beta,
                density_sigma_min=self.density_sigma_min, density_sigma_max=self.density_sigma_max,
            )
        ).unsqueeze(0)  # (1, crop_h, crop_w)

        # exemplar boxes (num_objects, 4)
        if self.zero_shot:
            bboxes = torch.zeros(self.num_objects, 4)
        elif eval_inds is not None and all_boxes.numel():
            # val/test: the fixed 3 least-occluded instances
            keep = [i for i in eval_inds if 0 <= i < all_boxes.shape[0]]
            ex = all_boxes[keep] if keep else torch.zeros((0, 4))
            bboxes = select_exemplars(ex, self.num_objects, randomize=False)
        else:
            # train: random boxes with occ < train_occ_limit, else the visible ones
            if all_boxes.numel():
                low = all_boxes[torch.from_numpy(occ < self.train_occ_limit)]
                pool = low if low.shape[0] > 0 else gt_bboxes
            else:
                pool = all_boxes
            bboxes = select_exemplars(pool, self.num_objects, randomize=(self.split == 'train'))

        # cached depth (native full-res, [0,1]), center-cropped with the RGB and
        # appended as the 4th (+k) channel later. Keyed by folder, shared by its classes.
        depth_map = (load_depth_map(self.depthmaps_dir, self._cache_key(folder_id), (H0, W0))
                     if self.depthmaps_dir else None)
        if depth_map is not None and getattr(self, 'depthfeats_dir', ''):
            depth_map = torch.cat(
                [depth_map, load_depth_feats(self.depthfeats_dir, self._cache_key(folder_id), (H0, W0))], dim=0)
        if depth_map is not None and cs != -1 and (H0 > cs and W0 > cs):
            depth_map = depth_map[:, oy:oy + cs, ox:ox + cs].contiguous()

        if self.split == 'train':
            tiled = False
            longer_dimension = max(crop_h, crop_w)
            scaling_factor = self.image_size / longer_dimension
            bboxes_resized = bboxes * scaling_factor

            if self.crop_p > 0 and torch.rand(1) < self.crop_p:
                crop_px = int(torch.randint(
                    int(self.crop_min_px), int(self.crop_max_px) + 1, (1,)
                ).item())
                img, gt_bboxes, density_map, depth_map = crop_augmentation(
                    img, gt_bboxes, density_map, crop_px, depth_map=depth_map
                )
                if self.zero_shot or gt_bboxes.shape[0] == 0:
                    bboxes = torch.zeros(self.num_objects, 4)
                else:
                    bboxes = select_exemplars(gt_bboxes, self.num_objects, randomize=True)
                img = self.jitter(img)
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                    img, bboxes, density_map, gt_bboxes=gt_bboxes,
                    size=float(self.image_size), train=True,
                )
                if depth_map is not None:
                    depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)
            elif (
                (bboxes_resized[:, 2] - bboxes_resized[:, 0]).mean() > 30
                and (bboxes_resized[:, 3] - bboxes_resized[:, 1]).mean() > 30
                and torch.rand(1) < self.tiling_p
            ):
                tiled = True
                tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
                img, bboxes, density_map, gt_bboxes, depth_map = tiling_augmentation(
                    img, bboxes, self.resize,
                    self.jitter, tile_size, self.horizontal_flip_p,
                    gt_bboxes=gt_bboxes, density_map=density_map, depth_map=depth_map
                )
            else:
                img = self.jitter(img)
                img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                    img, bboxes, density_map, gt_bboxes=gt_bboxes,
                    size=float(self.image_size), train=True,
                )
                if depth_map is not None:
                    depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)

            if not tiled and torch.rand(1) < self.horizontal_flip_p:
                img = TVF.hflip(img)
                density_map = TVF.hflip(density_map)
                if depth_map is not None:
                    depth_map = TVF.hflip(depth_map)
                bboxes[:, [0, 2]] = self.image_size - bboxes[:, [2, 0]]
                gt_bboxes[:, [0, 2]] = self.image_size - gt_bboxes[:, [2, 0]]
        else:
            img, bboxes, density_map, gt_bboxes, scaling_factor, padwh = resize_and_pad(
                img, bboxes, density_map, gt_bboxes=gt_bboxes,
                size=float(self.image_size)
            )
            if depth_map is not None:
                depth_map = _resize_pad_depth(depth_map, scaling_factor, padwh, self.image_size)

        original_sum = density_map.sum()
        density_map = self.density_resize(density_map)
        if density_map.sum() > 0:
            density_map = density_map / density_map.sum() * original_sum

        gt_bboxes = torch.clamp(gt_bboxes, min=0, max=self.image_size)

        img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if depth_map is not None:
            if depth_map.shape[-2:] != img.shape[-2:]:
                depth_map = torch.nn.functional.interpolate(
                    depth_map.unsqueeze(0), size=img.shape[-2:], mode='bilinear', align_corners=False)[0]
            img = torch.cat([img, depth_map.to(img.dtype)], dim=0)

        if self.split == 'train' or self.training:
            return img, bboxes, density_map, torch.tensor(idx), gt_bboxes
        else:
            return img, bboxes, density_map, torch.tensor(idx), gt_bboxes, torch.tensor(scaling_factor), padwh

    def _cache_key(self, folder_id):
        """Depth-cache file id '<split>_<folder_id>', folder ids repeat across splits."""
        return f"{self.split}_{folder_id}"

    def id_to_imgpath(self):
        """{cache_key: native RGB path}, one entry per folder so the pre-pass generates
        a single map per image. Keyed like __getitem__'s load_depth_map call."""
        out = {}
        for folder_id, _obj_id, _inds in self.samples:
            key = self._cache_key(folder_id)
            if key not in out:
                out[key] = str(self.split_dir / folder_id / 'img.png')
        return out

    def __len__(self):
        return len(self.samples)
