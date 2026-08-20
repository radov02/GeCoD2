"""Frozen monocular depth models (DAv2 dpt head by default, AnyDepth sdt head)."""

import argparse
import os
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.env import load_env
load_env()

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import hf_hub_download
from PIL import Image

from .AnyDepth import FeaturesToDepth
from .AnyDepth.sdt_head import SDTHead


_DINOV2_REPO = "facebookresearch/dinov2"
_DINOV2_CACHE_DIRNAME = "facebookresearch_dinov2_main"
_MAX_RETRIES = 6
_BACKOFF_BASE_SECONDS = 1.0


def _ensure_hub_dir():
    """Point torch.hub's cache at TORCH_HUB_DIR."""
    hub_dir = os.environ.get("TORCH_HUB_DIR") or os.path.expanduser("~/.cache/torch/hub")
    os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    return hub_dir


def _load_dinov2_arch(hub_name="dinov2_vitl14"):
    """DINOv2 arch via torch.hub, weights come from the SDT checkpoint."""
    hub_dir = _ensure_hub_dir()
    cached_repo = os.path.join(hub_dir, _DINOV2_CACHE_DIRNAME)

    if os.path.isfile(os.path.join(cached_repo, "hubconf.py")):
        print(f"[load_depther] reusing cached dinov2 repo at {cached_repo} (no network)")
        return torch.hub.load(cached_repo, hub_name, source="local", pretrained=False)

    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return torch.hub.load(
                _DINOV2_REPO, hub_name,
                source="github", pretrained=False,
                trust_repo=True, skip_validation=True,
            )
        except Exception as e:
            last_err = e
            if attempt == _MAX_RETRIES:
                break
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"[load_depther] torch.hub.load attempt {attempt}/{_MAX_RETRIES} "
                  f"failed ({type(e).__name__}: {e}); retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"failed to fetch {_DINOV2_REPO} after {_MAX_RETRIES} attempts "
        f"({type(last_err).__name__}: {last_err}), run once on a node with internet to warm the cache"
    ) from last_err


def _hf_download_with_retries(repo_id, filename):
    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            last_err = e
            if attempt == _MAX_RETRIES:
                break
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"[load_depther] hf_hub_download attempt {attempt}/{_MAX_RETRIES} "
                  f"failed ({type(e).__name__}: {e}); retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"failed to download {repo_id}/{filename} after {_MAX_RETRIES} attempts "
        f"({type(last_err).__name__}: {last_err}), run once on a node with internet to warm the cache"
    ) from last_err


class _BackboneWithPretrained(torch.nn.Module):
    """Mirrors the checkpoint's 'model.backbone.pretrained.*' attribute path."""

    def __init__(self, vit):
        super().__init__()
        self.pretrained = vit


# build configs keyed by checkpoint filename (DEPTH_CHECKPOINT)
# disparity: large = near, depth: large = far
_DEPTH_CONFIGS = {
    "depth_anything_v2_vitb.pth": dict(
        repo_id="depth-anything/Depth-Anything-V2-Base", head="dpt",
        dav2_encoder="vitb", dpt_features=128, dpt_out_channels=(96, 192, 384, 768),
        output="disparity",
    ),
    "depth_anything_v2_vits.pth": dict(
        repo_id="depth-anything/Depth-Anything-V2-Small", head="dpt",
        dav2_encoder="vits", dpt_features=64, dpt_out_channels=(48, 96, 192, 384),
        output="disparity",
    ),
    # default, 256-ch decoder tap for --depth_source decoder
    "depth_anything_v2_vitl.pth": dict(
        repo_id="depth-anything/Depth-Anything-V2-Large", head="dpt",
        dav2_encoder="vitl", dpt_features=256, dpt_out_channels=(256, 512, 1024, 1024),
        output="disparity",
    ),
    # floors the near field on close-ups, vit_norm=False matches its training wrapper
    "dav2_sdt_vitb.pth": dict(
        repo_id="AIGeeksGroup/AnyDepth", head="sdt",
        hub_name="dinov2_vitb14", embed_dim=768, layers=(2, 5, 8, 11),
        fusion_channels=128, use_cls_token=True, use_detail_enhancer=True,
        vit_norm=False, output="depth",
    ),
    # loads fully but output collapses on close-ups, reference only
    "da3_sdt_vitl.pth": dict(
        repo_id="AIGeeksGroup/AnyDepth", head="sdt",
        hub_name="dinov2_vitl14", embed_dim=1024, layers=(4, 11, 17, 23),
        fusion_channels=256, use_cls_token=False, use_detail_enhancer=False,
        vit_norm=False, output="depth",
    ),
}


def _remap_ckpt_key(k):
    """Remap ckpt keys to backbone.pretrained.* / head.* (da3 and dav2 prefixes differ)."""
    if k.startswith("model."):
        k = k[len("model."):]
    if k.startswith("depth_head."):
        k = "head." + k[len("depth_head."):]
    if k.startswith("pretrained."):
        k = "backbone.pretrained." + k[len("pretrained."):]
    return k


class SDTDepther(torch.nn.Module):
    """DINOv2 ViT/14 encoder + SDT decoder, normalized RGB in, 1-ch map out."""

    NATIVE_SIDE = 518  # ckpt pos_embed is 37x37 patches, default input_size

    def __init__(self, vit, head, cfg,
                 min_depth=0.001, max_depth=80.0,
                 target_size=None, input_size=None):
        super().__init__()
        self.backbone = _BackboneWithPretrained(vit)
        self.head = head
        self.cfg = cfg
        self.features_to_depth = FeaturesToDepth(min_depth=min_depth, max_depth=max_depth)
        self.target_size = target_size
        self.input_size = input_size or self.NATIVE_SIDE

    def forward(self, x, out_size=None):
        B, C, H, W = x.shape
        s = self.input_size / max(H, W)
        h14 = max(14, int(round(H * s / 14)) * 14)
        w14 = max(14, int(round(W * s / 14)) * 14)
        xr = F.interpolate(x, size=(h14, w14), mode="bilinear", align_corners=False)
        feats = self.backbone.pretrained.get_intermediate_layers(
            xr, n=self.cfg["layers"], reshape=True,
            return_class_token=self.cfg["use_cls_token"], norm=self.cfg["vit_norm"],
        )
        out = self.head(list(feats))  # (B, 1, 16*h14/14, 16*w14/14)
        out = self.features_to_depth(out)  # relu + min_depth
        # out_size is the caller's letterbox target, else square target_size or input size
        out_hw = out_size if out_size is not None else (
            (self.target_size, self.target_size) if self.target_size else (H, W))
        return F.interpolate(out, size=out_hw, mode="bilinear", align_corners=False)


def _load_dav2_class():
    """Import DepthAnythingV2 from the vendored clone (no __init__.py, so path hack)."""
    repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DepthAnythingV2")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from depth_anything_v2.dpt import DepthAnythingV2
    return DepthAnythingV2


class DAv2Depther(torch.nn.Module):
    """Official DepthAnythingV2 behind the same interface as SDTDepther."""

    def __init__(self, model, cfg,
                 min_depth=0.001, target_size=None,
                 input_size=None, return_path_1=False):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.min_depth = min_depth
        self.target_size = target_size
        self.input_size = input_size or 518  # dinoInSize
        self.return_path_1 = return_path_1
        self.decoder_feat_channels = cfg.get("dpt_features", 0)

    def forward(self, x, out_size=None):
        B, C, H, W = x.shape
        s = self.input_size / max(H, W)
        h14 = max(14, int(round(H * s / 14)) * 14)
        w14 = max(14, int(round(W * s / 14)) * 14)
        xr = F.interpolate(x, size=(h14, w14), mode="bilinear", align_corners=False)
        # out_size is the caller's letterbox target, else square target_size or input size
        out_hw = out_size if out_size is not None else (
            (self.target_size, self.target_size) if self.target_size else (H, W))
        if not self.return_path_1:
            d = self.model(xr).unsqueeze(1) + self.min_depth  # (B, 1, h14, w14)
            return F.interpolate(d, size=out_hw, mode="bilinear", align_corners=False)
        # path1_depth: DPTHead.forward returns (depth, path_1)
        d, feats = self.model(xr, return_path_1=True)
        depth = F.interpolate(d.unsqueeze(1) + self.min_depth, size=out_hw, mode="bilinear", align_corners=False)
        feats = F.interpolate(feats, size=out_hw, mode="bilinear", align_corners=False)
        return feats, depth


def load_depther(
    repo="AIGeeksGroup/AnyDepth",
    checkpoint=None,
    min_depth=0.001,
    max_depth=80.0,
    device="cuda",
    target_size=None,
    input_size=None,  # dinoInSize, None/0 = 518
    return_path_1=False,  # path1_depth, also return the decoder path_1 features (dpt head only)
    **legacy_kwargs,  # ignored
):
    """Build the requested depth model and fully load its checkpoint."""
    if legacy_kwargs:
        print(f"[load_depther] ignoring legacy kwargs: {sorted(legacy_kwargs)}")
    input_size = input_size or None  # 0 means default
    if checkpoint is None:
        checkpoint = os.environ.get("DEPTH_CHECKPOINT") or "depth_anything_v2_vitl.pth"
    if checkpoint not in _DEPTH_CONFIGS:
        raise ValueError(f"Unknown depth checkpoint '{checkpoint}'; known: {sorted(_DEPTH_CONFIGS)}")
    cfg = _DEPTH_CONFIGS[checkpoint]

    ckpt_path = _hf_download_with_retries(repo_id=cfg.get("repo_id", repo), filename=checkpoint)

    if cfg.get("head", "sdt") == "dpt":
        # official DepthAnythingV2, plain state dict, no key remap
        DepthAnythingV2 = _load_dav2_class()
        model = DepthAnythingV2(encoder=cfg["dav2_encoder"], features=cfg["dpt_features"],
                                out_channels=list(cfg["dpt_out_channels"]))
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"[depth] checkpoint did not load cleanly: missing={list(missing)[:8]} "
                f"(n={len(missing)}), unexpected={list(unexpected)[:8]} (n={len(unexpected)})."
            )
        depther = DAv2Depther(model, cfg, min_depth=min_depth, target_size=target_size,
                              input_size=input_size, return_path_1=return_path_1)
        print(f"[depth] loaded {checkpoint} FULLY: missing=0 unexpected=0 -- "
              f"1-ch {cfg['output']} model (official DepthAnythingV2, dpt head); "
              f"path1_depth={'ON' if return_path_1 else 'OFF'} "
              f"(decoder path_1 tap, {depther.decoder_feat_channels} ch) "
              f"dinoInSize={depther.input_size} (dino_input_size)")
        return depther.to(device).eval()

    if return_path_1:
        raise ValueError(
            f"path1_depth (--depth_source decoder) needs a DPT-head depth model (the decoder "
            f"path_1 tap); checkpoint '{checkpoint}' is an SDT head. Use a depth_anything_v2_vit* checkpoint."
        )
    vit = _load_dinov2_arch(cfg["hub_name"])
    head = SDTHead(
        in_channels=[cfg["embed_dim"]] * 4,
        fusion_channels=cfg["fusion_channels"],
        n_output_channels=1,
        use_cls_token=cfg["use_cls_token"],
        use_detail_enhancer=cfg["use_detail_enhancer"],
    )
    depther = SDTDepther(vit, head, cfg, min_depth=min_depth, max_depth=max_depth,
                         target_size=target_size, input_size=input_size)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt.get("state_dict", ckpt)
    state = {_remap_ckpt_key(k): v for k, v in state.items()}
    missing, unexpected = depther.load_state_dict(state, strict=False)
    allowed_missing = {"backbone.pretrained.mask_token"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"[depth] checkpoint did not load cleanly: missing={list(missing)[:8]} "
            f"(n={len(missing)}), unexpected={list(unexpected)[:8]} (n={len(unexpected)})."
        )
    print(f"[depth] loaded {checkpoint} FULLY: missing={len(missing)} "
          f"(only {sorted(set(missing))}) unexpected={len(unexpected)} -- "
          f"1-ch {cfg['output']} model (sdt head)")

    return depther.to(device).eval()


def preprocess(image):
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet stats
    ])
    return tfm(image).unsqueeze(0)


@torch.no_grad()
def predict_depth(depther, image_path):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).to(next(depther.parameters()).device)
    depth = depther(x).squeeze().cpu().numpy()
    return depth


def save_depth_visualization(depth, out_path):
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    Image.fromarray((d * 255).astype(np.uint8)).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="path to input RGB image")
    parser.add_argument("--checkpoint", type=str, default=None, choices=sorted(_DEPTH_CONFIGS),
                        help="depth model; default $DEPTH_CHECKPOINT, else depth_anything_v2_vitl.pth")
    parser.add_argument("--out", type=str, default="depth.png", help="visualization output path")
    parser.add_argument("--npy", type=str, default=None, help="optional path to dump raw depth as .npy")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    depther = load_depther(checkpoint=args.checkpoint, device=args.device)
    depth = predict_depth(depther, args.image)
    save_depth_visualization(depth, args.out)
    if args.npy:
        np.save(args.npy, depth)
    print(f"depth: shape={depth.shape} min={depth.min():.3f} max={depth.max():.3f} -> {args.out}")


if __name__ == "__main__":
    main()
