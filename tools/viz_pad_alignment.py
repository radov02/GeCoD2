#!/usr/bin/env python
"""Padding-alignment diagnostics: real content vs letterbox pad on the RGB
Hiera maps and the depth tensors they fuse with, per depth-fusion mode.
Run from the GECO2 root, PNGs land in ../diagrams/pad_alignment/ (OUT_DIR).
"""
import os
import sys

# sam2/sam2 is the real package but models.sam_mask imports it as sam2.sam2
PWD = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # GECO2 root
sys.path.insert(0, os.path.join(PWD, "sam2"))
sys.path.insert(0, PWD)
import sam2
sys.modules.setdefault("sam2.sam2", sam2)

# local HF cache only
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils.arg_parser import get_argparser
from models.counter import build_model, _masked_group_norm_fn, _rgb_pad_valid_mask

torch.set_num_threads(min(20, os.cpu_count() or 8))
torch.manual_seed(0)

IMG_PATH = os.path.join(PWD, "IOCfish5kDataset/latest/IOCfish5k/images/1209.jpg")
BODY_CKPT = os.path.join(PWD, "CNTQG_multitrain_ca44.pth")
OUT_DIR = os.path.abspath(os.path.join(PWD, "..", "diagrams", "pad_alignment"))
IMAGE_SIZE = 1024
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
os.makedirs(OUT_DIR, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def letterbox(img_path, size=IMAGE_SIZE, force_hw=None):
    """Dataset-style resize_and_pad + Normalize. force_hw=(H0,W0) first resizes
    the source to an awkward aspect to stress the rounding."""
    img = T.ToTensor()(Image.open(img_path).convert("RGB"))
    if force_hw is not None:
        img = F.interpolate(img.unsqueeze(0), size=force_hw, mode="bilinear",
                            align_corners=False)[0]
    _, H0, W0 = img.shape
    sf = size / max(H0, W0)
    r = F.interpolate(img.unsqueeze(0), scale_factor=sf, mode="bilinear",
                      align_corners=False)
    vh, vw = r.shape[2], r.shape[3]
    x01 = F.pad(r, (0, size - vw, 0, size - vh), value=0.0)[0]     # zero-pad R/B
    x = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(x01).unsqueeze(0)
    return x, x01, vh, vw


def denorm(x01):
    """(3,S,S) [0,1] -> HWC uint8 for display."""
    return (x01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def feat_norm(t):
    """(1,C,H,W) or (C,H,W) -> per-pixel L2 norm over channels, HxW numpy float."""
    if t.dim() == 4:
        t = t[0]
    return t.float().pow(2).sum(0).sqrt().cpu().numpy()


def geo_valid_mask(R, vh, vw, size=IMAGE_SIZE):
    """Content mask at resolution R: a pixel counts as content iff its center
    maps inside the valid input rectangle (vh, vw)."""
    s = size / R
    rr = (np.arange(R) + 0.5) * s < vh
    cc = (np.arange(R) + 0.5) * s < vw
    return np.outer(rr, cc)


def resample_mask(mask_bool, R):
    """(1,1,h,w) bool -> nearest-resampled to RxR bool (as the code does with >0.5)."""
    m = F.interpolate(mask_bool.float(), size=(R, R), mode="nearest") > 0.5
    return m[0, 0].cpu().numpy()


def _erode_pad(geo, k):
    """Pad pixels at least k away from the content boundary and the frame border."""
    pad = torch.from_numpy((~geo).astype(np.float32))[None, None]
    inv = 1.0 - pad
    dil = F.max_pool2d(inv, kernel_size=2 * k + 1, stride=1, padding=k)  # dilate content
    safe = ((pad > 0.5) & (dil[0, 0] < 0.5)).numpy()
    safe[:k, :] = False; safe[-k:, :] = False
    safe[:, :k] = False; safe[:, -k:] = False
    return safe


def content_ness(t, geo):
    """Per-pixel 'real vs pad' strength: L2 distance from the mean deep-pad
    feature. Returns (H,W) in [0,1] and a boolean empirical-valid mask."""
    if t.dim() == 4:
        t = t[0]
    C, H, W = t.shape
    tf = t.float()
    safe = _erode_pad(geo, max(1, min(H, W) // 32))
    if safe.sum() < 4:
        safe = ~geo
    ref = tf[:, torch.from_numpy(safe)].mean(dim=1) # (C,)
    dist = (tf - ref.view(C, 1, 1)).pow(2).sum(0).sqrt().cpu().numpy()
    m = dist.max()
    cn = dist / m if m > 0 else dist
    emp = cn > 0.10
    return cn, emp


AGREE_CMAP = ListedColormap(["#2b2b2b", "#d64545", "#3fb56b"])  # 0 pad/pad,1 mismatch,2 real/real


def agreement_map(a, b):
    """3-class agreement of two boolean masks: 0 both-pad, 1 mismatch, 2 both-real."""
    a = a.astype(bool); b = b.astype(bool)
    out = np.zeros(a.shape, dtype=np.int32)
    out[a & b] = 2
    out[a ^ b] = 1
    n_mismatch = int((a ^ b).sum())
    return out, n_mismatch


def leak_agreement(valid_clean, emp):
    """2=clean real, 0=clean pad, 1=pad pixel carrying real content (a leak).
    Interior content-ness holes are not flagged, only leaks into pad."""
    valid_clean = valid_clean.astype(bool); emp = emp.astype(bool)
    out = np.where(valid_clean, 2, 0).astype(np.int32)
    leak = (~valid_clean) & emp
    out[leak] = 1
    return out, int(leak.sum())


def draw_boundary(ax, R, vh, vw, size=IMAGE_SIZE, color="cyan", ls="-", label=None):
    """Draw the continuous content->pad boundary lines (row/col) at resolution R."""
    bh = vh / size * R
    bw = vw / size * R
    if bh < R - 0.5:
        ax.axhline(bh - 0.5, color=color, ls=ls, lw=1.6, label=label)
    if bw < R - 0.5:
        ax.axvline(bw - 0.5, color=color, ls=ls, lw=1.6)


def imshow(ax, data, title, cmap="magma", vmin=None, vmax=None):
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def build(use_depth, **overrides):
    ap = get_argparser()
    args = ap.parse_args([])
    args.use_depth = use_depth
    args.zero_shot = True
    args.image_size = IMAGE_SIZE
    for k, v in overrides.items():
        setattr(args, k, v)
    model = build_model(args).to("cpu").eval()
    # pretrained CNT body just so the maps show real structure, fusion params stay at init
    state = torch.load(BODY_CKPT, map_location="cpu")
    if isinstance(state, dict) and "model" in state and not any(
            k.startswith("backbone") for k in state):
        state = state["model"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f"  [build use_depth={use_depth}] body load: missing={len(missing)} "
        f"unexpected={len(unexpected)} dts={model.depth_target_size} "
        f"src={model.depth_source} adapt={model.depth_adapt}")
    return model, args


def capture_feature_fusion(model, x, vh, vw):
    """Run backbone + depth pipeline as CNT.forward does for modes 1/2/5,
    capturing the tensors at the fusion point and the adapter in/out."""
    cap = {}
    adapter = model.depth_channel_adapt

    def hook(_m, inp, out):
        cap["adapter_in"] = inp[0].detach()
        cap["adapter_out"] = out.detach()
    h = adapter.register_forward_hook(hook)
    with torch.no_grad():
        feats = model._run_backbone(x)
        depth_features, depth_valid, depth_1ch = model._depth_features(x, None)
    h.remove()

    src = feats["vision_features"]
    l1, l2 = feats["backbone_fpn"][0], feats["backbone_fpn"][1]
    levels = []
    for label, rgb in [("src (vision_features)", src), ("l2 (backbone_fpn[1])", l2),
                       ("l1 (backbone_fpn[0])", l1)]:
        R = rgb.shape[-1]
        d = F.interpolate(depth_features, size=rgb.shape[-2:], mode="bilinear",
                          align_corners=False)
        dv = resample_mask(depth_valid, R) # code's nearest resample
        levels.append(dict(label=label, R=R, rgb=rgb, depth=d, depth_valid=dv))
    cap.update(dict(levels=levels, depth_features=depth_features,
                    depth_valid=depth_valid, depth_1ch=depth_1ch,
                    S=depth_features.shape[-1]))
    return cap


def render_fusion_figure(name, title, cap, vh, vw, fuse_desc):
    levels = cap["levels"]
    nrow = len(levels)
    fig, axes = plt.subplots(nrow, 5, figsize=(18, 3.5 * nrow))
    if nrow == 1:
        axes = axes[None, :]
    total_mismatch = 0
    for i, lv in enumerate(levels):
        R = lv["R"]; s = IMAGE_SIZE / R
        geo = geo_valid_mask(R, vh, vw)
        dv = lv["depth_valid"]
        cn_rgb, emp_rgb = content_ness(lv["rgb"], geo)
        agr, nmis = agreement_map(geo, dv)
        total_mismatch += nmis
        n_rows_geo = int(geo.any(1).sum()); n_cols_geo = int(geo.any(0).sum())

        ax = axes[i]
        imshow(ax[0], feat_norm(lv["rgb"]), f"RGB feat |.|  {lv['label']}\n"
               f"stride {s:.0f}  {R}x{R}  content {n_rows_geo}x{n_cols_geo}px")
        draw_boundary(ax[0], R, vh, vw, color="cyan", ls="-", label="geo (RGB)")
        draw_boundary(ax[0], R, vh, vw, color="magenta", ls="--")

        imshow(ax[1], cn_rgb, "RGB content-ness (1=real, 0=pad)", cmap="viridis")
        draw_boundary(ax[1], R, vh, vw, color="cyan", ls="-")

        imshow(ax[2], feat_norm(lv["depth"]), f"depth feat |d|  (resized to {R})")
        draw_boundary(ax[2], R, vh, vw, color="magenta", ls="--", label="depth-valid")

        imshow(ax[3], dv.astype(float), "depth valid mask (code)", cmap="gray", vmin=0, vmax=1)
        draw_boundary(ax[3], R, vh, vw, color="cyan", ls="-")

        imshow(ax[4], agr, f"alignment  geo(RGB) vs depth-valid\nmismatch = {nmis}px",
               cmap=AGREE_CMAP, vmin=0, vmax=2)
    fig.suptitle(f"{title}\n{fuse_desc}   |   image content vh={vh} vw={vw} of {IMAGE_SIZE}"
                 f"   depth_target_size S={cap['S']}   total boundary mismatch = {total_mismatch}px",
                 fontsize=11)
    _finish(fig, name)
    return total_mismatch


def render_sep_hiera_figure(name, model, x, vh, vw):
    with torch.no_grad():
        feats = model._run_backbone(x)
        depth_features, depth_valid, depth_1ch = model._depth_features(x, None)
        d_full = depth_features if depth_features.shape[-2:] == x.shape[-2:] else \
            F.interpolate(depth_features, size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats_d = model.depth_backbone(d_full)
    src, l1, l2 = feats["vision_features"], feats["backbone_fpn"][0], feats["backbone_fpn"][1]
    dsrc = feats_d["vision_features"]; dl1 = feats_d["backbone_fpn"][0]; dl2 = feats_d["backbone_fpn"][1]
    pairs = [("src (vision_features)", src, dsrc), ("l2 (backbone_fpn[1])", l2, dl2),
             ("l1 (backbone_fpn[0])", l1, dl1)]
    fig, axes = plt.subplots(3, 5, figsize=(18, 10.5))
    total = 0
    for i, (label, rgb, dep) in enumerate(pairs):
        R = rgb.shape[-1]; s = IMAGE_SIZE / R
        geo = geo_valid_mask(R, vh, vw)
        cn_rgb, emp_rgb = content_ness(rgb, geo)
        cn_dep, emp_dep = content_ness(dep, geo)
        # both streams see the same letterboxed frame, valid regions coincide
        agr, _ = agreement_map(geo, geo)
        leak_dep = int(((~geo) & emp_dep).sum())
        leak_rgb = int(((~geo) & emp_rgb).sum())
        ax = axes[i]
        imshow(ax[0], feat_norm(rgb), f"RGB Hiera |.|  {label}\nstride {s:.0f}  {R}x{R}")
        draw_boundary(ax[0], R, vh, vw, color="cyan")
        imshow(ax[1], cn_rgb, "RGB Hiera content-ness", cmap="viridis")
        draw_boundary(ax[1], R, vh, vw, color="cyan")
        imshow(ax[2], feat_norm(dep), "depth Hiera |.| (2nd frozen Hiera)")
        draw_boundary(ax[2], R, vh, vw, color="magenta", ls="--")
        imshow(ax[3], cn_dep, "depth Hiera content-ness", cmap="viridis")
        draw_boundary(ax[3], R, vh, vw, color="magenta", ls="--")
        imshow(ax[4], agr, "valid regions COINCIDE (both Hiera): 0px offset\n"
               f"(shared pad-grid texture: RGB {leak_rgb} / depth {leak_dep} px)",
               cmap=AGREE_CMAP, vmin=0, vmax=2)
    fig.suptitle("sep_hiera (mode 4): RGB Hiera pyramid + depth Hiera pyramid, summed per level\n"
                 "both are Hiera(the SAME letterboxed frame) -> valid regions identical BY CONSTRUCTION "
                 f"(0px boundary offset at every level)   vh={vh} vw={vw}", fontsize=11)
    _finish(fig, name)
    return 0


def render_input_concat_figure(name, title, rgb01, depth_full, rgb_valid, depth_valid, vh, vw, extra=""):
    """1024-res input-level concat: RGB padded image vs depth padded map before
    initial_depth_fuse. rgb_valid/depth_valid: (H,W) bool at IMAGE_SIZE."""
    R = IMAGE_SIZE
    agr, nmis = agreement_map(rgb_valid, depth_valid)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    axes[0].imshow(denorm(rgb01), interpolation="nearest")
    axes[0].set_title(f"RGB input (padded)\ncontent vh={vh} vw={vw}", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    draw_boundary(axes[0], R, vh, vw, color="cyan")
    imshow(axes[1], rgb_valid.astype(float), "RGB valid mask\n(letterbox rectangle)", cmap="gray", vmin=0, vmax=1)
    imshow(axes[2], depth_full, "depth map fed to concat\n(1-ch, as fused)", cmap="turbo")
    draw_boundary(axes[2], R, vh, vw, color="magenta", ls="--")
    imshow(axes[3], depth_valid.astype(float), "depth valid mask", cmap="gray", vmin=0, vmax=1)
    imshow(axes[4], agr, f"alignment RGB vs depth\nmismatch = {nmis}px", cmap=AGREE_CMAP, vmin=0, vmax=2)
    fig.suptitle(f"{title}\n{extra}   mismatch = {nmis}px", fontsize=11)
    _finish(fig, name)
    return nmis


def render_adapter_figure(name, cap, vh, vw):
    ain = cap["adapter_in"]; aout = cap["adapter_out"]; S = ain.shape[-1]
    geo = geo_valid_mask(S, vh, vw)
    dv = resample_mask(cap["depth_valid"], S) # the code's own valid region at S
    cn_in, emp_in = content_ness(ain, geo)
    cn_out, emp_out = content_ness(aout, geo)
    # adapter is size-preserving, check whether the 3x3 convs bleed content past dv
    agr, leak_out = leak_agreement(dv, emp_out)
    leak_in = int(((~dv) & emp_in).sum())
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    imshow(axes[0], feat_norm(ain), f"adapter INPUT |.|\n({ain.shape[1]}-ch depth_main, {S}x{S})")
    draw_boundary(axes[0], S, vh, vw, color="cyan")
    imshow(axes[1], cn_in, "INPUT content-ness", cmap="viridis")
    draw_boundary(axes[1], S, vh, vw, color="cyan")
    imshow(axes[2], feat_norm(aout), f"adapter OUTPUT |.|\n({aout.shape[1]}-ch feats, {S}x{S})")
    draw_boundary(axes[2], S, vh, vw, color="magenta", ls="--")
    imshow(axes[3], cn_out, "OUTPUT content-ness", cmap="viridis")
    draw_boundary(axes[3], S, vh, vw, color="magenta", ls="--")
    imshow(axes[4], agr, "valid region preserved (size-preserving conv)\n"
           f"INPUT pad-leak={leak_in}px  OUTPUT pad-leak={leak_out}px", cmap=AGREE_CMAP, vmin=0, vmax=2)
    fig.suptitle("_ConvDepthAdapter: does the valid (real) region survive the 3x3 conv stack?\n"
                 f"conv 3x3 -> masked GroupNorm -> GELU -> conv 3x3 (masked_conv={cap.get('mc', '?')})   "
                 f"vh={vh} vw={vw}   OUTPUT pad-leak = {leak_out}px (0 = boundary perfectly preserved)",
                 fontsize=11)
    _finish(fig, name)
    return leak_out


def render_rounding_stress(name):
    """Depth valid boundary vs RGB geometric boundary over many content heights,
    pure geometry. torch mode='nearest' maps dst index d -> floor(d*S/R), not
    the pixel-center floor((d+0.5)*S/R)."""
    strides = [16, 8, 4] # src, l2, l1 for image_size 1024
    S_choices = [256, 512] # parser-auto vs IOC-shell depth_target_size
    aspects = np.linspace(0.35, 1.0, 260) # content height fraction
    fig, axes = plt.subplots(len(S_choices), 1, figsize=(11, 7), squeeze=False)
    for si, S in enumerate(S_choices):
        ax = axes[si][0]
        for stg in strides:
            R = IMAGE_SIZE // stg
            mism = []
            for f in aspects:
                vh = int(round(f * IMAGE_SIZE))
                # RGB path: pixel-center-in-content extent (same geo as the figures)
                rgb_rows = int(((np.arange(R) + 0.5) * stg < vh).sum())
                # depth path: th-block at S, then the model's torch nearest resample S->R
                th = max(1, int(round(vh / IMAGE_SIZE * S)))
                mS = torch.zeros(1, 1, S, S); mS[..., :th, :] = 1.0
                mR = F.interpolate(mS, size=(R, R), mode="nearest") > 0.5
                depth_rows = int(mR[0, 0, :, 0].sum())
                mism.append(abs(rgb_rows - depth_rows))
            ax.plot(aspects, mism, label=f"stride {stg} ({R}x{R})", lw=1.4)
        ax.set_title(f"depth_target_size S={S}: |RGB geo rows - depth valid rows (real torch nearest)| "
                     "vs content height fraction", fontsize=9)
        ax.set_xlabel("content height / 1024"); ax.set_ylabel("row mismatch (px)")
        ax.set_ylim(-0.2, 2.2); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Rounding stress: RGB Hiera geometric boundary vs depth resample boundary\n"
                 "(depth rows via the model's real F.interpolate nearest; 0 = exact alignment, "
                 "spikes = 1px boundary rounding disagreements)", fontsize=11)
    _finish(fig, name)


def _finish(fig, name):
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path}")


def main():
    log("=== padding-alignment diagnostics ===")
    cases = [("iocfish_1080p", None), ("awkward_aspect_827x611", (611, 827))]

    # modes 1/2/5 share one depth pipeline, mode-1 model with the current defaults + hires
    log("[build] feature-fusion model (conv_depth_add, decoder+conv+group, +hires)")
    m1, a1 = build(1, depth_hires_fusion=1, depth_hires_norm=1)
    # mode 4 needs depth_feat_channels=3, idinit matches the real recipe
    log("[build] sep_hiera model (mode 4)")
    m4, a4 = build(4, depth_feat_channels=3, depth_fuse_identity_init=1, depth_feat_norm="group")

    summary = {}
    for cname, force_hw in cases:
        log(f"\n----- case {cname} force_hw={force_hw} -----")
        x, x01, vh, vw = letterbox(IMG_PATH, force_hw=force_hw)
        log(f"  input {tuple(x.shape)}  content vh={vh} vw={vw}")
        cap = capture_feature_fusion(m1, x, vh, vw)
        cap["mc"] = getattr(m1.depth_channel_adapt, "masked_conv", None)
        S = cap["S"]

        # conv_depth_add / conv_hiera / ffm: same feature geometry, different op
        summary[f"{cname}/conv_depth_add"] = render_fusion_figure(
            f"{cname}__conv_depth_add.png",
            "conv_depth_add (mode 1): src/l1/l2 = src + depth_proj(d)", cap, vh, vw,
            "fusion = ADD; depth_features resized (bilinear) to each level, no mask at the add")
        summary[f"{cname}/conv_hiera"] = render_fusion_figure(
            f"{cname}__conv_hiera.png",
            "conv_hiera (mode 2): src/l1/l2 = conv(cat[src, d])", cap, vh, vw,
            "fusion = CONCAT+conv; depth_features resized to each level, concatenated on channels")
        summary[f"{cname}/ffm"] = render_fusion_figure(
            f"{cname}__ffm.png",
            "ffm (mode 5): BiSeNet FeatureFusionModule per level (uses explicit v_src/v_l1/v_l2)", cap, vh, vw,
            "fusion = FFM; the depth-valid mask (col 4) IS consumed by the module's masked conv/GN/pool")

        # _ConvDepthAdapter input vs output
        summary[f"{cname}/adapter"] = render_adapter_figure(f"{cname}__conv_adapter_io.png", cap, vh, vw)

        # depth_dim (mode 3): input-level concat at full res
        d_full = F.interpolate(cap["depth_1ch"], size=(IMAGE_SIZE, IMAGE_SIZE),
                               mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        rgb_rect = geo_valid_mask(IMAGE_SIZE, vh, vw)
        dv_full = resample_mask(cap["depth_valid"], IMAGE_SIZE)
        summary[f"{cname}/depth_dim"] = render_input_concat_figure(
            f"{cname}__depth_dim.png",
            "depth_dim (mode 3): rgbd = initial_depth_fuse(cat[x(3), depth(1)]) at full 1024",
            x01, d_full, rgb_rect, dv_full, vh, vw,
            extra=f"depth upsampled S={S}->1024; RGB valid = letterbox rectangle")

        # _hires: replicate the hires depth normalization + concat geometry
        with torch.no_grad():
            d1 = F.interpolate(cap["depth_1ch"], size=(IMAGE_SIZE, IMAGE_SIZE),
                               mode="bilinear", align_corners=False)
            valid = _rgb_pad_valid_mask(x)
            rows = valid.any(dim=3, keepdim=True).to(torch.uint8)
            cols = valid.any(dim=2, keepdim=True).to(torch.uint8)
            valid_rect = (rows.flip(2).cummax(2).values.flip(2)
                          & cols.flip(3).cummax(3).values.flip(3)).bool()
            if getattr(m1, "depth_hires_norm", 0):
                d1n = _masked_group_norm_fn(d1, valid_rect, m1.initial_depth_fuse_norm)
            else:
                d1n = d1
        summary[f"{cname}/hires"] = render_input_concat_figure(
            f"{cname}__hires_concat.png",
            "_hires (modes 1/2/5): rgbd = initial_depth_fuse(cat[x(3), hires-depth(1)]) before Hiera",
            x01, d1n[0, 0].cpu().numpy(), rgb_rect, valid_rect[0, 0].cpu().numpy(), vh, vw,
            extra="depth: full-res, masked per-image GroupNorm (flip+cummax letterbox rect) then concat")

        # sep_hiera (mode 4)
        summary[f"{cname}/sep_hiera"] = render_sep_hiera_figure(f"{cname}__sep_hiera.png", m4, x, vh, vw)

    render_rounding_stress("rounding_stress_analysis.png")

    log("\n=== alignment summary (mismatch pixels) ===")
    for k in sorted(summary):
        log(f"  {k:42s} {summary[k]}")
    log(f"\nAll figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
