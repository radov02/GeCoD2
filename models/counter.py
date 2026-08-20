import numpy as np
import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from Depther.infer_depth import load_depther
from utils.box_ops import boxes_with_scores
from .prompt_encoder import PromptEncoder
from .query_generator import C_base
from .regression_head import DensityHead, DensityDecoder, DensityGuidedDetection
from .sam_mask import MaskProcessor


# depth-fusion modes (args.use_depth), one experiment dir per nonzero value
DEPTH_OFF = 0
CONV_DEPTH_ADD = 1
CONV_HIERA = 2
DEPTH_DIM = 3
SEP_HIERA = 4
FFM_FUSE = 5


# trainable depth-fusion param prefixes (not the frozen depth_model/depth_backbone).
# used by the --depth_fuse_lr group, the probe phase and the deviation log.
DEPTH_FUSION_PARAM_PREFIXES = (
    'depth_proj_', # mode 1
    'depth_fuse_src', 'depth_fuse_l1', 'depth_fuse_l2', # mode 2
    'depth_ffm_', # mode 5
    'initial_depth_fuse', # mode 3 + hires fusion (prefix also catches initial_depth_fuse_norm)
    'depth_channel_adapt', # depth adapter
    'depth_pyramid_gamma', # mode 4 gate
    'depth_feat_norm_layer',
)


def is_depth_fusion_param(name: str) -> bool:
    """True for depth-fusion param names (with or without the DDP 'module.' prefix)."""
    if name.startswith('module.'):
        name = name[len('module.'):]
    return name.startswith(DEPTH_FUSION_PARAM_PREFIXES)


def _pick_norm_groups(num_channels: int, override: int = 0) -> int:
    """First of (32,16,8,4,2,1) that divides num_channels, unless override>0."""
    if override > 0:
        return override
    for g in (32, 16, 8, 4, 2, 1):
        if num_channels % g == 0:
            return g
    return 1


def _sobel_laplacian_kernels() -> torch.Tensor:
    """Fixed 3x3 [Sobel-dx, Sobel-dy, Laplacian] kernels, stacked (3,1,3,3)."""
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    sobel_y = sobel_x.t()
    laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    return torch.stack([sobel_x, sobel_y, laplacian]).unsqueeze(1)  # (3,1,3,3)


def _masked_group_norm_fn(feats, valid_mask, gn):
    """GroupNorm with mean/var over valid (non-pad) pixels only, using gn's
    groups/eps/affine params (gn.forward is never called). pad pixels come out
    at the affine bias."""
    B, C, H, W = feats.shape
    G = gn.num_groups
    cg = C // G
    orig_dtype = feats.dtype
    x = feats.float().view(B, G, cg, H, W)
    m = valid_mask.view(B, 1, 1, H, W).float() # 1 = valid
    valid_pix = m.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    denom = (valid_pix * cg).view(B, 1) # elements per (sample, group)
    mean = (x * m).sum(dim=(2, 3, 4)) / denom # (B, G)
    mean_ = mean.view(B, G, 1, 1, 1)
    var = (((x - mean_) ** 2) * m).sum(dim=(2, 3, 4)) / denom
    x = (x - mean_) / torch.sqrt(var.view(B, G, 1, 1, 1) + gn.eps)
    x = (x * m).view(B, C, H, W) # pad -> 0
    if gn.affine:
        x = x * gn.weight.float().view(1, C, 1, 1) + gn.bias.float().view(1, C, 1, 1)
    return x.to(orig_dtype)


def _rgb_pad_valid_mask(x):
    """(B,1,H,W) valid mask of a normalised RGB batch. exact-black content also
    tests as pad, _letterbox_rect_mask fixes that up."""
    pad_vals = x.new_tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(1, 3, 1, 1)
    is_pad = ((x - pad_vals).abs().amax(dim=1, keepdim=True) < 1e-4) | (x.abs().amax(dim=1, keepdim=True) < 1e-12)
    return ~is_pad


def _letterbox_rect_mask(valid):
    """Grow a per-pixel valid mask to the full letterbox rectangle so exact-black
    content pixels survive."""
    rows = valid.any(dim=3, keepdim=True).to(torch.uint8) # (B,1,H,1)
    cols = valid.any(dim=2, keepdim=True).to(torch.uint8) # (B,1,1,W)
    return (rows.flip(2).cummax(2).values.flip(2)
            & cols.flip(3).cummax(3).values.flip(3)).bool() # (B,1,H,W)


def _masked_conv2d(conv, x, valid_mask):
    """Partial conv (Liu et al. 2018): renormalize each window by its valid input fraction."""
    m = valid_mask.to(x.dtype)
    out = conv(x * m)
    kh, kw = conv.kernel_size
    ph, pw = conv.padding
    ones = m.new_ones(1, 1, kh, kw)
    m_pad = F.pad(m, (pw, pw, ph, ph), value=1.0) # border = valid
    valid = F.conv2d(m_pad, ones, stride=conv.stride, padding=0) # (B,1,H',W')
    scale = (kh * kw) / valid.clamp_min(1.0)
    if conv.bias is not None:
        b = conv.bias.view(1, -1, 1, 1)
        return (out - b) * scale + b
    return out * scale


class _ConvDepthAdapter(nn.Module):
    """Conv depth adapter (modes 1/2, --depth_adapt conv): conv3x3 -> masked
    GroupNorm -> GELU -> conv3x3. depth_cues 'fixed' keeps the precomputed
    Sobel/Laplacian input channels so old conv-adapter checkpoints still load
    (conv1's input width differs)."""

    def __init__(self, in_ch, mid, out_ch, orthogonal_init=False, masked_conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(_pick_norm_groups(mid, 0), mid)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid, out_ch, kernel_size=3, padding=1)
        # --depth_adapt_masked_conv: partial conv instead of plain Conv2d
        self.masked_conv = masked_conv
        if orthogonal_init:
            nn.init.orthogonal_(self.conv1.weight)
            nn.init.orthogonal_(self.conv2.weight)
        # plain attr: stays out of state_dict and DDP sync
        self.last_health = None
        self._health_capture_failed = False

    def forward(self, x, valid_mask):
        x = _masked_conv2d(self.conv1, x, valid_mask) if self.masked_conv else self.conv1(x)
        x = _masked_group_norm_fn(x, valid_mask, self.gn)
        act = self.act(x)
        out = _masked_conv2d(self.conv2, act, valid_mask) if self.masked_conv else self.conv2(act)
        if self.masked_conv:
            # no masked norm after conv2, content bleeds ~1 row into the pad band, zero it
            out = out * valid_mask.to(out.dtype)
        if not self._health_capture_failed:
            try:
                self._capture_health(act, out, valid_mask)
            except Exception as e:
                # one-shot disable: drop the stale stats and stop capturing
                self._health_capture_failed = True
                self.last_health = None
                print(f"[conv-adapter health capture disabled after error: {e}]", flush=True)
        return out

    @torch.no_grad()
    def _capture_health(self, act, out, valid_mask):
        """Adapter output stats over valid pixels, ~32x32 subsample per image.
        eff_rank = trace(cov)^2 / ||cov||_F^2 in [1, C], near 1 = channel collapse."""
        C = out.shape[1]
        S = out.shape[-1]
        stride = max(1, S // 32)
        o = out[:, :, ::stride, ::stride]
        a = act[:, :, ::stride, ::stride]
        mv = valid_mask[:, :, ::stride, ::stride]
        X = o.permute(1, 0, 2, 3).reshape(C, -1).float() # (C, n)
        w = mv.reshape(1, -1).float() # (1, n)
        wsum = w.sum().clamp_min(1.0)
        absmean = (X.abs() * w).sum() / (wsum * C)
        mean = (X * w).sum() / (wsum * C)
        std = ((((X - mean) ** 2) * w).sum() / (wsum * C)).clamp_min(0).sqrt()
        naninf = (~torch.isfinite(X)).sum()
        # fraction of GELU-suppressed hidden activations
        A = a.permute(1, 0, 2, 3).reshape(a.shape[1], -1).float()
        dead = (((A <= 0).float()) * w).sum() / (wsum * a.shape[1])
        # effective rank via participation ratio
        mu = (X * w).sum(dim=1, keepdim=True) / wsum
        Xc = (X - mu) * w # invalid cols -> 0
        cov = (Xc @ Xc.t()) / wsum # (C, C)
        tr = torch.diagonal(cov).sum()
        fro2 = (cov * cov).sum().clamp_min(1e-12)
        eff_rank = (tr * tr) / fro2
        self.last_health = {
            'eff_rank': eff_rank.detach(),
            'act_absmean': absmean.detach(),
            'act_std': std.detach(),
            'dead_frac': dead.detach(),
            'naninf': naninf.detach(),
        }


class FeatureFusionModule(nn.Module):
    """BiSeNet Feature Fusion Module, the per-level fusion op of the 'ffm'
    experiment (use_depth=5). Under --depth_fuse_identity_init the output enters
    through a gamma gate (init 0.1), without it the zero-depth ablation raises."""

    def __init__(self, rgb_ch, depth_ch, out_ch, norm='group', residual_gate=False,
                 kernel_size=3):
        super().__init__()
        # BatchNorm has no masked form
        self.norm_is_group = (norm != 'batch')
        _act = (lambda: nn.ReLU(inplace=True)) if norm == 'batch' else (lambda: nn.GELU())
        norm_layer = (nn.BatchNorm2d(out_ch) if norm == 'batch'
                      else nn.GroupNorm(_pick_norm_groups(out_ch, 0), out_ch))
        # conv kept at conv_block[0], the masked forward pulls it out
        self.conv_block = nn.Sequential(
            nn.Conv2d(rgb_ch + depth_ch, out_ch, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            norm_layer,
            _act(),
        )
        # SE squeeze happens in forward so it can be masked
        self.attention = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            _act(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        # residual gate under identity-init, None = plain FFM
        self.gamma = nn.Parameter(torch.full((1,), 0.1)) if residual_gate else None

    def forward(self, rgb, depth, valid=None):
        """valid: (B,1,H,W) bool mask. GroupNorm stats then use valid pixels only."""
        masked = valid is not None and self.norm_is_group
        x = torch.cat([rgb, depth], dim=1)
        if masked:
            conv, norm, act = self.conv_block[0], self.conv_block[1], self.conv_block[2]
            f = act(_masked_group_norm_fn(_masked_conv2d(conv, x, valid), valid, norm))
            # pool over valid pixels only
            m = valid.to(f.dtype)
            pooled = (f * m).sum(dim=(2, 3), keepdim=True) \
                / m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        else:
            f = self.conv_block(x)
            pooled = f.mean(dim=(2, 3), keepdim=True)
        att = self.attention(pooled)
        f = f + f * att
        if self.gamma is not None:
            return rgb + self.gamma * f
        return f


def _pca_rgb_uint8(feats, valid_mask=None):
    """Top-3 PCA of a (C, H, W) feature map rendered as (H, W, 3) uint8 RGB.
    Stats use only the valid pixels when valid_mask is given, pad renders black."""
    C, H, W = feats.shape
    X = feats.detach().float().reshape(C, -1) # (C, N)
    v = (valid_mask.reshape(-1).bool() if valid_mask is not None
         else torch.ones(H * W, dtype=torch.bool, device=X.device))
    idx = v.nonzero(as_tuple=False).squeeze(1)
    if idx.numel() < 4: # degenerate frame
        return np.zeros((H, W, 3), dtype=np.uint8)
    # subsample valid pixels for the stats, the projection below stays full-res
    step = idx.numel() // 100_000 + 1
    idx_s = idx[::step] if step > 1 else idx
    Xs = X[:, idx_s]
    mu = Xs.mean(dim=1, keepdim=True)
    Xc = Xs - mu
    cov = (Xc @ Xc.t()) / max(1, Xs.shape[1] - 1) # (C, C)
    # eigh returns ascending eigenvalues
    _, vecs = torch.linalg.eigh(cov)
    k = min(3, C)
    comps = vecs[:, -k:].flip(-1) # (C, k), PC1 first
    proj = ((comps.t() @ X) - (comps.t() @ mu)).reshape(k, H, W) # (k, H, W)
    out = torch.zeros(3, H, W, device=X.device)
    vmap = v.reshape(H, W).float()
    for i in range(k):
        p = proj[i].reshape(-1)[idx_s]
        lo, hi = torch.quantile(p, 0.02), torch.quantile(p, 0.98)
        out[i] = ((proj[i] - lo) / (hi - lo + 1e-8)).clamp(0, 1) * vmap
    return (out * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


class CNT(nn.Module):

    def __init__(
            self,
            image_size: int,
            num_objects: int,
            emb_dim: int,
            kernel_dim: int,
            reduction: int,
            zero_shot: bool,
            training: bool = False,
            use_depth: int = 0,
            depth_kernel_size: int = 3,
            depth_feat_channels: int = 16, # matches the arg_parser default
            use_density: int = 0,
            density_head_type: str = 'simple',
            density_guided: int = 0,
            density_detach: int = 0,
            unfreeze_last_hiera: int = 0,
            depth_fuse_identity_init: int = 0,
            depth_feat_norm: str = 'group',
            depth_feat_norm_groups: int = 0,
            depth_target_size: int = 0,
            depth_source: str = 'scalar',
            depth_adapt: str = 'linear',
            depth_cues: str = 'learned',
            depth_adapt_init: str = 'orthogonal',
            depth_adapt_masked_conv: int = 1,
            sep_hiera_input: str = 'cues',
            sep_hiera_fullres: int = 1,
            sep_hiera_per_level_gate: int = 1,
            ffm_norm: str = 'group',
            dino_input_size: int = 0,
            external_depth: bool = False,
            external_depth_feats: int = 0,
            depth_hires_fusion: int = 0,
            depth_hires_norm: int = 1,
    ):
        super(CNT, self).__init__()
        self.inference = not training
        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot # prototypes learned directly as tokens
        self.pretrain = False
        self.use_depth = use_depth
        self.depth_kernel_size = depth_kernel_size
        self.depth_feat_channels = depth_feat_channels
        self.use_density = use_density
        self.density_head_type = density_head_type
        self.density_guided = density_guided
        self.density_detach = int(density_detach)
        self.unfreeze_last_hiera = unfreeze_last_hiera
        self.depth_fuse_identity_init = depth_fuse_identity_init
        self.depth_feat_norm = depth_feat_norm
        self.depth_feat_norm_groups = depth_feat_norm_groups
        # multi-channel depth options: modes 1/2/5 only, modes 3/4 forced to
        # scalar/linear. depth_cues 'fixed' only exists to load old conv-adapter
        # checkpoints (input width differs). run-name tags: _lcues, _oinit, _ffmbn.
        self.depth_source = depth_source if use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) else 'scalar'
        self.depth_adapt = depth_adapt if use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) else 'linear'
        self.depth_cues = (depth_cues
                           if use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) and self.depth_adapt == 'conv'
                           else 'fixed')
        self.depth_adapt_init = (depth_adapt_init
                                 if use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) and self.depth_adapt == 'conv'
                                 else 'default')
        self.depth_adapt_masked_conv = bool(depth_adapt_masked_conv)
        # mode-4 second-Hiera input: 'cues' = [disparity, grad-mag, laplacian] stack,
        # 'replicate' = legacy 1-ch disparity copied to 3ch
        self.sep_hiera_input = sep_hiera_input if use_depth == SEP_HIERA else 'replicate'
        # --sep_hiera_fullres: build the cue stack/adapter/norm at image_size instead
        # of depth_target_size, so the edge cues land on the Hiera's own grid
        self.sep_hiera_fullres = bool(sep_hiera_fullres) if use_depth == SEP_HIERA else False
        # --sep_hiera_per_level_gate: one gamma per fused pyramid level instead of a
        # shared scalar (needs identity-init, the gate only exists then)
        self.sep_hiera_per_level_gate = bool(sep_hiera_per_level_gate) if use_depth == SEP_HIERA else False
        if self.sep_hiera_per_level_gate and use_depth == SEP_HIERA and not depth_fuse_identity_init:
            print("[counter][WARN] --sep_hiera_per_level_gate needs --depth_fuse_identity_init 1 "
                  "(the depth_pyramid_gamma gate only exists under identity-init; without it the "
                  "depth pyramid is summed ungated at weight 1.0) -> per-level gate has no effect.",
                  flush=True)
        self.ffm_norm = ffm_norm
        # external_depth: dataset supplies the cached map as the image's 4th channel,
        # so the in-model depth ViT is never built. external_depth_feats = k adds the
        # _pdf<k> cache channels (4..3+k), keeping a decoder-like source with C_dec = k.
        self.external_depth = bool(external_depth)
        self.external_depth_feats = int(external_depth_feats) if self.external_depth else 0
        if self.external_depth_feats > 0 and use_depth not in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE):
            raise ValueError(
                f"--depthfeats_dir (the PCA-k decoder-feature cache) is decoder-family "
                f"and only applies to use_depth modes 1/2/5; got use_depth={use_depth}.")
        if self.external_depth:
            self.depth_source = 'decoder' if self.external_depth_feats > 0 else 'scalar'
        # hi-res input-level depth injection, modes 1/2/5 only
        self.depth_hires_fusion = int(depth_hires_fusion) if use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) else 0
        # masked per-image GroupNorm (G=1) on the 1-ch hires depth before initial_depth_fuse
        self.depth_hires_norm = int(depth_hires_norm) if self.depth_hires_fusion else 0
        if int(depth_hires_norm) and not self.depth_hires_fusion:
            print("[counter][WARN] --depth_hires_norm 1 needs --depth_hires_fusion 1 "
                  "(use_depth modes 1/2/5): there is no input-level hires path to "
                  "normalise -> ignoring the flag.", flush=True)
        self.unfreeze_backbone = False
        if self.use_depth > 0:
            pad = depth_kernel_size // 2
            # resolution the depth map is predicted and fused at, default 256 for a
            # 1024 image (l1's stride-4 res). A cached mode-3 map bypasses it and
            # fuses at image_size. No parameter shape depends on it, but train and
            # inference must use the same value.
            self.depth_target_size = depth_target_size if depth_target_size > 0 else (image_size * 4 // reduction)
            # dino_input_size = resolution the depth ViT runs at (the real detail
            # ceiling), 0 = auto = max(518, depth_target_size). Train/inference must match.
            self.dino_input_size = dino_input_size if dino_input_size > 0 else max(518, self.depth_target_size)

            if self.use_depth == CONV_DEPTH_ADD:
                # project depth (depth_feat_channels -> emb_dim) and add to vision features
                self.depth_proj_src = nn.Conv2d(depth_feat_channels, self.emb_dim, depth_kernel_size, padding=pad)
                self.depth_proj_l1 = nn.Conv2d(depth_feat_channels, self.emb_dim, depth_kernel_size, padding=pad)
                self.depth_proj_l2 = nn.Conv2d(depth_feat_channels, self.emb_dim, depth_kernel_size, padding=pad)
            elif self.use_depth == CONV_HIERA:
                # concat depth (depth_feat_channels) with vision features (emb_dim), then convolve back to emb_dim
                self.depth_fuse_src = nn.Conv2d(self.emb_dim + depth_feat_channels, self.emb_dim, depth_kernel_size, padding=pad)
                self.depth_fuse_l1 = nn.Conv2d(self.emb_dim + depth_feat_channels, self.emb_dim, depth_kernel_size, padding=pad)
                self.depth_fuse_l2 = nn.Conv2d(self.emb_dim + depth_feat_channels, self.emb_dim, depth_kernel_size, padding=pad)
            elif self.use_depth == FFM_FUSE:
                # 'ffm' experiment: one BiSeNet FeatureFusionModule per level
                _ffm = lambda: FeatureFusionModule(
                    self.emb_dim, depth_feat_channels, self.emb_dim,
                    norm=ffm_norm,
                    residual_gate=bool(depth_fuse_identity_init),
                    kernel_size=depth_kernel_size)
                self.depth_ffm_src = _ffm()
                self.depth_ffm_l1 = _ffm()
                self.depth_ffm_l2 = _ffm()
            elif self.use_depth == DEPTH_DIM:
                # input-level fusion: concat RGB + 1-ch depth, 1x1 conv back to 3 ch
                # for the frozen Hiera. Kernel hardcoded 1x1, --depth_kernel_size does
                # not apply (shells force _k1).
                self.initial_depth_fuse = nn.Conv2d(3 + 1, 3, kernel_size=1)
            elif self.use_depth == SEP_HIERA:
                # mode 4: second Hiera on the depth stream, pyramids summed. The second
                # Hiera is instantiated after self.backbone (same config/checkpoint).
                assert depth_feat_channels == 3, ("use_depth=4 feeds depth into a separate Hiera; depth_feat_channels must be 3.")

            if self.depth_hires_fusion and self.use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE):
                # hi-res input-level depth injection on top of the feature fusion: 1x1
                # conv fuses full-res depth into the RGB input (the mode-3 path).
                # Always identity-init (RGB passthrough, depth weight 0), regardless
                # of --depth_fuse_identity_init.
                self.initial_depth_fuse = nn.Conv2d(3 + 1, 3, kernel_size=1)
                with torch.no_grad():
                    self.initial_depth_fuse.weight.zero_()
                    self.initial_depth_fuse.bias.zero_()
                    for o in range(3):
                        self.initial_depth_fuse.weight[o, o, 0, 0] = 1.0
                if self.depth_hires_norm:
                    # --depth_hires_norm: masked per-image GroupNorm (G=1) of the 1-ch
                    # hires depth before initial_depth_fuse. The name prefix keeps it
                    # in the depth-fusion LR group. Applied via _masked_group_norm_fn
                    # so the letterbox pad does not skew the stats.
                    self.initial_depth_fuse_norm = nn.GroupNorm(1, 1)

            # modes 1/2 downsample to stride-4 anyway, so a huge depth_target_size only
            # burns memory. --depth_hires_fusion is the 1-channel path for hi-res depth.
            if self.use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) and self.depth_target_size >= 1024:
                print(
                    f"[counter][WARN] depth_target_size={self.depth_target_size} with use_depth="
                    f"{self.use_depth} (conv_depth_add/conv_hiera): the {depth_feat_channels}-channel "
                    f"depth feature map is held at {self.depth_target_size}^2 (~"
                    f"{depth_feat_channels * self.depth_target_size**2 * 4 / 1e9:.1f} GB/sample fp32) "
                    f"and is then DOWNSAMPLED to the stride-4 FPN level (~{image_size//4}), so the "
                    f"extra resolution is wasted and may OOM. Use --depth_hires_fusion 1 to feed "
                    f"high-def depth to these modes via the 1-channel input path instead.")

            # frozen depth model (Depther/infer_depth.py), default DAv2 Base (relative
            # disparity, large = near), other checkpoints via DEPTH_CHECKPOINT. With
            # depth_source='decoder' it also returns the DPT path_1 feature map.
            if self.external_depth:
                # dataset supplies the depth (image channels 3..3+k), so skip building
                # the in-model depth model. predict_depth_map is never called here.
                self.depth_model = None
                C_dec = self.external_depth_feats if self.external_depth_feats > 0 else 1
            else:
                self.depth_model = load_depther(
                    device='cpu', target_size=self.depth_target_size,
                    input_size=self.dino_input_size,
                    return_path_1=(self.depth_source == 'decoder'),
                )
                self.depth_model.requires_grad_(False)
                self.depth_model.eval()

                # adapter input channels: decoder = path_1 width (256 for ViT-L), scalar = 1
                C_dec = getattr(self.depth_model, 'decoder_feat_channels', 0) if self.depth_source == 'decoder' else 1
                if self.depth_source == 'decoder' and not C_dec:
                    raise ValueError("depth_source='decoder' needs a DPT-head depth model exposing decoder_feat_channels.")

            # depth adapter -> depth_feat_channels. mode 3 = none, mode 4 = 1x1 to 3ch,
            # modes 1/2 = 1x1 (linear) or the conv stack. Every variant is named
            # 'depth_channel_adapt*' so it stays in the depth-fusion LR group.
            if self.use_depth == DEPTH_DIM:
                self.depth_channel_adapt = None
                norm_ch = 1
            elif self.use_depth == SEP_HIERA:
                # 'cues': [disparity, Sobel grad-mag, Laplacian] (3ch) -> depth_feat_channels(=3).
                # 'replicate' (legacy): 1-ch disparity -> 3ch. Input width differs (3 vs 1).
                _sep_in = 3 if self.sep_hiera_input == 'cues' else 1
                self.depth_channel_adapt = nn.Conv2d(_sep_in, depth_feat_channels, kernel_size=1)
                if self.sep_hiera_input == 'cues':
                    self.register_buffer('_depth_cue_kernels', _sobel_laplacian_kernels(), persistent=False)
                norm_ch = depth_feat_channels
            elif self.depth_adapt == 'conv':
                # conv -> masked GroupNorm -> GELU -> conv, see _ConvDepthAdapter.
                # depth_cues 'learned' feeds the raw signal, 'fixed' the legacy
                # [signal + Sobel/Lap] input (needed to load old conv-adapter ckpts).
                in_ch = C_dec if self.depth_cues == 'learned' else C_dec + 3
                mid = max(64, depth_feat_channels // 2)
                self.depth_channel_adapt = _ConvDepthAdapter(
                    in_ch, mid, depth_feat_channels,
                    orthogonal_init=(self.depth_adapt_init == 'orthogonal'),
                    masked_conv=self.depth_adapt_masked_conv)
                # fixed (non-learned) edge kernels, persistent=False keeps them out of state_dict
                self.register_buffer('_depth_cue_kernels', _sobel_laplacian_kernels(), persistent=False)
                norm_ch = depth_feat_channels
            else:  # modes 1/2 + linear
                self.depth_channel_adapt = nn.Conv2d(C_dec, depth_feat_channels, kernel_size=1)
                norm_ch = depth_feat_channels

            # optional per-image GroupNorm on the adapted depth features (relative
            # depth has arbitrary per-image scale/offset). New params land in the
            # main --lr group, loaded strict=False.
            if self.depth_feat_norm == 'group':
                self.depth_feat_norm_layer = nn.GroupNorm(
                    _pick_norm_groups(norm_ch, self.depth_feat_norm_groups),
                    norm_ch,
                )

            if self.use_depth == SEP_HIERA:
                # warm-start the 1x1 adapter to pass its input through unchanged, so
                # the frozen Hiera starts on an in-distribution image-like input:
                # 'cues' -> identity, 'replicate' -> disparity copied to all 3 channels.
                with torch.no_grad():
                    self.depth_channel_adapt.bias.zero_()
                    if self.sep_hiera_input == 'cues':
                        self.depth_channel_adapt.weight.zero_()
                        for o in range(self.depth_channel_adapt.out_channels):
                            self.depth_channel_adapt.weight[o, o, 0, 0] = 1.0
                    else:
                        self.depth_channel_adapt.weight.fill_(1.0)

            # identity-at-init fusion: start from the pretrained RGB model, depth
            # contributes ~0 at epoch 0. Survives --init_from_pretrained (the fusion
            # convs are not in the upstream checkpoint), overwritten by --resume_training.
            if self.depth_fuse_identity_init:
                if self.use_depth == SEP_HIERA:
                    # gamma starts at 0.1, not 0: a zero gamma would also zero the
                    # gradient into the whole depth branch, so it could never learn.
                    # per-level gate = 3-vector [src, l1, l2], else one shared scalar.
                    # Extra backbone_fpn levels reuse the last entry in forward.
                    n_gamma = 3 if self.sep_hiera_per_level_gate else 1
                    self.depth_pyramid_gamma = nn.Parameter(torch.full((n_gamma,), 0.1))
                elif self.use_depth == FFM_FUSE:
                    # FFM near-identity is the gamma gate inside each FeatureFusionModule.
                    # Do not call apply_depth_fusion_identity() here, it would zero the
                    # gammas (that is the --depth_zero_ablation path).
                    pass
                else:
                    self.apply_depth_fusion_identity()

            # config log, lands in the .out
            _p1 = ('OFF (1-ch scalar)' if self.depth_source != 'decoder' else
                   f"ON ({C_dec}ch "
                   + ('cached PCA feats)' if self.external_depth_feats > 0 else 'in-model path_1)'))
            # mode 3 fuses a cached map at image_size, so the logged depth_target_size
            # is not its fusion resolution
            _dts = (f"depth_target_size={self.depth_target_size}"
                    + (f" (mode 3: BYPASSED for provided/cached depth -> fuses at "
                       f"image_size={self.image_size}; applies only to the in-model fallback)"
                       if self.use_depth == DEPTH_DIM else ""))
            print(f"[counter] depth config: mode={self.use_depth} path1_depth={_p1} "
                  f"(depth_source={self.depth_source}) depth_adapt={self.depth_adapt} "
                  f"(cues={self.depth_cues}, init={self.depth_adapt_init}, "
                  f"masked_conv={self.depth_adapt_masked_conv})"
                  f"{f' ffm_norm={self.ffm_norm}' if self.use_depth == FFM_FUSE else ''} "
                  f"dinoInSize={'n/a (external depth)' if self.external_depth else self.dino_input_size} "
                  f"external_feats={self.external_depth_feats or 'off'} "
                  f"{_dts} "
                  f"depth_feat_channels={self.depth_feat_channels} "
                  f"hires_fusion={self.depth_hires_fusion} "
                  f"hires_norm={self.depth_hires_norm}"
                  f"{f' sep_hiera_input={self.sep_hiera_input} sep_hiera_fullres={self.sep_hiera_fullres} per_level_gate={self.sep_hiera_per_level_gate}' if self.use_depth == SEP_HIERA else ''}",
                  flush=True)

        self.class_embed = nn.Sequential(nn.Linear(emb_dim, 1), nn.LeakyReLU())
        self.bbox_embed = MLP(emb_dim, emb_dim, 4, 3)
        if not self.pretrain:
            self.class_embed_aux = nn.Sequential(nn.Linear(emb_dim, 1), nn.LeakyReLU())
            self.bbox_embed_aux = MLP(emb_dim, emb_dim, 4, 3)
        if self.use_density > 0:
            # density head on the adapted feature map, count = pred_density.sum().
            # 'density_head' in the param names puts it in its own higher-LR group.
            # 'simple' = shallow DensityHead (512^2), 'fpn' = DensityDecoder (1024^2).
            if self.density_head_type == 'fpn':
                self.density_head = DensityDecoder(self.emb_dim)
            else:
                self.density_head = DensityHead(self.emb_dim)
            if self.density_guided:
                # density-guided detection: detached density prior modulates the
                # detection features. 'density_guide*' params go in the density LR group.
                self.density_guide = DensityGuidedDetection(self.emb_dim)
        if self.zero_shot:
            # zero-shot prototypes learned directly as tokens: per feature level,
            # num_objects appearance + num_objects shape slots, matching the few-shot
            # (B, 2*num_objects, emb_dim) layout so nothing downstream changes.
            # Own optimizer group (--zs_proto_lr).
            self.zs_prototypes = nn.Parameter(
                torch.empty(3, 2 * self.num_objects, self.emb_dim))
            nn.init.trunc_normal_(self.zs_prototypes, std=0.02)

        self.adapt_features = C_base(
            transformer_dim=self.emb_dim,
            num_prototype_attn_steps=3,
            num_image_attn_steps=2,
        )
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.emb_dim,
            image_embedding_size=(
                self.image_size // self.reduction,
                self.image_size // self.reduction,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        config_name = '../configs/sam2_hiera_base_plus.yaml'
        cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        self.backbone = instantiate(cfg.backbone, _recursive_=True)
        checkpoint = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/segment_anything_2/072824/' + config_name.split('/')[-1].replace('.yaml',
                                                                                                             '.pt'),
            map_location="cpu"
        )['model']
        state_dict = {k.replace("image_encoder.", ""): v for k, v in checkpoint.items()}
        self.backbone.load_state_dict(state_dict, strict=False)

        if self.unfreeze_last_hiera > 0:
            # unfreeze only the last N Hiera stages (FPN neck + early stages stay
            # frozen). stage_ends marks the last block index of each stage.
            self.backbone.requires_grad_(False)
            trunk = self.backbone.trunk
            stage_ends = trunk.stage_ends
            n_stages = len(stage_ends)
            k = min(self.unfreeze_last_hiera, n_stages)
            first_block = stage_ends[n_stages - k - 1] + 1 if k < n_stages else 0
            for blk in trunk.blocks[first_block:]:
                blk.requires_grad_(True)
            self.unfreeze_backbone = True

        if self.use_depth == SEP_HIERA:
            # second Hiera for the depth stream (separate weights, same architecture and pretrained init)
            cfg_d = compose(config_name=config_name)
            OmegaConf.resolve(cfg_d)
            self.depth_backbone = instantiate(cfg_d.backbone, _recursive_=True)
            self.depth_backbone.load_state_dict(state_dict, strict=False)

        self.shape_or_objectness = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1 ** 2 * emb_dim)
        )

        if self.inference:
            self.sam_mask = MaskProcessor(self.emb_dim, self.image_size, reduction)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # refuse old box-based zero-shot checkpoints (prototypes would stay at random
        # init). Hook instead of a load_state_dict override so it also fires under DDP.
        if self.zero_shot and (prefix + 'trainable_exemplar_bboxes') in state_dict \
                and (prefix + 'zs_prototypes') not in state_dict:
            raise RuntimeError(
                'checkpoint has trainable_exemplar_bboxes but no zs_prototypes, so it '
                'predates the direct-prototype zero-shot. Evaluate it with the '
                'matching older code instead.')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def snapshot_zs_prototypes_init(self):
        """Reference copy of the zero-shot prototype tokens, taken after weight
        loading, for the per-epoch deviation log."""
        if self.zero_shot:
            self._zs_prototypes_init = self.zs_prototypes.detach().clone()

    def zs_prototypes_deviation(self):
        """L2 movement of the zero-shot prototype tokens from their init snapshot."""
        if not self.zero_shot or not hasattr(self, '_zs_prototypes_init'):
            return {}
        ref = self._zs_prototypes_init
        cur = self.zs_prototypes.detach()
        out = {'total': (cur - ref).norm().item(),
               'init_norm': ref.norm().item()}
        for i, lvl in enumerate(('src', 'l1', 'l2')):
            out[lvl] = (cur[i] - ref[i]).norm().item()
        return out

    def train(self, mode=True):
        super().train(mode)
        # keep the frozen depth model in eval (None under external_depth)
        if self.use_depth > 0 and self.depth_model is not None:
            self.depth_model.eval()
        return self

    def apply_depth_fusion_identity(self):
        """(Re-)set the depth-fusion weights so depth contributes nothing. Used by
        --depth_fuse_identity_init and by the --depth_zero_ablation."""
        c = self.depth_kernel_size // 2  # center tap (0 for 1x1)
        with torch.no_grad():
            if self.use_depth == FFM_FUSE:
                # mode 5: identity is only reachable through the gamma gates, which
                # exist only under --depth_fuse_identity_init
                for m in (self.depth_ffm_src, self.depth_ffm_l1, self.depth_ffm_l2):
                    if m.gamma is None:
                        raise RuntimeError(
                            'zero-depth ablation on use_depth=5 (ffm) needs the per-level '
                            'gamma gates, which only exist on models built with '
                            '--depth_fuse_identity_init 1.')
                    m.gamma.zero_()
            elif self.use_depth == CONV_DEPTH_ADD:
                # add-fusion: zero the projection so src + depth_proj(d) == src.
                for conv in (self.depth_proj_src, self.depth_proj_l1, self.depth_proj_l2):
                    conv.weight.zero_(); conv.bias.zero_()
            elif self.use_depth == CONV_HIERA:
                # concat-fusion (in = emb_dim vision + depth_feat_channels depth, out = emb_dim):
                # center-tap identity on the vision channels, zero on the depth channels.
                for conv in (self.depth_fuse_src, self.depth_fuse_l1, self.depth_fuse_l2):
                    conv.weight.zero_(); conv.bias.zero_()
                    for o in range(self.emb_dim):
                        conv.weight[o, o, c, c] = 1.0
            elif self.use_depth == DEPTH_DIM:
                # RGB passthrough, depth zeroed. The conv is 1x1, so use its own
                # kernel center rather than c.
                kc = self.initial_depth_fuse.kernel_size[0] // 2
                self.initial_depth_fuse.weight.zero_(); self.initial_depth_fuse.bias.zero_()
                for o in range(3):
                    self.initial_depth_fuse.weight[o, o, kc, kc] = 1.0
            elif self.use_depth == SEP_HIERA:
                if not hasattr(self, 'depth_pyramid_gamma'):
                    raise RuntimeError(
                        'zero-depth ablation on use_depth=4 needs the depth_pyramid_gamma '
                        'gate, which only exists on models built with '
                        '--depth_fuse_identity_init 1.')
                self.depth_pyramid_gamma.zero_()

            # hires fusion also carries an input-level initial_depth_fuse, reset it
            # to RGB-passthrough too
            if self.depth_hires_fusion and self.use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) \
                    and getattr(self, 'initial_depth_fuse', None) is not None:
                kc = self.initial_depth_fuse.kernel_size[0] // 2
                self.initial_depth_fuse.weight.zero_()
                self.initial_depth_fuse.bias.zero_()
                for o in range(3):
                    self.initial_depth_fuse.weight[o, o, kc, kc] = 1.0

    def snapshot_depth_fusion_init(self):
        """Snapshot the depth-fusion params as the reference for
        depth_fusion_deviation(). Call once after weight loading."""
        self._depth_fusion_ref = {
            n: p.detach().float().cpu().clone()
            for n, p in self.named_parameters() if is_depth_fusion_param(n)
        }

    def depth_fusion_deviation(self):
        """L2 distance of each depth-fusion module from its snapshot, plus 'total'."""
        ref = getattr(self, '_depth_fusion_ref', None)
        if not ref:
            return {}
        params = dict(self.named_parameters())
        per_module, total_sq = {}, 0.0
        for n, r in ref.items():
            if n not in params:
                continue
            d = (params[n].detach().float().cpu() - r).pow(2).sum().item()
            total_sq += d
            mod = n.rsplit('.', 1)[0] if '.' in n else n
            per_module[mod] = per_module.get(mod, 0.0) + d
        out = {k: v ** 0.5 for k, v in per_module.items()}
        out['total'] = total_sq ** 0.5
        return out

    def density_guide_stats(self):
        """Gate stats for density-guided detection: gamma magnitudes plus
        'rel_energy' (||gamma*mod|| / ||feat|| on the last batch). Values ~0
        mean the densg branch is doing nothing yet."""
        if not (self.use_density > 0 and self.density_guided):
            return {}
        guide = getattr(self, 'density_guide', None)
        if guide is None:
            return {}
        g = guide.gamma.detach().float()
        out = {
            'gamma_abs_mean': g.abs().mean().item(),
            'gamma_abs_max': g.abs().max().item(),
            'gamma_l2': g.norm().item(),
        }
        rel = getattr(guide, 'last_rel_energy', None)
        if rel is not None:
            out['rel_energy'] = float(rel)
        return out

    def pyramid_gamma_stats(self):
        """Depth-pyramid gate values (mode 4): per-level gamma_src/l1/l2, or a
        single 'gamma' for a scalar gate."""
        g = getattr(self, 'depth_pyramid_gamma', None)
        if g is None:
            return {}
        g = g.detach().float().flatten()
        if g.numel() == 1:
            return {'gamma': g[0].item()}
        names = ['gamma_src', 'gamma_l1', 'gamma_l2']
        return {(names[i] if i < len(names) else f'gamma_fpn{i}'): g[i].item()
                for i in range(g.numel())}

    def depth_adapter_health(self):
        """Conv-adapter health stats from the last forward batch: eff_rank (near 1 =
        channel collapse), act_absmean, act_std, dead_frac, naninf. {} for
        non-conv configs."""
        if not (self.use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE) and self.depth_adapt == 'conv'):
            return {}
        adapter = getattr(self, 'depth_channel_adapt', None)
        health = getattr(adapter, 'last_health', None) if adapter is not None else None
        if not health:
            return {}
        out = {k: float(v) for k, v in health.items()}
        out['n_channels'] = float(adapter.conv2.out_channels)
        return out

    def save_train_cue_visual(self, x, out_path):
        """4-panel turbo image of the depth map + Sobel/Laplacian cues for one
        sample, written to out_path (see --vis_every). A path1 PCA panel is added
        under depth_source='decoder'."""
        if self.use_depth <= 0 or getattr(self, 'depth_model', None) is None:
            return False
        import os
        import cv2
        if x.dim() == 3:
            x = x.unsqueeze(0)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                depth_main, depth_1ch = self.predict_depth_map(x)  # (1,C,S,S), (1,1,S,S)
                cues = F.conv2d(depth_1ch, _sobel_laplacian_kernels().to(depth_1ch), padding=1)  # (1,3,S,S)
        finally:
            if was_training:
                self.train()
        d = depth_1ch[0, 0]
        valid = d > 1e-6
        if bool(valid.any()):
            rows = torch.where(valid.any(dim=1))[0]
            cols = torch.where(valid.any(dim=0))[0]
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
        else:
            r0, c0, r1, c1 = 0, 0, d.shape[0], d.shape[1]
        cfg = getattr(self.depth_model, 'cfg', None)
        invert_depth = bool(cfg and cfg.get('output') == 'depth')  # depth-like -> near warm

        def _panel(t2d, invert):
            p = t2d[r0:r1, c0:c1].float()
            # torch.quantile caps out near 16M elements, so subsample before the
            # percentile clip.
            flat = p.reshape(-1)
            if flat.numel() > 4_000_000:
                flat = flat[:: flat.numel() // 4_000_000 + 1]
            lo, hi = torch.quantile(flat, 0.02), torch.quantile(flat, 0.98)
            p = p.clamp(lo, hi)
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            if invert:
                p = 1.0 - p
            p8 = (p * 255.0).to(torch.uint8).cpu().numpy()
            return cv2.applyColorMap(p8, cv2.COLORMAP_TURBO)

        # depth oriented by the checkpoint's output type. The signed Sobel/Laplacian
        # cues are derivatives, never inverted.
        panels = [_panel(d, invert_depth),
                  _panel(cues[0, 0], False),
                  _panel(cues[0, 1], False),
                  _panel(cues[0, 2], False)]
        labels = ['depth', 'sobel-dx', 'sobel-dy', 'laplacian']
        if depth_main.shape[1] > 1:
            # path1_depth: PCA-RGB of the decoder path_1 map, same valid-region crop
            # as the other panels, RGB -> BGR for cv2
            pca = _pca_rgb_uint8(depth_main[0, :, r0:r1, c0:c1], valid[r0:r1, c0:c1])
            panels.append(np.ascontiguousarray(pca[:, :, ::-1]))
            labels.append(f'path1-pca ({depth_main.shape[1]}ch)')
        for img, lab in zip(panels, labels):
            cv2.putText(img, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, cv2.hconcat(panels))
        return True

    def save_depth_feature_visual(self, x, pad_wh, orig_w, orig_h, out_path):
        """PCA-RGB image of the path_1 decoder feature map, pad-cropped and resized
        to the original frame. Returns False when there is no multi-channel map
        (scalar source / external depth / RGB-only)."""
        if (self.use_depth <= 0 or getattr(self, 'depth_model', None) is None
                or self.depth_source != 'decoder'):
            return False
        import os
        import cv2
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            depth_main, depth_1ch = self.predict_depth_map(x[:, :3])
        if depth_main.shape[1] <= 1:
            return False
        hd, wd = depth_main.shape[-2:]
        in_h, in_w = x.shape[-2], x.shape[-1]
        if pad_wh is not None:
            # same pad-crop convention as utils/viz.py save_depth_visual
            vw = max(1, int(round((in_w - float(pad_wh[0])) / in_w * wd)))
            vh = max(1, int(round((in_h - float(pad_wh[1])) / in_h * hd)))
        else:
            vh, vw = hd, wd
        rgb = _pca_rgb_uint8(depth_main[0, :, :vh, :vw], depth_1ch[0, 0, :vh, :vw] > 1e-6)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if orig_w and orig_h:
            bgr = cv2.resize(bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(out_path, bgr)
        return True

    def _run_backbone(self, x):
        """RGB Hiera, no_grad unless --unfreeze_last_hiera>0. Mode 3 runs the
        backbone itself in forward instead."""
        if self.unfreeze_backbone:
            return self.backbone(x)
        with torch.no_grad():
            return self.backbone(x)

    def predict_depth_map(self, x):
        """Run the frozen depth model on the valid (unpadded) region of each sample
        and re-embed the result into the padded frame (the ViT attends globally,
        so feeding it the letterbox pad corrupts the map). Returns
        (depth_main, depth_1ch), (B, C, S, S) and (B, 1, S, S), no grad. For the
        scalar source both are the same disparity map."""
        B, _, H, W = x.shape
        S = self.depth_target_size
        decoder = self.depth_source == 'decoder'
        C = getattr(self.depth_model, 'decoder_feat_channels', 1) if decoder else 1
        depth_main = x.new_zeros((B, C, S, S))
        depth_1ch = x.new_zeros((B, 1, S, S))
        # the dataset pads with 0 before Normalize, so pad pixels arrive as
        # (0 - mean) / std. Exact-0 pixels count as pad too.
        pad_vals = x.new_tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(1, 3, 1, 1) # ImageNet mean/std
        is_pad = ((x - pad_vals).abs().amax(dim=1) < 1e-4) | (x.abs().amax(dim=1) < 1e-12)
        nonzero = ~is_pad # (B, H, W), True = real image pixel
        for i in range(B):
            rows = nonzero[i].any(dim=1)
            cols = nonzero[i].any(dim=0)
            vh = int(rows.nonzero().max().item()) + 1 if rows.any() else H
            vw = int(cols.nonzero().max().item()) + 1 if cols.any() else W
            # letterbox target inside the S x S frame, same convention as
            # resize_and_pad. Emitting directly at (th, tw) avoids a double resample.
            th = max(1, int(round(vh / H * S)))
            tw = max(1, int(round(vw / W * S)))
            out_i = self.depth_model(x[i:i + 1, :, :vh, :vw], out_size=(th, tw))  # decoder returns (depth_main, depth_1ch), else depth_1ch
            if decoder:
                depth_main_i, depth_1ch_i = out_i
            else:
                depth_1ch_i = out_i
                depth_main_i = depth_1ch_i
            depth_main[i, :, :th, :tw] = depth_main_i[0]
            depth_1ch[i, :, :th, :tw] = depth_1ch_i[0]
        return depth_main, depth_1ch

    def _depth_cues(self, d1):
        """Fixed Sobel-dx/dy + Laplacian of the disparity -> (B,3,H,W) boundary cues."""
        return F.conv2d(d1, self._depth_cue_kernels.to(d1.dtype), padding=1)

    def _sep_hiera_cue_stack(self, d1, valid_mask):
        """[disparity, Sobel grad-mag, Laplacian] stack for the mode-4 second Hiera,
        each channel standardised over valid pixels, pad set to 0. (B,1,H,W) ->
        (B,3,H,W)."""
        c = F.conv2d(d1, self._depth_cue_kernels.to(d1.dtype), padding=1) # [sx, sy, lap]
        grad = torch.sqrt(c[:, 0:1] ** 2 + c[:, 1:2] ** 2 + 1e-6)
        stack = torch.cat([d1, grad, c[:, 2:3]], dim=1) # (B,3,H,W)
        m = valid_mask.to(stack.dtype)
        n = m.sum(dim=(2, 3), keepdim=True).clamp_min(1.0) # valid px per sample
        mean = (stack * m).sum(dim=(2, 3), keepdim=True) / n # (B,3,1,1) per channel
        var = (((stack - mean) ** 2) * m).sum(dim=(2, 3), keepdim=True) / n
        return ((stack - mean) / (var + 1e-5).sqrt()) * m # standardise, pad -> 0

    def _depth_features(self, x, provided_depth=None):
        """Fusion-ready depth features from the in-model depth model or a provided
        (cached) map. Returns (depth_features, depth_valid, depth_1ch). Mode 3
        with a provided map stays at image_size, everything else at
        depth_target_size."""
        with torch.no_grad():
            if provided_depth is not None:
                # dataset-provided depth stack: channel 0 = the 1-ch map, channels 1..k
                # = cached decoder features when present. Mode 3 keeps it at native
                # image_size (it feeds the full-res RGB concat), the other modes
                # resample to depth_target_size.
                tgt = tuple(x.shape[-2:]) if (self.use_depth == DEPTH_DIM
                        or (self.use_depth == SEP_HIERA and self.sep_hiera_fullres)) \
                    else (self.depth_target_size, self.depth_target_size)
                d = provided_depth
                if d.shape[-2:] != tgt:
                    d = F.interpolate(d, size=tgt, mode='bilinear', align_corners=False)
                d = d.to(x.dtype)
                depth_1ch = d[:, :1]
                depth_main = d[:, 1:] if d.shape[1] > 1 else depth_1ch
            else:
                depth_main, depth_1ch = self.predict_depth_map(x)
        if provided_depth is not None:
            # valid mask from the RGB pad, not the depth value (a far pixel can be 0).
            # _letterbox_rect_mask keeps exact-black content pixels from counting as pad.
            _pv = _letterbox_rect_mask(_rgb_pad_valid_mask(x))
            depth_valid = F.interpolate(_pv.float(),
                                        size=depth_1ch.shape[-2:], mode='nearest') > 0.5
        else:
            depth_valid = depth_1ch > 1e-6
        if self.use_depth == SEP_HIERA and self.sep_hiera_fullres \
                and depth_1ch.shape[-2:] != x.shape[-2:]:
            # --sep_hiera_fullres: lift the in-model disparity + valid mask to
            # image_size so the cue stack / adapter / norm run on the Hiera's own grid
            tgt = tuple(x.shape[-2:])
            depth_1ch = F.interpolate(depth_1ch, size=tgt, mode='bilinear', align_corners=False)
            depth_valid = F.interpolate(depth_valid.float(), size=tgt, mode='nearest') > 0.5
            depth_main = depth_1ch
        if self.use_depth == DEPTH_DIM:
            feats = depth_1ch
        elif self.depth_adapt == 'conv':
            # conv adapter takes (x, valid_mask). 'fixed' cues concatenate the
            # precomputed Sobel/Laplacian channels.
            if self.depth_cues == 'learned':
                feats = self.depth_channel_adapt(depth_main, depth_valid)
            else:
                feats = self.depth_channel_adapt(torch.cat([depth_main, self._depth_cues(depth_1ch)], dim=1), depth_valid)
        else:
            # sep_hiera 'cues': swap the raw disparity for the [disparity, grad-mag,
            # Laplacian] stack before the 1x1 adapter
            if self.use_depth == SEP_HIERA and self.sep_hiera_input == 'cues':
                depth_main = self._sep_hiera_cue_stack(depth_1ch, depth_valid)
            feats = self.depth_channel_adapt(depth_main)
        if self.depth_feat_norm == 'group':
            feats = self._masked_group_norm(feats, depth_valid)
        return feats, depth_valid, depth_1ch

    def _masked_group_norm(self, feats, depth_valid):
        """Masked GroupNorm for the final depth features (depth_feat_norm_layer)."""
        return _masked_group_norm_fn(feats, depth_valid, self.depth_feat_norm_layer)

    def forward(self, x, bboxes, tiled=False):
        num_objects = self.num_objects if self.zero_shot else bboxes.size(1)

        # the dataset may carry the cached depth as a 4th (+k) channel, split it off
        # so RGB consumers see a 3-channel image
        provided_depth = None
        if x.shape[1] > 3:
            provided_depth = x[:, 3:]
            x = x[:, :3].contiguous()

        if self.use_depth > 0:
            # hi-res input fusion (modes 1/2/5): the backbone input depends on depth,
            # so the RGB Hiera waits for the depth stream like mode 3
            _hires = bool(self.depth_hires_fusion) and self.use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE)
            if x.is_cuda:
                # depth pipeline on its own CUDA stream so it overlaps with the RGB Hiera
                depth_stream = torch.cuda.Stream(device=x.device)
                depth_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(depth_stream):
                    depth_features, depth_valid, depth_1ch = self._depth_features(x, provided_depth)
                    if self.use_depth == SEP_HIERA:
                        # under --sep_hiera_fullres the features are already at
                        # image_size and the upsample is skipped
                        d_full = depth_features if depth_features.shape[-2:] == x.shape[-2:] \
                            else F.interpolate(depth_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
                        # run the frozen depth Hiera with grad in training: it is the
                        # adapter/norm's only consumer, under no_grad they could never learn
                        if self.training and torch.is_grad_enabled():
                            feats_d = self.depth_backbone(d_full)
                        else:
                            with torch.no_grad():
                                feats_d = self.depth_backbone(d_full)
                if self.use_depth != DEPTH_DIM and not _hires:
                    feats = self._run_backbone(x)
                torch.cuda.current_stream().wait_stream(depth_stream)
            else:
                depth_features, depth_valid, depth_1ch = self._depth_features(x, provided_depth)
                if self.use_depth == SEP_HIERA:
                    d_full = depth_features if depth_features.shape[-2:] == x.shape[-2:] \
                        else F.interpolate(depth_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
                    # see the GPU path above
                    if self.training and torch.is_grad_enabled():
                        feats_d = self.depth_backbone(d_full)
                    else:
                        with torch.no_grad():
                            feats_d = self.depth_backbone(d_full)
                if self.use_depth != DEPTH_DIM and not _hires:
                    feats = self._run_backbone(x)

            if self.use_depth == DEPTH_DIM:
                # only the in-model fallback (at depth_target_size) needs the upsample,
                # a cached map is already at x's resolution
                d_full = depth_features if depth_features.shape[-2:] == x.shape[-2:] \
                    else F.interpolate(depth_features, size=x.shape[-2:], mode='bilinear', align_corners=False)
                rgbd = self.initial_depth_fuse(torch.cat([x, d_full], dim=1))
                # run the frozen backbone with grad in training so the gradient
                # reaches initial_depth_fuse through it. Under no_grad the fusion
                # would never train.
                if self.training and torch.is_grad_enabled():
                    feats = self.backbone(rgbd)
                else:
                    with torch.no_grad():
                        feats = self.backbone(rgbd)

            if _hires:
                # fuse the full-res depth into the RGB input (like mode 3), preferring
                # the provided map. The feature-level fusion below still runs on these feats.
                depth_1ch = provided_depth[:, :1] if provided_depth is not None else depth_1ch
                depth_1ch = F.interpolate(depth_1ch, size=x.shape[-2:], mode='bilinear', align_corners=False)
                if self.depth_hires_norm:
                    # per-image standardisation over valid pixels, mask from the RGB
                    # letterbox pad (a far depth pixel can legitimately be 0)
                    valid = _letterbox_rect_mask(_rgb_pad_valid_mask(x))
                    depth_1ch = _masked_group_norm_fn(depth_1ch, valid,
                                                      self.initial_depth_fuse_norm)
                rgbd = self.initial_depth_fuse(torch.cat([x, depth_1ch], dim=1))
                if self.training and torch.is_grad_enabled():
                    feats = self.backbone(rgbd)
                else:
                    with torch.no_grad():
                        feats = self.backbone(rgbd)

            if self.use_depth == SEP_HIERA:
                # sum the two Hieras' pyramids, through the gamma gate under
                # identity-init (ungated g=1.0 otherwise)
                if self.depth_fuse_identity_init:
                    gamma = self.depth_pyramid_gamma
                    last = gamma.shape[0] - 1
                    g_src = gamma[0]
                    g_fpn = [gamma[min(i + 1, last)] for i in range(len(feats['backbone_fpn']))]
                else:
                    g_src = 1.0
                    g_fpn = [1.0] * len(feats['backbone_fpn'])
                feats = {
                    'vision_features': feats['vision_features'] + g_src * feats_d['vision_features'],
                    'backbone_fpn': [a + gf * b for gf, a, b in zip(g_fpn, feats['backbone_fpn'], feats_d['backbone_fpn'])],
                    'vision_pos_enc': feats['vision_pos_enc'],
                }
        else:
            depth_features = None
            feats = self._run_backbone(x)

        src = feats['vision_features']
        fpn = list(feats['backbone_fpn'])
        l1, l2 = fpn[0], fpn[1]
        batch_size, c, w, h = src.shape
        # reduction from the actual feature map, a hardcoded 1024/w breaks for
        # image_size != 1024
        self.reduction = self.image_size / w

        if self.use_depth in (CONV_DEPTH_ADD, CONV_HIERA, FFM_FUSE):
            # bilinearly resize depth features to match spatial dimensions of src, l1, l2:
            d_src = F.interpolate(depth_features, size=src.shape[-2:], mode='bilinear', align_corners=False)
            d_l1 = F.interpolate(depth_features, size=l1.shape[-2:], mode='bilinear', align_corners=False)
            d_l2 = F.interpolate(depth_features, size=l2.shape[-2:], mode='bilinear', align_corners=False)
            if self.use_depth == FFM_FUSE:
                # mode 5: FeatureFusionModule per level, with the valid mask resampled
                # per level so its conv/norm/pool ignore the letterbox pad
                v_src = F.interpolate(depth_valid.float(), size=src.shape[-2:], mode='nearest') > 0.5
                v_l1 = F.interpolate(depth_valid.float(), size=l1.shape[-2:], mode='nearest') > 0.5
                v_l2 = F.interpolate(depth_valid.float(), size=l2.shape[-2:], mode='nearest') > 0.5
                src_rgbd = self.depth_ffm_src(src, d_src, v_src)
                l1_rgbd = self.depth_ffm_l1(l1, d_l1, v_l1)
                l2_rgbd = self.depth_ffm_l2(l2, d_l2, v_l2)
            elif self.use_depth == CONV_DEPTH_ADD:
                # project depth (depth_feat_channels -> emb_dim) and add to vision features:
                src_rgbd = src + self.depth_proj_src(d_src)
                l1_rgbd = l1 + self.depth_proj_l1(d_l1)
                l2_rgbd = l2 + self.depth_proj_l2(d_l2)
            else:
                # concatenate depth with vision features, then convolve back to emb_dim:
                src_rgbd = self.depth_fuse_src(torch.cat([src, d_src], dim=1))
                l1_rgbd = self.depth_fuse_l1(torch.cat([l1, d_l1], dim=1))
                l2_rgbd = self.depth_fuse_l2(torch.cat([l2, d_l2], dim=1))
            # fused features replace src/l1/l2 for the shared downstream
            src, l1, l2 = src_rgbd, l1_rgbd, l2_rgbd
            fpn[0], fpn[1] = l1, l2


        self.kernel_dim = 1
        if self.zero_shot:
            # zero-shot: the learned tokens stand in for the roi_align + shape-MLP
            # prototypes, one set per level shared across the batch. The loader's
            # dummy zero boxes are ignored.
            prototype_embeddings = self.zs_prototypes[0].unsqueeze(0).expand(batch_size, -1, -1)
            prototype_embeddings_l1 = self.zs_prototypes[1].unsqueeze(0).expand(batch_size, -1, -1)
            prototype_embeddings_l2 = self.zs_prototypes[2].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # few-shot: roi_align appearance + shape-MLP prototypes, unchanged
            # upstream GeCo2 behavior
            bboxes_roi = torch.cat([
                torch.arange(
                    batch_size, requires_grad=False
                ).to(bboxes.device).repeat_interleave(num_objects).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ], dim=1)

            exemplars = roi_align(
                src,
                boxes=bboxes_roi, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction, aligned=True
            ).permute(0, 2, 3, 1).reshape(batch_size, num_objects * self.kernel_dim ** 2, self.emb_dim)

            exemplars_l1 = roi_align(
                l1,
                boxes=bboxes_roi, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction * 2 * 2, aligned=True
            ).permute(0, 2, 3, 1).reshape(batch_size, num_objects * self.kernel_dim ** 2, self.emb_dim)

            exemplars_l2 = roi_align(
                l2,
                boxes=bboxes_roi, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction * 2, aligned=True
            ).permute(0, 2, 3, 1).reshape(batch_size, num_objects * self.kernel_dim ** 2, self.emb_dim)

            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]

            # Encode shape
            shape = self.shape_or_objectness(box_hw).reshape(
                batch_size, -1, self.emb_dim
            )

            prototype_embeddings = torch.cat([exemplars, shape], dim=1)
            prototype_embeddings_l1 = torch.cat([exemplars_l1, shape], dim=1)
            prototype_embeddings_l2 = torch.cat([exemplars_l2, shape], dim=1)
        hq_prototype_embeddings = [prototype_embeddings_l1, prototype_embeddings_l2]

        # adapt image feature with prototypes
        adapted_f, adapted_f_aux = self.adapt_features(
            image_embeddings=src,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototype_embeddings,
            hq_features=fpn,
            hq_prototypes=hq_prototype_embeddings,
            hq_pos=feats['vision_pos_enc'],
        )

        # Predict class [fg, bg] and l,r,t,b
        batch_size, c, w, h = adapted_f.shape
        # density head reads the spatial adapted maps before they are flattened for
        # the detection heads. GT density is at image_size//2, the 2x 'fpn' output
        # makes DensityLoss upsample the GT.
        if self.use_density > 0:
            # --density_detach 1: DensityLoss trains the head alone, not the shared trunk
            _dens_in = adapted_f.detach() if self.density_detach else adapted_f
            _dens_in_aux = (adapted_f_aux.detach()
                            if (self.density_detach and adapted_f_aux is not None) else adapted_f_aux)
            if self.density_head_type == 'fpn':
                density_pred = self.density_head(_dens_in, _dens_in_aux)
            else:
                density_pred = self.density_head(_dens_in)
        else:
            density_pred = None
        # density-guided detection modulates the main detection map (identity at
        # init, see DensityGuidedDetection). Main branch only.
        if density_pred is not None and self.density_guided:
            adapted_f = self.density_guide(adapted_f, density_pred)
        adapted_f = adapted_f.view(batch_size, self.emb_dim, -1).permute(0, 2, 1)
        centerness = self.class_embed(adapted_f).view(batch_size, w, h, 1).permute(0, 3, 1, 2)
        outputs_coord = self.bbox_embed(adapted_f).sigmoid().view(batch_size, w, h, 4).permute(0, 3, 1, 2)
        outputs, ref_points = boxes_with_scores(centerness, outputs_coord, sort=False, validate=self.inference)
        if density_pred is not None:
            # train/inference read outputs[i]['pred_density']
            for i in range(len(outputs)):
                outputs[i]['pred_density'] = density_pred[i]

        if not self.pretrain:
            adapted_f_aux = adapted_f_aux.view(batch_size, self.emb_dim, -1).permute(0, 2, 1)
            centerness_aux = self.class_embed_aux(adapted_f_aux).view(batch_size, w, h, 1).permute(0, 3, 1, 2)
            outputs_coord_aux = self.bbox_embed_aux(adapted_f_aux).sigmoid().view(batch_size, w, h, 4).permute(0, 3, 1, 2)
            outputs_aux, ref_points_aux = boxes_with_scores(centerness_aux, outputs_coord_aux, sort=False, validate=self.inference)

        if self.inference:
            # mask processing
            masks, ious, corrected_bboxes = self.sam_mask(feats, outputs)
            for i in range(len(outputs)):
                outputs[i]["scores"] = ious[i]
                outputs[i]["pred_boxes"] = corrected_bboxes[i].to(outputs[i]["pred_boxes"].device).unsqueeze(0) / x.shape[-1]
            return outputs, ref_points, centerness, outputs_coord, masks

        else:
            for i in range(len(outputs)):
                outputs[i]["scores"] = outputs[i]["box_v"]

        if self.pretrain:
            return outputs, ref_points, centerness, outputs_coord
        else:
            return outputs, ref_points, centerness, outputs_coord, (outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _resolve_depthfeats_channels(args):
    """Adapter input width for the cached decoder-feature path: PCA-k when >0, the
    checkpoint's path_1 width when -1. Lazy import avoids a load-time cycle."""
    from utils.depth_recipe import resolve_depthfeats_spec
    return resolve_depthfeats_spec(getattr(args, 'decoder_feat_channels_PCA', 16))[0]


def build_model(args):
    assert args.reduction in [4, 8, 16]

    return CNT(
        image_size=args.image_size,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        reduction=args.reduction,
        kernel_dim=args.kernel_dim,
        training=getattr(args, 'training', False),
        use_depth=getattr(args, 'use_depth', 0),
        depth_kernel_size=getattr(args, 'depth_kernel_size', 3),
        depth_feat_channels=getattr(args, 'depth_feat_channels', 16),
        use_density=getattr(args, 'use_density', 0),
        density_head_type=getattr(args, 'density_head_type', 'simple'),
        density_guided=getattr(args, 'density_guided', 0),
        density_detach=getattr(args, 'density_detach', 0),
        unfreeze_last_hiera=getattr(args, 'unfreeze_last_hiera', 0),
        depth_fuse_identity_init=getattr(args, 'depth_fuse_identity_init', 0),
        depth_feat_norm=getattr(args, 'depth_feat_norm', 'group'),
        depth_feat_norm_groups=getattr(args, 'depth_feat_norm_groups', 0),
        depth_target_size=getattr(args, 'depth_target_size', 0),
        depth_source=getattr(args, 'depth_source', 'decoder'), # fallbacks match the arg_parser
        depth_adapt=getattr(args, 'depth_adapt', 'conv'),
        depth_cues=getattr(args, 'depth_cues', 'learned'),
        depth_adapt_init=getattr(args, 'depth_adapt_init', 'orthogonal'),
        depth_adapt_masked_conv=getattr(args, 'depth_adapt_masked_conv', 1),
        sep_hiera_input=getattr(args, 'sep_hiera_input', 'cues'),
        sep_hiera_fullres=getattr(args, 'sep_hiera_fullres', 1),
        sep_hiera_per_level_gate=getattr(args, 'sep_hiera_per_level_gate', 1),
        ffm_norm=getattr(args, 'ffm_norm', 'group'),
        dino_input_size=getattr(args, 'dino_input_size', 0),
        # external depth: on when a depthmaps cache dir is set and depth fusion is on
        external_depth=(bool(getattr(args, 'depthmaps_dir', '')) and getattr(args, 'use_depth', 0) > 0),
        # _pdf<k> cache: k decoder-feature channels after the 1-ch map, only when
        # both caches are configured. build_model runs before the cache is made, so
        # the width is resolved deterministically here.
        external_depth_feats=(_resolve_depthfeats_channels(args)
                              if (bool(getattr(args, 'depthfeats_dir', ''))
                                  and bool(getattr(args, 'depthmaps_dir', ''))
                                  and getattr(args, 'use_depth', 0) > 0)
                              else 0),
        depth_hires_fusion=getattr(args, 'depth_hires_fusion', 0),
        depth_hires_norm=getattr(args, 'depth_hires_norm', 1),
    )
