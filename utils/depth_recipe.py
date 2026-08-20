"""pseudo-depth maps (IOCfish5k-D recipe: DAv2-Large at 4 scales, averaged,
min-max to [0,1], 8-bit grayscale JPG, near = bright) plus the on-disk caches."""

import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCALES = (512, 768, 1024, 1280)
# fp16 keeps the ViT-L pass in memory at the largest scale

_MAX_RETRIES = 6
_BACKOFF_BASE_SECONDS = 1.0


def _hf_download_with_retries(repo_id, filename):
    """hf_hub_download with exponential backoff."""
    from huggingface_hub import hf_hub_download
    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            last_err = e
            if attempt == _MAX_RETRIES:
                break
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"[depth_recipe] hf_hub_download attempt {attempt}/{_MAX_RETRIES} "
                  f"failed ({type(e).__name__}: {e}); retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(
        f"Failed to download {repo_id}/{filename} after {_MAX_RETRIES} attempts. "
        f"Last error: {type(last_err).__name__}: {last_err}. "
        f"Pre-warm the HF cache from a node with internet access."
    ) from last_err


def _load_dav2_large(device):
    """Depth Anything V2 Large (DPT head) from the vendored repo clone, weights loaded."""
    project = Path(__file__).resolve().parents[1] # GECO2
    repo = project / 'Depther' / 'DepthAnythingV2'
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from depth_anything_v2.dpt import DepthAnythingV2
    ckpt = _hf_download_with_retries('depth-anything/Depth-Anything-V2-Large',
                                     'depth_anything_v2_vitl.pth')
    model = DepthAnythingV2(encoder='vitl', features=256,
                            out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.to(device).eval()


def _load_dav2_for_feats(device, ckpt_name):
    """DAv2 DPT model for this run's DEPTH_CHECKPOINT -> (model, tap_width).
    SDT heads have no path_1, so the cache only supports the DPT checkpoints."""
    from Depther.infer_depth import _DEPTH_CONFIGS
    cfg = _DEPTH_CONFIGS.get(ckpt_name)
    if cfg is None or cfg.get('head') != 'dpt':
        raise SystemExit(
            f"[depthfeats][FATAL] DEPTH_CHECKPOINT={ckpt_name} has no DPT path_1 tap "
            f"(unknown checkpoint or SDT head); the decoder-feature cache supports the "
            f"original DAv2 DPT checkpoints only.")
    project = Path(__file__).resolve().parents[1] # GECO2
    repo = project / 'Depther' / 'DepthAnythingV2'
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from depth_anything_v2.dpt import DepthAnythingV2
    ckpt = _hf_download_with_retries(cfg['repo_id'], ckpt_name)
    model = DepthAnythingV2(encoder=cfg['dav2_encoder'], features=cfg['dpt_features'],
                            out_channels=list(cfg['dpt_out_channels']))
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.to(device).eval(), int(cfg['dpt_features'])


@torch.no_grad()
def compute_paper_depth(model, bgr, device='cuda', scales=_SCALES):
    """The recipe on one BGR (cv2) image -> float32 HxW in [0,1], near = bright."""
    H, W = bgr.shape[:2]
    acc = np.zeros((H, W), np.float64)
    for l in scales:
        if device == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                d = model.infer_image(bgr, input_size=l) # native HxW disparity
        else:
            d = model.infer_image(bgr, input_size=l)
        acc += d.astype(np.float64)
    d = acc / len(scales)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return d.astype(np.float32)


def _timed_subtask(name):
    """Bracket a cache-prep call with [time] START/END lines on rank 0, in the shells'
    stage_start/stage_end format. rank and log are read from kwargs only."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            log = kwargs.get('log', print)
            quiet = kwargs.get('rank', 0) != 0
            t0 = time.time()
            if not quiet:
                log(f"[time] ---- {name} START {time.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                return fn(*args, **kwargs)
            finally:
                if not quiet:
                    dt = int(time.time() - t0)
                    log(f"[time] ---- {name} END   {time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"(elapsed {dt // 3600:02d}:{dt % 3600 // 60:02d}:{dt % 60:02d})")
        return wrapper
    return deco


@_timed_subtask('depthmap cache build/check')
def prepare_depthmaps(depthmaps_dir, id_to_imgpath, use_available,
                      device='cuda', rank=0, log=print):
    """Build the depth cache for every id in id_to_imgpath ({id: rgb_path}).
    use_available=True never generates, otherwise rank 0 fills in missing <id>.jpg."""
    depthmaps_dir = Path(depthmaps_dir)
    n_needed = len(id_to_imgpath)

    def _exists(i):
        return (depthmaps_dir / f'{i}.jpg').exists() or (depthmaps_dir / f'{i}.png').exists()

    if use_available:
        n_files = len(list(depthmaps_dir.glob('*'))) if depthmaps_dir.is_dir() else 0
        if n_files == 0:
            log("[depth] ============================================================")
            log(f"[depth][FATAL] --use_available_depthmaps is set but '{depthmaps_dir}' "
                f"has no files. Generate them first (run without the flag) or upload them. "
                f"Aborting the job.")
            log("[depth] ============================================================")
            sys.exit(2)
        n_present = sum(1 for i in id_to_imgpath if _exists(i))
        log("[depth] ============================================================")
        log("[depth] source = PRECOMPUTED depth-map cache (--use_available_depthmaps)")
        log(f"[depth]   dir    = {depthmaps_dir}")
        log(f"[depth]   loaded = {n_present}/{n_needed} maps needed for this split "
            f"(dir holds {n_files} files total)")
        if n_present < n_needed:
            log(f"[depth][WARN] {n_needed - n_present} needed maps are MISSING from the "
                f"cache -- those images will fail to load. Re-generate (drop "
                f"--use_available_depthmaps) to fill the gaps.")
        log("[depth]   the in-model depth model is NOT run (depth read from disk).")
        log("[depth] ============================================================")
        return

    depthmaps_dir.mkdir(parents=True, exist_ok=True)
    missing = [(i, p) for i, p in id_to_imgpath.items() if not _exists(i)]
    n_present = n_needed - len(missing)
    log("[depth] ============================================================")
    if not missing:
        log("[depth] source = ON-THE-FLY paper recipe -> ALL maps already cached")
        log(f"[depth]   dir    = {depthmaps_dir}")
        log(f"[depth]   loaded = {n_needed}/{n_needed} maps from cache (no generation)")
        log("[depth] ============================================================")
        return
    log("[depth] source = ON-THE-FLY generation (paper recipe: DAv2-Large, "
        f"short-side scales {list(_SCALES)}, averaged + normalised)")
    log(f"[depth]   dir    = {depthmaps_dir}  (caching maps as they are generated)")
    log(f"[depth]   cached = {n_present}/{n_needed} present; GENERATING {len(missing)} "
        f"missing on the fly{' (rank 0 only)' if rank == 0 else ''}, then reused every epoch")
    log("[depth] ============================================================")
    if rank != 0:
        return  # rank 0 generates, the others wait at the caller's barrier

    import cv2
    from PIL import Image
    model = _load_dav2_large(device)
    done = 0
    for i, p in missing:
        bgr = cv2.imread(str(p))
        if bgr is None:
            log(f"[depth][warn] cannot read image {p}; skipping {i}")
            continue
        d = compute_paper_depth(model, bgr, device=device)
        Image.fromarray((d * 255.0).astype(np.uint8)).save(
            depthmaps_dir / f'{i}.jpg', quality=95)
        done += 1
        if done % 200 == 0:
            log(f"[depth]   generated {done}/{len(missing)} ...")
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    log(f"[depth] done: generated {done} maps into {depthmaps_dir} (now cached for reuse).")


def attach_depthmaps(dataset, args, device='cuda', rank=0, log=print):
    """Check/generate the maps and point dataset at the cache (__getitem__ then
    appends the map as the 4th channel). Returns the effective dir, '' if unused."""
    if getattr(args, 'use_depth', 0) <= 0:
        return ''  # RGB-only model
    dm = getattr(args, 'depthmaps_dir', '')
    if not dm:
        log("[depth] source = IN-MODEL on-the-fly prediction (legacy; no --depthmaps_dir). "
            "Depth is predicted inside the model each forward and not cached.")
        return ''
    prepare_depthmaps(dm, dataset.id_to_imgpath(),
                      getattr(args, 'use_available_depthmaps', False),
                      device=device, rank=rank, log=log)
    dataset.depthmaps_dir = dm
    return dm


# PCA-k cached decoder features (_pdf<k>): the DPT path_1 tap on disk, PCA-projected
# to k channels. depth map sits at channel 3, features at 4..3+k. dinoInSize is
# stamped into pca_basis.npz, so a 1024 run cannot consume a 518 cache.
# Files: <id>.npy, fp16, (k, h', w').

_DEPTHFEATS_DINO_SIZE = 518 # fallback dinoInSize (DAv2 native)
_DEPTHFEATS_FIT_IMAGES = 48 # images the PCA basis is fit on
_DEPTHFEATS_FIT_PIX = 8000 # subsampled pixels per fit image


def resolve_dino_input_size(args):
    """The dinoInSize the in-model depth ViT would use, same rule as counter.py
    CNT.__init__. The feature cache has to be generated at this size."""
    dis = getattr(args, 'dino_input_size', 0)
    if dis and dis > 0:
        return int(dis)
    dfs = getattr(args, 'depth_target_size', 0)
    target = dfs if dfs and dfs > 0 else (
        getattr(args, 'image_size', 1024) * 4 // getattr(args, 'reduction', 16))
    return max(_DEPTHFEATS_DINO_SIZE, int(target))


def resolve_depthfeats_spec(depthfeats_channels):
    """--decoder_feat_channels_PCA -> (stored_channels, pca_on): >0 = that many
    principal components, -1 = every path_1 channel with PCA off."""
    if depthfeats_channels is not None and int(depthfeats_channels) < 0:
        from Depther.infer_depth import _DEPTH_CONFIGS
        ckpt = os.environ.get('DEPTH_CHECKPOINT') or 'depth_anything_v2_vitl.pth'
        width = int(_DEPTH_CONFIGS.get(ckpt, {}).get('dpt_features', 256))
        return width, False
    return int(depthfeats_channels), True


def _pca_basis_path(depthfeats_dir):
    return Path(depthfeats_dir) / 'pca_basis.npz'


@torch.no_grad()
def compute_path1(model, bgr, device='cuda', input_size=_DEPTHFEATS_DINO_SIZE):
    """DAv2 on one BGR image -> path_1 map, float32 CPU (dpt_features, h', w').
    Preprocessing follows DAv2Depther so the cache matches the in-model tap."""
    import cv2
    import torch.nn.functional as F
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(device)
    H, W = x.shape[-2:]
    s = input_size / max(H, W)
    h14 = max(14, int(round(H * s / 14)) * 14)
    w14 = max(14, int(round(W * s / 14)) * 14)
    xr = F.interpolate(x, size=(h14, w14), mode='bilinear', align_corners=False)
    if device == 'cuda':
        with torch.autocast('cuda', dtype=torch.float16):
            _, path_1 = model(xr, return_path_1=True)
    else:
        _, path_1 = model(xr, return_path_1=True)
    return path_1[0].float().cpu() # (dpt_features, h', w')


def _fit_pca_basis(model, fit_paths, k, input_size, device, log, width):
    """Fit the (mean, top-k eigenvector) PCA basis of the path_1 channels on a fixed
    image sample. fit_paths should be train-only so the basis never sees val/test."""
    import cv2
    n = 0
    mu_acc = torch.zeros(width, dtype=torch.float64)
    cov_acc = torch.zeros(width, width, dtype=torch.float64)
    used = 0
    for p in fit_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        feats = compute_path1(model, bgr, device=device, input_size=input_size)
        X = feats.reshape(width, -1).double()
        step = max(1, X.shape[1] // _DEPTHFEATS_FIT_PIX)
        X = X[:, ::step]
        mu_acc += X.sum(dim=1)
        cov_acc += X @ X.t()
        n += X.shape[1]
        used += 1
    if n < width:
        raise RuntimeError(f"[depthfeats] PCA fit failed: only {n} pixels from "
                           f"{used} readable images.")
    mean = mu_acc / n
    cov = cov_acc / n - torch.outer(mean, mean)
    evals, vecs = torch.linalg.eigh(cov) # ascending
    comps = vecs[:, -k:].flip(-1).contiguous() # (width, k), PC1 first
    log(f"[depthfeats]   PCA basis fit on {used} images / {n} pixels -> top-{k} of {width} ch "
        f"(dinoInSize {input_size})")
    # explained variance, logged at fit time only
    ev = evals.flip(0).clamp_min(0)
    evr = (ev / ev.sum()).tolist()
    cum = 0.0
    cums = []
    for v in evr[:k]:
        cum += v
        cums.append(cum)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    log("[depthfeats]   explained variance: "
        + " ".join(f"PC{i + 1}={evr[i] * 100:.1f}%" for i in range(min(4, k)))
        + f" ... cum@{k}={cums[-1] * 100:.1f}% (participation ratio {pr:.2f})")
    return mean.float(), comps.float()


def _validate_basis(basis, k, pca, input_size, where, log, ckpt=None):
    """Abort when the cache was built with a different k/pca/dinoInSize/checkpoint."""
    problems = []
    if int(basis['k']) != k:
        problems.append(f"channels={int(basis['k'])} (cache) vs {k} (this run)")
    cache_pca = bool(basis['pca']) if 'pca' in basis else True # unstamped caches = PCA
    if cache_pca != bool(pca):
        problems.append(f"pca={cache_pca} (cache) vs pca={bool(pca)} (this run)")
    if int(basis['input_size']) != int(input_size):
        problems.append(f"dinoInSize={int(basis['input_size'])} (cache) vs {int(input_size)} "
                        f"(this run's resolved --dino_input_size)")
    if ckpt is not None and 'checkpoint' in basis and str(basis['checkpoint']) != str(ckpt):
        problems.append(f"checkpoint={basis['checkpoint']} (cache) vs {ckpt} "
                        f"(this run's DEPTH_CHECKPOINT)")
    if problems:
        log("[depthfeats] ============================================================")
        log(f"[depthfeats][FATAL] cache at {where} does not match this run: "
            + "; ".join(problems) + ". Use a matching --depthfeats_dir (one cache dir "
            "per (k, dinoInSize) combination; the shells derive "
            "depthfeats<k|raw>[_dino<size>], e.g. .../depthfeats16_dino1024).")
        log("[depthfeats] ============================================================")
        sys.exit(2)


@_timed_subtask('depthfeats cache build/check')
def prepare_depthfeats(depthfeats_dir, id_to_imgpath, use_available, k=16, pca=True,
                       input_size=0, fit_ids=None, device='cuda', rank=0, log=print):
    """Build the decoder-feature cache for every id in id_to_imgpath.
    pca=False stores the raw tap instead of the top-k components. input_size is the
    resolved dinoInSize (0 -> 518), fit_ids the train ids the PCA fit is limited to."""
    depthfeats_dir = Path(depthfeats_dir)
    input_size = int(input_size) or _DEPTHFEATS_DINO_SIZE
    # resolve DEPTH_CHECKPOINT once, the cache is supplier-specific
    ckpt_name = os.environ.get('DEPTH_CHECKPOINT') or 'depth_anything_v2_vitl.pth'
    n_needed = len(id_to_imgpath)
    basis_p = _pca_basis_path(depthfeats_dir)
    mode = f"PCA->{k} ch" if pca else f"RAW {k} ch (PCA OFF)"

    def _exists(i):
        return (depthfeats_dir / f'{i}.npy').exists()

    if use_available:
        if not basis_p.exists():
            log("[depthfeats] ============================================================")
            log(f"[depthfeats][FATAL] --use_available_depthfeats is set but '{basis_p}' "
                f"is missing. Generate the cache first (run without the flag). Aborting.")
            log("[depthfeats] ============================================================")
            sys.exit(2)
        _validate_basis(np.load(basis_p), k, pca, input_size, depthfeats_dir, log, ckpt=ckpt_name)
        n_present = sum(1 for i in id_to_imgpath if _exists(i))
        log("[depthfeats] ============================================================")
        log("[depthfeats] source = PRECOMPUTED decoder-feature cache "
            "(--use_available_depthfeats)")
        log(f"[depthfeats]   dir    = {depthfeats_dir} ({mode}, dinoInSize={input_size})")
        log(f"[depthfeats]   loaded = {n_present}/{n_needed} feature files needed")
        if n_present < n_needed:
            log(f"[depthfeats][WARN] {n_needed - n_present} needed feature files are "
                f"MISSING -- those images will fail to load. Re-generate (drop "
                f"--use_available_depthfeats) to fill the gaps.")
        log("[depthfeats]   NOTE: cached features are computed on the un-augmented image "
            "and warped by the augs (speed-for-fidelity trade vs the in-model tap).")
        log("[depthfeats] ============================================================")
        return

    depthfeats_dir.mkdir(parents=True, exist_ok=True)
    missing = [(i, p) for i, p in id_to_imgpath.items() if not _exists(i)]
    log("[depthfeats] ============================================================")
    if not missing and basis_p.exists():
        _validate_basis(np.load(basis_p), k, pca, input_size, depthfeats_dir, log, ckpt=ckpt_name)
        log("[depthfeats] source = decoder-feature cache -> ALL files already cached")
        log(f"[depthfeats]   dir    = {depthfeats_dir} ({mode}, dinoInSize={input_size})")
        log(f"[depthfeats]   loaded = {n_needed}/{n_needed} from cache (no generation)")
        log("[depthfeats] ============================================================")
        return
    mb = 2 * k * (input_size * 8 // 14) ** 2 / 1e6
    log(f"[depthfeats] source = ON-THE-FLY generation ({ckpt_name} path_1 tap @ "
        f"dinoInSize {input_size}, {mode}, fp16; grid ~ {input_size * 8 // 14} px "
        f"long side, ~{mb:.1f} MB/image upper bound)")
    if not pca:
        log(f"[depthfeats][WARN] RAW mode (PCA OFF) stores every path_1 channel: ~"
            f"{mb * n_needed / 1e3:.0f} GB for this split alone -- far larger than the "
            f"PCA cache and likely over the cluster quota. This is information-equivalent "
            f"to the in-model tap (USE_DEPTHFEATS=0); use it only to remove the PCA step "
            f"as an ablation.")
    log(f"[depthfeats]   dir    = {depthfeats_dir}")
    log(f"[depthfeats]   cached = {n_needed - len(missing)}/{n_needed} present; "
        f"GENERATING {len(missing)} missing (rank 0 only), then reused every epoch")
    log("[depthfeats]   NOTE: cached features are computed on the un-augmented image "
        "and warped by the augs (speed-for-fidelity trade vs the in-model tap).")
    log("[depthfeats] ============================================================")
    if rank != 0:
        return

    import cv2
    model, tap_width = _load_dav2_for_feats(device, ckpt_name)
    if not pca and k != tap_width:
        log(f"[depthfeats][FATAL] RAW mode width mismatch: k={k} vs the {tap_width}-ch "
            f"path_1 tap of {ckpt_name} (resolve_depthfeats_spec and prepare_depthfeats "
            f"resolved different DEPTH_CHECKPOINT values).")
        sys.exit(2)
    mean = comps = None
    if pca:
        if basis_p.exists():
            b = np.load(basis_p)
            _validate_basis(b, k, pca, input_size, depthfeats_dir, log, ckpt=ckpt_name)
            mean = torch.from_numpy(b['mean'])
            comps = torch.from_numpy(b['comps'])
            log(f"[depthfeats]   reusing PCA basis from {basis_p}")
        else:
            if fit_ids is not None:
                fit_src = [(i, id_to_imgpath[i]) for i in fit_ids if i in id_to_imgpath]
            else:
                log("[depthfeats][WARN] no fit_ids given -> fitting the PCA basis on ALL "
                    "provided ids (may include val/test). Pass the train ids for a "
                    "train-only, leak-free basis.")
                fit_src = list(id_to_imgpath.items())
            fit_paths = [p for _, p in sorted(fit_src)][:_DEPTHFEATS_FIT_IMAGES]
            mean, comps = _fit_pca_basis(model, fit_paths, k, input_size, device, log,
                                         width=tap_width)
            np.savez(basis_p, mean=mean.numpy(), comps=comps.numpy(), k=k, pca=True,
                     input_size=input_size, checkpoint=ckpt_name)
            log(f"[depthfeats]   PCA basis saved to {basis_p}")
    elif not basis_p.exists():
        # raw mode: no basis arrays, but stamp the mode/shape for validation
        np.savez(basis_p, k=k, pca=False, input_size=input_size,
                 checkpoint=ckpt_name)
        log(f"[depthfeats]   raw-mode marker saved to {basis_p} (no PCA basis)")
    else:
        _validate_basis(np.load(basis_p), k, pca, input_size, depthfeats_dir, log, ckpt=ckpt_name)
    done, total_bytes = 0, 0
    for i, p in missing:
        bgr = cv2.imread(str(p))
        if bgr is None:
            log(f"[depthfeats][warn] cannot read image {p}; skipping {i}")
            continue
        feats = compute_path1(model, bgr, device=device, input_size=input_size)  # (tap_width,h,w)
        if pca:
            X = feats.reshape(feats.shape[0], -1)
            Y = (comps.t() @ X) - (comps.t() @ mean).unsqueeze(1) # (k, h*w)
            out = Y.reshape(k, feats.shape[-2], feats.shape[-1])
        else:
            out = feats # raw (k == tap width)
        np.save(depthfeats_dir / f'{i}.npy', out.to(torch.float16).numpy())
        total_bytes += (k * feats.shape[-2] * feats.shape[-1] * 2)
        done += 1
        if done % 100 == 0:
            log(f"[depthfeats]   generated {done}/{len(missing)} "
                f"(~{total_bytes / 1e9:.1f} GB so far) ...")
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    log(f"[depthfeats] done: generated {done} feature files "
        f"(~{total_bytes / 1e9:.1f} GB) into {depthfeats_dir}.")


def load_depth_feats(depthfeats_dir, img_id, target_hw):
    """Load <depthfeats_dir>/<id>.npy as a (k, H, W) float32 tensor resized to
    target_hw, so it aligns with the native RGB before the geometric augs."""
    import torch.nn.functional as F
    path = Path(depthfeats_dir) / f'{img_id}.npy'
    if not path.exists():
        raise FileNotFoundError(
            f"[depthfeats] missing feature file for id '{img_id}' in {depthfeats_dir} "
            f"(expected {img_id}.npy). Generate the cache (run without "
            f"--use_available_depthfeats) before training/inference.")
    arr = np.load(path).astype(np.float32) # (k, h, w)
    d = torch.from_numpy(arr).unsqueeze(0) # (1, k, h, w)
    H, W = target_hw
    if d.shape[-2:] != (H, W):
        d = F.interpolate(d, size=(H, W), mode='bilinear', align_corners=False)
    return d[0] # (k, H, W)


def attach_depthfeats(dataset, args, device='cuda', rank=0, log=print):
    """Same as attach_depthmaps but for the feature cache (channels 4..3+k).
    Needs the depth-map cache attached first."""
    if getattr(args, 'use_depth', 0) <= 0:
        return ''
    df = getattr(args, 'depthfeats_dir', '')
    if not df:
        return ''
    if not getattr(dataset, 'depthmaps_dir', ''):
        raise SystemExit(
            "[depthfeats][FATAL] --depthfeats_dir requires the 1-ch depth-map cache "
            "(--depthmaps_dir / attach_depthmaps first): the feature cache rides as "
            "channels 4..3+k ON TOP of the 4th-channel depth map.")
    k, pca = resolve_depthfeats_spec(getattr(args, 'decoder_feat_channels_PCA', 16))
    # the basis already exists from training, so no fit_ids needed here
    prepare_depthfeats(df, dataset.id_to_imgpath(),
                       getattr(args, 'use_available_depthfeats', False),
                       k=k, pca=pca, input_size=resolve_dino_input_size(args),
                       device=device, rank=rank, log=log)
    dataset.depthfeats_dir = df
    return df


# depth image -> single channel. Maps arrive as greyscale, greyscale-in-RGB (the
# released IOCfish5k-D ones), or a colormapped viz. .convert('L') would scramble the
# ordering of the last kind, so the best-fitting colormap is inverted instead.
# grey vs colour: mean channel spread is 0 grey-in-RGB, ~90-110 colormapped.
_GRAY_SPREAD_TOL = 6.0
# colormaps a depth viz plausibly uses, lowest-residual LUT wins
_DEPTH_CMAPS = ('turbo', 'inferno', 'magma', 'viridis', 'plasma', 'jet', 'gray')
_CMAP_LUTS = {} # (name, n) -> (n, 3) float32 LUT, built lazily
_COLORMAP_WARNED = set() # directories already warned about


def _cmap_lut(name, n=256):
    key = (name, n)
    lut = _CMAP_LUTS.get(key)
    if lut is None:
        try:
            from matplotlib import colormaps as _cmaps
            cmap = _cmaps[name]
        except Exception:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(name)
        lut = cmap(np.linspace(0.0, 1.0, n))[:, :3].astype(np.float32)
        _CMAP_LUTS[key] = lut
    return lut


def _nearest_lut(pts, lut, chunk=50000):
    """Index of the nearest LUT row (M,3) for each point (N,3), chunked to keep the
    distance tensor small."""
    out = np.empty(pts.shape[0], np.int64)
    for s in range(0, pts.shape[0], chunk):
        c = pts[s:s + chunk]
        d = ((c[:, None, :] - lut[None, :, :]) ** 2).sum(2)
        out[s:s + chunk] = d.argmin(1)
    return out


def _invert_colormap(rgb01):
    """rgb01: (H,W,3) float32 in [0,1] from a colormapped depth viz -> (gray (H,W)
    float32 in [0,1], info_str). Without matplotlib it uses plain luminance."""
    try:
        import matplotlib
    except Exception:
        gray = rgb01 @ np.array([0.299, 0.587, 0.114], np.float32)
        return gray, 'luminance-fallback(no matplotlib)'
    h, w, _ = rgb01.shape
    flat = rgb01.reshape(-1, 3)
    # pick the colormap on a cheap subsample
    step = max(1, flat.shape[0] // 20000)
    sub = flat[::step]
    best_name, best_res = None, None
    for nm in _DEPTH_CMAPS:
        lut = _cmap_lut(nm)
        idx = _nearest_lut(sub, lut)
        res = float(np.mean((lut[idx] - sub) ** 2))
        if best_res is None or res < best_res:
            best_name, best_res = nm, res
    # invert the full map with the chosen LUT
    lut = _cmap_lut(best_name)
    idx = _nearest_lut(flat, lut)
    gray = (idx.astype(np.float32) / (lut.shape[0] - 1)).reshape(h, w)
    return gray, f"cmap='{best_name}' residual={best_res:.4f}"


def depth_image_to_gray(img, src_dir=None):
    """PIL depth image -> (H, W) float32 in [0,1], near = bright (see the block
    comment above). src_dir only de-duplicates the colour warning."""
    mode = img.mode
    if mode in ('I', 'I;16', 'I;16B', 'F'):
        # higher bit-depth / float depth: absolute scale is unknown -> min-max to [0,1]
        a = np.asarray(img, np.float32)
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-8)
    if mode == 'L': # true greyscale
        return np.asarray(img, np.float32) / 255.0
    rgb = np.asarray(img.convert('RGB'), np.float32) # 'P'/'LA'/'RGB'/'RGBA' -> RGB
    spread = rgb.max(2) - rgb.min(2)
    mean_spread = float(spread.mean())
    if mean_spread <= _GRAY_SPREAD_TOL: # greyscale in a colour container
        return rgb.mean(2) / 255.0
    gray, info = _invert_colormap(rgb / 255.0) # colormapped visualization
    if src_dir is not None and src_dir not in _COLORMAP_WARNED:
        _COLORMAP_WARNED.add(src_dir)
        print(f"[depthmaps][WARN] '{src_dir}' holds COLOUR (colormapped) depth maps, "
              f"not greyscale (mean channel-spread {mean_spread:.0f}). Inverting the "
              f"colour map to recover the scalar depth ({info}). If this is a depth "
              f"*visualization* directory (e.g. IOCfish5k-D/color), point "
              f"--depthmaps_dir at the raw greyscale maps instead (e.g. .../depthmaps); "
              f"verify the near/far orientation, since a colormap's direction is "
              f"ambiguous (we assume colormap-parameter high = near = bright).")
    return gray


def load_depth_map(depthmaps_dir, img_id, target_hw):
    """Load <depthmaps_dir>/<id>.{jpg,png} as a (1, H, W) float tensor in [0,1]
    (near=bright), resized to target_hw to align with the native RGB before the augs."""
    from PIL import Image
    import torch.nn.functional as F
    depthmaps_dir = Path(depthmaps_dir)
    path = depthmaps_dir / f'{img_id}.jpg'
    if not path.exists():
        path = depthmaps_dir / f'{img_id}.png'
    if not path.exists():
        raise FileNotFoundError(
            f"[depthmaps] missing depth map for id '{img_id}' in {depthmaps_dir} "
            f"(expected {img_id}.jpg or .png). Generate the cache (run without "
            f"--use_available_depthmaps) before training/inference.")
    arr = depth_image_to_gray(Image.open(path), src_dir=str(depthmaps_dir))
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    d = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) # (1,1,h,w)
    H, W = target_hw
    if d.shape[-2:] != (H, W):
        d = F.interpolate(d, size=(H, W), mode='bilinear', align_corners=False)
    return d[0] # (1, H, W)
