"""Full PCA spectrum of the cached DAv2 path_1 depth features (_pdf8 cache),
checked against the shipped pca_basis.npz, plus PC/residual maps for val
images. Cluster paths hardcoded, run via sbatch tools/hpc_pca_dim_analysis.sh.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from utils.depth_recipe import (_DEPTHFEATS_FIT_IMAGES, _DEPTHFEATS_FIT_PIX,
                                _load_dav2_for_feats)

HOME = '/d/hpc/home/er52565/GECO2'
DATASET_SPECS = {
    'FSCD147': dict(
        data_path='/d/hpc/projects/FRI/pelhanj/fsc147',
        feats_dir=f'{HOME}/FSCD147Dataset/depthfeats8_dino512',
    ),
    'IOCfish5k': dict(
        data_path=f'{HOME}/IOCfish5kDataset/dataset/IOCfish5k',
        feats_dir=f'{HOME}/IOCfish5kDataset/dataset/IOCfish5k-D/depthfeats8_dino512',
    ),
    'MCAC': dict(
        data_path='/d/hpc/projects/FRI/pelhanj/MCAC',
        feats_dir=f'{HOME}/MCACDataset/depthfeats8_dino512',
    ),
}

# diverging cmap for the signed PC maps, sequential for magnitudes
CMAP_SIGNED = 'RdBu_r'
CMAP_DEPTH = 'inferno'
CMAP_MAG = 'magma'
BAR_COLOR = '#4477AA'
LINE_COLOR = '#333333'


def _log(msg):
    print(msg, flush=True)


def build_id_lists(name, spec):
    """(fit_paths, val id->path), fit set = first-48 sorted train ids as in
    prepare_depthfeats."""
    from utils.data import FSC147DATASET, IOCFish5kDataset, MCACDataset
    cls = {'FSCD147': FSC147DATASET, 'IOCfish5k': IOCFish5kDataset,
           'MCAC': MCACDataset}[name]
    train = cls(spec['data_path'], 1024, split='train', training=True)
    val = cls(spec['data_path'], 1024, split='val', training=False)
    t = train.id_to_imgpath()
    fit_paths = [p for _, p in sorted(t.items())][:_DEPTHFEATS_FIT_IMAGES]
    return fit_paths, dict(sorted(val.id_to_imgpath().items()))


@torch.no_grad()
def forward_depth_and_path1(model, bgr, device, input_size):
    """compute_path1's preprocessing, but returns (disparity, path_1)."""
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
            depth, path_1 = model(xr, return_path_1=True)
    else:
        depth, path_1 = model(xr, return_path_1=True)
    return depth.squeeze().float().cpu(), path_1[0].float().cpu()


def fit_full_spectrum(model, fit_paths, width, input_size, device):
    """Same accumulation as _fit_pca_basis but keeps the full eigen-spectrum."""
    import cv2
    n, used = 0, 0
    mu_acc = torch.zeros(width, dtype=torch.float64)
    cov_acc = torch.zeros(width, width, dtype=torch.float64)
    for p in fit_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            _log(f'  [warn] unreadable fit image {p}')
            continue
        _, feats = forward_depth_and_path1(model, bgr, device, input_size)
        X = feats.reshape(width, -1).double()
        step = max(1, X.shape[1] // _DEPTHFEATS_FIT_PIX)
        X = X[:, ::step]
        mu_acc += X.sum(dim=1)
        cov_acc += X @ X.t()
        n += X.shape[1]
        used += 1
    mean = mu_acc / n
    cov = cov_acc / n - torch.outer(mean, mean)
    evals, vecs = torch.linalg.eigh(cov) # ascending
    evals = evals.flip(0).clamp_min(0) # descending, PC1 first
    vecs = vecs.flip(-1) # col j = PC j+1
    return mean, evals, vecs, used, n


def spectrum_stats(evals, kmax=16):
    tot = float(evals.sum())
    evr = (evals / tot).numpy()
    cum = np.cumsum(evr)
    pr_full = float(evals.sum() ** 2 / (evals ** 2).sum())
    top8 = evals[:8]
    pr_top8 = float(top8.sum() ** 2 / (top8 ** 2).sum())
    return dict(
        evr=[float(v) for v in evr[:kmax]],
        cum=[float(v) for v in cum[:kmax]],
        pr_full=pr_full,
        pr_top8=pr_top8,
        var_within_top8=[float(v) for v in (top8 / top8.sum()).numpy()],
    )


def project(feats, mean_t, comps_t):
    """Project onto the shipped top-8 basis.
    Returns (Y (8,h,w), per-pixel centered energy (h,w))."""
    C, H, W = feats.shape
    X = feats.reshape(C, -1)
    Xc = X - mean_t.unsqueeze(1)
    Y = comps_t.t() @ Xc
    return Y.reshape(-1, H, W), (Xc ** 2).sum(0).reshape(H, W)


def robust_sym(a, q=99.0):
    v = float(np.percentile(np.abs(a), q))
    return v if v > 0 else 1.0


def render_spectrum_fig(plt, stats, name, out_png):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ks = np.arange(1, len(stats['evr']) + 1)
    evr_pct = np.array(stats['evr']) * 100
    cum_pct = np.array(stats['cum']) * 100
    ax.bar(ks, evr_pct, color=BAR_COLOR, width=0.72, label='per-PC variance')
    ax.plot(ks, cum_pct, color=LINE_COLOR, lw=2, marker='o', ms=4,
            label='cumulative')
    for k in (2, 5, 8):
        ax.annotate(f'{cum_pct[k - 1]:.1f}%', (k, cum_pct[k - 1]),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9, color=LINE_COLOR)
    ax.set_xticks(ks)
    ax.set_xlabel('principal component of the 256-ch path_1 depth features')
    ax.set_ylabel('% of total variance')
    ax.set_ylim(0, 105)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', color='0.9', lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='center right')
    ax.set_title(f'{name}: depth-feature PCA spectrum '
                 f'(fit images of pca_basis.npz; PR={stats["pr_full"]:.2f})',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    fig.savefig(str(out_png).replace('.png', '.pdf'))
    plt.close(fig)


def _imshow(ax, img, cmap=None, vmin=None, vmax=None, title=''):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=8.5)
    ax.axis('off')


def render_image_figs(plt, name, img_id, rgb, disp, Y, xc_sq, stats, out_dir,
                      crosscheck=None):
    """Overview + first-2/3/4/5 figures for one image. Y = (8,h,w) projections,
    xc_sq = per-pixel centered feature energy."""
    import matplotlib.gridspec as gridspec
    Yn = Y.numpy()
    xc_sq_n = xc_sq.numpy()
    norm_map = np.sqrt(xc_sq_n)
    tot_energy = float(xc_sq_n.sum())
    cum_energy = np.cumsum([float((Yn[j] ** 2).sum()) for j in range(8)])
    captured = cum_energy / tot_energy # per-image, k=1..8
    resid = {k: np.sqrt(np.clip(xc_sq_n - np.sum(Yn[:k] ** 2, axis=0), 0, None))
             for k in (2, 3, 4, 5, 8)}
    vmax_mag = float(np.percentile(norm_map, 99))
    disp_n = disp.numpy()
    disp_n = (disp_n - disp_n.min()) / (np.ptp(disp_n) + 1e-9)
    cum_pct = np.array(stats['cum']) * 100

    # overview fig
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(3, 8, figure=fig, hspace=0.28, wspace=0.06)
    ax = fig.add_subplot(gs[0, 0:2]); _imshow(ax, rgb, title=f'{name} {img_id} - RGB')
    ax = fig.add_subplot(gs[0, 2:4]); _imshow(ax, disp_n, cmap=CMAP_DEPTH,
                                              title='DAv2 disparity (same forward)')
    ax = fig.add_subplot(gs[0, 4:7])
    ks = np.arange(1, 17)
    ax.bar(ks, np.array(stats['evr']) * 100, color=BAR_COLOR, width=0.72,
           label='per-PC (dataset)')
    ax.plot(ks, cum_pct, color=LINE_COLOR, lw=1.8, marker='o', ms=3.5,
            label='cumulative (dataset)')
    ax.plot(np.arange(1, 9), captured * 100, color='#EE6677', lw=1.8,
            marker='s', ms=3.5, label='cumulative (this image)')
    ax.set_xticks(ks); ax.set_ylim(0, 105)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', color='0.9', lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='center right')
    ax.set_title('explained variance / captured energy [%]', fontsize=9)
    ax = fig.add_subplot(gs[0, 7]); ax.axis('off')
    txt = (f"PR (256-ch)  = {stats['pr_full']:.2f}\n"
           f"PR (top-8)   = {stats['pr_top8']:.2f}\n"
           f"cum dataset:\n"
           f"  k=1 {cum_pct[0]:5.1f}%   k=2 {cum_pct[1]:5.1f}%\n"
           f"  k=3 {cum_pct[2]:5.1f}%   k=4 {cum_pct[3]:5.1f}%\n"
           f"  k=5 {cum_pct[4]:5.1f}%   k=8 {cum_pct[7]:5.1f}%\n"
           f"this image:\n"
           f"  k=2 {captured[1] * 100:5.1f}%   k=5 {captured[4] * 100:5.1f}%\n"
           f"  k=8 {captured[7] * 100:5.1f}%")
    if crosscheck is not None:
        txt += f"\ncache check:\n  max|diff|={crosscheck['max_abs_diff']:.3g}\n" \
               f"  mean r={crosscheck['mean_corr']:.4f}"
    ax.text(0.0, 0.98, txt, va='top', ha='left', fontsize=8.5, family='monospace')
    for j in range(8):
        ax = fig.add_subplot(gs[1, j])
        v = robust_sym(Yn[j])
        _imshow(ax, Yn[j], cmap=CMAP_SIGNED, vmin=-v, vmax=v,
                title=f'PC{j + 1} ({stats["evr"][j] * 100:.1f}% var)')
    ax = fig.add_subplot(gs[2, 0])
    _imshow(ax, norm_map, cmap=CMAP_MAG, vmin=0, vmax=vmax_mag,
            title='||centered feats|| per px')
    for i, k in enumerate((2, 3, 4, 5, 8)):
        ax = fig.add_subplot(gs[2, i + 1])
        _imshow(ax, resid[k], cmap=CMAP_MAG, vmin=0, vmax=vmax_mag,
                title=f'resid. after {k} PCs\n'
                      f'({captured[k - 1] * 100:.1f}% captured)')
    ax = fig.add_subplot(gs[2, 6:8]); ax.axis('off')
    fig.suptitle(f'{name} image {img_id}: PCA of 256-ch DAv2 path_1 depth '
                 f'features (the _pdf8 cache basis)', fontsize=12, y=0.99)
    fig.savefig(out_dir / f'{img_id}_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # first-k figs
    for k in (2, 3, 4, 5):
        ncol = k + 3
        fig, axes = plt.subplots(1, ncol, figsize=(2.9 * ncol, 3.3))
        _imshow(axes[0], rgb, title='RGB')
        _imshow(axes[1], disp_n, cmap=CMAP_DEPTH, title='DAv2 disparity')
        for j in range(k):
            v = robust_sym(Yn[j])
            _imshow(axes[2 + j], Yn[j], cmap=CMAP_SIGNED, vmin=-v, vmax=v,
                    title=f'PC{j + 1} ({stats["evr"][j] * 100:.1f}% var)')
        _imshow(axes[k + 2], resid[k], cmap=CMAP_MAG, vmin=0, vmax=vmax_mag,
                title=f'residual after {k} PCs')
        fig.suptitle(
            f'{name} {img_id} - first {k} PCs: {stats["cum"][k - 1] * 100:.1f}% '
            f'of dataset variance, {captured[k - 1] * 100:.1f}% of this image\'s '
            f'feature energy', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(out_dir / f'{img_id}_first{k}.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
    return captured


def crosscheck_cache(feats_dir, img_id, Y):
    """Compare our projection against the cached <id>.npy the training runs
    consumed (fp16, tiny diffs expected)."""
    p = Path(feats_dir) / f'{img_id}.npy'
    if not p.exists():
        return None
    cached = np.load(p).astype(np.float32)
    ours = Y.to(torch.float16).float().numpy()
    if cached.shape != ours.shape:
        return dict(max_abs_diff=float('nan'), mean_corr=float('nan'),
                    note=f'shape {cached.shape} vs {ours.shape}')
    corrs = []
    for j in range(cached.shape[0]):
        a, b = cached[j].ravel(), ours[j].ravel()
        sa, sb = a.std(), b.std()
        corrs.append(float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else 1.0)
    return dict(max_abs_diff=float(np.abs(cached - ours).max()),
                mean_corr=float(np.mean(corrs)))


def run_dataset(name, spec, args, model, device, plt):
    import cv2
    out_dir = Path(args.out) / name
    cand_dir = out_dir / 'candidates'
    cand_dir.mkdir(parents=True, exist_ok=True)

    basis_p = Path(spec['feats_dir']) / 'pca_basis.npz'
    b = np.load(basis_p)
    assert int(b['k']) == 8 and int(b['input_size']) == args.input_size, \
        f'{name}: unexpected basis (k={int(b["k"])}, dino={int(b["input_size"])})'
    shipped_mean = torch.from_numpy(b['mean']).double()
    shipped_comps = torch.from_numpy(b['comps']).float() # (256, 8)
    width = shipped_comps.shape[0]

    idlists_p = out_dir / 'idlists.json'
    if idlists_p.exists():
        cached = json.load(open(idlists_p))
        fit_paths, val_ids = cached['fit_paths'], cached['val_ids']
        _log(f'[{name}] id lists loaded from {idlists_p}')
    else:
        _log(f'[{name}] building id lists ...')
        fit_paths, val_ids = build_id_lists(name, spec)
        with open(idlists_p, 'w') as f:
            json.dump(dict(fit_paths=[str(p) for p in fit_paths],
                           val_ids=val_ids), f)
    _log(f'[{name}] {len(fit_paths)} fit images, {len(val_ids)} val images')

    _log(f'[{name}] refitting full covariance spectrum on the fit set ...')
    mean, evals, vecs, used, npix = fit_full_spectrum(
        model, fit_paths, width, args.input_size, device)
    stats = spectrum_stats(evals)
    cos = [abs(float(shipped_comps[:, j].double() @ vecs[:, j]))
           for j in range(8)]
    mean_rel = float((mean - shipped_mean).norm() / shipped_mean.norm())
    stats.update(eigvec_cos_vs_shipped=cos, mean_rel_diff=mean_rel,
                 fit_images_used=used, fit_pixels=npix,
                 basis_file=str(basis_p), width=width,
                 checkpoint=str(b['checkpoint']) if 'checkpoint' in b else '?')
    _log(f'[{name}] PR={stats["pr_full"]:.2f}  cum@2={stats["cum"][1]:.3f}  '
         f'cum@5={stats["cum"][4]:.3f}  cum@8={stats["cum"][7]:.3f}  '
         f'min|cos|={min(cos):.4f}  mean_rel={mean_rel:.3g}')
    render_spectrum_fig(plt, stats, name, out_dir / 'spectrum.png')

    # candidates: explicit --ids or a val-split scan
    mean_f = shipped_mean.float()
    forced = (args.ids or {}).get(name)
    scan = []
    if forced:
        missing = [i for i in forced if i not in val_ids]
        if missing:
            raise SystemExit(f'[{name}] --ids not in the val split: {missing}')
        chosen = forced
    else:
        scan_ids = list(val_ids)[:args.n_scan]
        for i, img_id in enumerate(scan_ids):
            bgr = cv2.imread(str(val_ids[img_id]))
            if bgr is None:
                continue
            disp, feats = forward_depth_and_path1(model, bgr, device,
                                                  args.input_size)
            Y, xc_sq = project(feats, mean_f, shipped_comps)
            d = disp.numpy()
            d = (d - d.min()) / (np.ptp(d) + 1e-9)
            cap = np.cumsum([(Y[j] ** 2).sum().item() for j in range(8)])
            cap = cap / xc_sq.sum().item()
            scan.append(dict(id=img_id, depth_std=float(d.std()),
                             captured2=float(cap[1]), captured5=float(cap[4]),
                             captured8=float(cap[7])))
            if (i + 1) % 10 == 0:
                _log(f'[{name}] scanned {i + 1}/{len(scan_ids)} val images')
        scan.sort(key=lambda r: -r['depth_std'])
        top = [r['id'] for r in scan[:4]]
        rest = [r['id'] for r in scan[4:]]
        spread = [rest[i] for i in
                  np.linspace(0, len(rest) - 1, min(4, len(rest))).astype(int)] \
            if rest else []
        chosen = top + [i for i in spread if i not in top]
    _log(f'[{name}] rendering candidates: {chosen}')

    per_image = []
    for img_id in chosen:
        bgr = cv2.imread(str(val_ids[img_id]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        disp, feats = forward_depth_and_path1(model, bgr, device,
                                              args.input_size)
        Y, xc_sq = project(feats, mean_f, shipped_comps)
        cc = crosscheck_cache(spec['feats_dir'], img_id, Y)
        captured = render_image_figs(plt, name, img_id, rgb, disp, Y, xc_sq,
                                     stats, cand_dir, crosscheck=cc)
        per_image.append(dict(id=img_id,
                              captured=[float(c) for c in captured],
                              crosscheck=cc))
        _log(f'[{name}]   {img_id}: captured@2={captured[1]:.3f} '
             f'@5={captured[4]:.3f} @8={captured[7]:.3f}'
             + (f'  cache r={cc["mean_corr"]:.4f}' if cc else '  (no cache file)'))

    with open(out_dir / 'numbers.json', 'w') as f:
        json.dump(dict(dataset=name, spectrum=stats, scan=scan,
                       rendered=per_image), f, indent=2)
    _log(f'[{name}] done -> {out_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=f'{HOME}/pca_dim_analysis_out')
    ap.add_argument('--datasets', nargs='+', default=list(DATASET_SPECS),
                    choices=list(DATASET_SPECS))
    ap.add_argument('--n-scan', type=int, default=40,
                    help='val images scanned when picking candidates')
    ap.add_argument('--input-size', type=int, default=512,
                    help='dinoInSize of the cache (config DINO_INPUT_SIZE)')
    ap.add_argument('--ids', nargs='+', default=None, metavar='DS:ID1,ID2',
                    help='render exactly these val ids per dataset instead of scanning')
    args = ap.parse_args()
    if args.ids:
        args.ids = {e.split(':', 1)[0]: e.split(':', 1)[1].split(',')
                    for e in args.ids}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = os.environ.get('DEPTH_CHECKPOINT') or 'depth_anything_v2_vitl.pth'
    _log(f'[setup] device={device} ckpt={ckpt} input_size={args.input_size}')
    model, tap_width = _load_dav2_for_feats(device, ckpt)
    _log(f'[setup] tap width = {tap_width}')

    for name in args.datasets:
        run_dataset(name, DATASET_SPECS[name], args, model, device, plt)
    _log('[done] all datasets analyzed')


if __name__ == '__main__':
    main()
