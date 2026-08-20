"""Shared parsers for the *_results.txt files written by the inference scripts."""
import os
import re
import sys
import glob


def _f(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def _parse_full_row(block_text):
    """'full' row of a metric block. Trailing SRE appears only in MCAC files."""
    m = re.search(r"full\s+n=\s*(\d+)\s+MAE=\s*([\d.]+)\s+MSE\*=\s*([\d.]+)\s+NAE=\s*([\d.]+)"
                  r"(?:\s+SRE=\s*([\d.]+))?", block_text)
    if not m:
        return None
    return {'n': int(m.group(1)), 'mae': float(m.group(2)),
            'rmse': float(m.group(3)), 'nae': float(m.group(4)),
            'sre': float(m.group(5)) if m.group(5) is not None else None}


def _sre_from_rows(rows, gt_key, pred_key):
    """SRE (MCAC Eq. 6) from per-image rows. DPred is rounded to 2 decimals,
    so density SRE can be ~1e-3 off the printed value."""
    pairs = [(r[gt_key], r[pred_key]) for r in rows if r.get(pred_key) is not None]
    if not pairs:
        return None
    return (sum((g - p) ** 2 / max(g, 1) for g, p in pairs) / len(pairs)) ** 0.5


def _slice(text, start_pat, stop_pats):
    """text from the first start_pat match up to the next stop pattern."""
    sm = re.search(start_pat, text)
    if not sm:
        return None
    rest = text[sm.end():]
    ends = [m.start() for p in stop_pats for m in [re.search(p, rest)] if m]
    return rest[:min(ends)] if ends else rest


_STOPS = [r"\nCount metrics", r"\nBox-count metrics", r"\nDensity-integral count metrics",
          r"\nBbox-quality", r"\nPer-threshold", r"\n\s*Idx\s+Image ID"]


def parse_results_file(path):
    """Parse one *_results.txt into a dict."""
    text = open(path).read()
    d = {'path': path, 'file': os.path.basename(path)}

    # summary block
    d['model'] = _f(r"Model path:\s*(\S+)", text, str)
    d['exp_tag'] = _f(r"Experiment tag:\s*(\S+)", text, str)
    d['mode'] = _f(r"\n\s*Mode:\s*(\S+)", text, str) or _f(r"Training regime:\s*(\S+)", text, str)
    d['split'] = _f(r"(?:Eval|Test) split:\s*(\S+)", text, str)
    d['infer'] = _f(r"Inference type:\s*(\S+)", text, str)
    d['best_epoch'] = _f(r"Best epoch:\s*(\d+)", text, int)
    d['epochs'] = _f(r"Epochs \(max\):\s*(\d+)", text, int)
    d['n_images'] = _f(r"(?:Test images|images):\s*(\d+)", text, int)
    d['headline_mae'] = _f(r"\n\s*Test MAE:\s*([\d.]+)", text)
    d['headline_rmse'] = _f(r"\n\s*Test MSE\*:\s*([\d.]+)", text)
    d['headline_nae'] = _f(r"\n\s*Test NAE:\s*([\d.]+)", text)
    d['headline_sre'] = _f(r"\n\s*Test SRE:\s*([\d.]+)", text)  # MCAC files only
    # headline paradigm: taken from the file when stated, else density
    d['primary'] = 'box' if re.search(r"Primary metric:\s*box-count", text) else None

    # count blocks, either layout
    box, dens = None, None
    prim = _slice(text, r"Count metrics \(", _STOPS)
    prim_row = _parse_full_row(prim) if prim else None
    if prim is not None and re.search(r"Count metrics \(box-count", text):
        box = prim_row
        if d['primary'] is None:
            d['primary'] = 'box'
    elif prim is not None:
        # primary "Count metrics (density-integral...)"
        dens = prim_row
        if d['primary'] is None:
            d['primary'] = 'density'
    sec_box = _slice(text, r"Box-count metrics", _STOPS)
    if sec_box:
        box = box or _parse_full_row(sec_box)
    sec_dens = _slice(text, r"Density-integral count metrics", _STOPS)
    if sec_dens:
        dens = dens or _parse_full_row(sec_dens)
    d['box'] = box
    d['density'] = dens

    # bbox-quality scalars
    d['iou_tp'] = _f(r"Mean IoU \(TP\) @0\.5:\s*([\d.]+)", text)
    d['giou_tp'] = _f(r"Mean GIoU \(TP\) @0\.5:\s*([\d.]+)", text)
    d['precision'] = _f(r"Precision @0\.5:\s*([\d.]+)", text)
    d['recall'] = _f(r"Recall\s+@0\.5:\s*([\d.]+)", text)
    d['f1'] = _f(r"F1\s+@0\.5:\s*([\d.]+)", text)
    d['ap50'] = _f(r"AP @0\.5:\s*([\d.]+)", text)
    d['ap5095'] = _f(r"AP @\[\.5:\.95\]:\s*([\d.]+)", text)

    # per-image table
    rows = []
    for ln in text.splitlines():
        # optional groups: Time(s), then the MCAC DPred/DAE pair
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(nan|[\d.]+)\s+(nan|[\d.]+)"
                     r"(?:\s+(nan|[\d.]+))?(?:\s+(nan|[\d.]+)\s+(nan|[\d.]+))?\s*$", ln)
        if not m:
            continue
        g = m.groups()
        iou = float('nan') if g[8] == 'nan' else float(g[8])
        giou = float('nan') if g[9] == 'nan' else float(g[9])
        time_s = None if g[10] is None else (float('nan') if g[10] == 'nan' else float(g[10]))
        density_count = None if g[11] is None else (float('nan') if g[11] == 'nan' else float(g[11]))
        density_ae = None if g[12] is None else (float('nan') if g[12] == 'nan' else float(g[12]))
        rows.append({'idx': int(g[0]), 'image_id': g[1], 'gt': int(g[2]),
                     'pred': int(g[3]), 'ae': float(g[4]), 'tp': int(g[5]),
                     'fp': int(g[6]), 'fn': int(g[7]), 'iou': iou, 'giou': giou,
                     'time_s': time_s, 'density_count': density_count,
                     'density_ae': density_ae})
    d['per_image'] = rows

    # some MCAC files print SRE only in the Summary, recompute from the rows
    if rows:
        if d.get('box') and d['box'].get('sre') is None:
            d['box']['sre'] = _sre_from_rows(rows, 'gt', 'pred')
        if d.get('density') and d['density'].get('sre') is None:
            d['density']['sre'] = _sre_from_rows(rows, 'gt', 'density_count')
    return d


def _run_recency(entry):
    """Sort key for duplicate copies of one results file, newest run last:
    subfolder beats flat, then SLURM job id, then mtime."""
    depth, path = entry
    parent = os.path.basename(os.path.dirname(path))
    m = re.match(r'(\d+)', parent)
    job_id = int(m.group(1)) if m else -1
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (depth, job_id, mtime, path)


def find_results(paths, pattern='*_results.txt'):
    """Expand files/dirs into a sorted list of results-txt paths, skipping
    _zerodepth artifacts. Duplicate names keep only the newest copy."""
    cands = []  # (depth, path), depth 1 = per-run subfolder
    for p in paths:
        if os.path.isdir(p):
            cands += [(0, f) for f in glob.glob(os.path.join(p, pattern))]
            cands += [(1, f) for f in glob.glob(os.path.join(p, '*', pattern))]
        elif os.path.isfile(p):
            cands.append((0, p))
    # a path can be collected at both depths, subfolder wins
    depth_of = {}
    for depth, f in cands:
        if '_zerodepth' in os.path.basename(f):
            continue
        depth_of[f] = max(depth, depth_of.get(f, 0))
    by_name = {}
    for f in sorted(depth_of):
        by_name.setdefault(os.path.basename(f), []).append((depth_of[f], f))
    out = []
    for name, group in sorted(by_name.items()):
        group.sort(key=_run_recency)
        if len(group) > 1:
            print('[find_results] %d copies of %s -- keeping %s (shadowed: %s)'
                  % (len(group), name, group[-1][1],
                     ', '.join(f for _, f in group[:-1])), file=sys.stderr)
        out.append(group[-1][1])
    return sorted(out)


def model_key(rec):
    """Short label: tag + split + infer, or the filename stem."""
    if rec.get('exp_tag'):
        bits = [rec['exp_tag']]
        if rec.get('split'):
            bits.append(rec['split'])
        if rec.get('infer'):
            bits.append(rec['infer'])
        return '_'.join(bits)
    return rec['file'].replace('_results.txt', '')


def is_geco2(rec):
    """True for the finetuned RGB baseline (geco2/geco2_finetuned tags,
    not geco2_pretrained)."""
    if is_geco2_pretrained(rec):
        return False
    tag = (rec.get('exp_tag') or rec.get('file') or '')
    return tag.startswith('geco2') or '_geco2_' in tag or '_geco2' in tag


def is_geco2_pretrained(rec):
    """True for the released-weights (not finetuned) geco2 reference rows."""
    tag = (rec.get('exp_tag') or rec.get('file') or '')
    return 'geco2_pretrained' in tag
