"""count metrics (MAE/RMSE/NAE/SRE) with optional GT-count stratification.
MSE* = the crowd-counting RMSE convention."""


def count_metrics(gt_pred_pairs):
    """None if empty. SRE = MCAC Eq. 6, same max(gt, 1) clamp as NAE."""
    pairs = list(gt_pred_pairs)
    n = len(pairs)
    if n == 0:
        return None
    ae = sum(abs(g - p) for g, p in pairs)
    se = sum((g - p) ** 2 for g, p in pairs)
    nae = sum(abs(g - p) / max(g, 1) for g, p in pairs)
    sre = sum((g - p) ** 2 / max(g, 1) for g, p in pairs)
    return {'n': n, 'mae': ae / n, 'rmse': (se / n) ** 0.5, 'nae': nae / n,
            'sre': (sre / n) ** 0.5}


def stratified_rows(records, count_key, thresholds=(), gt_key='gt_count'):
    """full set first, then one row per threshold N over the subset gt < N."""
    rows = [('full', count_metrics((r[gt_key], r[count_key]) for r in records))]
    for n in sorted({int(t) for t in thresholds}):
        sub = [r for r in records if r[gt_key] < n]
        rows.append((f'GT<{n}', count_metrics((r[gt_key], r[count_key]) for r in sub)))
    return rows


def format_rows(rows, indent='  ', with_sre=False):
    """stratified_rows -> aligned text lines, with_sre adds the SRE column."""
    out = []
    for label, m in rows:
        if m is None:
            out.append(f"{indent}{label:<10s} n=    0  (no images in subset)")
        else:
            line = (f"{indent}{label:<10s} n={m['n']:>5d}  "
                    f"MAE={m['mae']:>8.3f}  MSE*={m['rmse']:>9.3f}  NAE={m['nae']:>6.3f}")
            if with_sre:
                line += f"  SRE={m['sre']:>7.3f}"
            out.append(line)
    return out
