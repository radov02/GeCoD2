#!/usr/bin/env python3
"""geco2 baseline vs depth variants, one CSV. d_* = geco2 - depth
(positive = depth helps), grouped by (split, infer).
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.results_lib import parse_results_file, find_results, model_key, is_geco2, is_geco2_pretrained

FIELDS = ['group', 'model', 'is_baseline',
          'box_MAE', 'box_RMSE', 'box_NAE', 'box_SRE',
          'd_box_MAE', 'd_box_RMSE', 'd_box_NAE',
          'dens_MAE', 'dens_RMSE', 'dens_NAE', 'dens_SRE',
          'AP50', 'AP_5095', 'F1', 'precision', 'recall', 'best_epoch']


def _get(block, k):
    return block[k] if block else None


def _delta(base, val):
    """baseline error - variant error, positive = depth helps."""
    if base is None or val is None:
        return None
    return round(base - val, 3)


def main():
    ap = argparse.ArgumentParser(description="geco2 vs depth comparison CSV")
    ap.add_argument('paths', nargs='*')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(here, '..', 'expresults', 'results'))
    paths = args.paths or [default_dir]
    out = args.out or os.path.join(here, 'geco2_vs_depth.csv')

    files = find_results(paths)
    if not files:
        print(f"[parse_geco2_vs_depth] no *_results.txt found under {paths}")
        return
    recs = [parse_results_file(fp) for fp in files]

    # group by (split, infer)
    groups = {}
    for r in recs:
        groups.setdefault((r.get('split'), r.get('infer')), []).append(r)

    rows = []
    for (split, infer), grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        base = next((r for r in grp if is_geco2(r)), None)
        gname = f"{split or '?'}/{infer or '?'}"
        b_box = base['box'] if base else None
        for r in sorted(grp, key=lambda x: (not is_geco2(x), model_key(x))):
            box, dens = r.get('box'), r.get('density')
            rows.append({
                'group': gname, 'model': model_key(r),
                'is_baseline': 'yes' if is_geco2(r) else ('pretrained-ref' if is_geco2_pretrained(r) else ''),
                'box_MAE': _get(box, 'mae'), 'box_RMSE': _get(box, 'rmse'), 'box_NAE': _get(box, 'nae'),
                'box_SRE': _get(box, 'sre'),
                'd_box_MAE': None if (is_geco2(r) or not b_box) else _delta(_get(b_box, 'mae'), _get(box, 'mae')),
                'd_box_RMSE': None if (is_geco2(r) or not b_box) else _delta(_get(b_box, 'rmse'), _get(box, 'rmse')),
                'd_box_NAE': None if (is_geco2(r) or not b_box) else _delta(_get(b_box, 'nae'), _get(box, 'nae')),
                'dens_MAE': _get(dens, 'mae'), 'dens_RMSE': _get(dens, 'rmse'), 'dens_NAE': _get(dens, 'nae'),
                'dens_SRE': _get(dens, 'sre'),
                'AP50': r.get('ap50'), 'AP_5095': r.get('ap5095'), 'F1': r.get('f1'),
                'precision': r.get('precision'), 'recall': r.get('recall'),
                'best_epoch': r.get('best_epoch'),
            })
        if base is None:
            print(f"[warn] group {gname}: no geco2 baseline found -- depth deltas left blank")
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[parse_geco2_vs_depth] {len(rows)} rows from {len(files)} file(s) -> {out}")
    print("  d_box_* = geco2_minus_depth on the PRIMARY box-count (positive = depth helps).")


if __name__ == '__main__':
    main()
