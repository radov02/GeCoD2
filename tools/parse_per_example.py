#!/usr/bin/env python3
"""Dump every per-image row of the *_results.txt files into one CSV.
'pred' follows the file's 'Primary metric' line.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.results_lib import parse_results_file, find_results, model_key

FIELDS = ['model', 'split', 'infer', 'primary', 'image_id', 'gt', 'pred', 'ae',
          'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'iou_tp', 'giou_tp',
          'time_s']


def main():
    ap = argparse.ArgumentParser(description="per-example metrics CSV")
    ap.add_argument('paths', nargs='*', help="results.txt files and/or dirs")
    ap.add_argument('-o', '--out', default=None, help="output CSV path")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(here, '..', 'expresults', 'results'))
    paths = args.paths or [default_dir]
    out = args.out or os.path.join(here, 'per_example.csv')

    files = find_results(paths)
    if not files:
        print(f"[parse_per_example] no *_results.txt found under {paths}")
        return
    rows = []
    for fp in files:
        rec = parse_results_file(fp)
        key = model_key(rec)
        for r in rec['per_image']:
            tp, fp_, fn = r['tp'], r['fp'], r['fn']
            prec = tp / (tp + fp_) if (tp + fp_) else float('nan')
            rec_ = tp / (tp + fn) if (tp + fn) else float('nan')
            f1 = (2 * prec * rec_ / (prec + rec_)) if (prec + rec_) else float('nan')
            rows.append({
                'model': key, 'split': rec.get('split'), 'infer': rec.get('infer'),
                'primary': rec.get('primary'), 'image_id': r['image_id'],
                'gt': r['gt'], 'pred': r['pred'], 'ae': r['ae'],
                'tp': tp, 'fp': fp_, 'fn': fn,
                'precision': round(prec, 4), 'recall': round(rec_, 4), 'f1': round(f1, 4),
                'iou_tp': r['iou'], 'giou_tp': r['giou'],
                'time_s': r.get('time_s'),
            })
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[parse_per_example] {len(rows)} rows from {len(files)} file(s) -> {out}")


if __name__ == '__main__':
    main()
