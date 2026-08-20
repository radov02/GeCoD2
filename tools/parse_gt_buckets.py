#!/usr/bin/env python3
"""Bucket the per-image tables by GT count, one CSV row per (model, bucket).
Default buckets 0-10, 10-50, 50-200, 200-500, 500+.
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.results_lib import parse_results_file, find_results, model_key

FIELDS = ['model', 'split', 'infer', 'bucket', 'n',
          'MAE', 'RMSE', 'NAE', 'mean_gt', 'mean_pred',
          'mean_tp', 'mean_fp', 'mean_fn', 'precision', 'recall']


def _stats(rows):
    n = len(rows)
    if n == 0:
        return None
    ae = [r['ae'] for r in rows]
    se = [r['ae'] ** 2 for r in rows]
    nae = [abs(r['gt'] - r['pred']) / max(r['gt'], 1) for r in rows]
    tp = sum(r['tp'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    return {
        'n': n,
        'MAE': round(sum(ae) / n, 3),
        'RMSE': round(math.sqrt(sum(se) / n), 3),
        'NAE': round(sum(nae) / n, 4),
        'mean_gt': round(sum(r['gt'] for r in rows) / n, 1),
        'mean_pred': round(sum(r['pred'] for r in rows) / n, 1),
        'mean_tp': round(tp / n, 1), 'mean_fp': round(fp / n, 1), 'mean_fn': round(fn / n, 1),
        'precision': round(tp / (tp + fp), 4) if (tp + fp) else float('nan'),
        'recall': round(tp / (tp + fn), 4) if (tp + fn) else float('nan'),
    }


def main():
    ap = argparse.ArgumentParser(description="GT-count bucketed metrics CSV")
    ap.add_argument('paths', nargs='*')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--edges', type=int, nargs='+', default=[0, 10, 50, 200, 500],
                    help="bucket edges; last bucket is [last_edge, inf)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(here, '..', 'expresults', 'results'))
    paths = args.paths or [default_dir]
    out = args.out or os.path.join(here, 'gt_buckets.csv')

    edges = sorted(set(args.edges))
    buckets = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)] + [(edges[-1], float('inf'))]

    files = find_results(paths)
    if not files:
        print(f"[parse_gt_buckets] no *_results.txt found under {paths}")
        return
    rows_out = []
    for fp in files:
        rec = parse_results_file(fp)
        key = model_key(rec)
        per = rec['per_image']
        if not per:
            continue

        def emit(name, subset):
            st = _stats(subset)
            if st is None:
                return
            rows_out.append({'model': key, 'split': rec.get('split'),
                             'infer': rec.get('infer'), 'bucket': name, **st})

        emit('full', per)
        for lo, hi in buckets:
            name = f"{lo}-{'inf' if hi == float('inf') else hi}"
            emit(name, [r for r in per if lo <= r['gt'] < hi])
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    print(f"[parse_gt_buckets] {len(rows_out)} rows from {len(files)} file(s) -> {out}")


if __name__ == '__main__':
    main()
