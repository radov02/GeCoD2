#!/usr/bin/env python3
"""All depth runs into one CSV: *_results.txt metrics, FSCD coco AP,
*_metrics.txt convergence, deltas vs the finetuned geco2 baseline.
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_lib import parse_results_file, find_results, is_geco2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DIRS = [
    os.path.join(ROOT, 'expresults', 'results'),
    os.path.join(ROOT, 'expresults_fscd', 'results_FSCD147'),
]

# lever tags as columns, _hiresn replaces _hires in the name (hires=1 set below)
LEVERS = ['pdm', 'pdmgen', 'dconv', 'dnorm', 'idinit', 'hires', 'hiresn', 'k1']


def experiment_of(tag):
    """Model tag to fusion-mode name. A bare 'geco2' counts as finetuned."""
    for e in ('conv_depth_add', 'conv_hiera', 'depth_dim', 'sep_hiera', 'ffm',
              'geco2_pretrained', 'geco2_finetuned', 'geco2'):
        if tag.startswith(e) or ('_' + e + '_') in ('_' + tag + '_'):
            return e
    return tag.split('_')[0]


def dataset_of(path):
    return 'FSCD147' if 'FSCD147' in path or 'fscd' in path.lower() else 'IOCfish'


def metrics_path_for(results_path):
    """Map a *_results.txt path to its sibling *_metrics.txt."""
    base = os.path.basename(results_path)
    stem = re.sub(r'_(TEST|VAL)_.*results\.txt$', '', base)
    return os.path.join(os.path.dirname(results_path), stem + '_metrics.txt')


def parse_metrics(path):
    """Pull convergence + depth-fusion health from a *_metrics.txt."""
    out = {}
    if not os.path.isfile(path):
        return out
    text = open(path).read()
    m = re.search(r'Best model .* epoch (\d+):', text)
    if m:
        out['best_epoch'] = int(m.group(1))
    for key, pat in [('trainval_mae', r'Val MAE:\s*([\d.]+)'),
                     ('trainval_rmse', r'Val RMSE:\s*([\d.]+)'),
                     ('train_mae_best', r'Train MAE:\s*([\d.]+)'),
                     ('best_epoch_time_s', r'Epoch time:\s*([\d.]+)s')]:
        mm = re.search(pat, text[text.find('Best model'):]) if 'Best model' in text else None
        if mm:
            out[key] = float(mm.group(1))
    # stop epoch = last numbered data row
    last = None
    for ln in text.splitlines():
        mm = re.match(r'\s*(\d+)\s+[\d.]+\s+[\d.]+\s', ln)
        if mm:
            last = int(mm.group(1))
    if last is not None:
        out['stop_epoch'] = last
    rank = re.findall(r'conv-depth-adapter @epoch \d+: eff_rank=([\d.]+)/(\d+)\s+\|act\|=([\d.eE+-]+)\s+std=([\d.eE+-]+)\s+dead=([\d.]+)', text)
    if rank:
        r0, r_last = rank[0], rank[-1]
        out['eff_rank_first'] = float(r0[0])
        out['eff_rank_last'] = float(r_last[0])
        out['adapter_channels'] = int(r_last[1])
        out['dead_frac_last'] = float(r_last[4])
    dev = re.findall(r'depth-fusion deviation @epoch \d+: total_L2=([\d.eE+-]+)', text)
    if dev:
        out['fusion_dev_first'] = float(dev[0])
        out['fusion_dev_last'] = float(dev[-1])
    return out


def parse_cocoeval(results_path):
    """Pull official COCO AP from the sibling *_cocoeval.txt (FSCD only)."""
    coco = results_path.replace('_results.txt', '_cocoeval.txt')
    if not os.path.isfile(coco):
        return {}
    text = open(coco).read()
    m = re.search(r"'AP':\s*([\d.]+),\s*'AP50':\s*([\d.]+),\s*'AP75':\s*([\d.]+)", text)
    if not m:
        return {'coco_eval_status': 'CRASHED' if 'Traceback' in text else 'no-AP'}
    return {'coco_ap': round(float(m.group(1)), 2), 'coco_ap50': round(float(m.group(2)), 2),
            'coco_ap75': round(float(m.group(3)), 2), 'coco_eval_status': 'ok'}


def build_rows(dirs):
    rows = []
    for path in find_results(dirs):
        rec = parse_results_file(path)
        tag = rec.get('exp_tag') or os.path.basename(path)
        row = {
            'dataset': dataset_of(path),
            'experiment': experiment_of(tag),
            'tag': tag,
            'split': rec.get('split'),
            'infer': rec.get('infer'),
            'best_epoch': rec.get('best_epoch'),
            'n_images': rec.get('n_images'),
            'box_mae': (rec['box'] or {}).get('mae'),
            'box_rmse': (rec['box'] or {}).get('rmse'),
            'box_nae': (rec['box'] or {}).get('nae'),
            'box_sre': (rec['box'] or {}).get('sre'),
            'dens_mae': (rec['density'] or {}).get('mae'),
            'dens_rmse': (rec['density'] or {}).get('rmse'),
            'dens_nae': (rec['density'] or {}).get('nae'),
            'dens_sre': (rec['density'] or {}).get('sre'),
            'precision': rec.get('precision'),
            'recall': rec.get('recall'),
            'f1': rec.get('f1'),
            'ap50_greedy': rec.get('ap50'),
            'ap5095_greedy': rec.get('ap5095'),
        }
        for lev in LEVERS:
            row[lev] = int(('_' + lev + '_') in ('_' + tag + '_') or tag.endswith('_' + lev))
        if row['hiresn']:
            row['hires'] = 1  # hiresn implies hires
        times = [r['time_s'] for r in rec['per_image'] if r.get('time_s') is not None]
        row['infer_ms'] = round(1000 * sum(times) / len(times), 1) if times else None
        row.update(parse_metrics(metrics_path_for(path)))
        row.update(parse_cocoeval(path))
        rows.append(row)
    # deltas vs the finetuned baseline per (dataset, split, infer)
    base = {}
    for r in rows:
        if is_geco2({'exp_tag': r['tag']}):
            base[(r['dataset'], r['split'], r.get('infer'))] = r
    for r in rows:
        if is_geco2({'exp_tag': r['tag']}):
            continue
        b = base.get((r['dataset'], r['split'], r.get('infer')))
        if b is None:
            sys.stderr.write(
                "[parse_depth_runs][WARN] no same-infer geco2 baseline for "
                f"(dataset={r['dataset']}, split={r['split']}, infer={r.get('infer')}); "
                f"leaving depth-delta columns empty for tag '{r['tag']}'.\n")
            continue
        for k in ('box_mae', 'box_rmse', 'box_nae', 'dens_mae', 'dens_rmse', 'dens_nae'):
            if r.get(k) is not None and b.get(k) is not None:
                r['d_' + k] = round(b[k] - r[k], 3)  # positive = depth helps
    return rows


COLUMNS = ['dataset', 'experiment', 'tag', 'split', 'best_epoch', 'stop_epoch',
           'n_images', 'box_mae', 'box_rmse', 'box_nae', 'box_sre',
           'dens_mae', 'dens_rmse', 'dens_nae', 'dens_sre',
           'd_box_mae', 'd_box_rmse', 'd_dens_mae', 'd_dens_rmse',
           'precision', 'recall', 'f1', 'ap50_greedy', 'ap5095_greedy',
           'coco_ap', 'coco_ap50', 'coco_ap75', 'coco_eval_status',
           'trainval_mae', 'train_mae_best', 'best_epoch_time_s', 'infer_ms',
           'eff_rank_first', 'eff_rank_last', 'adapter_channels', 'dead_frac_last',
           'fusion_dev_first', 'fusion_dev_last'] + LEVERS + ['infer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='*', default=DEFAULT_DIRS)
    ap.add_argument('-o', '--out', default=os.path.join(HERE, 'depth_runs_summary.csv'))
    args = ap.parse_args()
    rows = build_rows(args.dirs or DEFAULT_DIRS)
    rows.sort(key=lambda r: (r['dataset'], r['experiment'], r['tag'], str(r['split'])))
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote', args.out, '(%d rows)' % len(rows))


if __name__ == '__main__':
    main()
