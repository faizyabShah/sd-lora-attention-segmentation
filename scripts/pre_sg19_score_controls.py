#!/usr/bin/env python3
"""CPU-only CA/middle/late score controls on saved SG18 image maps."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finalize_pre_sg19_late_support import auc_ap
from src.segmentation import locked_refinements as lr
from src.segmentation.pre_sg19_late_support import comparison_masks

SOURCE = ROOT / 'outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed/cache'
OUT = ROOT / 'outputs/segmentation/analysis/pre_sg19_score_controls'
NAMES = ('ca', 'middle', 'late', 'min_middle_late', 'mean_middle_late')
CONTRASTS = {'tp_vs_fp': 1, 'tp_vs_fp_gt16': 2,
             'tp_vs_fp_gt64': 3, 'tp_vs_fp_separate': 4}


def image_scores(path: Path, threshold: float):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        maps = data['maps'].copy()
        seed = data['seed'].copy()
        truth = data['truth'].copy()
    if maps.shape != (5, 4096) or seed.shape != (4096,) or truth.shape != (512, 512):
        raise ValueError(f'invalid SG18 map cache: {path}')
    if meta['dataset'] != path.parent.name or meta['spec']['names'] != ['W/REF','W/T1','W/T2','W/T3','W/P1']:
        raise ValueError(f'foreign SG18 map cache: {path}')
    middle = maps[1]
    # W/T2 is the average of 11 middle and 10 late calls, with eight heads per call.
    late = ((21.0 * maps[2].astype(np.float64) -
             11.0 * middle.astype(np.float64)) / 10.0).astype(np.float32)
    if not np.isfinite(late).all():
        raise ValueError(f'nonfinite late score: {path}')
    raw = torch.from_numpy(np.stack((seed, middle, late)).reshape(3, 1, 64, 64))
    raw_full = F.interpolate(raw, size=truth.shape, mode='bilinear', align_corners=False)[:, 0].numpy()
    c, m, l = raw_full
    normalized = lr.full_scores(torch.from_numpy(np.stack((seed, middle))), truth.shape).numpy()
    eligible, labels = comparison_masks(truth, normalized[1] >= threshold,
                                         normalized[0] >= 0.45, m > c)
    labels = np.stack(labels, axis=1).astype(np.uint8)
    scores = np.stack((c[eligible], m[eligible], l[eligible],
                       np.minimum(m[eligible], l[eligible]),
                       (m[eligible] + l[eligible]) / 2), axis=1).astype(np.float32)
    if len(scores) != len(labels) or np.any(labels[:, 0] + labels[:, 1] != 1):
        raise ValueError(f'invalid candidate labels: {path}')
    return meta['image_id'], scores, labels


def evaluate(scores: np.ndarray, labels: np.ndarray):
    result = {}
    for index, name in enumerate(NAMES):
        result[name] = {}
        for contrast, col in CONTRASTS.items():
            eligible = (labels[:, 0] > 0) | (labels[:, col] > 0)
            y = labels[eligible, 0].astype(bool)
            roc, ap = auc_ap(scores[eligible, index], y)
            result[name][contrast] = {'roc_auc': roc, 'pr_auc': ap,
                                       'tp_pixels': int(y.sum()),
                                       'fp_pixels': int((~y).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--threshold', type=float, default=0.425)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prior = json.loads((ROOT/'outputs/segmentation/analysis/pre_sg19_late_support/results.json').read_text())
    if args.threshold != prior['spec']['primary_middle_threshold']:
        raise ValueError('the control must use the Pre-SG19 primary threshold')
    datasets = {}
    for dataset, expected in (('voc', 490), ('coco', 707)):
        files = sorted((args.source/dataset).glob('*.npz'))
        if len(files) != expected:
            raise ValueError(f'{dataset}: expected {expected} SG18 cache files, found {len(files)}')
        if args.smoke:
            files = files[:2]
        all_scores, all_labels, ids = [], [], set()
        for index, path in enumerate(files, 1):
            image_id, scores, labels = image_scores(path, args.threshold)
            if image_id in ids:
                raise ValueError(f'duplicate image ID: {image_id}')
            ids.add(image_id)
            all_scores.append(scores)
            all_labels.append(labels)
            if index % 100 == 0 or index == len(files):
                print(f'{dataset}: {index}/{len(files)} cached maps processed', flush=True)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        datasets[dataset] = {'images': len(files), 'candidate_pixels': len(labels),
                             'new_tp': int(labels[:, 0].sum()),
                             'new_fp': int(labels[:, 1].sum()),
                             'control_metrics': evaluate(scores, labels),
                             'pre_sg19_replay_candidate_pixels': prior['results'][dataset][str(args.threshold)]['pixels'],
                             'pre_sg19_replay_new_tp': prior['results'][dataset][str(args.threshold)]['new_tp'],
                             'pre_sg19_replay_new_fp': prior['results'][dataset][str(args.threshold)]['new_fp'],
                             'pre_sg19_replay_late_auc': prior['results'][dataset][str(args.threshold)]['feature_auc']['late_score']['tp_vs_fp']['roc_auc']}
        del scores, labels, all_scores, all_labels
    result = {'source': 'existing SG18 cached 64x64 maps, not Pre-SG19 GPU replay',
              'candidate_definition': 'M>c; minmax-then-bilinear CA <0.45, middle >=0.425; annotation resolution',
              'same_candidate_pixels_for_all_five_scores_within_each_SG18_cache_image': True,
              'scores_are_raw_bilinearly_resized_values': True,
              'no_generation_or_new_method': True, 'datasets': datasets}
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = 'smoke_' if args.smoke else ''
    (args.output/f'{prefix}results.json').write_text(json.dumps(result, indent=2) + '\n')
    if args.smoke:
        print('CPU score-control smoke passed', flush=True)
        return
    with (args.output/'auc_table.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('dataset','score','contrast','roc_auc','pr_auc','tp_pixels','fp_pixels'))
        for dataset, d in datasets.items():
            for score, contrasts in d['control_metrics'].items():
                for contrast, metrics in contrasts.items():
                    writer.writerow((dataset, score, contrast, metrics['roc_auc'], metrics['pr_auc'],
                                     metrics['tp_pixels'], metrics['fp_pixels']))
    lines = ['# Pre-SG19 score controls', '',
             'CPU-only analysis of saved SG18 maps. All five scores use the same candidate pixels within each image. '
             'The earlier Pre-SG19 replayed maps differ slightly from SG18 caches, so its reported candidate counts and late-score AUC may differ.', '',
             '| Score | VOC TP/FP ROC | COCO TP/FP ROC | COCO TP vs >16 FP | COCO TP vs separate FP |',
             '|---|---:|---:|---:|---:|']
    for name in NAMES:
        v = datasets['voc']['control_metrics'][name]
        c = datasets['coco']['control_metrics'][name]
        lines.append(f"| {name} | {v['tp_vs_fp']['roc_auc']:.4f} | {c['tp_vs_fp']['roc_auc']:.4f} | "
                     f"{c['tp_vs_fp_gt16']['roc_auc']:.4f} | {c['tp_vs_fp_separate']['roc_auc']:.4f} |")
    lines += ['', 'Exact ROC-AUC, PR-AUC, all four contrasts and support counts are in `auc_table.csv` and `results.json`.', '']
    (args.output/'report.md').write_text('\n'.join(lines))
    (args.output/'completed.json').write_text(json.dumps({'images': {'voc': 490, 'coco': 707},
                                                           'all_required_stages_complete': True}, indent=2) + '\n')
    print('Pre-SG19 CPU score controls completed', flush=True)


if __name__ == '__main__':
    main()
