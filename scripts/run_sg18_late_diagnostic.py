#!/usr/bin/env python3
"""Diagnose late 21–30 calls from exact SG18 cached maps; no selection."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sg18_temporal_windowed as sg18
from scripts.select_propagation_cv import pooled_miou
from src.segmentation import sg18_temporal as sg
from src.segmentation import locked_refinements as lr
from src.segmentation import controlled_cv as cv

SPEC = ROOT / 'configs/sg18_late_diagnostic.json'
SOURCE = ROOT / 'outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed'
OUT = ROOT / 'outputs/segmentation/analysis/sg18_late_diagnostic'
NAMES = ('middle', 'late', 'middle_plus_late')


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def score_one(cache: Path, row: dict, dataset: str, source_spec: dict, thresholds: tuple[float, ...]) -> dict:
    with np.load(cache, allow_pickle=False) as data:
        sg18.verify_cache(data, row, dataset, source_spec)
        t1 = np.asarray(data['maps'][sg.NAMES.index('W/T1')], np.float32)
        t2 = np.asarray(data['maps'][sg.NAMES.index('W/T2')], np.float32)
        seed = np.asarray(data['seed'], np.float32)
        truth = np.asarray(data['truth'], bool)
    if len(t1) != 4096 or len(t2) != 4096:
        raise ValueError('invalid cached 64x64 map')
    # T2 is the equal mean of 11 middle and 10 late calls, each with 8 heads.
    late = (21.0 * t2.astype(np.float64) - 11.0 * t1.astype(np.float64)) / 10.0
    maps = {'middle': t1, 'late': late.astype(np.float32), 'middle_plus_late': t2}
    full = lr.full_scores(torch.as_tensor(np.stack([seed, *maps.values()])).reshape(4, 64, 64), truth.shape).cpu().numpy()
    ca_score = full[0]
    scores = dict(zip(NAMES, full[1:]))
    distance = ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape, np.inf)
    output = {'image_id': row['image_id'], 'dataset': dataset, 'truth_pixels': int(truth.sum()), 'conditions': {}}
    for threshold in thresholds:
        ca = ca_score >= 0.45
        masks = {name: value >= threshold for name, value in scores.items()}
        per_method = {}
        for name, mask in masks.items():
            tp = int((mask & truth).sum()); fp = int((mask & ~truth).sum())
            fn = int((~mask & truth).sum()); tn = int((~mask & ~truth).sum())
            event, _ = sg.events(ca, mask, truth, distance)
            per_method[name] = {'counts': [tp, fp, fn, tn],
                                'ca_relative_events': event.tolist()}
        middle, together = masks['middle'], masks['middle_plus_late']
        gained = together & ~middle
        lost = middle & ~together
        change = {'tp_gained': int((gained & truth).sum()),
                  'tp_lost': int((lost & truth).sum()),
                  'fp_gained': int((gained & ~truth).sum()),
                  'fp_removed': int((lost & ~truth).sum()),
                  'distant_fp_gained_gt16': int((gained & ~truth & (distance > 16)).sum()),
                  'distant_fp_removed_gt16': int((lost & ~truth & (distance > 16)).sum()),
                  'foreground_gained': int(gained.sum()), 'foreground_lost': int(lost.sum())}
        output['conditions'][str(threshold)] = {'methods': per_method, 'middle_to_middle_plus_late': change}
    return output


def summarize(rows: list[tuple[str, dict, dict]], thresholds: tuple[float, ...]) -> dict:
    output = {}
    for dataset in ('voc', 'coco'):
        records = [(cls, rec) for ds, cls, rec in rows if ds == dataset]
        output[dataset] = {'images': len(records), 'conditions': {}}
        for threshold in thresholds:
            key = str(threshold)
            methods = {}
            classes = np.array([cls for cls, _ in records])
            for name in NAMES:
                counts = np.array([rec['conditions'][key]['methods'][name]['counts'] for _, rec in records], np.int64)
                events = np.array([rec['conditions'][key]['methods'][name]['ca_relative_events'] for _, rec in records], np.int64).sum(0)
                tp, fp, fn, _ = counts.sum(0)
                per_class = {}
                for cls in sorted(set(classes)):
                    a, b, c, _ = counts[classes == cls].sum(0)
                    per_class[cls] = {'iou': float(a/max(a+b+c, 1)),
                                      'precision': float(a/max(a+b, 1)),
                                      'recall': float(a/max(a+c, 1))}
                methods[name] = {'miou': pooled_miou(counts, classes),
                                 'precision': float(tp/max(tp+fp, 1)),
                                 'recall': float(tp/max(tp+fn, 1)),
                                 'pixel_iou': float(tp/max(tp+fp+fn, 1)),
                                 'ca_relative_events': sg.rates(events),
                                 'per_class': per_class}
            changes = {}
            for _, rec in records:
                for key2, value in rec['conditions'][key]['middle_to_middle_plus_late'].items():
                    changes[key2] = changes.get(key2, 0) + value
            changes['net_true_positives'] = changes['tp_gained'] - changes['tp_lost']
            changes['net_false_positives'] = changes['fp_gained'] - changes['fp_removed']
            output[dataset]['conditions'][key] = {'methods': methods, 'middle_to_middle_plus_late': changes}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    source_spec = json.loads((ROOT / 'configs/sg18_temporal_windowed_propagation.json').read_text())
    if spec['windows'] != {'middle': [10, 20], 'late': [21, 30], 'middle_plus_late': [10, 30]}:
        raise ValueError('late diagnostic window contract changed')
    if not args.smoke:
        completed = json.loads((args.source_dir / 'completed.json').read_text())
        if not completed['all_required_stages_complete'] or completed['images'] != {'voc': 490, 'coco': 707}:
            raise ValueError('SG18 source coverage incomplete')
    thresholds = tuple(float(t) for t in spec['threshold_conditions'])
    rows = []
    for dataset in ('voc', 'coco'):
        source_rows = sg18.rows_for(dataset, first_only=args.smoke)
        for cls, row in source_rows:
            cache, _ = sg18.paths(args.source_dir, dataset, row['image_id'])
            path = args.output_dir / 'per_image' / dataset / (sg18.key(row['image_id'])+'.json')
            if path.exists():
                record = json.loads(path.read_text())
                if record['image_id'] != row['image_id'] or record['dataset'] != dataset:
                    raise ValueError('foreign diagnostic checkpoint')
            else:
                record = score_one(cache, row, dataset, source_spec, thresholds)
                atomic_json(path, record)
            rows.append((dataset, cls, record))
        print(f'{dataset}: {len(source_rows)} cached images diagnosed', flush=True)
    summary = {'spec': spec, 'source_sg18_job': 235290, 'source_is_cached_sg18_maps': True,
               'no_selection_or_promotion': True, 'results': summarize(rows, thresholds)}
    atomic_json(args.output_dir / ('smoke_results.json' if args.smoke else 'results.json'), summary)
    if args.smoke:
        print('SG18 late-call diagnostic smoke passed', flush=True)
        return
    with (args.output_dir / 'per_image_changes.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['dataset', 'class', 'image_id', 'threshold', 'tp_gained', 'tp_lost', 'fp_gained', 'fp_removed',
                         'distant_fp_gained_gt16', 'distant_fp_removed_gt16'])
        for dataset, cls, rec in rows:
            for threshold in thresholds:
                x = rec['conditions'][str(threshold)]['middle_to_middle_plus_late']
                writer.writerow([dataset, cls, rec['image_id'], threshold,
                                 *(x[k] for k in ['tp_gained', 'tp_lost', 'fp_gained', 'fp_removed',
                                                  'distant_fp_gained_gt16', 'distant_fp_removed_gt16'])])
    lines = ['# SG18 middle versus late call diagnostic', '',
             'Late 21–30 is a diagnostic-only map, not a selectable method. All values use the original SG18 cached maps.','']
    for dataset in ('voc', 'coco'):
        for threshold in thresholds:
            result = summary['results'][dataset]['conditions'][str(threshold)]
            lines.append(f'## {dataset.upper()} at fixed threshold {threshold}')
            lines.append('')
            lines.append('| Window | mIoU | Precision | Recall | CA-relative TP add | FP add | distant FP >16 | separate FP |')
            lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
            for name in NAMES:
                v = result['methods'][name]; e = v['ca_relative_events']
                lines.append(f"| {name} | {v['miou']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | "
                             f"{e['tp_add']:,} | {e['fp_add']:,} | {e['fp_gt16']:,} | {e['fp_separate']:,} |")
            c = result['middle_to_middle_plus_late']
            lines += ['', f"Middle → middle+late threshold crossings: TP gained {c['tp_gained']:,}, "
                      f"TP lost {c['tp_lost']:,}, FP gained {c['fp_gained']:,}, "
                      f"FP removed {c['fp_removed']:,}. Distant FP gained/removed (>16 px): "
                      f"{c['distant_fp_gained_gt16']:,}/{c['distant_fp_removed_gt16']:,}.", '']
    (args.output_dir / 'report.md').write_text('\n'.join(lines))
    atomic_json(args.output_dir / 'completed.json', {'images': {'voc': 490, 'coco': 707},
                'all_required_stages_complete': True})
    print('SG18 late-call diagnostic completed', flush=True)


if __name__ == '__main__':
    main()
