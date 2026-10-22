#!/usr/bin/env python3
"""Five-fold VOC selection and frozen COCO transfer for locked SG18 T1+CRF."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_propagation_cv import best_threshold, class_ious, image_ious, pooled_miou, run_screen
from src.segmentation import controlled_cv as cv

SPEC = ROOT / 'configs/sg18_t1_crf.json'
OUT = ROOT / 'outputs/segmentation/controlled_attention_cv/sg18_t1_crf'


def load(out: Path, dataset: str, spec: dict, smoke: bool) -> cv.Results:
    files = sorted((out / dataset / 'shards').glob('class_*.npz'))
    result = cv.load_results(files)
    if result.configurations != tuple(spec['names']) or result.thresholds != tuple(spec['thresholds']):
        raise ValueError('result configuration or threshold mismatch')
    if result.settings['spec'] != spec or result.settings['dataset'] != dataset:
        raise ValueError('foreign result settings')
    if not smoke:
        if len(files) != 20 or len(result.image_ids) != (490 if dataset == 'voc' else 707):
            raise ValueError(f'incomplete {dataset} coverage')
        if len(result.complete) != 20 or not all(result.complete):
            raise ValueError('unfinished class shard')
    if dataset == 'voc':
        expected = cv.read_folds(ROOT / 'configs/voc_sim_folds.csv')
        if any(result.folds[i] != expected[image_id] for i, image_id in enumerate(result.image_ids)):
            raise ValueError('VOC fold mismatch')
    return result


def rates(counts: np.ndarray) -> dict:
    tp, fp, fn, _ = counts.sum(0)
    return {'precision': float(tp / max(tp + fp, 1)),
            'recall': float(tp / max(tp + fn, 1)),
            'pixel_iou': float(tp / max(tp + fp + fn, 1)),
            'mean_image_iou': float(image_ious(counts).mean())}


def forced_oof(voc: cv.Results, candidate: str, reference: str) -> dict:
    a = np.zeros((len(voc.image_ids), 4), np.int64)
    b = np.zeros_like(a)
    folds = []
    for fold in range(5):
        held = voc.folds == fold
        ta, _ = best_threshold(voc, voc.index(candidate), ~held)
        tb, _ = best_threshold(voc, voc.index(reference), ~held)
        a[held] = voc.counts[voc.index(candidate), held, ta]
        b[held] = voc.counts[voc.index(reference), held, tb]
        folds.append({'fold': fold, 'candidate_threshold': voc.thresholds[ta],
                      'reference_threshold': voc.thresholds[tb]})
    cm = pooled_miou(a, voc.classes)
    rm = pooled_miou(b, voc.classes)
    return {'candidate_miou': cm, 'reference_miou': rm, 'gain': cm-rm,
            'fold_thresholds': folds, 'paired_image_iou': cv.paired_bootstrap(image_ious(a), image_ious(b))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    voc = load(args.output_dir, 'voc', spec, args.smoke)
    coco = load(args.output_dir, 'coco', spec, args.smoke)
    if args.smoke:
        if len(voc.image_ids) != 1 or len(coco.image_ids) != 1:
            raise ValueError('smoke must contain one image per dataset')
        (args.output_dir / 'smoke_validation.json').write_text(json.dumps({
            'passed': True, 'voc_image': voc.image_ids[0], 'coco_image': coco.image_ids[0],
            'all_four_cells_scored': True}, indent=2)+'\n')
        print('SG18 T1+CRF live VOC/COCO smoke passed', flush=True)
        return
    screen = run_screen(voc, spec['reference'], [spec['candidate']], {spec['candidate']: 0})
    oof = forced_oof(voc, spec['candidate'], spec['reference'])
    full = np.ones(len(voc.image_ids), bool)
    thresholds = {n: best_threshold(voc, voc.index(n), full)[0] for n in spec['names']}
    rows = []
    cells = {}
    reference = spec['reference']
    for dataset, result in [('voc', voc), ('coco', coco)]:
        cells[dataset] = {}
        for name in spec['names']:
            ti = thresholds[name]
            counts = result.counts[result.index(name), :, ti]
            ref_counts = result.counts[result.index(reference), :, thresholds[reference]]
            per_class = class_ious(result, result.index(name), ti, full if dataset == 'voc' else np.ones(len(result.image_ids), bool))
            ref_classes = class_ious(result, result.index(reference), thresholds[reference],
                                     full if dataset == 'voc' else np.ones(len(result.image_ids), bool))
            cells[dataset][name] = {'images': len(result.image_ids), 'voc_frozen_threshold': result.thresholds[ti],
                                    'miou': pooled_miou(counts, result.classes), **rates(counts),
                                    'per_class_iou': per_class,
                                    'per_class_delta_vs_reference': {k: per_class[k]-ref_classes[k] for k in per_class},
                                    'paired_image_iou_vs_reference': cv.paired_bootstrap(image_ious(counts), image_ious(ref_counts))}
            for cls in per_class:
                rows.append({'dataset': dataset, 'method': name, 'class': cls,
                             'iou': per_class[cls], 'delta_vs_reference': per_class[cls]-ref_classes[cls]})
    fixed = {}
    for dataset, result in [('voc', voc), ('coco', coco)]:
        fixed[dataset] = {}
        for name in spec['names']:
            ti = result.thresholds.index(spec['locked_reference_threshold'])
            counts = result.counts[result.index(name), :, ti]
            fixed[dataset][name] = {'miou': pooled_miou(counts, result.classes), **rates(counts)}
    summary = {'spec': spec, 'screen': screen, 'forced_oof': oof, 'cells': cells,
               'fixed_0p45': fixed, 'thresholds_selected_on_voc_only': True,
               'no_coco_selection': True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.output_dir / 'per_class_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# SG18 T1 + unchanged DenseCRF', '',
             f"VOC fold votes: {screen['selection_counts']}; frozen: {screen['frozen']}",
             f"Forced OOF T1+CRF {oof['candidate_miou']:.4f} versus P1+CRF {oof['reference_miou']:.4f} ({oof['gain']:+.4f})", '',
             '| Method | VOC threshold | VOC mIoU | COCO frozen mIoU | VOC @0.45 | COCO @0.45 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in spec['names']:
        lines.append(f"| {name} | {cells['voc'][name]['voc_frozen_threshold']:.3f} | "
                     f"{cells['voc'][name]['miou']:.4f} | {cells['coco'][name]['miou']:.4f} | "
                     f"{fixed['voc'][name]['miou']:.4f} | {fixed['coco'][name]['miou']:.4f} |")
    lines += ['', 'Full precision, recall, per-class values and paired intervals are in results.json. '
              'The fixed-0.45 column includes the locked final-pipeline P1+CRF reference.','']
    (args.output_dir / 'report.md').write_text('\n'.join(lines))
    (args.output_dir / 'completed.json').write_text(json.dumps({
        'voc_images': len(voc.image_ids), 'coco_images': len(coco.image_ids),
        'frozen': screen['frozen'], 'all_required_stages_complete': True}, indent=2)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
