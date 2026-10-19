#!/usr/bin/env python3
"""Check locked SG2 references and compare SG2+CRF with the final pipeline."""
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
from src.segmentation import controlled_cv as cv
from scripts.select_propagation_cv import best_threshold, class_ious, pooled_miou, image_ious, run_screen
from scripts.narmyn.attention.an1_propagation_depth import rates
from scripts.narmyn.attention.finalize_sg5 import load as load_sg5

BASE = ROOT/'outputs/segmentation/controlled_attention_cv'
OUT = BASE/'sg2_densecrf_confirmation'
SPEC = ROOT/'configs/sg2_densecrf_confirmation.json'


def load(directory):
    return cv.load_results(sorted((directory/'shards').glob('class_*.npz')))


def validate(directory, dataset, spec, smoke):
    result = load(directory/dataset)
    old = load(BASE/'sg2_consensus_edges'/dataset)
    if result.configurations != tuple(spec['configurations']) or result.thresholds != tuple(spec['thresholds']):
        raise ValueError('configuration/threshold contract differs')
    if result.settings['spec'] != spec:
        raise ValueError('result specification differs')
    if not smoke and (len(result.complete) != 20 or not all(result.complete) or set(result.image_ids) != set(old.image_ids)):
        raise ValueError(f'{dataset}: need 20 complete, disjoint shards and exactly the original SG2 images')
    indices = {name: i for i, name in enumerate(old.image_ids)}
    order = [indices[n] for n in result.image_ids]
    if dataset == 'voc' and not np.array_equal(result.folds, old.folds[order]):
        raise ValueError('VOC fold assignments differ')
    checks = {}
    # SG5 has legitimate class-specific synonyms/prompt metadata; use its
    # existing shared-settings-validated loader, rather than weakening cv.load_results.
    other = load_sg5(dataset)
    for new, historical, source in [('W/REF', 'W/REF', old), ('W/P1', 'W/P1', old),
                                   ('W/SG2', 'W/G_cv_g20_n1', old), ('W/REF+R2', 'W/R2', other),
                                   ('W/P1+R2', 'W/P1+R2', other)]:
        order = [{n: i for i, n in enumerate(source.image_ids)}[n] for n in result.image_ids]
        a, b = result.counts[result.index(new)], source.counts[source.index(historical), order]
        maximum = float(np.max(np.abs(a-b)/np.maximum(a.sum(-1)[..., None], 1)))
        checks[new] = {'historical_cell': historical, 'max_pixel_difference': int(np.abs(a-b).max()), 'max_fraction_of_image': maximum}
        if maximum > .002:
            raise ValueError(f'{dataset} {new}: historical-reference mismatch {maximum:.5f}')
    return result, checks


def forced_oof(result, reference, candidate):
    """Evaluate the locked candidate on every held-out image, even if marginal.

    Distinct from protocol selection, which may retain REF in some/all folds.
    """
    a, b = np.zeros((len(result.image_ids), 4), np.int64), np.zeros((len(result.image_ids), 4), np.int64)
    folds = []
    for fold in range(5):
        held = result.folds == fold
        ta, _ = best_threshold(result, result.index(candidate), ~held)
        tb, _ = best_threshold(result, result.index(reference), ~held)
        a[held] = result.counts[result.index(candidate), held, ta]
        b[held] = result.counts[result.index(reference), held, tb]
        folds.append({'fold': fold, 'candidate_threshold': result.thresholds[ta], 'reference_threshold': result.thresholds[tb]})
    return {'reference': reference, 'candidate': candidate, 'reference_oof': pooled_miou(b, result.classes),
            'candidate_oof': pooled_miou(a, result.classes), 'gain': pooled_miou(a, result.classes)-pooled_miou(b, result.classes),
            'paired_image_iou': cv.paired_bootstrap(image_ious(a), image_ious(b)), 'fold_thresholds': folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    voc, vc = validate(args.output_dir, 'voc', spec, args.smoke)
    coco, cc = validate(args.output_dir, 'coco', spec, args.smoke)
    if args.smoke:
        (args.output_dir/'smoke_validation.json').write_text(json.dumps({'voc': vc, 'coco': cc, 'passed': True}, indent=2)+'\n')
        print('LIVE SMOKE PASSED: SG2, one/two-pass and DenseCRF historical controls on VOC and COCO')
        return
    screens = {name: run_screen(voc, reference, [spec['candidate']], {spec['candidate']: 0})
               for name, reference in [('primary_vs_FTF', spec['primary_reference']), ('secondary_vs_one_pass_CRF', spec['secondary_reference'])]}
    paired_oof = {name: forced_oof(voc, reference, candidate) for name, reference, candidate in
                  [('primary_vs_FTF', 'W/P1+R2', 'W/SG2+R2'), ('secondary_vs_one_pass_CRF', 'W/REF+R2', 'W/SG2+R2'),
                   ('SG2_without_CRF_vs_REF', 'W/REF', 'W/SG2')]}
    full = np.ones(len(voc.image_ids), bool)
    thresholds = {n: best_threshold(voc, voc.index(n), full)[0] for n in voc.configurations}
    cells, class_rows = {}, []
    for name in voc.configurations:
        ti = thresholds[name]
        cell = {'voc_threshold': voc.thresholds[ti]}
        for ds, result in [('voc', voc), ('coco', coco)]:
            counts = result.counts[result.index(name), :, ti]
            fixed = result.counts[result.index(name), :, result.thresholds.index(.45)]
            cell[ds] = {'at_voc_threshold': rates(counts, np.asarray(result.classes)),
                        'fixed_0.45': rates(fixed, np.asarray(result.classes)),
                        'fixed_threshold_miou': {str(t): pooled_miou(result.counts[result.index(name), :, result.thresholds.index(t)], result.classes) for t in (.35, .45, .55)}}
            values = class_ious(result, result.index(name), ti, np.ones(len(result.image_ids), bool))
            baseline = class_ious(result, result.index('W/P1+R2'), thresholds['W/P1+R2'], np.ones(len(result.image_ids), bool))
            cell[ds]['per_class_iou'] = values
            cell[ds]['per_class_delta_vs_FTF'] = {c: values[c]-baseline[c] for c in values}
            for c in values:
                class_rows.append({'dataset': ds, 'cell': name, 'class': c, 'threshold': voc.thresholds[ti],
                                   'iou': values[c], 'delta_vs_FTF': values[c]-baseline[c]})
        cells[name] = cell
    transfer = {}
    for key, screen in screens.items():
        reference, candidate = screen['reference'], spec['candidate']
        a = coco.counts[coco.index(candidate), :, thresholds[candidate]]
        b = coco.counts[coco.index(reference), :, thresholds[reference]]
        transfer[key] = {'candidate': candidate, 'reference': reference, 'candidate_miou': pooled_miou(a, coco.classes),
                         'reference_miou': pooled_miou(b, coco.classes), 'delta': pooled_miou(a, coco.classes)-pooled_miou(b, coco.classes),
                         'paired_image_iou': cv.paired_bootstrap(image_ious(a), image_ious(b)), 'protocol_frozen': screen['frozen']}
    summary = {'spec': spec, 'voc_images': len(voc.image_ids), 'coco_images': len(coco.image_ids),
               'reference_checks': {'voc': vc, 'coco': cc}, 'screens': screens, 'forced_candidate_oof': paired_oof,
               'cells': cells, 'transfer': transfer,
               'note': 'No parameter or threshold chosen on COCO. Existing FTF at its locked .45 threshold is reported alongside the framework-selected thresholds. CRF cells have no continuous-map AP.'}
    for filename, value in [('summary.json', summary), ('fold_results.json', {'screens': screens, 'forced_candidate_oof': paired_oof})]:
        (args.output_dir/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    with (args.output_dir/'per_class_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader(); writer.writerows(class_rows)
    lines = ['# Locked SG2 + DenseCRF confirmation', '', summary['note'], '',
             '| Comparison | OOF reference | Locked candidate OOF | Delta | Protocol frozen | Candidate fold votes |',
             '|---|---:|---:|---:|---|---:|']
    for key, s in screens.items():
        oof = paired_oof[key]
        lines.append(f"| {key} | {oof['reference_oof']:.4f} | {oof['candidate_oof']:.4f} | {oof['gain']:+.4f} | {s['frozen']} | {s['selection_counts'].get(spec['candidate'], 0)}/5 |")
    lines += ['', '| Cell | VOC threshold | VOC mIoU | COCO mIoU | VOC @.45 | COCO @.45 |', '|---|---:|---:|---:|---:|---:|']
    for name, c in cells.items():
        lines.append(f"| {name} | {c['voc_threshold']:.3f} | {c['voc']['at_voc_threshold']['miou']:.4f} | {c['coco']['at_voc_threshold']['miou']:.4f} | {c['voc']['fixed_0.45']['miou']:.4f} | {c['coco']['fixed_0.45']['miou']:.4f} |")
    lines += ['', 'OOF rows evaluate the locked candidate in every held-out fold with training-fold-selected thresholds. Protocol selection is separately reported and can retain FTF for a marginal gain. All class IoUs/deltas, recall/precision and paired intervals are in summary.json and per_class_results.csv.', '']
    (args.output_dir/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
