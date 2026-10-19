#!/usr/bin/env python3
"""Validate, select, report and render SG11/SG12 with compact SG-Diag outputs."""
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
from scripts.select_propagation_cv import best_threshold, pooled_miou, run_screen
from src.segmentation.observation_reliability import configuration_names
from src.segmentation.propagation_diagnostics import FEATURES, EVENTS, BOUNDS, histogram_auc

DEFAULT = ROOT/'outputs/segmentation/controlled_attention_cv/sg11_sg12_observation_reliability'
SPEC = ROOT/'configs/sg11_sg12_observation_reliability.json'
SG8 = ROOT/'outputs/segmentation/controlled_attention_cv/sg8_observation_consensus'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_result(directory, dataset, spec, smoke=False):
    paths = [p for p in sorted((directory/dataset/'shards').glob('class_*.npz')) if '.diagnostics.' not in p.name]
    result = cv.load_results(paths)
    if result.configurations != configuration_names(spec) or result.thresholds != tuple(spec['thresholds']):
        raise ValueError('configuration/threshold contract differs')
    historical = cv.load_results([p for p in sorted((SG8/dataset/'shards').glob('class_*.npz')) if '.diagnostics.' not in p.name])
    if not smoke:
        if len(paths) != 20 or not all(result.complete) or set(result.image_ids) != set(historical.image_ids):
            raise ValueError(f'{dataset}: expected 20 complete shards and the full SG8 manifest')
        if len(result.image_ids) != (490 if dataset == 'voc' else 707):
            raise ValueError('unexpected dataset size')
    historic_indices = {name: i for i, name in enumerate(historical.image_ids)}
    order = [historic_indices[name] for name in result.image_ids]
    if dataset == 'voc' and not np.array_equal(result.folds, historical.folds[order]):
        raise ValueError('fold assignments differ from SG8')
    checks = {}
    for name in result.configurations[:4]:
        a = result.counts[result.index(name)]
        b = historical.counts[historical.index(name), order]
        pixels = a.sum(-1)
        relative_error = float(np.max(np.abs(a-b)/np.maximum(pixels[..., None], 1)))
        checks[name] = {'max_count_difference': int(np.max(np.abs(a-b))), 'max_fraction_of_image': relative_error}
        if relative_error > .002:
            raise ValueError(f'{dataset}: {name} SG8 counts differ by {relative_error:.5f} of image')
    return result, checks


def rates(counts):
    tp, fp, fn = counts[:, :3].sum(0)
    return {'recall': float(tp/max(tp+fn, 1)), 'precision': float(tp/max(tp+fp, 1))}


def per_class(counts, classes):
    classes = np.asarray(classes)
    return {c: float((s := counts[classes == c, :3].sum(0))[0]/max(s.sum(), 1)) for c in sorted(set(classes))}


def diagnostic_auc(hist):
    output = {}
    for label, pos, neg in [('true_vs_false_additions', hist[0], hist[1]),
                            ('useful_vs_harmful_changes', hist[0]+hist[2], hist[1]+hist[3])]:
        output[label] = {feature: {'higher_predicts_useful': histogram_auc(pos[i], neg[i]),
                                  'lower_predicts_useful': histogram_auc(pos[i][::-1], neg[i][::-1])}
                         for i, feature in enumerate(FEATURES)}
    return output


def finalize_diagnostics(directory, dataset, result, spec):
    output = directory/'diagnostics'
    output.mkdir(exist_ok=True)
    by_class, reliability, selected, pair_sums, image_ids = {}, [], [], [], []
    all_hist, all_totals = None, None
    class_reliability_rows = []
    for path in sorted((directory/dataset/'shards').glob('*.diagnostics.npz')):
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data['spec'])) != spec:
                raise ValueError('diagnostic spec differs')
            ids = data['image_ids'].tolist()
            classname = result.classes[result.image_ids.index(ids[0])]
            if any(result.classes[result.image_ids.index(i)] != classname for i in ids):
                raise ValueError('mixed classes in diagnostic shard')
            hist, totals = data['histogram'], data['event_totals']
            # Every feature must account for the same eligible event pixels.
            if not np.all(hist.sum(-1) == hist.sum(-1)[..., :1]):
                raise ValueError('feature histogram population mismatch')
            by_class[classname] = {str(t): diagnostic_auc(hist[i]) for i, t in enumerate(spec['diagnostic']['thresholds'])}
            all_hist = hist.copy() if all_hist is None else all_hist+hist
            all_totals = totals.copy() if all_totals is None else all_totals+totals
            class_scores = data['reliability'].mean(0)
            class_frequency = data['selected'].mean(0)
            for observation in range(spec['sa']['observations']):
                class_reliability_rows.append({'class': classname, 'module_index': 15, 'call': observation//8, 'head': observation%8, 'mean_reliability': float(class_scores[observation]), **{f'selected_top{round(100*f)}': float(class_frequency[j, observation]) for j, f in enumerate(spec['sg12']['fractions'])}})
            reliability.append(data['reliability'])
            selected.append(data['selected'])
            pair_sums.append(data['pairwise_sum'])
            image_ids += ids
    if set(image_ids) != set(result.image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError('diagnostic/result image IDs differ')
    scores = np.concatenate(reliability)
    selection = np.concatenate(selected)
    pairwise = sum(pair_sums)/len(image_ids)
    avg = scores.mean(0)
    frequency = selection.mean(0)
    rows = []
    for i in range(spec['sa']['observations']):
        rows.append({'module_index': 15, 'module': result.settings['selected_sa_module'], 'call': i//8, 'head': i%8,
                     'mean_reliability': float(avg[i]), 'std_across_images': float(scores[:, i].std()),
                     **{f'selected_top{round(100*f)}': float(frequency[j, i]) for j, f in enumerate(spec['sg12']['fractions'])}})
    write_csv(output/f'{dataset}_observation_reliability.csv', rows)
    write_csv(output/f'{dataset}_per_class_observation_reliability.csv', class_reliability_rows)
    hist_counts, hist_edges = np.histogram(scores.ravel(), bins=50)
    np.savez_compressed(output/f'{dataset}_aggregates.npz', pairwise_agreement=pairwise, histograms=all_hist,
                       reliability_hist_counts=hist_counts, reliability_hist_edges=hist_edges,
                       feature_names=np.asarray(FEATURES), event_names=np.asarray(EVENTS), bounds=np.asarray(BOUNDS),
                       thresholds=np.asarray(spec['diagnostic']['thresholds']), reliability_mean=avg, selected_frequency=frequency)
    return {'pixels': {str(t): {'appreciable': int(all_totals[i, 0]), 'binary_flips': int(all_totals[i, 1]),
                                'eligible_flips': int(all_totals[i, 2]),
                                'event_counts': dict(zip(EVENTS, map(int, all_hist[i, :, 0].sum(-1))))}
                       for i, t in enumerate(spec['diagnostic']['thresholds'])},
            'auc': {str(t): diagnostic_auc(all_hist[i]) for i, t in enumerate(spec['diagnostic']['thresholds'])},
            'per_class_auc': by_class,
            'head_mean_reliability': avg.reshape(31, 8).mean(0).tolist(),
            'call_mean_reliability': avg.reshape(31, 8).mean(1).tolist(),
            'lowest_reliability_observations': sorted(rows, key=lambda r: r['mean_reliability'])[:16],
            'auc_estimator': '1024-bin ROC-AUC and step PR-AUC/AP; all eligible pixels, no sampling; raw and reversed directions are labelled; no model fitted'}


def render_visuals(directory, spec, summary):
    from PIL import Image, ImageDraw, ImageFont
    from src.segmentation.sg_visual_audit import resize_map, _overlay, _mask_rgb, _iou
    visual = directory/'visuals'
    visual.mkdir(exist_ok=True)
    caches = sorted((directory/'visual_cache').glob('*.npz'))
    index = []
    for path in caches:
        with np.load(path, allow_pickle=False) as data:
            names, classname, image_id = data['names'].tolist(), str(data['classname']), str(data['image_id'])
            maps, seed, original, truth = data['maps'], data['seed'], data['image'], data['target'].astype(bool)
        shape = truth.shape
        original = np.asarray(Image.fromarray(original).resize((shape[1], shape[0])))
        image = original
        ca = _overlay(original, resize_map(seed, shape))
        def value(name):
            return maps[names.index(name)]
        def overlay(name):
            return _overlay(original, resize_map(value(name), shape))
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', 14)
        except OSError:
            font = ImageFont.load_default()
        def write_sheet(filename, rows):
            tile, row_height = 220, 280
            canvas = Image.new('RGB', (6*tile, 35+row_height*len(rows)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 8), f'{classname} / {image_id} / VOC-selected thresholds', font=font, fill='black')
            for ri, (name, panels) in enumerate(rows):
                y = 35+ri*row_height
                draw.text((5, y), name, font=font, fill='black')
                for ci, (label, panel) in enumerate(panels):
                    x = ci*tile
                    draw.text((x+4, y+20), label, font=font, fill='black')
                    canvas.paste(Image.fromarray(panel).resize((tile, tile)), (x, y+45))
            canvas.save(visual/filename, quality=92)
        rows = []
        for name in names:
            if '/SG11/' not in name:
                continue
            robust = 'W/A/median' if '/median/' in name else 'W/A/trim20'
            full = resize_map(value(name), shape)
            threshold = summary['cells'][name]['voc_threshold']
            prediction = full >= threshold
            # TP green, FP red, FN blue, with original image visible underneath.
            colors = np.zeros_like(original)
            colors[prediction & truth] = (0, 255, 0)
            colors[prediction & ~truth] = (255, 0, 0)
            colors[~prediction & truth] = (0, 100, 255)
            mask_overlay = original.copy()
            region = prediction | truth
            mask_overlay[region] = (.45*original[region]+.55*colors[region]).astype(np.uint8)
            rows.append((name, [('Image', image), ('CA seed', ca), ('SG8 C rho=.80', overlay('W/C/rho0p80')),
                                (robust, overlay(robust)), ('SG11', overlay(name)),
                                (f'GT/pred IoU={_iou(prediction, truth):.3f}', mask_overlay)]))
        write_sheet(f"{classname.replace(' ', '_')}_sg11.jpg", rows)
        frozen = summary['screens']['SG12']['frozen']
        chosen = [n for n in names if '/SG12/' in n]
        rows = []
        for name in chosen:
            threshold = summary['cells'][name]['voc_threshold']
            pred = resize_map(value(name), shape) >= threshold
            rows.append((name, [('Image', image), ('REF', overlay('W/REF')), ('Top 50%', overlay('W/SG12/top50')),
                                ('Top 70%', overlay('W/SG12/top70')), (f'Frozen: {frozen}', overlay(frozen)),
                                (f'{name} IoU={_iou(pred, truth):.3f}', _mask_rgb(pred, truth))]))
        write_sheet(f"{classname.replace(' ', '_')}_sg12.jpg", rows)
        index.append({'class': classname, 'image_id': image_id, 'sampling': spec['visual_sampling'],
                      'sg11_sheet': f"{classname.replace(' ', '_')}_sg11.jpg", 'sg12_sheet': f"{classname.replace(' ', '_')}_sg12.jpg"})
    expected = set(spec['visual_classes'])
    if {r['class'] for r in index} != expected:
        # A finalized/restarted job can use its already complete visual index.
        old = visual/'index.json'
        if not caches and old.exists() and {r['class'] for r in json.loads(old.read_text())} == expected:
            return
        raise ValueError('missing preregistered visual examples')
    dump(visual/'index.json', index)
    # Delete only this package's seven temporary visual caches after rendering.
    for path in caches:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    parser.add_argument('--smoke', action='store_true', help='validate partial VOC/COCO and SG8 controls without selecting')
    args = parser.parse_args()
    directory = args.output_dir
    spec = json.loads(SPEC.read_text())
    voc, vc = load_result(directory, 'voc', spec, args.smoke)
    coco, cc = load_result(directory, 'coco', spec, args.smoke)
    if args.smoke:
        # Also validate diagnostic population alignment on the tiny live runs.
        for dataset, results in [('voc', voc), ('coco', coco)]:
            ids = []
            for path in (directory/dataset/'shards').glob('*.diagnostics.npz'):
                with np.load(path, allow_pickle=False) as data:
                    ids += data['image_ids'].tolist()
                    if not np.isfinite(data['reliability']).all():
                        raise ValueError('nonfinite reliability')
            if set(ids) != set(results.image_ids):
                raise ValueError('smoke diagnostic IDs differ')
        dump(directory/'smoke_validation.json', {'voc': vc, 'coco': cc, 'status': 'passed'})
        print('LIVE SMOKE PASSED: both datasets, all four SG8 controls, diagnostics', flush=True)
        return
    screens = {}
    for family in ('SG11', 'SG12'):
        candidates = [n for n in voc.configurations if f'/{family}/' in n]
        preference = {n: (int(n[-2:]) if family == 'SG12' else float(n.rsplit('rho', 1)[1].replace('p', '.'))*10 + (.1 if '/trim20/' in n else 0)) for n in candidates}
        screens[family] = run_screen(voc, 'W/REF', candidates, preference)
        screen = screens[family]
        screen['chatgpt_voc_target_met'] = bool(screen['oof_miou_selection'] > .7970 and screen['vote_winner'] != 'W/REF' and screen['vote_winner_votes'] >= 4)
    thresholds = {n: best_threshold(voc, voc.index(n), np.ones(len(voc.image_ids), bool))[0] for n in voc.configurations}
    cells, class_rows = {}, []
    for name in voc.configurations:
        ti = thresholds[name]
        cell = {'voc_threshold': voc.thresholds[ti]}
        for dataset, result in [('voc', voc), ('coco', coco)]:
            counts = result.counts[result.index(name), :, ti]
            reference = result.counts[result.index('W/REF'), :, thresholds['W/REF']]
            values, base = per_class(counts, result.classes), per_class(reference, result.classes)
            cell[f'{dataset}_miou'] = pooled_miou(counts, result.classes)
            cell[f'{dataset}_delta_reference'] = cell[f'{dataset}_miou']-pooled_miou(reference, result.classes)
            cell[f'{dataset}_rates'] = rates(counts)
            cell[f'{dataset}_fixed_threshold_miou'] = {str(t): pooled_miou(result.counts[result.index(name), :, result.thresholds.index(t)], result.classes) for t in (.35, .45, .55)}
            cell[f'{dataset}_mean_ap_64'] = float(np.nanmean(result.ap[result.index(name)]))
            cell[f'{dataset}_per_class'] = values
            cell[f'{dataset}_per_class_delta'] = {c: values[c]-base[c] for c in values}
            for c in values:
                class_rows.append({'dataset': dataset, 'cell': name, 'class': c, 'threshold': voc.thresholds[ti],
                                   'iou': values[c], 'reference_iou': base[c], 'delta': values[c]-base[c]})
        cells[name] = cell
    summary = {'spec': spec, 'voc_images': len(voc.image_ids), 'coco_images': len(coco.image_ids),
               'reference_validation_against_sg8': {'voc': vc, 'coco': cc}, 'screens': screens, 'cells': cells,
               'historical_sg8_screens': {k: v for k, v in json.loads((SG8/'summary.json').read_text())['screens'].items() if k in ('W-A', 'W-C')},
               'reference_context': 'W/REF is one-pass without DenseCRF. Existing final FTF is two passes + DenseCRF (COCO 0.6370). This screen does not replace FTF.',
               'diagnostics': {d: finalize_diagnostics(directory, d, r, spec) for d, r in [('voc', voc), ('coco', coco)]}}
    for family, screen in screens.items():
        screen['coco_frozen'] = cells[screen['frozen']]['coco_miou']
        screen['chatgpt_coco_target_met_descriptive_only'] = screen['coco_frozen'] > .625
    dump(directory/'summary.json', summary)
    dump(directory/'fold_results.json', screens)
    write_csv(directory/'per_class_results.csv', class_rows)
    write_csv(directory/'fold_winners.csv', [{'family': family, **row} for family, screen in screens.items() for row in screen['folds']])
    lines = ['# SG11 / SG12 / SG-Diag', '', summary['reference_context'], '',
             'VOC-Sim selects; COCO-Cap transfers only. Class words, no optimized tokens.', '',
             '| Family | Vote winner | Votes | OOF REF | OOF selection | Delta | Frozen | ChatGPT VOC target |',
             '|---|---|---:|---:|---:|---:|---|---|']
    for family, s in screens.items():
        lines.append(f"| {family} | {s['vote_winner']} | {s['vote_winner_votes']}/5 | {s['oof_miou_reference']:.4f} | {s['oof_miou_selection']:.4f} | {s['oof_gain']:+.4f} | {s['frozen']} | {s['chatgpt_voc_target_met']} |")
    lines += ['', 'Historical SG8 W-C selected rho=.80 in 4/5 folds: OOF 0.797014 (REF 0.787508). SG8 W-A split its votes and did not freeze a robust aggregator. These historical results are context, not new selection.', '', '## All cells and controls', '', '| Cell | VOC threshold | VOC mIoU | Δ REF | VOC recall / precision | COCO mIoU | Δ REF | COCO recall / precision |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, c in cells.items():
        vr, cr = c['voc_rates'], c['coco_rates']
        lines.append(f"| {name} | {c['voc_threshold']:.3f} | {c['voc_miou']:.4f} | {c['voc_delta_reference']:+.4f} | {vr['recall']:.4f} / {vr['precision']:.4f} | {c['coco_miou']:.4f} | {c['coco_delta_reference']:+.4f} | {cr['recall']:.4f} / {cr['precision']:.4f} |")
    for family, screen in screens.items():
        lines += ['', f'## {family}: fold winners', '', '| Fold | Winner | Threshold | Held-out chosen / REF |', '|---:|---|---:|---:|']
        for row in screen['folds']:
            lines.append(f"| {row['fold']} | {row['chosen']} | {row['chosen_threshold']:.3f} | {row['heldout_miou_chosen']:.4f} / {row['heldout_miou_reference']:.4f} |")
    lines += ['', '## SG-Diag', '', 'AUCs use all appreciable binary flips and 1024-bin histograms. PR-AUC is step average precision. GT enters diagnostics only. Both feature orientations are reported; these scores cannot select methods. No fitted combination or classifier.', '']
    for dataset in ('voc', 'coco'):
        for threshold, diagnostics in summary['diagnostics'][dataset]['auc'].items():
            population = summary['diagnostics'][dataset]['pixels'][threshold]
            lines += [f'### {dataset}, fixed diagnostic threshold {threshold}', '',
                      f"Eligible flips: {population['eligible_flips']}; events: {population['event_counts']}.", '', '| Signal (higher predicts useful) | Addition ROC / PR | All-change ROC / PR |', '|---|---:|---:|']
            for feature in FEATURES:
                a = diagnostics['true_vs_false_additions'][feature]['higher_predicts_useful']
                b = diagnostics['useful_vs_harmful_changes'][feature]['higher_predicts_useful']
                def fmt(x):
                    return 'undefined' if x['roc_auc'] is None else f"{x['roc_auc']:.4f} / {x['pr_auc']:.4f}"
                lines.append(f'| {feature} | {fmt(a)} | {fmt(b)} |')
    lines += ['', '## Class IoUs and deltas for every variant', '']
    for dataset in ('voc', 'coco'):
        for name, c in cells.items():
            lines += [f'### {dataset}: {name}', '', '| Class | IoU | Delta REF |', '|---|---:|---:|']
            lines += [f"| {cls} | {iou:.4f} | {c[f'{dataset}_per_class_delta'][cls]:+.4f} |" for cls, iou in c[f'{dataset}_per_class'].items()]
    lines += ['', 'Reliability by module/call/head, top-K selection frequencies, pairwise matrices and plots are in `diagnostics/`. Visual sheets are in `visuals/`.', '']
    (directory/'report.md').write_text('\n'.join(lines))
    (directory/'CHATGPT_RESULTS.md').write_text('\n'.join(lines))
    render_visuals(directory, spec, summary)
    print(json.dumps(screens, indent=2))


if __name__ == '__main__':
    main()
