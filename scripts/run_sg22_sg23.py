#!/usr/bin/env python3
"""Resumable class shard for the locked SG22 or SG23 factorial."""
from __future__ import annotations

import argparse
import gc
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
from src.segmentation import controlled_cv as cv
from src.segmentation import locked_refinements as lr
from src.segmentation import sg18_temporal as temporal
from src.segmentation import sg22_sg23 as operators
from src.segmentation.ovam_full import _import_densecrf, OVAM_CRF_PARAMETERS

BASE = ROOT/'outputs/segmentation/controlled_attention_cv'
DIAG_FIELDS = ('tp_gained','tp_lost','fp_gained','fp_removed',
               'fp_gt16_gained','fp_gt16_removed','fp_gt64_gained','fp_gt64_removed',
               'fp_separate_gained','fp_separate_removed')


def names_for(experiment):
    raw = operators.SG22_RAW if experiment == 'sg22' else (*operators.SG23_RAW, operators.SG23_AUX)
    return tuple(name for r in raw for name in (r, r+'+R2'))


def transition(before, after, truth, distance, where=None):
    """Fixed-threshold changed pixels; component membership belongs to each mask."""
    eligible = np.ones_like(truth,bool) if where is None else np.asarray(where,bool)
    gained = after & ~before & eligible
    removed = before & ~after & eligible
    def separate(mask):
        labels, _ = ndimage.label(mask)
        touches = np.unique(labels[mask & truth])
        return mask & ~np.isin(labels, touches[touches > 0])
    bs, an = separate(before), separate(after)
    fg, fr = gained & ~truth, removed & ~truth
    return np.asarray(((gained & truth).sum(), (removed & truth).sum(), fg.sum(), fr.sum(),
                       (fg & (distance > 16)).sum(), (fr & (distance > 16)).sum(),
                       (fg & (distance > 64)).sum(), (fr & (distance > 64)).sum(),
                       (fg & an).sum(), (fr & bs).sum()), dtype=np.int64)


def diagnostics(experiment, maps, seed, truth):
    raw = operators.SG22_RAW if experiment == 'sg22' else operators.SG23_RAW
    scores = lr.full_scores(torch.stack((seed, *(maps[n] for n in raw))), truth.shape).cpu().numpy()
    masks = {n: scores[i+1] >= .45 for i, n in enumerate(raw)}
    ca = scores[0] >= .45
    distance = ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape, np.inf)
    events = np.stack([temporal.events(ca, masks[n], truth, distance)[0] for n in raw])
    if experiment == 'sg22':
        pairs = [(f'{w}-P{d}', f'{w}-P{d+1}') for w in ('ALL','MID') for d in (1,2)]
    else:
        pairs = [(f'MID-R0', f'MID-{method}') for method in ('SG2','SG11','SG12')]
    changed = np.stack([transition(masks[a], masks[b], truth, distance,
                                   None if experiment == 'sg22' else ~ca) for a,b in pairs])
    return events, changed


def process(row, dataset, pipe, ca_modules, selected, ca_spec, device, spec, crf_site):
    manifest = sg18.transfer.EVAL/('ovam/voc_sim' if dataset == 'voc' else 'ovam/coco_captions')
    sg18.transfer._ATTRIBUTION_PROMPT[0] = row['attribution_prompt']
    sg18.common._RECORDERS.clear()
    sample = sg18.common.live.generate_attention_sample(pipe, row, manifest, 'float16', cross_map_kind='conditional')
    recorder = sg18.common._RECORDERS[-1]
    try:
        resolutions = {n:r for r, modules in sample.modules_by_resolution.items() for n in modules}
        seed = sg18.common.ca_seed(recorder, ca_modules, resolutions, ca_spec, device).reshape(-1)
        maps = operators.aggregate(recorder.sa_qk[selected], seed, spec['experiment'])
        raw = operators.SG22_RAW if spec['experiment'] == 'sg22' else (*operators.SG23_RAW, operators.SG23_AUX)
        values = torch.stack([maps[n] for n in raw]).reshape(-1, 64, 64).cpu()
        truth = np.asarray(sample.target_mask, bool)
        image = np.asarray(sample.generated_image, np.uint8)
        counts, ap = cv.score_maps(values, truth, 'minmax', spec['thresholds'])
        crf = np.concatenate([sg18.transfer.crf_counts(v, truth, image, crf_site) for v in values])
        output_counts = np.stack([x for i in range(len(raw)) for x in (counts[i], crf[i])])
        output_ap = np.asarray([x for i in range(len(raw)) for x in (ap[i], np.nan)])
        events, changes = diagnostics(spec['experiment'], maps, seed, truth)
        if output_counts.shape != (len(spec['names']),17,4) or not np.all(output_counts.sum(-1) == truth.size):
            raise ValueError('invalid raw/CRF counts')
        return dict(counts=output_counts, ap=output_ap, events=events, changes=changes)
    finally:
        recorder.ca_calls.clear(); recorder.sa_qk.clear(); sample.clear()
        sg18.common._RECORDERS.clear()
        del recorder, sample
        gc.collect(); torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', choices=('sg22','sg23'), required=True)
    p.add_argument('--dataset', choices=('voc','coco'), required=True)
    p.add_argument('--class-shard', type=int, choices=range(20), required=True)
    p.add_argument('--max-samples', type=int)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    p.add_argument('--device', default='cuda')
    p.add_argument('--crf-site', type=Path, default=ROOT/'outputs/dependencies/simplecrf_py310')
    args = p.parse_args()
    spec = json.loads((ROOT/f'configs/{args.experiment}_factorial.json').read_text())
    frozen = json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    locked = json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    if spec['names'] != list(names_for(args.experiment)) or spec['thresholds'] != locked['thresholds']:
        raise ValueError('changed locked cell/threshold contract')
    if locked['sa']['module_index'] != 15 or locked['sa']['observations'] != 248 or not frozen['frozen']:
        raise ValueError('changed frozen CA/SA contract')
    _import_densecrf(args.crf_site)
    if tuple(spec['thresholds']) != tuple(sg18.transfer.THRESHOLDS):
        raise ValueError('CRF helper threshold grid differs')
    out = args.output_dir or BASE/args.experiment
    classname, all_rows = (sg18.transfer.voc_rows if args.dataset == 'voc' else sg18.transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    pipe, ca_modules, selected, ca_spec = sg18.make_pipe(args.model,args.device,locked,frozen)
    path = out/args.dataset/'shards'/f'class_{args.class_shard:02d}_{classname.replace(" ", "_")}.npz'
    settings = {'experiment':args.experiment,'spec':spec,'dataset':args.dataset,'model':args.model,
                'ca_seed':ca_spec,'selected_sa_module':selected,'crf_parameters':list(OVAM_CRF_PARAMETERS),
                'diagnostic_threshold':.45,'diagnostic_ca_threshold':.45}
    folds = cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if args.dataset == 'voc' else {r['image_id']:0 for r in all_rows}
    writer = cv.ShardWriter(path, spec['names'], folds, settings, spec['thresholds'])
    diag_path = path.with_suffix('.diagnostics.npz')
    if diag_path.exists():
        with np.load(diag_path,allow_pickle=False) as d:
            ids = list(map(str,d['image_ids']))
            diagnostic = {str(i):(e,c) for i,e,c in zip(ids,d['events'],d['changes'])}
    else:
        diagnostic = {}
    if writer.completed-set(diagnostic):
        raise ValueError('metric shard ahead of diagnostic shard')
    for row in rows:
        if row['image_id'] in writer.completed:
            continue
        result = process(row,args.dataset,pipe,ca_modules,selected,ca_spec,args.device,spec,args.crf_site)
        diagnostic[row['image_id']] = (result['events'],result['changes'])
        diag_path.parent.mkdir(parents=True,exist_ok=True)
        temp = diag_path.with_name(diag_path.name+'.tmp.npz')
        np.savez_compressed(temp,image_ids=np.asarray(list(diagnostic)),
                            events=np.stack([v[0] for v in diagnostic.values()]),
                            changes=np.stack([v[1] for v in diagnostic.values()]))
        temp.replace(diag_path)
        writer.add(row['image_id'],classname,result['counts'],result['ap'])
        print(f'[{args.experiment}:{args.dataset}:{classname}] {row["image_id"]} '
              f'{len(writer.completed)}/{len(rows)}',flush=True)
        del result
    if args.max_samples is None:
        if writer.completed != {r['image_id'] for r in all_rows}:
            raise ValueError('incomplete class shard')
        writer.finish()


if __name__ == '__main__':
    main()
