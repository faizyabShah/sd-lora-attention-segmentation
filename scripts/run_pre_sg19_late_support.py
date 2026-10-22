#!/usr/bin/env python3
"""Resumable live late-observation diagnostic; stores candidate-pixel features only."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sg18_temporal_windowed as sg18
from src.segmentation import locked_refinements as lr
from src.segmentation import pre_sg19_late_support as late

SPEC = ROOT / 'configs/pre_sg19_late_support.json'
OUT = ROOT / 'outputs/segmentation/analysis/pre_sg19_late_support'
SOURCE = ROOT / 'outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed'


def upsample(x, shape):
    return F.interpolate(x.reshape(-1, 1, 64, 64), size=shape,
                         mode='bilinear', align_corners=False)[:, 0]


def process(row, dataset, pipe, ca_modules, selected, ca_spec, device, spec):
    manifest = sg18.transfer.EVAL / ('ovam/voc_sim' if dataset == 'voc' else 'ovam/coco_captions')
    sg18.transfer._ATTRIBUTION_PROMPT[0] = row['attribution_prompt']
    sg18.common._RECORDERS.clear()
    sample = sg18.common.live.generate_attention_sample(pipe, row, manifest, 'float16', cross_map_kind='conditional')
    recorder = sg18.common._RECORDERS[-1]
    try:
        resolution = {name: size for size, names in sample.modules_by_resolution.items() for name in names}
        seed = sg18.common.ca_seed(recorder, ca_modules, resolution, ca_spec, device).reshape(-1)
        middle, late_mean, observations = late.propagate(recorder.sa_qk[selected], seed)
        truth = np.asarray(sample.target_mask, bool)
        normalized = lr.full_scores(torch.stack((seed, middle)), truth.shape).cpu().numpy()
        ca_mask = normalized[0] >= spec['ca_threshold']
        raw = upsample(torch.stack((seed, middle, late_mean)).float(), truth.shape).cpu().numpy()
        c_full, m_full, l_full = raw
        positive = m_full > c_full
        eligible_by_threshold = {}
        labels_by_threshold = {}
        for threshold in spec['middle_thresholds']:
            middle_mask = normalized[1] >= threshold
            eligible, label_columns = late.comparison_masks(truth, middle_mask, ca_mask, positive)
            eligible_by_threshold[threshold] = eligible
            labels_by_threshold[threshold] = np.stack(label_columns, axis=1).astype(np.uint8)
        union = np.logical_or.reduce(list(eligible_by_threshold.values()))
        union_idx = np.flatnonzero(union.ravel())
        full_observations = upsample(observations, truth.shape).reshape(80,-1)
        selected_observations = full_observations[:,torch.as_tensor(union_idx,device=full_observations.device)].cpu().numpy()
        union_features = late.features_at_pixels(selected_observations,c_full[union],m_full[union],spec['epsilon'])
        if not np.allclose(union_features[:,0],l_full[union],rtol=1e-4,atol=2e-6):
            raise ValueError('late observation mean disagrees with late map')
        if not np.isfinite(union_features).all():
            raise ValueError('nonfinite support feature')
        arrays = {}
        for threshold in spec['middle_thresholds']:
            token = str(threshold).replace('.', 'p')
            eligible = eligible_by_threshold[threshold]
            labels = labels_by_threshold[threshold]
            values = union_features[eligible[union]]
            if len(values) != len(labels) or (labels[:, 0]+labels[:, 1] != 1).any():
                raise ValueError('candidate labels invalid')
            arrays['features_'+token] = values
            arrays['labels_'+token] = labels
        # Cross-run drift is recorded rather than silently conflating trajectories.
        source_cache, _ = sg18.paths(SOURCE, dataset, row['image_id'])
        with np.load(source_cache, allow_pickle=False) as src:
            source_spec = json.loads((ROOT/'configs/sg18_temporal_windowed_propagation.json').read_text())
            sg18.verify_cache(src, row, dataset, source_spec)
            cached_middle = src['maps'][1]
            cached_late = (21*src['maps'][2].astype(np.float64)-11*cached_middle.astype(np.float64))/10
            drift = [float(np.mean(np.abs(middle.cpu().numpy()-cached_middle))),
                     float(np.mean(np.abs(late_mean.cpu().numpy()-cached_late)))]
        return arrays, drift
    finally:
        recorder.ca_calls.clear(); recorder.sa_qk.clear(); sample.clear()
        sg18.common._RECORDERS.clear(); del sample, recorder
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    if tuple(spec['features']) != late.FEATURES or spec['middle_calls_inclusive'] != [10,20] or spec['late_calls_inclusive'] != [21,30]:
        raise ValueError('locked feature/window contract changed')
    completed = json.loads((SOURCE/'completed.json').read_text())
    if not completed['all_required_stages_complete'] or completed['images'] != {'voc':490,'coco':707}:
        raise ValueError('SG18 cached map source incomplete')
    locked = json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    frozen = json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not frozen['frozen'] or locked['sa']['module_index'] != 15 or locked['sa']['observations'] != 248:
        raise ValueError('frozen CA/SA contract changed')
    pipe, ca_modules, selected, ca_spec = sg18.make_pipe(args.model, args.device, locked, frozen)
    for dataset in ('voc','coco'):
        rows = sg18.rows_for(dataset, first_only=args.smoke)
        for index, (classname, row) in enumerate(rows):
            path = args.output_dir/'cache'/dataset/(sg18.key(row['image_id'])+'.npz')
            if path.exists():
                with np.load(path, allow_pickle=False) as d:
                    meta = json.loads(str(d['meta']))
                    if meta['image_id'] != row['image_id'] or meta['spec'] != spec:
                        raise ValueError('foreign pre-SG19 checkpoint')
                continue
            arrays, drift = process(row, dataset, pipe, ca_modules, selected, ca_spec, args.device, spec)
            meta = {'image_id':row['image_id'], 'dataset':dataset, 'class':classname,
                    'spec':spec, 'sg18_cache_middle_late_mae':drift}
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name+'.tmp.npz')
            np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
            tmp.replace(path)
            print(f'[{dataset}:{classname}] {index+1}/{len(rows)} {row["image_id"]}', flush=True)
    print('Pre-SG19 live feature extraction completed', flush=True)


if __name__ == '__main__':
    main()
