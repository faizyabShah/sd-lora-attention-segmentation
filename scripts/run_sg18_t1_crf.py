#!/usr/bin/env python3
"""Generate a class shard and score locked T1/P1 with unchanged DenseCRF."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sg18_temporal_windowed as sg18
from src.segmentation import controlled_cv as cv
from src.segmentation.ovam_full import _import_densecrf, OVAM_CRF_PARAMETERS

SPEC = ROOT / 'configs/sg18_t1_crf.json'
OUT = ROOT / 'outputs/segmentation/controlled_attention_cv/sg18_t1_crf'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('voc', 'coco'), required=True)
    parser.add_argument('--class-shard', type=int, choices=range(20), required=True)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--crf-site', type=Path, default=ROOT / 'outputs/dependencies/simplecrf_py310')
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    locked = json.loads((ROOT / 'configs/sg12_crf_sg14_sg15.json').read_text())
    frozen = json.loads((ROOT / 'configs/controlled_sa_seed_frozen.json').read_text())
    if not frozen['frozen'] or locked['sa']['module_index'] != 15 or locked['sa']['observations'] != 248:
        raise ValueError('frozen CA/SA contract changed')
    if tuple(spec['names']) != ('W/T1', 'W/T1+R2', 'W/P1', 'W/P1+R2'):
        raise ValueError('locked cell names changed')
    if tuple(spec['thresholds']) != tuple(sg18.transfer.THRESHOLDS):
        raise ValueError('DenseCRF threshold grid changed')
    _import_densecrf(args.crf_site)
    classname, all_rows = (sg18.transfer.voc_rows if args.dataset == 'voc' else sg18.transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    folds = cv.read_folds(ROOT / 'configs/voc_sim_folds.csv') if args.dataset == 'voc' else {r['image_id']: 0 for r in all_rows}
    pipe, ca_modules, selected, ca_spec = sg18.make_pipe(args.model, args.device, locked, frozen)
    settings = {'package': 'sg18_t1_crf_v1', 'spec': spec, 'dataset': args.dataset,
                'ca_seed': ca_spec, 'selected_sa_module': selected, 'model': args.model,
                'model_dtype': 'float32', 'qk_storage_dtype': 'float16',
                'crf_parameters': list(OVAM_CRF_PARAMETERS)}
    path = args.output_dir / args.dataset / 'shards' / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    writer = cv.ShardWriter(path, tuple(spec['names']), folds, settings, spec['thresholds'])
    for row in rows:
        if row['image_id'] in writer.completed:
            continue
        maps, _, truth, image, _ = sg18.live_maps(
            pipe, ca_modules, selected, ca_spec, row, args.dataset, args.device,
            names=('W/T1', 'W/P1'))
        values = torch.as_tensor(np.stack([maps['W/T1'], maps['W/P1']]).reshape(2, 64, 64))
        counts, ap = cv.score_maps(values, truth, 'minmax', spec['thresholds'])
        crf = np.concatenate([sg18.transfer.crf_counts(m, truth, image, args.crf_site) for m in values])
        ordered_counts = np.stack((counts[0], crf[0], counts[1], crf[1]))
        ordered_ap = np.array((ap[0], np.nan, ap[1], np.nan))
        if ordered_counts.shape != (4, 17, 4) or not np.all(ordered_counts.sum(-1) == truth.size):
            raise ValueError('invalid T1/P1 plus DenseCRF counts')
        writer.add(row['image_id'], classname, ordered_counts, ordered_ap)
        print(f'[{args.dataset}:{classname}] {row["image_id"]} {len(writer.completed)}/{len(rows)}', flush=True)
        del maps, truth, image, values, counts, ap, crf, ordered_counts
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    if args.max_samples is None:
        if writer.completed != {r['image_id'] for r in all_rows}:
            raise ValueError('incomplete or foreign class shard')
        writer.finish()


if __name__ == '__main__':
    main()
