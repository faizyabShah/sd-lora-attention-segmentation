#!/usr/bin/env python3
"""Locked, class-word SG2 + fixed DenseCRF confirmation on a class shard."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_sg11_sg12_observation_reliability import WordObservationRecorder, transfer, common
from src.segmentation import controlled_cv as cv
from src.segmentation.consensus_graph import locked_sg2_maps
from src.segmentation.ovam_full import _import_densecrf, OVAM_CRF_PARAMETERS

SPEC = ROOT/'configs/sg2_densecrf_confirmation.json'
OUT = ROOT/'outputs/segmentation/controlled_attention_cv/sg2_densecrf_confirmation'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('voc', 'coco'), required=True)
    parser.add_argument('--class-shard', type=int, choices=range(20), required=True)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--crf-site', type=Path, default=ROOT/'outputs/dependencies/simplecrf_py310')
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    seed_json = json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not seed_json.get('frozen'):
        raise ValueError('CA must be frozen')
    _import_densecrf(args.crf_site)
    classname, all_rows = (transfer.voc_rows if args.dataset == 'voc' else transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    manifest = transfer.EVAL / ('ovam/voc_sim' if args.dataset == 'voc' else 'ovam/coco_captions')
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32,
                                                  local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    transfer._PIPE = pipe
    processors = [n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith('attn2')))
    sa_modules = tuple(sorted(n for n in processors if n.endswith('attn1')))
    selected = sa_modules[spec['sa']['module_index']]
    WordObservationRecorder.selected_sa = selected
    ca_spec = common.resolve_seed_spec(seed_json, ca_modules, spec['sa']['calls'])
    names, thresholds = tuple(spec['configurations']), tuple(spec['thresholds'])
    folds = cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if args.dataset == 'voc' else {r['image_id']: 0 for r in all_rows}
    settings = {'package': 'sg2_densecrf_confirmation_v1', 'spec': spec, 'dataset': args.dataset,
                'ca_seed': ca_spec, 'selected_sa_module': selected, 'model': args.model,
                'model_dtype': 'float32', 'qk_storage_dtype': 'float16', 'crf_parameters': list(OVAM_CRF_PARAMETERS)}
    path = args.output_dir/args.dataset/'shards'/f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    writer = cv.ShardWriter(path, names, folds, settings, thresholds)
    for row in rows:
        if row['image_id'] in writer.completed:
            continue
        transfer._ATTRIBUTION_PROMPT[0] = row['attribution_prompt']
        common._RECORDERS.clear()
        sample = common.live.generate_attention_sample(pipe, row, manifest, 'float16', cross_map_kind='conditional')
        recorder = common._RECORDERS[-1]
        try:
            resolutions = {n: r for r, modules in sample.modules_by_resolution.items() for n in modules}
            seed = common.ca_seed(recorder, ca_modules, resolutions, ca_spec, args.device)
            calls = recorder.sa_qk[selected]
            if len(calls) != 31 or any(q.shape[0] != 8 for q, k in calls):
                raise ValueError('expected frozen 31 calls x 8 heads')
            maps = locked_sg2_maps(calls, seed, spec['gamma']).reshape(3, 64, 64)
            recorder.sa_qk.clear()
            counts, ap = cv.score_maps(maps, sample.target_mask, 'minmax', thresholds)
            crf = np.concatenate([transfer.crf_counts(m, sample.target_mask, sample.generated_image, args.crf_site) for m in maps])
            writer.add(row['image_id'], classname, np.concatenate((counts, crf)), np.concatenate((ap, [np.nan]*3)))
            print(f"[{args.dataset}:{classname}] {row['image_id']} {len(writer.completed)}/{len(rows)}", flush=True)
            del seed, maps, counts, crf, ap, calls
        finally:
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            sample.clear()
            del sample, recorder
            common._RECORDERS.clear()
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_samples is None:
        if writer.completed != {r['image_id'] for r in all_rows}:
            raise ValueError('incomplete or foreign image ids')
        writer.finish()


if __name__ == '__main__':
    main()
