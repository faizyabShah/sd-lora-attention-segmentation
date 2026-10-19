#!/usr/bin/env python3
"""Stream SG11/SG12 and SG-Diag on one class; no optimized-token path."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts.run_coco_cap_transfer as transfer
import scripts.run_sa_package_b as common
from src.segmentation import controlled_cv as cv
from src.segmentation.observation_reliability import configuration_names, reliability_maps, propagate_qk
from src.segmentation.propagation_diagnostics import ObservationDiagnosticWriter, change_histograms

SPEC = ROOT / 'configs/sg11_sg12_observation_reliability.json'
OUT = ROOT / 'outputs/segmentation/controlled_attention_cv/sg11_sg12_observation_reliability'


class WordObservationRecorder(transfer.TransferRecorder):
    """Same padded word CA as transfer; retain QK only for the frozen SA module."""
    selected_sa = ''

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key)
        if kind != 'cross' and module_name != self.selected_sa:
            self.sa_qk.pop(module_name, None)


common.live.CompactAttentionRecorder = WordObservationRecorder


def save_visual_cache(path, sample, seed, maps, names, image_id, classname):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(temporary, image=sample.generated_image, target=sample.target_mask,
                       seed=seed.cpu().numpy(), maps=maps.cpu().numpy(), names=np.asarray(names),
                       image_id=np.array(image_id), classname=np.array(classname))
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('voc', 'coco'), required=True)
    parser.add_argument('--class-shard', type=int, choices=range(20), required=True)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    seed_json = json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not seed_json.get('frozen'):
        raise ValueError('CA must be frozen')
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
    names, thresholds = configuration_names(spec), tuple(spec['thresholds'])
    folds = cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if args.dataset == 'voc' else {r['image_id']: 0 for r in all_rows}
    settings = {'package': 'sg11_sg12_observation_reliability_v1', 'spec': spec, 'dataset': args.dataset,
                'ca_seed': ca_spec, 'selected_sa_module': selected, 'model': args.model,
                'model_dtype': 'float32', 'qk_storage_dtype': 'float16'}
    base = args.output_dir/args.dataset/'shards'/f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    writer = cv.ShardWriter(base.with_suffix('.npz'), names, folds, settings, thresholds)
    diag = ObservationDiagnosticWriter(base.with_suffix('.diagnostics.npz'), spec)
    if writer.completed - set(diag.ids):
        raise RuntimeError('metric checkpoint contains images missing diagnostics')
    for row in rows:
        if row['image_id'] in writer.completed:
            continue
        transfer._ATTRIBUTION_PROMPT[0] = row['attribution_prompt']
        common._RECORDERS.clear()
        sample = common.live.generate_attention_sample(pipe, row, manifest, 'float16', cross_map_kind='conditional')
        recorder = common._RECORDERS[-1]
        try:
            resolutions = {name: resolution for resolution, modules in sample.modules_by_resolution.items() for name in modules}
            seed = common.ca_seed(recorder, ca_modules, resolutions, ca_spec, args.device)
            if seed.numel() != 64*64:
                raise RuntimeError('unexpected seed resolution')
            observations, reference = propagate_qk(recorder.sa_qk[selected], seed, spec['sa'], args.device)
            recorder.sa_qk.clear()
            maps, diagnostics = reliability_maps(observations, seed, spec)
            if float((maps[0]-reference).abs().max()) > 2e-6:
                raise RuntimeError('one-pass reference algebra failed')
            counts, ap = cv.score_maps(maps.reshape(-1, 64, 64), sample.target_mask, 'minmax', thresholds)
            hist, totals = change_histograms(seed, diagnostics, sample.target_mask, spec['diagnostic'])
            if args.dataset == 'voc' and classname in spec['visual_classes'] and row['image_id'] == all_rows[0]['image_id']:
                save_visual_cache(args.output_dir/'visual_cache'/f'{args.class_shard:02d}.npz',
                                  sample, seed, maps, names, row['image_id'], classname)
            diag.add(row['image_id'], hist, totals, diagnostics)
            writer.add(row['image_id'], classname, counts, ap)
            print(f"[{args.dataset}:{classname}] {row['image_id']} {len(writer.completed)}/{len(rows)}", flush=True)
            del maps, diagnostics, observations, reference, counts, ap, seed, hist, totals
        finally:
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            sample.clear()
            del sample, recorder
            common._RECORDERS.clear()
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_samples is None:
        if writer.completed != {r['image_id'] for r in all_rows} or set(diag.ids) != writer.completed:
            raise RuntimeError('incomplete or foreign image ids')
        writer.finish()


if __name__ == '__main__':
    main()
