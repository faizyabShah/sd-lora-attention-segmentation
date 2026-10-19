#!/usr/bin/env python3
"""SG13 target-conditioned region inference on a streaming class shard."""
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
from src.segmentation.target_region_inference import observation_statistics, region_quantities, score_regions, oracle_diagnostic
from src.segmentation.attention_regions import spectral_partition
from src.segmentation.region_diagnostics import RegionDiagnosticWriter

SPEC = ROOT/'configs/sg13_target_region_inference.json'
OUT = ROOT/'outputs/segmentation/controlled_attention_cv/sg13_target_region_inference'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('voc', 'coco'), required=True)
    parser.add_argument('--class-shard', type=int, choices=range(20), required=True)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--model', default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
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
    names = tuple(spec['configurations'])
    thresholds = tuple(round(i*.005, 3) for i in range(201))
    folds = cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if args.dataset == 'voc' else {r['image_id']: 0 for r in all_rows}
    settings = {'package': 'sg13_target_region_inference_v1', 'spec': spec, 'dataset': args.dataset,
                'ca_seed': ca_spec, 'selected_sa_module': selected, 'model': args.model,
                'model_dtype': 'float32', 'qk_storage_dtype': 'float16'}
    path = args.output_dir/args.dataset/'shards'/f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    writer = cv.ShardWriter(path, names, folds, settings, thresholds)
    diag = RegionDiagnosticWriter(path.with_suffix('.regions.npz'), spec)
    if writer.completed-set(diag.records):
        raise ValueError('metric records missing diagnostic sidecars')
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
            graph, propagated, two, consensus = observation_statistics(calls, seed)
            labels = spectral_partition(graph, 32)
            del graph, calls
            recorder.sa_qk.clear()
            quantities, scores, area = region_quantities(labels, seed, propagated, consensus)
            legacy = quantities[:, 1][labels]
            sg8c = torch.where(consensus >= .8, propagated, seed.reshape(-1))
            controls = torch.stack((propagated, two, legacy, sg8c)).reshape(4, 64, 64)
            counts, ap = cv.score_maps(controls, sample.target_mask, 'minmax', thresholds)
            region_counts = score_regions(labels, scores, sample.target_mask, thresholds)
            fraction, oracle_counts = oracle_diagnostic(labels, sample.target_mask)
            region_ap = cv.average_precision_batch(scores[:, labels], cv.target_at_size(sample.target_mask, 64)).numpy()
            diag.add(row['image_id'], quantities, scores, area, fraction, oracle_counts)
            if args.dataset == 'voc' and classname in spec['visual_classes'] and row['image_id'] == all_rows[0]['image_id']:
                cache = args.output_dir/'visual_cache'/f'{args.class_shard:02d}.npz'
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_name(cache.name+'.tmp.npz')
                np.savez_compressed(temporary, image=sample.generated_image, target=sample.target_mask,
                                   labels=labels.cpu().numpy(), seed=seed.cpu().numpy(), propagated=propagated.cpu().numpy(),
                                   consensus=consensus.cpu().numpy(), scores=scores.cpu().numpy(),
                                   oracle_region_positive=((fraction >= .5)&(area > 0)).cpu().numpy(),
                                   classname=np.array(classname), image_id=np.array(row['image_id']))
                temporary.replace(cache)
            writer.add(row['image_id'], classname, np.concatenate((counts, region_counts)), np.concatenate((ap, region_ap)))
            print(f"[{args.dataset}:{classname}] {row['image_id']} {len(writer.completed)}/{len(rows)}", flush=True)
            del seed, propagated, two, consensus, labels, quantities, scores, area, controls, counts, ap, region_ap, region_counts, fraction, oracle_counts
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
        if set(diag.records) != writer.completed:
            raise ValueError('diagnostic and result image IDs differ')
        writer.finish()


if __name__ == '__main__':
    main()
