#!/usr/bin/env python3
"""Resumable SG21 per-class GPU shard; save only reduced per-image results."""
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
from src.segmentation import sg18_temporal as sg
from src.segmentation import sg21_structural as structural
from src.segmentation import temporal_head_diagnostics as diag

SPEC = ROOT/'configs/sg21_structural.json'
OUT = ROOT/'outputs/segmentation/controlled_attention_cv/sg21_structural'
SOURCE = ROOT/'outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed'
HEAD_EVENT_FIELDS = ('tp_add','fp_add','fp_gt16','useful_remove','harmful_remove')


def save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def head_events(observations: torch.Tensor, seed: torch.Tensor, truth: np.ndarray,
                ca_mask: np.ndarray, distance: np.ndarray, device: str) -> np.ndarray:
    """Fixed-0.45 observation behavior used only after VOC promotion."""
    shape = truth.shape
    events = np.zeros((len(structural.CALLS), structural.HEADS, len(HEAD_EVENT_FIELDS)), np.int64)
    far = distance > 16
    for t in range(len(structural.CALLS)):
        z = observations[t].to(device)
        masks = (lr.full_scores(z, shape) >= .45).cpu().numpy()
        added = masks & ~ca_mask
        removed = ~masks & ca_mask
        tp = added & truth
        fp = added & ~truth
        values = (tp.sum((1,2)), fp.sum((1,2)), (fp & far).sum((1,2)),
                  (removed & ~truth).sum((1,2)), (removed & truth).sum((1,2)))
        events[t] = np.stack(values, -1)
        del z, masks, added, removed, tp, fp
    return events


def process(row, dataset, pipe, ca_modules, selected, ca_spec, geometry, device, spec):
    manifest = sg18.transfer.EVAL/('ovam/voc_sim' if dataset=='voc' else 'ovam/coco_captions')
    sg18.transfer._ATTRIBUTION_PROMPT[0] = row['attribution_prompt']
    sg18.common._RECORDERS.clear()
    sample = sg18.common.live.generate_attention_sample(pipe,row,manifest,'float16',cross_map_kind='conditional')
    recorder = sg18.common._RECORDERS[-1]
    try:
        resolution = {name:size for size,names in sample.modules_by_resolution.items() for name in names}
        seed = sg18.common.ca_seed(recorder,ca_modules,resolution,ca_spec,device).reshape(-1)
        maps, weights, descriptors, observations = structural.aggregate(recorder.sa_qk[selected],seed,geometry)
        truth = np.asarray(sample.target_mask,bool)
        counts, ap = cv.score_maps(maps.reshape(len(structural.NAMES),64,64),truth,
                                   'minmax',spec['thresholds'])
        map_dict = {name:maps[i].cpu().numpy() for i,name in enumerate(structural.NAMES)}
        ca, predictions = sg.masks(map_dict,seed.cpu().numpy(),truth,
                                   {name:spec['diagnostic_threshold'] for name in structural.NAMES},
                                   spec['diagnostic_ca_threshold'])
        distance = ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
        mechanism = np.stack([sg.events(ca,predictions[name],truth,distance)[0]
                              for name in structural.NAMES])
        heads = head_events(observations,seed,truth,ca,distance,device)
        with np.load(sg18.paths(SOURCE,dataset,row['image_id'])[0],allow_pickle=False) as source:
            previous = source['maps'][1]
        drift = float(np.mean(np.abs(maps[0].cpu().numpy()-previous)))
        if counts.shape != (len(structural.NAMES),len(spec['thresholds']),4):
            raise ValueError('incorrect SG21 count shape')
        if mechanism.shape != (len(structural.NAMES),len(sg.EVENT_NAMES)):
            raise ValueError('incorrect mechanism count shape')
        if heads.shape != (11,8,len(HEAD_EVENT_FIELDS)):
            raise ValueError('incorrect head event shape')
        return {'counts':counts, 'ap':ap, 'mechanism':mechanism,
                'weights':weights.numpy().astype(np.float32),
                'descriptors':descriptors.numpy().astype(np.float32),
                'head_events':heads, 't1_replay_mae':np.array(drift,np.float32)}
    finally:
        recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
        sg18.common._RECORDERS.clear();del sample,recorder
        gc.collect();torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('voc','coco'),required=True)
    parser.add_argument('--class-shard',type=int,choices=range(20),required=True)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--model',default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text())
    frozen = json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    locked = json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    if spec['names'] != list(structural.NAMES) or spec['calls_inclusive'] != [10,20] or spec['heads'] != 8:
        raise ValueError('SG21 locked candidate contract changed')
    if tuple(spec['thresholds']) != tuple(locked['thresholds']) or locked['sa']['module_index'] != 15 or not frozen['frozen']:
        raise ValueError('SG21 frozen base contract changed')
    if not json.loads((SOURCE/'completed.json').read_text())['all_required_stages_complete']:
        raise ValueError('SG18 source incomplete')
    classname, all_rows = (sg18.transfer.voc_rows if args.dataset=='voc' else sg18.transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    expected = {'spec':spec,'dataset':args.dataset,'class_shard':args.class_shard,
                'class':classname,'model':args.model}
    base = args.output_dir/'cache'/args.dataset/f'class_{args.class_shard:02d}'
    pipe, ca_modules, selected, ca_spec = sg18.make_pipe(args.model,args.device,locked,frozen)
    geometry = diag.Geometry(args.device)
    for index,row in enumerate(rows,1):
        path = base/(sg18.key(row['image_id'])+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                meta = json.loads(str(old['meta']))
                if meta != {**expected,'image_id':row['image_id']}:
                    raise ValueError(f'{path}: incompatible SG21 checkpoint')
            continue
        arrays = process(row,args.dataset,pipe,ca_modules,selected,ca_spec,geometry,args.device,spec)
        save(path,meta=np.array(json.dumps({**expected,'image_id':row['image_id']},sort_keys=True)),**arrays)
        print(f'[{args.dataset}:{classname}] {index}/{len(rows)} {row["image_id"]}',flush=True)
    if args.max_samples is None:
        found = {p.stem for p in base.glob('*.npz')}
        wanted = {sg18.key(r['image_id']) for r in all_rows}
        if found != wanted:
            raise ValueError(f'incomplete/foreign SG21 class shard {args.dataset}/{args.class_shard}')
        print(f'completed {args.dataset} class {args.class_shard}: {len(found)}',flush=True)


if __name__=='__main__':
    main()
