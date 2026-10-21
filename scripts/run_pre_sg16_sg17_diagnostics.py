#!/usr/bin/env python3
"""Stream both prerequisite diagnostics for one VOC/COCO class shard."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_sg11_sg12_observation_reliability import WordObservationRecorder,transfer,common
from src.segmentation import locked_refinements as lr
from src.segmentation.attention_regions import spectral_partition
from src.segmentation.target_region_inference import expanded_labels
from src.segmentation.pre_sg_diagnostics import collect,region_records,pixel_features,pixel_histograms

SPEC=ROOT/'configs/pre_sg16_sg17_diagnostics.json'
OUT=ROOT/'outputs/segmentation/analysis/pre_sg16_sg17'


def image_path(base,image_id):
    return base/(hashlib.sha256(image_id.encode()).hexdigest()[:20]+'.npz')


def save(path,meta,regions,region_image,pixel_image,hist):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(temp,meta=np.array(json.dumps(meta,sort_keys=True)),
                        regions=np.array(json.dumps(regions)),
                        region_image=np.array(json.dumps(region_image)),
                        pixel_image=np.array(json.dumps(pixel_image)),hist=hist)
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('voc','coco'),required=True)
    parser.add_argument('--class-shard',type=int,choices=range(20),required=True)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--model',default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    spec=json.loads(SPEC.read_text());locked=json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    thresholds=json.loads((ROOT/'outputs/segmentation/controlled_attention_cv/sg15/summary.json').read_text())
    frozen=json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not frozen.get('frozen') or spec['observation_count']!=locked['sa']['observations']:
        raise ValueError('frozen experimental contract changed')
    names=lr.PRIORS
    prior_threshold=np.array([thresholds['cells'][n]['voc_threshold'] for n in names])
    classname,allrows=(transfer.voc_rows if args.dataset=='voc' else transfer.coco_rows)(args.class_shard)
    rows=allrows[:args.max_samples] if args.max_samples else allrows
    base=args.output_dir/args.dataset/'shards'/f'class_{args.class_shard:02d}'
    expected={'spec':spec,'locked':locked,'model':args.model,'dataset':args.dataset,
              'class_shard':args.class_shard,'class':classname,'prior_thresholds':prior_threshold.tolist()}
    manifest=transfer.EVAL/('ovam/voc_sim' if args.dataset=='voc' else 'ovam/coco_captions')
    pipe=StableDiffusionPipeline.from_pretrained(args.model,torch_dtype=torch.float32,local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True);transfer._PIPE=pipe
    modules=[n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca_modules=tuple(sorted(n for n in modules if n.endswith('attn2')))
    selected=tuple(sorted(n for n in modules if n.endswith('attn1')))[locked['sa']['module_index']]
    WordObservationRecorder.selected_sa=selected
    ca_spec=common.resolve_seed_spec(frozen,ca_modules,locked['sa']['calls'])
    for index,row in enumerate(rows):
        path=image_path(base,row['image_id'])
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                meta=json.loads(str(d['meta']))
                if meta['settings']!=expected or meta['image_id']!=row['image_id']:
                    raise ValueError(f'{path}: incompatible checkpoint')
            continue
        transfer._ATTRIBUTION_PROMPT[0]=row['attribution_prompt'];common._RECORDERS.clear()
        sample=common.live.generate_attention_sample(pipe,row,manifest,'float16',cross_map_kind='conditional')
        recorder=common._RECORDERS[-1]
        try:
            resolution={n:r for r,mods in sample.modules_by_resolution.items() for n in mods}
            seed=common.ca_seed(recorder,ca_modules,resolution,ca_spec,args.device)
            calls=recorder.sa_qk[selected]
            if len(calls)!=31 or any(q.shape[0]!=8 for q,k in calls):
                raise ValueError('unexpected SA observations')
            first,second,graph=collect(calls,seed)
            one=first.mean(0);two=second.mean(0)
            maps,details=lr.selected_consensus_maps(first,seed,locked)
            torch.testing.assert_close(one,maps[lr.REF],rtol=1e-5,atol=2e-6)
            if index==0:
                oldone,oldtwo=transfer.propagate(recorder,selected,seed,args.device)
                torch.testing.assert_close(one,oldone.reshape(-1),rtol=1e-5,atol=2e-6)
                torch.testing.assert_close(two,oldtwo.reshape(-1),rtol=1e-5,atol=2e-6)
            labels=spectral_partition(graph,32)
            truth=np.asarray(sample.target_mask,bool)
            full_labels=expanded_labels(labels,truth.shape).cpu().numpy()
            maps[lr.P1]=two
            full_scores=lr.full_scores(torch.stack([maps[n] for n in names]),truth.shape).cpu().numpy()
            masks=full_scores>=prior_threshold[:,None,None]
            rr,ri=region_records(labels.cpu().numpy(),full_labels,truth,masks,
                                 seed.cpu().numpy(),first.cpu().numpy(),details['selected_indices'].cpu().numpy())
            pf=pixel_features(seed,first,second,details['reliability'],details['selected_indices'])
            hist,pi=pixel_histograms(pf,one,two,truth,bins=spec['pixel_histogram_bins'],
                                     threshold=spec['second_pass_threshold'])
            if sum(x['region_area_pixels'] for x in rr)!=truth.size:
                raise ValueError('SPEC32 regions do not cover annotation pixels')
            if pi['useful']+pi['harmful']!=pi['additions']+pi['removals']:
                raise ValueError('second-pass event partition failed')
            meta={'settings':expected,'image_id':row['image_id'],'class':classname,
                  'seed':row['seed'],'prompt':row['prompt']}
            save(path,meta,rr,ri,pi,hist)
            print(f'[{args.dataset}:{classname}] {row["image_id"]} {index+1}/{len(rows)} '
                  f'regions={len(rr)} changes={pi["useful"]+pi["harmful"]}',flush=True)
            del seed,first,second,graph,maps,details,labels,full_labels,full_scores,masks,pf,hist,rr,ri,pi
        finally:
            recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
            common._RECORDERS.clear();del sample,recorder
            gc.collect();torch.cuda.empty_cache()
    if args.max_samples is None:
        completed={}
        for p in base.glob('*.npz'):
            with np.load(p,allow_pickle=False) as d:
                m=json.loads(str(d['meta']))
            if m['settings']!=expected:raise ValueError(f'{p}: foreign checkpoint')
            completed[m['image_id']]=p
        if set(completed)!={r['image_id'] for r in allrows}:
            raise ValueError(f'{args.dataset} shard {args.class_shard}: incomplete/foreign images')
        print(f'Complete shard {args.dataset} {args.class_shard}: {len(completed)} images',flush=True)

if __name__=='__main__':main()
