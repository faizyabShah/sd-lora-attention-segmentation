#!/usr/bin/env python3
"""One resumable VOC/COCO class shard of temporal/head SA diagnostics."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_sg11_sg12_observation_reliability import WordObservationRecorder,transfer,common
from src.segmentation import temporal_head_diagnostics as diag

SPEC=ROOT/'configs/pre_sg18_sg21_diagnostics.json'
OUT=ROOT/'outputs/segmentation/analysis/pre_sg18_sg21'


def image_path(base,image_id):
    return base/(hashlib.sha256(image_id.encode()).hexdigest()[:20]+'.npz')


def save(path,meta,features,descriptors,head_counts,call_counts,stage_counts,
         call_support,hist,reference_counts):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(temporary,meta=np.array(json.dumps(meta,sort_keys=True)),
                        descriptors=descriptors.astype(np.float32),
                        head_counts=head_counts.astype(np.int64),
                        call_counts=call_counts.astype(np.int64),
                        stage_counts=stage_counts.astype(np.int64),
                        call_support=call_support.astype(np.float64),
                        hist=hist.astype(np.int32),
                        reference_counts=reference_counts.astype(np.int64))
    temporary.replace(path)


def process(seed,observations,descriptors,truth,spec):
    if observations.shape!=(31,8,4096):raise ValueError('incorrect observation grid')
    truth=np.asarray(truth,bool);shape=truth.shape;t=spec['threshold']
    ca=diag.binary_masks(seed.reshape(1,4096),shape,t)[0]
    ordinary=observations.mean((0,1))
    ref=diag.binary_masks(ordinary.reshape(1,4096),shape,t)[0]
    reference=diag.event_context(ca,ref,truth)
    features=diag.temporal_features(observations,seed)
    hist=diag.feature_histograms(features,reference,shape,spec['pixel_histogram_bins'])
    head_counts=np.zeros((31,8,len(diag.COUNTS)),np.int64)
    call_counts=np.zeros((31,len(diag.COUNTS)),np.int64)
    call_support=np.zeros((31,2),np.float64)
    seed_flat=seed.reshape(-1).float()
    for call in range(31):
        masks=diag.binary_masks(observations[call],shape,t)
        for head in range(8):
            head_counts[call,head]=diag.event_counts(ca,masks[head],truth,reference)
        call_map=observations[call].mean(0)
        call_mask=diag.binary_masks(call_map.reshape(1,4096),shape,t)[0]
        call_counts[call]=diag.event_counts(ca,call_mask,truth,reference)
        support=F.interpolate((call_map>seed_flat).reshape(1,1,64,64).float(),
                              size=shape,mode='bilinear',align_corners=False)[0,0].numpy()
        call_support[call]=[float(support[truth].mean()) if truth.any() else np.nan,
                            float(support[~truth].mean()) if (~truth).any() else np.nan]
    stage_counts=np.zeros((3,len(diag.COUNTS)),np.int64)
    for i,(start,stop) in enumerate(spec['time_stages'].values()):
        stage=observations[start:stop].mean((0,1))
        mask=diag.binary_masks(stage.reshape(1,4096),shape,t)[0]
        stage_counts[i]=diag.event_counts(ca,mask,truth,reference)
    reference_counts=diag.event_counts(ca,ref,truth)
    if not np.array_equal(hist[:,0,:,].sum(-1)[:,0],
                          [reference_counts[0]]*4+[reference_counts[0]+reference_counts[5]]):
        raise ValueError('temporal positive histogram event totals differ')
    if not np.array_equal(hist[:,1,:,].sum(-1)[:,0],
                          [reference_counts[1],reference_counts[2],reference_counts[3],
                           reference_counts[4],reference_counts[1]+reference_counts[6]]):
        raise ValueError('temporal negative histogram event totals differ')
    return features,head_counts,call_counts,stage_counts,call_support,hist,reference_counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('voc','coco'),required=True)
    p.add_argument('--class-shard',type=int,choices=range(20),required=True)
    p.add_argument('--max-samples',type=int)
    p.add_argument('--output-dir',type=Path,default=OUT)
    p.add_argument('--model',default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    p.add_argument('--device',default='cuda')
    args=p.parse_args();spec=json.loads(SPEC.read_text())
    frozen=json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    locked=json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    if not frozen['frozen'] or locked['sa']['module_index']!=spec['sa_module_index']:
        raise ValueError('frozen SA/CA contract changed')
    classname,allrows=(transfer.voc_rows if args.dataset=='voc' else transfer.coco_rows)(args.class_shard)
    rows=allrows[:args.max_samples] if args.max_samples else allrows
    base=args.output_dir/args.dataset/'shards'/f'class_{args.class_shard:02d}'
    expected={'spec':spec,'dataset':args.dataset,'class_shard':args.class_shard,
              'class':classname,'model':args.model}
    manifest=transfer.EVAL/('ovam/voc_sim' if args.dataset=='voc' else 'ovam/coco_captions')
    pipe=StableDiffusionPipeline.from_pretrained(args.model,torch_dtype=torch.float32,
                                                  local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True);transfer._PIPE=pipe
    modules=[n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca_modules=tuple(sorted(n for n in modules if n.endswith('attn2')))
    selected=tuple(sorted(n for n in modules if n.endswith('attn1')))[spec['sa_module_index']]
    WordObservationRecorder.selected_sa=selected
    ca_spec=common.resolve_seed_spec(frozen,ca_modules,locked['sa']['calls'])
    geometry=diag.Geometry(args.device)
    for index,row in enumerate(rows):
        path=image_path(base,row['image_id'])
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                m=json.loads(str(d['meta']))
                if m['settings']!=expected or m['image_id']!=row['image_id']:
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
                raise ValueError('expected 31 calls x 8 heads')
            observations,descriptors=diag.observations(calls,seed,geometry)
            if index==0:
                old,_=transfer.propagate(recorder,selected,seed,args.device)
                torch.testing.assert_close(observations.mean((0,1)),old.reshape(-1).cpu(),
                                           rtol=1e-5,atol=2e-6)
            truth=np.asarray(sample.target_mask,bool)
            result=process(seed.cpu(),observations,descriptors,truth,spec)
            meta={'settings':expected,'image_id':row['image_id'],'class':classname,
                  'seed':row['seed'],'prompt':row['prompt']}
            save(path,meta,*result[:1],descriptors,*result[1:])
            print(f'[{args.dataset}:{classname}] {index+1}/{len(rows)} {row["image_id"]}',flush=True)
            del seed,observations,descriptors,result
        finally:
            recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
            common._RECORDERS.clear();del sample,recorder
            gc.collect();torch.cuda.empty_cache()
    if args.max_samples is None:
        existing={}
        for path in base.glob('*.npz'):
            with np.load(path,allow_pickle=False) as d:m=json.loads(str(d['meta']))
            if m['settings']!=expected:raise ValueError(f'{path}: foreign checkpoint')
            existing[m['image_id']]=path
        if set(existing)!={r['image_id'] for r in allrows}:
            raise ValueError('incomplete class shard')
        print(f'Completed {args.dataset} class {args.class_shard}: {len(existing)}',flush=True)

if __name__=='__main__':main()
