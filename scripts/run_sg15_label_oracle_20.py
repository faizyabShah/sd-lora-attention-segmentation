#!/usr/bin/env python3
"""Generate 20 fixed VOC examples and compare locked SG15 labels with oracle labels."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import subprocess
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

SPEC=ROOT/'configs/sg15_label_oracle_20.json'
OUT=ROOT/'outputs/segmentation/analysis/sg15_label_oracle_20'
ENV_PLOT=Path('/home/shafa/miniconda3/envs/ovam_sdxl/bin/python')


def selected_rows():
    result=[]
    for shard in range(20):
        classname,rows=transfer.voc_rows(shard)
        if not rows:raise ValueError(f'no annotated VOC row for {classname}')
        result.append((classname,rows[0]))
    if len({r['image_id'] for _,r in result})!=20:raise ValueError('expected 20 unique images')
    return result


def metric(mask,truth):
    tp=np.count_nonzero(mask&truth);union=np.count_nonzero(mask|truth)
    return float(tp/union) if union else 1.


def compare(labels,truth,prior_masks):
    gt_fraction,area=lr.region_fractions(labels,truth)
    gt_positive=(gt_fraction>=.5)&(area>0)
    oracle=gt_positive[labels]
    current=[];snaps=[];rows=[]
    for before in prior_masks:
        majority,fraction,_=lr.refine_regions(labels,before,'A050')
        snap,_,_=lr.refine_regions(labels,before,'B')
        pred_region=(fraction>=.5)&(area>0)
        changed=(majority!=oracle)
        rows.append({'prior_iou':metric(before,truth),'current_A050_iou':metric(majority,truth),
                     'current_B_iou':metric(snap,truth),'oracle_iou':metric(oracle,truth),
                     'A050_oracle_disagreement_pixels':int(changed.sum()),
                     'B_oracle_disagreement_pixels':int((snap!=oracle).sum()),
                     'oracle_positive_regions':int(gt_positive.sum()),
                     'A050_oracle_false_positive_regions':int((pred_region&~gt_positive).sum()),
                     'A050_oracle_missed_regions':int((~pred_region&gt_positive).sum())})
        current.append(majority);snaps.append(snap)
    return np.stack(current),np.stack(snaps),oracle,rows


def render(cache,figure):
    env=dict(os.environ,MPLCONFIGDIR=str(cache.parent/'matplotlib_config'))
    subprocess.run([str(ENV_PLOT),str(ROOT/'scripts/plot_sg15_label_oracle_20.py'),
                    '--cache',str(cache),'--output',str(figure)],check=True,env=env)
    if not figure.is_file() or figure.stat().st_size<1000:raise RuntimeError('visual sheet missing')


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--max-classes',type=int,choices=range(1,21),default=20)
    parser.add_argument('--model',default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    locked=json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    sg15=json.loads((ROOT/'outputs/segmentation/controlled_attention_cv/sg15/summary.json').read_text())
    names=tuple(lr.PRIORS)
    if list(names)!=spec['priors']:raise ValueError('source priors differ from locked study')
    thresholds=np.array([sg15['cells'][n]['voc_threshold'] for n in names])
    plan={'spec':spec,'locked':locked,'model':args.model,'names':list(names),'thresholds':thresholds.tolist()}
    out=args.output_dir
    state=out/'results.json';records={}
    if state.exists():
        loaded=json.loads(state.read_text())
        if loaded['plan']!=plan:raise ValueError('resume plan changed')
        records=loaded['records']
    selected=selected_rows()[:args.max_classes]
    if not selected:raise ValueError('empty selection')
    if all(r['image_id'] in records and (out/'figures'/f'{i:02d}_{r["classname"].replace(" ","_")}.png').is_file()
           for i,(_,r) in enumerate(selected)):
        print('All requested example sheets already complete',flush=True)
        return
    frozen=json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not frozen.get('frozen'):raise ValueError('CA seed is not frozen')
    pipe=StableDiffusionPipeline.from_pretrained(args.model,torch_dtype=torch.float32,local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True);transfer._PIPE=pipe
    modules=[n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca_modules=tuple(sorted(n for n in modules if n.endswith('attn2')))
    selected_sa=tuple(sorted(n for n in modules if n.endswith('attn1')))[locked['sa']['module_index']]
    WordObservationRecorder.selected_sa=selected_sa
    ca_spec=common.resolve_seed_spec(frozen,ca_modules,locked['sa']['calls'])
    manifest=transfer.EVAL/'ovam/voc_sim'
    for i,(classname,row) in enumerate(selected):
        image_id=row['image_id'];figure=out/'figures'/f'{i:02d}_{classname.replace(" ","_")}.png'
        if image_id in records and figure.is_file():continue
        cache=out/'temporary'/f'{i:02d}.npz'
        if cache.exists() and image_id in records:
            render(cache,figure);cache.unlink();continue
        transfer._ATTRIBUTION_PROMPT[0]=row['attribution_prompt'];common._RECORDERS.clear()
        sample=common.live.generate_attention_sample(pipe,row,manifest,'float16',cross_map_kind='conditional')
        recorder=common._RECORDERS[-1]
        try:
            resolution={n:r for r,mods in sample.modules_by_resolution.items() for n in mods}
            seed=common.ca_seed(recorder,ca_modules,resolution,ca_spec,args.device)
            calls=recorder.sa_qk[selected_sa]
            if len(calls)!=31 or any(q.shape[0]!=8 for q,k in calls):raise ValueError('unexpected SA observations')
            obs,one,two,graph=lr.collect_observations(calls,seed,need_two=True,need_graph=True)
            maps,details=lr.selected_consensus_maps(obs,seed,locked)
            torch.testing.assert_close(maps[lr.REF],one,rtol=1e-5,atol=2e-6)
            maps[lr.P1]=two
            if i==0:
                original_one,original_two=transfer.propagate(recorder,selected_sa,seed,args.device)
                torch.testing.assert_close(one,original_one.reshape(-1),rtol=1e-5,atol=2e-6)
                torch.testing.assert_close(two,original_two.reshape(-1),rtol=1e-5,atol=2e-6)
            partition=spectral_partition(graph,32)
            truth=np.asarray(sample.target_mask,bool)
            labels=expanded_labels(partition,truth.shape).cpu().numpy()
            full=lr.full_scores(torch.stack([maps[name] for name in names]),truth.shape).cpu().numpy()
            masks=full>=thresholds[:,None,None]
            current,snaps,oracle,comparisons=compare(labels,truth,masks)
            record={'image_id':image_id,'class':classname,'prompt':row['prompt'],'seed':row['seed'],
                    'figure':figure.name,'comparisons':dict(zip(names,comparisons)),
                    'source_same_generation_error_max':float((one-maps[lr.REF]).abs().max()),
                    'rank_count':len(details['selected_indices'])}
            cache.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache.with_name(cache.name+'.tmp.npz')
            np.savez_compressed(tmp,image=sample.generated_image,truth=truth,labels=labels,
                                prior=masks,majority=current,snapped=snaps,oracle=oracle,
                                names=np.asarray(names),thresholds=thresholds,image_id=np.array(image_id),classname=np.array(classname))
            tmp.replace(cache)
            records[image_id]=record
            atomic_json(state,{'plan':plan,'records':records})
            render(cache,figure)
            cache.unlink()
            print(f'[{i+1}/{len(selected)}] {image_id} oracle IoU {metric(oracle,truth):.3f}',flush=True)
            del obs,one,two,graph,partition,seed,full,maps,details,labels,masks,current,snaps,oracle
        finally:
            recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
            common._RECORDERS.clear();del sample,recorder
            gc.collect();torch.cuda.empty_cache()
    # A compact comparison table uses the same fixed 20 images, no selection.
    csvpath=out/'per_image_iou.csv'
    with csvpath.open('w',newline='') as handle:
        fields=['class','image_id','prior','prior_iou','current_A050_iou','current_B_iou','oracle_iou',
                'A050_oracle_disagreement_pixels','B_oracle_disagreement_pixels','oracle_positive_regions',
                'A050_oracle_false_positive_regions','A050_oracle_missed_regions']
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for _,row in selected:
            for name,comparison in records[row['image_id']]['comparisons'].items():
                writer.writerow({'class':row['classname'],'image_id':row['image_id'],'prior':name,**comparison})
    if len(list((out/'figures').glob('*.png')))!=args.max_classes:raise ValueError('figure count differs from selected classes')
    if len(records)<args.max_classes:raise ValueError('incomplete result records')
    print(f'Completed {args.max_classes} fixed-class SG15 current/oracle sheets at {out}',flush=True)

if __name__=='__main__':main()
