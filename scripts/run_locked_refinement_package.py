#!/usr/bin/env python3
"""Stream a locked SG12+CRF, SG14 or SG15 class shard; discard native attention."""
from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from diffusers import StableDiffusionPipeline

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_sg11_sg12_observation_reliability import WordObservationRecorder,transfer,common
from src.segmentation import controlled_cv as cv
from src.segmentation import locked_refinements as lr
from src.segmentation.observation_reliability import reliability_maps
from src.segmentation.attention_regions import spectral_partition
from src.segmentation.target_region_inference import expanded_labels,oracle_diagnostic
from src.segmentation.experiment_records import NumericSidecar
from src.segmentation.ovam_full import _import_densecrf,apply_ovam_densecrf,OVAM_CRF_PARAMETERS
from scripts.narmyn.attention.an1_spatial import transition

SPEC=ROOT/'configs/sg12_crf_sg14_sg15.json'
BASE=ROOT/'outputs/segmentation/controlled_attention_cv'


def timer(device):
    lr.synchronize(device)
    return time.perf_counter()


def visual_cache(path,sample,labels,maps,image_id,classname,extreme=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(temp,image=sample.generated_image,target=sample.target_mask,
                        labels=labels.cpu().numpy(),maps=torch.stack([maps[p] for p in lr.PRIORS]).cpu().numpy(),
                        names=np.asarray(lr.PRIORS),image_id=np.array(image_id),classname=np.array(classname),
                        extreme=np.asarray(extreme if extreme is not None else [0,0,0]))
    temp.replace(path)


def crf_evaluate(maps,sample,thresholds,site):
    truth=np.asarray(sample.target_mask,bool)
    scores=lr.full_scores(torch.stack((maps[lr.P1],maps[lr.SG12])),truth.shape).cpu().numpy()
    counts=np.zeros((2,len(thresholds),4),np.int64)
    transitions=np.zeros((len(thresholds),4),np.int64)
    spatial=np.zeros((len(thresholds),8),np.int64)
    times=np.zeros((2,len(thresholds)))
    dgt=ndimage.distance_transform_edt(~truth)
    boundary=ndimage.binary_dilation(truth,iterations=4)^ndimage.binary_erosion(truth,iterations=4)
    for i in range(2):
        for j,t in enumerate(thresholds):
            before=scores[i]>=t
            start=time.perf_counter();after=apply_ovam_densecrf(sample.generated_image,before,site)
            times[i,j]=time.perf_counter()-start
            counts[i,j]=lr.mask_counts(after,truth)
            if i==1:
                transitions[j]=lr.changes(before,after,truth)
                dprev=ndimage.distance_transform_edt(~before) if before.any() else np.full(truth.shape,1e6)
                an1=transition(before,after,truth,dprev,dgt)
                # New FP: attached, detached, >16px from GT, >64px from GT;
                # errors before/after inside/outside a fixed 4px GT boundary band.
                spatial[j]=[an1['new_fp_bleeding'],an1['new_fp_jumping'],
                            sum(an1['new_fp_dist_to_object'][2:]),an1['new_fp_dist_to_object'][3],
                            ((before!=truth)&boundary).sum(),((after!=truth)&boundary).sum(),
                            ((before!=truth)&~boundary).sum(),((after!=truth)&~boundary).sum()]
    return counts,{'crf_changes':transitions,'crf_spatial':spatial,'crf_seconds':times}


def region_evaluate(maps,labels,truth,thresholds):
    labels=expanded_labels(labels,truth.shape).cpu().numpy()
    scores=lr.full_scores(torch.stack([maps[p] for p in lr.PRIORS]),truth.shape).cpu().numpy()
    gt_fraction,area=lr.region_fractions(labels,truth)
    oracle=(gt_fraction>=.5)&(area>0)
    oracle_counts=lr.mask_counts(oracle[labels],truth)
    counts=[];regions=[];changes=[]
    for pi,prior in enumerate(lr.PRIORS):
        for kind in ('A025','A050','A075','B'):
            cs=[];rs=[];ds=[]
            for ti,t in enumerate(thresholds):
                before=scores[pi]>=t
                after,fraction,_=lr.refine_regions(labels,before,kind)
                cs.append(lr.mask_counts(after,truth));rs.append(lr.region_diagnostic(labels,after,truth))
                # All changes, high-FG, high-BG and mixed; majority projection
                # total is explicitly retained for every q, even in mixed regions.
                high=fraction[labels]>=.75;low=fraction[labels]<=.25
                ds.append(np.stack([lr.changes(before,after,truth,m) for m in (None,high,low,~(high|low))]))
            counts.append(cs);regions.append(rs);changes.append(ds)
    prior_regions=np.stack([[lr.region_diagnostic(labels,s>=t,truth) for t in thresholds] for s in scores])
    return np.asarray(counts),{'region_stats':np.asarray(regions),'prior_region_stats':prior_regions,
                              'region_changes':np.asarray(changes),'oracle_counts':oracle_counts,
                              'region_gt_positive':np.array(oracle.sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('sg12_crf','sg14','sg15'),required=True)
    parser.add_argument('--dataset',choices=('voc','coco'),required=True)
    parser.add_argument('--class-shard',type=int,choices=range(20),required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--model',default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--crf-site',type=Path,default=ROOT/'outputs/dependencies/simplecrf_py310')
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    out=args.output_dir or BASE/args.stage
    frozen=json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not frozen.get('frozen'):raise ValueError('CA seed is not frozen')
    classname,allrows=(transfer.voc_rows if args.dataset=='voc' else transfer.coco_rows)(args.class_shard)
    rows=allrows[:args.max_samples] if args.max_samples else allrows
    manifest=transfer.EVAL/('ovam/voc_sim' if args.dataset=='voc' else 'ovam/coco_captions')
    if args.stage=='sg12_crf':_import_densecrf(args.crf_site)
    pipe=StableDiffusionPipeline.from_pretrained(args.model,torch_dtype=torch.float32,local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True);transfer._PIPE=pipe
    processors=[n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca_modules=tuple(sorted(n for n in processors if n.endswith('attn2')))
    selected=tuple(sorted(n for n in processors if n.endswith('attn1')))[spec['sa']['module_index']]
    WordObservationRecorder.selected_sa=selected
    ca_spec=common.resolve_seed_spec(frozen,ca_modules,spec['sa']['calls'])
    folds=cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if args.dataset=='voc' else {r['image_id']:0 for r in allrows}
    names=lr.names_for(args.stage);thresholds=tuple(spec['thresholds'])
    settings={'package':'locked_refinements_v1','stage':args.stage,'spec':spec,'dataset':args.dataset,
              'ca_seed':ca_spec,'selected_sa_module':selected,'model':args.model,'model_dtype':'float32',
              'qk_storage_dtype':'float16','crf_parameters':list(OVAM_CRF_PARAMETERS)}
    path=out/args.dataset/'shards'/f'class_{args.class_shard:02d}.npz'
    writer=cv.ShardWriter(path,names,folds,settings,thresholds)
    diag=NumericSidecar(path.with_suffix('.diagnostics.npz'),settings)
    if writer.completed-set(diag.records):raise ValueError('metrics missing sidecar')
    # Retain only two running extreme visual examples per VOC shard. Their
    # scores choose illustrations only; no method settings depend on them.
    extremes={key:None for key in ('best','worst')}
    for key in extremes:
        cp=out/'visual_cache'/f'{args.class_shard:02d}_{key}.npz'
        if cp.exists():
            with np.load(cp) as d:extremes[key]=float(d['extreme'][0])
    for row in rows:
        if row['image_id'] in writer.completed:continue
        transfer._ATTRIBUTION_PROMPT[0]=row['attribution_prompt'];common._RECORDERS.clear()
        start=timer(args.device)
        sample=common.live.generate_attention_sample(pipe,row,manifest,'float16',cross_map_kind='conditional')
        generation=timer(args.device)-start;recorder=common._RECORDERS[-1]
        try:
            res={n:r for r,modules in sample.modules_by_resolution.items() for n in modules}
            start=timer(args.device);seed=common.ca_seed(recorder,ca_modules,res,ca_spec,args.device)
            seed_seconds=timer(args.device)-start;calls=recorder.sa_qk[selected]
            if len(calls)!=31 or any(q.shape[0]!=8 for q,k in calls):raise ValueError('not 31 calls x8 heads')
            truth=np.asarray(sample.target_mask,bool)
            methodtimes=np.zeros(4)
            # Isolated ordinary two-pass operator for runtime and independent
            # same-generation reference, not derived from candidate output.
            if args.stage=='sg12_crf':
                start=timer(args.device);independent_one,two=transfer.propagate(recorder,selected,seed,args.device)
                methodtimes[0]=timer(args.device)-start
            start=timer(args.device)
            obs,one,collected_two,graph=lr.collect_observations(calls,seed,need_two=args.stage=='sg15',need_graph=args.stage=='sg15')
            methodtimes[1]=timer(args.device)-start
            start=timer(args.device);maps,details=lr.selected_consensus_maps(obs,seed,spec)
            methodtimes[2]=details['ranking_seconds'];methodtimes[3]=details['selected_mean_seconds']
            torch.testing.assert_close(one,maps[lr.REF],rtol=1e-5,atol=2e-6)
            reference_error=float((one-maps[lr.REF]).abs().max())
            if args.stage=='sg12_crf':
                torch.testing.assert_close(one,independent_one.reshape(-1),rtol=1e-5,atol=2e-6)
                reference_error=max(reference_error,float((one-independent_one.reshape(-1)).abs().max()))
                maps[lr.P1]=two.reshape(-1)
            elif args.stage=='sg15':maps[lr.P1]=collected_two
            # Smoke independently reproduces the frozen parents using their
            # original implementation, including top50 exact selected identity.
            parent_error=0.
            if args.max_samples:
                oldspec=json.loads((ROOT/'configs/sg11_sg12_observation_reliability.json').read_text())
                oldmaps,olddetails=reliability_maps(obs,seed,oldspec)
                from src.segmentation.observation_reliability import configuration_names
                oldnames=configuration_names(oldspec)
                for new,old in ((lr.REF,'W/REF'),(lr.SG11,'W/SG11/trim20/rho0p80'),(lr.SG12,'W/SG12/top50')):
                    torch.testing.assert_close(maps[new],oldmaps[oldnames.index(old)],rtol=1e-5,atol=2e-6)
                    parent_error=max(parent_error,float((maps[new]-oldmaps[oldnames.index(old)]).abs().max()))
                expected=torch.where(olddetails['selected'][0])[0].sort().values
                if not torch.equal(details['selected_indices'].sort().values,expected):raise ValueError('SG12 selection differs')
                del oldmaps,olddetails
            diagnostics={'generation_seed_seconds':np.array([generation,seed_seconds]),'method_seconds':methodtimes,
                         'audit_errors':np.array([reference_error,parent_error]),
                         'selected_indices':details['selected_indices'].cpu().numpy(),
                         'reliability':details['reliability'].cpu().numpy()}
            del calls;recorder.sa_qk.clear()
            if args.stage=='sg12_crf':
                raw_names=(lr.REF,lr.P1,lr.SG12)
                counts,ap=cv.score_maps(torch.stack([maps[n] for n in raw_names]).reshape(-1,64,64),truth,'minmax',thresholds)
                crf,ds=crf_evaluate(maps,sample,thresholds,args.crf_site)
                counts=np.concatenate((counts,crf));ap=np.concatenate((ap,[np.nan,np.nan]));diagnostics.update(ds)
            elif args.stage=='sg14':
                counts,ap=cv.score_maps(torch.stack([maps[n] for n in names]).reshape(-1,64,64),truth,'minmax',thresholds)
                diagnostics.update(lr.complementarity_diagnostics(maps,obs,seed,details,truth,thresholds,spec['diagnostic_change_tolerance']))
            else:
                labels=spectral_partition(graph,32);del graph
                controls,ap=cv.score_maps(torch.stack([maps[n] for n in names[:4]]).reshape(-1,64,64),truth,'minmax',thresholds)
                refined,ds=region_evaluate(maps,labels,truth,thresholds)
                counts=np.concatenate((controls,refined));ap=np.concatenate((ap,np.full(12,np.nan)));diagnostics.update(ds)
                _,legacy_oracle=oracle_diagnostic(labels,truth);diagnostics['legacy_sg13_oracle_counts']=legacy_oracle
                if args.dataset=='voc':
                    if classname in spec['visual_classes'] and row['image_id']==allrows[0]['image_id']:
                        visual_cache(out/'visual_cache'/f'{args.class_shard:02d}_representative.npz',sample,labels,maps,row['image_id'],classname)
                    ti=thresholds.index(.45)
                    for ci,name in enumerate(names[4:]):
                        pi=ci//4;prior_index=names.index(lr.PRIORS[pi])
                        a=refined[ci,ti];b=controls[prior_index,ti]
                        gain=a[0]/max(a[:3].sum(),1)-b[0]/max(b[:3].sum(),1)
                        for key in ('best','worst'):
                            if extremes[key] is None or (gain>extremes[key] if key=='best' else gain<extremes[key]):
                                extremes[key]=gain
                                visual_cache(out/'visual_cache'/f'{args.class_shard:02d}_{key}.npz',sample,labels,maps,row['image_id'],classname,[gain,pi,ci%4])
                del labels
            diag.add(row['image_id'],diagnostics)
            writer.add(row['image_id'],classname,counts,ap)
            print(f'[{args.stage}:{args.dataset}:{classname}] {row["image_id"]} {len(writer.completed)}/{len(rows)}',flush=True)
            del maps,obs,details,seed,counts,ap,diagnostics,one,collected_two
        finally:
            recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
            common._RECORDERS.clear();del sample,recorder
            gc.collect();torch.cuda.empty_cache()
    if args.max_samples is None:
        if writer.completed!={r['image_id'] for r in allrows} or writer.completed!=set(diag.records):raise ValueError('incomplete IDs or sidecars')
        writer.finish()

if __name__=='__main__':main()
