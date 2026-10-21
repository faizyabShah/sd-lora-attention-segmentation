#!/usr/bin/env python3
"""Resumable locked SG17 VOC selection, frozen COCO transfer and optional CRF."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
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
from scripts.select_propagation_cv import run_screen,best_threshold,pooled_miou,image_ious,class_ious
from src.segmentation import controlled_cv as cv
from src.segmentation import locked_refinements as lr
from src.segmentation import sg17_reliability as sg
from src.segmentation.ovam_full import apply_ovam_densecrf,OVAM_CRF_PARAMETERS

SPEC=ROOT/'configs/sg17_reliability_filtered_additions.json'
BASE=ROOT/'outputs/segmentation/controlled_attention_cv/sg17'
PLOT_ENV=Path('/home/shafa/miniconda3/envs/ovam_sdxl/bin/python')


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def key(image_id):return hashlib.sha256(image_id.encode()).hexdigest()[:20]


def npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)


def dataset_rows(dataset,max_images=None):
    rows=[]
    for i in range(20):
        classname,rs=(transfer.voc_rows if dataset=='voc' else transfer.coco_rows)(i)
        rows.extend((classname,r) for r in (rs[:1] if max_images else rs))
    return rows


def paths(out,dataset,image_id):
    stem=key(image_id)
    return out/'cache'/dataset/(stem+'.npz'),out/'screen'/dataset/(stem+'.npz'),out/'crf'/dataset/(stem+'.npz')


def check_meta(data,row,dataset,spec):
    meta=json.loads(str(data['meta']))
    if meta!={'image_id':row['image_id'],'dataset':dataset,'spec':spec}:
        raise ValueError('foreign or incompatible image checkpoint')


def make_pipe(model,device,locked,frozen):
    pipe=StableDiffusionPipeline.from_pretrained(model,torch_dtype=torch.float32,local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True);transfer._PIPE=pipe
    modules=[n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca_modules=tuple(sorted(n for n in modules if n.endswith('attn2')))
    selected=tuple(sorted(n for n in modules if n.endswith('attn1')))[locked['sa']['module_index']]
    WordObservationRecorder.selected_sa=selected
    ca_spec=common.resolve_seed_spec(frozen,ca_modules,locked['sa']['calls'])
    return pipe,ca_modules,selected,ca_spec


def generate(pipe,ca_modules,selected,ca_spec,row,dataset,device,verify=False):
    manifest=transfer.EVAL/('ovam/voc_sim' if dataset=='voc' else 'ovam/coco_captions')
    transfer._ATTRIBUTION_PROMPT[0]=row['attribution_prompt'];common._RECORDERS.clear()
    sample=common.live.generate_attention_sample(pipe,row,manifest,'float16',cross_map_kind='conditional')
    recorder=common._RECORDERS[-1]
    try:
        resolution={n:r for r,mods in sample.modules_by_resolution.items() for n in mods}
        seed=common.ca_seed(recorder,ca_modules,resolution,ca_spec,device)
        calls=recorder.sa_qk[selected]
        if len(calls)!=31 or any(q.shape[0]!=8 for q,k in calls):raise ValueError('unexpected SA calls/heads')
        one,two,q=sg.observations(calls,seed)
        if verify:
            old_one,old_two=transfer.propagate(recorder,selected,seed,device)
            torch.testing.assert_close(one,old_one.reshape(-1),rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(two,old_two.reshape(-1),rtol=1e-5,atol=2e-6)
        return (one.cpu().numpy(),two.cpu().numpy(),q.cpu().numpy(),
                np.asarray(sample.target_mask,bool),np.asarray(sample.generated_image,np.uint8),
                seed.reshape(-1).cpu().numpy())
    finally:
        recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
        common._RECORDERS.clear();del sample,recorder
        gc.collect();torch.cuda.empty_cache()


def capture(out,dataset,rows,pipe,ca_modules,selected,ca_spec,device,spec):
    for index,(classname,row) in enumerate(rows):
        path,_,_=paths(out,dataset,row['image_id'])
        if path.exists():
            with np.load(path,allow_pickle=False) as d:check_meta(d,row,dataset,spec)
            continue
        one,two,q,truth,_,_=generate(pipe,ca_modules,selected,ca_spec,row,dataset,device,index==0)
        if one.shape!=(4096,) or two.shape!=one.shape or q.max()>248 or not(np.isfinite(one).all() and np.isfinite(two).all()):
            raise ValueError('invalid SG17 cached maps')
        meta={'image_id':row['image_id'],'dataset':dataset,'spec':spec}
        npz(path,meta=np.array(json.dumps(meta,sort_keys=True)),one=one.astype(np.float32),
            two=two.astype(np.float32),q_count=q.astype(np.uint8),target=truth.astype(np.bool_))
        print(f'capture {dataset} {index+1}/{len(rows)} {row["image_id"]}',flush=True)


def calibrated_cutoffs(out,rows,spec):
    path=out/'cutoffs.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state['spec']!=spec or state['voc_image_ids']!=[r['image_id'] for _,r in rows]:
            raise ValueError('VOC calibration state changed')
        return state
    hist=np.zeros(249,np.int64)
    for _,row in rows:
        p,_,_=paths(out,'voc',row['image_id'])
        with np.load(p,allow_pickle=False) as d:
            check_meta(d,row,'voc',spec)
            hist+=sg.native_positive_hist(d['one'],d['two'],d['q_count'])
    cutoffs=sg.quantile_cutoffs(hist,spec['quantiles'])
    cumulative=np.cumsum(hist)
    edges=[0]+[int(np.searchsorted(cumulative,q*hist.sum(),side='left'))
               for q in np.arange(.1,1.,.1)]+[249]
    state={'spec':spec,'source':'VOC only; raw positive second-pass increments at 64x64',
           'voc_image_ids':[r['image_id'] for _,r in rows],
           'positive_increment_pixels':int(hist.sum()),'q_count_histogram':hist.tolist(),
           'cutoffs':cutoffs,'decile_edges':edges}
    atomic_json(path,state)
    print('Frozen VOC cutoffs',cutoffs,flush=True)
    return state


def score_dataset(out,dataset,rows,state,spec,thresholds):
    for index,(classname,row) in enumerate(rows):
        cache,screen,_=paths(out,dataset,row['image_id'])
        if screen.exists():
            with np.load(screen,allow_pickle=False) as d:
                if str(d['source_hash'])!=key(json.dumps(state['cutoffs'],sort_keys=True)):
                    raise ValueError('score checkpoint uses different cutoffs')
            continue
        with np.load(cache,allow_pickle=False) as d:
            check_meta(d,row,dataset,spec)
            maps=sg.candidate_maps(d['one'],d['two'],d['q_count'],state['cutoffs'])
            truth=d['target']
        counts,ap=cv.score_maps(maps,truth,'minmax',thresholds)
        if counts.shape!=(4,len(thresholds),4):raise ValueError('score shape')
        npz(screen,counts=counts,ap=ap,source_hash=np.array(key(json.dumps(state['cutoffs'],sort_keys=True))),
            image_id=np.array(row['image_id']))
        print(f'screen {dataset} {index+1}/{len(rows)} {row["image_id"]}',flush=True)


def results_object(out,dataset,rows,state,thresholds):
    folds=cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if dataset=='voc' else {}
    counts=[];aps=[]
    for _,row in rows:
        _,path,_=paths(out,dataset,row['image_id'])
        with np.load(path,allow_pickle=False) as d:
            if str(d['image_id'])!=row['image_id']:raise ValueError('score image id differs')
            counts.append(d['counts']);aps.append(d['ap'])
    return cv.Results(sg.NAMES,tuple(thresholds),tuple(r['image_id'] for _,r in rows),
                      tuple(c for c,_ in rows),
                      np.array([folds[r['image_id']] if dataset=='voc' else 0 for _,r in rows],np.int64),
                      np.stack(counts).transpose(1,0,2,3),np.stack(aps).T,
                      {'cutoffs':state['cutoffs'],'dataset':dataset},(True,))


def forced_oof(result,name,ref='W/P1'):
    i=result.index(name);r=result.index(ref)
    candidate=np.zeros((len(result.image_ids),4),np.int64)
    reference=np.zeros_like(candidate);choices=[]
    for fold in range(5):
        train=result.folds!=fold;held=~train
        ti,_=best_threshold(result,i,train);rt,_=best_threshold(result,r,train)
        candidate[held]=result.counts[i,held,ti];reference[held]=result.counts[r,held,rt]
        choices.append({'fold':fold,'candidate_threshold':result.thresholds[ti],
                        'reference_threshold':result.thresholds[rt]})
    return {'candidate_miou':pooled_miou(candidate,result.classes),
            'reference_miou':pooled_miou(reference,result.classes),
            'gain':pooled_miou(candidate,result.classes)-pooled_miou(reference,result.classes),
            'thresholds':choices,
            'paired_mean_image_iou':cv.paired_bootstrap(image_ious(candidate),image_ious(reference))}


def select_voc(out,result,spec,state):
    path=out/'selection.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['cutoffs']!=state['cutoffs'] or saved['spec']!=spec:raise ValueError('selection changed')
        return saved
    names=sg.NAMES[1:]
    screen=run_screen(result,sg.NAMES[0],list(names),
                      {'W/SG17-Q70':3,'W/SG17-Q80':2,'W/SG17-Q90':1})
    allmask=np.ones(len(result.image_ids),bool)
    full={}
    for name in sg.NAMES:
        ti,_=best_threshold(result,result.index(name),allmask)
        full[name]={'threshold':result.thresholds[ti],
                    'voc_miou':pooled_miou(result.counts[result.index(name),:,ti],result.classes)}
    forced={name:forced_oof(result,name) for name in names}
    illustrative=max(names,key=lambda name:(forced[name]['candidate_miou'],
                                           state['cutoffs'][name.rsplit('-',1)[-1]]['actual_retained_fraction']))
    saved={'spec':spec,'cutoffs':state['cutoffs'],'screen':screen,'forced_oof':forced,
           'full_voc':full,'illustrative_variant':screen['frozen'] if screen['frozen']!=sg.NAMES[0] else illustrative,
           'promoted':screen['frozen']!=sg.NAMES[0]}
    atomic_json(path,saved);return saved


def full_metrics(result,name,threshold,reference_name,reference_threshold):
    ci=result.index(name);ri=result.index(reference_name)
    ti=result.thresholds.index(threshold);rt=result.thresholds.index(reference_threshold)
    a=result.counts[ci,:,ti];b=result.counts[ri,:,rt]
    per=class_ious(result,ci,ti,np.ones(len(result.image_ids),bool))
    ref=class_ious(result,ri,rt,np.ones(len(result.image_ids),bool))
    tp,fp,fn,_=a.sum(0)
    return {'images':len(result.image_ids),'threshold':threshold,
            'reference_threshold':reference_threshold,'miou':pooled_miou(a,result.classes),
            'reference_miou':pooled_miou(b,result.classes),
            'gain':pooled_miou(a,result.classes)-pooled_miou(b,result.classes),
            'mean_image_iou':float(image_ious(a).mean()),
            'recall':float(tp/max(tp+fn,1)),'precision':float(tp/max(tp+fp,1)),
            'per_class_iou':per,'per_class_delta':{c:per[c]-ref[c] for c in per},
            'paired_mean_image_iou':cv.paired_bootstrap(image_ious(a),image_ious(b))}


def summarize_mechanism(out,dataset,rows,selection,result,state):
    summary={};per_image=[]
    ref_threshold=selection['full_voc']['W/P1']['threshold']
    selected=selection['illustrative_variant']
    bins=np.zeros((10,5),np.int64)
    raw_hist=np.zeros(249,np.int64)
    for name in sg.NAMES[1:]:
        summary[name]={'retained':{k:0 for k in ('tp','fp','fp_gt16','fp_gt64','fp_separate')},
                       'rejected':{k:0 for k in ('tp','fp','fp_gt16','fp_gt64','fp_separate')},
                       'raw_gate_rejected_positive_increment_pixels':0,
                       'p1_additions_outside_raw_positive_increment':0,
                       'candidate_new_additions_outside_p1':0,
                       'p1_removals_reversed_by_candidate':0}
    for index,(classname,row) in enumerate(rows):
        cache,_,_=paths(out,dataset,row['image_id'])
        with np.load(cache,allow_pickle=False) as d:
            one=d['one'];two=d['two'];q=d['q_count'];truth=d['target']
        raw_hist+=sg.native_positive_hist(one,two,q)
        maps=sg.candidate_maps(one,two,q,state['cutoffs'])
        for ci,name in enumerate(sg.NAMES[1:],start=1):
            t=selection['full_voc'][name]['threshold']
            b,p,c=sg.full_masks(one,two,maps[ci],truth,ref_threshold,t)
            m=sg.mechanism(b,p,c,truth,q,state['cutoffs'][name.rsplit('-',1)[-1]]['count_cutoff'],two-one)
            for kind in ('retained','rejected'):
                for k,v in m[kind].items():summary[name][kind][k]+=v
            for k in ('raw_gate_rejected_positive_increment_pixels','p1_additions_outside_raw_positive_increment',
                      'candidate_new_additions_outside_p1','p1_removals_reversed_by_candidate'):
                summary[name][k]+=m[k]
            if name==selected:
                counts=result.counts[result.index(name),index,result.thresholds.index(t)]
                prior=result.counts[0,index,result.thresholds.index(ref_threshold)]
                ai=image_ious(counts[None])[0];bi=image_ious(prior[None])[0]
                per_image.append({'dataset':dataset,'image_id':row['image_id'],'class':classname,
                    'P1_iou':float(bi),'SG17_iou':float(ai),'iou_delta':float(ai-bi),
                    'rejected_positive_increment_pixels':m['raw_gate_rejected_positive_increment_pixels'],
                    'rejected_P1_addition_pixels':int(m['rejected']['tp']+m['rejected']['fp']),
                    'rejected_TP_additions':m['rejected']['tp'],'rejected_FP_additions':m['rejected']['fp'],
                    'rejected_distant_FP_gt16':m['rejected']['fp_gt16'],
                    'rejected_distant_FP_gt64':m['rejected']['fp_gt64'],
                    'mean_q_rejected_additions':m['mean_q_rejected_p1_additions'],
                    'mean_q_retained_additions':m['mean_q_retained_p1_additions']})
        b,p,_=sg.full_masks(one,two,maps[0],truth,ref_threshold,ref_threshold)
        bins+=sg.reliability_bins(b,p,truth,q,state['decile_edges'])
    for name,record in summary.items():
        tp=record['retained']['tp'];fp=record['rejected']['fp']
        record['TP_retention_rate']=tp/max(tp+record['rejected']['tp'],1)
        record['FP_rejection_rate']=fp/max(fp+record['retained']['fp'],1)
    curve=[]
    for i,(total,tp,fp,far16,far64) in enumerate(bins):
        low,high=state['decile_edges'][i:i+2]
        curve.append({'decile':i+1,'q_count_low':state['decile_edges'][i],
                      'q_count_high_exclusive':state['decile_edges'][i+1],
                      'raw_positive_increment_pixels':int(raw_hist[low:high].sum()),
                      'positive_P1_additions':int(total),'TP':int(tp),'FP':int(fp),
                      'TP_fraction':float(tp/total) if total else None,
                      'FP_fraction':float(fp/total) if total else None,
                      'TP_FP_ratio':float(tp/fp) if fp else None,
                      'distant_FP_gt16':int(far16),'distant_FP_gt64':int(far64),
                      'distant_FP_fraction':float(far16/total) if total else None})
    return summary,per_image,curve


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def final_screen(out,voc,coco,voc_rows,coco_rows,selection,state):
    path=out/'results.json'
    if path.exists():return json.loads(path.read_text())
    metrics={};mech={};rows=[];curves={}
    for dataset,result,manifest in (('voc',voc,voc_rows),('coco',coco,coco_rows)):
        metrics[dataset]={}
        for name in sg.NAMES:
            metrics[dataset][name]=full_metrics(result,name,
                selection['full_voc'][name]['threshold'],'W/P1',
                selection['full_voc']['W/P1']['threshold'])
        mech[dataset],records,curves[dataset]=summarize_mechanism(out,dataset,manifest,selection,result,state)
        rows.extend(records)
    if len(rows)!=len(voc_rows)+len(coco_rows):raise ValueError('per-image diagnostic coverage')
    write_csv(out/'per_image_selected.csv',rows)
    summary={'cutoffs':state,'selection':selection,'metrics':metrics,
             'mechanism':mech,'reliability_bins':curves,
             'per_image_csv':'per_image_selected.csv',
             'visual_method_is_diagnostic_if_not_promoted':not selection['promoted']}
    atomic_json(path,summary)
    lines=['# SG17 locked screen','',f"VOC fold winners: {[r['chosen'] for r in selection['screen']['folds']]}",
           f"Frozen: {selection['screen']['frozen']}; promoted: {selection['promoted']}",
           f"OOF reference {selection['screen']['oof_miou_reference']:.4f}, selection "
           f"{selection['screen']['oof_miou_selection']:.4f}, gain {selection['screen']['oof_gain']:+.4f}",
           '', '| Method | VOC full mIoU | COCO frozen mIoU | VOC OOF forced gain |',
           '|---|---:|---:|---:|']
    for name in sg.NAMES:
        gain=selection['forced_oof'][name]['gain'] if name!='W/P1' else 0.
        lines.append(f"| {name} | {metrics['voc'][name]['miou']:.4f} | "
                     f"{metrics['coco'][name]['miou']:.4f} | {gain:+.4f} |")
    lines+=['','## Mechanism: rejected P1 additions','',
            '| Dataset | Method | TP retention | FP rejection | Rejected FP >16 px | Rejected FP >64 px |',
            '|---|---|---:|---:|---:|---:|']
    for dataset in ('voc','coco'):
        for name in sg.NAMES[1:]:
            z=mech[dataset][name]
            lines.append(f"| {dataset} | {name} | {z['TP_retention_rate']:.3f} | "
                         f"{z['FP_rejection_rate']:.3f} | {z['rejected']['fp_gt16']} | "
                         f"{z['rejected']['fp_gt64']} |")
    lines+=['','All five folds, candidate thresholds, bootstrap intervals, per-class IoUs, '
            'retained/rejected events, and VOC-frozen COCO reliability bins are in `results.json`.','']
    (out/'report.md').write_text('\n'.join(lines))
    return summary


def visual_ids(out,voc_rows):
    desired=('bird','bicycle','cow','dining-table','person','chair')
    first={}
    for classname,row in voc_rows:first.setdefault(classname,row['image_id'])
    records=list(csv.DictReader(open(out/'per_image_selected.csv')))
    v=[r for r in records if r['dataset']=='voc']
    best=max(v,key=lambda r:float(r['iou_delta']))['image_id']
    worst=min(v,key=lambda r:float(r['iou_delta']))['image_id']
    return [(c,first[c]) for c in desired]+[('largest_gain',best),('largest_loss',worst)]


def visuals(out,voc_rows,report,pipe,ca_modules,selected,ca_spec,device):
    chosen=report['selection']['illustrative_variant']
    if chosen=='W/P1':return
    lookup={r['image_id']:(c,r) for c,r in voc_rows}
    ids=visual_ids(out,voc_rows)
    t=report['selection']['full_voc'][chosen]['threshold']
    rt=report['selection']['full_voc']['W/P1']['threshold']
    cutoff=report['cutoffs']['cutoffs'][chosen.rsplit('-',1)[-1]]['count_cutoff']
    for i,(role,image_id) in enumerate(ids):
        figure=out/'visuals'/f'{i:02d}_{role.replace(" ","_")}.png'
        if figure.exists():continue
        classname,row=lookup[image_id]
        one,two,q,truth,image,seed=generate(pipe,ca_modules,selected,ca_spec,row,'voc',device)
        candidate=sg.candidate_maps(one,two,q,report['cutoffs']['cutoffs'])[sg.NAMES.index(chosen)].numpy()
        _,p1,pred=sg.full_masks(one,two,candidate,truth,rt,t)
        cache=out/'visuals'/'temporary'/f'{i:02d}.npz'
        npz(cache,image=image,truth=truth,seed=seed.reshape(64,64),one=one.reshape(64,64),
            two=two.reshape(64,64),q=q.reshape(64,64),candidate=candidate,
            p1=p1,pred=pred,cutoff=np.array(cutoff),image_id=np.array(image_id),
            classname=np.array(classname),method=np.array(chosen))
        env=dict(os.environ,MPLCONFIGDIR=str(out/'visuals'/'temporary'/'matplotlib'))
        subprocess.run([str(PLOT_ENV),str(ROOT/'scripts/plot_sg17_reliability_filtered.py'),
                        '--cache',str(cache),'--output',str(figure)],env=env,check=True)
        cache.unlink();print(f'visual {image_id}',flush=True)


def confirmation(out,rows_by_dataset,selection,pipe,ca_modules,selected,ca_spec,device):
    if not selection['promoted']:
        atomic_json(out/'crf_skipped.json',{'reason':'no SG17 variant passed VOC promotion; no CRF generated'})
        return
    method=selection['screen']['frozen'];cutoff=selection['cutoffs'][method.rsplit('-',1)[-1]]['count_cutoff']
    ref_t=selection['full_voc']['W/P1']['threshold'];cand_t=selection['full_voc'][method]['threshold']
    result={}
    for dataset,rows in rows_by_dataset.items():
        ids=[];classes=[];counts=[]
        for index,(classname,row) in enumerate(rows):
            _,_,path=paths(out,dataset,row['image_id'])
            if path.exists():
                with np.load(path,allow_pickle=False) as d:
                    if str(d['image_id'])!=row['image_id']:raise ValueError('CRF resume image mismatch')
                    c=d['counts']
            else:
                one,two,q,truth,image,_=generate(pipe,ca_modules,selected,ca_spec,row,dataset,device,index==0)
                candidate=sg.candidate_maps(one,two,q,selection['cutoffs'])[sg.NAMES.index(method)]
                _,p1,pred=sg.full_masks(one,two,candidate,truth,ref_t,cand_t)
                ref=apply_ovam_densecrf(image,p1,out.parent.parent.parent/'dependencies/simplecrf_py310')
                can=apply_ovam_densecrf(image,pred,out.parent.parent.parent/'dependencies/simplecrf_py310')
                c=np.stack((lr.mask_counts(ref,truth),lr.mask_counts(can,truth)))
                npz(path,image_id=np.array(row['image_id']),counts=c)
                print(f'CRF {dataset} {index+1}/{len(rows)} {row["image_id"]}',flush=True)
            ids.append(row['image_id']);classes.append(classname);counts.append(c)
        stack=np.stack(counts);a=stack[:,1];b=stack[:,0]
        def per_class(x):
            return {c:float(x[np.array(classes)==c,:3].sum(0)[0]/max(x[np.array(classes)==c,:3].sum(),1))
                    for c in sorted(set(classes))}
        ac=per_class(a);bc=per_class(b)
        tp,fp,fn,_=a.sum(0)
        result[dataset]={'images':len(ids),'method':method,
                         'candidate_miou':float(np.mean(list(ac.values()))),
                         'P1_densecrf_miou':float(np.mean(list(bc.values()))),
                         'gain':float(np.mean(list(ac.values()))-np.mean(list(bc.values()))),
                         'candidate_mean_image_iou':float(image_ious(a).mean()),
                         'P1_densecrf_mean_image_iou':float(image_ious(b).mean()),
                         'candidate_recall':float(tp/max(tp+fn,1)),
                         'candidate_precision':float(tp/max(tp+fp,1)),
                         'candidate_per_class_iou':ac,'P1_densecrf_per_class_iou':bc,
                         'per_class_delta':{c:ac[c]-bc[c] for c in ac},
                         'paired_mean_image_iou':cv.paired_bootstrap(image_ious(a),image_ious(b))}
        atomic_json(out/'crf_results.json',{'selection':selection['screen']['frozen'],
                   'fixed_parameters':OVAM_CRF_PARAMETERS,'reference_threshold':ref_t,
                   'candidate_threshold':cand_t,'datasets':result})


def smoke(args,spec,locked,frozen):
    out=args.output_dir;pipe,ca_modules,selected,ca_spec=make_pipe(args.model,args.device,locked,frozen)
    rows={d:dataset_rows(d,max_images=1)[:1] for d in ('voc','coco')}
    capture(out,'voc',rows['voc'],pipe,ca_modules,selected,ca_spec,args.device,spec)
    state=calibrated_cutoffs(out,rows['voc'],spec)
    capture(out,'coco',rows['coco'],pipe,ca_modules,selected,ca_spec,args.device,spec)
    for d in rows:score_dataset(out,d,rows[d],state,spec,locked['thresholds'])
    smoke_visual=None
    for d in rows:
        _,row=rows[d][0];cache,_,_=paths(out,d,row['image_id'])
        with np.load(cache) as data:
            maps=sg.candidate_maps(data['one'],data['two'],data['q_count'],state['cutoffs'])
            before,p1,candidate=sg.full_masks(data['one'],data['two'],maps[1],data['target'],.45,.45)
            diag=sg.mechanism(before,p1,candidate,data['target'],data['q_count'],
                              state['cutoffs']['Q70']['count_cutoff'],data['two']-data['one'])
            assert diag['retained']['tp']+diag['rejected']['tp']>=0
            if d=='voc':
                one,two,q,truth,image,seed=generate(pipe,ca_modules,selected,ca_spec,row,d,args.device)
                assert np.allclose(one,data['one'],rtol=1e-5,atol=2e-6)
                candidate_map=sg.candidate_maps(one,two,q,state['cutoffs'])[1].numpy()
                _,p1,pred=sg.full_masks(one,two,candidate_map,truth,.45,.45)
                crf_site=ROOT/'outputs/dependencies/simplecrf_py310'
                crf_ref=apply_ovam_densecrf(image,p1,crf_site)
                crf_candidate=apply_ovam_densecrf(image,pred,crf_site)
                assert crf_ref.shape==truth.shape and crf_candidate.shape==truth.shape
                smoke_visual=out/'smoke_visual.png'
                temporary=out/'smoke_visual_input.npz'
                npz(temporary,image=image,truth=truth,seed=seed.reshape(64,64),
                    one=one.reshape(64,64),two=two.reshape(64,64),q=q.reshape(64,64),
                    candidate=candidate_map,p1=p1,pred=pred,
                    cutoff=np.array(state['cutoffs']['Q70']['count_cutoff']),
                    image_id=np.array(row['image_id']),classname=np.array(rows['voc'][0][0]),
                    method=np.array('W/SG17-Q70'))
                env=dict(os.environ,MPLCONFIGDIR=str(out/'matplotlib'))
                subprocess.run([str(PLOT_ENV),str(ROOT/'scripts/plot_sg17_reliability_filtered.py'),
                                '--cache',str(temporary),'--output',str(smoke_visual)],
                               env=env,check=True)
                temporary.unlink()
                assert smoke_visual.exists() and smoke_visual.stat().st_size>0
    atomic_json(out/'smoke_validation.json',{'passed':True,'cutoffs':state['cutoffs'],
               'voc_image_id':rows['voc'][0][1]['image_id'],
               'coco_image_id':rows['coco'][0][1]['image_id'],
               'plot_and_fixed_densecrf_passed':smoke_visual is not None})
    print('SG17 live smoke passed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=BASE)
    p.add_argument('--model',default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();spec=json.loads(SPEC.read_text())
    locked=json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    frozen=json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not frozen.get('frozen') or locked['sa']['observations']!=spec['observations']:
        raise ValueError('frozen model contract changed')
    if args.smoke:return smoke(args,spec,locked,frozen)
    out=args.output_dir;voc_rows=dataset_rows('voc');coco_rows=dataset_rows('coco')
    pipe,ca_modules,selected,ca_spec=make_pipe(args.model,args.device,locked,frozen)
    capture(out,'voc',voc_rows,pipe,ca_modules,selected,ca_spec,args.device,spec)
    state=calibrated_cutoffs(out,voc_rows,spec)
    score_dataset(out,'voc',voc_rows,state,spec,locked['thresholds'])
    voc=results_object(out,'voc',voc_rows,state,locked['thresholds'])
    selection=select_voc(out,voc,spec,state)
    capture(out,'coco',coco_rows,pipe,ca_modules,selected,ca_spec,args.device,spec)
    score_dataset(out,'coco',coco_rows,state,spec,locked['thresholds'])
    coco=results_object(out,'coco',coco_rows,state,locked['thresholds'])
    report=final_screen(out,voc,coco,voc_rows,coco_rows,selection,state)
    visuals(out,voc_rows,report,pipe,ca_modules,selected,ca_spec,args.device)
    confirmation(out,{'voc':voc_rows,'coco':coco_rows},selection,pipe,ca_modules,selected,ca_spec,args.device)
    atomic_json(out/'completed.json',{'screen_images':{'voc':len(voc_rows),'coco':len(coco_rows)},
               'frozen':selection['screen']['frozen'],'crf_completed_or_skipped':True,
               'visual_sheets':len(list((out/'visuals').glob('*.png')))})
    print('SG17 completed',flush=True)

if __name__=='__main__':main()
