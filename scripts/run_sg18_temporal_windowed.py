#!/usr/bin/env python3
"""Resumable locked SG18 VOC selection and frozen COCO transfer."""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_sg11_sg12_observation_reliability import WordObservationRecorder,transfer,common
from scripts.select_propagation_cv import run_screen,best_threshold,pooled_miou,image_ious,class_ious
from src.segmentation import controlled_cv as cv
from src.segmentation import sg18_temporal as sg

SPEC=ROOT/'configs/sg18_temporal_windowed_propagation.json'
BASE=ROOT/'outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed'
PLOT_ENV=Path('/home/shafa/miniconda3/envs/ovam_sdxl/bin/python')


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)


def key(image_id):return hashlib.sha256(image_id.encode()).hexdigest()[:20]


def rows_for(dataset,first_only=False):
    rows=[]
    for index in range(20):
        classname,items=(transfer.voc_rows if dataset=='voc' else transfer.coco_rows)(index)
        rows.extend((classname,r) for r in (items[:1] if first_only else items))
    return rows[:1] if first_only else rows


def paths(out,dataset,image_id):
    stem=key(image_id)
    return out/'cache'/dataset/(stem+'.npz'),out/'screen'/dataset/(stem+'.npz')


def make_pipe(model,device,locked,frozen):
    pipe=StableDiffusionPipeline.from_pretrained(model,torch_dtype=torch.float32,local_files_only=True).to(device)
    pipe.set_progress_bar_config(disable=True);transfer._PIPE=pipe
    modules=[n.removesuffix('.processor') for n in pipe.unet.attn_processors]
    ca_modules=tuple(sorted(n for n in modules if n.endswith('attn2')))
    selected=tuple(sorted(n for n in modules if n.endswith('attn1')))[locked['sa']['module_index']]
    WordObservationRecorder.selected_sa=selected
    ca_spec=common.resolve_seed_spec(frozen,ca_modules,locked['sa']['calls'])
    return pipe,ca_modules,selected,ca_spec


def live_maps(pipe,ca_modules,selected,ca_spec,row,dataset,device,names=sg.NAMES,verify=False):
    manifest=transfer.EVAL/('ovam/voc_sim' if dataset=='voc' else 'ovam/coco_captions')
    transfer._ATTRIBUTION_PROMPT[0]=row['attribution_prompt'];common._RECORDERS.clear()
    if device.startswith('cuda'):torch.cuda.synchronize()
    start=time.perf_counter()
    sample=common.live.generate_attention_sample(pipe,row,manifest,'float16',cross_map_kind='conditional')
    recorder=common._RECORDERS[-1]
    try:
        resolution={n:r for r,mods in sample.modules_by_resolution.items() for n in mods}
        seed=common.ca_seed(recorder,ca_modules,resolution,ca_spec,device)
        if device.startswith('cuda'):torch.cuda.synchronize()
        capture_seconds=time.perf_counter()-start
        maps,aggregate_seconds=sg.aggregate(recorder.sa_qk[selected],seed,names)
        if device.startswith('cuda'):torch.cuda.synchronize()
        total_seconds=time.perf_counter()-start
        if verify and {'W/REF','W/P1'}<=set(names):
            old_ref,old_p1=transfer.propagate(recorder,selected,seed,device)
            torch.testing.assert_close(maps['W/REF'],old_ref.reshape(-1),rtol=1e-5,atol=2e-6)
            torch.testing.assert_close(maps['W/P1'],old_p1.reshape(-1),rtol=1e-5,atol=2e-6)
        return ({n:maps[n].cpu().numpy() for n in names},seed.reshape(-1).cpu().numpy(),
                np.asarray(sample.target_mask,bool),np.asarray(sample.generated_image,np.uint8),
                {'capture_seconds':capture_seconds,'aggregation_seconds':aggregate_seconds,
                 'end_to_end_seconds':total_seconds})
    finally:
        recorder.ca_calls.clear();recorder.sa_qk.clear();sample.clear()
        common._RECORDERS.clear();del sample,recorder
        gc.collect();torch.cuda.empty_cache()


def verify_cache(data,row,dataset,spec):
    meta=json.loads(str(data['meta']))
    if meta!={'image_id':row['image_id'],'dataset':dataset,'spec':spec}:
        raise ValueError('foreign SG18 image checkpoint')


def capture(out,dataset,rows,pipe,ca_modules,selected,ca_spec,device,spec):
    for index,(_,row) in enumerate(rows):
        path,_=paths(out,dataset,row['image_id'])
        if path.exists():
            with np.load(path,allow_pickle=False) as d:verify_cache(d,row,dataset,spec)
            continue
        maps,seed,truth,_,timing=live_maps(pipe,ca_modules,selected,ca_spec,row,dataset,device,
                                           verify=index==0)
        array=np.stack([maps[n] for n in sg.NAMES]).astype(np.float32)
        if array.shape!=(5,4096) or not np.isfinite(array).all():raise ValueError('invalid SG18 maps')
        meta={'image_id':row['image_id'],'dataset':dataset,'spec':spec}
        npz(path,meta=np.array(json.dumps(meta,sort_keys=True)),maps=array,
            seed=seed.astype(np.float32),truth=truth.astype(np.bool_),
            timing=np.array([timing['capture_seconds'],timing['aggregation_seconds'],
                             timing['end_to_end_seconds']],np.float64))
        print(f'capture {dataset} {index+1}/{len(rows)} {row["image_id"]}',flush=True)


def score_dataset(out,dataset,rows,spec,thresholds):
    for index,(_,row) in enumerate(rows):
        cache,screen=paths(out,dataset,row['image_id'])
        if screen.exists():
            with np.load(screen,allow_pickle=False) as d:
                if str(d['image_id'])!=row['image_id'] or tuple(d['thresholds'])!=tuple(thresholds):
                    raise ValueError('incompatible score checkpoint')
            continue
        with np.load(cache,allow_pickle=False) as d:
            verify_cache(d,row,dataset,spec)
            maps=torch.as_tensor(d['maps']).reshape(5,64,64);truth=d['truth']
        counts,ap=cv.score_maps(maps,truth,'minmax',thresholds)
        if counts.shape!=(5,len(thresholds),4):raise ValueError('invalid SG18 count shape')
        npz(screen,image_id=np.array(row['image_id']),thresholds=np.asarray(thresholds),
            counts=counts,ap=ap)
        print(f'score {dataset} {index+1}/{len(rows)} {row["image_id"]}',flush=True)


def result_object(out,dataset,rows,thresholds):
    folds=cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if dataset=='voc' else {}
    counts=[];aps=[]
    for _,row in rows:
        _,path=paths(out,dataset,row['image_id'])
        with np.load(path,allow_pickle=False) as d:
            if str(d['image_id'])!=row['image_id']:raise ValueError('wrong score image')
            counts.append(d['counts']);aps.append(d['ap'])
    return cv.Results(sg.NAMES,tuple(thresholds),tuple(r['image_id'] for _,r in rows),
                      tuple(c for c,_ in rows),
                      np.array([folds[r['image_id']] if dataset=='voc' else 0 for _,r in rows],np.int64),
                      np.stack(counts).transpose(1,0,2,3),np.stack(aps).T,
                      {'dataset':dataset,'method':'SG18 locked'},(True,))


def forced_oof(result,name):
    ci=result.index(name);ri=result.index('W/REF')
    cand=np.zeros((len(result.image_ids),4),np.int64);ref=np.zeros_like(cand);folds=[]
    for fold in range(5):
        train=result.folds!=fold;held=~train
        ti,_=best_threshold(result,ci,train);rt,_=best_threshold(result,ri,train)
        cand[held]=result.counts[ci,held,ti];ref[held]=result.counts[ri,held,rt]
        folds.append({'fold':fold,'candidate_threshold':result.thresholds[ti],
                      'reference_threshold':result.thresholds[rt]})
    cm=pooled_miou(cand,result.classes);rm=pooled_miou(ref,result.classes)
    return {'candidate_miou':cm,'reference_miou':rm,'gain':cm-rm,'fold_thresholds':folds,
            'paired_mean_image_iou':cv.paired_bootstrap(image_ious(cand),image_ious(ref))}


def select_voc(out,result,spec):
    path=out/'selection.json'
    if path.exists():
        data=json.loads(path.read_text())
        if data['spec']!=spec:raise ValueError('selection specification changed')
        return data
    names=('W/T1','W/T2','W/T3')
    screen=run_screen(result,'W/REF',list(names),{'W/T3':3,'W/T2':2,'W/T1':1})
    allmask=np.ones(len(result.image_ids),bool)
    full={}
    for name in sg.NAMES:
        ti,_=best_threshold(result,result.index(name),allmask)
        full[name]={'threshold':result.thresholds[ti],
                    'voc_miou':pooled_miou(result.counts[result.index(name),:,ti],result.classes)}
    forced={name:forced_oof(result,name) for name in sg.NAMES[1:]}
    illustrative=max(names,key=lambda n:(forced[n]['candidate_miou'],
                                         sg.WINDOWS[n][1]-sg.WINDOWS[n][0]))
    saved={'spec':spec,'screen':screen,'forced_oof':forced,'full_voc':full,
           'promoted':screen['frozen']!='W/REF',
           'illustrative_window':screen['frozen'] if screen['frozen']!='W/REF' else illustrative,
           'illustrative_is_diagnostic':screen['frozen']=='W/REF'}
    atomic_json(path,saved);return saved


def full_metrics(result,name,threshold,reference_threshold):
    ci=result.index(name);ri=result.index('W/REF')
    ti=result.thresholds.index(threshold);rt=result.thresholds.index(reference_threshold)
    a=result.counts[ci,:,ti];b=result.counts[ri,:,rt]
    c=class_ious(result,ci,ti,np.ones(len(result.image_ids),bool))
    r=class_ious(result,ri,rt,np.ones(len(result.image_ids),bool))
    tp,fp,fn,_=a.sum(0)
    return {'images':len(result.image_ids),'threshold':threshold,
            'reference_threshold':reference_threshold,
            'miou':pooled_miou(a,result.classes),
            'gain_vs_ref':pooled_miou(a,result.classes)-pooled_miou(b,result.classes),
            'mean_image_iou':float(image_ious(a).mean()),
            'pixel_iou':float(tp/max(tp+fp+fn,1)),
            'recall':float(tp/max(tp+fn,1)),'precision':float(tp/max(tp+fp,1)),
            'per_class_iou':c,'per_class_delta_vs_ref':{k:c[k]-r[k] for k in c},
            'paired_mean_image_iou_vs_ref':cv.paired_bootstrap(image_ious(a),image_ious(b))}


def mechanism(out,dataset,rows,selection,spec):
    selected=selection['illustrative_window']
    output={};per_image=[]
    threshold_sets={'voc_frozen':{n:selection['full_voc'][n]['threshold'] for n in sg.NAMES},
                    'fixed_0p45':{n:spec['diagnostic_fixed_threshold'] for n in sg.NAMES}}
    for condition,thresholds in threshold_sets.items():
        event_sum={n:np.zeros(len(sg.EVENT_NAMES),np.int64) for n in sg.NAMES}
        comparison={n:{} for n in sg.NAMES[1:4]}
        for index,(classname,row) in enumerate(rows):
            cache,_=paths(out,dataset,row['image_id'])
            with np.load(cache,allow_pickle=False) as d:
                verify_cache(d,row,dataset,spec)
                maps={n:d['maps'][i] for i,n in enumerate(sg.NAMES)}
                seed=d['seed'];truth=d['truth']
            ca,pred=sg.masks(maps,seed,truth,thresholds,spec['diagnostic_ca_threshold'])
            distance=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
            for name in sg.NAMES:
                count,_=sg.events(ca,pred[name],truth,distance)
                event_sum[name]+=count
            for name in sg.NAMES[1:4]:
                changes=sg.compare(pred['W/REF'],pred[name],truth,ca,distance)
                for k,v in changes.items():comparison[name][k]=comparison[name].get(k,0)+v
                if condition=='voc_frozen' and name==selected:
                    def iou(mask):return float((mask&truth).sum()/max((mask|truth).sum(),1))
                    per_image.append({'dataset':dataset,'image_id':row['image_id'],
                                      'class':classname,'REF_iou':iou(pred['W/REF']),
                                      'SG18_iou':iou(pred[name]),
                                      'iou_delta':iou(pred[name])-iou(pred['W/REF']),
                                      **changes})
        details={}
        ref=event_sum['W/REF']
        for name in sg.NAMES[1:4]:
            z=comparison[name]
            details[name]={**z,
                           'tp_retention_fraction':z['tp_retained']/max(int(ref[0]),1),
                           'fp_rejection_fraction':z['fp_removed']/max(int(ref[1]),1),
                           'reference_new_tp_support_fraction':z['reference_new_tp_supported']/max(int(ref[0]),1),
                           'reference_new_fp_support_fraction':z['reference_new_fp_supported']/max(int(ref[1]),1),
                           'reference_distant_fp_support_fraction':z['reference_distant_fp_supported']/max(int(ref[2]),1)}
        output[condition]={'ca_threshold':spec['diagnostic_ca_threshold'],
                           'method_thresholds':thresholds,
                           'events':{n:sg.rates(c) for n,c in event_sum.items()},
                           'relative_to_ref':details}
        print(f'mechanism {dataset} {condition} {len(rows)} images',flush=True)
    return output,per_image


def write_csv(path,rows):
    if not rows:raise ValueError('empty CSV')
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def final_report(out,voc,coco,voc_rows,coco_rows,selection,spec):
    path=out/'results.json'
    if path.exists():return json.loads(path.read_text())
    metrics={};diagnostics={};records=[]
    for dataset,result,rows in (('voc',voc,voc_rows),('coco',coco,coco_rows)):
        metrics[dataset]={n:full_metrics(result,n,selection['full_voc'][n]['threshold'],
                                         selection['full_voc']['W/REF']['threshold']) for n in sg.NAMES}
        diagnostics[dataset],per_image=mechanism(out,dataset,rows,selection,spec)
        records.extend(per_image)
    if len(records)!=len(voc_rows)+len(coco_rows):raise ValueError('per-image coverage incomplete')
    write_csv(out/'per_image_selected.csv',records)
    summary={'spec':spec,'selection':selection,'metrics':metrics,'mechanism':diagnostics,
             'per_image_csv':'per_image_selected.csv',
             'selected_window_is_diagnostic':selection['illustrative_is_diagnostic']}
    atomic_json(path,summary)
    lines=['# SG18 locked temporal-windowed propagation','',
           f"VOC fold winners: {[x['chosen'] for x in selection['screen']['folds']]}",
           f"Frozen: {selection['screen']['frozen']}; promoted: {selection['promoted']}",
           f"OOF selection {selection['screen']['oof_miou_selection']:.4f}; REF "
           f"{selection['screen']['oof_miou_reference']:.4f}; gain "
           f"{selection['screen']['oof_gain']:+.4f}",'',
           '| Cell | VOC forced OOF | OOF Δ REF | Fold wins | VOC full | VOC threshold | COCO frozen |',
           '|---|---:|---:|---:|---:|---:|---:|']
    votes=selection['screen']['selection_counts']
    for n in sg.NAMES:
        forced=selection['forced_oof'].get(n)
        oof=forced['candidate_miou'] if forced else selection['screen']['oof_miou_reference']
        gain=forced['gain'] if forced else 0.
        lines.append(f"| {n} | {oof:.4f} | {gain:+.4f} | {votes.get(n,0)}/5 | "
                     f"{metrics['voc'][n]['miou']:.4f} | {metrics['voc'][n]['threshold']:.3f} | "
                     f"{metrics['coco'][n]['miou']:.4f} |")
    lines+=['','Mechanism at VOC-frozen thresholds and controlled fixed 0.45, '
            'all 20 per-class values, bootstrap intervals and image-level rows '
            'are in `results.json` and `per_image_selected.csv`.','']
    (out/'report.md').write_text('\n'.join(lines))
    return summary


def visual_ids(out,voc_rows):
    first={}
    for cls,row in voc_rows:first.setdefault(cls,row['image_id'])
    selected=list(csv.DictReader(open(out/'per_image_selected.csv')))
    selected=[r for r in selected if r['dataset']=='voc']
    gain=max(selected,key=lambda r:float(r['iou_delta']))['image_id']
    loss=min(selected,key=lambda r:float(r['iou_delta']))['image_id']
    return [(cls,first[cls]) for cls in ('bird','bicycle','cow','person','chair','dining-table')]+[
        ('largest_gain',gain),('largest_loss',loss)]


def visuals(out,voc_rows,report,pipe,ca_modules,selected,ca_spec,device):
    lookup={r['image_id']:(c,r) for c,r in voc_rows}
    threshold={n:report['selection']['full_voc'][n]['threshold'] for n in sg.NAMES}
    chosen=report['selection']['illustrative_window']
    for i,(role,image_id) in enumerate(visual_ids(out,voc_rows)):
        figure=out/'visuals'/f'{i:02d}_{role}.png'
        if figure.exists() and figure.stat().st_size>0:continue
        classname,row=lookup[image_id]
        maps,seed,truth,image,_=live_maps(pipe,ca_modules,selected,ca_spec,row,'voc',device)
        ca,pred=sg.masks(maps,seed,truth,threshold)
        temporary=out/'visuals'/'temporary'/f'{i:02d}.npz'
        npz(temporary,image=image,truth=truth,seed=seed.reshape(64,64),
            maps=np.stack([maps[n] for n in sg.NAMES]).reshape(5,64,64),
            masks=np.stack([pred[n] for n in sg.NAMES]),
            ca_mask=ca,names=np.array(sg.NAMES),chosen=np.array(chosen),
            diagnostic=np.array(report['selected_window_is_diagnostic']),
            image_id=np.array(image_id),classname=np.array(classname))
        env=dict(os.environ,MPLCONFIGDIR=str(out/'visuals'/'temporary'/'matplotlib'))
        subprocess.run([str(PLOT_ENV),str(ROOT/'scripts/plot_sg18_temporal_windowed.py'),
                        '--cache',str(temporary),'--output',str(figure)],env=env,check=True)
        temporary.unlink()
        if not figure.exists() or figure.stat().st_size==0:raise ValueError('empty SG18 visual')
        print(f'visual {image_id}',flush=True)


def benchmark(out,row,pipe,ca_modules,selected,ca_spec,device,method,repeats=3):
    path=out/'timing.json'
    if path.exists():return json.loads(path.read_text())
    methods=('W/REF',method,'W/P1')
    values={}
    # Every trial executes the same UNet and hook capture. The only selective
    # work is post-capture A@c aggregation; no inference speedup is claimed.
    for name in methods:
        measurements=[]
        for j in range(repeats):
            _,_,_,_,timing=live_maps(pipe,ca_modules,selected,ca_spec,row,'voc',device,
                                     names=(name,))
            measurements.append(timing)
            print(f'timing {name} {j+1}/{repeats}: {timing}',flush=True)
        values[name]={'trials':measurements,
                      'median_end_to_end_seconds':float(np.median([x['end_to_end_seconds'] for x in measurements])),
                      'median_capture_seconds':float(np.median([x['capture_seconds'] for x in measurements])),
                      'median_aggregation_seconds':float(np.median([x['aggregation_seconds'] for x in measurements]))}
    result={'image_id':row['image_id'],'repeats':repeats,
            'selected_window':method,'selected_is_diagnostic':method!=json.loads((out/'selection.json').read_text())['screen']['frozen'],
            'same_UNet_and_attention_capture_for_all_methods':True,
            'no_inference_speedup_claim':True,'methods':values}
    atomic_json(path,result)
    return result


def smoke(args,spec,locked,frozen):
    out=args.output_dir;pipe,ca_modules,selected,ca_spec=make_pipe(args.model,args.device,locked,frozen)
    rows={d:rows_for(d,first_only=True) for d in ('voc','coco')}
    for d in rows:
        capture(out,d,rows[d],pipe,ca_modules,selected,ca_spec,args.device,spec)
        score_dataset(out,d,rows[d],spec,locked['thresholds'])
        _,row=rows[d][0];cache,_=paths(out,d,row['image_id'])
        with np.load(cache,allow_pickle=False) as data:
            maps={n:data['maps'][i] for i,n in enumerate(sg.NAMES)}
            seed=data['seed'];truth=data['truth']
        ca,pred=sg.masks(maps,seed,truth,{n:.45 for n in sg.NAMES})
        ref_count,_=sg.events(ca,pred['W/REF'],truth)
        for n in sg.NAMES[1:4]:
            candidate_count,_=sg.events(ca,pred[n],truth)
            c=sg.compare(pred['W/REF'],pred[n],truth,ca)
            assert c['tp_retained']+c['tp_lost']==ref_count[0]
            assert c['fp_retained']+c['fp_removed']==ref_count[1]
            assert c['tp_retained']+c['tp_new']==candidate_count[0]
    # Test the real plotting path with the live VOC sample.
    cls,row=rows['voc'][0]
    maps,seed,truth,image,_=live_maps(pipe,ca_modules,selected,ca_spec,row,'voc',args.device)
    ca,pred=sg.masks(maps,seed,truth,{n:.45 for n in sg.NAMES})
    temporary=out/'smoke_visual_input.npz';figure=out/'smoke_visual.png'
    npz(temporary,image=image,truth=truth,seed=seed.reshape(64,64),
        maps=np.stack([maps[n] for n in sg.NAMES]).reshape(5,64,64),
        masks=np.stack([pred[n] for n in sg.NAMES]),ca_mask=ca,names=np.array(sg.NAMES),
        chosen=np.array('W/T3'),diagnostic=np.array(True),
        image_id=np.array(row['image_id']),classname=np.array(cls))
    env=dict(os.environ,MPLCONFIGDIR=str(out/'matplotlib'))
    subprocess.run([str(PLOT_ENV),str(ROOT/'scripts/plot_sg18_temporal_windowed.py'),
                    '--cache',str(temporary),'--output',str(figure)],env=env,check=True)
    temporary.unlink();assert figure.stat().st_size>0
    atomic_json(out/'smoke_validation.json',{'passed':True,'voc_image':rows['voc'][0][1]['image_id'],
                                              'coco_image':rows['coco'][0][1]['image_id']})
    print('SG18 live smoke passed',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=BASE)
    p.add_argument('--model',default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();spec=json.loads(SPEC.read_text())
    locked=json.loads((ROOT/'configs/sg12_crf_sg14_sg15.json').read_text())
    frozen=json.loads((ROOT/'configs/controlled_sa_seed_frozen.json').read_text())
    if not frozen['frozen'] or locked['sa']['module_index']!=15 or locked['sa']['observations']!=248:
        raise ValueError('frozen model contract changed')
    if tuple(spec['names'])!=sg.NAMES or {k:tuple(v) for k,v in spec['windows'].items()}!=sg.WINDOWS:
        raise ValueError('locked SG18 window contract changed')
    if args.smoke:return smoke(args,spec,locked,frozen)
    out=args.output_dir;voc_rows=rows_for('voc');coco_rows=rows_for('coco')
    pipe,ca_modules,selected,ca_spec=make_pipe(args.model,args.device,locked,frozen)
    capture(out,'voc',voc_rows,pipe,ca_modules,selected,ca_spec,args.device,spec)
    score_dataset(out,'voc',voc_rows,spec,locked['thresholds'])
    voc=result_object(out,'voc',voc_rows,locked['thresholds'])
    selection=select_voc(out,voc,spec)
    capture(out,'coco',coco_rows,pipe,ca_modules,selected,ca_spec,args.device,spec)
    score_dataset(out,'coco',coco_rows,spec,locked['thresholds'])
    coco=result_object(out,'coco',coco_rows,locked['thresholds'])
    report=final_report(out,voc,coco,voc_rows,coco_rows,selection,spec)
    visuals(out,voc_rows,report,pipe,ca_modules,selected,ca_spec,args.device)
    benchmark(out,voc_rows[0][1],pipe,ca_modules,selected,ca_spec,args.device,
              selection['illustrative_window'])
    atomic_json(out/'completed.json',{'images':{'voc':len(voc_rows),'coco':len(coco_rows)},
               'frozen':selection['screen']['frozen'],'visual_sheets':len(list((out/'visuals').glob('*.png'))),
               'timing_complete':True,'all_required_stages_complete':True})
    print('SG18 completed',flush=True)

if __name__=='__main__':main()
