#!/usr/bin/env python3
"""Frozen VOC selection, COCO transfer and reduced SG21 diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sg18_temporal_windowed as sg18
from scripts.select_propagation_cv import best_threshold, class_ious, image_ious, pooled_miou, run_screen
from src.segmentation import controlled_cv as cv
from src.segmentation import sg18_temporal as sg
from src.segmentation import sg21_structural as structural

SPEC = ROOT/'configs/sg21_structural.json'
OUT = ROOT/'outputs/segmentation/controlled_attention_cv/sg21_structural'
SG18 = ROOT/'outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed'


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def load_dataset(out: Path, dataset: str, spec: dict, smoke: bool):
    rows = sg18.rows_for(dataset, first_only=smoke)
    folds = cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if dataset=='voc' else {}
    files = list((out/'cache'/dataset).glob('class_*/*.npz'))
    lookup = {path.stem:path for path in files}
    if len(lookup)!=len(files) or len(files)!=len(rows):
        raise ValueError(f'{dataset}: wrong number of unique SG21 image checkpoints')
    all_counts, all_ap, all_mechanism, all_weights, all_head_events, all_drift = [], [], [], [], [], []
    for classname,row in rows:
        path = lookup.get(sg18.key(row['image_id']))
        if path is None:
            raise ValueError(f'{dataset}/{row["image_id"]}: missing SG21 checkpoint')
        with np.load(path,allow_pickle=False) as data:
            meta = json.loads(str(data['meta']))
            if meta['spec']!=spec or meta['dataset']!=dataset or meta['image_id']!=row['image_id'] or meta['class']!=classname:
                raise ValueError(f'{path}: incompatible SG21 checkpoint')
            counts,ap = data['counts'],data['ap']
            mechanism,weights = data['mechanism'],data['weights']
            head_events,drift = data['head_events'],data['t1_replay_mae']
        if counts.shape!=(len(structural.NAMES),len(spec['thresholds']),4) or ap.shape!=(len(structural.NAMES),):
            raise ValueError(f'{path}: wrong count/AP dimensions')
        if mechanism.shape!=(len(structural.NAMES),len(sg.EVENT_NAMES)) or weights.shape!=(4,11,8) or head_events.shape!=(11,8,5):
            raise ValueError(f'{path}: wrong diagnostics dimensions')
        all_counts.append(counts);all_ap.append(ap);all_mechanism.append(mechanism)
        all_weights.append(weights);all_head_events.append(head_events);all_drift.append(float(drift))
    if not smoke and len(rows)!=({'voc':490,'coco':707}[dataset]):
        raise ValueError(f'{dataset}: incomplete dataset')
    result=cv.Results(structural.NAMES,tuple(spec['thresholds']),
                      tuple(r['image_id'] for _,r in rows),tuple(c for c,_ in rows),
                      np.array([folds[r['image_id']] for _,r in rows]) if dataset=='voc' else np.zeros(len(rows),int),
                      np.stack(all_counts).transpose(1,0,2,3),np.stack(all_ap).T,
                      {'dataset':dataset,'spec':spec},(True,))
    return result,np.stack(all_mechanism),np.stack(all_weights),np.stack(all_head_events),np.asarray(all_drift)


def forced_oof(result: cv.Results, name: str):
    ci=result.index(name);ri=result.index('SG18/T1')
    cand=np.zeros((len(result.image_ids),4),np.int64)
    ref=np.zeros_like(cand)
    folds=[]
    for fold in range(5):
        train=result.folds!=fold;held=~train
        ti,_=best_threshold(result,ci,train);rt,_=best_threshold(result,ri,train)
        cand[held]=result.counts[ci,held,ti]
        ref[held]=result.counts[ri,held,rt]
        folds.append({'fold':fold,'candidate_threshold':result.thresholds[ti],
                      'reference_threshold':result.thresholds[rt],
                      'heldout_miou_candidate':pooled_miou(cand[held],np.asarray(result.classes)[held]),
                      'heldout_miou_reference':pooled_miou(ref[held],np.asarray(result.classes)[held])})
    cm=pooled_miou(cand,result.classes);rm=pooled_miou(ref,result.classes)
    ciou=image_ious(cand);riou=image_ious(ref)
    return {'candidate_miou':cm,'reference_miou':rm,'gain':cm-rm,'folds':folds,
            'heldout_fold_wins':sum(f['heldout_miou_candidate']>f['heldout_miou_reference'] for f in folds),
            'paired_mean_image_iou':cv.paired_bootstrap(ciou,riou)}


def full_metrics(result: cv.Results, name: str, threshold: float, ref_threshold: float):
    ci=result.index(name);ri=result.index('SG18/T1')
    ti=result.thresholds.index(threshold);rt=result.thresholds.index(ref_threshold)
    candidate=result.counts[ci,:,ti];reference=result.counts[ri,:,rt]
    keep=np.ones(len(result.image_ids),bool)
    classes=class_ious(result,ci,ti,keep)
    baseline=class_ious(result,ri,rt,keep)
    tp,fp,fn,_=candidate.sum(0)
    return {'images':len(result.image_ids),'threshold':threshold,
            'miou':pooled_miou(candidate,result.classes),
            'gain_vs_t1':pooled_miou(candidate,result.classes)-pooled_miou(reference,result.classes),
            'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),
            'pixel_iou':float(tp/max(tp+fp+fn,1)),
            'mean_image_iou':float(image_ious(candidate).mean()),
            'per_class_iou':classes,
            'per_class_delta_vs_t1':{k:classes[k]-baseline[k] for k in classes},
            'paired_mean_image_iou_vs_t1':cv.paired_bootstrap(image_ious(candidate),image_ious(reference))}


def weight_summary(weights: np.ndarray):
    """[image, candidate, call, head] reduced diagnostics by dataset."""
    answer={}
    for index,name in enumerate(structural.WEIGHTED):
        w=weights[:,index]
        effective=1/np.square(w).sum(-1)
        top=np.argmax(w,axis=-1)
        answer[name]={'effective_heads':{'mean':float(effective.mean()),
                                         'median':float(np.median(effective)),
                                         'min':float(effective.min()),'max':float(effective.max()),
                                         'per_call':[
                                             {'call':call,'mean':float(effective[:,i].mean()),
                                              'median':float(np.median(effective[:,i])),
                                              'min':float(effective[:,i].min()),
                                              'max':float(effective[:,i].max())}
                                             for i,call in enumerate(structural.CALLS)]},
                      'top_head_fraction':(np.bincount(top.ravel(),minlength=8)/top.size).tolist(),
                      'top_head_fraction_per_call':[
                          {'call':call,'fractions':(np.bincount(top[:,i],minlength=8)/len(top)).tolist()}
                          for i,call in enumerate(structural.CALLS)],
                      'fraction_images_head_changes_across_calls':float(np.mean(np.any(top[:,1:]!=top[:,:-1],axis=1))),
                      'fraction_adjacent_call_transitions_change_head':float(np.mean(top[:,1:]!=top[:,:-1])),
                      'weight_mean_by_head':w.mean((0,1)).tolist(),
                      'weight_std_by_head':w.std((0,1)).tolist()}
    return answer


def quartile_validation(voc_weights,weights_by_dataset,events_by_dataset,promoted):
    index=structural.WEIGHTED.index(promoted)
    edges=np.quantile(voc_weights[:,index].ravel(),[.25,.5,.75])
    report={'voc_quartile_cuts':edges.tolist(),'datasets':{}}
    for dataset in ('voc','coco'):
        w=weights_by_dataset[dataset][:,index].ravel()
        e=events_by_dataset[dataset].reshape(-1,5)
        bins=np.searchsorted(edges,w,side='right')
        rows=[]
        for q in range(4):
            picked=e[bins==q]
            totals=picked.sum(0) if len(picked) else np.zeros(5,int)
            tp,fp,far,useful,harmful=map(int,totals)
            rows.append({'quartile':q+1,'observations':len(picked),'tp_add':tp,'fp_add':fp,
                         'fp_gt16':far,'useful_remove':useful,'harmful_remove':harmful,
                         'tp_fp_ratio':float(tp/fp) if fp else None,
                         'distant_fp_fraction':float(far/fp) if fp else None,
                         'useful_change_fraction':float((tp+useful)/(tp+fp+useful+harmful))
                         if tp+fp+useful+harmful else None})
        report['datasets'][dataset]=rows
    return report


def write_class_csv(path,metrics):
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle,lineterminator='\n')
        writer.writerow(('dataset','method','class','iou','delta_vs_t1'))
        for dataset in ('voc','coco'):
            for name in structural.NAMES:
                m=metrics[dataset][name]
                for cls,value in m['per_class_iou'].items():
                    writer.writerow((dataset,name,cls,value,m['per_class_delta_vs_t1'][cls]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    spec=json.loads(SPEC.read_text())
    loaded={dataset:load_dataset(args.output_dir,dataset,spec,args.smoke) for dataset in ('voc','coco')}
    if args.smoke:
        for dataset,(result,mech,w,heads,drift) in loaded.items():
            assert len(result.image_ids)==1 and mech.shape==(1,5,7) and w.shape==(1,4,11,8)
            assert heads.shape==(1,11,8,5) and np.isfinite(drift).all()
        print('SG21 finalizer smoke passed',flush=True)
        return
    voc=loaded['voc'][0];coco=loaded['coco'][0]
    screen=run_screen(voc,'SG18/T1',list(structural.WEIGHTED),
                      {name:4-i for i,name in enumerate(structural.WEIGHTED)})
    forced={name:forced_oof(voc,name) for name in structural.WEIGHTED}
    screen_frozen=screen['frozen']
    frozen=screen_frozen
    promoted=frozen!='SG18/T1' and forced[frozen]['gain']>0 and screen['selection_counts'].get(frozen,0)>=4
    if not promoted:
        frozen='SG18/T1'
    thresholds={}
    full=np.ones(len(voc.image_ids),bool)
    for name in structural.NAMES:
        ti,_=best_threshold(voc,voc.index(name),full)
        thresholds[name]=voc.thresholds[ti]
    metrics={dataset:{name:full_metrics(result,name,thresholds[name],thresholds['SG18/T1'])
                      for name in structural.NAMES}
             for dataset,result in (('voc',voc),('coco',coco))}
    event_totals={dataset:{name:sg.rates(mechanism[:,i].sum(0))
                           for i,name in enumerate(structural.NAMES)}
                  for dataset,(_,mechanism,_,_,_) in loaded.items()}
    weights={dataset:weight_summary(data[2]) for dataset,data in loaded.items()}
    quartiles=None
    if promoted:
        quartiles=quartile_validation(loaded['voc'][2],
                                     {d:loaded[d][2] for d in ('voc','coco')},
                                     {d:loaded[d][3] for d in ('voc','coco')},frozen)
    historical=json.loads((SG18/'results.json').read_text())
    p1={'source':'historical SG18 run; external context, not same-generation SG21 control',
        'voc_oof_miou':historical['selection']['forced_oof']['W/P1']['candidate_miou'],
        'full_voc':historical['metrics']['voc']['W/P1']['miou'],
        'voc_threshold':historical['metrics']['voc']['W/P1']['threshold'],
        'frozen_coco':historical['metrics']['coco']['W/P1']['miou']}
    result={'spec':spec,'images':{'voc':490,'coco':707},
            'selection':{'screen':screen,'forced_oof':forced,'frozen':frozen,'promoted':promoted,
                         'screen_frozen_before_forced_oof_guard':screen_frozen,
                         'promotion_reason':('passed fold, OOF and train-fold magnitude/class-stability guards'
                                             if promoted else 'no candidate passed all frozen promotion guards'),
                         'full_voc_thresholds':thresholds,
                         'reference_oof_miou':next(iter(forced.values()))['reference_miou']},
            'metrics':metrics,'mechanism_fixed_0p45':event_totals,'weights':weights,
            'promoted_weight_quartiles':quartiles,'external_w_p1':p1,
            't1_replay_mae':{dataset:{'mean':float(data[4].mean()),'max':float(data[4].max())}
                             for dataset,data in loaded.items()}}
    out=args.output_dir
    write_json(out/'results.json',result)
    write_json(out/'selection.json',result['selection'])
    write_json(out/'weight_diagnostics.json',{'weights':weights,'promoted_weight_quartiles':quartiles})
    write_class_csv(out/'per_class_iou.csv',metrics)
    lines=['# SG21 structural head weighting','',
           f"Frozen: {frozen}; promoted: {promoted}; fold choices: {[x['chosen'] for x in screen['folds']]}",
           f"Reference T1 OOF mIoU: {result['selection']['reference_oof_miou']:.4f}",'',
           '| Method | VOC forced OOF | Δ vs T1 | Fold votes | Full VOC | VOC threshold | Frozen COCO |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in structural.NAMES:
        f=forced.get(name)
        oof=f['candidate_miou'] if f else result['selection']['reference_oof_miou']
        gain=f['gain'] if f else 0.
        lines.append(f"| {name} | {oof:.4f} | {gain:+.4f} | {screen['selection_counts'].get(name,0)}/5 | "
                     f"{metrics['voc'][name]['miou']:.4f} | {thresholds[name]:.3f} | "
                     f"{metrics['coco'][name]['miou']:.4f} |")
    lines.append(f"| W/P1 (historical) | {p1['voc_oof_miou']:.4f} | — | — | "
                 f"{p1['full_voc']:.4f} | {p1['voc_threshold']:.3f} | {p1['frozen_coco']:.4f} |")
    lines+=['','W/P1 is external context from SG18, generated in a separate run. '
            'Full VOC and COCO precision/recall, pixel and mean-image IoU, all 20 class IoUs/deltas, '
            '20,000-resample paired intervals and better/worse/tied image counts are in `results.json`. '
            'Fixed-0.45 CA-relative events, per-call effective head counts, winning physical heads, '
            'weight spread and any promoted weight-quartile diagnostics are also there.','']
    (out/'report.md').write_text('\n'.join(lines))
    write_json(out/'completed.json',{'images':{'voc':490,'coco':707},'frozen':frozen,
                                      'promoted':promoted,'all_required_stages_complete':True})
    print('SG21 finalization completed',flush=True)


if __name__=='__main__':
    main()
