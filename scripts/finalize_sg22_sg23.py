#!/usr/bin/env python3
"""Five-fold VOC selection, frozen COCO transfer, and factorial diagnostics."""
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
from scripts.select_propagation_cv import best_threshold, class_ious, image_ious, pooled_miou, run_screen
from src.segmentation import controlled_cv as cv
from src.segmentation import sg22_sg23 as operators
from src.segmentation import sg18_temporal as temporal
from scripts.run_sg22_sg23 import DIAG_FIELDS

BASE = ROOT/'outputs/segmentation/controlled_attention_cv'


def load(out, dataset, spec, smoke):
    files = sorted(p for p in (out/dataset/'shards').glob('class_*.npz')
                   if not p.name.endswith(('.diagnostics.npz','.tmp.npz')))
    result = cv.load_results(files)
    if result.configurations != tuple(spec['names']) or result.thresholds != tuple(spec['thresholds']):
        raise ValueError('unexpected cells or thresholds')
    if result.settings['spec'] != spec or result.settings['dataset'] != dataset:
        raise ValueError('foreign result settings')
    if not smoke:
        expected = 490 if dataset == 'voc' else 707
        if len(files) != 20 or len(result.image_ids) != expected or not all(result.complete):
            raise ValueError(f'{dataset} missing complete class shards')
    folds = cv.read_folds(ROOT/'configs/voc_sim_folds.csv') if dataset == 'voc' else None
    if folds is not None and any(result.folds[i] != folds[image_id] for i,image_id in enumerate(result.image_ids)):
        raise ValueError('VOC fold mismatch')
    diag = {}
    for file in files:
        path = file.with_suffix('.diagnostics.npz')
        with np.load(path,allow_pickle=False) as data:
            for image_id,events,changes in zip(data['image_ids'],data['events'],data['changes']):
                if str(image_id) in diag: raise ValueError('duplicate diagnostic')
                diag[str(image_id)] = (events,changes)
    if set(diag) != set(result.image_ids):
        raise ValueError('diagnostic coverage mismatch')
    return result, np.stack([diag[i][0] for i in result.image_ids]), np.stack([diag[i][1] for i in result.image_ids])


def forced_oof(result, name, reference):
    a = np.zeros((len(result.image_ids),4),np.int64)
    b = np.zeros_like(a)
    folds=[]
    for fold in range(5):
        held=result.folds==fold;train=~held
        ta,_=best_threshold(result,result.index(name),train)
        tb,_=best_threshold(result,result.index(reference),train)
        a[held]=result.counts[result.index(name),held,ta]
        b[held]=result.counts[result.index(reference),held,tb]
        folds.append({'fold':fold,'candidate_threshold':result.thresholds[ta],
                      'reference_threshold':result.thresholds[tb],
                      'candidate_miou':pooled_miou(a[held],np.asarray(result.classes)[held]),
                      'reference_miou':pooled_miou(b[held],np.asarray(result.classes)[held])})
    return {'miou':pooled_miou(a,result.classes),'reference_miou':pooled_miou(b,result.classes),
            'delta':pooled_miou(a,result.classes)-pooled_miou(b,result.classes),
            'fold_wins':sum(f['candidate_miou']>f['reference_miou'] for f in folds),
            'folds':folds,'paired_mean_image_iou':cv.paired_bootstrap(image_ious(a),image_ious(b))}


def metrics(result,name,reference,thresholds):
    t=thresholds[name];rt=thresholds[reference]
    counts=result.counts[result.index(name),:,t]
    ref=result.counts[result.index(reference),:,rt]
    tp,fp,fn,_=counts.sum(0)
    keep=np.ones(len(result.image_ids),bool)
    pc=class_ious(result,result.index(name),t,keep)
    pr=class_ious(result,result.index(reference),rt,keep)
    return {'images':len(result.image_ids),'voc_frozen_threshold':result.thresholds[t],
            'miou':pooled_miou(counts,result.classes),'delta_vs_reference':pooled_miou(counts,result.classes)-pooled_miou(ref,result.classes),
            'precision':float(tp/max(tp+fp,1)),'recall':float(tp/max(tp+fn,1)),
            'pixel_iou':float(tp/max(tp+fp+fn,1)),'mean_image_iou':float(image_ious(counts).mean()),
            'per_class_iou':pc,'per_class_delta_vs_reference':{k:pc[k]-pr[k] for k in pc},
            'paired_mean_image_iou_vs_reference':cv.paired_bootstrap(image_ious(counts),image_ious(ref))}


def add_interactions(cells,oof,kind):
    values={}
    for method in ('SG2','SG11','SG12'):
        def gain(w):
            n=f'{w}-{method}'+kind;r=f'{w}-R0'+kind
            if oof: return cells[n]['miou']-cells[r]['miou']
            return cells[n]['miou']-cells[r]['miou']
        a,b=gain('ALL'),gain('MID')
        values[method]={'gain_all':a,'gain_mid':b,'interaction':b-a}
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=('sg22','sg23'),required=True)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    spec=json.loads((ROOT/f'configs/{args.experiment}_factorial.json').read_text())
    out=args.output_dir or BASE/args.experiment
    voc,ve,vc=load(out,'voc',spec,args.smoke)
    coco,ce,cc=load(out,'coco',spec,args.smoke)
    if args.smoke:
        if len(voc.image_ids)!=1 or len(coco.image_ids)!=1: raise ValueError('smoke expects one VOC and one COCO image')
        (out/'smoke_validation.json').write_text(json.dumps({'passed':True,'experiment':args.experiment,
            'cells':list(voc.configurations),'voc_image':voc.image_ids[0],'coco_image':coco.image_ids[0]},indent=2)+'\n')
        print(f'{args.experiment} live VOC/COCO smoke passed',flush=True)
        return
    rawref='ALL-P2' if args.experiment=='sg22' else 'MID-R0'
    crfref='ALL-P2+R2'
    raw=[n for n in spec['names'] if not n.endswith('+R2') and n!=operators.SG23_AUX]
    if args.experiment == 'sg22':
        raw_candidates=['MID-P2','MID-P3','ALL-P3']
        crf_candidates=[n+'+R2' for n in raw_candidates]
        preference={'ALL-P3':31,'MID-P2':11.1,'MID-P3':11}
    else:
        raw_candidates=['MID-SG2','MID-SG11','MID-SG12']
        crf_candidates=[n+'+R2' for n in operators.SG23_RAW]
        preference={f'{w}-{m}':(31 if w=='ALL' else 11)+{'R0':.4,'SG2':.3,'SG11':.2,'SG12':.1}[m]
                    for w in ('ALL','MID') for m in ('R0','SG2','SG11','SG12')}
    raw_screen=run_screen(voc,rawref,raw_candidates,preference)
    crf_screen=run_screen(voc,crfref,crf_candidates,{n+'+R2':v for n,v in preference.items()})
    keep=np.ones(len(voc.image_ids),bool)
    thresholds={n:best_threshold(voc,voc.index(n),keep)[0] for n in spec['names']}
    cells={'voc':{},'coco':{}}
    per_class=[]
    for dataset,result in (('voc',voc),('coco',coco)):
        for n in spec['names']:
            ref=crfref if n.endswith('+R2') else rawref
            cells[dataset][n]=metrics(result,n,ref,thresholds)
            for cls,iou in cells[dataset][n]['per_class_iou'].items():
                per_class.append({'dataset':dataset,'method':n,'class':cls,'iou':iou,
                    'delta_vs_reference':cells[dataset][n]['per_class_delta_vs_reference'][cls]})
    oof={n:forced_oof(voc,n,crfref if n.endswith('+R2') else rawref) for n in spec['names']}
    diagnostics={}
    for dataset,events,changes in (('voc',ve,vc),('coco',ce,cc)):
        diagnostics[dataset]={'ca_relative':{n:temporal.rates(events[:,i].sum(0)) for i,n in enumerate(raw)},
                              'transitions':{}}
        pairs=([f'{w}-P{d}_to_P{d+1}' for w in ('ALL','MID') for d in (1,2)] if args.experiment=='sg22'
               else [f'MID-R0_to_{m}' for m in ('SG2','SG11','SG12')])
        for i,pair in enumerate(pairs):
            row={k:int(v) for k,v in zip(DIAG_FIELDS,changes[:,i].sum(0))}
            if args.experiment == 'sg23':
                reference = events[:,4].sum(0)  # MID-R0, in SG23_RAW order.
                row['tp_additions_retained'] = int(reference[0]-row['tp_lost'])
                row['fp_additions_retained'] = int(reference[1]-row['fp_removed'])
            row['fp_removed_per_tp_lost']=float(row['fp_removed']/row['tp_lost']) if row['tp_lost'] else None
            diagnostics[dataset]['transitions'][pair]=row
    interactions={}
    if args.experiment=='sg23':
        for suffix in ('','+R2'):
            interactions['raw' if not suffix else 'crf']={
                'voc_oof':add_interactions(oof,oof=True,kind=suffix),
                'full_voc':add_interactions(cells['voc'],oof=False,kind=suffix),
                'frozen_coco':add_interactions(cells['coco'],oof=False,kind=suffix)}
    summary={'spec':spec,'selection':{'raw':raw_screen,'crf':crf_screen},'forced_oof':oof,
             'cells':cells,'diagnostics_fixed_0p45':diagnostics,'interactions':interactions,
             'voc_thresholds_frozen_for_coco':True,'same_generation_references':True}
    out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'per_class_iou.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_class[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(per_class)
    lines=[f'# {args.experiment.upper()} locked factorial', '',
           f"Raw VOC fold votes: {raw_screen['selection_counts']}; frozen: {raw_screen['frozen']}",
           f"CRF VOC fold votes: {crf_screen['selection_counts']}; frozen: {crf_screen['frozen']}", '',
           '| Cell | VOC OOF | Δ OOF | Fold wins | Full VOC | VOC threshold | Frozen COCO |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for n in spec['names']:
        item=oof[n]
        lines.append(f"| {n} | {item['miou']:.4f} | {item['delta']:+.4f} | {item['fold_wins']}/5 | "
                     f"{cells['voc'][n]['miou']:.4f} | {cells['voc'][n]['voc_frozen_threshold']:.3f} | {cells['coco'][n]['miou']:.4f} |")
    lines.extend(['','Full metrics, 20-class deltas, 20k paired intervals, fixed-threshold transitions, '
                  'and SG23 interactions are in results.json.',''])
    (out/'report.md').write_text('\n'.join(lines))
    (out/'completed.json').write_text(json.dumps({'voc_images':len(voc.image_ids),'coco_images':len(coco.image_ids),
         'all_required_stages_complete':True,'raw_frozen':raw_screen['frozen'],'crf_frozen':crf_screen['frozen']},indent=2)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
