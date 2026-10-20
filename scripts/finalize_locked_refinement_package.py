#!/usr/bin/env python3
"""Same-generation checks and VOC-only selection for the locked refinement package."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.segmentation import controlled_cv as cv
from src.segmentation import locked_refinements as lr
from src.segmentation.experiment_records import NumericSidecar
from scripts.select_propagation_cv import run_screen,best_threshold,pooled_miou,image_ious,class_ious
from scripts.narmyn.attention.an1_propagation_depth import rates
from scripts.run_coco_cap_transfer import voc_rows,coco_rows

BASE=ROOT/'outputs/segmentation/controlled_attention_cv'
SPEC=ROOT/'configs/sg12_crf_sg14_sg15.json'


def load(directory,dataset,stage,spec,smoke=False):
    paths=sorted((directory/dataset/'shards').glob('class_[0-9][0-9].npz'))
    result=cv.load_results(paths)
    if result.configurations!=lr.names_for(stage) or result.thresholds!=tuple(spec['thresholds']):raise ValueError('configuration contract differs')
    if result.settings['spec']!=spec or result.settings['stage']!=stage:raise ValueError('spec differs')
    expected={r['image_id']:(classname,r) for i in range(20) for classname,rows in [(voc_rows if dataset=='voc' else coco_rows)(i)] for r in rows}
    if not smoke and (len(paths)!=20 or not all(result.complete) or set(result.image_ids)!=set(expected)):
        raise ValueError(f'{dataset}: incomplete/foreign manifest')
    folds=cv.read_folds(ROOT/'configs/voc_sim_folds.csv')
    diagnostics={}
    for path in paths:
        diag=NumericSidecar(path.with_suffix('.diagnostics.npz'),result.settings)
        for image_id,record in diag.records.items():
            if image_id in diagnostics:raise ValueError('duplicate diagnostic id')
            diagnostics[image_id]=record
    if set(diagnostics)!=set(result.image_ids):raise ValueError('sidecar/result IDs differ')
    for i,image_id in enumerate(result.image_ids):
        if result.classes[i]!=expected[image_id][0]:raise ValueError('class differs')
        if dataset=='voc' and result.folds[i]!=folds[image_id]:raise ValueError('fold differs')
        d=diagnostics[image_id]
        if not np.isfinite(d['audit_errors']).all() or np.max(d['audit_errors'])>2e-6:raise ValueError('same-generation operator audit failed')
        if len(d['selected_indices'])!=124 or len(np.unique(d['selected_indices']))!=124:raise ValueError('wrong observation selection')
        rank=np.argsort(-d['reliability'],kind='stable')[:124]
        if not np.array_equal(rank,d['selected_indices']):raise ValueError('ranking mismatch')
        c=result.counts[:,i]
        if np.any(c<0) or not np.all(c.sum(-1)==c[0,0].sum()):raise ValueError('area invariant failed')
        if not np.all(c[:,:,0]+c[:,:,2]==c[0,0,0]+c[0,0,2]):raise ValueError('GT invariant failed')
    return result,diagnostics


def threshold_index(result,name,mask,sources):
    return best_threshold(result,result.index(sources.get(name,name)),mask)[0]


def oof_counts(result,name,sources):
    out=np.zeros((len(result.image_ids),4),np.int64);rows=[]
    for f in range(5):
        held=result.folds==f
        ti=threshold_index(result,name,~held,sources)
        out[held]=result.counts[result.index(name),held,ti]
        rows.append({'fold':f,'threshold':result.thresholds[ti],
                     'heldout_miou':pooled_miou(out[held],np.asarray(result.classes)[held])})
    return out,rows


def comparison(a,b,classes):
    return {'candidate_miou':pooled_miou(a,classes),'reference_miou':pooled_miou(b,classes),
            'delta':pooled_miou(a,classes)-pooled_miou(b,classes),
            'paired_mean_image_iou':cv.paired_bootstrap(image_ious(a),image_ious(b),resamples=20000)}


def full_metrics(counts,classes):
    out=rates(counts,np.asarray(classes));out['mean_image_iou']=float(image_ious(counts).mean())
    tp,fp,fn,tn=counts.sum(0)
    out.update(pixel_iou=float(tp/max(tp+fp+fn,1)),pixel_recall=float(tp/max(tp+fn,1)),pixel_precision=float(tp/max(tp+fp,1)))
    return out


def region_rates(values):
    fg,tp,fp,fn,pos=values.sum(0)
    precision=float(tp/max(tp+fp,1));recall=float(tp/max(tp+fn,1))
    return {'foreground_regions_per_image':float(values[:,0].mean()),'region_precision':precision,'region_recall':recall,
            'region_f1':2*precision*recall/max(precision+recall,1e-30),
            'oracle_positive_regions_missed':int(fn),'oracle_negative_regions_selected':int(fp),'oracle_positive_regions':int(pos)}


def historical_drift(result,dataset,stage):
    packages=[('sg11_sg12_observation_reliability',lr.REF),('sg13_target_region_inference',lr.REF)]
    output={}
    for package,name in packages:
        paths=[p for p in sorted((BASE/package/dataset/'shards').glob('class_*.npz')) if '.diagnostics.' not in p.name and '.regions.' not in p.name]
        old=cv.load_results(paths);lookup={x:i for i,x in enumerate(old.image_ids)}
        ids=[lookup[x] for x in result.image_ids]
        a=result.counts[result.index(name)]
        ti=[old.thresholds.index(t) for t in result.thresholds]
        b=old.counts[old.index(name),ids][:,ti]
        delta=np.abs(a-b)
        output[package]={'max_pixels':int(delta.max()),'max_fraction':float((delta/a.sum(-1)[...,None]).max()),
                         'diagnostic_only':True,'not_claiming_identical_historical_masks':True}
    return output


def mechanism(stage,dataset,result,diagnostics,thresholds):
    rows=[diagnostics[i] for i in result.image_ids];out={}
    if stage=='sg12_crf':
        ti=thresholds[lr.CRF_CAND];rawti=thresholds[lr.SG12]
        ch=np.stack([r['crf_changes'][ti] for r in rows]).sum(0)
        spatial=np.stack([r['crf_spatial'][ti] for r in rows]).sum(0)
        out={'same_input_threshold':result.thresholds[ti],
             'FN_to_TP':int(ch[0]),'FP_to_TN':int(ch[1]),'TP_to_FN':int(ch[2]),'TN_to_FP':int(ch[3]),
             'new_fp_attached':int(spatial[0]),'new_fp_separate_component':int(spatial[1]),
             'new_fp_more_than_16px_from_GT':int(spatial[2]),'new_fp_more_than_64px_from_GT':int(spatial[3]),
             'boundary_error_pixels_before_after':spatial[4:6].tolist(),'nonboundary_error_pixels_before_after':spatial[6:8].tolist(),
             'raw_to_crf_same_threshold':comparison(result.counts[result.index(lr.CRF_CAND),:,ti],result.counts[result.index(lr.SG12),:,ti],result.classes),
             'raw_to_crf_individually_VOC_selected_thresholds':comparison(result.counts[result.index(lr.CRF_CAND),:,ti],result.counts[result.index(lr.SG12),:,rawti],result.classes)}
        gen=np.stack([r['generation_seed_seconds'] for r in rows]);method=np.stack([r['method_seconds'] for r in rows])
        crf=np.stack([r['crf_seconds'] for r in rows])
        out['runtime']={'images':len(rows),'shared_generation_and_seed_mean_seconds':gen.mean(0).tolist(),
            'ordinary_two_pass_mean_seconds':float(method[:,0].mean()),'SG12_observation_propagation_mean_seconds':float(method[:,1].mean()),
            'SG12_ranking_mean_seconds':float(method[:,2].mean()),'SG12_selected_average_mean_seconds':float(method[:,3].mean()),
            'crf_reference_mean_seconds':float(crf[:,0,thresholds[lr.CRF_REF]].mean()),'crf_candidate_mean_seconds':float(crf[:,1,ti].mean()),
            'reference_inclusive_per_image_latency_mean_seconds':float((gen.sum(1)+method[:,0]+crf[:,0,thresholds[lr.CRF_REF]]).mean()),
            'candidate_inclusive_per_image_latency_mean_seconds':float((gen.sum(1)+method[:,1:4].sum(1)+crf[:,1,ti]).mean()),
            'measurement_scope':'synchronized isolated operators plus shared live generation/CA seed and one selected-threshold CRF; excludes model loading/metric/report overhead; shared generation is charged in full to each. Not independent end-to-end job benchmarks: no speedup claim.'}
    elif stage=='sg14':
        ti=result.thresholds.index(.45)
        out={'diagnostic_threshold':.45,'group_names':['SG12_only','SG11_only','both_same_direction','both_opposite_directions','neither'],
             'transition_methods':[lr.SG11,lr.SG12,lr.SG14],
             'change_order':['FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP'],
             'binary_group_changes_at_0p45':np.stack([r['binary_group_changes'][ti] for r in rows]).sum(0).tolist(),
             'binary_group_pixels_at0p45':np.stack([r['binary_group_pixels'][ti] for r in rows]).sum(0).tolist(),
             'continuous_group_changes_at0p45':np.stack([r['continuous_group_changes'][ti] for r in rows]).sum(0).tolist(),
             'continuous_full_resolution_group_pixels':np.stack([r['continuous_full_group_pixels'] for r in rows]).sum(0).tolist(),
             'continuous_raw_group_pixels':np.stack([r['continuous_group_pixels'] for r in rows]).sum(0).tolist(),
             'binary_opposite_direction_note':'Two binary predictions cannot move in opposite directions from the same binary REF; that group is exactly zero. Continuous raw-map directions are recorded separately.',
             'mean_consensus_fractions':np.stack([r['consensus_fractions'] for r in rows]).mean(0).tolist(),
             'consensus_fraction_order':['observations_rejected','all_observation_pixels_gate_rejected','selected_observation_pixels_gate_rejected','gate_membership_changed'],
             'mean_per_image_reliability_correlations':np.stack([r['reliability_correlations'] for r in rows]).mean(0).tolist(),
             'correlation_order':['raw_upward_on_GT_downward_on_BG_usefulness','raw_harmfulness','agreement_with_useful_REF_vs_CA_flips_at0p45','agreement_with_harmful_REF_vs_CA_flips_at0p45'],
             'all_threshold_group_counts_retained_in_sidecars':True}
        for name in (lr.SG11,lr.SG12,lr.SG14):
            ti=thresholds[name]
            out.setdefault('group_changes_at_each_VOC_selected_threshold',{})[name]=np.stack([r['binary_group_changes'][ti] for r in rows]).sum(0).tolist()
    else:
        oracle=np.stack([r['oracle_counts'] for r in rows]);oracle_mean=float(image_ious(oracle).mean())
        legacy=np.stack([r['legacy_sg13_oracle_counts'] for r in rows])
        out={'oracle_mean_image_iou':oracle_mean,'legacy_SG13_nearest64_GT_oracle_mean_image_iou':float(image_ious(legacy).mean()),
             'oracle_note':'Main diagnostic GT fractions are evaluated over annotation-resolution regions, as are prior occupancies. Legacy nearest64-GT SG13 oracle is also retained, never used by prediction.',
             'region_selected_definition':'at least 50% output occupancy in nonempty region; for majority projection equals its chosen label, for B a diagnostic summary of mixed regions',
             'cells':{}}
        for j,name in enumerate(result.configurations[4:]):
            prior=lr.PRIORS[j//4];ti=thresholds[prior]
            before=result.counts[result.index(prior),:,ti];after=result.counts[result.index(name),:,ti]
            bi=float(image_ious(before).mean());ai=float(image_ious(after).mean());gap=oracle_mean-bi
            ch=np.stack([r['region_changes'][j,ti] for r in rows]).sum(0)
            cell={'region_quality':region_rates(np.stack([r['region_stats'][j,ti] for r in rows])),
                  'prior_region_quality':region_rates(np.stack([r['prior_region_stats'][j//4,ti] for r in rows])),
                  'unrefined_mean_image_iou':bi,'refined_mean_image_iou':ai,'oracle_gap_before':gap,'oracle_gap_after':oracle_mean-ai,
                  'fraction_oracle_gap_closed':(ai-bi)/gap if abs(gap)>1e-12 else None,
                  'changes':{}}
            for group,values in zip(('all','high_FG','high_BG','mixed'),ch):
                cell['changes'][group]=dict(zip(('FN_to_TP','FP_to_TN','TP_to_FN','TN_to_FP'),map(int,values)))
                cell['changes'][group].update(useful=int(values[0]+values[1]),harmful=int(values[2]+values[3]))
            out['cells'][name]=cell
    return out


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('sg12_crf','sg14','sg15'),required=True)
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();directory=args.output_dir or BASE/args.stage;spec=json.loads(SPEC.read_text())
    voc,vd=load(directory,'voc',args.stage,spec,args.smoke);coco,cd=load(directory,'coco',args.stage,spec,args.smoke)
    drift={ds:historical_drift(r,ds,args.stage) for ds,r in (('voc',voc),('coco',coco))}
    if args.smoke:
        summary={'passed':True,'voc_images':len(voc.image_ids),'coco_images':len(coco.image_ids),'historical_drift':drift,
                 'same_generation_audits':'passed; frozen original parent operators and top50 identities checked on live inputs'}
        (directory/'smoke_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return
    sources=lr.threshold_sources(args.stage)
    if args.stage=='sg12_crf':screens={'primary':run_screen(voc,lr.CRF_REF,[lr.CRF_CAND],{})}
    elif args.stage=='sg14':
        screens={f'versus_{p}':run_screen(voc,p,[lr.SG14],{}) for p in (lr.REF,lr.SG11,lr.SG12)}
        screens['joint']=run_screen(voc,lr.REF,[lr.SG11,lr.SG12,lr.SG14],{lr.SG11:2,lr.SG12:1,lr.SG14:0})
    else:
        screens={}
        for prior in lr.PRIORS:
            for family,kinds in [('A',('A025','A050','A075')),('B',('B',))]:
                screens[f'{prior}_SG15-{family}']=run_screen(voc,prior,[lr.refined_name(prior,k) for k in kinds],
                    {lr.refined_name(prior,'A050'):2,lr.refined_name(prior,'A075'):1},threshold_sources=sources)
    oof={};oof_arrays={};fullmask=np.ones(len(voc.image_ids),bool)
    thresholds={n:threshold_index(voc,n,fullmask,sources) for n in voc.configurations}
    for name in voc.configurations:
        a,rows=oof_counts(voc,name,sources);oof_arrays[name]=a
        oof[name]={'miou':pooled_miou(a,voc.classes),'folds':rows,'mean_image_iou':float(image_ious(a).mean())}
    comparisons={}
    for key,screen in screens.items():
        comparisons[key]={candidate:comparison(oof_arrays[candidate],oof_arrays[screen['reference']],voc.classes) for candidate in screen['candidates']}
    cells={};classrows=[];imagerows=[]
    for name in voc.configurations:
        ti=thresholds[name];reference=(sources.get(name) or (lr.CRF_REF if args.stage=='sg12_crf' else lr.REF))
        cells[name]={'voc_threshold':voc.thresholds[ti],'reference':reference}
        for ds,result in [('voc',voc),('coco',coco)]:
            a=result.counts[result.index(name),:,ti];b=result.counts[result.index(reference),:,thresholds[reference]]
            values=class_ious(result,result.index(name),ti,np.ones(len(result.image_ids),bool))
            baseline=class_ious(result,result.index(reference),thresholds[reference],np.ones(len(result.image_ids),bool))
            cells[name][ds]={'metrics':full_metrics(a,result.classes),'per_class_iou':values,
                             'per_class_delta':{c:values[c]-baseline[c] for c in values},'versus_matching_reference':comparison(a,b,result.classes)}
            for c in values:classrows.append({'dataset':ds,'cell':name,'reference':reference,'class':c,'threshold':voc.thresholds[ti],
                                             'iou':values[c],'delta':values[c]-baseline[c]})
            for i,image_id in enumerate(result.image_ids):imagerows.append({'dataset':ds,'cell':name,'reference':reference,'image_id':image_id,
                             'class':result.classes[i],'image_iou':image_ious(a)[i],'reference_image_iou':image_ious(b)[i],
                             'delta':image_ious(a)[i]-image_ious(b)[i]})
    diagnostics={ds:mechanism(args.stage,ds,r,d,thresholds) for ds,r,d in [('voc',voc,vd),('coco',coco,cd)]}
    summary={'spec':spec,'stage':args.stage,'voc_images':len(voc.image_ids),'coco_images':len(coco.image_ids),
             'environments':{'voc':voc.environments,'coco':coco.environments},'historical_drift':drift,
             'screens':screens,'forced_cell_oof':oof,'forced_oof_comparisons':comparisons,'cells':cells,'mechanism':diagnostics,
             'validation':'Same-generation audited controls; historical cross-hardware discrepancies are diagnostic only. Does not certify historical bit-identical reproduction.',
             'selection_note':'VOC only selects. SG15 source thresholds selected on four training folds; refined outputs inherit them. Full-VOC choices frozen before COCO metrics are interpreted. Diagnostic variants never promoted on COCO.'}
    if args.stage=='sg14':
        summary['full_dataset_pairwise']={ds:{p:comparison(r.counts[r.index(lr.SG14),:,thresholds[lr.SG14]],r.counts[r.index(p),:,thresholds[p]],r.classes) for p in (lr.REF,lr.SG11,lr.SG12)} for ds,r in [('voc',voc),('coco',coco)]}
    for filename,value in [('summary.json',summary),('fold_results.json',{'screens':screens,'forced_cell_oof':oof,'forced_oof_comparisons':comparisons})]:
        (directory/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    write_csv(directory/'per_class_results.csv',classrows);write_csv(directory/'per_image_results.csv',imagerows)
    lines=[f'# Locked {args.stage} results','',summary['selection_note'],'',summary['validation'],'',
           '| Cell | VOC OOF mIoU | Full VOC threshold | Full VOC mIoU | COCO mIoU |', '|---|---:|---:|---:|---:|']
    for name,cell in cells.items():lines.append(f"| {name} | {oof[name]['miou']:.6f} | {cell['voc_threshold']:.3f} | {cell['voc']['metrics']['miou']:.6f} | {cell['coco']['metrics']['miou']:.6f} |")
    lines+=['','## Protocol selection','','| Screen | Winner | Votes | Selected OOF / REF | Gain | Frozen |','|---|---|---:|---:|---:|---|']
    for key,s in screens.items():lines.append(f"| {key} | {s['vote_winner']} | {s['vote_winner_votes']}/5 | {s['oof_miou_selection']:.6f} / {s['oof_miou_reference']:.6f} | {s['oof_gain']:+.6f} | {s['frozen']} |")
    lines+=['','Forced candidate OOF and protocol-selected OOF are distinct. All five fold winners, thresholds, class IoUs, deltas, paired 20k-bootstrap intervals and improved/worsened/tied counts are in summary.json and fold_results.json.','',
            '## Mechanism diagnostics','', '```json',json.dumps(diagnostics,indent=2,allow_nan=False),'```','']
    report='\n'.join(lines)
    (directory/'report.md').write_text(report);(directory/'CHATGPT_RESULTS.md').write_text(report+'\n## All class results\n\n'+(directory/'per_class_results.csv').read_text())
    print('\n'.join(lines[:len(cells)+8]),flush=True)

if __name__=='__main__':main()
