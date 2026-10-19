#!/usr/bin/env python3
"""SG13 selection, untouched transfer, region errors and oracle-gap analysis."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.segmentation import controlled_cv as cv
from scripts.select_propagation_cv import best_threshold, run_screen, class_ious, pooled_miou, image_ious
from scripts.narmyn.attention.an1_propagation_depth import rates

BASE=ROOT/'outputs/segmentation/controlled_attention_cv'
OUT=BASE/'sg13_target_region_inference'
SPEC=ROOT/'configs/sg13_target_region_inference.json'


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load(directory):
    return cv.load_results([p for p in sorted((directory/'shards').glob('*.npz')) if '.regions.' not in p.name and '.diagnostics.' not in p.name])


def validate(directory,dataset,spec,smoke):
    result=load(directory/dataset)
    old=load(BASE/'sg4_region_labels'/dataset)
    if result.settings.get('partition_validation') != 'canonical_sg4_same_generation_v2':
        raise ValueError('old or unaudited SG13 checkpoint; use a fresh corrected run directory')
    if result.configurations!=tuple(spec['configurations']) or result.settings['spec']!=spec:
        raise ValueError('SG13 settings/configuration contract differs')
    if result.thresholds!=tuple(round(i*.005,3) for i in range(201)):
        raise ValueError('SG13 threshold grid differs')
    if not smoke and (len(result.complete)!=20 or not all(result.complete) or set(result.image_ids)!=set(old.image_ids)):
        raise ValueError('need all 20 complete shards and exact original dataset IDs')
    order=[old.image_ids.index(n) for n in result.image_ids]
    if dataset=='voc' and not np.array_equal(result.folds,old.folds[order]):
        raise ValueError('fold assignments differ')
    checks={}
    for name,historical,source in [('W/REF','W/REF',old),('W/SG4/SPEC32_rmean','W/SPEC32_rmean',old),
                                   ('W/P1','W/P1',load(BASE/'sg2_consensus_edges'/dataset)),
                                   ('W/C/rho0p80','W/C/rho0p80',load(BASE/'sg8_observation_consensus'/dataset))]:
        order=[source.image_ids.index(n) for n in result.image_ids]
        new_indices=[result.thresholds.index(t) for t in source.thresholds]
        a=result.counts[result.index(name)][:,new_indices]
        b=source.counts[source.index(historical),order]
        error=float(np.max(np.abs(a-b)/np.maximum(a.sum(-1)[...,None],1)))
        # Across environments spectral clustering is not pixel-stable. Keep
        # historical drift visible; correctness is enforced on the same live
        # generation/partition by the canonical operator audits below.
        previous_limit=.01 if name.startswith('W/SG4') else .002
        checks[name]={'max_count_difference':int(np.abs(a-b).max()),'max_fraction_of_image':error,
                      'previous_historical_limit':previous_limit,'previous_limit_exceeded':bool(error>previous_limit),
                      'role':'cross-environment diagnostic; not a same-input identity check'}
    return result,checks


def load_regions(directory,dataset,result,spec):
    records={}
    for path in sorted((directory/dataset/'shards').glob('*.regions.npz')):
        with np.load(path,allow_pickle=False) as data:
            if json.loads(str(data['spec']))!=spec:
                raise ValueError('region specification differs')
            for i,image_id in enumerate(data['image_ids'].tolist()):
                if image_id in records:raise ValueError('duplicate region record')
                records[image_id]={k:data[k][i] for k in ('quantities','scores','area64','gt_fraction','oracle_counts','operator_audit_errors')}
    if set(records)!=set(result.image_ids):raise ValueError('metric/region image IDs differ')
    audit=np.stack([records[n]['operator_audit_errors'] for n in result.image_ids])
    if not np.isfinite(audit).all() or np.max(audit)>1.2e-5:
        raise ValueError('same-generation SG4/linear operator audit failed')
    return {k:np.stack([records[n][k] for n in result.image_ids]) for k in next(iter(records.values()))}


def region_statistics(scores,area,fraction,thresholds):
    valid=area>0
    truth=(fraction>=.5)&valid
    prediction=(scores>=np.asarray(thresholds)[:,None])&valid
    tp=(prediction&truth).sum(1);fp=(prediction&~truth).sum(1);fn=(~prediction&truth).sum(1)
    metrics={'mean_fg_regions':float(prediction.sum(1).mean()),'mean_occupied_regions':float(valid.sum(1).mean()),
             'mean_gt_positive_regions':float(truth.sum(1).mean()),'region_precision':float(tp.sum()/max((tp+fp).sum(),1)),
             'region_recall':float(tp.sum()/max((tp+fn).sum(),1)),
             'oracle_positive_missed':int(fn.sum()),'oracle_negative_selected':int(fp.sum()),
             'true_positive_regions':int(tp.sum())}
    rows=[{'predicted_fg_regions':int(prediction[i].sum()),'gt_positive_regions':int(truth[i].sum()),
           'true_positive_regions':int(tp[i]),'false_inclusion_regions':int(fp[i]),'missed_positive_regions':int(fn[i])}
          for i in range(len(scores))]
    return metrics,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    spec=json.loads(SPEC.read_text())
    voc,vc=validate(args.output_dir,'voc',spec,args.smoke)
    coco,cc=validate(args.output_dir,'coco',spec,args.smoke)
    regions={d:load_regions(args.output_dir,d,r,spec) for d,r in [('voc',voc),('coco',coco)]}
    if args.smoke:
        dump(args.output_dir/'smoke_validation.json',{'voc':vc,'coco':cc,'passed':True, 'same_generation_audit_maxima':{d:regions[d]['operator_audit_errors'].max(0).tolist() for d in regions}})
        print('SG13 live checks passed: reference curves, SPEC32 and region checkpoints')
        return
    candidates=list(voc.configurations[4:])
    allowed={n:([voc.thresholds.index(t) for t in spec['control_thresholds']] if n not in candidates else list(range(201))) for n in voc.configurations}
    screens={rule:run_screen(voc,'W/REF',[name],{},threshold_indices=allowed) for rule,name in zip(('R1','R2','R3'),candidates)}
    screens['joint']=run_screen(voc,'W/REF',candidates,{n:3-i for i,n in enumerate(candidates)},threshold_indices=allowed)
    full=np.ones(len(voc.image_ids),bool)
    threshold_indices={n:best_threshold(voc,voc.index(n),full,allowed[n])[0] for n in voc.configurations}
    cells,class_rows,region_rows,region_report={},[],[],{}
    for name in voc.configurations:
        ti=threshold_indices[name]
        cell={'voc_threshold':voc.thresholds[ti]}
        for dataset,result in [('voc',voc),('coco',coco)]:
            a=result.counts[result.index(name),:,ti]
            b=result.counts[result.index('W/REF'),:,threshold_indices['W/REF']]
            mask=np.ones(len(result.image_ids),bool)
            values=class_ious(result,result.index(name),ti,mask)
            base=class_ious(result,result.index('W/REF'),threshold_indices['W/REF'],mask)
            cell[dataset]={'miou':pooled_miou(a,result.classes),'delta_vs_REF':pooled_miou(a,result.classes)-pooled_miou(b,result.classes),
                           'mean_image_iou':float(image_ious(a).mean()),'rates':rates(a,np.asarray(result.classes)),
                           'per_class_iou':values,'per_class_delta_vs_REF':{c:values[c]-base[c] for c in values},
                           'mean_ap_64':float(np.nanmean(result.ap[result.index(name)]))}
            for c in values:
                class_rows.append({'dataset':dataset,'cell':name,'class':c,'threshold':voc.thresholds[ti],
                                   'iou':values[c],'delta_vs_REF':values[c]-base[c]})
        cells[name]=cell
    historical=json.loads((BASE/'sg4_region_labels'/'summary.json').read_text())
    region_class_rows=[]
    for dataset,result in [('voc',voc),('coco',coco)]:
        diag=regions[dataset]
        oracle_mean=float(image_ious(diag['oracle_counts']).mean())
        old_oracle=historical['diagnostics'][dataset]['oracle_iou_mean']['SPEC32']
        baseline=cells['W/SG4/SPEC32_rmean'][dataset]['mean_image_iou']
        region_report[dataset]={'majority_oracle_mean_image_iou':oracle_mean,'historical_sg4_SPEC32_oracle_mean_image_iou':old_oracle,
                                'historical_oracle_drift':oracle_mean-old_oracle,
                                'legacy_sg4_mean_image_iou':baseline,
                                'oracle_note':'SG4 majority-label oracle, not an IoU-optimal mathematical upper bound; GT diagnostic only', 'rules':{}}
        for rule_index,name in enumerate(candidates):
            thresholds=np.full(len(result.image_ids),voc.thresholds[threshold_indices[name]])
            metrics,rows=region_statistics(diag['scores'][:,rule_index],diag['area64'],diag['gt_fraction'],thresholds)
            actual=cells[name][dataset]['mean_image_iou']
            metrics.update({'actual_mean_image_iou':actual,'gap_to_oracle':oracle_mean-actual,
                            'oracle_gap_fraction_closed_vs_legacy_sg4':(actual-baseline)/(oracle_mean-baseline) if oracle_mean!=baseline else None})
            region_report[dataset]['rules'][name]=metrics
            for image_id,classname,threshold,row in zip(result.image_ids,result.classes,thresholds,rows):
                region_rows.append({'dataset':dataset,'evaluation':'full_VOC_threshold','rule':name,'image_id':image_id,'class':classname,'threshold':float(threshold),**row})
            for classname in sorted(set(result.classes)):
                keep=np.asarray(result.classes)==classname
                cm,_=region_statistics(diag['scores'][keep,rule_index],diag['area64'][keep],diag['gt_fraction'][keep],thresholds[keep])
                region_class_rows.append({'dataset':dataset,'rule':name,'class':classname,**cm})
            if dataset=='voc':
                oof_thresholds=np.zeros(len(result.image_ids))
                oof_counts=np.zeros((len(result.image_ids),4),np.int64)
                for fold in range(5):
                    held=result.folds==fold
                    ti,_=best_threshold(result,result.index(name),~held,allowed[name])
                    oof_thresholds[held]=result.thresholds[ti]
                    oof_counts[held]=result.counts[result.index(name),held,ti]
                om,orows=region_statistics(diag['scores'][:,rule_index],diag['area64'],diag['gt_fraction'],oof_thresholds)
                om['candidate_forced_oof_miou']=pooled_miou(oof_counts,result.classes)
                om['candidate_forced_oof_mean_image_iou']=float(image_ious(oof_counts).mean())
                region_report[dataset]['rules'][name]['forced_oof']=om
                for image_id,classname,threshold,row in zip(result.image_ids,result.classes,oof_thresholds,orows):
                    region_rows.append({'dataset':dataset,'evaluation':'OOF_training_fold_threshold','rule':name,'image_id':image_id,'class':classname,'threshold':float(threshold),**row})
    sg2=json.loads((BASE/'sg2_consensus_edges'/'summary.json').read_text())
    historical_sg2=sg2['cells']['W/G_cv_g20_n1']
    for key,screen in screens.items():
        frozen=screen['frozen']
        screen['coco_frozen_miou']=cells[frozen]['coco']['miou']
        screen['coco_delta_vs_REF']=cells[frozen]['coco']['delta_vs_REF']
        screen['user_promising_target_met']=bool(frozen!='W/REF' and screen['vote_winner_votes']>=4 and screen['oof_gain']>=.005 and screen['coco_delta_vs_REF']>0)
    summary={'spec':spec,'voc_images':len(voc.image_ids),'coco_images':len(coco.image_ids),
             'reference_checks':{'voc':vc,'coco':cc},'same_generation_audit_maxima':{d:regions[d]['operator_audit_errors'].max(0).tolist() for d in regions},'screens':screens,'cells':cells,'region_diagnostics':region_report,
             'historical_selected_SG2':{'source':'Section 85, different package/environment; context only, no SG2 inference added to SG13',
                                      'cell':'W/G_cv_g20_n1','voc_threshold':historical_sg2['voc_threshold'],
                                      'voc_miou':historical_sg2['voc_miou@threshold'],'coco_miou':historical_sg2['coco_miou@voc_threshold']}}
    dump(args.output_dir/'summary.json',summary)
    dump(args.output_dir/'fold_results.json',screens)
    write_csv(args.output_dir/'per_class_results.csv',class_rows)
    write_csv(args.output_dir/'per_image_region_diagnostics.csv',region_rows)
    write_csv(args.output_dir/'per_class_region_diagnostics.csv',region_class_rows)
    lines=['# SG13 target-conditioned region inference','','Fixed SPEC32; literal raw region scores; nearest-expanded region masks. VOC selects, COCO transfers only.',
           'Candidate score thresholds: 0–1 by .005; legacy controls retain .30–.70 by .025. No per-image normalization of SG13 region scores.','',
           '| Screen | Vote winner | Votes | OOF REF | OOF selection | Gain | Frozen | Promising target |','|---|---|---:|---:|---:|---:|---|---|']
    for key,s in screens.items():
        lines.append(f"| {key} | {s['vote_winner']} | {s['vote_winner_votes']}/5 | {s['oof_miou_reference']:.4f} | {s['oof_miou_selection']:.4f} | {s['oof_gain']:+.4f} | {s['frozen']} | {s['user_promising_target_met']} |")
    lines += ['', 'Historical pixel-count drift is reported in summary.json. These are cross-environment diagnostics; all methods and the canonical SG4 comparator use the same newly generated inputs and partition. Same-generation linear and region-score identities are enforced for every image.', '']
    lines+=['','| Cell | VOC threshold | VOC mIoU | VOC recall / precision | COCO mIoU | COCO recall / precision |','|---|---:|---:|---:|---:|---:|']
    for name,c in cells.items():
        vr,cr=c['voc']['rates'],c['coco']['rates']
        lines.append(f"| {name} | {c['voc_threshold']:.3f} | {c['voc']['miou']:.4f} | {vr['recall']:.4f}/{vr['precision']:.4f} | {c['coco']['miou']:.4f} | {cr['recall']:.4f}/{cr['precision']:.4f} |")
    lines += [f"| Historical selected SG2 (Section 85) | {historical_sg2['voc_threshold']:.3f} | {historical_sg2['voc_miou@threshold']:.4f} | — | {historical_sg2['coco_miou@voc_threshold']:.4f} | — |",'']
    for key,s in screens.items():
        lines += [f'## {key}: folds','','| Fold | Winner | Threshold | Held-out chosen / REF |','|---:|---|---:|---:|']
        for row in s['folds']:
            lines.append(f"| {row['fold']} | {row['chosen']} | {row['chosen_threshold']:.3f} | {row['heldout_miou_chosen']:.4f}/{row['heldout_miou_reference']:.4f} |")
    for dataset,diag in region_report.items():
        lines += ['',f'## {dataset}: region labelling and majority oracle','',f"Current oracle mean-image IoU: {diag['majority_oracle_mean_image_iou']:.4f}; historical SG4 oracle: {diag['historical_sg4_SPEC32_oracle_mean_image_iou']:.4f}. These are mean-image IoUs, not class-mIoU.",'',
                  '| Rule | FG regions/image | Region precision | Region recall | Missed positive | Selected negative | Actual mean-image IoU | Oracle gap | Gap fraction closed vs legacy SG4 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,m in diag['rules'].items():
            gap='undefined' if m['oracle_gap_fraction_closed_vs_legacy_sg4'] is None else f"{m['oracle_gap_fraction_closed_vs_legacy_sg4']:.4f}"
            lines.append(f"| {name} | {m['mean_fg_regions']:.2f} | {m['region_precision']:.4f} | {m['region_recall']:.4f} | {m['oracle_positive_missed']} | {m['oracle_negative_selected']} | {m['actual_mean_image_iou']:.4f} | {m['gap_to_oracle']:.4f} | {gap} |")
    for dataset in ('voc','coco'):
        for name in candidates:
            lines += ['',f'## {dataset}: {name}, class IoUs','','| Class | IoU | Delta REF |','|---|---:|---:|']
            lines += [f"| {c} | {value:.4f} | {cells[name][dataset]['per_class_delta_vs_REF'][c]:+.4f} |" for c,value in cells[name][dataset]['per_class_iou'].items()]
    lines += ['','Per-image region counts and OOF diagnostics are in the CSV/JSON files. The majority oracle uses GT and is diagnostic only. Negative gap fractions are retained; no value is clipped.','']
    (args.output_dir/'report.md').write_text('\n'.join(lines))
    (args.output_dir/'CHATGPT_RESULTS.md').write_text('\n'.join(lines))
    from src.segmentation.region_visualization import render_region_sheets
    render_region_sheets(args.output_dir,spec,summary)
    print(json.dumps({k:{f:v for f,v in s.items() if f in ('frozen','oof_gain','vote_winner_votes')} for k,s in screens.items()},indent=2))


if __name__=='__main__':main()
