#!/usr/bin/env python3
"""Validate the 40 diagnostic shards and write ChatGPT handoff tables."""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_coco_cap_transfer import voc_rows,coco_rows
from scripts.run_pre_sg18_sg21_diagnostics import SPEC,OUT,image_path
from src.segmentation import temporal_head_diagnostics as d
from src.segmentation.propagation_diagnostics import histogram_auc


def write_csv(path,rows):
    if not rows:raise ValueError(f'no rows for {path}')
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def ratio(a,b):return float(a/b) if b else None


def rates(counts,reference=None):
    c=np.asarray(counts,dtype=np.int64)
    v={name:int(c[i]) for i,name in enumerate(d.COUNTS)}
    tp,fp,far16,far64,sep,ur,hr=[v[k] for k in d.COUNTS[:7]]
    v.update({'tp_fp_ratio':ratio(tp,fp),'tp_fraction_additions':ratio(tp,tp+fp),
              'fp_gt16_fraction':ratio(far16,fp),'fp_gt64_fraction':ratio(far64,fp),
              'fp_separate_fraction':ratio(sep,fp),
              'distant_fp_fraction_additions':ratio(far16,tp+fp),
              'useful_change_fraction':ratio(tp+ur,tp+fp+ur+hr)})
    if reference is not None:
        v['reference_new_tp_support_fraction']=ratio(v['reference_new_tp_supported'],reference[0])
        v['reference_new_fp_support_fraction']=ratio(v['reference_new_fp_supported'],reference[1])
    return v


def spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y)
    if valid.sum()<3 or np.std(x[valid])==0 or np.std(y[valid])==0:
        return {'n':int(valid.sum()),'rho':None,'p':None}
    result=spearmanr(x[valid],y[valid])
    return {'n':int(valid.sum()),'rho':float(result.statistic),'p':float(result.pvalue)}


def metric(c,name):
    v=rates(c)
    return v[name] if v[name] is not None else np.nan


def load_dataset(base,dataset,spec,smoke):
    n=1 if smoke else 20
    hist=np.zeros((len(d.ENDPOINTS),2,len(d.FEATURES),spec['pixel_histogram_bins']),np.int64)
    by_class={};head=np.zeros((31,8,len(d.COUNTS)),np.int64)
    call=np.zeros((31,len(d.COUNTS)),np.int64)
    stage=np.zeros((3,len(d.COUNTS)),np.int64)
    descriptor=np.zeros((31,8,len(d.DESCRIPTORS)),np.float64)
    support_sum=np.zeros((31,2),np.float64);support_n=np.zeros((31,2),np.int64)
    reference=np.zeros(len(d.COUNTS),np.int64)
    seen=set();expected=set()
    for shard in range(n):
        classname,rows=(voc_rows if dataset=='voc' else coco_rows)(shard)
        if smoke:rows=rows[:1]
        ch=np.zeros_like(hist)
        for row in rows:
            image_id=row['image_id'];expected.add(image_id)
            path=image_path(base/dataset/'shards'/f'class_{shard:02d}',image_id)
            if not path.exists():raise ValueError(f'missing {path}')
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['meta']))
                if meta['image_id']!=image_id or meta['class']!=classname or meta['settings']['spec']!=spec:
                    raise ValueError(f'foreign/incompatible {path}')
                if image_id in seen:raise ValueError(f'duplicate {image_id}')
                seen.add(image_id)
                h=data['hist'];hc=data['head_counts'];cc=data['call_counts']
                sc=data['stage_counts'];desc=data['descriptors']
                su=data['call_support'];ref=data['reference_counts']
                if h.shape!=hist.shape or hc.shape!=head.shape or cc.shape!=call.shape or sc.shape!=stage.shape:
                    raise ValueError(f'wrong array shape {path}')
                if desc.shape!=descriptor.shape or su.shape!=support_sum.shape or ref.shape!=reference.shape:
                    raise ValueError(f'wrong descriptor/support shape {path}')
                if not np.isfinite(desc).all():raise ValueError(f'nonfinite descriptor {path}')
                hist+=h;ch+=h;head+=hc;call+=cc;stage+=sc;descriptor+=desc;reference+=ref
                finite=np.isfinite(su);support_sum+=np.where(finite,su,0);support_n+=finite
        by_class[classname]=ch
    if seen!=expected or (not smoke and len(seen)!=spec['images'][dataset]):
        raise ValueError(f'{dataset}: incomplete image coverage')
    count=len(seen)
    for ei in range(len(d.ENDPOINTS)):
        for fi in range(len(d.FEATURES)):
            if hist[ei,0,fi].sum()!=hist[ei,0,0].sum() or hist[ei,1,fi].sum()!=hist[ei,1,0].sum():
                raise ValueError('feature histogram event totals mismatch')
    if int(hist[0,0,0].sum())!=reference[0] or int(hist[0,1,0].sum())!=reference[1]:
        raise ValueError('reference event totals mismatch')
    return {'images':count,'hist':hist,'by_class':by_class,'head':head,'call':call,
            'stage':stage,'descriptor':descriptor/count,
            'support':np.divide(support_sum,support_n,out=np.full_like(support_sum,np.nan),
                                where=support_n>0),'reference':reference}


def temporal_auc(dataset,data):
    records=[]
    for ei,endpoint in enumerate(d.ENDPOINTS):
        for fi,feature in enumerate(d.FEATURES):
            p,n=data['hist'][ei,:,fi]
            raw=histogram_auc(p,n);inverse=histogram_auc(p[::-1],n[::-1])
            class_roc=[];class_inverse_roc=[]
            for ch in data['by_class'].values():
                pp,nn=ch[ei,:,fi]
                score=histogram_auc(pp,nn)['roc_auc']
                inverse_score=histogram_auc(pp[::-1],nn[::-1])['roc_auc']
                if score is not None:
                    class_roc.append(score);class_inverse_roc.append(inverse_score)
            records.append({'dataset':dataset,'endpoint':endpoint,'feature':feature,
                            'positive_pixels':raw['positive_pixels'],
                            'negative_pixels':raw['negative_pixels'],
                            'roc_auc':raw['roc_auc'],'pr_auc':raw['pr_auc'],
                            'inverse_roc_auc':inverse['roc_auc'],
                            'inverse_pr_auc':inverse['pr_auc'],
                            'class_balanced_roc_auc':float(np.mean(class_roc)) if class_roc else None,
                            'class_balanced_inverse_roc_auc':float(np.mean(class_inverse_roc)) if class_inverse_roc else None,
                            'class_balanced_classes':len(class_roc)})
    return records


def build_head_rows(dataset,data,spec):
    rows=[];summary=[];stages=[];calls=[]
    ref=data['reference']
    for t in range(31):
        for h in range(8):
            row={'dataset':dataset,'call':t,'head':h,**rates(data['head'][t,h],ref)}
            row.update({name:float(data['descriptor'][t,h,i]) for i,name in enumerate(d.DESCRIPTORS)})
            rows.append(row)
        c={'dataset':dataset,'call':t,**rates(data['call'][t],ref),
           'mean_up_support_gt_foreground':float(data['support'][t,0]),
           'mean_up_support_gt_background':float(data['support'][t,1])}
        calls.append(c)
    for h in range(8):
        counts=data['head'][:,h].sum(0)
        summary.append({'dataset':dataset,'head':h,**rates(counts,ref),
                        **{name:float(data['descriptor'][:,h,i].mean())
                           for i,name in enumerate(d.DESCRIPTORS)}})
        for stage_name,(start,stop) in spec['time_stages'].items():
            c=data['head'][start:stop,h].sum(0)
            stages.append({'dataset':dataset,'stage':stage_name,'head':h,**rates(c,ref),
                           **{name:float(data['descriptor'][start:stop,h,i].mean())
                              for i,name in enumerate(d.DESCRIPTORS)}})
    fixed=[{'dataset':dataset,'stage':name,'start':start,'stop_exclusive':stop,
            **rates(data['stage'][i],ref)} for i,(name,(start,stop)) in enumerate(spec['time_stages'].items())]
    return rows,summary,stages,calls,fixed


def structural_correlations(dataset,head_rows):
    outcomes=('tp_fp_ratio','fp_gt16_fraction','fp_separate_fraction')
    result=[]
    for name in d.DESCRIPTORS:
        x=[r[name] for r in head_rows]
        for outcome in outcomes:
            z=spearman(x,[r[outcome] if r[outcome] is not None else np.nan for r in head_rows])
            result.append({'dataset':dataset,'descriptor':name,'outcome':outcome,**z})
    return result


def stability(voc,coco):
    fields=('tp_fp_ratio','useful_change_fraction','fp_gt16_fraction','fp_separate_fraction')
    return {f:spearman([r[f] if r[f] is not None else np.nan for r in voc],
                       [r[f] if r[f] is not None else np.nan for r in coco]) for f in fields}


def md_table(rows,columns):
    def fmt(x):
        if x is None:return '—'
        if isinstance(x,float):return f'{x:.3f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |',
                      '| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(fmt(r.get(c)) for c in columns)+' |' for r in rows])


def reports(out,summary,feature_rows,head_rows,call_rows,stage_rows,correlations):
    lookup={(r['dataset'],r['endpoint'],r['feature']):r for r in feature_rows}
    ranked=[]
    for feature in d.FEATURES:
        v=lookup['voc','new_tp_vs_fp',feature]
        c=lookup['coco','new_tp_vs_fp',feature]
        best=max((v['roc_auc'] or 0), (v['inverse_roc_auc'] or 0))
        ranked.append({'feature':feature,'VOC ROC':best,'COCO ROC':c['roc_auc'],
                       'COCO inverse ROC':c['inverse_roc_auc'],
                       'COCO TP vs >16 FP':lookup['coco','new_tp_vs_fp_gt16',feature]['roc_auc'],
                       'COCO useful/harmful':lookup['coco','useful_vs_harmful',feature]['roc_auc']})
    ranked=sorted(ranked,key=lambda r:r['VOC ROC'],reverse=True)
    lines=['# Pre-SG18 / Pre-SG21 analysis for ChatGPT','',
           'Diagnostic only: frozen W CA, module-15 SA, 31 calls × 8 heads, '
           'threshold 0.45; no method or threshold was selected.',
           f"VOC {summary['datasets']['voc']['images']} images; "
           f"COCO {summary['datasets']['coco']['images']} images.",
           '', '## Strongest VOC temporal rankings', '',
           'Both raw and inverse ROC are supplied in the CSV. This table ranks '
           'the better VOC direction; COCO columns remain raw so direction can be checked.',
           '',md_table(ranked[:12],('feature','VOC ROC','COCO ROC','COCO inverse ROC',
                                  'COCO TP vs >16 FP','COCO useful/harmful')),
           '', '## Fixed stage maps versus CA', '',
           md_table(stage_rows,('dataset','stage','tp_add','fp_add','tp_fp_ratio',
                                'fp_gt16_fraction','fp_separate_fraction')),
           '', '## Cross-dataset cell stability', '',
           md_table([{'metric':k,**v} for k,v in summary['cross_dataset_stability'].items()],
                    ('metric','n','rho','p')),
           '', '## Structural correlations', '',
           md_table(sorted(correlations,key=lambda r:abs(r['rho'] or 0),reverse=True)[:15],
                    ('dataset','descriptor','outcome','n','rho')),
           '', '## Interpretation limits', '',
           '- Observation maps are each normalized independently before applying '
           'the fixed diagnostic threshold. Their event totals measure each observation '
           'against the same CA prediction; they do not sum to the ordinary mean effect.',
           '- Stage maps use the fixed early/middle/late windows and are scored the same way.',
           '- TP/FP ratios can be undefined when FP=0. Null AUC denotes a missing class.',
           '- Histogram AUC uses 512 fixed bins; correlated pixels make these descriptive '
           'rankings rather than independent-sample significance tests.',
           '- GT is used only for event labels, distance, connected components, and diagnostics.',
           '', 'Full tables: `results.json`, `temporal_feature_auc.csv`, '
           '`head_call_metrics.csv`, `head_summary.csv`, `head_stage_summary.csv`, '
           '`call_summary.csv`, `structural_correlations.csv`, and PNG heatmaps/curves.','']
    (out/'ANALYSIS_FOR_CHATGPT.md').write_text('\n'.join(lines))
    handoff=['# Pre-SG18 / Pre-SG21 diagnostic handoff','',
             'Question: does useful one-pass SA propagation show stable temporal or head-role structure?',
             '',f"Coverage: VOC {summary['datasets']['voc']['images']} / 490; "
             f"COCO {summary['datasets']['coco']['images']} / 707 images.",
             'The package contains all five endpoint ROC/PR results, class-balanced ROC, '
             'fixed stage counts, 31-call curves, 31×8 head maps, structural correlations, '
             'and VOC-to-COCO cell rank stability. No SG18/SG21 method was implemented.',
             '', 'Read `ANALYSIS_FOR_CHATGPT.md` first, then `results.json` and CSVs. '
             'Use the frozen go/no-go criteria in the supplied diagnostic specification.',
             '', 'Files: `ANALYSIS_FOR_CHATGPT.md`, `results.json`, '
             '`temporal_feature_auc.csv`, `head_call_metrics.csv`, `head_summary.csv`, '
             '`head_stage_summary.csv`, `call_summary.csv`, '
             '`structural_correlations.csv`, `heatmaps_*.png`, `temporal_curves_*.png`.','']
    (out/'CHATGPT_HANDOFF.md').write_text('\n'.join(handoff))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();spec=json.loads(SPEC.read_text())
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    all_data={dataset:load_dataset(out,dataset,spec,args.smoke) for dataset in spec['datasets']}
    feature_rows=[];head_rows=[];summary_rows=[];head_stage_rows=[];call_rows=[];stage_rows=[];corr=[]
    summary={'spec':spec,'datasets':{},'cross_dataset_stability':{}}
    for dataset,data in all_data.items():
        feature_rows+=temporal_auc(dataset,data)
        h,s,hs,c,st=build_head_rows(dataset,data,spec)
        head_rows+=h;summary_rows+=s;head_stage_rows+=hs;call_rows+=c;stage_rows+=st
        corr+=structural_correlations(dataset,h)
        summary['datasets'][dataset]={'images':data['images'],
            'ordinary_reference_counts':rates(data['reference']),
            'fixed_stages':st,'head_summary':s,'call_summary':c}
    voc=[r for r in head_rows if r['dataset']=='voc']
    coco=[r for r in head_rows if r['dataset']=='coco']
    summary['cross_dataset_stability']=stability(voc,coco)
    summary['temporal_feature_auc']=feature_rows
    summary['structural_correlations']=corr
    write_csv(out/'temporal_feature_auc.csv',feature_rows)
    write_csv(out/'head_call_metrics.csv',head_rows)
    write_csv(out/'head_summary.csv',summary_rows)
    write_csv(out/'head_stage_summary.csv',head_stage_rows)
    write_csv(out/'call_summary.csv',call_rows)
    write_csv(out/'fixed_stage_summary.csv',stage_rows)
    write_csv(out/'structural_correlations.csv',corr)
    (out/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    reports(out,summary,feature_rows,head_rows,call_rows,stage_rows,corr)
    print(f'Merged {all_data["voc"]["images"]} VOC and {all_data["coco"]["images"]} COCO images',flush=True)


if __name__=='__main__':main()
