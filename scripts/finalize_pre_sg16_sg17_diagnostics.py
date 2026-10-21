#!/usr/bin/env python3
"""Validate and summarize the full preregistered Pre-SG16/Pre-SG17 package."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr,rankdata

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_coco_cap_transfer import voc_rows,coco_rows
from scripts.run_pre_sg16_sg17_diagnostics import image_path,SPEC,OUT
from src.segmentation.pre_sg_diagnostics import PIXEL_FEATURES,PIXEL_BOUNDS,ENDPOINTS
from src.segmentation.propagation_diagnostics import histogram_auc


def auc_exact(y,x):
    y=np.asarray(y,bool);x=np.asarray(x,float)
    if len(y)==0 or y.all() or (~y).all():return None
    npos=int(y.sum());nneg=len(y)-npos
    roc=float((rankdata(x,method='average')[y].sum()-npos*(npos+1)/2)/(npos*nneg))
    ordered=np.argsort(-x,kind='stable');z=x[ordered];labels=y[ordered]
    ends=np.r_[np.flatnonzero(z[:-1]!=z[1:])+1,len(z)]
    tp=np.cumsum(labels)[ends-1];precision=tp/ends
    increments=np.diff(np.r_[0,tp])
    ap=float((increments*precision).sum()/npos)
    return {'roc_auc':roc,'pr_auc':ap,'positive':npos,'negative':nneg,'prevalence':float(npos/len(y))}


def correlations(rows,target,features):
    y=np.array([r[target] for r in rows],float);out={}
    for key in features:
        x=np.array([r[key] if r[key] is not None else np.nan for r in rows],float)
        valid=np.isfinite(x)&np.isfinite(y)
        if valid.sum()<3 or np.std(x[valid])==0 or np.std(y[valid])==0:
            out[key]={'n':int(valid.sum()),'pearson':None,'spearman':None}
        else:
            out[key]={'n':int(valid.sum()),'pearson':float(pearsonr(x[valid],y[valid]).statistic),
                      'spearman':float(spearmanr(x[valid],y[valid]).statistic)}
    return out


def region_summary(rows,spec):
    features=[k for k in rows[0] if k not in
              ('region_id','oracle','gt_fraction','region_area_pixels','region_area_fraction',
               'object_area_fraction','image_id','class')]
    subsets={
        'all':lambda r:True,
        'ambiguous_p1':lambda r:.2<=r['p1_occupancy']<=.8,
        'wrong_p1':lambda r:(r['p1_occupancy']>=.5)!=bool(r['oracle']),
        'priors_agree':lambda r:r['majority_votes'] in (0,3),
        'priors_disagree':lambda r:r['majority_votes'] in (1,2)}
    def score(group):
        y=np.array([r['oracle'] for r in group],bool)
        out={}
        for f in features:
            x=np.array([r[f] for r in group],float)
            raw=auc_exact(y,x);reverse=auc_exact(y,-x)
            out[f]={'raw':raw,'inverse':reverse}
        return {'regions':len(group),'oracle_positive':int(y.sum()),
                'oracle_negative':int((~y).sum()),'features':out}
    by_subset={name:score([r for r in rows if predicate(r)]) for name,predicate in subsets.items()}
    by_class={c:score([r for r in rows if r['class']==c]) for c in sorted({r['class'] for r in rows})}
    def bins(key,edges):
        return {f'[{edges[i]},{edges[i+1]}{"," if i<len(edges)-2 else "]"}':
                score([r for r in rows if edges[i]<=r[key]<(edges[i+1] if i<len(edges)-2 else edges[i+1]+1e-9)])
                for i in range(len(edges)-1)}
    area=bins('region_area_fraction',spec['region_area_fraction_bins'])
    object_area=bins('object_area_fraction',spec['object_area_fraction_bins'])
    balanced={}
    for f in features:
        values=[by_class[c]['features'][f]['raw']['roc_auc'] for c in by_class
                if by_class[c]['features'][f]['raw'] is not None]
        balanced[f]=float(np.mean(values)) if values else None
    return {'subsets':by_subset,'per_class':by_class,'class_balanced_roc_auc':balanced,
            'region_area_bins':area,'object_area_bins':object_area}


def pixel_summary(hist,images):
    out={}
    for ei,endpoint in enumerate(ENDPOINTS):
        out[endpoint]={}
        for fi,feature in enumerate(PIXEL_FEATURES):
            pos,neg=hist[ei,:,fi]
            out[endpoint][feature]={
                'raw':histogram_auc(pos,neg),
                'inverse':histogram_auc(pos[::-1],neg[::-1])}
    fields=['second_q_direction_mean','second_q_direction_median',
            'top50_q_direction_mean','top50_q_direction_median',
            'support_reliability_mean','support_reliability_median',
            'first_r_max_mean','first_r_max_median','useful','harmful',
            'new_fp_gt16','new_fp_gt64']
    return {'auc':out,'image_correlations':correlations(images,'two_pass_iou_gain',fields),
            'image_benefit_roc_auc':{k:auc_exact([r['two_pass_iou_gain']>0 for r in images],
                  [r[k] if r[k] is not None else 0 for r in images]) for k in fields if k.endswith(('_mean','_median'))}}


def binned_quantile(positive,negative,q):
    counts=positive+negative;total=counts.sum()
    if total==0:return None
    return int(np.searchsorted(np.cumsum(counts),q*total,side='left'))


def deciles(hist,feature,direction='raw',edges=None):
    fi=PIXEL_FEATURES.index(feature);base=hist[0,:,fi]
    if direction=='inverse':base=base[:,::-1]
    if edges is None:
        edges=[0]+[binned_quantile(base[0],base[1],q/10) for q in range(1,10)]+[hist.shape[-1]]
    rows=[]
    for i in range(10):
        lo,hi=edges[i],edges[i+1]
        if lo is None or hi is None:continue
        u,h=base[:,lo:hi].sum(1)
        def endpoints(index):
            p,n=hist[index,:,fi]
            if direction=='inverse':p,n=p[::-1],n[::-1]
            return int(p[lo:hi].sum()),int(n[lo:hi].sum())
        tp,fp=endpoints(3);_,far16=endpoints(4);_,far64=endpoints(5)
        rows.append({'decile':i+1,'bin_low':int(lo),'bin_high_exclusive':int(hi),
                     'changed_pixels':int(u+h),'useful':int(u),'harmful':int(h),
                     'useful_fraction':float(u/(u+h)) if u+h else None,
                     'new_tp':tp,'new_fp':fp,'tp_fp_ratio':float(tp/fp) if fp else None,
                     'distant_fp_gt16':far16,'distant_fp_gt64':far64,
                     'distant_fp_fraction_among_additions':float(far16/(tp+fp)) if tp+fp else None})
    return rows,edges


def quantiles_from_hist(hist,feature,endpoint=0):
    fi=PIXEL_FEATURES.index(feature);out={}
    for label,values in zip(('useful','harmful'),hist[endpoint,:,fi]):
        out[label]={str(q):binned_quantile(values,np.zeros_like(values),q)
                    for q in (0,.1,.25,.5,.75,.9,1)}
    lo,hi=PIXEL_BOUNDS[feature]
    return {'quantile_score':{label:{q:(None if b is None else float(lo+(b+.5)*(hi-lo)/hist.shape[-1]))
                               for q,b in values.items()} for label,values in out.items()},
            'score_range':[lo,hi]}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    summary={'spec':spec,'datasets':{}}
    all_hist={}
    for dataset in spec['datasets']:
        regions=[];region_images=[];pixel_images=[]
        hist=np.zeros((len(ENDPOINTS),2,len(PIXEL_FEATURES),spec['pixel_histogram_bins']),np.int64)
        found=set();expected=set();shards=range(1) if args.smoke else range(20)
        for shard in shards:
            classname,rows=(voc_rows if dataset=='voc' else coco_rows)(shard)
            if args.smoke:rows=rows[:1]
            for row in rows:
                image_id=row['image_id'];expected.add(image_id)
                path=image_path(args.output_dir/dataset/'shards'/f'class_{shard:02d}',image_id)
                if not path.exists():raise ValueError(f'missing {path}')
                with np.load(path,allow_pickle=False) as d:
                    meta=json.loads(str(d['meta']))
                    if meta['image_id']!=image_id or meta['class']!=classname or meta['settings']['spec']!=spec:
                        raise ValueError(f'{path}: wrong id/class/spec')
                    rr=json.loads(str(d['regions']));ri=json.loads(str(d['region_image']))
                    pi=json.loads(str(d['pixel_image']));h=d['hist']
                if h.shape!=hist.shape or not np.issubdtype(h.dtype,np.integer):raise ValueError('wrong histogram shape')
                if image_id in found:raise ValueError(f'duplicate {image_id}')
                found.add(image_id);hist+=h
                regions.extend([dict(r,image_id=image_id,**{'class':classname}) for r in rr])
                region_images.append(dict(ri,image_id=image_id,**{'class':classname}))
                pixel_images.append(dict(pi,image_id=image_id,**{'class':classname}))
        if found!=expected:raise ValueError(f'{dataset}: incomplete/foreign images')
        if not regions:raise ValueError('empty region diagnostics')
        analysis={'images':len(found),'regions':len(regions),
                  'region':region_summary(regions,spec),'pixel':pixel_summary(hist,pixel_images),
                  'oracle_gain_correlations':correlations(region_images,'oracle_gain_vs_p1',
                     ('object_area_fraction','boundary_length_over_area','compactness',
                      'oracle_positive_regions','object_crossed_regions','p1_iou')),
                  'pixel_event_totals':{name:{'positive':int(hist[ei,0,0].sum()),
                                              'negative':int(hist[ei,1,0].sum())}
                                         for ei,name in enumerate(ENDPOINTS)}}
        summary['datasets'][dataset]=analysis;all_hist[dataset]=hist
        args.output_dir.mkdir(parents=True,exist_ok=True)
        write_csv(args.output_dir/f'{dataset}_regions.csv',regions)
        write_csv(args.output_dir/f'{dataset}_region_images.csv',region_images)
        write_csv(args.output_dir/f'{dataset}_second_pass_images.csv',pixel_images)
        print(f'{dataset}: {len(found)} images, {len(regions)} regions',flush=True)
    # Exploratory ranking uses VOC only; the same feature/direction and VOC bin
    # edges are applied to COCO. This is diagnostic, not a pipeline choice.
    voc=summary['datasets']['voc']['pixel']['auc']['useful_all']
    eligible=('support_reliability','all_top50_agreement',
              'second_q_direction','second_median_direction','second_trim10_direction',
              'second_std','second_mad','second_iqr','second_mean_abs',
              'top50_q_direction','top50_median_direction','top50_trim10_direction',
              'top50_std','top50_mad','top50_iqr','top50_mean_abs')
    ranked=[(v['roc_auc'],k,direction) for k in eligible
            for direction,v in voc[k].items() if v['roc_auc'] is not None]
    _,best,direction=max(ranked) if ranked else (float('nan'),'second_q_direction','raw')
    summary['exploratory_best_reliability']={'feature':best,'direction':direction,
                                            'selected_using':'VOC diagnostic AUC only; no method or threshold selection'}
    voc_deciles,edges=deciles(all_hist['voc'],best,direction)
    summary['datasets']['voc']['pixel']['best_reliability_deciles']=voc_deciles
    summary['datasets']['coco']['pixel']['best_reliability_deciles']=deciles(all_hist['coco'],best,direction,edges)[0]
    summary['datasets']['voc']['pixel']['best_reliability_quantiles_bins']=quantiles_from_hist(all_hist['voc'],best)
    summary['datasets']['coco']['pixel']['best_reliability_quantiles_bins']=quantiles_from_hist(all_hist['coco'],best)
    result=args.output_dir/'results.json'
    temporary=result.with_suffix('.json.tmp');temporary.write_text(json.dumps(summary,indent=2)+'\n')
    temporary.replace(result)
    lines=['# Pre-SG16 and Pre-SG17 diagnostics','',
           'Full details: `results.json`; individual region and image records: CSV files. '+
           'Region AUC is exact. Pixel AUC/AP uses 1024 fixed bins and all changed pixels.',
           'No classifier, threshold, segmentation gate, or final pipeline was selected.','',
           '## Pre-SG16: region separability','',
           '| Feature | VOC all ROC / AP | VOC ambiguous ROC / AP | VOC wrong-P1 ROC / AP | COCO all ROC / AP | COCO ambiguous ROC / AP |',
           '|---|---:|---:|---:|---:|---:|']
    def auc_text(dataset,subset,feature):
        v=summary['datasets'][dataset]['region']['subsets'][subset]['features'][feature]['raw']
        return 'undefined' if v is None else f"{v['roc_auc']:.3f} / {v['pr_auc']:.3f}"
    for f in ('p1_occupancy','occupancy_mean','ca_mean','p1_mean','obs_all_mean',
              'obs_all_trim20','obs_all_r_max','obs_top50_mean','obs_top50_trim20','obs_top50_r_max'):
        lines.append('| '+f+' | '+' | '.join(auc_text(d,s,f) for d,s in
             (('voc','all'),('voc','ambiguous_p1'),('voc','wrong_p1'),
              ('coco','all'),('coco','ambiguous_p1')))+' |')
    lines+=['','Subset sizes: '+', '.join(f"{d} "+', '.join(f"{s}={v['regions']}" for s,v in
              summary['datasets'][d]['region']['subsets'].items()) for d in ('voc','coco')), '',
            '## Pre-SG17: second-pass changes','',
            '| Signal | VOC useful ROC / AP | COCO useful ROC / AP | COCO TP vs FP ROC / AP | COCO TP vs distant FP >16 ROC / AP |',
            '|---|---:|---:|---:|---:|']
    def px(dataset,endpoint,feature):
        v=summary['datasets'][dataset]['pixel']['auc'][endpoint][feature]['raw']
        return 'undefined' if v['roc_auc'] is None else f"{v['roc_auc']:.3f} / {v['pr_auc']:.3f}"
    for f in ('first_r_max','support_reliability','second_q_direction',
              'second_median_direction','second_trim10_direction','top50_q_direction',
              'top50_median_direction','top50_trim10_direction'):
        lines.append('| '+f+' | '+' | '.join(px(d,s,f) for d,s in
              (('voc','useful_all'),('coco','useful_all'),('coco','new_tp_vs_fp'),
               ('coco','new_tp_vs_fp_gt16')))+' |')
    lines+=['',f"Exploratory VOC best reliability: **{best}**, {direction} direction. "+
            'This choice is for reading the diagnostic curve only.','',
            'Full endpoint tables, region per-class and area-bin AUCs, class-balanced AUCs, '+
            'oracle-gain correlations, image-level correlations, changed-pixel score deciles and '+
            'per-image records are in `results.json` and CSV files.','']
    (args.output_dir/'CHATGPT_HANDOFF.md').write_text('\n'.join(lines))
    print(f'Wrote {result} and CHATGPT_HANDOFF.md',flush=True)

if __name__=='__main__':main()
