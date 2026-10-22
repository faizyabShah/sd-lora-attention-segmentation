#!/usr/bin/env python3
"""Exact pixel AUCs and VOC-defined reliability bins for Pre-SG19."""
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
from src.segmentation import pre_sg19_late_support as late

SPEC = ROOT/'configs/pre_sg19_late_support.json'
OUT = ROOT/'outputs/segmentation/analysis/pre_sg19_late_support'
NEGATIVE_COLUMNS = {'tp_vs_fp':1,'tp_vs_fp_gt16':2,'tp_vs_fp_gt64':3,'tp_vs_fp_separate':4}
CLASSES = ('person','chair','bird','bicycle','cow','dining table','tv')


def auc_ap(scores: np.ndarray, positives: np.ndarray) -> tuple[float | None,float | None]:
    """Exact tie-aware ROC-AUC and average precision, descending score groups."""
    n=len(scores); p=int(positives.sum())
    if p==0 or p==n:return None,None
    order=np.argsort(scores,kind='stable')[::-1]
    x=scores[order]; y=positives[order]
    starts=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1]
    group_tp=np.add.reduceat(y.astype(np.int64),starts)
    size=np.diff(np.r_[starts,n]);group_fp=size-group_tp
    tp=group_tp.cumsum();fp=group_fp.cumsum();negative=n-p
    last_tp=np.r_[0,tp[:-1]];last_fp=np.r_[0,fp[:-1]]
    roc=float(np.sum((fp-last_fp)/negative*(tp+last_tp)/(2*p)))
    ap=float(np.sum(group_tp/p*tp/(tp+fp)))
    return roc,ap


def load_dataset(out: Path, dataset: str, token: str, spec: dict, smoke=False):
    chunks=[];labels=[];classes=[];drift=[]
    rows=sg18.rows_for(dataset,first_only=smoke)
    for classname,row in rows:
        path=out/'cache'/dataset/(sg18.key(row['image_id'])+'.npz')
        with np.load(path,allow_pickle=False) as d:
            meta=json.loads(str(d['meta']))
            if meta['image_id']!=row['image_id'] or meta['dataset']!=dataset or meta['spec']!=spec:
                raise ValueError('foreign or mismatched feature checkpoint')
            f=d['features_'+token];lab=d['labels_'+token]
        if f.shape!=(len(lab),len(late.FEATURES)) or lab.shape!=(len(f),5):
            raise ValueError('invalid per-image feature shape')
        chunks.append(f);labels.append(lab)
        classes.append(np.full(len(f),classname,dtype='U20'))
        drift.append(meta['sg18_cache_middle_late_mae'])
    return np.concatenate(chunks),np.concatenate(labels),np.concatenate(classes),np.array(drift)


def auc_table(features, labels, sign):
    result={}
    for i,name in enumerate(late.FEATURES):
        value=features[:,i]*sign[name]
        result[name]={}
        for comparison,col in NEGATIVE_COLUMNS.items():
            eligible=(labels[:,0]>0)|(labels[:,col]>0)
            y=labels[eligible,0].astype(bool)
            auc,ap=auc_ap(value[eligible],y)
            result[name][comparison]={'roc_auc':auc,'pr_auc':ap,
                                      'tp_pixels':int(y.sum()),'fp_pixels':int((~y).sum())}
    return result


def splits(features,labels):
    output={}
    for name in ('late_minus_ca','late_minus_middle'):
        values=features[:,late.FEATURES.index(name)]
        total=len(values);parts={}
        for label,mask in [('positive',values>0),('negative',values<0),('zero',values==0)]:
            counts=labels[mask].sum(0).astype(int)
            n=int(mask.sum())
            parts[label]={'pixels':n,'fraction_of_additions':n/max(total,1),
                          'tp_rate':float(counts[0]/n) if n else None,
                          'fp_rate':float(counts[1]/n) if n else None,
                          'distant_fp_gt16_rate':float(counts[2]/n) if n else None,
                          'distant_fp_gt64_rate':float(counts[3]/n) if n else None,
                          'separate_fp_rate':float(counts[4]/n) if n else None}
        output[name]=parts
    return output


def bins_for(values,labels,edges):
    # Values outside the VOC edge range enter the end bins on COCO.
    bucket=np.searchsorted(edges[1:-1],values,side='right')
    output=[]
    for b in range(len(edges)-1):
        hit=bucket==b;n=int(hit.sum());c=labels[hit].sum(0).astype(int)
        output.append({'bin':b+1,'voc_lower':float(edges[b]),'voc_upper':float(edges[b+1]),
                       'pixels':n,'tp_pixels':int(c[0]),'fp_pixels':int(c[1]),
                       'tp_fraction':float(c[0]/n) if n else None,
                       'fp_fraction':float(c[1]/n) if n else None,
                       'tp_fp_ratio':float(c[0]/c[1]) if c[1] else None,
                       'distant_fp_gt16_fraction':float(c[2]/n) if n else None,
                       'distant_fp_gt64_fraction':float(c[3]/n) if n else None,
                       'separate_fp_fraction':float(c[4]/n) if n else None})
    return output


def class_auc(features,labels,classes,sign):
    output={}
    for wanted in CLASSES:
        keep=np.isin(classes,['dining-table','dining table']) if wanted=='dining table' else classes==wanted
        if not keep.any():continue
        rows=[]
        for i,name in enumerate(late.FEATURES):
            roc,ap=auc_ap(features[keep,i]*sign[name],labels[keep,0].astype(bool))
            rows.append({'feature':name,'roc_auc':roc,'pr_auc':ap})
        output[wanted]={'tp_pixels':int(labels[keep,0].sum()),
                        'fp_pixels':int(labels[keep,1].sum()),
                        'features':rows}
    return output


def plot_bins(out,bin_results,selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(selected),figsize=(5*len(selected),4),sharey=False)
    if len(selected)==1:axes=[axes]
    for ax,name in zip(axes,selected):
        for ds,color in [('voc','tab:blue'),('coco','tab:orange')]:
            values=bin_results[ds][name]
            ratio=[v['tp_fp_ratio'] if v['tp_fp_ratio'] is not None else np.nan for v in values]
            ax.plot(range(1,len(ratio)+1),ratio,'o-',label=ds.upper(),color=color)
        ax.set(title=name,xlabel='VOC-defined late-support bin',ylabel='TP / FP')
        ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'late_support_tp_fp_ratio.png',dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();spec=json.loads(SPEC.read_text())
    thresholds=spec['middle_thresholds'];primary=spec['primary_middle_threshold']
    primary_token=str(primary).replace('.','p')
    # Select feature orientation and plotted features on VOC only.
    voc_primary,voc_labels,voc_classes,voc_drift=load_dataset(args.output_dir,'voc',primary_token,spec,args.smoke)
    sign={}
    for i,name in enumerate(late.FEATURES):
        auc,_=auc_ap(voc_primary[:,i],voc_labels[:,0].astype(bool))
        sign[name]=1 if auc is None or auc>=.5 else -1
    voc_primary_auc=auc_table(voc_primary,voc_labels,sign)
    selected=sorted(late.FEATURES,key=lambda n:voc_primary_auc[n]['tp_vs_fp']['roc_auc'] or -1,
                    reverse=True)[:spec['bin_feature_count']]
    edges={name:np.quantile(voc_primary[:,late.FEATURES.index(name)]*sign[name],np.linspace(0,1,11))
           for name in selected}
    # Repeated quantiles (notably discrete consensus) are merged; no COCO choice.
    edges={name:np.unique(value) for name,value in edges.items()}
    for name,value in edges.items():
        if len(value)<2:raise ValueError(f'constant selected feature {name}')
    datasets={};bin_results={};class_results={};drift_results={}
    for ds in ('voc','coco'):
        datasets[ds]={};bin_results[ds]={}
        for threshold in thresholds:
            token=str(threshold).replace('.','p')
            if ds=='voc' and threshold==primary:
                f,l,c,drift=voc_primary,voc_labels,voc_classes,voc_drift
                table=voc_primary_auc
            else:
                f,l,c,drift=load_dataset(args.output_dir,ds,token,spec,args.smoke)
                table=auc_table(f,l,sign)
            datasets[ds][str(threshold)]={'pixels':len(l),'new_tp':int(l[:,0].sum()),
                                          'new_fp':int(l[:,1].sum()),
                                          'feature_auc':table,'direct_splits':splits(f,l)}
            drift_results[ds]={'middle_mean_absolute':float(drift[:,0].mean()),
                               'late_mean_absolute':float(drift[:,1].mean()),
                               'middle_max_absolute':float(drift[:,0].max()),
                               'late_max_absolute':float(drift[:,1].max())}
            if threshold==primary:
                class_results[ds]=class_auc(f,l,c,sign)
                for name in selected:
                    values=f[:,late.FEATURES.index(name)]*sign[name]
                    bin_results[ds][name]=bins_for(values,l,edges[name])
    # VOC chooses best per-class feature; COCO only evaluates that choice.
    class_transfer={}
    for cls in CLASSES:
        voc=class_results['voc'].get(cls);coco=class_results['coco'].get(cls)
        if not voc or not coco:continue
        best=max(voc['features'],key=lambda x:x['roc_auc'] if x['roc_auc'] is not None else -1)
        other=next(r for r in coco['features'] if r['feature']==best['feature'])
        class_transfer[cls]={'voc_chosen_feature':best['feature'],
                             'voc_roc_auc':best['roc_auc'],'voc_pr_auc':best['pr_auc'],
                             'coco_roc_auc':other['roc_auc'],'coco_pr_auc':other['pr_auc'],
                             'voc_tp_fp':[voc['tp_pixels'],voc['fp_pixels']],
                             'coco_tp_fp':[coco['tp_pixels'],coco['fp_pixels']]}
    summary={'spec':spec,'images':{'voc':1 if args.smoke else 490,'coco':1 if args.smoke else 707},
             'pixel_auc_is_exact_with_tie_groups':True,
             'feature_sign_chosen_on_voc_only':sign,'selected_bin_features_voc_only':selected,
             'voc_bin_edges':{k:v.tolist() for k,v in edges.items()},
             'results':datasets,'reliability_bins':bin_results,'class_transfer':class_transfer,
             'sg18_cache_drift':drift_results}
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    (out/('smoke_results.json' if args.smoke else 'results.json')).write_text(json.dumps(summary,indent=2)+'\n')
    if args.smoke:
        print('Pre-SG19 finalizer smoke passed',flush=True)
        return
    with (out/'auc_table.csv').open('w',newline='') as h:
        w=csv.writer(h,lineterminator='\n');w.writerow(['dataset','threshold','feature','comparison','roc_auc','pr_auc','tp_pixels','fp_pixels'])
        for ds in ('voc','coco'):
            for threshold in thresholds:
                for feature,cells in datasets[ds][str(threshold)]['feature_auc'].items():
                    for comparison,x in cells.items():
                        w.writerow([ds,threshold,feature,comparison,x['roc_auc'],x['pr_auc'],x['tp_pixels'],x['fp_pixels']])
    with (out/'reliability_bins.csv').open('w',newline='') as h:
        w=csv.writer(h,lineterminator='\n');keys=list(bin_results['voc'][selected[0]][0])
        w.writerow(['dataset','feature',*keys])
        for ds in ('voc','coco'):
            for feature in selected:
                for row in bin_results[ds][feature]:w.writerow([ds,feature,*(row[k] for k in keys)])
    plot_bins(out,bin_results,selected)
    lines=['# Pre-SG19 late support diagnostic','',
           'Middle-added pixels require raw M>c, CA background at 0.45 and middle foreground at the listed threshold. No segmentation method or cutoff is selected. Feature direction and reliability bins are defined on VOC only.','',
           '## Primary threshold 0.425','',
           '| Feature | Direction | VOC TP/FP ROC | COCO TP/FP ROC | COCO TP vs >16 FP | COCO TP vs separate FP |',
           '|---|---|---:|---:|---:|---:|']
    for name in late.FEATURES:
        v=datasets['voc'][str(primary)]['feature_auc'][name]
        c=datasets['coco'][str(primary)]['feature_auc'][name]
        fmt=lambda x: 'NA' if x is None else f'{x:.3f}'
        lines.append(f"| {name} | {'higher' if sign[name]>0 else 'lower'} | "
                     f"{fmt(v['tp_vs_fp']['roc_auc'])} | {fmt(c['tp_vs_fp']['roc_auc'])} | "
                     f"{fmt(c['tp_vs_fp_gt16']['roc_auc'])} | {fmt(c['tp_vs_fp_separate']['roc_auc'])} |")
    lines+=['','Exact ROC and PR AUC for all four contrasts and both thresholds are in `auc_table.csv`. '
            'Bin counts, TP/FP ratios, distant/separate FP rates and direct L>c/L>M splits are in `results.json`.','',
            '## VOC-selected per-class feature','',
            '| Class | Feature | VOC ROC | COCO ROC |', '|---|---|---:|---:|']
    for cls,x in class_transfer.items():
        fmt=lambda n:'NA' if n is None else f'{n:.3f}'
        lines.append(f"| {cls} | {x['voc_chosen_feature']} | {fmt(x['voc_roc_auc'])} | {fmt(x['coco_roc_auc'])} |")
    lines+=['','Reliability plot: `late_support_tp_fp_ratio.png`. Temporary candidate-pixel feature caches are deleted after successful finalization; only reduced reports remain.','']
    (out/'report.md').write_text('\n'.join(lines))
    (out/'completed.json').write_text(json.dumps({'images':{'voc':490,'coco':707},
                                                  'all_required_stages_complete':True},indent=2)+'\n')
    for p in (out/'cache').glob('*/*.npz'):p.unlink()
    for d in (out/'cache').glob('*'):d.rmdir()
    (out/'cache').rmdir()
    print('Pre-SG19 analysis completed',flush=True)


if __name__=='__main__':main()
