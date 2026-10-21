#!/usr/bin/env python3
"""Plot comparable VOC/COCO 31x8 heatmaps and 31-call trajectories."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    with path.open(newline='') as handle:return list(csv.DictReader(handle))


def value(row,name):
    try:return float(row[name])
    except (KeyError,ValueError,TypeError):return np.nan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir
    results=json.loads((out/'results.json').read_text())
    n={k:v['images'] for k,v in results['datasets'].items()}
    heads=read(out/'head_call_metrics.csv');calls=read(out/'call_summary.csv')
    metrics=[('tp_add','TP additions per image'),('fp_add','FP additions per image'),
             ('tp_fp_ratio','TP / FP'),('fp_gt16_fraction','FP >16 px fraction'),
             ('fp_gt64_fraction','FP >64 px fraction'),
             ('fp_separate_fraction','GT-disjoint FP fraction'),
             ('useful_change_fraction','Useful-change fraction'),
             ('mean_distance','Mean attention distance'),
             ('entropy','Mean row entropy'),('reciprocity','Reciprocity')]
    for metric,title in metrics:
        fig,axes=plt.subplots(1,2,figsize=(11,9),layout='constrained')
        arrays=[]
        for dataset in ('voc','coco'):
            a=np.full((31,8),np.nan)
            for row in heads:
                if row['dataset']==dataset:
                    v=value(row,metric)
                    if metric in ('tp_add','fp_add'):v/=n[dataset]
                    a[int(row['call']),int(row['head'])]=v
            arrays.append(a)
        finite=np.concatenate([a[np.isfinite(a)] for a in arrays])
        vmin=float(finite.min()) if len(finite) else 0
        vmax=float(finite.max()) if len(finite) else 1
        if vmax==vmin:vmax=vmin+1
        for ax,dataset,a in zip(axes,('VOC-Sim','COCO-Cap'),arrays):
            im=ax.imshow(a,aspect='auto',interpolation='nearest',cmap='viridis',vmin=vmin,vmax=vmax)
            ax.set_title(dataset);ax.set_xlabel('SA head (0–7)');ax.set_ylabel('Diffusion call (0–30)')
            ax.set_xticks(range(8));ax.set_yticks(range(0,31,5))
        fig.colorbar(im,ax=axes,label=title,shrink=.7)
        fig.suptitle(f'Frozen module 15: {title}')
        fig.savefig(out/f'heatmaps_{metric}.png',dpi=160)
        plt.close(fig)
    curves=[('tp_add','TP additions per image'),('fp_add','FP additions per image'),
            ('tp_fp_ratio','TP / FP'),('fp_gt16_fraction','FP >16 px fraction'),
            ('fp_separate_fraction','GT-disjoint FP fraction'),
            ('reference_new_tp_support_fraction','Fraction of P1 new TP supported'),
            ('reference_new_fp_support_fraction','Fraction of P1 new FP supported'),
            ('mean_up_support_gt_foreground','Mean upward support on GT foreground'),
            ('mean_up_support_gt_background','Mean upward support on GT background')]
    fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    for ax,(metric,title) in zip(axes.flat,curves):
        for dataset,color in (('voc','#247f58'),('coco','#a33b76')):
            rows=sorted((r for r in calls if r['dataset']==dataset),key=lambda r:int(r['call']))
            values=np.array([value(r,metric) for r in rows],float)
            if metric in ('tp_add','fp_add'):values/=n[dataset]
            ax.plot(range(31),values,label=dataset.upper(),color=color)
        ax.axvline(9.5,color='gray',ls=':',lw=.8);ax.axvline(20.5,color='gray',ls=':',lw=.8)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Call index');ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.suptitle('Fixed early (0–9), middle (10–20), late (21–30) call trajectories')
    fig.savefig(out/'temporal_curves_all.png',dpi=160)
    plt.close(fig)
    print('Wrote 10 heatmaps and temporal curves',flush=True)

if __name__=='__main__':main()
