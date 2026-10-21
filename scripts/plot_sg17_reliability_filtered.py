#!/usr/bin/env python3
"""Eight-panel SG17 diagnostic sheet from one transient same-generation cache."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import zoom


def normalize(x):
    x=np.asarray(x,float);lo=x.min();hi=x.max()
    return (x-lo)/(hi-lo) if hi>lo else np.zeros_like(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(args.cache,allow_pickle=False) as d:
        image=d['image'];truth=d['truth'].astype(bool);seed=d['seed'];one=d['one'];two=d['two']
        q=d['q'];candidate=d['candidate'];pred=d['pred'].astype(bool);p1=d['p1'].astype(bool)
        cutoff=int(d['cutoff']);classname=str(d['classname']);image_id=str(d['image_id']);method=str(d['method'])
    scale=(truth.shape[0]//64,truth.shape[1]//64)
    rejected=(two>one)&(q<cutoff)
    full=np.repeat(np.repeat(rejected,scale[0],axis=0),scale[1],axis=1)
    overlay=np.asarray(image).copy()
    tp=full&truth;fp=full&~truth
    overlay[tp]=(.35*overlay[tp]+.65*np.array([36,200,82])).astype(np.uint8)
    overlay[fp]=(.35*overlay[fp]+.65*np.array([230,50,145])).astype(np.uint8)
    def iou(mask):
        return (mask&truth).sum()/max((mask|truth).sum(),1)
    fig,ax=plt.subplots(2,4,figsize=(15,7.3),layout='constrained')
    panels=[('Generated image',image,None),('CA seed',normalize(seed),'magma'),
            ('One-pass W/REF',normalize(one),'magma'),
            (f'Ordinary P1 · IoU {iou(p1):.3f}',normalize(two),'magma'),
            (f'Second-pass q-up · cutoff {cutoff}/248',q/248,'viridis'),
            ('Rejected raw increments\ngreen=GT TP, pink=GT FP',overlay,None),
            (f'{method} mask · IoU {iou(pred):.3f}',pred,'gray'),
            ('Original GT mask',truth,'gray')]
    for a,(title,img,cmap) in zip(ax.flat,panels):
        a.imshow(img,cmap=cmap,vmin=0 if cmap else None,vmax=1 if cmap else None)
        a.set_title(title,fontsize=10);a.axis('off')
    fig.suptitle(f'{classname} · {image_id}',fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    plt.close(fig)

if __name__=='__main__':main()
