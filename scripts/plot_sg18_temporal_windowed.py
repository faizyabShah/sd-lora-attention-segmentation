#!/usr/bin/env python3
"""SG18 image | CA | REF | T1 | T2 | T3 | GT | selected-vs-REF sheet."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def normalize(value):
    x=np.asarray(value,float);lo=x.min();hi=x.max()
    return (x-lo)/(hi-lo) if hi>lo else np.zeros_like(x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with np.load(args.cache,allow_pickle=False) as data:
        image=data['image'];truth=data['truth'].astype(bool);seed=data['seed']
        maps=data['maps'];masks=data['masks'].astype(bool);names=[str(x) for x in data['names']]
        chosen=str(data['chosen']);diagnostic=bool(data['diagnostic'])
        image_id=str(data['image_id']);classname=str(data['classname'])
    if image.shape[:2]!=truth.shape or maps.shape!=(5,64,64) or masks.shape!=(5,*truth.shape):
        raise ValueError('unexpected SG18 visual cache shapes')
    def iou(mask):return float((mask&truth).sum()/max((mask|truth).sum(),1))
    fig,axes=plt.subplots(2,4,figsize=(16,8.5),layout='constrained')
    panels=[('Generated image',image,None,None),('CA seed',normalize(seed.reshape(64,64)),'magma',None)]
    for name in ('W/REF','W/T1','W/T2','W/T3'):
        j=names.index(name)
        panels.append((f'{name} · IoU {iou(masks[j]):.3f}',normalize(maps[j]),'magma',masks[j]))
    panels.append(('Original GT',truth,'gray',None))
    ref=masks[names.index('W/REF')];can=masks[names.index(chosen)]
    overlay=image.copy()
    gained=can&~ref;removed=ref&~can
    overlay[gained]=(.35*overlay[gained]+.65*np.array([40,190,75])).astype(np.uint8)
    overlay[removed]=(.35*overlay[removed]+.65*np.array([225,65,145])).astype(np.uint8)
    panels.append((f'{chosen} vs REF · green gained / pink removed',overlay,None,None))
    for ax,(title,data,cmap,mask) in zip(axes.flat,panels):
        ax.imshow(data,cmap=cmap,vmin=0 if cmap else None,vmax=1 if cmap else None)
        if mask is not None:
            # Contour at annotation size; the heatmap remains visually readable.
            from scipy.ndimage import zoom
            display=np.asarray(mask)[::8,::8] if mask.shape==(512,512) else zoom(mask,(64/mask.shape[0],64/mask.shape[1]),order=0)
            ax.contour(display,levels=[.5],colors='cyan',linewidths=.55)
        ax.set_title(title,fontsize=10);ax.axis('off')
    label='diagnostic only' if diagnostic else 'VOC-frozen selected'
    fig.suptitle(f'{classname} · {image_id} · {label}',fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=150)
    plt.close(fig)

if __name__=='__main__':main()
