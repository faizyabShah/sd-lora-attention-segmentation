#!/usr/bin/env python3
"""Render one fixed SG15 current/oracle comparison sheet from one transient NPZ."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def iou(prediction, truth):
    union=np.count_nonzero(prediction | truth)
    return np.count_nonzero(prediction & truth)/union if union else 1.


def difference(prediction, oracle):
    # 0 shared background, 1 shared foreground, 2 false region inclusion,
    # 3 missed oracle-positive pixels. Evaluated against oracle, not GT.
    return (prediction & oracle).astype(np.uint8) + 2*(prediction & ~oracle).astype(np.uint8) + 3*(~prediction & oracle).astype(np.uint8)


def plot(cache:Path, output:Path):
    with np.load(cache,allow_pickle=False) as data:
        image=data['image'];truth=data['truth'].astype(bool);labels=data['labels'].astype(np.int32)
        names=data['names'].tolist();thresholds=data['thresholds'];priors=data['prior'].astype(bool)
        majority=data['majority'].astype(bool);snapped=data['snapped'].astype(bool);oracle=data['oracle'].astype(bool)
        image_id=str(data['image_id']);classname=str(data['classname'])
    assert priors.shape==majority.shape==snapped.shape==(3,*truth.shape)
    assert labels.shape==oracle.shape==truth.shape
    color=ListedColormap(['#f5f5f5','#187b46','#d62828','#2563eb'])
    fig,axes=plt.subplots(3,9,figsize=(27,10),squeeze=False)
    for row,name in enumerate(names):
        original=priors[row];current=majority[row];snap=snapped[row]
        images=(image,truth,labels,original,current,snap,oracle,difference(current,oracle),difference(snap,oracle))
        titles=('Generated RGB','Original GT mask','SPEC32 regions',
                f'{name}\nprior t={thresholds[row]:.3f}; IoU={iou(original,truth):.3f}',
                f'Current A050\nIoU={iou(current,truth):.3f}',
                f'Current B\nIoU={iou(snap,truth):.3f}',
                f'GT-majority oracle\nIoU={iou(oracle,truth):.3f}',
                'A050 vs oracle\nred extra / blue missing',
                'B vs oracle\nred extra / blue missing')
        for col,(value,title) in enumerate(zip(images,titles)):
            kwargs={'interpolation':'nearest'}
            if col==2:kwargs.update(cmap='tab20',vmin=0,vmax=31)
            elif col in (7,8):kwargs.update(cmap=color,vmin=0,vmax=3)
            elif col not in (0,):kwargs.update(cmap='gray',vmin=0,vmax=1)
            axes[row,col].imshow(value,**kwargs);axes[row,col].set_title(title,fontsize=8)
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    fig.suptitle(f'{classname}: {image_id} | same generated image and SPEC32 regions across three frozen priors',fontsize=13)
    fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=130);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot(args.cache,args.output)
