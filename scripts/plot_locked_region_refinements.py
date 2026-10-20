#!/usr/bin/env python3
"""Small SG15 presentation sheets; no torch/model dependencies."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_package(directory,summary,keep_cache=False):
    out=directory/'visuals';out.mkdir(parents=True,exist_ok=True)
    cache=directory/'visual_cache'
    fixed=sorted(cache.glob('*_representative.npz'))
    candidates=[]
    for key,reverse in [('best',True),('worst',False)]:
        paths=list(cache.glob(f'*_{key}.npz'))
        scores=[]
        for p in paths:
            with np.load(p) as data:scores.append((float(data['extreme'][0]),p))
        if scores:candidates.append(sorted(scores,key=lambda x:x[0],reverse=reverse)[0][1])
    selected=fixed+candidates
    index=[]
    for path in selected:
        with np.load(path) as d:
            image,truth,smalllabels,maps=d['image'],d['target'].astype(bool),d['labels'],d['maps']
            names=d['names'].tolist();image_id=str(d['image_id']);classname=str(d['classname']);extreme=d['extreme']
        labels=zoom(smalllabels.reshape(64,64),np.array(truth.shape)/64,order=0,grid_mode=True,mode='nearest')
        # Use exact half-pixel bilinear coordinates (torch align_corners=False).
        from scipy.ndimage import map_coordinates
        y=(np.arange(truth.shape[0])+.5)*64/truth.shape[0]-.5
        x=(np.arange(truth.shape[1])+.5)*64/truth.shape[1]-.5
        yy,xx=np.meshgrid(y,x,indexing='ij')
        fig,axes=plt.subplots(3,8,figsize=(24,9),squeeze=False)
        kinds=('A025','A050','A075','B')
        usekind=kinds[int(extreme[2])] if path in candidates else None
        for pi,name in enumerate(names):
            raw=maps[pi].reshape(64,64);den=raw.max()-raw.min()
            norm=(raw-raw.min())/den if den>0 else np.zeros_like(raw)
            score=map_coordinates(norm,[yy,xx],order=1,mode='nearest')
            t=summary['cells'][name]['voc_threshold'];before=score>=t
            area=np.bincount(labels.ravel(),minlength=32)
            fraction=np.bincount(labels[before],minlength=32)/np.maximum(area,1)
            # For fixed representatives use the family-selected output if any
            # passed; otherwise show B explicitly as a failed diagnostic.
            kind=usekind or 'B'
            if kind=='B':
                after=before.copy();after[fraction[labels]>=.75]=True;after[fraction[labels]<=.25]=False
            else:after=fraction[labels]>=dict(A025=.25,A050=.5,A075=.75)[kind]
            change=np.ones((*truth.shape,3),float)*.93
            change[(~before&after&truth)|(before&~after&~truth)]=[.1,.7,.2]
            change[(before&~after&truth)|(~before&after&~truth)]=[.9,.1,.2]
            def iou(m):return (m&truth).sum()/max((m|truth).sum(),1)
            fields=(image,labels,score,before,fraction[labels],after,truth,change)
            titles=('RGB','SPEC32',f'{name}\nscore',f'Prior t={t:.3f}\nIoU={iou(before):.3f}',
                    'Region FG fraction',f'SG15-{kind}\nIoU={iou(after):.3f}','GT','Changes: green useful / red harmful')
            for j,(v,title) in enumerate(zip(fields,titles)):
                axes[pi,j].imshow(v,cmap=('tab20' if j==1 else 'magma' if j in (2,4) else 'gray'),vmin=0 if j!=1 else None,vmax=1 if j in (2,4) else None)
                axes[pi,j].set_title(title,fontsize=8);axes[pi,j].axis('off')
        fig.suptitle(f'{classname}: {image_id}; {path.stem}; diagnostic illustrations, not COCO selection',fontsize=11)
        fig.tight_layout();dest=out/(path.stem+'.jpg');fig.savefig(dest,dpi=130);plt.close(fig)
        index.append({'file':dest.name,'image_id':image_id,'class':classname,'selection':path.stem,
                      'extreme_sampling':'largest/smallest VOC image-IoU change across all refinements at fixed .45; shown at frozen prior thresholds'})
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    if not keep_cache:
        for p in cache.glob('*.npz'):p.unlink()
    print(f'SG15: {len(index)} sheets; temporary image/map caches removed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--keep-cache',action='store_true')
    a=p.parse_args();plot_package(a.output_dir,json.loads((a.output_dir/'summary.json').read_text()),a.keep_cache)
