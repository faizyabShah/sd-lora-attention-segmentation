"""Diagnostic measurements of frozen W self-attention observations.

No ground truth enters attention reconstruction, temporal features, or
structural descriptors. GT is used only in event labels and accounting.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from src.segmentation import locked_refinements as lr

FEATURE_BOUNDS={
    'time_up_fraction':(0,1),'time_down_fraction':(0,1),
    'longest_up_run':(0,31),'longest_down_run':(0,31),
    'sign_change_count':(0,30),
    'early_up_fraction':(0,1),'middle_up_fraction':(0,1),'late_up_fraction':(0,1),
    'early_down_fraction':(0,1),'middle_down_fraction':(0,1),'late_down_fraction':(0,1),
    'late_minus_early':(-1,1),'temporal_mean':(0,1),'temporal_median':(0,1),
    'temporal_std':(0,1),'temporal_mad':(0,1),'temporal_iqr':(0,1),
    'temporal_slope':(-1/30,1/30),'max_deviation_ca':(0,1),
    'final_minus_initial':(-1,1)
}
FEATURES=tuple(FEATURE_BOUNDS)
ENDPOINTS=('new_tp_vs_fp','new_tp_vs_fp_gt16','new_tp_vs_fp_gt64',
           'new_tp_vs_fp_separate','useful_vs_harmful')
DESCRIPTORS=('entropy','median_row_entropy','diagonal_mass','local_mass_r1',
             'local_mass_r2','local_mass_r4','mean_distance','top1_mass',
             'top5_mass','top10_mass','reciprocity','row_entropy_variance',
             'effective_support')
COUNTS=('tp_add','fp_add','fp_gt16','fp_gt64','fp_separate','useful_remove',
        'harmful_remove','reference_new_tp_supported','reference_new_fp_supported')


def _longest(sign):
    # [31,4096] boolean trajectory, then one pass over time.
    run=torch.zeros(sign.shape[1],device=sign.device,dtype=torch.int16)
    best=torch.zeros_like(run)
    for row in sign:
        run=torch.where(row,run+1,0)
        best=torch.maximum(best,run)
    return best.float()


@torch.inference_mode()
def temporal_features(calls,seed):
    """Direct-CA-relative per-pixel statistics of 31 head-averaged maps."""
    x=calls.float().mean(1)  # [31,4096]
    c=seed.reshape(-1).float()
    up=x>c;down=x<c
    median=x.median(0).values
    ordered=x.sort(0).values
    # For 31 values, linear quartile positions are 7.5 and 22.5.
    iqr=.5*(ordered[22]+ordered[23]-ordered[7]-ordered[8])
    t=torch.arange(31,device=x.device,dtype=x.dtype)-15
    slope=(x*t[:,None]).sum(0)/(t*t).sum()
    signs=up.to(torch.int8)-down.to(torch.int8)
    return {
        'time_up_fraction':up.float().mean(0),
        'time_down_fraction':down.float().mean(0),
        'longest_up_run':_longest(up),'longest_down_run':_longest(down),
        'sign_change_count':(signs[1:]!=signs[:-1]).float().sum(0),
        'early_up_fraction':up[:10].float().mean(0),
        'middle_up_fraction':up[10:21].float().mean(0),
        'late_up_fraction':up[21:].float().mean(0),
        'early_down_fraction':down[:10].float().mean(0),
        'middle_down_fraction':down[10:21].float().mean(0),
        'late_down_fraction':down[21:].float().mean(0),
        'late_minus_early':x[21:].mean(0)-x[:10].mean(0),
        'temporal_mean':x.mean(0),'temporal_median':median,
        'temporal_std':x.std(0,correction=0),
        'temporal_mad':(x-median).abs().median(0).values,
        'temporal_iqr':iqr,'temporal_slope':slope,
        'max_deviation_ca':(x-c).abs().amax(0),
        'final_minus_initial':x[-1]-x[0]
    }


class Geometry:
    """One reusable 64x64 query-key geometry per worker."""
    def __init__(self,device):
        coord=torch.stack(torch.meshgrid(torch.arange(64,device=device),
                                         torch.arange(64,device=device),indexing='ij'),-1).reshape(-1,2).float()
        self.distance=torch.cdist(coord,coord)
        self.local={r:self.distance<=r for r in (1,2,4)}


@torch.inference_mode()
def structure(a,geometry):
    """Exact per-head structural descriptors from row-stochastic [8,4096,4096]."""
    n=a.shape[-1]
    entropy=-(a*a.clamp_min(1e-30).log()).sum(-1)
    top=a.topk(10,dim=-1).values
    result={
        'entropy':entropy.mean(-1),
        'median_row_entropy':entropy.median(-1).values,
        'diagonal_mass':a.diagonal(dim1=-2,dim2=-1).mean(-1),
        'mean_distance':(a*geometry.distance).sum((-1,-2))/n,
        'top1_mass':top[...,:1].sum(-1).mean(-1),
        'top5_mass':top[...,:5].sum(-1).mean(-1),
        'top10_mass':top.sum(-1).mean(-1),
        'reciprocity':torch.minimum(a,a.transpose(-1,-2)).sum((-1,-2))/a.sum((-1,-2)),
        'row_entropy_variance':entropy.var(-1,correction=0),
        'effective_support':(1/a.square().sum(-1).clamp_min(1e-30)).mean(-1)
    }
    for r,mask in geometry.local.items():
        result[f'local_mass_r{r}']=(a*mask).sum((-1,-2))/n
    return torch.stack([result[k] for k in DESCRIPTORS],-1).cpu().numpy()


@torch.inference_mode()
def observations(calls,seed,geometry):
    """Stream attention matrices; return 31x8 propagation and descriptors."""
    c=seed.reshape(-1).float();maps=[];descriptors=[]
    for query,key in calls:
        q=query.to(c.device).float();k=key.to(c.device).float()
        a=torch.softmax(q@k.transpose(-1,-2)/(q.shape[-1]**.5),-1)
        z=a@c
        maps.append(z.cpu());descriptors.append(structure(a,geometry))
        del q,k,a,z
    x=torch.stack(maps)
    d=np.stack(descriptors)
    if x.shape!=(31,8,4096) or d.shape!=(31,8,len(DESCRIPTORS)):
        raise ValueError(f'wrong observation shapes {x.shape}, {d.shape}')
    return x,d


def event_context(ca_mask,ordinary_mask,truth):
    """Fixed GT-distance and GT-disjoint-component definitions."""
    truth=np.asarray(truth,bool);ca=np.asarray(ca_mask,bool);after=np.asarray(ordinary_mask,bool)
    add=after&~ca;remove=ca&~after;fp=add&~truth
    distance=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
    labels,_=ndimage.label(after)
    touch=np.unique(labels[after&truth]);touch=touch[touch>0]
    separate=after&~np.isin(labels,touch)
    return {'tp':add&truth,'fp':fp,'fp_gt16':fp&(distance>16),
            'fp_gt64':fp&(distance>64),'fp_separate':fp&separate,
            'useful':(add&truth)|(remove&~truth),
            'harmful':(add&~truth)|(remove&truth),
            'distance':distance,'separate':separate}


def event_counts(ca_mask,mask,truth,reference=None):
    e=event_context(ca_mask,mask,truth)
    ca=np.asarray(ca_mask,bool);after=np.asarray(mask,bool);truth=np.asarray(truth,bool)
    remove=ca&~after
    if reference is None:
        supported_tp=supported_fp=0
    else:
        supported_tp=int((after&reference['tp']).sum())
        supported_fp=int((after&reference['fp']).sum())
    counts=(int(e['tp'].sum()),int(e['fp'].sum()),int(e['fp_gt16'].sum()),
            int(e['fp_gt64'].sum()),int(e['fp_separate'].sum()),
            int((remove&~truth).sum()),int((remove&truth).sum()),
            supported_tp,supported_fp)
    return np.asarray(counts,np.int64)


def feature_histograms(features,events,shape,bins):
    """All changed full-resolution pixels, five preregistered endpoints."""
    pos=[events['tp']]*4+[events['useful']]
    neg=[events['fp'],events['fp_gt16'],events['fp_gt64'],
         events['fp_separate'],events['harmful']]
    hist=np.zeros((len(ENDPOINTS),2,len(FEATURES),bins),np.int64)
    for j,name in enumerate(FEATURES):
        field=F.interpolate(features[name].reshape(1,1,64,64).float(),size=shape,
                            mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        lo,hi=FEATURE_BOUNDS[name]
        quant=np.clip(((field-lo)/(hi-lo)*bins).astype(np.int32),0,bins-1)
        for i in range(len(ENDPOINTS)):
            hist[i,0,j]=np.bincount(quant[pos[i]],minlength=bins)
            hist[i,1,j]=np.bincount(quant[neg[i]],minlength=bins)
    return hist


def binary_masks(maps,shape,threshold):
    scores=lr.full_scores(torch.as_tensor(maps).reshape(-1,4096),shape)
    return (scores>=threshold).cpu().numpy()


def row_rates(counts):
    c=np.asarray(counts)
    tp,fp,far16,far64,sep,ur,hr=[float(c[...,i]) for i in range(7)]
    return {'tp_fp_ratio':tp/fp if fp else None,
            'tp_fraction_additions':tp/(tp+fp) if tp+fp else None,
            'fp_gt16_fraction':far16/fp if fp else None,
            'fp_gt64_fraction':far64/fp if fp else None,
            'fp_separate_fraction':sep/fp if fp else None,
            'useful_change_fraction':(tp+ur)/(tp+fp+ur+hr) if tp+fp+ur+hr else None}
