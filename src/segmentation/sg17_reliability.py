"""Literal SG17 continuous-score gate and same-input mechanism accounting."""
from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage
from src.segmentation import locked_refinements as lr

N=248
NAMES=('W/P1','W/SG17-Q70','W/SG17-Q80','W/SG17-Q90')


@torch.inference_mode()
def observations(calls,seed):
    """Reconstruct one affinity at a time; return y1, y2, integer q_up count."""
    c=seed.reshape(-1).float()
    first=torch.zeros_like(c);second=torch.zeros_like(c)
    count=torch.zeros_like(c,dtype=torch.int16)
    n=0
    for query,key in calls:
        q=query.to(c.device).float();k=key.to(c.device).float()
        a=torch.softmax(q@k.transpose(-1,-2)/(q.shape[-1]**.5),-1)
        z=a@c;zz=(a@z[...,None])[...,0]
        first+=z.sum(0);second+=zz.sum(0);count+=(zz>z).sum(0).short();n+=len(q)
        del q,k,a,z,zz
    if n!=N:raise ValueError(f'expected {N} observations, got {n}')
    return first/n,second/n,count.to(torch.uint8)


def quantile_cutoffs(hist,quantiles=(.70,.80,.90)):
    """No GT input: discrete upper-tail cutoffs from raw positive VOC increments."""
    hist=np.asarray(hist,np.int64)
    if hist.shape!=(N+1,) or hist.sum()<=0 or (hist<0).any():raise ValueError('invalid VOC q histogram')
    tail=hist[::-1].cumsum()[::-1]
    out={}
    for q in quantiles:
        want=1-q
        # Integer agreement counts can make the requested fraction impossible.
        # Keep a nonempty tail and disclose the attained fraction.
        candidates=np.flatnonzero(tail>0)
        cut=int(min(candidates,key=lambda k:(abs(tail[k]/hist.sum()-want),-k)))
        out[f'Q{int(round(100*q))}']={'count_cutoff':cut,'q_cutoff':cut/N,
            'requested_top_fraction':want,'actual_retained_fraction':float(tail[cut]/hist.sum())}
    if not (out['Q70']['count_cutoff']<=out['Q80']['count_cutoff']<=out['Q90']['count_cutoff']):
        raise ValueError('quantile cutoffs not ordered')
    return out


def candidate_maps(first,second,count,cutoffs):
    """Gate only positive raw second-pass increments, before normalization."""
    first=torch.as_tensor(first).reshape(-1).float()
    second=torch.as_tensor(second).reshape(-1).float()
    count=torch.as_tensor(count).reshape(-1)
    if first.numel()!=4096 or second.shape!=first.shape or count.shape!=first.shape:
        raise ValueError('expected 64x64 input maps')
    delta=second-first
    maps=[second]
    for name in ('Q70','Q80','Q90'):
        keep=count>=cutoffs[name]['count_cutoff']
        maps.append(torch.where((delta>0)&~keep,first,second))
    return torch.stack(maps).reshape(4,64,64)


def native_positive_hist(first,second,count):
    delta=np.asarray(second).reshape(-1)-np.asarray(first).reshape(-1)
    q=np.asarray(count).reshape(-1)
    return np.bincount(q[delta>0].astype(np.int64),minlength=N+1).astype(np.int64)


def full_masks(first,second,candidate,truth,threshold_ref,threshold_candidate):
    """Use the existing independent minmax->bilinear->threshold convention."""
    scores=lr.full_scores(torch.stack([torch.as_tensor(x).reshape(-1) for x in
                                      (first,second,candidate)]),np.asarray(truth).shape).cpu().numpy()
    return scores[0]>=threshold_ref,scores[1]>=threshold_ref,scores[2]>=threshold_candidate


def mechanism(first_mask,p1_mask,candidate_mask,truth,q_native,cutoff,delta_native):
    """Account for all P1 binary additions; GT enters only this diagnostic."""
    truth=np.asarray(truth,bool);one=np.asarray(first_mask,bool)
    p1=np.asarray(p1_mask,bool);cand=np.asarray(candidate_mask,bool)
    add=p1&~one;retained=add&cand;rejected=add&~cand
    gt_dist=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
    labels,_=ndimage.label(p1)
    touched=np.unique(labels[p1&truth]);touched=touched[touched>0]
    separate=p1&~np.isin(labels,touched)
    import torch.nn.functional as F
    q=torch.as_tensor(np.asarray(q_native).reshape(1,1,64,64),dtype=torch.float32)/N
    qfull=F.interpolate(q,size=truth.shape,mode='nearest')[0,0].numpy()
    native_rejected=(np.asarray(delta_native).reshape(64,64)>0)&(np.asarray(q_native).reshape(64,64)<cutoff)
    raw_rejected=F.interpolate(torch.as_tensor(native_rejected)[None,None].float(),size=truth.shape,
                               mode='nearest')[0,0].numpy().astype(bool)
    def counts(mask):
        fp=mask&~truth
        return {'tp':int((mask&truth).sum()),'fp':int(fp.sum()),
                'fp_gt16':int((fp&(gt_dist>16)).sum()),
                'fp_gt64':int((fp&(gt_dist>64)).sum()),
                'fp_separate':int((fp&separate).sum())}
    def mean(mask):return float(qfull[mask].mean()) if mask.any() else None
    result={'p1_new_additions':int(add.sum()),'retained':counts(retained),'rejected':counts(rejected),
            'raw_gate_rejected_positive_increment_pixels':int(raw_rejected.sum()),
            'p1_additions_outside_raw_positive_increment':int((add&~F.interpolate(
               torch.as_tensor(np.asarray(delta_native).reshape(64,64)>0)[None,None].float(),
               size=truth.shape,mode='nearest')[0,0].numpy().astype(bool)).sum()),
            'mean_q_rejected_p1_additions':mean(rejected),
            'mean_q_retained_p1_additions':mean(retained),
            'candidate_new_additions_outside_p1':int((cand&~p1).sum()),
            'p1_removals_reversed_by_candidate':int((~p1&one&cand).sum())}
    if result['retained']['tp']+result['rejected']['tp']!=int((add&truth).sum()):
        raise ValueError('TP addition accounting failed')
    if result['retained']['fp']+result['rejected']['fp']!=int((add&~truth).sum()):
        raise ValueError('FP addition accounting failed')
    return result


def reliability_bins(one_mask,p1_mask,truth,q_native,edges):
    """GT-only breakdown of P1 binary additions by native-q score bins."""
    truth=np.asarray(truth,bool);add=np.asarray(p1_mask,bool)&~np.asarray(one_mask,bool)
    import torch.nn.functional as F
    q=F.interpolate(torch.as_tensor(np.asarray(q_native).reshape(1,1,64,64)).float(),
                    size=truth.shape,mode='nearest')[0,0].numpy()
    gt_dist=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
    hist=np.zeros((10,5),np.int64)  # total, TP, FP, FP >16, FP >64
    for i in range(10):
        selected=add&(q>=edges[i])&(q<(edges[i+1] if i<9 else N+1))
        fp=selected&~truth
        hist[i]=[selected.sum(),(selected&truth).sum(),fp.sum(),
                 (fp&(gt_dist>16)).sum(),(fp&(gt_dist>64)).sum()]
    if hist[:,0].sum()!=add.sum():raise ValueError('reliability bins do not partition additions')
    return hist
