"""Locked one-pass temporal windows and segmentation-change accounting."""
from __future__ import annotations
import time
import numpy as np
import torch
from scipy import ndimage
from src.segmentation import locked_refinements as lr

NAMES=('W/REF','W/T1','W/T2','W/T3','W/P1')
WINDOWS={'W/T1':(10,21),'W/T2':(10,31),'W/T3':(6,31)}
EVENT_NAMES=('tp_add','fp_add','fp_gt16','fp_gt64','fp_separate',
             'useful_remove','harmful_remove')


@torch.inference_mode()
def aggregate(calls,seed,names=NAMES):
    """Reconstruct each call once and average exactly the registered heads/calls.

    P1 is the observation-wise A_i^2 c mean. Unrequested methods are not
    computed, which permits honest post-capture runtime benchmarking.
    """
    requested=set(names)
    if not requested or not requested<=set(NAMES):raise ValueError('unknown SG18 cell')
    c=seed.reshape(-1).float();sums={n:torch.zeros_like(c) for n in requested}
    started=time.perf_counter()
    if len(c)!=4096 or len(calls)!=31:raise ValueError('expected 31 calls at 64x64')
    for t,(query,key) in enumerate(calls):
        if query.shape[0]!=8:raise ValueError('expected eight SA heads')
        active=[n for n in requested if n in WINDOWS and WINDOWS[n][0]<=t<WINDOWS[n][1]]
        if 'W/REF' not in requested and 'W/P1' not in requested and not active:
            continue
        q=query.to(c.device).float();k=key.to(c.device).float()
        a=torch.softmax(q@k.transpose(-1,-2)/(q.shape[-1]**.5),-1)
        z=a@c
        if 'W/REF' in requested:sums['W/REF']+=z.sum(0)
        for n in active:sums[n]+=z.sum(0)
        if 'W/P1' in requested:sums['W/P1']+=(a@z[...,None])[...,0].sum(0)
        del q,k,a,z
    if c.is_cuda:torch.cuda.synchronize(c.device)
    elapsed=time.perf_counter()-started
    denominators={'W/REF':31*8,'W/P1':31*8,
                  **{n:(stop-start)*8 for n,(start,stop) in WINDOWS.items()}}
    return {n:sums[n]/denominators[n] for n in names},elapsed


def masks(maps,seed,truth,thresholds,ca_threshold=.45):
    """Use the unchanged native minmax and bilinear mask convention."""
    names=list(maps);allmaps=torch.stack([torch.as_tensor(seed).reshape(-1)]+[torch.as_tensor(maps[n]).reshape(-1) for n in names])
    scores=lr.full_scores(allmaps,np.asarray(truth).shape).cpu().numpy()
    ca=scores[0]>=ca_threshold
    return ca,{n:scores[i+1]>=thresholds[n] for i,n in enumerate(names)}


def events(ca,mask,truth,distance=None):
    """CA-to-method additions/removals at annotation resolution."""
    ca=np.asarray(ca,bool);mask=np.asarray(mask,bool);truth=np.asarray(truth,bool)
    add=mask&~ca;remove=ca&~mask;fp=add&~truth
    if distance is None:
        distance=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
    labels,_=ndimage.label(mask)
    touching=np.unique(labels[mask&truth]);touching=touching[touching>0]
    separate=mask&~np.isin(labels,touching)
    count=np.array([(add&truth).sum(),fp.sum(),(fp&(distance>16)).sum(),
                    (fp&(distance>64)).sum(),(fp&separate).sum(),
                    (remove&~truth).sum(),(remove&truth).sum()],np.int64)
    return count,{'add':add,'tp_add':add&truth,'fp_add':fp,
                  'fp_gt16':fp&(distance>16),'fp_gt64':fp&(distance>64),
                  'fp_separate':fp&separate}


def rates(count):
    c=np.asarray(count,np.int64);tp,fp,far16,far64,separate,ur,hr=map(int,c)
    div=lambda a,b:float(a/b) if b else None
    return {'tp_add':tp,'fp_add':fp,'fp_gt16':far16,'fp_gt64':far64,
            'fp_separate':separate,'useful_remove':ur,'harmful_remove':hr,
            'tp_fp_ratio':div(tp,fp),'tp_fraction_additions':div(tp,tp+fp),
            'fp_gt16_fraction':div(far16,fp),'fp_gt64_fraction':div(far64,fp),
            'fp_separate_fraction':div(separate,fp),
            'useful_change_fraction':div(tp+ur,tp+fp+ur+hr)}


def compare(reference,candidate,truth,ca,distance=None):
    """Direct CA-addition overlap, including retained/lost/new TP and FP."""
    truth=np.asarray(truth,bool);ca=np.asarray(ca,bool)
    ref=np.asarray(reference,bool)&~ca;cand=np.asarray(candidate,bool)&~ca
    if distance is None:
        distance=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape,np.inf)
    _,ref_events=events(ca,reference,truth,distance)
    removed=ref&~cand;introduced=cand&~ref;overlap=ref&cand
    return {'tp_retained':int((overlap&truth).sum()),
            'tp_lost':int((removed&truth).sum()),
            'tp_new':int((introduced&truth).sum()),
            'fp_retained':int((overlap&~truth).sum()),
            'fp_removed':int((removed&~truth).sum()),
            'fp_new':int((introduced&~truth).sum()),
            'distant_fp_gt16_removed':int((removed&ref_events['fp_gt16']).sum()),
            'distant_fp_gt64_removed':int((removed&ref_events['fp_gt64']).sum()),
            'separate_component_fp_removed':int((removed&ref_events['fp_separate']).sum()),
            'reference_new_tp_supported':int((cand&ref_events['tp_add']).sum()),
            'reference_new_fp_supported':int((cand&ref_events['fp_add']).sum()),
            'reference_distant_fp_supported':int((cand&ref_events['fp_gt16']).sum()),
            'reference_separate_fp_supported':int((cand&ref_events['fp_separate']).sum())}
