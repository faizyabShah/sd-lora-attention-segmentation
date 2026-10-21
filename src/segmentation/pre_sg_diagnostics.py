"""Training-free measurements for the prerequisites of SG16 and SG17.

Ground truth is used only after all continuous signals and frozen masks exist.
The pixel AUC accumulator uses 1024 fixed bins, all eligible pixels, and no
sampling. Region AUC is computed later from all individual region records.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from src.segmentation import locked_refinements as lr

PIXEL_BOUNDS = {
    'first_r_up': (0,1), 'first_r_down': (0,1), 'first_r_max': (0,1),
    'first_std': (0,1), 'first_mad': (0,1), 'first_iqr': (0,1),
    'first_mean_median_gap': (0,1), 'first_magnitude': (0,1),
    'first_mean': (0,1), 'ca_seed': (0,1),
    'support_reliability': (-1,1), 'top50_support_fraction': (0,1),
    'all_top50_agreement': (0,1), 'second_q_up': (0,1),
    'second_q_down': (0,1), 'second_q_direction': (0,1),
    'second_median_direction': (-1,1), 'second_trim10_direction': (-1,1),
    'second_std': (0,1), 'second_mad': (0,1), 'second_iqr': (0,1),
    'second_mean_abs': (0,1), 'top50_q_direction': (0,1),
    'top50_median_direction': (-1,1), 'top50_trim10_direction': (-1,1),
    'top50_std': (0,1), 'top50_mad': (0,1), 'top50_iqr': (0,1),
    'top50_mean_abs': (0,1)
}
PIXEL_FEATURES = tuple(PIXEL_BOUNDS)
ENDPOINTS = ('useful_all','useful_additions','useful_removals',
             'new_tp_vs_fp','new_tp_vs_fp_gt16','new_tp_vs_fp_gt64',
             'new_tp_vs_fp_separate')
REGION_SUBSETS = ('all','ambiguous_p1','wrong_p1','priors_agree','priors_disagree')


@torch.inference_mode()
def collect(calls, seed):
    """Original call-major/head-minor A_i c and A_i^2 c, plus SG15 graph."""
    seed=seed.reshape(-1).float()
    graph=torch.zeros((seed.numel(),seed.numel()),device=seed.device)
    first=[];second=[]
    for query,key in calls:
        q=query.to(seed.device).float();k=key.to(seed.device).float()
        a=torch.softmax(q @ k.transpose(-1,-2)/math.sqrt(q.shape[-1]),-1)
        z=a @ seed
        first.append(z)
        second.append((a @ z[...,None])[...,0])
        graph+=a.mean(0)
        del q,k,a,z
    first=torch.cat(first);second=torch.cat(second)
    if first.shape!=(248,4096) or second.shape!=first.shape:
        raise ValueError(f'unexpected observations {first.shape}, {second.shape}')
    return first,second,graph/len(calls)


def quantile(x,p,axis=0):
    return np.quantile(x,p,axis=axis,method='linear')


def observation_stats(values,prefix):
    """Statistics across individual observation values at each region."""
    ordered=np.sort(values,axis=0)
    n=len(values);median=np.median(ordered,axis=0)
    return {
        f'{prefix}_mean':values.mean(0),
        f'{prefix}_median':median,
        f'{prefix}_std':values.std(0),
        f'{prefix}_mad':np.median(np.abs(values-median[None]),axis=0),
        f'{prefix}_iqr':quantile(values,.75)-quantile(values,.25),
        f'{prefix}_trim10':ordered[int(.1*n):n-int(.1*n)].mean(0),
        f'{prefix}_trim20':ordered[int(.2*n):n-int(.2*n)].mean(0),
        f'{prefix}_min':ordered[0],f'{prefix}_max':ordered[-1]
    }


def region_records(labels64,labels_full,truth,prior_masks,seed,first,selected):
    """One exact row per nonempty annotation-resolution SPEC32 region."""
    labels64=np.asarray(labels64,dtype=np.int32).reshape(-1)
    labels_full=np.asarray(labels_full,dtype=np.int32)
    truth=np.asarray(truth,bool)
    seed=np.asarray(seed).reshape(-1);first=np.asarray(first)
    selected=np.asarray(selected,dtype=np.int32)
    one=first.mean(0)
    prior=np.asarray(prior_masks,bool)
    gt_fraction,area=lr.region_fractions(labels_full,truth)
    occupancies=np.stack([lr.region_fractions(labels_full,m)[0] for m in prior])
    records=[]
    for k in np.flatnonzero(area):
        member=labels64==k;full=labels_full==k
        s=first[:,member].mean(1)
        so=s[selected]
        c=seed[member];p=one[member]
        ck=float(c.mean());f=occupancies[:,k]
        rec={'region_id':int(k),'oracle':int(gt_fraction[k]>=.5),
             'gt_fraction':float(gt_fraction[k]),'region_area_pixels':int(area[k]),
             'region_area_fraction':float(area[k]/truth.size),
             'object_area_fraction':float(truth.mean()),
             'p1_occupancy':float(f[0]),'sg11_occupancy':float(f[1]),
             'sg12_occupancy':float(f[2]),'occupancy_mean':float(f.mean()),
             'occupancy_min':float(f.min()),'occupancy_max':float(f.max()),
             'occupancy_std':float(f.std()),'majority_votes':int((f>=.5).sum()),
             'ca_mean':ck,'ca_median':float(np.median(c)),
             'ca_max':float(c.max()),'ca_p90':float(quantile(c,.9)),
             'p1_mean':float(p.mean()),'p1_median':float(np.median(p)),
             'p1_max':float(p.max()),'p1_p90':float(quantile(p,.9))}
        rec.update({n:float(v) for n,v in observation_stats(s[:,None],'obs_all').items()})
        rec.update({n:float(v) for n,v in observation_stats(so[:,None],'obs_top50').items()})
        for prefix,v in [('obs_all',s),('obs_top50',so)]:
            rec[f'{prefix}_r_up']=float(np.mean(v>ck))
            rec[f'{prefix}_r_down']=float(np.mean(v<ck))
            rec[f'{prefix}_r_max']=max(rec[f'{prefix}_r_up'],rec[f'{prefix}_r_down'])
        records.append(rec)
    oracle=(gt_fraction>=.5)&(area>0)
    oracle_mask=oracle[labels_full]
    p1=prior[0]
    def iou(mask):
        union=np.count_nonzero(mask|truth)
        return float(np.count_nonzero(mask&truth)/union) if union else 1.
    edge=truth & ~ndimage.binary_erosion(truth)
    perimeter=int(edge.sum());object_area=int(truth.sum())
    image={'oracle_gain_vs_p1':iou(oracle_mask)-iou(p1),
           'oracle_iou':iou(oracle_mask),'p1_iou':iou(p1),
           'object_area_fraction':float(truth.mean()),
           'boundary_length_over_area':float(perimeter/max(object_area,1)),
           'compactness':float(4*np.pi*object_area/max(perimeter**2,1)),
           'oracle_positive_regions':int(oracle.sum()),
           'object_crossed_regions':int(np.count_nonzero(np.bincount(labels_full[truth],minlength=32)))}
    return records,image


@torch.inference_mode()
def pixel_features(seed,first,second,reliability,selected):
    """All pixel features at native 64x64, independent of GT and predictions.

    Directional signed features are converted to the proposed binary change
    direction in pixel_histograms, after the frozen thresholded masks exist.
    """
    seed=seed.reshape(-1).float()
    selected=selected.long()
    d=second-first
    ordered=first.sort(0).values
    median=(ordered[123]+ordered[124])/2
    dsort=d.sort(0).values
    dmedian=(dsort[123]+dsort[124])/2
    td=d[selected];tsort=td.sort(0).values
    tmedian=(tsort[61]+tsort[62])/2
    def q_iqr(sorted_values):
        n=len(sorted_values)
        lo=(n-1)*.25;hi=(n-1)*.75
        def interp(v):
            i=int(v);return sorted_values[i]*(1-(v-i))+sorted_values[i+1]*(v-i)
        return interp(hi)-interp(lo)
    first_mean=first.mean(0)
    up=(first>seed).float().mean(0);down=(first<seed).float().mean(0)
    support_up=d>0;support_down=d<0
    rel=reliability.reshape(-1,1)
    supup=(support_up*rel).sum(0)/support_up.sum(0).clamp_min(1)
    supdown=(support_down*rel).sum(0)/support_down.sum(0).clamp_min(1)
    out={
        'first_r_up':up,'first_r_down':down,'first_r_max':torch.maximum(up,down),
        'first_std':first.std(0,correction=0),
        'first_mad':torch.quantile((first-median).abs(),.5,dim=0),
        'first_iqr':q_iqr(ordered),
        'first_mean_median_gap':(first_mean-median).abs(),
        'first_magnitude':(first_mean-seed).abs(),
        'first_mean':first_mean,'ca_seed':seed,
        'support_reliability_up':supup,'support_reliability_down':supdown,
        'top50_support_up':(td>0).float().mean(0),
        'top50_support_down':(td<0).float().mean(0),
        'second_q_up':support_up.float().mean(0),
        'second_q_down':support_down.float().mean(0),
        'second_median':dmedian,'second_trim10':dsort[24:224].mean(0),
        'second_std':d.std(0,correction=0),
        'second_mad':torch.quantile((d-dmedian).abs(),.5,dim=0),
        'second_iqr':q_iqr(dsort),'second_mean_abs':d.abs().mean(0),
        'top50_median':tmedian,'top50_trim10':tsort[12:112].mean(0),
        'top50_std':td.std(0,correction=0),
        'top50_mad':torch.quantile((td-tmedian).abs(),.5,dim=0),
        'top50_iqr':q_iqr(tsort),'top50_mean_abs':td.abs().mean(0)
    }
    return out


def pixel_histograms(features,one,two,truth,bins=1024,threshold=.45):
    """Histogram every changed annotation pixel for seven diagnostic endpoints."""
    truth=np.asarray(truth,bool);shape=truth.shape
    from src.segmentation import controlled_cv as cv
    scores=lr.full_scores(torch.stack((one, two)),shape).cpu().numpy()
    before,after=scores>=threshold
    add=after&~before;remove=before&~after;changed=add|remove
    useful=(add&truth)|(remove&~truth)
    harmful=changed&~useful
    newtp=add&truth;newfp=add&~truth
    distance=ndimage.distance_transform_edt(~truth) if truth.any() else np.full(shape,np.inf)
    components,_=ndimage.label(after)
    touch=np.unique(components[after&truth]);touch=touch[touch>0]
    separate=after&~np.isin(components,touch)
    negative=[harmful, harmful&add,harmful&remove,newfp,newfp&(distance>16),
              newfp&(distance>64),newfp&separate]
    positive=[useful,useful&add,useful&remove,newtp,newtp,newtp,newtp]
    assert len(negative)==len(ENDPOINTS)
    hist=np.zeros((len(ENDPOINTS),2,len(PIXEL_FEATURES),bins),np.int64)
    def up(v):
        if isinstance(v,np.ndarray):v=torch.as_tensor(v,device=one.device)
        return F.interpolate(v.reshape(1,1,64,64).float(),size=shape,mode='bilinear',align_corners=False)[0,0]
    # Direction is assigned at the annotation pixel after interpolation.
    direction_np=np.where(add,1.,-1.)
    for fi,name in enumerate(PIXEL_FEATURES):
        if name in ('support_reliability','top50_support_fraction','second_q_direction','top50_q_direction'):
            source={'support_reliability':('support_reliability_up','support_reliability_down'),
                    'top50_support_fraction':('top50_support_up','top50_support_down'),
                    'second_q_direction':('second_q_up','second_q_down'),
                    'top50_q_direction':('top50_support_up','top50_support_down')}[name]
            field=np.where(add,up(features[source[0]]).cpu().numpy(),up(features[source[1]]).cpu().numpy())
        elif name=='all_top50_agreement':
            allq=np.where(add,up(features['second_q_up']).cpu().numpy(),up(features['second_q_down']).cpu().numpy())
            topq=np.where(add,up(features['top50_support_up']).cpu().numpy(),up(features['top50_support_down']).cpu().numpy())
            field=1-np.abs(allq-topq)
        elif name.endswith('_direction'):
            field=up(features[name.removesuffix('_direction')]).cpu().numpy()*direction_np
        else:
            field=up(features[name]).cpu().numpy()
        lo,hi=PIXEL_BOUNDS[name]
        quant=np.clip(((field-lo)/(hi-lo)*bins).astype(np.int32),0,bins-1)
        for ei in range(len(ENDPOINTS)):
            hist[ei,0,fi]=np.bincount(quant[positive[ei]],minlength=bins)
            hist[ei,1,fi]=np.bincount(quant[negative[ei]],minlength=bins)
    first_count=lr.mask_counts(before,truth);second_count=lr.mask_counts(after,truth)
    def iou(c):return float(c[0]/max(c[:3].sum(),1))
    image={'useful':int(useful.sum()),'harmful':int(harmful.sum()),
           'additions':int(add.sum()),'removals':int(remove.sum()),
           'new_tp':int(newtp.sum()),'new_fp':int(newfp.sum()),
           'new_fp_gt16':int((newfp&(distance>16)).sum()),
           'new_fp_gt64':int((newfp&(distance>64)).sum()),
           'new_fp_separate':int((newfp&separate).sum()),
           'one_pass_iou':iou(first_count),'two_pass_iou':iou(second_count),
           'two_pass_iou_gain':iou(second_count)-iou(first_count)}
    # All changed pixels, with the same full-resolution interpolation as AUC.
    for name in ('second_q_direction','top50_q_direction','support_reliability','first_r_max'):
        if name=='first_r_max':f=up(features[name]).cpu().numpy()
        else:
            keys={'second_q_direction':('second_q_up','second_q_down'),
                  'top50_q_direction':('top50_support_up','top50_support_down'),
                  'support_reliability':('support_reliability_up','support_reliability_down')}[name]
            f=np.where(add,up(features[keys[0]]).cpu().numpy(),up(features[keys[1]]).cpu().numpy())
        image[name+'_mean']=float(f[changed].mean()) if changed.any() else None
        image[name+'_median']=float(np.median(f[changed])) if changed.any() else None
    return hist,image
