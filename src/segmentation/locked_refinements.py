"""Frozen SG12/SG14 propagation and prior-driven SG15 region refinement.

Operators take no ground truth. Labels enter only the separate diagnostics.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from src.segmentation import controlled_cv as cv
from src.segmentation.target_region_inference import expanded_labels

REF, P1, SG11, SG12, SG14 = 'W/REF', 'W/P1', 'W/SG11/trim20/rho0p80', 'W/SG12/top50', 'W/SG14'
CRF_REF, CRF_CAND = P1+'+R2', SG12+'+R2'
PRIORS = (P1, SG11, SG12)


def names_for(stage):
    if stage == 'sg12_crf':
        return (REF, P1, SG12, CRF_REF, CRF_CAND)
    if stage == 'sg14':
        return (REF, SG11, SG12, SG14)
    if stage == 'sg15':
        return (REF, *PRIORS, *[refined_name(p, kind) for p in PRIORS for kind in ('A025','A050','A075','B')])
    raise ValueError(stage)


def refined_name(prior, kind):
    return f'{prior}/SG15-{kind}'


def threshold_sources(stage):
    return {refined_name(p, k): p for p in PRIORS for k in ('A025','A050','A075','B')} if stage == 'sg15' else {}


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def collect_observations(calls, seed, *, need_two=False, need_graph=False):
    """One affinity tensor at a time; optional mean graph for canonical SPEC32."""
    seed = seed.reshape(-1).float()
    one, two = torch.zeros_like(seed), torch.zeros_like(seed)
    graph = torch.zeros((seed.numel(), seed.numel()), device=seed.device) if need_graph else None
    vectors = []
    for query, key in calls:
        q, k = query.to(seed.device).float(), key.to(seed.device).float()
        a = torch.softmax(q @ k.transpose(-1,-2) / math.sqrt(q.shape[-1]), -1)
        z = a @ seed
        vectors.append(z)
        one += z.mean(0)
        if need_two:
            two += (a @ z[...,None])[...,0].mean(0)
        if need_graph:
            graph += a.mean(0)
        del q, k, a, z
    obs = torch.cat(vectors)
    one /= len(calls)
    torch.testing.assert_close(obs.mean(0), one, rtol=1e-5, atol=2e-6)
    if graph is not None:
        graph /= len(calls)
        torch.testing.assert_close(graph @ seed, one, rtol=1e-5, atol=2e-6)
    return obs, one, two / len(calls) if need_two else None, graph


@torch.inference_mode()
def selected_consensus_maps(obs, seed, spec):
    """Exact existing SG12 ranking; SG14 gate and trim use ONLY selected vectors."""
    seed = seed.reshape(-1)
    if obs.shape[0] != spec['sa']['observations'] or not torch.isfinite(obs).all():
        raise ValueError('expected 248 finite observations')
    synchronize(obs.device); start = time.perf_counter()
    unit = F.normalize(obs, p=2, dim=1, eps=1e-12)
    pairwise = unit @ unit.T
    reliability = (pairwise.sum(1)-pairwise.diagonal())/(len(obs)-1)
    ranking = torch.argsort(reliability, descending=True, stable=True)
    k = math.ceil(spec['top_fraction'] * len(obs))
    if k != spec['selected_observations']:
        raise ValueError('locked SG12 must keep exactly 124 observations')
    keep = ranking[:k]
    synchronize(obs.device); ranking_seconds = time.perf_counter()-start
    all_consensus = torch.maximum((obs>seed).float().mean(0), (obs<seed).float().mean(0))
    selected = obs[keep]
    consensus = torch.maximum((selected>seed).float().mean(0), (selected<seed).float().mean(0))
    def trimmed(x):
        cut = int(spec['trim_fraction_per_tail'] * len(x))
        return x.sort(dim=0).values[cut:len(x)-cut].mean(0)
    rho = spec['consensus_rho']
    synchronize(obs.device); reduction_start = time.perf_counter()
    selected_mean = selected.mean(0)
    synchronize(obs.device); reduction_seconds = time.perf_counter()-reduction_start
    maps = {REF: obs.mean(0), SG11: torch.where(all_consensus>=rho,trimmed(obs),seed),
            SG12: selected_mean, SG14: torch.where(consensus>=rho,trimmed(selected),seed)}
    return maps, {'reliability': reliability, 'selected_indices': keep,
                  'all_consensus': all_consensus, 'selected_consensus': consensus,
                  'ranking_seconds': ranking_seconds, 'selected_mean_seconds': reduction_seconds}


@torch.inference_mode()
def full_scores(maps, shape):
    normalized = cv.normalize_batch(maps.reshape(-1,64,64),'minmax')
    return F.interpolate(normalized[:,None],size=shape,mode='bilinear',align_corners=False)[:,0]


def mask_counts(mask, truth):
    tp=int(np.count_nonzero(mask & truth)); fp=int(np.count_nonzero(mask & ~truth))
    fn=int(np.count_nonzero(~mask & truth)); tn=int(np.count_nonzero(~mask & ~truth))
    return np.asarray((tp,fp,fn,tn),np.int64)


def changes(before, after, truth, where=None):
    """FN->TP, FP->TN, TP->FN, TN->FP, in that order."""
    eligible = np.ones_like(truth) if where is None else where
    add=after & ~before & eligible; remove=before & ~after & eligible
    return np.array([(add&truth).sum(),(remove&~truth).sum(),(remove&truth).sum(),(add&~truth).sum()],np.int64)


def region_fractions(labels, mask, regions=32):
    area=np.bincount(labels.ravel(),minlength=regions)
    fg=np.bincount(labels[mask],minlength=regions)
    return np.divide(fg,area,out=np.zeros(regions,float),where=area>0), area


def refine_regions(labels, prior_mask, kind):
    fraction,area=region_fractions(labels,prior_mask)
    if kind == 'B':
        result=prior_mask.copy()
        result[fraction[labels]>=.75]=True
        result[fraction[labels]<=.25]=False
    else:
        q={'A025':.25,'A050':.5,'A075':.75}[kind]
        result=fraction[labels]>=q
    return result,fraction,area


def region_diagnostic(labels, prediction, truth):
    f,area=region_fractions(labels,prediction)
    gt,_=region_fractions(labels,truth)
    positive=(gt>=.5)&(area>0); selected=(f>=.5)&(area>0)
    # Snapping can retain mixed regions: >=.5 predicted occupancy defines a
    # selected region for this diagnostic, never changes its actual mask.
    return np.array([selected.sum(),(selected&positive).sum(),(selected&~positive&(area>0)).sum(),
                     (~selected&positive).sum(),positive.sum()],np.int64)


def complementarity_diagnostics(maps, obs, seed, diag, truth, thresholds, epsilon):
    """Thresholded mask groups plus GT-only per-observation usefulness correlations."""
    names=(REF,SG11,SG12,SG14)
    full=full_scores(torch.stack([maps[n] for n in names]),truth.shape).cpu().numpy()
    groups=np.zeros((len(thresholds),5,3,4),np.int64)
    population=np.zeros((len(thresholds),5),np.int64)
    raw_delta=torch.stack((maps[SG11]-maps[REF],maps[SG12]-maps[REF])).reshape(2,1,64,64)
    raw_delta=F.interpolate(raw_delta,size=truth.shape,mode='bilinear',align_corners=False)[:,0].cpu().numpy()
    a11,a12=np.abs(raw_delta)>epsilon
    raw_categories=((~a11&a12),(a11&~a12),(a11&a12&(raw_delta[0]*raw_delta[1]>0)),
                    (a11&a12&(raw_delta[0]*raw_delta[1]<0)),(~a11&~a12))
    raw_changes=np.zeros_like(groups)
    for j,t in enumerate(thresholds):
        ref,s11,s12,s14=full>=t
        d11=s11.astype(np.int8)-ref; d12=s12.astype(np.int8)-ref
        categories=( (d12!=0)&(d11==0), (d11!=0)&(d12==0),
                     (d11!=0)&(d12==d11), (d11*d12<0), (d11==0)&(d12==0))
        for g,sel in enumerate(categories):
            population[j,g]=sel.sum()
            for m,pred in enumerate((s11,s12,s14)):
                groups[j,g,m]=changes(ref,pred,truth,sel)
                raw_changes[j,g,m]=changes(ref,pred,truth,raw_categories[g])
        # Opposite binary directions cannot occur relative to the same binary
        # REF; leave its exact zero explicitly, do not invent a nonzero count.
    smalltruth=cv.target_at_size(truth,64).to(seed.device).reshape(-1)
    c=seed.reshape(-1); delta=obs-c
    changed=delta.abs()>epsilon
    correct_direction=torch.where(smalltruth,delta>epsilon,delta < -epsilon)
    useful=(changed&correct_direction).float().sum(1)/changed.sum(1).clamp_min(1)
    harmful=(changed&~correct_direction).float().sum(1)/changed.sum(1).clamp_min(1)
    def correlation(a,b):
        a=a.double()-a.double().mean();b=b.double()-b.double().mean()
        denom=a.norm()*b.norm()
        return float((a@b)/denom) if denom>0 else 0.
    # Continuous directions relative to REF are separately recorded, since
    # unlike binary transitions they can genuinely oppose one another.
    d11=maps[SG11]-maps[REF];d12=maps[SG12]-maps[REF]
    active11=d11.abs()>epsilon;active12=d12.abs()>epsilon
    rawgroups=torch.stack(((~active11&active12),(active11&~active12),
                            (active11&active12&(d11*d12>0)),(active11&active12&(d11*d12<0)),(~active11&~active12))).sum(1)
    # Per-observation agreement specifically with useful/harmful REF-vs-seed
    # changes at the fixed, preregistered diagnostic threshold .45.
    norm=cv.normalize_batch(torch.stack((c,maps[REF])).reshape(2,64,64),'minmax').reshape(2,-1)
    before,after=norm>=.45
    useful_ref=(before!=after)&(after==smalltruth);harmful_ref=(before!=after)&(after!=smalltruth)
    direction=maps[REF]-c
    agree=delta*direction>0
    fractions=[]
    for selected_pixels in (useful_ref,harmful_ref):
        fractions.append(agree[:,selected_pixels].float().mean(1) if selected_pixels.any() else torch.zeros(len(obs),device=obs.device))
    reliability=diag['reliability']
    return {'binary_group_changes':groups,'binary_group_pixels':population,
            'continuous_group_changes':raw_changes, 'continuous_full_group_pixels':np.array([g.sum() for g in raw_categories]),
            'continuous_group_pixels':rawgroups.cpu().numpy(),
            'reliability_correlations':np.array([correlation(reliability,useful),correlation(reliability,harmful),
                                               correlation(reliability,fractions[0]),correlation(reliability,fractions[1])]),
            'reliability_agreement_vectors':torch.stack((reliability,useful,harmful,*fractions)).cpu().numpy(),
            'consensus_fractions':np.array([.5,float((diag['all_consensus']<.8).float().mean()),
                       float((diag['selected_consensus']<.8).float().mean()),
                       float(((diag['all_consensus']>=.8)!=(diag['selected_consensus']>=.8)).float().mean())])}
