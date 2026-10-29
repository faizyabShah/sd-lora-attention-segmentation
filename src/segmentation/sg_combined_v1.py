"""SG-combined-v1: conservative hierarchical transport and budgeted completion.

All transport states are nonnegative distributions.  Segmentation-space
normalization is applied only to a finished 64x64 state, never between
constrained resolution transitions.
"""
from __future__ import annotations

import torch

from src.segmentation import controlled_cv as cv
from src.segmentation.sg25_sa import _apply, _attention, _resize
from src.segmentation.sg32_adaptive import background_support

NAMES=tuple(f"C{i}" for i in range(10))
EPS=1e-12


def distribution(value:torch.Tensor)->torch.Tensor:
    x=value.float().clamp_min(0).reshape(-1)+EPS
    return x/x.sum()


def score(value:torch.Tensor)->torch.Tensor:
    return cv.normalize_batch(value.reshape(1,64,64),"minmax")[0].reshape(-1)


def group_sum(children:torch.Tensor,coarse:int)->torch.Tensor:
    fine=coarse*2
    return children.reshape(fine,fine).reshape(coarse,2,coarse,2).sum((1,3))


def project(parent:torch.Tensor,proposal:torch.Tensor,coarse:int,*,rho:float=1.,eps:float=EPS):
    """Closed-form KL projection onto 2x2 parent marginals.

    `proposal` and `parent` are transport-space distributions.  The exact
    output preserves every parent; rho=.5 blends two distributions and is
    intentionally not required to preserve each parent.
    """
    if rho not in (0.,.5,1.) or parent.numel()!=coarse*coarse or proposal.numel()!=(2*coarse)**2:
        raise ValueError("locked SG-combined-v1 parent/proposal/rho mismatch")
    p=distribution(parent).reshape(coarse,coarse)
    q=proposal.float().clamp_min(0).reshape(coarse,2,coarse,2)
    qsum=q.sum()
    if bool(qsum>0):q=q/qsum
    total=q.sum((1,3))
    fallback=total<=eps
    scale=p/total.clamp_min(eps)
    projected=q*scale[:,None,:,None]
    projected=torch.where(fallback[:,None,:,None],p[:,None,:,None]/4,projected)
    exact=projected.reshape(-1)
    relaxed=(1-rho)*q.reshape(-1)+rho*exact
    error=(group_sum(relaxed,coarse)-p).abs()
    # The projection must preserve within-parent proposal ratios; compare
    # the first/last child in each positive group in log space.
    first=q[:,0,:,0];last=q[:,1,:,1]
    zfirst=projected[:,0,:,0];zlast=projected[:,1,:,1]
    valid=(first>eps)&(last>eps)&(zfirst>eps)&(zlast>eps)
    ratio_error=(torch.log(zfirst[valid]/zlast[valid])-torch.log(first[valid]/last[valid])).abs()
    if not ratio_error.numel():ratio_error=torch.zeros(1,device=q.device)
    def ent(x):
        cond=x/x.sum((1,3),keepdim=True).clamp_min(eps)
        return float(-(cond*cond.clamp_min(eps).log()).sum((1,3)).mean())
    # The upsampled prior is the distribution before the fine SA proposal.
    prior=torch.full_like(q,.25)*p[:,None,:,None]
    stats={"error":(float(error.mean()),float(error.max()),float(torch.quantile(error.reshape(-1),.95)),int(fallback.sum())),
           "ratio":(float(ratio_error.mean()),float(ratio_error.max()),float(torch.quantile(ratio_error,.95))),
           "entropy":(ent(prior),ent(q),ent(projected))}
    return relaxed,stats


def completion(core:torch.Tensor,graph:torch.Tensor,background:torch.Tensor,mode:str,eps=1e-6):
    """A bounded *positive* residual; output is pre-final-normalization."""
    p1=score(core)
    p2=score(graph@p1)
    delta=(p2-p1).clamp_min(0)
    if mode=="plain":return p2,p1,p2,delta,torch.ones_like(delta)
    if mode=="background":weight=1-background
    elif mode=="odds":weight=(p1+eps)/(p1+background+2*eps)
    else:raise ValueError(mode)
    weight=weight.clamp(0,1)
    y=p1+weight*delta
    return y,p1,p2,delta,weight


@torch.inference_mode()
def aggregate(sa_qk:dict[int,list],seed:torch.Tensor):
    if seed.shape!=(64,64) or any(len(sa_qk.get(s,()))!=11 for s in (16,32,64)):
        raise ValueError("SG-combined-v1 needs 11 matched SA calls at 16/32/64")
    c64=seed.float().reshape(-1)
    c16=_resize(c64,64,16).reshape(-1)
    sums={name:torch.zeros(4096,device=seed.device) for name in NAMES}
    stages=torch.zeros((3,3,64,64),device=seed.device) # C1/C4/C5 x 16/32/64
    transport_stats=torch.zeros((4,2,4),dtype=torch.float64) # C2-C5 x 2 x error/fallback
    ratio_stats=torch.zeros((2,3),dtype=torch.float64)
    entropy_stats=torch.zeros((2,3),dtype=torch.float64)
    first=torch.zeros((5,64,64),device=seed.device) # core, P2, delta, B, G
    beta=torch.zeros((3,),dtype=torch.float64)
    fallback_bg=0;bg_fraction=0.
    for call in range(11):
        a16=_attention(sa_qk[16][call],16).mean(0)
        a32=_attention(sa_qk[32][call],32).mean(0)
        fine_heads=_attention(sa_qk[64][call],64)
        a64=fine_heads.mean(0)
        u=_apply(fine_heads,c64)
        sums["C0"]+=_apply(fine_heads,u).mean(0)/11
        fine_first=score(u.mean(0))
        background,fb,fraction=background_support(fine_heads,fine_first)
        fallback_bg+=int(fb);bg_fraction+=fraction/11
        x16=distribution(a16@c16)
        # All proposals use exactly the established SG25 resize/alignment.
        q32=distribution(a32@_resize(x16,16,32).reshape(-1))
        p32,st32=project(x16,q32,16,rho=1.)
        r32,st32r=project(x16,q32,16,rho=.5)
        q64_ordinary=distribution(a64@_resize(q32,32,64).reshape(-1))
        q64_from_projected=distribution(a64@_resize(p32,32,64).reshape(-1))
        q64_from_relaxed=distribution(a64@_resize(r32,32,64).reshape(-1))
        c3,st3=project(q32,q64_ordinary,32,rho=1.)
        c4,st4=project(p32,q64_from_projected,32,rho=1.)
        c5,st5=project(r32,q64_from_relaxed,32,rho=.5)
        cores={"C1":q64_ordinary,"C2":q64_from_projected,
               "C3":c3,"C4":c4,"C5":c5}
        for name,value in cores.items():sums[name]+=value/11
        # C2/C3/C4/C5 have one or two constraints.  Unused transitions
        # remain NaN in final diagnostics, never interpreted as zero error.
        for ci,transition,st in ((0,0,st32),(1,1,st3),(2,0,st32),
                                 (2,1,st4),(3,0,st32r),(3,1,st5)):
            transport_stats[ci,transition]+=torch.tensor(st["error"],dtype=torch.float64)/11
        ratio_stats[0]+=torch.tensor(st32["ratio"],dtype=torch.float64)/11
        ratio_stats[1]+=torch.tensor(st4["ratio"],dtype=torch.float64)/11
        entropy_stats[0]+=torch.tensor(st32["entropy"],dtype=torch.float64)/11
        entropy_stats[1]+=torch.tensor(st4["entropy"],dtype=torch.float64)/11
        for si,values in enumerate(((x16,q32,q64_ordinary),(x16,p32,c4),(x16,r32,c5))):
            for j,(value,side) in enumerate(zip(values,(16,32,64))):
                stages[si,j]+=_resize(value,side,64).reshape(64,64)/11
        # Completion is computed from each same-call semantic core, not from
        # an across-call mean graph.  Background follows SG32 exactly: rows
        # selected from first fine A@c, never from the transported core.
        plain,p1,p2,delta,_=completion(c4,a64,background,"plain")
        y7,_,_,_,g7=completion(c4,a64,background,"background")
        y8,_,_,_,g8=completion(c4,a64,background,"odds")
        y9,p1r,p2r,deltar,g9=completion(c5,a64,background,"odds")
        for name,value in (("C6",plain),("C7",y7),("C8",y8),("C9",y9)):
            sums[name]+=value/11
        for i,(d,g) in enumerate(((delta,g7),(delta,g8),(deltar,g9))):
            beta[i]+=float((d*g).sum()/d.sum().clamp_min(EPS))/11
        first+=torch.stack((p1,p2,delta,background,g8)).reshape(5,64,64)/11
        del a16,a32,a64,fine_heads,u
    maps={name:value.reshape(64,64) for name,value in sums.items()}
    transport_stats[0,1]=float("nan") # C2 has no 32→64 constraint
    transport_stats[1,0]=float("nan") # C3 has no 16→32 constraint
    transient={"stages":stages,"transport":transport_stats.numpy(),
               "ratio":ratio_stats.numpy(),"entropy":entropy_stats.numpy(),
               "completion_maps":first.cpu(),"beta":beta.numpy(),
               "background_fallbacks":fallback_bg,"background_seed_fraction":bg_fraction}
    return maps,transient
