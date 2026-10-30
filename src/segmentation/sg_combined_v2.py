"""SG-combined-v2: one-sided coarse caps on the raw 16→32→64 hierarchy.

The cap is applied in raw (non-normalized) measure space, on 2x2 group means:

    z_j = q_j * min(1, p_g / (mean_{k in g} q_k + eps)).

It is the generalized-KL projection onto groupwise upper-mass constraints,
``sum_{k in g} z_k <= 4 p_g`` (bilinear upsampling preserves local means, so
4 p_g is the parent's mass at the child resolution). Clipped mass is
discarded, not redistributed; this is not a projection on the probability
simplex. With no binding group the capped hierarchy equals the ordinary raw
hierarchy V1 exactly, so V3−V1 isolates the cap. No call is renormalized,
so relative call weights match V1 (unlike SG-combined-v1 C2–C5).
"""
from __future__ import annotations

import math

import torch

from src.segmentation.sg25_sa import _apply, _attention, _resize
from src.segmentation.sg32_adaptive import background_support
from src.segmentation.sg_combined_v1 import completion, distribution, project, score

NAMES=tuple(f"V{i}" for i in range(7))
EPS=1e-12
ROUTE_PAIRS=(("V0","V2"),("V0","V1"),("V0","V3"),("V0","V5"))  # first pair is primary


def cap(parent:torch.Tensor,proposal:torch.Tensor,coarse:int,*,scale:float=1.,eps:float=EPS):
    """One-sided generalized-KL cap of a fine proposal by its coarse parent.

    Returns the capped fine map and (binding-group fraction, removed-mass
    fraction, mean factor over binding groups). ``scale=inf`` disables the
    cap and exists only for the V3=V1 identity check.
    """
    if parent.numel()!=coarse*coarse or proposal.numel()!=(2*coarse)**2:
        raise ValueError("SG-combined-v2 cap parent/proposal mismatch")
    p=parent.float().clamp_min(0).reshape(coarse,coarse)*scale
    q=proposal.float().clamp_min(0)
    m=q.reshape(coarse,2,coarse,2).mean((1,3))
    factor=torch.where(m>p,p/(m+eps),torch.ones_like(m)) if math.isfinite(scale) else torch.ones_like(m)
    z=(q.reshape(coarse,2,coarse,2)*factor[:,None,:,None]).reshape(-1)
    binding=factor<1
    total=float(q.sum())
    stats=(float(binding.float().mean()),
           float((q.sum()-z.sum())/total) if total>0 else 0.,
           float(factor[binding].mean()) if bool(binding.any()) else 1.)
    return z,stats


def bayes_completion(core:torch.Tensor,graph:torch.Tensor,background:torch.Tensor):
    """Admit the whole positive residual where P1 >= B (G >= 1/2), else none."""
    p1=score(core)
    p2=score(graph@p1)
    delta=(p2-p1).clamp_min(0)
    return p1+(p1>=background).float()*delta


@torch.inference_mode()
def aggregate(sa_qk:dict[int,list],seed:torch.Tensor,*,cap_scale:float=1.):
    if seed.shape!=(64,64) or any(len(sa_qk.get(s,()))!=11 for s in (16,32,64)):
        raise ValueError("SG-combined-v2 needs the A3 seed and 11 matched SA calls at 16/32/64")
    if cap_scale not in (1.,math.inf):
        raise ValueError("SG-combined-v2 cap scale is locked (inf only for the identity check)")
    c64=seed.float().reshape(-1)
    c16=_resize(c64,64,16).reshape(-1)
    sums={name:torch.zeros(4096,device=seed.device) for name in NAMES}
    cap_stats=torch.zeros((2,3),dtype=torch.float64)  # 16→32 (V3/V4), 32→64 (V4)
    background_mean=torch.zeros(4096,device=seed.device)
    fallback_bg=0;bg_fraction=0.
    for call in range(11):
        heads16=_attention(sa_qk[16][call],16)
        heads32=_attention(sa_qk[32][call],32)
        fine_heads=_attention(sa_qk[64][call],64)
        a32=heads32.mean(0);a64=fine_heads.mean(0)
        u=_apply(fine_heads,c64)
        sums["V0"]+=_apply(fine_heads,u).mean(0)/11
        background,fb,fraction=background_support(fine_heads,score(u.mean(0)))
        background_mean+=background/11;fallback_bg+=int(fb);bg_fraction+=fraction/11
        # V1 is the raw SG31 S2 / SG-combined-v1 C1 hierarchy; per-call raw
        # maps are averaged over calls before any normalization.
        raw16=_apply(heads16,c16).mean(0)
        raw32=_apply(heads32,_resize(raw16,16,32).reshape(-1)).mean(0)
        raw64=_apply(fine_heads,_resize(raw32,32,64).reshape(-1)).mean(0)
        sums["V1"]+=raw64/11
        # V3: cap at 16→32 only; V4: cap at 16→32 and 32→64.
        z32,st32=cap(raw16,raw32,16,scale=cap_scale)
        v3=_apply(fine_heads,_resize(z32,32,64).reshape(-1)).mean(0)
        z64,st64=cap(z32,v3,32,scale=cap_scale)
        sums["V3"]+=v3/11;sums["V4"]+=z64/11
        cap_stats[0]+=torch.tensor(st32,dtype=torch.float64)/11
        cap_stats[1]+=torch.tensor(st64,dtype=torch.float64)/11
        # V2 replays SG-combined-v1 C6 exactly: exact KL core at both
        # transitions in transport space, then unconditional completion.
        x16=distribution(raw16)
        q32=distribution(a32@_resize(x16,16,32).reshape(-1))
        p32,_=project(x16,q32,16,rho=1.)
        q64=distribution(a64@_resize(p32,32,64).reshape(-1))
        c4,_=project(p32,q64,32,rho=1.)
        sums["V2"]+=completion(c4,a64,background,"plain")[0]/11
        # V5/V6: the same completion conventions on the V3 core.
        sums["V5"]+=completion(v3,a64,background,"plain")[0]/11
        sums["V6"]+=bayes_completion(v3,a64,background)/11
        del heads16,heads32,fine_heads,a32,a64,u
    maps={name:value.reshape(64,64) for name,value in sums.items()}
    transient={"cap":cap_stats.numpy(),"background":background_mean.reshape(64,64),
               "background_fallbacks":fallback_bg,"background_seed_fraction":bg_fraction}
    return maps,transient


def route_scores(fine:torch.Tensor,hier:torch.Tensor,background:torch.Tensor,eps:float=1e-8):
    """R (background-weighted fine-exclusive support), E and D on [0,1] maps."""
    fine,hier,background=(x.float().cpu().reshape(-1) for x in (fine,hier,background))
    extra=(fine-hier).clamp_min(0)
    return (float((background*extra).sum()/(extra.sum()+eps)),
            float(extra.mean()),float((fine-hier).abs().mean()))
