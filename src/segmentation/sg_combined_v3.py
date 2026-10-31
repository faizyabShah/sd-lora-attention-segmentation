"""SG-combined-v3: three single changes to the SG-combined-v2 V3 capped hierarchy.

All cells share one historical A3 seed and the v2 one-sided cap
(``sg_combined_v2.cap``). With r16_t = mean_h A16_{t,h} c16, the reference
R1 is v2 V3: z32_t = cap(r16_t, mean_h A32_{t,h} U r16_t) and
R1 = mean_t mean_h A64_{t,h} U z32_t. Each W cell changes exactly one thing:

* W1 caps every 32-px head against the head-mean parent before averaging,
  z32_t = mean_h cap(r16_t, A32_{t,h} U r16_t). By concavity of min(., p),
  W1 removes at least as much mass as R1; no head pairing across modules.
* W2 starts the hierarchy from c16*, the native 8/16-px share of the A3 sum
  (same weights, S = S_64 + S_<=16), instead of the downsampled 64-px seed.
* W3 adds a temporal cap: call t > 10 is capped by the area-pooled running
  mean of R1's earlier per-call maps, z_t = cap(D(mean_{t'<t} v_t'), v_t).

GT enters only the diagnostic cap-loss statistics, never a map.
"""
from __future__ import annotations

import math

import torch

from src.segmentation.sg25_sa import _apply, _attention, _resize
from src.segmentation.sg_combined_v2 import EPS, cap

NAMES=("R0","R1","W1","W2","W3")
CAP_ROWS=("R1_16_to_32","W1_16_to_32_headwise","W2_16_to_32","W3_temporal_32_to_64")
GT_ROWS=("R1","W1","W2")


def gt_cap_loss(parent:torch.Tensor,proposal:torch.Tensor,capped:torch.Tensor,coarse:int,gt:torch.Tensor):
    """(gamma_GT, L_GT, L_background) for one cap at ``2*coarse`` resolution.

    ``gt`` is the GT foreground fraction area-pooled to (2*coarse)^2.
    gamma_GT is the GT-weighted share of children whose 2x2 group binds;
    L_GT / L_background are the shares of GT / background proposal mass removed.
    """
    fine=2*coarse
    q=proposal.float().clamp_min(0)
    binding=(q.reshape(coarse,2,coarse,2).mean((1,3))>parent.float().clamp_min(0).reshape(coarse,coarse))
    child=binding[:,None,:,None].expand(coarse,2,coarse,2).reshape(-1).float()
    g=gt.float().reshape(-1);b=1-g;removed=(q-capped).clamp_min(0)
    if g.numel()!=fine*fine:raise ValueError("GT cap-loss resolution mismatch")
    def share(num,den):return float(num/den) if float(den)>0 else float("nan")
    return (share((g*child).sum(),g.sum()),share((g*removed).sum(),(g*q).sum()),
            share((b*removed).sum(),(b*q).sum()))


@torch.inference_mode()
def aggregate(sa_qk:dict[int,list],seed:torch.Tensor,coarse_seed:torch.Tensor,*,
              gt32:torch.Tensor|None=None,cap_scale:float=1.):
    if seed.shape!=(64,64) or coarse_seed.shape!=(16,16) or any(len(sa_qk.get(s,()))!=11 for s in (16,32,64)):
        raise ValueError("SG-combined-v3 needs the A3 seed, its 16x16 share and 11 SA calls at 16/32/64")
    if cap_scale not in (1.,math.inf):
        raise ValueError("SG-combined-v3 cap scale is locked (inf only for identity checks)")
    c64=seed.float().reshape(-1)
    c16=_resize(c64,64,16).reshape(-1)
    c16w=coarse_seed.float().reshape(-1).to(c64.device)
    sums={name:torch.zeros(4096,device=seed.device) for name in NAMES}
    cap_stats=torch.zeros((len(CAP_ROWS),3),dtype=torch.float64)
    gt_stats=torch.zeros((len(GT_ROWS),3),dtype=torch.float64)
    history=torch.zeros(4096,device=seed.device)
    for call in range(11):
        heads16=_attention(sa_qk[16][call],16)
        heads32=_attention(sa_qk[32][call],32)
        fine_heads=_attention(sa_qk[64][call],64)
        sums["R0"]+=_apply(fine_heads,_apply(fine_heads,c64)).mean(0)/11
        # R1 (v2 V3) and W1 share the head-mean parent r16 and 32-px proposals.
        r16=_apply(heads16,c16).mean(0)
        q32h=_apply(heads32,_resize(r16,16,32).reshape(-1))
        z32,st=cap(r16,q32h.mean(0),16,scale=cap_scale)
        cap_stats[0]+=torch.tensor(st,dtype=torch.float64)/11
        head_caps=[cap(r16,q32h[h],16,scale=cap_scale) for h in range(q32h.shape[0])]
        z32_w1=torch.stack([z for z,_ in head_caps]).mean(0)
        cap_stats[1]+=torch.tensor([sum(s[i] for _,s in head_caps)/len(head_caps) for i in range(3)],
                                   dtype=torch.float64)/11
        v=_apply(fine_heads,_resize(z32,32,64).reshape(-1)).mean(0)
        sums["R1"]+=v/11
        sums["W1"]+=_apply(fine_heads,_resize(z32_w1,32,64).reshape(-1)).mean(0)/11
        # W2: the same chain from the native coarse seed.
        r16w=_apply(heads16,c16w).mean(0)
        q32w=_apply(heads32,_resize(r16w,16,32).reshape(-1)).mean(0)
        z32w,stw=cap(r16w,q32w,16,scale=cap_scale)
        cap_stats[2]+=torch.tensor(stw,dtype=torch.float64)/11
        sums["W2"]+=_apply(fine_heads,_resize(z32w,32,64).reshape(-1)).mean(0)/11
        # W3: temporal cap by the running mean of earlier R1 calls (call 10 free).
        if call==0:
            sums["W3"]+=v/11
        else:
            parent=_resize(history/call,64,32).reshape(-1)
            z3,st3=cap(parent,v,32,scale=cap_scale)
            sums["W3"]+=z3/11
            cap_stats[3]+=torch.tensor(st3,dtype=torch.float64)/10
        history+=v
        if gt32 is not None:
            g=gt32.to(c64.device)
            gt_stats[0]+=torch.tensor(gt_cap_loss(r16,q32h.mean(0),z32,16,g),dtype=torch.float64)/11
            per_head=torch.tensor([gt_cap_loss(r16,q32h[h],head_caps[h][0],16,g)
                                   for h in range(q32h.shape[0])],dtype=torch.float64)
            gt_stats[1]+=per_head.nanmean(0)/11
            gt_stats[2]+=torch.tensor(gt_cap_loss(r16w,q32w,z32w,16,g),dtype=torch.float64)/11
        del heads16,heads32,fine_heads,q32h
    maps={name:value.reshape(64,64) for name,value in sums.items()}
    a,b=c16w,c16
    seed_stats=(float(torch.corrcoef(torch.stack((a,b)))[0,1]),float((a-b).abs().mean()))
    return maps,{"cap":cap_stats.numpy(),"gt_cap":gt_stats.numpy(),"seed":seed_stats}
