"""SG32: reliability-conditioned second SA pass; first pass remains A @ c."""
from __future__ import annotations

import torch

from src.segmentation import controlled_cv as cv
from src.segmentation.sg25_sa import _apply, _attention, _resize

NAMES = tuple(f"S{i}" for i in range(12))
SIGNALS = ("c16", "pair", "range", "bg", "pair_bg", "range_bg")
ADAPTIVE = {"S2": ("c16", "convex"), "S3": ("pair", "convex"),
            "S4": ("bg", "convex"), "S5": ("pair_bg", "convex"),
            "S6": ("range_bg", "convex"), "S7": ("c16", "temperature"),
            "S8": ("pair", "temperature"), "S9": ("bg", "temperature"),
            "S10": ("pair_bg", "temperature"), "S11": ("range_bg", "temperature")}


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return cv.normalize_batch(vector.reshape(1, 64, 64), "minmax")[0].reshape(-1)


def background_support(graph: torch.Tensor, fine: torch.Tensor):
    """SeeDiff selected-row background expansion from first-pass fine map."""
    selected = fine <= .5
    fallback = not bool(selected.any())
    if fallback:
        selected[int(fine.argmin())] = True
    # Mean over heads and selected query rows; destination dimension remains.
    value = graph[:, selected, :].mean((0, 1))
    return normalize(value), fallback, float(selected.float().mean())


def reliability(fine, mid, coarse, background):
    pair = 1 - ((fine-mid).abs() + (fine-coarse).abs() + (mid-coarse).abs()) / 3
    range_ = 1 - (torch.stack((fine,mid,coarse)).amax(0)
                  - torch.stack((fine,mid,coarse)).amin(0))
    c16 = 1 - (fine-coarse).abs()
    bg = 1 - background
    values = {"c16":c16,"pair":pair,"range":range_,"bg":bg}
    values["pair_bg"] = pair*bg
    values["range_bg"] = range_*bg
    return {name:value.clamp(0,1) for name,value in values.items()}


def row_temperature(graph: torch.Tensor, exponent: torch.Tensor):
    """Row-wise power, then row normalization; never average heads first."""
    if exponent.shape != (graph.shape[1],):
        raise ValueError("SG32 exponent must be one value per query row")
    powered = graph.pow(exponent.reshape(1,-1,1))
    return powered / powered.sum(-1,keepdim=True).clamp_min(torch.finfo(powered.dtype).tiny)


@torch.inference_mode()
def aggregate(sa_qk: dict[int,list], seed: torch.Tensor, r_max: float = 2.6):
    """Return twelve 64² maps plus small transient first-pass/reliability maps.

    Transient maps are for same-image diagnostics only; they are not saved.
    """
    if seed.shape != (64,64) or any(len(sa_qk.get(s,())) != 11 for s in (16,32,64)):
        raise ValueError("SG32 requires one A3 seed and 11 matched calls at each SA scale")
    if r_max != 2.6:
        raise ValueError("SG32 locked r_max changed")
    c64 = seed.float().reshape(-1)
    c16 = _resize(c64,64,16).reshape(-1)
    c32 = _resize(c64,64,32).reshape(-1)
    sums = {name:torch.zeros_like(c64) for name in NAMES}
    rel_calls=[];first_calls=[];p2_calls=[];entropy_calls=[];bg_log=[]
    sample_rows=torch.linspace(0,4095,64,device=c64.device).long()
    for call in range(11):
        a16=_attention(sa_qk[16][call],16)
        a32=_attention(sa_qk[32][call],32)
        a=_attention(sa_qk[64][call],64)
        u=_apply(a,c64)
        ordinary=_apply(a,u)
        sharp=row_temperature(a,torch.full((4096,),r_max,device=c64.device))
        sharpened=_apply(sharp,u)
        sums["S0"] += ordinary.mean(0)/11
        sums["S1"] += sharpened.mean(0)/11
        fine=normalize(u.mean(0))
        mid=normalize(_resize(_apply(a32,c32).mean(0),32,64).reshape(-1))
        coarse=normalize(_resize(_apply(a16,c16).mean(0),16,64).reshape(-1))
        bg,fallback,bg_fraction=background_support(a,fine)
        rel=reliability(fine,mid,coarse,bg)
        bg_log.append((int(fallback),bg_fraction))
        sampled=a[:,sample_rows,:]
        ent0=-(sampled*sampled.clamp_min(1e-30).log()).sum(-1)
        ent_stat=[]
        for name,(signal,form) in ADAPTIVE.items():
            r=rel[signal]
            if form=="convex":
                value=r[None]*ordinary+(1-r[None])*sharpened
            else:
                exponent=1+(r_max-1)*(1-r)
                adjusted=row_temperature(a,exponent)
                value=_apply(adjusted,u)
                sampled_adjusted=adjusted[:,sample_rows,:]
                ent1=-(sampled_adjusted*sampled_adjusted.clamp_min(1e-30).log()).sum(-1)
                loss=(ent0-ent1).mean(0)
                strength=(1-r[sample_rows])
                centered_x=strength-strength.mean()
                centered_y=loss-loss.mean()
                corr=(centered_x*centered_y).mean()/(
                    centered_x.square().mean().sqrt()*centered_y.square().mean().sqrt()).clamp_min(1e-8)
                ent_stat.append((float(ent0.mean()),float(ent1.mean()),
                                 float((ent1-ent0).mean()),float(corr)))
                del adjusted,sampled_adjusted
            sums[name] += value.mean(0)/11
        rel_calls.append(torch.stack([rel[n].float().cpu() for n in SIGNALS]))
        first_calls.append(fine.float().cpu())
        p2_calls.append(normalize(ordinary.mean(0)).float().cpu())
        entropy_calls.append(ent_stat)
        del a,a16,a32,u,ordinary,sharp,sharpened,sampled
    maps={name:value.reshape(64,64) for name,value in sums.items()}
    transient={"reliability":torch.stack(rel_calls),"first":torch.stack(first_calls),
               "p2":torch.stack(p2_calls),"entropy":torch.tensor(entropy_calls),
               "background_log":torch.tensor(bg_log)}
    return maps,transient
