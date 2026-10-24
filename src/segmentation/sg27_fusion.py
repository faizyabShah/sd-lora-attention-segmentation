"""Parameter-free late fusion of the three locked SG25 propagation maps."""
from __future__ import annotations

import torch

from src.segmentation import controlled_cv as cv
from src.segmentation.sg25_sa import _apply, _attention, _resize

NAMES = tuple(f"S{i}" for i in range(46))
EPS = 1e-8


@torch.inference_mode()
def base_paths(sa_qk: dict[int,list[tuple[torch.Tensor,torch.Tensor]]], seed: torch.Tensor) -> dict[str,torch.Tensor]:
    """Recompute SG25 FINE-P2, 16→64 and 32→64 without other SG25 cells."""
    if seed.shape!=(64,64) or any(len(sa_qk.get(side,[]))!=11 for side in (16,32,64)):
        raise ValueError("SG27 requires one A3 seed and 11 SA calls at each scale")
    c64=seed.float().reshape(-1)
    c16=_resize(c64,64,16).reshape(-1)
    c32=_resize(c64,64,32).reshape(-1)
    sums=torch.zeros((3,64,64),device=seed.device,dtype=torch.float32)
    for call in range(11):
        a16=_attention(sa_qk[16][call],16)
        a32=_attention(sa_qk[32][call],32)
        a64=_attention(sa_qk[64][call],64)
        fine_once=_apply(a64,c64)
        coarse16_once=_apply(a16,c16).mean(0)
        coarse32_once=_apply(a32,c32).mean(0)
        sums[0]+=_apply(a64,fine_once).sum(0).reshape(64,64)
        sums[1]+=_apply(a64,_resize(coarse16_once,16,64).reshape(-1)).sum(0).reshape(64,64)
        sums[2]+=_apply(a64,_resize(coarse32_once,32,64).reshape(-1)).sum(0).reshape(64,64)
        del a16,a32,a64,fine_once,coarse16_once,coarse32_once
    return {name:sums[i]/88 for i,name in enumerate(("FINE","C16","C32"))}


def _pair_formulas(a:torch.Tensor,b:torch.Tensor) -> tuple[torch.Tensor,...]:
    """Arithmetic, geometric, harmonic, RMS, max, min, product, OR, agreement."""
    arithmetic=(a+b)/2
    maximum=torch.maximum(a,b)
    minimum=torch.minimum(a,b)
    agreement=1-(a-b).abs()
    return (arithmetic,torch.sqrt(a*b),2*a*b/(a+b+EPS),
            torch.sqrt((a.square()+b.square())/2),maximum,minimum,a*b,
            a+b-a*b,a*agreement,arithmetic*agreement)


@torch.inference_mode()
def proposed_maps(base:dict[str,torch.Tensor]) -> dict[str,torch.Tensor]:
    """All 46 proposed formulas, including aliases for the smoke duplicate scan.

    The same final minmax operation is applied to every formula. Aliases
    use the exact same tensor expression so algebraic duplicates are exact.
    """
    if set(base)!={"FINE","C16","C32"}:
        raise ValueError("SG27 base map set changed")
    normalized=cv.normalize_batch(torch.stack([base[n] for n in ("FINE","C16","C32")]),"minmax")
    f,c16,c32=normalized
    p16=_pair_formulas(f,c16)
    p32=_pair_formulas(f,c32)
    # Pair tuple: mean, geom, harmonic, RMS, max, min, product, OR,
    # agreement-weighted fine, agreement-weighted mean.
    raw={"S0":f,"S1":c16,"S2":c32,
         "S3":p16[0],"S4":p16[1],"S5":p16[2],"S6":p16[3],
         "S7":p16[4],"S8":p16[5],"S9":p16[6],"S10":p16[7],
         "S11":p16[4],"S12":p16[5],"S13":p16[8],"S14":p16[9],"S15":p16[4],
         "S16":p32[0],"S17":p32[1],"S18":p32[2],"S19":p32[3],
         "S20":p32[4],"S21":p32[5],"S22":p32[6],"S23":p32[7],
         "S24":p32[4],"S25":p32[8],"S26":p32[9]}
    coarse=_pair_formulas(c16,c32)
    raw.update({"S27":coarse[0],"S28":coarse[1],"S29":coarse[2],
                "S30":coarse[4],"S31":coarse[5],"S32":coarse[7]})
    values=torch.stack((f,c16,c32))
    ordered=values.sort(dim=0).values
    raw.update({"S33":values.mean(0),"S34":(f*c16*c32).clamp_min(0).pow(1/3),
                "S35":3/(1/(f+EPS)+1/(c16+EPS)+1/(c32+EPS)),
                "S36":values.square().mean(0).sqrt(),"S37":values.max(0).values,
                "S38":values.min(0).values,
                "S39":1-(1-f)*(1-c16)*(1-c32),
                "S40":ordered[1],"S41":(ordered[1]+ordered[2])/2,
                "S42":(ordered[0]+ordered[1])/2})
    coarse_average=(c16+c32)/2
    agree16=1-(f-c16).abs()
    agree32=1-(f-c32).abs()
    raw.update({"S43":f*(1-(f-coarse_average).abs()),
                "S44":f*torch.sqrt(agree16*agree32),
                "S45":torch.maximum(f,coarse_average)})
    if set(raw)!=set(NAMES):
        raise ValueError("SG27 proposed formula coverage changed")
    final=cv.normalize_batch(torch.stack([raw[name] for name in NAMES]),"minmax")
    return {name:final[i] for i,name in enumerate(NAMES)}


def duplicate_groups(max_abs_differences, threshold:float=1e-8) -> tuple[list[str],dict[str,str],list[list[str]]]:
    """Freeze the earliest-name representative of every smoke-equivalent group."""
    if tuple(max_abs_differences.shape)!=(len(NAMES),len(NAMES)):
        raise ValueError("SG27 duplicate matrix has unexpected shape")
    groups=[]
    aliases={}
    for i,name in enumerate(NAMES):
        representative=None
        for group in groups:
            k=NAMES.index(group[0])
            if float(max_abs_differences[i,k])<threshold:
                representative=group[0]
                group.append(name)
                break
        if representative is None:
            groups.append([name])
        else:
            aliases[name]=representative
    return [group[0] for group in groups],aliases,[group for group in groups if len(group)>1]
