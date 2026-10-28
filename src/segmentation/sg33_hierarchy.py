"""SG33 sequential residual hierarchy and class-seeded coarse Markov maps."""
from __future__ import annotations

import torch

from src.segmentation import controlled_cv as cv
from src.segmentation.sg25_sa import _apply, _attention, _resize

NAMES=("S0","H0","H1","H2","H3","H4","M1","M2","M3","C1","C2","C3")
MARKOV=("M1","M2","M3")
IPF_FIELDS=("before_row_error","before_col_error","before_col_variance",
            "after_row_error","after_col_error","after_col_variance",
            "iterations","converged","stationary_entropy_before",
            "stationary_entropy_after")
TEMP_FIELDS=("row_entropy_before","row_entropy_after","effective_neighbors_before",
             "effective_neighbors_after","top10_mass_before","top10_mass_after")


def norm(x:torch.Tensor)->torch.Tensor:
    return cv.normalize_batch(x.reshape(1,*x.shape[-2:]),"minmax")[0].reshape(x.shape)


def _errors(graph):
    return (float((graph.sum(-1)-1).abs().max()),
            float((graph.sum(0)-1).abs().max()),
            float(graph.sum(0).var(unbiased=False)))


def ipf(graph:torch.Tensor,tolerance=1e-5,max_iterations=100):
    """Alternating row/column scaling; return a Markov row-stochastic matrix."""
    graph=graph.float().clamp_min(1e-30)
    before=_errors(graph)
    scaled=graph.clone()
    converged=False
    for iteration in range(1,max_iterations+1):
        scaled=scaled/scaled.sum(-1,keepdim=True).clamp_min(1e-30)
        scaled=scaled/scaled.sum(0,keepdim=True).clamp_min(1e-30)
        row_error,col_error,_=_errors(scaled)
        if max(row_error,col_error)<=tolerance:
            converged=True
            break
    # Column-last IPF can leave a tiny row error. The final transition must
    # be row-stochastic for p_{t+1}=p_t P; log both post-IPF errors.
    scaled=scaled/scaled.sum(-1,keepdim=True).clamp_min(1e-30)
    after=_errors(scaled)
    return scaled,before,after,iteration,converged and max(after[:2])<=tolerance


def stationary_entropy(graph,steps=256):
    p=torch.full((graph.shape[0],),1/graph.shape[0],device=graph.device)
    for _ in range(steps):p=p@graph
    return float(-(p*p.clamp_min(1e-30).log()).sum())


def temperature(graph,value=.65):
    transformed=graph.float().clamp_min(1e-30).pow(1/value)
    return transformed/transformed.sum(-1,keepdim=True).clamp_min(1e-30)


def temperature_diagnostics(before,after):
    def stats(a):
        ent=-(a*a.clamp_min(1e-30).log()).sum(-1)
        return (float(ent.mean()),float(ent.exp().mean()),
                float(a.topk(10,dim=-1).values.sum(-1).mean()))
    x=stats(before);y=stats(after)
    return (x[0],y[0],x[1],y[1],x[2],y[2])


def markov(graph,p0,tau=.3,cap=256):
    """Integer first relative-saturation step, t=0 included.

    Unreached nodes receive the locked cap. GT never enters this operator.
    """
    if graph.shape!=(256,256) or p0.shape!=(256,):
        raise ValueError("SG33 Markov graph/seed must have 256 nodes")
    p=p0.float().clamp_min(0)
    if not bool(p.sum()>0):p=torch.full_like(p,1/256)
    else:p=p/p.sum()
    arrival=torch.full((256,),cap,dtype=torch.int16,device=graph.device)
    seen=torch.zeros(256,dtype=torch.bool,device=graph.device)
    for t in range(cap+1):
        reached=(p/p.max().clamp_min(1e-30)>tau)&~seen
        arrival[reached]=t
        seen|=reached
        if bool(seen.all()):break
        if t<cap:p=p@graph
    m=arrival.float()
    support=1-(m-m.min())/(m.max()-m.min()+1e-8)
    return support.reshape(16,16),arrival.reshape(16,16),~seen.reshape(16,16)


def propagate(mid:torch.Tensor,fine:torch.Tensor,start:torch.Tensor):
    """Sequential head-mean 16→32→64; start is the 16² semantic state."""
    u32=_apply(mid,_resize(start.reshape(-1),16,32).reshape(-1)).mean(0).reshape(32,32)
    u64=_apply(fine,_resize(u32.reshape(-1),32,64).reshape(-1)).mean(0).reshape(64,64)
    return u32,u64


def residual(states:tuple[torch.Tensor,torch.Tensor,torch.Tensor],cell:str):
    r16,r32,r64=(norm(x) for x in states)
    if cell in ("H1","C1","C2","C3"):
        return (r16+r32+r64)/3
    if cell=="H2":return .2*r16+.3*r32+.5*r64
    if cell=="H3":return r16/9+5*r32/9+3*r64/9
    if cell=="H4":return torch.maximum(r64,torch.maximum(.5*r32,.25*r16))
    raise ValueError(cell)


@torch.inference_mode()
def aggregate(sa_qk:dict[int,list],seed:torch.Tensor,*,tau=.3,temperature_value=.65,cap=256):
    if seed.shape!=(64,64) or any(len(sa_qk.get(s,()))!=11 for s in (16,32,64)):
        raise ValueError("SG33 requires A3 seed and 11 matched calls at 16/32/64")
    if (tau,temperature_value,cap)!=(.3,.65,256):
        raise ValueError("SG33 locked Markov settings changed")
    c64=seed.float().reshape(-1)
    c16=_resize(c64,64,16).reshape(-1)
    sums={name:torch.zeros((64,64),device=seed.device) for name in NAMES}
    stages=torch.zeros((3,64,64),device=seed.device)
    markov_logs=[];ipf_logs=[];temperature_logs=[];coarse_logs=[]
    visual=None
    for call in range(11):
        a16=_attention(sa_qk[16][call],16).mean(0)
        a32=_attention(sa_qk[32][call],32).mean(0)
        fine_heads=_attention(sa_qk[64][call],64)
        a64=fine_heads.mean(0)
        first=_apply(fine_heads,c64)
        sums["S0"]+=_apply(fine_heads,first).mean(0).reshape(64,64)/11
        u16=(a16@c16).reshape(16,16)
        u32,u64=propagate(a32[None],a64[None],u16)
        sums["H0"]+=u64/11
        ordinary_states=(_resize(u16.reshape(-1),16,64).reshape(64,64),
                         _resize(u32.reshape(-1),32,64).reshape(64,64),u64)
        stages+=torch.stack(ordinary_states)/11
        for name in ("H1","H2","H3","H4"):
            sums[name]+=residual(ordinary_states,name)/11
        p0=c16/c16.sum().clamp_min(1e-30)
        tempered=temperature(a16,temperature_value)
        p2,before2,after2,iterations2,converged2=ipf(a16)
        p3,before3,after3,iterations3,converged3=ipf(tempered)
        ipf_logs.append(((*_errors(a16),*_errors(a16),0,1,stationary_entropy(a16),stationary_entropy(a16)),
                         (*before2,*after2,iterations2,int(converged2),stationary_entropy(a16),stationary_entropy(p2)),
                         (*before3,*after3,iterations3,int(converged3),stationary_entropy(tempered),stationary_entropy(p3))))
        temperature_logs.append(temperature_diagnostics(a16,tempered))
        log=[];visual_markov={}
        for name,graph in (("M1",a16),("M2",p2),("M3",p3)):
            support,arrival,unreached=markov(graph,p0,tau,cap)
            # Always compute cap-512 diagnostic when the 256-step result
            # leaves >5% unsaturated; it is never an eligible method.
            if float(unreached.float().mean())>.05:
                _,_,unreached512=markov(graph,p0,tau,512)
                remain512=float(unreached512.float().mean())
            else:remain512=float("nan")
            mid,fine=propagate(a32[None],a64[None],support)
            sums[name]+=fine/11
            states=(_resize(support.reshape(-1),16,64).reshape(64,64),
                    _resize(mid.reshape(-1),32,64).reshape(64,64),fine)
            combo="C"+name[-1]
            sums[combo]+=residual(states,combo)/11
            log.append((arrival.float().cpu(),unreached.cpu(),remain512))
            visual_markov[name]=(support.cpu(),fine.cpu())
        markov_logs.append(log)
        coarse_logs.append((u16.cpu(),*(visual_markov[name][0] for name in MARKOV)))
        if call==0:
            visual={"seed":seed.cpu(),"ordinary16":u16.cpu(),
                    **{f"{name}_16":item[0] for name,item in visual_markov.items()},
                    "M3_final64":visual_markov["M3"][1]}
        del a16,a32,a64,fine_heads,first,p2,p3,tempered
    maps={name:value for name,value in sums.items()}
    transient={"stages":stages,"markov":markov_logs,"coarse":coarse_logs,
               "ipf":torch.tensor(ipf_logs),"temperature":torch.tensor(temperature_logs),
               "visual":visual}
    return maps,transient
