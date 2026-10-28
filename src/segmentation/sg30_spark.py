"""Locked SPARK-inspired graph sharpening for the historical A3/P2 pipeline.

SPARK itself clusters an unlabeled diffusion-feature graph. This experiment
keeps our class-word CA seed and original SD softmax SA as the global graph.
All transforms are observation-wise: never average SA matrices first.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.segmentation.sg25_sa import _apply, _attention, SG25Recorder
from src.segmentation.sg29_nerve import neighbor_index

MAIN = tuple(f"S{i}" for i in range(8))
DIAGNOSTIC = ("S5-I2", "S5-I3")
ALL = MAIN + DIAGNOSTIC
GRAPH_FIELDS = ("row_entropy_mean", "row_entropy_median", "effective_neighbors",
                "top1_mass", "top5_mass", "top10_mass", "sparsity",
                "mean_transition_distance", "dominant_neighbor_changed",
                "pruned_entry_fraction", "pruned_inflated_mass", "zero_row_fallback_fraction")
TRANSITION_FIELDS = ("foreground_to_foreground", "foreground_to_background",
                     "background_to_foreground", "background_to_background")
ITER_FIELDS = ("mean_row_l1_change", "row_entropy", "sparsity",
               "effective_neighbors", "near_deterministic_rows")


class SG30Recorder(SG25Recorder):
    """Capture only the selected SA module's pre-Q/K input feature vectors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sa_features: list[torch.Tensor] = []

    def capture_sa_features(self, module_name: str, hidden: torch.Tensor) -> None:
        step = self.call_counts[module_name]
        if module_name == self.selected_sa and 10 <= step <= 20:
            if hidden.ndim != 3 or hidden.shape[0] < 2 or hidden.shape[1] != 4096:
                raise ValueError("SG30 selected SA input feature shape changed")
            self.sa_features.append(hidden[1].detach().half().clone())

    def clear(self):
        super().clear()
        self.sa_features.clear()


def row_normalize(values: torch.Tensor, fallback: torch.Tensor | None = None):
    """Normalize last dimension; use the pre-pruned row when pruning zeroes it."""
    totals = values.sum(-1, keepdim=True)
    zero = totals <= 0
    if zero.any() and fallback is None:
        raise ValueError("zero row without an explicit pre-pruned fallback")
    normalized = values / totals.clamp_min(torch.finfo(values.dtype).tiny)
    if fallback is not None:
        normalized = torch.where(zero, fallback, normalized)
    return normalized, zero.squeeze(-1)


def inflate(graph: torch.Tensor, power: float) -> torch.Tensor:
    if power <= 1 or torch.any(graph < 0):
        raise ValueError("inflation requires a nonnegative graph and r>1")
    out, _ = row_normalize(graph.pow(power))
    return out


def sharpen(graph: torch.Tensor, power: float, threshold: float,
            *, expanded: torch.Tensor | None = None):
    """Exactly expand² → inflate^r → prune τ → row-normalize, with fallback."""
    squared = torch.bmm(graph, graph) if expanded is None else expanded
    inflated = squared.pow(power)
    unpruned, _ = row_normalize(inflated)
    # τ is applied to the *unnormalized* inflated value per the locked spec.
    keep = inflated >= threshold
    pruned = torch.where(keep, inflated, torch.zeros_like(inflated))
    result, fallback = row_normalize(pruned, unpruned)
    removed_mass = torch.where(keep, torch.zeros_like(inflated), inflated).sum(-1)
    removed_mass = removed_mass / inflated.sum(-1).clamp_min(torch.finfo(inflated.dtype).tiny)
    stats = torch.stack(((~keep).float().mean((-1,-2)), removed_mass.mean(-1),
                         fallback.float().mean(-1)), dim=-1)
    return result, unpruned, stats


def feature_local_graph(features: torch.Tensor, index: torch.Tensor,
                        valid: torch.Tensor) -> torch.Tensor:
    """SPARK-feature local graph: ReLU cosine neighbors, self=1, row normalize.

    The paper specifies cosine plus epsilon but its unreleased code leaves
    signed-value handling unresolved. Clamping negative cosine to zero is the
    explicit nonnegative adaptation; no signed row normalization occurs.
    """
    features = F.normalize(features.float(),dim=-1,eps=1e-8)
    gathered = features[index.reshape(-1)].reshape(features.shape[0],9,features.shape[1])
    weights = (features[:,None,:]*gathered).sum(-1).clamp_min(0)
    weights *= valid
    weights[:,4] = 1.0
    weights, _ = row_normalize(weights)
    dense = torch.zeros((features.shape[0],features.shape[0]),
                        device=features.device,dtype=torch.float32)
    dense.scatter_add_(1,index,weights)
    return dense


def diagnostic_rows(side: int, truth: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    n=side*side
    uniform=torch.linspace(0,n-1,32,device=device).long()
    if truth is None:
        return uniform
    gt=truth.to(device).reshape(-1).bool()
    parts=[uniform]
    for mask in (gt,~gt):
        ids=torch.nonzero(mask,as_tuple=False).reshape(-1)
        if ids.numel():
            parts.append(ids[torch.linspace(0,ids.numel()-1,16,device=device).long()])
    return torch.unique(torch.cat(parts),sorted=True)


def graph_diagnostics(graph: torch.Tensor, reference: torch.Tensor,
                      rows: torch.Tensor, truth: torch.Tensor | None,
                      side: int, prune_stats: torch.Tensor | None = None):
    """Per-head metrics on deterministic sampled source rows; GT is post hoc."""
    sampled = graph[:,rows]
    top = sampled.topk(10,dim=-1).values
    entropy = -(sampled*sampled.clamp_min(1e-30).log()).sum(-1)
    ids=torch.arange(side*side,device=graph.device)
    x=ids%side;y=torch.div(ids,side,rounding_mode="floor")
    distance=torch.sqrt((x[rows,None]-x[None]).float().square()+
                        (y[rows,None]-y[None]).float().square())
    out=torch.stack((entropy.mean(-1),entropy.median(-1).values,
                     entropy.exp().mean(-1),top[:,:,0].mean(-1),
                     top[:,:,:5].sum(-1).mean(-1),top.sum(-1).mean(-1),
                     (sampled==0).float().mean((-1,-2)),
                     (sampled*distance).sum(-1).mean(-1),
                     (sampled.argmax(-1)!=reference[:,rows].argmax(-1)).float().mean(-1)),dim=-1)
    if prune_stats is None:
        prune_stats=torch.zeros((graph.shape[0],3),device=graph.device)
    graph_stats=torch.cat((out,prune_stats),dim=-1)
    transition=torch.zeros((graph.shape[0],4),device=graph.device)
    if truth is not None:
        gt=truth.to(graph.device).reshape(-1).bool()
        to_fg=(sampled*gt[None,None].float()).sum(-1)
        fg=gt[rows];bg=~fg
        if fg.any():
            transition[:,0]=to_fg[:,fg].mean(-1)
            transition[:,1]=1-transition[:,0]
        else:
            transition[:,:2]=float("nan")
        if bg.any():
            transition[:,2]=to_fg[:,bg].mean(-1)
            transition[:,3]=1-transition[:,2]
        else:
            transition[:,2:]=float("nan")
    return graph_stats.cpu(),transition.cpu()


def edge_quartile_mass(original: torch.Tensor, transformed: torch.Tensor,
                       rows: torch.Tensor) -> torch.Tensor:
    """Original-edge quartile bins and probability mass before/after inflation."""
    source=original[:,rows]
    target=transformed[:,rows]
    bounds=torch.quantile(source,torch.tensor((.25,.5,.75),device=source.device),dim=-1)
    bins=(source<bounds[0,:,:,None],
          (source>=bounds[0,:,:,None])&(source<bounds[1,:,:,None]),
          (source>=bounds[1,:,:,None])&(source<bounds[2,:,:,None]),
          source>=bounds[2,:,:,None])
    return torch.stack((torch.stack([(source*m).sum(-1).mean(-1) for m in bins],dim=-1),
                        torch.stack([(target*m).sum(-1).mean(-1) for m in bins],dim=-1)),dim=1).cpu()


@torch.inference_mode()
def aggregate(sa_qk: list[tuple[torch.Tensor,torch.Tensor]],
              features: list[torch.Tensor], seed: torch.Tensor, *, power: float,
              threshold: float, beta: float,
              truth64: torch.Tensor | None = None):
    if len(sa_qk)!=11 or len(features)!=11 or seed.shape!=(64,64):
        raise ValueError("SG30 requires 11 matched 64x64 Q/K and feature observations")
    if power<=1 or threshold<=0 or not 0<=beta<=1:
        raise ValueError("invalid locked SPARK adaptation constants")
    c=seed.float().reshape(-1)
    index,valid=neighbor_index(64,c.device)
    row_ids=diagnostic_rows(64,truth64,c.device)
    sums={name:torch.zeros_like(c) for name in ALL}
    graph_stats=torch.empty((11,8,len(MAIN),len(GRAPH_FIELDS)),dtype=torch.float32)
    transitions=torch.empty((11,8,len(MAIN),len(TRANSITION_FIELDS)),dtype=torch.float32)
    edge=torch.empty((11,8,2,2,4),dtype=torch.float32)
    iteration=torch.empty((11,8,3,len(ITER_FIELDS)),dtype=torch.float32)
    for call,(pair,feature) in enumerate(zip(sa_qk,features)):
        a=_attention(pair,64)
        if a.shape!=(8,4096,4096) or feature.shape[0]!=4096:
            raise ValueError("SG30 selected SA observation shape changed")
        ac=_apply(a,c)
        a2=torch.bmm(a,a)
        values={"S0":_apply(a,ac).mean(0)}
        graphs={0:a}
        inflated=inflate(a,power)
        values["S1"]=_apply(inflated,_apply(inflated,c)).mean(0)
        values["S2"]=_apply(inflated,ac).mean(0)
        edge[call,:,0]=edge_quartile_mass(a,inflated,row_ids)
        graphs[1]=inflated;graphs[2]=inflated
        expanded_inflated=inflate(a2,power)
        values["S3"]=_apply(expanded_inflated,c).mean(0)
        values["S4"]=_apply(expanded_inflated,ac).mean(0)
        edge[call,:,1]=edge_quartile_mass(a,expanded_inflated,row_ids)
        graphs[3]=expanded_inflated;graphs[4]=expanded_inflated
        p1,prepruned,pstats=sharpen(a,power,threshold,expanded=a2)
        values["S5"]=_apply(p1,c).mean(0)
        values["S6"]=_apply(p1,ac).mean(0)
        graphs[5]=p1;graphs[6]=p1
        local=feature_local_graph(feature,index,valid)
        fused=beta*a+(1-beta)*local[None]
        p7,_,p7stats=sharpen(fused,power,threshold)
        values["S7"]=_apply(p7,c).mean(0)
        graphs[7]=p7
        for cell,graph in graphs.items():
            stats,gt=graph_diagnostics(graph,a,row_ids,truth64,64,
                                       pstats if cell in (5,6) else p7stats if cell==7 else None)
            graph_stats[call,:,cell]=stats
            transitions[call,:,cell]=gt
        prev=a
        for iteration_index,current in enumerate((p1,)):
            stats,_=graph_diagnostics(current,a,row_ids,None,64)
            iteration[call,:,iteration_index]=torch.stack((
                (current[:,row_ids]-prev[:,row_ids]).abs().sum(-1).mean(-1).cpu(),
                stats[:,0],stats[:,6],stats[:,2],
                (current[:,row_ids].amax(-1)>.9).float().mean(-1).cpu()),dim=-1)
            prev=current
        p2,_,_=sharpen(p1,power,threshold)
        values["S5-I2"]=_apply(p2,c).mean(0)
        stats,_=graph_diagnostics(p2,a,row_ids,None,64)
        iteration[call,:,1]=torch.stack(((p2[:,row_ids]-p1[:,row_ids]).abs().sum(-1).mean(-1).cpu(),
             stats[:,0],stats[:,6],stats[:,2],(p2[:,row_ids].amax(-1)>.9).float().mean(-1).cpu()),dim=-1)
        p3,_,_=sharpen(p2,power,threshold)
        values["S5-I3"]=_apply(p3,c).mean(0)
        stats,_=graph_diagnostics(p3,a,row_ids,None,64)
        iteration[call,:,2]=torch.stack(((p3[:,row_ids]-p2[:,row_ids]).abs().sum(-1).mean(-1).cpu(),
             stats[:,0],stats[:,6],stats[:,2],(p3[:,row_ids].amax(-1)>.9).float().mean(-1).cpu()),dim=-1)
        for name in ALL:
            sums[name]+=values[name]/11
        del a,a2,inflated,expanded_inflated,p1,p2,p3,p7,prepruned,local,fused,graphs
    output={name:value.reshape(64,64) for name,value in sums.items()}
    diagnostics={"graph":graph_stats,"transition":transitions,
                 "edge_quartile_mass":edge,"iteration":iteration}
    return output,diagnostics
