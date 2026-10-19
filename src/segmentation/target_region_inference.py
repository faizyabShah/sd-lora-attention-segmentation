"""Literal SG13 region scores; labels enter only the separate diagnostics."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from src.segmentation.attention_regions import spectral_partition
from src.segmentation import controlled_cv as cv


@torch.inference_mode()
def observation_statistics(calls, seed):
    """One mean graph, raw mean propagation, two-pass control, directional consensus."""
    seed = seed.reshape(-1).float()
    graph = torch.zeros((seed.numel(), seed.numel()), device=seed.device)
    one, two = torch.zeros_like(seed), torch.zeros_like(seed)
    upward, downward = torch.zeros_like(seed), torch.zeros_like(seed)
    heads = calls[0][0].shape[0]
    for query, key in calls:
        q, k = query.to(seed.device).float(), key.to(seed.device).float()
        affinity = (q @ k.transpose(-1, -2)/math.sqrt(q.shape[-1])).softmax(-1)
        propagated = affinity @ seed
        graph += affinity.mean(0)
        one += propagated.mean(0)
        two += (affinity @ propagated[..., None])[..., 0].mean(0)
        upward += (propagated > seed).sum(0)
        downward += (propagated < seed).sum(0)
        del q, k, affinity, propagated
    n = len(calls)
    consensus = torch.maximum(upward, downward)/(n*heads)
    return graph/n, one/n, two/n, consensus


@torch.inference_mode()
def region_quantities(labels, seed, propagated, consensus):
    """Columns C, P, Q, M, and the three exact raw region-score rows."""
    labels = labels.reshape(-1).long()
    regions = int(labels.max())+1
    area = torch.bincount(labels, minlength=regions)
    quantities = torch.zeros((regions, 4), device=seed.device)
    for index, values in enumerate((seed, propagated, consensus)):
        quantities[:, index].index_add_(0, labels, values.reshape(-1))
        quantities[:, index] /= area.clamp_min(1)
    quantities[:, 3].scatter_reduce_(0, labels, seed.reshape(-1), reduce='amax', include_self=True)
    p, q, core = quantities[:, 1], quantities[:, 2], quantities[:, 3]
    scores = torch.stack((p, p*q, .5*core+.5*p*q))
    return quantities, scores, area


def expanded_labels(labels, shape):
    side = math.isqrt(labels.numel())
    return F.interpolate(labels.reshape(1, 1, side, side).float(), shape, mode='nearest')[0, 0].long()


@torch.inference_mode()
def score_regions(labels, scores, target, thresholds):
    """Exact full-resolution pixel counts from region areas; no saved binary masks.

    Threshold region scores, never normalize them or blur across a region edge.
    """
    truth = torch.as_tensor(np.asarray(target), device=scores.device).bool()
    full_labels = expanded_labels(labels, truth.shape)
    k = scores.shape[1]
    area = torch.bincount(full_labels.reshape(-1), minlength=k)
    positive = torch.bincount(full_labels[truth], minlength=k)
    pred = scores[:, None, :] >= torch.as_tensor(thresholds, device=scores.device)[None, :, None]
    tp = (pred*positive).sum(-1)
    fp = (pred*(area-positive)).sum(-1)
    fn = positive.sum()-tp
    tn = area.sum()-tp-fp-fn
    counts = torch.stack((tp, fp, fn, tn), -1).cpu().numpy().astype(np.int64)
    return counts


@torch.inference_mode()
def oracle_diagnostic(labels, target):
    truth = torch.as_tensor(np.asarray(target), device=labels.device).float()
    side = math.isqrt(labels.numel())
    small = F.interpolate(truth[None, None], (side, side), mode='nearest')[0, 0].reshape(-1)
    area = torch.bincount(labels)
    positives = torch.zeros(len(area), device=labels.device).index_add_(0, labels, small)
    fraction = positives/area.clamp_min(1)
    oracle = (fraction >= .5) & (area > 0)
    full = oracle[expanded_labels(labels, truth.shape)].cpu().numpy()
    count = cv.score_masks(full[None], target, thresholds=(.5,))[0, 0]
    return fraction, count
