"""Observation-level late support for the fixed SG18 middle proposal."""
from __future__ import annotations

import numpy as np
import torch

FEATURES = ('late_score', 'late_minus_ca', 'late_minus_middle', 'retention_ratio',
            'late_up_consensus', 'late_vs_middle_consensus', 'late_median',
            'late_trimmed_mean', 'late_std', 'late_mad')


@torch.inference_mode()
def propagate(calls, seed):
    """Return 64x64 M, L, and the 80 late values without retaining affinities."""
    c = seed.reshape(-1).float()
    if c.numel() != 4096 or len(calls) != 31:
        raise ValueError('expected 31 calls at 64x64')
    middle = torch.zeros_like(c)
    late = []
    for t in range(10, 31):
        q, k = calls[t]
        if q.shape[0] != 8:
            raise ValueError('expected eight heads')
        q = q.to(c.device).float(); k = k.to(c.device).float()
        affinity = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1]**.5, dim=-1)
        z = affinity @ c
        if t <= 20:
            middle += z.sum(0)
        else:
            late.append(z)
        del q, k, affinity, z
    middle /= 88
    observations = torch.cat(late, dim=0)
    if observations.shape != (80, 4096):
        raise ValueError('expected 80 late observations')
    late_mean = observations.mean(0)
    return middle, late_mean, observations


def features_at_pixels(observations: np.ndarray, ca: np.ndarray, middle: np.ndarray,
                       epsilon: float) -> np.ndarray:
    """Compute exact output-pixel features after bilinear resizing each observation.

    ``observations`` has shape [80, N] and the other arrays have shape [N].
    The caller keeps only middle-added pixels, so M-c is strictly positive.
    """
    x=np.asarray(observations,np.float32)
    if x.ndim!=2 or x.shape[0]!=80:
        raise ValueError('expected 80 late observations')
    ca=np.asarray(ca,np.float32);middle=np.asarray(middle,np.float32)
    if x.shape[1]!=len(ca) or len(ca)!=len(middle) or np.any(middle<=ca):
        raise ValueError('features require middle-positive candidate pixels')
    mean=x.mean(0)
    sorted_values=np.sort(x,axis=0)
    median=sorted_values[39:41].mean(0)
    mad=np.sort(np.abs(x-median),axis=0)[39:41].mean(0)
    values=(mean,mean-ca,mean-middle,(mean-ca)/(middle-ca+epsilon),
            (x>ca).mean(0),(x>middle).mean(0),median,
            sorted_values[16:64].mean(0),x.std(0),mad)
    return np.stack(values,axis=1).astype(np.float32)


def comparison_masks(truth, middle_mask, ca_mask, raw_positive):
    """Label only middle foreground newly added to CA and with raw M>c."""
    from scipy import ndimage
    truth = np.asarray(truth, bool)
    eligible = np.asarray(middle_mask, bool) & ~np.asarray(ca_mask, bool) & np.asarray(raw_positive, bool)
    tp = eligible & truth
    fp = eligible & ~truth
    distance = ndimage.distance_transform_edt(~truth) if truth.any() else np.full(truth.shape, np.inf)
    components, _ = ndimage.label(middle_mask)
    touching = np.unique(components[middle_mask & truth]); touching = touching[touching > 0]
    separate = fp & ~np.isin(components, touching)
    return eligible, (tp[eligible], fp[eligible], (fp & (distance>16))[eligible],
                      (fp & (distance>64))[eligible], separate[eligible])
