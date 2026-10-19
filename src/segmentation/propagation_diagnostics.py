"""Bounded-storage SG-Diag histograms and resumable observation statistics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

FEATURES = ('r_up', 'r_down', 'consensus', 'directional_consensus', 'std', 'mad', 'iqr',
            'median_minus_seed', 'mean_minus_seed', 'mean_median_disagreement',
            'fraction_close_to_seed', 'propagation_magnitude', 'consensus_agreement_combination')
EVENTS = ('tp_addition', 'fp_addition', 'correct_fg_removal', 'incorrect_fg_removal')
BOUNDS = tuple((-1., 1.) if k in ('median_minus_seed', 'mean_minus_seed') else (0., 1.) for k in FEATURES)


@torch.inference_mode()
def change_histograms(seed, diagnostics, target, spec):
    """Histogram every eligible pixel; AUC is binned, with no pixel sampling.

    Correction events use normalized masks; feature values use raw A_i c scale.
    Scores are diagnostic only and cannot feed candidate/fold selection.
    """
    device = seed.device
    truth = torch.as_tensor(np.asarray(target), device=device).bool()
    shape = truth.shape
    def up(x):
        return F.interpolate(x.reshape(1, 1, 64, 64).float(), shape, mode='bilinear', align_corners=False)[0, 0]
    def normalize(x):
        return (x-x.min()) / (x.max()-x.min()).clamp_min(1e-12)
    base, mean = up(normalize(seed)), up(normalize(diagnostics['mean']))
    appreciable = (mean-base).abs() > spec['change_epsilon']
    bins = spec['histogram_bins']
    hist = np.zeros((len(spec['thresholds']), 4, len(FEATURES), bins), dtype=np.int64)
    totals = np.zeros((len(spec['thresholds']), 3), dtype=np.int64)
    fields = {k: up(diagnostics[k]) for k in FEATURES if k not in
              ('directional_consensus', 'propagation_magnitude', 'consensus_agreement_combination')}
    fields['propagation_magnitude'] = (mean-base).abs()
    for ti, threshold in enumerate(spec['thresholds']):
        before, after = base >= threshold, mean >= threshold
        add, remove = after & ~before, before & ~after
        fields['directional_consensus'] = torch.where(add, fields['r_up'], fields['r_down'])
        fields['consensus_agreement_combination'] = fields['directional_consensus'] / (
            1 + fields['mean_median_disagreement'] / spec['near_seed_epsilon'])
        regions = (add & truth, add & ~truth, remove & ~truth, remove & truth)
        totals[ti] = (int(appreciable.sum()), int((before != after).sum()), int(((before != after) & appreciable).sum()))
        for fi, (name, (low, high)) in enumerate(zip(FEATURES, BOUNDS)):
            quantized = (((fields[name]-low)/(high-low)).clamp(0, 1)*bins).long().clamp_max(bins-1)
            for ei, region in enumerate(regions):
                selected = quantized[region & appreciable]
                hist[ti, ei, fi] = torch.bincount(selected, minlength=bins).cpu().numpy()
    return hist, totals


class ObservationDiagnosticWriter:
    """One atomic checkpoint per class; diagnostic commit precedes count commit.

    An interrupted image is regenerated if its metric counts were not committed.
    Re-adding its diagnostic is idempotent, so histogram sums never double count.
    """
    def __init__(self, path: Path, spec: dict):
        self.path, self.spec = path, spec
        n, bins = spec['sa']['observations'], spec['diagnostic']['histogram_bins']
        self.ids, self.scores, self.selection = [], [], []
        self.hist = np.zeros((len(spec['diagnostic']['thresholds']), 4, len(FEATURES), bins), dtype=np.int64)
        self.totals = np.zeros((len(spec['diagnostic']['thresholds']), 3), dtype=np.int64)
        self.pairwise_sum = np.zeros((n, n), dtype=np.float64)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data['spec']) != json.dumps(spec, sort_keys=True):
                    raise ValueError(f'{path}: diagnostic specification differs')
                self.ids = data['image_ids'].tolist()
                self.scores = list(data['reliability'])
                self.selection = list(data['selected'])
                self.hist, self.totals = data['histogram'], data['event_totals']
                self.pairwise_sum = data['pairwise_sum']

    def add(self, image_id, histogram, totals, diagnostics):
        if image_id in self.ids:
            return
        self.ids.append(image_id)
        self.scores.append(diagnostics['reliability'].cpu().numpy())
        self.selection.append(diagnostics['selected'].cpu().numpy())
        self.hist += histogram
        self.totals += totals
        self.pairwise_sum += diagnostics['pairwise'].cpu().numpy()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name+'.tmp.npz')
        np.savez_compressed(temporary, image_ids=np.asarray(self.ids), reliability=np.stack(self.scores),
                           selected=np.stack(self.selection), histogram=self.hist, event_totals=self.totals,
                           pairwise_sum=self.pairwise_sum, spec=np.array(json.dumps(self.spec, sort_keys=True)))
        os.replace(temporary, self.path)


def histogram_auc(positive, negative):
    """ROC-AUC (half credit to ties), and step PR-AUC / average precision.

    Both raw and reversed feature directions are returned, explicitly labelled.
    Null is returned for an absent positive or negative class.
    """
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    p, n = positive.sum(), negative.sum()
    if p == 0 or n == 0:
        return {'positive_pixels': int(p), 'negative_pixels': int(n), 'roc_auc': None, 'pr_auc': None}
    tp, fp = np.cumsum(positive[::-1]), np.cumsum(negative[::-1])
    tpr, fpr = np.r_[0, tp/p], np.r_[0, fp/n]
    roc = float(np.trapz(tpr, fpr))
    precision = tp / np.maximum(tp+fp, 1)
    ap = float(np.sum((positive[::-1]/p)*precision))
    return {'positive_pixels': int(p), 'negative_pixels': int(n), 'prevalence': float(p/(p+n)),
            'roc_auc': roc, 'pr_auc': ap}
