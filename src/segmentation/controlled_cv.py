"""Shared cross-validation infrastructure for the controlled CA/SA protocol.

Implements Sections 2, 3, 5, 5a and the Section 9 result contract of
``docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md``:

* ``score_maps`` turns a batch of 64x64 candidate maps into per-threshold
  confusion counts at annotation resolution, exactly as ``attention_to_mask``
  followed by ``BinarySegmentationMetrics`` would, plus the threshold-free
  64x64 average precision;
* ``ShardWriter`` stores those per-image records for one class shard in a
  compact, resumable ``.npz``; ``load_results`` reads one or more shards;
* ``decide`` applies the Section 5 decision rule, ``cross_validate`` the
  Section 5a fold procedure, and ``paired_bootstrap`` the paired per-image
  interval.

Counts are indexed ``[configuration, image, threshold, (tp, fp, fn, tn)]``.
Selection always uses threshold 0.45; 0.35 and 0.55 are calibration
diagnostics only.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

THRESHOLDS = (0.35, 0.45, 0.55)
SELECTION_THRESHOLD = 0.45
SELECTION_INDEX = THRESHOLDS.index(SELECTION_THRESHOLD)
COUNT_FIELDS = ("tp", "fp", "fn", "tn")
FOLD_COUNT = 5
FOLD_SEED = 20261013
BOOTSTRAP_RESAMPLES = 20_000

PROMOTE = 0.0100
CONDITIONAL = 0.0050
CONDITIONAL_MIN_CLASSES = 12
CONDITIONAL_MAX_CLASS_LOSS = 0.050
TIE_MARGIN = 0.0020
FREEZE_MIN_FOLDS = 4


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def make_folds(
    rows: Iterable[Mapping[str, str]], folds: int = FOLD_COUNT, seed: int = FOLD_SEED
) -> list[dict[str, object]]:
    """Assign annotated images to folds, stratified by class.

    Images of each class are shuffled with a fixed seed, then dealt to folds
    with one counter that continues across classes (classes in sorted order),
    so every class is split as evenly as possible and fold sizes differ by at
    most one image.
    """
    annotated = [r for r in rows if str(r["annotated"]).lower() == "true"]
    by_class: dict[str, list[str]] = {}
    for row in annotated:
        by_class.setdefault(row["classname"], []).append(row["image_id"])
    rng = np.random.default_rng(seed)
    out: list[dict[str, object]] = []
    counter = 0
    for classname in sorted(by_class):
        ids = sorted(by_class[classname], key=_natural_key)
        for index in rng.permutation(len(ids)):
            out.append({"image_id": ids[index], "classname": classname, "fold": counter % folds})
            counter += 1
    return out


def write_folds(path: Path, assignments: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "classname", "fold"))
        writer.writeheader()
        writer.writerows(sorted(assignments, key=lambda r: _natural_key(str(r["image_id"]))))


def read_folds(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image_id"]: int(row["fold"]) for row in csv.DictReader(handle)}


def _natural_key(value: str) -> tuple:
    head, _, tail = value.rpartition("-")
    return (head, int(tail)) if tail.isdigit() else (value, -1)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def normalize_batch(maps: torch.Tensor, method: str) -> torch.Tensor:
    """Batched ``attention_aggregation.normalize_map`` for ``[C, H, W]`` maps."""
    flat = maps.float().reshape(maps.shape[0], -1)
    if method == "minmax":
        low = flat.amin(dim=1, keepdim=True)
        scale = flat.amax(dim=1, keepdim=True) - low
    elif method == "percentile_01_99":
        quantiles = torch.quantile(flat, torch.tensor([0.01, 0.99], device=flat.device), dim=1)
        low, scale = quantiles[0][:, None], (quantiles[1] - quantiles[0])[:, None]
    else:
        raise ValueError(f"Unsupported normalization: {method}")
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    out = torch.where(scale > 0, (flat - low) / safe, torch.zeros_like(flat))
    if method == "percentile_01_99":
        out = out.clamp(0, 1)
    return out.reshape(maps.shape)


def target_at_size(target: np.ndarray | torch.Tensor, size: int = 64) -> torch.Tensor:
    """Nearest downsampling used for the 64x64 diagnostics (as ``attention_analysis``)."""
    value = torch.as_tensor(np.asarray(target), dtype=torch.float32)[None, None]
    return F.interpolate(value, size=(size, size), mode="nearest")[0, 0].bool()


def average_precision_batch(scores: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Average precision of each row of ``scores`` [C, P] against ``truth`` [P].

    Same definition and tie handling as ``attention_analysis._average_precision``
    (stable descending order). NaN when the target has no positive pixel.
    """
    truth = truth.reshape(-1).bool()
    positives = int(truth.sum())
    if positives == 0:
        return torch.full((scores.shape[0],), float("nan"), dtype=torch.float64)
    order = torch.sort(scores.reshape(scores.shape[0], -1), dim=1, descending=True, stable=True).indices
    hits = truth.to(order.device)[order].double()
    ranks = torch.arange(1, hits.shape[1] + 1, device=hits.device, dtype=torch.float64)
    precision = hits.cumsum(dim=1) / ranks
    return ((precision * hits).sum(dim=1) / positives).cpu()


@torch.inference_mode()
def score_maps(
    maps: torch.Tensor,
    target: np.ndarray | torch.Tensor,
    normalization: str = "minmax",
    thresholds: Sequence[float] = THRESHOLDS,
    chunk: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Score a batch of 64x64 maps against one annotation.

    Returns ``counts`` int64 ``[C, len(thresholds), 4]`` at annotation
    resolution (normalize at 64x64, bilinear resize, ``>= threshold``, the
    ``attention_to_mask`` rule) and ``ap`` float64 ``[C]`` at 64x64.
    """
    if maps.ndim != 3:
        raise ValueError(f"Expected [C, H, W] maps, got {tuple(maps.shape)}")
    device = maps.device
    truth = torch.as_tensor(np.asarray(target), device=device).bool()
    shape = tuple(truth.shape)
    normalized = normalize_batch(maps, normalization)
    counts = np.zeros((maps.shape[0], len(thresholds), 4), dtype=np.int64)
    truth_pixels = int(truth.sum())
    total = truth.numel()
    for start in range(0, maps.shape[0], chunk):
        block = normalized[start : start + chunk]
        resized = F.interpolate(block[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0]
        for t_index, threshold in enumerate(thresholds):
            prediction = resized >= threshold
            tp = (prediction & truth).sum(dim=(1, 2))
            predicted = prediction.sum(dim=(1, 2))
            fp = predicted - tp
            fn = truth_pixels - tp
            tn = total - tp - fp - fn
            counts[start : start + chunk, t_index] = torch.stack((tp, fp, fn, tn), dim=1).cpu().numpy()
    small_truth = target_at_size(truth.cpu().numpy(), maps.shape[-1])
    ap = average_precision_batch(normalized.reshape(maps.shape[0], -1), small_truth)
    return counts, ap.numpy()


def score_masks(
    masks: torch.Tensor | np.ndarray, target: np.ndarray | torch.Tensor,
    thresholds: Sequence[float] = THRESHOLDS,
) -> np.ndarray:
    """Counts ``[C, T, 4]`` for already-binary full-resolution masks (e.g. after DenseCRF).

    A binary mask has no threshold, so the same counts are repeated for every
    threshold slot; store NaN as its AP.
    """
    pred = torch.as_tensor(np.asarray(masks)).bool()
    truth = torch.as_tensor(np.asarray(target)).bool()
    tp = (pred & truth).sum(dim=(1, 2))
    fp = (pred & ~truth).sum(dim=(1, 2))
    fn = (~pred & truth).sum(dim=(1, 2))
    tn = (~pred & ~truth).sum(dim=(1, 2))
    counts = torch.stack((tp, fp, fn, tn), dim=1).numpy().astype(np.int64)
    return np.repeat(counts[:, None], len(thresholds), axis=1)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
class ShardWriter:
    """Resumable per-image result store for one class shard.

    The configuration list is fixed when the shard is created; every image
    must provide counts for every configuration, in that order. Each ``add``
    rewrites the shard atomically, so a killed job loses at most one image.
    ``settings`` must match on resume; the Git commit and the runtime
    environment (``runtime_environment``) are recorded separately, one entry
    per session, so that a documentation-only pull or a library update does
    not block resuming but is still visible in the shard.
    """

    def __init__(
        self,
        path: Path,
        configurations: Sequence[str],
        folds: Mapping[str, int],
        settings: Mapping[str, object],
        thresholds: Sequence[float] = THRESHOLDS,
    ) -> None:
        self.path = Path(path)
        self.configurations = tuple(configurations)
        if len(set(self.configurations)) != len(self.configurations):
            raise ValueError("Configuration names must be unique")
        self.thresholds = tuple(float(t) for t in thresholds)
        self.folds = dict(folds)
        self.settings = dict(settings)
        self.commits: list[str] = []
        self.environments: list[dict] = []
        self.image_ids: list[str] = []
        self.classes: list[str] = []
        self.counts: list[np.ndarray] = []
        self.ap: list[np.ndarray] = []
        if self.path.exists():
            self._resume()
        commit = git_commit()
        if not self.commits or self.commits[-1] != commit:
            self.commits.append(commit)
        environment = {"commit": commit, **runtime_environment()}
        if not self.environments or _without_job(self.environments[-1]) != _without_job(environment):
            self.environments.append(environment)

    def _resume(self) -> None:
        loaded = load_results([self.path])
        if loaded.configurations != self.configurations:
            raise ValueError(f"{self.path}: configuration list differs from this run")
        if loaded.thresholds != self.thresholds:
            raise ValueError(f"{self.path}: thresholds differ from this run")
        if loaded.settings != _jsonable(self.settings):
            raise ValueError(f"{self.path}: settings differ from this run")
        self.commits = list(loaded.commits)
        self.environments = [dict(e) for e in loaded.environments]
        self.image_ids = list(loaded.image_ids)
        self.classes = list(loaded.classes)
        self.counts = [loaded.counts[:, i] for i in range(loaded.counts.shape[1])]
        self.ap = [loaded.ap[:, i] for i in range(loaded.ap.shape[1])]

    @property
    def completed(self) -> set[str]:
        return set(self.image_ids)

    def add(self, image_id: str, classname: str, counts: np.ndarray, ap: np.ndarray | None = None) -> None:
        if image_id in self.completed:
            raise ValueError(f"{image_id} is already stored in {self.path}")
        if image_id not in self.folds:
            raise KeyError(f"{image_id} is not in the fold file")
        counts = np.asarray(counts, dtype=np.int64)
        expected = (len(self.configurations), len(self.thresholds), 4)
        if counts.shape != expected:
            raise ValueError(f"Counts for {image_id} have shape {counts.shape}, expected {expected}")
        if np.any(counts < 0):
            raise ValueError(f"Negative counts for {image_id}")
        ap = np.full(len(self.configurations), np.nan) if ap is None else np.asarray(ap, dtype=np.float64)
        if ap.shape != (len(self.configurations),):
            raise ValueError(f"AP for {image_id} has shape {ap.shape}")
        self.image_ids.append(image_id)
        self.classes.append(classname)
        self.counts.append(counts)
        self.ap.append(ap)
        self.save()

    def save(self, complete: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp.npz")
        counts = np.stack(self.counts, axis=1) if self.counts else np.zeros(
            (len(self.configurations), 0, len(self.thresholds), 4), dtype=np.int64
        )
        ap = np.stack(self.ap, axis=1) if self.ap else np.zeros((len(self.configurations), 0))
        np.savez_compressed(
            temporary,
            configurations=np.array(self.configurations),
            image_ids=np.array(self.image_ids, dtype=str),
            classes=np.array(self.classes, dtype=str),
            folds=np.array([self.folds[i] for i in self.image_ids], dtype=np.int8),
            thresholds=np.array(self.thresholds),
            counts=counts.astype(np.int32 if counts.size == 0 or counts.max() < 2**31 else np.int64),
            ap=ap.astype(np.float32),
            settings=np.array(json.dumps(_jsonable(self.settings), sort_keys=True)),
            commits=np.array(self.commits, dtype=str),
            environments=np.array(json.dumps(_jsonable(self.environments), sort_keys=True)),
            complete=np.array(bool(complete)),
        )
        os.replace(temporary, self.path)

    def finish(self) -> None:
        self.save(complete=True)


@dataclass
class Results:
    """Per-image results of one or more shards with the same configurations."""

    configurations: tuple[str, ...]
    thresholds: tuple[float, ...]
    image_ids: tuple[str, ...]
    classes: tuple[str, ...]
    folds: np.ndarray
    counts: np.ndarray  # [C, N, T, 4] int64
    ap: np.ndarray  # [C, N] float64
    settings: dict
    complete: tuple[bool, ...] = field(default=())
    commits: tuple[str, ...] = field(default=())
    environments: tuple[dict, ...] = field(default=())

    def index(self, name: str) -> int:
        return self.configurations.index(name)

    def subset(self, mask: np.ndarray) -> "Results":
        mask = np.asarray(mask, dtype=bool)
        return Results(
            self.configurations, self.thresholds,
            tuple(np.array(self.image_ids)[mask]), tuple(np.array(self.classes)[mask]),
            self.folds[mask], self.counts[:, mask], self.ap[:, mask], self.settings, self.complete,
            self.commits, self.environments,
        )

    def records(self, threshold: float = SELECTION_THRESHOLD) -> Iterable[dict]:
        """Row view matching the Section 9 contract (for inspection and export)."""
        t = self.thresholds.index(threshold)
        for c, name in enumerate(self.configurations):
            for i, image_id in enumerate(self.image_ids):
                tp, fp, fn, tn = (int(v) for v in self.counts[c, i, t])
                union = tp + fp + fn
                yield {
                    "image_id": image_id, "class": self.classes[i], "fold": int(self.folds[i]),
                    "configuration": name, "threshold": threshold,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "image_iou": tp / union if union else 1.0, "ap": float(self.ap[c, i]),
                }


def load_results(paths: Iterable[Path]) -> Results:
    """Concatenate shards; configurations, thresholds and settings must agree."""
    parts = []
    for path in sorted(Path(p) for p in paths):
        with np.load(path, allow_pickle=False) as data:
            parts.append({key: data[key] for key in data.files} | {"path": path})
    if not parts:
        raise FileNotFoundError("No result shards given")
    first = parts[0]
    configurations = tuple(str(v) for v in first["configurations"])
    thresholds = tuple(float(v) for v in first["thresholds"])
    settings = json.loads(str(first["settings"]))
    for part in parts[1:]:
        if tuple(str(v) for v in part["configurations"]) != configurations:
            raise ValueError(f"{part['path']}: configuration list differs from {first['path']}")
        if tuple(float(v) for v in part["thresholds"]) != thresholds:
            raise ValueError(f"{part['path']}: thresholds differ")
        if json.loads(str(part["settings"])) != settings:
            raise ValueError(f"{part['path']}: settings differ")
    image_ids = tuple(str(v) for p in parts for v in p["image_ids"])
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("An image appears in more than one shard")
    return Results(
        configurations=configurations,
        thresholds=thresholds,
        image_ids=image_ids,
        classes=tuple(str(v) for p in parts for v in p["classes"]),
        folds=np.concatenate([p["folds"].astype(np.int64) for p in parts]),
        counts=np.concatenate([p["counts"].astype(np.int64) for p in parts], axis=1),
        ap=np.concatenate([p["ap"].astype(np.float64) for p in parts], axis=1),
        settings=settings,
        complete=tuple(bool(p["complete"]) for p in parts),
        commits=tuple(dict.fromkeys(str(c) for p in parts for c in p.get("commits", []))),
        environments=_unique_environments(p.get("environments") for p in parts),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def class_ious(results: Results, configuration: int, threshold_index: int = SELECTION_INDEX,
               mask: np.ndarray | None = None) -> dict[str, float]:
    """Pooled foreground IoU per class (``BinarySegmentationMetrics`` definition)."""
    counts = results.counts[configuration, :, threshold_index]
    classes = np.array(results.classes)
    if mask is not None:
        counts, classes = counts[mask], classes[mask]
    out = {}
    for name in sorted(set(classes)):
        tp, fp, fn = counts[classes == name, :3].sum(axis=0)
        union = tp + fp + fn
        out[name] = float(tp / union) if union else 1.0
    return out


def miou(results: Results, configuration: int, threshold_index: int = SELECTION_INDEX,
         mask: np.ndarray | None = None) -> float:
    return float(np.mean(list(class_ious(results, configuration, threshold_index, mask).values())))


def image_ious(results: Results, configuration: int, threshold_index: int = SELECTION_INDEX) -> np.ndarray:
    tp, fp, fn = (results.counts[configuration, :, threshold_index, k] for k in range(3))
    union = tp + fp + fn
    return np.where(union > 0, tp / np.maximum(union, 1), 1.0)


def summary_row(results: Results, configuration: int, mask: np.ndarray | None = None) -> dict:
    keep = np.ones(len(results.image_ids), bool) if mask is None else np.asarray(mask, bool)
    row = {"configuration": results.configurations[configuration], "images": int(keep.sum())}
    for t_index, threshold in enumerate(results.thresholds):
        counts = results.counts[configuration, keep, t_index]
        tp, fp, fn = counts[:, :3].sum(axis=0)
        row[f"miou@{threshold:.2f}"] = miou(results, configuration, t_index, keep)
        row[f"pixel_iou@{threshold:.2f}"] = float(tp / max(tp + fp + fn, 1))
    row["mean_image_iou@0.45"] = float(image_ious(results, configuration)[keep].mean())
    row["mean_ap"] = float(np.nanmean(results.ap[configuration, keep])) if keep.any() else float("nan")
    return row


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = FOLD_SEED) -> dict[str, float]:
    """Mean and 95% interval of the paired per-image difference ``a - b``."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        means[start:stop] = diff[rng.integers(0, len(diff), size=(stop - start, len(diff)))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(diff.mean()), "ci_low": float(low), "ci_high": float(high),
            "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()), "ties": int((diff == 0).sum())}


# --------------------------------------------------------------------------- #
# Decision rule (Section 5) and cross-validation (Section 5a)
# --------------------------------------------------------------------------- #
def assess(results: Results, reference: int, candidate: int, mask: np.ndarray | None = None) -> dict:
    """Classify one candidate against the reference on the selected images."""
    ref = class_ious(results, reference, SELECTION_INDEX, mask)
    cand = class_ious(results, candidate, SELECTION_INDEX, mask)
    delta = float(np.mean(list(cand.values())) - np.mean(list(ref.values())))
    class_deltas = {c: cand[c] - ref[c] for c in ref}
    improved = sum(d > 0 for d in class_deltas.values())
    worst = min(class_deltas.values())
    if delta >= PROMOTE:
        verdict = "promote"
    elif delta >= CONDITIONAL and improved >= CONDITIONAL_MIN_CLASSES and worst >= -CONDITIONAL_MAX_CLASS_LOSS:
        verdict = "conditional"
    else:
        verdict = "marginal"
    return {"configuration": results.configurations[candidate], "delta": delta,
            "classes_improved": int(improved), "worst_class_delta": float(worst), "verdict": verdict}


def decide(results: Results, reference: str, candidates: Sequence[str],
           inclusiveness: Mapping[str, float] | None = None, mask: np.ndarray | None = None) -> dict:
    """Section 5: choose among candidates, or keep the reference.

    Eligible candidates are those with verdict ``promote`` or ``conditional``.
    Among eligible candidates within ``TIE_MARGIN`` mIoU of the best one, the
    most inclusive (largest ``inclusiveness``; ties by higher mIoU) is chosen.
    """
    ref = results.index(reference)
    assessments = [assess(results, ref, results.index(c), mask) for c in candidates if c != reference]
    eligible = [a for a in assessments if a["verdict"] != "marginal"]
    if not eligible:
        return {"chosen": reference, "reason": "no eligible candidate", "assessments": assessments}
    best = max(a["delta"] for a in eligible)
    near = [a for a in eligible if a["delta"] >= best - TIE_MARGIN]
    weight = inclusiveness or {}
    chosen = max(near, key=lambda a: (weight.get(a["configuration"], 0.0), a["delta"]))
    return {"chosen": chosen["configuration"], "reason": chosen["verdict"], "assessments": assessments}


def cross_validate(results: Results, reference: str, candidates: Sequence[str],
                   inclusiveness: Mapping[str, float] | None = None,
                   folds: int = FOLD_COUNT, freeze_min_folds: int = FREEZE_MIN_FOLDS) -> dict:
    """Section 5a: select on four folds, score on the fifth, rotate, then freeze.

    A change is frozen only if it wins at least ``freeze_min_folds`` fold votes
    **and** the pooled out-of-fold mIoU of the selection procedure exceeds the
    reference's pooled out-of-fold mIoU (the OOF guard, protocol Section 5a
    step 3). Otherwise the reference is kept.
    """
    if set(np.unique(results.folds)) != set(range(folds)):
        raise ValueError(f"Results must contain all {folds} folds")
    ref = results.index(reference)
    per_fold = []
    oof = np.zeros((len(results.image_ids), 4), dtype=np.int64)  # held-out counts at 0.45
    oof_ref = results.counts[ref, :, SELECTION_INDEX]
    for k in range(folds):
        held = results.folds == k
        decision = decide(results, reference, candidates, inclusiveness, mask=~held)
        chosen = results.index(decision["chosen"])
        oof[held] = results.counts[chosen, held, SELECTION_INDEX]
        per_fold.append({
            "fold": k,
            "chosen": decision["chosen"],
            "reason": decision["reason"],
            "train_miou_chosen": miou(results, chosen, mask=~held),
            "train_miou_reference": miou(results, ref, mask=~held),
            "heldout_miou_chosen": miou(results, chosen, mask=held),
            "heldout_miou_reference": miou(results, ref, mask=held),
        })
    picks: dict[str, int] = {}
    for row in per_fold:
        picks[row["chosen"]] = picks.get(row["chosen"], 0) + 1

    def pooled(counts: np.ndarray) -> float:
        classes = np.array(results.classes)
        values = []
        for name in sorted(set(classes)):
            tp, fp, fn = counts[classes == name, :3].sum(axis=0)
            values.append(tp / (tp + fp + fn) if tp + fp + fn else 1.0)
        return float(np.mean(values))

    oof_selection, oof_reference = pooled(oof), pooled(oof_ref)
    vote_winner, votes = max(picks.items(), key=lambda kv: (kv[1], kv[0] == reference))
    vote_passed = vote_winner != reference and votes >= freeze_min_folds
    oof_guard_passed = oof_selection > oof_reference
    if vote_winner == reference:
        frozen, freeze_reason = reference, "reference won the fold vote"
    elif not vote_passed:
        frozen, freeze_reason = reference, f"fewer than {freeze_min_folds} fold votes"
    elif not oof_guard_passed:
        frozen, freeze_reason = reference, "OOF guard failed: no out-of-fold gain over the reference"
    else:
        frozen, freeze_reason = vote_winner, "fold vote and OOF guard passed"

    f = results.index(frozen)
    calibration = {
        f"delta_miou@{t:.2f}": miou(results, f, i) - miou(results, ref, i) for i, t in enumerate(results.thresholds)
    }
    signs = {np.sign(round(v, 12)) for v in calibration.values()}
    return {
        "reference": reference,
        "candidates": list(candidates),
        "folds": per_fold,
        "selection_counts": picks,
        "vote_winner": vote_winner,
        "vote_winner_votes": votes,
        "vote_passed": vote_passed,
        "oof_guard_passed": oof_guard_passed,
        "frozen": frozen,
        "frozen_votes": picks.get(frozen, 0),
        "freeze_reason": freeze_reason,
        "out_of_fold_miou_selection": oof_selection,
        "out_of_fold_miou_reference": oof_reference,
        "oof_gain": oof_selection - oof_reference,
        "in_sample_miou_frozen": miou(results, f),
        "in_sample_miou_reference": miou(results, ref),
        "paired_bootstrap_frozen_vs_reference": paired_bootstrap(image_ious(results, f), image_ious(results, ref)),
        "calibration": calibration,
        "calibration_dependent": frozen != reference and len(signs - {0.0}) > 1,
        "mean_ap_frozen": float(np.nanmean(results.ap[f])),
        "mean_ap_reference": float(np.nanmean(results.ap[ref])),
    }


# --------------------------------------------------------------------------- #
# Settings helpers
# --------------------------------------------------------------------------- #
def runtime_environment() -> dict:
    """Versions and hardware of the current process (protocol Section 22)."""
    import platform
    import socket
    from importlib import metadata

    def version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "not installed"

    gpu = "none"
    if torch.cuda.is_available():
        try:
            gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        except RuntimeError:
            gpu = "unavailable"
    job = os.environ.get("SLURM_JOB_ID", "")
    if os.environ.get("SLURM_ARRAY_JOB_ID"):
        job = f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '')}"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "diffusers": version("diffusers"),
        "transformers": version("transformers"),
        "numpy": np.__version__,
        "gpu": gpu,
        "node": socket.gethostname().split(".")[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "slurm_job": job,
    }


def _without_job(environment: Mapping) -> dict:
    return {k: v for k, v in environment.items() if k != "slurm_job"}


def _unique_environments(values) -> tuple[dict, ...]:
    seen: dict[str, dict] = {}
    for value in values:
        if value is None:
            continue  # shards written before environments were recorded
        for environment in json.loads(str(value)):
            seen.setdefault(json.dumps(_without_job(environment), sort_keys=True), environment)
    return tuple(seen.values())


def git_commit(root: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root or Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _jsonable(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))
