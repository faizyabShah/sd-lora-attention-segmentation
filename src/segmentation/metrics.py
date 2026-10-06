"""Compact segmentation accumulators used by offline ablations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    image_iou_sum: float = 0.0
    images: int = 0


class BinarySegmentationMetrics:
    """Accumulate foreground confusion counts independently for each class."""

    def __init__(self) -> None:
        self.counts: dict[str, _Counts] = defaultdict(_Counts)

    def update(self, classname: str, prediction: torch.Tensor, target: np.ndarray) -> None:
        pred = prediction.detach().cpu().numpy().astype(bool, copy=False)
        truth = target.astype(bool, copy=False)
        if pred.shape != truth.shape:
            raise ValueError(f"Prediction {pred.shape} and target {truth.shape} differ")
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        tn = int(np.count_nonzero(~pred & ~truth))
        item = self.counts[classname]
        item.tp += tp
        item.fp += fp
        item.fn += fn
        item.tn += tn
        union = tp + fp + fn
        item.image_iou_sum += tp / union if union else 1.0
        item.images += 1

    def result(self) -> dict:
        per_class: dict[str, dict] = {}
        foreground_ious: list[float] = []
        background_ious: list[float] = []
        image_iou_sum = 0.0
        image_count = 0
        for classname in sorted(self.counts):
            value = self.counts[classname]
            foreground_union = value.tp + value.fp + value.fn
            background_union = value.tn + value.fp + value.fn
            foreground_iou = value.tp / foreground_union if foreground_union else 1.0
            background_iou = value.tn / background_union if background_union else 1.0
            per_class[classname] = {
                "iou": foreground_iou,
                "background_iou": background_iou,
                "mean_image_iou": value.image_iou_sum / value.images,
                "images": value.images,
                "tp": value.tp,
                "fp": value.fp,
                "fn": value.fn,
                "tn": value.tn,
            }
            foreground_ious.append(foreground_iou)
            background_ious.append(background_iou)
            image_iou_sum += value.image_iou_sum
            image_count += value.images
        return {
            "miou": float(np.mean(foreground_ious)),
            "mean_background_iou": float(np.mean(background_ious)),
            "mean_image_iou": image_iou_sum / image_count,
            "images": image_count,
            "classes": len(per_class),
            "per_class": per_class,
        }
