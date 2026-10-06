"""Segmentation package."""
"""Attention-based segmentation and ablation utilities."""

from .ablation_runner import (
    SegmentationSettings,
    run_resolution_ablation,
    run_time_window_ablation,
)
from .captured_dataset import CapturedAttentionDataset, CapturedSample

__all__ = [
    "CapturedAttentionDataset",
    "CapturedSample",
    "SegmentationSettings",
    "run_resolution_ablation",
    "run_time_window_ablation",
]
