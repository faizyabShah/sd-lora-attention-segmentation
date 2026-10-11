"""Segmentation package."""
"""Attention-based segmentation and ablation utilities."""

from .ablation_runner import (
    SegmentationSettings,
    run_cross_attention_layer_ablation,
    run_resolution_ablation,
    run_time_resolution_ablation,
    run_threshold_normalization_ablation,
    run_time_window_ablation,
)
from .captured_dataset import CapturedAttentionDataset, CapturedSample
from .exhaustive_layer_search import ExhaustiveLayerResults, run_exhaustive_layer_search

__all__ = [
    "CapturedAttentionDataset",
    "CapturedSample",
    "ExhaustiveLayerResults",
    "SegmentationSettings",
    "run_cross_attention_layer_ablation",
    "run_exhaustive_layer_search",
    "run_resolution_ablation",
    "run_time_resolution_ablation",
    "run_threshold_normalization_ablation",
    "run_time_window_ablation",
]
