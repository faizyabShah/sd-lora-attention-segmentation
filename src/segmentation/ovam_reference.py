"""Published OVAM postprocessing using the authors' released VOC-Sim artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def published_ovam_fused_map(
    cross_attention: torch.Tensor,
    self_attention: torch.Tensor,
    output_shape: tuple[int, int],
    self_attention_min: float = 0.85,
) -> torch.Tensor:
    """Reproduce Eq. 8-9 fusion before thresholding and DenseCRF.

    OVAM first resizes both 64x64 maps, max-normalizes cross-attention,
    min-max normalizes self-attention into ``[self_attention_min, 1]``, and
    multiplies them.
    """
    ca = F.interpolate(
        cross_attention.float()[None, None], size=output_shape, mode="bilinear",
        align_corners=False,
    )[0, 0]
    sa = F.interpolate(
        self_attention.float()[None, None], size=output_shape, mode="bilinear",
        align_corners=False,
    )[0, 0]
    ca_max = ca.max()
    ca = ca / ca_max if ca_max > 0 else torch.zeros_like(ca)
    sa_min, sa_max = sa.min(), sa.max()
    if sa_max > sa_min:
        sa = (sa - sa_min) / (sa_max - sa_min)
        sa = sa * (1.0 - self_attention_min) + self_attention_min
    else:
        sa = torch.ones_like(sa)
    return ca * sa


def load_released_ovam_reference(
    row: Mapping[str, str], manifest_root: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the full unoptimized OVAM fused map and released post-CRF mask."""
    image_stem = Path(row["image"]).stem
    heatmap_path = manifest_root / "heatmaps" / f"{image_stem}_heatmap_token0.npy"
    evaluation_root = manifest_root.parents[1]
    sa_path = (
        evaluation_root / "ovam_sa" / "voc_sim" / "sa"
        / f"sa15_{row['classname']}_{row['seed']}_sa.npy"
    )
    mask_path = (
        evaluation_root / "ovam_sa" / "voc_sim" / "mask"
        / f"{image_stem}_ovamsa_crf_mask.png"
    )
    missing = [path for path in (heatmap_path, sa_path, mask_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing released OVAM artifact: {missing[0]}")

    stored_heatmap = np.load(heatmap_path)
    if stored_heatmap.ndim != 3 or stored_heatmap.shape[0] < 2:
        raise ValueError(f"Unexpected OVAM heatmap shape {stored_heatmap.shape}: {heatmap_path}")
    cross_attention = torch.from_numpy(stored_heatmap[1])
    self_attention = torch.from_numpy(np.load(sa_path))
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"))
    prediction = torch.from_numpy(np.any(mask_rgb != 0, axis=-1))
    fused = published_ovam_fused_map(
        cross_attention, self_attention, tuple(prediction.shape), self_attention_min=0.85
    )
    return fused, prediction
