"""Complete OVAM VOC-Sim pseudo-mask pipeline from live attention.

This module mirrors the authors' released implementation: ordinary OVAM
cross-attention, their self-attention hook reduction, multiplication after
resizing to image size, a peak-relative threshold, and SimpleCRF/DenseCRF.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .attention_aggregation import resize_map
from .ovam_baseline import ovam_cross_attention_map

OVAM_SA_MIN = 0.85
OVAM_THRESHOLD = 0.40
OVAM_SIZE = 64
OVAM_CRF_PARAMETERS = (10.0, 80, 13, 3.0, 3, 5.0)


def _highest_resolution_self_modules(sample: Any) -> tuple[str, ...]:
    """Return the W×H self-attention blocks specified by OVAM Eq. 8."""
    if not sample.self_modules_by_resolution:
        return ()
    highest_resolution = max(sample.self_modules_by_resolution)
    return tuple(sorted(sample.self_modules_by_resolution[highest_resolution]))


@torch.inference_mode()
def ovam_self_attention_map(sample: Any) -> torch.Tensor:
    """Reproduce ``StableDiffusionHookerSA.get_self_attention_map``.

    The per-forward values were computed by ``CompactAttentionRecorder`` with
    the same SDPA call as the authors' ``SelfAttentionHooker``.  This function
    performs their timestep/head/block aggregation without retaining a full
    self-attention matrix.
    """
    module_maps: list[torch.Tensor] = []
    for module_name in _highest_resolution_self_modules(sample):
        resolution = next(
            size
            for size, names in sample.self_modules_by_resolution.items()
            if module_name in names
        )
        observations: list[torch.Tensor] = []
        for step_index in range(len(sample.timesteps)):
            with sample.open_attention(step_index) as tensors:
                observations.append(
                    tensors.get_tensor(f"{module_name}.ovam_self_map").float()
                )
        # Authors: stack(time).mean(time).sum(head), then resize each block.
        value = torch.stack(observations).mean(dim=0).sum(dim=0).reshape(
            resolution, resolution
        )
        module_maps.append(resize_map(value, OVAM_SIZE))
    if not module_maps:
        raise RuntimeError("No highest-resolution self-attention modules were captured")
    return -torch.stack(module_maps).mean(dim=0)


def scale_self_attention(
    self_attention: torch.Tensor, minimum: float = OVAM_SA_MIN
) -> torch.Tensor:
    value = self_attention.float()
    span = value.max() - value.min()
    if not torch.isfinite(span) or span <= 0:
        raise RuntimeError("OVAM self-attention map is constant or non-finite")
    return (value - value.min()) / span * (1.0 - minimum) + minimum


def fuse_ovam_maps(
    cross_attention: torch.Tensor,
    self_attention: torch.Tensor,
    output_shape: tuple[int, int],
    self_attention_minimum: float = OVAM_SA_MIN,
) -> torch.Tensor:
    """Return the exact full-resolution CA×SA score used before thresholding."""
    ca = F.interpolate(
        cross_attention.float()[None, None],
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    sa = F.interpolate(
        self_attention.float()[None, None],
        size=output_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    ca = ca / ca.max().clamp_min(torch.finfo(ca.dtype).eps)
    sa = scale_self_attention(sa, self_attention_minimum)
    return ca * sa


def _import_densecrf(extra_site_packages: Path | None = None):
    try:
        import denseCRF  # type: ignore
    except ImportError:
        if extra_site_packages is None:
            raise RuntimeError(
                "The OVAM DenseCRF extension is unavailable. Pass its site-packages path."
            ) from None
        sys.path.append(str(extra_site_packages))
        import denseCRF  # type: ignore
    return denseCRF


def apply_ovam_densecrf(
    image: np.ndarray,
    binary_mask: np.ndarray,
    extra_site_packages: Path | None = None,
) -> np.ndarray:
    """Apply the authors' SimpleCRF wrapper and fixed VOC-Sim parameters."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("DenseCRF image must be uint8 RGB")
    probabilities = np.stack((~binary_mask, binary_mask), axis=-1).astype(np.float32)
    densecrf = _import_densecrf(extra_site_packages)
    return np.asarray(
        densecrf.densecrf(image, probabilities, OVAM_CRF_PARAMETERS), dtype=bool
    )


@torch.inference_mode()
def ovam_full_prediction(
    sample: Any,
    densecrf_site_packages: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute live CA, live SA, fused score, and post-CRF OVAM mask."""
    if sample.generated_image is None:
        raise ValueError("The sample has no generated image for DenseCRF")
    ca = ovam_cross_attention_map(sample)
    sa = ovam_self_attention_map(sample)
    fused = fuse_ovam_maps(ca, sa, sample.generated_image.shape[:2])
    thresholded = fused > OVAM_THRESHOLD
    prediction = apply_ovam_densecrf(
        sample.generated_image,
        thresholded.cpu().numpy(),
        densecrf_site_packages,
    )
    return ca, sa, fused, torch.from_numpy(prediction)
