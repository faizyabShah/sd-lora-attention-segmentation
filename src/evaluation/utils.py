"""Shared helpers for evaluation metrics."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def build_image_transform(size: int = 512) -> transforms.Compose:
    """Create the evaluation image transform pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )


def load_image_tensor(path: str | Path, transform: transforms.Compose, device: str) -> torch.Tensor:
    """Load image and return BCHW tensor on target device."""
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def to_uint8_batch(batch: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] float tensor batch to uint8 image range for FID metric."""
    return (batch * 255.0).clamp(0, 255).to(torch.uint8)
