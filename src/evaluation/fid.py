"""FID metric computation."""

from __future__ import annotations

from src.dataset.loader import load_images_from_folder
from src.evaluation.utils import build_image_transform, load_image_tensor, to_uint8_batch


def compute_fid(real_dir: str, generated_dir: str, max_images: int | None = 2500) -> float:
    """Compute Frechet Inception Distance between two image directories."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = metric.to(device)
    transform = build_image_transform(size=512)

    real_paths = load_images_from_folder(real_dir, max_images=max_images, extensions=(".png", ".jpg", ".jpeg"))
    gen_paths = load_images_from_folder(generated_dir, max_images=max_images, extensions=(".png", ".jpg", ".jpeg"))
    limit = min(len(real_paths), len(gen_paths))
    if limit == 0:
        raise ValueError("No image pairs available for FID.")
    for idx in range(limit):
        real_tensor = load_image_tensor(real_paths[idx], transform, device)
        gen_tensor = load_image_tensor(gen_paths[idx], transform, device)
        metric.update(to_uint8_batch(real_tensor), real=True)
        metric.update(to_uint8_batch(gen_tensor), real=False)

    return float(metric.compute().item())
