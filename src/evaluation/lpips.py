"""LPIPS metric computation."""

from __future__ import annotations

from src.dataset.loader import load_images_from_folder
from src.evaluation.utils import build_image_transform, load_image_tensor


def compute_lpips(reference_dir: str, generated_dir: str, max_images: int | None = 2500) -> float:
    """Compute LPIPS similarity metric."""
    import lpips
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = lpips.LPIPS(net="alex").to(device)
    transform = build_image_transform(size=512)

    ref_paths = load_images_from_folder(reference_dir, max_images=max_images)
    gen_paths = load_images_from_folder(generated_dir, max_images=max_images)
    limit = min(len(ref_paths), len(gen_paths))
    if limit == 0:
        raise ValueError("No image pairs available for LPIPS.")

    scores: list[float] = []
    for idx in range(limit):
        ref = load_image_tensor(ref_paths[idx], transform, device) * 2 - 1
        gen = load_image_tensor(gen_paths[idx], transform, device) * 2 - 1
        score = scorer(ref, gen)
        scores.append(float(score.item()))
    return float(sum(scores) / len(scores))
