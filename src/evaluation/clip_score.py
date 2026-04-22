"""CLIP score computation."""

from __future__ import annotations

from PIL import Image

from src.dataset.loader import load_images_from_folder


def compute_clip_score(images_dir: str, captions: list[str], max_images: int | None = 2500) -> float:
    """Compute CLIP score for image-caption alignment."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    image_paths = load_images_from_folder(images_dir, max_images=max_images)
    limit = min(len(image_paths), len(captions))
    if limit == 0:
        raise ValueError("No image-caption pairs available for CLIP score.")

    scores: list[float] = []
    for idx in range(limit):
        image = Image.open(image_paths[idx]).convert("RGB")
        inputs = processor(text=[captions[idx]], images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        scores.append(float(outputs.logits_per_image.item()))

    return float(sum(scores) / len(scores))
