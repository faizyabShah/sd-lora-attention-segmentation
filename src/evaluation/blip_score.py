"""BLIP-based scoring utilities."""

from __future__ import annotations

from PIL import Image

from src.dataset.loader import load_images_from_folder


def _overlap_ratio(reference: str, generated: str) -> float:
    ref_words = set(reference.lower().split())
    if not ref_words:
        return 0.0
    gen_words = set(generated.lower().split())
    return len(ref_words & gen_words) / len(ref_words)


def compute_blip_score(images_dir: str, captions: list[str], max_images: int | None = 2500) -> float:
    """Compute BLIP overlap score against reference captions."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

    image_paths = load_images_from_folder(images_dir, max_images=max_images)
    limit = min(len(image_paths), len(captions))
    if limit == 0:
        raise ValueError("No image-caption pairs available for BLIP score.")

    scores: list[float] = []
    for idx in range(limit):
        image = Image.open(image_paths[idx]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=40)
        generated = processor.decode(out[0], skip_special_tokens=True)
        scores.append(_overlap_ratio(captions[idx], generated))

    return float(sum(scores) / len(scores))
