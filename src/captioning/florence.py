"""Florence captioning entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

from src.captioning.utils import clean_caption
from src.dataset.loader import get_image_paths


def _fixed_get_imports(filename: str) -> list[str]:
    imports = get_imports(filename)
    if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_florence_model(
    model_id: str = "microsoft/Florence-2-base",
    device: str | None = None,
) -> tuple[AutoProcessor, Any, str]:
    """Load Florence model and processor with flash-attn fallback."""
    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    resolved_device = requested_device
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        )
    model = model.to(resolved_device).eval()
    return processor, model, resolved_device


def generate_caption(
    image_path: str | Path,
    processor: AutoProcessor,
    model: Any,
    device: str,
    prompt: str = "<DETAILED_CAPTION>",
    max_new_tokens: int = 80,
    num_beams: int = 3,
) -> str:
    """Generate a caption for a single image path."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return clean_caption(processor.batch_decode(generated, skip_special_tokens=True)[0])


def run_florence_captioning(
    images_dir: str | Path,
    output_csv: str | Path,
    model_id: str = "microsoft/Florence-2-base",
    prompt: str = "<DETAILED_CAPTION>",
    device: str | None = None,
) -> pd.DataFrame:
    """Run captioning over a folder of images and save CSV output."""
    paths = get_image_paths(images_dir)
    if not paths:
        raise ValueError(f"No images found in {images_dir}")

    processor, model, resolved_device = load_florence_model(model_id=model_id, device=device)

    rows: list[dict[str, str]] = []
    for image_path in tqdm(paths, desc="Captioning images"):
        caption = generate_caption(
            image_path=image_path,
            processor=processor,
            model=model,
            device=resolved_device,
            prompt=prompt,
        )
        rows.append({"image_path": str(image_path), "image_name": image_path.name, "caption": caption})

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
