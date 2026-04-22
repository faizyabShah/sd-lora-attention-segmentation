"""Image generation entrypoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from src.generation.pipeline import PromptTemplate


def load_lora_pipeline(base_model: str, lora_path: str, device: str | None = None) -> tuple[StableDiffusionPipeline, str]:
    """Load Stable Diffusion pipeline and attach LoRA weights."""
    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    resolved_device = requested_device
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe = pipe.to(resolved_device)
    pipe.safety_checker = None
    pipe.load_lora_weights(lora_path)
    return pipe, resolved_device


def generate_images(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    output_dir: str | Path,
    negative_prompt: str,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    device: str,
) -> pd.DataFrame:
    """Generate images from prompts and save outputs to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    image_index = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating images"):
        prompt_batch = prompts[i : i + batch_size]
        generators = [
            torch.Generator(device=device).manual_seed(42 + i + j)
            for j in range(len(prompt_batch))
        ]
        result = pipe(
            prompt_batch,
            negative_prompt=[negative_prompt] * len(prompt_batch),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generators,
        )

        for j, image in enumerate(result.images):
            name = f"img_{image_index:05d}.png"
            path = output_dir / name
            image.save(path)
            rows.append({"image_name": name, "image_path": str(path), "prompt": prompt_batch[j]})
            image_index += 1

    return pd.DataFrame(rows)


def generate(config_path: str = "configs/generation.yaml") -> pd.DataFrame:
    """Run generation from config and return output dataframe."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = config["model"]
    inf_cfg = config["inference"]
    prompt_cfg = config.get("prompts", {})

    prompt_template = PromptTemplate(prompt_cfg.get("components"))
    prompts = prompt_template.generate_many(int(inf_cfg["num_images"]))

    pipe, device = load_lora_pipeline(
        base_model=model_cfg["base_model"],
        lora_path=model_cfg["lora_weights"],
        device=inf_cfg.get("device"),
    )

    df = generate_images(
        pipe=pipe,
        prompts=prompts,
        output_dir=inf_cfg["output_dir"],
        negative_prompt=config.get("negative_prompt", ""),
        batch_size=int(inf_cfg.get("batch_size", 4)),
        num_inference_steps=int(inf_cfg.get("steps", 30)),
        guidance_scale=float(inf_cfg.get("guidance_scale", 7.5)),
        device=device,
    )

    csv_path = Path(inf_cfg.get("output_csv", "outputs/images/generated_dataset.csv"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
