"""Generate one diffusion sample and expose a compact in-memory attention source."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from src.segmentation.attention_aggregation import find_token_subsequence
from src.segmentation.in_memory_dataset import InMemoryAttentionSample

from .memory import CompactAttentionRecorder
from .recorder import RecordingAttnProcessor


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _install(pipe: StableDiffusionPipeline, recorder: CompactAttentionRecorder) -> None:
    processors = {}
    for processor_name, installed in pipe.unet.attn_processors.items():
        original = installed.original if isinstance(installed, RecordingAttnProcessor) else installed
        module_name = processor_name.removesuffix(".processor")
        kind = "self" if module_name.endswith(".attn1") else "cross" if module_name.endswith(".attn2") else None
        if kind is None:
            raise RuntimeError(f"Cannot classify attention processor: {processor_name}")
        module = pipe.unet.get_submodule(module_name)
        recorder.register_module(
            module_name, kind, int(module.heads), float(module.scale), original.__class__.__name__
        )
        processors[processor_name] = RecordingAttnProcessor(original, recorder, module_name, kind)
    pipe.unet.set_attn_processor(processors)


@torch.inference_mode()
def generate_attention_sample(
    pipe: StableDiffusionPipeline,
    row: dict[str, str],
    manifest_dir: Path,
    storage_dtype: str = "float16",
    cross_map_kind: str = "ovam",
) -> InMemoryAttentionSample:
    """Run one complete denoising trajectory without saving an image or tensors."""
    if cross_map_kind not in ("ovam", "conditional"):
        raise ValueError(f"Unknown cross-map kind: {cross_map_kind}")
    prompt = row["prompt"]
    separate_attribution_prompt = row.get("attribution_prompt", "").strip()
    attribution_prompt = separate_attribution_prompt or prompt
    classname = row["classname"]
    seed = int(row["seed"])
    steps = int(row["num_inferece_steps"])
    annotation_value = row.get("annotation", "")
    annotation_path = manifest_dir / annotation_value
    if not annotation_value or not annotation_path.is_file():
        raise FileNotFoundError(f"No annotation for {row['image_id']}: {annotation_path}")

    tokenized = pipe.tokenizer(
        attribution_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    class_ids = pipe.tokenizer(classname, add_special_tokens=False)["input_ids"]
    target_positions = find_token_subsequence(tokenized.input_ids[0].tolist(), class_ids)
    short_tokens = pipe.tokenizer(attribution_prompt, padding=False, return_tensors="pt")
    prompt_token_count = int(short_tokens.input_ids.shape[1])
    attribution_hidden_states = None
    if separate_attribution_prompt:
        attribution_inputs = {
            key: value.to(pipe.device) for key, value in short_tokens.items()
        }
        attribution_hidden_states = pipe.text_encoder(**attribution_inputs)[0]

    # The released implementations use different indexing in these two cases:
    # VOC selects len(prompt.split()), while COCO's separately encoded class
    # selects -2 (the final token before EOT).
    ovam_target_position = (
        prompt_token_count - 2
        if separate_attribution_prompt
        else len(prompt.split())
    )

    recorder = CompactAttentionRecorder(
        target_positions=target_positions,
        prompt_token_count=prompt_token_count,
        ovam_target_position=ovam_target_position,
        attribution_hidden_states=attribution_hidden_states,
        storage_dtype=storage_dtype,
    )
    _install(pipe, recorder)
    _seed_everything(seed)

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
    )
    latent_shape = (
        1,
        pipe.unet.config.in_channels,
        512 // pipe.vae_scale_factor,
        512 // pipe.vae_scale_factor,
    )
    initial_noise = torch.randn(latent_shape, device=pipe.device, dtype=prompt_embeds.dtype)
    observed_timesteps: list[int] = []

    def on_step_end(
        _pipe: Any, step: int, timestep: torch.Tensor, callback_kwargs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        timestep_value = int(timestep.item())
        observed_timesteps.append(timestep_value)
        recorder.flush_step(step, timestep_value, callback_kwargs["latents"])
        return callback_kwargs

    result = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        latents=initial_noise,
        height=512,
        width=512,
        num_inference_steps=steps,
        guidance_scale=7.5,
        output_type="pil",
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    recorder.finalize()
    generated_image = np.asarray(result.images[0].convert("RGB")).copy()
    # The recorder is retained by the installed attention processors. Drop the
    # separate attribution encoding as soon as its final cross-attention map is
    # captured so it cannot accumulate across streaming samples.
    recorder.attribution_hidden_states = None
    del result, initial_noise, prompt_embeds, negative_prompt_embeds, attribution_hidden_states

    cross: dict[int, list[str]] = {}
    self_attention: dict[int, list[str]] = {}
    for name, metadata in recorder.module_metadata.items():
        resolution = int(metadata["spatial_height"])
        destination = self_attention if metadata["kind"] == "self" else cross
        destination.setdefault(resolution, []).append(name)
    pixels = np.asarray(Image.open(annotation_path).convert("RGB"))
    target = np.any(pixels != 0, axis=-1)
    return InMemoryAttentionSample(
        sample_id=row["image_id"],
        classname=classname,
        prompt=prompt,
        token_ids=tuple(tokenized.input_ids[0].tolist()),
        attention_mask=tuple(tokenized.attention_mask[0].tolist()),
        timesteps=tuple(observed_timesteps),
        modules_by_resolution={key: tuple(sorted(value)) for key, value in cross.items()},
        self_modules_by_resolution={key: tuple(sorted(value)) for key, value in self_attention.items()},
        cross_map_key="ovam_target_map" if cross_map_kind == "ovam" else "target_maps",
        target_mask=target,
        generated_image=generated_image,
        steps=recorder.steps,
    )

