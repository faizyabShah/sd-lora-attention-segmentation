"""Regenerate VOC-Sim and retain fine-grained UNet attention information."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import diffusers
import numpy as np
import torch
import transformers
from diffusers import StableDiffusionPipeline
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm

# Direct execution sets sys.path to scripts/. Add the repository root so the
# source package works without requiring an editable installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.attention_capture import AttentionRecorder, RecordingAttnProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def read_rows(path: Path, start: int, count: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = rows[start : start + count]
    if len(selected) != count:
        raise ValueError(f"Requested {count} rows from {start}, but only {len(selected)} are available")
    return selected


def install_recorders(pipe: StableDiffusionPipeline, recorder: AttentionRecorder) -> None:
    processors = {}
    for processor_name, installed in pipe.unet.attn_processors.items():
        # A new recorder is used for every image. Unwrap the recorder installed
        # for the preceding image rather than nesting recorder processors.
        original = installed.original if isinstance(installed, RecordingAttnProcessor) else installed
        module_name = processor_name.removesuffix(".processor")
        if module_name.endswith(".attn1"):
            kind = "self"
        elif module_name.endswith(".attn2"):
            kind = "cross"
        else:
            raise RuntimeError(f"Cannot classify attention processor: {processor_name}")
        module = pipe.unet.get_submodule(module_name)
        recorder.register_module(
            module_name,
            kind,
            int(module.heads),
            float(module.scale),
            original.__class__.__name__,
        )
        processors[processor_name] = RecordingAttnProcessor(original, recorder, module_name, kind)
    pipe.unet.set_attn_processor(processors)


def ensure_free_space(path: Path, minimum_gb: float) -> None:
    free = shutil.disk_usage(path).free / 1024**3
    if free < minimum_gb:
        raise RuntimeError(f"Only {free:.1f} GiB free at {path}; minimum is {minimum_gb:.1f} GiB")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_one(
    pipe: StableDiffusionPipeline,
    row: dict[str, str],
    manifest_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = row["image_id"]
    final_dir = output_root / sample_id
    partial_dir = output_root / f".{sample_id}.partial"
    if final_dir.exists() and (final_dir / "COMPLETE").exists() and not args.overwrite:
        return json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
    if final_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Incomplete output exists: {final_dir}; use --overwrite")
        shutil.rmtree(final_dir)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    seed = int(row["seed"])
    steps = int(row["num_inferece_steps"])
    prompt = row["prompt"]
    set_seed(seed)

    tokenized = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
    )
    save_file(
        {
            "conditional": prompt_embeds.detach().cpu().contiguous(),
            "unconditional": negative_prompt_embeds.detach().cpu().contiguous(),
        },
        str(partial_dir / "text_embeddings.safetensors"),
    )

    height = width = 512
    latent_shape = (1, pipe.unet.config.in_channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    initial_noise = torch.randn(latent_shape, device=pipe.device, dtype=prompt_embeds.dtype)
    save_file({"initial_noise": initial_noise.detach().cpu().contiguous()}, str(partial_dir / "initial_noise.safetensors"))

    recorder = AttentionRecorder(partial_dir, args.storage_dtype)
    install_recorders(pipe, recorder)
    observed_timesteps: list[int] = []

    def on_step_end(_pipe: Any, step: int, timestep: torch.Tensor, callback_kwargs: dict[str, torch.Tensor]):
        timestep_value = int(timestep.item())
        observed_timesteps.append(timestep_value)
        recorder.flush_step(step, timestep_value, callback_kwargs["latents"])
        return callback_kwargs

    result = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        latents=initial_noise,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=7.5,
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    result.images[0].save(partial_dir / "generated.png")
    recorder.finalize()

    reference = manifest_dir / row["image"]
    if reference.exists():
        shutil.copy2(reference, partial_dir / "reference.png")
    annotation_value = row.get("annotation", "")
    annotation = manifest_dir / annotation_value if annotation_value else None
    if annotation is not None and annotation.exists():
        shutil.copy2(annotation, partial_dir / "annotation.png")

    generated_path = partial_dir / "generated.png"
    comparison: dict[str, Any] = {}
    if reference.exists():
        reference_pixels = np.asarray(Image.open(reference).convert("RGB"), dtype=np.int16)
        generated_pixels = np.asarray(Image.open(generated_path).convert("RGB"), dtype=np.int16)
        difference = np.abs(reference_pixels - generated_pixels)
        comparison = {
            "reference_sha256": sha256(reference),
            "generated_sha256": sha256(generated_path),
            "exact_pixels": bool(np.array_equal(reference_pixels, generated_pixels)),
            "pixel_mae": float(difference.mean()),
            "pixel_rmse": float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
            "pixel_max_error": int(difference.max()),
        }

    metadata = {
        "image_id": sample_id,
        "classname": row["classname"],
        "prompt": prompt,
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": 7.5,
        "height": height,
        "width": width,
        "model_requested": args.model,
        "model_recorded_by_ovam": row["model"],
        "annotated": row["annotated"].lower() == "true",
        "token_ids": tokenized.input_ids[0].tolist(),
        "tokens": pipe.tokenizer.convert_ids_to_tokens(tokenized.input_ids[0].tolist()),
        "attention_mask": tokenized.attention_mask[0].tolist(),
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": jsonable(dict(pipe.scheduler.config)),
        "pipeline_config": jsonable(dict(pipe.config)),
        "unet_config": jsonable(dict(pipe.unet.config)),
        "vae_config": jsonable(dict(pipe.vae.config)),
        "text_encoder_config": jsonable(pipe.text_encoder.config.to_dict()),
        "model_commit_hash": getattr(pipe.config, "_commit_hash", None),
        "model_dtype": str(pipe.unet.dtype),
        "timesteps": observed_timesteps,
        "cfg_batch_order": ["unconditional", "conditional"],
        "torch_version": torch.__version__,
        "diffusers_version": diffusers.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "storage_dtype": args.storage_dtype,
        "attention_bytes": recorder.bytes_written,
        "reference_comparison": comparison,
    }
    (partial_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (partial_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    partial_dir.rename(final_dir)
    return metadata


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest, args.start_index, args.count)

    pipe = StableDiffusionPipeline.from_pretrained(args.model)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    run_metadata = {
        "manifest": str(args.manifest),
        "start_index": args.start_index,
        "count": args.count,
        "model": args.model,
        "storage_dtype": args.storage_dtype,
    }
    (args.output_dir / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    manifest_output = args.output_dir / "capture_manifest.jsonl"
    completed: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="Capturing VOC-Sim attention"):
        ensure_free_space(args.output_dir, args.min_free_gb)
        metadata = capture_one(pipe, row, args.manifest.parent, args.output_dir, args)
        completed.append(metadata)
        manifest_output.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in completed),
            encoding="utf-8",
        )
        size_gib = sum(p.stat().st_size for p in (args.output_dir / row["image_id"]).rglob("*") if p.is_file()) / 1024**3
        print(f"{row['image_id']}: {size_gib:.2f} GiB")


if __name__ == "__main__":
    main()
