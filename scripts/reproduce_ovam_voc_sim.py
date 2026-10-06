"""Regenerate a small VOC-Sim sample and compare it with OVAM images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import diffusers
import numpy as np
import torch
import transformers
from diffusers import StableDiffusionPipeline
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"
        ),
    )
    parser.add_argument(
        "--model",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="SD 1.5 model repository or local path.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ovam_reproduction"))
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_ovam_seed(seed: int) -> None:
    """Match ovam.utils.set_seed used by the paper's generation scripts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_different_classes(manifest: Path, count: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["classname"] in seen or not row["image"]:
                continue
            selected.append(row)
            seen.add(row["classname"])
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"Could only select {len(selected)} distinct classes")
    return selected


def compare_images(reference: Path, generated: Path) -> dict[str, object]:
    ref = np.asarray(Image.open(reference).convert("RGB"), dtype=np.int16)
    gen = np.asarray(Image.open(generated).convert("RGB"), dtype=np.int16)
    if ref.shape != gen.shape:
        return {"same_shape": False, "reference_shape": list(ref.shape), "generated_shape": list(gen.shape)}
    delta = np.abs(ref - gen)
    return {
        "same_shape": True,
        "exact_pixels": bool(np.array_equal(ref, gen)),
        "different_values": int(np.count_nonzero(delta)),
        "different_pixels": int(np.count_nonzero(np.any(delta != 0, axis=2))),
        "mae": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "max_error": int(delta.max()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = select_different_classes(args.manifest, args.count)

    print(f"torch={torch.__version__}")
    print(f"diffusers={diffusers.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")

    # OVAM loaded the pipeline without a dtype override and used its defaults.
    pipe = StableDiffusionPipeline.from_pretrained(args.model)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"scheduler={pipe.scheduler.__class__.__name__}")
    print(f"scheduler_config={dict(pipe.scheduler.config)}")

    results: list[dict[str, object]] = []
    for row in samples:
        seed = int(row["seed"])
        steps = int(row["num_inferece_steps"])
        set_ovam_seed(seed)
        image = pipe(row["prompt"], num_inference_steps=steps).images[0]

        generated = args.output_dir / Path(row["image"]).name
        image.save(generated)
        reference = args.manifest.parent / row["image"]
        comparison = compare_images(reference, generated)
        result = {
            "image_id": row["image_id"],
            "classname": row["classname"],
            "prompt": row["prompt"],
            "seed": seed,
            "steps": steps,
            "reference": str(reference),
            "generated": str(generated),
            "reference_sha256": sha256(reference),
            "generated_sha256": sha256(generated),
            "exact_file": sha256(reference) == sha256(generated),
            **comparison,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    report = args.output_dir / "comparison.json"
    report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"report={report}")
    print(f"exact_pixels={sum(bool(r.get('exact_pixels')) for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
