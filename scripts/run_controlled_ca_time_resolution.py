#!/usr/bin/env python3
"""Run official controlled CA time/resolution screens on one VOC-Sim class."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_streaming_voc_ablations import read_rows  # noqa: E402
from src.attention_capture.live import generate_attention_sample  # noqa: E402
from src.segmentation.ablation_runner import target_token_positions  # noqa: E402
from src.segmentation.controlled_ca import (  # noqa: E402
    build_time_resolution_maps,
    configuration_names,
    resolution_configurations,
    time_windows,
)
from src.segmentation.controlled_cv import ShardWriter, read_folds, score_maps  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/"
            "ovam_voc_sim.csv"
        ),
    )
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/controlled_attention_cv/ca_time_resolution"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--model-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    folds = read_folds(args.folds)
    windows = time_windows()
    resolutions = resolution_configurations()
    names = configuration_names(windows, resolutions)
    settings = {
        "package": "controlled_ca_time_resolution_v1",
        "protocol": "v1.2",
        "manifest": str(args.manifest),
        "fold_file": str(args.folds),
        "model": args.model,
        "inference_steps": 30,
        "recorded_calls": "0-30 inclusive",
        "guidance_scale": 7.5,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "cross_attention": "conditional CFG query and key; padded 77-token softmax",
        "target_tokens": "all tokenizer pieces of complete class name, averaged",
        "module_weighting": "equal per module-call observation",
        "working_size": 64,
        "normalization": "minmax",
        "thresholds": [0.35, 0.45, 0.55],
        "selection_threshold": 0.45,
        "self_attention": "none",
        "postprocessing": "none",
        "time_windows": {name: list(values) for name, values in windows.items()},
        "resolution_sets": {name: list(values) for name, values in resolutions.items()},
    }
    shard_path = args.output_dir / "shards" / (
        f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    )
    writer = ShardWriter(shard_path, names, folds, settings)

    dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    expected = len(rows)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        print(
            f"[{classname}] {row['image_id']} ({len(writer.completed) + 1}/{expected})",
            flush=True,
        )
        sample = generate_attention_sample(
            pipe,
            row,
            args.manifest.parent,
            args.storage_dtype,
            cross_map_kind="conditional",
        )
        try:
            positions = target_token_positions(sample, pipe.tokenizer)
            observed_names, maps = build_time_resolution_maps(
                sample, positions, windows, resolutions, working_size=64, branch=1
            )
            if observed_names != names:
                raise RuntimeError("Configuration order differs from the result writer")
            counts, ap = score_maps(maps, sample.target_mask, normalization="minmax")
            writer.add(row["image_id"], classname, counts, ap)
            del maps, counts, ap
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    if len(writer.completed) != expected:
        raise RuntimeError(f"Shard has {len(writer.completed)} of {expected} expected images")
    writer.finish()
    print(f"Completed {expected} controlled CA samples for {classname}", flush=True)


if __name__ == "__main__":
    main()
