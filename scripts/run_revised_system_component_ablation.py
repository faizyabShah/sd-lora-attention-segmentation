#!/usr/bin/env python3
"""Run leave-one-change ablations around the revised system on one class."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_component_ablation import (
    MANIFESTS,
    atomic_torch_save,
    empty_count,
    read_rows,
    update_count,
    write_results,
)
from src.attention_capture.live import generate_attention_sample
from src.segmentation.best_method import (
    REVISED_COMPONENT_ABLATION_LABELS,
    frozen_revised_component_ablations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MANIFESTS), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/segmentation/revised_system_component_ablation"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--model-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--fixed-stage1-results",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_stage1_suite/results.json"),
    )
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = args.manifest or MANIFESTS[args.dataset]
    classname, rows = read_rows(manifest, args.class_shard, args.max_samples)
    shard_dir = (
        args.output_root
        / args.dataset
        / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = shard_dir / "checkpoint.pt"
    rankings = json.loads(args.fixed_stage1_results.read_text(encoding="utf-8"))["settings"][
        "head_rankings"
    ]
    signature = {
        "suite": "revised_system_component_leave_one_change_v1",
        "dataset": args.dataset,
        "manifest": str(manifest),
        "model": args.model,
        "classname": classname,
        "class_shard": args.class_shard,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "fixed_stage1_results": str(args.fixed_stage1_results),
        "revised_baseline": "one SA pass; all 16 CA layers; remaining choices unchanged",
        "selection_source": "remaining choices frozen from VOC-Sim",
    }
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {
            "signature": signature,
            "completed": [],
            "metrics": {key: empty_count() for key in REVISED_COMPONENT_ABLATION_LABELS},
        }
    completed = set(state["completed"])

    dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    for row in rows:
        if row["image_id"] in completed:
            continue
        print(f"[{args.dataset}/{classname}] {row['image_id']} ({len(completed)+1}/{len(rows)})", flush=True)
        sample = generate_attention_sample(
            pipe, row, manifest.parent, args.storage_dtype, cross_map_kind="ovam"
        )
        try:
            predictions = frozen_revised_component_ablations(
                sample, pipe.tokenizer, rankings, args.device
            )
            target = sample.load_target_mask()
            for key, prediction in predictions.items():
                update_count(state["metrics"][key], prediction, target)
            state["completed"].append(row["image_id"])
            completed.add(row["image_id"])
            atomic_torch_save(checkpoint, state)
            write_results(
                shard_dir / "results.json",
                state,
                len(rows),
                REVISED_COMPONENT_ABLATION_LABELS,
            )
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {len(completed)} samples for {args.dataset}/{classname}")


if __name__ == "__main__":
    main()
