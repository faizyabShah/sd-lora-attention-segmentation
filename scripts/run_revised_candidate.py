#!/usr/bin/env python3
"""Evaluate one parameterized revised-system candidate on a class shard."""

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
from src.segmentation.best_method import frozen_revised_candidate


METHOD_KEY = "ca_steps_10_30_all_sa_heads"
METHOD_LABELS = {
    METHOD_KEY: "CA calls 10--30 and all SA heads",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MANIFESTS), default="voc_sim")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/segmentation/ca10_30_all_sa_heads"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-step-start", type=int, default=10)
    parser.add_argument("--method-key", default=METHOD_KEY)
    parser.add_argument(
        "--method-label",
        default=METHOD_LABELS[METHOD_KEY],
        help="Human-readable label written to the shard results.",
    )
    parser.add_argument(
        "--sa-heads",
        default="all",
        help="'all' or comma-separated zero-based SA head indices.",
    )
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


def parse_sa_heads(value: str) -> tuple[int, ...] | None:
    if value.strip().lower() == "all":
        return None
    heads = tuple(int(item) for item in value.split(","))
    if not heads or len(set(heads)) != len(heads) or min(heads) < 0 or max(heads) >= 8:
        raise ValueError(f"Invalid SA head selection: {value}")
    return heads


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sa_heads = parse_sa_heads(args.sa_heads)
    method_labels = {args.method_key: args.method_label}
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
        "suite": "revised_parameterized_candidate_v1",
        "dataset": args.dataset,
        "manifest": str(manifest),
        "model": args.model,
        "classname": classname,
        "class_shard": args.class_shard,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "fixed_stage1_results": str(args.fixed_stage1_results),
        "ca_steps": [args.ca_step_start, 30],
        "sa_heads": "all" if sa_heads is None else list(sa_heads),
        "fixed_settings": (
            "all 16 CA layers; frozen CA head filtering; one final-64 SA pass; "
            "SA calls 15--30; percentile 1--99; threshold 0.45; 1% cleanup"
        ),
    }
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {
            "signature": signature,
            "completed": [],
            "metrics": {args.method_key: empty_count()},
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
        print(
            f"[{args.dataset}/{classname}] {row['image_id']} "
            f"({len(completed) + 1}/{len(rows)})",
            flush=True,
        )
        sample = generate_attention_sample(
            pipe, row, manifest.parent, args.storage_dtype, cross_map_kind="ovam"
        )
        try:
            prediction = frozen_revised_candidate(
                sample,
                pipe.tokenizer,
                rankings,
                args.device,
                ca_step_start=args.ca_step_start,
                sa_heads=sa_heads,
            )
            update_count(
                state["metrics"][args.method_key], prediction, sample.load_target_mask()
            )
            state["completed"].append(row["image_id"])
            completed.add(row["image_id"])
            atomic_torch_save(checkpoint, state)
            write_results(shard_dir / "results.json", state, len(rows), method_labels)
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {len(completed)} samples for {args.dataset}/{classname}")


if __name__ == "__main__":
    main()
