#!/usr/bin/env python3
"""Run final-system component isolation on one dataset/class shard."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention_capture.live import generate_attention_sample
from src.segmentation.best_method import (
    COMPONENT_ABLATION_LABELS,
    frozen_component_ablations,
)


MANIFESTS = {
    "voc_sim": Path(
        "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"
    ),
    "coco_cap": Path(
        "datasets/ovam_experiment_with_dataset/evaluation/ovam/"
        "coco_captions/ovam_coco_cap.csv"
    ),
}
COUNT_KEYS = ("tp", "fp", "fn", "tn", "images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MANIFESTS), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/segmentation/final_component_ablation"),
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


def read_rows(path: Path, shard: int, limit: int | None) -> tuple[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("annotated", "true").lower() == "true"]
    classes = sorted({row["classname"] for row in rows})
    if len(classes) != 20:
        raise ValueError(f"Expected 20 classes in {path}, found {len(classes)}")
    classname = classes[shard]
    selected = [row for row in rows if row["classname"] == classname]
    return classname, selected[:limit] if limit is not None else selected


def empty_count() -> dict[str, int | float]:
    return {**{key: 0 for key in COUNT_KEYS}, "image_iou_sum": 0.0}


def update_count(count: dict[str, int | float], prediction: Any, target: np.ndarray) -> None:
    pred = np.asarray(
        prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else prediction
    ).astype(bool, copy=False)
    truth = target.astype(bool, copy=False)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction {pred.shape} and target {truth.shape} differ")
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    count["tp"] += tp
    count["fp"] += fp
    count["fn"] += fn
    count["tn"] += tn
    count["images"] += 1
    union = tp + fp + fn
    count["image_iou_sum"] += tp / union if union else 1.0


def summarize(count: dict[str, int | float]) -> dict[str, int | float]:
    fg_union = int(count["tp"] + count["fp"] + count["fn"])
    bg_union = int(count["tn"] + count["fp"] + count["fn"])
    return {
        **count,
        "iou": count["tp"] / fg_union if fg_union else 1.0,
        "background_iou": count["tn"] / bg_union if bg_union else 1.0,
        "mean_image_iou": count["image_iou_sum"] / count["images"],
    }


def atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_results(
    path: Path,
    state: dict,
    expected: int,
    method_labels: dict[str, str] = COMPONENT_ABLATION_LABELS,
) -> None:
    payload = {
        "settings": {
            **state["signature"],
            "completed": len(state["completed"]),
            "expected": expected,
            "component_labels": method_labels,
        },
        "results": {key: summarize(value) for key, value in state["metrics"].items()},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
        "suite": "final_component_leave_one_change_v1",
        "dataset": args.dataset,
        "manifest": str(manifest),
        "model": args.model,
        "classname": classname,
        "class_shard": args.class_shard,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "fixed_stage1_results": str(args.fixed_stage1_results),
        "selection_source": "all method choices frozen from VOC-Sim",
    }
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {
            "signature": signature,
            "completed": [],
            "metrics": {key: empty_count() for key in COMPONENT_ABLATION_LABELS},
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
            predictions = frozen_component_ablations(
                sample, pipe.tokenizer, rankings, args.device
            )
            target = sample.load_target_mask()
            for key, prediction in predictions.items():
                update_count(state["metrics"][key], prediction, target)
            state["completed"].append(row["image_id"])
            completed.add(row["image_id"])
            atomic_torch_save(checkpoint, state)
            write_results(shard_dir / "results.json", state, len(rows))
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {len(completed)} samples for {args.dataset}/{classname}")


if __name__ == "__main__":
    main()
