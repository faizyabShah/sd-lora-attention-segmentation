#!/usr/bin/env python3
"""Evaluate paper OVAM and the frozen best VOC method on COCO-Cap."""

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
from src.segmentation.best_method import frozen_best_stages
from src.segmentation.ovam_full import ovam_full_prediction


METHODS = ("ovam", "frozen_best_voc")
COUNT_KEYS = ("tp", "fp", "fn", "tn", "images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/"
            "coco_captions/ovam_coco_cap.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/coco_cap_generalization"),
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
    parser.add_argument(
        "--densecrf-site-packages",
        type=Path,
        default=Path("/home/shafa/miniconda3/envs/ovam_sdxl/lib/python3.10/site-packages"),
    )
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def read_rows(path: Path, shard: int, limit: int | None) -> tuple[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    classes = sorted({row["classname"] for row in rows})
    if len(classes) != 20:
        raise ValueError(f"Expected 20 COCO-Cap classes, found {len(classes)}")
    classname = classes[shard]
    selected = [row for row in rows if row["classname"] == classname]
    return classname, selected[:limit] if limit is not None else selected


def _empty_count() -> dict[str, int | float]:
    return {**{key: 0 for key in COUNT_KEYS}, "image_iou_sum": 0.0}


def update_count(count: dict[str, int | float], prediction: Any, target: np.ndarray) -> None:
    pred = np.asarray(prediction.detach().cpu() if isinstance(prediction, torch.Tensor) else prediction)
    pred = pred.astype(bool, copy=False)
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


def atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def summarized(count: dict[str, int | float]) -> dict[str, int | float]:
    foreground_union = int(count["tp"] + count["fp"] + count["fn"])
    background_union = int(count["tn"] + count["fp"] + count["fn"])
    return {
        **count,
        "iou": count["tp"] / foreground_union if foreground_union else 1.0,
        "background_iou": count["tn"] / background_union if background_union else 1.0,
        "mean_image_iou": count["image_iou_sum"] / count["images"],
    }


def save_results(path: Path, state: dict, signature: dict, expected: int) -> None:
    results = {
        method: summarized(state["metrics"][method]) for method in METHODS
    }
    subgroup_results = {
        method: {
            subgroup: summarized(count)
            for subgroup, count in state["subgroups"][method].items()
            if count["images"]
        }
        for method in METHODS
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "settings": {
                    **signature,
                    "completed": len(state["completed"]),
                    "expected": expected,
                },
                "results": results,
                "caption_subgroups": subgroup_results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    shard_dir = args.output_dir / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = shard_dir / "checkpoint.pt"
    rankings = json.loads(args.fixed_stage1_results.read_text(encoding="utf-8"))["settings"][
        "head_rankings"
    ]
    signature = {
        "suite": "coco_cap_untuned_generalization_v1",
        "manifest": str(args.manifest),
        "model": args.model,
        "classname": classname,
        "class_shard": args.class_shard,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "attribution": "separate lowercase category encoding",
        "ovam_parameters": {"threshold": 0.50, "self_attention_minimum": 0.91},
        "best_method_source": "frozen VOC-Sim selections; no COCO-Cap tuning",
        "fixed_stage1_results": str(args.fixed_stage1_results),
    }
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {
            "signature": signature,
            "completed": [],
            "metrics": {method: _empty_count() for method in METHODS},
            "subgroups": {
                method: {"class_present": _empty_count(), "class_absent": _empty_count()}
                for method in METHODS
            },
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
        print(f"[{classname}] {row['image_id']} ({len(completed) + 1}/{len(rows)})", flush=True)
        sample = generate_attention_sample(
            pipe, row, args.manifest.parent, args.storage_dtype, cross_map_kind="ovam"
        )
        try:
            _, _, _, ovam_prediction = ovam_full_prediction(
                sample,
                args.densecrf_site_packages,
                self_attention_minimum=0.91,
                threshold=0.50,
            )
            best_prediction = frozen_best_stages(
                sample, pipe.tokenizer, rankings, args.device
            )[-1].prediction
            target = sample.load_target_mask()
            subgroup = (
                "class_present"
                if row["word_included"].strip().lower() == "true"
                else "class_absent"
            )
            for method, prediction in (
                ("ovam", ovam_prediction),
                ("frozen_best_voc", best_prediction),
            ):
                update_count(state["metrics"][method], prediction, target)
                update_count(state["subgroups"][method][subgroup], prediction, target)
            state["completed"].append(row["image_id"])
            completed.add(row["image_id"])
            atomic_torch_save(checkpoint, state)
            save_results(shard_dir / "results.json", state, signature, len(rows))
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {len(completed)} annotated COCO-Cap samples for {classname}")


if __name__ == "__main__":
    main()

