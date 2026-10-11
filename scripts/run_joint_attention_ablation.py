"""Run a compact joint timestep-window and attention-resolution ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers import CLIPTokenizer

from src.segmentation import (
    CapturedAttentionDataset,
    SegmentationSettings,
    run_time_resolution_ablation,
)


TIME_WINDOWS = {
    "all_00_30": range(0, 31),
    "steps_00_04": range(0, 5),
    "steps_05_09": range(5, 10),
    "steps_10_14": range(10, 15),
    "steps_15_20": range(15, 21),
    "steps_21_25": range(21, 26),
    "steps_26_30": range(26, 31),
}

RESOLUTION_CONFIGURATIONS = {
    "r16": (16,),
    "r16_r64": (16, 64),
    "r16_r32_r64": (16, 32, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_joint_time_resolution"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--normalization", choices=("minmax", "max", "none"), default="minmax")
    parser.add_argument("--branch", type=int, choices=(0, 1), default=1)
    parser.add_argument("--working-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, results: dict[str, dict]) -> None:
    classes = sorted({name for result in results.values() for name in result["per_class"]})
    fields = ["configuration", "time_window", "resolutions", "miou", "mean_image_iou", "images"]
    fields.extend(f"iou_{name.replace(' ', '_')}" for name in classes)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for configuration, result in results.items():
            time_window, resolution_name = configuration.split("__", maxsplit=1)
            row = {
                "configuration": configuration,
                "time_window": time_window,
                "resolutions": "+".join(
                    str(value) for value in RESOLUTION_CONFIGURATIONS[resolution_name]
                ),
                "miou": result["miou"],
                "mean_image_iou": result["mean_image_iou"],
                "images": result["images"],
            }
            row.update(
                {
                    f"iou_{name.replace(' ', '_')}": result["per_class"][name]["iou"]
                    for name in classes
                }
            )
            writer.writerow(row)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CapturedAttentionDataset(args.capture_dir, require_annotation=True)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        dataset.samples = dataset.samples[: args.max_samples]

    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", local_files_only=True
    )
    settings = SegmentationSettings(
        threshold=args.threshold,
        normalization=args.normalization,
        branch=args.branch,
        working_size=args.working_size,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    results = run_time_resolution_ablation(
        dataset,
        tokenizer,
        TIME_WINDOWS,
        RESOLUTION_CONFIGURATIONS,
        settings,
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "capture_dir": str(args.capture_dir),
        "samples": len(dataset),
        "classes": list(dataset.classes),
        "model": args.model,
        "threshold": args.threshold,
        "normalization": args.normalization,
        "branch": args.branch,
        "working_size": args.working_size,
        "time_windows": {name: list(values) for name, values in TIME_WINDOWS.items()},
        "resolution_configurations": RESOLUTION_CONFIGURATIONS,
        "primary_metric": "macro mean of accumulated foreground IoU across target classes",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }
    atomic_json(args.output_dir / "results.json", {"settings": metadata, "results": results})
    write_csv(args.output_dir / "results.csv", results)


if __name__ == "__main__":
    main()
