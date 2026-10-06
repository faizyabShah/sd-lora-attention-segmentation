"""Run compact offline timestep and resolution segmentation ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers import CLIPTokenizer

from src.segmentation import (
    CapturedAttentionDataset,
    SegmentationSettings,
    run_resolution_ablation,
    run_time_window_ablation,
)


DEFAULT_WINDOWS = {
    "all": range(0, 31),
    "early_00_09": range(0, 10),
    "middle_10_20": range(10, 21),
    "late_21_30": range(21, 31),
}

DEFAULT_RESOLUTIONS = {
    "r16": (16,),
    "r32": (32,),
    "r64": (64,),
    "r16_r32": (16, 32),
    "r16_r64": (16, 64),
    "r32_r64": (32, 64),
    "r16_r32_r64": (16, 32, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/segmentation/voc_sim_100_ablations")
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--normalization", choices=("minmax", "max", "none"), default="minmax")
    parser.add_argument("--branch", type=int, choices=(0, 1), default=1)
    parser.add_argument("--working-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=("time_window", "resolution"),
        default=("time_window", "resolution"),
    )
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, experiment: str, results: dict[str, dict]) -> None:
    classes = sorted({name for result in results.values() for name in result["per_class"]})
    fields = ["experiment", "configuration", "miou", "mean_image_iou", "images"] + [
        f"iou_{name.replace(' ', '_')}" for name in classes
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for configuration, result in results.items():
            row = {
                "experiment": experiment,
                "configuration": configuration,
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
    # Generation already cached this tokenizer. Offline loading avoids network
    # retries on compute nodes and guarantees the same tokenizer files are used.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", local_files_only=True
    )
    settings = SegmentationSettings(
        threshold=args.threshold,
        normalization=args.normalization,
        branch=args.branch,
        working_size=args.working_size,
    )
    run_info = {
        "capture_dir": str(args.capture_dir),
        "samples": len(dataset),
        "classes": list(dataset.classes),
        "model": args.model,
        "threshold": args.threshold,
        "normalization": args.normalization,
        "branch": args.branch,
        "working_size": args.working_size,
        "ablations": list(args.ablations),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "macro mean of accumulated foreground IoU across target classes",
    }
    atomic_json(args.output_dir / "run.json", run_info)

    combined: dict[str, Any] = {"run": run_info, "experiments": {}}
    if "time_window" in args.ablations:
        time_results = run_time_window_ablation(
            dataset,
            tokenizer,
            DEFAULT_WINDOWS,
            resolutions=(16, 32, 64),
            settings=settings,
        )
        payload = {
            "settings": {**run_info, "windows": {k: list(v) for k, v in DEFAULT_WINDOWS.items()}},
            "results": time_results,
        }
        atomic_json(args.output_dir / "time_window.json", payload)
        write_csv(args.output_dir / "time_window.csv", "time_window", time_results)
        combined["experiments"]["time_window"] = payload

    if "resolution" in args.ablations:
        resolution_results = run_resolution_ablation(
            dataset,
            tokenizer,
            DEFAULT_RESOLUTIONS,
            settings=settings,
        )
        payload = {
            "settings": {**run_info, "configurations": DEFAULT_RESOLUTIONS},
            "results": resolution_results,
        }
        atomic_json(args.output_dir / "resolution.json", payload)
        write_csv(args.output_dir / "resolution.csv", "resolution", resolution_results)
        combined["experiments"]["resolution"] = payload

    combined["run"]["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.output_dir / "all_results.json", combined)


if __name__ == "__main__":
    main()
