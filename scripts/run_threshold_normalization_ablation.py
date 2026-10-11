"""Calibrate mask normalization and threshold for selected layer configurations."""

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
    run_threshold_normalization_ablation,
)


THRESHOLDS = tuple(round(value / 100, 2) for value in range(10, 91, 5))
NORMALIZATIONS = ("minmax", "max", "percentile_01_99")
BEST_12_INDICES = (0, 2, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_threshold_normalization"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--branch", type=int, choices=(0, 1), default=1)
    parser.add_argument("--working-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_layer_configurations(dataset: CapturedAttentionDataset):
    modules = tuple(
        sorted(
            name
            for names in dataset.samples[0].modules_by_resolution.values()
            for name in names
        )
    )
    if len(modules) != 16:
        raise ValueError(f"Expected 16 cross-attention modules, found {len(modules)}")
    configurations = {
        "best_12_layers": tuple(modules[index] for index in BEST_12_INDICES),
        "all_16_layers": modules,
        "previous_15_layers": tuple(module for index, module in enumerate(modules) if index != 6),
    }
    return configurations, modules


def write_csv(path: Path, results: dict[str, dict]) -> None:
    classes = sorted({name for result in results.values() for name in result["per_class"]})
    fields = ["configuration", "layer_configuration", "normalization", "threshold", "miou", "mean_image_iou", "images"]
    fields.extend(f"iou_{name.replace(' ', '_')}" for name in classes)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for configuration, result in results.items():
            layer_configuration, normalization, threshold_name = configuration.split("__")
            row = {
                "configuration": configuration,
                "layer_configuration": layer_configuration,
                "normalization": normalization,
                "threshold": float(threshold_name[1:]),
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
    layer_configurations, modules = build_layer_configurations(dataset)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", local_files_only=True
    )
    settings = SegmentationSettings(branch=args.branch, working_size=args.working_size)
    started_at = datetime.now(timezone.utc).isoformat()
    results = run_threshold_normalization_ablation(
        dataset,
        tokenizer,
        layer_configurations,
        THRESHOLDS,
        NORMALIZATIONS,
        settings,
    )
    finished_at = datetime.now(timezone.utc).isoformat()
    ranked = sorted(results, key=lambda name: results[name]["miou"], reverse=True)
    metadata = {
        "capture_dir": str(args.capture_dir),
        "samples": len(dataset),
        "classes": list(dataset.classes),
        "model": args.model,
        "branch": args.branch,
        "working_size": args.working_size,
        "thresholds": THRESHOLDS,
        "normalizations": NORMALIZATIONS,
        "percentile_definition": "clip to 1st and 99th percentiles, then scale to [0, 1]",
        "layer_index": {str(index): name for index, name in enumerate(modules)},
        "layer_configurations": layer_configurations,
        "configuration_count": len(results),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }
    payload = {
        "settings": metadata,
        "best_overall": {"configuration": ranked[0], **results[ranked[0]]},
        "best_per_layer_configuration": {
            layer_name: {
                "configuration": next(
                    name for name in ranked if name.startswith(f"{layer_name}__")
                ),
                **results[next(name for name in ranked if name.startswith(f"{layer_name}__"))],
            }
            for layer_name in layer_configurations
        },
        "results": results,
    }
    atomic_json(args.output_dir / "results.json", payload)
    write_csv(args.output_dir / "results.csv", results)


if __name__ == "__main__":
    main()
