"""Evaluate U-Net cross-attention modules individually and by region."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
    run_cross_attention_layer_ablation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_cross_attention_layers"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--normalization", choices=("minmax", "max", "none"), default="minmax")
    parser.add_argument("--branch", type=int, choices=(0, 1), default=1)
    parser.add_argument("--working-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def layer_key(module_name: str) -> str:
    return "layer__" + re.sub(r"[^a-zA-Z0-9]+", "_", module_name).strip("_")


def build_configurations(dataset: CapturedAttentionDataset) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    first = dataset.samples[0]
    resolution_by_module = {
        name: resolution
        for resolution, names in first.modules_by_resolution.items()
        for name in names
    }
    modules = tuple(sorted(resolution_by_module))
    down = tuple(name for name in modules if name.startswith("down_blocks."))
    mid = tuple(name for name in modules if name.startswith("mid_block."))
    up = tuple(name for name in modules if name.startswith("up_blocks."))
    configurations = {
        "all_layers": modules,
        "previous_baseline_r16_r32_r64": tuple(
            name for name in modules if resolution_by_module[name] in (16, 32, 64)
        ),
        "down_only": down,
        "mid_only": mid,
        "up_only": up,
        "down_mid": down + mid,
        "mid_up": mid + up,
        "down_up": down + up,
    }
    configurations.update({layer_key(name): (name,) for name in modules})
    return configurations, resolution_by_module


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(
    path: Path,
    results: dict[str, dict],
    configurations: dict[str, tuple[str, ...]],
    resolution_by_module: dict[str, int],
) -> None:
    classes = sorted({name for result in results.values() for name in result["per_class"]})
    fields = ["configuration", "module_count", "modules", "resolutions", "miou", "mean_image_iou", "images"]
    fields.extend(f"iou_{name.replace(' ', '_')}" for name in classes)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for configuration, result in results.items():
            modules = configurations[configuration]
            row = {
                "configuration": configuration,
                "module_count": len(modules),
                "modules": ";".join(modules),
                "resolutions": ";".join(str(resolution_by_module[name]) for name in modules),
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

    configurations, resolution_by_module = build_configurations(dataset)
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
    results = run_cross_attention_layer_ablation(
        dataset, tokenizer, configurations, settings
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
        "timesteps": "all captured U-Net invocations",
        "configurations": configurations,
        "resolution_by_module": resolution_by_module,
        "primary_metric": "macro mean of accumulated foreground IoU across target classes",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
    }
    atomic_json(args.output_dir / "results.json", {"settings": metadata, "results": results})
    write_csv(args.output_dir / "results.csv", results, configurations, resolution_by_module)


if __name__ == "__main__":
    main()
