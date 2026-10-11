"""Run the complete Stage-1 cross-attention ablation suite."""

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

from src.segmentation import CapturedAttentionDataset, SegmentationSettings
from src.segmentation.stage1_suite import run_stage1_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/segmentation/voc_sim_100_stage1_suite")
    )
    parser.add_argument(
        "--exhaustive-summary",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_exhaustive_layer_subsets/summary.json"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, results: dict[str, dict]) -> None:
    classes = sorted(
        {classname for group in results.values() for value in group.values() for classname in value["per_class"]}
    )
    fields = ["experiment", "configuration", "miou", "mean_image_iou", "images"]
    fields.extend(f"iou_{name}" for name in classes)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment, group in results.items():
            for configuration, value in group.items():
                row = {
                    "experiment": experiment,
                    "configuration": configuration,
                    "miou": value["miou"],
                    "mean_image_iou": value["mean_image_iou"],
                    "images": value["images"],
                }
                row.update({f"iou_{name}": value["per_class"][name]["iou"] for name in classes})
                writer.writerow(row)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CapturedAttentionDataset(args.capture_dir, require_annotation=True)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", local_files_only=True)
    started_at = datetime.now(timezone.utc).isoformat()
    results, details = run_stage1_suite(
        dataset,
        tokenizer,
        SegmentationSettings(branch=1, working_size=64),
        args.exhaustive_summary,
    )
    metadata = {
        "capture_dir": str(args.capture_dir),
        "samples": len(dataset),
        "classes": list(dataset.classes),
        "normalization": "percentile_01_99",
        "threshold": 0.45,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    best = {
        group: max(values, key=lambda name: values[name]["miou"])
        for group, values in results.items()
    }
    atomic_json(args.output_dir / "results.json", {"settings": metadata, "best": best, "results": results})
    write_csv(args.output_dir / "results.csv", results)


if __name__ == "__main__":
    main()
