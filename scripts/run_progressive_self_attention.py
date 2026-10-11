"""Run one progressive multi-resolution self-attention refinement sequence."""

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

from src.segmentation import CapturedAttentionDataset
from src.segmentation.self_attention_refinement import run_progressive_self_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, help="Comma-separated resolutions, e.g. 8,16,32,64")
    parser.add_argument("--capture-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = tuple(int(value) for value in args.sequence.replace("-", ",").split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CapturedAttentionDataset(args.capture_dir, require_annotation=True)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: args.max_samples]
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer", local_files_only=True)
    started_at = datetime.now(timezone.utc).isoformat()
    results, details = run_progressive_self_attention(dataset, tokenizer, sequence, args.device)
    ranked = sorted(results, key=lambda name: results[name]["miou"], reverse=True)
    payload = {
        "settings": {
            "capture_dir": str(args.capture_dir), "samples": len(dataset), "classes": list(dataset.classes),
            "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(), **details,
        },
        "best": ranked[0], "top_20": ranked[:20], "results": results,
    }
    temporary = args.output_dir / "results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "results.json")
    classes = sorted(dataset.classes)
    fields = ["configuration", "miou", "mean_image_iou", "images"] + [f"iou_{name}" for name in classes]
    temporary_csv = args.output_dir / "results.csv.tmp"
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name, value in results.items():
            row = {"configuration": name, "miou": value["miou"], "mean_image_iou": value["mean_image_iou"], "images": value["images"]}
            row.update({f"iou_{classname}": value["per_class"][classname]["iou"] for classname in classes})
            writer.writerow(row)
    os.replace(temporary_csv, args.output_dir / "results.csv")


if __name__ == "__main__":
    main()
