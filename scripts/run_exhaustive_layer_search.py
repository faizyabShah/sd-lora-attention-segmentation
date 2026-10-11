"""Evaluate all nonempty subsets of captured cross-attention layers."""

from __future__ import annotations

import argparse
import csv
import gzip
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
    run_exhaustive_layer_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, default=Path("outputs/attention/voc_sim"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_exhaustive_layer_subsets"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--branch", type=int, choices=(0, 1), default=1)
    parser.add_argument("--working-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def subset_category(mask: int, module_count: int) -> str:
    selected = mask.bit_count()
    if selected == module_count:
        return "all_layers"
    if selected == module_count - 1:
        return "leave_one_out"
    if selected == module_count - 2:
        return "leave_two_out"
    return "other_subset"


def result_record(results, mask: int, class_ious, mious) -> dict:
    selected = [index for index in range(len(results.module_names)) if mask & (1 << index)]
    omitted = [index for index in range(len(results.module_names)) if not mask & (1 << index)]
    record = {
        "subset_mask": mask,
        "subset_hex": f"0x{mask:04x}",
        "category": subset_category(mask, len(results.module_names)),
        "selected_count": len(selected),
        "selected_layer_indices": selected,
        "omitted_layer_indices": omitted,
        "miou": float(mious[mask - 1]),
    }
    for class_index, classname in enumerate(results.classes):
        record[f"iou_{classname}"] = float(class_ious[mask, class_index])
        record[f"tp_{classname}"] = int(results.tp[mask, class_index])
        record[f"fp_{classname}"] = int(results.fp[mask, class_index])
        record[f"fn_{classname}"] = int(results.fn[mask, class_index])
        record[f"tn_{classname}"] = int(results.tn[mask, class_index])
    return record


def write_results(output_dir: Path, results, metadata: dict) -> None:
    class_ious = results.per_class_iou()
    mious = results.miou()
    ranked_masks = (mious.argsort(descending=True) + 1).tolist()
    module_count = len(results.module_names)
    all_mask = (1 << module_count) - 1
    leave_one_masks = [all_mask ^ (1 << index) for index in range(module_count)]
    leave_two_masks = [
        all_mask ^ (1 << first) ^ (1 << second)
        for first in range(module_count)
        for second in range(first + 1, module_count)
    ]
    best_by_size = {}
    for size in range(1, module_count + 1):
        candidates = [mask for mask in ranked_masks if mask.bit_count() == size]
        best_by_size[str(size)] = result_record(results, candidates[0], class_ious, mious)

    summary = {
        "settings": metadata,
        "layer_index": {str(i): name for i, name in enumerate(results.module_names)},
        "all_layers": result_record(results, all_mask, class_ious, mious),
        "leave_one_out": [result_record(results, mask, class_ious, mious) for mask in leave_one_masks],
        "leave_two_out": [result_record(results, mask, class_ious, mious) for mask in leave_two_masks],
        "best_by_selected_layer_count": best_by_size,
        "top_100": [result_record(results, mask, class_ious, mious) for mask in ranked_masks[:100]],
    }
    atomic_json(output_dir / "summary.json", summary)

    path = output_dir / "all_subsets.csv.gz"
    temporary = output_dir / "all_subsets.csv.gz.tmp"
    fields = [
        "subset_mask", "subset_hex", "category", "selected_count",
        "selected_layer_indices", "omitted_layer_indices", "miou",
    ]
    for classname in results.classes:
        fields.extend(
            [f"iou_{classname}", f"tp_{classname}", f"fp_{classname}", f"fn_{classname}", f"tn_{classname}"]
        )
    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mask in range(1, all_mask + 1):
            record = result_record(results, mask, class_ious, mious)
            record["selected_layer_indices"] = ";".join(map(str, record["selected_layer_indices"]))
            record["omitted_layer_indices"] = ";".join(map(str, record["omitted_layer_indices"]))
            writer.writerow(record)
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
        normalization="minmax",
        branch=args.branch,
        working_size=args.working_size,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    results = run_exhaustive_layer_search(
        dataset, tokenizer, settings, device=args.device, batch_size=args.batch_size
    )
    metadata = {
        "capture_dir": str(args.capture_dir),
        "samples": len(dataset),
        "classes": list(dataset.classes),
        "model": args.model,
        "threshold": args.threshold,
        "normalization": "minmax",
        "branch": args.branch,
        "working_size": args.working_size,
        "device": args.device,
        "batch_size": args.batch_size,
        "module_count": len(results.module_names),
        "nonempty_subsets": results.subset_count,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_results(args.output_dir, results, metadata)


if __name__ == "__main__":
    main()
