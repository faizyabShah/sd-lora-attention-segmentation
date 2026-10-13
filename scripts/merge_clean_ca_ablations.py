#!/usr/bin/env python3
"""Merge the 20 controlled VOC-Sim CA-only class shards."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_streaming_voc_ablations import merge_metrics
from scripts.run_exhaustive_layer_search import write_results as write_exhaustive_results
from scripts.run_streaming_voc_ablations import finalized
from src.segmentation.exhaustive_layer_search import ExhaustiveLayerResults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_clean_ca_ablations"),
    )
    parser.add_argument("--expected-shards", type=int, default=20)
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    shards = sorted(
        path
        for path in args.input_dir.glob("class_*")
        if path.is_dir() and (path / "COMPLETE").is_file()
    )
    if len(shards) != args.expected_shards:
        raise RuntimeError(
            f"Expected {args.expected_shards} complete shards, found {len(shards)}"
        )

    metrics: dict = {}
    parts: list[tuple[str, dict]] = []
    signatures: list[dict] = []
    total_images = 0
    for shard in shards:
        state = torch.load(shard / "checkpoint.pt", map_location="cpu")
        signatures.append(state["signature"])
        classname = state["signature"]["classname"]
        if len(state["completed"]) == 0:
            raise RuntimeError(f"Empty shard: {shard}")
        total_images += len(state["completed"])
        merge_metrics(metrics, state["metrics"])
        if state["exhaustive"] is None:
            raise RuntimeError(f"Missing exhaustive layer results: {shard}")
        parts.append((classname, state["exhaustive"]))

    comparable_keys = (
        "suite",
        "manifest",
        "model",
        "storage_dtype",
        "model_dtype",
        "token_attribution",
        "branch",
        "normalization",
        "threshold",
        "working_size",
        "postprocessing",
        "self_attention",
        "densecrf",
    )
    common = {key: signatures[0][key] for key in comparable_keys}
    for signature in signatures[1:]:
        if {key: signature[key] for key in comparable_keys} != common:
            raise RuntimeError("Shard settings differ")

    results = finalized(metrics)
    payload = {
        "settings": {**common, "images": total_images, "classes": len(parts)},
        "results": results,
    }
    args.input_dir.mkdir(parents=True, exist_ok=True)
    temporary_gz = args.input_dir / "all_results.json.gz.tmp"
    with gzip.open(temporary_gz, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(temporary_gz, args.input_dir / "all_results.json.gz")

    ordered = sorted(parts)
    module_names = tuple(ordered[0][1]["module_names"])

    def stack(key: str) -> torch.Tensor:
        return torch.stack([part[key] for _, part in ordered], dim=1)

    exhaustive = ExhaustiveLayerResults(
        module_names=module_names,
        classes=tuple(name for name, _ in ordered),
        tp=stack("tp"),
        fp=stack("fp"),
        fn=stack("fn"),
        tn=stack("tn"),
        image_iou_sum=stack("image_iou_sum"),
        image_counts=torch.tensor([int(part["images"]) for _, part in ordered]),
    )
    exhaustive_dir = args.input_dir / "layers_exhaustive"
    exhaustive_dir.mkdir(exist_ok=True)
    write_exhaustive_results(
        exhaustive_dir,
        exhaustive,
        {
            **common,
            "images": total_images,
            "module_count": len(module_names),
            "nonempty_subsets": exhaustive.subset_count,
        },
    )
    exhaustive_summary = json.loads(
        (exhaustive_dir / "summary.json").read_text(encoding="utf-8")
    )

    winners = {}
    for family, configurations in results.items():
        key = max(configurations, key=lambda name: configurations[name]["miou"])
        winners[family] = {
            "configuration": key,
            "miou": configurations[key]["miou"],
            "mean_image_iou": configurations[key]["mean_image_iou"],
        }
    summary = {
        "settings": payload["settings"],
        "best_by_family": winners,
        "named_controls": {
            "time_all": results["time"]["all_00_30"],
            "resolution_all": results["resolution"]["r8_r16_r32_r64"],
            "layers_all": results["layers_structured"]["all_16"],
            "layers_non_middle": results["layers_structured"]["non_middle_15"],
            "heads_all": results["heads"]["heads_0_1_2_3_4_5_6_7"],
        },
        "exhaustive_layers": {
            "all_layers": exhaustive_summary["all_layers"],
            "best": exhaustive_summary["top_100"][0],
            "best_by_selected_layer_count": exhaustive_summary[
                "best_by_selected_layer_count"
            ],
        },
    }
    atomic_json(args.input_dir / "summary.json", summary)

    lines = [
        "# Clean CA-only ablations on full VOC-Sim",
        "",
        f"Images: **{total_images}**; classes: **{len(parts)}**.",
        "",
        "Fixed for every comparison: conditional CA; every tokenizer piece of the "
        "class name; equal averaging; min-max normalization; threshold 0.45; no SA, "
        "DenseCRF, learned weights, or component cleanup.",
        "",
        "| Family | Best configuration | mIoU | Mean image IoU |",
        "|---|---|---:|---:|",
    ]
    for family, value in sorted(winners.items()):
        lines.append(
            f"| {family} | `{value['configuration']}` | {value['miou']:.5f} | "
            f"{value['mean_image_iou']:.5f} |"
        )
    best_layer = exhaustive_summary["top_100"][0]
    lines.extend(
        [
            f"| layers_exhaustive | `{best_layer['subset_hex']}` "
            f"({best_layer['selected_layer_indices']}) | {best_layer['miou']:.5f} | --- |",
            "",
            "All raw regular results are in `all_results.json.gz`; all 65,535 layer "
            "subsets are in `layers_exhaustive/all_subsets.csv.gz`.",
        ]
    )
    (args.input_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Merged {len(shards)} shards and {total_images} images into {args.input_dir}")


if __name__ == "__main__":
    main()
