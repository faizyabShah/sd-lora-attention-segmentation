#!/usr/bin/env python3
"""Merge the combined one-pass/no-CA-layer-filter experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path


OUTPUT_ROOT = Path("outputs/segmentation/one_pass_no_ca_layer_filter")
REFERENCE_ROOT = Path("outputs/segmentation/final_component_ablation")
DATASETS = ("voc_sim", "coco_cap")
METHOD_KEY = "one_sa_pass_without_ca_layer_filter"
RAW_KEYS = ("tp", "fp", "fn", "tn", "images", "image_iou_sum")


def summarize(count: dict) -> dict:
    union = count["tp"] + count["fp"] + count["fn"]
    return {
        **count,
        "iou": count["tp"] / union if union else 1.0,
        "mean_image_iou": count["image_iou_sum"] / count["images"],
    }


def merge_dataset(dataset: str) -> dict:
    directory = OUTPUT_ROOT / dataset
    shards = sorted(path for path in directory.glob("class_*") if path.is_dir())
    complete = [path for path in shards if (path / "COMPLETE").is_file()]
    if len(complete) != 20:
        raise RuntimeError(f"{dataset}: expected 20 complete shards, found {len(complete)}")
    per_class: dict[str, dict] = {}
    settings: dict | None = None
    for shard in complete:
        payload = json.loads((shard / "results.json").read_text(encoding="utf-8"))
        config = payload["settings"]
        if config["completed"] != config["expected"]:
            raise RuntimeError(f"Incomplete shard: {shard}")
        classname = config["classname"]
        if classname in per_class:
            raise RuntimeError(f"Duplicate class {classname} in {dataset}")
        value = payload["results"][METHOD_KEY]
        per_class[classname] = {key: value[key] for key in RAW_KEYS}
        comparable = {
            key: config[key]
            for key in (
                "suite", "dataset", "manifest", "model", "storage_dtype",
                "model_dtype", "fixed_stage1_results", "selection_source",
            )
        }
        if settings is None:
            settings = comparable
        elif comparable != settings:
            raise RuntimeError(f"Settings differ in {shard}")

    finalized = {name: summarize(value) for name, value in sorted(per_class.items())}
    total = {key: sum(value[key] for value in per_class.values()) for key in RAW_KEYS}
    miou = sum(value["iou"] for value in finalized.values()) / len(finalized)
    reference = json.loads(
        (REFERENCE_ROOT / dataset / "results.json").read_text(encoding="utf-8")
    )["results"]["full_system"]
    result = {
        "miou": miou,
        "change_from_full": miou - reference["miou"],
        "full_system_miou": reference["miou"],
        "overall_pixel_iou": summarize(total)["iou"],
        "mean_image_iou": total["image_iou_sum"] / total["images"],
        "images": total["images"],
        "classes": len(finalized),
        "per_class": finalized,
    }
    payload = {"settings": settings, "results": {METHOD_KEY: result}}
    (directory / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (directory / "per_class_iou.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "iou"))
        writer.writerows((name, value["iou"]) for name, value in finalized.items())
    return result


def main() -> None:
    merged = {dataset: merge_dataset(dataset) for dataset in DATASETS}
    lines = [
        "# One SA pass without CA layer filtering",
        "",
        "All other final-system settings are unchanged.",
        "",
        "| Dataset | Images | Full system | Combined candidate | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        value = merged[dataset]
        lines.append(
            f"| {dataset} | {value['images']} | {value['full_system_miou']:.4f} | "
            f"{value['miou']:.4f} | {value['change_from_full']:+.4f} |"
        )
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Merged combination results into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
