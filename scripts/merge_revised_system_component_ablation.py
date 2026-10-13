#!/usr/bin/env python3
"""Merge revised-system component-isolation results for both datasets."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.best_method import REVISED_COMPONENT_ABLATION_LABELS


OUTPUT_ROOT = Path("outputs/segmentation/revised_system_component_ablation")
DATASETS = ("voc_sim", "coco_cap")
RAW_KEYS = ("tp", "fp", "fn", "tn", "images", "image_iou_sum")


def summarize(count: dict) -> dict:
    fg_union = count["tp"] + count["fp"] + count["fn"]
    bg_union = count["tn"] + count["fp"] + count["fn"]
    return {
        **count,
        "iou": count["tp"] / fg_union if fg_union else 1.0,
        "background_iou": count["tn"] / bg_union if bg_union else 1.0,
        "mean_image_iou": count["image_iou_sum"] / count["images"],
    }


def aggregate(per_class: dict[str, dict]) -> dict:
    finalized = {name: summarize(value) for name, value in sorted(per_class.items())}
    total = {key: sum(value[key] for value in per_class.values()) for key in RAW_KEYS}
    return {
        "miou": sum(value["iou"] for value in finalized.values()) / len(finalized),
        "overall_pixel_iou": summarize(total)["iou"],
        "mean_image_iou": total["image_iou_sum"] / total["images"],
        "images": total["images"],
        "classes": len(finalized),
        "per_class": finalized,
    }


def merge_dataset(dataset: str) -> dict:
    directory = OUTPUT_ROOT / dataset
    complete = sorted(
        path
        for path in directory.glob("class_*")
        if path.is_dir() and (path / "COMPLETE").is_file()
    )
    if len(complete) != 20:
        raise RuntimeError(f"{dataset}: expected 20 complete shards, found {len(complete)}")
    by_method = {key: {} for key in REVISED_COMPONENT_ABLATION_LABELS}
    seen: set[str] = set()
    common: dict | None = None
    for shard in complete:
        payload = json.loads((shard / "results.json").read_text(encoding="utf-8"))
        settings = payload["settings"]
        if settings["completed"] != settings["expected"]:
            raise RuntimeError(f"Incomplete shard: {shard}")
        classname = settings["classname"]
        if classname in seen:
            raise RuntimeError(f"Duplicate class {classname} in {dataset}")
        seen.add(classname)
        comparable = {
            key: settings[key]
            for key in (
                "suite", "dataset", "manifest", "model", "storage_dtype",
                "model_dtype", "fixed_stage1_results", "revised_baseline",
                "selection_source",
            )
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise RuntimeError(f"Settings differ in {shard}")
        for method in REVISED_COMPONENT_ABLATION_LABELS:
            value = payload["results"][method]
            by_method[method][classname] = {key: value[key] for key in RAW_KEYS}

    methods = {key: aggregate(values) for key, values in by_method.items()}
    full = methods["revised_full_system"]["miou"]
    for value in methods.values():
        value["change_from_full"] = value["miou"] - full
        value["impact_if_removed"] = full - value["miou"]
    output = {
        "settings": {**(common or {}), "component_labels": REVISED_COMPONENT_ABLATION_LABELS},
        "results": methods,
    }
    (directory / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (directory / "component_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("configuration", "label", "miou", "change_from_full", "impact_if_removed"))
        for key, label in REVISED_COMPONENT_ABLATION_LABELS.items():
            value = methods[key]
            writer.writerow((key, label, value["miou"], value["change_from_full"], value["impact_if_removed"]))
    return output


def main() -> None:
    merged = {dataset: merge_dataset(dataset) for dataset in DATASETS}
    lines = [
        "# Revised-system component isolation",
        "",
        "Baseline: one SA pass and all 16 CA layers. All other final choices are unchanged.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend((f"## {dataset}", "", "| Configuration | mIoU | Change from revised full | Impact |", "|---|---:|---:|---:|"))
        for key, label in REVISED_COMPONENT_ABLATION_LABELS.items():
            value = merged[dataset]["results"][key]
            lines.append(
                f"| {label} | {value['miou']:.4f} | {value['change_from_full']:+.4f} | {value['impact_if_removed']:+.4f} |"
            )
        lines.append("")
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Merged revised-system component results into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
