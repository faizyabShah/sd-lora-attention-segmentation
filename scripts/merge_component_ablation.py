#!/usr/bin/env python3
"""Merge final-system component isolation shards for VOC-Sim and COCO-Cap."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.best_method import COMPONENT_ABLATION_LABELS


OUTPUT_ROOT = Path("outputs/segmentation/final_component_ablation")
DATASETS = ("voc_sim", "coco_cap")
RAW_KEYS = ("tp", "fp", "fn", "tn", "images", "image_iou_sum")


def raw(value: dict) -> dict:
    return {key: value[key] for key in RAW_KEYS}


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
    input_dir = OUTPUT_ROOT / dataset
    shards = sorted(path for path in input_dir.glob("class_*") if path.is_dir())
    complete = [path for path in shards if (path / "COMPLETE").is_file()]
    if len(complete) != 20:
        raise RuntimeError(f"{dataset}: expected 20 complete shards, found {len(complete)}")
    by_method = {key: {} for key in COMPONENT_ABLATION_LABELS}
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
                "model_dtype", "fixed_stage1_results", "selection_source",
            )
        }
        if common is None:
            common = comparable
        elif comparable != common:
            raise RuntimeError(f"Settings differ in {shard}")
        for method in COMPONENT_ABLATION_LABELS:
            by_method[method][classname] = raw(payload["results"][method])

    methods = {key: aggregate(values) for key, values in by_method.items()}
    full = methods["full_system"]["miou"]
    for key, value in methods.items():
        value["change_from_full"] = value["miou"] - full
        value["impact_if_removed"] = full - value["miou"]
    output = {
        "settings": {**(common or {}), "component_labels": COMPONENT_ABLATION_LABELS},
        "results": methods,
    }
    (input_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (input_dir / "component_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("configuration", "label", "miou", "change_from_full", "impact_if_removed"))
        for key, label in COMPONENT_ABLATION_LABELS.items():
            value = methods[key]
            writer.writerow((key, label, value["miou"], value["change_from_full"], value["impact_if_removed"]))
    return output


def main() -> None:
    merged = {dataset: merge_dataset(dataset) for dataset in DATASETS}
    lines = [
        "# Final-system component isolation",
        "",
        "Each row changes one final-system choice while holding the remaining choices fixed.",
        "Positive impact means that removing or replacing the component lowered mIoU.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend((f"## {dataset}", "", "| Configuration | mIoU | Change from full | Impact |", "|---|---:|---:|---:|"))
        for key, label in COMPONENT_ABLATION_LABELS.items():
            value = merged[dataset]["results"][key]
            lines.append(
                f"| {label} | {value['miou']:.4f} | {value['change_from_full']:+.4f} | {value['impact_if_removed']:+.4f} |"
            )
        lines.append("")
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Merged component isolation results into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
