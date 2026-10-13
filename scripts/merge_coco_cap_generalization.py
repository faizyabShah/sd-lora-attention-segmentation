#!/usr/bin/env python3
"""Merge strict COCO-Cap class shards into tables and a short report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHOD_LABELS = {
    "ovam": "OVAM (paper COCO-Cap parameters)",
    "frozen_best_voc": "Our best configuration (frozen on VOC-Sim)",
}
RAW_KEYS = ("tp", "fp", "fn", "tn", "images", "image_iou_sum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/segmentation/coco_cap_generalization"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/"
            "coco_captions/ovam_coco_cap.csv"
        ),
    )
    parser.add_argument("--expected-shards", type=int, default=20)
    return parser.parse_args()


def add_counts(destination: dict, source: dict) -> None:
    for key in RAW_KEYS:
        destination[key] = destination.get(key, 0) + source[key]


def summarize(count: dict) -> dict:
    fg_union = count["tp"] + count["fp"] + count["fn"]
    bg_union = count["tn"] + count["fp"] + count["fn"]
    return {
        **count,
        "iou": count["tp"] / fg_union if fg_union else 1.0,
        "background_iou": count["tn"] / bg_union if bg_union else 1.0,
        "mean_image_iou": count["image_iou_sum"] / count["images"],
    }


def aggregate(classes: dict[str, dict]) -> dict:
    per_class = {name: summarize(count) for name, count in sorted(classes.items())}
    total: dict = {}
    for count in classes.values():
        add_counts(total, count)
    return {
        "miou": sum(item["iou"] for item in per_class.values()) / len(per_class),
        "overall_pixel_iou": summarize(total)["iou"],
        "mean_image_iou": total["image_iou_sum"] / total["images"],
        "images": total["images"],
        "classes": len(per_class),
        "per_class": per_class,
    }


def raw(value: dict) -> dict:
    return {key: value[key] for key in RAW_KEYS}


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    expected_by_class: dict[str, int] = {}
    for row in manifest_rows:
        expected_by_class[row["classname"]] = expected_by_class.get(row["classname"], 0) + 1

    shard_dirs = sorted(path for path in args.input_dir.glob("class_*") if path.is_dir())
    complete = [path for path in shard_dirs if (path / "COMPLETE").is_file()]
    if len(complete) != args.expected_shards:
        raise RuntimeError(
            f"Expected {args.expected_shards} complete shards, found {len(complete)}"
        )

    by_method: dict[str, dict[str, dict]] = {key: {} for key in METHOD_LABELS}
    subgroup: dict[str, dict[str, dict[str, dict]]] = {
        key: {"class_present": {}, "class_absent": {}} for key in METHOD_LABELS
    }
    seen_classes: set[str] = set()
    settings: list[dict] = []
    shared_signature: dict | None = None
    for shard in complete:
        payload = json.loads((shard / "results.json").read_text(encoding="utf-8"))
        config = payload["settings"]
        classname = config["classname"]
        if classname in seen_classes:
            raise RuntimeError(f"Duplicate completed shard for {classname}")
        seen_classes.add(classname)
        if config["completed"] != config["expected"]:
            raise RuntimeError(f"Incomplete results inside {shard}")
        if config["expected"] != expected_by_class.get(classname):
            raise RuntimeError(f"Manifest count mismatch for {classname}")
        comparable = {
            key: config[key]
            for key in (
                "suite",
                "manifest",
                "model",
                "storage_dtype",
                "model_dtype",
                "attribution",
                "ovam_parameters",
                "best_method_source",
                "fixed_stage1_results",
            )
        }
        if shared_signature is None:
            shared_signature = comparable
        elif comparable != shared_signature:
            raise RuntimeError(f"Evaluation settings differ in {shard}")
        settings.append(config)
        for method in METHOD_LABELS:
            by_method[method][classname] = raw(payload["results"][method])
            for name in ("class_present", "class_absent"):
                value = payload["caption_subgroups"][method].get(name)
                if value is not None:
                    subgroup[method][name][classname] = raw(value)

    if seen_classes != set(expected_by_class):
        raise RuntimeError(f"Class mismatch: found {sorted(seen_classes)}")

    methods = {method: aggregate(classes) for method, classes in by_method.items()}
    subgroups = {
        method: {
            name: aggregate(classes) for name, classes in values.items() if classes
        }
        for method, values in subgroup.items()
    }
    output = {
        "settings": {
            "dataset": "OVAM COCO-Cap annotated subset",
            "images": len(manifest_rows),
            "classes": len(expected_by_class),
            "no_coco_tuning": True,
            "ovam_parameters": settings[0]["ovam_parameters"],
            "best_method_source": settings[0]["best_method_source"],
        },
        "results": methods,
        "caption_subgroups": subgroups,
    }
    args.input_dir.mkdir(parents=True, exist_ok=True)
    (args.input_dir / "results.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (args.input_dir / "per_class_iou.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "images", *METHOD_LABELS))
        for classname in sorted(expected_by_class):
            writer.writerow(
                (
                    classname,
                    expected_by_class[classname],
                    *(methods[key]["per_class"][classname]["iou"] for key in METHOD_LABELS),
                )
            )

    lines = [
        "# COCO-Cap untuned generalization",
        "",
        f"Evaluated **{len(manifest_rows)} annotated images** from **{len(expected_by_class)} classes**. "
        "All choices in our method were frozen from VOC-Sim before this evaluation.",
        "",
        "## Overall results",
        "",
        "| Method | mIoU (20-class macro) | Mean image IoU | Overall pixel IoU |",
        "|---|---:|---:|---:|",
    ]
    for method, label in METHOD_LABELS.items():
        value = methods[method]
        lines.append(
            f"| {label} | {value['miou']:.4f} | {value['mean_image_iou']:.4f} | "
            f"{value['overall_pixel_iou']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Caption wording split",
            "",
            "This split tests whether the method still localizes the annotated category when the exact "
            "category word is absent from the generation caption. Both methods receive the category as "
            "a separate attribution prompt, matching OVAM's COCO-Cap protocol.",
            "",
            "| Method | Exact class word present | Exact class word absent |",
            "|---|---:|---:|",
        ]
    )
    for method, label in METHOD_LABELS.items():
        present = subgroups[method]["class_present"]
        absent = subgroups[method]["class_absent"]
        lines.append(
            f"| {label} | {present['miou']:.4f} ({present['images']} images) | "
            f"{absent['miou']:.4f} ({absent['images']} images) |"
        )
    lines.extend(
        [
            "",
            "## Per-class foreground IoU",
            "",
            "| Class | Images | OVAM | Frozen best VOC | Difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for classname in sorted(expected_by_class):
        ovam = methods["ovam"]["per_class"][classname]["iou"]
        best = methods["frozen_best_voc"]["per_class"][classname]["iou"]
        lines.append(
            f"| {classname} | {expected_by_class[classname]} | {ovam:.4f} | "
            f"{best:.4f} | {best - ovam:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed evaluation settings",
            "",
            "- OVAM uses the paper's COCO-Cap self-attention floor of 0.91 and threshold of 0.50, followed by its DenseCRF.",
            "- Our method uses the VOC-selected 12 CA layers and top four CA heads per layer, final 64×64 SA block, four propagation passes, SA heads 0/1/2/4/5/6, denoising calls 15 onward, percentile 1–99 normalization, threshold 0.45, and 1% component removal.",
            "- The generation caption and seed come from the released COCO-Cap data. The class name is encoded separately for attribution, as in OVAM's released COCO-Cap script.",
            "",
        ]
    )
    (args.input_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Merged {len(manifest_rows)} images; report: {args.input_dir / 'report.md'}")


if __name__ == "__main__":
    main()

