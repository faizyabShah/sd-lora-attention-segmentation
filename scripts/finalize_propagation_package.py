#!/usr/bin/env python3
"""Add frozen-winner and per-class diagnostics to propagation results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_propagation_cv import (  # noqa: E402
    best_threshold,
    image_ious,
    pooled_miou,
)
from src.segmentation import controlled_cv as cv  # noqa: E402


def crossfit(results, configuration: str, reference: str) -> dict:
    selected = results.index(configuration)
    ref = results.index(reference)
    oof = np.zeros((len(results.image_ids), 4), dtype=np.int64)
    oof_ref = np.zeros_like(oof)
    thresholds = []
    for fold in range(cv.FOLD_COUNT):
        held = results.folds == fold
        train = ~held
        selected_t, _ = best_threshold(results, selected, train)
        ref_t, _ = best_threshold(results, ref, train)
        oof[held] = results.counts[selected, held, selected_t]
        oof_ref[held] = results.counts[ref, held, ref_t]
        thresholds.append(
            {
                "fold": fold,
                "selected": results.thresholds[selected_t],
                "reference": results.thresholds[ref_t],
            }
        )
    classes = np.asarray(results.classes)
    per_class = {}
    for classname in sorted(set(classes)):
        keep = classes == classname
        selected_counts = oof[keep, :3].sum(axis=0)
        ref_counts = oof_ref[keep, :3].sum(axis=0)
        selected_iou = float(selected_counts[0] / max(selected_counts.sum(), 1))
        ref_iou = float(ref_counts[0] / max(ref_counts.sum(), 1))
        per_class[classname] = {
            "reference": ref_iou,
            "selected": selected_iou,
            "delta": selected_iou - ref_iou,
        }
    return {
        "configuration": configuration,
        "reference": reference,
        "thresholds": thresholds,
        "oof_miou_selected": pooled_miou(oof, results.classes),
        "oof_miou_reference": pooled_miou(oof_ref, results.classes),
        "paired_bootstrap": cv.paired_bootstrap(image_ious(oof), image_ious(oof_ref)),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("outputs/segmentation/controlled_attention_cv/propagation_package"),
    )
    args = parser.parse_args()
    package = args.package_dir
    results = cv.load_results(sorted((package / "shards").glob("*.npz")))
    folds = json.loads((package / "fold_results.json").read_text(encoding="utf-8"))
    diagnostics = {}
    for screen_name, screen in folds["screens"].items():
        if screen["frozen"] != screen["reference"]:
            diagnostics[screen_name] = crossfit(results, screen["frozen"], screen["reference"])
    (package / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    marker = "## Frozen-winner diagnostics"
    lines = ["", marker]
    for screen_name, item in diagnostics.items():
        gain = item["oof_miou_selected"] - item["oof_miou_reference"]
        bootstrap = item["paired_bootstrap"]
        lines += [
            "",
            f"### {screen_name}: `{item['configuration']}`",
            "",
            f"Pure frozen-candidate OOF mIoU: {item['oof_miou_reference']:.4f} → "
            f"{item['oof_miou_selected']:.4f} ({gain:+.4f}). Paired mean image-IoU "
            f"{bootstrap['mean']:+.4f} [{bootstrap['ci_low']:+.4f}, "
            f"{bootstrap['ci_high']:+.4f}]; {bootstrap['better']} better, "
            f"{bootstrap['worse']} worse.",
            "",
            "| Class | Reference OOF IoU | Selected OOF IoU | Δ |",
            "|---|---:|---:|---:|",
        ]
        for classname, row in sorted(item["per_class"].items(), key=lambda pair: pair[1]["delta"]):
            lines.append(
                f"| {classname} | {row['reference']:.4f} | {row['selected']:.4f} | "
                f"{row['delta']:+.4f} |"
            )
    report_path = package / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.split(marker, 1)[0].rstrip() + "\n" + "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
