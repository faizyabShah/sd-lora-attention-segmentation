#!/usr/bin/env python3
"""Finalize locked token-SG3 VOC comparisons and untouched COCO transfer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_propagation_cv import best_threshold, image_ious, pooled_miou  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

CONFIG = ROOT / "configs/token_sg3_confirmation.json"
OUT = ROOT / "outputs/segmentation/controlled_attention_cv/token_sg3_confirmation"


def load(dataset: str) -> cv.Results:
    result = cv.load_results(sorted((OUT / dataset / "shards").glob("*.npz")))
    if not all(result.complete) or len(set(result.classes)) != 20:
        raise SystemExit(f"{dataset}: incomplete package")
    return result


def fold_comparison(results: cv.Results, candidate: str, reference: str) -> dict:
    selected = np.zeros((len(results.image_ids), 4), dtype=np.int64)
    baseline = np.zeros_like(selected)
    folds = []
    for fold in range(cv.FOLD_COUNT):
        held = results.folds == fold
        train = ~held
        ct = best_threshold(results, results.index(candidate), train)[0]
        rt = best_threshold(results, results.index(reference), train)[0]
        selected[held] = results.counts[results.index(candidate), held, ct]
        baseline[held] = results.counts[results.index(reference), held, rt]
        folds.append({"fold": fold, "candidate_threshold": results.thresholds[ct],
                      "reference_threshold": results.thresholds[rt]})
    classes = np.asarray(results.classes)
    a, b = pooled_miou(selected, classes), pooled_miou(baseline, classes)
    return {"candidate": candidate, "reference": reference, "folds": folds,
            "oof_candidate": a, "oof_reference": b, "oof_gain": a-b,
            "paired_image_iou": cv.paired_bootstrap(image_ious(selected), image_ious(baseline))}


def main() -> None:
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    voc, coco = load("voc"), load("coco")
    comparisons = {f"{a}_vs_{b}": fold_comparison(voc, a, b) for a, b in spec["comparisons"]}
    full = np.ones(len(voc.image_ids), dtype=bool)
    voc_classes, coco_classes = np.asarray(voc.classes), np.asarray(coco.classes)
    cells = {}
    for name in voc.configurations:
        vt = best_threshold(voc, voc.index(name), full)[0]
        vc = voc.counts[voc.index(name), :, vt]
        cc = coco.counts[coco.index(name), :, vt]
        cells[name] = {"voc_threshold": voc.thresholds[vt],
                       "voc_miou": pooled_miou(vc, voc_classes),
                       "coco_miou_at_voc_threshold": pooled_miou(cc, coco_classes)}
    output = {"protocol": "Section 72", "voc_images": len(voc.image_ids),
              "coco_images": len(coco.image_ids), "comparisons": comparisons, "cells": cells}
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = ["# Locked token-SG3 confirmation", "",
             f"VOC-Sim {len(voc.image_ids)} images; COCO-Cap {len(coco.image_ids)} images.", "",
             "| Cell | VOC threshold | VOC mIoU | COCO mIoU |", "|---|---:|---:|---:|"]
    for name, row in cells.items():
        lines.append(f"| `{name}` | {row['voc_threshold']:.3f} | {row['voc_miou']:.4f} | {row['coco_miou_at_voc_threshold']:.4f} |")
    lines += ["", "| OOF comparison | Reference -> candidate | Gain | Paired image-IoU [95% CI] |",
              "|---|---:|---:|---:|"]
    for row in comparisons.values():
        p = row["paired_image_iou"]
        lines.append(f"| `{row['candidate']}` vs `{row['reference']}` | {row['oof_reference']:.4f} -> {row['oof_candidate']:.4f} | "
                     f"{row['oof_gain']:+.4f} | {p['mean']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
