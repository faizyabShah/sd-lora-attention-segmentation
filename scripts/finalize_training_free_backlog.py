#!/usr/bin/env python3
"""Cross-validate training-free backlog families and report untouched COCO transfer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_propagation_cv import (  # noqa: E402
    best_threshold, image_ious, miou, pooled_miou, run_screen,
)
from src.segmentation import controlled_cv as cv  # noqa: E402

OUT = ROOT / "outputs/segmentation/controlled_attention_cv/training_free_backlog"
SPEC = ROOT / "configs/training_free_backlog.json"
REFERENCE = "REF/one_pass"
PREFIXES = ("P3/", "P5/", "P9/", "F2/", "BG1/", "W1/", "W2/", "W4/", "H1/",
            "G1/", "G4/", "M1/", "M2/", "R1/")


def load(dataset: str) -> cv.Results:
    paths = sorted((OUT / dataset / "shards").glob("*.npz"))
    result = cv.load_results(paths)
    expected = 490 if dataset == "voc" else 707
    if len(paths) != 20 or len(result.image_ids) != expected or not all(result.complete):
        raise SystemExit(f"{dataset}: expected 20 complete shards/{expected} images, found {len(paths)}/{len(result.image_ids)}")
    return result


def family_screens(voc: cv.Results) -> dict[str, dict]:
    output = {}
    for prefix in PREFIXES:
        candidates = [name for name in voc.configurations if name.startswith(prefix)]
        if not candidates:
            raise RuntimeError(f"No configurations for {prefix}")
        # Larger top-k and less restrictive parameters are treated as more inclusive;
        # REF still wins every within-band tie through the registered decision rule.
        inclusiveness = {name: index for index, name in enumerate(candidates)}
        output[prefix.rstrip("/")] = run_screen(voc, REFERENCE, candidates, inclusiveness)
    return output


def rates(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn = counts[..., :3].sum(0)
    return {"recall": float(tp / max(tp + fn, 1)), "precision": float(tp / max(tp + fp, 1))}


def class_threshold_diagnostic(voc: cv.Results) -> dict:
    """C1: label-using per-class threshold selection, reported only as a diagnostic."""
    reference = voc.index(REFERENCE)
    classes = np.asarray(voc.classes)
    selected = np.zeros((len(voc.image_ids), 4), dtype=np.int64)
    global_ref = np.zeros_like(selected)
    choices = []
    for fold in range(cv.FOLD_COUNT):
        held, train = voc.folds == fold, voc.folds != fold
        global_t, _ = best_threshold(voc, reference, train)
        for classname in sorted(set(classes)):
            class_train = train & (classes == classname)
            class_held = held & (classes == classname)
            threshold, _ = best_threshold(voc, reference, class_train)
            selected[class_held] = voc.counts[reference, class_held, threshold]
            global_ref[class_held] = voc.counts[reference, class_held, global_t]
            choices.append({"fold": fold, "class": classname, "threshold": voc.thresholds[threshold]})
    return {
        "label_using_diagnostic": True,
        "oof_miou_per_class_threshold": pooled_miou(selected, voc.classes),
        "oof_miou_global_threshold": pooled_miou(global_ref, voc.classes),
        "gain": pooled_miou(selected, voc.classes) - pooled_miou(global_ref, voc.classes),
        "choices": choices,
    }


def class_operator_diagnostic(voc: cv.Results, screens: dict[str, dict]) -> dict:
    """C2: choose among independently frozen family cells for each class/fold."""
    candidates = list(dict.fromkeys([REFERENCE] + [screen["frozen"] for screen in screens.values()]))
    classes = np.asarray(voc.classes)
    selected = np.zeros((len(voc.image_ids), 4), dtype=np.int64)
    reference_oof = np.zeros_like(selected)
    choices = []
    for fold in range(cv.FOLD_COUNT):
        held, train = voc.folds == fold, voc.folds != fold
        ref_t, _ = best_threshold(voc, voc.index(REFERENCE), train)
        reference_oof[held] = voc.counts[voc.index(REFERENCE), held, ref_t]
        for classname in sorted(set(classes)):
            class_train, class_held = train & (classes == classname), held & (classes == classname)
            scored = []
            for name in candidates:
                index = voc.index(name)
                threshold, score = best_threshold(voc, index, class_train)
                scored.append((score, name == REFERENCE, name, index, threshold))
            _, _, name, index, threshold = max(scored)
            selected[class_held] = voc.counts[index, class_held, threshold]
            choices.append({"fold": fold, "class": classname, "configuration": name,
                            "threshold": voc.thresholds[threshold]})
    return {
        "label_using_diagnostic": True, "candidate_cells": candidates,
        "oof_miou_per_class_operator": pooled_miou(selected, voc.classes),
        "oof_miou_reference": pooled_miou(reference_oof, voc.classes),
        "gain": pooled_miou(selected, voc.classes) - pooled_miou(reference_oof, voc.classes),
        "choices": choices,
    }


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    voc, coco = load("voc"), load("coco")
    if voc.configurations != coco.configurations or voc.thresholds != coco.thresholds:
        raise RuntimeError("VOC and COCO configuration contracts differ")
    screens = family_screens(voc)
    full = np.ones(len(voc.image_ids), dtype=bool)
    cells, transfer = {}, {}
    for index, name in enumerate(voc.configurations):
        threshold, score = best_threshold(voc, index, full)
        coco_counts = coco.counts[coco.index(name), :, threshold]
        cells[name] = {
            "voc_threshold": voc.thresholds[threshold], "voc_miou": score,
            "voc_per_class_iou": cv.class_ious(voc, index, threshold),
            "coco_miou_at_voc_threshold": pooled_miou(coco_counts, coco.classes),
            "voc_rates": rates(voc.counts[index, :, threshold]), "coco_rates": rates(coco_counts),
        }
    for family, screen in screens.items():
        frozen, reference = screen["frozen"], screen["reference"]
        frozen_t = voc.thresholds.index(screen["full_threshold_frozen"])
        reference_t = voc.thresholds.index(screen["full_threshold_reference"])
        frozen_counts = coco.counts[coco.index(frozen), :, frozen_t]
        reference_counts = coco.counts[coco.index(reference), :, reference_t]
        transfer[family] = {
            "frozen": frozen, "reference": reference,
            "frozen_voc_threshold": voc.thresholds[frozen_t],
            "reference_voc_threshold": voc.thresholds[reference_t],
            "coco_miou_frozen": pooled_miou(frozen_counts, coco.classes),
            "coco_miou_reference": pooled_miou(reference_counts, coco.classes),
            "paired_image_iou": cv.paired_bootstrap(image_ious(frozen_counts), image_ious(reference_counts)),
        }
    diagnostics = {
        "C1_per_class_threshold": class_threshold_diagnostic(voc),
        "C2_per_class_operator": class_operator_diagnostic(voc, screens),
    }
    output = {
        "protocol": spec["protocol"], "spec": spec, "voc_images": len(voc.image_ids),
        "coco_images": len(coco.image_ids), "configurations": len(voc.configurations),
        "screens": screens, "transfer": transfer, "cells": cells, "diagnostics": diagnostics,
    }
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    (OUT / "fold_results.json").write_text(json.dumps({
        "protocol": spec["protocol"], "images": len(voc.image_ids), "classes": len(set(voc.classes)),
        "configurations": len(voc.configurations), "thresholds": list(voc.thresholds), "screens": screens,
        "diagnostics": diagnostics,
    }, indent=2) + "\n", encoding="utf-8")

    lines = ["# Training-free backlog screen", "",
             f"VOC-Sim {len(voc.image_ids)} images selects; COCO-Cap {len(coco.image_ids)} images transfers only. "
             f"{len(voc.configurations)} cells, class-word attribution only.", "",
             "| Family | Vote winner | Votes | OOF REF → selection | Gain | Frozen | COCO frozen / REF |",
             "|---|---|---:|---:|---:|---|---:|"]
    for family, screen in screens.items():
        item = transfer[family]
        lines.append(
            f"| {family} | `{screen['vote_winner']}` | {screen['vote_winner_votes']}/5 | "
            f"{screen['oof_miou_reference']:.4f} → {screen['oof_miou_selection']:.4f} | "
            f"{screen['oof_gain']:+.4f} | `{screen['frozen']}` | "
            f"{item['coco_miou_frozen']:.4f} / {item['coco_miou_reference']:.4f} |"
        )
    lines += ["", "## Label-using diagnostics (not eligible for the label-free pipeline)", "",
              f"- C1 per-class threshold: {diagnostics['C1_per_class_threshold']['oof_miou_global_threshold']:.4f} → "
              f"{diagnostics['C1_per_class_threshold']['oof_miou_per_class_threshold']:.4f} "
              f"({diagnostics['C1_per_class_threshold']['gain']:+.4f}).",
              f"- C2 per-class operator: {diagnostics['C2_per_class_operator']['oof_miou_reference']:.4f} → "
              f"{diagnostics['C2_per_class_operator']['oof_miou_per_class_operator']:.4f} "
              f"({diagnostics['C2_per_class_operator']['gain']:+.4f}).", "",
              "## Every cell", "", "| Cell | VOC threshold | VOC mIoU | COCO mIoU |", "|---|---:|---:|---:|"]
    for name, cell in cells.items():
        lines.append(f"| `{name}` | {cell['voc_threshold']:.3f} | {cell['voc_miou']:.4f} | "
                     f"{cell['coco_miou_at_voc_threshold']:.4f} |")
    lines += ["", "C1/C2 use labels for fold-wise selection and are diagnostics only. No token, token optimization, "
              "LoRA, external segmenter, generated image, attention tensor, graph, cluster, or mask is retained."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7 + len(screens)]))


if __name__ == "__main__":
    main()
