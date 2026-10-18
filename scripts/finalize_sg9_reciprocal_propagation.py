#!/usr/bin/env python3
"""Finalize SG9 VOC selection, COCO transfer, and reciprocal diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_propagation_cv import best_threshold, image_ious, pooled_miou, run_screen  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

SPEC_PATH = ROOT / "configs/sg9_reciprocal_propagation.json"
OUT = ROOT / "outputs/segmentation/controlled_attention_cv/sg9_reciprocal_propagation"


def load(dataset: str) -> cv.Results:
    paths = [p for p in sorted((OUT / dataset / "shards").glob("class_*.npz"))
             if not p.name.endswith(".diagnostics.npz")]
    result = cv.load_results(paths)
    if not all(result.complete) or len(set(result.classes)) != 20:
        raise SystemExit(f"{dataset}: incomplete result shards")
    return result


def load_diagnostics(dataset: str, image_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    records = {}
    for path in sorted((OUT / dataset / "shards").glob("*.diagnostics.npz")):
        with np.load(path, allow_pickle=False) as data:
            for i, image_id in enumerate(data["image_ids"].tolist()):
                if image_id in records:
                    raise SystemExit(f"duplicate diagnostic record {image_id}")
                records[image_id] = (data["changes"][i], data["spatial"][i], data["mechanism"][i])
    missing, extra = set(image_ids) - records.keys(), records.keys() - set(image_ids)
    if missing or extra:
        raise SystemExit(f"{dataset}: diagnostic/result mismatch: {len(missing)} missing, {len(extra)} extra")
    ordered = [records[i] for i in image_ids]
    return {"changes": np.stack([x[0] for x in ordered]), "spatial": np.stack([x[1] for x in ordered]),
            "mechanism": np.stack([x[2] for x in ordered])}


def rates(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn = counts[..., :3].sum(axis=0)
    return {"recall": float(tp / max(tp + fn, 1)), "precision": float(tp / max(tp + fp, 1))}


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    voc, coco = load("voc"), load("coco")
    vd, cd = load_diagnostics("voc", voc.image_ids), load_diagnostics("coco", coco.image_ids)
    screens = {}
    for track in ("W", "T"):
        candidates = [f"{track}/R1_geometric", f"{track}/R2_minimum", f"{track}/R3_soft_gate"]
        preference = {candidates[0]: 2, candidates[1]: 1, candidates[2]: 3}
        screens[track] = run_screen(voc, f"{track}/REF", candidates, preference)
    full = np.ones(len(voc.image_ids), dtype=bool)
    thresholds = {name: best_threshold(voc, voc.index(name), full)[0] for name in voc.configurations}
    cells, transfer = {}, {}
    for name in voc.configurations:
        index, threshold = voc.index(name), thresholds[name]
        vc, cc = voc.counts[index, :, threshold], coco.counts[coco.index(name), :, threshold]
        cells[name] = {"voc_threshold": voc.thresholds[threshold], "voc_miou": pooled_miou(vc, voc.classes),
                       "voc_rates": rates(vc), "coco_miou": pooled_miou(cc, coco.classes), "coco_rates": rates(cc)}
    for track, screen in screens.items():
        frozen, reference = screen["frozen"], screen["reference"]
        fi, ft, rt = voc.index(frozen), thresholds[frozen], thresholds[reference]
        a, b = coco.counts[coco.index(frozen), :, ft], coco.counts[coco.index(reference), :, rt]
        transfer[track] = {"frozen": frozen, "reference": reference,
                           "coco_miou_frozen": pooled_miou(a, coco.classes),
                           "coco_miou_reference": pooled_miou(b, coco.classes),
                           "paired_image_iou": cv.paired_bootstrap(image_ious(a), image_ious(b)),
                           "changes_from_seed": {"fields": ["added_tp", "added_fp", "removed_tp", "removed_fp"],
                               "voc": vd["changes"][:, fi, ft].sum(0).astype(int).tolist(),
                               "coco": cd["changes"][:, coco.index(frozen), ft].sum(0).astype(int).tolist()},
                           "spatial_new_fp": {"fields": ["distant_gt16", "attached", "separate_component"],
                               "voc": vd["spatial"][:, fi, ft].sum(0).astype(int).tolist(),
                               "coco": cd["spatial"][:, coco.index(frozen), ft].sum(0).astype(int).tolist()}}
    regions = ("added_tp", "added_fp", "distant_added_fp_gt16", "unchanged_membership")
    mechanism = {}
    for dataset, diag in (("voc", vd), ("coco", cd)):
        mechanism[dataset] = {}
        for track, track_index in (("W", 0), ("T", 1)):
            mechanism[dataset][track] = {}
            for region_index, region in enumerate(regions):
                sums = diag["mechanism"][:, track_index, region_index].sum(0)
                mechanism[dataset][track][region] = {
                    "mean_seed_weighted_edge_asymmetry": float(sums[0] / max(sums[2], 1)),
                    "mean_two_step_return_probability": float(sums[1] / max(sums[2], 1)),
                    "pixels": int(sums[2]),
                }
    output = {"protocol": "Section 74", "spec": spec, "voc_images": len(voc.image_ids),
              "coco_images": len(coco.image_ids), "screens": screens, "transfer": transfer,
              "cells": cells, "mechanism": mechanism}
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = ["# SG9 reciprocal self-attention propagation", "",
             f"VOC-Sim {len(voc.image_ids)} images (selection); COCO-Cap {len(coco.image_ids)} images (transfer only).", "",
             "| Track | Winner | Votes | OOF reference → selection | Gain | Frozen | COCO frozen/reference |",
             "|---|---|---:|---:|---:|---|---:|"]
    for track, screen in screens.items():
        tr = transfer[track]
        lines.append(f"| {track} | `{screen['vote_winner']}` | {screen['vote_winner_votes']}/5 | "
                     f"{screen['oof_miou_reference']:.4f} → {screen['oof_miou_selection']:.4f} | "
                     f"{screen['oof_gain']:+.4f} | `{screen['frozen']}` | "
                     f"{tr['coco_miou_frozen']:.4f}/{tr['coco_miou_reference']:.4f} |")
    lines += ["", "Every cell:", "", "| Cell | VOC threshold | VOC mIoU | VOC R/P | COCO mIoU | COCO R/P |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, cell in cells.items():
        lines.append(f"| `{name}` | {cell['voc_threshold']:.3f} | {cell['voc_miou']:.4f} | "
                     f"{cell['voc_rates']['recall']:.3f}/{cell['voc_rates']['precision']:.3f} | "
                     f"{cell['coco_miou']:.4f} | {cell['coco_rates']['recall']:.3f}/{cell['coco_rates']['precision']:.3f} |")
    lines += ["", "AN1-style change/spatial counts and reciprocal-edge mechanism diagnostics are in `summary.json`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7 + len(screens)]))


if __name__ == "__main__":
    main()
