#!/usr/bin/env python3
"""Finalize SG8 fold selection, untouched transfer, and compact diagnostics."""

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

SPEC_PATH = ROOT / "configs/sg8_observation_consensus.json"
OUT = ROOT / "outputs/segmentation/controlled_attention_cv/sg8_observation_consensus"


def load(dataset: str) -> cv.Results:
    shards = [path for path in sorted((OUT / dataset / "shards").glob("class_*.npz"))
              if not path.name.endswith(".diagnostics.npz")]
    result = cv.load_results(shards)
    if not all(result.complete) or len(set(result.classes)) != 20:
        raise SystemExit(f"{dataset}: incomplete result shards")
    return result


def load_diagnostics(dataset: str, image_ids: tuple[str, ...]) -> dict[str, object]:
    records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fields = None
    for path in sorted((OUT / dataset / "shards").glob("*.diagnostics.npz")):
        with np.load(path, allow_pickle=False) as data:
            current_fields = tuple(data["agreement_fields"].tolist())
            fields = current_fields if fields is None else fields
            if current_fields != fields:
                raise SystemExit(f"{path}: diagnostic fields differ")
            for i, image_id in enumerate(data["image_ids"].tolist()):
                if image_id in records:
                    raise SystemExit(f"duplicate diagnostic record {image_id}")
                records[image_id] = (data["agreement"][i], data["changes"][i], data["mechanism"][i])
    missing, extra = set(image_ids) - records.keys(), records.keys() - set(image_ids)
    if missing or extra:
        raise SystemExit(f"{dataset}: diagnostic/result mismatch: {len(missing)} missing, {len(extra)} extra")
    ordered = [records[i] for i in image_ids]
    return {"fields": fields, "agreement": np.stack([x[0] for x in ordered]),
            "changes": np.stack([x[1] for x in ordered]), "mechanism": np.stack([x[2] for x in ordered])}


def screen_specs(configurations: tuple[str, ...]) -> list[dict]:
    specs = []
    for track in ("W", "T"):
        reference = f"{track}/REF"
        for family in ("A", "B", "C", "D"):
            candidates = [name for name in configurations if name.startswith(f"{track}/{family}/")]
            if family == "A":
                preference = {f"{track}/A/trim10": 3, f"{track}/A/trim20": 2, f"{track}/A/median": 1}
            else:
                preference = {name: i for i, name in enumerate(candidates, start=1)}
            specs.append({"name": f"{track}-{family}", "reference": reference,
                          "candidates": candidates, "inclusiveness": preference})
    return specs


def rates(counts: np.ndarray) -> dict[str, float]:
    tp, fp, fn = counts[..., :3].sum(axis=0)
    return {"recall": float(tp / max(tp + fn, 1)), "precision": float(tp / max(tp + fp, 1))}


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    voc, coco = load("voc"), load("coco")
    voc_diag = load_diagnostics("voc", voc.image_ids)
    coco_diag = load_diagnostics("coco", coco.image_ids)
    screens = {row["name"]: run_screen(voc, row["reference"], row["candidates"], row["inclusiveness"])
               for row in screen_specs(voc.configurations)}
    full = np.ones(len(voc.image_ids), dtype=bool)
    thresholds = {name: best_threshold(voc, voc.index(name), full)[0] for name in voc.configurations}
    cells, transfer = {}, {}
    for name in voc.configurations:
        index, threshold = voc.index(name), thresholds[name]
        vc, cc = voc.counts[index, :, threshold], coco.counts[coco.index(name), :, threshold]
        cells[name] = {"voc_threshold": voc.thresholds[threshold],
                       "voc_miou": pooled_miou(vc, voc.classes), "voc_rates": rates(vc),
                       "coco_miou": pooled_miou(cc, coco.classes), "coco_rates": rates(cc)}
    for key, screen in screens.items():
        frozen, reference = screen["frozen"], screen["reference"]
        fi, ft = voc.index(frozen), thresholds[frozen]
        rt = thresholds[reference]
        a = coco.counts[coco.index(frozen), :, ft]
        b = coco.counts[coco.index(reference), :, rt]
        transfer[key] = {"frozen": frozen, "reference": reference,
                         "coco_miou_frozen": pooled_miou(a, coco.classes),
                         "coco_miou_reference": pooled_miou(b, coco.classes),
                         "paired_image_iou": cv.paired_bootstrap(image_ious(a), image_ious(b)),
                         "changes_from_seed_at_selected_threshold": {
                             "voc": voc_diag["changes"][:, fi, ft].sum(axis=0).astype(int).tolist(),
                             "coco": coco_diag["changes"][:, coco.index(frozen), ft].sum(axis=0).astype(int).tolist(),
                         }}
    agreement = {}
    regions = ("added_tp", "added_fp", "distant_added_fp_gt16", "unchanged_membership")
    for dataset, diag in (("voc", voc_diag), ("coco", coco_diag)):
        agreement[dataset] = {}
        for track, index in (("W", 0), ("T", 1)):
            mean_values = diag["agreement"][:, index].mean(axis=0)
            region_values = {}
            for region_index, region in enumerate(regions):
                sums = diag["mechanism"][:, index, region_index].sum(axis=0)
                region_values[region] = {"mean_sigma": float(sums[0] / max(sums[2], 1)),
                                         "mean_direction_consensus": float(sums[1] / max(sums[2], 1)),
                                         "pixels": int(sums[2])}
            agreement[dataset][track] = {
                "mean_per_image": {field: float(value) for field, value in zip(diag["fields"], mean_values)},
                "reference_regions_at_0.45": region_values,
            }
    output = {"protocol": "Section 73", "spec": spec, "voc_images": len(voc.image_ids),
              "coco_images": len(coco.image_ids), "screens": screens, "transfer": transfer,
              "cells": cells, "agreement_diagnostics": agreement,
              "change_fields": ["added_tp", "added_fp", "removed_tp", "removed_fp"]}
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = ["# SG8 observation-consensus propagation", "",
             f"VOC-Sim {len(voc.image_ids)} images (selection); COCO-Cap {len(coco.image_ids)} images (transfer only).", "",
             "| Screen | Winner | Votes | OOF reference → selection | Gain | Frozen | COCO frozen/reference |",
             "|---|---|---:|---:|---:|---|---:|"]
    for key, screen in screens.items():
        tr = transfer[key]
        lines.append(f"| {key} | `{screen['vote_winner']}` | {screen['vote_winner_votes']}/5 | "
                     f"{screen['oof_miou_reference']:.4f} → {screen['oof_miou_selection']:.4f} | "
                     f"{screen['oof_gain']:+.4f} | `{screen['frozen']}` | "
                     f"{tr['coco_miou_frozen']:.4f}/{tr['coco_miou_reference']:.4f} |")
    lines += ["", "Every cell:", "", "| Cell | VOC threshold | VOC mIoU | COCO mIoU |", "|---|---:|---:|---:|"]
    for name, cell in cells.items():
        lines.append(f"| `{name}` | {cell['voc_threshold']:.3f} | {cell['voc_miou']:.4f} | {cell['coco_miou']:.4f} |")
    lines += ["", "Agreement, propagation-change, and regional leakage diagnostics are in `summary.json`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:7 + len(screens)]))


if __name__ == "__main__":
    main()
