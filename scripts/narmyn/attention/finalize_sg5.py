#!/usr/bin/env python3
"""Finalize SG5 (protocol Section 66): fold selection on VOC-Sim, AN1 diagnostics, COCO-Cap transfer.

Selection follows protocol v1.2 exactly (``scripts/select_propagation_cv.run_screen``),
one screen per row of ``configs/sg5_contrastive_seed.json``, each against its
own reference. COCO-Cap is never used to select: every cell is scored at the
threshold chosen for it on all of VOC-Sim, and the screens' frozen cells are
compared with their references there (paired bootstrap over images).

    python scripts/narmyn/attention/finalize_sg5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.narmyn.attention.an1_propagation_depth import rates  # noqa: E402
from scripts.select_propagation_cv import best_threshold, image_ious, pooled_miou, run_screen  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

CONFIG = ROOT / "configs" / "sg5_contrastive_seed.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "sg5_contrastive_seed"


def load(dataset: str) -> cv.Results | None:
    shards = sorted((OUT / dataset / "shards").glob("*.npz"))
    if not shards:
        return None
    results = cv.load_results(shards)
    if not all(results.complete) or len(set(results.classes)) != 20:
        raise SystemExit(f"{dataset}: shards are incomplete")
    return results


def main() -> None:
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    voc, coco = load("voc"), load("coco")
    if voc is None:
        raise SystemExit("No VOC-Sim shards")
    full = np.ones(len(voc.image_ids), dtype=bool)
    t45 = voc.thresholds.index(0.45)
    voc_classes = np.asarray(voc.classes)
    screens = {s["name"]: run_screen(voc, s["reference"], s["candidates"], s.get("inclusiveness", {}))
               for s in spec["screens"]}
    voc_threshold = {name: best_threshold(voc, voc.index(name), full)[0] for name in voc.configurations}
    cells = {}
    for name in voc.configurations:
        i = voc.index(name)
        cells[name] = {"voc_miou@0.45": pooled_miou(voc.counts[i, :, t45], voc_classes),
                       "voc_full_threshold": voc.thresholds[voc_threshold[name]],
                       "voc_miou@full_threshold": pooled_miou(voc.counts[i, :, voc_threshold[name]], voc_classes),
                       "voc_an1@0.45": rates(voc.counts[i, :, t45], voc_classes)}
    transfer = {}
    if coco is not None:
        coco_classes = np.asarray(coco.classes)
        for name in coco.configurations:
            i, t = coco.index(name), voc_threshold[name]
            cells[name].update({"coco_miou@voc_threshold": pooled_miou(coco.counts[i, :, t], coco_classes),
                                "coco_an1@0.45": rates(coco.counts[i, :, t45], coco_classes)})
        for screen, item in screens.items():
            ref, frozen = item["reference"], item["frozen"]
            a = coco.counts[coco.index(frozen), :, voc_threshold[frozen]]
            b = coco.counts[coco.index(ref), :, voc_threshold[ref]]
            transfer[screen] = {"frozen": frozen, "reference": ref,
                                "coco_miou_frozen": pooled_miou(a, coco_classes), "coco_miou_reference": pooled_miou(b, coco_classes),
                                "paired_image_iou": cv.paired_bootstrap(image_ious(a), image_ious(b))}
    output = {"protocol": "Section 66", "voc_images": len(voc.image_ids), "coco_images": len(coco.image_ids) if coco else 0,
              "screens": screens, "coco_transfer": transfer, "cells": cells}
    (OUT / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = ["# SG5 contrastive CA seed (protocol Section 66)", "",
             f"VOC-Sim {output['voc_images']} images (selection, 5 folds); COCO-Cap {output['coco_images']} images (transfer only).", "",
             "| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Why |", "|---|---|---:|---:|---:|---|---|"]
    for name, item in screens.items():
        lines.append(f"| {name} | `{item['vote_winner']}` | {item['vote_winner_votes']}/5 | {item['oof_miou_reference']:.4f} → "
                     f"{item['oof_miou_selection']:.4f} | {item['oof_gain']:+.4f} | `{item['frozen']}` | {item['freeze_reason']} |")
    if transfer:
        lines += ["", "COCO-Cap transfer of the frozen cells (VOC full-set thresholds):", "",
                  "| Screen | Frozen | COCO frozen | COCO reference | Paired Δ image IoU [95% CI] |", "|---|---|---:|---:|---:|"]
        for name, t in transfer.items():
            p = t["paired_image_iou"]
            lines.append(f"| {name} | `{t['frozen']}` | {t['coco_miou_frozen']:.4f} | {t['coco_miou_reference']:.4f} | "
                         f"{p['mean']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] |")
    lines += ["", "Every cell (AN1 metrics at 0.45: recall / precision):", "",
              "| Cell | VOC mIoU@0.45 | VOC thr | VOC mIoU@thr | VOC R / P | COCO mIoU@VOC thr | COCO R / P |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, c in cells.items():
        v = c["voc_an1@0.45"]
        row = (f"| `{name}` | {c['voc_miou@0.45']:.4f} | {c['voc_full_threshold']:.3f} | {c['voc_miou@full_threshold']:.4f} | "
               f"{v['recall']:.3f} / {v['precision']:.3f} |")
        if "coco_an1@0.45" in c:
            k = c["coco_an1@0.45"]
            row += f" {c['coco_miou@voc_threshold']:.4f} | {k['recall']:.3f} / {k['precision']:.3f} |"
        else:
            row += " – | – |"
        lines.append(row)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(screens) + 2]))


if __name__ == "__main__":
    main()
