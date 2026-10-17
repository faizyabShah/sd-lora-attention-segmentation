#!/usr/bin/env python3
"""Summarize the COCO-Cap transfer (phase 3 registry Section 7; protocol Sections 48, 54).

Every stage is read at its VOC-frozen threshold (0.45 for all stages here;
protocol Sections 11 and 51); nothing is chosen on COCO-Cap. Reports pooled
class mIoU on all 707 images and split by whether the caption contains the
class word, and paired bootstraps between stages. Writes ``summary.json``.

    python scripts/narmyn/attention/summarize_coco_transfer.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import controlled_cv as cv  # noqa: E402

PKG = ROOT / "outputs" / "segmentation" / "coco_cap_transfer" / "coco_cap"
CAPTIONS = ROOT / "datasets" / "ovam_experiment_with_dataset" / "evaluation" / "coco_captions_sampled.csv"
FROZEN_THRESHOLD = {"CA/seed": 0.45, "SA/REF": 0.45, "P3/P1": 0.45, "P3/R2": 0.45, "P3/P1+R2": 0.45}
PAIRS = (("SA/REF", "CA/seed"), ("P3/P1", "SA/REF"), ("P3/R2", "SA/REF"), ("P3/P1+R2", "SA/REF"), ("P3/P1+R2", "P3/R2"))


def main() -> None:
    r = cv.load_results(sorted((PKG / "shards").glob("*.npz")))
    if len(r.image_ids) != 707 or not all(r.complete):
        raise SystemExit(f"{len(r.image_ids)} images; complete={all(r.complete)}")
    with CAPTIONS.open(newline="", encoding="utf-8") as handle:
        included = {row["caption_id"]: row["word_included"].strip().lower() == "true" for row in csv.DictReader(handle)}
    classes = np.asarray(r.classes)
    inc = np.array([included[i.split("-")[2]] for i in r.image_ids])
    thresholds = list(r.thresholds)

    def counts(stage):
        return r.counts[r.index(stage), :, thresholds.index(FROZEN_THRESHOLD[stage])]

    def pooled(c, m):
        return float(np.mean([c[m & (classes == k), 0].sum() / c[m & (classes == k), :3].sum() for k in sorted(set(classes[m]))]))

    iou = lambda c: c[:, 0] / np.maximum(c[:, :3].sum(axis=1), 1)  # noqa: E731
    everyone = np.ones(len(classes), bool)
    stages = {s: {"threshold": FROZEN_THRESHOLD[s], "miou_all": pooled(counts(s), everyone),
                  "miou_caption_includes_class": pooled(counts(s), inc), "miou_caption_excludes_class": pooled(counts(s), ~inc),
                  "mean_image_iou": float(iou(counts(s)).mean())} for s in FROZEN_THRESHOLD}
    pairs = [{"stage": a, "versus": b, "delta_miou": stages[a]["miou_all"] - stages[b]["miou_all"],
              "paired_bootstrap": cv.paired_bootstrap(iou(counts(a)), iou(counts(b)))} for a, b in PAIRS]
    out = {"images": len(r.image_ids), "caption_includes_class": int(inc.sum()), "caption_excludes_class": int((~inc).sum()),
           "commits": list(r.commits), "environments": list(r.environments),
           "note": "VOC-frozen thresholds; nothing chosen on COCO-Cap", "stages": stages, "pairs": pairs}
    (PKG / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for s, v in stages.items():
        print(f"{s:10s} {v['miou_all']:.4f}  (includes {v['miou_caption_includes_class']:.4f}, excludes {v['miou_caption_excludes_class']:.4f})")
    for p in pairs:
        b = p["paired_bootstrap"]
        print(f"{p['stage']} vs {p['versus']}: {p['delta_miou']:+.4f}; paired {b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] {b['better']}/{b['worse']}")


if __name__ == "__main__":
    main()
