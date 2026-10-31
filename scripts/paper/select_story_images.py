#!/usr/bin/env python3
"""Rule-based image choice for the paper's coarse-to-fine figures (no image is looked at).

Pipelines and full-VOC thresholds (frozen for COCO), from committed count shards:
  V3+R2 = SG-combined-v3 R1+R2 at 0.55; V3 = R1 at 0.55; R0+R2 = FINE-P2+R2 at 0.425;
  cap contrast = SG-combined-v2 V3 (0.55) vs V1 (uncapped hierarchy, 0.525), raw.

  stages : VOC classes at the best, upper-third, lower-third and worst V3+R2 class IoU;
           COCO classes at the best, median and worst; per class the image whose V3+R2
           image IoU is the class median (lower median for an even count)
  cap    : the image at the 95th percentile of the V3 - V1 image-IoU change, on VOC and on
           COCO, and the VOC image at the 5th percentile
  crf    : the image at the 90th percentile of the V3+R2 - V3 change, on VOC and on COCO,
           and the VOC image at the 10th percentile
  sheep  : the VOC sheep image at the median V3+R2 - R0+R2 change among sheep images

Writes ``outputs/segmentation/paper_figures/story_selection.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.segmentation import controlled_cv as cv  # noqa: E402

CV = ROOT / "outputs/segmentation/controlled_attention_cv"
OUT = ROOT / "outputs/segmentation/paper_figures/story_selection.json"


def load(package, dataset):
    files = [p for p in sorted((CV / package / dataset / "shards").glob("class_*.npz")) if p.name.count(".") == 1]
    return cv.load_results(files)


def ious(result, name, threshold):
    c = result.counts[result.index(name), :, result.thresholds.index(threshold)]
    return c, c[:, 0] / np.maximum(c[:, :3].sum(1), 1)


def at_quantile(values, q):
    order = np.argsort(values, kind="stable")
    return int(order[int(round(q * (len(order) - 1)))])


def pick(result, i, **extra):
    return {"image_id": result.image_ids[i], "class": result.classes[i], **extra}


def main():
    out = {"rule": __doc__.split("\n\n")[1].strip(), "groups": {}}
    v3 = {ds: load("sg_combined_v3", ds) for ds in ("voc", "coco")}
    v2 = {ds: load("sg_combined_v2", ds) for ds in ("voc", "coco")}
    stages = []
    for ds, quantiles in (("voc", (1.0, 2 / 3, 1 / 3, 0.0)), ("coco", (1.0, 0.5, 0.0))):
        r = v3[ds]
        c, iou = ious(r, "R1+R2", 0.55)
        cls = np.asarray(r.classes)
        class_iou = {k: float(c[cls == k, 0].sum() / c[cls == k, :3].sum()) for k in sorted(set(cls))}
        order = sorted(class_iou, key=class_iou.get)
        for q in quantiles:
            k = order[int(round(q * (len(order) - 1)))]
            idx = np.where(cls == k)[0]
            m = idx[np.argsort(iou[idx], kind="stable")][(len(idx) - 1) // 2]
            stages.append(pick(r, m, dataset=ds, class_quantile=round(q, 3), class_iou=round(class_iou[k], 4),
                               image_iou=round(float(iou[m]), 4)))
    out["groups"]["stages"] = stages
    cap = []
    for ds, q in (("voc", .95), ("coco", .95), ("voc", .05)):
        r = v2[ds]
        _, a = ious(r, "V3", 0.55)
        _, b = ious(r, "V1", 0.525)
        i = at_quantile(a - b, q)
        cap.append(pick(r, i, dataset=ds, quantile=q, v3_iou=round(float(a[i]), 4), v1_iou=round(float(b[i]), 4)))
    out["groups"]["cap"] = cap
    crf = []
    for ds, q in (("voc", .9), ("coco", .9), ("voc", .1)):
        r = v3[ds]
        _, a = ious(r, "R1+R2", 0.55)
        _, b = ious(r, "R1", 0.55)
        i = at_quantile(a - b, q)
        crf.append(pick(r, i, dataset=ds, quantile=q, r2_iou=round(float(a[i]), 4), raw_iou=round(float(b[i]), 4)))
    out["groups"]["crf"] = crf
    r = v3["voc"]
    _, a = ious(r, "R1+R2", 0.55)
    _, b = ious(r, "R0+R2", 0.425)
    sheep = np.where(np.asarray(r.classes) == "sheep")[0]
    i = int(sheep[at_quantile((a - b)[sheep], .5)])
    out["groups"]["sheep"] = [pick(r, i, dataset="voc", quantile=.5, v3_r2_iou=round(float(a[i]), 4),
                                   r0_r2_iou=round(float(b[i]), 4))]
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["groups"], indent=1))


if __name__ == "__main__":
    main()
