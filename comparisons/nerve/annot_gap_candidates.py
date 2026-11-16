"""Rank items where a confident NERVE region lies outside the ground-truth mask (possible annotation gaps).

Read only: uses saved soft maps and the benchmark annotations, changes nothing. For every item, the 0.5 min-max mask
of the chosen variant is split into connected components. A component counts as "outside" if under 5% of its
pixels fall in the GT mask. Items are ranked by the largest outside component relative to GT area, keeping only items
whose GT recall is >= 0.8 (the object itself is found, so the extra region is less likely to be a plain miss).
The ranked candidates are only a shortlist; each one is checked by eye on its overlay before it is listed.
Overlay panels: image | GT (green) | prediction (red) with the outside components outlined in yellow.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def gt_mask(p):
    a = np.asarray(Image.open(p))
    return (a.sum(-1) if a.ndim == 3 else a) != 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", required=True)
    ap.add_argument("--manifest", required=True, help="comma list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    vd, out = Path(a.variant_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for m in a.manifest.split(",") for r in csv.DictReader(open(m))]
    cand = []
    for r in rows:
        f = vd / f"{r['key']}__soft.npy"
        if not f.exists():
            continue
        s = np.load(f).astype(np.float32)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        pred = s > 0.5
        gt = gt_mask(r["annotation"])
        if gt.sum() == 0:
            continue
        recall = (pred & gt).sum() / gt.sum()
        n, lab = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
        outside = []
        for k in range(1, n):
            comp = lab == k
            if (comp & gt).sum() < 0.05 * comp.sum():
                outside.append(int(comp.sum()))
        big = max(outside) if outside else 0
        cand.append({"key": r["key"], "dataset": r["dataset"], "classname": r["classname"],
                     "recall": float(recall), "gt_area": int(gt.sum()), "largest_outside_px": big,
                     "largest_outside_rel_gt": big / gt.sum(), "n_outside": len(outside),
                     "image": r["image"], "annotation": r["annotation"]})
    ranked = sorted([c for c in cand if c["recall"] >= 0.8 and c["largest_outside_px"] >= 300],
                    key=lambda c: -c["largest_outside_rel_gt"])
    for c in ranked[: a.top]:
        img = np.asarray(Image.open(c["image"]).convert("RGB").resize((512, 512))).astype(np.float32)
        gt = gt_mask(c["annotation"])
        s = np.load(vd / f"{c['key']}__soft.npy").astype(np.float32)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        pred = s > 0.5
        n, lab = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
        outm = np.zeros_like(pred)
        for k in range(1, n):
            comp = lab == k
            if (comp & gt).sum() < 0.05 * comp.sum():
                outm |= comp
        g = img.copy(); g[gt] = 0.45 * g[gt] + 0.55 * np.array([0, 255, 0])
        p = img.copy(); p[pred] = 0.45 * p[pred] + 0.55 * np.array([255, 0, 0])
        cnt, _ = cv2.findContours(outm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        p = np.ascontiguousarray(p.clip(0, 255).astype(np.uint8))
        cv2.drawContours(p, cnt, -1, (255, 255, 0), 2)
        Image.fromarray(np.concatenate([img.astype(np.uint8), g.clip(0, 255).astype(np.uint8), p], 1)).save(
            out / f"{c['key']}__gapcheck.png")
    (out / "candidates.json").write_text(json.dumps({"rule": __doc__, "variant_dir": str(vd),
                                                     "ranked": ranked[: a.top]}, indent=1))
    for c in ranked[: a.top]:
        print(c["key"], c["classname"], "recall=%.2f outside/gt=%.2f n_out=%d" % (
            c["recall"], c["largest_outside_rel_gt"], c["n_outside"]))


if __name__ == "__main__":
    main()
