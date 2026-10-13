#!/usr/bin/env python3
"""Summarise run_token_crf.py output: mIoU, pooled precision/recall, paired bootstrap tests.

Reads outputs/segmentation/narmyn_token_crf/<dataset>/class_*.jsonl and, for
VOC-Sim, the committed scores of the authors' released OVAM masks
(outputs/segmentation/ovam_paper_baseline). Writes report_<dataset>.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RELEASED = {
    "released OVAM (unoptimized)": "summary.json",
    "released OVAM (token-optimized)": "optimized_summary.json",
}
PAIRS = (
    ("plain_top4/sa_iter1", "ovam_implemented"),
    ("opt_top4/sa_iter1", "ovam_implemented"),
    ("opt_top4/sa_iter1+crf", "ovam_implemented"),
    ("opt_top4/sa_iter1", "plain_top4/sa_iter1"),
    ("opt_all16/sa_iter1", "opt_top4/sa_iter1"),
    ("plain_top4/sa_iter1+crf", "plain_top4/sa_iter1"),
    ("opt_top4/sa_iter1+crf", "opt_top4/sa_iter1"),
    ("opt_top4/sa_iter4", "plain_top4/sa_iter4"),
    ("opt_top4/sa_iter4+crf", "opt_top4/sa_iter4"),
    ("plain_top4/sa_iter1", "released OVAM (token-optimized)"),
    ("opt_top4/sa_iter1", "released OVAM (token-optimized)"),
    ("opt_top4/sa_iter1+crf", "released OVAM (token-optimized)"),
    ("opt_all16/sa_iter1+crf", "released OVAM (token-optimized)"),
    ("opt_top4/sa_iter4+crf", "released OVAM (token-optimized)"),
)
CLASS_TABLE = (
    "ovam_implemented", "plain_top4/sa_iter1", "plain_top4/sa_iter1+crf",
    "opt_top4/sa_iter1", "opt_top4/sa_iter1+crf", "opt_all16/sa_iter1+crf",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("voc_sim", "coco_cap"))
    parser.add_argument("--root", type=Path, default=Path("outputs/segmentation/narmyn_token_crf"))
    parser.add_argument("--released", type=Path, default=Path("outputs/segmentation/ovam_paper_baseline"))
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = args.root / args.dataset
    recs = [json.loads(line) for f in sorted(folder.glob("class_*.jsonl")) for line in f.open()]
    complete = len(list(folder.glob("*.COMPLETE")))
    classes = sorted({r["classname"] for r in recs})
    ids = [r["image_id"] for r in recs]
    keys = sorted(recs[0]["results"])

    released = {}
    if args.dataset == "voc_sim":
        for name, file in RELEASED.items():
            d = json.loads((args.released / file).read_text())
            per = {p["image_id"]: p["iou"] for p in d["per_image"]}
            released[name] = (d, np.array([per[i] for i in ids]))

    def pooled(key, subset=None):
        c = defaultdict(lambda: [0, 0, 0])
        for r in subset if subset is not None else recs:
            v = r["results"][key]
            c[r["classname"]][0] += v["tp"]
            c[r["classname"]][1] += v["fp"]
            c[r["classname"]][2] += v["fn"]
        per = {k: tp / (tp + fp + fn) for k, (tp, fp, fn) in c.items()}
        tp, fp, fn = (sum(v[i] for v in c.values()) for i in range(3))
        return per, float(np.mean(list(per.values()))), tp / (tp + fp), tp / (tp + fn)

    per_image = {k: np.array([r["results"][k]["iou"] for r in recs]) for k in keys}
    title = "VOC-Sim" if args.dataset == "voc_sim" else "COCO-Cap"
    lines = [
        f"# {title}: OVAM optimized token and DenseCRF on the earlier final method",
        "",
        f"{len(recs)} images, {len(classes)} classes ({complete}/20 shards complete). "
        "Exploratory, pre-protocol; all settings frozen, nothing selected here.",
        "",
        "| Method | mIoU | Mean image IoU | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (d, arr) in released.items():
        miou = np.mean([d["per_class"][c]["iou"] for c in classes])
        lines.append(f"| {name} | {miou:.4f} | {arr.mean():.4f} | — | — |")
    for k in keys:
        _, miou, precision, recall = pooled(k)
        lines.append(f"| `{k}` | {miou:.4f} | {per_image[k].mean():.4f} | {precision:.3f} | {recall:.3f} |")

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(recs), size=(20000, len(recs)))
    lines += ["", "## Paired per-image IoU differences (bootstrap over images, 20,000 resamples)", "",
              "| A | B | Mean(A−B) | 95% CI | A better / B better |", "|---|---|---:|---|---|"]
    for a, b in PAIRS:
        if not all(k in per_image or k in released for k in (a, b)):
            continue
        x = per_image[a] if a in per_image else released[a][1]
        y = per_image[b] if b in per_image else released[b][1]
        diff = x - y
        lo, hi = np.quantile(diff[idx].mean(1), [0.025, 0.975])
        lines.append(f"| `{a}` | `{b}` | {diff.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {(diff > 0).sum()} / {(diff < 0).sum()} |")

    if args.dataset == "coco_cap":
        lines += ["", "## Class word present in the caption?", "",
                  "| Method | Present | Absent |", "|---|---:|---:|"]
        for k in CLASS_TABLE:
            cells = []
            for flag in ("True", "False"):
                subset = [r for r in recs if str(r.get("word_included")) == flag]
                cells.append(f"{pooled(k, subset)[1]:.4f} ({len(subset)})")
            lines.append(f"| `{k}` | " + " | ".join(cells) + " |")

    show = list(released) + list(CLASS_TABLE)
    lines += ["", "## Pooled class IoU", "",
              "| Class | " + " | ".join(s.replace("released OVAM", "rel. OVAM") for s in show) + " |",
              "|---|" + "---:|" * len(show)]
    for c in classes:
        cells = [f"{released[s][0]['per_class'][c]['iou']:.3f}" if s in released else f"{pooled(s)[0][c]:.3f}"
                 for s in show]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "Released-OVAM rows are the authors' masks on their original renders; every other row uses our "
              "regenerated images (same prompts and seeds) scored on the same released annotations."]
    out = args.root / f"report_{args.dataset}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
