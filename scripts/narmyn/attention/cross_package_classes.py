#!/usr/bin/env python3
"""Cross-package per-class analysis of the SG/Track-A results (analysis only; nothing is selected).

Every comparison is taken **within one package** (candidate and reference from the
same run), from the committed per-class tables, so no environment drift enters.
Per-class IoU deltas are computed from the stored IoUs (candidate - reference), on
VOC-Sim (each cell at its full-VOC threshold) and COCO-Cap (transfer).

Two groups:
  raw     operators on the propagated map, against the one-pass REF (or P1 where stated)
  final   full pipelines with DenseCRF, against the final pipeline of that package

Writes ``outputs/segmentation/analysis/cross_package_classes/{summary.json,report.md,deltas.csv,class_deltas.{png,pdf}}``.

    python scripts/narmyn/attention/cross_package_classes.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
B = ROOT / "outputs" / "segmentation" / "controlled_attention_cv"
OUT = ROOT / "outputs" / "segmentation" / "analysis" / "cross_package_classes"

# label, group, package file, candidate cell, reference cell
COMPARISONS = [
    ("P1 two passes", "raw", "sg12_crf/per_class_results.csv", "W/P1", "W/REF"),
    ("SG2 consensus graph", "raw", "sg2_densecrf_confirmation/per_class_results.csv", "W/SG2", "W/REF"),
    ("SG8 sign consensus", "raw", "sg11_sg12_observation_reliability/per_class_results.csv", "W/C/rho0p80", "W/REF"),
    ("SG11 robust consensus", "raw", "sg11_sg12_observation_reliability/per_class_results.csv", "W/SG11/trim20/rho0p80", "W/REF"),
    ("SG12 top-50% observations", "raw", "sg11_sg12_observation_reliability/per_class_results.csv", "W/SG12/top50", "W/REF"),
    ("SG13 region labels (R1)", "raw", "sg13_target_region_inference/per_class_results.csv", "W/SG13/R1", "W/REF"),
    ("SG18 middle calls (T1) vs P1", "raw", "sg18_t1_crf/per_class_results.csv", "W/T1", "W/P1"),
    ("SG2 + CRF vs FTF", "final", "sg2_densecrf_confirmation/per_class_results.csv", "W/SG2+R2", "W/P1+R2"),
    ("SG12 + CRF vs FTF", "final", "sg12_crf/per_class_results.csv", "W/SG12/top50+R2", "W/P1+R2"),
    ("T1 + CRF vs FTF", "final", "sg18_t1_crf/per_class_results.csv", "W/T1+R2", "W/P1+R2"),
    ("SG22 MID-P2 + CRF vs FTF", "final", "sg22/per_class_iou.csv", "MID-P2+R2", "ALL-P2+R2"),
    ("SG24 A3 vs A0 (both MID-P2 + CRF)", "final", "sg24_a/per_class_iou.csv", "A3/FINAL", "A0/FINAL"),
    ("SG25 16->64 + CRF vs fine", "final", "sg25/per_class_iou.csv", "16-to-64+R2", "FINE-P2+R2"),
    ("SG26 S5 + CRF vs S0", "final", "sg26/per_class_iou.csv", "S5+R2", "S0+R2"),
]


def load(path: Path, cell: str) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(path.open()))
    key = "cell" if "cell" in rows[0] else "method"
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r[key] == cell:
            out.setdefault(r["dataset"], {})[r["class"]] = float(r["iou"])
    if not out:
        raise SystemExit(f"{path}: no rows for {cell}")
    return out


def main() -> None:
    results, classes = [], None
    for label, group, f, cand, ref in COMPARISONS:
        a, b = load(B / f, cand), load(B / f, ref)
        entry = {"label": label, "group": group, "source": f"outputs/segmentation/controlled_attention_cv/{f}",
                 "candidate": cand, "reference": ref}
        for ds in ("voc", "coco"):
            if ds in a and ds in b:
                cls = sorted(set(a[ds]) & set(b[ds]))
                classes = classes or cls
                d = {c: a[ds][c] - b[ds][c] for c in cls}
                entry[ds] = {"mean_delta": float(np.mean(list(d.values()))), "per_class": d,
                             "improved": sum(v > 0 for v in d.values())}
        results.append(entry)
    voc_classes = sorted(results[0]["voc"]["per_class"])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "deltas.csv").open("w", newline="") as h:
        w = csv.writer(h, lineterminator="\n")
        w.writerow(["comparison", "group", "dataset"] + voc_classes + ["mean"])
        for e in results:
            for ds in ("voc", "coco"):
                if ds in e:
                    pc = e[ds]["per_class"]
                    w.writerow([e["label"], e["group"], ds] + [f"{pc.get(c, float('nan')):.4f}" for c in voc_classes] + [f"{e[ds]['mean_delta']:.4f}"])
    # Class consistency: how often each class gains / loses across comparisons.
    consistency = {}
    for ds in ("voc", "coco"):
        rows = [e[ds]["per_class"] for e in results if ds in e]
        names = sorted(set().union(*rows))
        consistency[ds] = {c: {"gains": sum(r.get(c, 0) > 0.005 for r in rows), "losses": sum(r.get(c, 0) < -0.005 for r in rows),
                               "mean": float(np.mean([r[c] for r in rows if c in r]))} for c in names}
    # VOC -> COCO agreement of per-class deltas, per comparison.
    for e in results:
        if "voc" in e and "coco" in e:
            common = [c for c in e["voc"]["per_class"] if c.replace("-", " ") in {k.replace("-", " ") for k in e["coco"]["per_class"]}]
            cmap = {k.replace("-", " "): v for k, v in e["coco"]["per_class"].items()}
            x = np.array([e["voc"]["per_class"][c] for c in common]); y = np.array([cmap[c.replace("-", " ")] for c in common])
            e["voc_coco_class_correlation"] = float(np.corrcoef(x, y)[0, 1]) if len(common) > 2 and x.std() > 0 and y.std() > 0 else None
    summary = {"note": "analysis of committed per-class tables; comparisons within one package; nothing selected",
               "comparisons": results, "class_consistency": consistency}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    labels = [e["label"] for e in results]
    for ax, ds in zip(axes, ("voc", "coco")):
        mat = np.array([[e[ds]["per_class"].get(c, e[ds]["per_class"].get(c.replace("-", " "), np.nan)) if ds in e else np.nan
                         for c in voc_classes] for e in results])
        im = ax.imshow(mat, cmap="RdBu", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(voc_classes)), [c.replace("potted-plant", "plant").replace("dining-table", "table") for c in voc_classes],
                      rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.axhline(sum(e["group"] == "raw" for e in results) - 0.5, color="k", lw=0.8)
        ax.set_title({"voc": "VOC-Sim (selection set)", "coco": "COCO-Cap (transfer)"}[ds], fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Δ class IoU (candidate − reference)")
    fig.savefig(OUT / "class_deltas.png", dpi=170, bbox_inches="tight"); fig.savefig(OUT / "class_deltas.pdf", bbox_inches="tight")

    lines = ["# Cross-package per-class analysis", "", "Comparisons within one package, from the committed per-class tables; nothing is selected.", "",
             "| Comparison | Group | VOC mean Δ | VOC classes improved | COCO mean Δ | COCO classes improved | VOC–COCO class correlation |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for e in results:
        v, c = e.get("voc"), e.get("coco")
        r = e.get("voc_coco_class_correlation")
        lines.append(f"| {e['label']} | {e['group']} | {v['mean_delta']:+.4f} | {v['improved']}/20 | "
                     + (f"{c['mean_delta']:+.4f} | {c['improved']}/20 |" if c else "– | – |") + (f" {r:+.2f} |" if r is not None else " – |"))
    for ds in ("voc", "coco"):
        cons = consistency[ds]
        top = sorted(cons.items(), key=lambda kv: -kv[1]["mean"])
        lines += ["", f"Classes on {ds.upper()}, by mean Δ over all {len([e for e in results if ds in e])} comparisons (gains / losses = comparisons with Δ > +0.005 / < −0.005):", "",
                  "| Class | Mean Δ | Gains | Losses |", "|---|---:|---:|---:|"]
        lines += [f"| {k} | {x['mean']:+.4f} | {x['gains']} | {x['losses']} |" for k, x in top]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5 + len(results)]))


if __name__ == "__main__":
    main()
