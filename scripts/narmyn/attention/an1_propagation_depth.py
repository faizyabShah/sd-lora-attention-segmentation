#!/usr/bin/env python3
"""AN1 (registry Section 11): why does a second SA pass help, and when does it stop helping?

Analysis only; nothing is selected. Every number comes from committed per-image
counts:

  propagation package   seed (CHECK/ca_seed), one pass (REF) and 2-4 passes, VOC-Sim
  token package         class word vs OVAM token: seed, one and two passes, VOC-Sim
  COCO-Cap transfers    class word and token: seed, one and two passes

Questions:
  1. error decomposition by pass: recall (object filled) vs precision (leakage)
  2. does the second-pass gain depend on how much of the object the seed covers?
  3. does the token's seed cover more, explaining why the second pass stops helping?
  4. which classes and object sizes carry the gain?

Thresholds: 0.45 (the frozen CA/SA threshold) for every cell, so that passes are
compared on equal terms; the per-pass best in-sample threshold is also reported.
Writes ``outputs/segmentation/analysis/an1_propagation_depth/summary.json``.

    python scripts/narmyn/attention/an1_propagation_depth.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import controlled_cv as cv  # noqa: E402

CV = ROOT / "outputs" / "segmentation" / "controlled_attention_cv"
TRANSFER = ROOT / "outputs" / "segmentation" / "coco_cap_transfer"
OUT = ROOT / "outputs" / "segmentation" / "analysis" / "an1_propagation_depth"


def load(path: Path) -> cv.Results:
    return cv.load_results(sorted((path / "shards").glob("*.npz")))


def at(r: cv.Results, name: str, t: float = 0.45) -> np.ndarray:
    return r.counts[r.index(name), :, list(r.thresholds).index(t)].astype(np.int64)


def pooled_miou(c: np.ndarray, classes: np.ndarray) -> float:
    return float(np.mean([c[classes == k, 0].sum() / c[classes == k, :3].sum() for k in sorted(set(classes))]))


def rates(c: np.ndarray, classes: np.ndarray) -> dict:
    """Pooled per-class recall, precision and FP relative to the object area, macro-averaged."""
    rec, pre, fpr, iou = [], [], [], []
    for k in sorted(set(classes)):
        tp, fp, fn = (c[classes == k, j].sum() for j in range(3))
        rec.append(tp / (tp + fn)); pre.append(tp / max(tp + fp, 1)); fpr.append(fp / (tp + fn)); iou.append(tp / (tp + fp + fn))
    return {"miou": float(np.mean(iou)), "recall": float(np.mean(rec)), "precision": float(np.mean(pre)),
            "fp_per_object_pixel": float(np.mean(fpr))}


def image_iou(c: np.ndarray) -> np.ndarray:
    return c[:, 0] / np.maximum(c[:, :3].sum(axis=1), 1)


def image_recall(c: np.ndarray) -> np.ndarray:
    return c[:, 0] / np.maximum(c[:, 0] + c[:, 2], 1)


def coverage_analysis(seed: np.ndarray, one: np.ndarray, two: np.ndarray) -> dict:
    """Second-pass gain per image against the seed's recall (how much of the object the seed covers)."""
    gain = image_iou(two) - image_iou(one)
    sr = image_recall(seed)
    q = np.quantile(sr, [1 / 3, 2 / 3])
    bins = {"low": sr <= q[0], "mid": (sr > q[0]) & (sr <= q[1]), "high": sr > q[1]}
    return {"seed_recall_mean": float(sr.mean()), "seed_recall_terciles": [float(x) for x in q],
            "second_pass_gain_mean": float(gain.mean()),
            # scipy assigns average ranks to ties.  Nested argsort does not and
            # can therefore change slightly across NumPy sorting implementations.
            "spearman_seed_recall_vs_gain": float(spearmanr(sr, gain).statistic),
            "gain_by_seed_recall_tercile": {k: {"images": int(m.sum()), "mean_gain": float(gain[m].mean()),
                                                "share_improved": float((gain[m] > 0).mean())} for k, m in bins.items()}}


def size_analysis(one: np.ndarray, two: np.ndarray) -> dict:
    area = one[:, 0] + one[:, 2]  # object pixels (tp + fn) at annotation resolution
    gain = image_iou(two) - image_iou(one)
    q = np.quantile(area, [1 / 3, 2 / 3])
    bins = {"small": area <= q[0], "medium": (area > q[0]) & (area <= q[1]), "large": area > q[1]}
    return {"area_terciles_pixels": [float(x) for x in q],
            "gain_by_object_size": {k: {"images": int(m.sum()), "mean_gain": float(gain[m].mean())} for k, m in bins.items()}}


def main() -> None:
    out = {"threshold": 0.45, "note": "analysis only; nothing selected"}

    # 1. Error decomposition by pass (propagation package, A100, class word).
    p = load(CV / "propagation_package")
    pc = np.asarray(p.classes)
    cells = {"seed": "CHECK/ca_seed", "1 pass": "REF/one_pass", "2 passes": "P1/obs_n2", "3 passes": "P1/obs_n3", "4 passes": "P1/obs_n4"}
    thresholds = list(p.thresholds)
    out["by_pass_voc_word"] = {}
    for label, name in cells.items():
        best = max(range(len(thresholds)), key=lambda t: pooled_miou(p.counts[p.index(name), :, t], pc))
        out["by_pass_voc_word"][label] = {**rates(at(p, name), pc), "best_threshold_in_sample": thresholds[best],
                                          "miou_at_best_threshold": pooled_miou(p.counts[p.index(name), :, best], pc)}
    one, two = at(p, "REF/one_pass"), at(p, "P1/obs_n2")
    out["per_class_second_pass_voc_word"] = {
        k: {"delta_recall": float(two[pc == k, 0].sum() / (two[pc == k, 0] + two[pc == k, 2]).sum()
                                  - one[pc == k, 0].sum() / (one[pc == k, 0] + one[pc == k, 2]).sum()),
            "delta_iou": float(two[pc == k, 0].sum() / two[pc == k, :3].sum() - one[pc == k, 0].sum() / one[pc == k, :3].sum())}
        for k in sorted(set(pc))}
    out["size_voc_word"] = size_analysis(one, two)

    # 2-3. Seed coverage and the second-pass gain: class word vs token (token package, V100), and COCO-Cap.
    tk = load(CV / "tokens")
    out["coverage"] = {
        "voc_word": coverage_analysis(at(tk, "W/seed"), at(tk, "W/REF"), at(tk, "W/P1")),
        "voc_token": coverage_analysis(at(tk, "TC/seed"), at(tk, "TC/REF"), at(tk, "TC/P1")),
    }
    c_word, c_tok = load(TRANSFER / "coco_cap"), load(TRANSFER / "coco_cap_token")
    out["coverage"]["coco_word"] = coverage_analysis(at(c_word, "CA/seed"), at(c_word, "SA/REF"), at(c_word, "P3/P1"))
    out["coverage"]["coco_token"] = coverage_analysis(at(c_tok, "T/seed"), at(c_tok, "T/REF"), at(c_tok, "T/P1"))
    for key, (r, s, o, t2) in {"voc_token": (tk, "TC/seed", "TC/REF", "TC/P1"), "coco_word": (c_word, "CA/seed", "SA/REF", "P3/P1"),
                               "coco_token": (c_tok, "T/seed", "T/REF", "T/P1")}.items():
        cls = np.asarray(r.classes)
        out.setdefault("by_pass_other", {})[key] = {lab: rates(at(r, n), cls) for lab, n in (("seed", s), ("1 pass", o), ("2 passes", t2))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    print("1. VOC-Sim, class word, by pass (threshold 0.45):")
    for k, v in out["by_pass_voc_word"].items():
        print(f"   {k:9s} mIoU {v['miou']:.4f} recall {v['recall']:.4f} precision {v['precision']:.4f} FP/object {v['fp_per_object_pixel']:.3f} "
              f"| best threshold {v['best_threshold_in_sample']} -> {v['miou_at_best_threshold']:.4f}")
    for key, v in out["by_pass_other"].items():
        print(f"   {key:10s} " + " | ".join(f"{k}: mIoU {x['miou']:.3f} R {x['recall']:.3f} P {x['precision']:.3f}" for k, x in v.items()))
    print("2-3. Seed coverage vs second-pass gain:")
    for k, v in out["coverage"].items():
        g = v["gain_by_seed_recall_tercile"]
        print(f"   {k:10s} seed recall {v['seed_recall_mean']:.3f}; mean gain {v['second_pass_gain_mean']:+.4f}; Spearman {v['spearman_seed_recall_vs_gain']:+.3f}; "
              f"gain low/mid/high coverage {g['low']['mean_gain']:+.4f} / {g['mid']['mean_gain']:+.4f} / {g['high']['mean_gain']:+.4f}")
    s = out["size_voc_word"]["gain_by_object_size"]
    print(f"4. VOC word, second-pass gain by object size: small {s['small']['mean_gain']:+.4f}, medium {s['medium']['mean_gain']:+.4f}, large {s['large']['mean_gain']:+.4f}")
    top = sorted(out["per_class_second_pass_voc_word"].items(), key=lambda kv: -kv[1]["delta_iou"])
    print("   classes (Δ IoU, Δ recall):", [(k, round(v["delta_iou"], 3), round(v["delta_recall"], 3)) for k, v in top[:4]], "...",
          [(k, round(v["delta_iou"], 3), round(v["delta_recall"], 3)) for k, v in top[-3:]])


if __name__ == "__main__":
    main()
