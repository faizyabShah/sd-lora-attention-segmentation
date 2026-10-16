#!/usr/bin/env python3
"""DAAM with its mask threshold chosen inside our VOC-Sim folds (registry Q2, rule R3).

DAAM's released heatmaps share our images and annotations (byte-identical), so
DAAM can be selected exactly like our methods: in each outer fold the threshold
is chosen on the other four folds and scored on the held-out fold. DAAM's own
binarization is kept (channel 1, bilinear to 512, divide by the maximum,
``> t``); only ``t`` is chosen. The grid is 0.30-0.90 in steps of 0.025, wider
than R3's 0.30-0.70 so that it contains the 0.8 that OVAM uses for the
token-optimized heatmaps. COCO-Cap is never used to choose: it is scored with
the threshold chosen on all 490 VOC-Sim images, and with OVAM's fixed values.

    python scripts/narmyn/attention/daam_fold_threshold.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation import controlled_cv as cv  # noqa: E402

EVAL = ROOT / "datasets" / "ovam_experiment_with_dataset" / "evaluation"
OUT = ROOT / "outputs" / "segmentation" / "comparisons" / "daam_fold_threshold"
THRESHOLDS = tuple(round(0.30 + 0.025 * i, 3) for i in range(25))
OVAM_FIXED = {"training_free": 0.4, "token_optimized": 0.8}


def heat512(path: Path) -> torch.Tensor:
    h = torch.from_numpy(np.load(path, allow_pickle=False)[1].astype(np.float32))[None, None]
    h = torch.nn.functional.interpolate(h, size=(512, 512), mode="bilinear")[0, 0]
    return h / h.max()


def counts_for(heat: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    out = np.zeros((len(THRESHOLDS), 4), np.int64)
    for i, t in enumerate(THRESHOLDS):
        p = heat > t
        tp = int((p & target).sum()); fp = int((p & ~target).sum()); fn = int((~p & target).sum())
        out[i] = (tp, fp, fn, target.numel() - tp - fp - fn)
    return out


def binary(path: Path) -> torch.Tensor:
    a = np.asarray(Image.open(path))
    return torch.from_numpy(((a.sum(axis=-1) if a.ndim == 3 else a) != 0))


def pooled(counts: np.ndarray, classes: np.ndarray) -> float:
    return float(np.mean([counts[classes == c, 0].sum() / counts[classes == c, :3].sum() for c in sorted(set(classes))]))


def voc_rows():
    with (EVAL / "ovam/voc_sim/ovam_voc_sim.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["annotated"].lower() == "true"]
    for r in rows:
        stem = Path(r["annotation"]).stem.removesuffix("_annotation")
        yield r["image_id"], r["classname"], EVAL / "ovam/voc_sim" / r["annotation"], {
            "training_free": EVAL / f"daam/voc_sim/heatmaps/{stem}_heatmap_token0.npy",
            "token_optimized": EVAL / f"daam/voc_sim/heatmaps/{stem}_heatmap_token1.npy"}


def coco_rows():
    for a in sorted((EVAL / "ovam/coco_captions/annotations").iterdir()):
        cls, model, _, caption, _, seed = a.stem.split("_")
        stem = f"{cls}_{model.replace('-', '')}_caption{caption}_seed{seed}"
        yield stem, cls, a, {"training_free": EVAL / f"daam/coco_captions/heatmaps/{stem}_heatmap_classname.npy",
                             "token_optimized": EVAL / f"daam/coco_captions/heatmaps/{stem}_heatmap_token1.npy"}


def score(rows) -> tuple[list, list, dict[str, np.ndarray]]:
    ids, classes, counts = [], [], {"training_free": [], "token_optimized": []}
    for key, cls, annotation, heatmaps in rows:
        target = binary(annotation)
        ids.append(key)
        classes.append(cls.replace(" ", "-"))
        for variant, path in heatmaps.items():
            counts[variant].append(counts_for(heat512(path), target))
    return ids, classes, {k: np.stack(v) for k, v in counts.items()}


def main() -> None:
    folds = cv.read_folds(ROOT / "configs" / "voc_sim_folds.csv")
    v_ids, v_cls, v_counts = score(voc_rows())
    c_ids, c_cls, c_counts = score(coco_rows())
    v_cls, c_cls = np.asarray(v_cls), np.asarray(c_cls)
    fold = np.array([folds[i] for i in v_ids])
    iou = lambda k: k[:, 0] / np.maximum(k[:, :3].sum(axis=1), 1)  # noqa: E731
    out = {"thresholds": list(THRESHOLDS), "grid_note": "0.30-0.90 (R3 grid extended to include OVAM's 0.8)", "variants": {}}
    for variant in ("training_free", "token_optimized"):
        counts = v_counts[variant]
        fixed_i = THRESHOLDS.index(OVAM_FIXED[variant])
        oof = np.zeros((len(v_ids), 4), np.int64)
        per_fold = []
        for k in sorted(set(fold.tolist())):
            train, held = fold != k, fold == k
            t = max(range(len(THRESHOLDS)), key=lambda i: (pooled(counts[train, i], v_cls[train]), -abs(THRESHOLDS[i] - 0.45)))
            oof[held] = counts[held, t]
            per_fold.append({"fold": k, "threshold": THRESHOLDS[t], "train_miou": pooled(counts[train, t], v_cls[train]),
                             "heldout_miou": pooled(counts[held, t], v_cls[held]),
                             "heldout_miou_ovam_fixed": pooled(counts[held, fixed_i], v_cls[held])})
        full_t = max(range(len(THRESHOLDS)), key=lambda i: (pooled(counts[:, i], v_cls), -abs(THRESHOLDS[i] - 0.45)))
        fixed = counts[:, fixed_i]
        out["variants"][variant] = {
            "voc_sim": {"images": len(v_ids), "ovam_fixed_threshold": OVAM_FIXED[variant], "miou_ovam_fixed": pooled(fixed, v_cls),
                        "oof_miou_fold_chosen": pooled(oof, v_cls),
                        "paired_bootstrap_fold_chosen_vs_fixed": cv.paired_bootstrap(iou(oof), iou(fixed)),
                        "folds": per_fold, "full_threshold": THRESHOLDS[full_t], "miou_full_threshold_in_sample": pooled(counts[:, full_t], v_cls),
                        "miou_by_threshold": {f"{t:.3f}": pooled(counts[:, i], v_cls) for i, t in enumerate(THRESHOLDS)}},
            "coco_cap": {"images": len(c_ids), "miou_ovam_fixed": pooled(c_counts[variant][:, fixed_i], c_cls),
                         "threshold_from_voc": THRESHOLDS[full_t], "miou_threshold_from_voc": pooled(c_counts[variant][:, full_t], c_cls)}}
        v, c = out["variants"][variant]["voc_sim"], out["variants"][variant]["coco_cap"]
        print(f"{variant}: VOC fixed {OVAM_FIXED[variant]} {v['miou_ovam_fixed']:.4f} | fold-chosen OOF {v['oof_miou_fold_chosen']:.4f} "
              f"(thresholds {[f['threshold'] for f in per_fold]}) | COCO fixed {c['miou_ovam_fixed']:.4f}, VOC-chosen {c['threshold_from_voc']} -> {c['miou_threshold_from_voc']:.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    np.savez_compressed(OUT / "counts.npz", thresholds=np.array(THRESHOLDS), voc_ids=np.array(v_ids), voc_classes=v_cls,
                        voc_folds=fold, coco_ids=np.array(c_ids), coco_classes=c_cls,
                        **{f"voc_{k}": v for k, v in v_counts.items()}, **{f"coco_{k}": v for k, v in c_counts.items()})


if __name__ == "__main__":
    main()
