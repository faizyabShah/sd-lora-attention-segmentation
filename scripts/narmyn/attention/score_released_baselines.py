#!/usr/bin/env python3
"""Score the released baseline masks and heatmaps of OVAM's public archive (registry Q1-Q5).

Each method is scored on its own released images and annotations, with the
procedure of OVAM's evaluation notebooks (github.com/vpulab/ovam,
``experiments/evaluation/<method>/evaluation_<method>.ipynb``, commit
``12afd87``):

  DAAM            heatmap channel 1, bilinear to 512, / max, > 0.4 (token-optimized: > 0.8)
  Attn2Mask       heatmap[0][-2] (optimized: [0, 1]), min-max, background 1 - h - beta
                  (beta 0.2; optimized -0.2), bilinear, argmax; DenseCRF on COCO-Cap only
  Grounded Diff.  released binary masks (gd / gdopt)
  DatasetDM       released binary masks (masks / masks_optimized)
  OVAM            released DenseCRF masks (``ovamsa_crf_mask``; token-optimized on VOC-Sim only)

Metric: per-class pooled IoU (sum of intersections / sum of unions), macro mean
over classes, as in the notebooks and ``src/segmentation/metrics.py``. COCO-Cap
is also split by whether the caption contains the class word, as OVAM reports.

    python scripts/narmyn/attention/score_released_baselines.py --crf-site <dir with denseCRF>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402

EVAL = ROOT / "datasets" / "ovam_experiment_with_dataset" / "evaluation"


def binary(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    return (a.sum(axis=-1) if a.ndim == 3 else a) != 0


def bilinear(maps: np.ndarray) -> np.ndarray:
    """``torch.nn.functional.interpolate`` to 512x512, as in the notebooks ([C, 64, 64] or [64, 64])."""
    t = torch.from_numpy(np.ascontiguousarray(maps, dtype=np.float32))
    t = t[None, None] if t.ndim == 2 else t[None]
    return torch.nn.functional.interpolate(t, size=(512, 512), mode="bilinear")[0].squeeze(0).numpy()


def daam_mask(path: Path, threshold: float) -> np.ndarray:
    heat = bilinear(np.load(path, allow_pickle=False)[1])
    return heat / heat.max() > threshold


def attn2mask_mask(path: Path, optimized: bool, image_path: Path | None, crf_site: Path | None) -> np.ndarray:
    raw = np.load(path, allow_pickle=False)
    heat = raw[0, 1] if optimized else raw[0][-2]
    heat = (heat - heat.min()) / (heat.max() - heat.min())
    beta = -0.2 if optimized else 0.2
    mask = bilinear(np.stack([1 - heat - beta, heat])).argmax(axis=0).astype(bool)
    if image_path is not None:
        mask = apply_ovam_densecrf(np.asarray(Image.open(image_path).convert("RGB")), mask, crf_site)
    return mask


def coco_included() -> dict[int, bool]:
    with (EVAL / "coco_captions_sampled.csv").open(newline="", encoding="utf-8") as handle:
        return {int(r["caption_id"]): r["word_included"].strip().lower() == "true" for r in csv.DictReader(handle)}


def items(method: str, dataset: str, crf_site: Path | None):
    """Yield (image key, class, caption id or None, annotation path, {variant: prediction callable})."""
    if dataset == "voc_sim":
        if method == "daam":
            for a in sorted((EVAL / "ovam/voc_sim/annotations").iterdir()):  # byte-identical to DAAM's copy
                cls, model, seed, _ = a.stem.split("_")
                stem = f"{cls}_{model}_{seed}"
                h = EVAL / "daam/voc_sim/heatmaps"
                yield stem, cls, None, a, {"training_free": lambda h=h, s=stem: daam_mask(h / f"{s}_heatmap_token0.npy", 0.4),
                                           "token_optimized": lambda h=h, s=stem: daam_mask(h / f"{s}_heatmap_token1.npy", 0.8)}
        elif method == "attn2mask":
            for a in sorted((EVAL / "attn2mask/voc_sim/annotations").iterdir()):
                model, cls, seed, _ = a.stem.split("_")
                stem = f"{model}_{cls}_{seed}"
                h = EVAL / "attn2mask/voc_sim/voc_sim_fixed"
                yield stem, cls, None, a, {
                    "training_free": lambda h=h, s=stem: attn2mask_mask(h / f"{s}_heatmap.npy", False, None, None),
                    "token_optimized": lambda h=h, s=stem: attn2mask_mask(h / f"{s}_heatmapopt.npy", True, None, None)}
        elif method == "grounded_diffusion":
            for a in sorted((EVAL / "grounded_diffusion/voc_sim/annotations").iterdir()):
                cls, model, seed, _ = a.stem.split("_")
                stem = f"{cls.replace(' ', '-')}_{model}_{seed}"
                d = EVAL / "grounded_diffusion/voc_sim"
                yield stem, cls, None, a, {"training_free": lambda d=d, s=stem: binary(d / f"masks/{s}_gd_mask.png"),
                                           "token_optimized": lambda d=d, s=stem: binary(d / f"masks_optimized/{s}_gdopt_mask.png")}
        elif method == "datasetdm":
            for a in sorted((EVAL / "datasetdm/voc_sim/annotations").iterdir()):
                cls = a.stem.split("_")[0]
                d = EVAL / "datasetdm/voc_sim"
                yield a.stem, cls, None, a, {"training_free": lambda d=d, n=a.name: binary(d / f"masks/{n}"),
                                             "token_optimized": lambda d=d, n=a.name: binary(d / f"masks_optimized/{n}")}
        elif method == "ovam":
            with (EVAL / "ovam/voc_sim/ovam_voc_sim.csv").open(newline="", encoding="utf-8") as handle:
                rows = [r for r in csv.DictReader(handle) if r["annotated"].lower() == "true"]
            d = EVAL / "ovam_sa/voc_sim"
            for r in rows:
                stem = Path(r["annotation"]).stem.removesuffix("_annotation")
                yield stem, r["classname"], None, EVAL / "ovam/voc_sim" / r["annotation"], {
                    "training_free": lambda d=d, s=stem: binary(d / f"mask/{s}_ovamsa_crf_mask.png"),
                    "token_optimized": lambda d=d, s=stem: binary(d / f"mask_optimized/{s}_ovamsa_crf_mask.png")}
        return
    # COCO-Cap
    if method in ("daam", "ovam"):
        for a in sorted((EVAL / "ovam/coco_captions/annotations").iterdir()):  # byte-identical to DAAM's copy (707)
            cls, model, _, caption, _, seed = a.stem.split("_")
            stem = f"{cls}_{model.replace('-', '')}_caption{caption}_seed{seed}"
            if method == "daam":
                h = EVAL / "daam/coco_captions/heatmaps"
                yield stem, cls, int(caption), a, {
                    "training_free": lambda h=h, s=stem: daam_mask(h / f"{s}_heatmap_classname.npy", 0.4),
                    "token_optimized": lambda h=h, s=stem: daam_mask(h / f"{s}_heatmap_token1.npy", 0.8)}
            else:
                m = EVAL / "ovam_sa/coco_cap/mask"
                yield stem, cls, int(caption), a, {"training_free": lambda m=m, s=stem: binary(m / f"{s}_ovamsa_crf_mask.png")}
    elif method == "attn2mask":
        d = EVAL / "attn2mask/coco_cap"
        for a in sorted((d / "annotations").iterdir()):
            model, cls, caption, seed, _ = a.stem.split("_")
            stem = f"{model.replace('-', '')}_{cls}_{caption}_{seed}"
            image = d / f"images/{stem}_image.png"
            yield stem, cls, int(caption.replace("caption", "")), a, {
                "training_free": lambda d=d, s=stem, i=image: attn2mask_mask(d / f"heatmaps/{s}_heatmap.npy", False, i, crf_site),
                "token_optimized": lambda d=d, s=stem, i=image: attn2mask_mask(d / f"heatmaps_optimized/{s}_heatmapopt.npy", True, i, crf_site)}
    elif method == "grounded_diffusion":
        d = EVAL / "grounded_diffusion/coco_captions"
        for a in sorted((d / "annotations").iterdir()):
            cls, model, caption, seed, _ = a.stem.split("_")
            stem = f"{cls}_{model}_{caption}_{seed}"
            yield stem, cls, int(caption.replace("caption", "")), a, {
                "training_free": lambda d=d, s=stem: binary(d / f"masks/{s}_gd_mask.png"),
                "token_optimized": lambda d=d, s=stem: binary(d / f"masks_optimized/{s}_gdopt_mask.png")}
    elif method == "datasetdm":
        d = EVAL / "datasetdm/coco_captions"
        for a in sorted((d / "annotations").iterdir()):
            cls, _, _, caption, _ = a.stem.split("_")
            yield a.stem, cls, int(caption), a, {"training_free": lambda d=d, n=a.name: binary(d / f"masks/{n}"),
                                                 "token_optimized": lambda d=d, n=a.name: binary(d / f"masks_optimized/{n}")}


def summarize(rows: list[dict]) -> dict:
    by_class = defaultdict(lambda: [0, 0])
    for r in rows:
        by_class[r["class"]][0] += r["intersection"]
        by_class[r["class"]][1] += r["union"]
    ious = {c: i / u for c, (i, u) in sorted(by_class.items()) if u}
    return {"images": len(rows), "classes": len(ious), "miou": float(np.mean(list(ious.values()))) if ious else float("nan"),
            "mean_image_iou": float(np.mean([r["iou"] for r in rows])) if rows else float("nan"), "class_iou": ious}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crf-site", type=Path, help="directory holding SimpleCRF's denseCRF (Attn2Mask on COCO-Cap)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/segmentation/comparisons/released_baselines")
    args = parser.parse_args()
    included = coco_included()
    out, per_image = {}, []
    for dataset in ("voc_sim", "coco_cap"):
        for method in ("ovam", "daam", "attn2mask", "grounded_diffusion", "datasetdm"):
            rows = defaultdict(list)
            for key, cls, caption, annotation, variants in items(method, dataset, args.crf_site):
                target = binary(annotation)
                for variant, predict in variants.items():
                    pred = predict()
                    if pred.shape != target.shape:
                        raise ValueError(f"{method}/{key}: shape {pred.shape} vs {target.shape}")
                    i, u = int((pred & target).sum()), int((pred | target).sum())
                    row = {"dataset": dataset, "method": method, "variant": variant, "image": key, "class": cls.replace(" ", "-"),
                           "caption_id": caption, "caption_includes_class": None if caption is None else included[caption],
                           "intersection": i, "union": u, "iou": i / u if u else 1.0}
                    rows[variant].append(row)
                    per_image.append(row)
            for variant, rs in rows.items():
                entry = {"all": summarize(rs)}
                if dataset == "coco_cap":
                    entry["caption_includes_class"] = summarize([r for r in rs if r["caption_includes_class"]])
                    entry["caption_excludes_class"] = summarize([r for r in rs if r["caption_includes_class"] is False])
                out.setdefault(dataset, {}).setdefault(method, {})[variant] = entry
                print(f"{dataset:8s} {method:18s} {variant:16s} images={entry['all']['images']:4d} mIoU={entry['all']['miou']:.4f}"
                      + (f"  included={entry['caption_includes_class']['miou']:.4f} excluded={entry['caption_excludes_class']['miou']:.4f}"
                         if dataset == "coco_cap" else ""), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source": "OVAM public archive (www-vpu.eps.uam.es/publications/ovam/ovam_experiment_with_dataset.zip)",
            "procedure": "OVAM evaluation notebooks, github.com/vpulab/ovam commit 12afd87",
            "note": "Each method is scored on its own released images and annotations; only DAAM and OVAM share ours."}
    (args.output_dir / "summary.json").write_text(json.dumps({**meta, "results": out}, indent=2) + "\n")
    with (args.output_dir / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)


if __name__ == "__main__":
    main()
