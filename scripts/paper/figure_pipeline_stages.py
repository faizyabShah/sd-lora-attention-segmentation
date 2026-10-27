#!/usr/bin/env python3
"""Paper Figure 1: what each stage of the training-free pipeline (FTF) contributes.

Columns are the images chosen by ``scripts/narmyn/attention/select_figure_images.py``
(fixed rule, no image inspected): four VOC-Sim classes from best to worst FTF class IoU,
then three COCO-Cap classes, each the class's median-IoU image. Rows:

  generated image | cross-attention seed c | after two SA passes, mean_i(A_i^2 c) |
  final mask (DenseCRF at 0.45) with the ground-truth outline and the image IoU

Maps come from ``figure_maps.py`` (cache outside the repository). Output:
``paper/figures/pipeline_stages.pdf`` (+ ``.png`` preview).

    python scripts/paper/figure_pipeline_stages.py --cache /path/to/cache
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PALETTE = {"mask": (0.16, 0.47, 0.84), "gt": (1.0, 0.85, 0.1)}


def resize(m: np.ndarray, shape) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    t = torch.as_tensor(m, dtype=torch.float32)[None, None]
    t = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-12)
    return F.interpolate(t, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from scipy import ndimage

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "paper/figures/pipeline_stages")
    p.add_argument("--selection", type=Path, default=ROOT / "outputs/segmentation/paper_figures/selection.json",
                   help="gallery_selection.json for the supplementary failure gallery")
    args = p.parse_args()
    sel = json.loads(args.selection.read_text())
    picks = [("VOC-Sim", x) for x in sel["datasets"]["voc"]] + [("COCO-Cap", x) for x in sel["datasets"]["coco"]]
    rows = ["Generated image", "Cross-attention seed", "+ self-attention (2 passes)", "+ DenseCRF (final)"]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(1.32 * len(picks), 1.38 * len(rows) + 0.25))
    for j, (ds, pick) in enumerate(picks):
        d = np.load(args.cache / f"{pick['image_id']}.npz")
        img, gt = d["image"].astype(np.float32) / 255, d["gt"]
        shape = gt.shape
        if img.shape[:2] != shape:
            from PIL import Image
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(shape[::-1])) / 255
        heat = lambda m: 0.45 * img + 0.55 * cm.magma(resize(m, shape))[..., :3]  # noqa: E731
        mask = d["ftf"]
        over = img.copy()
        over[mask] = 0.45 * over[mask] + 0.55 * np.array(PALETTE["mask"])
        edge = gt ^ ndimage.binary_erosion(gt, iterations=3)
        over[edge] = PALETTE["gt"]
        iou = (mask & gt).sum() / max((mask | gt).sum(), 1)
        tiles = [img, heat(d["seed"]), heat(d["p1"]), over]
        for i, t in enumerate(tiles):
            ax = axes[i, j]
            ax.imshow(np.clip(t, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=6.2, rotation=90, labelpad=3)
        cname = pick["class"].replace("-", " ")
        axes[0, j].set_title(f"{ds}: {cname}", fontsize=6.6, pad=2.5)
        axes[-1, j].set_xlabel(f"IoU {iou:.2f}", fontsize=6.4, labelpad=2)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.05, wspace=0.03, hspace=0.04)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(args.out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
