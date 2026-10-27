#!/usr/bin/env python3
"""Paper figure: how far should evidence propagate? One, two and three SA passes.

Left: images chosen by ``scripts/narmyn/attention/figure_passes_maps.py`` (fixed rule from the
SG22 all-call raw cells). Columns: generated image with the ground-truth outline; raw mask after one, two
and three observation-wise passes at threshold 0.45 (image IoU below); first-arrival depth of every
accepted pixel (pass 1 / 2 / 3), with the ground-truth outline.

Right: true-positive rate of the pixels first accepted at each depth, pooled over all images, from the
SG25 first-arrival diagnostic (A3 seed, calls 10-20, threshold 0.425, depths 1-5; a different setting
from the left panel, stated in the caption).

    python scripts/paper/figure_passes.py --cache /path/to/cache                     # main: two rows
    python scripts/paper/figure_passes.py --cache /path/to/cache --out paper/figures/supp_propagation_depth \
        --keys voc_completion voc_leakage coco_completion coco_leakage            # supplementary
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MASK, GT = (0.16, 0.47, 0.84), (1.0, 0.85, 0.1)
DEPTH = [(0.16, 0.47, 0.84), (0.93, 0.55, 0.0), (0.84, 0.15, 0.16)]  # first accepted at pass 1, 2, 3
TITLES = {"voc_completion": "VOC-Sim: completion", "voc_leakage": "VOC-Sim: leakage",
          "coco_completion": "COCO-Cap: completion", "coco_leakage": "COCO-Cap: leakage"}


def arrival_rates(ds: str) -> np.ndarray:
    """Pooled TP / pixels for pixels first accepted at depth 1..5 (SG25 sidecars)."""
    tot = 0
    for f in sorted(glob.glob(str(ROOT / f"outputs/segmentation/controlled_attention_cv/sg25/{ds}/shards/*.arrival.npz"))):
        tot = tot + np.load(f)["counts"].sum(0)
    return tot[:, 1] / np.maximum(tot[:, 0], 1)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from scipy import ndimage

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "paper/figures/propagation_depth")
    p.add_argument("--keys", nargs="+", default=["voc_completion", "coco_leakage"],
                   help="rows to show (the supplementary uses all four)")
    args = p.parse_args()
    picks = json.loads((ROOT / "outputs/segmentation/paper_figures/passes_selection.json").read_text())["picks"]
    sel = {k: picks[k] for k in args.keys}

    def full(m, shape):
        t = torch.as_tensor(m, dtype=torch.float32)[None, None]
        t = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-12)
        return F.interpolate(t, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()

    fig = plt.figure(figsize=(7.0, 1.05 * len(sel) + 0.55))
    gs = fig.add_gridspec(len(sel), 7, width_ratios=[1, 1, 1, 1, 1, 0.42, 1.75], wspace=0.03, hspace=0.06)
    cols = ["Generated image", "1 pass", "2 passes", "3 passes", "First accepted at"]
    for r, (key, pick) in enumerate(sel.items()):
        d = np.load(args.cache / f"passes_{pick['image_id']}.npz")
        img, gt = d["image"].astype(np.float32) / 255, d["gt"]
        if img.shape[:2] != gt.shape:
            from PIL import Image
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(gt.shape[::-1])) / 255
        edge = gt ^ ndimage.binary_erosion(gt, iterations=3)
        masks = [full(d[f"p{k}"], gt.shape) >= 0.45 for k in (1, 2, 3)]
        tiles, ious = [img.copy()], [None]
        tiles[0][edge] = GT
        for m in masks:
            t = img.copy()
            t[m] = 0.45 * t[m] + 0.55 * np.array(MASK)
            t[edge] = GT
            tiles.append(t); ious.append((m & gt).sum() / max((m | gt).sum(), 1))
        arr = 0.35 * img + 0.65
        seen = np.zeros_like(gt)
        for k, m in enumerate(masks):
            first = m & ~seen
            arr[first] = DEPTH[k]
            seen |= m
        arr[edge] = (0.05, 0.05, 0.05)
        tiles.append(arr); ious.append(None)
        for c, (t, iou) in enumerate(zip(tiles, ious)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(np.clip(t, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=6.4, pad=2.5)
            if c == 0:
                ax.set_ylabel(f"{TITLES[key]}\n({pick['class'].replace('_', ' ')})", fontsize=5.8, labelpad=2)
            if iou is not None:
                ax.text(0.03, 0.04, f"{iou:.2f}", transform=ax.transAxes, fontsize=5.6, color="white",
                        bbox=dict(boxstyle="square,pad=0.15", fc=(0, 0, 0, 0.55), lw=0))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=DEPTH[k], label=f"pass {k + 1}") for k in range(3)] +
               [Patch(facecolor="white", edgecolor=GT, lw=1.2, label="ground truth")],
               loc="lower left", ncol=4, fontsize=5.8, frameon=False, bbox_to_anchor=(0.03, -0.005))

    ax = fig.add_subplot(gs[:, 6])
    depths = np.arange(1, 6)
    for ds, col, off, lab in (("voc", "#2a78d6", -0.18, "VOC-Sim"), ("coco", "#eda100", 0.18, "COCO-Cap")):
        rate = arrival_rates(ds)
        ax.bar(depths + off, rate, width=0.34, color=col, label=lab)
        for x, y in zip(depths + off, rate):
            ax.text(x, y + 0.012, f"{100 * y:.0f}", ha="center", fontsize=5.0)
    ax.set_xticks(depths); ax.set_ylim(0, 1); ax.set_xlabel("first-arrival depth (passes)", fontsize=6.2)
    ax.set_ylabel("TP rate of newly accepted pixels", fontsize=6.2, labelpad=1)
    ax.tick_params(labelsize=5.8); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.8, frameon=False, loc="upper right")
    ax.set_title("All images (SG25)", fontsize=6.4, pad=2.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(args.out.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
