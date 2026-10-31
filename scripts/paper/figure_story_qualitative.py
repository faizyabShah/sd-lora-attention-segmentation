#!/usr/bin/env python3
"""Qualitative figures of the coarse-to-fine pipeline from the ``figure_story_maps.py`` cache (CPU).

Images are those of ``story_selection.json`` (fixed rules, no image inspected):

  pipeline_c2f.pdf     generated image, A3 CA seed, coarse 16-px SA, cap + fine SA (V3), final
                       mask after DenseCRF (stages group); supp_pipeline_c2f.pdf adds the 32-px
                       proposal and the capped 32-px map
  method_overview.pdf  the pipeline with one image's real intermediate maps and the operators
  cap_mechanism.pdf    uncapped hierarchy vs exact mass conservation vs one-sided cap (cap group)
  crf_effect.pdf       V3 map, its thresholded mask and the DenseCRF mask (crf group)
  supp_ca_seeds.pdf    per-resolution CA, uniform and A3-weighted seeds (stages group)
  supp_sheep_case.pdf  FINE-P2+R2 vs V3+R2 on the median sheep image (sheep group)

    MPLCONFIGDIR=/tmp ~/miniconda3/envs/lab1/bin/python scripts/paper/figure_story_qualitative.py \
        --cache /home/shafa/paper_cache/story_maps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper/figures"
SELECTION = ROOT / "outputs/segmentation/paper_figures/story_selection.json"
MASK, GT, REMOVED = np.array((0.16, 0.47, 0.84)), np.array((1.0, 0.85, 0.1)), np.array((0.85, 0.15, 0.15))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})


def resize(m, shape, normalize=True):
    t = torch.as_tensor(np.asarray(m, np.float32))[None, None]
    if normalize:
        t = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-12)
    return F.interpolate(t, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()


def erode(mask, k=3):
    m = mask.copy()
    for _ in range(k):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


class Item:
    def __init__(self, cache, image_id):
        self.d = np.load(cache / f"{image_id}.npz")
        self.gt = self.d["gt"]
        self.shape = self.gt.shape
        img = self.d["image"]
        if img.shape[:2] != self.shape:
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize(self.shape[::-1]))
        self.img = img.astype(np.float32) / 255

    def heat(self, key, cmap="magma"):
        return .42 * self.img + .58 * getattr(cm, cmap)(resize(self.d[key], self.shape))[..., :3]

    def mask(self, key, outline=True):
        m = self.d[key].astype(bool)
        over = self.img.copy()
        over[m] = .45 * over[m] + .55 * MASK
        if outline:
            over[self.gt ^ erode(self.gt)] = GT
        return over, float((m & self.gt).sum() / max((m | self.gt).sum(), 1))

    def outlined(self):
        over = self.img.copy()
        over[self.gt ^ erode(self.gt)] = GT
        return over

    def removed(self):
        r = np.clip(resize(self.d["removed32"], self.shape, normalize=False), 0, None)
        r = r / max(r.max(), 1e-12)
        return np.clip(self.img * (1 - .85 * r[..., None]) + .85 * r[..., None] * REMOVED, 0, 1)


def grid(rows, cols, w, h):
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    return fig, axes


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.01)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def label(item):
    ds = "VOC-Sim" if item["dataset"] == "voc" else "COCO-Cap"
    return f"{ds}\n{item['class'].replace('-', ' ')}"


def teaser(cache, picks, keys, rows, name):
    fig, axes = grid(len(rows), len(picks), 1.0, 1.0)
    for j, p in enumerate(picks):
        it = Item(cache, p["image_id"])
        over, iou = it.mask("crf_V3")
        tiles = [it.img] + [it.heat(k) for k in keys] + [over]
        for i, t in enumerate(tiles):
            axes[i, j].imshow(np.clip(t, 0, 1))
            if j == 0:
                axes[i, j].set_ylabel(rows[i], fontsize=6.3)
        axes[0, j].set_title(label(p), fontsize=6.2, pad=2)
        axes[-1, j].set_xlabel(f"IoU {iou:.2f}", fontsize=6.2, labelpad=1.5)
    fig.subplots_adjust(wspace=.03, hspace=.04)
    save(fig, name)


def overview(cache, pick):
    it = Item(cache, pick["image_id"])
    fig = plt.figure(figsize=(7.0, 1.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(.5, 3.5); ax.axis("off")
    over, _ = it.mask("crf_V3")
    panels = [(it.img, "generated image\n+ class word $w$"), (it.heat("seed_a3"), "CA seed $c$\n(A3, 8/16/64 px)"),
              (it.heat("r16"), "coarse SA 16 px\n$r_{16}=\\bar A^{16}c_{16}$"),
              (it.heat("q32"), "32-px proposal\n$q=\\bar A^{32}U r_{16}$"),
              (it.heat("z32"), "one-sided cap\n$z=\\mathrm{cap}(r_{16},q)$"),
              (it.heat("V3"), "fine SA 64 px\n$\\bar A^{64}U z$"), (over, "threshold\n+ DenseCRF")]
    x0, w, gap = .1, 1.72, .27
    for k, (tile, text) in enumerate(panels):
        x = x0 + k * (w + gap)
        ax.imshow(np.clip(tile, 0, 1), extent=(x, x + w, 1.35, 1.35 + w), zorder=2)
        ax.add_patch(FancyBboxPatch((x - .04, 1.31), w + .08, w + .08, boxstyle="round,pad=0.02", lw=.8,
                                    ec="#38546b", fc="none", zorder=3))
        ax.text(x + w / 2, 1.18, text, ha="center", va="top", fontsize=6.3)
        if k:
            ax.add_patch(FancyArrowPatch((x - gap + .02, 1.35 + w / 2), (x - .06, 1.35 + w / 2),
                                         arrowstyle="-|>", mutation_scale=8, lw=.9, color="#3a6075", zorder=4))
    ax.text(x0 + 1 * (w + gap) + w / 2, 3.32, "semantic seed", ha="center", fontsize=6.8, weight="bold", color="#30465d")
    ax.text(x0 + 3.5 * (w + gap) + w / 2, 3.32, "coarse $\\rightarrow$ fine propagation, calls 10–20",
            ha="center", fontsize=6.8, weight="bold", color="#30465d")
    ax.text(x0 + 6 * (w + gap) + w / 2, 3.32, "boundary", ha="center", fontsize=6.8, weight="bold", color="#30465d")
    save(fig, "method_overview")


def cap_mechanism(cache, picks):
    cols = ["image + GT", "uncapped\nhierarchy", "exact mass\nconservation", "one-sided cap\n(ours)", "mass removed\nby the cap"]
    fig, axes = grid(len(picks), len(cols), 1.25, 1.18)
    for i, p in enumerate(picks):
        it = Item(cache, p["image_id"])
        tiles = [(it.outlined(), None), it.mask("mask_V1"), it.mask("mask_C4"), it.mask("mask_V3"), (it.removed(), None)]
        for j, (t, iou) in enumerate(tiles):
            axes[i, j].imshow(np.clip(t, 0, 1))
            if iou is not None:
                axes[i, j].set_xlabel(f"IoU {iou:.2f}", fontsize=6.2, labelpad=1.5)
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=6.6, pad=2)
        axes[i, 0].set_ylabel(label(p), fontsize=6.2)
    fig.subplots_adjust(wspace=.03, hspace=.18)
    save(fig, "cap_mechanism")


def crf_effect(cache, picks):
    cols = ["image + GT", "V3 map", "V3 mask ($\\tau$=0.55)", "+ DenseCRF"]
    fig, axes = grid(len(picks), len(cols), 1.2, 1.15)
    for i, p in enumerate(picks):
        it = Item(cache, p["image_id"])
        tiles = [(it.outlined(), None), (it.heat("V3"), None), it.mask("mask_V3"), it.mask("crf_V3")]
        for j, (t, iou) in enumerate(tiles):
            axes[i, j].imshow(np.clip(t, 0, 1))
            if iou is not None:
                axes[i, j].set_xlabel(f"IoU {iou:.2f}", fontsize=6.2, labelpad=1.5)
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=6.6, pad=2)
        axes[i, 0].set_ylabel(label(p), fontsize=6.2)
    fig.subplots_adjust(wspace=.03, hspace=.18)
    save(fig, "crf_effect")


def ca_seeds(cache, picks):
    cols = ["image + GT", "CA 8 px", "CA 16 px", "CA 64 px", "uniform seed", "A3 seed"]
    fig, axes = grid(len(picks), len(cols), 1.05, 1.05)
    for i, p in enumerate(picks):
        it = Item(cache, p["image_id"])
        tiles = [it.outlined(), it.heat("ca_8"), it.heat("ca_16"), it.heat("ca_64"), it.heat("seed_a0"), it.heat("seed_a3")]
        for j, t in enumerate(tiles):
            axes[i, j].imshow(np.clip(t, 0, 1))
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=6.6, pad=2)
        axes[i, 0].set_ylabel(label(p), fontsize=6)
    fig.subplots_adjust(wspace=.03, hspace=.05)
    save(fig, "supp_ca_seeds")


def sheep_case(cache, pick):
    it = Item(cache, pick["image_id"])
    cols = ["image + GT", "FINE-P2 map", "FINE-P2 + R2 ($\\tau$=0.425)", "V3 map", "V3 + R2 ($\\tau$=0.55)"]
    fig, axes = grid(1, len(cols), 1.3, 1.3)
    tiles = [(it.outlined(), None), (it.heat("R0"), None), it.mask("crf_R0"), (it.heat("V3"), None), it.mask("crf_V3")]
    for j, (t, iou) in enumerate(tiles):
        axes[0, j].imshow(np.clip(t, 0, 1)); axes[0, j].set_title(cols[j], fontsize=6.6, pad=2)
        if iou is not None:
            axes[0, j].set_xlabel(f"IoU {iou:.2f}", fontsize=6.2, labelpad=1.5)
    fig.subplots_adjust(wspace=.03)
    save(fig, "supp_sheep_case")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=Path("/home/shafa/paper_cache/story_maps"))
    args = p.parse_args()
    groups = json.loads(SELECTION.read_text())["groups"]
    teaser(args.cache, groups["stages"], ("seed_a3", "r16", "V3"),
           ["Generated image", "CA seed $c$", "coarse SA (16 px)", "cap + fine SA", "+ DenseCRF"], "pipeline_c2f")
    teaser(args.cache, groups["stages"], ("seed_a3", "r16", "q32", "z32", "V3"),
           ["Generated image", "CA seed $c$", "coarse SA (16 px)", "32-px proposal", "after the cap",
            "fine SA (64 px)", "+ DenseCRF"], "supp_pipeline_c2f")
    overview(args.cache, groups["stages"][1])
    cap_mechanism(args.cache, groups["cap"])
    crf_effect(args.cache, groups["crf"])
    ca_seeds(args.cache, groups["stages"])
    sheep_case(args.cache, groups["sheep"][0])


if __name__ == "__main__":
    main()
