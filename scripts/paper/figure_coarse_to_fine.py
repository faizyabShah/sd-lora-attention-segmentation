#!/usr/bin/env python3
"""Quantitative figures of the coarse-to-fine story, from committed count shards (CPU, no inference).

  scale_profiles.pdf   precision and recall across the threshold grid for fine-only (64 px),
                       coarse-only (16 px) and coarse-to-fine (16->64) propagation (SG25), and the
                       share of object vs background mass removed by the 16->32 cap (SG-combined-v3)
  threshold_curves.pdf class-macro mIoU across the grid for FINE-P2 (R0) and V3, raw and +R2,
                       with the per-image best-threshold ceiling (SG-combined-v3, SG-threshold)
  a3_weights.pdf       A3 CA module weights by module and by denoising call (SG24-A)
  supp_sheep_curves.pdf  sheep IoU and false-positive share across the grid (SG-combined-v3)
  supp_crf_classes.pdf   per-class IoU change from DenseCRF on fine-only and capped maps (SG-combined-v3)

    MPLCONFIGDIR=/tmp ~/miniconda3/envs/lab1/bin/python scripts/paper/figure_coarse_to_fine.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs/segmentation/controlled_attention_cv"
FIG = ROOT / "paper/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
                     "xtick.major.width": .6, "ytick.major.width": .6, "legend.frameon": False})
BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#dd8a2e", "#2f9e73", "#8a8f98", "#c2413a"


def load(package, dataset):
    """Concatenated counts [C, N, T, 4], configurations, classes, thresholds."""
    files = [p for p in sorted((CV / package / dataset / "shards").glob("class_*.npz")) if p.name.count(".") == 1]
    parts = [np.load(p) for p in files]
    cfg = [str(x) for x in parts[0]["configurations"]]
    counts = np.concatenate([p["counts"].astype(np.int64) for p in parts], axis=1)
    classes = np.concatenate([p["classes"] for p in parts]).astype(str)
    return counts, cfg, classes, np.round(parts[0]["thresholds"], 3)


def class_miou(c, classes):
    """Class-macro mIoU per threshold; c is [N, T, 4]."""
    vals = []
    for k in sorted(set(classes)):
        s = c[classes == k].sum(0)
        vals.append(s[:, 0] / np.maximum(s[:, :3].sum(1), 1))
    return np.mean(vals, 0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.01)
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def scale_profiles():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    cells = (("FINE-P2", "fine only (64 px)", BLUE, "-"), ("COARSE16-P2", "coarse only (16 px)", ORANGE, "-"),
             ("16-to-64", "coarse $\\rightarrow$ fine", GREEN, "-"))
    for ax, ds, title in ((axes[0], "voc", "VOC-Sim"), (axes[1], "coco", "COCO-Cap")):
        counts, cfg, _, thr = load("sg25", ds)
        for name, label, color, ls in cells:
            s = counts[cfg.index(name)].sum(0).astype(float)
            p, r = s[:, 0] / (s[:, 0] + s[:, 1]), s[:, 0] / (s[:, 0] + s[:, 2])
            ax.plot(r, p, ls, color=color, lw=1.3, label=label)
            ax.plot(r[[0, -1]], p[[0, -1]], "o", color=color, ms=2.4)
        ax.set_xlabel("recall"); ax.set_title(title, fontsize=8)
        ax.grid(alpha=.25, lw=.4)
    axes[0].set_ylabel("precision")
    axes[0].legend(loc="lower left", fontsize=6.4, handlelength=1.4)
    # cap: removed object vs background mass by class
    res = json.loads((CV / "sg_combined_v3/results.json").read_text())
    by = res["diagnostics"]["voc"]["gt_cap_loss_16_to_32"]["R1"]["by_class"]
    names = sorted(by, key=lambda k: by[k]["L_GT"])
    x = np.arange(len(names))
    ax = axes[2]
    ax.bar(x - .2, [100 * by[k]["L_background"] for k in names], .4, color=GREY, label="background mass removed")
    ax.bar(x + .2, [100 * by[k]["L_GT"] for k in names], .4, color=RED, label="object mass removed")
    ax.set_xticks(x); ax.set_xticklabels([k.replace("-", " ") for k in names], rotation=90, fontsize=5.4)
    ax.set_ylabel("% of 32-px proposal mass"); ax.set_title("16$\\rightarrow$32 cap (VOC-Sim)", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout(w_pad=.8)
    save(fig, "scale_profiles")


def threshold_curves():
    tres = json.loads((CV / "sg_threshold/results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=False)
    for ax, ds, title in ((axes[0], "voc", "VOC-Sim"), (axes[1], "coco", "COCO-Cap")):
        counts, cfg, classes, thr = load("sg_combined_v3", ds)
        for name, label, color, ls in (("R0", "FINE-P2", BLUE, "--"), ("R0+R2", "FINE-P2 + R2", BLUE, "-"),
                                       ("R1", "V3", GREEN, "--"), ("R1+R2", "V3 + R2", GREEN, "-")):
            ax.plot(thr, class_miou(counts[cfg.index(name)], classes), ls, color=color, lw=1.2, label=label)
        ceil = tres["ceiling_diagnostic_only"][ds]
        for name, color in (("R0+R2", BLUE), ("R1+R2", GREEN)):
            ax.axhline(ceil[name]["pooled_miou"], color=color, lw=.7, ls=":")
        ax.axvline(.425, color=BLUE, lw=.5, alpha=.5); ax.axvline(.55, color=GREEN, lw=.5, alpha=.5)
        ax.set_title(title, fontsize=7); ax.set_xlabel("global threshold $\\tau$", fontsize=6.5)
        ax.grid(alpha=.25, lw=.4); ax.tick_params(labelsize=5.8); ax.set_xticks([.3, .4, .5, .6, .7])
    axes[0].set_ylabel("class mIoU")
    axes[0].text(.302, tres["ceiling_diagnostic_only"]["voc"]["R1+R2"]["pooled_miou"] - .018,
                 "oracle (per-image $\\tau$)", fontsize=5.2, color=GREY)
    axes[1].legend(fontsize=5.2, loc="lower right", ncol=1, handlelength=1.6)
    fig.tight_layout(w_pad=.4)
    save(fig, "threshold_curves")


def a3_weights():
    res = json.loads((CV / "sg24_a/results.json").read_text())
    rows = sorted(res["layer_weights"]["voc"]["A"], key=lambda w: (-int(w["resolution"]), int(w["index"])))
    selected = json.loads((ROOT / "configs/controlled_sa_seed_frozen.json").read_text())["modules"]
    block = lambda m: "down" if m < 6 else ("mid" if m == 6 else "up")  # noqa: E731
    labels = [f"{block(selected[int(w['index'])])} {w['resolution']} px (m{selected[int(w['index'])]})" for w in rows]
    by_call = np.array([w["mean_by_call"] for w in rows], float)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [1, 2.4]})
    colors = {"64": BLUE, "16": GREEN, "8": ORANGE}
    axes[0].barh(np.arange(len(rows)), [w["mean"] for w in rows], color=[colors[str(w["resolution"])] for w in rows])
    axes[0].axvline(1 / len(rows), color=GREY, lw=.7, ls="--")
    axes[0].text(1 / len(rows), len(rows) - .2, " uniform", fontsize=5.8, color=GREY, va="top")
    axes[0].set_yticks(np.arange(len(rows))); axes[0].set_yticklabels(labels, fontsize=5.8)
    axes[0].invert_yaxis(); axes[0].set_xlabel("mean A3 weight")
    im = axes[1].imshow(by_call, aspect="auto", cmap="viridis")
    axes[1].set_yticks(np.arange(len(rows))); axes[1].set_yticklabels(labels, fontsize=5.8)
    axes[1].set_xlabel("denoising call (0 = first, noisiest)")
    cb = fig.colorbar(im, ax=axes[1], fraction=.03, pad=.01); cb.ax.tick_params(labelsize=5.5)
    cb.set_label("weight", fontsize=6)
    fig.tight_layout(w_pad=.6)
    save(fig, "a3_weights")


def sheep_curves():
    counts, cfg, classes, thr = load("sg_combined_v3", "voc")
    m = classes == "sheep"
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.8))
    for name, label, color, ls in (("R0+R2", "FINE-P2 + R2", BLUE, "-"), ("R1+R2", "V3 + R2", GREEN, "-")):
        c = counts[cfg.index(name)][m].astype(float)
        iou = (c[..., 0] / np.maximum(c[..., :3].sum(-1), 1)).mean(0)
        fp = c[..., 1].sum(0) / (c[..., 0] + c[..., 2]).sum(0)
        axes[0].plot(thr, iou, ls, color=color, lw=1.2, label=label)
        axes[1].plot(thr, fp, ls, color=color, lw=1.2)
    for ax in axes:
        ax.axvline(.425, color=BLUE, lw=.5, alpha=.5); ax.axvline(.55, color=GREEN, lw=.5, alpha=.5)
        ax.set_xlabel("global threshold $\\tau$"); ax.grid(alpha=.25, lw=.4)
    axes[0].set_ylabel("sheep mean image IoU"); axes[1].set_ylabel("false positives / GT area")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "supp_sheep_curves")


def crf_classes():
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.2))
    for ax, ds, title in ((axes[0], "voc", "VOC-Sim"), (axes[1], "coco", "COCO-Cap")):
        counts, cfg, classes, thr = load("sg_combined_v3", ds)
        names = sorted(set(classes))
        t = list(thr)
        def per_class(cell, tau):
            c = counts[cfg.index(cell), :, t.index(tau)]
            return np.array([c[classes == k, 0].sum() / c[classes == k, :3].sum() for k in names])
        fine = per_class("R0+R2", .425) - per_class("R0", .45)
        c2f = per_class("R1+R2", .55) - per_class("R1", .55)
        order = np.argsort(c2f)
        x = np.arange(len(names))
        ax.bar(x - .2, fine[order], .4, color=BLUE, label=f"fine only (mean {fine.mean():+.3f})")
        ax.bar(x + .2, c2f[order], .4, color=GREEN, label=f"coarse-to-fine + cap (mean {c2f.mean():+.3f})")
        ax.axhline(0, color="k", lw=.5)
        ax.set_xticks(x); ax.set_xticklabels([names[i].replace("-", " ") for i in order], rotation=90, fontsize=5.4)
        ax.set_ylabel("IoU change from R2", fontsize=6.5); ax.tick_params(axis="y", labelsize=5.8)
        ax.set_title(title, fontsize=7); ax.legend(fontsize=5.6, loc="upper left")
    fig.tight_layout(h_pad=.6)
    save(fig, "supp_crf_classes")


if __name__ == "__main__":
    scale_profiles(); threshold_curves(); a3_weights(); sheep_curves(); crf_classes()
