"""Build the three paper figures from committed, provenance-tracked evidence.

No new evaluation is performed. The numeric points are transcribed from the
SG22 same-generation report; the CA tiles are cropped from the archived,
common-colour-scale module visualization.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False})

# Process diagram: conceptual representation of the implemented operations.
fig, ax = plt.subplots(figsize=(11.6, 2.3))
ax.set_xlim(0, 11.6); ax.set_ylim(0, 2.3); ax.axis("off")
nodes = [
    (0.15, "Class word\nattribution"),
    (2.35, "CA seed\n8, 16, 64 px"),
    (4.65, "SA module 15\n64 px"),
    (6.95, "Two passes\nper observation"),
    (9.25, "Threshold\n+ DenseCRF"),
]
colors = ["#e7eef8", "#d9e9f5", "#d9ebed", "#e2e8f6", "#e6f2e8"]
for j, ((x, label), color) in enumerate(zip(nodes, colors)):
    ax.add_patch(FancyBboxPatch((x, 0.77), 1.95, 0.96, boxstyle="round,pad=0.06",
                                linewidth=1.25, edgecolor="#38546b", facecolor=color))
    ax.text(x + 0.975, 1.25, label, ha="center", va="center", weight="semibold", fontsize=10)
    if j:
        ax.add_patch(FancyArrowPatch((x - 0.22, 1.25), (x - 0.02, 1.25),
                                     arrowstyle="-|>", mutation_scale=13, lw=1.4, color="#3a6075"))
ax.text(5.8, 2.10, "Frozen on VOC-Sim folds", ha="center", fontsize=10, color="#30465d")
ax.text(5.8, 0.35, "Transferred to COCO-Cap without selecting a new operator or threshold",
        ha="center", fontsize=9, color="#50677b")
fig.savefig(FIG / "method_overview.pdf", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# Common-scale source, all head-averaged maps, all calls, airplane example.
source = ROOT / "outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution/figures/ovam-voc-sim-0_ca_modules.png"
im = Image.open(source).convert("RGB")
assert im.size == (1650, 1155)
tiles = [
    ((19, 61, 271, 313), "Generated image", None),
    ((19, 336, 271, 588), "Annotation", None),
    ((835, 61, 1087, 313), "64 px: module 13", "retained"),
    ((1107, 336, 1359, 588), "32 px: module 11", "excluded"),
    ((291, 611, 543, 863), "16 px: module 4", "retained"),
    ((291, 886, 543, 1138), "8 px: module 6", "retained"),
]
fig, axes = plt.subplots(1, 6, figsize=(12.8, 2.65))
for ax, (box, title, status) in zip(axes, tiles):
    ax.imshow(im.crop(box)); ax.axis("off")
    ax.set_title(title, fontsize=9, pad=5)
    if status: ax.text(.5, -.075, status, transform=ax.transAxes, ha="center",
                       fontsize=8, color=("#207157" if status == "retained" else "#a44732"))
fig.subplots_adjust(wspace=.055, left=.01, right=.99, bottom=.14, top=.86)
fig.savefig(FIG / "ca_airplane.pdf", bbox_inches="tight", pad_inches=.04)
plt.close(fig)

# Same-study scores from SG22 report, not a sum of independent ablations.
stages = ["One pass", "Two passes", "+ DenseCRF", "Calls 10–20\n+ DenseCRF"]
scores = [.7873, .7997, .8132, .8168]
fig, ax = plt.subplots(figsize=(5.25, 2.55))
ax.plot(range(4), scores, color="#286492", lw=2.1, marker="o", ms=6)
for i, score in enumerate(scores):
    ax.annotate(f"{score:.3f}", (i, score), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color="#1e425e")
ax.set_xticks(range(4), stages)
ax.set_ylim(.778, .831); ax.set_xlim(-.2, 3.2)
ax.set_ylabel("VOC-Sim OOF mIoU")
ax.grid(axis="y", alpha=.23)
fig.tight_layout()
fig.savefig(FIG / "validated_progression.pdf", bbox_inches="tight", pad_inches=.04)
plt.close(fig)

# Diagnostic extremes of the independent SG18 one-pass raw-map temporal study.
# A gain and a loss illustrate the tradeoff without presenting one favorable
# image as the mechanism or as the final two-pass/CRF result.
examples = [
    ("06_largest_gain", "bird: middle helps", (.486, .763)),
    ("07_largest_loss", "person: middle hurts", (.592, .480)),
]
fig, axes = plt.subplots(2, 5, figsize=(11.3, 4.8))
for row, (stem, label, ious) in enumerate(examples):
    sheet = Image.open(ROOT / "outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed/visuals" / f"{stem}.png").convert("RGB")
    boxes = [(7, 55, 595, 654), (607, 55, 1195, 654),
             (1207, 55, 1795, 654), (1807, 55, 2395, 654),
             (1207, 681, 1795, 1261)]
    titles = ["Generated image", "CA seed", f"All calls, IoU {ious[0]:.3f}",
              f"Calls 10–20, IoU {ious[1]:.3f}", "Annotation"]
    for col, (box, title) in enumerate(zip(boxes, titles)):
        ax = axes[row, col]
        ax.imshow(sheet.crop(box)); ax.axis("off")
        ax.set_title(title, fontsize=8.7, pad=4)
    axes[row, 0].text(-.04, .5, label, transform=axes[row, 0].transAxes,
                      rotation=90, va="center", ha="right", fontsize=9, weight="bold")
fig.subplots_adjust(wspace=.035, hspace=.13, left=.09, right=.995, top=.93, bottom=.02)
fig.savefig(FIG / "temporal_tradeoff.pdf", bbox_inches="tight", pad_inches=.05)
plt.close(fig)

# Direct same-study per-class differences in the SG22 window comparison.
with (ROOT / "outputs/segmentation/controlled_attention_cv/sg22/per_class_iou.csv").open() as handle:
    records = list(csv.DictReader(handle))
fig, axes = plt.subplots(1, 2, figsize=(8.9, 5.25), sharey=True)
for ax, dataset, title in zip(axes, ("voc", "coco"), ("VOC-Sim", "COCO-Cap")):
    ref = {r["class"]: float(r["iou"]) for r in records
           if r["dataset"] == dataset and r["method"] == "ALL-P2+R2"}
    mid = {r["class"]: float(r["iou"]) for r in records
           if r["dataset"] == dataset and r["method"] == "MID-P2+R2"}
    assert len(ref) == len(mid) == 20
    names = sorted(ref, key=lambda name: name.replace("-", " "))
    changes = [mid[name] - ref[name] for name in names]
    ax.barh(range(20), changes,
            color=["#247d67" if value >= 0 else "#b35b46" for value in changes], height=.72)
    ax.axvline(0, lw=.8, color="#51606c")
    ax.set_yticks(range(20), [name.replace("-", " ") for name in names], fontsize=7.4)
    ax.invert_yaxis(); ax.set_xlim(-.03, .035)
    ax.set_title(title, fontsize=10, weight="semibold")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
axes[1].tick_params(axis="y", labelleft=False)
fig.supxlabel("Class IoU change: calls 10–20 minus calls 0–30 (P2+R2)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "temporal_class_effects.pdf", bbox_inches="tight", pad_inches=.04)
plt.close(fig)
