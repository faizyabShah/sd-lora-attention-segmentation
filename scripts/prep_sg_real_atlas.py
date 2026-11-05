#!/usr/bin/env python3
"""SG-real-atlas image sets and grouped folds (Section 137).

D: the 200 VOC train ids of the Section 136 pilot. S: 300 more VOC train ids (seed 137), disjoint from D,
sealed. C: the Section 136 COCO val200 ids. G: 5 VOC-Sim rows per class (seed 137).
Folds for D and S: five folds grouped by photograph, greedy class stratification (photos with rarer
classes placed first, each into the fold that most lowers the squared deviation of per-class counts).
"""
import json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_coco_cap_transfer as transfer  # noqa: E402

VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"; COCO_LAB = ROOT / "datasets/coco2017/sg_real_data_labels"
OUT = ROOT / "outputs/segmentation/sg_real_atlas"; OUT.mkdir(parents=True, exist_ok=True)
RD = ROOT / "outputs/segmentation/sg_real_data"


def classes_of(path):
    return sorted(int(v) for v in np.unique(np.asarray(Image.open(path))) if v not in (0, 255))


def folds(ids, cls_of, seed):
    rng = random.Random(seed); order = list(ids); rng.shuffle(order)
    freq = {}
    for i in order:
        for k in cls_of[i]:
            freq[k] = freq.get(k, 0) + 1
    order.sort(key=lambda i: min(freq[k] for k in cls_of[i]))
    counts = np.zeros((5, 81)); sizes = np.zeros(5); out = {}
    for i in order:
        def cost(f):
            c = counts.copy(); c[f, cls_of[i]] += 1
            return ((c - c.mean(0)) ** 2).sum() + 0.1 * ((sizes + (np.arange(5) == f)) ** 2).sum()
        f = min(range(5), key=cost); out[i] = f; counts[f, cls_of[i]] += 1; sizes[f] += 1
    return out


train = (VOC / "ImageSets/Segmentation/train.txt").read_text().split()
D = (RD / "voc_train_pilot_ids.txt").read_text().split()
rest = sorted(set(train) - set(D)); rng = random.Random(137); rng.shuffle(rest); S = sorted(rest[:300])
C = (RD / "coco_val200_ids.txt").read_text().split()
voc_cls = {i: classes_of(VOC / f"SegmentationClass/{i}.png") for i in D + S}
coco_cls = {i: classes_of(COCO_LAB / f"{i}.png") for i in C}
G = []
for s in range(20):
    name, rows = transfer.voc_rows(s)
    for r in random.Random(137 + s).sample(rows, 5):
        G.append(r)
fD, fS = folds(D, voc_cls, 1), folds(S, voc_cls, 2)
manifest = {
    "D": [{"set": "D", "image": i, "dataset": "voc", "classes": voc_cls[i], "fold": fD[i],
           "all_qk": n < 50, "ddim_qk": True} for n, i in enumerate(D)],
    "S": [{"set": "S", "image": i, "dataset": "voc", "classes": voc_cls[i], "fold": fS[i], "ddim_qk": False} for i in S],
    "C": [{"set": "C", "image": i, "dataset": "coco", "classes": coco_cls[i], "ddim_qk": False} for i in C],
    "G": [{"set": "G", "image": r["image_id"], "dataset": "vocsim", "classes": [1], "row": r, "ddim_qk": True} for r in G],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
for name in ("D", "S", "C", "G"):
    m = manifest[name]
    pairs = sum(len(e["classes"]) for e in m)
    extra = ""
    if name in ("D", "S"):
        fp = np.bincount([e["fold"] for e in m for _ in e["classes"]], minlength=5)
        extra = f"; pairs per fold {fp.tolist()}"
    print(f"{name}: {len(m)} images, {pairs} pairs{extra}")
