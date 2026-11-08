#!/usr/bin/env python3
"""Fresh confirmation set F (2026-11-08): the next 200 VOC train ids in the same seed-137 shuffle that produced S
(S = rest[:300], F = rest[300:500]), so F is disjoint from D and S by construction and has never been captured
or inspected. Folds: the same grouped, class-stratified rule as D/S (prep_sg_real_atlas.folds), seed 3.
Adds manifest["F"] only; D/S/C/G are left byte-identical in content."""
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from PIL import Image

VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
OUT = ROOT / "outputs/segmentation/sg_real_atlas"; RD = ROOT / "outputs/segmentation/sg_real_data"


# copied verbatim from prep_sg_real_atlas.py (importing it would rewrite the manifest)
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

man = json.loads((OUT / "manifest.json").read_text())
train = (VOC / "ImageSets/Segmentation/train.txt").read_text().split()
D = (RD / "voc_train_pilot_ids.txt").read_text().split()
rest = sorted(set(train) - set(D)); rng = random.Random(137); rng.shuffle(rest)
assert sorted(rest[:300]) == sorted(e["image"] for e in man["S"]), "S reconstruction mismatch"
F = sorted(rest[300:500])
assert not set(F) & set(D) and not set(F) & {e["image"] for e in man["S"]}
cls = {i: classes_of(VOC / f"SegmentationClass/{i}.png") for i in F}
fF = folds(F, cls, 3)
man["F"] = [{"set": "F", "image": i, "dataset": "voc", "classes": cls[i], "fold": fF[i], "ddim_qk": False} for i in F]
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print(f"F: {len(F)} photos, {sum(len(c) for c in cls.values())} pairs")
