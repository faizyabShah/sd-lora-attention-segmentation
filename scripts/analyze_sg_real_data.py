#!/usr/bin/env python3
"""SG-real-data analysis: class-macro mIoU (counts pooled per class) per mode x variant at the frozen
tau = 0.55, the best grid tau (diagnostic only: chosen on the same images), and by object size
(thirds of the GT share of valid pixels). For COCO also the 20 VOC-equivalent classes.
Usage: analyze_sg_real_data.py RUN_DIR [voc|coco]"""
import glob, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]); DS = sys.argv[2] if len(sys.argv) > 2 else "voc"
NAMES = ("CA", "C16", "V0", "V1", "V3", "V0+R2", "V3+R2")
LABEL = {"CA": "CA seed (A3)", "C16": "coarse only (16 px)", "V0": "fine only", "V1": "coarse-to-fine, no cap",
         "V3": "BRACE raw", "V0+R2": "fine only + CRF", "V3+R2": "BRACE + CRF"}
THR = np.round(0.30 + 0.025 * np.arange(17), 3); G = 10
VOC20 = {"airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
         "dog", "horse", "motorcycle", "person", "potted plant", "sheep", "couch", "train", "tv"}


def miou(c, k):
    s = np.stack([c[k == j].sum(0) for j in np.unique(k)])
    return (s[..., 0] / np.maximum(s[..., :3].sum(-1), 1)).mean(0)  # [variants, T]


def report(C, K, A, title):
    print(f"\n### {title}: {len(C)} image-class pairs, {len(np.unique(K))} classes")
    m = miou(C, K); small, large = A < np.quantile(A, 1 / 3), A > np.quantile(A, 2 / 3)
    ms, ml = miou(C[small], K[small]), miou(C[large], K[large])
    print("| variant | mIoU @0.55 | best τ (diag.) | mIoU @best | small third @0.55 | large third @0.55 |")
    print("|---|---:|---:|---:|---:|---:|")
    for i, n in enumerate(NAMES):
        b = int(m[i].argmax())
        print(f"| {LABEL[n]} | {m[i, G]:.3f} | {THR[b]:.3f} | {m[i, b]:.3f} | {ms[i, G]:.3f} | {ml[i, G]:.3f} |")


for mode in sorted(p.name for p in D.iterdir() if p.is_dir() and p.name != "tmp"):
    C, K, A = [], [], []
    for f in sorted(glob.glob(str(D / mode / "*.npz"))):
        e = np.load(f)
        for key in sorted({k.rsplit("/", 1)[0] for k in e.files}):
            C.append(e[f"{key}/counts"]); K.append(int(key.split("/")[1])); A.append(float(e[f"{key}/area"]))
    C, K, A = np.array(C), np.array(K), np.array(A)
    print(f"\n## {mode}")
    if DS == "coco":
        names = (D.parent / "coco_classes.txt").read_text().splitlines()
        voc = np.array([names[k - 1] in VOC20 for k in K])
        report(C, K, A, "all 80 classes"); report(C[voc], K[voc], A[voc], "20 VOC-equivalent classes")
    else:
        report(C, K, A, "20 VOC classes")
