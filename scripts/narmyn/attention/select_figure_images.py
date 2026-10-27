#!/usr/bin/env python3
"""Rule-based image choice for the paper's qualitative figures (no image is looked at).

Pipeline: FTF (two observation-wise SA passes + OVAM DenseCRF) at its frozen threshold
0.45, from the committed SG5 package (``W/P1+R2``; same environment as the regeneration).

  VOC-Sim  : the classes at the best, upper-third, lower-third and worst FTF class IoU
  COCO-Cap : the classes at the best, median and worst FTF class IoU
  per class: the image whose FTF image IoU is the median of that class (lower median for
             an even count)

Writes ``outputs/segmentation/paper_figures/selection.json``. With ``--gallery`` (supplementary
failure gallery): the four lowest FTF class IoUs on each dataset, same per-class median rule,
written to ``gallery_selection.json``.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import scripts.narmyn.attention.finalize_sg5 as f5

CELL, T = "W/P1+R2", 0.45
out = {"rule": __doc__.split("\n\n")[1].strip(), "pipeline": CELL, "threshold": T, "datasets": {}}
GALLERY = "--gallery" in sys.argv
RANKS = {"voc": (0, 1, 2, 3), "coco": (0, 1, 2, 3)}  # gallery: positions in the ascending class order
for ds, quantiles in (("voc", (1.0, 2 / 3, 1 / 3, 0.0)), ("coco", (1.0, 0.5, 0.0))):
    if GALLERY:
        quantiles = tuple(k / 19 for k in RANKS[ds])
    r = f5.load(ds)
    c = r.counts[r.index(CELL), :, r.thresholds.index(T)]
    iou = c[:, 0] / np.maximum(c[:, :3].sum(1), 1)
    cls = np.asarray(r.classes)
    class_iou = {k: float(c[cls == k, 0].sum() / c[cls == k, :3].sum()) for k in sorted(set(cls))}
    order = sorted(class_iou, key=lambda k: class_iou[k])  # ascending
    picks = []
    for q in quantiles:
        k = order[int(round(q * (len(order) - 1)))]
        idx = np.where(cls == k)[0]
        ranked = idx[np.argsort(iou[idx], kind="stable")]
        m = ranked[(len(ranked) - 1) // 2]
        picks.append({"class": k, "class_quantile": round(q, 3), "class_iou": round(class_iou[k], 4),
                      "image_id": r.image_ids[m], "image_iou": round(float(iou[m]), 4), "images_in_class": int(len(idx))})
    out["datasets"][ds] = picks
if GALLERY:
    out["rule"] = "four lowest FTF class IoUs per dataset; per class the median-IoU image"
(ROOT / "outputs/segmentation/paper_figures" / ("gallery_selection.json" if GALLERY else "selection.json")).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["datasets"], indent=1))
