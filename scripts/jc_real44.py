#!/usr/bin/env python3
"""Run 44 (paper2 Table 3): real-photo comparison on PASCAL VOC val, PASCAL Context val (59) and COCO-Object (val2017).

Setup stage only (runs/44_real_benchmarks/PLAN.md); captures, baselines and scoring are added after the pre-registration.

items   (CPU) every (photo, class) item of the three columns with its class area; primary = area >= 1%.
              -> <WORK>/items.json  {"voc": [...], "context": [...], "coco": [...], "classes": {...}}
Ground truth (scoring only):
  voc      datasets/voc2012/VOCdevkit/VOC2012/SegmentationClass/<id>.png, index = VOC position + 1
  context  datasets/pascal_context/SegmentationClass/<id>.png (iSeg's released 459 -> 59 converter), index = CTX59 position + 1
  coco     run 42's items (val2017 minus the Section 136 set; masks from the instances json, as runs 40-42)
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = CACHE / "jc_real44"
MIN_AREA = 0.01
VOC20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse",
         "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv"]
# PASCAL Context 59 in the label order of iSeg's converter (data/voc_context/context_name59.py ORI_CONTEXT_NAME), with the
# 20 VOC classes written as in the VOC column. Checked by name against the 459-class labels.txt (2026-11-25).
CTX59 = VOC20 + ["bag", "bed", "bench", "book", "building", "cabinet", "ceiling", "cloth", "computer", "cup", "door", "fence",
                 "floor", "flower", "food", "grass", "ground", "keyboard", "light", "mountain", "mouse", "curtain", "platform",
                 "sign", "plate", "road", "rock", "shelves", "sidewalk", "sky", "snow", "bedclothes", "track", "tree", "truck",
                 "wall", "water", "window", "wood"]
VOC_SEG = ROOT / "datasets/voc2012/VOCdevkit/VOC2012/SegmentationClass"
VOC_JPG = ROOT / "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
CTX_SEG = ROOT / "datasets/pascal_context/SegmentationClass"
CTX_VAL = ROOT / "other repositories/nerve/data/voc/context_val.txt"      # the standard 5,104-image list
assert len(CTX59) == 59 and len(set(CTX59)) == 59


def items(a):
    import numpy as np
    from PIL import Image
    out = {"classes": {"voc": VOC20, "context": CTX59}}
    voc = json.loads((CACHE / "jc_realvoc" / "items_val.json").read_text())        # runs 32-37 VOC val items (2,148)
    V = []
    for x in voc:
        m = np.array(Image.open(VOC_SEG / f"{x['img']}.png")); ar = float((m == x["ci"]).mean())
        assert VOC20[x["ci"] - 1].replace(" ", "")[:4] == x["cls"].replace(" ", "")[:4] or x["cls"] in ("tvmonitor", "tv"), x
        V.append({"id": f"voc_{x['img']}_{x['ci']:02d}", "img": x["img"], "path": str((VOC_JPG / f"{x['img']}.jpg").relative_to(ROOT)),
                  "cls": VOC20[x["ci"] - 1], "ci": x["ci"], "area": ar, "primary": ar >= MIN_AREA})
    C = []
    for v in [l.strip() for l in open(CTX_VAL) if l.strip()]:
        m = np.array(Image.open(CTX_SEG / f"{v}.png"))
        for i in np.unique(m):
            i = int(i)
            if 0 < i <= 59:
                ar = float((m == i).mean())
                C.append({"id": f"ctx_{v}_{i:02d}", "img": v, "path": str((VOC_JPG / f"{v}.jpg").relative_to(ROOT)),
                          "cls": CTX59[i - 1], "ci": i, "area": ar, "primary": ar >= MIN_AREA})
    coco = json.loads((CACHE / "jc_ov42" / "items_val.json").read_text())
    K = [dict(x, id="coco_" + x["id"], primary=x["area"] >= MIN_AREA) for x in coco]
    out.update(voc=V, context=C, coco=K)
    out["classes"]["coco"] = sorted({x["cls"] for x in K})
    for k in ("voc", "context", "coco"):
        L = out[k]; P = [x for x in L if x["primary"]]
        print(f"{k}: items {len(L)}, primary {len(P)}, photos {len({x['img'] for x in P})}, classes {len({x['cls'] for x in P})}")
        assert all(Path(ROOT / x["path"]).exists() for x in P[:50]) if k != "coco" else True
    WORK.mkdir(parents=True, exist_ok=True); (WORK / "items.json").write_text(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items")
    a = ap.parse_args(); {"items": items}[a.cmd](a)
