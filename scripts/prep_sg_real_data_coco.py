#!/usr/bin/env python3
"""SG-real-data COCO set: pick 200 COCO val2017 images covering all 80 classes (>=2 images per class where possible, rarest
class first, seeded), fill the rest at random, and write semantic label PNGs (0 bg, 1..80 = COCO
category order, 255 = crowd/ignore). Larger instances are drawn first so small objects stay on top.
No pycocotools: polygons via PIL, crowd uncompressed RLE decoded directly."""
import json, random
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

C = Path("/home/shafa/sd-lora-attention-segmentation/datasets/coco2017"); OUT = Path("/home/shafa/sd-lora-attention-segmentation/outputs/segmentation/sg_real_data"); LAB = C / "sg_real_data_labels"
d = json.load(open(C / "annotations/instances_val2017.json"))
cats = sorted(d["categories"], key=lambda c: c["id"]); idx = {c["id"]: i + 1 for i, c in enumerate(cats)}
(OUT / "coco_classes.txt").write_text("\n".join(c["name"] for c in cats) + "\n")
imgs = {i["id"]: i for i in d["images"]}; anns = defaultdict(list)
for a in d["annotations"]:
    anns[a["image_id"]].append(a)
has = defaultdict(set)
for iid, al in anns.items():
    for a in al:
        if not a["iscrowd"]:
            has[idx[a["category_id"]]].add(iid)
rng = random.Random(2017); chosen = []
count = defaultdict(int)
for k in sorted(has, key=lambda k: len(has[k])):
    pool = sorted(has[k] - set(chosen)); rng.shuffle(pool)
    while count[k] < 2 and pool:
        iid = pool.pop(); chosen.append(iid)
        for kk in {idx[a["category_id"]] for a in anns[iid] if not a["iscrowd"]}:
            count[kk] += 1
rest = sorted(set(anns) - set(chosen)); rng.shuffle(rest)
chosen += rest[: 200 - len(chosen)]
assert len(chosen) == 200 and all(count[k] >= 1 for k in range(1, 81))

def rle_decode(rle, h, w):
    flat = np.zeros(h * w, np.uint8); pos, val = 0, 0
    for n in rle["counts"]:
        flat[pos:pos + n] = val; pos += n; val ^= 1
    return flat.reshape(w, h).T.astype(bool)

LAB.mkdir(exist_ok=True)
for iid in chosen:
    im = imgs[iid]; h, w = im["height"], im["width"]
    lab = np.zeros((h, w), np.uint8)
    for a in sorted(anns[iid], key=lambda a: -a["area"]):
        if a["iscrowd"]:
            m = rle_decode(a["segmentation"], h, w) if isinstance(a["segmentation"]["counts"], list) else None
            if m is not None:
                lab[m & (lab == 0)] = 255
            continue
        canvas = Image.new("L", (w, h), 0); dr = ImageDraw.Draw(canvas)
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                dr.polygon(poly, fill=1)
        lab[np.asarray(canvas, bool)] = idx[a["category_id"]]
    Image.fromarray(lab).save(LAB / f"{im['file_name'][:-4]}.png")
(OUT / "coco_val200_ids.txt").write_text("\n".join(imgs[i]["file_name"][:-4] for i in sorted(chosen)) + "\n")
per = [sum(1 for i in chosen if i in has[k]) for k in range(1, 81)]
print(f"200 images, {sum(len({idx[a['category_id']] for a in anns[i] if not a['iscrowd']}) for i in chosen)} image-class pairs; "
      f"images per class min {min(per)} median {int(np.median(per))} max {max(per)} (person {per[0]})")
