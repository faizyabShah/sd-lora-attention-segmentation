#!/usr/bin/env python3
"""Full GoCA manifests (stdlib only): outputs/manifest_full_{voc_sim,coco_cap}.csv.

Enumeration follows the shared repo exactly:
  VOC-Sim  rows of evaluation/ovam/voc_sim/ovam_voc_sim.csv with annotated == True (490),
           in numeric image-id order; image/annotation paths from the csv columns.
  COCO-Cap every evaluation/ovam/coco_captions/annotations/*.png (707), sorted, parsed and
           joined with coco_captions_sampled.csv as scripts/build_coco_cap_manifest.py does
           (same regex, same image name <cls>_sd15_caption<cap>_seed<seed>.png, same
           image_id ovam-coco-<cap>-<cls>-<seed>, prompt = caption.lower()).
Columns match scripts/goca_classword.py (key, dataset, classname, image, image_source,
annotation) plus gen_prompt, seed and, for COCO-Cap, caption_id and word_included.
Every image and annotation must exist (stored benchmark images, no regeneration).
"""
import csv
import re
import sys
from pathlib import Path

EVAL = Path("/home/sidna/TRDP/sd-lora-attention-segmentation/datasets/ovam_experiment_with_dataset/evaluation")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/sidna/TRDP/external/n_side/outputs")
PAT = re.compile(r"^(?P<cls>.+)_sd-15_caption_(?P<cap>\d+)_(?P=cls)_(?P<seed>\d+)$")  # build_coco_cap_manifest.py:16-19


def write(rows, path):
    missing = [x for r in rows for x in (r["image"], r["annotation"]) if not Path(x).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} missing files, e.g. {missing[:3]}")
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(path, len(rows), "rows,", len({r['classname'] for r in rows}), "classes")


voc = [r for r in csv.DictReader(open(EVAL / "ovam/voc_sim/ovam_voc_sim.csv", newline="")) if r["annotated"].strip().lower() == "true"]
voc.sort(key=lambda r: int(r["image_id"].rsplit("-", 1)[1]))
write([{"key": r["image_id"], "dataset": "voc_sim", "classname": r["classname"],
        "image": str(EVAL / "ovam/voc_sim" / r["image"]), "image_source": "stored",
        "annotation": str(EVAL / "ovam/voc_sim" / r["annotation"]), "gen_prompt": r["prompt"], "seed": r["seed"]}
       for r in voc], OUT / "manifest_full_voc_sim.csv")

caps = {r["caption_id"]: r for r in csv.DictReader(open(EVAL / "coco_captions_sampled.csv", newline="", encoding="utf-8"))}
coco = []
for a in sorted((EVAL / "ovam/coco_captions/annotations").glob("*.png")):
    m = PAT.fullmatch(a.stem)
    if m is None:
        raise SystemExit(f"unexpected annotation name {a.name}")
    cls, cap, seed = m["cls"], m["cap"], m["seed"]
    src = caps[cap]
    if src["category_name"] != cls:
        raise SystemExit(f"class mismatch {a.name}: {src['category_name']}")
    coco.append({"key": f"ovam-coco-{cap}-{cls.replace(' ', '_')}-{seed}", "dataset": "coco_cap", "classname": cls,
                 "image": str(EVAL / f"ovam/coco_captions/images/{cls}_sd15_caption{cap}_seed{seed}.png"),
                 "image_source": "stored", "annotation": str(a), "gen_prompt": src["caption"].lower(), "seed": seed,
                 "caption_id": cap, "word_included": src["word_included"]})
if len({(r["key"], r["classname"]) for r in coco}) != len(coco):
    raise SystemExit("duplicate COCO-Cap (key, class) pairs")
write(coco, OUT / "manifest_full_coco_cap.csv")
