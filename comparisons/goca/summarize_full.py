#!/usr/bin/env python3
"""Sanity summary of a goca_full.py output dir (stdlib + numpy).

NOT the official score: official scoring waits for Entry 1's shared scorer.
Fixed threshold 0.5 on the min-max-normalised map, class-macro of pooled per-class IoU.

    python summarize_full.py <out_dir>   ->  <out_dir>/sanity_summary.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
N = Path("/home/sidna/TRDP/external/n_side/outputs")
items = []
for m in ("manifest_full_voc_sim.csv", "manifest_full_coco_cap.csv"):
    items += list(csv.DictReader(open(N / m, newline="")))
recs, missing = [], []
for r in items:
    p = OUT / "records" / f"{r['key']}__{r['classname']}.json"
    if p.exists():
        x = json.loads(p.read_text())
        x["word_included"] = r.get("word_included")
        recs.append(x)
    else:
        missing.append(f"{r['key']}__{r['classname']}")


def macro(rs, name):
    by = {}
    for x in rs:
        a = by.setdefault(x["classname"], [0, 0])
        a[0] += x[f"inter_{name}"]
        a[1] += x[f"union_{name}"]
    ious = {c: i / u for c, (i, u) in sorted(by.items()) if u}
    return {"miou": float(np.mean(list(ious.values()))) if ious else float("nan"), "classes": len(ious),
            "items": len(rs), "class_iou": ious}


out = {"label": "sanity only, fixed threshold 0.5 on min-max map; NOT the official score "
                "(official scoring waits for Entry 1's shared scorer)",
       "out_dir": str(OUT), "items_expected": len(items), "items_done": len(recs), "missing": missing,
       "per_dataset": {}}
for ds in ("voc_sim", "coco_cap"):
    rs = [x for x in recs if x["dataset"] == ds]
    if not rs:
        continue
    sec = np.array([x["seconds"] for x in rs])
    d = {"post_affinity": macro(rs, "post"), "pre_affinity": macro(rs, "pre"),
         "nan_maps_post": sum(x["nan_post"] for x in rs), "nan_maps_pre": sum(x["nan_pre"] for x in rs),
         "constant_maps_post": sum(x.get("constant_post", False) for x in rs),
         "constant_maps_pre": sum(x.get("constant_pre", False) for x in rs),
         "seconds_mean": float(sec.mean()), "seconds_median": float(np.median(sec)), "seconds_max": float(sec.max()),
         "peak_alloc_mib_max": max(x["peak_alloc_mib"] for x in rs),
         "peak_reserved_mib_max": max(x["peak_reserved_mib"] for x in rs),
         "gpus": sorted({f"{x['gpu']} ({x['node']}, job {x['slurm_job']})" for x in rs}),
         "captions_example": sorted({x["caption"] for x in rs})[:3]}
    if ds == "coco_cap":
        d["post_word_included"] = macro([x for x in rs if str(x["word_included"]).lower() == "true"], "post")
        d["post_word_excluded"] = macro([x for x in rs if str(x["word_included"]).lower() == "false"], "post")
    out["per_dataset"][ds] = d
(OUT / "sanity_summary.json").write_text(json.dumps(out, indent=1))
print(out["label"])
print("done", len(recs), "/", len(items), "missing", len(missing))
for ds, d in out["per_dataset"].items():
    print(ds, {k: (round(v["miou"], 4) if isinstance(v, dict) else v) for k, v in d.items()
               if k not in ("gpus", "captions_example")})
