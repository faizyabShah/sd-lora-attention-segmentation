"""Export N-side comparison results into the shared repo layout (small, auditable files only).

For each run it writes per_item_counts.csv (key, dataset, class, TP/FP/FN of the 0.5 threshold
on the min-max soft map, runtime, peak memory) and copies the run's sanity summary.
Soft maps stay local (git-ignored by repo policy)."""
import csv, json, os, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

N = Path("/home/sidna/TRDP/external/n_side")
REPO = Path("/home/sidna/TRDP/sd-lora-attention-segmentation")
OUT = REPO / "outputs/segmentation/comparisons/n_side"
rows = {d: list(csv.DictReader(open(N / f"outputs/manifest_full_{d}.csv"))) for d in ("voc_sim", "coco_cap")}


def mm(x):
    x = x.astype(np.float32); lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-12)


def goca(name):
    base = N / f"outputs/goca_{name}_full"
    def item(d, r):
        k, c = r["key"], r["classname"]
        rec = json.load(open(base / f"records/{k}__{c}.json"))
        return base / f"maps/{k}__{c}__post.npy", rec["seconds"], rec["peak_alloc_mib"]
    return base / "sanity_summary.json", item


def diffseg():
    base = N / "diffseg/outputs/full_sd15/none"
    def item(d, r):
        ku, cu = r["key"].replace(" ", "_"), r["classname"].replace(" ", "_")
        rec = json.load(open(base / d / f"{d}__{ku}__{cu}.json"))
        t = rec.get("runtime_s", rec.get("seconds")); mem = rec.get("peak_alloc_MiB", rec.get("peak_alloc_mib"))
        return base / d / f"{d}__{ku}__{cu}.npy", t, mem
    return base / "summary.json", item


def nerve(v):
    base = N / f"nerve/outputs/full_sd15/variant_{v}"
    recs = {}
    for line in open(base / "records.jsonl"):
        x = json.loads(line); recs[(x["key"], x["classname"])] = x
    def item(d, r):
        x = recs[(r["key"], r["classname"])]
        return base / f"{r['key']}__soft.npy", x["runtime_s"], x["peak_alloc_MiB"]
    return base / "sanity_summary.json", item


RUNS = {"goca_sd15": goca("sd15"), "goca_sdxl": goca("sdxl"), "diffsegmenter_sd15": diffseg(),
        "nerve_sd15_A": nerve("A"), "nerve_sd15_V": nerve("V"), "nerve_sd15_W": nerve("W")}

for run, (summary, item) in RUNS.items():
    od = OUT / run; od.mkdir(parents=True, exist_ok=True)
    with open(od / "per_item_counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["key", "dataset", "classname", "tp_t0.5", "fp_t0.5", "fn_t0.5", "seconds", "peak_alloc_mib"])
        agg = {}
        for d, rs in rows.items():
            for r in rs:
                p, t, mem = item(d, r)
                g = np.asarray(Image.open(r["annotation"]).convert("RGB")).max(-1) > 0
                b = mm(np.load(p)) >= 0.5
                tp, fp, fn = int((b & g).sum()), int((b & ~g).sum()), int((~b & g).sum())
                w.writerow([r["key"], d, r["classname"], tp, fp, fn, round(float(t), 4), round(float(mem), 1)])
                a = agg.setdefault((d, r["classname"]), [0, 0, 0]); a[0] += tp; a[1] += fp; a[2] += fn
    if run.startswith("diffseg"):  # its summary embeds every item; keep only the header-level fields
        s = json.load(open(summary)); s = {k: v for k, v in s.items() if not isinstance(v, list)}
        json.dump(s, open(od / "sanity_summary.json", "w"), indent=1)
    else:
        shutil.copy(summary, od / "sanity_summary.json")
    res = {d: round(float(np.mean([v[0] / sum(v) for (dd, c), v in agg.items() if dd == d])), 4) for d in rows}
    print(run, res, flush=True)
