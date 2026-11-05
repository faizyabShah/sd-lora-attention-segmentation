#!/usr/bin/env python3
"""SG-real-atlas replay checks (Section 137), run before the full capture.

1. Same-run replay: V0/V1/V3 rebuilt on CPU from the stored BRACE Q/K and the stored A3 seed must match
   the live GPU maps (max |diff| on min-max maps is reported; fp32 CPU vs GPU roundoff only).
2. Cross-run agreement: renoise/n0 vs Section 136 (VOC pilot / COCO val200, V100), and generate vs the
   SG-combined-v2 VOC-Sim shards. Reports IoU@0.55 of raw V0 and V3 under identical scoring, plus
   min-max map differences where the old map is stored.
3. Inventory: stored bytes per pair, condition timings, peak memory from the log, reconstruction metrics.
Usage: check_sg_real_atlas_replay.py CAPTURE_DIR
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.segmentation import sg_combined_v2  # noqa: E402

CAP = Path(sys.argv[1]); RD = ROOT / "outputs/segmentation/sg_real_data"
V2 = ROOT / "outputs/segmentation/controlled_attention_cv/sg_combined_v2/voc/shards"
THR = np.round(0.30 + 0.025 * np.arange(17), 3); G = 10


def mm(x):
    x = torch.as_tensor(x, dtype=torch.float32); return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def iou_at(map64, truth, valid):
    f = F.interpolate(mm(map64)[None, None], size=truth.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    p = (f >= THR[G]) & valid; t = truth & valid
    return (p & t).sum() / max((p | t).sum(), 1)


def old_pilot(iid_k, dataset):
    root = RD / ("voc_train_pilot/renoise" if dataset == "voc" else "../../../../real_coco/run200/renoise")
    files = glob.glob(str(RD / "voc_train_pilot/renoise/*.npz")) if dataset == "voc" else \
        glob.glob("/home/shafa/real_coco/run200/renoise/*.npz")
    for f in files:
        e = np.load(f)
        if f"{iid_k}/counts" in e.files:
            c = e[f"{iid_k}/counts"][:, G]                     # names CA,C16,V0,V1,V3,V0+R2,V3+R2
            return {"V0": c[2, 0] / max(c[2, :3].sum(), 1), "V3": c[4, 0] / max(c[4, :3].sum(), 1),
                    "V3map": e[f"{iid_k}/map"].astype(np.float32)}
    return None


def old_vocsim(iid):
    for f in glob.glob(str(V2 / "class_*.npz")):
        if "diagnostics" in f:
            continue
        s = np.load(f); ids = list(s["image_ids"].astype(str))
        if iid in ids:
            cfg = [str(c) for c in s["configurations"]]; j = ids.index(iid)
            out = {}
            for n in ("V0", "V3"):
                c = s["counts"][cfg.index(n), j, G]; out[n] = c[0] / max(c[:3].sum(), 1)
            return out
    return None


rows = []
for d in sorted(glob.glob(str(CAP / "*/*/*/done.json"))):
    pdir = Path(d).parent; meta = json.loads((pdir / "meta.json").read_text()); maps = np.load(pdir / "maps.npz")
    st, iid, k = pdir.parts[-3], pdir.parts[-2], int(pdir.parts[-1])
    rec = {"pair": f"{st}/{iid}/{k}", "class": meta["class"], "MB": sum(meta["bytes"].values()) / 1e6,
           "pair_s": meta["time"]["pair_total"], "image_once_s": meta["time"]["image_setup_once"]}
    # 1. same-run replay from stored Q/K
    for cond in ("renoise/n0", "generate", "ddim"):
        qf = pdir / f"qk_{cond.replace('/', '_')}.npz"
        if not qf.exists():
            continue
        q = np.load(qf)
        sa = {s: [(torch.from_numpy(q[f"s{s}"][c, 0]), torch.from_numpy(q[f"s{s}"][c, 1])) for c in range(11)]
              for s in (16, 32, 64)}
        seed = torch.from_numpy(maps[f"{cond}/A3"])
        v2, _ = sg_combined_v2.aggregate(sa, seed)
        rec[f"replay {cond}"] = max(float((mm(v2[n]) - mm(maps[f"{cond}/{n}"])).abs().max()) for n in ("V0", "V1", "V3"))
    # 2. cross-run agreement
    if st in ("D", "C"):
        from PIL import Image
        lab = np.asarray(Image.open(ROOT / ("datasets/voc2012/VOCdevkit/VOC2012/SegmentationClass" if st == "D"
                                            else "datasets/coco2017/sg_real_data_labels") / f"{iid}.png"))
        truth, valid = lab == k, lab != 255
        old = old_pilot(f"{iid}/{k}", "voc" if st == "D" else "coco")
        for n in ("V0", "V3"):
            rec[f"IoU {n} new"] = iou_at(maps[f"renoise/n0/{n}"], truth, valid)
            if old:
                rec[f"IoU {n} old"] = old[n]
        if old:
            rec["V3 map |diff|"] = float((mm(old["V3map"]) - mm(maps["renoise/n0/V3"])).abs().max())
    if st == "G":
        old = old_vocsim(iid)
        from PIL import Image
        truth = np.asarray(Image.open(sorted(pdir.glob("*.png"))[0])) > 0 if list(pdir.glob("*.png")) else None
        if old:
            rec.update({f"IoU {n} old (v2)": old[n] for n in ("V0", "V3")})
        if truth is not None:
            for n in ("V0", "V3"):
                for cond in ("generate", "renoise/n0", "ddim"):
                    rec[f"IoU {n} {cond}"] = iou_at(maps[f"{cond}/{n}"], truth, np.ones_like(truth))
    rec["recon"] = meta["image"]
    rec["cond_s"] = {c: round(sum(v.values()), 1) for c, v in meta["time"]["conditions"].items()}
    rows.append(rec)

for r in rows:
    print(json.dumps(r, default=lambda x: round(float(x), 4)))
print(f"\npairs {len(rows)}; mean MB/pair {np.mean([r['MB'] for r in rows]):.0f}; "
      f"mean s/pair {np.mean([r['pair_s'] for r in rows]):.0f}; mean image setup s {np.mean([r['image_once_s'] for r in rows]):.0f}")

# ------------------------------------------------------------------ gate (exit 1 blocks the full capture)
if "--gate" in sys.argv:
    problems = []
    rep = [v for r in rows for key, v in r.items() if key.startswith("replay ")]
    if not rep or max(rep) > 1e-3:
        problems.append(f"same-run replay max |diff| {max(rep) if rep else 'missing'} > 1e-3")
    for n in ("V0", "V3"):
        d = [abs(r[f"IoU {n} new"] - r[f"IoU {n} old"]) for r in rows if f"IoU {n} old" in r]
        if d and np.median(d) > 0.01:
            problems.append(f"{n} renoise vs Section 136 median |dIoU| {np.median(d):.4f} > 0.01")
        g = [abs(r[f"IoU {n} generate"] - r[f"IoU {n} old (v2)"]) for r in rows if f"IoU {n} old (v2)" in r]
        if g and np.median(g) > 0.01:
            problems.append(f"{n} generate vs SG-combined-v2 median |dIoU| {np.median(g):.4f} > 0.01")
    if not any(f"IoU V0 old" in r for r in rows) or not any("IoU V0 old (v2)" in r for r in rows):
        problems.append("cross-run comparison missing for real or VOC-Sim pairs")
    # projected storage by component (measured on the smoke pairs), applied per manifest entry flags:
    #   base = everything except the all-module Q/K and the DDIM BRACE Q/K; allqk = qk/* arrays in the
    #   renoise/n0 atlas (first 50 D images); ddimqk = qk_ddim.npz (sets with ddim_qk)
    comp = {"base": [], "allqk": [], "ddimqk": [], "generate": []}
    for d in sorted(glob.glob(str(CAP / "*/*/*/done.json"))):
        pdir = Path(d).parent
        total_b = sum(f.stat().st_size for f in pdir.iterdir() if f.is_file())
        a = np.load(pdir / "atlas_renoise_n0.npz")
        allqk = sum(a[k].nbytes for k in a.files if k.startswith("qk/"))
        ddimqk = (pdir / "qk_ddim.npz").stat().st_size if (pdir / "qk_ddim.npz").exists() else 0
        gen = sum(f.stat().st_size for f in pdir.glob("*generate*"))
        comp["base"].append(total_b - allqk - ddimqk - gen)
        if allqk: comp["allqk"].append(allqk)
        if ddimqk: comp["ddimqk"].append(ddimqk)
        if gen: comp["generate"].append(gen)
    mb = {k: (np.mean(v) / 1e6 if v else 0.0) for k, v in comp.items()}
    if not comp["allqk"]:
        problems.append("all-module Q/K size not measured (no first-50 D pair in the smoke)")
    m = json.loads((ROOT / "outputs/segmentation/sg_real_atlas/manifest.json").read_text())
    total = 0.0
    for s, entries in m.items():
        for e in entries:
            per = mb["base"] + (mb["allqk"] if e.get("all_qk") else 0) + (mb["ddimqk"] if e.get("ddim_qk") else 0) \
                  + (mb["generate"] if e["dataset"] == "vocsim" else 0)
            total += per * len(e["classes"])
    print(f"projected storage {total / 1e3:.0f} GB (component MB per pair: "
          + ", ".join(f"{k} {v:.0f}" for k, v in mb.items()) + ")")
    if total > 500e3:  # owner accepted ~460 GB (5 Nov 2026)
        problems.append(f"projected storage {total / 1e3:.0f} GB > 500 GB")
    print("GATE FAIL:\n  " + "\n  ".join(problems) if problems else "GATE PASS")
    sys.exit(1 if problems else 0)
