#!/usr/bin/env python3
"""Post-hoc mechanism check (exploratory, not a method test): does the pixel gate's per-map median normalization
switch off clean, coarse CA layers inside the object and hand weight to the noisy 64-res layers?

Written and committed before running. 10 VOC-Sim + 10 COCO-Cap images, chosen by crc32(image_id) order (not by
outcome). One live regeneration each (frozen runtime, runs/20 prompts, safety checker off), so maps differ slightly
from the runs/21 capture; the check compares variants within this capture only.

Per image, per CA layer resolution group (64 / 16 / 8):
- sigma: mean per-map median of |A - A P^T| (the gate's normaliser)
- mean reliability r in object interior, next-to-object background, far background (GT64, scoring only)
- effective layer weight share inside the object: sum_g w r / sum w r (gate) vs sum_g w / sum w (BRACE)
Counterfactual (mechanism only, never a candidate): one shared sigma per image (median over all 341 maps) instead
of per-map. Seeds BR, Gpx, Gpx-shared through seed and BRACE hierarchy V3: pixel AUC and mass share on GT.
Prediction if the hypothesis holds: gate interior r for 16/8 layers << far-bg r; 64-res share inside object rises
under Gpx; the shared-sigma gate removes the interior drop and does not lose AUC/mass vs BR.
"""
from __future__ import annotations

import json, sys, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_layergate as lg  # noqa: E402
import jc_newrerun as nr  # noqa: E402

RES = torch.tensor([64, 64, 16, 16, 8, 16, 16, 16, 64, 64, 64])   # selected CA modules, sorted names
OUT = Path(__file__).with_name("why_check.json")


def mm(x):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def auc(s, g):
    s, g = s.ravel(), g.ravel(); pos, neg = s[g], s[~g]
    rk = rankdata(np.concatenate([pos, neg]))
    return float((rk[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


@torch.no_grad()
def main(n_per, model):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2, sg25_sa
    runtime = sg27_runtime.make_runtime(model, recorder_class=lg.recorder_class()); pipe = runtime["pipe"]
    pipe.safety_checker = None; dev = pipe.device
    res = {}
    for ds in ("voc", "coco"):
        R = sorted(nr.rows(ds), key=lambda r: zlib.crc32(r["image_id"].encode()))[:n_per]
        res[ds] = []
        for r in R:
            iid = r["image_id"]
            with sg27_runtime.live_maps(runtime, dict(r), ds) as (sample, sa, seed, _b, _p):
                rec = runtime["recorders"][-1]
                A = torch.stack(rec.lmaps).float().reshape(31, 11, 4096); W = torch.stack(rec.lw).float().to(dev)
                truth = np.asarray(sample.target_mask, bool)
                P = torch.zeros(4096, 4096, device=dev)
                for c in range(11):
                    P += sg25_sa._attention(sa[64][c], 64).mean(0) / 11
                Af = A.reshape(-1, 4096); w = W.reshape(-1, 1)
                num = (Af - Af @ P.T).abs(); med = num.median(1, keepdim=True).values
                r_map = torch.exp(-num / (med + 1e-12))                       # per-map sigma (the registered gate)
                r_sh = torch.exp(-num / (med.median() + 1e-12))               # one shared sigma
                br = (w * Af).sum(0) / 31
                seeds = {"BR": mm(br), "Gpx": mm((w * r_map * Af).sum(0) / (w * r_map).sum(0).clamp_min(1e-20)),
                         "Gpx-shared": mm((w * r_sh * Af).sum(0) / (w * r_sh).sum(0).clamp_min(1e-20))}
                rchk = float((seeds["BR"].reshape(64, 64) - seed.float()).abs().max())
                v3 = {k: mm(sg_combined_v2.aggregate(sa, v.reshape(64, 64))[0]["V3"].reshape(-1)) for k, v in seeds.items()}
                del P
            H = truth.shape[0] // 64
            g = truth.reshape(64, H, 64, H).mean((1, 3)) >= 0.5
            if g.sum() < 4 or (~g).sum() < 4:
                continue
            inner = ndi.binary_erosion(g, iterations=2); inner = inner if inner.any() else g
            near = ndi.binary_dilation(g, iterations=4) & ~g; far = ~ndi.binary_dilation(g, iterations=4)
            regs = {k: torch.from_numpy(m.reshape(-1)).to(dev) for k, m in (("interior", inner), ("near", near), ("far", far))}
            grp = RES.repeat(31).to(dev)                                       # [341]
            o = {"id": iid, "cls": r["label"], "area": float(g.mean()), "recon": rchk}
            for gr in (64, 16, 8):
                k = grp == gr
                o[f"sigma|{gr}"] = float(med[k].mean())
                o[f"auc_layer|{gr}"] = auc(((w[k] * Af[k]).sum(0) / w[k].sum()).cpu().numpy(), g)
                o[f"share_global|{gr}"] = float(w[k].sum() / w.sum())
                for nm, rr in (("map", r_map), ("shared", r_sh)):
                    for rg, m in regs.items():
                        if m.any():
                            o[f"r_{nm}|{gr}|{rg}"] = float((w[k] * rr[k][:, m]).sum() / (w[k].sum() * m.sum()))
                    m = regs["interior"]
                    o[f"share_{nm}_interior|{gr}"] = float(((w[k] * rr[k])[:, m]).sum(0).div((w * rr)[:, m].sum(0)).mean())
            for s in seeds:
                for st, x in (("seed", seeds[s]), ("V3", v3[s])):
                    xx = x.cpu().numpy().reshape(64, 64)
                    o[f"{s}|{st}|auc"] = auc(xx, g); o[f"{s}|{st}|mass"] = float(xx[g].sum() / xx.sum())
                    for rg, m in (("interior", inner), ("near", near), ("far", far)):
                        o[f"{s}|{st}|{rg}"] = float(xx[m].mean()) if m.any() else float("nan")
            res[ds].append(o)
            print(ds, iid, r["label"], f"recon {rchk:.1e}",
                  " ".join(f"{s}:{o[f'{s}|V3|auc']:.3f}" for s in seeds), flush=True)
            torch.cuda.empty_cache()
    summ = {}
    for ds, rows in res.items():
        keys = [k for k in rows[0] if k not in ("id", "cls")]
        summ[ds] = {k: float(np.nanmean([x[k] for x in rows if k in x])) for k in keys}; summ[ds]["n"] = len(rows)
    OUT.write_text(json.dumps({"summary": summ, "per_image": res}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(n, "stable-diffusion-v1-5/stable-diffusion-v1-5")
