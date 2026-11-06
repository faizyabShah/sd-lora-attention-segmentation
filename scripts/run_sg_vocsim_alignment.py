#!/usr/bin/env python3
"""VOC-Sim replication of the SG-real-atlas alignment diagnostics (exploratory; nothing large is stored).

For every annotated VOC-Sim row (490): regenerate the image with the row's seed (the native SD1.5 trajectory,
the trajectory the VOC-Sim benchmark is scored on), then replay it as renoise/n0 and DDIM exactly as the real-
image capture does. The three conditions run the unchanged A3/SG25 recorder with the atlas hooks of
run_sg_real_atlas.py; the atlas stays in memory and only per-pair statistics are written:
  - the alignment statistics of analyze_sg_real_atlas_alignment.py (CA atlas, SA keep/leak, A^t seed curves,
    error breakdown) for generate, renoise_n0 and ddim;
  - the spectral statistics of analyze_sg_real_atlas_spectral.py for generate and renoise_n0 (eigh on GPU);
  - map scores: tp/fp/fn/tn of A3, C16, V0, V1, V3 at the 17 taus (raw) and V0/V3 + DenseCRF at 5 taus.
The VOC-Sim mask only measures the maps; it never builds a prediction.
Usage: run_sg_vocsim_alignment.py OUT_DIR [--shard i --num-shards n]
"""
from __future__ import annotations

import argparse, csv, json, math, os, sys, time, zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from scripts import run_sg_real_atlas as cap, sg27_runtime  # noqa: E402
from src.segmentation import sg_combined_v2  # noqa: E402
import analyze_sg_real_atlas_alignment as al  # noqa: E402
import analyze_sg_real_atlas_spectral as sp  # noqa: E402

VS = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim"
THR = np.round(0.30 + 0.025 * np.arange(17), 3)
THR_CRF = (0.50, 0.55, 0.60, 0.65, 0.70)
MAPS = ("A3", "C16", "V0", "V1", "V3")
CONDS = ("generate", "renoise/n0", "ddim")
al.ERR = (("gen_V3R2", "generate", "V3", 0.55, True), ("gen_V0R2", "generate", "V0", 0.65, True),
          ("gen_C16", "generate", "C16", 0.675, False), ("n0_V0R2", "renoise/n0", "V0", 0.65, True),
          ("ddim_C16", "ddim", "C16", 0.675, False), ("ddim_V0", "ddim", "V0", 0.625, False))


class Arr(dict):
    @property
    def files(self):
        return list(self.keys())


def spectrum_gpu(A):
    W = torch.from_numpy(A).cuda().double(); W = (W + W.T) / 2; d = W.sum(1); dm = d.clamp_min(1e-12).rsqrt()
    L = torch.eye(len(W), device=W.device, dtype=W.dtype) - dm[:, None] * W * dm[None]
    w, U = torch.linalg.eigh(L)
    n = min(sp.NEIG, len(W))
    return w[:n].cpu().numpy(), U[:, :n].cpu().numpy(), d.cpu().numpy()


sp.spectrum = spectrum_gpu


def rows():
    with (VS / "ovam_voc_sim.csv").open(newline="", encoding="utf-8") as h:
        r = [dict(x, attribution_prompt=x["prompt"]) for x in csv.DictReader(h) if x["annotated"].lower() == "true"]
    return sorted(r, key=lambda x: x["image_id"])


def gen_condition(runtime, pipe, row, masks):
    """Native generation, full level, all-module Q/K kept for the spectral analysis."""
    rc = runtime["recorder_class"]
    rc.level, rc.rl_masks, rc.all_qk = "full", masks, True
    rc.n_tokens = len(pipe.tokenizer(row["attribution_prompt"], padding=False)["input_ids"])
    with sg27_runtime.live_maps(runtime, row, "voc") as (sample, sa, seed, base, previous):
        rec = runtime["recorders"][-1]
        v2, _ = sg_combined_v2.aggregate(sa, seed)
        maps = {"A3": seed.float().cpu().numpy(), "C16": base["C16"].reshape(64, 64).float().cpu().numpy(),
                "V0": v2["V0"].float().cpu().numpy(), "V1": v2["V1"].float().cpu().numpy(),
                "V3": v2["V3"].float().cpu().numpy()}
        out = {"maps": maps, "atlas": rec.atlas, "img": np.asarray(sample.generated_image, np.uint8).copy(),
               "qk": {s: torch.stack([torch.stack(p) for p in sa[s]]).cpu() for s in (16, 32, 64)}}
    return out


def spectral(atlas, seed64, truth, valid):
    res = {}
    s64 = torch.from_numpy(seed64).float()[None, None]
    for key in sorted(x for x in atlas.files if x.startswith("qk/")):
        mod = key[3:]; s = al.side_of(mod)
        if s == 8:
            continue
        y, keep = al.gt_at(truth, valid, s)
        seed = (F.interpolate(s64, size=(s, s), mode="area") if s < 64 else s64)[0, 0].numpy().ravel().astype(np.float64)
        qk = atlas[key]
        for c in sp.CALLS:
            q, kk = torch.from_numpy(qk[c, 0]).cuda().float(), torch.from_numpy(qk[c, 1]).cuda().float()
            A = torch.softmax(torch.bmm(q, kk.transpose(1, 2)) / math.sqrt(q.shape[-1]), -1).double().cpu().numpy()
            graphs = {"mean": A.mean(0)}
            if s < 64:
                graphs.update({f"h{h}": A[h] for h in range(8)})
            for g, Ag in graphs.items():
                for name, v in sp.analyse(Ag, y, keep, seed, s).items():
                    res[f"{mod}|{c}|{g}|{name}"] = np.asarray(v)
    return res


def score_maps(img, truth, valid, maps):
    from scripts.analyze_sg_real_atlas_baseline import counts
    from src.segmentation.ovam_full import apply_ovam_densecrf
    out = {}
    for cond in CONDS:
        for m in MAPS:
            x = al.up(al.mm(maps[f"{cond}/{m}"]), truth.shape)
            out[f"{cond}/{m}"] = counts(x[None] >= THR[:, None, None], truth, valid)
            if m in ("V0", "V3"):
                ms = np.stack([apply_ovam_densecrf(img, x >= t, al.CRF_SITE) for t in THR_CRF]).astype(bool)
                out[f"{cond}/{m}+R2"] = counts(ms, truth, valid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    todo = rows()[a.shard::a.num_shards]
    todo = todo[: a.limit] if a.limit else todo
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "spectral").mkdir(exist_ok=True)
    from src.segmentation.ovam_full import _import_densecrf
    _import_densecrf(al.CRF_SITE)
    runtime = sg27_runtime.make_runtime(a.model, recorder_class=cap.AtlasRecorder)
    pipe = runtime["pipe"]
    pipe.scheduler.set_timesteps(30, device=pipe.device)
    ts_unique = sorted(set(pipe.scheduler.timesteps.tolist()))
    empty = pipe.text_encoder(pipe.tokenizer([""], padding="max_length", max_length=77,
                                             return_tensors="pt").input_ids.to(pipe.device))[0].to(pipe.unet.dtype)
    for row in todo:
        iid = row["image_id"]; outp = a.out / f"{iid}.npz"
        if outp.exists():
            continue
        t0 = time.perf_counter()
        ann = np.asarray(Image.open(VS / row["annotation"]).convert("RGB"))
        truth = np.any(ann != 0, axis=-1); valid = np.ones_like(truth)
        gt_masks = cap.rl_masks(truth, valid, pipe.device)
        n_all = len(pipe.tokenizer(row["attribution_prompt"], padding=False)["input_ids"])
        maps, res, spec, qks, atl = {}, {}, {}, {}, {}
        g = gen_condition(runtime, pipe, dict(row), gt_masks)
        img = g["img"]
        if img.shape[:2] != truth.shape:
            raise ValueError(f"{iid}: image {img.shape} vs mask {truth.shape}")
        maps.update({f"generate/{n}": v for n, v in g["maps"].items()})
        atl["generate"] = Arr(cap.pack_atlas(g["atlas"], "full")); qks["generate"] = g["qk"]; del g
        z0 = cap.encode(pipe, cap.to512(img))
        table, _ = cap.ddim_invert(pipe, z0, ts_unique, empty)
        ref = dict(row, image_id=f"{iid}/x")
        n0 = cap.noise_for(iid, 0, z0)
        for cond, lat, allqk in (("renoise/n0", cap.renoise_fn(z0, n0), True), ("ddim", cap.table_fn(table), False)):
            r = cap.run_condition(runtime, pipe, ref, lat, "full", gt_masks, all_qk=allqk, sets_qk=True)
            maps.update({f"{cond}/{n}": v for n, v in r["maps"].items()})
            atl[cond] = Arr(cap.pack_atlas(r["atlas"], "full")); qks[cond] = r["qk"]; del r
        t_gpu = time.perf_counter() - t0
        y64, keep64 = al.gt_at(truth, valid, 64)
        for cond in CONDS:
            tag = cond.replace("/", "_"); A = atl[cond]
            mods, auc, rp, vn, kn, tk = al.ca_atlas(A, n_all, truth, valid)
            res.update({f"{tag}/ca_auc": auc, f"{tag}/ca_rprec": rp, f"{tag}/ca_vnorm": vn, f"{tag}/ca_knorm": kn,
                        f"{tag}/tok_auc": tk})
            sa_mods = sorted(k[3:] for k in A.files if k.startswith("rl/"))
            res[f"{tag}/rl"] = np.stack([A[f"rl/{m}"].astype(np.float32) for m in sa_mods])
            qk = {f"s{s}": v.numpy() for s, v in qks[cond].items()}
            curves, heads2, c16p = al.diffusion(qk, maps[f"{cond}/A3"], y64, keep64)
            for s in curves:
                res[f"{tag}/diff{s}"] = curves[s]
            for s in heads2:
                res[f"{tag}/heads2_{s}"] = heads2[s]
            res[f"{tag}/c16path"] = c16p
            res[f"{tag}/seed_auc"] = al.auroc(maps[f"{cond}/A3"].ravel()[keep64][None], y64[keep64])
            if cond in ("generate", "renoise/n0"):
                s_ = spectral(A, maps[f"{cond}/A3"], truth, valid)
                spec.update({(f"{k}" if cond == "renoise/n0" else f"gen:{k}"): v for k, v in s_.items()})
        t = torch.as_tensor(truth, dtype=torch.float32)[None, None]
        t64 = F.interpolate(t, size=(64, 64), mode="area"); base = (t64 >= .5).float()
        st3 = torch.cat([t64, 1 - F.max_pool2d(1 - base, 5, 1, 2), F.max_pool2d(base, 5, 1, 2)], 1)
        res["area"] = np.array([[float((F.interpolate(st3, size=(s, s), mode="area") if s < 64 else st3)[0, j].mean())
                                 for j in range(3)] for s in (8, 16, 32, 64)], np.float32)
        err = al.errors(img, truth.astype(np.uint8), truth.astype(np.uint8), 1, maps)
        sc = score_maps(img, truth, valid, maps)
        area_full = float(truth.mean())
        tmp = outp.with_name(outp.stem + ".tmp.npz")
        np.savez(tmp, **res, **{f"counts/{k}": v for k, v in sc.items()}, mods=np.array(mods),
                 sa_mods=np.array(sa_mods), err=json.dumps(err), area_full=area_full, k=1,
                 classname=row["classname"])
        os.replace(tmp, outp)
        tmp = a.out / "spectral" / f"{iid}.tmp.npz"
        np.savez(tmp, **{k: v for k, v in spec.items() if not k.startswith("gen:")}, area=area_full, k=1)
        os.replace(tmp, a.out / "spectral" / f"{iid}.npz")
        (a.out / "spectral_gen").mkdir(exist_ok=True)
        tmp = a.out / "spectral_gen" / f"{iid}.tmp.npz"
        np.savez(tmp, **{k[4:]: v for k, v in spec.items() if k.startswith("gen:")}, area=area_full, k=1)
        os.replace(tmp, a.out / "spectral_gen" / f"{iid}.npz")
        print(f"[vocsim] {iid} {row['classname']} gpu {t_gpu:.0f}s total {time.perf_counter() - t0:.0f}s "
              f"seed AUROC gen {float(res['generate/seed_auc'][0]):.3f}", flush=True)
        del atl, qks, maps, res, spec


if __name__ == "__main__":
    main()
