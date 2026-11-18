#!/usr/bin/env python3
"""runs/22_rel_gate/PREREG.md: relative-inconsistency pixel gate, 30 + 30 image pilot.

capture (GPU)  one live generation per image; seeds BR, UN, Gabs, Gabsx, Grel, Grelx x {seed, V0, V3, walk};
               mechanism diagnostics M1-M3 per image
score   (CPU)  counts raw and + DenseCRF at 17 thresholds (runs/21 scorer)
report  (CPU)  runs/22_rel_gate/result{tag}.md
"""
from __future__ import annotations

import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_class_comp as cc  # noqa: E402
import nerve_walk_port as port  # noqa: E402
import jc_srcrule as sr  # noqa: E402
import jc_newrerun as nr  # noqa: E402
import jc_layergate as lg  # noqa: E402
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_relgate"
OUT = ROOT / "runs/22_rel_gate"
SEEDS = ("BR", "UN", "Gabs", "Gabsx", "Grel", "Grelx")
CELLS = lg.CELLS
LAB = nr.LAB
NAME = nr.NAME
THR = lg.THR
RES = torch.tensor([64, 64, 16, 16, 8, 16, 16, 16, 64, 64, 64])   # selected CA modules (sorted names)
DESC = {"BR": "BRACE seed (global layer weights)", "UN": "uniform layer weights",
        "Gabs": "pixel gate, absolute inconsistency (runs/21)", "Gabsx": "absolute gate, shuffled",
        "Grel": "pixel gate, relative inconsistency", "Grelx": "relative gate, shuffled"}
mm = lg.mm


def pick(ds, start, n, ids=""):
    R = nr.rows(ds, ids=ids)
    return R if ids else sorted(R, key=lambda r: zlib.crc32(r["image_id"].encode()))[start:start + n]


def spearman_rows(x, y):
    rx = x.argsort(1).argsort(1).float(); ry = y.argsort(1).argsort(1).float()
    rx = rx - rx.mean(1, keepdim=True); ry = ry - ry.mean(1, keepdim=True)
    return (rx * ry).sum(1) / (rx.norm(dim=1) * ry.norm(dim=1)).clamp_min(1e-12)


@torch.no_grad()
def build(A, W, P, iid):
    T, Lm, N = A.shape; Af = A.reshape(T * Lm, N); w = W.reshape(-1, 1)
    PA = Af @ P.T; num = (Af - PA).abs()
    e_abs = num / (num.median(1, keepdim=True).values + 1e-12)
    eps = (Af + PA).median()
    e_rel = num / (Af + PA + eps); e_rel = e_rel / e_rel.median().clamp_min(1e-12)
    g = torch.Generator(device="cpu").manual_seed(zlib.crc32(iid.encode())); perm = torch.randperm(N, generator=g).to(Af.device)
    out = {"UN": Af.sum(0) / (T * Lm)}; R = {}
    for nm, e in (("abs", e_abs), ("rel", e_rel)):
        r = torch.exp(-e); R[nm] = r
        for sfx, rr in (("", r), ("x", r[:, perm])):
            out[f"G{nm}{sfx}"] = (w * rr * Af).sum(0) / (w * rr).sum(0).clamp_min(1e-20)
    out = {k: mm(v).reshape(64, 64) for k, v in out.items()}
    return out, R, Af, w


def mech(R, Af, w, seeds, g):
    inner = ndi.binary_erosion(g, iterations=2); inner = inner if inner.any() else g
    near = ndi.binary_dilation(g, iterations=4) & ~g; far = ~ndi.binary_dilation(g, iterations=4)
    g, inner, near, far = (x.reshape(-1) for x in (g, inner, near, far))
    dev = Af.device; regs = {k: torch.from_numpy(m.reshape(-1)).to(dev) for k, m in (("interior", inner), ("near", near), ("far", far))}
    grp = RES.repeat(Af.shape[0] // 11).to(dev); o = {}
    strong = (grp == 16) | (grp == 8)
    o["share_global|8+16"] = float(w[strong].sum() / w.sum())
    for nm, r in R.items():
        rho = spearman_rows(r, Af); o[f"M1rho|{nm}"] = float((w[:, 0] * rho).sum() / w.sum())
        m = regs["interior"]; wr = (w * r)[:, m]
        o[f"M2share|{nm}|8+16"] = float((wr[strong].sum(0) / wr.sum(0)).mean())
        for gr in (64, 16, 8):
            k = grp == gr
            for rg, mk in regs.items():
                if mk.any():
                    o[f"r|{nm}|{gr}|{rg}"] = float((w[k] * r[k][:, mk]).sum() / (w[k].sum() * mk.sum()))
    for s, x in seeds.items():
        xx = mm(x).cpu().numpy()
        o[f"M3mass|{s}"] = float(xx[g].sum() / xx.sum())
        for rg, mk in (("interior", inner), ("near", near), ("far", far)):
            o[f"val|{s}|{rg}"] = float(xx[mk].mean()) if mk.any() else float("nan")
    return o


@torch.no_grad()
def capture(a):
    from PIL import Image
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2, sg25_sa
    runtime = sg27_runtime.make_runtime(a.model, recorder_class=lg.recorder_class()); pipe = runtime["pipe"]
    pipe.safety_checker = None; dev = pipe.device
    out = WORK / (a.dataset + a.tag); out.mkdir(parents=True, exist_ok=True)
    for r in pick(a.dataset, a.start, a.n, a.ids):
        iid = r["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        with sg27_runtime.live_maps(runtime, dict(r), a.dataset) as (sample, sa, seed, _b, _p):
            rec = runtime["recorders"][-1]
            A = torch.stack(rec.lmaps).float().reshape(31, 11, 4096); W = torch.stack(rec.lw).float().to(dev)
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            P = torch.zeros(4096, 4096, device=dev)
            for c in range(11):
                P += sg25_sa._attention(sa[64][c], 64).mean(0) / 11
            sd, R, Af, w = build(A, W, P, iid); del P
            br = mm((w * Af).sum(0) / 31).reshape(64, 64)
            rchk = float((br - seed.float()).abs().max())
            if rchk > 1e-5:
                raise ValueError(f"{iid}: reconstruction differs from runtime seed by {rchk:.2e}")
            sd["BR"] = seed.float().clone()
            H = truth.shape[0] // 64; g = truth.reshape(64, H, 64, H).mean((1, 3)) >= 0.5
            mo = mech(R, Af, w, {s: sd[s].reshape(-1) for s in ("BR", "Gabs", "Grel")}, g) if g.any() and (~g).any() else {}
            del R, Af
            res = {s: {"seed": sd[s]} for s in SEEDS}
            for s in SEEDS:
                v, _ = sg_combined_v2.aggregate(sa, mm(sd[s]).reshape(64, 64)); res[s]["V0"], res[s]["V3"] = v["V0"], v["V3"]; del v
        nstore = sr.noised_pass(pipe, Image.fromarray(img), sr.NERVE_LAYERS)
        d = int(nstore[sr.NERVE_LAYERS[0]][0]["query"].shape[-1])
        op = port.walk_operator(port.chained_attention_product([nstore[n][0] for n in sr.NERVE_LAYERS],
                                                                torch.ones(4096, 1, device=dev), d=d))
        for s in SEEDS:
            res[s]["W"] = port.apply_walk(op, mm(sd[s]).reshape(1, 64, 64))[0]
        del op, nstore
        maps = np.stack([np.stack([res[s][c].float().reshape(64, 64).cpu().numpy() for c in CELLS]) for s in SEEDS])
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=r["label"], changed=r["changed"],
                 off_diff=float(np.abs(img.astype(int) - nr.official(a.dataset, r).astype(int)).mean()),
                 recon_check=rchk, img_vs_run20=float("nan"), v3_vs_run20=float("nan"),
                 mech=json.dumps(mo), truth=truth, image=img)
        os.replace(tmp, o)
        print(f"[rg:{a.dataset}] {iid} {r['label']} recon {rchk:.1e} rho abs {mo.get('M1rho|abs', np.nan):+.3f} "
              f"rel {mo.get('M1rho|rel', np.nan):+.3f} {time.perf_counter() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


def score(a):
    lg.SEEDS = SEEDS; lg.WORK = WORK                                    # reuse the runs/21 scorer unchanged
    for ds in a.sets:
        sc = WORK / f"{ds}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / (ds + a.tag)).glob("*.npz")) if not f.stem.endswith(".tmp")]
        with Pool(a.workers) as p:
            for n, _ in enumerate(p.imap_unordered(lg.score_one, jobs, chunksize=1), 1):
                if n % 10 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    M = {ds: [json.loads(str(np.load(f)["mech"])) for f in sorted((WORK / (ds + a.tag)).glob("*.npz")) if not f.stem.endswith(".tmp")]
         for ds in S}
    n = {ds: len(S[ds]["grp"]) for ds in S}; one = {ds: np.ones(n[ds], bool) for ds in S}
    t55 = int(np.flatnonzero(np.isclose(THR, 0.55))[0])
    k = lambda s, c, crf: f"{s}|{c}{'+R2' if crf else ''}|raw"
    mi = lambda ds, P: cc.miou(P[:, None], S[ds]["cls"], one[ds])[0]
    tb = {key: int(np.argmax(cc.miou(S["voc"][key], S["voc"]["cls"], one["voc"]))) for key in S["voc"] if "|" in key and key.endswith("|raw")}
    at = lambda ds, key, mode: S[ds][key][:, t55 if mode == "0.55" else tb[key]]

    def dci(ds, Pa, Pb):
        pt = mi(ds, Pa) - mi(ds, Pb); rng = np.random.default_rng(0); d = []
        for _ in range(a.boot):
            s = rng.integers(0, n[ds], n[ds]); o1 = np.ones(n[ds], bool)
            d.append(cc.miou(Pa[s][:, None], S[ds]["cls"][s], o1)[0] - cc.miou(Pb[s][:, None], S[ds]["cls"][s], o1)[0])
        lo, hi = np.percentile(d, [2.5, 97.5]); return pt, lo, hi

    L = ["# Relative-inconsistency pixel gate: 30 + 30 pilot (`PREREG.md`)\n",
         f"`scripts/jc_relgate.py`. Images: crc32 order positions 10–39 per set (VOC-Sim {n['voc']}, COCO-Cap {n['coco']}), one "
         "live capture each, compared within this capture only. Class-macro mIoU. *VOC30-sel.* = threshold best on these "
         "VOC-Sim images (in-sample on VOC-Sim, applied unchanged to COCO-Cap). Paired image-bootstrap 95% intervals "
         f"({a.boot}). Small-sample exploratory pilot.\n"]
    for ds in S:
        rc = [float(np.load(f)["recon_check"]) for f in sorted((WORK / (ds + a.tag)).glob("*.npz")) if not f.stem.endswith(".tmp")]
        L.append(f"- {NAME[ds]}: reconstruction vs runtime seed max |Δ| {max(rc):.1e}.")
    L.append("\n## Mechanism diagnostics (mean over images)\n")
    L.append("| quantity | VOC-Sim abs | VOC-Sim rel | COCO-Cap abs | COCO-Cap rel |"); L.append("|---|---:|---:|---:|---:|")
    av = lambda ds, key: float(np.nanmean([m[key] for m in M[ds] if key in m]))
    L.append(f"| M1 ρ(r, A), w-weighted | {av('voc', 'M1rho|abs'):+.3f} | {av('voc', 'M1rho|rel'):+.3f} | {av('coco', 'M1rho|abs'):+.3f} | {av('coco', 'M1rho|rel'):+.3f} |")
    L.append(f"| M2 8+16-res share inside object (global {av('voc', 'share_global|8+16'):.3f} / {av('coco', 'share_global|8+16'):.3f}) | "
             f"{av('voc', 'M2share|abs|8+16'):.3f} | {av('voc', 'M2share|rel|8+16'):.3f} | {av('coco', 'M2share|abs|8+16'):.3f} | {av('coco', 'M2share|rel|8+16'):.3f} |")
    for gr in (64, 16, 8):
        for rg in ("interior", "near", "far"):
            L.append(f"| r, {gr}-res layers, {rg} | " + " | ".join(f"{av(ds, f'r|{nm}|{gr}|{rg}'):.3f}" for ds in S for nm in ("abs", "rel")) + " |")
    L.append("\n| seed (min–max) | VOC-Sim mass on GT | interior | near | far | COCO-Cap mass on GT | interior | near | far |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in ("BR", "Gabs", "Grel"):
        L.append(f"| {s} | " + " | ".join(f"{av(ds, f'M3mass|{s}'):.3f} | " + " | ".join(f"{av(ds, f'val|{s}|{rg}'):.3f}" for rg in ("interior", "near", "far")) for ds in S) + " |")
    for mode, ttl in (("sel", "VOC30-sel. threshold"), ("0.55", "fixed 0.55")):
        for crf in (True, False):
            for ds in S:
                L.append(f"\n**{NAME[ds]}, {ttl}, {'with DenseCRF' if crf else 'raw'}**\n")
                L.append("| Seed | " + " | ".join(LAB[c] for c in CELLS) + " |"); L.append("|---|" + "---:|" * len(CELLS))
                for s in SEEDS:
                    L.append(f"| {s}: {DESC[s]} | " + " | ".join(f"{mi(ds, at(ds, k(s, c, crf), mode)):.3f}" for c in CELLS) + " |")
    L.append("\n## Contrasts (with DenseCRF)\n")
    L.append("| Contrast | cell | VOC-Sim, VOC30-sel. | COCO-Cap, VOC30-sel. | VOC-Sim, 0.55 | COCO-Cap, 0.55 |"); L.append("|---|---|---|---|---|---|")
    C = {}
    for an, bn in (("Grel", "BR"), ("Grel", "Gabs"), ("Grel", "Grelx"), ("Gabs", "BR"), ("Gabs", "Gabsx")):
        for c in CELLS:
            cells = []
            for mode in ("sel", "0.55"):
                for ds in S:
                    C[(an, bn, c, ds, mode)] = dci(ds, at(ds, k(an, c, True), mode), at(ds, k(bn, c, True), mode))
            cells = [C[(an, bn, c, ds, mode)] for mode in ("sel", "0.55") for ds in S]
            L.append(f"| {an} − {bn} | {LAB[c]} | " + " | ".join(f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]" for p, lo, hi in cells) + " |")
    chk = {}
    for ds in S:
        chk[f"M1 {ds}"] = av(ds, "M1rho|rel") >= -0.1
        chk[f"M2 {ds}"] = av(ds, "M2share|rel|8+16") >= av(ds, "share_global|8+16") - 0.02
        chk[f"M3 mass {ds}"] = av(ds, "M3mass|Grel") >= av(ds, "M3mass|BR")
        chk[f"M3 far {ds}"] = av(ds, "val|Grel|far") <= av(ds, "val|BR|far") + 0.01
        chk[f"M4 {ds}"] = C[("Grel", "Grelx", "V3", ds, "sel")][0] > 0
        for mode in ("sel", "0.55"):
            chk[f"Grel−BR hier+CRF {ds} {mode}"] = C[("Grel", "BR", "V3", ds, mode)][0] >= 0
    L.append("\n## Pre-registered reading\n")
    L += [f"- {kk}: {'pass' if v else 'fail'}" for kk, v in chk.items()]
    L.append(f"\n**Grel {'promising (licenses a separate full-set pre-registration)' if all(chk.values()) else 'not promising: pixel-gate direction closed'}.**")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--start", type=int, default=10); p.add_argument("--n", type=int, default=30); p.add_argument("--ids", default="")
    p.add_argument("--tag", default=""); p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"]); p.add_argument("--tag", default="")
        p.add_argument("--workers", type=int, default=16); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)
