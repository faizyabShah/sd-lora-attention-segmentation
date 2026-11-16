#!/usr/bin/env python3
"""Two diagnostics (pre-registered in runs/09_equiv_noise/PREREG.md), SD1.5, BRACE frozen:
A  transformation equivariance: is BRACE's bleed less consistent than correct object cells under flip / 16 px / 4 px shifts?
B  noise variability: is it more variable across noise realizations at BRACE's 11 attention timesteps?
Primary maps: hierarchy-free class-token cross-attention re-derived on each version; secondary: V3 with re-captured SA.

python scripts/equiv_noise.py run --dataset voc|coco [--limit k] [--tag t]   GPU -> CACHE/equiv_noise[_t]/<ds>/<id>.npz
python scripts/equiv_noise.py score [--tag t] [--workers 16]               CPU -> runs/09_equiv_noise/result.md
"""
from __future__ import annotations

import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
from brace_paths import CACHE  # noqa: E402

RUN = ROOT / "runs/09_equiv_noise"; DUMP = CACHE / "ana_dump"
SEEDS = range(6); TRANS = ("flip", "shift16", "shift4")
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


def work(tag):
    return CACHE / ("equiv_noise" + (f"_{tag}" if tag else ""))


# ------------------------------------------------------------------ transforms (image 512, maps 64)
def t_img(img, T):
    if T == "flip":
        return img[:, ::-1].copy()
    s = 16 if T == "shift16" else 4
    return np.pad(img, ((s, 0), (s, 0), (0, 0)), mode="reflect")[:512, :512].copy()


def t_map(m, T):
    from scipy.ndimage import shift as nshift
    if T == "flip":
        return m[:, ::-1].copy()
    if T == "shift16":
        return np.pad(m, ((2, 0), (2, 0)), mode="reflect")[:64, :64].copy()
    return nshift(m, (0.5, 0.5), order=1, mode="nearest")


def back(m, T):
    """map in transformed coordinates -> original coordinates, plus validity mask."""
    from scipy.ndimage import shift as nshift
    v = np.ones((64, 64), bool)
    if T == "flip":
        return m[:, ::-1].copy(), v
    if T == "shift16":
        o = np.full((64, 64), np.nan, np.float32); o[:62, :62] = m[2:, 2:]; v[62:, :] = False; v[:, 62:] = False
        return o, v
    o = nshift(m, (-0.5, -0.5), order=1, mode="nearest"); v[63, :] = False; v[:, 63] = False
    return o, v


# ------------------------------------------------------------------ GPU capture
class Cap:
    """Stores per-head (q, k) at the three frozen SA modules and the class-token CA map of every attn2 layer."""

    def __init__(self, side, tok, store):
        self.side, self.tok, self.store = side, tok, store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        import torch, torch.nn.functional as F
        from diffusers.models.attention_processor import AttnProcessor
        if self.side is not None:
            q = attn.head_to_batch_dim(attn.to_q(hidden_states)); k = attn.head_to_batch_dim(attn.to_k(hidden_states))
            self.store["sa"][self.side] = (q.detach(), k.detach())
        elif encoder_hidden_states is not None:
            q = attn.head_to_batch_dim(attn.to_q(hidden_states)); k = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
            P = attn.get_attention_scores(q, k, None)[..., self.tok].sum(-1).mean(0)           # [HW]
            s = int(P.numel() ** 0.5)
            self.store["ca"].append(F.interpolate(P.reshape(1, 1, s, s).float(), size=(64, 64), mode="bilinear", align_corners=False)[0, 0])
        return AttnProcessor()(attn, hidden_states, encoder_hidden_states, attention_mask, temb)


def run(a):
    import torch
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    import idea1_knockout as k1
    from ideas_grouped import screen_ids
    ids = sorted(screen_ids(a.dataset))[: a.limit or None]
    rows = {r["image_id"]: r for r in cc.gen_rows(a.dataset) if r["image_id"] in set(ids)}
    runtime = sg27_runtime.make_runtime(k1.MODEL); pipe = runtime["pipe"]; modules = runtime["modules"]; dev = pipe.device
    sched = pipe.scheduler; sched.set_timesteps(30, device=dev); TS = [int(t) for t in sched.timesteps[10:21]]
    ac = sched.alphas_cumprod.to(dev)
    out = work(a.tag) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    saved = dict(pipe.unet.attn_processors)
    for iid in ids:
        o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); z = np.load(DUMP / a.dataset / f"{iid}.npz", allow_pickle=True); row = rows[iid]
        img = z["image"]; seed = z["seed"].astype(np.float32); tok = [int(t) for t in np.atleast_1d(z["tokens"])]
        pe, _ = pipe.encode_prompt(prompt=row["prompt"], device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        store = {"sa": {}, "ca": []}; new = {}
        for name, p in saved.items():
            mod = name.removesuffix(".processor"); side = next((s for s, m in modules.items() if m == mod), None)
            new[name] = Cap(side, tok, store) if (side is not None or mod.endswith("attn2")) else getattr(p, "original", p)
        pipe.unet.set_attn_processor(new)
        res = {"cls": str(z["cls"]), "TS": np.array(TS)}
        try:
            conds = [(f"id_s{s}", None, s) for s in SEEDS] + [(T, T, 0) for T in TRANS]
            for nm, T, s in conds:
                im = img if T is None else t_img(img, T); sd = seed if T is None else t_map(seed, T)
                x = torch.from_numpy(im).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
                with torch.no_grad():
                    z0 = pipe.vae.encode(x.to(pipe.vae.dtype)).latent_dist.mean * pipe.vae.config.scaling_factor
                g = torch.Generator(device="cpu").manual_seed(s); eps = torch.randn(z0.shape, generator=g)
                if T == "flip":
                    eps = eps.flip(-1)
                elif T == "shift16":
                    eps = eps.roll((2, 2), (-2, -1))
                eps = eps.to(dev, z0.dtype)
                sa = {16: [], 32: [], 64: []}; store["ca"].clear(); ca_sum = 0
                for t in TS:
                    store["sa"].clear(); n0 = len(store["ca"])
                    zt = ac[t].sqrt() * z0 + (1 - ac[t]).sqrt() * eps
                    with torch.no_grad():
                        pipe.unet(zt, torch.tensor(t, device=dev), encoder_hidden_states=pe, return_dict=False)
                    for r in (16, 32, 64):
                        sa[r].append(store["sa"][r])
                    ca_sum = ca_sum + torch.stack(store["ca"][n0:]).mean(0)
                ca = (ca_sum / len(TS)).cpu().numpy()
                with torch.no_grad():
                    v3 = sg_combined_v2.aggregate(sa, torch.from_numpy(sd.copy()).to(dev))[0]["V3"].float().reshape(64, 64).cpu().numpy()
                if T is not None:
                    ca, vm = back(ca, T); v3, _ = back(v3, T); res[f"valid/{nm}"] = vm
                res[f"ca/{nm}"] = ca.astype(np.float32); res[f"v3/{nm}"] = v3.astype(np.float32)
                del sa; torch.cuda.empty_cache()
        finally:
            pipe.unet.set_attn_processor(dict(saved))
        res["sec"] = time.perf_counter() - t0
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)
        print(f"[eqn:{a.dataset}] {iid} {res['cls']} tokens {tok} {res['sec']:.1f}s", flush=True)


# ------------------------------------------------------------------ CPU scoring
def mm_valid(m):
    x = np.asarray(m, np.float64); f = np.isfinite(x)
    lo, hi = x[f].min(), x[f].max(); return (x - lo) / max(hi - lo, 1e-12)


def one(job):
    ds, iid, W = job
    import feat_bleed as fb, sa_boundary_diag as sbd, ana_ops_common as oc
    from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
    oc.crf_init()
    z = np.load(DUMP / ds / f"{iid}.npz", allow_pickle=True); c = np.load(Path(W) / ds / f"{iid}.npz")
    img, truth = z["image"], z["truth"].astype(np.float64); v = st.mm(z["v3"])
    M = oc.crf(img, st.up(v, truth.shape) >= 0.55); Mc = sbd.pool(M.astype(float), 64) >= 0.5; gt = sbd.pool(truth, 64)
    F = v >= 0.9
    if not F.any():
        F = v >= np.quantile(v, 0.95)
    B = (v <= 0.1) & ~Mc
    if F.sum() < fb.MINCORE or B.sum() < fb.MINCORE:
        return {"id": iid, "ok": False}
    L = sbd.lab(img); Lc = sbd.pool(L, 64); Ls = np.sqrt(np.maximum(sbd.pool(L ** 2, 64) - Lc ** 2, 0))
    Lg = gaussian_filter(L, (1, 1, 0)); G = np.sqrt(sum(sobel(Lg[..., k], 0) ** 2 + sobel(Lg[..., k], 1) ** 2 for k in range(3)))
    dF, (iy, ix) = distance_transform_edt(~F, return_indices=True); depth = distance_transform_edt(Mc)
    cells = np.argwhere(np.ones((64, 64), bool)); near = np.c_[iy.ravel(), ix.ravel()]
    contour = sbd.contour_max(G, cells.astype(float), near.astype(float), 8).reshape(64, 64)
    Fv, Bv = F.ravel(), B.ravel(); tr = Fv | Bv
    X6 = np.c_[Lc.reshape(-1, 3), Ls.reshape(-1, 3)]; s_rgb = fb.apply(fb.wlogit(X6[tr], Fv[tr].astype(float)), X6)
    CA = {k[3:]: mm_valid(c[k]) for k in c.files if k.startswith("ca/")}; V3 = {k[3:]: mm_valid(c[k]) for k in c.files if k.startswith("v3/")}
    ids_ = [f"id_s{s}" for s in SEEDS]

    def equiv(D, Ts=TRANS):
        diffs = []
        for T in Ts:
            d = np.abs(D[T] - D["id_s0"]); d[~c[f"valid/{T}"]] = np.nan; diffs.append(d)
        return np.nanmean(np.stack(diffs), 0)
    feats = {"dLab_F": np.linalg.norm(Lc - Lc[F].mean(0), axis=-1), "dLab_B": np.linalg.norm(Lc - Lc[B].mean(0), axis=-1),
             "contour": contour, "dist_F": dF, "conf": v, "depth": depth, "area": np.full((64, 64), Mc.mean()),
             "s_rgb": s_rgb.reshape(64, 64), "mean_ca": np.mean([CA[k] for k in ids_], 0), "mean_v3": np.mean([V3[k] for k in ids_], 0),
             "e_ca": equiv(CA), "e_v3": equiv(V3), "v_ca": np.std([CA[k] for k in ids_], 0), "v_v3": np.std([V3[k] for k in ids_], 0)}
    for T in TRANS:
        feats[f"e_ca_{T}"] = equiv(CA, (T,))
    Q = (Mc & ~F & ((gt >= 0.75) | (gt <= 0.25))).ravel(); y = (gt <= 0.25).ravel()[Q]
    tb = truth.astype(bool); iou = lambda m: float((m & tb).sum() / max((m | tb).sum(), 1))
    mask = lambda m64: oc.crf(img, st.up(st.mm(m64), truth.shape) >= 0.55)
    aug = {"brace": iou(M), "rn_id0": iou(mask(c["v3/id_s0"])),
           "flip_avg": iou(mask(np.nan_to_num(c["v3/id_s0"]) + np.nan_to_num(c["v3/flip"]))),
           "noise_avg": iou(mask(np.mean([c[f"v3/{k}"] for k in ids_], 0)))}
    caq = sbd.auc((gt >= 0.5).ravel(), feats["mean_ca"].ravel())
    inside = {k: (float(np.nanmean(feats[k][Mc])), float(np.nanmean(feats[k][~Mc]))) for k in ("e_ca", "v_ca", "e_v3", "v_v3")}
    return {"id": iid, "ok": True, "Q": Q, "y": y, "X": {k: np.asarray(f, np.float64).ravel()[Q] for k, f in feats.items()},
            "aug": aug, "ca_auc": caq, "inside": inside, "sec": float(c["sec"])}


def score(a):
    import sa_boundary_diag as sbd
    sbd.NBOOT = 2000
    W = str(work(a.tag)); res = {}
    BASE = ["dLab_F", "dLab_B", "contour", "dist_F", "conf", "depth", "area", "s_rgb", "mean_ca", "mean_v3"]
    L = ["# Equivariance (A) and noise variability (B) of SD1.5 attention at BRACE's bleed (diagnostics; `PREREG.md`)\n",
         "Stored rich capture images; 9 conditions per image (6 noise seeds; flip, 16 px and 4 px shifts); BRACE's 11 "
         "attention timesteps, one conditional call each. Decisive: COCO-Cap, primary CA statistics.\n"]
    for ds in ("voc", "coco"):
        jobs = [(ds, f.stem, W) for f in sorted((Path(W) / ds).glob("*.npz")) if not f.stem.endswith(".tmp")]
        with Pool(a.workers) as p:
            R = p.map(one, jobs, chunksize=1)
        ok = [r for r in R if r["ok"] and len(r["y"])]
        y = np.concatenate([r["y"] for r in ok]); grp = np.concatenate([np.full(len(r["y"]), r["id"]) for r in ok])
        X = {k: np.concatenate([r["X"][k] for r in ok]) for k in ok[0]["X"]}
        fin = np.isfinite(np.stack(list(X.values()), 1)).all(1); y, grp = y[fin], grp[fin]; X = {k: v[fin] for k, v in X.items()}
        S = lambda ks: np.stack([X[k] for k in ks], 1)
        g = {nm: sbd.compare(S(BASE), S(BASE + [f]), y, grp) for nm, f in
             (("A_ca", "e_ca"), ("B_ca", "v_ca"), ("A_v3", "e_v3"), ("B_v3", "v_v3"),
              ("A_ca_flip", "e_ca_flip"), ("A_ca_shift16", "e_ca_shift16"), ("A_ca_shift4", "e_ca_shift4"))}
        single = {k: sbd.auc(y, X[k]) for k in X}
        aug = {k: float(np.mean([r["aug"][k] for r in ok])) for k in ok[0]["aug"]}
        ins = {k: [float(np.mean([r["inside"][k][j] for r in ok])) for j in (0, 1)] for k in ok[0]["inside"]}
        res[ds] = dict(images=len(ok), excluded=len(R) - len(ok), cells=int(len(y)), bleed=float(y.mean()), gates=g, single=single,
                       aug=aug, inside=ins, ca_auc=float(np.nanmean([r["ca_auc"] for r in ok])), sec=float(np.mean([r["sec"] for r in ok])))
    c = res["coco"]["gates"]; pa = bool(c["A_ca"]["delta"] >= 0.05 and c["A_ca"]["lo"] > 0); pb = bool(c["B_ca"]["delta"] >= 0.05 and c["B_ca"]["lo"] > 0)
    res.update(A_pass=pa, B_pass=pb)
    f4 = lambda x: f"{x:+.4f}"
    L += [f"## Verdicts (COCO-Cap, primary CA statistic): A equivariance **{'PASS' if pa else 'CLOSE'}**, B noise variability **{'PASS' if pb else 'CLOSE'}**\n",
          "Bar: Δ OOF AUROC ≥ +0.05 with CI > 0 over baseline (RGB, contour, distance, BRACE confidence, depth, area, RGB classifier, mean CA and mean V3 maps).\n",
          "| set | images | cells (bleed) | baseline | A: + e_CA Δ (CI) | B: + v_CA Δ (CI) | A: + e_V3 Δ | B: + v_V3 Δ | A per transform (flip / shift16 / shift4) |",
          "|---|---|---|---:|---|---|---:|---:|---|"]
    for ds in ("voc", "coco"):
        r = res[ds]; G = r["gates"]; ci = lambda k: f"{f4(G[k]['delta'])} [{f4(G[k]['lo'])}, {f4(G[k]['hi'])}]"
        L.append(f"| {NAME[ds]} | {r['images']} ({r['excluded']} excl.) | {r['cells']:,} ({r['bleed']:.1%}) | {G['A_ca']['rgb']:.3f} | {ci('A_ca')} | {ci('B_ca')} | "
                 f"{f4(G['A_v3']['delta'])} | {f4(G['B_v3']['delta'])} | {f4(G['A_ca_flip']['delta'])} / {f4(G['A_ca_shift16']['delta'])} / {f4(G['A_ca_shift4']['delta'])} |")
    L += ["\n## Single-feature AUROC for bleed vs correct (higher = feature larger on bleed)\n",
          "| feature | VOC-Sim | COCO-Cap |", "|---|---:|---:|"]
    for k in res["voc"]["single"]:
        L.append(f"| {k} | {res['voc']['single'][k]:.3f} | {res['coco']['single'][k]:.3f} |")
    L += ["\n## Augmentation baselines (mean per-image IoU, τ 0.55 + R2) and protocol checks\n",
          "| set | BRACE (live) | re-noised id_s0 V3 | flip-average | noise-average (6) | mean CA map pixel AUROC (cells) | mean e_CA in / out of M | mean v_CA in / out of M | s / image |",
          "|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for ds in ("voc", "coco"):
        r = res[ds]; A = r["aug"]; I = r["inside"]
        L.append(f"| {NAME[ds]} | {A['brace']:.4f} | {A['rn_id0']:.4f} | {A['flip_avg']:.4f} | {A['noise_avg']:.4f} | {r['ca_auc']:.3f} | "
                 f"{I['e_ca'][0]:.3f} / {I['e_ca'][1]:.3f} | {I['v_ca'][0]:.3f} / {I['v_ca'][1]:.3f} | {r['sec']:.0f} |")
    L.append("\nDiagnostics only; GT scores only. Augmentation rows are descriptive (not a selection).")
    out = RUN / (f"result_{a.tag}" if a.tag else "result"); RUN.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("\n".join(L) + "\n"); out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--dataset", choices=("voc", "coco"), required=True); p.add_argument("--limit", type=int); p.add_argument("--tag", default="")
    p = sub.add_parser("score"); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(); {"run": run, "score": score}[a.cmd](a)
