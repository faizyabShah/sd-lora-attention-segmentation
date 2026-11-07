#!/usr/bin/env python3
"""Class competition for the A3 seed on real VOC photos (exploratory; D = selection, S = untouched holdout).

capture (GPU): per photo, DDIM-invert once (empty prompt, prompt-independent trajectory), then run the frozen
  SG25/A3 pipeline on that trajectory for every vocabulary word ("a photograph of a {w}"): the 20 VOC classes
  plus a generic, non-VOC word list. For every annotated target class of the photo it also runs one joint
  prompt that lists all 20 VOC classes (shared CA softmax). Competitive seeds are built from the per-word A3
  seeds and propagated through the TARGET run's own SA graphs with the frozen functions (V0/V3 aggregate,
  C16 base path, P1 one fine step). Stores per pair: every word's A3 seed and the 5 maps of every variant.
  Masks are never used at inference; they are only read by `score`.
score (CPU): per pair and map, threshold counts (+ other-object FP) over 0.30-0.90, DenseCRF on P1/V3,
  and pixel-level separation stats (target vs other annotated object / vs background).
report: D (selection) tables, then S evaluated with the variant chosen on D (pre-registered rule below).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402

GENERIC = ["wall", "floor", "ground", "sky", "grass", "tree", "building", "road"]
THR = np.round(0.30 + 0.025 * np.arange(25), 3)          # 0.30 .. 0.90
THR_CRF = THR[::2]                                         # 0.30 .. 0.90 step 0.05
MAPS = ("seed", "P1", "V0", "V3", "C16")
CRF_MAPS = ("P1", "V3")
EPS = 1e-8


def voc_classes():
    from scripts import run_coco_cap_transfer as transfer
    return [transfer.voc_rows(i)[0] for i in range(20)]


# ------------------------------------------------------------------ competition variants (label-free)
def norm(x, how):
    if how == "raw":
        return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo).clamp_min(1e-12)


def variants(seeds, raw, target, voc, joint):
    """seeds: word -> [64,64] min-max A3; raw: word -> un-normalised A3 sum (CA softmax probability scale).
    Returns name -> seed [64,64], each min-max normalised before propagation."""
    out = {"base": seeds[target], "joint": joint}
    others = [w for w in voc if w != target]
    peak = {w: float(raw[w].max()) for w in others}
    top3 = sorted(others, key=lambda w: -peak[w])[:3]          # label-free: strongest competitors in this image
    sets = {"voc": others, "gen": GENERIC, "top3": top3}
    for how in ("mm", "raw"):
        src = seeds if how == "mm" else raw
        s = src[target]
        for sname, ws in sets.items():
            C = torch.stack([src[w] for w in ws])
            cmax, csum = C.max(0).values, C.sum(0)
            out[f"marg1/{how}/{sname}"] = (s - cmax).clamp_min(0)
            out[f"marg.5/{how}/{sname}"] = (s - 0.5 * cmax).clamp_min(0)
            out[f"rsum/{how}/{sname}"] = s * s / (s + csum + EPS)
            out[f"rmax/{how}/{sname}"] = s * s / (s + cmax + EPS)
    return {n: norm(v, "mm") for n, v in out.items()}


@torch.inference_mode()
def propagate(sa, seed):
    from src.segmentation import sg_combined_v2, sg27_fusion
    from src.segmentation.sg25_sa import _apply, _attention
    seed = seed.float()
    v2, _ = sg_combined_v2.aggregate(sa, seed)
    base = sg27_fusion.base_paths(sa, seed)
    c = seed.reshape(-1); p1 = torch.zeros(4096, device=seed.device)
    for pair in sa[64]:
        p1 += _apply(_attention(pair, 64), c).mean(0) / len(sa[64])
    return torch.stack([seed, p1.reshape(64, 64), v2["V0"].float(), v2["V3"].float(),
                        base["C16"].reshape(64, 64).float()]).cpu().numpy().astype(np.float32)


def joint_prompt(voc):
    return "a photograph of " + ", ".join(f"a {c}" for c in voc[:-1]) + f" and a {voc[-1]}"


# ------------------------------------------------------------------ capture (GPU)
def capture(a):
    import run_sg_real_atlas as ra          # patches the recording processor so ddim_invert can pause it
    from scripts import sg27_runtime
    voc = voc_classes(); vocab = voc + GENERIC
    manifest = json.loads(st.MANIFEST.read_text())
    entries = manifest[a.set][a.shard::a.num_shards]
    if a.max_images:
        entries = entries[: a.max_images]
    runtime = sg27_runtime.make_runtime(a.model)
    pipe = runtime["pipe"]
    pipe.scheduler.set_timesteps(30, device=pipe.device)
    ts_unique = sorted(set(pipe.scheduler.timesteps.tolist()))
    empty = pipe.text_encoder(pipe.tokenizer([""], padding="max_length", max_length=77,
                                             return_tensors="pt").input_ids.to(pipe.device))[0].to(pipe.unet.dtype)
    jp = joint_prompt(voc)
    n_tok = len(pipe.tokenizer(jp)["input_ids"])
    if n_tok > 77:
        raise ValueError(f"joint prompt {n_tok} tokens")
    a.out.mkdir(parents=True, exist_ok=True)
    scratch = a.out / f".gt_{a.set}_{a.shard}.png"
    stored = ROOT.parent / "sg_real_atlas" / a.set
    for entry in entries:
        iid, ks = entry["image"], entry["classes"]
        if all((a.out / f"{iid}__{k}.npz").exists() for k in ks):
            continue
        t0 = time.perf_counter()
        img, lab, names = ra.load_pair_source(entry)
        Image.fromarray((lab == ks[0]).astype(np.uint8) * 255).save(scratch)   # only satisfies the loader
        z0 = ra.encode(pipe, ra.to512(img))
        table, _ = ra.ddim_invert(pipe, z0, ts_unique, empty)
        targets = {names[k - 1]: k for k in ks}

        def run(prompt, cls, keep_sa):
            row = {"image_id": f"{iid}/{cls}", "classname": cls, "prompt": prompt, "attribution_prompt": prompt,
                   "seed": "0", "num_inferece_steps": "30", "annotation": str(scratch)}
            with ra.trajectory(ra.table_fn(table)), sg27_runtime.live_maps(runtime, row, "voc") as (s_, sa, seed, base, _p):
                seed = seed.float().clone()
                raw = runtime["recorders"][-1].sums["selected/A3"].float().clone()
                keep = {s: [(q.clone(), k.clone()) for q, k in sa[s]] for s in (16, 32, 64)} if keep_sa else None
            return seed, raw, keep

        seeds, raws, sas = {}, {}, {}
        for w in vocab:
            seeds[w], raws[w], sas_w = run(f"a photograph of a {w}", w, w in targets)
            if sas_w is not None:
                sas[w] = sas_w
        for cls, k in targets.items():
            joint, _, _ = run(jp, cls, False)
            V = variants(seeds, raws, cls, voc, joint)
            maps = {name: propagate(sas[cls], s) for name, s in V.items()}
            check = float("nan")
            if (stored / iid / str(k) / "maps.npz").exists():
                sm = np.load(stored / iid / str(k) / "maps.npz")
                check = max(float(np.abs(st.mm(maps["base"][i]) - st.mm(sm[f"ddim/{m}"])).max())
                            for i, m in ((0, "A3"), (2, "V0"), (3, "V3"), (4, "C16")))
            out = a.out / f"{iid}__{k}.npz"; tmp = out.with_name(out.stem + ".tmp.npz")
            np.savez(tmp, **{f"map/{n}": v for n, v in maps.items()},
                     **{f"word/{w}": s.cpu().numpy() for w, s in raws.items()},
                     cls=cls, k=k, check=check, set=a.set)
            os.replace(tmp, out)
            print(f"[comp:{a.set}] {iid}/{k} {cls} check {check:.1e} seed range "
                  f"{float(seeds[cls].min()):.2e}..{float(seeds[cls].max()):.2e}", flush=True)
        del sas
        torch.cuda.empty_cache()
        print(f"[comp:{a.set}] {iid} {len(ks)} pairs {time.perf_counter() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ score (CPU)
def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg]); r = np.empty(len(x)); o = np.argsort(x, kind="mergesort")
    r[o] = np.arange(1, len(x) + 1)
    # average ranks for ties
    xs = x[o]; b = np.r_[0, np.flatnonzero(np.diff(xs)) + 1, len(xs)]
    for i in range(len(b) - 1):
        if b[i + 1] - b[i] > 1:
            r[o[b[i]:b[i + 1]]] = (b[i] + 1 + b[i + 1]) / 2
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ap(score, y):
    """Average precision with ties handled as one threshold (a constant map scores the positive rate)."""
    if not y.any():
        return float("nan")
    o = np.argsort(-score, kind="mergesort"); sc, yy = score[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(sc)), len(sc) - 1]               # end of each tie group
    tp = np.cumsum(yy)[last]; n = last + 1
    rec = tp / yy.sum(); prec = tp / n
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def counts(stack, truth, valid, other):
    p = stack & valid; t = truth & valid
    tp = (p & t).sum((1, 2)); fp = (p & ~t).sum((1, 2)); fn = (~p & t[None]).sum((1, 2))
    fpo = (p & other).sum((1, 2))
    return np.stack([tp, fp, fn, valid.sum() - tp - fp - fn, fpo], 1).astype(np.int64)


def score_one(job):
    f, out = Path(job[0]), Path(job[1])
    if out.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); iid, k = f.stem.split("__"); k = int(k)
    img = np.asarray(Image.open(st.VOC / f"JPEGImages/{iid}.jpg").convert("RGB"))
    lab = np.asarray(Image.open(st.VOC / f"SegmentationClass/{iid}.png"))
    truth, valid = lab == k, lab != 255
    other = valid & (lab != 0) & ~truth; bg = lab == 0
    rng = np.random.default_rng(zlib.crc32(f.stem.encode()))
    sub = lambda m: np.flatnonzero(m.ravel()) if m.sum() <= 20000 else rng.choice(np.flatnonzero(m.ravel()), 20000, replace=False)
    it, io, ib = sub(truth & valid), sub(other), sub(bg)
    res = {"area": float(truth.sum() / valid.sum()), "cls": str(z["cls"]), "check": float(z["check"]),
           "has_other": bool(other.any()), "other_area": float(other.sum() / valid.sum())}
    for key in (n for n in z.files if n.startswith("map/")):
        name = key[4:]
        for i, m in enumerate(MAPS):
            x = st.up(st.mm(z[key][i]), truth.shape); xr = x.ravel()
            res[f"{name}|{m}|raw"] = counts(x[None] >= THR[:, None, None], truth, valid, other)
            ally = np.concatenate([it, io, ib]); yy = np.r_[np.ones(len(it), bool), np.zeros(len(io) + len(ib), bool)]
            res[f"{name}|{m}|sep"] = np.array([auroc(xr[it], xr[io]), auroc(xr[it], xr[ib]), ap(xr[ally], yy),
                                               float(xr[io].mean()) if len(io) else np.nan,
                                               float(xr[ib].mean()), float(xr[it].mean())])
            if m in CRF_MAPS:
                ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR_CRF]).astype(bool)
                res[f"{name}|{m}+R2|raw"] = counts(ms, truth, valid, other)
    tmp = out.with_name(out.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, out)


def score(a):
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(a.out / f.name)) for f in sorted(a.maps.glob("*.npz")) if not f.stem.endswith(".tmp")]
    print(len(jobs), "pairs", flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(score_one, jobs), 1):
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------ report
def load(d, setname):
    fs = sorted(f for f in d.glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    fo = {e["image"]: e["fold"] for e in json.loads(st.MANIFEST.read_text())[setname]}
    grp = np.array([f.stem.split("__")[0] for f in fs])
    S = {"cls": np.array([str(z["cls"]) for z in Z]), "grp": grp, "fold": np.array([fo[g] for g in grp]),
         "area": np.array([float(z["area"]) for z in Z]), "check": np.array([float(z["check"]) for z in Z]),
         "has_other": np.array([bool(z["has_other"]) for z in Z])}
    for key in Z[0].files:
        if "|" in key:
            S[key] = np.stack([z[key] for z in Z])
    return S


def miou(C, cls, sel):
    out = []
    for c in np.unique(cls[sel]):
        x = C[sel & (cls == c)].sum(0); out.append(x[:, 0] / np.maximum(x[:, :3].sum(1), 1))
    return np.mean(out, 0)


def oof(S, X):
    P = np.zeros_like(X[:, 0]); taus = []
    for fo in range(5):
        tr, te = S["fold"] != fo, S["fold"] == fo
        ti = int(np.argmax(miou(X, S["cls"], tr))); P[te] = X[te, ti]; taus.append(ti)
    return miou(P[:, None], S["cls"], np.ones(len(P), bool))[0], P, taus


def boot(S, Pa, Pb, n=1000):
    rng = np.random.default_rng(0); g = np.unique(S["grp"]); idx = {x: np.where(S["grp"] == x)[0] for x in g}; d = []
    for _ in range(n):
        s = np.concatenate([idx[x] for x in rng.choice(g, len(g))]); sel = np.ones(len(s), bool)
        d.append(miou(Pa[s][:, None], S["cls"][s], sel)[0] - miou(Pb[s][:, None], S["cls"][s], sel)[0])
    return np.percentile(d, [2.5, 97.5])


def prf(P):
    x = P.sum(0); return x[0] / max(x[0] + x[1], 1), x[0] / max(x[0] + x[2], 1), x[4] / max(x[1], 1)


SCORED = ("C16", "P1", "P1+R2", "V3+R2", "V0")
PRIMARY = "C16"          # pre-registered selection map (best real-photo map in earlier analyses)


def report(a):
    L = []; w = L.append
    w("# Class competition for the A3 seed on real VOC photos (exploratory)\n")
    w("Every word's A3 seed comes from a fresh frozen-pipeline run on the photo's DDIM trajectory (empty-prompt "
      "inversion, so the trajectory is the same for every word); the propagation graphs are the target run's own SA. "
      "Variants: **base** = the target's A3; **joint** = A3 of the target inside one prompt listing all 20 VOC "
      "classes (shared CA softmax); `marg1` = s − max_c, `marg.5` = s − ½·max_c (clipped at 0), `rsum` = s·s/(s+Σc), "
      "`rmax` = s·s/(s+max_c); normalisation `mm` = per-word min-max, `raw` = CA probability; competitor set `voc` = "
      "the other 19 VOC classes (fixed vocabulary), `gen` = 8 generic non-VOC words (" + ", ".join(GENERIC) + "), "
      "`top3` = the 3 VOC words with the highest raw seed peak in this photo (label-free). Competition only ever lowers "
      "the target seed. Class-macro mIoU, τ grid 0.30–0.90 (CRF every 0.05), OOF τ over 5 grouped folds. "
      f"Selection rule fixed before seeing results: the competition variant with the best D OOF mIoU on {PRIMARY}; "
      "S is scored once with that variant.\n")
    chosen = None
    for setname, d in (("D", a.D), ("S", a.S)):
        if d is None or not Path(d).exists() or not any(Path(d).glob("*.npz")):
            w(f"## {setname}: no scores\n"); continue
        S = load(Path(d), setname)
        names = sorted({k.split("|")[0] for k in S if "|" in k}, key=lambda n: (n not in ("base", "joint"), n))
        w(f"## {setname} ({len(S['cls'])} pairs, {len(np.unique(S['grp']))} photos; "
          f"{S['has_other'].sum()} pairs with another annotated object)\n")
        if setname == "D":
            c = S["check"][np.isfinite(S["check"])]
            w(f"Fresh base run vs stored D ddim maps (A3/V0/V3/C16, min-max), max abs diff: median {np.median(c):.1e}, "
              f"max {c.max():.1e} over {len(c)} pairs.\n")
        # 1. where does the other-object confusion appear (base)
        ho = S["has_other"]
        w("### Where does other-object confusion appear? (base seed and its propagated maps)\n")
        w("Pixel AUROC target vs other annotated object (pairs with one), target vs background, AP target vs rest; "
          "mean min-max score on other-object / background / target pixels.\n")
        w("| map | AUROC tgt vs other | AUROC tgt vs bg | AP | mean other | mean bg | mean tgt |\n|---|---:|---:|---:|---:|---:|---:|")
        for m in MAPS:
            X = S[f"base|{m}|sep"]
            w(f"| {m} | {np.nanmean(X[ho, 0]):.3f} | {np.nanmean(X[:, 1]):.3f} | {np.nanmean(X[:, 2]):.3f} | "
              f"{np.nanmean(X[ho, 3]):.3f} | {np.nanmean(X[:, 4]):.3f} | {np.nanmean(X[:, 5]):.3f} |")
        # 2. seed-level separation per variant
        w("\n### Seed-level separation by variant (before propagation)\n")
        w("| variant | AUROC tgt vs other | Δ | AUROC tgt vs bg | Δ | AP | Δ | C16 AUROC tgt vs other |\n|---|---:|---:|---:|---:|---:|---:|---:|")
        b0 = S["base|seed|sep"]
        for n in names:
            X = S[f"{n}|seed|sep"]; Xc = S[f"{n}|C16|sep"]
            w(f"| {n} | {np.nanmean(X[ho, 0]):.3f} | {np.nanmean(X[ho, 0] - b0[ho, 0]):+.3f} | {np.nanmean(X[:, 1]):.3f} | "
              f"{np.nanmean(X[:, 1] - b0[:, 1]):+.3f} | {np.nanmean(X[:, 2]):.3f} | {np.nanmean(X[:, 2] - b0[:, 2]):+.3f} | "
              f"{np.nanmean(Xc[ho, 0]):.3f} |")
        # 3. final mIoU
        w("\n### Segmentation (OOF mIoU; Δ vs base with photo-bootstrap 95% CI)\n")
        w("| variant | " + " | ".join(SCORED) + f" | {PRIMARY} prec / rec / FP-on-other |\n|---|" + "---:|" * (len(SCORED) + 1))
        P = {}; best = (-1, None)
        for n in names:
            cells = []
            for m in SCORED:
                mi, Pp, taus = oof(S, S[f"{n}|{m}|raw"]); P[(n, m)] = Pp
                if n == "base":
                    tau = (THR_CRF if "+R2" in m else THR)[int(np.median(taus))]
                    cells.append(f"{mi:.3f} (τ≈{tau:.2f})")
                else:
                    allp = np.ones(len(Pp), bool)
                    dm = miou(Pp[:, None], S["cls"], allp)[0] - miou(P[("base", m)][:, None], S["cls"], allp)[0]
                    lo, hi = boot(S, Pp, P[("base", m)], n=300 if n != "joint" else 1000)
                    cells.append(f"{mi:.3f} ({dm:+.3f}; {lo:+.3f}…{hi:+.3f})")
                if m == PRIMARY and n != "base" and setname == "D" and mi > best[0]:
                    best = (mi, n)
            pr, rc, fo_ = prf(P[(n, PRIMARY)])
            w(f"| {n} | " + " | ".join(cells) + f" | {pr:.3f} / {rc:.3f} / {fo_:.0%} |")
        if setname == "D":
            chosen = best[1]
            w(f"\n**Chosen on D for S:** `{chosen}` ({PRIMARY} OOF {best[0]:.3f}).\n")
        else:
            w(f"\n**Pre-registered holdout comparison on S:** `{chosen}` vs base (τ chosen OOF within S; "
              "the variant itself was chosen on D only).\n")
            if chosen is not None:
                for m in SCORED:
                    allp = np.ones(len(S["cls"]), bool)
                    dm = miou(P[(chosen, m)][:, None], S["cls"], allp)[0] - miou(P[("base", m)][:, None], S["cls"], allp)[0]
                    lo, hi = boot(S, P[(chosen, m)], P[("base", m)], n=2000)
                    w(f"- {m}: {dm:+.3f} (95% CI {lo:+.3f} … {hi:+.3f})")
                if "joint" != chosen:
                    for m in SCORED:
                        lo, hi = boot(S, P[("joint", m)], P[("base", m)], n=2000)
                        dm = miou(P[("joint", m)][:, None], S["cls"], allp)[0] - miou(P[("base", m)][:, None], S["cls"], allp)[0]
                        w(f"- joint, {m}: {dm:+.3f} (95% CI {lo:+.3f} … {hi:+.3f})")
        # best τ at grid edge?
        X = S[f"base|{PRIMARY}|raw"]; ti = int(np.argmax(miou(X, S["cls"], np.ones(len(X), bool))))
        w(f"\nBase {PRIMARY} best global τ on {setname}: {THR[ti]:.3f} (grid 0.30–0.90).\n")
    a.out_md.write_text("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap_ = argparse.ArgumentParser(); sub = ap_.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--set", default="D"); p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1); p.add_argument("--max-images", type=int)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p = sub.add_parser("score"); p.add_argument("maps", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("report"); p.add_argument("--D", type=Path); p.add_argument("--S", type=Path)
    p.add_argument("out_md", type=Path)
    a = ap_.parse_args()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
