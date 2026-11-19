#!/usr/bin/env python3
"""runs/30_query_gate_mechanism/PREREG.md: which part of the query-profile gate works, where, and class-adaptive.

Arms (seed): A BRACE | NC absent-class gate | NH NC at half strength | NO all-pixel control | PD object direction
             only | PI object direction, image's own anchor.   Maps: seed, V0 (fine-only), V3 (hierarchy).
Profiles: run 29's statistics (<CACHE>/jc_absent/fit.pt), cross-fitted by fold.

capture  (GPU, sharded, --set voc|coco) -> <CACHE>/jc_qgate/{set}{tag}/
score    (CPU, sharded) counts + diagnostics
report   runs/30_query_gate_mechanism/result{tag}.md
"""
from __future__ import annotations

import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_absent as AB  # noqa: E402

T, J, cc, st, nr = AB.T, AB.J, AB.cc, AB.st, AB.nr
WORK = J.CACHE / "jc_qgate"
OUT = ROOT / "runs/30_query_gate_mechanism"
ARMS = ("A", "NC", "NH", "NO", "PD", "PI")
GARMS = ("NC", "NO", "PD", "PI")                      # gate maps saved for diagnostics (NH = NC ** 0.5)
MAPS = ("seed", "V0", "V3")
THR, T55 = J.THR, J.T55
SETS = ("voc", "coco")
NEAR = 6


def sig(q, w, pp, pm):
    return torch.sigmoid(T.GAIN * (q @ w - (pp + pm) / 2) / max(pp - pm, 1e-6))


def proto_gates(R, ds, iid, lab, Q, arm):
    """PD: w = m_o/|m_o|, negative anchor w·m_all (cross-fitted). PI: same w, anchor = image's own mean projection."""
    Fs = R[ds]; folds = [f for f in range(5) if f != AB.fold(iid)]
    g, miss = {}, 0
    for m, q in Q.items():
        vo, no = AB._sum(Fs, "obj", (lab, m), folds); va, na = AB._sum(Fs, "all", m, folds)
        if vo is None or no <= 0 or va is None or na <= 0:
            g[m] = torch.ones(q.shape[0], device=q.device); miss += 1; continue
        mo = (vo / no).to(q.device); w = F.normalize(mo, dim=0); pp = float(w @ mo)
        pm = float(w @ (va / na).to(q.device)) if arm == "PD" else float((q @ w).mean())
        g[m] = sig(q, w, pp, pm)
    return g, miss


def gate_map(g):
    acc = 0
    for v in g.values():
        s = int(round(v.numel() ** 0.5))
        acc = acc + F.interpolate(v.reshape(1, 1, s, s).float(), size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
    return acc / len(g)


@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(a.model); R = torch.load(AB.WORK / "fit.pt")
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in nr.rows(a.set, a.shard, a.num_shards, a.max_images):
        o = out / f"{row['image_id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, a.set); lab = AB.label(row); iid = row["image_id"]
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); Q = T.qbar(rec); tgt = (None, list(rec.target))
        G = {"NC": AB.gates(R, a.set, iid, lab, Q, "NC"), "NO": AB.gates(R, a.set, iid, lab, Q, "NO"),
             "PD": proto_gates(R, a.set, iid, lab, Q, "PD"), "PI": proto_gates(R, a.set, iid, lab, Q, "PI")}
        G["NH"] = ({m: v.sqrt() for m, v in G["NC"][0].items()}, G["NC"][1])
        S = T.ca_seeds(rec, {arm: tgt for arm in ARMS}, gate={arm: G[arm][0] for arm in G})
        fid = float((S["A"] - seed).abs().max())
        if fid > 1e-2:
            raise ValueError(f"recomputed seed differs from runtime seed by {fid:.2e} for {iid}")
        S["A"] = seed; maps = []
        for arm in ARMS:
            v = sg_combined_v2.aggregate(sa, S[arm])[0]
            maps.append(np.stack([x.float().reshape(64, 64).cpu().numpy() for x in (S[arm], v["V0"], v["V3"])]))
        maps = np.stack(maps).astype(np.float32)
        gm = np.stack([gate_map(G[k][0]).cpu().numpy() for k in GARMS]).astype(np.float32)
        if not (np.isfinite(maps).all() and np.isfinite(gm).all()):
            raise ValueError(f"non-finite map for {iid}")
        ncomp = sum(T.mentions(row["prompt"], c) for c in T.CLASSES if c != lab) if a.set == "coco" else 0
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), gates=gm.astype(np.float16), cls=lab,
                 truth=np.asarray(sample.target_mask, bool), image=np.asarray(sample.generated_image, np.uint8),
                 fid_seed=fid, ncomp=ncomp, **{f"gate_miss_{k}": G[k][1] for k in G})
        os.replace(tmp, o)
        print(f"[qgate:{a.set}] {iid} fid {fid:.1e} miss " + " ".join(f"{k}{G[k][1]}" for k in G) +
              f" {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


# ---------------------------------------------------------------- score
def regions(truth):
    t = torch.from_numpy(truth.astype(np.float32))[None, None]
    fg = (F.adaptive_avg_pool2d(t, 64)[0, 0] > 0.5)
    near = (F.max_pool2d(fg.float()[None, None], 2 * NEAR + 1, 1, NEAR)[0, 0] > 0) & ~fg
    return fg.numpy(), near.numpy(), (~fg & ~near).numpy()


def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    fg, near, far = regions(truth); M = z["maps"].astype(np.float32); Gm = z["gates"].astype(np.float32)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "fid_seed": float(z["fid_seed"]), "ncomp": int(z["ncomp"]),
           "nfg": int(fg.sum()), "nnear": int(near.sum()), "nfar": int(far.sum())}
    for gi, k in enumerate(GARMS):
        for rn, r in (("fg", fg), ("near", near), ("far", far)):
            res[f"gate_{k}_{rn}"] = float(Gm[gi][r].mean()) if r.any() else np.nan
    for ai, arm in enumerate(ARMS):
        s = st.mm(M[ai, 0]); tot = max(float(s.sum()), 1e-8)
        for rn, r in (("fg", fg), ("near", near), ("far", far)):
            res[f"mass_{arm}_{rn}"] = float(s[r].sum()) / tot
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(M[ai, mi]), truth.shape)
            res[f"{arm}|{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
            res[f"{arm}|{m}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    jobs = []
    for s in SETS:
        sc = WORK / f"{s}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs += [(str(f), str(sc / f.name)) for f in sorted((WORK / f"{s}{a.tag}").glob("*.npz")) if not f.stem.endswith(".tmp")]
    jobs = jobs[a.shard::a.num_shards]; print(len(jobs), flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- report
def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def report(a):
    S = {s: cc.load_gen(WORK / f"{s}{a.tag}_scores") for s in SETS}
    D = {s: [dict(np.load(f)) for f in sorted((WORK / f"{s}{a.tag}_scores").glob("*.npz")) if not f.stem.endswith(".tmp")] for s in SETS}
    one = {s: np.ones(len(S[s]["grp"]), bool) for s in SETS}
    mi = lambda s, P: cc.miou(P[:, None], S[s]["cls"], one[s])[0]
    K = lambda arm, m="V3", crf=True: f"{arm}|{m}{'+R2' if crf else ''}|raw"
    vsel = lambda k: int(np.argmax(cc.miou(S["voc"][k], S["voc"]["cls"], one["voc"])))
    prim = {"voc": "oof", "coco": "vsel"}; rname = {"oof": "OOF", "vsel": "VOC-Sim-selected", "55": "0.55"}

    def pred(s, k, rule):
        if rule == "oof":
            return cc.oof(S[s], S[s][k])[1]
        if rule == "55":
            return S[s][k][:, T55]
        return S[s][k][:, vsel(k)]

    # class-adaptive composites (hierarchy + CRF)
    def pcls(s, P, sel):
        return {c: cc.miou(P[sel & (S[s]["cls"] == c)][:, None], S[s]["cls"][sel & (S[s]["cls"] == c)],
                           np.ones(int((sel & (S[s]["cls"] == c)).sum()), bool))[0] for c in np.unique(S[s]["cls"][sel])}

    voof = {arm: pred("voc", K(arm), "oof") for arm in ARMS}
    choice = {}

    def composite(name, cand):
        cv = S["voc"]["cls"]; fo = S["voc"]["fold"]; Pv = np.zeros_like(voof["A"])
        for f in range(5):
            tr = fo != f; sc = {arm: pcls("voc", voof[arm], tr) for arm in cand}
            for c in np.unique(cv):
                best = max(cand, key=lambda arm: (round(sc[arm].get(c, -1), 6), -cand.index(arm)))
                sel = (fo == f) & (cv == c); Pv[sel] = voof[best][sel]
        sc = {arm: pcls("voc", voof[arm], one["voc"]) for arm in cand}
        ch = {c: max(cand, key=lambda arm: (round(sc[arm][c], 6), -cand.index(arm))) for c in np.unique(cv)}
        choice[name] = ch; cc_ = S["coco"]["cls"]; Pc = np.zeros_like(S["coco"][K("A")][:, 0])
        for c in np.unique(cc_):
            arm = ch.get(c, "A"); sel = cc_ == c; Pc[sel] = S["coco"][K(arm)][sel, vsel(K(arm))]
        return {"voc": Pv, "coco": Pc}

    CP = {"CAg": composite("CAg", ["A", "NH", "NC"]), "CAa": composite("CAa", list(("A", "NC", "NH", "NO", "PD", "PI")))}

    def P_of(s, x, m="V3"):
        return CP[x][s] if x in CP else pred(s, K(x, m), prim[s])

    def dci(s, x, y, m="V3"):
        Pa, Pb = P_of(s, x, m), P_of(s, y, m); lo, hi = cc.boot(S[s], Pa, Pb, n=a.boot); d = mi(s, Pa) - mi(s, Pb)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    name = {"A": "A BRACE", "NC": "NC absent-class gate", "NH": "NH NC half strength", "NO": "NO all-pixel contrast",
            "PD": "PD object direction, all-pixel anchor", "PI": "PI object direction, image's own anchor",
            "CAg": "CAg class-adaptive strength", "CAa": "CAa class-adaptive any arm"}
    mname = {"seed": "seed", "V0": "fine-only", "V3": "hierarchy"}
    L = ["# Query-profile gate: mechanism and class-adaptive follow-up (`PREREG.md`)\n",
         "`scripts/jc_qgate.py`. **Exploratory.** One live SD1.5 generation per image on renyi (A100); all arms paired. "
         "Profiles from run 29's statistics, cross-fitted. VOC-Sim primary = OOF thresholds; COCO-Cap (**screened**) "
         f"primary = VOC-Sim-selected thresholds. Paired image bootstrap ({a.boot}).\n", "## Checks\n"]
    for s in SETS:
        L.append(f"- {s}: {len(D[s])} images; recomputed vs runtime seed max |Δ| {max(d['fid_seed'] for d in D[s]):.1e}.")
    for s in SETS:
        L.append(f"\n## {'VOC-Sim' if s == 'voc' else 'COCO-Cap (screened)'}: mIoU, DenseCRF\n")
        L.append(f"| Seed | Seed ({rname[prim[s]]}) | Fine-only ({rname[prim[s]]}) | Hierarchy ({rname[prim[s]]}) | Hierarchy (0.55) | Hierarchy raw ({rname[prim[s]]}) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            L.append(f"| {name[arm]} | " + " | ".join(f"{mi(s, pred(s, K(arm, m), prim[s])):.3f}" for m in MAPS) +
                     f" | {mi(s, pred(s, K(arm), '55')):.3f} | {mi(s, pred(s, K(arm, 'V3', False), prim[s])):.3f} |")
        for x in CP:
            L.append(f"| {name[x]} | — | — | {mi(s, CP[x][s]):.3f} | — | — |")
    L.append("\n## Comparisons (DenseCRF, primary thresholds; exploratory)\n")
    L.append("| Comparison | Map | VOC-Sim | COCO-Cap (screened) |")
    L.append("|---|---|---|---|")
    rows_ = [(x, "A", "V3") for x in ARMS[1:]] + [(x, "NC", "V3") for x in ("NO", "PD", "PI", "NH")] + \
            [("CAg", "A", "V3"), ("CAa", "A", "V3")] + [(x, "A", m) for m in ("seed", "V0") for x in ARMS[1:]]
    for x, y, m in rows_:
        L.append(f"| {x} − {y} | {mname[m]} | {dci('voc', x, y, m)} | {dci('coco', x, y, m)} |")
    L.append("\nClass-adaptive choices (from VOC-Sim, applied to COCO-Cap): " +
             "; ".join(f"**{k}**: " + ", ".join(f"{c} {v}" for c, v in ch.items() if v != "A") + (" (rest A)") for k, ch in choice.items()) + "\n")
    # diagnostics
    for s in SETS:
        d = D[s]; cl = np.array([x["cls"] for x in d])
        L.append(f"\n## Diagnostics, {'VOC-Sim' if s == 'voc' else 'COCO-Cap'}\n")
        L.append("D1 mean gate by region (image mean, then mean over images):\n")
        L.append("| Gate | fg | near bg | far bg | fg − far |"); L.append("|---|---:|---:|---:|---:|")
        for k in GARMS:
            v = [np.nanmean([x[f"gate_{k}_{r}"] for x in d]) for r in ("fg", "near", "far")]
            L.append(f"| {k} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {v[0] - v[2]:+.3f} |")
        L.append("\nD2 share of seed mass by region:\n")
        L.append("| Seed | fg | near bg | far bg |"); L.append("|---|---:|---:|---:|")
        for arm in ARMS:
            L.append(f"| {arm} | " + " | ".join(f"{np.mean([x[f'mass_{arm}_{r}'] for x in d]):.3f}" for r in ("fg", "near", "far")) + " |")
        L.append("\nD1 per class, NC gate fg / near / far (and NO fg / far):\n")
        L.append("| Class | NC fg | NC near | NC far | NO fg | NO far | PI fg | PI far |"); L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for c in np.unique(cl):
            sel = [x for x in d if x["cls"] == c]; f_ = lambda k, r: np.nanmean([x[f"gate_{k}_{r}"] for x in sel])
            L.append(f"| {c} | {f_('NC', 'fg'):.2f} | {f_('NC', 'near'):.2f} | {f_('NC', 'far'):.2f} | {f_('NO', 'fg'):.2f} | "
                     f"{f_('NO', 'far'):.2f} | {f_('PI', 'fg'):.2f} | {f_('PI', 'far'):.2f} |")
        PA, PN = pred(s, K("A"), prim[s]), pred(s, K("NC"), prim[s])
        iou = lambda P: P[:, 0] / np.maximum(P[:, :3].sum(1), 1)
        gain = iou(PN) - iou(PA)
        feats = {"object area": np.array([x["area"] for x in d]), "far-bg seed mass of A": np.array([x["mass_A_far"] for x in d])}
        if s == "coco":
            feats["other classes in caption"] = np.array([x["ncomp"] for x in d], float)
        L.append(f"\nD3 per-image IoU gain NC − A (hierarchy + CRF, {rname[prim[s]]}); mean {gain.mean():+.4f}:\n")
        L.append("| Feature | Spearman ρ | gain, low tertile | mid | high |"); L.append("|---|---:|---:|---:|---:|")
        for fn, v in feats.items():
            if fn == "other classes in caption":
                bins = [v == 0, v == 1, v >= 2]; lab_ = " (0 / 1 / ≥2)"
            else:
                q = np.quantile(v, [1 / 3, 2 / 3]); bins = [v <= q[0], (v > q[0]) & (v <= q[1]), v > q[1]]; lab_ = ""
            L.append(f"| {fn}{lab_} | {spearman(v, gain):+.3f} | " + " | ".join(
                f"{gain[b].mean():+.4f} (n={int(b.sum())})" if b.any() else "—" for b in bins) + " |")
    for s in SETS:
        cl = S[s]["cls"]; arms = list(ARMS) + list(CP)
        L.append(f"\nPer class, {s}, hierarchy + DenseCRF, {rname[prim[s]]} (" + " / ".join(arms) + "): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(P_of(s, x)[:, None], cl, cl == c)[0]:.2f}" for x in arms) for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture"); p.add_argument("--set", choices=SETS, required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default="")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"capture": capture, "score": score, "report": report}[a.cmd](a)
