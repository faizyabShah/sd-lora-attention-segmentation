#!/usr/bin/env python3
"""runs/29_absent_class_study/PREREG.md: absent-class calibration as a component, VOC-Sim and COCO-Cap.

Arms (seed):  A BRACE | NC cross-fitted profiles (primary) | NO all-pixel control | NL leave-one-out (run 28)
              | NX profiles from the other benchmark.     Maps: seed, V0 (fine-only), V3 (hierarchy).

stats    (GPU, sharded, --set voc|coco) per-image query statistics -> <CACHE>/jc_absent/stats_{set}{tag}/
fit      (CPU) fold sums for both sets                               -> <CACHE>/jc_absent/fit{tag}.pt
capture  (GPU, sharded, --set) maps [5, 3, 64, 64]                   -> <CACHE>/jc_absent/{set}{tag}/
score    (CPU, sharded) raw and + DenseCRF R2 counts at 0.30-0.70
report   (CPU) runs/29_absent_class_study/result{tag}.md
"""
from __future__ import annotations

import argparse, os, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_trio as T  # noqa: E402

J, cc, st, nr = T.J, T.cc, T.st, T.nr
WORK = J.CACHE / "jc_absent"
OUT = ROOT / "runs/29_absent_class_study"
ARMS = ("A", "NC", "NO", "NL", "NX")
MAPS = ("seed", "V0", "V3")
THR, T55 = J.THR, J.T55
SETS = ("voc", "coco")


def fold(iid):
    return zlib.crc32(iid.encode()) % 5


def label(row):
    return row["label"].replace("-", " ")


def rows(a):
    return nr.rows(a.set, a.shard, a.num_shards, a.max_images)


# ---------------------------------------------------------------- statistics pass
@torch.no_grad()
def stats(a):
    rt = J.make_runtime(a.model); K = T.class_keys(rt, T.CLASSES)
    out = WORK / f"stats_{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows(a):
        o = out / f"{row['image_id']}.pt"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, a.set)
        Q = T.qbar(rec); _, Tm = T.ca_seeds(rec, K, tbar=True); lab = label(row); txt = row["prompt"]
        res = {"label": lab, "obj": {}, "spur": {}, "all": {}}
        for m in Q:
            res["all"][m] = (Q[m].sum(0).cpu(), Q[m].shape[0])
            t = Tm[(lab, m)]; idx = t.topk(max(1, int(round(T.TOP_OBJ * t.numel())))).indices
            res["obj"][m] = (Q[m][idx].sum(0).cpu(), len(idx))
            for c in T.CLASSES:
                if c == lab or T.mentions(txt, c):
                    continue
                t = Tm[(c, m)]; idx = t.topk(max(1, int(round(T.TOP_SPUR * t.numel())))).indices
                res["spur"][(c, m)] = (Q[m][idx].sum(0).cpu(), len(idx))
        torch.save(res, o)
        print(f"[absent:stats:{a.set}] {row['image_id']} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


def _add(d, key, v, n):
    s = d.setdefault(key, [0, 0]); s[0] = s[0] + v; s[1] += n


def fit(a):
    """Per set and fold: sums for obj[(c, m)], spur[(c, m)] and all[m] (all pixels of every image, for the NO
    control), plus each image's own object sums (leave-one-out, NL)."""
    R = {}
    for s in SETS:
        fs = sorted((WORK / f"stats_{s}{a.tag}").glob("*.pt"))
        fo = {f: {k: {} for k in ("obj", "spur", "all")} for f in range(5)}
        img = {}
        for p in fs:
            z = torch.load(p); f = fold(p.stem); lab = z["label"]; img[p.stem] = (lab, z["obj"])
            for m, (v, n) in z["obj"].items():
                _add(fo[f]["obj"], (lab, m), v, n)
            for key, (v, n) in z["spur"].items():
                _add(fo[f]["spur"], key, v, n)
            for m, (v, n) in z["all"].items():
                _add(fo[f]["all"], m, v, n)
        R[s] = {"fold": fo, "img": img, "n": len(fs)}
        print(f"fit {s}: {len(fs)} images; per-fold image counts "
              f"{[sum(fold(p.stem) == f for p in fs) for f in range(5)]}", flush=True)
    torch.save(R, WORK / f"fit{a.tag}.pt")


def _sum(Fs, kind, key, folds):
    v, n = 0, 0
    for f in folds:
        x = Fs["fold"][f][kind].get(key)
        if x is not None:
            v = v + x[0]; n += x[1]
    return (v, n) if n > 0 else (None, 0)


def _gate(mo, ms, q):
    w = F.normalize(mo - ms, dim=0); po, ps = float(w @ mo), float(w @ ms)
    return torch.sigmoid(T.GAIN * (q @ w - (po + ps) / 2) / max(po - ps, 1e-6))


def gates(R, ds, iid, lab, Q, arm):
    other = "coco" if ds == "voc" else "voc"
    Fs = R[other] if arm == "NX" else R[ds]
    folds = [f for f in range(5) if f != fold(iid)] if arm in ("NC", "NO") else list(range(5))
    g, miss = {}, 0
    for m, q in Q.items():
        vo, no = _sum(Fs, "obj", (lab, m), folds)
        if arm == "NL" and vo is not None and iid in Fs["img"]:
            vi, ni = Fs["img"][iid][1][m]; vo = vo - vi; no -= ni
        vs, ns = _sum(Fs, "all", m, folds) if arm == "NO" else _sum(Fs, "spur", (lab, m), folds)
        if vo is None or no <= 0 or vs is None or ns <= 0:
            g[m] = torch.ones(q.shape[0], device=q.device); miss += 1; continue
        g[m] = _gate((vo / no).to(q.device), (vs / ns).to(q.device), q)
    return g, miss


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(a.model); R = torch.load(WORK / f"fit{a.tag}.pt")
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows(a):
        o = out / f"{row['image_id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, a.set); lab = label(row); iid = row["image_id"]
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); Q = T.qbar(rec); tgt = (None, list(rec.target))
        G = {arm: gates(R, a.set, iid, lab, Q, arm) for arm in ARMS[1:]}
        S = T.ca_seeds(rec, {"A": tgt, **{arm: tgt for arm in ARMS[1:]}}, gate={arm: G[arm][0] for arm in G})
        fid = float((S["A"] - seed).abs().max())
        if fid > 1e-2:
            raise ValueError(f"recomputed seed differs from runtime seed by {fid:.2e} for {iid}")
        S["A"] = seed
        maps = []
        for arm in ARMS:
            v = sg_combined_v2.aggregate(sa, S[arm])[0]
            maps.append(np.stack([x.float().reshape(64, 64).cpu().numpy() for x in (S[arm], v["V0"], v["V3"])]))
        maps = np.stack(maps).astype(np.float32)
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        gm = {arm: float(np.mean([float(x.mean()) for x in G[arm][0].values()])) for arm in G}
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=lab, truth=np.asarray(sample.target_mask, bool),
                 image=np.asarray(sample.generated_image, np.uint8), fid_seed=fid,
                 **{f"gate_mean_{k}": v for k, v in gm.items()}, **{f"gate_miss_{k}": G[k][1] for k in G})
        os.replace(tmp, o)
        print(f"[absent:{a.set}] {iid} fid {fid:.1e} gates " + " ".join(f"{k} {v:.2f}/{G[k][1]}" for k, v in gm.items())
              + f" {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


# ---------------------------------------------------------------- score / report
def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "fid_seed": float(z["fid_seed"]),
           **{k: z[k] for k in z.files if k.startswith("gate_")}}
    for ai, arm in enumerate(ARMS):
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(z["maps"][ai, mi].astype(np.float32)), truth.shape)
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
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


def report(a):
    S = {s: cc.load_gen(WORK / f"{s}{a.tag}_scores") for s in SETS}
    Z = {s: [np.load(f) for f in sorted((WORK / f"{s}{a.tag}_scores").glob("*.npz")) if not f.stem.endswith(".tmp")] for s in SETS}
    one = {s: np.ones(len(S[s]["grp"]), bool) for s in SETS}
    mi = lambda s, P: cc.miou(P[:, None], S[s]["cls"], one[s])[0]
    key = lambda arm, m, crf: f"{arm}|{m}{'+R2' if crf else ''}|raw"
    vsel = lambda k: int(np.argmax(cc.miou(S["voc"][k], S["voc"]["cls"], one["voc"])))

    def pred(s, k, rule):
        if rule == "oof":
            return cc.oof(S[s], S[s][k])[1]
        if rule == "55":
            return S[s][k][:, T55]
        if rule == "vsel":
            return S[s][k][:, vsel(k)]
        return S[s][k][:, int(np.argmax(cc.miou(S[s][k], S[s]["cls"], one[s])))]       # own best (in-sample)

    def dci(s, kx, ky, rule):
        Pa, Pb = pred(s, kx, rule), pred(s, ky, rule); lo, hi = cc.boot(S[s], Pa, Pb, n=a.boot); d = mi(s, Pa) - mi(s, Pb)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", d, lo

    prim = {"voc": "oof", "coco": "vsel"}
    rname = {"oof": "OOF", "55": "0.55", "vsel": "VOC-Sim-selected", "own": "own best"}
    lab = {"A": "A BRACE seed", "NC": "NC calibrated, cross-fitted", "NO": "NO control (all-pixel mean)",
           "NL": "NL calibrated, leave-one-out", "NX": "NX calibrated, other benchmark's profiles"}
    mname = {"seed": "seed", "V0": "fine-only", "V3": "hierarchy"}
    L = ["# Absent-class calibration: component study (`PREREG.md`)\n",
         "`scripts/jc_absent.py`. One live SD1.5 generation per image, all arms from it (paired). Class-macro mIoU; "
         f"paired image-bootstrap 95% intervals ({a.boot}). VOC-Sim = development. **COCO-Cap = screened** (run 28 "
         "already looked at this idea there); NX on COCO-Cap uses VOC-Sim profiles only.\n", "## Data and checks\n"]
    for s in SETS:
        z = Z[s]
        L.append(f"- {s}: {len(z)} images, {len(np.unique(S[s]['cls']))} classes; recomputed vs runtime seed max |Δ| "
                 f"{max(float(x['fid_seed']) for x in z):.1e}; mean gate " +
                 ", ".join(f"{k} {np.mean([float(x[f'gate_mean_{k}']) for x in z]):.3f}" for k in ARMS[1:]) +
                 "; modules without a profile " + ", ".join(f"{k} {sum(int(x[f'gate_miss_{k}']) for x in z)}" for k in ARMS[1:])
                 + f" of {11 * len(z)}.")
    for crf in (True, False):
        for s in SETS:
            rules = ("oof", "55", "own") if s == "voc" else ("vsel", "55", "own")
            L.append(f"\n## {'VOC-Sim' if s == 'voc' else 'COCO-Cap (screened)'}, {'DenseCRF' if crf else 'raw'}\n")
            L.append("| Seed | Map | " + " | ".join(rname[r] for r in rules) + " |")
            L.append("|---|---|" + "---:|" * len(rules))
            for m in MAPS:
                for arm in ARMS:
                    k = key(arm, m, crf); cells = []
                    for r in rules:
                        v = f"{mi(s, pred(s, k, r)):.3f}"
                        if r == "own":
                            v += f" ({THR[int(np.argmax(cc.miou(S[s][k], S[s]['cls'], one[s])))]:.3f})"
                        if r == "vsel":
                            v += f" ({THR[vsel(k)]:.3f})"
                        cells.append(v)
                    L.append(f"| {lab[arm]} | {mname[m]} | " + " | ".join(cells) + " |")
    L.append("\n## Registered comparisons (DenseCRF)\n")
    L.append("| ID | Benchmark | Comparison | Threshold | Δ [95% CI] | Interval above zero? |")
    L.append("|---|---|---|---|---|---|")
    K = lambda arm, m="V3": key(arm, m, True)
    tests = [("P1 (primary)", "voc", "NC − A, hierarchy", K("NC"), K("A"), "oof"),
             ("P1b", "voc", "NC − A, hierarchy", K("NC"), K("A"), "55"),
             ("P2", "coco", "NX − A, hierarchy", K("NX"), K("A"), "vsel"),
             ("P3", "coco", "NC − A, hierarchy", K("NC"), K("A"), "vsel")]
    for s in SETS:
        tests.append(("P4", s, "NC − NO, hierarchy", K("NC"), K("NO"), prim[s]))
    for s in SETS:
        for m in ("seed", "V0"):
            tests.append(("P5", s, f"NC − A, {mname[m]}", K("NC", m), K("A", m), prim[s]))
        for arm in ("A", "NC"):
            tests.append(("P5", s, f"hierarchy − fine-only, {arm}", K(arm), K(arm, "V0"), prim[s]))
    for s in SETS:
        tests.append(("P6", s, "NL − NC, hierarchy", K("NL"), K("NC"), prim[s]))
    tests += [("—", "coco", "NC − A, hierarchy (run 28 convention)", K("NC"), K("A"), "55"),
              ("—", "coco", "NL − A, hierarchy (run 28 convention)", K("NL"), K("A"), "55"),
              ("—", "coco", "NX − A, hierarchy", K("NX"), K("A"), "55")]
    res = {}
    for tid, s, lb, kx, ky, r in tests:
        c = dci(s, kx, ky, r); res[(tid, s, lb, r)] = c
        L.append(f"| {tid} | {'VOC-Sim' if s == 'voc' else 'COCO-Cap'} | {lb} | {rname[r]} | {c[0]} | "
                 f"{'yes' if round(c[2], 3) > 0 else 'no'} |")
    p1 = res[("P1 (primary)", "voc", "NC − A, hierarchy", "oof")][2]
    p2 = res[("P2", "coco", "NX − A, hierarchy", "vsel")][2]
    p4 = res[("P4", "voc", "NC − NO, hierarchy", "oof")][2]
    L.append("\n## Reading (rule fixed in `PREREG.md`)\n")
    L.append(f"- Development (P1): **{'pass' if round(p1, 3) > 0 else 'fail'}**.")
    L.append(f"- Absent-class set matters on VOC-Sim (P4): **{'supported' if round(p4, 3) > 0 else 'not supported'}**.")
    L.append(f"- P1 and P2 both above zero (bring to owner): **{'yes' if round(p1, 3) > 0 and round(p2, 3) > 0 else 'no'}**.\n")
    for s in SETS:
        cl = S[s]["cls"]
        L.append(f"Per class, {s}, hierarchy + DenseCRF, {rname[prim[s]]} (A / NC / NO / NL / NX): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(pred(s, K(arm), prim[s])[:, None], cl, cl == c)[0]:.2f}" for arm in ARMS)
            for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    M = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    for nm in ("stats", "capture"):
        p = sub.add_parser(nm); p.add_argument("--set", choices=SETS, required=True)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default=""); p.add_argument("--model", default=M)
    p = sub.add_parser("fit"); p.add_argument("--tag", default="")
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"stats": stats, "fit": fit, "capture": capture, "score": score, "report": report}[a.cmd](a)
