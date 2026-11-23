#!/usr/bin/env python3
"""runs/37_realvoc_gates/PREREG.md: run 36 gates (MK4, SN, MK4-LR, MK4-RW) on real PASCAL VOC photos (exploratory).

capture  (GPU) one SD1.5 call per item at t = 100 (run 32); seeds {u,w} x ({-} + {A,C} x gates); seed/V0/V1/V3
                                                                             -> <WORK>/cap_{split}{tag}/
score    (CPU) raw counts; val only: DenseCRF on V0/V1/V3 at the train-selected threshold and at 0.55
select   (CPU) train-selected threshold per map                              -> <WORK>/thr{tag}.json
report   (CPU)                                                               -> runs/37_realvoc_gates/result{tag}.md

Profiles: run 33's t = 100 fit (sources A, C). Gate strengths and RW weights: run 36 regime X (stage1.json).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_realvoc as R  # noqa: E402  (run 32)
import jc_gate36 as G36  # noqa: E402  (run 36: consensus)

M, T, cc, st = R.M, R.T, R.cc, R.st
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_realvoc37"
FIT33 = CACHE / "jc_realvoc_t" / "fit_t100.pt"
S36 = ROOT / "runs/36_sn_consensus/stage1.json"
OUT = ROOT / "runs/37_realvoc_gates"
VOC, CLASSES, THR, T55, DEV = R.VOC, R.CLASSES, R.THR, R.T55, R.DEV
SELF_FRAC = 0.30
GATES = ("M", "S", "L", "R")                       # MK4, SN, MK4-LR, MK4-RW
GNAME = {"-": "none", "M": "MK4", "S": "SN", "L": "MK4-LR", "R": "MK4-RW"}
G36KEY = {"M": "MK4", "S": "SN", "L": "MK4-LR", "R": "MK4-RW"}
SRCS = ("C", "A")
SEEDS = ("u-", "w-") + tuple(f"{w}{s}{g}" for w in "uw" for s in SRCS for g in GATES)
PROPS = ("seed", "V0", "V1", "V3")
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in PROPS)
CRF_MAPS = frozenset(f"{s}|{p}" for s in SEEDS for p in ("V0", "V1", "V3"))


def seed_name(W, src, g):
    return f"{'w' if W else 'u'}{'-' if g == '-' else src + g}"


# ---------------------------------------------------------------- gates (run 33 profile layout)
def g_mk4(P, c, m, q, extra=None):
    mo, neg = P["mo"].get((c, m)), P["neg"].get((c, m))
    if mo is None or neg is None:
        return None
    C = neg[4] if extra is None else torch.cat([neg[4], extra[None]])
    return M.g_multi(q, mo, C)


def gates_for(P, c, Q, sneg, Wrw, GS):
    """{gate: {module: g^gamma}}; modules without a profile get g = 1."""
    base, sn = {}, {}
    for m, q in Q.items():
        g = g_mk4(P, c, m, q)
        if g is not None:
            base[m] = g; sn[m] = g_mk4(P, c, m, q, sneg[m])
    res = {m: math.isqrt(q.shape[0]) for m, q in base.items()}
    raw = {"M": base, "S": sn, "L": G36.consensus(base, {m: (1.0 if r < 64 else 0.0) for m, r in res.items()}),
           "R": G36.consensus(base, Wrw)}
    out = {}
    for k, gd in raw.items():
        gm = float(GS[G36KEY[k]]); d = {}
        for m, q in Q.items():
            g = gd.get(m)
            d[m] = torch.ones(q.shape[0], device=q.device) if g is None else g.clamp(0, 1) ** gm
            if not (torch.isfinite(d[m]).all() and float(d[m].min()) >= 0 and float(d[m].max()) <= 1):
                raise ValueError(f"bad gate {k} {m}")
        out[k] = d
    return out, len(Q) - len(base)


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention
    pipe, rec = R.load(); Wm = R.word_map(); K = T.class_keys({"pipe": pipe, "ca_modules": rec.ca_order}, CLASSES)
    S6 = json.loads(S36.read_text()); GS = S6["gstar"]["X"]; Wrw = S6["weights"]["X"]["MK4"]["all"]
    if set(Wrw) != set(rec.ca_order):
        raise ValueError(f"RW module names differ: {sorted(set(Wrw) ^ set(rec.ca_order))}")
    print("gamma", {k: GS[v] for k, v in G36KEY.items()}, flush=True)
    FIT = torch.load(FIT33, weights_only=False)["fit"]; dev_p = {}
    prof = lambda s: dev_p.setdefault(s, M.to_dev(FIT[s]))
    out = WORK / f"cap_{a.split}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=DEV); checked = False
    for it in R.load_items(a.split, a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]; ci = CLASSES.index(c)
        R.one_pass(pipe, rec, np.asarray(Image.open(VOC / f"JPEGImages/{it['img']}.jpg").convert("RGB")),
                   f"A photograph of a {Wm[c]}", Wm[c])
        w_a3 = R.layer_weights(rec); Q = R.qbar(rec); Tm = R.tbar(rec, K)
        sneg = {m: q[Tm[m][ci].float().topk(max(1, round(SELF_FRAC * q.shape[0])), largest=False).indices].mean(0) for m, q in Q.items()}
        sets = {"A": "A_all" if a.split == "val" else ("A_f1" if it["fold"] == 0 else "A_f0"), "C": "C_all"}
        G, miss, gmean = {}, {}, {}
        for src, sname in sets.items():
            P = prof(sname); G[src], miss[src] = gates_for(P, c, Q, sneg, Wrw, GS)
            for k in GATES:
                gmean[f"{src}{k}"] = float(torch.stack([G[src][k][m].mean() for m in Q]).mean())
            if a.tag and not checked:   # smoke: SN with a duplicated centroid == MK4
                d = max(float((g_mk4(P, c, m, q, P["neg"][(c, m)][4][0]) - g_mk4(P, c, m, q)).abs().max())
                        for m, q in Q.items() if (c, m) in P["neg"] and (c, m) in P["mo"])
                print(f"identity SN(duplicate centroid) vs MK4: {d:.2e}", flush=True); assert d < 1e-5
        H = {g: _attention(rec.grid_qk[g], g) for g in R.SA}; maps = {}
        for nm in SEEDS:
            gd = None if nm[1] == "-" else G[nm[1]][nm[2]]
            s = R.make_seed(rec, w_a3 if nm[0] == "w" else w_u, gd); maps[f"{nm}|seed"] = s
            for k, v in R.propagate(H, s, v3=True).items():
                maps[f"{nm}|{k}"] = v
        if a.tag and not checked:   # smoke: identity with aggregate
            from src.segmentation import sg_combined_v2
            ref, _ = sg_combined_v2.aggregate({g: [rec.grid_qk[g]] * 11 for g in R.SA}, maps["w-|seed"])
            worst = max(float(np.abs(st.mm(maps[f"w-|{k}"].cpu().numpy()) - st.mm(ref[k].float().cpu().numpy())).max())
                        for k in ("V0", "V1", "V3"))
            print(f"identity vs sg_combined_v2.aggregate (V0, V1, V3): max |d| {worst:.2e}", flush=True)
            assert worst < 1e-5; checked = True
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {it['id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), cls=c, ci=it["ci"], img=it["img"], fold=it["fold"],
                 split=a.split, w_a3=w_a3.cpu().numpy(), **{f"miss_{k}": v for k, v in miss.items()},
                 **{f"gmean_{k}": v for k, v in gmean.items()})
        os.replace(tmp, o)
        print(f"[rv37:{a.split}] {it['id']} {c} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- score / select (as run 33)
def score_one(job):
    f, o, thr = Path(job[0]), Path(job[1]), job[2]
    if o.exists():
        return
    torch.set_num_threads(1)
    from PIL import Image
    z = np.load(f); img = str(z["img"]); ci = int(z["ci"])
    gt = np.asarray(Image.open(VOC / f"SegmentationClass/{img}.png")).astype(np.int64)
    truth = gt == ci; valid = gt != 255; other = valid & (gt > 0) & ~truth
    res = {"cls": str(z["cls"]), "img": img, "fold": int(z["fold"])}
    res.update({k: z[k] for k in z.files if k.startswith("gmean_")})
    rgb = None
    for i, nm in enumerate(str(x) for x in z["names"]):
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if thr is not None and nm in CRF_MAPS:
            from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
            _import_densecrf(st.CRF_SITE)
            if rgb is None:
                rgb = np.asarray(Image.open(VOC / f"JPEGImages/{img}.jpg").convert("RGB"), np.uint8)
            ms = np.stack([apply_ovam_densecrf(rgb, x >= t, st.CRF_SITE) for t in (thr[nm], float(THR[T55]))]).astype(bool)
            res[f"{nm}|crf"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    thr = json.loads((WORK / f"thr{a.tag}.json").read_text())["thr"] if a.split == "val" else None
    sc = WORK / f"cap_{a.split}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(sc / f.name), thr) for f in sorted((WORK / f"cap_{a.split}{a.tag}").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
    for attempt in range(5):
        todo = [j for j in jobs if not Path(j[1]).exists()]; print(f"attempt {attempt}: {len(todo)} to score", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers) as ex:
                for n, _ in enumerate(ex.map(score_one, todo, chunksize=1), 1):
                    if n % 100 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)
        except BrokenProcessPool:
            print("worker died; restarting pool", flush=True)
    left = [j for j in jobs if not Path(j[1]).exists()]
    if left:
        raise SystemExit(f"{len(left)} items not scored")


def load_scores(split, tag):
    fs = sorted(f for f in (WORK / f"cap_{split}{tag}_scores").glob("*.npz") if not f.stem.endswith(".tmp"))
    Z = [np.load(f) for f in fs]
    S = {"id": np.array([f.stem for f in fs]), "cls": np.array([str(z["cls"]) for z in Z]),
         "grp": np.array([str(z["img"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k or k.startswith("gmean_"):
            S[k] = np.stack([z[k] for z in Z])
    return S


def select(a):
    S = load_scores("train", a.tag); one = np.ones(len(S["cls"]), bool); thr, tr = {}, {}
    for nm in MAPS:
        cv = cc.miou(S[f"{nm}|raw"], S["cls"], one); i = int(np.argmax(cv))
        thr[nm] = float(THR[i]); tr[nm] = float(cv[i])
    (WORK / f"thr{a.tag}.json").write_text(json.dumps({"thr": thr, "train_miou": tr, "items": int(len(S["cls"]))}, indent=1))
    print(f"selected {len(thr)} thresholds")


# ---------------------------------------------------------------- report
def report(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr = Tr["thr"]
    S = load_scores("val", a.tag); BS = R.Boot(S, a.boot); L = []; w = L.append
    S6 = json.loads(S36.read_text()); GS = S6["gstar"]["X"]
    ti = {nm: int(np.flatnonzero(np.isclose(THR, thr[nm]))[0]) for nm in MAPS}
    RULES = {"sel_crf": "+ CRF, train thr (primary)", "055_crf": "+ CRF, 0.55", "sel_raw": "raw, train thr", "055_raw": "raw, 0.55"}

    def P(nm, rule):
        if rule == "sel_raw":
            return S[f"{nm}|raw"][:, ti[nm]]
        if rule == "055_raw":
            return S[f"{nm}|raw"][:, T55]
        return S[f"{nm}|crf"][:, 0 if rule == "sel_crf" else 1]
    cache = {}

    def m(nm, rule="sel_crf"):
        if (nm, rule) not in cache:
            cache[(nm, rule)] = BS.stat(P(nm, rule))
        return cache[(nm, rule)]
    ci = lambda d: (float(d[0]), float(np.percentile(d[1], 2.5)), float(np.percentile(d[1], 97.5)))
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    verdict = lambda x: "**helps**" if x[1] > 0 else "**hurts**" if x[2] < 0 else "not resolved"
    diff = lambda x, y, rule="sel_crf": ci((m(x, rule)[0] - m(y, rule)[0], m(x, rule)[1] - m(y, rule)[1]))
    HN = {"V0": "fine-only (V0)", "V3": "capped hierarchy (V3)", "V1": "uncapped hierarchy (V1)", "seed": "seed only"}

    nimg = len(np.unique(S["grp"]))
    w("# Run 37: run 36 gates on real PASCAL VOC photos (`PREREG.md`)\n")
    w(f"**Exploratory; third look at VOC val.** VOC 2012 val: {nimg} photos, {len(S['cls'])} items; thresholds chosen on VOC train "
      f"({Tr['items']} items, raw maps). One SD1.5 call per item at t = 100. Class-macro mIoU; 95% photo-bootstrap intervals "
      f"({a.boot} samples, paired). Gate strengths fixed from run 36 regime X: "
      + ", ".join(f"{GNAME[k]} γ = {GS[v]:.3f}" for k, v in G36KEY.items()) + ". MK4-RW weights from run 36 regime X.\n")

    w("\n## 1. Primary contrasts (val, + DenseCRF, train-selected threshold; source C, SA weights on)\n")
    w("| # | Contrast | Hierarchy | Δ mIoU [95% CI] | Reading |"); w("|---|---|---|---|---|")
    pri = {}
    for k, (g, h) in enumerate((("L", "V3"), ("L", "V0"), ("R", "V3"), ("R", "V0"), ("S", "V3"), ("S", "V0")), 1):
        d = diff(f"wC{g}|{h}", f"wCM|{h}"); pri[(g, h)] = d
        w(f"| P{k} | {GNAME[g]} − MK4 | {HN[h]} | {f3(d)} | {verdict(d)} |")
    w("")
    for g in ("L", "R", "S"):
        a3, a0 = pri[(g, "V3")], pri[(g, "V0")]
        holds = (a3[1] > 0 or a0[1] > 0) and not (a3[2] < 0 or a0[2] < 0)
        w(f"- {GNAME[g]}: {'**holds on real photos**' if holds else 'does not hold on real photos'} (helps on V3 or V0, hurts on neither).")

    for src in SRCS:
        w(f"\n## {'2' if src == 'C' else '3'}. Source {src} ({'real COCO photos, captions; label-free' if src == 'C' else 'VOC train labels; reference'})\n")
        w("mIoU, primary rule (seed only: raw, train threshold):\n")
        w("| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |"); w("|:-:|---|---:|---:|---:|---:|---:|")
        for W_ in (False, True):
            for h in ("V0", "V3", "V1", "seed"):
                rule = "sel_raw" if h == "seed" else "sel_crf"
                w(f"| {'✓' if W_ else ''} | {HN[h]} | " + " | ".join(f"{m(seed_name(W_, src, g) + '|' + h, rule)[0]:.3f}" for g in ("-",) + GATES) + " |")
        w("\nPaired differences (primary rule; seed only: raw):\n")
        w("| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |"); w("|:-:|---|---|---|---|---|")
        for W_ in (False, True):
            for h in ("V0", "V3", "V1", "seed"):
                rule = "sel_raw" if h == "seed" else "sel_crf"; x = lambda g: seed_name(W_, src, g) + "|" + h
                w(f"| {'✓' if W_ else ''} | {HN[h]} | {f3(diff(x('M'), x('-'), rule))} | " +
                  " | ".join(f3(diff(x(g), x("M"), rule)) for g in ("S", "L", "R")) + " |")
        w("\nComponent effects for MK4-LR (primary rule): SA weights and hierarchy, with the gate fixed:\n")
        w("| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |"); w("|---|---|---|---|---|")
        for g in ("-", "M", "L"):
            n = lambda W_, h: seed_name(W_, src, g) + "|" + h
            w(f"| {GNAME[g]} | {f3(diff(n(True, 'V0'), n(False, 'V0')))} | {f3(diff(n(True, 'V3'), n(False, 'V3')))} | "
              f"{f3(diff(n(False, 'V3'), n(False, 'V0')))} | {f3(diff(n(True, 'V3'), n(True, 'V0')))} |")

    w("\n## 4. Other rules, source C, SA weights on\n")
    w("| Map | Gate | " + " | ".join(RULES.values()) + " | train thr |"); w("|---|---|" + "---:|" * len(RULES) + "---:|")
    for h in ("V0", "V3"):
        for g in ("-",) + GATES:
            nm = seed_name(True, "C", g) + "|" + h
            w(f"| {HN[h]} | {GNAME[g]} | " + " | ".join(f"{m(nm, r)[0]:.3f}" for r in RULES) + f" | {thr[nm]:.3f} |")
    w("\nMK4-LR − MK4 under the other rules (source C, SA weights on):\n")
    w("| Map | " + " | ".join(RULES.values()) + " |"); w("|---|" + "---|" * len(RULES))
    for h in ("V0", "V3"):
        w(f"| {HN[h]} | " + " | ".join(f3(diff(f"wCL|{h}", f"wCM|{h}", r)) for r in RULES) + " |")

    w("\n## 5. Per class IoU (primary rule, source C, SA weights on)\n")
    hdr = [(f"w-|V3", "none V3"), ("wCM|V3", "MK4 V3"), ("wCS|V3", "SN V3"), ("wCL|V3", "LR V3"), ("wCR|V3", "RW V3"),
           ("wCM|V0", "MK4 V0"), ("wCL|V0", "LR V0")]
    w("| Class | items | " + " | ".join(h for _, h in hdr) + " |"); w("|---|---:|" + "---:|" * len(hdr))
    for c in CLASSES:
        sel = S["cls"] == c
        if sel.any():
            w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{cc.miou(P(nm, 'sel_crf')[:, None], S['cls'], sel)[0]:.3f}" for nm, _ in hdr) + " |")
    w("\nMean gate value over modules and pixels (val): " + ", ".join(
        f"{src}/{GNAME[g]} {float(np.mean(S[f'gmean_{src}{g}'])):.3f}" for src in SRCS for g in GATES) + ".")
    w("\nRun 33 t = 100 (different capture, descriptive only, primary rule): all off 0.596, BRACE 0.608, SA weights + uncapped "
      "hierarchy 0.610, full stack A 0.646, full stack C 0.645 (uncapped hierarchy + MK4).\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("capture", "score", "select", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("--split", choices=("train", "val"), default="val")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"capture": capture, "score": score, "select": select, "report": report}[a.cmd](a)
