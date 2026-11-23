#!/usr/bin/env python3
"""runs/37_realvoc_gates/PREREG.md (amended): run 36 gates (MK4, SN, MK4-LR, MK4-RW) on real PASCAL VOC photos, with
gate strength and RW module weights fitted on real VOC train photos (exploratory).

fitw     (GPU) VOC train: per-module AUROC (object vs far) of the MK4 gate, per source  -> <WORK>/fitw{tag}/
capture  (GPU) one SD1.5 call per item at t = 100 (run 32); seeds {u,w} x ({-} + {A,C} x gates x gamma grid);
               maps seed / V0 / V3                                                     -> <WORK>/cap_{split}{tag}/
score    (CPU) raw counts; val only: DenseCRF on the train-selected and gamma = 1 maps
select   (CPU) per map: train threshold; per family and map: train-selected gamma     -> <WORK>/thr{tag}.json
report   (CPU)                                                                         -> runs/37_realvoc_gates/result{tag}.md

Profiles: run 33's t = 100 fit (sources A, C), already fitted on real photos.
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

WORK = CACHE / "jc_realvoc37b"
FIT33 = CACHE / "jc_realvoc_t" / "fit_t100.pt"
OUT = ROOT / "runs/37_realvoc_gates"
VOC, CLASSES, THR, T55, DEV = R.VOC, R.CLASSES, R.THR, R.T55, R.DEV
SELF_FRAC = 0.30
GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
K1 = GRID.index(1.0)
GATES = ("M", "S", "L", "R")                       # MK4, SN, MK4-LR, MK4-RW
GNAME = {"-": "none", "M": "MK4", "S": "SN", "L": "MK4-LR", "R": "MK4-RW"}
SRCS = ("C", "A")
FAMS = tuple(f"{w}{s}{g}" for w in "uw" for s in SRCS for g in GATES)
SEEDS = ("u-", "w-") + tuple(f"{f}{k}" for f in FAMS for k in range(len(GRID)))
PROPS = ("seed", "V0", "V3")
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in PROPS)


def fam(W, src, g):
    return f"{'w' if W else 'u'}-" if g == "-" else f"{'w' if W else 'u'}{src}{g}"


def other(fold):
    return "f1" if fold == 0 else "f0"


def prof_set(split, src, fold):
    return "C_all" if src == "C" else ("A_all" if split == "val" else f"A_{other(fold)}")


# ---------------------------------------------------------------- gates (run 33 profile layout)
def g_mk4(P, c, m, q, extra=None):
    mo, neg = P["mo"].get((c, m)), P["neg"].get((c, m))
    if mo is None or neg is None:
        return None
    C = neg[4] if extra is None else torch.cat([neg[4], extra[None]])
    return M.g_multi(q, mo, C)


def base_gates(P, c, Q, sneg, Wrw):
    """{gate: {module: g}} at gamma = 1 (modules without a profile are left out)."""
    base, sn = {}, {}
    for m, q in Q.items():
        g = g_mk4(P, c, m, q)
        if g is not None:
            base[m] = g; sn[m] = g_mk4(P, c, m, q, sneg[m])
    res = {m: math.isqrt(q.shape[0]) for m, q in base.items()}
    return {"M": base, "S": sn, "L": G36.consensus(base, {m: (1.0 if r < 64 else 0.0) for m, r in res.items()}),
            "R": G36.consensus(base, Wrw)}, len(Q) - len(base)


def powered(gd, Q, gm):
    d = {}
    for m, q in Q.items():
        g = gd.get(m)
        d[m] = torch.ones(q.shape[0], device=q.device) if g is None else g.clamp(0, 1) ** gm
        if not (torch.isfinite(d[m]).all() and float(d[m].min()) >= 0 and float(d[m].max()) <= 1):
            raise ValueError(f"bad gate at {m}")
    return d


def truth_of(img, ci):
    from PIL import Image
    gt = np.asarray(Image.open(VOC / f"SegmentationClass/{img}.png")).astype(np.int64)
    return gt == ci, gt


# ---------------------------------------------------------------- fitw: RW weights from VOC train
@torch.no_grad()
def fitw(a):
    from PIL import Image
    pipe, rec = R.load(); Wm = R.word_map(); FIT = torch.load(FIT33, weights_only=False)["fit"]; dev_p = {}
    prof = lambda s: dev_p.setdefault(s, M.to_dev(FIT[s]))
    out = WORK / f"fitw{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for it in R.load_items("train", a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]
        R.one_pass(pipe, rec, np.asarray(Image.open(VOC / f"JPEGImages/{it['img']}.jpg").convert("RGB")),
                   f"A photograph of a {Wm[c]}", Wm[c])
        Q = R.qbar(rec); truth, _ = truth_of(it["img"], it["ci"])
        fg64 = (F.adaptive_avg_pool2d(torch.from_numpy(truth.astype(np.float32))[None, None], 64)[0, 0] > 0.5).numpy()
        d = {"cls": c, "fold": it["fold"], "modules": np.array(rec.ca_order)}
        for src in SRCS:
            P = prof(prof_set("train", src, it["fold"])); au = np.full(len(rec.ca_order), np.nan)
            for mi, m in enumerate(rec.ca_order):
                g = g_mk4(P, c, m, Q[m])
                if g is None:
                    continue
                fg, _, far = M._regions_at(fg64, math.isqrt(g.shape[0]))
                au[mi] = M.auroc(g[fg.to(g.device)].cpu().numpy(), g[far.to(g.device)].cpu().numpy())
            d[f"auc_{src}"] = au
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **d); os.replace(tmp, o)
        print(f"[rv37:fitw] {it['id']} {c} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear()


def rw_weights(tag):
    """{src: {subset f0/f1/all: {module: w}}}, w = clip(2*(AUC - 0.5), 0, 1), class-balanced AUROC over VOC train items."""
    fs = sorted(f for f in (WORK / f"fitw{tag}").glob("*.npz") if not f.stem.endswith(".tmp"))
    Z = [np.load(f) for f in fs]; mods = [str(x) for x in Z[0]["modules"]]
    cls = np.array([str(z["cls"]) for z in Z]); fold = np.array([int(z["fold"]) for z in Z]); out, aucs = {}, {}
    for src in SRCS:
        A = np.stack([z[f"auc_{src}"] for z in Z]); out[src], aucs[src] = {}, {}
        for sub, sel in (("f0", fold == 0), ("f1", fold == 1), ("all", np.ones(len(fold), bool))):
            au = np.nanmean(np.stack([np.nanmean(A[sel & (cls == c)], 0) for c in np.unique(cls[sel])]), 0)
            out[src][sub] = {m: float(np.clip(2 * (x - 0.5), 0, 1)) for m, x in zip(mods, au)}
            aucs[src][sub] = {m: float(x) for m, x in zip(mods, au)}
    return out, aucs, len(Z)


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention
    pipe, rec = R.load(); Wm = R.word_map(); K = T.class_keys({"pipe": pipe, "ca_modules": rec.ca_order}, CLASSES)
    RW, _, nw = rw_weights(a.tag)
    if set(RW["C"]["all"]) != set(rec.ca_order):
        raise ValueError("RW module names differ from the recorder")
    print(f"RW weights from {nw} VOC train items", flush=True)
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
        B, miss, gmean = {}, {}, {}
        for src in SRCS:
            P = prof(prof_set(a.split, src, it["fold"]))
            wsub = "all" if a.split == "val" else other(it["fold"])
            B[src], miss[src] = base_gates(P, c, Q, sneg, RW[src][wsub])
            for g in GATES:
                gmean[f"{src}{g}"] = float(torch.stack([B[src][g][m].mean() for m in B[src][g]]).mean()) if B[src][g] else 1.0
            if a.tag and not checked:   # smoke: SN with a duplicated centroid == MK4
                d = max(float((g_mk4(P, c, m, q, P["neg"][(c, m)][4][0]) - g_mk4(P, c, m, q)).abs().max())
                        for m, q in Q.items() if (c, m) in P["neg"] and (c, m) in P["mo"])
                print(f"identity SN(duplicate centroid) vs MK4: {d:.2e}", flush=True); assert d < 1e-5
        H = {g: _attention(rec.grid_qk[g], g) for g in R.SA}; maps = {}
        for nm in SEEDS:
            gd = None if nm[1] == "-" else powered(B[nm[1]][nm[2]], Q, GRID[int(nm[3:])])
            s = R.make_seed(rec, w_a3 if nm[0] == "w" else w_u, gd); maps[f"{nm}|seed"] = s
            pr = R.propagate(H, s, v3=True); maps[f"{nm}|V0"] = pr["V0"]; maps[f"{nm}|V3"] = pr["V3"]
        if a.tag and not checked:   # smoke: identity with aggregate
            from src.segmentation import sg_combined_v2
            ref, _ = sg_combined_v2.aggregate({g: [rec.grid_qk[g]] * 11 for g in R.SA}, maps["w-|seed"])
            worst = max(float(np.abs(st.mm(maps[f"w-|{k}"].cpu().numpy()) - st.mm(ref[k].float().cpu().numpy())).max())
                        for k in ("V0", "V3"))
            print(f"identity vs sg_combined_v2.aggregate (V0, V3): max |d| {worst:.2e}", flush=True)
            assert worst < 1e-5; checked = True
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {it['id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), cls=c, ci=it["ci"], img=it["img"], fold=it["fold"],
                 split=a.split, **{f"miss_{k}": v for k, v in miss.items()}, **{f"gmean_{k}": v for k, v in gmean.items()})
        os.replace(tmp, o)
        print(f"[rv37:{a.split}] {it['id']} {c} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- score / select
def score_one(job):
    f, o, thr, crf = Path(job[0]), Path(job[1]), job[2], job[3]
    if o.exists():
        return
    torch.set_num_threads(1)
    from PIL import Image
    z = np.load(f); img = str(z["img"]); ci = int(z["ci"])
    truth, gt = truth_of(img, ci); valid = gt != 255; other_ = valid & (gt > 0) & ~truth
    res = {"cls": str(z["cls"]), "img": img, "fold": int(z["fold"])}
    res.update({k: z[k] for k in z.files if k.startswith("gmean_")})
    rgb = None
    for i, nm in enumerate(str(x) for x in z["names"]):
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other_)
        if thr is not None and nm in crf:
            from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
            _import_densecrf(st.CRF_SITE)
            if rgb is None:
                rgb = np.asarray(Image.open(VOC / f"JPEGImages/{img}.jpg").convert("RGB"), np.uint8)
            ms = np.stack([apply_ovam_densecrf(rgb, x >= t, st.CRF_SITE) for t in (thr[nm], float(THR[T55]))]).astype(bool)
            res[f"{nm}|crf"] = cc.counts(ms, truth, valid, other_)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    thr, crf = None, frozenset()
    if a.split == "val":
        Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr, crf = Tr["thr"], frozenset(Tr["crf"])
    sc = WORK / f"cap_{a.split}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(sc / f.name), thr, crf) for f in sorted((WORK / f"cap_{a.split}{a.tag}").glob("*.npz"))
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
    """Per map: train threshold. Per family and map: gamma with the best train mIoU (raw maps, own best threshold)."""
    S = load_scores("train", a.tag); one = np.ones(len(S["cls"]), bool); thr, tr = {}, {}
    for nm in MAPS:
        cv = cc.miou(S[f"{nm}|raw"], S["cls"], one); i = int(np.argmax(cv))
        thr[nm] = float(THR[i]); tr[nm] = float(cv[i])
    gam = {}
    for f in FAMS:
        for p in PROPS:
            gam[f"{f}|{p}"] = int(np.argmax([tr[f"{f}{k}|{p}"] for k in range(len(GRID))]))
    crf = sorted({f"{s}|{p}" for s in ("u-", "w-") for p in ("V0", "V3")} |
                 {f"{f}{k}|{p}" for f in FAMS for p in ("V0", "V3") for k in (gam[f"{f}|{p}"], K1)})
    (WORK / f"thr{a.tag}.json").write_text(json.dumps({"thr": thr, "train_miou": tr, "gamma": gam, "crf": crf,
                                                       "items": int(len(S["cls"]))}, indent=1))
    print(f"selected {len(thr)} thresholds; gamma: " + ", ".join(f"{k} {GRID[v]}" for k, v in gam.items()), flush=True)


# ---------------------------------------------------------------- report
def report(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr, gam = Tr["thr"], Tr["gamma"]
    S = load_scores("val", a.tag); BS = R.Boot(S, a.boot); L = []; w = L.append
    ti = {nm: int(np.flatnonzero(np.isclose(THR, thr[nm]))[0]) for nm in MAPS}
    RULES = {"sel_crf": "+ CRF, train thr (primary)", "055_crf": "+ CRF, 0.55", "sel_raw": "raw, train thr", "055_raw": "raw, 0.55"}

    def name(W, src, g, p, fixed=False):
        f = fam(W, src, g)
        if g == "-":
            return f"{f}|{p}"
        return f"{f}{K1 if fixed else gam[f'{f}|{p}']}|{p}"

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
    rl = lambda p: "sel_raw" if p == "seed" else "sel_crf"
    HN = {"V0": "fine-only (V0)", "V3": "capped hierarchy (V3)", "seed": "seed only"}
    _, aucs, nw = rw_weights(a.tag)

    nimg = len(np.unique(S["grp"]))
    w("# Run 37: run 36 gates on real PASCAL VOC photos, strengths and RW weights fitted on VOC train (`PREREG.md`)\n")
    w(f"**Exploratory; third look at VOC val.** VOC 2012 val: {nimg} photos, {len(S['cls'])} items. Fitted on VOC train "
      f"({Tr['items']} items): thresholds per map, gate strength γ per gate family and map (grid {list(GRID)}), RW module "
      f"weights ({nw} items). One SD1.5 call per item at t = 100. Class-macro mIoU; 95% photo-bootstrap intervals "
      f"({a.boot} samples, paired).\n")

    w("\n## 1. Primary contrasts (val, + DenseCRF, train-selected threshold and γ; source C, SA weights on)\n")
    w("| # | Contrast | Hierarchy | Δ mIoU [95% CI] | Reading |"); w("|---|---|---|---|---|")
    pri = {}
    for k, (g, h) in enumerate((("L", "V3"), ("L", "V0"), ("R", "V3"), ("R", "V0"), ("S", "V3"), ("S", "V0")), 1):
        d = diff(name(True, "C", g, h), name(True, "C", "M", h)); pri[(g, h)] = d
        w(f"| P{k} | {GNAME[g]} − MK4 | {HN[h]} | {f3(d)} | {verdict(d)} |")
    w("")
    for g in ("L", "R", "S"):
        a3, a0 = pri[(g, "V3")], pri[(g, "V0")]
        holds = (a3[1] > 0 or a0[1] > 0) and not (a3[2] < 0 or a0[2] < 0)
        w(f"- {GNAME[g]}: {'**holds on real photos**' if holds else 'does not hold on real photos'} (helps on V3 or V0, hurts on neither).")

    w("\n## 2. Train-selected γ\n")
    w("| Source | SA weights | Map | MK4 | SN | MK4-LR | MK4-RW |"); w("|---|:-:|---|---:|---:|---:|---:|")
    for src in SRCS:
        for W_ in (False, True):
            for p in PROPS:
                w(f"| {src} | {'✓' if W_ else ''} | {HN[p]} | " + " | ".join(f"{GRID[gam[fam(W_, src, g) + '|' + p]]}" for g in GATES) + " |")

    for si, src in enumerate(SRCS):
        w(f"\n## {3 + si}. Source {src} ({'real COCO photos, captions; label-free profiles' if src == 'C' else 'VOC train labels; reference'})\n")
        w("mIoU, primary rule (seed only: raw, train threshold), train-selected γ:\n")
        w("| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |"); w("|:-:|---|---:|---:|---:|---:|---:|")
        for W_ in (False, True):
            for p in ("V0", "V3", "seed"):
                w(f"| {'✓' if W_ else ''} | {HN[p]} | " + " | ".join(f"{m(name(W_, src, g, p), rl(p))[0]:.3f}" for g in ("-",) + GATES) + " |")
        w("\nPaired differences, train-selected γ:\n")
        w("| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |"); w("|:-:|---|---|---|---|---|")
        for W_ in (False, True):
            for p in ("V0", "V3", "seed"):
                x = lambda g: name(W_, src, g, p)
                w(f"| {'✓' if W_ else ''} | {HN[p]} | {f3(diff(x('M'), x('-'), rl(p)))} | " +
                  " | ".join(f3(diff(x(g), x("M"), rl(p))) for g in ("S", "L", "R")) + " |")
        w("\nComponent effects with the gate fixed (primary rule):\n")
        w("| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |"); w("|---|---|---|---|---|")
        for g in ("-",) + GATES:
            n = lambda W_, h: name(W_, src, g, h)
            w(f"| {GNAME[g]} | {f3(diff(n(True, 'V0'), n(False, 'V0')))} | {f3(diff(n(True, 'V3'), n(False, 'V3')))} | "
              f"{f3(diff(n(False, 'V3'), n(False, 'V0')))} | {f3(diff(n(True, 'V3'), n(True, 'V0')))} |")
        w("\nDescriptive: every gate at γ = 1 (untuned), SA weights on, primary rule:\n")
        w("| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4 |"); w("|---|---:|---:|---:|---:|---|")
        for p in ("V0", "V3"):
            w(f"| {HN[p]} | " + " | ".join(f"{m(name(True, src, g, p, True))[0]:.3f}" for g in GATES) +
              f" | {f3(diff(name(True, src, 'L', p, True), name(True, src, 'M', p, True)))} |")

    w("\n## 5. Other rules, source C, SA weights on (train-selected γ)\n")
    w("| Map | Gate | " + " | ".join(RULES.values()) + " |"); w("|---|---|" + "---:|" * len(RULES))
    for p in ("V0", "V3"):
        for g in ("-",) + GATES:
            nm = name(True, "C", g, p)
            w(f"| {HN[p]} | {GNAME[g]} | " + " | ".join(f"{m(nm, r)[0]:.3f}" for r in RULES) + " |")

    w("\n## 6. Per class IoU (primary rule, source C, SA weights on)\n")
    hdr = [(name(True, "C", g, p), f"{GNAME[g]} {p}") for p in ("V3", "V0") for g in ("-",) + GATES]
    w("| Class | items | " + " | ".join(h for _, h in hdr) + " |"); w("|---|---:|" + "---:|" * len(hdr))
    for c in CLASSES:
        sel = S["cls"] == c
        if sel.any():
            w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{cc.miou(P(nm, 'sel_crf')[:, None], S['cls'], sel)[0]:.3f}" for nm, _ in hdr) + " |")

    w("\n## 7. RW module weights (VOC train, all items; w = clip(2·(AUROC − 0.5), 0, 1) of the MK4 gate, object vs far)\n")
    mods = list(aucs["C"]["all"])
    w("| Source | " + " | ".join(str(k) for k in range(len(mods))) + " |"); w("|---|" + "---:|" * len(mods))
    for src in SRCS:
        w(f"| {src} AUROC | " + " | ".join(f"{aucs[src]['all'][x]:.3f}" for x in mods) + " |")
    w(f"\nModules: " + ", ".join(f"{k} = {x}" for k, x in enumerate(mods)))
    w("\nMean gate value at γ = 1 over modules and pixels (val): " + ", ".join(
        f"{src}/{GNAME[g]} {float(np.mean(S[f'gmean_{src}{g}'])):.3f}" for src in SRCS for g in GATES) + ".")
    w("\nRun 33 t = 100 (different capture, descriptive only, primary rule): all off 0.596, BRACE 0.608, full stack C 0.645 "
      "(uncapped hierarchy + MK4).\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("fitw", "capture", "score", "select", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("--split", choices=("train", "val"), default="val")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"fitw": fitw, "capture": capture, "score": score, "select": select, "report": report}[a.cmd](a)
