#!/usr/bin/env python3
"""runs/38_gen_gates_factorial/PREREG.md: run 37's factorial (SA weights x hierarchy x gate, gamma grid) on the
generated benchmarks VOC-Sim and COCO-Cap, regimes V and X, everything fitted on dev (exploratory).

capture  (GPU) all images, generation-time; seeds {u,w} x ({-} + {X,V} x gates x gamma grid); maps seed / V0 / V3
                                                                               -> <WORK>/{set}{tag}/
score    (CPU) raw counts, all images                                          -> <WORK>/{set}{tag}_scores/
select   (CPU) per regime and map: dev threshold; per regime x W x gate x map: dev gamma -> <WORK>/thr{tag}.json
crf      (CPU) test images: DenseCRF on V0 / V3 at the dev threshold and 0.55   -> <WORK>/{set}{tag}_crf/
report   (CPU)                                                                  -> runs/38_gen_gates_factorial/result{tag}.md

Profiles: run 35's fit. Gates: run 36 (jc_gate36.gate_set). RW weights: run 36 stage1.json (dev, held out).
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
import jc_gate36 as G36  # noqa: E402  (run 36: gates, consensus)
import jc_realvoc as R  # noqa: E402  (run 32: photo bootstrap)

M, G34, AB, T, J, cc, st = G36.M, G36.G34, G36.AB, G36.T, G36.J, G36.cc, G36.st
WORK = J.CACHE / "jc_gen38"
W35 = J.CACHE / "jc_gate35"
S36 = ROOT / "runs/36_sn_consensus/stage1.json"
OUT = ROOT / "runs/38_gen_gates_factorial"
SETS, CLASSES, THR, T55, REGIMES, DEV = M.SETS, M.CLASSES, M.THR, M.T55, M.REGIMES, M.DEV
SELF_FRAC = G36.SELF_FRAC
GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
K1 = GRID.index(1.0)
GATES = ("M", "S", "L", "R")
GNAME = {"-": "none", "M": "MK4", "S": "SN", "L": "MK4-LR", "R": "MK4-RW"}
FAMS = tuple(f"{r}{w}{g}" for r in REGIMES for w in "uw" for g in GATES)
SEEDS = ("u-", "w-") + tuple(f"{f}{k}" for f in FAMS for k in range(len(GRID)))
PROPS = ("seed", "V0", "V3")
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in PROPS)
SETN = {"voc": "VOC-Sim", "coco": "COCO-Cap"}


def fam(r, W, g):
    return f"{'w' if W else 'u'}-" if g == "-" else f"{r}{'w' if W else 'u'}{g}"


# ---------------------------------------------------------------- seeds and propagation
def tsums(rec):
    """Per module: A3-weighted and uniform sums over calls of the head-mean target attention (recorded target key)."""
    wmap = {s: w for s, lab, w in rec.weights if lab == "selected"}
    if len(wmap) != 31:
        raise ValueError("expected 31 selected weight vectors")
    mods = list(rec.selected_modules); tgt = list(rec.target); Tw, Tu = {}, {}
    for step, name, q, kst, mask, scale in rec.ca_store:
        s = q.float() @ kst.float().transpose(-1, -2) * scale
        if mask is not None:
            s = s + mask.float()
        t = s.softmax(-1)[..., tgt].mean(-1).mean(0); w = float(wmap[step][mods.index(name)])
        Tw[name] = Tw.get(name, 0) + t * w / 31; Tu[name] = Tu.get(name, 0) + t / 31
    return Tw, Tu


def make_seed(Ts, gate=None):
    acc = None
    for m, t in Ts.items():
        x = t if gate is None else t * gate[m]; side = math.isqrt(x.shape[0])
        u = F.interpolate(x.reshape(1, 1, side, side), size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        acc = u if acc is None else acc + u
    return J.mm_t(acc)


def powered(gd, Q, gm):
    d = {}
    for m, q in Q.items():
        g = gd.get(m)
        d[m] = torch.ones(q.shape[0], device=q.device) if g is None else g.clamp(0, 1) ** gm
        if not (torch.isfinite(d[m]).all() and float(d[m].min()) >= 0 and float(d[m].max()) <= 1):
            raise ValueError(f"bad gate at {m}")
    return d


def _cap_b(parent, proposal, coarse, eps=1e-12):
    """Batched sg_combined_v2.cap (scale 1): parent [B, c*c], proposal [B, (2c)^2]."""
    B = parent.shape[0]; p = parent.float().clamp_min(0).reshape(B, coarse, coarse); q = proposal.float().clamp_min(0)
    qq = q.reshape(B, coarse, 2, coarse, 2); m = qq.mean((2, 4))
    f = torch.where(m > p, p / (m + eps), torch.ones_like(m))
    return (qq * f[:, :, None, :, None]).reshape(B, -1)


def _app(A, V):
    """A [h, N, N]; V [B, N] (shared over heads) or [h, N, B] -> [h, N, B]."""
    if V.ndim == 2:
        V = V.T[None].expand(A.shape[0], -1, -1)
    return torch.bmm(A, V)


@torch.no_grad()
def propagate(sa, seeds):
    """Batched V0 (fine-only) and V3 (capped hierarchy), as sg_combined_v2.aggregate. seeds [B, 64, 64]."""
    from src.segmentation.sg25_sa import _attention, _resize
    c64 = seeds.float().reshape(seeds.shape[0], -1); c16 = _resize(c64, 64, 16)
    V0 = torch.zeros_like(c64); V3 = torch.zeros_like(c64)
    for call in range(11):
        h16, h32, h64 = _attention(sa[16][call], 16), _attention(sa[32][call], 32), _attention(sa[64][call], 64)
        u = _app(h64, c64); V0 += _app(h64, u).mean(0).T / 11
        raw16 = _app(h16, c16).mean(0).T
        raw32 = _app(h32, _resize(raw16, 16, 32)).mean(0).T
        z32 = _cap_b(raw16, raw32, 16)
        V3 += _app(h64, _resize(z32, 32, 64)).mean(0).T / 11
        del h16, h32, h64, u
    return V0.reshape(-1, 64, 64), V3.reshape(-1, 64, 64)


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(G36.MODEL); K = T.class_keys(rt, CLASSES)
    FIT = torch.load(W35 / "fit.pt")["fit"]; W = json.loads(S36.read_text())["weights"]; Sp = M.load_split(); cache = {}
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True); checked = False
    for row in M.rows_for(a, "all"):
        iid = row["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); sp = Sp[iid]; rec, sample = J.generate(rt, row, a.set); lab = AB.label(row)
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        a3 = rec.seeds()["selected/A3"].float(); Q = T.qbar(rec); tgt = (None, list(rec.target))
        _, Tm = T.ca_seeds(rec, {lab: K[lab]}, tbar=True)
        sneg = {m: q[Tm[(lab, m)].float().topk(max(1, round(SELF_FRAC * q.shape[0])), largest=False).indices].mean(0) for m, q in Q.items()}
        Tw, Tu = tsums(rec); B = {}; miss = 0; gmean = {}
        for r in REGIMES:
            sub = "all" if (sp["part"] == "test" or (r == "V" and sp["ds"] == "coco")) else ("dB" if sp["fold"] == "dA" else "dA")
            if (r, sub) not in cache:
                cache[(r, sub)] = G34.to_dev(FIT[r][sub])
            P = cache[(r, sub)]
            Gs, mi_ = G36.gate_set(P, lab, Q, sneg, W[r], sub); miss += mi_
            for g in GATES:
                B[(r, g)] = Gs[GNAME[g]]
                gmean[f"{r}{g}"] = float(torch.stack([x.mean() for x in Gs[GNAME[g]].values()]).mean()) if Gs[GNAME[g]] else 1.0
            if a.tag and not checked:   # smoke: SN with a duplicated centroid == MK4
                d = max(float((G36.G35.gate(P, "SN", lab, m, Q[m], P["C4"][(lab, m)][0]) - Gs["MK4"][m]).abs().max()) for m in Gs["MK4"])
                print(f"identity SN(duplicate centroid) vs MK4: {d:.2e}", flush=True); assert d < 1e-5
        seeds = []
        for nm in SEEDS:
            if nm[1] == "-":
                seeds.append(make_seed(Tw if nm[0] == "w" else Tu))
            else:
                seeds.append(make_seed(Tw if nm[1] == "w" else Tu, powered(B[(nm[0], nm[2])], Q, GRID[int(nm[3:])])))
        seeds = torch.stack(seeds).float()
        fid = float((seeds[1] - a3).abs().max())
        if fid > 1e-2:
            raise ValueError(f"seed fidelity {fid:.2e} for {iid}")
        if a.tag and not checked:   # smoke: seeds == jc_trio.ca_seeds; propagation == aggregate
            gd = powered(B[("X", "M")], Q, 1.0); ref = T.ca_seeds(rec, {"x": tgt, "g": tgt}, gate={"g": gd})
            d0 = float((seeds[1] - ref["x"]).abs().max()); d1 = float((make_seed(Tw, gd) - ref["g"]).abs().max())
            print(f"identity seed vs ca_seeds: ungated {d0:.2e}, gated {d1:.2e}; A3 fidelity {fid:.2e}", flush=True)
            assert d0 < 1e-5 and d1 < 1e-5
        V0, V3 = propagate(sa, seeds)
        if a.tag and not checked:
            ref = sg_combined_v2.aggregate(sa, seeds[1])[0]; mm = lambda x: (x - x.min()) / (x.max() - x.min()).clamp_min(1e-20)
            d = max(float((mm(V0[1]) - mm(ref["V0"].float())).abs().max()), float((mm(V3[1]) - mm(ref["V3"].float())).abs().max()))
            print(f"identity batched propagation vs aggregate (V0, V3): {d:.2e}", flush=True); assert d < 1e-5; checked = True
        maps = torch.stack([seeds, V0, V3], 1).reshape(-1, 64, 64).cpu().numpy()     # SEEDS-major, PROPS-minor = MAPS order
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), names=np.array(MAPS), cls=lab, truth=np.asarray(sample.target_mask, bool),
                 image=np.asarray(sample.generated_image, np.uint8), part=sp["part"], fold=str(sp["fold"]), gate_miss=miss,
                 **{f"gmean_{k}": v for k, v in gmean.items()})
        os.replace(tmp, o)
        print(f"[g38:{a.set}] {iid} {sp['part']} fid {fid:.1e} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


# ---------------------------------------------------------------- score (raw) / select / crf
def _run_pool(jobs, fn, workers):
    for attempt in range(5):
        todo = [j for j in jobs if not Path(j[1]).exists()]; print(f"attempt {attempt}: {len(todo)} to score", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(workers) as ex:
                for n, _ in enumerate(ex.map(fn, todo, chunksize=1), 1):
                    if n % 100 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)
        except BrokenProcessPool:
            print("worker died; restarting pool", flush=True)
    left = [j for j in jobs if not Path(j[1]).exists()]
    if left:
        raise SystemExit(f"{len(left)} items not scored")


def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    z = np.load(f); truth = z["truth"].astype(bool); valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"cls": str(z["cls"]), "part": str(z["part"]), "fold": str(z["fold"])}
    res.update({k: z[k] for k in z.files if k.startswith("gmean_")})
    for i, nm in enumerate(str(x) for x in z["names"]):
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    jobs = []
    for s in SETS:
        sc = WORK / f"{s}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs += [(str(f), str(sc / f.name)) for f in sorted((WORK / f"{s}{a.tag}").glob("*.npz")) if not f.stem.endswith(".tmp")]
    _run_pool(jobs[a.shard::a.num_shards], score_one, a.workers)


def load_scores(ds, kind, tag):
    fs = sorted(f for f in (WORK / f"{ds}{tag}_{kind}").glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    S = {"id": np.array([f.stem for f in fs]), "grp": np.array([f.stem for f in fs]),
         "cls": np.array([str(z["cls"]) for z in Z]), "part": np.array([str(z["part"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k or k.startswith("gmean_"):
            S[k] = np.stack([z[k] for z in Z])
    return S


def select(a):
    """Per regime and map: dev threshold (V: VOC-Sim dev; X: mean of both dev curves). Per regime x W x gate x map: gamma."""
    S = {ds: load_scores(ds, "scores", a.tag) for ds in SETS}
    dev = {ds: S[ds]["part"] == "dev" for ds in SETS}
    curve = lambda ds, nm: cc.miou(S[ds][f"{nm}|raw"], S[ds]["cls"], dev[ds])
    thr, dm, gam = {r: {} for r in REGIMES}, {r: {} for r in REGIMES}, {r: {} for r in REGIMES}
    for nm in MAPS:
        cv = {ds: curve(ds, nm) for ds in SETS}
        for r in REGIMES:
            c = cv["voc"] if r == "V" else (cv["voc"] + cv["coco"]) / 2; i = int(np.argmax(c))
            thr[r][nm] = float(THR[i]); dm[r][nm] = float(c[i])
    for r in REGIMES:
        for f in (x for x in FAMS if x[0] == r):
            for p in PROPS:
                gam[r][f"{f}|{p}"] = int(np.argmax([dm[r][f"{f}{k}|{p}"] for k in range(len(GRID))]))
    crf = {r: sorted({f"{s}|{p}" for s in ("u-", "w-") for p in ("V0", "V3")} |
                     {f"{f}{k}|{p}" for f in FAMS if f[0] == r for p in ("V0", "V3") for k in (gam[r][f"{f}|{p}"], K1)}) for r in REGIMES}
    (WORK / f"thr{a.tag}.json").write_text(json.dumps({"thr": thr, "dev_miou": dm, "gamma": gam, "crf": crf,
                                                       "dev_items": {ds: int(dev[ds].sum()) for ds in SETS}}, indent=1))
    print("gamma: " + ", ".join(f"{r}:{k} {GRID[v]}" for r in REGIMES for k, v in gam[r].items()), flush=True)


def crf_one(job):
    f, o, todo = Path(job[0]), Path(job[1]), job[2]
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool); valid = np.ones_like(truth); other = np.zeros_like(truth)
    idx = {str(x): i for i, x in enumerate(z["names"])}; res = {"cls": str(z["cls"]), "part": str(z["part"])}; memo = {}
    for key, nm, t in todo:      # key = "{regime}|{map}|crf", thresholds (dev, 0.55)
        x = st.up(st.mm(z["maps"][idx[nm]].astype(np.float32)), truth.shape); ms = []
        for tv in t:
            if (nm, tv) not in memo:
                memo[(nm, tv)] = apply_ovam_densecrf(img, x >= tv, st.CRF_SITE).astype(bool)
            ms.append(memo[(nm, tv)])
        res[key] = cc.counts(np.stack(ms), truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def crf(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text())
    todo = [(f"{r}|{nm}|crf", nm, (Tr["thr"][r][nm], float(THR[T55]))) for r in REGIMES for nm in Tr["crf"][r]]
    Sp = M.load_split(); jobs = []
    for s in SETS:
        sc = WORK / f"{s}{a.tag}_crf"; sc.mkdir(parents=True, exist_ok=True)
        jobs += [(str(f), str(sc / f.name), todo) for f in sorted((WORK / f"{s}{a.tag}").glob("*.npz"))
                 if not f.stem.endswith(".tmp") and Sp[f.stem]["part"] == "test"]
    _run_pool(jobs[a.shard::a.num_shards], crf_one, a.workers)


# ---------------------------------------------------------------- report
def report(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr, gam = Tr["thr"], Tr["gamma"]
    S = {}
    for ds in SETS:
        raw = load_scores(ds, "scores", a.tag); tst = raw["part"] == "test"; S[ds] = {k: (v[tst] if isinstance(v, np.ndarray) else v) for k, v in raw.items()}
        cr = load_scores(ds, "crf", a.tag)
        if list(cr["id"]) != list(S[ds]["id"]):
            raise ValueError(f"{ds}: CRF and raw test items differ")
        S[ds].update({k: v for k, v in cr.items() if k.endswith("|crf")})
    BS = {ds: R.Boot(S[ds], a.boot) for ds in SETS}; L = []; w = L.append
    ti = {r: {nm: int(np.flatnonzero(np.isclose(THR, thr[r][nm]))[0]) for nm in MAPS} for r in REGIMES}
    RULES = {"sel_crf": "+ CRF, dev thr (primary)", "055_crf": "+ CRF, 0.55", "sel_raw": "raw, dev thr", "055_raw": "raw, 0.55"}
    HN = {"V0": "fine-only (V0)", "V3": "capped hierarchy (V3)", "seed": "seed only"}

    def name(r, W, g, p, fixed=False):
        f = fam(r, W, g)
        return f"{f}|{p}" if g == "-" else f"{f}{K1 if fixed else gam[r][f'{f}|{p}']}|{p}"

    def P(ds, r, nm, rule):
        D = S[ds]
        if rule == "sel_raw":
            return D[f"{nm}|raw"][:, ti[r][nm]]
        if rule == "055_raw":
            return D[f"{nm}|raw"][:, T55]
        return D[f"{r}|{nm}|crf"][:, 0 if rule == "sel_crf" else 1]
    cache = {}

    def m(ds, r, nm, rule="sel_crf"):
        k = (ds, r, nm, rule)
        if k not in cache:
            cache[k] = BS[ds].stat(P(ds, r, nm, rule))
        return cache[k]
    ci = lambda d: (float(d[0]), float(np.percentile(d[1], 2.5)), float(np.percentile(d[1], 97.5)))
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    rd = lambda x: "helps" if x[1] > 0 else "hurts" if x[2] < 0 else "n.r."
    diff = lambda ds, r, x, y, rule="sel_crf": ci((m(ds, r, x, rule)[0] - m(ds, r, y, rule)[0], m(ds, r, x, rule)[1] - m(ds, r, y, rule)[1]))
    rl = lambda p: "sel_raw" if p == "seed" else "sel_crf"
    ntest = {ds: len(S[ds]["cls"]) for ds in SETS}

    w("# Run 38: SA weights × hierarchy × gate on VOC-Sim and COCO-Cap, everything fitted on dev (`PREREG.md`)\n")
    w(f"**Exploratory.** Test halves: VOC-Sim {ntest['voc']}, COCO-Cap {ntest['coco']} images. Fitted on dev "
      f"(VOC-Sim {Tr['dev_items']['voc']}, COCO-Cap {Tr['dev_items']['coco']}): thresholds per regime and map, gate strength γ per "
      f"regime × SA weights × gate × map (grid {list(GRID)}). RW weights from run 36 (dev, held out). Regime V: profiles and fits "
      "from VOC-Sim dev only (COCO-Cap is untuned transfer). Regime X: VOC-Sim + COCO-Cap dev. One generation-time capture for "
      f"every map. Class-macro mIoU; 95% image-bootstrap intervals ({a.boot} samples, paired). DenseCRF at the dev threshold "
      "chosen on raw maps (run 37 rule).\n")

    w("\n## 1. Primary contrasts (test, + DenseCRF, dev-selected threshold and γ, SA weights on)\n")
    w("\"Improves\" = COCO-Cap CI lower bound > 0 and VOC-Sim CI lower bound ≥ −0.005 (run 36 rule).\n")
    w("| Regime | # | Contrast | Map | VOC-Sim Δ [95% CI] | COCO-Cap Δ [95% CI] | reading |"); w("|---|---|---|---|---|---|---|")
    pri = {}
    for r in REGIMES:
        for k, (g, h) in enumerate((("L", "V3"), ("L", "V0"), ("R", "V3"), ("R", "V0"), ("S", "V3"), ("S", "V0")), 1):
            d = {ds: diff(ds, r, name(r, True, g, h), name(r, True, "M", h)) for ds in SETS}; pri[(r, g, h)] = d
            imp = d["coco"][1] > 0 and d["voc"][1] >= -0.005
            w(f"| {r} | P{k} | {GNAME[g]} − MK4 | {HN[h]} | {f3(d['voc'])} ({rd(d['voc'])}) | {f3(d['coco'])} ({rd(d['coco'])}) | "
              f"{'**improves**' if imp else 'does not improve'} |")
    w("")
    for r in REGIMES:
        for g in ("L", "R", "S"):
            imp = any(pri[(r, g, h)]["coco"][1] > 0 and pri[(r, g, h)]["voc"][1] >= -0.005 for h in ("V3", "V0"))
            hurt = any(pri[(r, g, h)][ds][2] < 0 for h in ("V3", "V0") for ds in SETS)
            w(f"- Regime {r}, {GNAME[g]}: {'**holds**' if imp and not hurt else 'does not hold'} "
              f"(improves on V3 or V0: {'yes' if imp else 'no'}; hurts somewhere: {'yes' if hurt else 'no'}).")

    w("\n## 2. Dev-selected γ\n")
    w("| Regime | SA weights | Map | MK4 | SN | MK4-LR | MK4-RW |"); w("|---|:-:|---|---:|---:|---:|---:|")
    for r in REGIMES:
        for W_ in (False, True):
            for p in PROPS:
                w(f"| {r} | {'✓' if W_ else ''} | {HN[p]} | " + " | ".join(f"{GRID[gam[r][fam(r, W_, g) + '|' + p]]}" for g in GATES) + " |")

    sec = 3
    for r in REGIMES:
        w(f"\n## {sec}. Regime {r} ({'VOC-Sim dev profiles; COCO-Cap is transfer' if r == 'V' else 'VOC-Sim + COCO-Cap dev profiles'})\n"); sec += 1
        w("mIoU, VOC-Sim / COCO-Cap test, primary rule (seed only: raw, dev threshold), dev-selected γ:\n")
        w("| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |"); w("|:-:|---|---:|---:|---:|---:|---:|")
        for W_ in (False, True):
            for p in ("V0", "V3", "seed"):
                w(f"| {'✓' if W_ else ''} | {HN[p]} | " + " | ".join(
                    f"{m('voc', r, name(r, W_, g, p), rl(p))[0]:.3f} / {m('coco', r, name(r, W_, g, p), rl(p))[0]:.3f}" for g in ("-",) + GATES) + " |")
        for ds in SETS:
            w(f"\nPaired differences, {SETN[ds]} test, dev-selected γ:\n")
            w("| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |"); w("|:-:|---|---|---|---|---|")
            for W_ in (False, True):
                for p in ("V0", "V3", "seed"):
                    x = lambda g: name(r, W_, g, p)
                    w(f"| {'✓' if W_ else ''} | {HN[p]} | {f3(diff(ds, r, x('M'), x('-'), rl(p)))} | " +
                      " | ".join(f3(diff(ds, r, x(g), x('M'), rl(p))) for g in ("S", "L", "R")) + " |")
        for ds in SETS:
            w(f"\nComponent effects with the gate fixed, {SETN[ds]} test (primary rule):\n")
            w("| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |")
            w("|---|---|---|---|---|")
            for g in ("-",) + GATES:
                n = lambda W_, h: name(r, W_, g, h)
                w(f"| {GNAME[g]} | {f3(diff(ds, r, n(True, 'V0'), n(False, 'V0')))} | {f3(diff(ds, r, n(True, 'V3'), n(False, 'V3')))} | "
                  f"{f3(diff(ds, r, n(False, 'V3'), n(False, 'V0')))} | {f3(diff(ds, r, n(True, 'V3'), n(True, 'V0')))} |")
        w("\nDescriptive: every gate at γ = 1 (untuned), SA weights on, primary rule, VOC-Sim / COCO-Cap:\n")
        w("| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4, VOC-Sim | MK4-LR − MK4, COCO-Cap |"); w("|---|---:|---:|---:|---:|---|---|")
        for p in ("V0", "V3"):
            w(f"| {HN[p]} | " + " | ".join(f"{m('voc', r, name(r, True, g, p, True))[0]:.3f} / {m('coco', r, name(r, True, g, p, True))[0]:.3f}"
                                           for g in GATES) + " | " +
              " | ".join(f3(diff(ds, r, name(r, True, 'L', p, True), name(r, True, 'M', p, True))) for ds in SETS) + " |")

    w(f"\n## {sec}. Other rules, SA weights on (dev-selected γ), VOC-Sim / COCO-Cap\n"); sec += 1
    w("| Regime | Map | Gate | " + " | ".join(RULES.values()) + " |"); w("|---|---|---|" + "---:|" * len(RULES))
    for r in REGIMES:
        for p in ("V0", "V3"):
            for g in ("-",) + GATES:
                nm = name(r, True, g, p)
                w(f"| {r} | {HN[p]} | {GNAME[g]} | " + " | ".join(f"{m('voc', r, nm, ru)[0]:.3f} / {m('coco', r, nm, ru)[0]:.3f}" for ru in RULES) + " |")

    w(f"\n## {sec}. Per class IoU (primary rule, SA weights on, dev-selected γ)\n"); sec += 1
    for r in REGIMES:
        for ds in SETS:
            w(f"\nRegime {r}, {SETN[ds]} test:\n")
            hdr = [(name(r, True, g, p), f"{GNAME[g]} {p}") for p in ("V3", "V0") for g in ("-",) + GATES]
            w("| Class | items | " + " | ".join(h for _, h in hdr) + " |"); w("|---|---:|" + "---:|" * len(hdr))
            for c in CLASSES:
                sel = S[ds]["cls"] == c
                if sel.any():
                    w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{cc.miou(P(ds, r, nm, 'sel_crf')[:, None], S[ds]['cls'], sel)[0]:.3f}"
                                                                  for nm, _ in hdr) + " |")

    w(f"\n## {sec}. RW module weights used (run 36, all dev of the regime; MK4 gate AUROC object vs far)\n"); sec += 1
    Wt = json.loads(S36.read_text())["weights"]; mods = list(Wt["X"]["MK4"]["all"])
    w("| Regime | " + " | ".join(str(k) for k in range(len(mods))) + " |"); w("|---|" + "---:|" * len(mods))
    for r in REGIMES:
        w(f"| {r} AUROC | " + " | ".join(f"{Wt[r]['MK4']['all_auc'][x]:.3f}" for x in mods) + " |")
        w(f"| {r} weight | " + " | ".join(f"{Wt[r]['MK4']['all'][x]:.2f}" for x in mods) + " |")
    w(f"\nModules: " + ", ".join(f"{k} = {x}" for k, x in enumerate(mods)))
    w("\nMean gate value at γ = 1 over modules and pixels (test, VOC-Sim / COCO-Cap): " + ", ".join(
        f"{r}/{GNAME[g]} {float(np.mean(S['voc'][f'gmean_{r}{g}'])):.3f} / {float(np.mean(S['coco'][f'gmean_{r}{g}'])):.3f}"
        for r in REGIMES for g in GATES) + ".")
    w("\nBRACE (W on, no gate, V3 + DenseCRF) is the 'none / capped hierarchy / ✓' cell. BRACE stays frozen; any change to the "
      "paper's method is a separate owner decision.\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("capture", "score", "select", "crf", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--set", choices=SETS, default="voc")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"capture": capture, "score": score, "select": select, "crf": crf, "report": report}[a.cmd](a)
