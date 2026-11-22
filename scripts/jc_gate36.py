#!/usr/bin/env python3
"""runs/36_sn_consensus/PREREG.md: self-negative gate end to end, plus module-consensus gates (exploratory).

stage1   (CPU) run 31 dev dumps: MK4, SN and consensus gates (AV / LR / RW on each), retention curves, pass rule,
         operating strength gamma*, RW module weights                     -> runs/36_sn_consensus/stage1{tag}.{md,json}
capture  (GPU) all images, generation-time, arms A + {X,V} x ARMS_G      -> <WORK>/{set}{tag}/
score    (CPU) DenseCRF on V3 at the full grid                            -> <WORK>/{set}{tag}_scores/
report   (CPU)                                                            -> runs/36_sn_consensus/result{tag}.md

Profiles: run 35's fit (<CACHE>/jc_gate35/fit.pt). RW weights: run 35's held-out stage-1 AUROC rows.
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
import jc_mneg as M  # noqa: E402  (run 31)
import jc_gate34 as G34  # noqa: E402  (run 34)
import jc_gate35 as G35  # noqa: E402  (run 35)

AB, T, J, cc, st = M.AB, M.T, M.J, M.cc, M.st
WORK = J.CACHE / "jc_gate36"
W35 = J.CACHE / "jc_gate35"
OUT = ROOT / "runs/36_sn_consensus"
SETS, CLASSES, THR, T55, REGIMES = M.SETS, M.CLASSES, M.THR, M.T55, M.REGIMES
MODEL = G34.MODEL
GAMMAS, SETTINGS = G34.GAMMAS, G34.SETTINGS
SELF_FRAC = G35.SELF_FRAC
BASES = ("MK4", "SN")
CONS = ("AV", "LR", "RW")
GATES = BASES + tuple(f"{b}-{c}" for b in BASES for c in CONS)
CANDS = GATES[2:]
ARMS_G = ("MK4", "SN", "SN1") + CANDS
ARMS = ("A",) + tuple(f"{r}:{g}" for r in REGIMES for g in ARMS_G)
MAPS = ("seed", "V0", "V3")
DEV = M.DEV
WSET = {"V": (("V", "voc"),), "X": (("X", "voc"), ("X", "coco"))}     # RW weights: settings pooled per regime


# ---------------------------------------------------------------- consensus
def consensus(gd, w):
    """gd {module: g [N]}; w {module: weight}. Weighted mean of bilinear-upsampled gates at 64, area-resized back."""
    num, den = None, 0.0
    for m, g in gd.items():
        wm = float(w.get(m, 0.0))
        if wm <= 0:
            continue
        r = math.isqrt(g.shape[0])
        u = F.interpolate(g.float().view(1, 1, r, r), size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        num = wm * u if num is None else num + wm * u; den += wm
    if num is None:
        return dict(gd)
    cb = (num / den)[None, None]
    return {m: F.adaptive_avg_pool2d(cb, math.isqrt(g.shape[0]))[0, 0].reshape(-1).clamp(0, 1) for m, g in gd.items()}


def cons_w(kind, mods_res, Wrw):
    if kind == "AV":
        return {m: 1.0 for m in mods_res}
    if kind == "LR":
        return {m: (1.0 if r < 64 else 0.0) for m, r in mods_res.items()}
    return Wrw


def rw_weights(mods, Mt, fs, Sp):
    """{regime: {base: {subset: {module: w}}}} from run 35's held-out per-module AUROC (class-balanced)."""
    R35 = torch.load(W35 / "stage1_rows.pt")["R"]; out = {}
    for r in REGIMES:
        out[r] = {}
        for b in BASES:
            out[r][b] = {}
            for sub in ("dA", "dB", "all"):
                vals, cls = [], []
                for f, x in zip(fs, Mt):
                    for s_ in WSET[r]:
                        if x["ds"] != s_[1] or (s_, f.stem) not in R35 or ("raw", b) not in R35[(s_, f.stem)]:
                            continue
                        if sub != "all" and Sp[f.stem]["fold"] != sub:
                            continue
                        vals.append(R35[(s_, f.stem)][("raw", b)]["auc"]); cls.append(x["label"])
                A = np.stack(vals); cls = np.array(cls)
                auc = np.nanmean(np.stack([np.nanmean(A[cls == c], 0) for c in np.unique(cls)]), 0)
                out[r][b][sub] = {m: float(np.clip(2 * (a_ - 0.5), 0, 1)) for m, a_ in zip(mods, auc)}
                out[r][b][sub + "_auc"] = {m: float(a_) for m, a_ in zip(mods, auc)}
    return out


def gate_set(P, lab, Q, sneg, Wr, sub, gst=None):
    """All gates for one image under one profile set. Q {module: q}; Wr = weights[r]. gst None: gamma 1 (stage 1)."""
    base = {b: {} for b in BASES}; miss = 0; mres = {m: math.isqrt(q.shape[0]) for m, q in Q.items()}
    for m, q in Q.items():
        g = G35.gate(P, "MK4", lab, m, q, None)
        if g is None:
            miss += 1; continue
        base["MK4"][m] = g; base["SN"][m] = G35.gate(P, "SN", lab, m, q, sneg[m])
    G = {"MK4": base["MK4"], "SN": base["SN"]}
    for b in BASES:
        for c in CONS:
            G[f"{b}-{c}"] = consensus(base[b], cons_w(c, {m: mres[m] for m in base[b]}, Wr[b][sub]))
    return G, miss


# ---------------------------------------------------------------- stage 1
def stage1(a):
    torch.set_num_threads(a.workers)
    fs = G34.dumps(a); Mt = M._meta(fs); Sp = M.load_split(); mods = Mt[0]["modules"]
    FIT = torch.load(W35 / "fit.pt")["fit"]; fa = M.dump_files(""); W = rw_weights(mods, M._meta(fa), fa, Sp)   # weights: all dev dumps
    nG, nM = len(GAMMAS), len(mods); R = {}; checked = False
    for i, f in enumerate(fs):
        t0 = time.perf_counter(); z = np.load(f); ds = Mt[i]["ds"]; c = Mt[i]["label"]; ci = CLASSES.index(c); iid = f.stem
        seed = torch.from_numpy(z["seed"]).float(); Q, SNg, REG, MASS = {}, {}, {}, {}
        for mi, m in enumerate(mods):
            q = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float(), dim=1); t = torch.from_numpy(z[f"t_{mi:02d}"]).float()
            N = q.shape[0]; res = math.isqrt(N); Q[m] = q
            SNg[m] = q[t[ci].topk(max(1, round(SELF_FRAC * N)), largest=False).indices].mean(0)
            REG[m] = M._regions_at(z["fg64"], res); MASS[m] = F.adaptive_avg_pool2d(seed[None, None], res)[0, 0].reshape(-1).clamp_min(0)
        for s_ in [x for x in SETTINGS if x[1] == ds]:
            r, _ = s_; sub = G34.held_out(Sp, iid, r, ds); P = FIT[r][sub]
            Gs, _ = gate_set(P, c, Q, SNg, W[r], sub); R[(s_, iid)] = {}
            if a.tag and not checked:   # smoke: AV == RW with all weights 1; MK4 == run 31 g_multi
                av = consensus(Gs["MK4"], {m: 1.0 for m in Gs["MK4"]}); rw1 = consensus(Gs["MK4"], {m: 1.0 for m in mods})
                d1 = max(float((av[m] - rw1[m]).abs().max()) for m in av)
                d2 = max(float((Gs["MK4"][m] - M.g_multi(Q[m], P["mo"][(c, m)], P["C4"][(c, m)])).abs().max()) for m in Gs["MK4"])
                print(f"identity AV vs RW(w=1): {d1:.2e}; MK4 vs g_multi: {d2:.2e}", flush=True); assert d1 < 1e-6 and d2 < 1e-6; checked = True
            for g, gd in Gs.items():
                E = R[(s_, iid)].setdefault(("raw", g), {"pix": np.full((nM, 3, nG), np.nan), "mass": np.full((nM, 3, nG), np.nan),
                                                        "auc": np.full(nM, np.nan)})
                for mi, m in enumerate(mods):
                    if m not in gd:
                        continue
                    gv = gd[m]; assert torch.isfinite(gv).all() and float(gv.min()) >= 0 and float(gv.max()) <= 1, (g, iid)
                    regs = REG[m]; mass = MASS[m]; Gg = torch.stack([gv ** gm for gm in GAMMAS], 1)
                    for ri, rg in enumerate(regs):
                        if rg.sum() == 0:
                            continue
                        E["pix"][mi, ri] = Gg[rg].mean(0).numpy(); wm = mass[rg]
                        if float(wm.sum()) > 0:
                            E["mass"][mi, ri] = ((Gg[rg] * wm[:, None]).sum(0) / wm.sum()).numpy()
                    E["auc"][mi] = M.auroc(gv[regs[0]].numpy(), gv[regs[2]].numpy())
        if i % 50 == 0:
            print(f"stage1 image {i}/{len(fs)} {iid}: {time.perf_counter() - t0:.1f}s", flush=True)
    torch.save({"R": R}, WORK / f"stage1_rows{a.tag}.pt")
    stage1_report(a, R, Mt, fs, W, mods)


def gstar_of(curve, target):
    """gamma at which the far-retention curve [3, nG] equals target; log-linear, clipped to the grid."""
    far = np.asarray(curve)[2]; lg = np.log(np.array(GAMMAS))
    return float(np.exp(np.interp(-target, -far, lg)))


def stage1_report(a, R, Mt, fs, W, mods):
    L = []; w = L.append; J_ = {}; rng = np.random.default_rng(0)
    qq = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
    w("# Run 36, stage 1: module-consensus gates (`PREREG.md` §3)\n")
    w("**Exploratory.** Run 31 dev dumps; every image is gated with profiles (and RW weights) that never saw it. Retention = "
      f"Σ w·g^γ / Σ w per region, mean over modules, then images, then classes. 95% CIs from {a.boot} image-bootstrap samples. "
      "Candidates are compared with their own base gate (MK4-x vs MK4, SN-x vs SN) at matched far-background suppression.\n")
    passed = {}; curves = {}
    for s_ in SETTINGS:
        allids = [f.stem for f, x in zip(fs, Mt) if x["ds"] == s_[1]]; allcls = np.array([x["label"] for x in Mt if x["ds"] == s_[1]])
        rule = s_ in SETTINGS[:2]
        w(f"\n## regime {s_[0]}, {'VOC-Sim' if s_[1] == 'voc' else 'COCO-Cap'} dev{' (pass rule)' if rule else ' (descriptive)'}\n")
        for excl in (False, True):
            sel = (allcls != "person") if excl else np.ones(len(allids), bool)
            ids = [x for x, k in zip(allids, sel) if k]; cls = allcls[sel]
            Wb = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(a.boot)]).astype(float)
            if excl:
                w("\nPerson excluded (mass-weighted):\n")
            w("| Gate | vs | weight | object kept, γ=1 | near bg kept | far bg kept | AUROC obj vs far | ΔR [95% CI] | ΔS [95% CI] |")
            w("|---|---|---|---:|---:|---:|---:|---|---|")
            wts = ("mass",) if excl else ("mass", "pix")
            S0, Sb = {}, {}
            for g in GATES:
                for wt in wts:
                    S0[(g, wt)] = G34._summ(R, s_, ("raw", g), wt, ids, cls); Sb[(g, wt)] = G34._summ(R, s_, ("raw", g), wt, ids, cls, Wb)
            pairs = [(g, "MK4") for g in ("MK4", "SN")] + [(g, g.split("-")[0]) for g in CANDS] + \
                    [(f"{b}-{c}", f"{b}-AV") for b in BASES for c in ("LR", "RW")]
            for g, ref in pairs:
                auc0 = G34._summ(R, s_, ("raw", g), "auc", ids, cls); aucb = G34._summ(R, s_, ("raw", g), "auc", ids, cls, Wb)
                for wt in wts:
                    p0 = S0[(g, wt)]; dR0, dS0 = G34.matched(p0, S0[(ref, wt)]); dRb, dSb = G34.matched(Sb[(g, wt)], Sb[(ref, wt)])
                    cR, cS = qq(dRb), qq(dSb); g1 = GAMMAS.index(1.0)
                    w(f"| {g} | {ref} | {wt} | {p0[0, g1]:.3f} | {p0[1, g1]:.3f} | {p0[2, g1]:.3f} | {auc0:.3f} [{qq(aucb)[0]:.3f}, {qq(aucb)[1]:.3f}] | "
                      f"{dR0:+.3f} [{cR[0]:+.3f}, {cR[1]:+.3f}] | {dS0:+.3f} [{cS[0]:+.3f}, {cS[1]:+.3f}] |")
                    key = f"{s_[0]}:{s_[1]}:{g}:vs:{ref}:{wt}{':noperson' if excl else ''}"
                    J_[key] = {"dR": dR0, "dR_ci": cR, "dS": dS0, "dS_ci": cS, "auc": float(auc0), "auc_ci": qq(aucb), "curve": p0.tolist()}
                    if not excl and wt == "mass":
                        curves[(s_, g)] = p0
                    if rule and not excl and wt == "mass" and g in CANDS and ref == g.split("-")[0]:
                        passed.setdefault(g, []).append(bool(dR0 >= 0.01 and cR[0] > 0))
    verdict = {g: all(v) and len(v) == 2 for g, v in passed.items()}
    w("\n## Pass rule (mass-weighted far background; ΔR ≥ +0.01 and CI > 0 vs own base on VOC-Sim dev / V and COCO-Cap dev / X)\n")
    for g in CANDS:
        w(f"- {g} vs {g.split('-')[0]}: {'**passes**' if verdict.get(g) else 'does not pass'}")
    # operating strength
    GS = {}
    for r in REGIMES:
        sts = [x for x in WSET[r]]
        cv = lambda g: np.mean([curves[(s_, g)] for s_ in sts], 0)
        tgt = float(cv("MK4")[2, GAMMAS.index(1.0)]); GS[r] = {"MK4": 1.0, "SN1": 1.0, "_target_far": tgt}
        for g in ("SN",) + CANDS:
            GS[r][g] = gstar_of(cv(g), tgt)
    w("\n## Operating strength γ* (far retention matched to MK4 at γ = 1; regime V: VOC-Sim / V; X: mean of VOC-Sim / X and COCO-Cap / X)\n")
    w("| Regime | MK4 far kept at γ=1 | " + " | ".join(("SN",) + CANDS) + " |"); w("|---|---:|" + "---:|" * (1 + len(CANDS)))
    for r in REGIMES:
        w(f"| {r} | {GS[r]['_target_far']:.3f} | " + " | ".join(f"{GS[r][g]:.3f}" for g in ("SN",) + CANDS) + " |")
    w("\n## RW module weights, all dev (w = clip(2·(AUC − 0.5), 0, 1); AUC from run 35 held-out rows)\n")
    w("| Regime | base | " + " | ".join(str(k) for k in range(len(mods))) + " |"); w("|---|---|" + "---:|" * len(mods))
    for r in REGIMES:
        for b in BASES:
            w(f"| {r} | {b} | " + " | ".join(f"{W[r][b]['all'][m]:.2f}" for m in mods) + " |")
    J_["verdict"] = verdict; J_["gstar"] = GS; J_["weights"] = W
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stage1{a.tag}.json").write_text(json.dumps(J_, indent=1, default=float))
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"stage1{a.tag}.md").write_text(txt)


# ---------------------------------------------------------------- stage 2
@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(MODEL); K = T.class_keys(rt, CLASSES)
    FIT = torch.load(W35 / "fit.pt")["fit"]; S1 = json.loads((OUT / f"stage1{a.tag}.json").read_text())
    GS, W = S1["gstar"], S1["weights"]; Sp = M.load_split(); cache = {}
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True); checked = False
    for row in M.rows_for(a, "all"):
        iid = row["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); sp = Sp[iid]; rec, sample = J.generate(rt, row, a.set); lab = AB.label(row)
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); Q = T.qbar(rec); tgt = (None, list(rec.target)); G = {}; miss = 0
        _, Tm = T.ca_seeds(rec, {lab: K[lab]}, tbar=True)
        sneg = {m: q[Tm[(lab, m)].float().topk(max(1, round(SELF_FRAC * q.shape[0])), largest=False).indices].mean(0) for m, q in Q.items()}
        for r in REGIMES:
            sub = "all" if (sp["part"] == "test" or (r == "V" and sp["ds"] == "coco")) else ("dB" if sp["fold"] == "dA" else "dA")
            if (r, sub) not in cache:
                cache[(r, sub)] = G34.to_dev(FIT[r][sub])
            P = cache[(r, sub)]
            Gs, mi_ = gate_set(P, lab, Q, sneg, W[r], sub); miss += mi_
            for arm in ARMS_G:
                src = "SN" if arm == "SN1" else arm; gm = float(GS[r][arm])
                gd = {}
                for m, q in Q.items():
                    g = Gs[src].get(m)
                    gd[m] = torch.ones(q.shape[0], device=q.device) if g is None else g.clamp(0, 1) ** gm
                    if not (torch.isfinite(gd[m]).all() and float(gd[m].min()) >= 0 and float(gd[m].max()) <= 1):
                        raise ValueError(f"bad gate {arm} {m} {iid}")
                G[f"{r}:{arm}"] = gd
            if a.tag and not checked:   # smoke: SN with a duplicated centroid == MK4
                d = max(float((G35.gate(P, "SN", lab, m, Q[m], P["C4"][(lab, m)][0]) - Gs["MK4"][m]).abs().max()) for m in Gs["MK4"])
                print(f"identity SN(duplicate centroid) vs MK4: {d:.2e}", flush=True); assert d < 1e-5; checked = True
        if a.tag:   # smoke, descriptive: capture-time SN vs dump-based SN (regime X profiles)
            df = M.WORK / f"dump_{a.set}" / f"{iid}.npz"
            if df.exists():
                z = np.load(df); mods = list(z["modules"]); ci = CLASSES.index(lab); dd = []
                r = "X"; sub = "all" if sp["part"] == "test" else ("dB" if sp["fold"] == "dA" else "dA"); P = cache[(r, sub)]
                for mi, m in enumerate(mods):
                    q = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float(), dim=1).to(DEV); t = torch.from_numpy(z[f"t_{mi:02d}"]).float()
                    sn = q[t[ci].topk(max(1, round(SELF_FRAC * q.shape[0])), largest=False).indices.to(DEV)].mean(0)
                    g0 = G35.gate(P, "SN", lab, m, q, sn)
                    g1 = G35.gate(P, "SN", lab, m, Q[m], sneg[m])
                    if g0 is not None:
                        dd.append(float((g0 - g1).abs().mean()))
                print(f"smoke: {iid} capture-time vs dump SN gate, mean |Δ| per module max {max(dd):.3e}", flush=True)
        S = T.ca_seeds(rec, {arm: tgt for arm in ARMS}, gate=G)
        fid = float((S["A"] - seed).abs().max())
        if fid > 1e-2:
            raise ValueError(f"seed fidelity {fid:.2e} for {iid}")
        S["A"] = seed; maps = []
        for arm in ARMS:
            v = sg_combined_v2.aggregate(sa, S[arm])[0]
            maps.append(np.stack([x.float().reshape(64, 64).cpu().numpy() for x in (S[arm], v["V0"], v["V3"])]))
        maps = np.stack(maps).astype(np.float32)
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=lab, truth=np.asarray(sample.target_mask, bool),
                 image=np.asarray(sample.generated_image, np.uint8), fid_seed=fid, part=sp["part"], fold=str(sp["fold"]), gate_miss=miss)
        os.replace(tmp, o)
        print(f"[g36:{a.set}] {iid} {sp['part']} fid {fid:.1e} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"cls": str(z["cls"]), "part": str(z["part"]), "fold": str(z["fold"])}
    for ai, arm in enumerate(ARMS):
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(z["maps"][ai, mi].astype(np.float32)), truth.shape)
            res[f"{arm}|{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            if m == "V3":
                ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
                res[f"{arm}|V3+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    jobs = []
    for s in SETS:
        sc = WORK / f"{s}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs += [(str(f), str(sc / f.name)) for f in sorted((WORK / f"{s}{a.tag}").glob("*.npz")) if not f.stem.endswith(".tmp")]
    jobs = jobs[a.shard::a.num_shards]
    for attempt in range(5):
        todo = [j for j in jobs if not Path(j[1]).exists()]; print(f"attempt {attempt}: {len(todo)} to score", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers) as ex:
                for n, _ in enumerate(ex.map(score_one, todo, chunksize=1), 1):
                    if n % 50 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)
        except BrokenProcessPool:
            print("worker died; restarting pool", flush=True)
    if any(not Path(j[1]).exists() for j in jobs):
        raise SystemExit("items not scored")


def _load_scores(ds, tag):
    fs = sorted(f for f in (WORK / f"{ds}{tag}_scores").glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    S = {"cls": np.array([str(z["cls"]) for z in Z]), "grp": np.array([f.stem for f in fs]), "part": np.array([str(z["part"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k:
            S[k] = np.stack([z[k] for z in Z])
    return S


def report(a):
    S = {ds: _load_scores(ds, a.tag) for ds in SETS}
    S1 = json.loads((OUT / f"stage1{a.tag}.json").read_text()); V1 = S1["verdict"]; GS = S1["gstar"]
    sub = lambda s, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in s.items()}
    dev = {ds: sub(S[ds], S[ds]["part"] == "dev") for ds in SETS}; tst = {ds: sub(S[ds], S[ds]["part"] == "test") for ds in SETS}
    mi = lambda D, P: cc.miou(P[:, None], D["cls"], np.ones(len(D["cls"]), bool))[0] if len(D["cls"]) else np.nan
    curve = lambda D, k: cc.miou(D[k], D["cls"], np.ones(len(D["cls"]), bool)) if len(D["cls"]) else np.zeros(len(THR))

    def thr(reg, k):
        cv = curve(dev["voc"], k)
        return int(np.argmax(cv if reg == "V" else (cv + curve(dev["coco"], k)) / 2))

    def pred(reg, ds, k, rule="dev"):
        D = tst[ds]
        return D[k][:, T55] if rule == "55" else D[k][:, thr(reg, k)]

    def dci(ds, Pa, Pb):
        if len(tst[ds]["cls"]) < 2:
            return "n/a", np.nan, np.nan, np.nan
        lo, hi = cc.boot(tst[ds], Pa, Pb, n=a.boot); d = mi(tst[ds], Pa) - mi(tst[ds], Pb)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", d, lo, hi
    K = lambda arm, m="V3+R2": f"{arm}|{m}|raw"
    L = ["# Run 36, stage 2: self-negative gate and module-consensus gates, full pipeline (`PREREG.md`)\n",
         "Test halves; thresholds chosen on dev. **Exploratory.** One generation-time capture for every arm. Regime V = VOC-Sim-only "
         "profiles (COCO-Cap is transfer); regime X = VOC-Sim + COCO-Cap dev profiles (label-free domain adaptation). Every gate except "
         "MK4 and SN1 runs at its dev-matched strength γ* (`stage1.md`).\n",
         f"Test images: VOC-Sim {len(tst['voc']['cls'])}, COCO-Cap {len(tst['coco']['cls'])}. Stage 1 verdicts (vs own base): "
         + ", ".join(f"{g} {'pass' if V1.get(g) else 'fail'}" for g in CANDS) + ". SN passed run 35 stage 1.\n"]
    imp = {}
    for reg in REGIMES:
        L.append(f"\n## Regime {reg}\n")
        L.append("| Arm | γ | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |"); L.append("|---|---:|---:|---:|---:|---:|")
        for arm in ("A",) + tuple(f"{reg}:{g}" for g in ARMS_G):
            gm = "—" if arm == "A" else f"{GS[reg][arm.split(':')[1]]:.2f}"
            L.append(f"| {arm} | {gm} | {mi(tst['voc'], pred(reg, 'voc', K(arm))):.3f} | {mi(tst['coco'], pred(reg, 'coco', K(arm))):.3f} | "
                     f"{mi(tst['voc'], pred(reg, 'voc', K(arm), '55')):.3f} | {mi(tst['coco'], pred(reg, 'coco', K(arm), '55')):.3f} |")
        L.append("\nHierarchy + DenseCRF. Paired image bootstrap; \"improves\" = COCO-Cap CI lower bound > 0 and VOC-Sim CI lower bound ≥ −0.005.\n")
        L.append("| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |")
        L.append("|---|---|---|---|---|---|---|")
        rows = [("SN", "MK4", True, "pass (run 35)")] + [(g, g.split("-")[0], V1.get(g, False), "pass" if V1.get(g) else "fail") for g in CANDS]
        for g, ref, ok, s1 in rows:
            x, y = f"{reg}:{g}", f"{reg}:{ref}"; c = [dci(ds, pred(reg, ds, K(x)), pred(reg, ds, K(y))) for ds in SETS]
            c5 = [dci(ds, pred(reg, ds, K(x), "55"), pred(reg, ds, K(y), "55")) for ds in SETS]
            im = c[1][2] > 0 and c[0][2] >= -0.005; imp[(reg, g)] = bool(ok and im)
            rd = (f"**improves on {ref}**" if im else f"does not improve on {ref}") if ok else "descriptive (failed stage 1)"
            L.append(f"| {g} − {ref} | {s1} | {c[0][0]} | {c[1][0]} | {c5[0][0]} | {c5[1][0]} | {rd} |")
        desc = [("SN1", "MK4"), ("SN", "A"), ("MK4", "A")] + [(f"{b}-{c}", f"{b}-AV") for b in BASES for c in ("LR", "RW")]
        for x0, y0 in desc:
            x = f"{reg}:{x0}"; y = "A" if y0 == "A" else f"{reg}:{y0}"
            c = [dci(ds, pred(reg, ds, K(x)), pred(reg, ds, K(y))) for ds in SETS]
            c5 = [dci(ds, pred(reg, ds, K(x), "55"), pred(reg, ds, K(y), "55")) for ds in SETS]
            L.append(f"| {x0} − {y0} | — | {c[0][0]} | {c[1][0]} | {c5[0][0]} | {c5[1][0]} | descriptive |")
        L.append("\nSN − MK4 before the hierarchy (descriptive; dev-selected thresholds on each map):\n")
        L.append("| Map | VOC-Sim test | COCO-Cap test |"); L.append("|---|---|---|")
        for mp in ("seed", "V0", "V3", "V3+R2"):
            c = [dci(ds, pred(reg, ds, K(f"{reg}:SN", mp)), pred(reg, ds, K(f"{reg}:MK4", mp))) for ds in SETS]
            L.append(f"| {mp} | {c[0][0]} | {c[1][0]} |")
        cl = tst["coco"]["cls"]; arms = ["A", f"{reg}:MK4", f"{reg}:SN", f"{reg}:SN-LR", f"{reg}:SN-RW", f"{reg}:MK4-LR", f"{reg}:MK4-RW"]
        L.append("\nPer class, COCO-Cap test, dev-selected (" + " / ".join(x.split(':')[-1] for x in arms) + "): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(pred(reg, 'coco', K(x))[:, None], cl, cl == c)[0]:.2f}" for x in arms) for c in np.unique(cl)) + "\n")
    L.append("\n## Combined reading (`PREREG.md` §4)\n")
    for reg in REGIMES:
        sn = imp[(reg, "SN")]; hx = [g for g in CANDS if g.startswith("SN-") and imp[(reg, g)]]
        mx = [g for g in CANDS if g.startswith("MK4-") and imp[(reg, g)]]
        L.append(f"- Regime {reg}: SN − MK4 {'improves' if sn else 'does not improve'}; SN consensus improving on SN: {', '.join(hx) or 'none'}; "
                 f"MK4 consensus improving on MK4: {', '.join(mx) or 'none'}.")
    L.append("\nBRACE stays frozen; any change to the paper's method is a separate owner decision.\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("stage1", "capture", "score", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--set", choices=SETS, default="voc")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"stage1": stage1, "capture": capture, "score": score, "report": report}[a.cmd](a)
