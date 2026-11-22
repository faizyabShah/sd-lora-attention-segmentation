#!/usr/bin/env python3
"""runs/35_gate_negatives/PREREG.md: better negative evidence in the absent-class gate (stage 1 + N4 diagnostics, exploratory).

fit      (CPU) profiles from the run 31 dev dumps: MK4 pools plus IB object means and PD persistent negatives; N4 cluster
         statistics                                                           -> <WORK>/fit{tag}.pt
stage1   (CPU) retention curves, matched operating range, pass rule; N4 module agreement and object size
                                                                              -> runs/35_gate_negatives/{stage1,diag}{tag}.{md,json}
Run 31 (scripts/jc_mneg.py) supplies the dumps, split, pools, clustering and calibration; run 34 (scripts/jc_gate34.py)
the stage-1 summaries.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_mneg as M  # noqa: E402  (run 31)
import jc_gate34 as G34  # noqa: E402  (run 34)

T, J = M.T, M.J
WORK = J.CACHE / "jc_gate35"
OUT = ROOT / "runs/35_gate_negatives"
CLASSES, REGIMES = M.CLASSES, M.REGIMES
GATES = ("NC", "MK4", "SN", "IB", "GM", "PD")
CANDS = ("SN", "IB", "GM", "PD")
GAMMAS, SETTINGS = G34.GAMMAS, G34.SETTINGS
SELF_FRAC, REL, NMIN, PD_FRAC = 0.30, 0.5, 5.0, 0.10


def neff(img):
    if len(img) == 0:
        return 0.0
    p = np.bincount(img).astype(float); p = p[p > 0] / p.sum()
    return float(1.0 / (p ** 2).sum())


def persistent(X, al, C, img, nmin):
    """Drop clusters supported by fewer than nmin effective images; re-assign members to the kept centroids."""
    if C.shape[0] == 1:
        return C, C.shape[0]
    al = al.cpu().numpy()
    keep = [k for k in range(C.shape[0]) if neff(img[al == k]) >= nmin]
    if len(keep) == C.shape[0]:
        return C, len(keep)
    if not keep:
        return X.mean(0, keepdim=True), 0
    a2 = (X @ F.normalize(C[keep], dim=1).T).argmax(1)
    return torch.stack([X[a2 == k].mean(0) for k in range(len(keep)) if bool((a2 == k).any())]), len(keep)


def _top(v):
    u, n = np.unique(v, return_counts=True); j = int(np.argmax(n))
    return str(u[j]), float(n[j] / n.sum())


# ---------------------------------------------------------------- fit
def fit(a):
    torch.set_num_threads(a.workers)
    fs = G34.dumps(a); Mt = M._meta(fs); Sp = M.load_split(); SUB = M.subsets(Mt, Sp); mods = Mt[0]["modules"]
    FIT = {r: {s: {"mo": {}, "mo_ib": {}, "C1": {}, "C4": {}, "C4d": {}} for s in SUB[r]} for r in REGIMES}
    STAT = {r: {} for r in REGIMES}
    print(f"fit: {len(fs)} dump images; " + str({r: {s: len(v) for s, v in SUB[r].items()} for r in REGIMES}), flush=True)
    for mi, m in enumerate(mods):
        t0 = time.perf_counter(); objv, objib, ib0, negv, negatt = {}, {}, {}, {}, {}
        for i, f in enumerate(fs):
            z = np.load(f); q = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float(), dim=1); t = torch.from_numpy(z[f"t_{mi:02d}"]).float()
            N = q.shape[0]; ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
            lab = Mt[i]["label"]; ci = CLASSES.index(lab)
            idx = t[ci].topk(ko).indices; objv[i] = q[idx]
            keep = t[ci][idx] >= REL * t[ci].max(); keep[0] = True
            objib[i] = q[idx[keep]].mean(0); ib0[i] = q[idx].mean(0)
            for cj, c in enumerate(CLASSES):
                if c != lab and not Mt[i]["mention"][cj]:
                    nidx = t[cj].topk(kn).indices; negv[(c, i)] = q[nidx]
                    tt = t[:, nidx].clone(); tt[cj] = -float("inf"); negatt[(c, i)] = tt.argmax(0).numpy()
        for r in REGIMES:
            for s, ids in SUB[r].items():
                P = FIT[r][s]
                for c in CLASSES:
                    oi = [i for i in ids if Mt[i]["label"] == c]
                    if oi:
                        P["mo"][(c, m)] = torch.cat([objv[i] for i in oi]).mean(0)
                        P["mo_ib"][(c, m)] = torch.stack([objib[i] for i in oi]).mean(0)
                        if a.tag:   # smoke: relative threshold 0 == MK4 object mean
                            d = float((torch.stack([ib0[i] for i in oi]).mean(0) - P["mo"][(c, m)]).abs().max()); assert d < 1e-5, d
                    nk = [(c, i) for i in ids if (c, i) in negv]
                    if not nk:
                        continue
                    X = torch.cat([negv[k] for k in nk]); img = np.concatenate([[k[1]] * negv[k].shape[0] for k in nk])
                    P["C1"][(c, m)] = X.mean(0, keepdim=True)
                    C, al, _ = M.cluster(X, 4); P["C4"][(c, m)] = C
                    P["C4d"][(c, m)], kept = persistent(X, al, C, img, NMIN)
                    if a.tag:   # smoke: n_min = 0 == MK4 centroids
                        C0, _ = persistent(X, al, C, img, 0.0); assert float((C0 - C).abs().max()) < 1e-6
                    if s != "all":
                        continue
                    labs = np.array([Mt[i]["label"] for i in img]); att = np.array([CLASSES[x] for x in np.concatenate([negatt[k] for k in nk])])
                    alv = al.cpu().numpy(); cl = []
                    for k in range(C.shape[0]):
                        sel = alv == k
                        if not sel.any():
                            continue
                        lt, ls = _top(labs[sel]); at, as_ = _top(att[sel]); cnt = np.bincount(img[sel])
                        cl.append({"n": int(sel.sum()), "neff": neff(img[sel]), "maxshare": float(cnt.max() / cnt.sum()),
                                   "lab": lt, "lab_share": ls, "att": at, "att_share": as_})
                    STAT[r][(c, m)] = {"clusters": cl, "pool_lab_share": _top(labs)[1], "pool_att_share": _top(att)[1],
                                       "pool_n": int(X.shape[0]), "pd_kept": kept}
        print(f"fit module {mi} ({m}): {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save({"fit": FIT, "stat": STAT, "modules": mods}, WORK / f"fit{a.tag}.pt")


# ---------------------------------------------------------------- gates
def g_margin(q, mo, C):
    o = F.normalize(mo, dim=0); Cn = F.normalize(C, dim=1); sn = q @ Cn.T; k = sn.argmax(1)
    z = q @ o - sn.max(1).values; co = Cn @ o; zo = 1 - co.max(); zn = (co - 1)[k]
    return torch.sigmoid(T.GAIN * (z - (zo + zn) / 2) / (zo - zn).clamp(min=1e-6))


def gate(P, name, c, m, q, selfneg):
    mo, C4 = P["mo"].get((c, m)), P["C4"].get((c, m))
    if mo is None or C4 is None:
        return None
    if name == "NC":
        return M.g_single(q, mo, P["C1"][(c, m)][0])
    if name == "MK4":
        return M.g_multi(q, mo, C4)
    if name == "SN":
        return M.g_multi(q, mo, torch.cat([C4, selfneg[None]]))
    if name == "IB":
        return M.g_multi(q, P["mo_ib"][(c, m)], C4)
    if name == "GM":
        return g_margin(q, mo, C4)
    return M.g_multi(q, mo, P["C4d"][(c, m)])


# ---------------------------------------------------------------- stage 1
def stage1(a):
    torch.set_num_threads(a.workers)
    fs = G34.dumps(a); Mt = M._meta(fs); Sp = M.load_split(); mods = Mt[0]["modules"]
    FIT = torch.load(WORK / f"fit{a.tag}.pt")
    nG, nM = len(GAMMAS), len(mods); R, D = {}, {}; checked = False
    for i, f in enumerate(fs):
        t0 = time.perf_counter(); z = np.load(f); ds = Mt[i]["ds"]; c = Mt[i]["label"]; ci = CLASSES.index(c); iid = f.stem
        fg64 = torch.from_numpy(np.asarray(z["fg64"]).astype(bool)).reshape(-1); seed = torch.from_numpy(z["seed"]).float()
        sts = [s_ for s_ in SETTINGS if s_[1] == ds]; GA = {s_: [] for s_ in sts}
        for s_ in sts:
            R[(s_, iid)] = {}
        for mi, m in enumerate(mods):
            q = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float(), dim=1); t = torch.from_numpy(z[f"t_{mi:02d}"]).float()
            N = q.shape[0]; res = math.isqrt(N); fg, near, far = M._regions_at(z["fg64"], res); regs = (fg, near, far)
            mass = F.adaptive_avg_pool2d(seed[None, None], res)[0, 0].reshape(-1).clamp_min(0)
            sneg = q[t[ci].topk(max(1, round(SELF_FRAC * N)), largest=False).indices].mean(0)
            for s_ in sts:
                r, _ = s_; P = FIT["fit"][r][G34.held_out(Sp, iid, r, ds)]
                for g in GATES:
                    gv = gate(P, g, c, m, q, sneg)
                    if gv is None:
                        continue
                    assert torch.isfinite(gv).all() and float(gv.min()) >= 0 and float(gv.max()) <= 1, (g, iid)
                    if g == "MK4":
                        if a.tag and not checked:   # smoke: SN with a duplicated centroid == MK4
                            d = float((gate(P, "SN", c, m, q, P["C4"][(c, m)][0]) - gv).abs().max())
                            print(f"identity SN(duplicate centroid) vs MK4: {d:.2e}", flush=True); assert d < 1e-5; checked = True
                        u = 64 // res; GA[s_].append(gv.view(res, res).repeat_interleave(u, 0).repeat_interleave(u, 1).reshape(-1))
                    E = R[(s_, iid)].setdefault(("raw", g), {"pix": np.full((nM, 3, nG), np.nan), "mass": np.full((nM, 3, nG), np.nan),
                                                            "auc": np.full(nM, np.nan)})
                    Gg = torch.stack([gv ** gm for gm in GAMMAS], 1)
                    for ri, rg in enumerate(regs):
                        if rg.sum() == 0:
                            continue
                        E["pix"][mi, ri] = Gg[rg].mean(0).numpy(); wm = mass[rg]
                        if float(wm.sum()) > 0:
                            E["mass"][mi, ri] = ((Gg[rg] * wm[:, None]).sum(0) / wm.sum()).numpy()
                    E["auc"][mi] = M.auroc(gv[fg].numpy(), gv[far].numpy())
        for s_ in sts:
            if len(GA[s_]) != nM:
                continue
            Gm = torch.stack(GA[s_]); C = np.corrcoef(Gm.numpy()); off = C[~np.eye(nM, dtype=bool)]
            med, mn = Gm.median(0).values, Gm.min(0).values
            fgb = fg64; farb = M._regions_at(z["fg64"], 64)[2]
            D[(s_, iid)] = {"corr": float(np.nanmean(off)), "area": float(fgb.float().mean()),
                            "rescue_obj": float(((mn < 0.5) & (med >= 0.5))[fgb].float().mean()) if fgb.any() else np.nan,
                            "any_obj": float((mn < 0.5)[fgb].float().mean()) if fgb.any() else np.nan,
                            "med_far": float((med < 0.5)[farb].float().mean()) if farb.any() else np.nan,
                            "any_far": float((mn < 0.5)[farb].float().mean()) if farb.any() else np.nan}
        if i % 50 == 0:
            print(f"stage1 image {i}/{len(fs)} {iid}: {time.perf_counter() - t0:.1f}s", flush=True)
    torch.save({"R": R, "D": D}, WORK / f"stage1_rows{a.tag}.pt")
    pd_ok = diag_report(a, R, D, Mt, fs, FIT)
    stage1_report(a, R, Mt, fs, pd_ok)


def _cls_mean(vals, cls):
    v = np.array(vals, float); out = [np.nanmean(v[cls == c]) for c in np.unique(cls) if np.isfinite(v[cls == c]).any()]
    return float(np.mean(out)) if out else np.nan


def _ret1(R, s_, iid, g):
    E = R[(s_, iid)].get(("raw", g))
    return np.nan if E is None else float(np.nanmean(E["mass"][:, 0, GAMMAS.index(1.0)]))


def _spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    rx, ry = x.argsort().argsort(), y.argsort().argsort()
    return float(np.corrcoef(rx, ry)[0, 1])


def diag_report(a, R, D, Mt, fs, FIT):
    L = []; w = L.append; JD = {}; mods = FIT["modules"]; ST = FIT["stat"]
    w("# Run 35, N4 diagnostics (`PREREG.md` §3)\n")
    w("**Descriptive.** Cluster statistics from profiles fitted on all dev images of the regime; agreement and size from held-out "
      "MK4 gates on the run 31 dev dumps.\n")
    w("\n## 1. Support of the MK4 absent centroids\n")
    w("| Regime | centroids | median pixels | median n_eff | n_eff < 5 | median largest single-image share | PD kept (mean per class×module) |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    pd_ok = False
    for r in REGIMES:
        cl = [x for v in ST[r].values() for x in v["clusters"]]
        ne = np.array([x["neff"] for x in cl]); fr = float((ne < NMIN).mean()) if len(ne) else np.nan
        pd_ok = pd_ok or (fr >= PD_FRAC)
        w(f"| {r} | {len(cl)} | {np.median([x['n'] for x in cl]):.0f} | {np.median(ne):.1f} | {fr:.3f} | "
          f"{np.median([x['maxshare'] for x in cl]):.3f} | {np.mean([v['pd_kept'] for v in ST[r].values()]):.2f} |")
        JD[f"support:{r}"] = {"n": len(cl), "median_neff": float(np.median(ne)), "frac_lt5": fr}
    w("\nPer module (regime X): median n_eff / fraction n_eff < 5\n")
    w("| " + " | ".join(f"{k}" for k in range(len(mods))) + " |"); w("|" + "---:|" * len(mods))
    row = []
    for m in mods:
        ne = np.array([x["neff"] for (c, mm), v in ST["X"].items() if mm == m for x in v["clusters"]])
        row.append(f"{np.median(ne):.1f} / {(ne < NMIN).mean():.2f}" if len(ne) else "—")
    w("| " + " | ".join(row) + " |")
    w(f"\nModules: " + ", ".join(f"{k} = {m}" for k, m in enumerate(mods)))
    w(f"\n**PD reading rule (≥ {PD_FRAC:.0%} of centroids with n_eff < {NMIN:.0f} in regime V or X): "
      f"{'met, PD is read' if pd_ok else 'not met, PD is descriptive'}.**\n")
    JD["pd_read"] = bool(pd_ok)

    w("\n## 2. What the absent clusters contain\n")
    w("Share of the most common member-image label / most common other class by attention, per cluster vs its whole pool.\n")
    w("| Regime | cluster label share | pool label share | lift | cluster attention share | pool attention share | lift |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for r in REGIMES:
        rows = [(x, v) for v in ST[r].values() for x in v["clusters"]]
        cl_l = np.mean([x["lab_share"] for x, _ in rows]); po_l = np.mean([v["pool_lab_share"] for _, v in rows])
        cl_a = np.mean([x["att_share"] for x, _ in rows]); po_a = np.mean([v["pool_att_share"] for _, v in rows])
        w(f"| {r} | {cl_l:.3f} | {po_l:.3f} | {cl_l - po_l:+.3f} | {cl_a:.3f} | {po_a:.3f} | {cl_a - po_a:+.3f} |")
        JD[f"content:{r}"] = {"lab": [cl_l, po_l], "att": [cl_a, po_a]}
    m0 = mods[0]
    w(f"\nPer class, regime X, module {m0}: each cluster as `label (share) · attention (share) · n_eff`\n")
    w("| Class | clusters |"); w("|---|---|")
    for c in CLASSES:
        v = ST["X"].get((c, m0))
        if v:
            w(f"| {c} | " + "; ".join(f"{x['lab']} ({x['lab_share']:.2f}) · {x['att']} ({x['att_share']:.2f}) · {x['neff']:.0f}" for x in v["clusters"]) + " |")

    w("\n## 3. Module agreement (MK4, held out, 64 × 64 grid)\n")
    w("| Setting | mean pairwise corr of g | object px: some module < 0.5, median ≥ 0.5 | object px: some module < 0.5 | far px: median < 0.5 | far px: some module < 0.5 |")
    w("|---|---:|---:|---:|---:|---:|")
    for s_ in SETTINGS:
        ids = [f.stem for f, x in zip(fs, Mt) if x["ds"] == s_[1] and (s_, f.stem) in D]
        cls = np.array([x["label"] for f, x in zip(fs, Mt) if x["ds"] == s_[1] and (s_, f.stem) in D])
        vals = {k: _cls_mean([D[(s_, i)][k] for i in ids], cls) for k in ("corr", "rescue_obj", "any_obj", "med_far", "any_far")}
        w(f"| {s_[0]} on {s_[1]} | {vals['corr']:.3f} | {vals['rescue_obj']:.3f} | {vals['any_obj']:.3f} | {vals['med_far']:.3f} | {vals['any_far']:.3f} |")
        JD[f"agree:{s_[0]}:{s_[1]}"] = vals
    w("\nPer-module AUROC of MK4 (object vs far background), class-balanced:\n")
    w("| Setting | " + " | ".join(str(k) for k in range(len(mods))) + " |"); w("|---|" + "---:|" * len(mods))
    for s_ in SETTINGS:
        ids = [f.stem for f, x in zip(fs, Mt) if x["ds"] == s_[1]]; cls = np.array([x["label"] for x in Mt if x["ds"] == s_[1]])
        au = [_cls_mean([R[(s_, i)][("raw", "MK4")]["auc"][k] if ("raw", "MK4") in R[(s_, i)] else np.nan for i in ids], cls) for k in range(len(mods))]
        w(f"| {s_[0]} on {s_[1]} | " + " | ".join(f"{x:.3f}" for x in au) + " |"); JD[f"auc_module:{s_[0]}:{s_[1]}"] = au

    w("\n## 4. Object size (mass-weighted object retention at γ = 1, image means)\n")
    w("| Setting | gate | small third | middle third | large third | Spearman ρ (area, retention) |"); w("|---|---|---:|---:|---:|---:|")
    for s_ in SETTINGS[:2]:
        ids = [f.stem for f, x in zip(fs, Mt) if x["ds"] == s_[1] and (s_, f.stem) in D]
        area = np.array([D[(s_, i)]["area"] for i in ids]); cut = np.nanpercentile(area, [100 / 3, 200 / 3])
        terc = np.digitize(area, cut)
        for g in ("MK4", "IB", "NC"):
            ret = np.array([_ret1(R, s_, i, g) for i in ids])
            w(f"| {s_[0]} on {s_[1]} | {g} | " + " | ".join(f"{np.nanmean(ret[terc == k]):.3f}" for k in range(3)) + f" | {_spearman(area, ret):+.3f} |")
            JD[f"size:{s_[0]}:{s_[1]}:{g}"] = {"terciles": [float(np.nanmean(ret[terc == k])) for k in range(3)], "rho": _spearman(area, ret)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"diag{a.tag}.json").write_text(json.dumps(JD, indent=1, default=float))
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"diag{a.tag}.md").write_text(txt)
    return pd_ok


def stage1_report(a, R, Mt, fs, pd_ok):
    L = []; w = L.append; J_ = {}
    rng = np.random.default_rng(0)
    qq = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
    w("# Run 35, stage 1: better negative evidence in the absent-class gate (`PREREG.md`)\n")
    w("**Exploratory.** Run 31 dev dumps; every image is gated with profiles that never saw it. Retention = Σ w·g^γ / Σ w per region, "
      f"mean over modules, then images, then classes. 95% CIs from {a.boot} image-bootstrap samples. Mass weight = A3 seed at the module grid.\n")
    passed = {}
    for s_ in SETTINGS:
        allids = [f.stem for f, x in zip(fs, Mt) if x["ds"] == s_[1]]; allcls = np.array([x["label"] for x in Mt if x["ds"] == s_[1]])
        rule = s_ in SETTINGS[:2]
        w(f"\n## regime {s_[0]}, {'VOC-Sim' if s_[1] == 'voc' else 'COCO-Cap'} dev{' (pass rule)' if rule else ' (descriptive)'}\n")
        for excl in (False, True):
            sel = (allcls != "person") if excl else np.ones(len(allids), bool)
            ids = [x for x, k in zip(allids, sel) if k]; cls = allcls[sel]
            W = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(a.boot)]).astype(float)
            if excl:
                w("\nPerson excluded (mass-weighted):\n")
            w("| Gate | weight | object kept, γ=1 | near bg kept | far bg kept | AUROC obj vs far | ΔR vs MK4 [95% CI] | ΔS vs MK4 [95% CI] |")
            w("|---|---|---:|---:|---:|---:|---|---|")
            wts = ("mass",) if excl else ("mass", "pix")
            base = {wt: (G34._summ(R, s_, ("raw", "MK4"), wt, ids, cls), G34._summ(R, s_, ("raw", "MK4"), wt, ids, cls, W)) for wt in wts}
            for g in GATES:
                auc0 = G34._summ(R, s_, ("raw", g), "auc", ids, cls); aucb = G34._summ(R, s_, ("raw", g), "auc", ids, cls, W)
                for wt in wts:
                    p0 = G34._summ(R, s_, ("raw", g), wt, ids, cls); pb = G34._summ(R, s_, ("raw", g), wt, ids, cls, W)
                    dR0, dS0 = G34.matched(p0, base[wt][0]); dRb, dSb = G34.matched(pb, base[wt][1]); g1 = GAMMAS.index(1.0)
                    cR, cS = qq(dRb), qq(dSb)
                    w(f"| {g} | {wt} | {p0[0, g1]:.3f} | {p0[1, g1]:.3f} | {p0[2, g1]:.3f} | {auc0:.3f} [{qq(aucb)[0]:.3f}, {qq(aucb)[1]:.3f}] | "
                      f"{dR0:+.3f} [{cR[0]:+.3f}, {cR[1]:+.3f}] | {dS0:+.3f} [{cS[0]:+.3f}, {cS[1]:+.3f}] |")
                    J_[f"{s_[0]}:{s_[1]}:{g}:{wt}{':noperson' if excl else ''}"] = {"dR": dR0, "dR_ci": cR, "dS": dS0, "dS_ci": cS, "auc": float(auc0),
                                                                                    "auc_ci": qq(aucb), "curve": p0.tolist()}
                    if rule and not excl and wt == "mass" and g in CANDS:
                        passed.setdefault(g, []).append(bool(dR0 >= 0.01 and cR[0] > 0))
        if rule:
            w("\nPer class, mass-weighted object retention at γ = 1 (" + " / ".join(GATES) + "):\n")
            w("| Class | " + " | ".join(GATES) + " |"); w("|---|" + "---:|" * len(GATES))
            for c in np.unique(allcls):
                ii = [x for x, k in zip(allids, allcls) if k == c]
                w(f"| {c} | " + " | ".join(f"{np.nanmean([_ret1(R, s_, i, g) for i in ii]):.3f}" for g in GATES) + " |")
    verdict = {g: all(v) and len(v) == 2 for g, v in passed.items()}
    w(f"\n## Pass rule (mass-weighted, far background; ΔR ≥ +0.01 and CI > 0 on VOC-Sim dev / V and COCO-Cap dev / X)\n")
    for g in CANDS:
        note = ""
        if g == "PD" and not pd_ok:
            note = " — **descriptive only**: the N4 reading rule was not met (see `diag.md`)"
        w(f"- {g}: {'**passes**' if verdict.get(g) else 'does not pass'}{note}")
    verdict["PD_read"] = bool(pd_ok)
    J_["verdict"] = verdict
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stage1{a.tag}.json").write_text(json.dumps(J_, indent=1, default=float))
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"stage1{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("fit", "stage1"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"fit": fit, "stage1": stage1}[a.cmd](a)
