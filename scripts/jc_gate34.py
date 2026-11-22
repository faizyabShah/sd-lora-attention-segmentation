#!/usr/bin/env python3
"""runs/34_gate_discrim/PREREG.md: object-vs-leakage discrimination of the absent-class gate (exploratory).

fit      (CPU) profiles from the run 31 dev dumps, per representation (raw, pt = minus class-word key, pP = minus
         prompt-key span), regimes X / V, subsets all / dA / dB                    -> <WORK>/fit_{rep}{tag}.pt
stage1   (CPU) pixel- and mass-weighted retention curves, matched operating range, Q-K diagnostic
                                                                                  -> runs/34_gate_discrim/stage1{tag}.{md,json}
capture  (GPU) all images, generation-time, arms A + {X,V} x GATES                -> <WORK>/{set}{tag}/
score    (CPU) DenseCRF on V3 at the full grid                                     report -> runs/34_gate_discrim/result{tag}.md

Run 31 (scripts/jc_mneg.py) supplies the split, dumps, pools, clustering, calibration, capture and scoring pattern.
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

AB, T, J, cc, st, nr, QG = M.AB, M.T, M.J, M.cc, M.st, M.nr, M.QG
WORK = J.CACHE / "jc_gate34"
OUT = ROOT / "runs/34_gate_discrim"
SETS, CLASSES, THR, T55, REGIMES = M.SETS, M.CLASSES, M.THR, M.T55, M.REGIMES
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
GATES = ("NC", "MK4", "MP2", "MP4", "FD4", "FP2", "FP4")
REPS = ("raw", "pt", "pP")
GAMMAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
SHRINK = 0.5
ARMS = ("A",) + tuple(f"{r}:{g}" for r in REGIMES for g in GATES)
MAPS = ("seed", "V0", "V3")
DEV = M.DEV
EOS = 49407


# ---------------------------------------------------------------- prompt keys (for the Q-K residual reps)
def row_index():
    return {r["image_id"]: r for ds in SETS for r in nr.rows(ds)}


@torch.no_grad()
def prompt_keys(fs, mods):
    """{image id: {module: (target key [h, d], span basis list per head [h] of [d, r])}} plus rank fractions."""
    from diffusers import StableDiffusionPipeline
    from src.segmentation.attention_aggregation import find_token_subsequence
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32, safety_checker=None,
                                                   requires_safety_checker=False)
    tok, R = pipe.tokenizer, row_index(); out, ranks = {}, {}
    for f in fs:
        r = R[f.stem]; p = r["attribution_prompt"]
        ids = tok(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        emb = pipe.text_encoder(ids)[0]; il = ids[0].tolist(); n = il.index(EOS) + 1
        tgt = list(find_token_subsequence(il, tok(r["classname"], add_special_tokens=False)["input_ids"]))
        d = {}
        for m in mods:
            mod = pipe.unet.get_submodule(m); k = mod.to_k(emb); hd = mod.heads; dh = k.shape[-1] // hd
            k = k.view(77, hd, dh).transpose(0, 1).float()                           # [h, 77, d]
            kt = F.normalize(k[:, tgt].mean(1), dim=-1)                               # [h, d]
            Qs = []
            for hh in range(hd):
                Qm, Rm = torch.linalg.qr(k[hh, :n].T)                                  # [d, n]
                keep = Rm.diagonal().abs() > 1e-6 * Rm.diagonal().abs().max()
                Qs.append(Qm[:, keep]); ranks.setdefault(m, []).append(int(keep.sum()) / dh)
            d[m] = (kt, Qs)
        out[f.stem] = d
    return out, {m: float(np.mean(v)) for m, v in ranks.items()}


def residual(q, rep, kd):
    """q: [N, h*d] unit rows; remove per-head projection; re-normalise."""
    if rep == "raw":
        return q
    kt, Qs = kd; hd = kt.shape[0]; x = q.view(q.shape[0], hd, -1).clone()
    if rep == "pt":
        x = x - (x * kt[None]).sum(-1, keepdim=True) * kt[None]
    else:
        for hh in range(hd):
            Qm = Qs[hh]; x[:, hh] = x[:, hh] - (x[:, hh] @ Qm) @ Qm.T
    return F.normalize(x.reshape(q.shape[0], -1), dim=1)


def dumps(a):
    """Run 31 dev dumps (always the full-run dumps); smoke: an even subsample of --max-images per benchmark."""
    fs = M.dump_files("")
    if a.max_images:
        by = {ds: [f for f in fs if f.parent.name == f"dump_{ds}"] for ds in SETS}
        fs = [x for ds in SETS for x in by[ds][:: max(1, len(by[ds]) // a.max_images)][: a.max_images]]
    return fs


# ---------------------------------------------------------------- fit
def var_by(X, a, k):
    Y = X[a == k]
    return Y.var(0, unbiased=False) if Y.shape[0] > 1 else torch.zeros(X.shape[1])


def fit(a):
    torch.set_num_threads(a.workers)
    fs = dumps(a); Mt = M._meta(fs); Sp = M.load_split(); SUB = M.subsets(Mt, Sp); mods = Mt[0]["modules"]
    keys, ranks = (prompt_keys(fs, mods) if a.rep != "raw" else (None, {}))
    FIT = {r: {s: {"mo": {}, "mv": {}, "O": {2: {}, 4: {}}, "Ov": {2: {}, 4: {}}, "C1": {}, "C4": {}, "Cv4": {}} for s in SUB[r]}
           for r in REGIMES}
    print(f"fit {a.rep}: {len(fs)} dump images; " + str({r: {s: len(v) for s, v in SUB[r].items()} for r in REGIMES}), flush=True)
    worst = 0.0
    for mi, m in enumerate(mods):
        t0 = time.perf_counter(); objv, negv = {}, {}
        for i, f in enumerate(fs):
            z = np.load(f); q = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float(), dim=1); t = torch.from_numpy(z[f"t_{mi:02d}"]).float()
            q = residual(q, a.rep, keys[f.stem][m] if keys else None)
            if a.tag and a.rep != "raw" and i == 0:                                  # smoke: orthogonality
                kt, Qs = keys[f.stem][m]; x = q.view(q.shape[0], kt.shape[0], -1)
                c = (x * kt[None]).sum(-1).abs().max() if a.rep == "pt" else max(float((x[:, hh] @ Qs[hh]).abs().max()) for hh in range(len(Qs)))
                worst = max(worst, float(c))
            N = q.shape[0]; ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
            lab = Mt[i]["label"]; ci = CLASSES.index(lab)
            objv[i] = q[t[ci].topk(ko).indices]
            for cj, c in enumerate(CLASSES):
                if c != lab and not Mt[i]["mention"][cj]:
                    negv[(c, i)] = q[t[cj].topk(kn).indices]
        for r in REGIMES:
            for s, ids in SUB[r].items():
                P = FIT[r][s]
                for c in CLASSES:
                    oi = [i for i in ids if Mt[i]["label"] == c]
                    if oi:
                        X = torch.cat([objv[i] for i in oi]); P["mo"][(c, m)] = X.mean(0); P["mv"][(c, m)] = X.var(0, unbiased=False)
                        if a.rep == "raw":
                            for K in (2, 4):
                                C, al, _ = M.cluster(X, K); P["O"][K][(c, m)] = C
                                P["Ov"][K][(c, m)] = torch.stack([var_by(X, al, k) for k in range(C.shape[0])])
                    nk = [(c, i) for i in ids if (c, i) in negv]
                    if not nk:
                        continue
                    X = torch.cat([negv[k] for k in nk]); P["C1"][(c, m)] = X.mean(0, keepdim=True)
                    C, al, _ = M.cluster(X, 4); P["C4"][(c, m)] = C
                    P["Cv4"][(c, m)] = torch.stack([var_by(X, al, k) for k in range(C.shape[0])])
        print(f"fit {a.rep} module {mi} ({m}): {time.perf_counter() - t0:.0f}s", flush=True)
    if a.tag and a.rep != "raw":
        print(f"smoke: residual max |cos| to removed keys {worst:.2e}", flush=True); assert worst < 1e-4
    torch.save({"fit": FIT, "modules": mods, "ranks": ranks}, WORK / f"fit_{a.rep}{a.tag}.pt")


# ---------------------------------------------------------------- gates
def g_pair(q, O, Ov, C, Cv, fisher):
    """Nearest object centroid vs nearest absent centroid; run 31 calibration along w."""
    jo = (q @ F.normalize(O, dim=1).T).argmax(1); kn = (q @ F.normalize(C, dim=1).T).argmax(1)
    D = O[:, None] - C[None]
    if fisher:
        s = Ov[:, None] + Cv[None]; s = (1 - SHRINK) * s + SHRINK * s.mean(-1, keepdim=True); D = D / s.clamp_min(1e-12)
    W = F.normalize(D, dim=-1); pp = (W * O[:, None]).sum(-1); pm = (W * C[None]).sum(-1)
    z = (q * W[jo, kn]).sum(1); a, b = pp[jo, kn], pm[jo, kn]
    return torch.sigmoid(T.GAIN * (z - (a + b) / 2) / (a - b).clamp(min=1e-6))


def gate(P, name, c, m, q):
    mo, C4 = P["mo"].get((c, m)), P["C4"].get((c, m))
    if mo is None or C4 is None:
        return None
    if name == "NC":
        return M.g_single(q, mo, P["C1"][(c, m)][0])
    if name == "MK4":
        return M.g_multi(q, mo, C4)
    if name == "FD4":
        return g_pair(q, mo[None], P["mv"][(c, m)][None], C4, P["Cv4"][(c, m)], True)
    K = int(name[2:]); O = P["O"][K].get((c, m))
    if O is None:
        return None
    return g_pair(q, O, P["Ov"][K][(c, m)], C4, P["Cv4"][(c, m)], name.startswith("FP"))


def to_dev(P):
    def mv(x):
        if isinstance(x, torch.Tensor):
            return x.to(DEV)
        return {k: mv(v) for k, v in x.items()}
    return mv(P)


# ---------------------------------------------------------------- stage 1
def held_out(Sp, iid, r, ds):
    sp = Sp[iid]
    if r == "V" and ds == "coco":
        return "all"
    return "dB" if sp["fold"] == "dA" else "dA"


SETTINGS = (("V", "voc"), ("X", "coco"), ("X", "voc"), ("V", "coco"))     # first two: pass rule


def stage1(a):
    torch.set_num_threads(a.workers)
    fs = dumps(a); Mt = M._meta(fs); Sp = M.load_split(); mods = Mt[0]["modules"]
    FITS = {rep: torch.load(WORK / f"fit_{rep}{a.tag}.pt") for rep in REPS}
    keys, _ = prompt_keys(fs, mods)
    VAR = [("raw", g) for g in GATES] + [(rep, g) for rep in ("pt", "pP") for g in ("NC", "MK4")]
    nG, nM = len(GAMMAS), len(mods)
    # per image: ret[var][weight][module, region, gamma]; auc[var][module]
    R = {(st_, iid): {} for st_ in SETTINGS for iid in [f.stem for f in fs]}
    checked = False
    for mi, m in enumerate(mods):
        t0 = time.perf_counter()
        for i, f in enumerate(fs):
            z = np.load(f); ds = Mt[i]["ds"]; c = Mt[i]["label"]; iid = f.stem
            q0 = F.normalize(torch.from_numpy(z[f"q_{mi:02d}"]).float(), dim=1); N = q0.shape[0]; res = math.isqrt(N)
            fg, near, far = M._regions_at(z["fg64"], res)
            mass = F.adaptive_avg_pool2d(torch.from_numpy(z["seed"])[None, None], res)[0, 0].reshape(-1).clamp_min(0)
            regs = (fg, near, far); qs = {rep: residual(q0, rep, keys[iid][m]) for rep in REPS}
            for st_ in SETTINGS:
                r, dsx = st_
                if dsx != ds:
                    continue
                sub = held_out(Sp, iid, r, ds)
                for rep, g in VAR:
                    P = FITS[rep]["fit"][r][sub]; gv = gate(P, g, c, m, qs[rep])
                    if gv is None:
                        continue
                    if not checked and rep == "raw" and g == "MK4":                     # MP with 1 centroid == MK4
                        d = float((g_pair(qs[rep], P["mo"][(c, m)][None], None, P["C4"][(c, m)], None, False) - gv).abs().max())
                        print(f"identity g_pair(1 object, no Fisher) vs MK4: {d:.2e}", flush=True); assert d < 1e-5; checked = True
                    E = R[(st_, iid)].setdefault((rep, g), {"pix": np.full((nM, 3, nG), np.nan), "mass": np.full((nM, 3, nG), np.nan),
                                                         "auc": np.full(nM, np.nan)})
                    G = torch.stack([gv ** gm for gm in GAMMAS], 1)                        # [N, nG]
                    for ri, rg in enumerate(regs):
                        if rg.sum() == 0:
                            continue
                        E["pix"][mi, ri] = G[rg].mean(0).numpy()
                        wm = mass[rg]
                        if float(wm.sum()) > 0:
                            E["mass"][mi, ri] = ((G[rg] * wm[:, None]).sum(0) / wm.sum()).numpy()
                    E["auc"][mi] = M.auroc(gv[fg].numpy(), gv[far].numpy())
        print(f"stage1 module {mi} ({m}): {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save({"R": R, "meta": [(f.stem, x["ds"], x["label"]) for f, x in zip(fs, Mt)]}, WORK / f"stage1_rows{a.tag}.pt")
    stage1_report(a, R, Mt, fs, FITS)


def _summ(R, st_, var, w, ids, cls, W=None):
    """Class-balanced retention [region, gamma] (or AUROC) per bootstrap weight row W (images)."""
    X = np.stack([np.nanmean(R[(st_, i)][var][w], 0) if var in R[(st_, i)] else np.full((3, len(GAMMAS)), np.nan) for i in ids]) \
        if w != "auc" else np.array([np.nanmean(R[(st_, i)][var]["auc"]) if var in R[(st_, i)] else np.nan for i in ids])
    if W is None:
        W = np.ones((1, len(ids)))
    out = []
    for c in np.unique(cls):
        sel = (cls == c)
        Xi = X[sel]; ok = ~np.isnan(Xi.reshape(len(Xi), -1)).any(1); Ws = W[:, sel] * ok
        Xi = np.nan_to_num(Xi)
        num = np.tensordot(Ws, Xi, axes=(1, 0)); den = Ws.sum(1).reshape((-1,) + (1,) * (Xi.ndim - 1))
        out.append(np.where(den > 0, num / np.maximum(den, 1e-12), np.nan))
    return np.nanmean(np.stack(out), 0)


def matched(Rc, Rb):
    """Rc, Rb: [..., 3, nG] retention (candidate, MK4). Returns (dR at matched far suppression, dS at matched object retention)."""
    gi = [GAMMAS.index(0.5), GAMMAS.index(2.0)]

    def one(c, b):
        Sb, Ob, Sc, Oc = 1 - b[2], b[0], 1 - c[2], c[0]
        pts = np.linspace(Sb[gi[0]], Sb[gi[1]], 5)
        oc = np.interp(pts, np.sort(Sc), Oc[np.argsort(Sc)], left=np.nan, right=np.nan)
        ob = np.interp(pts, np.sort(Sb), Ob[np.argsort(Sb)])
        rp = np.linspace(Ob[gi[1]], Ob[gi[0]], 5)
        sc = np.interp(rp, np.sort(Oc), Sc[np.argsort(Oc)], left=np.nan, right=np.nan)
        sb = np.interp(rp, np.sort(Ob), Sb[np.argsort(Ob)])
        return np.mean(oc - ob), np.mean(sc - sb)
    if Rc.ndim == 2:
        return one(Rc, Rb)
    v = np.array([one(x, y) for x, y in zip(Rc, Rb)]); return v[:, 0], v[:, 1]


def stage1_report(a, R, Mt, fs, FITS):
    L = []; w = L.append; J_ = {}
    rng = np.random.default_rng(0)
    q = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
    w("# Run 34, stage 1: object vs leakage discrimination of the gate (`PREREG.md`)\n")
    w("**Exploratory.** Run 31 dev dumps; every image is gated with profiles that never saw it (other dev fold; regime V "
      "on COCO-Cap uses all VOC dev). Retention = Σ w·g^γ / Σ w per region, mean over modules, then images, then classes. "
      f"95% CIs from {a.boot} image-bootstrap samples. Mass weight = A3 seed at the module grid.\n")
    passed = {}
    for st_ in SETTINGS:
        ids = [f.stem for f, x in zip(fs, Mt) if x["ds"] == st_[1]]; cls = np.array([x["label"] for x in Mt if x["ds"] == st_[1]])
        W = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)) for _ in range(a.boot)]).astype(float)
        name = f"regime {st_[0]}, {'VOC-Sim' if st_[1] == 'voc' else 'COCO-Cap'} dev"
        rule = st_ in SETTINGS[:2]
        w(f"\n## {name}{' (pass rule)' if rule else ' (descriptive)'}\n")
        w("| Gate | weight | object kept, γ=1 | near bg kept | far bg kept | AUROC obj vs far | ΔR vs MK4 [95% CI] | ΔS vs MK4 [95% CI] |")
        w("|---|---|---:|---:|---:|---:|---|---|")
        base = {wt: (_summ(R, st_, ("raw", "MK4"), wt, ids, cls), _summ(R, st_, ("raw", "MK4"), wt, ids, cls, W)) for wt in ("mass", "pix")}
        for rep, g in [("raw", x) for x in GATES] + [(r_, x) for r_ in ("pt", "pP") for x in ("NC", "MK4")]:
            lab = g if rep == "raw" else f"{g} ⊥{'t' if rep == 'pt' else 'P'}"
            auc0 = _summ(R, st_, (rep, g), "auc", ids, cls); aucb = _summ(R, st_, (rep, g), "auc", ids, cls, W)
            for wt in ("mass", "pix"):
                p0 = _summ(R, st_, (rep, g), wt, ids, cls); pb = _summ(R, st_, (rep, g), wt, ids, cls, W)
                dR0, dS0 = matched(p0, base[wt][0]); dRb, dSb = matched(pb, base[wt][1])
                g1 = GAMMAS.index(1.0)
                cR, cS = q(dRb), q(dSb)
                w(f"| {lab} | {wt} | {p0[0, g1]:.3f} | {p0[1, g1]:.3f} | {p0[2, g1]:.3f} | "
                  f"{auc0:.3f} [{q(aucb)[0]:.3f}, {q(aucb)[1]:.3f}] | {dR0:+.3f} [{cR[0]:+.3f}, {cR[1]:+.3f}] | {dS0:+.3f} [{cS[0]:+.3f}, {cS[1]:+.3f}] |")
                J_[f"{st_[0]}:{st_[1]}:{rep}:{g}:{wt}"] = {"dR": dR0, "dR_ci": cR, "dS": dS0, "dS_ci": cS, "auc": float(auc0),
                                                          "auc_ci": q(aucb), "curve": p0.tolist()}
                if rule and wt == "mass" and rep == "raw" and g not in ("MK4", "NC"):
                    passed.setdefault(g, []).append(bool(dR0 >= 0.01 and cR[0] > 0))
        # Q-K ratio
        for g in ("NC", "MK4"):
            a0 = J_[f"{st_[0]}:{st_[1]}:raw:{g}:mass"]["auc"]; aP = J_[f"{st_[0]}:{st_[1]}:pP:{g}:mass"]["auc"]
            ratio = (aP - 0.5) / (a0 - 0.5) if abs(a0 - 0.5) > 1e-9 else np.nan
            beyond = ratio >= 0.5 and J_[f"{st_[0]}:{st_[1]}:pP:{g}:mass"]["auc_ci"][0] > 0.5
            w(f"\nQ–K diagnostic, {g}: (AUROC⊥P − 0.5)/(AUROC − 0.5) = {ratio:.2f}; "
              f"information beyond the prompt-key logits: {'**supported**' if beyond else 'not supported'}.")
            J_[f"{st_[0]}:{st_[1]}:qk:{g}"] = {"ratio": float(ratio), "beyond": bool(beyond)}
    verdict = {g: all(v) and len(v) == 2 for g, v in passed.items()}
    for K in (2, 4):
        verdict[f"FP{K}"] = verdict.get(f"FP{K}", False) and verdict.get(f"MP{K}", False) and verdict.get("FD4", False)
    w("\n## Pass rule (mass-weighted, far background; ΔR ≥ +0.01 and CI > 0 on VOC-Sim dev / V and COCO-Cap dev / X)\n")
    for g in GATES[2:]:
        extra = " (read only if its parts pass)" if g.startswith("FP") else ""
        w(f"- {g}: {'**passes**' if verdict.get(g) else 'does not pass'}{extra}")
    rk = FITS["pP"]["ranks"]
    w(f"\nRank fraction of the removed prompt-key span per head (mean over images): " + ", ".join(f"{m.split('.')[0][:4]}{i}: {v:.2f}" for i, (m, v) in enumerate(rk.items())) + "\n")
    J_["verdict"] = verdict
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stage1{a.tag}.json").write_text(json.dumps(J_, indent=1, default=float))
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"stage1{a.tag}.md").write_text(txt)


# ---------------------------------------------------------------- stage 2: capture / score / report (run 31 pattern)
@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(MODEL); FIT = torch.load(WORK / f"fit_raw{a.tag}.pt")["fit"]; Sp = M.load_split(); cache = {}
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in M.rows_for(a, "all"):
        iid = row["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); sp = Sp[iid]; rec, sample = J.generate(rt, row, a.set); lab = AB.label(row)
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); Q = T.qbar(rec); tgt = (None, list(rec.target)); G = {}; miss = 0
        for r in REGIMES:
            sub = "all" if (sp["part"] == "test" or (r == "V" and sp["ds"] == "coco")) else ("dB" if sp["fold"] == "dA" else "dA")
            if (r, sub) not in cache:
                cache[(r, sub)] = to_dev(FIT[r][sub])
            P = cache[(r, sub)]
            for gname in GATES:
                gd = {}
                for m, q in Q.items():
                    g = gate(P, gname, lab, m, q)
                    if g is None:
                        g = torch.ones(q.shape[0], device=q.device); miss += 1
                    gd[m] = g
                G[f"{r}:{gname}"] = gd
            if a.tag:   # smoke: one object centroid, no Fisher == MK4
                for m, q in Q.items():
                    if (lab, m) in P["mo"] and (lab, m) in P["C4"]:
                        d = float((g_pair(q, P["mo"][(lab, m)][None], None, P["C4"][(lab, m)], None, False) - G[f"{r}:MK4"][m]).abs().max())
                        assert d < 1e-5, f"g_pair != MK4 ({d:.2e})"
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
        print(f"[g34:{a.set}] {iid} {sp['part']} fid {fid:.1e} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
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
    V1 = json.loads((OUT / f"stage1{a.tag}.json").read_text())["verdict"]
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
    L = ["# Run 34, stage 2: full pipeline (`PREREG.md`)\n",
         "Test halves; thresholds chosen on dev. **Exploratory.** One generation-time capture for every arm. Regime V = VOC-Sim-only "
         "profiles (COCO-Cap is transfer); regime X = VOC-Sim + COCO-Cap dev profiles (label-free domain adaptation).\n",
         f"Test images: VOC-Sim {len(tst['voc']['cls'])}, COCO-Cap {len(tst['coco']['cls'])}. Stage 1 verdicts: "
         + ", ".join(f"{g} {'pass' if V1.get(g) else 'fail'}" for g in GATES[2:]) + ". A contrast is read only for gates that passed stage 1.\n"]
    for reg in REGIMES:
        L.append(f"\n## Regime {reg}\n")
        L.append("| Arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |"); L.append("|---|---:|---:|---:|---:|")
        for arm in ("A",) + tuple(f"{reg}:{g}" for g in GATES):
            L.append(f"| {arm} | {mi(tst['voc'], pred(reg, 'voc', K(arm))):.3f} | {mi(tst['coco'], pred(reg, 'coco', K(arm))):.3f} | "
                     f"{mi(tst['voc'], pred(reg, 'voc', K(arm), '55')):.3f} | {mi(tst['coco'], pred(reg, 'coco', K(arm), '55')):.3f} |")
        L.append("\nHierarchy + DenseCRF, dev-selected thresholds (0.55 in brackets column).\n")
        L.append("| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |")
        L.append("|---|---|---|---|---|---|---|")
        for g in GATES[2:]:
            x, y = f"{reg}:{g}", f"{reg}:MK4"; c = [dci(ds, pred(reg, ds, K(x)), pred(reg, ds, K(y))) for ds in SETS]
            c5 = [dci(ds, pred(reg, ds, K(x), "55"), pred(reg, ds, K(y), "55")) for ds in SETS]
            ok = V1.get(g, False); imp = c[1][2] > 0 and c[0][2] >= -0.005
            rd = ("**improves on MK4**" if imp else "does not improve on MK4") if ok else "descriptive (failed stage 1)"
            L.append(f"| {g} − MK4 | {'pass' if ok else 'fail'} | {c[0][0]} | {c[1][0]} | {c5[0][0]} | {c5[1][0]} | {rd} |")
        for lb, x, y in (("MK4 − NC", f"{reg}:MK4", f"{reg}:NC"), ("MK4 − A", f"{reg}:MK4", "A"), ("NC − A", f"{reg}:NC", "A")):
            c = [dci(ds, pred(reg, ds, K(x)), pred(reg, ds, K(y))) for ds in SETS]
            c5 = [dci(ds, pred(reg, ds, K(x), "55"), pred(reg, ds, K(y), "55")) for ds in SETS]
            L.append(f"| {lb} | — | {c[0][0]} | {c[1][0]} | {c5[0][0]} | {c5[1][0]} | — |")
        cl = tst["coco"]["cls"]; arms = ["A", f"{reg}:MK4"] + [f"{reg}:{g}" for g in GATES[2:]]
        L.append("\nPer class, COCO-Cap test, dev-selected (" + " / ".join(x.split(':')[-1] for x in arms) + "): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(pred(reg, 'coco', K(x))[:, None], cl, cl == c)[0]:.2f}" for x in arms) for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("fit", "stage1", "capture", "score", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--rep", choices=REPS, default="raw"); p.add_argument("--set", choices=SETS, default="voc")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"fit": fit, "stage1": stage1, "capture": capture, "score": score, "report": report}[a.cmd](a)
