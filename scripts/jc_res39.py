#!/usr/bin/env python3
"""runs/39_lowres_why/PREREG.md: why the low-resolution consensus gate works (stage-1 diagnostics, exploratory).

real    (GPU) VOC train items, one SD1.5 pass at t = 100; MK4 gates with profiles C (and A, cross-fitted)  -> <WORK>/real{tag}_{shard}.pt
gen     (CPU) run 31 dev dumps; MK4 gates with run 35 held-out profiles, run 35 settings                  -> <WORK>/gen{tag}_{shard}.pt
report  (CPU)                                                                                              -> runs/39_lowres_why/result{tag}.md
"""
from __future__ import annotations

import argparse, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_gate36 as G36  # noqa: E402

M, G34, G35 = G36.M, G36.G34, G36.G35
CACHE = M.J.CACHE
WORK = CACHE / "jc_res39"
OUT = ROOT / "runs/39_lowres_why"
GAMMAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
MATCH = [i for i, g in enumerate(GAMMAS) if 0.5 <= g <= 2.0]
CLASSES = M.CLASSES


def up64(g):
    r = math.isqrt(g.shape[0])
    return F.interpolate(g.float().view(1, 1, r, r), size=(64, 64), mode="bilinear", align_corners=False)[0, 0].reshape(-1)


def to_grid(g, res):
    r = math.isqrt(g.shape[0]); x = g.float().view(1, 1, r, r)
    if r == res:
        return x.reshape(-1)
    if r > res:
        return F.adaptive_avg_pool2d(x, res).reshape(-1)
    return F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False).reshape(-1)


def qpool(q, res=16):
    r = math.isqrt(q.shape[0]); d = q.shape[1]
    x = q.float().T.reshape(1, d, r, r)
    return F.normalize(F.adaptive_avg_pool2d(x, res).reshape(d, -1).T, dim=-1)


def variant_names(lo):
    return ["LR", "LR-med", "LR-trim", "AV", "HR", "HR-gp", "HR-qp"] + [f"only-{k}" for k in lo] + [f"LOO-{k}" for k in lo]


def measure(mods, Q, gfn, fg64, seed64, check=False):
    """Per-item rows. mods: ordered module names; Q {m: q [N, d]}; gfn(m, q) -> gate [N] or None."""
    nM = len(mods); res = {m: math.isqrt(Q[m].shape[0]) for m in mods}
    lo = [i for i, m in enumerate(mods) if res[m] < 64]; hi = [i for i, m in enumerate(mods) if res[m] == 64]
    G = {}
    for i, m in enumerate(mods):
        g = gfn(m, Q[m])
        if g is not None:
            g = g.float().clamp(0, 1)
            if not torch.isfinite(g).all():
                raise ValueError(f"non-finite gate at {m}")
            G[i] = g.cpu()
    reg = {r_: M._regions_at(fg64, r_) for r_ in (8, 16, 64)}
    natreg = {r_: M._regions_at(fg64, r_) for r_ in set(res.values())}
    au = lambda g, rg: M.auroc(g[rg[0]].numpy(), g[rg[2]].numpy())
    out = {"auc_nat": np.full(nM, np.nan), "auc16": np.full(nM, np.nan), "auc8": np.full(nM, np.nan), "auc16_qp": np.full(nM, np.nan)}
    for i, g in G.items():
        out["auc_nat"][i] = au(g, natreg[res[mods[i]]])
        out["auc16"][i] = au(to_grid(g, 16), reg[16]); out["auc8"][i] = au(to_grid(g, 8), reg[8])
        if i in hi:
            gq = gfn(mods[i], qpool(Q[mods[i]]).to(Q[mods[i]].device))
            if gq is not None:
                out["auc16_qp"][i] = au(gq.float().clamp(0, 1).cpu(), reg[16])
    U = {i: up64(g) for i, g in G.items()}
    mean_of = lambda ks: torch.stack([U[k] for k in ks]).mean(0) if all(k in U for k in ks) and ks else None
    LO = [k for k in lo if k in U]
    V = {"LR": mean_of(LO), "AV": mean_of(list(U)), "HR": mean_of([k for k in hi if k in U])}
    if len(LO) == len(lo):
        S = torch.stack([U[k] for k in lo]).sort(0).values
        V["LR-med"] = torch.quantile(torch.stack([U[k] for k in lo]), 0.5, dim=0); V["LR-trim"] = S[1:-1].mean(0)
    V["HR-gp"] = torch.stack([up64(to_grid(G[k], 16)) for k in hi if k in G]).mean(0) if any(k in G for k in hi) else None
    qp = []
    for k in hi:
        if k in G:
            gq = gfn(mods[k], qpool(Q[mods[k]]).to(Q[mods[k]].device))
            if gq is not None:
                qp.append(up64(gq.float().clamp(0, 1).cpu()))
    V["HR-qp"] = torch.stack(qp).mean(0) if qp else None
    for k in lo:
        V[f"only-{k}"] = U.get(k); V[f"LOO-{k}"] = mean_of([j for j in lo if j != k])
    if check:   # LR == jc_gate36.consensus at a 64 x 64 module
        cons = G36.consensus({mods[i]: G[i] for i in G}, {mods[i]: (1.0 if res[mods[i]] < 64 else 0.0) for i in G})
        m64 = mods[hi[0]]; d = float((cons[m64] - V["LR"].clamp(0, 1)).abs().max())
        print(f"identity LR vs jc_gate36.consensus: {d:.2e}", flush=True); assert d < 1e-6
    names = variant_names(lo); nG = len(GAMMAS)
    w64 = torch.from_numpy(np.asarray(seed64, np.float32)).reshape(-1).clamp_min(0)
    ret = np.full((len(names) + 1, 3, nG), np.nan); auc64 = np.full(len(names) + 1, np.nan)
    gam = torch.tensor(GAMMAS)[None]
    for vi, nm in enumerate(names):
        v = V.get(nm)
        if v is None:
            continue
        v = v.clamp(0, 1); auc64[vi] = au(v, reg[64]); Gg = v[:, None] ** gam
        for ri, rg in enumerate(reg[64]):
            wm = w64[rg]
            if float(wm.sum()) > 0:
                ret[vi, ri] = ((Gg[rg] * wm[:, None]).sum(0) / wm.sum()).numpy()
    # MK4 (per module, own grid), averaged over modules
    per = []
    for i, g in G.items():
        r_ = res[mods[i]]; wm_all = F.adaptive_avg_pool2d(w64.view(1, 1, 64, 64), r_).reshape(-1); Gg = g[:, None] ** gam
        x = np.full((3, nG), np.nan)
        for ri, rg in enumerate(natreg[r_]):
            wm = wm_all[rg]
            if float(wm.sum()) > 0:
                x[ri] = ((Gg[rg] * wm[:, None]).sum(0) / wm.sum()).numpy()
        per.append(x)
    ret[-1] = np.nanmean(np.stack(per), 0); auc64[-1] = np.nanmean(out["auc_nat"])
    out.update({"ret": ret, "auc64": auc64, "lo": np.array(lo), "hi": np.array(hi)})
    return out


# ---------------------------------------------------------------- real VOC train (GPU)
@torch.no_grad()
def real(a):
    from PIL import Image
    import jc_realvoc37 as R37
    R = R37.R
    pipe, rec = R.load(); Wm = R.word_map(); FIT = torch.load(R37.FIT33, weights_only=False)["fit"]; dev_p = {}
    prof = lambda s: dev_p.setdefault(s, M.to_dev(FIT[s]))
    rows = []; checked = False; t0 = time.perf_counter()
    for n, it in enumerate(R.load_items("train", a)):
        c = it["cls"]
        R.one_pass(pipe, rec, np.asarray(Image.open(R.VOC / f"JPEGImages/{it['img']}.jpg").convert("RGB")), f"A photograph of a {Wm[c]}", Wm[c])
        Q = R.qbar(rec); mods = list(rec.ca_order); seed = R.make_seed(rec, R.layer_weights(rec)).cpu().numpy()
        truth, _ = R37.truth_of(it["img"], it["ci"])
        fg64 = (F.adaptive_avg_pool2d(torch.from_numpy(truth.astype(np.float32))[None, None], 64)[0, 0] > 0.5).numpy()
        row = {"id": it["id"], "cls": c, "set": "real"}
        for src in ("C", "A"):
            P = prof(R37.prof_set("train", src, it["fold"]))
            row[src] = measure(mods, Q, lambda m, q: R37.g_mk4(P, c, m, q), fg64, seed, check=bool(a.tag) and not checked)
            checked = True
        rows.append(row); rec.clear()
        if n % 100 == 0:
            print(f"[r39:real] {n} {it['id']} {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save({"rows": rows, "modules": mods}, WORK / f"real{a.tag}_{a.shard}.pt")


# ---------------------------------------------------------------- generated dev dumps (CPU)
def gen(a):
    torch.set_num_threads(a.workers)
    fs = G34.dumps(a)[a.shard::a.num_shards]; Mt = M._meta(fs); Sp = M.load_split(); mods = list(Mt[0]["modules"])
    FIT = torch.load(CACHE / "jc_gate35" / "fit.pt")["fit"]; rows = []; checked = False; t0 = time.perf_counter()
    for n, f in enumerate(fs):
        z = np.load(f); ds = Mt[n]["ds"]; c = Mt[n]["label"]; iid = f.stem
        Q = {m: F.normalize(torch.from_numpy(z[f"q_{i:02d}"]).float(), dim=1) for i, m in enumerate(mods)}
        for s_ in [x for x in G34.SETTINGS if x[1] == ds]:
            r = s_[0]; P = FIT[r][G34.held_out(Sp, iid, r, ds)]
            row = {"id": iid, "cls": c, "set": f"{ds}/{r}"}
            row["G"] = measure(mods, Q, lambda m, q: G35.gate(P, "MK4", c, m, q, None), z["fg64"], z["seed"], check=bool(a.tag) and not checked)
            checked = True; rows.append(row)
        if n % 50 == 0:
            print(f"[r39:gen] {n}/{len(fs)} {iid} {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save({"rows": rows, "modules": mods}, WORK / f"gen{a.tag}_{a.shard}.pt")


# ---------------------------------------------------------------- report
class CB:
    """Class-balanced means with image-bootstrap draws (paired across quantities)."""

    def __init__(self, cls, nb):
        self.u, ci = np.unique(cls, return_inverse=True); self.O = np.eye(len(self.u))[ci]
        rng = np.random.default_rng(0); n = len(cls)
        self.W = np.vstack([np.ones(n), np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)])]).astype(np.float64)

    def __call__(self, X):
        """X [n, ...] -> [1 + nb, ...] (row 0 = point estimate). NaN entries are skipped."""
        sh = X.shape[1:]; X2 = X.reshape(len(X), -1); ok = np.isfinite(X2); Xs = np.where(ok, X2, 0.0)
        num = self.W @ (self.O[:, :, None] * Xs[:, None, :]).reshape(len(X), -1)
        den = self.W @ (self.O[:, :, None] * ok[:, None, :]).reshape(len(X), -1)
        k = Xs.shape[1]; num, den = num.reshape(-1, len(self.u), k), den.reshape(-1, len(self.u), k)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        return np.nanmean(cm, 1).reshape((-1,) + sh)


def matched(objv, farv, objr, farr):
    """Object retention of v at r's far retention, minus r's, averaged over MATCH. Inputs [B, nG]."""
    out = np.full(len(objv), np.nan)
    for b in range(len(objv)):
        xv, yv = farv[b][::-1], objv[b][::-1]
        if not (np.isfinite(xv).all() and np.isfinite(yv).all() and np.isfinite(farr[b]).all()):
            continue
        out[b] = np.mean([np.interp(farr[b][i], xv, yv) - objr[b][i] for i in MATCH])
    return out


def report(a):
    rows = []; mods = None
    for kind in ("real", "gen"):
        for f in sorted(WORK.glob(f"{kind}{a.tag}_*.pt")):
            d = torch.load(f, weights_only=False); rows += d["rows"]; mods = d["modules"]
    sets = [("real VOC train / C (primary)", "real", "C"), ("real VOC train / A (reference)", "real", "A"),
            ("VOC-Sim dev / V", "voc/V", "G"), ("COCO-Cap dev / X", "coco/X", "G"), ("VOC-Sim dev / X", "voc/X", "G"),
            ("COCO-Cap dev / V", "coco/V", "G")]
    L = []; w = L.append
    ex = next(r[k] for r in rows for k in ("C", "G") if k in r); lo, hi = list(ex["lo"]), list(ex["hi"]); names = variant_names(lo) + ["MK4"]
    w("# Run 39: why the low-resolution consensus gate works (`PREREG.md`)\n")
    w(f"**Exploratory, diagnostic.** MK4 gates at γ = 1 (curves over γ), held-out profiles. Class-balanced means; 95% CIs from "
      f"{a.boot} image-bootstrap samples. Low-resolution modules: {lo}; 64 × 64 modules: {hi}.\n")
    w("Modules: " + ", ".join(f"{i} = {m.split('.transformer')[0]}" for i, m in enumerate(mods)) + "\n")
    q = lambda x: (float(x[0]), float(np.nanpercentile(x[1:], 2.5)), float(np.nanpercentile(x[1:], 97.5)))
    f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"; fd = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    verdicts = {}
    for excl in (False, True):
        w(f"\n# {'Person excluded' if excl else 'All classes'}\n")
        for title, key, src in sets:
            R_ = [r[src] for r in rows if r["set"] == key and src in r and not (excl and r["cls"] == "person")]
            cls = np.array([r["cls"] for r in rows if r["set"] == key and src in r and not (excl and r["cls"] == "person")])
            if len(R_) < 2:
                continue
            cb = CB(cls, a.boot); st = lambda k: cb(np.stack([x[k] for x in R_]))
            w(f"\n## {title}: {len(R_)} items, {len(np.unique(cls))} classes\n")
            # A. resolution control
            An, A16, A8, Aq = st("auc_nat"), st("auc16"), st("auc8"), st("auc16_qp")
            w("### A. Per-module AUROC (object vs far), MK4 gate\n")
            w("| Module | grid | native | at 16 × 16 | at 8 × 8 | query-pooled, at 16 × 16 |"); w("|---|---:|---|---|---|---|")
            for i in range(len(mods)):
                w(f"| {i} | {64 if i in hi else ('8' if i == 4 else '16')} | {f3(q(An[:, i]))} | {f3(q(A16[:, i]))} | {f3(q(A8[:, i]))} | "
                  f"{f3(q(Aq[:, i])) if i in hi else '—'} |")
            a_ = np.nanmean(A16[:, lo], 1); hn = np.nanmean(An[:, hi], 1); hg = np.nanmean(A16[:, hi], 1); hq = np.nanmean(Aq[:, hi], 1)
            w("\n| Group mean | AUROC [95% CI] | minus low-res at 16 |"); w("|---|---|---|")
            for lab, x in (("a: low-res, at 16", a_), ("64 × 64, native", hn), ("h_g: 64 × 64, gate-pooled to 16", hg),
                           ("h_q: 64 × 64, query-pooled to 16", hq)):
                w(f"| {lab} | {f3(q(x))} | {fd(q(x - a_)) if lab[0] != 'a' else '—'} |")
            best = max(float(hg[0]), float(hq[0])); gap = float(a_[0]) - best
            rd = "spatial averaging" if gap <= 0.05 else "representation difference" if gap >= 0.15 else "partial"
            verdicts[(title, excl)] = (rd, gap)
            w(f"\nReading (`PREREG.md` §3A): a − max(h_g, h_q) = {gap:+.3f} → **{rd}**{' (primary set)' if src == 'C' else ' (descriptive)'}.\n")
            # B. consensus variants
            RT = st("ret"); A64 = st("auc64"); li = names.index("LR")
            w("### B. Consensus variants (64 × 64 map; mass-weighted retention)\n")
            w("| Variant | AUROC obj vs far | object kept, γ=1 | far kept, γ=1 | ΔR vs LR [95% CI] | ΔR vs MK4 [95% CI] |"); w("|---|---|---:|---:|---|---|")
            g1 = GAMMAS.index(1.0); mk = names.index("MK4")
            for vi, nm in enumerate(names):
                if not np.isfinite(RT[0, vi]).any():
                    continue
                dl = matched(RT[:, vi, 0], RT[:, vi, 2], RT[:, li, 0], RT[:, li, 2]) if nm != "LR" else None
                dm = matched(RT[:, vi, 0], RT[:, vi, 2], RT[:, mk, 0], RT[:, mk, 2]) if nm != "MK4" else None
                w(f"| {nm} | {f3(q(A64[:, vi]))} | {RT[0, vi, 0, g1]:.3f} | {RT[0, vi, 2, g1]:.3f} | {fd(q(dl)) if dl is not None else '—'} | "
                  f"{fd(q(dm)) if dm is not None else '—'} |")
                if nm in ("LR-med", "LR-trim") and not excl:
                    verdicts[(title, nm)] = q(dl)
    w("\n# Pass rule for robust consensus (`PREREG.md` §3B)\n")
    for nm in ("LR-med", "LR-trim"):
        ok = all((t, nm) in verdicts and verdicts[(t, nm)][0] >= 0.01 and verdicts[(t, nm)][1] > 0
                 for t in ("real VOC train / C (primary)", "COCO-Cap dev / X"))
        w(f"- {nm} vs LR: {'**passes**' if ok else 'does not pass'} (real VOC / C: {fd(verdicts.get(('real VOC train / C (primary)', nm), (np.nan,) * 3))}; "
          f"COCO-Cap / X: {fd(verdicts.get(('COCO-Cap dev / X', nm), (np.nan,) * 3))}).")
    w("\n# Resolution reading (`PREREG.md` §3A)\n")
    for (t, ex_), v in verdicts.items():
        if isinstance(ex_, bool):
            w(f"- {t}{', person excluded' if ex_ else ''}: **{v[0]}** (gap {v[1]:+.3f}).")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("real", "gen", "report"):
        p = sub.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--max-images", type=int, default=0)
        p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"real": real, "gen": gen, "report": report}[a.cmd](a)
