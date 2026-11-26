#!/usr/bin/env python3
"""runs/41_robust_consensus/PREREG.md: LR-med / LR-trim / LR without module 2 vs MK4-LR, end to end.

items    (CPU) new COCO train2017 confirmation set (3,500 photos, seed 41)          -> <WORK>/items_conf{tag}.json, masks/
capture  (GPU) --set S: one SD1.5 pass at t = 100; seeds w- + w{M,L,D,T,O}{gamma grid}; maps seed / V0 / V3 -> <WORK>/cap_S/
raw      (CPU) --set S: counts at every threshold                                    -> <WORK>/cap_S_raw/
select   (CPU) gamma per (gate, map) and VOC thresholds from vtrain; COCO thresholds from cfit -> <WORK>/sel.json
crf      (CPU) --set S: DenseCRF on the selected-gamma and gamma-1 V0 / V3 maps      -> <WORK>/cap_S_crf/
winner   (CPU) development rule on vtrain (PREREG section 4)                        -> <WORK>/winner.json
report   (CPU)                                                                       -> runs/41_robust_consensus/result.md

Sets: vtrain (VOC train, development), cfit (run 40 COCO train threshold items), conf (new COCO train set, confirmation),
cval (run 40 COCO val2017 set, reuse), vval (VOC val, reuse).
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_realvoc37 as R37  # noqa: E402
import jc_coco40 as C40  # noqa: E402
import jc_gate36 as G36  # noqa: E402

R, M, cc, st = R37.R, R37.M, R37.cc, R37.st
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_cons41"
OUT = ROOT / "runs/41_robust_consensus"
CLASSES, THR, T55, DEV, GRID, K1 = R.CLASSES, R37.THR, R37.T55, R.DEV, R37.GRID, R37.K1
COCO, VOC = C40.COCO, R.VOC
N_CONF, CONF_SEED, MIN_AREA = 3500, 41, C40.MIN_AREA
GATES = ("M", "L", "D", "T", "O")
CANDS = ("D", "T", "O")
GN = {"-": "none", "M": "MK4", "L": "MK4-LR", "D": "LR-med", "T": "LR-trim", "O": "LR without module 2"}
PROPS = ("seed", "V0", "V3")
SEEDS = ("w-",) + tuple(f"w{g}{k}" for g in GATES for k in range(len(GRID)))
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in PROPS)
MOD2 = "down_blocks.2.attentions.0"
TIE, O_MARGIN = 0.005, 0.005
SETS = {"vtrain": "voc", "vval": "voc", "cfit": "coco", "conf": "coco", "cval": "coco"}
SMOKE_SUB = {"conf": "cfit", "cval": "cfit", "vval": "vtrain"}      # smoke report only: evaluation sets are never touched


def items_file(s, tag):
    return {"vtrain": R.WORK / "items_train.json", "vval": R.WORK / "items_val.json",
            "cfit": C40.WORK / f"items_train.json", "cval": C40.WORK / "items_val.json",
            "conf": WORK / f"items_conf{tag}.json"}[s]


def mask_dir(s):
    return WORK / "masks" if s == "conf" else C40.WORK / "masks"


def load_items(s, a):
    X = json.loads(items_file(s, a.tag if s == "conf" else "").read_text())[a.shard::a.num_shards]
    if a.max_items:
        X = X[:: max(1, len(X) // a.max_items)][: a.max_items]
    return X


# ---------------------------------------------------------------- items
def items(a):
    prof = {int(r["path"].split("/")[1].split(".")[0]) for r in json.loads((CACHE / "jc_realvoc_t" / "items_C.json").read_text())}
    fit40 = {int(r["img"]) for r in json.loads((C40.WORK / "items_train.json").read_text())}
    d = json.loads((COCO / "annotations/instances_train2017.json").read_text())
    voc_ids = {c["id"] for c in d["categories"] if C40.C2V.get(c["name"], c["name"]) in C40.VOCN}
    elig = sorted({an["image_id"] for an in d["annotations"] if an["category_id"] in voc_ids and not an["iscrowd"]} - prof - fit40)
    del d
    chosen = set(random.Random(CONF_SEED).sample(elig, N_CONF))
    C40.WORK = WORK                                   # masks for the new set go to <WORK>/masks
    X = C40.build("ctr", "train2017", lambda i: i in chosen, max_items=a.max_items)
    assert not ({x["img"] for x in X} & (prof | fit40)), "confirmation photo in a fit set"
    (WORK / f"items_conf{a.tag}.json").write_text(json.dumps(X))
    print(f"eligible {len(elig)}; chosen {len(chosen)}; items {len(X)} from {len({x['img'] for x in X})} photos; "
          f"primary {sum(x['area'] >= MIN_AREA for x in X)}", flush=True)


# ---------------------------------------------------------------- gates
def up64(g):
    r = math.isqrt(g.shape[0])
    return F.interpolate(g.float().view(1, 1, r, r), size=(64, 64), mode="bilinear", align_corners=False)[0, 0]


def cons(base, how, check=False):
    """Low-resolution consensus combined per pixel at 64 x 64, area-resized back to every module (as jc_gate36.consensus)."""
    res = {m: math.isqrt(g.shape[0]) for m, g in base.items()}
    lo = [m for m in base if res[m] < 64 and not (how == "O" and m.startswith(MOD2))]
    if not lo:
        return dict(base), 0
    U = torch.stack([up64(base[m]) for m in lo]); full = sum(res[m] < 64 for m in base) == 6
    if how == "D" and full:
        cb = torch.quantile(U, 0.5, dim=0)
    elif how == "T" and full:
        cb = U.sort(0).values[1:-1].mean(0)
    else:
        cb = U.mean(0)
    if check and how in ("D", "T"):
        assert bool((cb >= U.min(0).values - 1e-6).all() and (cb <= U.max(0).values + 1e-6).all()), "robust consensus out of range"
    out = {m: F.adaptive_avg_pool2d(cb[None, None], res[m])[0, 0].reshape(-1).clamp(0, 1) for m in base}
    return out, int(full)


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention
    pipe, rec = R.load(); Wm = R.word_map()
    P = M.to_dev(torch.load(R37.FIT33, weights_only=False)["fit"]["C_all"])
    out = WORK / f"cap_{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True); checked = False
    for it in load_items(a.set, a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]
        path = VOC / f"JPEGImages/{it['img']}.jpg" if SETS[a.set] == "voc" else COCO / it["path"]
        R.one_pass(pipe, rec, np.asarray(Image.open(path).convert("RGB")), f"A photograph of a {Wm[c]}", Wm[c])
        if not checked:
            assert rec.ca_order[2].startswith(MOD2), rec.ca_order[2]
        w_a3 = R.layer_weights(rec); Q = R.qbar(rec)
        base = {m: g for m, q in Q.items() if (g := R37.g_mk4(P, c, m, q)) is not None}
        B, full = {"M": base}, {}
        for g in "LDTO":
            B[g], full[g] = cons(base, g, check=not checked)
        if not checked:   # identities (first item of every shard)
            ref = G36.consensus(base, {m: (1.0 if math.isqrt(x.shape[0]) < 64 else 0.0) for m, x in base.items()})
            dl = max(float((ref[m] - B["L"][m]).abs().max()) for m in base)
            print(f"identity LR vs jc_gate36.consensus: {dl:.2e}", flush=True); assert dl < 1e-6
        gmean = {g: float(torch.stack([x.mean() for x in B[g].values()]).mean()) if B[g] else 1.0 for g in B}
        H = {g: _attention(rec.grid_qk[g], g) for g in R.SA}; maps = {}
        for nm in SEEDS:
            gd = None if nm[1] == "-" else R37.powered(B[nm[1]], Q, GRID[int(nm[2:])])
            s = R.make_seed(rec, w_a3, gd); maps[f"{nm}|seed"] = s
            pr = R.propagate(H, s, v3=True); maps[f"{nm}|V0"] = pr["V0"]; maps[f"{nm}|V3"] = pr["V3"]
        if not checked:
            from src.segmentation import sg_combined_v2
            ref, _ = sg_combined_v2.aggregate({g: [rec.grid_qk[g]] * 11 for g in R.SA}, maps["w-|seed"])
            worst = max(float(np.abs(st.mm(maps[f"w-|{k}"].cpu().numpy()) - st.mm(ref[k].float().cpu().numpy())).max()) for k in ("V0", "V3"))
            print(f"identity vs sg_combined_v2.aggregate (V0, V3): max |d| {worst:.2e}", flush=True); assert worst < 1e-5
            checked = True
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {it['id']}")
        meta = {"cls": c, "img": str(it["img"]), "ci": int(it["ci"]), "area": float(it.get("area", np.nan)),
                "path": str(it.get("path", "")), "miss": len(Q) - len(base), "full6": full["D"]}
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), **meta, **{f"gmean_{k}": v for k, v in gmean.items()})
        os.replace(tmp, o)
        print(f"[c41:{a.set}] {it['id']} {c} miss {meta['miss']} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- scoring
def truth_for(s, z, stem):
    if SETS[s] == "voc":
        truth, gt = R37.truth_of(str(z["img"]), int(z["ci"])); valid = gt != 255
        return truth, valid, valid & (gt > 0) & ~truth, VOC / f"JPEGImages/{z['img']}.jpg"
    mk = np.load(mask_dir(s) / f"{stem}.npz")
    return mk["truth"], ~mk["ignore"], mk["other"], COCO / str(z["path"])


def score_one(job):
    s, f, o, kind, thr, crf = job[0], Path(job[1]), Path(job[2]), job[3], job[4], job[5]
    if o.exists():
        return
    torch.set_num_threads(1)
    from PIL import Image
    z = np.load(f); truth, valid, other, ipath = truth_for(s, z, f.stem)
    res = {"cls": str(z["cls"]), "img": str(z["img"]), "area": float(z["area"]), "full6": int(z["full6"]), "miss": int(z["miss"])}
    res.update({k: z[k] for k in z.files if k.startswith("gmean_")})
    rgb = None
    for i, nm in enumerate(str(x) for x in z["names"]):
        if kind == "crf" and nm not in crf:
            continue
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        if kind == "raw":
            res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        else:
            from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
            _import_densecrf(st.CRF_SITE)
            if rgb is None:
                rgb = np.asarray(Image.open(ipath).convert("RGB"), np.uint8)
            ms = np.stack([apply_ovam_densecrf(rgb, x >= t, st.CRF_SITE) for t in (thr[nm], float(THR[T55]))]).astype(bool)
            res[f"{nm}|crf"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def run_score(a, kind):
    thr, crf = None, None
    if kind == "crf":
        Sl = json.loads((WORK / f"sel{a.tag}.json").read_text())
        thr = Sl["thr_voc" if SETS[a.set] == "voc" else "thr_coco"]; crf = frozenset(Sl["crf"])
    od = WORK / f"cap_{a.set}{a.tag}_{kind}"; od.mkdir(parents=True, exist_ok=True)
    jobs = [(a.set, str(f), str(od / f.name), kind, thr, crf) for f in sorted((WORK / f"cap_{a.set}{a.tag}").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    for attempt in range(5):
        todo = [j for j in jobs if not Path(j[2]).exists()]; print(f"{kind} {a.set} attempt {attempt}: {len(todo)} to score", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers) as ex:
                for n, _ in enumerate(ex.map(score_one, todo, chunksize=1), 1):
                    if n % 200 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)
        except BrokenProcessPool:
            print("worker died; restarting pool", flush=True)
    left = [j for j in jobs if not Path(j[2]).exists()]
    if left:
        raise SystemExit(f"{len(left)} items not scored")


def load_scores(s, tag, kinds=("raw", "crf")):
    S = None
    for kind in kinds:
        d = WORK / f"cap_{s}{tag}_{kind}"
        if not d.exists():
            continue
        fs = sorted(f for f in d.glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
        if S is None:
            S = {"id": np.array([f.stem for f in fs]), "cls": np.array([str(z["cls"]) for z in Z]),
                 "grp": np.array([str(z["img"]) for z in Z]), "area": np.array([float(z["area"]) for z in Z]),
                 "full6": np.array([int(z["full6"]) for z in Z])}
            for k in Z[0].files:
                if k.startswith("gmean_"):
                    S[k] = np.array([float(z[k]) for z in Z])
        else:
            assert list(S["id"]) == [f.stem for f in fs], f"{s}: {kind} items differ from raw"
        for k in Z[0].files:
            if "|" in k:
                S[k] = np.stack([z[k] for z in Z])
    return S


# ---------------------------------------------------------------- select / winner
def select(a):
    V = load_scores("vtrain", a.tag, ("raw",)); Cf = load_scores("cfit", a.tag, ("raw",))
    def best(S):
        one = np.ones(len(S["cls"]), bool); t, m = {}, {}
        for nm in MAPS:
            cv = cc.miou(S[f"{nm}|raw"], S["cls"], one); i = int(np.argmax(cv)); t[nm] = float(THR[i]); m[nm] = float(cv[i])
        return t, m
    tv, mv = best(V); tc, _ = best(Cf)
    gam = {f"{g}|{p}": int(np.argmax([mv[f"w{g}{k}|{p}"] for k in range(len(GRID))])) for g in GATES for p in PROPS}
    crf = sorted({f"w-|{p}" for p in ("V0", "V3")} | {f"w{g}{k}|{p}" for g in GATES for p in ("V0", "V3") for k in (gam[f"{g}|{p}"], K1)})
    (WORK / f"sel{a.tag}.json").write_text(json.dumps({"thr_voc": tv, "thr_coco": tc, "train_miou_voc": mv, "gamma": gam, "crf": crf,
                                                       "items": {"vtrain": int(len(V["cls"])), "cfit": int(len(Cf["cls"]))}}, indent=1))
    print("gamma: " + ", ".join(f"{k} {GRID[v]}" for k, v in gam.items()) + f"; {len(crf)} CRF maps", flush=True)


def nm_of(g, p, gam, fixed=False):
    return f"w-|{p}" if g == "-" else f"w{g}{K1 if fixed else gam[f'{g}|{p}']}|{p}"


def winner(a):
    Sl = json.loads((WORK / f"sel{a.tag}.json").read_text()); gam = Sl["gamma"]
    S = load_scores("vtrain", a.tag); B = R.Boot(S, a.boot); d = {}
    for X in CANDS:
        pt, draws = [], []
        for p in ("V0", "V3"):
            x, y = B.stat(S[f"{nm_of(X, p, gam)}|crf"][:, 0]), B.stat(S[f"{nm_of('L', p, gam)}|crf"][:, 0])
            pt.append(x[0] - y[0]); draws.append(x[1] - y[1])
        dd = np.mean(draws, 0); d[X] = (float(np.mean(pt)), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))
    pick = "T" if abs(d["D"][0] - d["T"][0]) < TIE else max(("D", "T"), key=lambda k: d[k][0])
    if d["O"][0] >= d[pick][0] + O_MARGIN:
        pick = "O"
    out = {"d": d, "winner": pick, "dev_pass": bool(d[pick][1] > 0)}
    (WORK / f"winner{a.tag}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out), flush=True)


# ---------------------------------------------------------------- report
def report(a):
    Sl = json.loads((WORK / f"sel{a.tag}.json").read_text()); gam = Sl["gamma"]
    Wn = json.loads((WORK / f"winner{a.tag}.json").read_text()); X, dev_ok = Wn["winner"], Wn["dev_pass"]
    src = (lambda s: SMOKE_SUB.get(s, s)) if a.tag == "_smoke" else (lambda s: s)
    L = []; w = L.append
    sub = lambda S, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}
    HN = {"V0": "fine-only (V0)", "V3": "capped hierarchy (V3)", "seed": "seed only"}
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    rd = lambda x: "**helps**" if x[1] > 0 else "**hurts**" if x[2] < 0 else "not resolved"

    class E:
        def __init__(self, S, thr):
            self.S, self.B, self.c, self.thr = S, R.Boot(S, a.boot), {}, thr
            self.ti = {n: int(np.flatnonzero(np.isclose(THR, thr[n]))[0]) for n in MAPS}

        def P(self, n, rule):
            if rule == "sel_raw":
                return self.S[f"{n}|raw"][:, self.ti[n]]
            if rule == "055_raw":
                return self.S[f"{n}|raw"][:, T55]
            if isinstance(rule, int):
                return self.S[f"{n}|raw"][:, rule]
            return self.S[f"{n}|crf"][:, 0 if rule == "sel_crf" else 1]

        def m(self, n, rule="sel_crf"):
            if (n, rule) not in self.c:
                self.c[(n, rule)] = self.B.stat(self.P(n, rule))
            return self.c[(n, rule)]

        def d(self, x, y, rule="sel_crf"):
            A, B_ = self.m(x, rule), self.m(y, rule); dd = A[1] - B_[1]
            return float(A[0] - B_[0]), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
    rl = lambda p: "sel_raw" if p == "seed" else "sel_crf"

    def ev(s, primary_only):
        S = load_scores(src(s), a.tag); thr = Sl["thr_voc" if SETS[s] == "voc" else "thr_coco"]
        if primary_only:
            S = sub(S, S["area"] >= MIN_AREA)
        return S, E(S, thr)

    Sc_all = load_scores(src("conf"), a.tag); Sc, Ec = ev("conf", True)
    Ex = E(sub(Sc, Sc["cls"] != "person"), Sl["thr_coco"]); Ea = E(Sc_all, Sl["thr_coco"])
    w("# Run 41: median and trimmed low-resolution consensus vs MK4-LR, end to end (`PREREG.md`)\n")
    if a.tag == "_smoke":
        w("**SMOKE: evaluation sets replaced by training sets (conf, cval → cfit; vval → vtrain). Numbers are not results.**\n")
    w(f"**Development on VOC train, frozen confirmation on a new COCO train2017 set.** Confirmation set: {len(Sc_all['cls'])} items "
      f"({len(np.unique(Sc_all['grp']))} photos); primary items (class area ≥ {MIN_AREA:.0%}): {len(Sc['cls'])} "
      f"({len(np.unique(Sc['grp']))} photos). COCO thresholds from {Sl['items']['cfit']} run 40 threshold-fit items; VOC thresholds and "
      f"γ from {Sl['items']['vtrain']} VOC train items. Source C profiles, SA weights on. Class-macro mIoU; 95% photo-bootstrap "
      f"intervals ({a.boot} samples, paired).\n")

    w("\n## 1. Development on VOC train (PREREG §4)\n")
    w("Selected γ (VOC train, raw maps):\n")
    w("| Map | " + " | ".join(GN[g] for g in GATES) + " |"); w("|---|" + "---:|" * len(GATES))
    for p in PROPS:
        w(f"| {HN[p]} | " + " | ".join(f"{GRID[gam[f'{g}|{p}']]}" for g in GATES) + " |")
    Sv = load_scores("vtrain", a.tag); Ev = E(Sv, Sl["thr_voc"])
    w("\nVOC train mIoU, + DenseCRF at the VOC-train threshold, selected γ (in-sample):\n")
    w("| Map | none | " + " | ".join(GN[g] for g in GATES) + " |"); w("|---|---:|" + "---:|" * len(GATES))
    for p in ("V0", "V3"):
        w(f"| {HN[p]} | {Ev.m(nm_of('-', p, gam))[0]:.3f} | " + " | ".join(f"{Ev.m(nm_of(g, p, gam))[0]:.3f}" for g in GATES) + " |")
    w("\n| Candidate | d = mean over V0, V3 of (candidate − MK4-LR) [95% CI] |"); w("|---|---|")
    for k in CANDS:
        w(f"| {GN[k]} | {f3(Wn['d'][k])} |")
    w(f"\n- Winner by the fixed rule: **{GN[X]}**.")
    w(f"- Development: **{'passes' if dev_ok else 'fails'}** (the winner's CI {'is' if dev_ok else 'is not'} above 0).")
    w(f"- Items missing one of the six low-resolution gates (D / T fall back to the mean): VOC train {int((Sv['full6'] == 0).sum())}, "
      f"confirmation {int((Sc_all['full6'] == 0).sum())}.")

    w("\n## 2. Primary contrasts (new COCO set, primary items, SA weights on, selected γ, + DenseCRF at the COCO-fit threshold)\n")
    w("| # | Contrast | Map | Δ mIoU [95% CI] | Reading |"); w("|---|---|---|---|---|"); pc = {}
    for k, p in ((1, "V3"), (2, "V0")):
        pc[k] = Ec.d(nm_of(X, p, gam), nm_of("L", p, gam))
        w(f"| P{k} | {GN[X]} − MK4-LR | {HN[p]} | {f3(pc[k])} | {rd(pc[k])} |")
    ok = dev_ok and (pc[1][1] > 0 or pc[2][1] > 0) and not (pc[1][2] < 0 or pc[2][2] < 0)
    verdict = "**confirmed**" if ok else ("**not confirmed (failed development)**; P1 and P2 are descriptive" if not dev_ok else "**not confirmed**")
    w(f"\n- {GN[X]} over MK4-LR: {verdict} (rule: development passes, and P1 or P2 helps, and neither hurts).")

    w("\n## 3. mIoU, new COCO set, primary items (seed only: raw, COCO-fit threshold; V0 / V3: + DenseCRF)\n")
    w("| γ | Map | none | " + " | ".join(GN[g] for g in GATES) + " |"); w("|---|---|---:|" + "---:|" * len(GATES))
    for fixed in (False, True):
        for p in PROPS:
            w(f"| {'1' if fixed else 'selected'} | {HN[p]} | {Ec.m(nm_of('-', p, gam), rl(p))[0]:.3f} | " +
              " | ".join(f"{Ec.m(nm_of(g, p, gam, fixed), rl(p))[0]:.3f}" for g in GATES) + " |")

    w("\n## 4. Paired differences, new COCO set, primary items\n")
    w("| γ | Map | LR-med − MK4-LR | LR-trim − MK4-LR | LR without module 2 − MK4-LR | MK4-LR − MK4 | "
      f"{GN[X]} − none |"); w("|---|---|---|---|---|---|---|")
    for fixed in (False, True):
        for p in PROPS:
            n_ = lambda g: nm_of(g, p, gam, fixed and g != "-")
            w(f"| {'1' if fixed else 'selected'} | {HN[p]} | " + " | ".join(f3(Ec.d(n_(k), n_("L"), rl(p))) for k in CANDS) +
              f" | {f3(Ec.d(n_('L'), n_('M'), rl(p)))} | {f3(Ec.d(n_(X), n_('-'), rl(p)))} |")
    w("\nHierarchy (V3 − V0) per gate, selected γ, + DenseCRF:\n")
    w("| Gate | V3 − V0 |"); w("|---|---|")
    for g in ("-",) + GATES:
        w(f"| {GN[g]} | {f3(Ec.d(nm_of(g, 'V3', gam), nm_of(g, 'V0', gam)))} |")

    w("\n## 5. Person excluded, and all items (new COCO set, selected γ, + DenseCRF)\n")
    w("| Contrast | Map | person excluded | all items (incl. < 1%) |"); w("|---|---|---|---|")
    for g, ref in [(k, "L") for k in CANDS] + [("L", "M"), (X, "-")]:
        for p in ("V3", "V0"):
            w(f"| {GN[g]} − {GN[ref]} | {HN[p]} | {f3(Ex.d(nm_of(g, p, gam), nm_of(ref, p, gam)))} | "
              f"{f3(Ea.d(nm_of(g, p, gam), nm_of(ref, p, gam)))} |")

    w("\n## 6. Threshold rules (new COCO set, primary items, selected γ)\n")
    RULES = {"sel_crf": "+ CRF, COCO-fit thr (primary)", "055_crf": "+ CRF, 0.55", "sel_raw": "raw, COCO-fit thr", "055_raw": "raw, 0.55"}
    w("| Map | Gate | " + " | ".join(RULES.values()) + " |"); w("|---|---|" + "---:|" * len(RULES))
    for p in ("V0", "V3"):
        for g in ("-",) + GATES:
            w(f"| {HN[p]} | {GN[g]} | " + " | ".join(f"{Ec.m(nm_of(g, p, gam), r_)[0]:.3f}" for r_ in RULES) + " |")
    w("\nRaw maps, candidate − MK4-LR at every threshold (selected γ; † = 95% CI excludes 0):\n")
    w("| Contrast | Map | " + " | ".join(f"{t:.3f}" for t in THR) + " |"); w("|---|---|" + "---:|" * len(THR))
    for k in CANDS:
        for p in PROPS:
            row = []
            for i in range(len(THR)):
                x = Ec.d(nm_of(k, p, gam), nm_of("L", p, gam), i); row.append(f"{x[0]:+.3f}{'†' if x[1] > 0 or x[2] < 0 else ''}")
            w(f"| {GN[k]} − MK4-LR | {HN[p]} | " + " | ".join(row) + " |")

    w(f"\n## 7. Per class IoU (new COCO set, primary items, selected γ, + DenseCRF)\n")
    hdr = [(nm_of(g, p, gam), f"{GN[g]} {p}") for p in ("V3", "V0") for g in ("M", "L", X)]
    w("| Class | items | " + " | ".join(h for _, h in hdr) + " |"); w("|---|---:|" + "---:|" * len(hdr))
    better = {p: [0, 0] for p in ("V3", "V0")}
    for c in CLASSES:
        sel = Sc["cls"] == c
        if not sel.any():
            continue
        v = {n_: cc.miou(Ec.P(n_, "sel_crf")[:, None], Sc["cls"], sel)[0] for n_, _ in hdr}
        for p in ("V3", "V0"):
            dl = v[nm_of(X, p, gam)] - v[nm_of("L", p, gam)]; better[p][0] += dl > 0; better[p][1] += dl < 0
        w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{v[n_]:.3f}" for n_, _ in hdr) + " |")
    w(f"\nClasses where {GN[X]} beats / trails MK4-LR: " + "; ".join(f"{p}: {b[0]} / {b[1]}" for p, b in better.items()) + ".")
    w("\nMean gate value at γ = 1 (primary items): " + ", ".join(f"{GN[g]} {float(np.mean(Sc[f'gmean_{g}'])):.3f}" for g in GATES) + ".")

    w("\n## 8. Reuse sets (secondary; both evaluated before, so agreement only)\n")
    w("Selected γ, + DenseCRF. COCO val2017 (run 40 set): primary items, COCO-fit thresholds. VOC val: all items, VOC-train thresholds.\n")
    w("| Set | Map | LR-med − MK4-LR | LR-trim − MK4-LR | LR without module 2 − MK4-LR | MK4-LR − MK4 | MK4-LR mIoU |")
    w("|---|---|---|---|---|---|---:|")
    for s, prim, lab in (("cval", True, "COCO val2017"), ("vval", False, "VOC val")):
        _, Er = ev(s, prim)
        for p in ("V3", "V0"):
            w(f"| {lab} | {HN[p]} | " + " | ".join(f3(Er.d(nm_of(k, p, gam), nm_of("L", p, gam))) for k in CANDS) +
              f" | {f3(Er.d(nm_of('L', p, gam), nm_of('M', p, gam)))} | {Er.m(nm_of('L', p, gam))[0]:.3f} |")
    w("\nBRACE stays frozen; any change to the paper's method is a separate owner decision.\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("items", "capture", "raw", "select", "crf", "winner", "report"):
        p = sp.add_parser(cmd)
        p.add_argument("--set", choices=tuple(SETS), default="vtrain")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"items": items, "capture": capture, "raw": lambda a: run_score(a, "raw"), "select": select,
     "crf": lambda a: run_score(a, "crf"), "winner": winner, "report": report}[a.cmd](a)
