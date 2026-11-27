#!/usr/bin/env python3
"""Run 51: gate-guided refinements, and run 50's leads on fresh random photos (runs/51_gate_refine/PREREG.md).

Photos (items): the 100 hard photos of run 50 (copied from <W50>/r0) + 100 random photos of run 49's COCO train2017
development pool (run 49's permutation, seed 49), from position 1,000 on, with >= 1 primary item (area >= 1%; run 49's
rule). Run 49's rounds use the first ~70 positions and run 50 positions >= 2,000, so the halves never share a photo.
Every primary item. Each item carries half = "hard" | "random".

Passes (our pass code as run 50; SD 1.5 prompts T, S1, S2, S2n, B, F; SD 2.1 prompts T, S2; iSeg's own pass on SD 2.1).
Seeds: U- (no gate), UL (our MK4-LR gate), IL (iSeg's aggregation x the 64 x 64 gate map).
Refinements (all from the same pass):
  seed   none
  V0     ours (frozen): two head-wise steps of the 64 x 64 self-attention H, x <- H x, head mean
  ERS    iSeg's (COCO settings), as run 50
  A2/A5  gated propagation: 2 / 5 head-wise steps x <- H (g * x), g = the 64 x 64 MK4-LR gate map (floor 1e-3), so
         mass moves only out of locations the gate keeps; no row renormalisation; head mean
  C      region blend, then V0: spherical k-means (6 x 6 grid init, 20 iterations) of the gate's own image queries (all
         gate modules, each L2-normalised, upsampled to 64 x 64, concatenated); seed' = 0.5 seed + 0.5 region mean
  CA     region blend, then A2
Gate-mass statistics (T pass, per item): mean of the 64 x 64 gate map and its fraction >= 0.5 (size check, PREREG 6).

items    (CPU)            -> <W>/r0/items.json, photos.json, masks (hard half copied from run 50)
caption  (GPU)            BLIP-large unconditional captions (both halves) and prompt-B class lists (jc_iseg50.caption)
check    (GPU)            on 2 photos: gated propagation with g = 1 equals V0; region blend with one region is the seed;
                          our T|UL|V0 map equals run 50's saved map (hard photos, same machine)
capture  (GPU) --bb 15|21
score    (CPU)            -> <W>/r0/scores.json.gz (jc_iseg50.score_photo)
report   (CPU)            -> runs/51_gate_refine/result.md
viz      (CPU)            -> runs/51_gate_refine/viz/
"""
from __future__ import annotations

import argparse, gzip, json, math, os, random, shutil, sys, time
from collections import defaultdict
from functools import reduce
from operator import add
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_iseg50 as J  # noqa: E402

D, S, CACHE = J.D, J.S, J.CACHE
W50 = Path(os.environ.get("ISEG50_WORK", str(CACHE / "jc_iseg50")))
W = Path(os.environ.get("GATE51_WORK", str(CACHE / "jc_gate51")))
J.W = W; D.W = W                                       # J.rdir / J.score_photo / D.maps_of now read run 51's folders
OUT = ROOT / "runs/51_gate_refine"
N_RANDOM, START_RANDOM, MIN_AREA, SMALL = 100, 1000, 0.01, 0.05
PROMPTS = {"15": ("T", "S1", "S2", "S2n", "B", "F"), "21": ("T", "S2")}
SEEDS = ("U-", "UL", "IL")
REFS = ("seed", "V0", "ERS", "A2", "A5", "C", "CA")
REF = "15|T|UL|V0"
GFLOOR, KM_K, KM_IT, BLEND = 1e-3, 6, 20, 0.5
LEADS50 = ("15|S1|UL|ERS", "15|S2|UL|ERS", "15|S2|IL|ERS", "15|S2n|IL|V0", "15|B|IL|V0", "15|F|UL|V0", "15|F|IL|V0")
NEW = [(f"15|{p}|UL|{r}", f"15|{p}|UL|V0") for p in ("T", "F") for r in ("A2", "A5", "C", "CA")]
rdir = J.rdir


# ------------------------------------------------------------------ items
def items(a):
    import jc_ov42 as O
    C40 = O.C40
    hard = json.loads((W50 / "r0/items.json").read_text())
    md = W / "masks"; md.mkdir(parents=True, exist_ok=True)
    for x in hard:
        x["half"] = "hard"; shutil.copyfile(W50 / "masks" / f"{x['id']}.npz", md / f"{x['id']}.npz")
    hard_img = {x["img"] for x in hard}
    excl = {int(x["img"]) for x in json.loads((CACHE / "jc_ov42/items_prof.json").read_text())["items"]}
    excl |= {int(x["img"]) for x in json.loads((CACHE / "jc_real44/ours/items_prof44.json").read_text())["jobs"]}
    excl |= set(json.loads((D.SHIP / "fit40_photos.json").read_text()))
    d = json.loads((J.COCO / "annotations/instances_train2017.json").read_text())
    cat = {c["id"]: O.C2V.get(c["name"], c["name"]) for c in d["categories"]}; imgs = {i["id"]: i for i in d["images"]}
    by = defaultdict(list)
    for an in d["annotations"]:
        by[an["image_id"]].append((cat[an["category_id"]], an))
    pool = sorted(i for i in by if i not in excl); random.Random(D.PERM_SEED).shuffle(pool)       # run 49's order
    X, P, k = [], [], START_RANDOM
    while len(P) < N_RANDOM:
        i = pool[k]; k += 1; im = imgs[i]; h, w = im["height"], im["width"]; truth, crowd, inst = {}, {}, {}
        for nm, an in by[i]:
            m = C40.rasterize(an["segmentation"], h, w)
            if an["iscrowd"]:
                crowd[nm] = crowd.get(nm, np.zeros((h, w), bool)) | m
            else:
                truth[nm] = truth.get(nm, np.zeros((h, w), bool)) | m; inst.setdefault(nm, []).append(m)
        prim = sorted(c for c, t in truth.items() if t.mean() >= MIN_AREA)
        if not prim:
            continue
        assert i not in hard_img, f"random photo {i} is in run 50's set"
        allobj = np.zeros((h, w), bool)
        for t in truth.values():
            allobj |= t
        for c in prim:
            t = truth[c]; ign = crowd.get(c, np.zeros((h, w), bool)) & ~t; lab = np.zeros((h, w), np.int16)
            for j, m in enumerate(inst[c]):
                lab[m & (lab == 0)] = j + 1
            iid = f"d{i:012d}_{O.CLS80.index(c) + 1:02d}"
            np.savez_compressed(md / f"{iid}.npz", truth=t, ignore=ign, other=allobj & ~t & ~ign, inst=lab)
            X.append({"id": iid, "img": int(i), "path": f"train2017/{im['file_name']}", "cls": c, "area": float(t.mean()),
                      "ninst": len(inst[c]), "half": "random"})
        P.append({"path": f"train2017/{im['file_name']}", "img": int(i)})
    allX = hard + X; photos = sorted({x["img"]: x["path"] for x in allX}.items())
    rdir().mkdir(parents=True, exist_ok=True)
    (rdir() / "items.json").write_text(json.dumps(allX, indent=0))
    (rdir() / "photos.json").write_text(json.dumps([{"path": p, "img": i} for i, p in photos]))
    for hn, xs in (("hard", hard), ("random", X)):
        print(f"run 51 {hn}: {len({x['img'] for x in xs})} photos, {len(xs)} primary items "
              f"({sum(x['area'] < SMALL for x in xs)} small < 5%), {len({x['cls'] for x in xs})} classes", flush=True)
    print(f"random half: pool positions {START_RANDOM}-{k - 1}", flush=True)


def load_items(limit=0):
    X = json.loads((rdir() / "items.json").read_text())
    if limit:                                                    # the first `limit` photos of each half
        keep = set()
        for hn in ("hard", "random"):
            keep |= set(sorted({x["img"] for x in X if x["half"] == hn})[:limit])
        X = [x for x in X if x["img"] in keep]
    return X


J.load_items = load_items                                         # J.caption reads the run 51 items


# ------------------------------------------------------------------ new refinements
def gated_prop(H, g, v, steps):
    """x <- H (g * x), head-wise, `steps` times; H [8, N, N] row-stochastic, g [N] in [0, 1], v [N]; head mean."""
    from src.segmentation.sg25_sa import _apply
    gg = g.clamp(min=GFLOOR); x = _apply(H, v * gg)
    for _ in range(steps - 1):
        x = _apply(H, x * gg[None])
    return x.mean(0)


def regions(Q, side=64):
    """spherical k-means of the gate's image queries: {module: [n, d] L2-normalised} -> cluster id per 64 x 64 pixel."""
    import torch
    import torch.nn.functional as F
    fs = []
    for m in sorted(Q):
        q = Q[m].float(); s = math.isqrt(q.shape[0])
        q = F.interpolate(q.T.reshape(1, -1, s, s), size=(side, side), mode="bilinear", align_corners=False)[0]
        fs.append(F.normalize(q.reshape(q.shape[0], -1).T, dim=-1))
    f = F.normalize(torch.cat(fs, 1), dim=-1)                                    # [4096, sum d]
    g = torch.arange(KM_K, device=f.device) * (side // KM_K) + side // (2 * KM_K)
    c = f[(g[:, None] * side + g[None]).reshape(-1)].clone()                     # 6 x 6 grid init
    for _ in range(KM_IT):
        lab = (f @ c.T).argmax(1); keep = torch.unique(lab)
        c = F.normalize(torch.stack([f[lab == k].mean(0) for k in keep]), dim=-1)
    return (f @ c.T).argmax(1)


def region_blend(v, lab):
    """seed' = (1 - BLEND) * v + BLEND * (mean of v over v's region); v [N] min-maxed."""
    import torch
    n = int(lab.max()) + 1
    s = torch.zeros(n, device=v.device).index_add_(0, lab, v); c = torch.zeros(n, device=v.device).index_add_(0, lab, torch.ones_like(v))
    return (1 - BLEND) * v + BLEND * (s / c.clamp_min(1))[lab]


class R51(J.R50):
    def maps(self, x, sel, out, P):
        from src.segmentation.sg25_sa import _attention, _apply
        O, rec = self.O, self.rec
        key, _ = self.U.keyword("coco", x["cls"], self.Wm); tg = sel[x["cls"]]
        g64, Q, B = self.gate64(key); ga = None if not B else O.R37.powered(B["L"], Q, 1.0)
        H64 = _attention(rec.grid_qk[64], 64)
        cls = list(sel); full = {m: v[None] for m, v in rec.ca_all.items()}
        I = S.iseg_seed(full, "coco", [sel[c] for c in cls])[:, cls.index(x["cls"])].reshape(64, 64)
        u = self.our_seed(tg, None)
        seeds = {"U-": u, "UL": self.our_seed(tg, ga) if ga else u, "IL": I * g64 if g64 is not None else I}
        g = g64.reshape(-1).float() if g64 is not None else None
        lab = regions(Q)
        for nm, sd in seeds.items():
            v = J.mmt(sd).reshape(-1); k = f"{self.bb}|{P}|{nm}"
            out[f"{k}|seed"] = v.reshape(64, 64)
            out[f"{k}|V0"] = _apply(H64, _apply(H64, v)).mean(0).reshape(64, 64)
            out[f"{k}|ERS"] = S.ers(v.reshape(-1, 1), rec.pair_qk[S.ERS_SA], "coco")[:, 0].reshape(64, 64)
            vb = region_blend(v, lab)
            out[f"{k}|C"] = _apply(H64, _apply(H64, vb)).mean(0).reshape(64, 64)
            if g is not None:
                out[f"{k}|A2"] = gated_prop(H64, g, v, 2).reshape(64, 64)
                out[f"{k}|A5"] = gated_prop(H64, g, v, 5).reshape(64, 64)
                out[f"{k}|CA"] = gated_prop(H64, g, vb, 2).reshape(64, 64)
        if P == "T" and g is not None:
            self.gstat[x["id"]] = {"gmean": float(g.mean()), "g50": float((g >= 0.5).float().mean()), "nreg": int(lab.max()) + 1}
        return 0 if B else 1, ga

    def item(self, x, img, blist):
        from src.segmentation.attention_aggregation import find_token_subsequence
        tok = self.pipe.tokenizer; out = {}; miss = 0; prompts = {}
        _, word = self.U.keyword("coco", x["cls"], self.Wm)
        for P in PROMPTS[self.bb]:
            if P == "T":
                prompt = f"A photograph of a {word}"
                ids = tok(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
                sel = {x["cls"]: find_token_subsequence(ids, tok(word, add_special_tokens=False)["input_ids"])}
            else:
                cls = {"S1": [x["cls"]], "S2": [x["cls"]], "S2n": [x["cls"]], "B": blist,
                       "F": [x["cls"]] + [c for c in J.FIXED if c != x["cls"]][:4]}[P]
                prompt, sel = J.multi(cls, bg=P != "S1"); J.check_sel(tok, prompt, sel)
            enh = reduce(add, sel.values()) if P in ("S1", "S2", "B", "F") else []
            self.run_pass(img, prompt, sel[x["cls"]], enh)
            m, ga = self.maps(x, sel, out, P); miss += m; prompts[P] = prompt
            if P == "T":
                self.check_seed(sel[x["cls"]], ga)
            self.rec.clear()
        return out, prompts, miss


def runner(bb):
    rn = R51(bb); rn.gstat = {}; return rn


# ------------------------------------------------------------------ checks and capture
def check(a):
    """the new refinements reduce to known ones in their limits; the capture reproduces run 50 on its photos."""
    import torch
    from PIL import Image
    from src.segmentation.sg25_sa import _attention, _apply
    rn = runner(a.bb); X = load_items(1); BL = json.loads((rdir() / "blist.json").read_text())
    with torch.no_grad():
        for img_id, xs in D.by_photo(X):
            x = xs[0]; img = np.asarray(Image.open(J.COCO / x["path"]).convert("RGB"))
            out, _, _ = rn.item(x, img, BL[x["id"]]); H = _attention(rn.rec.grid_qk[64], 64)
            v = torch.rand(4096, device=H.device)
            d1 = float((gated_prop(H, torch.ones(4096, device=H.device), v, 2) - _apply(H, _apply(H, v)).mean(0)).abs().max())
            lab = torch.arange(4096, device=H.device); d2 = float((region_blend(v, lab) - v).abs().max())
            assert d1 < 1e-5 and d2 < 1e-6, (d1, d2)
            print(f"check {img_id} ({x['half']}): A with g = 1 vs V0 max |diff| {d1:.1e}; C with singleton regions vs seed {d2:.1e}; "
                  f"regions {rn.gstat.get(x['id'], {}).get('nreg')}", flush=True)
            f50 = W50 / "r0" / f"maps_{a.bb}" / f"{x['id']}.npz"
            if x["half"] == "hard" and f50.exists():
                z = np.load(f50); n50 = [str(n) for n in z["names"]]
                for nm in (f"{a.bb}|T|UL|V0", f"{a.bb}|S2|IL|ERS"):
                    d = float(np.abs(z["maps"][n50.index(nm)].astype(np.float32) - out[nm].float().cpu().numpy()).max())
                    assert d < 2e-3, (nm, d)
                    print(f"check {img_id}: {nm} vs run 50's saved map, max |diff| {d:.1e}", flush=True)
            rn.rec.clear()


def capture(a):
    import torch
    from PIL import Image
    X = load_items(a.limit); BL = json.loads((rdir() / "blist.json").read_text())
    rn = runner(a.bb); od = rdir() / f"maps_{a.bb}"; od.mkdir(parents=True, exist_ok=True)
    oi = rdir() / "maps_iseg"; oi.mkdir(parents=True, exist_ok=True); n = 0; t_all = time.perf_counter()
    with torch.no_grad():
        for img_id, xs in D.by_photo(X):
            if all((od / f"{x['id']}.npz").exists() for x in xs) and (a.bb != "21" or (oi / f"{img_id:012d}.npz").exists()):
                continue
            img = np.asarray(Image.open(J.COCO / xs[0]["path"]).convert("RGB")); t0 = time.perf_counter(); miss = 0
            for x in xs:
                o = od / f"{x['id']}.npz"
                if not o.exists():
                    mp, prompts, m = rn.item(x, img, BL[x["id"]]); miss += m
                    J.save(o, mp, {"prompts": json.dumps(prompts), "miss": m, "gstat": json.dumps(rn.gstat.pop(x["id"], {}))})
            if a.bb == "21":
                o = oi / f"{img_id:012d}.npz"
                if not o.exists():
                    mp, text, m = rn.own_iseg(xs, img); d = {}
                    for xid, mm in mp.items():
                        names = sorted(mm); d[f"m_{xid}"] = np.stack([mm[k].float().cpu().numpy() for k in names]).astype(np.float16)
                    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, names=np.array(names), text=text, miss=m, **d); os.replace(tmp, o)
            torch.cuda.empty_cache(); n += 1; dt = time.perf_counter() - t0
            npass = len(xs) * len(PROMPTS[a.bb]) + (a.bb == "21")
            print(f"[gate51:{a.bb}] photo {img_id} ({xs[0]['half']}): {len(xs)} items, {npass} passes, {dt:.1f}s "
                  f"({npass / dt:.2f} passes/s), miss {miss}", flush=True)
    print(f"[gate51:{a.bb}] done: {n} photos in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


def score(a):
    from multiprocessing import get_context
    X = load_items(a.limit); jobs = D.by_photo(X); recs = []
    with get_context("fork").Pool(a.workers) as p:
        for img_id, nx, nn, r in p.imap_unordered(J.score_photo, jobs):
            recs += r; print(f"[score51] photo {img_id}: {nx} items, {nn} maps", flush=True)
    half = {x["id"]: x["half"] for x in X}
    for r in recs:
        r["half"] = half[r["id"]]
    gs = {}
    for x in X:
        f = rdir() / "maps_15" / f"{x['id']}.npz"
        if f.exists():
            gs[x["id"]] = json.loads(str(np.load(f)["gstat"]))
    o = rdir() / f"scores{a.tag}.json.gz"
    with gzip.open(o, "wt") as f:
        json.dump({"recs": recs, "gstat": gs}, f)
    print(f"wrote {o}: {len(recs)} records", flush=True)


# ------------------------------------------------------------------ report
def delta_q(T, a, b, q):
    """T.delta with a (1 - 2q) interval: percentiles q and 1 - q (q in percent)."""
    (Ta, Ua, sa), (Tb, Ub, sb) = T.mats(*a), T.mats(*b); cs = (sa.sum(0) > 0) & (sb.sum(0) > 0)
    f = lambda Tm, Um: np.where((T.Wb @ Um)[:, cs] > 0, (T.Wb @ Tm)[:, cs] / np.maximum((T.Wb @ Um)[:, cs], 1), np.nan)
    pt = float((Ta.sum(0)[cs] / np.maximum(Ua.sum(0)[cs], 1)).mean() - (Tb.sum(0)[cs] / np.maximum(Ub.sum(0)[cs], 1)).mean())
    bs = np.nanmean(f(Ta, Ua), 1) - np.nanmean(f(Tb, Ub), 1); lo, hi = np.nanpercentile(bs, [q, 100 - q])
    return pt, float(lo), float(hi)


def report(a):
    from scipy.stats import spearmanr, rankdata
    with gzip.open(rdir() / f"scores{a.tag}.json.gz", "rt") as f:
        Z = json.load(f)
    recs = [dict(x, round=0) for x in Z["recs"]]; gs = Z["gstat"]
    sub = lambda f: [x for x in recs if f(x)]
    H = {hn: sub(lambda x, hn=hn: x["half"] == hn) for hn in ("random", "hard")}; H["pooled"] = recs
    T = {hn: D.Table(r, a.boot) for hn, r in H.items()}
    TS = {hn: D.Table([x for x in r if x["area"] < SMALL], a.boot) for hn, r in H.items()}
    TL = {hn: D.Table([x for x in r if x["area"] >= SMALL], a.boot) for hn, r in H.items()}
    have = lambda m, hn="random": bool(T[hn].by.get(m))
    f3 = lambda m, t, ro="t55": f"{t.miou(m, ro):.3f}" if t.by.get(m) else "–"
    fd = D.fmt_d
    nitem = {hn: len({x["id"] for x in r}) for hn, r in H.items()}
    nsm = {hn: len({x["id"] for x in r if x["area"] < SMALL}) for hn, r in H.items()}
    L = [f"# Run 51: gate-guided refinements, and run 50's leads on random photos ({'smoke, ' if a.tag else ''}`PREREG.md`)", "",
         f"COCO train2017 development photos. **Random half** (fresh, primary): {len(T['random'].photos)} photos, {nitem['random']} "
         f"items ({nsm['random']} small < 5%). **Hard half** (run 50's photos; run 50's leads were found on them): "
         f"{len(T['hard'].photos)} photos, {nitem['hard']} items ({nsm['hard']} small). Class-macro mIoU at 0.55 unless stated; "
         f"Δ [95% paired photo bootstrap, {a.boot}]; **+** / **−** = CI excludes 0.", "",
         "Refinements: V0 ours; ERS iSeg's; A2 / A5 gated propagation (2 / 5 steps); C region blend + V0; CA region blend + A2.", ""]
    # 1. references
    L += ["## 1. Reference cells (mIoU)", "", "| Cell | read-out | random | random small | random not small | hard | pooled |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for m, ro in ((REF, "t55"), ("15|F|UL|V0", "t55"), ("15|T|UL|ERS", "t55"), ("15|T|UL|A5", "t55"), ("21|T|UL|V0", "t55"),
                  ("21|S2|IL|ERS", "t55"), ("21|iseg|I-|ERS", "comp"), ("21|iseg|I-|ERS", "t55")):
        if have(m):
            L.append(f"| `{m}` | {ro} | {f3(m, T['random'], ro)} | {f3(m, TS['random'], ro)} | {f3(m, TL['random'], ro)} | "
                     f"{f3(m, T['hard'], ro)} | {f3(m, T['pooled'], ro)} |")
    # 2. confirmation of run 50's leads (registered)
    q7 = 2.5 / len(LEADS50)
    L += ["", f"## 2. Run 50's leads on the random half (registered; Δ vs `{REF}`)", "",
          f"Confirmed = Δ ≥ +0.02, lower end of the Bonferroni interval (7 leads: {100 - 2 * q7:.2f}%) > 0, and Δ on not-small "
          "items ≥ −0.01. The hard half repeats run 50 on the same photos.", "",
          "| Lead | random Δ (95%) | random Δ (Bonferroni) | random small | random not small | hard Δ | verdict |",
          "|---|---|---|---|---|---|---|"]
    for m in LEADS50:
        if not have(m):
            continue
        d = T["random"].delta((m, "t55"), (REF, "t55")); db = delta_q(T["random"], (m, "t55"), (REF, "t55"), q7)
        ds = TS["random"].delta((m, "t55"), (REF, "t55")); dl = TL["random"].delta((m, "t55"), (REF, "t55"))
        dh = T["hard"].delta((m, "t55"), (REF, "t55"))
        ok = d[0] >= 0.02 and db[1] > 0 and dl[0] >= -0.01
        L.append(f"| `{m}` | {fd(d)} | {fd(db)} | {fd(ds)} | {fd(dl)} | {fd(dh)} | {'**confirmed**' if ok else 'not confirmed'} |")
    # 3. new refinements (registered screen)
    L += ["", "## 3. New refinements vs V0 (registered screen; same prompt and seed)", "",
          "Lead = random-half Δ ≥ +0.02 with CI > 0 and not-small Δ ≥ −0.01 (then confirmed elsewhere before any claim).", "",
          "| Cell | vs | random Δ | random small | random not small | hard Δ | pooled Δ | lead |", "|---|---|---|---|---|---|---|---|"]
    for m, b in NEW:
        if not (have(m) and have(b)):
            continue
        d = T["random"].delta((m, "t55"), (b, "t55")); ds = TS["random"].delta((m, "t55"), (b, "t55"))
        dl = TL["random"].delta((m, "t55"), (b, "t55")); dh = T["hard"].delta((m, "t55"), (b, "t55"))
        dp = T["pooled"].delta((m, "t55"), (b, "t55")); ok = d[0] >= 0.02 and d[1] > 0 and dl[0] >= -0.01
        L.append(f"| `{m}` | `{b}` | {fd(d)} | {fd(ds)} | {fd(dl)} | {fd(dh)} | {fd(dp)} | {'**lead**' if ok else ''} |")
    # 4. refinement factor table (descriptive): each refinement vs V0, averaged over prompts and seeds, per backbone
    L += ["", "## 4. Each refinement vs V0, mean Δ over prompts and seeds (descriptive)", "",
          "| bb | refinement | random | random small | random not small | hard | cells |", "|---|---|---:|---:|---:|---:|---:|"]
    for bb in ("15", "21"):
        for rf in ("seed", "ERS", "A2", "A5", "C", "CA"):
            vals = defaultdict(list)
            for P in PROMPTS[bb]:
                for sd in SEEDS:
                    m, b = f"{bb}|{P}|{sd}|{rf}", f"{bb}|{P}|{sd}|V0"
                    for nm, t in (("r", T["random"]), ("rs", TS["random"]), ("rl", TL["random"]), ("h", T["hard"])):
                        if t.by.get(m) and t.by.get(b):
                            vals[nm].append(t.miou(m, "t55") - t.miou(b, "t55"))
            if vals["r"]:
                L.append(f"| {bb} | {rf} | {np.mean(vals['r']):+.3f} | {np.mean(vals['rs']):+.3f} | {np.mean(vals['rl']):+.3f} | "
                         f"{np.mean(vals['h']):+.3f} | {len(vals['r'])} |")
    # 5. label-free iSeg-style vs iSeg on random photos (descriptive)
    L += ["", "## 5. Label-free cells vs iSeg's own pass on the random half (descriptive)", "",
          "| Cell | vs iSeg own (comp) | vs iSeg own (t55) | vs ours |", "|---|---|---|---|"]
    for m in ("21|S2|IL|ERS", "21|S2|IL|A5", "21|S2|UL|A5", "15|F|UL|V0", "15|T|UL|A5", "15|F|UL|A5", REF):
        if have(m) and have("21|iseg|I-|ERS"):
            L.append(f"| `{m}` | {fd(T['random'].delta((m, 't55'), ('21|iseg|I-|ERS', 'comp')))} | "
                     f"{fd(T['random'].delta((m, 't55'), ('21|iseg|I-|ERS', 't55')))} | {fd(T['random'].delta((m, 't55'), (REF, 't55')))} |")
    # 6. size check with the gate's own mass (registered rule of runs/51_prep/SIZE_CHECK.md)
    L += ["", "## 6. Size check with the gate's mass (T pass; rule: ρ ≥ 0.6 and AUROC small ≥ 0.80, all items)", "",
          "| Items | n (small) | gate mean ρ / AUROC | gate ≥ 0.5 ρ / AUROC |", "|---|---|---|---|"]
    area = {x["id"]: x["area"] for x in recs}
    iou = {x["id"]: x["tp"][D.I55] / max(x["tp"][D.I55] + x["fp"][D.I55] + x["fn"][D.I55], 1) for x in recs if x["map"] == REF}
    def auc(y, s):
        r = rankdata(s); y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
        return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
    verdict = None
    hv = {x["id"] for x in H["random"]}
    for nm, keep in (("all", lambda i: True), ("random half", lambda i: i in hv),
                     ("map on object (IoU > 0.3)", lambda i: iou.get(i, 0) > 0.3), ("map off object", lambda i: iou.get(i, 0) <= 0.3)):
        ids = [i for i in gs if gs[i] and i in area and keep(i)]
        if len(ids) < 10:
            continue
        A = np.array([area[i] for i in ids]); sm = A < SMALL; row = f"| {nm} | {len(ids)} ({int(sm.sum())}) |"
        best = []
        for k in ("gmean", "g50"):
            s = np.array([gs[i][k] for i in ids]); rho = spearmanr(s, A)[0]; au = auc(sm, -s); best.append((rho, au))
            row += f" {rho:+.3f} / {au:.3f} |"
        L.append(row)
        if nm == "all":
            verdict = any(r >= 0.6 and u >= 0.80 for r, u in best)
    if verdict is not None:
        L += ["", f"Size check: **{'passes' if verdict else 'fails'}** (candidate B {'may go to a later run' if verdict else 'stays dropped'})."]
    # 7. error measures
    L += ["", "## 7. What goes wrong (random half, t55; item means; columns as run 49)", "",
          "| Cell | n | IoU | oracle | AUROC | point | mass | prec | recall | FP-obj | inst | IoU-nb | small | mid | large |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in (REF, "15|T|UL|ERS", "15|T|UL|A2", "15|T|UL|A5", "15|T|UL|C", "15|T|UL|CA", "15|F|UL|V0", "15|F|UL|A5", "21|S2|IL|ERS", "21|iseg|I-|ERS"):
        dr = D.diag_row(H["random"], m, "55")
        if dr:
            L.append(f"| `{m}` | {dr['n']} | {dr['iou']:.3f} | {dr['orc']:.3f} | {dr['auroc']:.3f} | {dr['point']:.2f} | {dr['mass']:.2f} | "
                     f"{dr['prec']:.2f} | {dr['rec']:.2f} | {dr['fpo']:.2f} | {dr['inst']:.2f} | {dr['biou']:.3f} | {dr['small']:.3f} | "
                     f"{dr['mid']:.3f} | {dr['large']:.3f} |")
    # 8. every cell
    L += ["", "## 8. Every cell (mIoU at t55; random / random small / random not small / hard / pooled)", "",
          "| Cell | random | small | not small | hard | pooled |", "|---|---:|---:|---:|---:|---:|"]
    for m in sorted(T["random"].by, key=lambda m: -T["random"].miou(m, "t55")):
        L.append(f"| `{m}` | {f3(m, T['random'])} | {f3(m, TS['random'])} | {f3(m, TL['random'])} | {f3(m, T['hard'])} | {f3(m, T['pooled'])} |")
    # panels: random-half items where A5 changes ours most
    by = defaultdict(dict)
    for x in H["random"]:
        by[x["id"]][x["map"]] = x
    io = lambda x: x["tp"][D.I55] / max(x["tp"][D.I55] + x["fp"][D.I55] + x["fn"][D.I55], 1)
    G = sorted(((io(d["15|T|UL|A5"]) - io(d[REF]), k) for k, d in by.items() if REF in d and "15|T|UL|A5" in d))
    pick = [k for _, k in G[-5:][::-1]] + [k for _, k in G[:5]]
    OUT.mkdir(parents=True, exist_ok=True); o = OUT / f"result{a.tag}.md"; o.write_text("\n".join(L) + "\n")
    (OUT / f"panels{a.tag}.json").write_text(json.dumps(pick)); print(f"wrote {o}", flush=True)


def viz(a):
    """photo | truth | ours (V0) | A5 | CA | ERS | F prompt V0 | iSeg own (comp): the 5 random-half items A5 helps most, 5 it hurts most."""
    from PIL import Image, ImageDraw
    G = json.loads((OUT / f"panels{a.tag}.json").read_text()); X = {x["id"]: x for x in load_items()}
    vd = OUT / "viz"; vd.mkdir(parents=True, exist_ok=True)
    for iid in G:
        x = X[iid]; xs = [y for y in X.values() if y["img"] == x["img"]]; M = D.maps_of(0, xs)
        img = Image.open(J.COCO / x["path"]).convert("RGB"); hw = (img.height, img.width)
        t = np.load(W / "masks" / f"{iid}.npz")["truth"]
        def mask(m, ro="t55"):
            if m not in M[iid]:
                return None
            if ro == "comp":
                cams = np.stack([D.mm(D.up(D.mm(M[y["id"]][m]), hw)) for y in xs]); j = [y["id"] for y in xs].index(iid)
                return S.cam_to_label(cams, 0.45) == j + 1
            return D.mm(D.up(D.mm(M[iid][m]), hw)) >= 0.55
        panels = [("photo", None), (f"truth: {x['cls']}", t), ("ours V0", mask(REF)), ("A5", mask("15|T|UL|A5")),
                  ("CA", mask("15|T|UL|CA")), ("ERS", mask("15|T|UL|ERS")), ("F prompt V0", mask("15|F|UL|V0")),
                  ("iSeg own", mask("21|iseg|I-|ERS", "comp"))]
        h = 224; w = round(img.width * h / img.height); canvas = Image.new("RGB", (w * len(panels), h + 18), "white")
        base = np.asarray(img.resize((w, h))).astype(np.float32)
        for k, (nm, m) in enumerate(panels):
            p = base.copy()
            if m is not None:
                mr = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h))) > 127
                p[mr] = 0.45 * p[mr] + 0.55 * np.array([255, 40, 40])
            canvas.paste(Image.fromarray(p.astype(np.uint8)), (k * w, 18)); ImageDraw.Draw(canvas).text((k * w + 3, 3), nm[:30], fill="black")
        o = vd / f"{iid}_{x['cls'].replace(' ', '_')}{a.tag}.jpg"; canvas.save(o, quality=80); print(f"wrote {o}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("items")
    q = sp.add_parser("caption"); q.add_argument("--limit", type=int, default=0)
    q = sp.add_parser("check"); q.add_argument("--bb", choices=("15", "21"), default="15")
    q = sp.add_parser("capture"); q.add_argument("--bb", choices=("15", "21"), required=True); q.add_argument("--limit", type=int, default=0)
    q = sp.add_parser("score"); q.add_argument("--limit", type=int, default=0); q.add_argument("--workers", type=int, default=16)
    q.add_argument("--tag", default="")
    q = sp.add_parser("report"); q.add_argument("--boot", type=int, default=2000); q.add_argument("--tag", default="")
    q = sp.add_parser("viz"); q.add_argument("--tag", default="")
    a = p.parse_args()
    {"items": items, "caption": J.caption, "check": check, "capture": capture, "score": score, "report": report, "viz": viz}[a.cmd](a)
