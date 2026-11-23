#!/usr/bin/env python3
"""runs/40_coco_confirm/PREREG.md: MK4-LR vs MK4 vs no gate on untouched real COCO val2017 photos (frozen method).

items    (CPU) val2017 items (minus the Section 136 set) and COCO train threshold-fit items; masks  -> <WORK>/items_{split}.json, masks/
capture  (GPU) one SD1.5 pass at t = 100; seeds {u,w} x ({-} + {MK4, LR, RW} x gamma 1) + W-on run 37 selected gamma -> <WORK>/cap_{split}/
score    (CPU) raw counts; val: DenseCRF on V0 / V3 at the train threshold and 0.55
select   (CPU) per map: train threshold                                                       -> <WORK>/thr.json
report   (CPU)                                                                                 -> runs/40_coco_confirm/result.md
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_realvoc37 as R37  # noqa: E402  (run 37: MK4 gate, RW weights, powered, selected gamma)
import jc_gate36 as G36  # noqa: E402

R, M, T, cc, st = R37.R, R37.M, R37.T, R37.cc, R37.st
from brace_paths import CACHE  # noqa: E402

COCO = ROOT / "datasets/coco2017"
WORK = CACHE / "jc_coco40"
OUT = ROOT / "runs/40_coco_confirm"
CLASSES, THR, T55, DEV, GRID, K1 = R.CLASSES, R37.THR, R37.T55, R.DEV, R37.GRID, R37.K1
C2V = {"airplane": "aeroplane", "motorcycle": "motorbike", "couch": "sofa"}
VOCN = set(CLASSES)
MIN_AREA = 0.01
N_FIT = 60
GN = {"-": "none", "M": "MK4", "L": "MK4-LR", "R": "MK4-RW"}
PROPS = ("seed", "V0", "V3")
GAM37 = json.loads((R37.WORK / "thr.json").read_text())["gamma"]          # run 37, VOC train selected (index into GRID)
SEL = {(g, p): GAM37[f"wC{g}|{p}"] for g in "MLR" for p in PROPS}
SEEDS = tuple(dict.fromkeys(("u-", "w-") + tuple(f"{w}{g}{K1}" for w in "uw" for g in "MLR") +
                            tuple(f"w{g}{SEL[(g, p)]}" for g in "MLR" for p in PROPS)))
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in PROPS)


# ---------------------------------------------------------------- items and masks
def rasterize(seg, h, w):
    if isinstance(seg, list):
        from PIL import Image, ImageDraw
        im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
        for poly in seg:
            if len(poly) >= 6:
                d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
        return np.asarray(im).astype(bool)
    cnt = seg["counts"]
    if isinstance(cnt, str):
        raise ValueError("compressed RLE not supported")
    flat = np.zeros(h * w, bool); pos = 0; val = False
    for c in cnt:
        flat[pos:pos + c] = val; pos += c; val = not val
    return flat.reshape(w, h).T


def build(split, ann, keep_img, per_class=None, seed=0, max_items=0):
    d = json.loads((COCO / f"annotations/instances_{ann}.json").read_text())
    cat = {c["id"]: C2V.get(c["name"], c["name"]) for c in d["categories"]}; imgs = {i["id"]: i for i in d["images"]}
    by = defaultdict(list)
    for an in d["annotations"]:
        nm = cat[an["category_id"]]
        if nm in VOCN:
            by[an["image_id"]].append((nm, an))
    ids = sorted(i for i in by if keep_img(i))
    if per_class:
        random.Random(seed).shuffle(ids)
    out, count = [], defaultdict(int); md = WORK / "masks"; md.mkdir(parents=True, exist_ok=True)
    for i in ids:
        im = imgs[i]; h, w = im["height"], im["width"]; anns = by[i]
        present = sorted({nm for nm, an in anns if not an["iscrowd"]})
        if per_class:
            present = [c for c in present if count[c] < per_class]
            if not present:
                continue
        masks = {}
        for nm, an in anns:
            k = "crowd" if an["iscrowd"] else "obj"
            masks.setdefault((nm, k), np.zeros((h, w), bool))
            masks[(nm, k)] |= rasterize(an["segmentation"], h, w)
        allobj = np.zeros((h, w), bool)
        for (nm, k), m in masks.items():
            if k == "obj":
                allobj |= m
        for c in present:
            truth = masks[(c, "obj")]; area = float(truth.mean())
            if truth.sum() == 0 or (per_class and area < MIN_AREA):
                continue
            ci = CLASSES.index(c) + 1; iid = f"{split}{i:012d}_{ci:02d}"
            ign = masks.get((c, "crowd"), np.zeros((h, w), bool)) & ~truth
            np.savez_compressed(md / f"{iid}.npz", truth=truth, ignore=ign, other=allobj & ~truth & ~ign)
            out.append({"id": iid, "path": f"{ann}/{im['file_name']}", "img": int(i), "cls": c, "ci": ci, "area": area, "split": split})
            count[c] += 1
        if max_items and len(out) >= max_items:
            break
    return out


def items(a):
    s136 = {int(p.stem) for p in (COCO / "sg_real_data_labels").glob("*.png")}
    prof = {int(r["path"].split("/")[1].split(".")[0]) for r in json.loads((CACHE / "jc_realvoc_t" / "items_C.json").read_text())}
    assert len(s136) == 200 and len(prof) > 0, (len(s136), len(prof))
    val = build("val", "val2017", lambda i: i not in s136, max_items=a.max_items)
    tr = build("train", "train2017", lambda i: i not in prof, per_class=N_FIT, max_items=a.max_items)
    assert not ({x["img"] for x in val} & s136) and not ({x["img"] for x in tr} & prof)
    for nm, x in (("val", val), ("train", tr)):
        (WORK / f"items_{nm}{a.tag}.json").write_text(json.dumps(x))
        cnt = defaultdict(int)
        for r in x:
            cnt[r["cls"]] += 1
        print(f"{nm}: {len(x)} items, {len({r['img'] for r in x})} images, primary (area >= {MIN_AREA}) "
              f"{sum(r['area'] >= MIN_AREA for r in x)}; per class {dict(cnt)}", flush=True)


def load_items(split, a):
    X = json.loads((WORK / f"items_{split}{a.tag}.json").read_text())[a.shard::a.num_shards]
    if a.max_items:
        X = X[:: max(1, len(X) // a.max_items)][: a.max_items]
    return X


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention
    pipe, rec = R.load(); Wm = R.word_map()
    RW, _, _ = R37.rw_weights("")
    if set(RW["C"]["all"]) != set(rec.ca_order):
        raise ValueError("RW module names differ from the recorder")
    P = M.to_dev(torch.load(R37.FIT33, weights_only=False)["fit"]["C_all"]); Wrw = RW["C"]["all"]
    out = WORK / f"cap_{a.split}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=DEV); checked = False
    for it in load_items(a.split, a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]
        R.one_pass(pipe, rec, np.asarray(Image.open(COCO / it["path"]).convert("RGB")), f"A photograph of a {Wm[c]}", Wm[c])
        w_a3 = R.layer_weights(rec); Q = R.qbar(rec)
        base = {m: g for m, q in Q.items() if (g := R37.g_mk4(P, c, m, q)) is not None}
        res = {m: int(round(q.shape[0] ** 0.5)) for m, q in Q.items()}
        B = {"M": base, "L": G36.consensus(base, {m: (1.0 if res[m] < 64 else 0.0) for m in base}), "R": G36.consensus(base, Wrw)}
        gmean = {g: float(torch.stack([x.mean() for x in B[g].values()]).mean()) if B[g] else 1.0 for g in B}
        H = {g: _attention(rec.grid_qk[g], g) for g in R.SA}; maps = {}
        for nm in SEEDS:
            gd = None if nm[1] == "-" else R37.powered(B[nm[1]], Q, GRID[int(nm[2:])])
            s = R.make_seed(rec, w_a3 if nm[0] == "w" else w_u, gd); maps[f"{nm}|seed"] = s
            pr = R.propagate(H, s, v3=True); maps[f"{nm}|V0"] = pr["V0"]; maps[f"{nm}|V3"] = pr["V3"]
        if a.tag and not checked:   # smoke: identity with aggregate
            from src.segmentation import sg_combined_v2
            ref, _ = sg_combined_v2.aggregate({g: [rec.grid_qk[g]] * 11 for g in R.SA}, maps["w-|seed"])
            worst = max(float(np.abs(st.mm(maps[f"w-|{k}"].cpu().numpy()) - st.mm(ref[k].float().cpu().numpy())).max()) for k in ("V0", "V3"))
            print(f"identity vs sg_combined_v2.aggregate (V0, V3): max |d| {worst:.2e}; seeds {SEEDS}", flush=True)
            assert worst < 1e-5; checked = True
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {it['id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), cls=c, img=it["img"], path=it["path"], area=it["area"],
                 miss=len(Q) - len(base), **{f"gmean_{k}": v for k, v in gmean.items()})
        os.replace(tmp, o)
        print(f"[c40:{a.split}] {it['id']} {c} miss {len(Q) - len(base)} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- score / select
def score_one(job):
    f, o, thr = Path(job[0]), Path(job[1]), job[2]
    if o.exists():
        return
    torch.set_num_threads(1)
    from PIL import Image
    z = np.load(f); mk = np.load(WORK / "masks" / f"{f.stem}.npz")
    truth, ign, other = mk["truth"], mk["ignore"], mk["other"]; valid = ~ign
    res = {"cls": str(z["cls"]), "img": int(z["img"]), "area": float(z["area"])}
    res.update({k: z[k] for k in z.files if k.startswith("gmean_")})
    rgb = None
    for i, nm in enumerate(str(x) for x in z["names"]):
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if thr is not None and not nm.endswith("|seed"):
            from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
            _import_densecrf(st.CRF_SITE)
            if rgb is None:
                rgb = np.asarray(Image.open(COCO / str(z["path"])).convert("RGB"), np.uint8)
            ms = np.stack([apply_ovam_densecrf(rgb, x >= t, st.CRF_SITE) for t in (thr[nm], float(THR[T55]))]).astype(bool)
            res[f"{nm}|crf"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    thr = json.loads((WORK / f"thr{a.tag}.json").read_text())["thr"] if a.split == "val" else None
    sc = WORK / f"cap_{a.split}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(sc / f.name), thr) for f in sorted((WORK / f"cap_{a.split}{a.tag}").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
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
         "grp": np.array([str(int(z["img"])) for z in Z]), "area": np.array([float(z["area"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k or k.startswith("gmean_"):
            S[k] = np.stack([z[k] for z in Z])
    return S


def select(a):
    S = load_scores("train", a.tag); one = np.ones(len(S["cls"]), bool); thr, tr = {}, {}
    for nm in MAPS:
        cv = cc.miou(S[f"{nm}|raw"], S["cls"], one); i = int(np.argmax(cv)); thr[nm] = float(THR[i]); tr[nm] = float(cv[i])
    (WORK / f"thr{a.tag}.json").write_text(json.dumps({"thr": thr, "train_miou": tr, "items": int(len(S["cls"]))}, indent=1))
    print(json.dumps(thr), flush=True)


# ---------------------------------------------------------------- report
def report(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr = Tr["thr"]
    Sall = load_scores("val", a.tag); L = []; w = L.append
    sub = lambda S, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}
    ti = {nm: int(np.flatnonzero(np.isclose(THR, thr[nm]))[0]) for nm in MAPS}
    HN = {"V0": "fine-only (V0)", "V3": "capped hierarchy (V3)", "seed": "seed only"}
    RULES = {"sel_crf": "+ CRF, train thr (primary)", "055_crf": "+ CRF, 0.55", "sel_raw": "raw, train thr", "055_raw": "raw, 0.55"}

    def nm(W, g, p, sel=False):
        if g == "-":
            return f"{'w' if W else 'u'}-|{p}"
        return f"w{g}{SEL[(g, p)]}|{p}" if sel else f"{'w' if W else 'u'}{g}{K1}|{p}"

    def P(S, n, rule):
        if rule == "sel_raw":
            return S[f"{n}|raw"][:, ti[n]]
        if rule == "055_raw":
            return S[f"{n}|raw"][:, T55]
        if isinstance(rule, int):
            return S[f"{n}|raw"][:, rule]
        return S[f"{n}|crf"][:, 0 if rule == "sel_crf" else 1]
    rl = lambda p: "sel_raw" if p == "seed" else "sel_crf"
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    rd = lambda x: "**helps**" if x[1] > 0 else "**hurts**" if x[2] < 0 else "not resolved"

    class E:
        def __init__(self, S):
            self.S, self.B, self.c = S, R.Boot(S, a.boot), {}

        def m(self, n, rule="sel_crf"):
            if (n, rule) not in self.c:
                self.c[(n, rule)] = self.B.stat(P(self.S, n, rule))
            return self.c[(n, rule)]

        def d(self, x, y, rule="sel_crf"):
            A, B_ = self.m(x, rule), self.m(y, rule); dd = A[1] - B_[1]
            return float(A[0] - B_[0]), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
    prim = Sall["area"] >= MIN_AREA; Sp = sub(Sall, prim); E1 = E(Sp)
    Ex = E(sub(Sp, Sp["cls"] != "person")); Ea = E(Sall)
    w("# Run 40: low-resolution consensus gate on untouched real COCO val2017 photos (`PREREG.md`)\n")
    w(f"**Confirmatory in intent; method frozen before capture.** COCO val2017 minus the Section 136 set: {len(Sall['cls'])} items "
      f"({len(np.unique(Sall['grp']))} photos); primary items (class area ≥ {MIN_AREA:.0%}): {int(prim.sum())} "
      f"({len(np.unique(Sp['grp']))} photos). Thresholds fitted on {Tr['items']} COCO train2017 items. Profiles: run 33 source C "
      f"(490 COCO train photos, captions). Gate strength γ = 1 (primary). Class-macro mIoU; 95% photo-bootstrap intervals "
      f"({a.boot} samples, paired).\n")
    w("\n## 1. Primary contrasts (primary items, SA weights on, γ = 1, + DenseCRF at the train threshold)\n")
    w("| # | Contrast | Map | Δ mIoU [95% CI] | Reading |"); w("|---|---|---|---|---|"); pc = {}
    for k, (g, ref, p) in enumerate((("L", "M", "V3"), ("L", "M", "V0"), ("L", "-", "V3"), ("L", "-", "V0")), 1):
        x = E1.d(nm(True, g, p), nm(True, ref, p)); pc[k] = x
        w(f"| C{k} | MK4-LR − {GN[ref]} | {HN[p]} | {f3(x)} | {rd(x)} |")
    conf = lambda i, j: (pc[i][1] > 0 or pc[j][1] > 0) and not (pc[i][2] < 0 or pc[j][2] < 0)
    w(f"\n- MK4-LR over MK4: {'**confirmed**' if conf(1, 2) else '**not confirmed**'} (C1 or C2 helps, neither hurts).")
    w(f"- MK4-LR over no gate: {'**confirmed**' if conf(3, 4) else '**not confirmed**'} (C3 or C4 helps, neither hurts).")

    w("\n## 2. mIoU, primary items (seed only: raw, train threshold; V0 / V3: + DenseCRF, train threshold)\n")
    w("| SA weights | γ | Map | none | MK4 | MK4-LR | MK4-RW |"); w("|:-:|---|---|---:|---:|---:|---:|")
    for W in (False, True):
        for p in ("seed", "V0", "V3"):
            w(f"| {'✓' if W else ''} | 1 | {HN[p]} | " + " | ".join(f"{E1.m(nm(W, g, p), rl(p))[0]:.3f}" for g in "-MLR") + " |")
    for p in ("seed", "V0", "V3"):
        w(f"| ✓ | run 37 selected ({', '.join(f'{GN[g]} {GRID[SEL[(g, p)]]}' for g in 'MLR')}) | {HN[p]} | {E1.m(nm(True, '-', p), rl(p))[0]:.3f} | " +
          " | ".join(f"{E1.m(nm(True, g, p, True), rl(p))[0]:.3f}" for g in "MLR") + " |")

    w("\n## 3. Paired differences, primary items\n")
    w("| SA weights | γ | Map | MK4 − none | MK4-LR − MK4 | MK4-RW − MK4 | MK4-LR − none |"); w("|:-:|---|---|---|---|---|---|")
    for W, sel in ((False, False), (True, False), (True, True)):
        for p in ("seed", "V0", "V3"):
            n_ = lambda g: nm(W, g, p, sel and g != "-")
            w(f"| {'✓' if W else ''} | {'selected' if sel else '1'} | {HN[p]} | {f3(E1.d(n_('M'), n_('-'), rl(p)))} | "
              f"{f3(E1.d(n_('L'), n_('M'), rl(p)))} | {f3(E1.d(n_('R'), n_('M'), rl(p)))} | {f3(E1.d(n_('L'), n_('-'), rl(p)))} |")
    w("\nHierarchy (V3 − V0) and SA weights (on − off) with the gate fixed, γ = 1, primary rule:\n")
    w("| Gate | hierarchy, SA weights on | hierarchy, uniform | SA weights, V3 | SA weights, V0 |"); w("|---|---|---|---|---|")
    for g in "-MLR":
        w(f"| {GN[g]} | {f3(E1.d(nm(True, g, 'V3'), nm(True, g, 'V0')))} | {f3(E1.d(nm(False, g, 'V3'), nm(False, g, 'V0')))} | "
          f"{f3(E1.d(nm(True, g, 'V3'), nm(False, g, 'V3')))} | {f3(E1.d(nm(True, g, 'V0'), nm(False, g, 'V0')))} |")

    w("\n## 4. Person excluded, and all items (SA weights on, γ = 1, + DenseCRF)\n")
    w("| Contrast | Map | person excluded | all items (incl. < 1%) |"); w("|---|---|---|---|")
    for g, ref in (("L", "M"), ("L", "-"), ("M", "-"), ("R", "M")):
        for p in ("V3", "V0"):
            w(f"| {GN[g]} − {GN[ref]} | {HN[p]} | {f3(Ex.d(nm(True, g, p), nm(True, ref, p)))} | {f3(Ea.d(nm(True, g, p), nm(True, ref, p)))} |")

    w("\n## 5. Other threshold rules (primary items, SA weights on, γ = 1)\n")
    w("| Map | Gate | " + " | ".join(RULES.values()) + " |"); w("|---|---|" + "---:|" * len(RULES))
    for p in ("V0", "V3"):
        for g in "-MLR":
            w(f"| {HN[p]} | {GN[g]} | " + " | ".join(f"{E1.m(nm(True, g, p), r_)[0]:.3f}" for r_ in RULES) + " |")
    w("\nRaw maps, difference at every threshold (primary items, SA weights on, γ = 1; † = 95% CI excludes 0):\n")
    w("| Contrast | Map | " + " | ".join(f"{t:.3f}" for t in THR) + " |"); w("|---|---|" + "---:|" * len(THR))
    for g, ref in (("L", "M"), ("L", "-")):
        for p in ("seed", "V0", "V3"):
            row = []
            for i in range(len(THR)):
                x = E1.d(nm(True, g, p), nm(True, ref, p), i); row.append(f"{x[0]:+.3f}{'†' if x[1] > 0 or x[2] < 0 else ''}")
            w(f"| {GN[g]} − {GN[ref]} | {HN[p]} | " + " | ".join(row) + " |")

    w("\n## 6. Per class IoU (primary items, SA weights on, γ = 1, + DenseCRF)\n")
    hdr = [(nm(True, g, p), f"{GN[g]} {p}") for p in ("V3", "V0") for g in "-MLR"]
    w("| Class | items | " + " | ".join(h for _, h in hdr) + " |"); w("|---|---:|" + "---:|" * len(hdr))
    better = {p: [0, 0] for p in ("V3", "V0")}
    for c in CLASSES:
        sel = Sp["cls"] == c
        if not sel.any():
            continue
        v = {n_: cc.miou(P(Sp, n_, "sel_crf")[:, None], Sp["cls"], sel)[0] for n_, _ in hdr}
        for p in ("V3", "V0"):
            dlt = v[nm(True, "L", p)] - v[nm(True, "M", p)]; better[p][0] += dlt > 0; better[p][1] += dlt < 0
        w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{v[n_]:.3f}" for n_, _ in hdr) + " |")
    w("\nClasses where MK4-LR beats / trails MK4: " + "; ".join(f"{p}: {b[0]} / {b[1]}" for p, b in better.items()) + ".")
    w("\nMean gate value at γ = 1 (primary items): " + ", ".join(f"{GN[g]} {float(np.mean(Sp[f'gmean_{g}'])):.3f}" for g in "MLR") + ".")
    w("\nBRACE stays frozen; any change to the paper's method is a separate owner decision.\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub_ = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("items", "capture", "score", "select", "report"):
        p = sub_.add_parser(cmd)
        p.add_argument("--split", choices=("train", "val"), default="val")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"items": items, "capture": capture, "score": score, "select": select, "report": report}[a.cmd](a)
