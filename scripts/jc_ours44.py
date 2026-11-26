#!/usr/bin/env python3
"""Run 44: our method (paper2 pipeline) on PASCAL VOC val, PASCAL Context val and COCO-Object (env trdp).

Method (frozen, as Table 2 / runs 42-43): one SD 1.5 pass per item at t = 100 (jc_realvoc.one_pass), prompt
"A photograph of a {word}"; uniform CA seed; MK4 gate (object profile + 4 negative centroids) -> MK4-LR (mean of the
gates below 64 x 64, jc_gate36.consensus) at gamma = 1; fine-only propagation (V0); min-max; fixed 0.55 (raw primary,
DenseCRF R2 secondary).

Profiles (captions of COCO train2017 photos only; no labels or masks):
  * the 80 COCO words: run 42's size-20 profiles (jc_ov42 fit.pt, "20"), unchanged;
  * PASCAL Context words that are not COCO words (33: 32 stuff/object words + "table"): the same recipe as run 42
    (20 caption-matched positive photos per word; 480 negative photos whose captions do not match the word, drawn from
    run 42's profile pool plus the new positives; object profile = mean of the top 5% queries by the word's attention;
    4 spherical k-means centroids of the top 1% queries on the negatives), with the caption rules SYN44 below.
items44   (CPU) new-word profile photos and dump jobs                      -> <WORK>/ours/items_prof44.json
dump44    (GPU) per job: one pass with the photo's own word; o_ for its own new word, n_ for its assigned new words
fit44     (CPU) -> <WORK>/ours/fit44.pt  {"mo": {(word, module): f}, "neg": {(word, module): {4: C}}}
capture   (GPU) --set voc|context|coco: maps u-, uM, uL as seed and V0     -> <WORK>/ours/cap_<set>/<item id>.npz
"""
from __future__ import annotations

import argparse, json, math, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_ov42 as O  # noqa: E402
from jc_real44 import VOC20, CTX59, WORK as W44  # noqa: E402

R, M, T, R37, C40, G36 = O.R, O.M, O.T, O.R37, O.C40, O.G36
COCO, CLS80, DEV = O.COCO, O.CLS80, O.DEV
WORK = W44 / "ours"
# Caption rules for the PASCAL Context words that are not COCO words (fixed before any dump or capture; regex
# alternatives, then an optional s / es, word boundaries, lower-case captions, as run 42).
SYN44 = {
    "bag": ["bag", "backpack", "handbag", "purse"], "building": ["building", "skyscraper"], "cabinet": ["cabinet", "cupboard"],
    "ceiling": ["ceiling"], "cloth": ["cloth", "towel", "tablecloth"], "computer": ["computer", "laptop"], "door": ["door"],
    "fence": ["fence"], "floor": ["floor"], "flower": ["flower"], "food": ["food", "meal"], "grass": ["grass", "lawn"],
    "ground": ["ground", "dirt"], "light": ["light", "lamp"], "mountain": ["mountain", "hill"], "curtain": ["curtain", "drape"],
    "platform": ["platform"], "sign": ["sign"], "plate": ["plate"], "road": ["road", "street", "highway"],
    "rock": ["rock", "boulder", "stone"], "shelves": ["shelf", "shelves", "bookshelf", "bookshelves"],
    "sidewalk": ["sidewalk", "pavement"], "sky": ["sky"], "snow": ["snow"],
    "bedclothes": ["sheet", "blanket", "comforter", "duvet", "bedspread", "bedding"], "track": ["track", "rail", "railroad", "railway"],
    "tree": ["tree"], "wall": ["wall"], "water": ["water", "ocean", "lake", "river", "sea"], "window": ["window"],
    "wood": ["wood", "wooden"], "table": ["table", "desk"]}
NEW = sorted(SYN44)
PAT44 = {w: re.compile(r"\b(" + "|".join(SYN44[w]) + r")(s|es)?\b") for w in NEW}
N_POS, N_NEG = 20, 480
SEEDS = ("u-", "uM", "uL")
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in ("seed", "V0"))


def keyword(s, c, Wm):
    """(profile key, prompt word) for canonical class c of column s."""
    if s == "context" and c == "dining table":
        return "table", "table"
    if c in CLS80:
        return c, O.pword(c, Wm)
    assert s == "context" and c in SYN44, (s, c)
    return c, c


def ment44(cap, w):
    return PAT44[w].search(cap.lower()) is not None


def items44(a):
    ctx_new = [c for c in CTX59 if c not in CLS80 and c != "dining table"] + ["table"]
    assert sorted(ctx_new) == NEW, sorted(set(ctx_new) ^ set(NEW))
    fit40 = {int(r["img"]) for r in json.loads((C40.WORK / "items_train.json").read_text())}
    P42 = json.loads((O.WORK / "items_prof.json").read_text())["items"]
    pool42 = sorted({p["img"]: p["cls"] for p in P42 if p["rank"] < 20}.items())       # run 42's negative pool (size 20)
    caps = defaultdict(list)
    for x in json.loads((COCO / "annotations/captions_train2017.json").read_text())["annotations"]:
        caps[x["image_id"]].append(x["caption"])
    used42 = {p["img"] for p in P42}
    ment = {i: [any(ment44(s, w) for s in cs) for w in NEW] for i, cs in caps.items() if i not in fit40}
    for k, w in enumerate(NEW):
        ex = [s for i in list(caps)[:3000] for s in caps[i] if ment44(s, w)][:2]
        print(f"rule {w!r}: {sum(m[k] for m in ment.values())} photos; e.g. {ex}", flush=True)
    rng = random.Random(44); used = set(used42); pos = {}
    for w in sorted(NEW, key=lambda w: (sum(m[NEW.index(w)] for m in ment.values()), w)):
        k = NEW.index(w); pool = sorted(i for i, m in ment.items() if m[k] and i not in used)
        rng.shuffle(pool); pos[w] = sorted(pool[:N_POS]); used.update(pos[w])
    own = dict(pool42); own.update({i: w for w in NEW for i in pos[w]})
    negpool = sorted(own)
    negw = defaultdict(list); nneg = {}
    for k, w in enumerate(NEW):
        cand = [i for i in negpool if i in ment and not ment[i][k]]
        pick = sorted(random.Random(4400 + k).sample(cand, min(N_NEG, len(cand)))); nneg[w] = len(pick)
        for i in pick:
            negw[i].append(w)
    jobs = []
    for i in sorted(set(negw) | {i for w in NEW for i in pos[w]}):
        jobs.append({"id": f"q{i:012d}", "img": i, "path": f"train2017/{i:012d}.jpg", "own": own[i],
                     "pos": own[i] if own[i] in SYN44 else None, "negw": sorted(negw.get(i, []))})
    assert not ({j["img"] for j in jobs} & fit40)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "items_prof44.json").write_text(json.dumps({"jobs": jobs, "npos": {w: len(pos[w]) for w in NEW}, "nneg": nneg}))
    print(f"jobs {len(jobs)}; positives per word min {min(len(v) for v in pos.values())} ({min(pos, key=lambda w: len(pos[w]))}); "
          f"negatives per word min {min(nneg.values())}", flush=True)


@torch.no_grad()
def dump44(a):
    from PIL import Image
    pipe, rec = R.load(); Wm = R.word_map()
    K = T.class_keys({"pipe": pipe, "ca_modules": rec.ca_order}, NEW)
    out = WORK / "dump44"; out.mkdir(parents=True, exist_ok=True)
    J = json.loads((WORK / "items_prof44.json").read_text())["jobs"][a.shard::a.num_shards]
    if a.limit: J = J[:a.limit]
    for j in J:
        o = out / f"{j['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); w = j["own"]; pw = w if w in SYN44 else O.pword(w, Wm)
        R.one_pass(pipe, rec, np.asarray(Image.open(COCO / j["path"]).convert("RGB")), f"A photograph of a {pw}", pw)
        Q = R.qbar(rec); d = {"own": w, "modules": np.array(rec.ca_order)}
        for mi, m in enumerate(rec.ca_order):
            N = Q[m].shape[0]; ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
            if j["pos"]:
                d[f"o_{mi:02d}"] = Q[m][O.word_attn(rec, K, j["pos"], m).topk(ko).indices].sum(0).cpu().numpy().astype(np.float32)
                d[f"ko_{mi:02d}"] = ko
            for nw in j["negw"]:
                d[f"n_{mi:02d}_{NEW.index(nw):02d}"] = Q[m][O.word_attn(rec, K, nw, m).topk(kn).indices].cpu().numpy().astype(np.float16)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **d); os.replace(tmp, o)
        print(f"[ours44:dump] {j['id']} own {w} pos {j['pos']} neg {len(j['negw'])} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear()


def fit44(a):
    Pj = json.loads((WORK / "items_prof44.json").read_text()); fs = sorted(f for f in (WORK / "dump44").glob("q*.npz") if not f.stem.endswith(".tmp"))
    assert a.limit or len(fs) == len(Pj["jobs"]), (len(fs), len(Pj["jobs"]))
    mods = [str(x) for x in np.load(fs[0])["modules"]]
    obj = defaultdict(list); neg = defaultdict(list)
    for f in fs:
        z = np.load(f); w = str(z["own"])
        for k in z.files:
            if k.startswith("o_"):
                mi = int(k[2:]); obj[(w, mi)].append((torch.from_numpy(z[k]), int(z[f"ko_{mi:02d}"])))
            elif k.startswith("n_"):
                _, mi, wk = k.split("_"); neg[(NEW[int(wk)], int(mi))].append(torch.from_numpy(z[k].astype(np.float32)))
    FIT = {"mo": {}, "neg": {}}; nq = {}
    for (w, mi), L in obj.items():
        FIT["mo"][(w, mods[mi])] = sum(x[0] for x in L) / sum(x[1] for x in L)
    for (w, mi), parts in neg.items():
        X = F.normalize(torch.cat(parts), dim=1); C4, _, _ = M.cluster(X, 4); FIT["neg"][(w, mods[mi])] = {4: C4}; nq[(w, mi)] = int(X.shape[0])
    torch.save({"fit": FIT, "modules": mods, "npos": Pj["npos"], "nneg": Pj["nneg"]}, WORK / "fit44.pt")
    print(f"fit44: object pairs {len(obj)}, negative pairs {len(neg)}, negative queries min {min(nq.values()) if nq else 0}", flush=True)


@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention, _apply
    pipe, rec = R.load(); Wm = R.word_map()
    P42 = torch.load(O.WORK / "fit.pt", weights_only=False)["fit"]["20"]
    P44 = torch.load(WORK / "fit44.pt", weights_only=False)["fit"] if a.set == "context" else {"mo": {}, "neg": {}}
    P = M.to_dev({"mo": {**P42["mo"], **P44["mo"]}, "neg": {**P42["neg"], **P44["neg"]}})
    it = [x for x in json.loads((W44 / "items.json").read_text())[a.set] if x["primary"]][a.shard::a.num_shards]
    if a.limit: it = it[:a.limit]
    out = WORK / f"cap_{a.set}"; out.mkdir(parents=True, exist_ok=True); checked = False
    w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=DEV)
    for x in it:
        o = out / f"{x['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); key, word = keyword(a.set, x["cls"], Wm)
        path = COCO / x["path"] if a.set == "coco" else ROOT / x["path"]
        R.one_pass(pipe, rec, np.asarray(Image.open(path).convert("RGB")), f"A photograph of a {word}", word)
        Q = R.qbar(rec); res = {m: math.isqrt(q.shape[0]) for m, q in Q.items()}
        base = {m: g for m, q in Q.items() if (g := R37.g_mk4(P, key, m, q)) is not None}
        B = {"M": base, "L": G36.consensus(base, {m: (1.0 if res[m] < 64 else 0.0) for m in base}) if base else {}}
        if not checked and base:
            lo = [m for m in base if res[m] < 64]; m64 = next(m for m in base if res[m] == 64)
            ref = torch.stack([F.interpolate(base[m].view(1, 1, res[m], res[m]), size=(64, 64), mode="bilinear",
                                             align_corners=False)[0, 0] for m in lo]).mean(0).reshape(-1).clamp(0, 1)
            dl = float((ref - B["L"][m64]).abs().max()); print(f"identity LR (independent mean) at 64: {dl:.2e}", flush=True); assert dl < 1e-6
        H64 = _attention(rec.grid_qk[64], 64); maps = {}
        for nm in SEEDS:
            gd = None if nm == "u-" else R37.powered(B[nm[1:]], Q, 1.0)
            s = R.make_seed(rec, w_u, gd); u = _apply(H64, s.float().reshape(-1))
            maps[f"{nm}|seed"] = s; maps[f"{nm}|V0"] = _apply(H64, u).mean(0).reshape(64, 64)
        if not checked and base:
            H = {g: _attention(rec.grid_qk[g], g) for g in R.SA}
            dv = float((R.propagate(H, maps["uL|seed"])["V0"] - maps["uL|V0"]).abs().max())
            print(f"identity V0 vs jc_realvoc.propagate: {dv:.2e}", flush=True); assert dv < 1e-5; checked = True; del H
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {x['id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), cls=x["cls"], key=key, word=word,
                 miss=len(Q) - len(base), area=float(x["area"]))
        os.replace(tmp, o)
        print(f"[ours44:{a.set}] {x['id']} {x['cls']} -> {key!r}/{word!r} miss {len(Q) - len(base)} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H64
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items44"); sub.add_parser("fit44").add_argument("--limit", type=int, default=0)
    for c in ("dump44", "capture"):
        p = sub.add_parser(c); p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--limit", type=int, default=0)
        if c == "capture":
            p.add_argument("--set", required=True, choices=["voc", "context", "coco"])
    a = ap.parse_args(); {"items44": items44, "dump44": dump44, "fit44": fit44, "capture": capture}[a.cmd](a)
