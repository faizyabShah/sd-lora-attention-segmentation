#!/usr/bin/env python3
"""runs/42_open_vocab/PREREG.md: MK4-LR on 60 COCO classes never used in development (O2) and profile size (O4).

items    (CPU) caption-only profile photos for 80 classes (sizes 5 / 20 / 100), negative pool, COCO val2017 items, masks
dump     (GPU) per profile photo: object statistics for its class, negative queries for its assigned words
fit      (CPU) object profiles at sizes 5 / 20 / 100 and 4 negative centroids per (word, module)  -> <WORK>/fit{tag}.pt
capture  (GPU) --set val / fit / strain: one SD1.5 pass at t = 100; seeds u-, u{M,L}{5,20,100,c}; maps seed / V0
score    (CPU) raw counts; with --crf: DenseCRF on u-|V0, uM20|V0, uL20|V0 at the seen-class threshold and 0.55
select   (CPU) threshold per map from seen-class COCO train items (run 40 fit set)               -> <WORK>/thr{tag}.json
report   (CPU)                                                                                    -> runs/42_open_vocab/result.md
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
import jc_realvoc37 as R37  # noqa: E402
import jc_coco40 as C40  # noqa: E402
import jc_gate36 as G36  # noqa: E402

R, M, T, cc, st = R37.R, R37.M, R37.R.T, R37.cc, R37.st
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_ov42"
OUT = ROOT / "runs/42_open_vocab"
COCO = C40.COCO
THR, T55, DEV = R37.THR, R37.T55, R.DEV
MIN_AREA = 0.01
SEEN = list(R.CLASSES)
C2V = C40.C2V
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
          "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
          "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
          "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
          "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
          "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]
UNSEEN = [c for c in COCO80 if C2V.get(c, c) not in SEEN]
CLS80 = SEEN + UNSEEN
assert len(UNSEEN) == 60 and len(set(CLS80)) == 80
# caption rules for the unseen classes (fixed before any capture): regex alternatives, then an optional s / es
SYN_NEW = {
    "truck": ["truck", "lorry", "pickup"], "traffic light": ["traffic light", "stoplight", "stop light", "traffic signal"],
    "fire hydrant": ["hydrant"], "stop sign": ["stop sign"], "parking meter": ["parking meter"], "bench": ["bench"],
    "elephant": ["elephant"], "bear": ["(?<!teddy )(?<!stuffed )bear"], "zebra": ["zebra"], "giraffe": ["giraffe"],
    "backpack": ["backpack", "back pack", "rucksack"], "umbrella": ["umbrella", "parasol"], "handbag": ["handbag", "hand bag", "purse"],
    "tie": ["tie", "necktie"], "suitcase": ["suitcase", "luggage", "baggage"], "frisbee": ["frisbee"], "skis": ["ski"],
    "snowboard": ["snowboard"], "sports ball": ["ball"], "kite": ["kite"], "baseball bat": ["bat"],
    "baseball glove": ["glove", "mitt"], "skateboard": ["skateboard"], "surfboard": ["surfboard", "surf board"],
    "tennis racket": ["racket", "racquet"], "wine glass": ["wine glass", "wineglass", "glass of wine"], "cup": ["cup", "mug"],
    "fork": ["fork"], "knife": ["knife", "knives"], "spoon": ["spoon"], "bowl": ["bowl"], "banana": ["banana"],
    "apple": ["apple"], "sandwich": ["sandwich"], "orange": ["orange"], "broccoli": ["broccoli"], "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"], "pizza": ["pizza"], "donut": ["donut", "doughnut"], "cake": ["cake", "cupcake"],
    "bed": ["bed"], "toilet": ["toilet"], "laptop": ["laptop"], "mouse": ["mouse", "mice"], "remote": ["remote", "controller"],
    "keyboard": ["keyboard"], "cell phone": ["phone", "cellphone", "cell phone", "smartphone"], "microwave": ["microwave"],
    "oven": ["oven", "stove"], "toaster": ["toaster"], "sink": ["sink"], "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"], "clock": ["clock"], "vase": ["vase"], "scissors": ["scissors"],
    "teddy bear": ["teddy", "teddy bear", "stuffed bear", "stuffed animal"],
    "hair drier": ["hair drier", "hair dryer", "hairdryer", "blow dryer", "blowdryer"], "toothbrush": ["toothbrush", "tooth brush"]}
assert set(SYN_NEW) == set(UNSEEN)
ALTS = {**{c: T.SYN[c].split() for c in SEEN}, **SYN_NEW}
PAT = {c: re.compile(r"\b(" + "|".join(ALTS[c]) + r")(s|es)?\b") for c in CLS80}
SIZES = (5, 20, 100)
PRIM = 20
N_NEG = 480
PSETS = ("5", "20", "100", "c")             # profile sets; c = run 33 C_all (seen classes only)
SEEDS = ("u-",) + tuple(f"u{g}{k}" for k in PSETS for g in "ML")
PROPS = ("seed", "V0")
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in PROPS)
CRF_MAPS = ("u-|V0", "uM20|V0", "uL20|V0")
GN = {"-": "none", "M": "MK4", "L": "MK4-LR"}


def mentions(cap, c):
    return PAT[c].search(cap.lower()) is not None


def pword(c, Wm):
    return Wm[c] if c in SEEN else c


def sizes(tag):
    return (2, 3, 6) if tag else SIZES


# ---------------------------------------------------------------- items
def build80(prefix, ann, keep_img, max_photos=0, seed=0):
    d = json.loads((COCO / f"annotations/instances_{ann}.json").read_text())
    cat = {c["id"]: C2V.get(c["name"], c["name"]) for c in d["categories"]}; imgs = {i["id"]: i for i in d["images"]}
    assert set(cat.values()) == set(CLS80)
    by = defaultdict(list)
    for an in d["annotations"]:
        by[an["image_id"]].append((cat[an["category_id"]], an))
    ids = sorted(i for i in by if keep_img(i))
    if max_photos:
        random.Random(seed).shuffle(ids); ids = sorted(ids[:max_photos])
    out = []; md = WORK / "masks"; md.mkdir(parents=True, exist_ok=True)
    for i in ids:
        im = imgs[i]; h, w = im["height"], im["width"]; anns = by[i]
        present = sorted({nm for nm, an in anns if not an["iscrowd"]})
        masks = {}
        for nm, an in anns:
            k = "crowd" if an["iscrowd"] else "obj"
            masks.setdefault((nm, k), np.zeros((h, w), bool))
            masks[(nm, k)] |= C40.rasterize(an["segmentation"], h, w)
        allobj = np.zeros((h, w), bool)
        for (nm, k), m in masks.items():
            if k == "obj":
                allobj |= m
        for c in present:
            truth = masks[(c, "obj")]
            if truth.sum() == 0:
                continue
            ci = CLS80.index(c) + 1; iid = f"{prefix}{i:012d}_{ci:02d}"
            ign = masks.get((c, "crowd"), np.zeros((h, w), bool)) & ~truth
            o = md / f"{iid}.npz"
            if not o.exists():
                np.savez_compressed(o, truth=truth, ignore=ign, other=allobj & ~truth & ~ign)
            out.append({"id": iid, "path": f"{ann}/{im['file_name']}", "img": int(i), "cls": c, "area": float(truth.mean()),
                        "seen": c in SEEN})
    return out


def items(a):
    tag = a.tag; n_pos = max(sizes(tag)); n_pool = sizes(tag)[1]; n_neg = 60 if tag else N_NEG
    fit40 = {int(r["img"]) for r in json.loads((C40.WORK / "items_train.json").read_text())}
    s136 = {int(p.stem) for p in (COCO / "sg_real_data_labels").glob("*.png")}
    assert len(s136) == 200
    for c, ex in (("bear", "a teddy bear sitting on a bed"), ("teddy bear", "a teddy bear sitting on a bed")):
        print(f"rule check: {c!r} on {ex!r}: {mentions(ex, c)}", flush=True)
    assert not mentions("a teddy bear sitting on a bed", "bear") and not mentions("a stuffed bear on the grass", "bear") \
        and mentions("a brown bear in the woods", "bear")
    caps = defaultdict(list)
    for x in json.loads((COCO / "annotations/captions_train2017.json").read_text())["annotations"]:
        caps[x["image_id"]].append(x["caption"])
    ment = {i: [any(mentions(s, c) for s in cs) for c in CLS80] for i, cs in sorted(caps.items()) if i not in fit40}
    head = [s for _, cs in sorted(caps.items())[:20000] for s in cs]
    for k, c in enumerate(CLS80):       # caption-rule examples
        ex = [s for s in head if mentions(s, c)][:2]
        print(f"rule {c!r}: {sum(m[k] for m in ment.values())} photos; e.g. {ex}", flush=True)
    rng = random.Random(42); used = set(); pos = {}
    pools = {c: sorted(i for i, m in ment.items() if m[k]) for k, c in enumerate(CLS80)}
    for c in sorted(CLS80, key=lambda c: (len(pools[c]), c)):
        pool = [i for i in pools[c] if i not in used]; rng.shuffle(pool); pos[c] = pool[:n_pos]; used.update(pos[c])
    pool = sorted({i for c in CLS80 for i in pos[c][:n_pool]})
    negw = defaultdict(list); nneg = {}
    for k, c in enumerate(CLS80):
        cand = [i for i in pool if not ment[i][k]]
        pick = sorted(random.Random(42 + k).sample(cand, min(n_neg, len(cand)))); nneg[c] = len(pick)
        for i in pick:
            negw[i].append(c)
    prof = []
    for c in CLS80:
        for r_, i in enumerate(pos[c]):
            prof.append({"id": f"p{i:012d}_{CLS80.index(c) + 1:02d}", "img": i, "path": f"train2017/{i:012d}.jpg", "cls": c,
                         "rank": r_, "negw": negw.get(i, [])})
    assert not ({p["img"] for p in prof} & fit40)
    assert sum(len(v) for v in negw.values()) == sum(nneg.values()) and set(negw) <= {p["img"] for p in prof}
    (WORK / f"items_prof{tag}.json").write_text(json.dumps({"items": prof, "npos": {c: len(pos[c]) for c in CLS80}, "nneg": nneg}))
    print(f"profile photos {len(prof)}; pool {len(pool)}; positives per class min {min(len(v) for v in pos.values())} "
          f"({min(pos, key=lambda c: len(pos[c]))}); negatives per class min {min(nneg.values())}", flush=True)
    profimg = {p["img"] for p in prof}
    if tag:     # smoke evaluation set: COCO train photos outside every fit set (no val2017 photo is used in the smoke)
        X = build80("str", "train2017", lambda i: i not in profimg and i not in fit40, max_photos=40, seed=1)
        (WORK / f"items_strain{tag}.json").write_text(json.dumps(X))
    else:
        X = build80("val", "val2017", lambda i: i not in s136)
        assert not ({x["img"] for x in X} & s136)
        (WORK / "items_val.json").write_text(json.dumps(X))
    print(f"evaluation items {len(X)} ({len({x['img'] for x in X})} photos); primary {sum(x['area'] >= MIN_AREA for x in X)}; "
          f"unseen primary {sum(x['area'] >= MIN_AREA and not x['seen'] for x in X)}", flush=True)


def items_of(s, a):
    f = {"val": WORK / "items_val.json", "strain": WORK / f"items_strain{a.tag}.json", "fit": C40.WORK / "items_train.json",
         "prof": WORK / f"items_prof{a.tag}.json"}[s]
    X = json.loads(f.read_text()); X = X["items"] if s == "prof" else X
    X = X[a.shard::a.num_shards]
    if a.max_items:
        X = X[:: max(1, len(X) // a.max_items)][: a.max_items]
    return X


# ---------------------------------------------------------------- dump (profile statistics)
def word_attn(rec, K, w, m):
    q = rec.q[m]
    return (q @ K[w][0][m].transpose(-1, -2) * rec.scale[m]).softmax(-1)[..., K[w][1]].mean(-1).mean(0)


@torch.no_grad()
def dump(a):
    from PIL import Image
    pipe, rec = R.load(); Wm = R.word_map(); K = T.class_keys({"pipe": pipe, "ca_modules": rec.ca_order}, CLS80)
    out = WORK / f"dump{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for it in items_of("prof", a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]
        R.one_pass(pipe, rec, np.asarray(Image.open(COCO / it["path"]).convert("RGB")), f"A photograph of a {pword(c, Wm)}", pword(c, Wm))
        Q = R.qbar(rec); d = {"cls": c, "rank": it["rank"], "modules": np.array(rec.ca_order)}
        for mi, m in enumerate(rec.ca_order):
            N = Q[m].shape[0]; ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
            d[f"o_{mi:02d}"] = Q[m][word_attn(rec, K, c, m).topk(ko).indices].sum(0).cpu().numpy().astype(np.float32)
            d[f"ko_{mi:02d}"] = ko
            for w in it["negw"]:
                d[f"n_{mi:02d}_{CLS80.index(w):02d}"] = Q[m][word_attn(rec, K, w, m).topk(kn).indices].cpu().numpy().astype(np.float16)
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **d); os.replace(tmp, o)
        print(f"[ov42:dump] {it['id']} {c} neg {len(it['negw'])} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear()


def fit(a):
    torch.set_num_threads(a.workers)
    Pj = json.loads((WORK / f"items_prof{a.tag}.json").read_text()); fs = sorted((WORK / f"dump{a.tag}").glob("p*.npz"))
    fs = [f for f in fs if not f.stem.endswith(".tmp")]
    assert len(fs) == len(Pj["items"]), (len(fs), len(Pj["items"]))
    mods = [str(x) for x in np.load(fs[0])["modules"]]; nM = len(mods)
    obj = defaultdict(list); neg = defaultdict(list)
    for f in fs:
        z = np.load(f); c = str(z["cls"]); r_ = int(z["rank"])
        for mi in range(nM):
            obj[(c, mi)].append((r_, torch.from_numpy(z[f"o_{mi:02d}"]), int(z[f"ko_{mi:02d}"])))
        for k in z.files:
            if k.startswith("n_"):
                _, mi, wk = k.split("_"); neg[(CLS80[int(wk)], int(mi))].append(torch.from_numpy(z[k].astype(np.float32)))
    FIT = {str(n): {"mo": {}, "neg": {}} for n in sizes(a.tag)}; nneg = {}
    for (c, mi), L in obj.items():
        L.sort(key=lambda x: x[0])
        for n in sizes(a.tag):
            sub = L[:n]; FIT[str(n)]["mo"][(c, mods[mi])] = sum(x[1] for x in sub) / sum(x[2] for x in sub)
    for (c, mi), parts in neg.items():
        X = F.normalize(torch.cat(parts), dim=1); C4, _, _ = M.cluster(X, 4); nneg[(c, mi)] = int(X.shape[0])
        for n in sizes(a.tag):
            FIT[str(n)]["neg"][(c, mods[mi])] = {4: C4}
    torch.save({"fit": FIT, "modules": mods, "npos": Pj["npos"], "nneg_photos": Pj["nneg"]}, WORK / f"fit{a.tag}.pt")
    print(f"fit: {len(obj)} object (class, module) pairs; {len(neg)} negative pairs; queries per negative set min "
          f"{min(nneg.values())}", flush=True)


# ---------------------------------------------------------------- capture
@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention, _apply
    pipe, rec = R.load(); Wm = R.word_map(); FT = torch.load(WORK / f"fit{a.tag}.pt", weights_only=False)
    sz = sizes(a.tag); P = {str(k): M.to_dev(FT["fit"][str(n)]) for k, n in zip(("5", "20", "100"), sz)}
    P["c"] = M.to_dev(torch.load(R37.FIT33, weights_only=False)["fit"]["C_all"])
    out = WORK / f"cap_{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True); checked = False
    w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=DEV)
    for it in items_of(a.set, a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]
        R.one_pass(pipe, rec, np.asarray(Image.open(COCO / it["path"]).convert("RGB")), f"A photograph of a {pword(c, Wm)}", pword(c, Wm))
        Q = R.qbar(rec); res = {m: math.isqrt(q.shape[0]) for m, q in Q.items()}; B, miss = {}, {}
        for k in PSETS:
            base = {m: g for m, q in Q.items() if (g := R37.g_mk4(P[k], c, m, q)) is not None}; miss[k] = len(Q) - len(base)
            B[f"M{k}"] = base; B[f"L{k}"] = G36.consensus(base, {m: (1.0 if res[m] < 64 else 0.0) for m in base}) if base else {}
            if not checked and k == "20" and base:
                lo = [m for m in base if res[m] < 64]; m64 = next(m for m in base if res[m] == 64)
                ref = torch.stack([F.interpolate(base[m].view(1, 1, res[m], res[m]), size=(64, 64), mode="bilinear",
                                                 align_corners=False)[0, 0] for m in lo]).mean(0).reshape(-1).clamp(0, 1)
                dl = float((ref - B["L20"][m64]).abs().max()); print(f"identity LR (independent mean) at 64: {dl:.2e}", flush=True)
                assert dl < 1e-6
        H64 = _attention(rec.grid_qk[64], 64); maps = {}
        for nm in SEEDS:
            gd = None if nm == "u-" else R37.powered(B[nm[1:]], Q, 1.0)
            s = R.make_seed(rec, w_u, gd); c64 = s.float().reshape(-1); u = _apply(H64, c64)
            maps[f"{nm}|seed"] = s; maps[f"{nm}|V0"] = _apply(H64, u).mean(0).reshape(64, 64)
        if not checked:
            H = {g: _attention(rec.grid_qk[g], g) for g in R.SA}
            dv = float((R.propagate(H, maps["uL20|seed"])["V0"] - maps["uL20|V0"]).abs().max())
            print(f"identity V0 vs jc_realvoc.propagate: {dv:.2e}", flush=True); assert dv < 1e-5; checked = True; del H
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {it['id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), cls=c, img=str(it["img"]), path=it["path"],
                 area=float(it["area"]), **{f"miss_{k}": v for k, v in miss.items()})
        os.replace(tmp, o)
        print(f"[ov42:{a.set}] {it['id']} {c} miss {miss} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H64
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- score / select
def score_one(job):
    s, f, o, thr = job[0], Path(job[1]), Path(job[2]), job[3]
    if o.exists():
        return
    torch.set_num_threads(1)
    from PIL import Image
    z = np.load(f); mk = np.load((C40.WORK if s == "fit" else WORK) / "masks" / f"{f.stem}.npz")
    truth, valid, other = mk["truth"], ~mk["ignore"], mk["other"]
    res = {"cls": str(z["cls"]), "img": str(z["img"]), "area": float(z["area"])}
    res.update({k: int(z[k]) for k in z.files if k.startswith("miss_")})
    rgb = None
    for i, nm in enumerate(str(x) for x in z["names"]):
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if thr is not None and nm in CRF_MAPS:
            from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
            _import_densecrf(st.CRF_SITE)
            if rgb is None:
                rgb = np.asarray(Image.open(COCO / str(z["path"])).convert("RGB"), np.uint8)
            ms = np.stack([apply_ovam_densecrf(rgb, x >= t, st.CRF_SITE) for t in (thr[nm], float(THR[T55]))]).astype(bool)
            res[f"{nm}|crf"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    thr = json.loads((WORK / f"thr{a.tag}.json").read_text())["thr"] if a.crf else None
    sc = WORK / f"cap_{a.set}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(a.set, str(f), str(sc / f.name), thr) for f in sorted((WORK / f"cap_{a.set}{a.tag}").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    for attempt in range(5):
        todo = [j for j in jobs if not Path(j[2]).exists()]; print(f"{a.set} attempt {attempt}: {len(todo)} to score", flush=True)
        if not todo:
            break
        try:
            with ProcessPoolExecutor(a.workers) as ex:
                for n, _ in enumerate(ex.map(score_one, todo, chunksize=1), 1):
                    if n % 500 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)
        except BrokenProcessPool:
            print("worker died; restarting pool", flush=True)
    if [j for j in jobs if not Path(j[2]).exists()]:
        raise SystemExit("items not scored")


def load_scores(s, tag):
    fs = sorted(f for f in (WORK / f"cap_{s}{tag}_scores").glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    S = {"id": np.array([f.stem for f in fs]), "cls": np.array([str(z["cls"]) for z in Z]), "grp": np.array([str(z["img"]) for z in Z]),
         "area": np.array([float(z["area"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k:
            S[k] = np.stack([z[k] for z in Z])
        elif k.startswith("miss_"):
            S[k] = np.array([int(z[k]) for z in Z])
    S["seen"] = np.isin(S["cls"], SEEN)
    return S


def select(a):
    S = load_scores("fit", a.tag); one = np.ones(len(S["cls"]), bool); thr = {}
    assert S["seen"].all()
    for nm in MAPS:
        cv = cc.miou(S[f"{nm}|raw"], S["cls"], one); thr[nm] = float(THR[int(np.argmax(cv))])
    (WORK / f"thr{a.tag}.json").write_text(json.dumps({"thr": thr, "items": int(len(S["cls"]))}, indent=1)); print(json.dumps(thr), flush=True)


# ---------------------------------------------------------------- report
class Boot:
    """Class-macro mIoU on photo-bootstrap resamples (as jc_realvoc.Boot, over the classes present)."""

    def __init__(self, S, n=2000):
        self.cl = sorted(set(S["cls"])); self.g, self.gi = np.unique(S["grp"], return_inverse=True)
        self.ci = np.array([self.cl.index(c) for c in S["cls"]]); G, C = len(self.g), len(self.cl)
        self.present = np.zeros((G, C)); np.add.at(self.present, (self.gi, self.ci), 1)
        rng = np.random.default_rng(0)
        self.W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(n)]).astype(np.float64)

    def miou(self, P, weights=None):
        X = np.zeros((len(self.g), len(self.cl), 3)); np.add.at(X, (self.gi, self.ci), P[:, :3].astype(np.float64))
        if weights is None:
            weights = np.ones((1, len(self.g)))
        tot = np.einsum("bg,gcx->bcx", weights, X); n = weights @ self.present
        return np.where(n > 0, tot[..., 0] / np.maximum(tot.sum(-1), 1), np.nan)

    def stat(self, P):
        return np.nanmean(self.miou(P)[0]), np.nanmean(self.miou(P, self.W), 1)


def report(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr = Tr["thr"]; ev = "strain" if a.tag else "val"
    Sall = load_scores(ev, a.tag); FT = torch.load(WORK / f"fit{a.tag}.pt", weights_only=False); npos = FT["npos"]
    sub = lambda S, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}
    ti = {n: int(np.flatnonzero(np.isclose(THR, thr[n]))[0]) for n in MAPS}
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    rd = lambda x: "**helps**" if x[1] > 0 else "**hurts**" if x[2] < 0 else "not resolved"
    L = []; w = L.append

    class E:
        def __init__(self, S):
            self.S, self.B, self.c = S, Boot(S, a.boot), {}

        def P(self, n, rule):
            if rule == "raw":
                return self.S[f"{n}|raw"][:, ti[n]]
            if rule == "055":
                return self.S[f"{n}|raw"][:, T55]
            if isinstance(rule, int):
                return self.S[f"{n}|raw"][:, rule]
            return self.S[f"{n}|crf"][:, 0 if rule == "crf" else 1]

        def m(self, n, rule="raw"):
            if (n, rule) not in self.c:
                self.c[(n, rule)] = self.B.stat(self.P(n, rule))
            return self.c[(n, rule)]

        def d(self, x, y, rule="raw"):
            A, B_ = self.m(x, rule), self.m(y, rule); dd = A[1] - B_[1]
            return float(A[0] - B_[0]), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
    nm = lambda g, k, p="V0": "u-|" + p if g == "-" else f"u{g}{k}|{p}"
    prim = Sall["area"] >= MIN_AREA; Sp = sub(Sall, prim)
    EU, ES = E(sub(Sp, ~Sp["seen"])), E(sub(Sp, Sp["seen"]))
    EUa, ESa = E(sub(Sall, ~Sall["seen"])), E(sub(Sall, Sall["seen"]))
    ESx = E(sub(Sp, Sp["seen"] & (Sp["cls"] != "person")))
    nU, nS = len(set(Sp["cls"][~Sp["seen"]])), len(set(Sp["cls"][Sp["seen"]]))
    w("# Run 42: the low-resolution consensus gate on 60 COCO classes never used in development (`PREREG.md`)\n")
    if a.tag:
        w("**SMOKE: evaluated on COCO train photos outside every fit set; profile sizes 2 / 3 / 6. Numbers are not results.**\n")
    w(f"**Confirmatory for the unseen-class question (O2); profile size (O4) descriptive.** Evaluation: "
      f"{'COCO val2017 minus the Section 136 set' if not a.tag else 'smoke set'}, {len(Sall['cls'])} items "
      f"({len(np.unique(Sall['grp']))} photos); primary items (class area ≥ {MIN_AREA:.0%}): {int(prim.sum())}, of which "
      f"{int((~Sp['seen']).sum())} unseen-class items over {nU} classes and {int(Sp['seen'].sum())} seen-class items over {nS} "
      f"classes. These val2017 photos were evaluated before for the 20 VOC classes only (runs 40, 41). Pipeline: uniform "
      f"cross-attention seed, γ = 1, fine-only propagation (V0), raw map at the threshold chosen on {Tr['items']} seen-class "
      f"COCO train items. Profiles from captioned COCO train photos (no labels). Class-macro mIoU; 95% photo-bootstrap "
      f"intervals ({a.boot} samples, paired).\n")
    k0 = str(PRIM)
    w(f"\n## 1. Primary contrasts (primary items, raw V0, profile size {PRIM if not a.tag else sizes(a.tag)[1]})\n")
    w("| # | Contrast | Classes | Δ mIoU [95% CI] | Reading |"); w("|---|---|---|---|---|")
    U1, U2 = EU.d(nm("L", "20"), nm("-", "")), EU.d(nm("L", "20"), nm("M", "20"))
    S1, S2 = ES.d(nm("L", "20"), nm("-", "")), ES.d(nm("L", "20"), nm("M", "20"))
    for lab, x, cl in (("U1", U1, f"unseen ({nU})"), ("U2", U2, f"unseen ({nU})"), ("S1", S1, f"seen ({nS})"), ("S2", S2, f"seen ({nS})")):
        w(f"| {lab} | MK4-LR − {'no gate' if lab[1] == '1' else 'MK4'} | {cl} | {f3(x)} | {rd(x)} |")
    w(f"\n- The gate generalises to unseen classes (U1 CI above 0): **{'yes' if U1[1] > 0 else 'no'}**.")
    w(f"- LR's advantage over MK4 generalises (U2 CI above 0): **{'yes' if U2[1] > 0 else 'no'}**.")
    if S1[0] != 0:
        w(f"- Unseen / seen gain over no gate: {U1[0] / S1[0]:.2f}.")

    w(f"\n## 2. mIoU, primary items (profile size 20)\n")
    w("| Classes | Map | Read-out | none | MK4 | MK4-LR |"); w("|---|---|---|---:|---:|---:|")
    for lab, E_ in (("unseen", EU), ("seen", ES)):
        for p, rule in (("seed", "raw"), ("V0", "raw"), ("V0", "055"), ("V0", "crf"), ("V0", "crf055")):
            if rule.startswith("crf") and f"{nm('-', '', p)}|crf" not in E_.S:
                continue
            rl = {"raw": "raw, seen-class thr (primary for V0)", "055": "raw, 0.55", "crf": "+ DenseCRF, seen-class thr",
                  "crf055": "+ DenseCRF, 0.55"}[rule]
            w(f"| {lab} | {p} | {rl} | " + " | ".join(f"{E_.m(nm(g, '20', p), rule)[0]:.3f}" for g in "-ML") + " |")
    w("\nSeen classes with run 33's profiles (`C_all`, 490 photos), same capture, raw V0: "
      f"MK4 {ES.m(nm('M', 'c'))[0]:.3f}, MK4-LR {ES.m(nm('L', 'c'))[0]:.3f}; this run's size-20 profiles: MK4 "
      f"{ES.m(nm('M', '20'))[0]:.3f}, MK4-LR {ES.m(nm('L', '20'))[0]:.3f}.")

    w("\n## 3. Profile size (O4; primary items, raw V0)\n")
    w("| Classes | Positive photos per class | MK4-LR − none | MK4 − none | MK4-LR − MK4 | MK4-LR mIoU |"); w("|---|---|---|---|---|---:|")
    for lab, E_ in (("unseen", EU), ("seen", ES)):
        for k, n in zip(("5", "20", "100"), sizes(a.tag)):
            w(f"| {lab} | {n} | {f3(E_.d(nm('L', k), nm('-', '')))} | {f3(E_.d(nm('M', k), nm('-', '')))} | "
              f"{f3(E_.d(nm('L', k), nm('M', k)))} | {E_.m(nm('L', k))[0]:.3f} |")
    w("\nClasses with fewer positive photos than the largest size: " +
      (", ".join(f"{c} {npos[c]}" for c in CLS80 if npos[c] < max(sizes(a.tag))) or "none") + ".")

    w("\n## 4. Secondary rules (profile size 20)\n")
    w("| Contrast | Classes | seed only (raw) | all items, raw V0 | + DenseCRF V0 | seen, person excluded |"); w("|---|---|---|---|---|---|")
    for g, ref in (("L", "-"), ("L", "M"), ("M", "-")):
        for lab, E_, Ea_ in (("unseen", EU, EUa), ("seen", ES, ESa)):
            crf = f3(E_.d(nm(g, "20"), nm(ref, "20"), "crf")) if f"{nm(g, '20')}|crf" in E_.S else "—"
            w(f"| {GN[g]} − {GN[ref]} | {lab} | {f3(E_.d(nm(g, '20', 'seed'), nm(ref, '20', 'seed')))} | "
              f"{f3(Ea_.d(nm(g, '20'), nm(ref, '20')))} | {crf} | {f3(ESx.d(nm(g, '20'), nm(ref, '20'))) if lab == 'seen' else '—'} |")
    w("\nRaw V0 at every threshold, unseen classes (profile size 20; † = 95% CI excludes 0):\n")
    w("| Contrast | " + " | ".join(f"{t:.3f}" for t in THR) + " |"); w("|---|" + "---:|" * len(THR))
    for g, ref in (("L", "-"), ("L", "M")):
        row = []
        for i in range(len(THR)):
            x = EU.d(nm(g, "20"), nm(ref, "20"), i); row.append(f"{x[0]:+.3f}{'†' if x[1] > 0 or x[2] < 0 else ''}")
        w(f"| {GN[g]} − {GN[ref]} | " + " | ".join(row) + " |")

    w("\n## 5. Per class (primary items, raw V0, profile size 20)\n")
    w("| Class | seen | items | positive photos | none | MK4 | MK4-LR | LR − none | LR − MK4 | modules without profile (mean) |")
    w("|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cnt = {"seen": [0, 0, 0, 0], "unseen": [0, 0, 0, 0]}
    for c in CLS80:
        sel = Sp["cls"] == c
        if not sel.any():
            continue
        v = {g: cc.miou(Sp[f"{nm(g, '20')}|raw"][:, ti[nm(g, '20')]][:, None], Sp["cls"], sel)[0] for g in "-ML"}
        k = "seen" if c in SEEN else "unseen"; d1, d2 = v["L"] - v["-"], v["L"] - v["M"]
        cnt[k][0] += d1 > 0; cnt[k][1] += d1 < 0; cnt[k][2] += d2 > 0; cnt[k][3] += d2 < 0
        w(f"| {c} | {'✓' if c in SEEN else ''} | {int(sel.sum())} | {npos[c]} | {v['-']:.3f} | {v['M']:.3f} | {v['L']:.3f} | "
          f"{d1:+.3f} | {d2:+.3f} | {float(np.mean(Sp['miss_20'][sel])):.1f} |")
    for k, x in cnt.items():
        w(f"\n- {k}: MK4-LR beats / trails no gate in {x[0]} / {x[1]} classes; beats / trails MK4 in {x[2]} / {x[3]}.")
    w("\nBRACE stays frozen; nothing here is a promotion.\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("items", "dump", "fit", "capture", "score", "select", "report"):
        p = sp.add_parser(cmd)
        p.add_argument("--set", choices=("val", "fit", "strain"), default="val"); p.add_argument("--crf", action="store_true")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"items": items, "dump": dump, "fit": fit, "capture": capture, "score": score, "select": select, "report": report}[a.cmd](a)
