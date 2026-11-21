#!/usr/bin/env python3
"""runs/33_realvoc_timesteps/PREREG.md: noise levels and profile sources on real PASCAL VOC photos (exploratory).

items    (CPU) source C (real COCO train2017 photos, captions only) and source S (A cut to B's size) -> <WORK>/items_{C,S}.json
dump     (GPU) profile-fit statistics at t = 100, 300, 500 (shared noise): --src A (VOC train), B (VOC-Sim), C (COCO)
fit      (CPU) object profiles and K = 4 absent-class centroids per t, for A_all/A_f0/A_f1/S_all/S_f0/S_f1/B_all/C_all
capture  (GPU) three calls per item; arms t100, t300, t500, M (multi)                    -> <WORK>/cap_{split}{tag}/
score    (CPU) raw counts; val only: DenseCRF at the train-selected threshold and at 0.55
select   (CPU) train-selected threshold per map                                         -> <WORK>/thr{tag}.json
report   (CPU)                                                                           -> runs/33_realvoc_timesteps/result{tag}.md

Run 32 code (scripts/jc_realvoc.py) is imported unchanged: recorder, A3 weights, q/t statistics, propagation, bootstrap.
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_realvoc as R  # noqa: E402  (run 32)

M, T, cc, st, nr, AB = R.M, R.T, R.cc, R.st, R.nr, R.AB
from brace_paths import CACHE  # noqa: E402

WORK = CACHE / "jc_realvoc_t"
OUT = ROOT / "runs/33_realvoc_timesteps"
COCO = ROOT / "datasets/coco2017"
VOC, VOCSIM, CLASSES, THR, T55, DEV = R.VOC, R.VOCSIM, R.CLASSES, R.THR, R.T55, R.DEV
TS = (100, 300, 500)
ARMS = ("t100", "t300", "t500", "M")
ARM_NAME = {"t100": "t = 100", "t300": "t = 300", "t500": "t = 500", "M": "multi (100+300+500)"}
SEEDS = (("u-", "u", None), ("w-", "w", None), ("uA4", "u", "A"), ("wA4", "w", "A"), ("uB4", "u", "B"), ("wB4", "w", "B"),
         ("uC4", "u", "C"), ("wC4", "w", "C"), ("wS4", "w", "S"))
BASE = tuple(f"{s}|{p}" for s, _, _ in SEEDS for p in ("seed", "V0", "V1")) + ("w-|V3",)
CRF_BASE = tuple(f"{s}|{p}" for s in ("u-", "w-", "uA4", "wA4", "uB4", "wB4", "uC4", "wC4") for p in ("V0", "V1")) \
    + ("w-|V3", "wS4|V1")
MAPS = tuple(f"{a}:{b}" for a in ARMS for b in BASE)
CRF_MAPS = frozenset(f"{a}:{b}" for a in ARMS for b in CRF_BASE)
COMP = R.COMP
CNAME = R.CNAME
SRC_NAME = {"A": "A: VOC train labels", "S": "S: VOC train labels, cut to 490", "B": "B: VOC-Sim generated, captions",
            "C": "C: real COCO photos, captions"}


def cell(arm, src, W, H, G):
    return f"{arm}:{R.cell(src, W, H, G)}"


# ---------------------------------------------------------------- items (sources C and S)
def n_b():
    return Counter(AB.label(r) for r in nr.rows("voc"))


def items(a):
    WORK.mkdir(parents=True, exist_ok=True); nB = n_b()
    # C: real COCO train2017 photos; mention = any of the photo's captions mentions the class
    caps = defaultdict(list)
    for x in json.loads((COCO / "annotations/captions_train2017.json").read_text())["annotations"]:
        caps[x["image_id"]].append(x["caption"])
    ment = {i: [any(T.mentions(s, c) for s in cs) for c in CLASSES] for i, cs in sorted(caps.items())}
    rng = random.Random(33); used = set(); out = []
    pools = {c: sorted(i for i, m in ment.items() if m[ci]) for ci, c in enumerate(CLASSES)}
    for c in sorted(CLASSES, key=lambda c: len(pools[c])):
        pool = [i for i in pools[c] if i not in used]; rng.shuffle(pool)
        for i in pool[: nB[c]]:
            used.add(i)
            out.append({"id": f"coco{i:012d}_{CLASSES.index(c) + 1:02d}", "path": f"train2017/{i:012d}.jpg", "cls": c,
                        "mention": ment[i], "caps": caps[i]})
        print(f"C {c}: pool {len(pools[c])}, taken {min(nB[c], len(pool))} / {nB[c]}", flush=True)
    out.sort(key=lambda x: x["id"])
    (WORK / "items_C.json").write_text(json.dumps(out))
    # S: A cut to B's per-class counts, within each fold and over all of train; seeded by crc32(item id)
    tr = json.loads((R.WORK / "items_train.json").read_text()); S = {}
    for name, keep in (("S_all", lambda x: True), ("S_f0", lambda x: x["fold"] == 0), ("S_f1", lambda x: x["fold"] == 1)):
        ids = []
        for c in CLASSES:
            pool = sorted((x for x in tr if x["cls"] == c and keep(x)), key=lambda x: (R.h(x["id"]), x["id"]))
            ids += [x["id"] for x in pool[: nB[c]]]
        S[name] = sorted(ids)
        print(name, len(ids), flush=True)
    (WORK / "items_S.json").write_text(json.dumps(S))


# ---------------------------------------------------------------- multi-call pass
@torch.no_grad()
def encode(pipe, rec, img_u8, prompt, word):
    from src.segmentation.attention_aggregation import find_token_subsequence
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float().to(DEV) / 127.5 - 1
    x = F.interpolate(x, size=(R.PX, R.PX), mode="bicubic", align_corners=False).clamp(-1, 1)
    z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(R.NOISE_SEED)).to(DEV, z0.dtype)
    pe, _ = pipe.encode_prompt(prompt=prompt, device=DEV, num_images_per_prompt=1, do_classifier_free_guidance=False)
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
    rec.target = find_token_subsequence(ids, pipe.tokenizer(word, add_special_tokens=False)["input_ids"])
    return z0, eps, pe.float()


@torch.no_grad()
def call(pipe, rec, z0, eps, pe, t):
    """As jc_realvoc.one_pass, at noise level t, with the shared noise draw."""
    ac = pipe.scheduler.alphas_cumprod.to(DEV)
    zt = ac[t].sqrt() * z0 + (1 - ac[t]).sqrt() * eps
    rec.clear()
    pipe.unet(zt, torch.tensor(t, device=DEV), encoder_hidden_states=pe, return_dict=False)
    if set(rec.ca) != rec.ca_names or rec.ref_q is None or set(rec.grid_qk) != set(R.SA) or set(rec.pair_qk) != rec.pair:
        raise ValueError("missing recorded attention")


# ---------------------------------------------------------------- dump
def dump_rows(a):
    W = R.word_map()
    if a.src in ("A", "B"):
        yield from R.dump_rows(a)
        return
    Rs = json.loads((WORK / "items_C.json").read_text())[a.shard::a.num_shards]
    if a.max_items:
        Rs = Rs[:: max(1, len(Rs) // a.max_items)][: a.max_items]
    for r in Rs:
        assert r["mention"][CLASSES.index(r["cls"])]
        yield r["id"], COCO / r["path"], f"A photograph of a {W[r['cls']]}", W[r["cls"]], r["cls"], r["mention"], -1


@torch.no_grad()
def dump(a):
    from PIL import Image
    pipe, rec = R.load(); K = T.class_keys({"pipe": pipe, "ca_modules": rec.ca_order}, CLASSES)
    outs = {t: WORK / f"dump_{a.src}_t{t}{a.tag}" for t in TS}
    for o in outs.values():
        o.mkdir(parents=True, exist_ok=True)
    for iid, path, prompt, word, lab, mention, fold in dump_rows(a):
        if all((outs[t] / f"{iid}.npz").exists() for t in TS):
            continue
        t0 = time.perf_counter(); ci = CLASSES.index(lab)
        z0, eps, pe = encode(pipe, rec, np.asarray(Image.open(path).convert("RGB")), prompt, word)
        for t in TS:
            call(pipe, rec, z0, eps, pe, t)
            Q = R.qbar(rec); Tm = R.tbar(rec, K)
            d = {"label": lab, "src": a.src, "fold": fold, "mention": np.array(mention), "prompt": prompt,
                 "modules": np.array(rec.ca_order), "t": t}
            for mi, m in enumerate(rec.ca_order):
                q, tt = Q[m], Tm[m]; N = q.shape[0]
                ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
                d[f"o_{mi:02d}"] = q[tt[ci].topk(ko).indices].sum(0).cpu().numpy().astype(np.float32); d[f"ko_{mi:02d}"] = ko
                for cj, c in enumerate(CLASSES):
                    if c != lab and not mention[cj]:
                        d[f"n_{mi:02d}_{cj:02d}"] = q[tt[cj].topk(kn).indices].cpu().numpy().astype(np.float16)
            o = outs[t] / f"{iid}.npz"; tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **d); os.replace(tmp, o)
        print(f"[realvoc_t:dump:{a.src}] {iid} {lab} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear()


# ---------------------------------------------------------------- fit (one noise level per job)
def fit(a):
    torch.set_num_threads(a.workers); t = a.t
    ls = lambda s: sorted(f for f in (WORK / f"dump_{s}_t{t}{a.tag}").glob("*.npz") if not f.stem.endswith(".tmp"))
    fA, fB, fC = ls("A"), ls("B"), ls("C")
    meta = {f: (str(np.load(f)["label"]), int(np.load(f)["fold"])) for f in fA + fB + fC}
    Sids = json.loads((WORK / "items_S.json").read_text())
    if a.tag:   # smoke: only part of A is dumped; S restricted to what exists
        have = {f.stem for f in fA}; Sids = {k: [i for i in v if i in have] for k, v in Sids.items()}
    byid = {f.stem: f for f in fA}
    SETS = {"A_all": fA, "A_f0": [f for f in fA if meta[f][1] == 0], "A_f1": [f for f in fA if meta[f][1] == 1],
            "S_all": [byid[i] for i in Sids["S_all"]], "S_f0": [byid[i] for i in Sids["S_f0"]],
            "S_f1": [byid[i] for i in Sids["S_f1"]], "B_all": fB, "C_all": fC}
    print(f"t={t} fit sets", {k: len(v) for k, v in SETS.items()}, flush=True)
    mods = [str(x) for x in np.load(fA[0])["modules"]]
    FIT = {s: {"mo": {}, "neg": {}} for s in SETS}; STAT = {s: {} for s in SETS}
    for mi, m in enumerate(mods):
        t0 = time.perf_counter(); obj, ko, neg = {}, {}, {}
        for f in fA + fB + fC:
            z = np.load(f); obj[f] = torch.from_numpy(z[f"o_{mi:02d}"]); ko[f] = int(z[f"ko_{mi:02d}"])
            neg[f] = {cj: torch.from_numpy(z[f"n_{mi:02d}_{cj:02d}"].astype(np.float32)) for cj in range(len(CLASSES))
                      if f"n_{mi:02d}_{cj:02d}" in z.files}
        for s, fs in SETS.items():
            for cj, c in enumerate(CLASSES):
                oi = [f for f in fs if meta[f][0] == c]
                if oi:
                    FIT[s]["mo"][(c, m)] = sum(obj[f] for f in oi) / sum(ko[f] for f in oi)
                parts = [neg[f][cj] for f in fs if cj in neg[f]]
                if not parts:
                    continue
                X = F.normalize(torch.cat(parts), dim=1)
                C4, lab4, _ = M.cluster(X, 4)
                FIT[s]["neg"][(c, m)] = {4: C4}
                STAT[s][(c, m)] = {"n": int(X.shape[0]), "k4": int(C4.shape[0]), "sizes4": torch.bincount(lab4).tolist(),
                                   "n_obj": len(oi)}
        print(f"t={t} fit module {mi} ({m}): {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save({"fit": FIT, "stat": STAT, "modules": mods, "sets": {k: len(v) for k, v in SETS.items()}},
               WORK / f"fit_t{t}{a.tag}.pt")


# ---------------------------------------------------------------- capture
def raw_seed(rec, w, gate=None):
    """jc_realvoc.make_seed before min-max."""
    s = torch.zeros((64, 64), device=DEV)
    for i, m in enumerate(rec.ca_order):
        x = rec.ca[m] if gate is None else rec.ca[m] * gate[m]
        side = math.isqrt(x.numel())
        s += F.interpolate(x.reshape(1, 1, side, side), size=(64, 64), mode="bilinear", align_corners=False)[0, 0] * w[i]
    return s


def mm(x):
    from src.segmentation import controlled_cv as cv
    return cv.normalize_batch(x[None], "minmax")[0]


def prop_multi(Hs, seed, v3):
    """sg_combined_v2.aggregate over the given calls: per-call propagation, averaged."""
    outs = [R.propagate(H, seed, v3=v3) for H in Hs]
    return {k: sum(o[k] for o in outs) / len(outs) for k in outs[0]}


@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation.sg25_sa import _attention
    pipe, rec = R.load(); W = R.word_map()
    PF = {t: M.to_dev(torch.load(WORK / f"fit_t{t}{a.tag}.pt")["fit"]) for t in TS}
    out = WORK / f"cap_{a.split}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=DEV)
    checked = False
    for it in R.load_items(a.split, a):
        o = out / f"{it['id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); c = it["cls"]; other = "f1" if it["fold"] == 0 else "f0"
        sets = {"A": "A_all" if a.split == "val" else f"A_{other}", "S": "S_all" if a.split == "val" else f"S_{other}",
                "B": "B_all", "C": "C_all"}
        z0, eps, pe = encode(pipe, rec, np.asarray(Image.open(VOC / f"JPEGImages/{it['img']}.jpg").convert("RGB")),
                             f"A photograph of a {W[c]}", W[c])
        RAW, QK, WA3, miss, gmean = {}, {}, {}, {}, {}
        for t in TS:
            call(pipe, rec, z0, eps, pe, t)
            w_a3 = R.layer_weights(rec); Q = R.qbar(rec); WA3[t] = w_a3.cpu().numpy()
            G = {}
            for src, sname in sets.items():
                P = PF[t][sname]; gd, n = {}, 0
                for m, q in Q.items():
                    g = R.gate_for(P, c, m, q, 4)
                    if g is None:
                        g = torch.ones(q.shape[0], device=DEV); n += 1
                    gd[m] = g
                G[src] = gd; miss[f"t{t}_{src}"] = n
                gmean[f"t{t}_{src}"] = float(torch.stack([gd[m].mean() for m in Q]).mean())
            for name, ca, src in SEEDS:
                RAW[(t, name)] = raw_seed(rec, w_a3 if ca == "w" else w_u, G[src] if src else None)
            QK[t] = {g: rec.grid_qk[g] for g in R.SA}
        maps = {}
        H = {t: {g: _attention(QK[t][g], g) for g in R.SA} for t in TS}
        for name, _, _ in SEEDS:
            for t in TS:
                s = mm(RAW[(t, name)]); maps[f"t{t}:{name}|seed"] = s
                for k, v in R.propagate(H[t], s, v3=(name == "w-")).items():
                    maps[f"t{t}:{name}|{k}"] = v
            s = mm(sum(RAW[(t, name)] for t in TS) / len(TS)); maps[f"M:{name}|seed"] = s
            for k, v in prop_multi([H[t] for t in TS], s, v3=(name == "w-")).items():
                maps[f"M:{name}|{k}"] = v
        if a.tag and not checked:                                                  # smoke checks
            from src.segmentation import sg_combined_v2
            ref, _ = sg_combined_v2.aggregate({g: [QK[100][g]] * 11 for g in R.SA}, maps["t100:w-|seed"])
            d1 = max(float(np.abs(st.mm(maps[f"t100:w-|{k}"].cpu().numpy()) - st.mm(ref[k].float().cpu().numpy())).max())
                     for k in ("V0", "V1", "V3"))
            s3 = mm(sum(RAW[(100, "wA4")] for _ in TS) / len(TS))
            p3 = prop_multi([H[100]] * 3, s3, v3=True)
            d2 = max([float((s3 - maps["t100:wA4|seed"]).abs().max())] +
                     [float(np.abs(st.mm(p3[k].cpu().numpy()) - st.mm(maps[f"t100:wA4|{k}"].cpu().numpy())).max())
                      for k in ("V0", "V1")])
            print(f"identity vs aggregate (t100 V0/V1/V3): {d1:.2e}; M with 3 identical calls vs t100: {d2:.2e}", flush=True)
            assert d1 < 1e-5 and d2 < 1e-5, "propagation identity failed"
            checked = True
        arr = np.stack([maps[k].float().cpu().numpy() for k in MAPS])
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite map for {it['id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=arr.astype(np.float16), names=np.array(MAPS), cls=c, ci=it["ci"], img=it["img"], fold=it["fold"],
                 split=a.split, **{f"w_a3_t{t}": WA3[t] for t in TS}, **{f"miss_{k}": v for k, v in miss.items()},
                 **{f"gmean_{k}": v for k, v in gmean.items()})
        os.replace(tmp, o)
        print(f"[realvoc_t:{a.split}] {it['id']} {c} {time.perf_counter() - t0:.1f}s", flush=True)
        rec.clear(); del H, QK, RAW
        if DEV.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- score / select (as run 32, larger map set)
def score_one(job):
    f, o, thr = Path(job[0]), Path(job[1]), job[2]
    if o.exists():
        return
    torch.set_num_threads(1)
    from PIL import Image
    z = np.load(f); img = str(z["img"]); ci = int(z["ci"])
    gt = np.asarray(Image.open(VOC / f"SegmentationClass/{img}.png")).astype(np.int64)
    truth = gt == ci; valid = gt != 255; other = valid & (gt > 0) & ~truth
    res = {"cls": str(z["cls"]), "img": img, "fold": int(z["fold"])}
    rgb = None
    for i, nm in enumerate(str(x) for x in z["names"]):
        x = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if thr is not None and nm in CRF_MAPS:
            from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
            _import_densecrf(st.CRF_SITE)
            if rgb is None:
                rgb = np.asarray(Image.open(VOC / f"JPEGImages/{img}.jpg").convert("RGB"), np.uint8)
            ms = np.stack([apply_ovam_densecrf(rgb, x >= t, st.CRF_SITE) for t in (thr[nm], float(THR[T55]))]).astype(bool)
            res[f"{nm}|crf"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    thr = json.loads((WORK / f"thr{a.tag}.json").read_text())["thr"] if a.split == "val" else None
    sc = WORK / f"cap_{a.split}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(sc / f.name), thr) for f in sorted((WORK / f"cap_{a.split}{a.tag}").glob("*.npz"))
            if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
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
         "grp": np.array([str(z["img"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k:
            S[k] = np.stack([z[k] for z in Z])
    return S


def select(a):
    S = load_scores("train", a.tag); one = np.ones(len(S["cls"]), bool); thr, tr = {}, {}
    for nm in MAPS:
        cv = cc.miou(S[f"{nm}|raw"], S["cls"], one); i = int(np.argmax(cv))
        thr[nm] = float(THR[i]); tr[nm] = float(cv[i])
    (WORK / f"thr{a.tag}.json").write_text(json.dumps({"thr": thr, "train_miou": tr, "items": int(len(S["cls"]))}, indent=1))
    print(f"selected {len(thr)} thresholds")


# ---------------------------------------------------------------- report
def report(a):
    Tr = json.loads((WORK / f"thr{a.tag}.json").read_text()); thr = Tr["thr"]
    S = load_scores("val", a.tag); BS = R.Boot(S, a.boot); L = []; w = L.append
    ti = {nm: int(np.flatnonzero(np.isclose(THR, thr[nm]))[0]) for nm in MAPS}
    RULES = {"sel_crf": "+ DenseCRF, train-selected (primary)", "055_crf": "+ DenseCRF, 0.55",
             "sel_raw": "raw, train-selected", "055_raw": "raw, 0.55"}

    def P(nm, rule):
        if rule == "sel_raw":
            return S[f"{nm}|raw"][:, ti[nm]]
        if rule == "055_raw":
            return S[f"{nm}|raw"][:, T55]
        return S[f"{nm}|crf"][:, 0 if rule == "sel_crf" else 1]
    cache = {}

    def m(nm, rule):
        if (nm, rule) not in cache:
            cache[(nm, rule)] = BS.stat(P(nm, rule))
        return cache[(nm, rule)]
    ci = lambda d: (float(d[0]), float(np.percentile(d[1], 2.5)), float(np.percentile(d[1], 97.5)))
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    verdict = lambda x: "**helps**" if x[1] > 0 else "**hurts**" if x[2] < 0 else "not resolved"

    def diff(x, y, rule="sel_crf"):
        return ci((m(x, rule)[0] - m(y, rule)[0], m(x, rule)[1] - m(y, rule)[1]))

    def effect(arm, src, comp, rule, kind):
        idx = COMP.index(comp)
        if kind == "alone":
            on = [False] * 3; on[idx] = True; pairs = [(tuple(on), (False,) * 3)]
        elif kind == "stack":
            off = [True] * 3; off[idx] = False; pairs = [((True,) * 3, tuple(off))]
        else:
            pairs = []
            for o1 in (False, True):
                for o2 in (False, True):
                    rest = iter((o1, o2)); base = [None if j == idx else next(rest) for j in range(3)]
                    on = list(base); on[idx] = True; off = list(base); off[idx] = False
                    pairs.append((tuple(on), tuple(off)))
        pt = np.mean([m(cell(arm, src, *x), rule)[0] - m(cell(arm, src, *y), rule)[0] for x, y in pairs])
        bs = np.mean([m(cell(arm, src, *x), rule)[1] - m(cell(arm, src, *y), rule)[1] for x, y in pairs], 0)
        return ci((pt, bs))

    nimg = len(np.unique(S["grp"]))
    w("# Run 33: noise levels and profile sources on real PASCAL VOC photos (`PREREG.md`)\n")
    w(f"**Exploratory; second look at VOC val** (run 32 reported it first). VOC 2012 val: {nimg} photos, {len(S['cls'])} items. "
      f"Thresholds chosen on VOC train ({Tr['items']} items, raw maps). One capture: three SD1.5 calls per item at "
      "t = 100, 300, 500 with one shared noise draw; arms read one call or all three. Class-macro mIoU over items; "
      f"95% photo-bootstrap intervals ({a.boot} samples, paired).\n")

    w("\n## 1. Primary contrasts (val, + DenseCRF, train-selected threshold)\n")
    w("| # | Contrast | Δ mIoU [95% CI] | Reading |"); w("|---|---|---|---|")
    PRI = [("P1", "multi − t100, full stack (source A)", "M:wA4|V1", "t100:wA4|V1"),
           ("P2", "multi − t100, BRACE (capped hierarchy, no gate)", "M:w-|V3", "t100:w-|V3")]
    for k, lab, x, y in PRI:
        d = diff(x, y); w(f"| {k} | {lab} | {f3(d)} | {verdict(d)} |")
    d = effect("M", "A", "H", "sel_crf", "avg"); w(f"| P3 | hierarchy, average effect, multi, source A | {f3(d)} | {verdict(d)} |")
    for k, lab, x, y in (("P4", "C − B: real vs generated fit images (both captions)", "t100:wC4|V1", "t100:wB4|V1"),
                         ("P5", "S − C: labels vs captions (real, 490 each)", "t100:wS4|V1", "t100:wC4|V1"),
                         ("P6", "A − S: 2,170 vs 490 fit items (labels)", "t100:wA4|V1", "t100:wS4|V1")):
        d = diff(x, y); w(f"| {k} | {lab}, t = 100, full stack | {f3(d)} | {verdict(d)} |")

    w("\n## 2. Noise levels\n")
    w("| Map | " + " | ".join(ARM_NAME[x] for x in ARMS) + " |"); w("|---|" + "---:|" * len(ARMS))
    for nm, lab in (("u-|V0", "all off (uniform, fine-only, no gate)"), ("w-|V3", "BRACE (capped hierarchy)"),
                    ("w-|V1", "SA weights + hierarchy"), ("wA4|V1", "full stack, source A"), ("wS4|V1", "full stack, source S"),
                    ("wB4|V1", "full stack, source B"), ("wC4|V1", "full stack, source C")):
        w(f"| {lab} | " + " | ".join(f"{m(f'{x}:{nm}', 'sel_crf')[0]:.3f}" for x in ARMS) + " |")
    w("\nΔ vs t = 100 (primary rule):\n")
    w("| Map | t = 300 | t = 500 | multi |"); w("|---|---|---|---|")
    for nm, lab in (("w-|V3", "BRACE"), ("wA4|V1", "full stack, A"), ("wC4|V1", "full stack, C"), ("u-|V0", "all off")):
        w(f"| {lab} | " + " | ".join(f3(diff(f"{x}:{nm}", f"t100:{nm}")) for x in ("t300", "t500", "M")) + " |")
    w("\nAverage effect of each component per noise level (primary rule):\n")
    w("| Component | Source | " + " | ".join(ARM_NAME[x] for x in ARMS) + " |"); w("|---|---|" + "---|" * len(ARMS))
    for comp in COMP:
        for src in (("A", "B", "C") if comp == "G" else ("A",)):
            w(f"| {CNAME[comp]} | {src} | " + " | ".join(f3(effect(x, src, comp, 'sel_crf', 'avg')) for x in ARMS) + " |")

    w("\n## 3. Profile sources\n")
    w("Full stack (SA weights + hierarchy + 4 profiles), primary rule:\n")
    w("| Source | " + " | ".join(ARM_NAME[x] for x in ARMS) + " |"); w("|---|" + "---:|" * len(ARMS))
    w("| no gate | " + " | ".join(f"{m(f'{x}:w-|V1', 'sel_crf')[0]:.3f}" for x in ARMS) + " |")
    for src in ("A", "S", "C", "B"):
        w(f"| {SRC_NAME[src]} | " + " | ".join(f"{m(f'{x}:w{src}4|V1', 'sel_crf')[0]:.3f}" for x in ARMS) + " |")
    w("\nPaired source contrasts, full stack, multi (secondary):\n")
    w("| Contrast | Δ [95% CI] | Reading |"); w("|---|---|---|")
    for lab, x, y in (("C − B", "M:wC4|V1", "M:wB4|V1"), ("S − C", "M:wS4|V1", "M:wC4|V1"), ("A − S", "M:wA4|V1", "M:wS4|V1")):
        d = diff(x, y); w(f"| {lab} | {f3(d)} | {verdict(d)} |")

    for arm in ("t100", "M"):
        for src in ("A", "B", "C"):
            w(f"\n### Factorial, {ARM_NAME[arm]}, source {SRC_NAME[src]}\n")
            w("| SA weights | Hierarchy | 4 profiles | " + " | ".join(RULES.values()) + " | train thr |")
            w("|:-:|:-:|:-:|" + "---:|" * len(RULES) + "---:|")
            for W_ in (False, True):
                for H_ in (False, True):
                    for G_ in (False, True):
                        nm = cell(arm, src, W_, H_, G_)
                        w(f"| {'✓' if W_ else ''} | {'✓' if H_ else ''} | {'✓' if G_ else ''} | "
                          + " | ".join(f"{m(nm, r)[0]:.3f}" for r in RULES) + f" | {thr[nm]:.3f} |")
            w("\n| Component | alone | in the stack | average |"); w("|---|---|---|---|")
            for comp in COMP:
                w(f"| {CNAME[comp]} | " + " | ".join(f3(effect(arm, src, comp, 'sel_crf', k)) for k in ("alone", "stack", "avg")) + " |")

    w("\n## 4. Descriptive\n")
    w("Seed only (raw, train-selected):\n")
    w("| Seed | " + " | ".join(ARM_NAME[x] for x in ARMS) + " |"); w("|---|" + "---:|" * len(ARMS))
    for nm, lab in (("u-|seed", "uniform"), ("w-|seed", "SA weights"), ("wA4|seed", "SA weights + gate A"), ("wC4|seed", "SA weights + gate C")):
        w(f"| {lab} | " + " | ".join(f"{m(f'{x}:{nm}', 'sel_raw')[0]:.3f}" for x in ARMS) + " |")
    w("\nPer class IoU, primary rule, full stack per source:\n")
    hdr = [("t100:u-|V0", "all off, t100"), ("t100:wA4|V1", "A, t100"), ("t100:wS4|V1", "S, t100"), ("t100:wC4|V1", "C, t100"),
           ("t100:wB4|V1", "B, t100"), ("M:wA4|V1", "A, multi"), ("M:wC4|V1", "C, multi")]
    w("| Class | items | " + " | ".join(h for _, h in hdr) + " |"); w("|---|---:|" + "---:|" * len(hdr))
    for c in CLASSES:
        sel = S["cls"] == c
        if sel.any():
            w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{cc.miou(P(nm, 'sel_crf')[:, None], S['cls'], sel)[0]:.3f}" for nm, _ in hdr) + " |")
    w("\nRun 32 (different capture, descriptive only, primary rule): all off 0.596, SA weights + hierarchy 0.610, "
      "full stack A 0.646, full stack B 0.623. Compare with the t = 100 column above.")
    fits = {t: torch.load(WORK / f"fit_t{t}{a.tag}.pt")["sets"] for t in TS}
    w(f"\nFit set sizes (items): {fits[100]}. Profiles with no absent-class pool fall back to g = 1.\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("items", "dump", "fit", "capture", "score", "select", "report"):
        p = sub.add_parser(cmd)
        p.add_argument("--src", choices=("A", "B", "C"), default="A"); p.add_argument("--split", choices=("train", "val"), default="val")
        p.add_argument("--t", type=int, choices=TS, default=100)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-items", type=int, default=0); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    {"items": items, "dump": dump, "fit": fit, "capture": capture, "score": score, "select": select, "report": report}[a.cmd](a)
