#!/usr/bin/env python3
"""runs/43_generated_benchmarks/PREREG.md: the new pipeline (uniform CA seed, MK4-LR at gamma 1, fine-only V0, fixed
0.55) on VOC-Sim and COCO-Cap, with NERVE (scripts/jc_nerve43.py) and SCLIP on the same regenerated PNGs.

split    (CPU)  halves, labels, word_included, prompt mentions                  -> <WORK>/split{tag}.json, labels{tag}.json
dump     (GPU)  pass 1: per image, per module, object-query sum and the top-1% queries of every class word
                                                                                  -> <WORK>/dump{tag}/{set}/{id}.npz
fit      (CPU)  profiles: H (other half, same set), V (all VOC-Sim, COCO-Cap only), F (per image: 5 pos + 100 neg)
                                                                                  -> <WORK>/fit{tag}.pt
capture  (GPU)  pass 2: maps {seed, V0, V3} for every seed arm; PNG for the baselines
                                                                                  -> <WORK>/{set}{tag}/, png{tag}/{set}/
sclip    (GPU)  SCLIP ViT-B/16 VOC21 as released (port sg_clip_dense.py)           -> <WORK>/sclip{tag}/{set}/
score    (CPU)  raw counts on the threshold grid, DenseCRF at 0.55, baselines      -> <WORK>/{set}{tag}_scores/
report   (CPU)                                                                    -> runs/43_generated_benchmarks/result{tag}.md
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, zlib
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_gen38 as G38  # noqa: E402  (run 38: seeds, propagation, pool runner)
import jc_realvoc as R  # noqa: E402  (photo bootstrap)

G36, M = G38.G36, G38.M
T, J, AB, cc, st, nr = M.T, M.J, M.AB, M.cc, M.st, M.nr
WORK = J.CACHE / "jc_eval43"
OUT = ROOT / "runs/43_generated_benchmarks"
SETS, CLASSES, THR, T55 = ("voc", "coco"), M.CLASSES, M.THR, M.T55
SETN = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
NPOS, NNEG, KC, GAMMA = 5, 100, 4, 1.0
SMOKE_NPOS, SMOKE_NNEG = 2, 6
DEV = M.DEV
SRCS = {"voc": ("H", "F"), "coco": ("H", "F", "V")}
PROPS = ("seed", "V0", "V3")
GN = {"-": "no gate", "M": "MK4", "L": "MK4-LR"}
h = lambda s: zlib.crc32(s.encode())


def seeds_for(ds):
    return ("u-", "w-") + tuple(f"u{s}{g}" for s in SRCS[ds] for g in "ML") + ("wHL",)


def maps_for(ds):
    return tuple(f"{s}|{p}" for s in seeds_for(ds) for p in PROPS)


def crf_maps(ds):
    return tuple(f"{s}|V0" for s in ("u-", "uHM", "uHL", "uFL") + (("uVL",) if ds == "coco" else ())) + ("u-|V3", "uHL|V3")


def npos_nneg(tag):
    return (SMOKE_NPOS, SMOKE_NNEG) if tag else (NPOS, NNEG)


# ---------------------------------------------------------------- split
def all_rows(ds, tag, n):
    Rw = nr.rows(ds)
    if tag:                                    # smoke: a fixed subset of n images per set
        Rw = sorted(Rw, key=lambda r: h(r["image_id"] + "smoke"))[:n]
    return Rw


def split(a):
    S, L = {}, {}
    for ds in SETS:
        by = {}
        for r in all_rows(ds, a.tag, a.max_images):
            lab = AB.label(r); wi = str(r.get("word_included", "True")).lower() == "true"
            S[r["image_id"]] = {"ds": ds, "cls": lab, "wi": wi, "mention": [bool(T.mentions(r["prompt"], c)) for c in CLASSES]}
            by.setdefault(lab, []).append(r["image_id"]); L[r["image_id"]] = lab
        for c, ids in by.items():
            for k, i in enumerate(sorted(ids, key=lambda x: h(x + "43"))):
                S[i]["half"] = "H1" if k % 2 == 0 else "H2"
        v = [x for x in S.values() if x["ds"] == ds]
        print(ds, len(v), {hh: sum(x["half"] == hh for x in v) for hh in ("H1", "H2")}, "word not in prompt:",
              sum(not x["wi"] for x in v), flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"split{a.tag}.json").write_text(json.dumps(S, sort_keys=True)); (WORK / f"labels{a.tag}.json").write_text(json.dumps(L))


def load_split(tag):
    return json.loads((WORK / f"split{tag}.json").read_text())


def my_rows(a):
    Sp = load_split(a.tag)
    return [r for r in nr.rows(a.set) if r["image_id"] in Sp][a.shard::a.num_shards]


# ---------------------------------------------------------------- pass 1: dump
@torch.no_grad()
def dump(a):
    rt = J.make_runtime(G36.MODEL); K = T.class_keys(rt, CLASSES); Sp = load_split(a.tag)
    out = WORK / f"dump{a.tag}" / a.set; out.mkdir(parents=True, exist_ok=True)
    for row in my_rows(a):
        iid = row["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, a.set); lab = AB.label(row)
        assert lab == Sp[iid]["cls"]
        Q = T.qbar(rec); mods = list(Q); _, Tm = T.ca_seeds(rec, K, tbar=True)
        d = {"modules": np.array(mods), "label": lab, "ds": a.set}
        for i, m in enumerate(mods):
            q = Q[m].float(); N = q.shape[0]; ko, kn = max(1, round(T.TOP_OBJ * N)), max(1, round(T.TOP_SPUR * N))
            d[f"o_{i:02d}"] = q[Tm[(lab, m)].float().topk(ko).indices].sum(0).cpu().numpy().astype(np.float32)
            d[f"ko_{i:02d}"] = ko
            d[f"n_{i:02d}"] = torch.stack([q[Tm[(c, m)].float().topk(kn).indices] for c in CLASSES]).half().cpu().numpy()
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **d); os.replace(tmp, o)
        print(f"[e43:dump:{a.set}] {iid} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


# ---------------------------------------------------------------- fit
_G = {}


def _prof(job):
    """job = (key, pos indices, neg indices, class index) -> (key, mo, C) on the current module's arrays."""
    key, pos, neg, ci = job
    torch.set_num_threads(1)
    O, KO, NR = _G["O"], _G["KO"], _G["NR"]
    mo = sum(O[j] for j in pos) / float(sum(KO[j] for j in pos)) if pos else None
    C = None
    if neg:
        X = F.normalize(torch.from_numpy(np.concatenate([NR[j][ci] for j in neg]).astype(np.float32)), dim=1)
        C, _, _ = M.cluster(X, KC)
    return key, (None if mo is None else torch.from_numpy(np.asarray(mo, np.float32))), C


def refs(Sp, ids, tag):
    """Reference sets: H per (set, half of the evaluated image); V; F per image. Lists of indices into ids."""
    npos, nneg = npos_nneg(tag); ix = {i: k for k, i in enumerate(ids)}
    meta = [Sp[i] for i in ids]; RF = {"H": {}, "V": {}, "F": {}}
    for ds in SETS:
        for hh in ("H1", "H2"):
            pool = [k for k, x in enumerate(meta) if x["ds"] == ds and x["half"] != hh]
            RF["H"][(ds, hh)] = pool
    RF["V"]["all"] = [k for k, x in enumerate(meta) if x["ds"] == "voc"]
    for i in ids:
        x = Sp[i]; ci = CLASSES.index(x["cls"]); rng = np.random.default_rng(h(i))
        pos = sorted(k for k, y in enumerate(meta) if y["ds"] == x["ds"] and y["cls"] == x["cls"] and ids[k] != i)
        neg = sorted(k for k, y in enumerate(meta) if y["ds"] == x["ds"] and y["cls"] != x["cls"] and not y["mention"][ci] and ids[k] != i)
        pos = [pos[k] for k in rng.permutation(len(pos))[:npos]]; neg = [neg[k] for k in rng.permutation(len(neg))[:nneg]]
        assert ix[i] not in pos and ix[i] not in neg
        RF["F"][i] = (pos, neg)
    return RF


def fit(a):
    Sp = load_split(a.tag)
    fs = sorted(f for ds in SETS for f in (WORK / f"dump{a.tag}" / ds).glob("*.npz") if not f.stem.endswith(".tmp"))
    ids = [f.stem for f in fs]; meta = [Sp[i] for i in ids]
    missing = sorted(set(Sp) - set(ids))
    if missing:
        raise SystemExit(f"{len(missing)} images without dump, e.g. {missing[:3]}")
    RF = refs(Sp, ids, a.tag); mods = [str(x) for x in np.load(fs[0])["modules"]]
    FIT = {"mods": mods, "H": {k: {"mo": {}, "C4": {}} for k in RF["H"]}, "V": {"mo": {}, "C4": {}},
           "F": {i: {"mo": {}, "C4": {}} for i in ids}, "refs_F": {i: ([ids[k] for k in p], [ids[k] for k in n]) for i, (p, n) in RF["F"].items()},
           "ncent": {}}
    for mi, m in enumerate(mods):
        t0 = time.perf_counter(); O, KO, NR = [], [], []
        for f in fs:
            z = np.load(f); O.append(z[f"o_{mi:02d}"]); KO.append(int(z[f"ko_{mi:02d}"])); NR.append(z[f"n_{mi:02d}"])
        _G.update(O=O, KO=KO, NR=NR); jobs = []
        for (ds, hh), pool in RF["H"].items():
            for ci, c in enumerate(CLASSES):
                pos = [k for k in pool if meta[k]["cls"] == c]; neg = [k for k in pool if meta[k]["cls"] != c and not meta[k]["mention"][ci]]
                jobs.append((("H", (ds, hh), c), pos, neg, ci))
        for ci, c in enumerate(CLASSES):
            pool = RF["V"]["all"]
            jobs.append((("V", "all", c), [k for k in pool if meta[k]["cls"] == c],
                         [k for k in pool if meta[k]["cls"] != c and not meta[k]["mention"][ci]], ci))
        for i, (pos, neg) in RF["F"].items():
            jobs.append((("F", i, Sp[i]["cls"]), pos, neg, CLASSES.index(Sp[i]["cls"])))
        with get_context("fork").Pool(a.workers) as pool:
            for (src, sub, c), mo, C in pool.imap_unordered(_prof, jobs, chunksize=8):
                if mo is None or C is None:
                    continue
                P = FIT["H"][sub] if src == "H" else FIT["V"] if src == "V" else FIT["F"][sub]
                k = (c, m) if src != "F" else m
                P["mo"][k] = mo; P["C4"][k] = C
                FIT["ncent"].setdefault((src, m), []).append(int(C.shape[0]))
        print(f"fit module {mi} ({m}): {len(jobs)} profiles, {time.perf_counter() - t0:.0f}s", flush=True)
    torch.save(FIT, WORK / f"fit{a.tag}.pt")


# ---------------------------------------------------------------- pass 2: capture
def profile_for(FIT, src, iid, x, lab, m):
    if src == "F":
        P = FIT["F"][iid]; return P["mo"].get(m), P["C4"].get(m)
    P = FIT["H"][(x["ds"], x["half"])] if src == "H" else FIT["V"]
    return P["mo"].get((lab, m)), P["C4"].get((lab, m))


@torch.no_grad()
def capture(a):
    from PIL import Image
    from src.segmentation import sg_combined_v2
    rt = J.make_runtime(G36.MODEL); FIT = torch.load(WORK / f"fit{a.tag}.pt", weights_only=False); Sp = load_split(a.tag)
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    pdir = WORK / f"png{a.tag}" / a.set; pdir.mkdir(parents=True, exist_ok=True)
    rows = my_rows(a)
    if a.tag and a.max_capture:                # smoke: half word-in-prompt, half not (COCO-Cap) when available
        yes = [r for r in rows if Sp[r["image_id"]]["wi"]]; no = [r for r in rows if not Sp[r["image_id"]]["wi"]]
        k = a.max_capture // 2 if no else a.max_capture; rows = yes[:k] + no[: a.max_capture - min(k, len(yes))]
    SEEDS = seeds_for(a.set); checked = False
    for row in rows:
        iid = row["image_id"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); x = Sp[iid]; rec, sample = J.generate(rt, row, a.set); lab = AB.label(row)
        if FIT["refs_F"][iid][0].count(iid) or FIT["refs_F"][iid][1].count(iid):
            raise AssertionError("image in its own few-shot reference set")
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        Q = {m: q.to(DEV) for m, q in T.qbar(rec).items()}; Tw, Tu = G38.tsums(rec); a3 = rec.seeds()["selected/A3"].float()
        mres = {m: math.isqrt(q.shape[0]) for m, q in Q.items()}; gates = {}; miss = {}
        for src in SRCS[a.set]:
            base = {}
            for m, q in Q.items():
                mo, C = profile_for(FIT, src, iid, x, lab, m)
                if mo is None or C is None:
                    continue
                base[m] = M.g_multi(q, mo.to(DEV), C.to(DEV))
            miss[src] = len(Q) - len(base)
            lr = G36.consensus(base, {m: (1.0 if mres[m] < 64 else 0.0) for m in base})
            gates[(src, "M")] = base; gates[(src, "L")] = lr
            for g in gates[(src, "M")].values():
                assert torch.isfinite(g).all() and float(g.min()) >= 0 and float(g.max()) <= 1
        seeds = []
        for nm in SEEDS:
            Ts = Tw if nm[0] == "w" else Tu
            seeds.append(G38.make_seed(Ts) if nm[1] == "-" else G38.make_seed(Ts, G38.powered(gates[(nm[1], nm[2])], Q, GAMMA)))
        seeds = torch.stack(seeds).float()
        fid = float((seeds[SEEDS.index("w-")] - a3).abs().max())
        if fid > 1e-2:
            raise ValueError(f"seed fidelity {fid:.2e} for {iid}")
        lr = gates[("H", "L")]; low = [m for m in gates[("H", "M")] if mres[m] < 64]
        ready = bool(low) and any(mres[m] == 64 for m in lr)
        if a.tag and not checked and ready:
            ind = torch.stack([F.interpolate(gates[("H", "M")][m].float().view(1, 1, mres[m], mres[m]), size=(64, 64),
                                             mode="bilinear", align_corners=False)[0, 0] for m in low]).mean(0).clamp(0, 1).reshape(-1)
            m64 = [m for m in lr if mres[m] == 64][0]; d_lr = float((lr[m64] - ind).abs().max())
            gd = G38.powered(lr, Q, GAMMA); tgt = (None, list(rec.target))
            ref = T.ca_seeds(rec, {"g": tgt}, gate={"g": gd}); d_s = float((seeds[SEEDS.index("wHL")] - ref["g"]).abs().max())
            print(f"identity: LR vs independent mean {d_lr:.2e}; gated seed vs ca_seeds {d_s:.2e}; A3 fidelity {fid:.2e}", flush=True)
            assert d_lr < 1e-6 and d_s < 1e-5
        V0, V3 = G38.propagate(sa, seeds)
        if a.tag and not checked and ready:
            k = SEEDS.index("uHL"); ref = sg_combined_v2.aggregate(sa, seeds[k])[0]
            mm = lambda v: (v - v.min()) / (v.max() - v.min()).clamp_min(1e-20)
            d = max(float((mm(V0[k]) - mm(ref["V0"].float())).abs().max()), float((mm(V3[k]) - mm(ref["V3"].float())).abs().max()))
            print(f"identity: batched propagation vs aggregate {d:.2e}", flush=True); assert d < 1e-5; checked = True
        maps = torch.stack([seeds, V0, V3], 1).reshape(-1, 64, 64).cpu().numpy()
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {iid}")
        img = np.asarray(sample.generated_image, np.uint8)
        Image.fromarray(img).save(pdir / f"{iid}.png")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), names=np.array(maps_for(a.set)), cls=lab, truth=np.asarray(sample.target_mask, bool),
                 image=img, half=x["half"], wi=x["wi"], **{f"miss_{s}": v for s, v in miss.items()})
        os.replace(tmp, o)
        print(f"[e43:cap:{a.set}] {iid} {x['half']} wi={x['wi']} miss {miss} fid {fid:.1e} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)
    if a.tag and rows and not checked:
        raise SystemExit("smoke: identity checks did not run (no image with H profiles)")


# ---------------------------------------------------------------- SCLIP
@torch.no_grad()
def sclip(a):
    from PIL import Image
    import sg_clip_dense as SC
    dev = torch.device("cuda"); Mdl = SC.DenseCLIP("B16", dev)
    n21, i21 = SC.split([SC.VOC21_BG] + SC.VOC20_NAMES); T21 = Mdl.text(n21)
    first = [n.split(",")[0] for n in SC.VOC20_NAMES]
    for c, n in zip(CLASSES, first):
        print(f"SCLIP name for {c}: {n}", flush=True)
    L = json.loads((WORK / f"labels{a.tag}.json").read_text())
    out = WORK / f"sclip{a.tag}" / a.set; out.mkdir(parents=True, exist_ok=True)
    for f in sorted((WORK / f"png{a.tag}" / a.set).glob("*.png"))[a.shard::a.num_shards]:
        o = out / (f.stem + ".npz")
        if o.exists():
            continue
        c = CLASSES.index(L[f.stem]); img = np.asarray(Image.open(f).convert("RGB"))
        lg = Mdl.logits(img, T21); pred, p = SC.postprocess(lg, i21, 65, 0.1, 0.1)     # repo VOC21 settings
        own = (pred == c + 1).cpu().numpy(); prob = p[c + 1].float().cpu().numpy()
        tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, own=own, prob=prob.astype(np.float16), cls=c); os.replace(tmp, o)
        print(f"[e43:sclip:{a.set}] {f.stem} {L[f.stem]} own {own.mean():.3f}", flush=True)


# ---------------------------------------------------------------- score
def score_one(job):
    f, o, ds, tag, do_crf = Path(job[0]), Path(job[1]), job[2], job[3], job[4]
    if o.exists():
        return
    torch.set_num_threads(1)
    z = np.load(f); truth = z["truth"].astype(bool); valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"cls": str(z["cls"]), "half": str(z["half"]), "wi": bool(z["wi"])}
    res.update({k: z[k] for k in z.files if k.startswith("miss_")})
    names = [str(x) for x in z["names"]]; X = {}
    for i, nm in enumerate(names):
        X[nm] = st.up(st.mm(z["maps"][i].astype(np.float32)), truth.shape)
        res[f"{nm}|raw"] = cc.counts(X[nm][None] >= THR[:, None, None], truth, valid, other)
    if do_crf:
        from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
        _import_densecrf(st.CRF_SITE)
        for nm in crf_maps(ds):
            m = apply_ovam_densecrf(z["image"], X[nm] >= THR[T55], st.CRF_SITE).astype(bool)
            res[f"{nm}|crf"] = cc.counts(m[None], truth, valid, other)[0]
    for b in ("nerve", "sclip"):
        zb = np.load(WORK / f"{b}{tag}" / ds / f.name)
        if int(zb["cls"]) != CLASSES.index(str(z["cls"])):
            raise ValueError(f"{b} class mismatch for {f.stem}")
        res[f"{b}|own"] = cc.counts(zb["own"].astype(bool)[None], truth, valid, other)[0]
        soft = zb["soft"] if b == "nerve" else zb["prob"]
        x = st.up(st.mm(soft.astype(np.float32)), truth.shape)
        res[f"{b}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
        if b == "nerve":
            res["nerve|nogate"] = cc.counts(zb["nogate"].astype(bool)[None], truth, valid, other)[0]
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    jobs = []
    for ds in SETS:
        sc = WORK / f"{ds}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs += [(str(f), str(sc / f.name), ds, a.tag, a.crf) for f in sorted((WORK / f"{ds}{a.tag}").glob("*.npz")) if not f.stem.endswith(".tmp")]
    G38._run_pool(jobs[a.shard::a.num_shards], score_one, a.workers)


# ---------------------------------------------------------------- report
def load_scores(ds, tag):
    fs = sorted(f for f in (WORK / f"{ds}{tag}_scores").glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    S = {"id": np.array([f.stem for f in fs]), "grp": np.array([f.stem for f in fs]), "cls": np.array([str(z["cls"]) for z in Z]),
         "wi": np.array([bool(z["wi"]) for z in Z])}
    for k in Z[0].files:
        if "|" in k or k.startswith("miss_"):
            S[k] = np.stack([z[k] for z in Z])
    return S


def report(a):
    Sall = {ds: load_scores(ds, a.tag) for ds in SETS}
    exp = {ds: len([1 for x in load_split(a.tag).values() if x["ds"] == ds]) for ds in SETS}
    for ds in SETS:
        if not a.tag and len(Sall[ds]["id"]) != exp[ds]:
            raise SystemExit(f"{ds}: {len(Sall[ds]['id'])} scored of {exp[ds]}")
    sub = lambda S, mk: {k: (v[mk] if isinstance(v, np.ndarray) else v) for k, v in S.items()}
    SUB = {("voc", "all"): Sall["voc"], ("coco", "all"): Sall["coco"],
           ("coco", "in"): sub(Sall["coco"], Sall["coco"]["wi"]), ("coco", "out"): sub(Sall["coco"], ~Sall["coco"]["wi"])}
    BS = {k: R.Boot(v, a.boot) for k, v in SUB.items()}; cache = {}
    COL = [("voc", "all"), ("coco", "all"), ("coco", "in"), ("coco", "out")]
    CN = {("voc", "all"): "VOC-Sim", ("coco", "all"): "COCO-Cap", ("coco", "in"): "COCO-Cap, word in prompt",
          ("coco", "out"): "COCO-Cap, word not in prompt"}

    def P(k, key, t=T55):
        D = SUB[k][key]
        return D[:, t] if D.ndim == 3 else D

    def m(k, key, t=T55):
        c = (k, key, t)
        if c not in cache:
            cache[c] = BS[k].stat(P(k, key, t))
        return cache[c]
    ci = lambda x, y: (x[0] - y[0], float(np.percentile(x[1] - y[1], 2.5)), float(np.percentile(x[1] - y[1], 97.5)))
    f3 = lambda d: f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
    rd = lambda d: "**helps**" if d[1] > 0 else "**hurts**" if d[2] < 0 else "not resolved"
    has = lambda k, key: key in SUB[k]
    L = []; w = L.append
    n = {k: (len(v["id"]), len(set(v["cls"]))) for k, v in SUB.items()}
    w("# Run 43: the new pipeline on VOC-Sim and COCO-Cap, with NERVE and SCLIP in the same capture (`PREREG.md`)\n")
    w(f"Images: VOC-Sim {n[('voc', 'all')][0]}, COCO-Cap {n[('coco', 'all')][0]} (word in prompt {n[('coco', 'in')][0]}, not in prompt "
      f"{n[('coco', 'out')][0]}). One regeneration (pass 2) gives every map and the PNG for NERVE and SCLIP. Ours: uniform "
      "cross-attention seed, MK4-LR at γ = 1, fine-only propagation (V0), min-max, **fixed threshold 0.55**. Class-macro mIoU; "
      f"95% image-bootstrap intervals ({a.boot} samples, paired).\n")

    w("\n## 1. Primary contrasts (source H, V0, raw at 0.55)\n")
    w("| # | Contrast | Set | Δ mIoU [95% CI] | Reading |\n|---|---|---|---|---|")
    PR = [("P1", "uHL|V0|raw", "u-|V0|raw", ("voc", "all"), "MK4-LR − no gate"), ("P2", "uHL|V0|raw", "u-|V0|raw", ("coco", "all"), "MK4-LR − no gate"),
          ("P3", "uHL|V0|raw", "nerve|own", ("voc", "all"), "MK4-LR − NERVE"), ("P4", "uHL|V0|raw", "nerve|own", ("coco", "all"), "MK4-LR − NERVE"),
          ("P5", "uHL|V0|raw", "u-|V0|raw", ("coco", "out"), "MK4-LR − no gate")]
    for p, x, y, k, nm in PR:
        d = ci(m(k, x), m(k, y)); w(f"| {p} | {nm} | {CN[k]} | {f3(d)} | {rd(d)} |")

    w("\n## 2. Main table (mIoU)\n")
    w("| Method | Read-out | " + " | ".join(CN[k] for k in COL) + " |\n|---|---|" + "---:|" * len(COL))
    ROWS = [("NERVE (as released, SD 2.1-base)", "own binarisation", "nerve|own"), ("SCLIP (ViT-B/16, VOC21)", "own prediction", "sclip|own"),
            ("No gate (seed + V0)", "raw 0.55", "u-|V0|raw"), ("MK4 (H)", "raw 0.55", "uHM|V0|raw"), ("**MK4-LR (H), ours**", "raw 0.55", "uHL|V0|raw"),
            ("No gate (seed + V0)", "+ DenseCRF 0.55", "u-|V0|crf"), ("MK4 (H)", "+ DenseCRF 0.55", "uHM|V0|crf"),
            ("**MK4-LR (H), ours**", "+ DenseCRF 0.55", "uHL|V0|crf")]
    for nm, ro, key in ROWS:
        w(f"| {nm} | {ro} | " + " | ".join(f"{m(k, key)[0]:.3f}" for k in COL) + " |")

    w("\n## 3. Differences for every subset (raw 0.55 unless stated)\n")
    w("| Contrast | " + " | ".join(CN[k] for k in COL) + " |\n|---|" + "---|" * len(COL))
    DF = [("MK4-LR − no gate", "uHL|V0|raw", "u-|V0|raw"), ("MK4-LR − MK4", "uHL|V0|raw", "uHM|V0|raw"), ("MK4 − no gate", "uHM|V0|raw", "u-|V0|raw"),
          ("MK4-LR − NERVE", "uHL|V0|raw", "nerve|own"), ("MK4-LR − SCLIP", "uHL|V0|raw", "sclip|own"),
          ("MK4-LR + CRF − NERVE", "uHL|V0|crf", "nerve|own"), ("MK4-LR + CRF − no gate + CRF", "uHL|V0|crf", "u-|V0|crf"),
          ("few-shot F MK4-LR − no gate", "uFL|V0|raw", "u-|V0|raw"), ("few-shot F − half split H (MK4-LR)", "uFL|V0|raw", "uHL|V0|raw")]
    for nm, x, y in DF:
        w(f"| {nm} | " + " | ".join(f3(ci(m(k, x), m(k, y))) for k in COL) + " |")
    w("| VOC-Sim → COCO-Cap transfer V, MK4-LR − no gate | — | " + " | ".join(f3(ci(m(k, "uVL|V0|raw"), m(k, "u-|V0|raw"))) for k in COL[1:]) + " |")
    w("| V − H (MK4-LR) | — | " + " | ".join(f3(ci(m(k, "uVL|V0|raw"), m(k, "uHL|V0|raw"))) for k in COL[1:]) + " |")

    w("\n## 4. Every arm (mIoU, raw 0.55)\n")
    w("| Seed | " + " | ".join(f"{p}, {CN[k]}" for p in PROPS for k in COL[:2]) + " |\n|---|" + "---:|" * (2 * len(PROPS)))
    for s in seeds_for("coco"):
        w(f"| {s} | " + " | ".join((f"{m(k, f'{s}|{p}|raw')[0]:.3f}" if has(k, f"{s}|{p}|raw") else "—") for p in PROPS for k in COL[:2]) + " |")
    w("\nSeed codes: u = uniform weights, w = SA-guided weights; - = no gate; H / F / V = profile source; M = MK4, L = MK4-LR.")
    w("\nDescriptive: hierarchy and SA weights added to MK4-LR (H), raw 0.55: " + "; ".join(
        f"{CN[k]}: V3 − V0 {f3(ci(m(k, 'uHL|V3|raw'), m(k, 'uHL|V0|raw')))}, SA weights {f3(ci(m(k, 'wHL|V0|raw'), m(k, 'uHL|V0|raw')))}" for k in COL[:2]) + ".")

    w("\n## 5. Baselines, secondary read-outs\n")
    w("| Read-out | " + " | ".join(CN[k] for k in COL) + " |\n|---|" + "---:|" * len(COL))
    for nm, key in (("NERVE own binarisation", "nerve|own"), ("NERVE without class-score gate", "nerve|nogate"),
                    ("NERVE soft map, min-max 0.55", "nerve|raw"), ("SCLIP own prediction", "sclip|own"), ("SCLIP probability, min-max 0.55", "sclip|raw")):
        w(f"| {nm} | " + " | ".join(f"{m(k, key)[0]:.3f}" for k in COL) + " |")

    w("\n## 6. Every threshold (V0, raw; † = 95% CI excludes 0)\n")
    w("| Set | Contrast | " + " | ".join(f"{t:.3f}" for t in THR) + " |\n|---|---|" + "---:|" * len(THR))
    for k in COL:
        for nm, x, y in (("MK4-LR − no gate", "uHL|V0|raw", "u-|V0|raw"), ("MK4-LR − MK4", "uHL|V0|raw", "uHM|V0|raw")):
            cells = []
            for t in range(len(THR)):
                d = ci(m(k, x, t), m(k, y, t)); cells.append(f"{d[0]:+.3f}{'†' if d[1] > 0 or d[2] < 0 else ''}")
            w(f"| {CN[k]} | {nm} | " + " | ".join(cells) + " |")
        w(f"| {CN[k]} | MK4-LR mIoU | " + " | ".join(f"{m(k, 'uHL|V0|raw', t)[0]:.3f}" for t in range(len(THR))) + " |")

    w("\n## 7. Per class (raw 0.55 for ours)\n")
    for ds in SETS:
        S = Sall[ds]; w(f"\n{SETN[ds]}:\n"); hdr = [("u-|V0|raw", "no gate"), ("uHM|V0|raw", "MK4"), ("uHL|V0|raw", "MK4-LR"),
                                                     ("nerve|own", "NERVE"), ("sclip|own", "SCLIP")]
        w("| Class | images | " + " | ".join(x for _, x in hdr) + " |\n|---|---:|" + "---:|" * len(hdr))
        better = [0, 0]
        for c in CLASSES:
            sel = S["cls"] == c
            if not sel.any():
                continue
            v = [cc.miou((S[key][:, T55] if S[key].ndim == 3 else S[key])[:, None], S["cls"], sel)[0] for key, _ in hdr]
            better[0] += v[2] > v[0]; better[1] += v[2] < v[0]
            w(f"| {c} | {int(sel.sum())} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
        w(f"\nMK4-LR beats / trails no gate in {better[0]} / {better[1]} classes.")

    FIT = torch.load(WORK / f"fit{a.tag}.pt", weights_only=False)
    w("\n## 8. Profiles\n")
    w("Centroids kept (mean over profiles) per source and module: " + "; ".join(
        f"{s}/{mi}: {np.mean(FIT['ncent'][(s, mm_)]):.2f}" for s in ("H", "V", "F") for mi, mm_ in enumerate(FIT["mods"]) if (s, mm_) in FIT["ncent"]) + ".")
    w("\nModules: " + ", ".join(f"{k} = {x}" for k, x in enumerate(FIT["mods"])))
    w("\nModules without a profile per image (mean): " + ", ".join(
        f"{SETN[ds]} {s} {float(np.mean(Sall[ds][f'miss_{s}'])):.2f}" for ds in SETS for s in SRCS[ds]) + ".")
    nm_ = json.loads((WORK / f"nerve{a.tag}" / "meta.json").read_text())
    w(f"\nNERVE: SD {nm_['sd']} ({', '.join(Path(p).name for p in nm_['sd_snapshot'])}), SFSA head size {nm_['head_dim']}, seed {nm_['seed']}.")
    w("\nBRACE stays frozen; nothing here is a promotion.\n")
    txt = "\n".join(L) + "\n"; print(txt); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub_ = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("split", "dump", "fit", "capture", "sclip", "score", "report"):
        p = sub_.add_parser(nm); p.add_argument("--tag", default=""); p.add_argument("--workers", type=int, default=8)
        p.add_argument("--set", choices=SETS, default="voc"); p.add_argument("--crf", action="store_true")
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--max-images", type=int, default=0, help="split, smoke only: images per set")
        p.add_argument("--max-capture", type=int, default=0, help="capture, smoke only"); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.tag and a.cmd == "split" and not a.max_images:
        raise SystemExit("smoke split needs --max-images")
    WORK.mkdir(parents=True, exist_ok=True)
    {"split": split, "dump": dump, "fit": fit, "capture": capture, "sclip": sclip, "score": score, "report": report}[a.cmd](a)
