#!/usr/bin/env python3
"""Run 45 pilot: BLIP-caption prompts for our method, on PASCAL Context TRAIN photos (development data; env trdp).

Question: does image-specific caption conditioning improve our maps, especially for stuff classes (sky, road, grass...)?
Everything except the prompt (and, for arm B, how the word is read) is run 44's frozen method: one SD 1.5 pass per item
at t = 100, noise seed 0; uniform CA seed; MK4 gate with run 44's Context profiles (jc_ov42 fit.pt "20" + jc_ours44
fit44.pt, built with the template prompt); MK4-LR at gamma 1; fine-only V0; min-max; fixed 0.55 (raw).

Arms (prompt | how the word's cross-attention is read):
  A   "A photograph of a {word}"  | in-prompt: the word's tokens in this prompt (= run 44)
  At  "A photograph of the {word}"| in-prompt (grammar check: "a sky" vs "the sky")
  C   BLIP-large caption conditioned on the prefix "a photograph of {word}" (DiffSegmenter's recipe, no "++")
                                  | in-prompt: first occurrence of the word (always present: the prefix)
  B   BLIP-large unconditional caption of the photo (same text for every class of the photo)
                                  | out-of-prompt (OVAM-style): image queries against the keys of the bare word prompt
                                    (jc_ov42.word_attn); the word need not be in the caption
Secondary maps from the same passes: A|op (template pass, out-of-prompt read: separates prompt from read-out) and
B|ip (in-prompt read, only when the caption contains the word's tokens).
Per pass the gate's queries are compared with arm A's (mean row cosine per module): the profiles were built with the
template prompt, so this measures how far a caption prompt moves the gate's input.

items    (CPU) 1,000 Context train photos (seed 45; not in the 5,104 val list or VOC2012 seg val); every class present,
               primary = area >= 1%                                             -> <WORK>/items.json
caption  (GPU) BLIP-large (Salesforce/blip-image-captioning-large, the DiffSegmenter driver's checkpoint), greedy, its
               default length: per photo (B) and per item (C)                   -> <WORK>/captions.json
capture  (GPU) every arm's maps (u-, uM, uL; seed and V0)                         -> <WORK>/cap/<item id>.npz
check    (GPU) arm A on 3 run 44 Context val items reproduces run 44's stored maps
score    (CPU) counts on the 0.30-0.70 grid; seed / V0 AUROC (target vs other valid pixels)  -> <WORK>/scores/
report   (CPU)                                                                  -> runs/45_blip_prompts/result.md
"""
from __future__ import annotations

import argparse, json, math, os, random, re, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
WORK = Path(os.environ.get("BLIP45_WORK", str(CACHE / "jc_blip45")))      # BLIP45_WORK: separate directory for the smoke
REPORT = ROOT / "runs/45_blip_prompts/result.md"
BLIP = "/home/shafa/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-large/snapshots/353689b859fcf0523410b1806dace5fb46ecdf41"
CTX_SEG = ROOT / "datasets/pascal_context/SegmentationClass"
VOC_JPG = ROOT / "datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"
CTX_VAL = ROOT / "other repositories/nerve/data/voc/context_val.txt"
VOC_VAL = ROOT / "datasets/voc2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
OVD_MM = ROOT / "other repositories/ovdiff/mmdata.py"
N_PHOTOS, SEED, MIN_AREA = 1000, 45, 0.01
ARMS = ("A|ip", "At|ip", "C|ip", "B|op", "A|op", "B|ip")
PRIMARY_ARMS = ("At|ip", "C|ip", "B|op")
SEEDS = ("u-", "uM", "uL")
MAPS = tuple(f"{s}|{p}" for s in SEEDS for p in ("seed", "V0"))


def groups():
    """Context class -> 'voc' | 'thing' | 'stuff' (stuff/thing from OVDiff's released Context table, fixed in PREREG)."""
    from jc_real44 import VOC20, CTX59
    txt = OVD_MM.read_text(); blk = txt[txt.index('CHAT_GPT_REPLY_LONGER = """') + 27:]; blk = blk[: blk.index('"""')]
    IS = {l.split(":")[0].strip(): l.split(":")[-1].strip() == "stuff" for l in blk.strip().splitlines()}
    G = {c: "voc" if c in VOC20 else ("stuff" if IS[c] else "thing") for c in CTX59}
    assert sum(v == "stuff" for v in G.values()) == 19 and sum(v == "thing" for v in G.values()) == 20, G
    return G


def items(a):
    from PIL import Image
    from jc_real44 import CTX59
    val = {l.strip() for l in open(CTX_VAL) if l.strip()} | {l.strip() for l in open(VOC_VAL) if l.strip()}
    ids = sorted(p.stem for p in CTX_SEG.glob("*.png") if p.stem not in val and (VOC_JPG / f"{p.stem}.jpg").exists())
    print(f"Context train photos (not in Context val / VOC seg val): {len(ids)}", flush=True)
    pick = sorted(random.Random(SEED).sample(ids, N_PHOTOS)); G = groups(); X = []
    for v in pick:
        m = np.array(Image.open(CTX_SEG / f"{v}.png"))
        for i in np.unique(m):
            i = int(i)
            if 0 < i <= 59:
                ar = float((m == i).mean())
                X.append({"id": f"ctx45_{v}_{i:02d}", "img": v, "path": str((VOC_JPG / f"{v}.jpg").relative_to(ROOT)),
                          "cls": CTX59[i - 1], "ci": i, "area": ar, "primary": ar >= MIN_AREA, "grp": G[CTX59[i - 1]]})
    P = [x for x in X if x["primary"]]
    WORK.mkdir(parents=True, exist_ok=True); (WORK / "items.json").write_text(json.dumps({"photos": pick, "items": X}))
    print(f"items {len(X)}, primary {len(P)} on {len({x['img'] for x in P})} photos; by group "
          + str({g: sum(x['grp'] == g for x in P) for g in ('voc', 'thing', 'stuff')}), flush=True)


def word_of(c):
    import jc_ours44 as U
    import jc_realvoc as R
    return U.keyword("context", c, R.word_map())[1]


def caption(a):
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    dev = torch.device("cuda")
    proc = BlipProcessor.from_pretrained(BLIP); model = BlipForConditionalGeneration.from_pretrained(BLIP).to(dev).eval()
    J = json.loads((WORK / "items.json").read_text()); X = [x for x in J["items"] if x["primary"]]
    photos = sorted({x["img"] for x in X})
    if a.limit: photos = photos[:a.limit]; X = [x for x in X if x["img"] in set(photos)]
    out = WORK / "captions.json"
    C = {"photo": {}, "item": {}}; img = {}
    with torch.no_grad():
        for v in photos:
            img[v] = Image.open(ROOT / f"datasets/voc2012/VOCdevkit/VOC2012/JPEGImages/{v}.jpg").convert("RGB")
            C["photo"][v] = proc.decode(model.generate(**proc(img[v], return_tensors="pt").to(dev))[0], skip_special_tokens=True)
        for x in X:
            base = f"a photograph of {word_of(x['cls'])}"
            C["item"][x["id"]] = proc.decode(model.generate(**proc(img[x["img"]], base, return_tensors="pt").to(dev))[0],
                                             skip_special_tokens=True)
    out.write_text(json.dumps(C, indent=0))
    for v in photos[:5]:
        print(v, "|", C["photo"][v], "|", [C["item"][x["id"]] for x in X if x["img"] == v][:3], flush=True)
    print(f"captions: {len(C['photo'])} photos, {len(C['item'])} items -> {out}", flush=True)


def prompts(x, cap, w):
    base = f"a photograph of {w}"; c = cap["item"][x["id"]]
    pc = c if c.startswith(base) else f"{base} {c}"
    return {"A": f"A photograph of a {w}", "At": f"A photograph of the {w}", "C": pc, "B": cap["photo"][x["img"]]}


class Runner:
    def __init__(self):
        import torch
        import jc_ov42 as O
        import jc_ours44 as U
        import jc_realvoc as R
        from jc_real44 import CTX59
        self.torch, self.O, self.U, self.R = torch, O, U, R
        self.pipe, self.rec = R.load()
        words = sorted({word_of(c) for c in CTX59})
        self.K = O.T.class_keys({"pipe": self.pipe, "ca_modules": self.rec.ca_order}, words)
        P42 = torch.load(O.WORK / "fit.pt", weights_only=False)["fit"]["20"]
        P44 = torch.load(U.WORK / "fit44.pt", weights_only=False)["fit"]
        self.P = O.M.to_dev({"mo": {**P42["mo"], **P44["mo"]}, "neg": {**P42["neg"], **P44["neg"]}})
        self.w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=R.DEV)

    def one_pass(self, img_u8, prompt, word):
        """jc_realvoc.one_pass, except that the word may be absent from the prompt (then rec.ca is not used)."""
        torch, R, pipe, rec = self.torch, self.R, self.pipe, self.rec
        import torch.nn.functional as F
        from src.segmentation.attention_aggregation import find_token_subsequence
        x = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1)[None].float().to(R.DEV) / 127.5 - 1
        x = F.interpolate(x, size=(R.PX, R.PX), mode="bicubic", align_corners=False).clamp(-1, 1)
        z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
        eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(R.NOISE_SEED)).to(R.DEV, z0.dtype)
        ac = pipe.scheduler.alphas_cumprod.to(R.DEV)
        zt = ac[R.T_NOISE].sqrt() * z0 + (1 - ac[R.T_NOISE]).sqrt() * eps
        pe, _ = pipe.encode_prompt(prompt=prompt, device=R.DEV, num_images_per_prompt=1, do_classifier_free_guidance=False)
        ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
        try:
            rec.target = find_token_subsequence(ids, pipe.tokenizer(word, add_special_tokens=False)["input_ids"]); found = True
        except ValueError:
            rec.target = (0,); found = False
        rec.clear()
        pipe.unet(zt, torch.tensor(R.T_NOISE, device=R.DEV), encoder_hidden_states=pe.float(), return_dict=False)
        return found

    def maps(self, key, word, reads):
        """{read: {map name: 64x64}} for this pass; reads in {'ip', 'op'}; plus the gate queries."""
        torch, O, R, rec = self.torch, self.O, self.R, self.rec
        import torch.nn.functional as F
        from src.segmentation.sg25_sa import _attention, _apply
        Q = R.qbar(rec); res = {m: math.isqrt(q.shape[0]) for m, q in Q.items()}
        base = {m: g for m, q in Q.items() if (g := O.R37.g_mk4(self.P, key, m, q)) is not None}
        B = {"M": base, "L": O.G36.consensus(base, {m: (1.0 if res[m] < 64 else 0.0) for m in base}) if base else {}}
        H64 = _attention(rec.grid_qk[64], 64); out = {}; ip = dict(rec.ca)
        for rd in reads:
            rec.ca = ip if rd == "ip" else {m: O.word_attn(rec, self.K, word, m) for m in rec.ca_order}
            mp = {}
            for nm in SEEDS:
                gd = None if nm == "u-" else O.R37.powered(B[nm[1:]], Q, 1.0)
                s = R.make_seed(rec, self.w_u, gd); u = _apply(H64, s.float().reshape(-1))
                mp[f"{nm}|seed"] = s; mp[f"{nm}|V0"] = _apply(H64, u).mean(0).reshape(64, 64)
            out[rd] = np.stack([mp[k].float().cpu().numpy() for k in MAPS])
        rec.ca = ip
        return out, Q, len(Q) - len(base)


def capture(a):
    import torch
    from PIL import Image
    J = json.loads((WORK / "items.json").read_text()); cap = json.loads((WORK / "captions.json").read_text())
    X = [x for x in J["items"] if x["primary"]][a.shard::a.num_shards]
    if a.limit: X = X[:a.limit]
    rn = Runner(); out = WORK / "cap"; out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for x in X:
            o = out / f"{x['id']}.npz"
            if o.exists() or (out / f"{x['id']}.drop.json").exists():
                continue
            t0 = time.perf_counter(); key, word = rn.U.keyword("context", x["cls"], rn.R.word_map())
            img = np.asarray(Image.open(ROOT / x["path"]).convert("RGB")); pr = prompts(x, cap, word); res = {}; qa = None; qcos = {}
            dropped = False
            for arm, reads in (("A", ("ip", "op")), ("At", ("ip",)), ("C", ("ip",)), ("B", ("op", "ip"))):
                found = rn.one_pass(img, pr[arm], word)
                if arm == "C" and not found:
                    # PREREG section 4a: BLIP rewrote the conditioning word ("bag" -> "baggy"); the item is dropped from
                    # every arm (paired), and the drop is counted in the report.
                    rn.rec.clear(); dropped = True; break
                if arm in ("A", "At"):
                    assert found, (arm, pr[arm], word)
                reads = tuple(r for r in reads if r == "op" or found)
                mp, Q, miss = rn.maps(key, word, reads)
                for r, v in mp.items():
                    res[f"{arm}|{r}"] = v
                if arm == "A":
                    qa = Q
                else:
                    qcos[arm] = float(np.mean([float((qa[m] * Q[m]).sum(-1).mean()) for m in Q]))
                rn.rec.clear()
            if dropped:
                (out / f"{x['id']}.drop.json").write_text(json.dumps({"id": x["id"], "cls": x["cls"], "word": word, "C": pr["C"]}))
                print(f"[blip45] {x['id']} {x['cls']!r} DROPPED (C lacks the word): {pr['C']!r}", flush=True)
                continue
            arr = np.stack([res[k] for k in ARMS if k in res])
            if not np.isfinite(arr).all():
                raise ValueError(f"non-finite map for {x['id']}")
            tmp = o.with_name(o.stem + ".tmp.npz")
            np.savez(tmp, maps=arr.astype(np.float16), arms=np.array([k for k in ARMS if k in res]), names=np.array(MAPS),
                     prompts=json.dumps(pr), qcos=json.dumps(qcos), miss=miss, cls=x["cls"], word=word)
            os.replace(tmp, o); torch.cuda.empty_cache()
            print(f"[blip45] {x['id']} {x['cls']!r} qcos {qcos} B|ip {'B|ip' in res} C: {pr['C'][:60]!r} "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)


def check(a):
    """Arm A (template, in-prompt) through this driver reproduces run 44's stored maps on 3 Context val items."""
    import torch
    from PIL import Image
    from jc_real44 import WORK as W44
    X = [x for x in json.loads((W44 / "items.json").read_text())["context"] if x["primary"]][:3]; rn = Runner()
    with torch.no_grad():
        for x in X:
            key, word = rn.U.keyword("context", x["cls"], rn.R.word_map())
            rn.one_pass(np.asarray(Image.open(ROOT / x["path"]).convert("RGB")), f"A photograph of a {word}", word)
            mp, _, _ = rn.maps(key, word, ("ip",)); rn.rec.clear()
            z = np.load(W44 / "ours" / "cap_context" / f"{x['id']}.npz"); names = [str(n) for n in z["names"]]
            ref = z["maps"].astype(np.float32); new = mp["ip"].astype(np.float16).astype(np.float32)
            d = {n: float(np.abs(new[MAPS.index(n)] - ref[names.index(n)]).max()) for n in names}
            print(x["id"], "max |diff| vs run 44:", d, flush=True); assert max(d.values()) < 2e-3, d
    print("check passed", flush=True)


def auroc(s, t, v, rng):
    idx = np.flatnonzero(v.reshape(-1))
    if len(idx) > 40000:
        idx = rng.choice(idx, 40000, replace=False)
    sc, lb = s.reshape(-1)[idx], t.reshape(-1)[idx]; npos = int(lb.sum()); nneg = len(lb) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = np.argsort(np.argsort(sc, kind="stable"), kind="stable") + 1
    return float((r[lb].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score_one(job):
    x, o = job; o = Path(o)
    if o.exists():
        return
    import jc_ov42 as O
    import jc_score44 as S44
    z = np.load(WORK / "cap" / f"{x['id']}.npz"); truth, valid, other = S44.truth_of("context", x); hw = truth.shape
    res = {"cls": x["cls"], "img": str(x["img"]), "qcos": str(z["qcos"])}; rng = np.random.default_rng(0)
    for ai, arm in enumerate(str(n) for n in z["arms"]):
        for mi, nm in enumerate(str(n) for n in z["names"]):
            v = O.st.up(O.st.mm(z["maps"][ai, mi].astype(np.float32)), hw)
            if nm.endswith("|V0"):
                res[f"{arm}|{nm}"] = O.cc.counts(v[None] >= O.THR[:, None, None], truth, valid, other)
            if nm in ("u-|seed", "u-|V0", "uL|V0"):
                res[f"auc:{arm}|{nm}"] = auroc(v, truth, valid, rng)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    from multiprocessing import get_context
    X = [x for x in json.loads((WORK / "items.json").read_text())["items"] if x["primary"]]
    if a.limit: X = X[:a.limit]
    out = WORK / "scores"; out.mkdir(parents=True, exist_ok=True)
    jobs = [(x, str(out / f"{x['id']}.npz")) for x in X if (WORK / "cap" / f"{x['id']}.npz").exists()]
    with get_context("fork").Pool(a.workers) as p:
        for _ in p.imap_unordered(score_one, jobs, chunksize=8):
            pass
    print(f"scored {len(jobs)} of {len(X)} items", flush=True)


def report(a):
    import jc_ov42 as O
    THR, T55 = O.THR, O.T55
    fs = sorted(f for f in (WORK / "scores").glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f, allow_pickle=False) for f in fs]
    X = {x["id"]: x for x in json.loads((WORK / "items.json").read_text())["items"] if x["primary"]}
    D = [json.loads(f.read_text()) for f in sorted((WORK / "cap").glob("*.drop.json"))]
    if not a.tag and len(Z) + len(D) != len(X):
        raise SystemExit(f"{len(Z)} scored + {len(D)} dropped of {len(X)}")
    S = {"cls": np.array([str(z["cls"]) for z in Z]), "grp": np.array([str(z["img"]) for z in Z]),
         "g": np.array([X[f.stem]["grp"] for f in fs])}
    keys = sorted({k for z in Z for k in z.files if "|" in k})
    for k in keys:
        if all(k in z.files for z in Z):
            S[k] = np.stack([z[k] for z in Z])
    has_bip = np.array(["B|ip|uL|V0" in z.files for z in Z])
    sub = lambda S, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}
    GR = {"all 59": lambda g: np.ones(len(g), bool), "stuff 19": lambda g: g == "stuff", "things 40": lambda g: g != "stuff",
          "VOC 20": lambda g: g == "voc", "other things 20": lambda g: g == "thing"}
    E = {gn: (sub(S, f(S["g"])), None) for gn, f in GR.items()}
    E = {gn: (Sg, O.Boot(Sg, a.boot)) for gn, (Sg, _) in E.items()}
    f3 = lambda d: f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
    rd = lambda d: "**helps**" if d[1] > 0 else "**hurts**" if d[2] < 0 else "not resolved"

    def stat(gn, k, t=T55):
        Sg, B = E[gn]; return B.stat(Sg[k][:, t])

    def dif(gn, x, y, t=T55):
        A, Bv = stat(gn, x, t), stat(gn, y, t); d = A[1] - Bv[1]
        return A[0] - Bv[0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    L = []; w = L.append
    w("# Run 45 pilot: BLIP-caption prompts (`runs/45_blip_prompts/PREREG.md`)\n")
    w(f"PASCAL Context **train** photos (development): {len(set(S['grp']))} photos, {len(Z)} primary items (area >= 1%). "
      "Run 44's method and profiles; only the prompt (and for B the read-out) changes. Class-macro mIoU, raw V0 at 0.55; "
      f"95% paired photo bootstrap ({a.boot}).\n")
    gx = lambda g: sum(1 for x in X.values() if (x["grp"] == "stuff") == (g == "stuff"))
    gd = lambda g: sum(1 for d in D if (X[d["id"]]["grp"] == "stuff") == (g == "stuff"))
    w(f"**Dropped (PREREG 4a): {len(D)} of {len(X)} items**, where BLIP's conditional caption no longer contains the word's "
      f"tokens. They are dropped from every arm: stuff {gd('stuff')} / {gx('stuff')}, things {gd('things')} / {gx('things')}. "
      "Every number below is on the kept items only, so arm C is conditioned on BLIP keeping the word.\n")
    from collections import Counter
    if D:
        w("Dropped by class: " + ", ".join(f"{c} {n}" for c, n in Counter(d["cls"] for d in D).most_common()) + ". Examples: "
          + "; ".join(f"'{d['C']}'" for d in D[:5]) + ".\n")
    w("\n## 1. Primary contrasts (MK4-LR, raw 0.55): arm - A\n")
    w("| Arm | stuff 19 | things 40 | all 59 | pass rule (stuff >= +0.02 with CI > 0; things CI low >= -0.01) |\n|---|---|---|---|---|")
    verdict = {}
    for arm in PRIMARY_ARMS:
        ds, dt, da = (dif(g, f"{arm}|uL|V0", "A|ip|uL|V0") for g in ("stuff 19", "things 40", "all 59"))
        ok = ds[0] >= 0.02 and ds[1] > 0 and dt[1] >= -0.01; verdict[arm] = (ok, da[0])
        w(f"| {arm} | {f3(ds)} | {f3(dt)} | {f3(da)} | {'**PASS**' if ok else 'fail'} |")
    passed = [k for k, (ok, _) in verdict.items() if ok]
    w("\nSelected for confirmation: " + (max(passed, key=lambda k: verdict[k][1]) + " (largest all-59 gain among passing arms)"
                                         if passed else "none (no arm passes)") + ".\n")
    w("\n## 2. mIoU by arm, gate and group (raw 0.55)\n")
    arms = [k for k in ARMS if f"{k}|uL|V0" in S]
    w("| Arm | gate | " + " | ".join(GR) + " |\n|---|---|" + "---:|" * len(GR))
    for arm in arms:
        for gname, gk in (("none", "u-"), ("MK4", "uM"), ("MK4-LR", "uL")):
            w(f"| {arm} | {gname} | " + " | ".join(f"{stat(g, f'{arm}|{gk}|V0')[0]:.3f}" for g in GR) + " |")
    w("\n## 3. Secondary contrasts (raw 0.55)\n")
    w("| Contrast | stuff 19 | things 40 | all 59 |\n|---|---|---|---|")
    for x, y, nm in [(f"{arm}|u-|V0", "A|ip|u-|V0", f"{arm} - A, no gate") for arm in PRIMARY_ARMS] + \
                    [("A|op|uL|V0", "A|ip|uL|V0", "A out-of-prompt - A in-prompt, MK4-LR (read-out only)"),
                     ("B|op|uL|V0", "A|op|uL|V0", "B - A, both out-of-prompt, MK4-LR (prompt only)")] + \
                    [(f"{arm}|uL|V0", f"{arm}|u-|V0", f"MK4-LR - none within {arm}") for arm in ("A|ip",) + PRIMARY_ARMS]:
        if x in S and y in S:
            w(f"| {nm} | " + " | ".join(f3(dif(g, x, y)) for g in ("stuff 19", "things 40", "all 59")) + " |")
    if has_bip.sum() > 20:
        S2 = sub(S, has_bip); B2 = O.Boot(S2, a.boot); A2, Bb = B2.stat(S2["B|ip|uL|V0"][:, T55]), B2.stat(S2["A|ip|uL|V0"][:, T55])
        dd = A2[1] - Bb[1]
        w(f"\nItems whose scene caption contains the word ({int(has_bip.sum())} of {len(has_bip)}): B in-prompt - A, MK4-LR "
          f"{A2[0] - Bb[0]:+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}].\n")
    w("\n## 4. Attention quality (mean per-item AUROC, target vs other valid pixels)\n")
    w("| Arm | map | stuff 19 | things 40 | all 59 |\n|---|---|---:|---:|---:|")
    for arm in arms:
        for nm in ("u-|seed", "u-|V0", "uL|V0"):
            k = f"auc:{arm}|{nm}"
            if k in S:
                w(f"| {arm} | {nm} | " + " | ".join(f"{np.nanmean(E[g][0][k]):.3f}" for g in ("stuff 19", "things 40", "all 59")) + " |")
    w("\n## 5. Gate input shift: mean row cosine of the gate queries vs arm A (per item, mean over modules)\n")
    Qc = [json.loads(str(z["qcos"])) for z in Z]
    for arm in ("At", "C", "B"):
        v = np.array([q[arm] for q in Qc if arm in q])
        w(f"- {arm}: median {np.median(v):.4f}, 5th percentile {np.percentile(v, 5):.4f}, min {v.min():.4f}")
    w("\n## 6. Every threshold (MK4-LR, raw; descriptive)\n")
    w("| Arm | group | " + " | ".join(f"{t:.3f}" for t in THR) + " |\n|---|---|" + "---:|" * len(THR))
    for arm in arms:
        for g in ("stuff 19", "things 40", "all 59"):
            w(f"| {arm} | {g} | " + " | ".join(f"{stat(g, f'{arm}|uL|V0', t)[0]:.3f}" for t in range(len(THR))) + " |")
    w("\n## 7. Per class (MK4-LR, raw 0.55)\n")
    Ball = E["all 59"][1]; cl = Ball.cl
    w("| Class | group | items | " + " | ".join(arms) + " |\n|---|---|---:|" + "---:|" * len(arms))
    per = {arm: Ball.miou(S[f"{arm}|uL|V0"][:, T55])[0] for arm in arms}
    gmap = dict(zip(S["cls"], S["g"]))
    for i, c in sorted(enumerate(cl), key=lambda t: (gmap[t[1]], t[1])):
        w(f"| {c} | {gmap[c]} | {int((S['cls'] == c).sum())} | " + " | ".join(f"{per[arm][i]:.3f}" for arm in arms) + " |")
    w("\nBRACE and the paper2 method stay frozen; this is a development pilot, nothing is promoted.\n")
    out = REPORT if not a.tag else REPORT.with_name(f"result_{a.tag}.md"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L)); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("items"); sub.add_parser("check")
    p = sub.add_parser("caption"); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("capture"); p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("score"); p.add_argument("--workers", type=int, default=16); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("report"); p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args(); {"items": items, "check": check, "caption": caption, "capture": capture, "score": score, "report": report}[a.cmd](a)
