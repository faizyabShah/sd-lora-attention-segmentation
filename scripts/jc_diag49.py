#!/usr/bin/env python3
"""Run 49: small-set component diagnostics, our method vs DiffSegmenter vs iSeg (COCO train2017 development photos).

Each round takes 30 fresh random COCO train2017 photos (never used for any profile or fit) and every primary item
(class area >= 1%) on them. All maps for an item come from passes on the same machine, so the comparisons are within one
capture. Exploratory: no method is promoted from this run (runs/49_diag/PREREG.md).

Passes (t = 100, our pass code: bicubic 512, VAE mean, CPU noise seed 0, fp32), per item, on SD 1.5 (bb 15) and SD 2.1-base
(bb 21):
  T  "A photograph of a {word}"                                         (ours)
  E  T with the word's token embeddings x 1.6                            (iSeg's category enhancement)
  C  "a photograph of {word}++" + tail of a BLIP-large caption conditioned on "a photograph of {word}"   (DiffSegmenter)
     word tokens = DiffSegmenter's positions (+ the next token when the 5th word ends in "ing"), as run 44's driver
plus, on SD 2.1 only, iSeg's own per-photo pass (jc_swap47.Runner.iseg: every primary class of the photo in one prompt,
x 1.6, CUDA seed 1101, its 16/32 modules, head mean, DFC_KL clustering).
Seeds on each T / E / C pass:
  U-  our uniform 11-module cross-attention            UL  U- with our MK4-LR gate (gamma 1)       = our seed
  D-  DiffSegmenter's aggregation: every cross-attention module, head and layer mean per resolution (8/16/32/64),
      min-max per resolution, weights 0.3/0.5/0.1/0.1  DL  D- x the 64 x 64 MK4-LR gate map
  iSeg pass: I- (iSeg's seed), IL (I- x the gate map)
Refinements, from the same pass: seed (none) | V0 ours (two 64 x 64 steps, sg25 module, head-wise) | DSR DiffSegmenter's
one step of the 64 x 64 self-attention averaged over every 64 x 64 layer and head | ERS iSeg's entropy-reduced
self-attention (up_blocks.3.attentions.1 heads 1-2, its COCO settings 10 / 0.2).
Read-outs (score): t55 ours (min-max, >= 0.55) | t50 DiffSegmenter's (min-max, > 0.5) | comp iSeg's cam_to_label over
the photo's primary classes (COCO background threshold 0.45; uses the photo's class list = label information) |
best (one fixed threshold chosen in-sample, diagnostic) | orc (per-item oracle threshold, diagnostic).

items    (CPU) --round r                 photos and items for round r, masks (truth, ignore, other objects, instances)
caption  (GPU) --round r                 BLIP-large conditional captions per item
capture  (GPU) --round r --bb 15|21      maps
score    (CPU) --round r                 counts and error measures per (item, map, read-out)
report   (CPU) [--rounds 0,1]            runs/49_diag/result_r<rounds>.md
viz      (CPU) --round r                 a few photo / truth / map panels for the largest gaps (runs/49_diag/viz/)
"""
from __future__ import annotations

import argparse, gzip, json, math, os, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_swap47 as S  # noqa: E402  (iSeg pieces, Runner.iseg / gate64, Full; imports jc_sd21_47)

P47 = S.P47
CACHE = P47.CACHE
W = Path(os.environ.get("DIAG49_WORK", str(CACHE / "jc_diag49")))
SHIP = CACHE / "jc_diag49"                                   # word_map.json, fit40_photos.json (shipped/cache)
OUT = ROOT / "runs/49_diag"
COCO = ROOT / "datasets/coco2017"
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_BLIP_LOCAL = "/home/shafa/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-large/snapshots/353689b859fcf0523410b1806dace5fb46ecdf41"
BLIP = os.environ.get("BLIP_MODEL", _BLIP_LOCAL if os.path.isdir(_BLIP_LOCAL) else "Salesforce/blip-image-captioning-large")
N_PHOTO, MIN_AREA, PERM_SEED = 30, 0.01, 49
DSW = {8: 0.3, 16: 0.5, 32: 0.1, 64: 0.1}                    # DiffSegmenter ptp_stable_best.py:452
ENH = 1.6
PROMPTS, SEEDS, REFS = ("T", "E", "C"), ("U-", "UL", "D-", "DL"), ("seed", "V0", "DSR", "ERS")
THR = np.round(np.arange(0.05, 0.951, 0.05), 2)
I55, I50 = int(np.argmin(abs(THR - 0.55))), int(np.argmin(abs(THR - 0.50)))
REF = "15|T|UL|V0"                                            # our method (paper 2, frozen), read out at t55


def rdir(r):
    return W / f"r{r}"


# ------------------------------------------------------------------ items
def items(a):
    import jc_ov42 as O
    C40 = O.C40
    excl = {int(x["img"]) for x in json.loads((CACHE / "jc_ov42/items_prof.json").read_text())["items"]}
    excl |= {int(x["img"]) for x in json.loads((CACHE / "jc_real44/ours/items_prof44.json").read_text())["jobs"]}
    excl |= set(json.loads((SHIP / "fit40_photos.json").read_text()))
    d = json.loads((COCO / "annotations/instances_train2017.json").read_text())
    cat = {c["id"]: O.C2V.get(c["name"], c["name"]) for c in d["categories"]}; imgs = {i["id"]: i for i in d["images"]}
    by = defaultdict(list)
    for an in d["annotations"]:
        by[an["image_id"]].append((cat[an["category_id"]], an))
    pool = sorted(i for i in by if i not in excl); random.Random(PERM_SEED).shuffle(pool)
    print(f"development pool: {len(pool)} train2017 photos (excluded {len(excl)} profile / fit photos)", flush=True)
    md = W / "masks"; md.mkdir(parents=True, exist_ok=True)
    need = N_PHOTO * (a.round + 1); elig = []; k = 0
    while len(elig) < need:
        i = pool[k]; k += 1
        im = imgs[i]; h, w = im["height"], im["width"]; anns = by[i]
        truth, crowd, inst = {}, {}, {}
        for nm, an in anns:
            m = C40.rasterize(an["segmentation"], h, w)
            if an["iscrowd"]:
                crowd[nm] = crowd.get(nm, np.zeros((h, w), bool)) | m
            else:
                truth[nm] = truth.get(nm, np.zeros((h, w), bool)) | m; inst.setdefault(nm, []).append(m)
        prim = sorted(c for c, t in truth.items() if t.mean() >= MIN_AREA)
        if not prim:
            continue
        if len(elig) >= need - N_PHOTO:
            allobj = np.zeros((h, w), bool)
            for t in truth.values():
                allobj |= t
            X = []
            for c in prim:
                t = truth[c]; ign = crowd.get(c, np.zeros((h, w), bool)) & ~t
                lab = np.zeros((h, w), np.int16)
                for j, m in enumerate(inst[c]):
                    lab[m & (lab == 0)] = j + 1
                iid = f"d{i:012d}_{O.CLS80.index(c) + 1:02d}"
                np.savez_compressed(md / f"{iid}.npz", truth=t, ignore=ign, other=allobj & ~t & ~ign, inst=lab)
                X.append({"id": iid, "img": int(i), "path": f"train2017/{im['file_name']}", "cls": c, "area": float(t.mean()),
                          "ninst": len(inst[c])})
            elig.append((i, X))
        else:
            elig.append((i, None))
    mine = elig[-N_PHOTO:]; X = [x for _, xs in mine for x in xs]
    rdir(a.round).mkdir(parents=True, exist_ok=True)
    (rdir(a.round) / "items.json").write_text(json.dumps(X, indent=0))
    (rdir(a.round) / "photos.json").write_text(json.dumps([{"path": xs[0]["path"]} for _, xs in mine]))
    print(f"round {a.round}: {len(mine)} photos, {len(X)} primary items, {len({x['cls'] for x in X})} classes", flush=True)


def load_items(r, limit=0):
    X = json.loads((rdir(r) / "items.json").read_text())
    if limit:
        keep = sorted({x["img"] for x in X})[:limit]; X = [x for x in X if x["img"] in set(keep)]
    return X


def by_photo(X):
    P = defaultdict(list)
    for x in X:
        P[x["img"]].append(x)
    return sorted(P.items())


# ------------------------------------------------------------------ captions (DiffSegmenter's recipe, run 44 driver)
def caption(a):
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    dev = torch.device("cuda")
    proc = BlipProcessor.from_pretrained(BLIP); model = BlipForConditionalGeneration.from_pretrained(BLIP).to(dev).eval()
    X = load_items(a.round, a.limit); C = {}
    with torch.no_grad():
        for x in X:
            img = Image.open(COCO / x["path"]).convert("RGB"); base = f"a photograph of {x['cls']}"
            C[x["id"]] = proc.decode(model.generate(**proc(img, base, return_tensors="pt").to(dev))[0], skip_special_tokens=True)
    (rdir(a.round) / "captions.json").write_text(json.dumps(C, indent=0))
    for x in X[:5]:
        print(x["id"], x["cls"], "|", C[x["id"]], flush=True)
    print(f"captions: {len(C)}", flush=True)


def ds_prompt(cls, cap):
    base = f"a photograph of {cls}"; lab = f"({cls})" if " " in cls else cls
    return f"a photograph of {lab}++" + cap[len(base):]


def ds_tokens(tok, prompt, cls):
    """comparisons/diffsegmenter/diffseg_classword.py class_token_positions + the 'ing' rule (jc_diffseg44.py)."""
    n_prefix = len(tok("a photograph of")["input_ids"]) - 2
    pos = [1 + n_prefix + j for j, t in enumerate(tok(cls)["input_ids"][1:-1])
           if (s := tok.decode([t]).strip()) and any(ch.isalnum() for ch in s)]
    words = prompt.split(" ")
    return pos + [pos[-1] + 1] if (len(words) > 4 and words[4].endswith("ing")) else pos


# ------------------------------------------------------------------ recorder additions
class DSRec:
    """Wraps a processor: when rec.want_ds, keeps every cross-attention's head-mean probabilities [N, 77] and the running
    sum of the head-mean 64 x 64 self-attention (DiffSegmenter's aggregation)."""

    def __init__(self, inner, name, rec):
        self.inner, self.name, self.rec = inner, name, rec

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        rec = self.rec
        if rec.want_ds:
            import torch
            cross = encoder_hidden_states is not None; n = hidden_states.shape[1]
            if cross or n == 4096:
                ctx = encoder_hidden_states if cross else hidden_states
                if cross and attn.norm_cross:
                    ctx = attn.norm_encoder_hidden_states(ctx)
                q, k = attn.to_q(hidden_states), attn.to_k(ctx); b = q.shape[0]; hd = attn.heads; d = q.shape[-1] // hd
                qh = q.view(b, -1, hd, d).transpose(1, 2)[0].float(); kh = k.view(b, -1, hd, d).transpose(1, 2)[0].float()
                if cross:
                    rec.ds_ca[self.name] = (qh @ kh.transpose(-1, -2) * attn.scale).softmax(-1).mean(0)
                else:
                    acc = torch.zeros((n, n), device=qh.device)
                    for h in range(hd):
                        acc += (qh[h] @ kh[h].T * attn.scale).softmax(-1)
                    rec.ds_sa = acc / hd if rec.ds_sa is None else rec.ds_sa + acc / hd; rec.ds_n += 1
        return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)


def ds_reset(rec):
    rec.ds_ca, rec.ds_sa, rec.ds_n = {}, None, 0


class Diag(S.Runner):
    def __init__(self, bb):
        import torch
        R, O, U = P47.setup(); self.torch, self.R, self.O, self.U, self.bb = torch, R, O, U, bb
        if bb == "15":
            from jc_real44 import WORK as W44
            R.MODEL = SD15; O.WORK = CACHE / "jc_ov42"; U.WORK = W44 / "ours"
        else:
            O.WORK = Path(os.environ.get("DIAG49_PROF21", str(P47.W / "ov42")))
        wm = SHIP / "word_map.json"
        if wm.exists():
            R.word_map = lambda: json.loads(wm.read_text())
        self.pipe, self.rec = R.load(); rec = self.rec; rec.full = {}; rec.want_full = False; rec.want_ds = False; ds_reset(rec)
        procs = dict(self.pipe.unet.attn_processors)
        for n in list(procs):
            nm = n.removesuffix(".processor"); p = procs[n]
            if nm in S.FULL:
                p = S.Full(p, nm, rec)
            procs[n] = DSRec(p, nm, rec)
        self.pipe.unet.set_attn_processor(procs)
        assert S.ERS_SA in rec.pair, "iSeg's self-attention layer is not recorded"
        fit = torch.load(O.WORK / "fit.pt", weights_only=False)
        print(f"profiles: {O.WORK / 'fit.pt'} (model {fit.get('model', 'SD 1.5')})", flush=True)
        self.P = O.M.to_dev({"mo": fit["fit"]["20"]["mo"], "neg": fit["fit"]["20"]["neg"]})
        self.w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=R.DEV); self.Wm = R.word_map()

    def run_pass(self, img, prompt, target, enhance):
        """jc_realvoc.one_pass with explicit word tokens and optional x 1.6 on their embeddings."""
        import torch
        import torch.nn.functional as F
        R, pipe, rec = self.R, self.pipe, self.rec
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().to(R.DEV) / 127.5 - 1
        x = F.interpolate(x, size=(R.PX, R.PX), mode="bicubic", align_corners=False).clamp(-1, 1)
        z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
        eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(R.NOISE_SEED)).to(R.DEV, z0.dtype)
        ac = pipe.scheduler.alphas_cumprod.to(R.DEV); zt = ac[R.T_NOISE].sqrt() * z0 + (1 - ac[R.T_NOISE]).sqrt() * eps
        pe, _ = pipe.encode_prompt(prompt=prompt, device=R.DEV, num_images_per_prompt=1, do_classifier_free_guidance=False)
        pe = pe.float().clone()
        if enhance:
            pe[:, list(target)] *= ENH
        rec.target = tuple(target); rec.clear(); ds_reset(rec); rec.want_ds = True
        pipe.unet(zt, torch.tensor(R.T_NOISE, device=R.DEV), encoder_hidden_states=pe, return_dict=False)
        rec.want_ds = False
        if set(rec.ca) != rec.ca_names or set(rec.grid_qk) != set(R.SA) or set(rec.pair_qk) != rec.pair or rec.ds_n == 0:
            raise ValueError("missing recorded attention")

    def ds_seed(self, target):
        import torch
        import torch.nn.functional as F
        rec = self.rec; by = defaultdict(list)
        for m, p in rec.ds_ca.items():
            by[math.isqrt(p.shape[0])].append(p[:, list(target)].mean(-1))
        assert set(by) == set(DSW), sorted(by)
        out = torch.zeros(64 * 64, device=self.R.DEV)
        for res, L in by.items():
            v = torch.stack(L).mean(0).reshape(1, 1, res, res)
            if res != 64:
                v = F.interpolate(v, size=(64, 64), mode="bilinear", align_corners=False)
            v = v.reshape(-1); out += (v - v.min()) / (v.max() - v.min()).clamp_min(1e-12) * DSW[res]
        return out.reshape(64, 64)

    def refine(self, sd, H64, out, nm):
        from src.segmentation.sg25_sa import _apply
        sd = sd.float(); sd = (sd - sd.min()) / (sd.max() - sd.min()).clamp_min(1e-12); v = sd.reshape(-1)
        out[f"{nm}|seed"] = sd
        out[f"{nm}|V0"] = _apply(H64, _apply(H64, v)).mean(0).reshape(64, 64)
        out[f"{nm}|DSR"] = (self.rec.ds_sa / self.rec.ds_n @ v).reshape(64, 64)
        out[f"{nm}|ERS"] = S.ers(v.reshape(-1, 1), self.rec.pair_qk[S.ERS_SA], "coco")[:, 0].reshape(64, 64)

    def item(self, x, img, cap):
        from src.segmentation.attention_aggregation import find_token_subsequence
        from src.segmentation.sg25_sa import _attention
        R, O, tok, out = self.R, self.O, self.pipe.tokenizer, {}
        key, word = self.U.keyword("coco", x["cls"], self.Wm)
        for P in PROMPTS:
            if P == "C":
                prompt = ds_prompt(x["cls"], cap); tgt = ds_tokens(tok, prompt, x["cls"])
            else:
                prompt = f"A photograph of a {word}"
                ids = tok(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
                tgt = find_token_subsequence(ids, tok(word, add_special_tokens=False)["input_ids"])
            self.run_pass(img, prompt, tgt, P == "E")
            g64, Q, B = self.gate64(key); H64 = _attention(self.rec.grid_qk[64], 64)
            ga = None if not B else O.R37.powered(B["L"], Q, 1.0)
            u = R.make_seed(self.rec, self.w_u, None); d_ = self.ds_seed(tgt)
            seeds = {"U-": u, "UL": R.make_seed(self.rec, self.w_u, ga) if ga else u, "D-": d_,
                     "DL": d_ * g64 if g64 is not None else d_}
            for nm, sd in seeds.items():
                self.refine(sd, H64, out, f"{self.bb}|{P}|{nm}")
            out[f"_prompt_{P}"] = prompt; out[f"_miss"] = 0 if B else 1
            self.rec.clear(); ds_reset(self.rec)
        return out

    def photo_iseg(self, X, img):
        """iSeg's pass (jc_swap47.Runner.iseg, COCO settings) + DiffSegmenter's refinement of its seeds."""
        rec = self.rec; ds_reset(rec); rec.want_ds = True
        mp, text, ids, miss = self.iseg("coco", None, X, img)
        rec.want_ds = False; out = {}
        for xid, m in mp.items():
            o = {}
            for nm in ("I", "IL"):
                sd = m[f"{nm}|seed"].float(); sd = (sd - sd.min()) / (sd.max() - sd.min()).clamp_min(1e-12)
                for r in ("seed", "V0", "ERS"):
                    o[f"21|iseg|{nm}{'-' if nm == 'I' else ''}|{r}"] = m[f"{nm}|{r}"]
                o[f"21|iseg|{nm}{'-' if nm == 'I' else ''}|DSR"] = (rec.ds_sa / rec.ds_n @ sd.reshape(-1)).reshape(64, 64)
            out[xid] = o
        ds_reset(rec); return out, text, miss


def save(o, mp, extra):
    names = sorted(k for k in mp if not k.startswith("_"))
    arr = np.stack([mp[k].float().cpu().numpy() for k in names]).astype(np.float16)
    if not np.isfinite(arr.astype(np.float32)).all():
        raise ValueError(f"non-finite map in {o.name}")
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, maps=arr, names=np.array(names), **extra); os.replace(tmp, o)


def capture(a):
    import torch
    from PIL import Image
    X = load_items(a.round, a.limit); cap = json.loads((rdir(a.round) / "captions.json").read_text())
    rn = Diag(a.bb); od = rdir(a.round) / f"maps_{a.bb}"; od.mkdir(parents=True, exist_ok=True)
    oi = rdir(a.round) / "maps_iseg"; oi.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for img_id, xs in by_photo(X):
            img = np.asarray(Image.open(COCO / xs[0]["path"]).convert("RGB")); t0 = time.perf_counter()
            for x in xs:
                o = od / f"{x['id']}.npz"
                if not o.exists():
                    mp = rn.item(x, img, cap[x["id"]])
                    save(o, mp, {f"prompt_{P}": mp[f"_prompt_{P}"] for P in PROMPTS} | {"miss": mp["_miss"]})
            if a.bb == "21":
                o = oi / f"{img_id:012d}.npz"
                if not o.exists():
                    mp, text, miss = rn.photo_iseg(xs, img); d = {}
                    for xid, m in mp.items():
                        names = sorted(m); d[f"m_{xid}"] = np.stack([m[k].float().cpu().numpy() for k in names]).astype(np.float16)
                    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, names=np.array(names), text=text, miss=miss, **d); os.replace(tmp, o)
            torch.cuda.empty_cache()
            print(f"[diag49:{a.bb}] photo {img_id} {len(xs)} items {time.perf_counter() - t0:.1f}s", flush=True)


# ------------------------------------------------------------------ scoring
def mm(v):
    return (v - v.min()) / max(float(v.max() - v.min()), 1e-12)


def up(v, hw):
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(v, np.float32))[None, None], size=hw, mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def auroc(v, t):
    from scipy.stats import rankdata
    P = int(t.sum()); N = t.size - P
    if P == 0 or N == 0:
        return float("nan")
    r = rankdata(v); return float((r[t].sum() - P * (P + 1) / 2) / (P * N))


def maps_of(r, xs):
    """{item id: {map name: 64 x 64}} for one photo (both backbones and iSeg's pass, when present)."""
    M = defaultdict(dict)
    for bb in ("15", "21"):
        for x in xs:
            f = rdir(r) / f"maps_{bb}" / f"{x['id']}.npz"
            if f.exists():
                z = np.load(f)
                for n, m in zip(z["names"], z["maps"]):
                    M[x["id"]][str(n)] = m.astype(np.float32)
    f = rdir(r) / "maps_iseg" / f"{xs[0]['img']:012d}.npz"
    if f.exists():
        z = np.load(f)
        for x in xs:
            for n, m in zip(z["names"], z[f"m_{x['id']}"]):
                M[x["id"]][str(n)] = m.astype(np.float32)
    return M


def score(a):
    from scipy.ndimage import binary_dilation, binary_erosion
    X = load_items(a.round, a.limit); recs = []
    for img_id, xs in by_photo(X):
        M = maps_of(a.round, xs); G = {}
        for x in xs:
            z = np.load(W / "masks" / f"{x['id']}.npz"); t = z["truth"]; valid = ~z["ignore"]
            b = max(1, round(0.015 * max(t.shape)))
            band = binary_dilation(t, iterations=b) & ~binary_erosion(t, iterations=b)
            rng = np.random.default_rng(int(x["id"][1:13]) * 100 + int(x["id"][-2:]))
            vi = np.flatnonzero(valid.reshape(-1)); sub = rng.choice(vi, size=min(20000, vi.size), replace=False)
            G[x["id"]] = dict(t=t, valid=valid, other=z["other"], inst=z["inst"], band=band, sub=sub)
        names = sorted(set.intersection(*[set(M[x["id"]]) for x in xs])) if all(x["id"] in M for x in xs) else []
        up_ = {(x["id"], n): mm(up(mm(M[x["id"]][n]), G[x["id"]]["t"].shape)) for x in xs for n in names}
        for n in names:
            comp = S.cam_to_label(np.stack([up_[(x["id"], n)] for x in xs]), 0.45)
            for j, x in enumerate(xs):
                g = G[x["id"]]; t, valid = g["t"], g["valid"]; v = up_[(x["id"], n)]
                pr = v[None] >= THR[:, None, None]; tv = t & valid
                tp = (pr & tv).sum((1, 2)); fp = (pr & valid & ~t).sum((1, 2)); fn = (~pr & tv).sum((1, 2))
                rec = dict(id=x["id"], img=x["img"], cls=x["cls"], area=x["area"], ninst=x["ninst"], map=n,
                           tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist())
                pc = (comp == j + 1); rec.update(ctp=int((pc & tv).sum()), cfp=int((pc & valid & ~t).sum()), cfn=int((~pc & tv).sum()))
                rec["auroc"] = auroc(v.reshape(-1)[g["sub"]], t.reshape(-1)[g["sub"]])
                rec["point"] = bool(t.reshape(-1)[int(np.argmax(v))])
                vv = v * valid; rec["mass"] = float((vv * t).sum() / max(vv.sum(), 1e-12))
                for k, i in (("55", I55), ("50", I50)):
                    p = pr[i] & valid
                    rec[f"fpo{k}"] = int((p & g["other"]).sum()); rec[f"fpb{k}"] = int((p & ~t & ~g["other"]).sum())
                    hit = [float((p & (g["inst"] == q)).sum()) / max(int((g["inst"] == q).sum()), 1) >= 0.5 for q in range(1, x["ninst"] + 1)]
                    rec[f"ih{k}"] = float(np.mean(hit)) if hit else float("nan")
                    keep = valid & ~g["band"]
                    rec[f"btp{k}"] = int((p & t & keep).sum()); rec[f"bun{k}"] = int(((p | t) & keep).sum())
                recs.append(rec)
        print(f"[score49] photo {img_id}: {len(xs)} items, {len(names)} maps", flush=True)
    o = rdir(a.round) / "scores.json.gz"
    with gzip.open(o, "wt") as f:
        json.dump(recs, f)
    print(f"wrote {o}: {len(recs)} records", flush=True)


# ------------------------------------------------------------------ report
def load_scores(rounds):
    R = []
    for r in rounds:
        with gzip.open(rdir(r) / "scores.json.gz", "rt") as f:
            R += [dict(x, round=r) for x in json.load(f)]
    return R


class Table:
    """Per (map, read-out): photo x class count matrices, for class-macro mIoU and a paired photo bootstrap."""

    def __init__(self, recs, boot, seed=0):
        self.photos = sorted({(x["round"], x["img"]) for x in recs}); self.classes = sorted({x["cls"] for x in recs})
        pi = {p: i for i, p in enumerate(self.photos)}; ci = {c: i for i, c in enumerate(self.classes)}
        self.by = defaultdict(list)
        for x in recs:
            self.by[x["map"]].append(x)
        self.pi, self.ci = pi, ci
        rng = np.random.default_rng(seed); n = len(self.photos)
        self.Wb = rng.multinomial(n, [1 / n] * n, size=boot).astype(np.float64)
        self.cache = {}

    def mats(self, m, ro):
        k = (m, ro)
        if k not in self.cache:
            T = np.zeros((len(self.photos), len(self.classes))); U = np.zeros_like(T); seen = np.zeros_like(T)
            for x in self.by[m]:
                i, j = self.pi[(x["round"], x["img"])], self.ci[x["cls"]]
                if ro in ("t55", "t50"):
                    q = I55 if ro == "t55" else I50; tp, un = x["tp"][q], x["tp"][q] + x["fp"][q] + x["fn"][q]
                elif ro == "comp":
                    tp, un = x["ctp"], x["ctp"] + x["cfp"] + x["cfn"]
                elif ro.startswith("thr"):
                    q = int(ro[3:]); tp, un = x["tp"][q], x["tp"][q] + x["fp"][q] + x["fn"][q]
                T[i, j] += tp; U[i, j] += un; seen[i, j] = 1
            self.cache[k] = (T, U, seen)
        return self.cache[k]

    def miou(self, m, ro):
        T, U, s = self.mats(m, ro)
        if not self.by[m]:
            return float("nan")
        cs = s.sum(0) > 0; return float((T.sum(0)[cs] / np.maximum(U.sum(0)[cs], 1)).mean())

    def best(self, m):
        v = [self.miou(m, f"thr{q}") for q in range(len(THR))]; q = int(np.nanargmax(v)); return v[q], THR[q]

    def delta(self, a, b):
        """class-macro mIoU(a) - mIoU(b), a and b = (map, read-out); 95% photo bootstrap CI."""
        (Ta, Ua, sa), (Tb, Ub, sb) = self.mats(*a), self.mats(*b)
        cs = ((sa.sum(0) > 0) & (sb.sum(0) > 0))
        def f(T, U, Wt):                                  # a class absent from a bootstrap sample is left out (nan)
            num, den = (Wt @ T)[:, cs], (Wt @ U)[:, cs]
            return np.where(den > 0, num / np.maximum(den, 1), np.nan)
        pt = float((Ta.sum(0)[cs] / np.maximum(Ua.sum(0)[cs], 1)).mean() - (Tb.sum(0)[cs] / np.maximum(Ub.sum(0)[cs], 1)).mean())
        bs = np.nanmean(f(Ta, Ua, self.Wb), 1) - np.nanmean(f(Tb, Ub, self.Wb), 1)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return pt, float(lo), float(hi)


def fmt_d(d):
    pt, lo, hi = d; tag = " **+**" if lo > 0 else (" **−**" if hi < 0 else "")
    return f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{tag}"


def diag_row(recs, m, k):
    X = [x for x in recs if x["map"] == m]
    if not X:
        return None
    q = I55 if k == "55" else I50
    tp = np.array([x["tp"][q] for x in X], float); fp = np.array([x["fp"][q] for x in X], float); fn = np.array([x["fn"][q] for x in X], float)
    iou = tp / np.maximum(tp + fp + fn, 1)
    orc = np.array([max(t / max(t + f + n, 1) for t, f, n in zip(x["tp"], x["fp"], x["fn"])) for x in X])
    fpo = np.array([x[f"fpo{k}"] for x in X], float); fpb = np.array([x[f"fpb{k}"] for x in X], float)
    bt = np.array([x[f"btp{k}"] for x in X], float); bu = np.array([x[f"bun{k}"] for x in X], float)
    sz = np.array([x["area"] for x in X])
    out = dict(n=len(X), iou=iou.mean(), orc=orc.mean(), auroc=np.nanmean([x["auroc"] for x in X]),
               point=np.mean([x["point"] for x in X]), mass=np.mean([x["mass"] for x in X]),
               prec=(tp / np.maximum(tp + fp, 1)).mean(), rec=(tp / np.maximum(tp + fn, 1)).mean(),
               fpo=fpo.sum() / max((fpo + fpb).sum(), 1), inst=np.nanmean([x[f"ih{k}"] for x in X]),
               biou=(bt / np.maximum(bu, 1)).mean())
    for nm, sel in (("small", sz < 0.05), ("mid", (sz >= 0.05) & (sz < 0.25)), ("large", sz >= 0.25)):
        out[nm] = iou[sel].mean() if sel.any() else float("nan")
    return out


def report(a):
    rounds = [int(r) for r in a.rounds.split(",")] if a.rounds else sorted(int(p.name[1:]) for p in W.glob("r*") if (p / "scores.json.gz").exists())
    recs = load_scores(rounds); T = Table(recs, a.boot); maps = sorted(T.by)
    ours, ours21 = (REF, "t55"), ("21|T|UL|V0", "t55")
    DS, IS = ("15|C|D-|DSR", "t50"), ("21|iseg|I-|ERS", "comp")
    L = [f"# Run 49 diagnostics, rounds {','.join(map(str, rounds))} ({'smoke, ' if a.tag else ''}exploratory; `PREREG.md`)", "",
         f"COCO train2017 development photos: {len(T.photos)} photos, {len({x['id'] for x in recs})} primary items, "
         f"{len(T.classes)} classes. Class-macro mIoU over the classes present; Δ [95% paired photo bootstrap, {a.boot}]; "
         "**+** / **−** = CI excludes 0. With 30-90 photos most CIs are wide: read directions and sizes, not verdicts.", "",
         "Map names: backbone | prompt | seed | refinement. Prompt T template, E template + x1.6 word embeddings, C DiffSegmenter "
         "BLIP caption, iseg = iSeg's multi-class pass. Seed U- ours ungated, UL ours gated (MK4-LR), D-/DL DiffSegmenter "
         "aggregation (all modules, resolution-weighted) without / with the gate map, I-/IL iSeg's seed. Refinement seed / V0 "
         "ours / DSR DiffSegmenter / ERS iSeg. Read-out t55 ours, t50 DiffSegmenter's, comp iSeg's competition over the "
         "photo's classes (uses the class list), best = best single threshold in-sample (diagnostic).", ""]
    have = lambda c: bool(T.by.get(c[0]))
    L += ["## 1. Reference pipelines", "", "| Pipeline | Cell | mIoU | best fixed threshold |", "|---|---|---:|---:|"]
    for nm, c in (("ours (SD 1.5, frozen)", ours), ("ours on SD 2.1", ours21), ("DiffSegmenter (transcribed, SD 1.5)", DS),
                  ("iSeg (transcribed, SD 2.1, class list given)", IS)):
        if have(c):
            b, th = T.best(c[0]); L.append(f"| {nm} | `{c[0]}` @{c[1]} | {T.miou(*c):.3f} | {b:.3f} @ {th:.2f} |")
    L += ["", "## 2. One change to ours (Δ vs ours, `15|T|UL|V0` @t55)", "", "| Change | Cell | Δ mIoU |", "|---|---|---:|"]
    swaps = [("backbone SD 2.1", ("21|T|UL|V0", "t55")), ("prompt: x1.6 enhancement", ("15|E|UL|V0", "t55")),
             ("prompt: DiffSegmenter caption", ("15|C|UL|V0", "t55")), ("no gate", ("15|T|U-|V0", "t55")),
             ("seed: DiffSegmenter aggregation (+ gate map)", ("15|T|DL|V0", "t55")),
             ("refinement: none", ("15|T|UL|seed", "t55")), ("refinement: DiffSegmenter", ("15|T|UL|DSR", "t55")),
             ("refinement: iSeg ERS", ("15|T|UL|ERS", "t55")), ("read-out: > 0.5", ("15|T|UL|V0", "t50")),
             ("read-out: iSeg competition (class list)", ("15|T|UL|V0", "comp")),
             ("SD 2.1 + ERS", ("21|T|UL|ERS", "t55")), ("SD 2.1 + x1.6 + ERS", ("21|E|UL|ERS", "t55")),
             ("SD 2.1 + x1.6 + ERS + competition", ("21|E|UL|ERS", "comp"))]
    for nm, c in swaps:
        if have(c):
            L.append(f"| {nm} | `{c[0]}` @{c[1]} | {fmt_d(T.delta(c, ours))} |")
    L += ["", "## 3. One change to DiffSegmenter toward ours (Δ vs `15|C|D-|DSR` @t50)", "", "| Change | Cell | Δ mIoU |", "|---|---|---:|"]
    for nm, c in (("prompt: template", ("15|T|D-|DSR", "t50")), ("seed: our aggregation", ("15|C|U-|DSR", "t50")),
                  ("+ our gate map", ("15|C|DL|DSR", "t50")), ("refinement: ours V0", ("15|C|D-|V0", "t50")),
                  ("refinement: iSeg ERS", ("15|C|D-|ERS", "t50")), ("read-out: >= 0.55", ("15|C|D-|DSR", "t55")),
                  ("backbone SD 2.1", ("21|C|D-|DSR", "t50"))):
        if have(c) and have(DS):
            L.append(f"| {nm} | `{c[0]}` @{c[1]} | {fmt_d(T.delta(c, DS))} |")
    L += ["", "## 4. One change to iSeg toward ours (Δ vs `21|iseg|I-|ERS` @comp)", "", "| Change | Cell | Δ mIoU |", "|---|---|---:|"]
    for nm, c in (("+ our gate map", ("21|iseg|IL|ERS", "comp")), ("refinement: ours V0", ("21|iseg|I-|V0", "comp")),
                  ("refinement: DiffSegmenter", ("21|iseg|I-|DSR", "comp")), ("read-out: ours >= 0.55, one class", ("21|iseg|I-|ERS", "t55")),
                  ("pass: our single-word pass, our seed (gated), ERS", ("21|T|UL|ERS", "comp")),
                  ("pass: our single-word pass with x1.6, gated, ERS", ("21|E|UL|ERS", "comp"))):
        if have(c) and have(IS):
            L.append(f"| {nm} | `{c[0]}` @{c[1]} | {fmt_d(T.delta(c, IS))} |")
    L += ["", "## 5. Factor averages (every combination of the other factors; t55; mean Δ over cells)", ""]
    grid = [m for m in maps if m.count("|") == 3 and m.split("|")[1] in PROMPTS]
    def fac(i, a_, b_):
        pairs = [(m, "|".join(p if k != i else b_ for k, p in enumerate(m.split("|")))) for m in grid if m.split("|")[i] == a_]
        d = [T.miou(p, "t55") - T.miou(m, "t55") for m, p in pairs if p in T.by]
        return (np.mean(d), len(d)) if d else (float("nan"), 0)
    L += ["| Factor change | mean Δ | cells |", "|---|---:|---:|"]
    for nm, i, a_, b_ in (("SD 1.5 → SD 2.1", 0, "15", "21"), ("T → E (x1.6)", 1, "T", "E"), ("T → C (caption)", 1, "T", "C"),
                          ("U- → UL (gate)", 2, "U-", "UL"), ("D- → DL (gate map)", 2, "D-", "DL"), ("U- → D- (aggregation)", 2, "U-", "D-"),
                          ("UL → DL", 2, "UL", "DL"), ("seed → V0", 3, "seed", "V0"), ("V0 → DSR", 3, "V0", "DSR"), ("V0 → ERS", 3, "V0", "ERS")):
        m_, n_ = fac(i, a_, b_); L.append(f"| {nm} | {m_:+.3f} | {n_} |")
    L += ["", "## 6. What goes wrong (t55 unless the read-out is t50; item means)", "",
          "IoU and oracle = item mean (oracle = best threshold per item); AUROC = target vs other valid pixels (ranking "
          "quality, threshold-free); point = map maximum inside the object; mass = share of the min-max map inside the "
          "object; FP-obj = share of false positives on other annotated objects (vs unannotated background); inst = share of "
          "instances at least half covered; IoU-nb = IoU without a band of 1.5% of the photo size around the object edge.", "",
          "| Cell | n | IoU | oracle | AUROC | point | mass | prec | recall | FP-obj | inst | IoU-nb | small | mid | large |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in (ours, ("15|T|U-|V0", "t55"), ("15|T|UL|seed", "t55"), ours21, ("21|E|UL|ERS", "t55"), DS, ("21|iseg|I-|ERS", "t55")):
        d = diag_row(recs, c[0], "50" if c[1] == "t50" else "55")
        if d:
            L.append(f"| `{c[0]}` @{c[1] if c[1] != 'comp' else 't55'} | {d['n']} | {d['iou']:.3f} | {d['orc']:.3f} | {d['auroc']:.3f} | "
                     f"{d['point']:.2f} | {d['mass']:.2f} | {d['prec']:.2f} | {d['rec']:.2f} | {d['fpo']:.2f} | {d['inst']:.2f} | "
                     f"{d['biou']:.3f} | {d['small']:.3f} | {d['mid']:.3f} | {d['large']:.3f} |")
    L += ["", "## 7. Every cell (mIoU; t55 / t50 / comp / best fixed threshold)", "", "| Cell | t55 | t50 | comp | best |", "|---|---:|---:|---:|---:|"]
    for m in sorted(maps, key=lambda m: -T.miou(m, "t55")):
        b, th = T.best(m); L.append(f"| `{m}` | {T.miou(m, 't55'):.3f} | {T.miou(m, 't50'):.3f} | {T.miou(m, 'comp'):.3f} | {b:.3f} @ {th:.2f} |")
    L += ["", "## 8. Largest per-item gaps (best competitor cell − ours, IoU at each cell's read-out)", "",
          "| Item | class | area | ours | DiffSegmenter | iSeg | best other cell |", "|---|---|---:|---:|---:|---:|---|"]
    def iou_of(x, ro):
        if ro == "comp":
            return x["ctp"] / max(x["ctp"] + x["cfp"] + x["cfn"], 1)
        q = I55 if ro == "t55" else I50; return x["tp"][q] / max(x["tp"][q] + x["fp"][q] + x["fn"][q], 1)
    by_item = defaultdict(dict)
    for x in recs:
        for ro in ("t55", "t50", "comp"):
            by_item[(x["round"], x["id"])][(x["map"], ro)] = (iou_of(x, ro), x)
    gaps = []
    for k, d in by_item.items():
        if ours in d:
            o = d[ours][0]; bc = max(d, key=lambda c: d[c][0])
            gaps.append((d[bc][0] - o, k, o, d.get(DS, (float("nan"),))[0], d.get(IS, (float("nan"),))[0], bc, d[ours][1]))
    gaps.sort(key=lambda g: -g[0])
    for g in gaps[:15]:
        L.append(f"| r{g[1][0]} `{g[1][1]}` | {g[6]['cls']} | {g[6]['area']:.3f} | {g[2]:.3f} | {g[3]:.3f} | {g[4]:.3f} | `{g[5][0]}` @{g[5][1]} {g[0] + g[2]:.3f} |")
    o = OUT / f"result_r{'-'.join(map(str, rounds))}{a.tag}.md"; OUT.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n"); print(f"wrote {o}", flush=True)
    (OUT / f"gaps_r{'-'.join(map(str, rounds))}{a.tag}.json").write_text(json.dumps([[g[1][0], g[1][1], g[5][0], g[5][1]] for g in gaps[:15]]))


def viz(a):
    """Panels for the largest gaps of this round: photo | truth | ours | DiffSegmenter | iSeg | best other cell (t55 / t50
    / comp masks drawn on the photo; 256 px high)."""
    from PIL import Image, ImageDraw
    gf = OUT / f"gaps_r{a.round}{a.tag}.json"; G = json.loads(gf.read_text())[: a.n]
    X = {x["id"]: x for x in load_items(a.round)}; vd = OUT / "viz"; vd.mkdir(parents=True, exist_ok=True)
    for r, iid, bm, bro in G:
        x = X[iid]; xs = [y for y in X.values() if y["img"] == x["img"]]; M = maps_of(a.round, xs)
        img = Image.open(COCO / x["path"]).convert("RGB"); hw = (img.height, img.width)
        t = np.load(W / "masks" / f"{iid}.npz")["truth"]
        def mask(m, ro):
            if m not in M[iid]:
                return None
            if ro == "comp":
                cams = np.stack([mm(up(mm(M[y["id"]][m]), hw)) for y in xs]); j = [y["id"] for y in xs].index(iid)
                return S.cam_to_label(cams, 0.45) == j + 1
            return mm(up(mm(M[iid][m]), hw)) >= (0.55 if ro == "t55" else 0.5)
        panels = [("photo", None), ("truth", t), ("ours", mask(REF, "t55")), ("DiffSeg", mask("15|C|D-|DSR", "t50")),
                  ("iSeg", mask("21|iseg|I-|ERS", "comp")), (f"{bm} @{bro}", mask(bm, bro))]
        h = 256; w = round(img.width * h / img.height); canvas = Image.new("RGB", (w * len(panels), h + 18), "white")
        base = np.asarray(img.resize((w, h))).astype(np.float32)
        for k, (nm, m) in enumerate(panels):
            p = base.copy()
            if m is not None:
                mr = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h))) > 127
                p[mr] = 0.45 * p[mr] + 0.55 * np.array([255, 40, 40])
            canvas.paste(Image.fromarray(p.astype(np.uint8)), (k * w, 18)); ImageDraw.Draw(canvas).text((k * w + 3, 3), nm[:34], fill="black")
        o = vd / f"r{r}{a.tag}_{iid}_{x['cls'].replace(' ', '_')}.jpg"; canvas.save(o, quality=80); print(f"wrote {o}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    for c in ("items", "caption", "capture", "score", "viz"):
        q = sp.add_parser(c); q.add_argument("--round", type=int, required=True); q.add_argument("--limit", type=int, default=0)
        q.add_argument("--tag", default=""); q.add_argument("--n", type=int, default=8)
        if c == "capture":
            q.add_argument("--bb", choices=("15", "21"), required=True)
    q = sp.add_parser("report"); q.add_argument("--rounds", default=""); q.add_argument("--boot", type=int, default=2000); q.add_argument("--tag", default="")
    a = p.parse_args()
    {"items": items, "caption": caption, "capture": capture, "score": score, "report": report, "viz": viz}[a.cmd](a)
