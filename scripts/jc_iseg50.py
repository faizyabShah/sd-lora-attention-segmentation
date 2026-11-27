#!/usr/bin/env python3
"""Run 50: what in iSeg beats us? Its seed taken apart one ingredient at a time, on 100 hard COCO train2017 photos.

Exploratory (runs/50_iseg_parts/PREREG.md). Every map compared comes from one machine; all passes except iSeg's own use our
pass code (bicubic 512, VAE mean, CPU noise seed 0, fp32, t = 100) so a prompt change is the only change.

Photos (items): COCO train2017 development pool of run 49 (profile / fit photos excluded), in run 49's permutation
(seed 49), from position 2,000 on (run 49's rounds use the start of that order). "Hard", fixed from annotations only: >= 3
primary classes (area >= 1%) and >= 1 primary item < 5% of the photo. The first 100 such photos; every primary item.

Prompts (one pass each; target = the item's class; iSeg's class names and token selection, data/coco14/coco_name80.py):
  T    "A photograph of a {word}"                                       ours (paper 2, frozen)
  TE   T, word embeddings x 1.6
  S1   "a photograph of {c} and other object and background", x 1.6      iSeg's format, target class only (label-free)
  S2   S1 + iSeg's COCO background words (bg_context as released)        (label-free)
  S2n  S2 without x 1.6
  B    S2 format with the target + every COCO class named in a BLIP-large unconditional caption of the photo (label-free)
  F    S2 format with the target + a fixed generic list (person, car, chair, dining table, cup, bottle; first 4 != target)
  K    S2 format with every primary class of the photo = iSeg's prompt text (uses the class list; one pass per photo)
  Kn   K without x 1.6
  iseg iSeg's own pass (SD 2.1 only; jc_swap47.Runner.iseg: its preprocessing, CUDA noise seed 1101, = K's text)
Seeds (from each pass, for the target's tokens):
  U- our uniform 11-module cross-attention      UL  U- with our MK4-LR gate
  I- iSeg's aggregation (16/32 modules 0.7/0.3, head mean, per-token sum normalisation, DFC_KL clustering, token mean)
  IL I- x the 64 x 64 MK4-LR gate map           In  I- without the clustering
Refinements: seed | V0 ours | ERS iSeg's (COCO settings). Read-outs at scoring as run 49: t55, t50, comp (class list), best.
Map name: <bb>|<prompt>|<seed>|<refinement>, bb 15 = SD 1.5, 21 = SD 2.1-base.

items    (CPU)              -> <W>/r0/items.json, photos.json, masks
caption  (GPU)              BLIP-large unconditional caption per photo; class lists for prompt B
capture  (GPU) --bb 15|21   maps (+ iSeg's own pass on 21); asserts our seed from the new recorder == make_seed
score    (CPU)              counts and error measures                      -> <W>/r0/scores.json.gz
report   (CPU)              runs/50_iseg_parts/result.md
viz      (CPU)              panels for the items where iSeg beats us most   -> runs/50_iseg_parts/viz/
"""
from __future__ import annotations

import argparse, gzip, json, math, os, random, re, sys, time
from collections import defaultdict
from functools import reduce
from operator import add
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_diag49 as D  # noqa: E402  (Table, diag_row, maps_of, masks layout; imports jc_swap47 and jc_sd21_47)

S, P47, CACHE = D.S, D.P47, D.CACHE
W = Path(os.environ.get("ISEG50_WORK", str(CACHE / "jc_iseg50")))
D.W = W                                                       # D.rdir(0) = W / "r0", D.maps_of reads W / "r0" / maps_*
OUT = ROOT / "runs/50_iseg_parts"
COCO = D.COCO
N_PHOTO, MIN_AREA, SMALL, MIN_CLS, START = 100, 0.01, 0.05, 3, 2000
ENH = 1.6
FIXED = ["person", "car", "chair", "dining table", "cup", "bottle"]
ITEM_PROMPTS = ("T", "TE", "S1", "S2", "S2n", "B", "F")
PHOTO_PROMPTS = ("K", "Kn")
SEEDS, REFS = ("U-", "UL", "I-", "IL", "In"), ("seed", "V0", "ERS")
REF = "15|T|UL|V0"
TAIL = " and other object and background"
PERSON = ["person", "persons", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children",
          "kid", "kids", "player", "players", "guy", "guys", "lady", "ladies", "skier", "skiers", "surfer", "surfers",
          "skateboarder", "rider", "riders", "baby", "crowd"]


def rdir():
    return W / "r0"


# ------------------------------------------------------------------ items
def items(a):
    import jc_ov42 as O
    C40 = O.C40
    excl = {int(x["img"]) for x in json.loads((CACHE / "jc_ov42/items_prof.json").read_text())["items"]}
    excl |= {int(x["img"]) for x in json.loads((CACHE / "jc_real44/ours/items_prof44.json").read_text())["jobs"]}
    excl |= set(json.loads((D.SHIP / "fit40_photos.json").read_text()))
    d = json.loads((COCO / "annotations/instances_train2017.json").read_text())
    cat = {c["id"]: O.C2V.get(c["name"], c["name"]) for c in d["categories"]}; imgs = {i["id"]: i for i in d["images"]}
    by = defaultdict(list)
    for an in d["annotations"]:
        by[an["image_id"]].append((cat[an["category_id"]], an))
    pool = sorted(i for i in by if i not in excl); random.Random(D.PERM_SEED).shuffle(pool)       # run 49's order
    print(f"development pool: {len(pool)} photos; run 50 starts at position {START}", flush=True)
    md = W / "masks"; md.mkdir(parents=True, exist_ok=True); X, P, k, seen = [], [], START, 0
    while len(P) < N_PHOTO:
        i = pool[k]; k += 1; im = imgs[i]; h, w = im["height"], im["width"]; anns = by[i]
        fa = defaultdict(float)                                         # fast filter on annotation areas
        for nm, an in anns:
            if not an["iscrowd"]:
                fa[nm] += an["area"] / (h * w)
        if sum(v >= MIN_AREA * 0.8 for v in fa.values()) < MIN_CLS:
            continue
        seen += 1; truth, crowd, inst = {}, {}, {}
        for nm, an in anns:
            m = C40.rasterize(an["segmentation"], h, w)
            if an["iscrowd"]:
                crowd[nm] = crowd.get(nm, np.zeros((h, w), bool)) | m
            else:
                truth[nm] = truth.get(nm, np.zeros((h, w), bool)) | m; inst.setdefault(nm, []).append(m)
        prim = sorted(c for c, t in truth.items() if t.mean() >= MIN_AREA)
        if len(prim) < MIN_CLS or not any(truth[c].mean() < SMALL for c in prim):
            continue
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
                      "ninst": len(inst[c])})
        P.append({"path": f"train2017/{im['file_name']}", "img": int(i)})
    rdir().mkdir(parents=True, exist_ok=True)
    (rdir() / "items.json").write_text(json.dumps(X, indent=0)); (rdir() / "photos.json").write_text(json.dumps(P))
    ns = sum(x["area"] < SMALL for x in X)
    print(f"run 50: {len(P)} photos (pool positions {START}-{k - 1}; {seen} passed the fast filter), {len(X)} primary items "
          f"({ns} small < 5%), {len({x['cls'] for x in X})} classes", flush=True)


def load_items(limit=0):
    X = json.loads((rdir() / "items.json").read_text())
    if limit:
        keep = set(sorted({x["img"] for x in X})[:limit]); X = [x for x in X if x["img"] in keep]
    return X


# ------------------------------------------------------------------ prompts
def iseg_table():
    from jc_names44 import method_index
    AT = S.tokens("coco"); names = list(AT)
    return AT, names, (lambda c: method_index("iseg", "coco", names, c))


def multi(classes, bg=True):
    """iSeg's prompt text for these canonical classes (in its index order) and each class's selected token positions
    (jc_swap47.Runner.iseg); bg = append its released COCO bg_context."""
    AT, names, idx = iseg_table()
    order = sorted({idx(c): c for c in classes}.items()); text = "a photograph of "; sel, start, last = {}, [], None
    for j, (i, c) in enumerate(order):
        nm = names[i]; text += nm + " and "
        start.append(start[j - 1] + 1 + AT[last][0] if j > 0 else 4); sel[c] = [start[-1] + t for t in AT[nm][1]]; last = nm
    return text[:-5] + TAIL + (S.CFG["coco"]["bg"] if bg else ""), sel


def surface_forms():
    """canonical COCO class -> words that name it in a caption (COCO name, our word, iSeg's name; plural forms)."""
    import jc_ov42 as O
    _, names, idx = iseg_table(); coco = json.loads((COCO / "annotations/instances_val2017.json").read_text())["categories"]
    F = defaultdict(set)
    for c in coco:
        F[O.C2V.get(c["name"], c["name"])].add(c["name"])
    for c in O.CLS80:
        F[c] |= {c, names[idx(c)]}
    F["person"] |= set(PERSON); F["dining table"].add("table"); F["tv"] |= {"television", "tv"}; F["cell phone"].add("phone")
    F["potted plant"].add("plant"); F["aeroplane"] |= {"airplane", "plane", "jet"}; F["motorbike"].add("motorcycle"); F["sofa"].add("couch")
    out = {}
    for c, ws in F.items():
        out[c] = sorted({w for v in ws for w in (v, v + "s", v + "es")})
    return out


def caption_classes(cap, forms):
    t = " " + re.sub(r"[^a-z ]", " ", cap.lower()) + " "
    return sorted(c for c, ws in forms.items() if any(f" {w} " in t for w in ws))


def caption(a):
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    dev = torch.device("cuda"); proc = BlipProcessor.from_pretrained(D.BLIP)
    model = BlipForConditionalGeneration.from_pretrained(D.BLIP).to(dev).eval()
    X = load_items(a.limit); P = sorted({x["img"]: x["path"] for x in X}.items()); C = {}
    with torch.no_grad():
        for i, p in P:
            img = Image.open(COCO / p).convert("RGB")
            C[str(i)] = proc.decode(model.generate(**proc(img, return_tensors="pt").to(dev))[0], skip_special_tokens=True)
    forms = surface_forms(); BL = {}
    for x in X:
        BL[x["id"]] = sorted(set(caption_classes(C[str(x["img"])], forms)) | {x["cls"]})
    (rdir() / "captions_uncond.json").write_text(json.dumps(C, indent=0)); (rdir() / "blist.json").write_text(json.dumps(BL, indent=0))
    gt = defaultdict(set)
    for x in X:
        gt[x["img"]].add(x["cls"])
    hit = np.mean([len(set(BL[x["id"]]) & gt[x["img"]] - {x["cls"]}) / max(len(gt[x["img"]]) - 1, 1) for x in X])
    extra = np.mean([len(set(BL[x["id"]]) - gt[x["img"]]) for x in X])
    for i, p in P[:5]:
        print(i, "|", C[str(i)], flush=True)
    print(f"captions: {len(C)} photos; prompt-B lists: other primary classes named {hit:.2f} on average, "
          f"{extra:.2f} named classes not among the photo's primary classes per item", flush=True)


# ------------------------------------------------------------------ recorder and runner
class CARec:
    """Wraps a processor: when rec.want_ca, keeps every cross-attention's head-mean probabilities [N, 77]."""

    def __init__(self, inner, name, rec):
        self.inner, self.name, self.rec = inner, name, rec

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        if self.rec.want_ca and encoder_hidden_states is not None:
            ctx = encoder_hidden_states
            if attn.norm_cross:
                ctx = attn.norm_encoder_hidden_states(ctx)
            q, k = attn.to_q(hidden_states), attn.to_k(ctx); b = q.shape[0]; hd = attn.heads; d = q.shape[-1] // hd
            qh = q.view(b, -1, hd, d).transpose(1, 2)[0].float(); kh = k.view(b, -1, hd, d).transpose(1, 2)[0].float()
            self.rec.ca_all[self.name] = (qh @ kh.transpose(-1, -2) * attn.scale).softmax(-1).mean(0)
        return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)


def iseg_nocluster(full, sel):
    """jc_swap47.iseg_seed (COCO: head mean) without the DFC_KL clustering. full: {module: [heads, N, 77]}."""
    import torch
    import torch.nn.functional as F
    wl = {16: 0.7, 32: 0.3}; out = []
    for res, mods in ((16, S.CA16), (32, S.CA32)):
        v = torch.cat([full[m + ".transformer_blocks.0.attn2"] for m in mods], 0)
        v = v.permute(0, 2, 1).reshape(v.shape[0], 77, res, res).mean(0)
        v = v / v.sum(dim=(-2, -1), keepdim=True)
        out.append(wl[res] * F.interpolate(v[None], size=(64, 64), mode="bilinear", align_corners=False)[0])
    ca = torch.stack(out).sum(0).flatten(-2, -1).permute(1, 0)                            # [4096, 77]
    return torch.stack([ca[:, t].mean(1) for t in sel], dim=1)


def mmt(v):
    v = v.float(); return (v - v.min()) / (v.max() - v.min()).clamp_min(1e-12)


class R50(S.Runner):
    def __init__(self, bb):
        import torch
        R, O, U = P47.setup(); self.torch, self.R, self.O, self.U, self.bb = torch, R, O, U, bb
        if bb == "15":
            from jc_real44 import WORK as W44
            R.MODEL = D.SD15; O.WORK = CACHE / "jc_ov42"; U.WORK = W44 / "ours"
        else:
            O.WORK = Path(os.environ.get("DIAG49_PROF21", str(P47.W / "ov42")))
        wm = D.SHIP / "word_map.json"
        if wm.exists():
            R.word_map = lambda: json.loads(wm.read_text())
        self.pipe, self.rec = R.load(); rec = self.rec; rec.full = {}; rec.want_full = False; rec.want_ca = False; rec.ca_all = {}
        procs = dict(self.pipe.unet.attn_processors)
        for n in list(procs):
            nm = n.removesuffix(".processor"); p = procs[n]
            if nm in S.FULL:
                p = S.Full(p, nm, rec)
            procs[n] = CARec(p, nm, rec)
        self.pipe.unet.set_attn_processor(procs)
        assert S.ERS_SA in rec.pair, "iSeg's self-attention layer is not recorded"
        fit = torch.load(O.WORK / "fit.pt", weights_only=False)
        print(f"profiles: {O.WORK / 'fit.pt'} (model {fit.get('model', 'SD 1.5')})", flush=True)
        self.P = O.M.to_dev({"mo": fit["fit"]["20"]["mo"], "neg": fit["fit"]["20"]["neg"]})
        self.w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=R.DEV); self.Wm = R.word_map()
        self.checked = False

    def run_pass(self, img, prompt, target, enh_tokens):
        """our pass (jc_diag49.Diag.run_pass) with an explicit token list scaled by x 1.6 (empty = none)."""
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
        if enh_tokens:
            pe[:, list(enh_tokens)] *= ENH
        rec.target = tuple(target); rec.clear(); rec.ca_all = {}; rec.want_ca = True
        pipe.unet(zt, torch.tensor(R.T_NOISE, device=R.DEV), encoder_hidden_states=pe, return_dict=False)
        rec.want_ca = False
        if set(rec.ca) != rec.ca_names or set(rec.grid_qk) != set(R.SA) or set(rec.pair_qk) != rec.pair or not rec.ca_all:
            raise ValueError("missing recorded attention")

    def our_seed(self, tg, gate):
        """jc_realvoc.make_seed for token list tg, from the head-mean probabilities of this pass."""
        import torch
        import torch.nn.functional as F
        s = torch.zeros((64, 64), device=self.R.DEV)
        for i, m in enumerate(self.rec.ca_order):
            t = self.rec.ca_all[m][:, list(tg)].mean(-1)
            t = t if gate is None else t * gate[m]; side = math.isqrt(t.numel())
            s += F.interpolate(t.reshape(1, 1, side, side), size=(64, 64), mode="bilinear", align_corners=False)[0, 0] * self.w_u[i]
        return mmt(s)

    def check_seed(self, tg, gate):
        """the new recorder reproduces make_seed (the recorder our method uses) on this pass."""
        a = self.our_seed(tg, gate); b = self.R.make_seed(self.rec, self.w_u, gate)
        d = float((a - b.float()).abs().max()); assert d < 1e-4, f"seed mismatch {d}"
        if not self.checked:
            print(f"check: our seed from the new recorder vs make_seed, max |diff| {d:.2e}", flush=True); self.checked = True

    def maps(self, x, sel, out, P):
        """every seed and refinement for item x from the current pass; sel = {class: token positions} of this prompt."""
        from src.segmentation.sg25_sa import _attention, _apply
        O, rec = self.O, self.rec
        key, _ = self.U.keyword("coco", x["cls"], self.Wm); tg = sel[x["cls"]]
        g64, Q, B = self.gate64(key); ga = None if not B else O.R37.powered(B["L"], Q, 1.0)
        H64 = _attention(rec.grid_qk[64], 64)
        cls = list(sel); full = {m: v[None] for m, v in rec.ca_all.items()}
        I = S.iseg_seed(full, "coco", [sel[c] for c in cls])[:, cls.index(x["cls"])].reshape(64, 64)
        In = iseg_nocluster(full, [sel[c] for c in cls])[:, cls.index(x["cls"])].reshape(64, 64)
        u = self.our_seed(tg, None)
        seeds = {"U-": u, "UL": self.our_seed(tg, ga) if ga else u, "I-": I, "IL": I * g64 if g64 is not None else I, "In": In}
        for nm, sd in seeds.items():
            v = mmt(sd).reshape(-1); k = f"{self.bb}|{P}|{nm}"
            out[f"{k}|seed"] = v.reshape(64, 64)
            out[f"{k}|V0"] = _apply(H64, _apply(H64, v)).mean(0).reshape(64, 64)
            out[f"{k}|ERS"] = S.ers(v.reshape(-1, 1), rec.pair_qk[S.ERS_SA], "coco")[:, 0].reshape(64, 64)
        return 0 if B else 1, ga

    def item(self, x, img, blist):
        from src.segmentation.attention_aggregation import find_token_subsequence
        tok = self.pipe.tokenizer; out = {}; miss = 0; prompts = {}
        _, word = self.U.keyword("coco", x["cls"], self.Wm)
        for P in ITEM_PROMPTS:
            if P in ("T", "TE"):
                prompt = f"A photograph of a {word}"
                ids = tok(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
                sel = {x["cls"]: find_token_subsequence(ids, tok(word, add_special_tokens=False)["input_ids"])}
            else:
                cls = {"S1": [x["cls"]], "S2": [x["cls"]], "S2n": [x["cls"]], "B": blist,
                       "F": [x["cls"]] + [c for c in FIXED if c != x["cls"]][:4]}[P]
                prompt, sel = multi(cls, bg=P != "S1")
            enh = reduce(add, sel.values()) if P in ("TE", "S1", "S2", "B", "F") else []
            self.run_pass(img, prompt, sel[x["cls"]], enh)
            m, ga = self.maps(x, sel, out, P); miss += m; prompts[P] = prompt
            if P == "T":
                self.check_seed(sel[x["cls"]], ga)
            self.rec.clear()
        return out, prompts, miss

    def photo(self, xs, img):
        """K and Kn: one pass per photo with every primary class (iSeg's text), maps for each item."""
        out = defaultdict(dict); miss = 0; prompts = {}; tok = self.pipe.tokenizer
        for P in PHOTO_PROMPTS:
            prompt, sel = multi([x["cls"] for x in xs])
            ids = tok(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
            for c, pos in sel.items():                                   # the selected tokens are the class name's words
                got = tok.decode([ids[p] for p in pos]).strip(); _, names, idx = iseg_table()
                assert got and got in names[idx(c)], (c, names[idx(c)], got, prompt)
            first = not getattr(self, "checked_full", False); self.rec.want_full = first; self.rec.full = {}
            self.run_pass(img, prompt, sel[xs[0]["cls"]], reduce(add, sel.values()) if P == "K" else [])
            self.rec.want_full = False
            if first:                                                    # head-mean recorder == iSeg's per-head recording
                cl = list(sel); a_ = iseg_nocluster(self.rec.full, [sel[c] for c in cl])
                b_ = iseg_nocluster({m: v[None] for m, v in self.rec.ca_all.items()}, [sel[c] for c in cl])
                d = float(((a_ - b_).abs().max() / a_.abs().max())); assert d < 1e-4, f"aggregation mismatch {d}"
                print(f"check: iSeg aggregation from head means vs per-head, max relative diff {d:.2e}; prompt {prompt!r}", flush=True)
                self.checked_full = True; self.rec.full = {}
            for x in xs:
                m, _ = self.maps(x, sel, out[x["id"]], P); miss += m
            prompts[P] = prompt; self.rec.clear()
        return out, prompts, miss

    def own_iseg(self, xs, img):
        """iSeg's own pass (SD 2.1): I- / IL with seed / V0 / ERS (jc_swap47.Runner.iseg); checks that the new recorder
        gives the same aggregation as iSeg's per-head recording, and that its text equals prompt K's."""
        mp, text, ids, miss = self.iseg("coco", None, xs, img)
        ktext, _ = multi([x["cls"] for x in xs]); assert text == ktext, (text, ktext)
        out = {}
        for xid, m in mp.items():
            out[xid] = {f"21|iseg|{nm}{'-' if nm == 'I' else ''}|{r}": m[f"{nm}|{r}"] for nm in ("I", "IL") for r in ("seed", "V0", "ERS")}
        return out, text, miss


def save(o, mp, extra):
    names = sorted(mp)
    arr = np.stack([mp[k].float().cpu().numpy() for k in names]).astype(np.float16)
    if not np.isfinite(arr.astype(np.float32)).all():
        raise ValueError(f"non-finite map in {o.name}")
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, maps=arr, names=np.array(names), **extra); os.replace(tmp, o)


def capture(a):
    import torch
    from PIL import Image
    X = load_items(a.limit); BL = json.loads((rdir() / "blist.json").read_text())
    rn = R50(a.bb); od = rdir() / f"maps_{a.bb}"; od.mkdir(parents=True, exist_ok=True)
    oi = rdir() / "maps_iseg"; oi.mkdir(parents=True, exist_ok=True); n = 0; t_all = time.perf_counter()
    with torch.no_grad():
        for img_id, xs in D.by_photo(X):
            if all((od / f"{x['id']}.npz").exists() for x in xs) and (a.bb != "21" or (oi / f"{img_id:012d}.npz").exists()):
                continue
            img = np.asarray(Image.open(COCO / xs[0]["path"]).convert("RGB")); t0 = time.perf_counter()
            pm, pp, pmiss = rn.photo(xs, img)
            for x in xs:
                o = od / f"{x['id']}.npz"
                if not o.exists():
                    mp, prompts, miss = rn.item(x, img, BL[x["id"]]); mp.update(pm[x["id"]])
                    save(o, mp, {"prompts": json.dumps(prompts | pp), "miss": miss + pmiss})
            if a.bb == "21":
                o = oi / f"{img_id:012d}.npz"
                if not o.exists():
                    mp, text, miss = rn.own_iseg(xs, img); d = {}
                    for xid, m in mp.items():
                        names = sorted(m); d[f"m_{xid}"] = np.stack([m[k].float().cpu().numpy() for k in names]).astype(np.float16)
                    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, names=np.array(names), text=text, miss=miss, **d); os.replace(tmp, o)
            torch.cuda.empty_cache(); n += 1; dt = time.perf_counter() - t0
            npass = len(xs) * len(ITEM_PROMPTS) + len(PHOTO_PROMPTS) + (a.bb == "21")
            print(f"[iseg50:{a.bb}] photo {img_id}: {len(xs)} items, {npass} passes, {dt:.1f}s ({npass / dt:.2f} passes/s), "
                  f"miss {pmiss}", flush=True)
    print(f"[iseg50:{a.bb}] done: {n} photos in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ scoring (jc_diag49.score, one photo per worker)
def score_photo(job):
    from scipy.ndimage import binary_dilation, binary_erosion
    img_id, xs = job; M = D.maps_of(0, xs); G = {}; recs = []
    for x in xs:
        z = np.load(W / "masks" / f"{x['id']}.npz"); t = z["truth"]; valid = ~z["ignore"]
        b = max(1, round(0.015 * max(t.shape)))
        band = binary_dilation(t, iterations=b) & ~binary_erosion(t, iterations=b)
        rng = np.random.default_rng(int(x["id"][1:13]) * 100 + int(x["id"][-2:]))
        vi = np.flatnonzero(valid.reshape(-1)); sub = rng.choice(vi, size=min(20000, vi.size), replace=False)
        G[x["id"]] = dict(t=t, valid=valid, other=z["other"], inst=z["inst"], band=band, sub=sub)
    names = sorted(set.intersection(*[set(M[x["id"]]) for x in xs])) if all(x["id"] in M for x in xs) else []
    for n in names:
        V = {x["id"]: D.mm(D.up(D.mm(M[x["id"]][n]), G[x["id"]]["t"].shape)) for x in xs}
        comp = S.cam_to_label(np.stack([V[x["id"]] for x in xs]), 0.45)
        for j, x in enumerate(xs):
            g = G[x["id"]]; t, valid = g["t"], g["valid"]; v = V[x["id"]]
            pr = v[None] >= D.THR[:, None, None]; tv = t & valid
            tp = (pr & tv).sum((1, 2)); fp = (pr & valid & ~t).sum((1, 2)); fn = (~pr & tv).sum((1, 2))
            rec = dict(id=x["id"], img=x["img"], cls=x["cls"], area=x["area"], ninst=x["ninst"], map=n,
                       tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist())
            pc = (comp == j + 1); rec.update(ctp=int((pc & tv).sum()), cfp=int((pc & valid & ~t).sum()), cfn=int((~pc & tv).sum()))
            rec["auroc"] = D.auroc(v.reshape(-1)[g["sub"]], t.reshape(-1)[g["sub"]])
            rec["point"] = bool(t.reshape(-1)[int(np.argmax(v))])
            vv = v * valid; rec["mass"] = float((vv * t).sum() / max(vv.sum(), 1e-12))
            for k, i in (("55", D.I55), ("50", D.I50)):
                p = pr[i] & valid
                rec[f"fpo{k}"] = int((p & g["other"]).sum()); rec[f"fpb{k}"] = int((p & ~t & ~g["other"]).sum())
                hit = [float((p & (g["inst"] == q)).sum()) / max(int((g["inst"] == q).sum()), 1) >= 0.5 for q in range(1, x["ninst"] + 1)]
                rec[f"ih{k}"] = float(np.mean(hit)) if hit else float("nan")
                keep = valid & ~g["band"]; rec[f"btp{k}"] = int((p & t & keep).sum()); rec[f"bun{k}"] = int(((p | t) & keep).sum())
            recs.append(rec)
    return img_id, len(xs), len(names), recs


def score(a):
    from multiprocessing import get_context
    X = load_items(a.limit); jobs = D.by_photo(X); recs = []
    with get_context("fork").Pool(a.workers) as p:
        for img_id, nx, nn, r in p.imap_unordered(score_photo, jobs):
            recs += r; print(f"[score50] photo {img_id}: {nx} items, {nn} maps", flush=True)
    o = rdir() / f"scores{a.tag}.json.gz"
    with gzip.open(o, "wt") as f:
        json.dump(recs, f)
    print(f"wrote {o}: {len(recs)} records", flush=True)


# ------------------------------------------------------------------ report
def report(a):
    with gzip.open(rdir() / f"scores{a.tag}.json.gz", "rt") as f:
        recs = [dict(x, round=0) for x in json.load(f)]
    T = D.Table(recs, a.boot); have = lambda c: bool(T.by.get(c[0]))
    small = [x for x in recs if x["area"] < SMALL]; large = [x for x in recs if x["area"] >= SMALL]
    TS, TL = D.Table(small, a.boot), D.Table(large, a.boot)
    nI = len({x["id"] for x in recs}); nS = len({x["id"] for x in small})
    f3 = lambda c, t=T: f"{t.miou(*c):.3f}" if t.by.get(c[0]) else "–"
    L = [f"# Run 50: iSeg's seed taken apart on 100 hard COCO photos ({'smoke, ' if a.tag else ''}exploratory; `PREREG.md`)", "",
         f"COCO train2017 development photos, hard by annotation (>= 3 primary classes, >= 1 item < 5%): {len(T.photos)} photos, "
         f"{nI} primary items ({nS} small < 5%), {len(T.classes)} classes. Class-macro mIoU; Δ [95% paired photo bootstrap, "
         f"{a.boot}]; **+** / **−** = CI excludes 0. The photos are selected to be hard, so levels are lower than on random "
         "photos; a lead needs a confirmation on random photos.", "",
         "Cells `bb|prompt|seed|refinement`. Prompts: T ours; TE ours x1.6; S1 iSeg's format with the target only (+ 'other "
         "object and background', x1.6); S2 S1 + iSeg's background words; S2n S2 without x1.6; B S2 with the classes a BLIP "
         "caption names; F S2 with a fixed generic list; K iSeg's prompt with every class of the photo (class list); Kn K "
         "without x1.6; iseg iSeg's own pass. Seeds: U-/UL ours without / with gate; I-/IL iSeg's aggregation without / with "
         "the gate map; In iSeg's without clustering. Read-outs: t55 ours; comp iSeg's competition (class list).", ""]
    ours = (REF, "t55")
    # 1. references
    L += ["## 1. Reference cells", "", "| Pipeline | Cell | read-out | all | small | not small |", "|---|---|---|---:|---:|---:|"]
    refs = [("ours (SD 1.5, frozen)", (REF, "t55")), ("ours on SD 2.1", ("21|T|UL|V0", "t55")),
            ("iSeg's prompt through our pass", ("21|K|I-|ERS", "comp")), ("iSeg's own pass", ("21|iseg|I-|ERS", "comp")),
            ("iSeg's own pass, our read-out", ("21|iseg|I-|ERS", "t55"))]
    for nm, c in refs:
        if have(c):
            L.append(f"| {nm} | `{c[0]}` | {c[1]} | {f3(c)} | {f3(c, TS)} | {f3(c, TL)} |")
    # 2. ladders
    for bb in ("21", "15"):
        steps = [("ours", (f"{bb}|T|UL|V0", "t55")),
                 ("+ iSeg's format: 'and other object and background', x1.6", (f"{bb}|S1|UL|V0", "t55")),
                 ("+ iSeg's background words", (f"{bb}|S2|UL|V0", "t55")),
                 ("+ iSeg's aggregation (gated)", (f"{bb}|S2|IL|V0", "t55")),
                 ("+ iSeg's refinement ERS", (f"{bb}|S2|IL|ERS", "t55")),
                 ("+ the photo's class list in the prompt [labels]", (f"{bb}|K|IL|ERS", "t55")),
                 ("− our gate (= iSeg's seed and refinement)", (f"{bb}|K|I-|ERS", "t55")),
                 ("+ iSeg's competition read-out [labels]", (f"{bb}|K|I-|ERS", "comp"))]
        if bb == "21":
            steps.append(("+ iSeg's own pass (preprocessing, noise)", ("21|iseg|I-|ERS", "comp")))
        L += ["", f"## 2{'a' if bb == '21' else 'b'}. Ladder from ours to iSeg, SD {'2.1' if bb == '21' else '1.5'} (each step Δ vs the step before)", "",
              "| Step | Cell | mIoU | Δ step | Δ vs ours | small | not small |", "|---|---|---:|---|---|---:|---:|"]
        prev = None
        for nm, c in steps:
            if not have(c):
                continue
            ds = D.fmt_d(T.delta(c, prev)) if prev else ""; dv = D.fmt_d(T.delta(c, steps[0][1])) if prev else ""
            L.append(f"| {nm} | `{c[0]}` @{c[1]} | {f3(c)} | {ds} | {dv} | {f3(c, TS)} | {f3(c, TL)} |"); prev = c
    # 3. class list vs label-free competitors
    L += ["", "## 3. Is it the class list? Competing words in the prompt (Δ vs S2, the same format with the target only; t55)", "",
          "| bb | seed | refinement | K − S2 (class list) [labels] | B − S2 (BLIP-caption classes) | F − S2 (fixed list) | K − S2, small items |",
          "|---|---|---|---|---|---|---|"]
    for bb in ("15", "21"):
        for sd in ("UL", "IL", "I-"):
            for rf in ("V0", "ERS"):
                b = (f"{bb}|S2|{sd}|{rf}", "t55")
                if have(b):
                    cell = lambda P: D.fmt_d(T.delta((f"{bb}|{P}|{sd}|{rf}", "t55"), b)) if have((f"{bb}|{P}|{sd}|{rf}", "t55")) else "–"
                    ks = D.fmt_d(TS.delta((f"{bb}|K|{sd}|{rf}", "t55"), b)) if TS.by.get(f"{bb}|K|{sd}|{rf}") else "–"
                    L.append(f"| {bb} | {sd} | {rf} | {cell('K')} | {cell('B')} | {cell('F')} | {ks} |")
    # 4. reverse ladder: one ingredient removed from iSeg (via our pass, SD 2.1, comp and t55)
    L += ["", "## 4. One ingredient removed from iSeg (SD 2.1, iSeg's prompt through our pass, `21|K|I-|ERS`)", "",
          "| Removed | Cell | Δ @comp | Δ @t55 |", "|---|---|---|---|"]
    base = "21|K|I-|ERS"
    for nm, m in (("the class list (target only, background words kept)", "21|S2|I-|ERS"), ("class list and background words", "21|S1|I-|ERS"),
                  ("x1.6", "21|Kn|I-|ERS"), ("the clustering", "21|K|In|ERS"), ("iSeg's aggregation (ours instead)", "21|K|U-|ERS"),
                  ("ERS (our V0 instead)", "21|K|I-|V0"), ("ERS (no refinement)", "21|K|I-|seed"), ("SD 2.1 (SD 1.5 instead)", "15|K|I-|ERS")):
        if have((m, "t55")) and have((base, "t55")):
            L.append(f"| {nm} | `{m}` | {D.fmt_d(T.delta((m, 'comp'), (base, 'comp')))} | {D.fmt_d(T.delta((m, 't55'), (base, 't55')))} |")
    # 5. factor averages
    grid = [m for m in T.by if m.count("|") == 3 and m.split("|")[1] != "iseg"]
    def fac(i, a_, b_, tab=T):
        d = []
        for m in grid:
            p = m.split("|")
            if p[i] == a_:
                q = "|".join(b_ if k == i else s for k, s in enumerate(p))
                if q in tab.by and tab.by.get(m):
                    d.append(tab.miou(q, "t55") - tab.miou(m, "t55"))
        return (np.mean(d), len(d)) if d else (float("nan"), 0)
    L += ["", "## 5. Factor averages (mean Δ at t55 over every combination of the other factors)", "",
          "| Change | all | small items | not small | cells |", "|---|---:|---:|---:|---:|"]
    for nm, i, a_, b_ in (("T → TE (x1.6 on ours)", 1, "T", "TE"), ("T → S1 (iSeg's format)", 1, "T", "S1"),
                          ("S1 → S2 (background words)", 1, "S1", "S2"), ("S2 → S2n (drop x1.6)", 1, "S2", "S2n"),
                          ("S2 → K (class list) [labels]", 1, "S2", "K"), ("S2 → B (caption classes)", 1, "S2", "B"),
                          ("S2 → F (fixed list)", 1, "S2", "F"), ("K → Kn (drop x1.6)", 1, "K", "Kn"),
                          ("U- → UL (gate)", 2, "U-", "UL"), ("I- → IL (gate map)", 2, "I-", "IL"), ("U- → I- (aggregation)", 2, "U-", "I-"),
                          ("I- → In (drop clustering)", 2, "I-", "In"), ("seed → V0", 3, "seed", "V0"), ("V0 → ERS", 3, "V0", "ERS"),
                          ("SD 1.5 → SD 2.1", 0, "15", "21")):
        (m_, n_), (ms, _), (ml, _) = fac(i, a_, b_), fac(i, a_, b_, TS), fac(i, a_, b_, TL)
        L.append(f"| {nm} | {m_:+.3f} | {ms:+.3f} | {ml:+.3f} | {n_} |")
    # 6. error measures
    L += ["", "## 6. What goes wrong (t55; item means; columns as run 49)", "",
          "| Cell | n | IoU | oracle | AUROC | point | mass | prec | recall | FP-obj | inst | IoU-nb | small | mid | large |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in (REF, "15|S2|UL|V0", "15|B|UL|V0", "15|K|UL|V0", "21|T|UL|V0", "21|S2|IL|ERS", "21|K|I-|ERS", "21|iseg|I-|ERS"):
        d = D.diag_row(recs, m, "55")
        if d:
            L.append(f"| `{m}` | {d['n']} | {d['iou']:.3f} | {d['orc']:.3f} | {d['auroc']:.3f} | {d['point']:.2f} | {d['mass']:.2f} | "
                     f"{d['prec']:.2f} | {d['rec']:.2f} | {d['fpo']:.2f} | {d['inst']:.2f} | {d['biou']:.3f} | {d['small']:.3f} | "
                     f"{d['mid']:.3f} | {d['large']:.3f} |")
    # 7. lead check (registered)
    L += ["", "## 7. Lead check: label-free changes to ours on SD 1.5 (Δ vs `15|T|UL|V0` @t55; lead = Δ >= +0.02 and CI low > 0)", "",
          "| Cell | Δ all | Δ small | lead |", "|---|---|---|---|"]
    for P in ("TE", "S1", "S2", "S2n", "B", "F"):
        for sd in ("UL", "IL"):
            for rf in ("V0", "ERS"):
                c = (f"15|{P}|{sd}|{rf}", "t55")
                if have(c):
                    d = T.delta(c, ours); ds = TS.delta(c, ours) if TS.by.get(c[0]) else None
                    L.append(f"| `{c[0]}` | {D.fmt_d(d)} | {D.fmt_d(ds) if ds else '–'} | {'**lead**' if d[0] >= 0.02 and d[1] > 0 else ''} |")
    # 8. every cell
    L += ["", "## 8. Every cell (mIoU; t55 / t50 / comp / best fixed threshold / small items t55)", "",
          "| Cell | t55 | t50 | comp | best | small |", "|---|---:|---:|---:|---:|---:|"]
    for m in sorted(T.by, key=lambda m: -T.miou(m, "t55")):
        b, th = T.best(m); L.append(f"| `{m}` | {T.miou(m, 't55'):.3f} | {T.miou(m, 't50'):.3f} | {T.miou(m, 'comp'):.3f} | {b:.3f} @ {th:.2f} | {f3((m, 't55'), TS)} |")
    # gaps: where iSeg (own pass, comp) beats ours most
    iou = lambda x, ro: (x["ctp"] / max(x["ctp"] + x["cfp"] + x["cfn"], 1)) if ro == "comp" else x["tp"][D.I55] / max(x["tp"][D.I55] + x["fp"][D.I55] + x["fn"][D.I55], 1)
    by = defaultdict(dict)
    for x in recs:
        by[x["id"]][x["map"]] = x
    G = []
    for k, d in by.items():
        if REF in d and "21|iseg|I-|ERS" in d:
            o, s = iou(d[REF], "t55"), iou(d["21|iseg|I-|ERS"], "comp")
            s2 = iou(d["15|S2|UL|V0"], "t55") if "15|S2|UL|V0" in d else float("nan")
            G.append((s - o, k, d[REF]["cls"], d[REF]["area"], o, s2, s))
    G.sort(key=lambda g: -g[0])
    L += ["", "## 9. Items where iSeg's own pass beats ours most (IoU)", "", "| Item | class | area | ours | ours, S2 prompt | iSeg |",
          "|---|---|---:|---:|---:|---:|"]
    for g in G[:15]:
        L.append(f"| `{g[1]}` | {g[2]} | {g[3]:.3f} | {g[4]:.3f} | {g[5]:.3f} | {g[6]:.3f} |")
    OUT.mkdir(parents=True, exist_ok=True); o = OUT / f"result{a.tag}.md"; o.write_text("\n".join(L) + "\n")
    (OUT / f"gaps{a.tag}.json").write_text(json.dumps([g[1] for g in G[:15]])); print(f"wrote {o}", flush=True)


def viz(a):
    """photo | truth | ours | ours with S2 | iSeg's prompt through our pass (comp) | iSeg's own pass (comp)."""
    from PIL import Image, ImageDraw
    G = json.loads((OUT / f"gaps{a.tag}.json").read_text())[: a.n]; X = {x["id"]: x for x in load_items()}
    vd = OUT / "viz"; vd.mkdir(parents=True, exist_ok=True)
    for iid in G:
        x = X[iid]; xs = [y for y in X.values() if y["img"] == x["img"]]; M = D.maps_of(0, xs)
        img = Image.open(COCO / x["path"]).convert("RGB"); hw = (img.height, img.width)
        t = np.load(W / "masks" / f"{iid}.npz")["truth"]
        def mask(m, ro):
            if m not in M[iid]:
                return None
            if ro == "comp":
                cams = np.stack([D.mm(D.up(D.mm(M[y["id"]][m]), hw)) for y in xs]); j = [y["id"] for y in xs].index(iid)
                return S.cam_to_label(cams, 0.45) == j + 1
            return D.mm(D.up(D.mm(M[iid][m]), hw)) >= 0.55
        panels = [("photo", None), (f"truth: {x['cls']}", t), ("ours", mask(REF, "t55")), ("ours, S2 prompt", mask("15|S2|UL|V0", "t55")),
                  ("iSeg prompt, our pass", mask("21|K|I-|ERS", "comp")), ("iSeg own", mask("21|iseg|I-|ERS", "comp"))]
        h = 256; w = round(img.width * h / img.height); canvas = Image.new("RGB", (w * len(panels), h + 18), "white")
        base = np.asarray(img.resize((w, h))).astype(np.float32)
        for k, (nm, m) in enumerate(panels):
            p = base.copy()
            if m is not None:
                mr = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h))) > 127
                p[mr] = 0.45 * p[mr] + 0.55 * np.array([255, 40, 40])
            canvas.paste(Image.fromarray(p.astype(np.uint8)), (k * w, 18)); ImageDraw.Draw(canvas).text((k * w + 3, 3), nm[:34], fill="black")
        o = vd / f"{iid}_{x['cls'].replace(' ', '_')}{a.tag}.jpg"; canvas.save(o, quality=80); print(f"wrote {o}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("items")
    q = sp.add_parser("caption"); q.add_argument("--limit", type=int, default=0)
    q = sp.add_parser("capture"); q.add_argument("--bb", choices=("15", "21"), required=True); q.add_argument("--limit", type=int, default=0)
    q = sp.add_parser("score"); q.add_argument("--limit", type=int, default=0); q.add_argument("--workers", type=int, default=16)
    q.add_argument("--tag", default="")
    q = sp.add_parser("report"); q.add_argument("--boot", type=int, default=2000); q.add_argument("--tag", default="")
    q = sp.add_parser("viz"); q.add_argument("--n", type=int, default=10); q.add_argument("--tag", default="")
    a = p.parse_args()
    {"items": items, "caption": caption, "capture": capture, "score": score, "report": report, "viz": viz}[a.cmd](a)
