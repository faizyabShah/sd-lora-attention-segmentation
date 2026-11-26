#!/usr/bin/env python3
"""Run 47: SD 2.1-base for everything; swap seed, refinement and read-out between our method and iSeg (env trdp).

Two SD 2.1-base passes (fp32, t = 100), both recorded with the same recorder (jc_realvoc.Proc on SD 2.1, plus the full
cross-attention of iSeg's 16 / 32 modules):
  P_ours  per (photo, class) item: our pass (jc_realvoc.one_pass: bicubic 512, VAE mean, CPU noise seed 0, prompt
          "A photograph of a {word}")
  P_iseg  per photo: iSeg's pass, transcribed from the release (ovs/diff_for_openvocabulary.py test_step + wsss/iSeg.py +
          base/stable_difusion.py): photo / 255, bilinear 512, 2x - 1, VAE mean x 0.18215, CUDA noise seed 1101 (fp16 draw),
          prompt "a photograph of <c1> and ... and other object and background" + bg_context with the photo's classes
          (= run 44's "iSeg, class given": every primary class of the photo), class-token embeddings x 1.6
          (category enhancement). The prompt text and class ids are asserted equal to run 44's saved iSeg outputs.
Seeds (64 x 64, from the seed's own pass):
  O-  our uniform cross-attention seed (11 modules), no gate               [P_ours]
  OL  O- with our MK4-LR gate (SD 2.1 profiles, gamma 1)                    [P_ours]
  I   iSeg's category-enhanced cross-attention (16/32 modules, weights 0.7/0.3, head subset on VOC, per-token sum
      normalisation, DFC_KL clustering smoothing, class-token mean)        [P_iseg]
  IL  I x our MK4-LR gate (the 64 x 64 consensus from P_iseg's own queries)  [P_iseg]
Refinements (on the same pass's self-attention):
  V0   ours: two steps of the 64 x 64 self-attention (sg25 module), from the min-max seed
  ERS  iSeg's entropy-reduced self-attention (up_blocks.3.attentions.1 attn1, heads 1-2), its released iter / ent per
       column, iterative min-subtract / sum-normalise
Read-outs (at scoring): bin = min-max, bilinear to the photo, >= 0.55 (ours); comp = iSeg's cam_to_label over the photo's
classes (released cam_bg_thr per column).

items     (CPU) per column: items (run 44 primary) and photos with their classes
capture   (GPU) --pass ours|iseg --set S                        -> <W>/swap_<pass>_<set>/<item id | photo>.npz
check     (GPU) P_ours maps == jc_ours44.capture on SD 2.1 (3 items); P_iseg text / ids == run 44; label agreement
score     (CPU) per photo: bin and comp counts for every cell   -> <W>/swap_scores_<set>/<photo>.npz
report    (CPU)                                                 -> runs/47_sd21_swap/result.md
Profiles: jc_sd21_47 prof / dump42 / fit42 / dump44 / fit44 (SD 2.1, the same photos and recipe as runs 42 / 44).
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from functools import reduce
from operator import add
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_sd21_47 as P47  # noqa: E402

W = P47.W
CACHE = P47.CACHE
ISEG = ROOT / "other repositories/iseg"
REPORT = ROOT / "runs/47_sd21_swap/result.md"
COCO_ROOT = ROOT / "datasets/coco2017"
CA16 = ["down_blocks.2.attentions.0", "down_blocks.2.attentions.1", "up_blocks.1.attentions.0", "up_blocks.1.attentions.1",
        "up_blocks.1.attentions.2"]
CA32 = ["down_blocks.1.attentions.1", "up_blocks.2.attentions.0", "up_blocks.2.attentions.1"]
FULL = [m + ".transformer_blocks.0.attn2" for m in CA16 + CA32]
ERS_SA = "up_blocks.3.attentions.1.transformer_blocks.0.attn1"
HEADS = {32: [22, 5, 21, 13, 8, 27, 28, 6, 25],
         16: [76, 1, 43, 17, 81, 6, 44, 27, 2, 8, 22, 20, 60, 78, 12, 83, 94, 47, 88, 96, 3, 33, 46, 52, 77, 93, 51, 58, 0, 13,
              14, 19, 34, 41, 59, 87, 16, 24, 28, 30, 32]}          # wsss/iSeg.py, used when att_mean is False (VOC)
# released per-column settings (ovs/arguments.py) and bg_context as the released code evaluates it (wsss/iSeg.py; the
# continuation lines there are separate statements, so only the first literal is assigned)
CFG = {"voc": dict(iter=8, ent=0.015, att_mean=False, bkg=0.0, tab="data/voc/voc_name20.py", var="VOC12_NAME",
                   bg="ground, land, grass, tree, building, wall, sky, lake, water"),
       "context": dict(iter=5, ent=0.002, att_mean=True, bkg=0.0, tab="data/voc_context/context_name59.py", var="VOC_CONTEXT_NAME", bg=""),
       "coco": dict(iter=10, ent=0.2, att_mean=True, bkg=0.45, tab="data/coco14/coco_name80.py", var="COCO14_NAME",
                    bg="ground, land, grass, tree, building, wall, sky, lake, water, river, sea, railway, ")}
ENHANCE, ISEG_SEED, T_NOISE = 1.6, 1101, 100
SEEDS = ("O-", "OL", "I", "IL")
CELLS = tuple(f"{s}|{r}" for s in SEEDS for r in ("seed", "V0", "ERS"))
COLS = [("voc", "PASCAL VOC (20)"), ("context", "PASCAL Context (59)"), ("coco", "COCO-Object (80)")]


def tokens(s):
    g = {}; exec((ISEG / CFG[s]["tab"]).read_text(), g); return g[CFG[s]["var"]]


def photos(s):
    from jc_real44 import WORK as W44
    P = {}
    for x in json.loads((W44 / "items.json").read_text())[s]:
        if x["primary"]:
            P.setdefault(str(x["img"]), []).append(x)
    return sorted(P.items())


def iname(s, img):
    return f"{int(img):012d}" if s == "coco" else str(img)


def photo_path(s, x):
    return COCO_ROOT / x["path"] if s == "coco" else ROOT / x["path"]


# ------------------------------------------------------------------ iSeg pieces (transcribed)
def kl_dis(anchor, attn):                       # util/attn.py
    import torch
    anchor, attn = anchor.type(torch.float16) + 1e-5, attn.type(torch.float16) + 1e-5
    anchor, attn = anchor.unsqueeze(-2), attn.unsqueeze(-3)
    q = torch.log(anchor) - torch.log(attn)
    return torch.sum(anchor * q, dim=-1) * 0.5 - torch.sum(attn * q, dim=-1) * 0.5


def dfc_kl(x, number_cluster=32, stage_num=20, fea_size=64):     # wsss/featurecluster.py DFC_KL(32, 20, 64)
    import torch
    sqrt = number_cluster ** 0.5
    grid = torch.arange(0, fea_size, fea_size // sqrt)[None].long(); grid = grid.T * fea_size + grid
    x = x.flatten(-2, -1).T; x = torch.where(x == 0, 1e-6, x)
    prototype, cluster = x[grid.flatten().to(x.device)], None
    for _ in range(stage_num):
        dis = kl_dis(prototype, x); _, cluster = torch.min(dis, dim=0); idx = cluster.unique()
        prototype = torch.stack([x[cluster == d].mean(0) for d in idx], dim=0)
    idx = cluster.unique()
    for i_ in idx:
        cluster[cluster == i_] = torch.nonzero(idx == i_).squeeze()
    return cluster, len(idx)


def iseg_seed(full, s, sel):
    """wsss/iSeg.py process_cross_att with clustering (no_use_cluster defaults to True). full: {module: [heads, N, 77]}."""
    import torch
    import torch.nn.functional as F
    wl = {16: 0.7, 32: 0.3}; out = []
    for res, mods in ((16, CA16), (32, CA32)):
        v = torch.cat([full[m + ".transformer_blocks.0.attn2"] for m in mods], 0)         # [H, N, 77], execution order
        v = v.permute(0, 2, 1).reshape(v.shape[0], 77, res, res)                          # [H, 77, h, w]
        v = v.mean(0) if CFG[s]["att_mean"] else v[HEADS[res]].mean(0)
        v = v / v.sum(dim=(-2, -1), keepdim=True)
        out.append(wl[res] * F.interpolate(v[None], size=(64, 64), mode="bilinear", align_corners=False)[0])
    ca = torch.stack(out).sum(0)                                                          # [77, 64, 64]
    clusters, n = dfc_kl(ca); one = F.one_hot(clusters, n); sa = one[:, clusters].float()
    ca = (sa @ ca.flatten(-2, -1).permute(1, 0)) / sa.sum(-1, keepdim=True)               # [4096, 77]
    return torch.stack([ca[:, t].mean(1) for t in sel], dim=1)                          # [4096, C]


def ers(seed_cols, qk, s):
    """wsss/iSeg.py get_att_map (no_use_self_ers): entropy-reduced self-attention, iterated. seed_cols [4096, C]."""
    import torch
    q, k = qk; q, k = q.float(), k.float()                                                # [heads, 4096, d]
    A = torch.stack([(q[h] @ k[h].T * (q.shape[-1] ** -0.5)).softmax(-1) for h in (1, 2)]).mean(0)   # [query, key]
    T = A / torch.amax(A, dim=-2, keepdim=True)
    T = T + torch.where(T == 0, 0, CFG[s]["ent"] * torch.log10(torch.e * T)); T = torch.clamp(T, min=0)
    T = T / T.sum(dim=-1, keepdim=True)
    x = seed_cols - seed_cols.amin(dim=-2, keepdim=True); x = x / x.sum(dim=-2, keepdim=True)
    for _ in range(CFG[s]["iter"]):
        x = T @ x; x = x - x.amin(dim=-2, keepdim=True); x = x / x.sum(dim=-2, keepdim=True)
    return x


def cam_to_label(cam, bkg):
    """util/cam.py cam_to_label for the photo's classes (all given, cls_thre irrelevant). cam [C, h, w] -> labels 0..C."""
    c = cam.reshape(cam.shape[0], -1).astype(np.float64)
    c = c - c.min(1, keepdims=True); c = c / (c.max(1, keepdims=True) + 1e-6)
    if bkg > 0:
        c[c < bkg] = 0; v = c.max(0); lab = c.argmax(0) + 1; lab[v == 0] = 0
    else:
        lab = np.concatenate([(1 - c.max(0, keepdims=True)) ** 2, c], 0).argmax(0)
    return lab.reshape(cam.shape[1:])


# ------------------------------------------------------------------ runner
class Runner:
    def __init__(self):
        import torch
        R, O, U = P47.setup(); self.torch, self.R, self.O, self.U = torch, R, O, U
        self.pipe, self.rec = R.load(); rec = self.rec; rec.full = {}; rec.want_full = False
        procs = dict(self.pipe.unet.attn_processors)
        for n in list(procs):
            nm = n.removesuffix(".processor")
            if nm in FULL:
                procs[n] = Full(procs[n], nm, rec)
        self.pipe.unet.set_attn_processor(procs)
        assert ERS_SA in rec.pair, "iSeg's self-attention layer is not recorded"
        P42 = torch.load(O.WORK / "fit.pt", weights_only=False)["fit"]["20"]
        P44 = torch.load(U.WORK / "fit44.pt", weights_only=False)["fit"]
        self.P = O.M.to_dev({"mo": {**P42["mo"], **P44["mo"]}, "neg": {**P42["neg"], **P44["neg"]}})
        self.w_u = torch.full((len(R.CA_IDX),), 1 / len(R.CA_IDX), device=R.DEV); self.Wm = R.word_map()

    def gate64(self, key):
        """MK4-LR consensus at 64 x 64 from this pass's queries (as jc_ours44.capture), or None if no profile."""
        O, R, rec = self.O, self.R, self.rec
        Q = R.qbar(rec); res = {m: math.isqrt(q.shape[0]) for m, q in Q.items()}
        base = {m: g for m, q in Q.items() if (g := O.R37.g_mk4(self.P, key, m, q)) is not None}
        if not base:
            return None, Q, base
        L = O.G36.consensus(base, {m: (1.0 if res[m] < 64 else 0.0) for m in base}); m64 = next(m for m in base if res[m] == 64)
        return O.R37.powered(L, Q, 1.0)[m64].reshape(64, 64), Q, {"L": L, "M": base}

    def ours(self, s, x, img):
        """P_ours for one item: O- and OL seeds with V0 and ERS."""
        from src.segmentation.sg25_sa import _attention, _apply
        torch, R, O, rec = self.torch, self.R, self.O, self.rec
        key, word = self.U.keyword(s, x["cls"], self.Wm)
        rec.want_full = False; R.one_pass(self.pipe, rec, img, f"A photograph of a {word}", word)
        _, Q, B = self.gate64(key); H64 = _attention(rec.grid_qk[64], 64); out = {}
        for nm, gk in (("O-", None), ("OL", "L")):
            gd = None if gk is None or not B else O.R37.powered(B[gk], Q, 1.0)
            sd = R.make_seed(rec, self.w_u, gd); u = _apply(H64, sd.float().reshape(-1))
            out[f"{nm}|seed"] = sd; out[f"{nm}|V0"] = _apply(H64, u).mean(0).reshape(64, 64)
            out[f"{nm}|ERS"] = ers(sd.reshape(-1, 1).float(), rec.pair_qk[ERS_SA], s)[:, 0].reshape(64, 64)
        rec.clear(); return out, key, word, (0 if B else 1)

    def iseg(self, s, name, X, img):
        """P_iseg for one photo: I and IL seeds for each of its classes, with V0 and ERS."""
        import torch.nn.functional as F
        from src.segmentation.attention_aggregation import find_token_subsequence  # noqa: F401
        from src.segmentation.sg25_sa import _attention, _apply
        from jc_names44 import method_index
        torch, R, rec, pipe = self.torch, self.R, self.rec, self.pipe
        AT = tokens(s); names = list(AT)
        ids = sorted(method_index("iseg", s, names, x["cls"]) for x in X)
        lab = [names[i] for i in ids]; text = "a photograph of "; start, sel, last = [], [], None
        for i, c in enumerate(lab):
            text += c + " and "
            start.append(start[i - 1] + 1 + AT[last][0] if i > 0 else 4); sel.append([start[-1] + j for j in AT[c][1]]); last = c
        text = text[:-5] + " and other object and background" + CFG[s]["bg"]
        tok = pipe.tokenizer(text, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = pipe.text_encoder(tok.input_ids.to(R.DEV), output_hidden_states=True)[0].float()
        emb[:, reduce(add, sel)] *= ENHANCE
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False).to(R.DEV)
        z0 = pipe.vae.encode(2 * x - 1).latent_dist.mean * 0.18215
        torch.cuda.manual_seed(ISEG_SEED); torch.manual_seed(ISEG_SEED)
        eps = torch.randn(z0.shape, device=R.DEV, dtype=torch.float16).float()
        ac = pipe.scheduler.alphas_cumprod.to(R.DEV); zt = ac[T_NOISE].sqrt() * z0 + (1 - ac[T_NOISE]).sqrt() * eps
        rec.target = (0,); rec.clear(); rec.full = {}; rec.want_full = True
        pipe.unet(zt, torch.tensor(T_NOISE, device=R.DEV), encoder_hidden_states=emb, return_dict=False)
        rec.want_full = False
        I = iseg_seed(rec.full, s, sel)                                                   # [4096, C]
        H64 = _attention(rec.grid_qk[64], 64); out = {}; miss = 0
        for j, xi in enumerate(sorted(X, key=lambda t: method_index("iseg", s, names, t["cls"]))):
            key, _ = self.U.keyword(s, xi["cls"], self.Wm); g, _, _ = self.gate64(key)
            if g is None:
                miss += 1; g = torch.ones(64, 64, device=R.DEV)
            mp = {}
            for nm, col in (("I", I[:, j]), ("IL", I[:, j] * g.reshape(-1))):
                sd = col.reshape(64, 64); mm = (sd - sd.min()) / (sd.max() - sd.min()).clamp_min(1e-12)
                u = _apply(H64, mm.reshape(-1)); mp[f"{nm}|seed"] = sd
                mp[f"{nm}|V0"] = _apply(H64, u).mean(0).reshape(64, 64)
                mp[f"{nm}|ERS"] = ers(col.reshape(-1, 1), rec.pair_qk[ERS_SA], s)[:, 0].reshape(64, 64)
            out[xi["id"]] = mp
        rec.clear(); rec.full = {}
        return out, text, ids, miss


class Full:
    """Wraps jc_realvoc.Proc: additionally keeps the full cross-attention probabilities [heads, N, 77] of iSeg's modules."""

    def __init__(self, inner, name, rec):
        self.inner, self.name, self.rec = inner, name, rec

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        if self.rec.want_full and encoder_hidden_states is not None:
            q, k = attn.to_q(hidden_states), attn.to_k(encoder_hidden_states)
            b = q.shape[0]; hd = attn.heads; d = q.shape[-1] // hd
            qh = q.view(b, -1, hd, d).transpose(1, 2)[0].float(); kh = k.view(b, -1, hd, d).transpose(1, 2)[0].float()
            self.rec.full[self.name] = (qh @ kh.transpose(-1, -2) * attn.scale).softmax(-1)
        return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)


def stack(mp):
    return np.stack([mp[c].float().cpu().numpy() for c in CELLS if c in mp]).astype(np.float16), np.array([c for c in CELLS if c in mp])


def capture(a):
    import torch
    from PIL import Image
    from jc_real44 import WORK as W44
    rn = Runner(); P = photos(a.set)[a.shard::a.num_shards]
    if a.limit: P = P[:a.limit]
    out = W / f"swap_{a.pass_}_{a.set}"; out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for img_id, X in P:
            img = np.asarray(Image.open(photo_path(a.set, X[0])).convert("RGB")); t0 = time.perf_counter()
            if a.pass_ == "ours":
                for x in X:
                    o = out / f"{x['id']}.npz"
                    if o.exists():
                        continue
                    mp, key, word, miss = rn.ours(a.set, x, img); arr, names = stack(mp)
                    if not np.isfinite(arr.astype(np.float32)).all():
                        raise ValueError(f"non-finite map {x['id']}")
                    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, maps=arr, names=names, key=key, word=word, miss=miss); os.replace(tmp, o)
            else:
                o = out / f"{iname(a.set, img_id)}.npz"
                if o.exists():
                    continue
                mp, text, ids, miss = rn.iseg(a.set, iname(a.set, img_id), X, img)
                ref = np.load(W44 / "iseg" / f"{a.set}_given" / f"{iname(a.set, img_id)}.npz")
                assert str(ref["text"]) == text and list(ref["cls_ids"]) == ids, (img_id, str(ref["text"]), text, list(ref["cls_ids"]), ids)
                d = {}
                for xid, m in mp.items():
                    arr, names = stack(m); d[f"m_{xid}"] = arr
                tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, names=names, text=text, ids=np.array(ids), miss=miss, **d); os.replace(tmp, o)
            torch.cuda.empty_cache()
            print(f"[swap47:{a.pass_}:{a.set}] {img_id} {len(X)} items {time.perf_counter() - t0:.1f}s", flush=True)


def check(a):
    """(1) P_ours O-/OL V0 == jc_ours44.capture on SD 2.1 (same code path, 3 items); (2) P_iseg I|ERS + comp label agrees
    with run 44's released iSeg (class given) label map on 3 photos per column (pixel agreement, reported)."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from jc_real44 import WORK as W44
    import jc_ov42 as O
    rn = Runner()
    with torch.no_grad():
        X = [x for x in json.loads((W44 / "items.json").read_text())["voc"] if x["primary"]][:3]
        P47.capture(argparse.Namespace(set="voc", shard=0, num_shards=1, limit=3))
        for x in X:
            mp, *_ = rn.ours("voc", x, np.asarray(Image.open(photo_path("voc", x)).convert("RGB")))
            z = np.load(rn.U.WORK / "cap_voc" / f"{x['id']}.npz"); nm = [str(n) for n in z["names"]]
            for c, r in (("O-|V0", "u-|V0"), ("OL|V0", "uL|V0")):
                dd = float(np.abs(mp[c].float().cpu().numpy().astype(np.float16).astype(np.float32) - z["maps"][nm.index(r)].astype(np.float32)).max())
                print(f"check ours {x['id']} {c} vs capture {r}: {dd:.2e}", flush=True); assert dd < 2e-3
        for s in ("voc", "context", "coco"):
            for img_id, X in photos(s)[:3]:
                img = np.asarray(Image.open(photo_path(s, X[0])).convert("RGB")); H, Wd = img.shape[:2]
                mp, text, ids, _ = rn.iseg(s, iname(s, img_id), X, img)
                ref = np.load(W44 / "iseg" / f"{s}_given" / f"{iname(s, img_id)}.npz")
                assert str(ref["text"]) == text and list(ref["cls_ids"]) == ids, (str(ref["text"]), text)
                order = sorted(X, key=lambda t: list(mp).index(t["id"]))
                cam = np.stack([O.st.up(O.st.mm(mp[x["id"]]["I|ERS"].float().cpu().numpy()), (H, Wd)) for x in order])
                lab = cam_to_label(cam, CFG[s]["bkg"]); rl = ref["label"].astype(np.int64)
                ours_lab = np.zeros_like(lab); rel = np.zeros_like(rl)
                for j, x in enumerate(order):                                        # compare by class (both label spaces)
                    ours_lab[lab == j + 1] = j + 1; rel[rl == ids[j] + 1] = j + 1
                print(f"check iseg {s} {img_id}: pixel agreement with run 44 released iSeg {float((ours_lab == rel).mean()):.3f}", flush=True)
    print("check done", flush=True)


def score_one(job):
    s, img_id, X, o = job; o = Path(o)
    if o.exists():
        return
    import jc_ov42 as O
    import jc_score44 as S44
    zi = np.load(W / f"swap_iseg_{s}" / f"{iname(s, img_id)}.npz"); names_i = [str(n) for n in zi["names"]]
    T = {x["id"]: S44.truth_of(s, x) for x in X}; hw = next(iter(T.values()))[0].shape
    maps = {}
    for x in X:
        zo = np.load(W / f"swap_ours_{s}" / f"{x['id']}.npz"); no = [str(n) for n in zo["names"]]
        for c in CELLS:
            if c in no:
                maps[(x["id"], c)] = O.st.up(O.st.mm(zo["maps"][no.index(c)].astype(np.float32)), hw)
            elif c in names_i:
                maps[(x["id"], c)] = O.st.up(O.st.mm(zi[f"m_{x['id']}"][names_i.index(c)].astype(np.float32)), hw)
    res = {}
    for x in X:
        truth, valid, other = T[x["id"]]; r = {}
        for c in CELLS:
            if not c.endswith("|seed"):
                r[f"{c}|bin"] = O.cc.counts(maps[(x["id"], c)][None] >= O.THR[:, None, None], truth, valid, other)
        res[x["id"]] = r
    for c in CELLS:
        if c.endswith("|seed"):
            continue
        lab = cam_to_label(np.stack([maps[(x["id"], c)] for x in X]), CFG[s]["bkg"])
        for j, x in enumerate(X):
            truth, valid, other = T[x["id"]]
            res[x["id"]][f"{c}|comp"] = O.cc.counts((lab == j + 1)[None], truth, valid, other)[0]
    flat = {f"{xid}::{k}": v for xid, r in res.items() for k, v in r.items()}
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, ids=np.array([x["id"] for x in X]), cls=np.array([x["cls"] for x in X]), **flat)
    os.replace(tmp, o)


def score(a):
    from multiprocessing import get_context
    out = W / f"swap_scores_{a.set}"; out.mkdir(parents=True, exist_ok=True)
    P = photos(a.set)
    if a.limit: P = P[:a.limit]
    jobs = [(a.set, i, X, str(out / f"{iname(a.set, i)}.npz")) for i, X in P]
    with get_context("fork").Pool(a.workers) as p:
        for _ in p.imap_unordered(score_one, jobs, chunksize=4):
            pass
    print(f"scored {len(jobs)} photos of {a.set}", flush=True)


def report(a):
    import jc_ov42 as O
    from jc_real44 import WORK as W44, VOC20
    T55 = O.T55; L = []; w = L.append; E = {}
    KEEP = {"ours:uL|V0|raw": "sd15:OL|V0|bin", "ours:u-|V0|raw": "sd15:O-|V0|bin", "nerve|own": "NERVE", "iseg|given": "iSeg44:given",
            "iseg|own": "iSeg44:own"}
    for s, n in COLS:
        P = photos(s); rows = {}
        for i, X in P:
            f = W / f"swap_scores_{s}" / f"{iname(s, i)}.npz"
            if not f.exists():
                if a.tag:
                    continue
                raise SystemExit(f"{s}: missing scores for {i}")
            z = np.load(f)
            for x in X:
                rows[x["id"]] = {k.split("::", 1)[1]: z[k] for k in z.files if k.startswith(x["id"] + "::")} | {"cls": x["cls"], "img": str(i)}
        ids = sorted(rows); S = {"cls": np.array([rows[k]["cls"] for k in ids]), "grp": np.array([rows[k]["img"] for k in ids])}
        for k in rows[ids[0]]:
            if "|" in k:
                v = np.stack([rows[i][k] for i in ids]); S[k] = v[:, T55] if v.ndim == 3 else v
        Z44 = [np.load(W44 / f"scores_{s}_pre_ovdiff" / f"{i}.npz") for i in ids]
        for k, nk in KEEP.items():
            v = np.stack([z[k] for z in Z44]); S[nk] = v[:, T55] if v.ndim == 3 else v
        E[s] = (S, O.Boot(S, a.boot))
    stat = lambda s, k: E[s][1].stat(E[s][0][k])

    def dif(s, x, y):
        A, B = stat(s, x), stat(s, y); d = A[1] - B[1]
        return A[0] - B[0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    f3 = lambda d: f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
    rd = lambda d: "**helps**" if d[1] > 0 else "**hurts**" if d[2] < 0 else "not resolved"
    w("# Run 47: SD 2.1-base component swap between our method and iSeg (`runs/47_sd21_swap/PREREG.md`)\n")
    w("Run 44's items and protocol (class word given; one mask per primary item). Every cell on SD 2.1-base; seeds from "
      "their own pass (O = ours, I = iSeg's category-enhanced cross-attention; L = our MK4-LR gate); refinement V0 (ours) "
      "or ERS (iSeg); read-out bin (ours, 0.55) or comp (iSeg's cam_to_label over the photo's classes). Class-macro mIoU; "
      f"95% paired photo bootstrap ({a.boot}).\n")
    w("\n## 1. Primary contrasts\n")
    w("| # | Contrast | " + " | ".join(n for _, n in COLS) + " |\n|---|---|" + "---|" * len(COLS))
    PC = [("Q1", "iSeg refinement on our seed: OL ERS - OL V0 (bin)", "OL|ERS|bin", "OL|V0|bin"),
          ("Q2", "our gate on iSeg: IL ERS - I ERS (comp)", "IL|ERS|comp", "I|ERS|comp"),
          ("Q3", "seed, iSeg refinement + read-out: OL ERS - I ERS (comp)", "OL|ERS|comp", "I|ERS|comp"),
          ("Q4", "competition for ours: OL V0 comp - OL V0 bin", "OL|V0|comp", "OL|V0|bin"),
          ("Q5", "backbone: ours SD 2.1 - ours SD 1.5 (OL V0 bin)", "OL|V0|bin", "sd15:OL|V0|bin")]
    for q, nm, x, y in PC:
        w(f"| {q} | {nm} | " + " | ".join(f"{f3(dif(s, x, y))} {rd(dif(s, x, y))}" for s, _ in COLS) + " |")
    w("\n## 2. Every cell (mIoU)\n")
    w("| Seed | Refinement | Read-out | " + " | ".join(n for _, n in COLS) + " |\n|---|---|---|" + "---:|" * len(COLS))
    for sd in SEEDS:
        for r in ("V0", "ERS"):
            for ro in ("bin", "comp"):
                k = f"{sd}|{r}|{ro}"; w(f"| {sd} | {r} | {ro} | " + " | ".join(f"{stat(s, k)[0]:.3f}" for s, _ in COLS) + " |")
    w("\nReferences (run 44, same items): " + "; ".join(
        f"{lab} " + "/".join(f"{stat(s, k)[0]:.3f}" for s, _ in COLS) for lab, k in
        (("ours SD 1.5 (MK4-LR V0 bin)", "sd15:OL|V0|bin"), ("NERVE", "NERVE"), ("iSeg released, class given", "iSeg44:given"),
         ("iSeg released, own labels", "iSeg44:own"))) + ".\n")
    w("\nReproduction: I ERS comp (this transcription, fp32) vs iSeg released, class given (run 44, fp16): " +
      " | ".join(f"{n} {f3(dif(s, 'I|ERS|comp', 'iSeg44:given'))}" for s, n in COLS) + ".\n")
    w("\n## 3. Secondary contrasts\n")
    w("| Contrast | " + " | ".join(n for _, n in COLS) + " |\n|---|" + "---|" * len(COLS))
    for nm, x, y in [("our gate, ours: OL - O- (V0 bin)", "OL|V0|bin", "O-|V0|bin"), ("our gate, ours: OL - O- (ERS comp)", "OL|ERS|comp", "O-|ERS|comp"),
                     ("seed: I - O- (V0 bin)", "I|V0|bin", "O-|V0|bin"), ("seed: IL - OL (V0 bin)", "IL|V0|bin", "OL|V0|bin"),
                     ("refinement, iSeg seed: I ERS - I V0 (comp)", "I|ERS|comp", "I|V0|comp"),
                     ("competition, iSeg: I ERS comp - I ERS bin", "I|ERS|comp", "I|ERS|bin"),
                     ("best ours-gated cell - iSeg released given", "IL|ERS|comp", "iSeg44:given"),
                     ("OL ERS comp - NERVE", "OL|ERS|comp", "NERVE"), ("IL ERS comp - NERVE", "IL|ERS|comp", "NERVE")]:
        w(f"| {nm} | " + " | ".join(f3(dif(s, x, y)) for s, _ in COLS) + " |")
    w("\n## 4. Subsets (Context / COCO-Object: 20 VOC classes vs the others)\n")
    sub = lambda S, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}
    for s in ("context", "coco"):
        S = E[s][0]
        for lab, fn in (("20 VOC classes", lambda c: c in VOC20), ("other classes", lambda c: c not in VOC20)):
            sel = np.array([fn(c) for c in S["cls"]]); S2 = sub(S, sel); B2 = O.Boot(S2, a.boot)
            w(f"- **{s}, {lab}** ({int(sel.sum())} items): " + ", ".join(
                f"{k} {B2.stat(S2[k])[0]:.3f}" for k in ("OL|V0|bin", "OL|ERS|comp", "I|ERS|comp", "IL|ERS|comp", "sd15:OL|V0|bin", "NERVE")))
    w("\n## 5. Per class (OL V0 bin | OL ERS comp | I ERS comp | IL ERS comp | NERVE)\n")
    for s, n in COLS:
        S, B = E[s]; cols = ("OL|V0|bin", "OL|ERS|comp", "I|ERS|comp", "IL|ERS|comp", "NERVE"); M = {k: B.miou(S[k])[0] for k in cols}
        w(f"\n{n}:\n\n| Class | items | " + " | ".join(cols) + " |\n|---|---:|" + "---:|" * len(cols))
        for i, c in enumerate(B.cl):
            w(f"| {c} | {int((S['cls'] == c).sum())} | " + " | ".join(f"{M[k][i]:.3f}" for k in cols) + " |")
    w("\nThe paper 2 method stays frozen; this is a component analysis, nothing is promoted.\n")
    out = REPORT if not a.tag else REPORT.with_name(f"result_{a.tag}.md"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L)); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("capture"); p.add_argument("--pass", dest="pass_", required=True, choices=["ours", "iseg"])
    p.add_argument("--set", required=True, choices=[c for c, _ in COLS]); p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("score"); p.add_argument("--set", required=True, choices=[c for c, _ in COLS])
    p.add_argument("--workers", type=int, default=16); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("report"); p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args(); {"check": check, "capture": capture, "score": score, "report": report}[a.cmd](a)
