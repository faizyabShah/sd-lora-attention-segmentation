#!/usr/bin/env python3
"""runs/26_shadow_attn/PREREG.md: supervised *shadow* attention copies for BRACE (exploratory, supervised).

The generator's attention is untouched. Queries/keys it computes are recorded (detached) and a shadow branch
recomputes attention from them with low-rank adapters  q' = q + (qA)B,  k' = k + (kA)B  (rank 8, B = 0 at init).
  A0  frozen BRACE                      B  CA copies (11 A3 modules)          C  CA copies + SA copies (16/32/64)
Shadow seed = A3 construction (layer weights from the ORIGINAL SA); shadow BRACE = differentiable V3 path.

split    runs/26_shadow_attn/split.json (COCO-Cap: 160 train / 40 val / 507 test; VOC-Sim all = transfer)
train    (GPU) one arm, checkpoints in <BRACE_CACHE_DIR>/jc_shadow/ckpt{tag}/
val      (GPU) checkpoint selection on the 40 validation images -> runs/26_shadow_attn/val{tag}.json
capture  (GPU, sharded) test maps [3 arms, 2 maps (seed, V3), 64, 64]
score    (CPU) raw and + DenseCRF R2 counts at 0.30-0.70 (step 0.025)
report   (CPU) runs/26_shadow_attn/result{tag}.md
"""
from __future__ import annotations

import argparse, gc, json, math, os, random, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402
import jc_newrerun as nr  # noqa: E402
from brace_paths import CACHE  # noqa: E402
from src.segmentation import sg25_sa  # noqa: E402
from src.segmentation.sg25_sa import _apply, _resize  # noqa: E402
from src.segmentation.sg_combined_v2 import cap  # noqa: E402

WORK = CACHE / "jc_shadow"
OUT = ROOT / "runs/26_shadow_attn"
ARMS = ("A0", "B", "C")
MAPS = ("seed", "V3")
THR = nr.THR
T55 = int(np.flatnonzero(np.isclose(THR, 0.55))[0])
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap test"}
RANK, LR, WD, CLIP, SEED_W = 8, 3e-4, 0.0, 1.0, 0.5


# ---------------------------------------------------------------- recorder and generation
class ShadowRecorder(sg25_sa.SG25Recorder):
    """Frozen SG25/A3 recorder that also keeps the conditional CA queries/keys of the 11 selected modules."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ca_store = []

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind != "self" and module_name in self.ca_modules:
            step = self.call_counts[module_name]
            k = attribution_key[0] if attribution_key is not None else key[1]
            m = None if (attribution_key is not None or attention_mask is None) else attention_mask[1].detach().clone()
            self.ca_store.append((step, module_name, query[1].detach().half().clone(), k.detach().half().clone(), m, float(scale)))
        super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key)

    def clear(self):
        super().clear(); self.ca_store.clear()


def make_runtime(model):
    from scripts import sg27_runtime
    rt = sg27_runtime.make_runtime(model, recorder_class=ShadowRecorder)
    rt["pipe"].safety_checker = None
    return rt


def generate(rt, row, ds):
    """sg27_runtime.live_maps without the unused SG27 fusion maps; returns (recorder, sample). Caller cleans up."""
    from scripts import run_coco_cap_transfer as transfer
    from src.attention_capture import live
    from src.segmentation import sg24_ca
    from src.segmentation.attention_aggregation import find_token_subsequence
    pipe = rt["pipe"]; prompt = row["attribution_prompt"]
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
    target = find_token_subsequence(ids, pipe.tokenizer(row["classname"], add_special_tokens=False)["input_ids"])
    rt["recorder_class"].content = sg24_ca.content_positions(pipe.tokenizer, prompt, tuple(target))
    rt["recorder_class"].attribution_prompt = prompt
    rt["recorders"].clear()
    manifest = transfer.EVAL / ("ovam/voc_sim" if ds == "voc" else "ovam/coco_captions")
    sample = live.generate_attention_sample(pipe, dict(row), manifest, "float16", cross_map_kind="conditional")
    return rt["recorders"][-1], sample


def cleanup(rt, rec, sample):
    rec.clear(); sample.clear(); rt["recorders"].clear()
    gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------- shadow branch
class Low(nn.Module):
    """q-space low-rank residual on head-merged features; B = 0 so the copy starts equal to SD."""

    def __init__(self, c, g):
        super().__init__()
        self.A = nn.Parameter((torch.rand(c, RANK, generator=g) * 2 - 1) / math.sqrt(c))
        self.B = nn.Parameter(torch.zeros(RANK, c))

    def forward(self, x):                       # x [heads, N, d]
        h, n, d = x.shape
        m = x.permute(1, 0, 2).reshape(n, h * d)
        m = m + (m @ self.A) @ self.B
        return m.reshape(n, h, d).permute(1, 0, 2)


def _k(name):
    return name.replace(".", "_")


class Shadow(nn.Module):
    def __init__(self, rt, arm):
        super().__init__()
        g = torch.Generator().manual_seed(0); pipe = rt["pipe"]
        self.arm = arm
        self.ca_q, self.ca_k, self.sa_q, self.sa_k = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for m in rt["ca_modules"]:                                   # CA first: B and C share the CA init
            c = pipe.unet.get_submodule(m).to_q.out_features
            self.ca_q[_k(m)] = Low(c, g); self.ca_k[_k(m)] = Low(c, g)
        if arm == "C":
            for r in (16, 32, 64):
                c = pipe.unet.get_submodule(rt["modules"][r]).to_q.out_features
                self.sa_q[str(r)] = Low(c, g); self.sa_k[str(r)] = Low(c, g)


def mm_t(x):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-20)


def shadow_seed(rec, sh):
    """A3 seed from recorded CA q/k; layer weights are the frozen SA-guided weights of each call. min-max."""
    wmap = {s: w for s, lab, w in rec.weights if lab == "selected"}
    if len(wmap) != 31:
        raise ValueError("shadow: expected 31 selected weight vectors")
    mods = list(rec.selected_modules); tgt = list(rec.target); dev = rec.pipe.device
    acc = torch.zeros((64, 64), device=dev)
    for step, name, q, k, mask, scale in rec.ca_store:
        q = q.float().clone(); k = k.float().clone()
        if sh is not None:
            q = sh.ca_q[_k(name)](q); k = sh.ca_k[_k(name)](k)
        s = q @ k.transpose(-1, -2) * scale
        if mask is not None:
            s = s + mask.float()
        t = s.softmax(-1)[..., tgt].mean(-1).mean(0)
        side = math.isqrt(t.shape[0])
        w = float(wmap[step][mods.index(name)])
        acc = acc + F.interpolate(t.reshape(1, 1, side, side), size=(64, 64), mode="bilinear", align_corners=False)[0, 0] * w / 31
    return mm_t(acc)


def _att(q, k, aq, ak):
    if aq is not None:
        q = aq(q); k = ak(k)
    return torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)


def _call_v3(c16, q16, k16, q32, k32, q64, k64, sh):
    ad = (lambda r: (sh.sa_q[str(r)], sh.sa_k[str(r)])) if sh is not None else (lambda r: (None, None))
    raw16 = _apply(_att(q16, k16, *ad(16)), c16).mean(0)
    raw32 = _apply(_att(q32, k32, *ad(32)), _resize(raw16, 16, 32).reshape(-1)).mean(0)
    z32, _ = cap(raw16, raw32, 16)
    return _apply(_att(q64, k64, *ad(64)), _resize(z32, 32, 64).reshape(-1)).mean(0)


def shadow_v3(rec, rt, seed, sh, grad):
    """V3 path of sg_combined_v2.aggregate (16 SA -> 32 SA -> cap 16->32 -> 64 SA, mean of 11 calls). Raw map."""
    md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
    if any(len(sa[r]) != 11 for r in sa):
        raise ValueError("shadow: expected 11 SA calls per resolution")
    c16 = _resize(seed.reshape(-1), 64, 16).reshape(-1); acc = 0
    for i in range(11):
        args = [x.float().clone() for r in (16, 32, 64) for x in sa[r][i]]
        v = checkpoint(_call_v3, c16, *args, sh, use_reentrant=False) if grad else _call_v3(c16, *args, sh)
        acc = acc + v / 11
    return acc.reshape(64, 64)


def seg_loss(p, t):
    p = p.clamp(1e-4, 1 - 1e-4)
    return F.binary_cross_entropy(p, t) + 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def truth64(truth):
    return F.interpolate(torch.from_numpy(truth.astype(np.float32))[None, None], size=(64, 64), mode="area")[0, 0]


def maps_for(rec, rt, sh):
    """(seed, raw V3) for an arm; sh None = frozen copies (A0)."""
    s = shadow_seed(rec, sh)
    return s, shadow_v3(rec, rt, s, sh if (sh is not None and sh.arm == "C") else None, grad=False)


# ---------------------------------------------------------------- split
def quota(counts, total):
    n = sum(counts.values()); raw = {c: total * v / n for c, v in counts.items()}
    q = {c: int(math.floor(x)) for c, x in raw.items()}
    for c in sorted(raw, key=lambda c: (-(raw[c] - q[c]), c))[: total - sum(q.values())]:
        q[c] += 1
    return q


def split(a):
    R = nr.rows("coco"); by = {}
    for r in R:
        by.setdefault(r["label"], []).append(r["image_id"])
    qp = quota({c: len(v) for c, v in by.items()}, 200); pool = {}
    for c, v in by.items():
        pool[c] = sorted(v, key=lambda i: zlib.crc32(i.encode()))[: qp[c]]
    qv = quota({c: len(v) for c, v in pool.items() if v}, 40); tr, va = [], []
    for c, v in pool.items():
        s = sorted(v, key=lambda i: zlib.crc32((i + "val").encode())); k = qv.get(c, 0)
        va += s[:k]; tr += s[k:]
    used = set(tr) | set(va); te = [r["image_id"] for r in R if r["image_id"] not in used]
    o = {"train": sorted(tr), "val": sorted(va), "coco_test": sorted(te), "n_coco": len(R),
         "classes": {c: len(v) for c, v in sorted(by.items())}}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "split.json").write_text(json.dumps(o, indent=1))
    print(len(tr), len(va), len(te), flush=True)


def load_split():
    return json.loads((OUT / "split.json").read_text())


def rows_for(ds, ids):
    keep = set(ids); return [r for r in nr.rows(ds) if r["image_id"] in keep]


# ---------------------------------------------------------------- train / val / capture
def train(a):
    sp = load_split(); ids = sp["train"][: a.n_train] if a.n_train else sp["train"]
    R = rows_for("coco", ids); rt = make_runtime(a.model); dev = rt["pipe"].device
    torch.manual_seed(0); sh = Shadow(rt, a.arm).to(dev)
    opt = torch.optim.AdamW(sh.parameters(), lr=LR, weight_decay=WD)
    ck = WORK / f"ckpt{a.tag}"; ck.mkdir(parents=True, exist_ok=True)
    ckpts = {int(x) for x in a.ckpts.split(",")}; order, rng, log = [], random.Random(0), []
    print(f"[shadow:{a.arm}] {len(R)} training images, {sum(p.numel() for p in sh.parameters())} parameters", flush=True)
    for step in range(1, a.steps + 1):
        if not order:
            order = list(range(len(R))); rng.shuffle(order)
        row = R[order.pop()]; t0 = time.perf_counter()
        rec, sample = generate(rt, row, "coco")
        t = truth64(np.asarray(sample.target_mask, bool)).to(dev)
        s = shadow_seed(rec, sh)
        v = mm_t(shadow_v3(rec, rt, s, sh if a.arm == "C" else None, grad=True))
        lv, ls = seg_loss(v, t), seg_loss(s, t); loss = lv + SEED_W * ls
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(sh.parameters(), CLIP)); opt.step()
        log.append({"step": step, "id": row["image_id"], "loss": float(loss), "l_v3": float(lv), "l_seed": float(ls),
                    "grad_norm": gn, "sec": time.perf_counter() - t0})
        print(f"[shadow:{a.arm}] step {step} loss {float(loss):.4f} (V3 {float(lv):.4f} seed {float(ls):.4f}) "
              f"|g| {gn:.3f} {log[-1]['sec']:.1f}s", flush=True)
        del s, v, loss, lv, ls; cleanup(rt, rec, sample)
        if step in ckpts:
            torch.save(sh.state_dict(), ck / f"{a.arm}_{step}.pt")
            (OUT / f"train_{a.arm}{a.tag}.json").write_text(json.dumps(log, indent=0))


def load_shadow(rt, arm, tag, step):
    sh = Shadow(rt, arm).to(rt["pipe"].device)
    sh.load_state_dict(torch.load(WORK / f"ckpt{tag}" / f"{arm}_{step}.pt", map_location=rt["pipe"].device)); sh.eval()
    return sh


def _counts(m, truth):
    x = st.up(st.mm(m.float().cpu().numpy()), truth.shape) >= THR[T55]
    return cc.counts(x[None], truth, np.ones_like(truth), np.zeros_like(truth))


@torch.no_grad()
def val(a):
    sp = load_split(); ids = sp["val"][: a.n_val] if a.n_val else sp["val"]
    R = rows_for("coco", ids); rt = make_runtime(a.model); ckpts = [int(x) for x in a.ckpts.split(",")]
    sh = {(arm, s): load_shadow(rt, arm, a.tag, s) for arm in ("B", "C") for s in ckpts}
    C = {k: [] for k in [("A0", 0)] + list(sh)}; cls = []
    for row in R:
        rec, sample = generate(rt, row, "coco"); truth = np.asarray(sample.target_mask, bool); cls.append(row["label"])
        C[("A0", 0)].append(_counts(maps_for(rec, rt, None)[1], truth))
        for k, m in sh.items():
            C[k].append(_counts(maps_for(rec, rt, m)[1], truth))
        print(f"[shadow:val] {row['image_id']}", flush=True); cleanup(rt, rec, sample)
    cls = np.array(cls); one = np.ones(len(cls), bool)
    res = {f"{k[0]}@{k[1]}": float(cc.miou(np.stack(v), cls, one)[0]) for k, v in C.items()}
    sel = {}
    for arm in ("B", "C"):
        best = max(ckpts, key=lambda s: (res[f"{arm}@{s}"], -s)); sel[arm] = best
    o = {"n": len(R), "metric": "class-macro mIoU, raw shadow BRACE map, threshold 0.55", "miou": res, "selected": sel}
    (OUT / f"val{a.tag}.json").write_text(json.dumps(o, indent=1)); print(json.dumps(o, indent=1), flush=True)


@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    sp = load_split()
    ids = sp["coco_test"] if a.dataset == "coco" else [r["image_id"] for r in nr.rows("voc")]
    R = rows_for(a.dataset, ids)[a.shard::a.num_shards]
    R = R[: a.max_images] if a.max_images else R
    rt = make_runtime(a.model); sel = json.loads((OUT / f"val{a.tag}.json").read_text())["selected"]
    sh = {arm: load_shadow(rt, arm, a.tag, sel[arm]) for arm in ("B", "C")}
    out = WORK / f"{a.dataset}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in R:
        o = out / f"{row['image_id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = generate(rt, row, a.dataset)
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); v3 = sg_combined_v2.aggregate(sa, seed)[0]["V3"].float()
        s0, v0 = maps_for(rec, rt, None)
        fid_s = float((s0 - seed).abs().max()); fid_v = float((mm_t(v0) - mm_t(v3)).abs().max())
        maps = [[seed, v3]] + [list(maps_for(rec, rt, sh[arm])) for arm in ("B", "C")]
        maps = np.stack([np.stack([x.float().cpu().numpy() for x in p]) for p in maps]).astype(np.float32)
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {row['image_id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=row["label"], truth=np.asarray(sample.target_mask, bool),
                 image=np.asarray(sample.generated_image, np.uint8), fid_seed=fid_s, fid_v3=fid_v)
        os.replace(tmp, o)
        print(f"[shadow:{a.dataset}] {row['image_id']} fid seed {fid_s:.1e} V3 {fid_v:.1e} {time.perf_counter() - t0:.1f}s", flush=True)
        cleanup(rt, rec, sample)


# ---------------------------------------------------------------- score / report
def score_one(job):
    f, o = Path(job[0]), Path(job[1])
    if o.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f); img, truth = z["image"], z["truth"].astype(bool)
    valid = np.ones_like(truth); other = np.zeros_like(truth)
    res = {"area": float(truth.mean()), "cls": str(z["cls"]), "fid_seed": float(z["fid_seed"]), "fid_v3": float(z["fid_v3"])}
    for ai, arm in enumerate(ARMS):
        for mi, m in enumerate(MAPS):
            x = st.up(st.mm(z["maps"][ai, mi].astype(np.float32)), truth.shape)
            res[f"{arm}|{m}|raw"] = cc.counts(x[None] >= THR[:, None, None], truth, valid, other)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in THR]).astype(bool)
            res[f"{arm}|{m}+R2|raw"] = cc.counts(ms, truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    for ds in a.sets:
        sc = WORK / f"{ds}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / (ds + a.tag)).glob("*.npz"))
                if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
        print(ds, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
                if n % 25 == 0 or n == len(jobs):
                    print(f"{ds} {n}/{len(jobs)}", flush=True)


def report(a):
    S = {ds: cc.load_gen(WORK / f"{ds}{a.tag}_scores") for ds in ("voc", "coco")}
    n = {ds: len(S[ds]["grp"]) for ds in S}; one = {ds: np.ones(n[ds], bool) for ds in S}
    mi = lambda ds, P: cc.miou(P[:, None], S[ds]["cls"], one[ds])[0]
    tb = lambda key: int(np.argmax(cc.miou(S["voc"][key], S["voc"]["cls"], one["voc"])))

    def at(ds, key, mode):
        X = S[ds][key]
        if mode == "0.55":
            return X[:, T55]
        return cc.oof(S["voc"], X)[1] if ds == "voc" else X[:, tb(key)]

    def dci(ds, Pa, Pb):
        lo, hi = cc.boot(S[ds], Pa, Pb, n=a.boot); d = mi(ds, Pa) - mi(ds, Pb)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", d, lo

    key = lambda arm, m, crf: f"{arm}|{m}{'+R2' if crf else ''}|raw"
    V = json.loads((OUT / f"val{a.tag}.json").read_text()); sp = load_split()
    lab = {"A0": "A0 frozen BRACE", "B": f"B CA copies (step {V['selected']['B']})",
           "C": f"C CA + SA copies (step {V['selected']['C']})"}
    L = ["# Supervised shadow attention for BRACE: CA-only vs CA + SA (`PREREG.md`)\n",
         "`scripts/jc_shadow.py`. **Exploratory, supervised** (human masks of the COCO-Cap training images). The "
         "generator is untouched: trained low-rank *copies* of the attention only produce the maps BRACE reads. All arms "
         "from one live generation per test image (paired). *VOC-sel.* = 5-fold OOF on VOC-Sim; the full-VOC-Sim best "
         f"threshold applied unchanged on COCO-Cap test. Class-macro mIoU; Δ with paired image-bootstrap 95% interval ({a.boot}).\n",
         "## Data and checks\n",
         f"- Split: {len(sp['train'])} train / {len(sp['val'])} validation / {len(sp['coco_test'])} COCO-Cap test; "
         "VOC-Sim (all) never trained on or used for selection."]
    for ds in S:
        Z = [np.load(f) for f in sorted((WORK / f"{ds}{a.tag}_scores").glob("*.npz")) if not f.stem.endswith(".tmp")]
        fs = max(float(z["fid_seed"]) for z in Z); fv = max(float(z["fid_v3"]) for z in Z)
        L.append(f"- {NAME[ds]}: {n[ds]} images. Zero-adapter shadow vs frozen BRACE: max |Δ| seed {fs:.1e}, BRACE map {fv:.1e}.")
    L.append("- Validation (40 images, raw shadow BRACE map @0.55): " + ", ".join(f"{k} {v:.3f}" for k, v in V["miou"].items())
             + f"; selected B step {V['selected']['B']}, C step {V['selected']['C']}.")
    for arm in ("B", "C"):
        f = OUT / f"train_{arm}{a.tag}.json"
        if f.exists():
            lg = json.loads(f.read_text()); k = max(1, len(lg) // 10)
            L.append(f"- Training {arm}: loss first {k} steps {np.mean([x['loss'] for x in lg[:k]]):.3f} → last {k} "
                     f"{np.mean([x['loss'] for x in lg[-k:]]):.3f}; {np.mean([x['sec'] for x in lg]):.1f} s/step.")
    for crf in (True, False):
        L.append(f"\n## {'With DenseCRF (primary)' if crf else 'Raw (secondary)'}\n")
        L.append("| Arm | Map | VOC-Sim, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap test, VOC-sel. | COCO-Cap test, 0.55 |")
        L.append("|---|---|---:|---:|---:|---:|")
        for arm in ARMS:
            for m, ml in (("seed", "seed"), ("V3", "BRACE hierarchy")):
                L.append(f"| {lab[arm]} | {ml} | " + " | ".join(f"{mi(ds, at(ds, key(arm, m, crf), mode)):.3f}"
                                                           for ds in ("voc", "coco") for mode in ("sel", "0.55")) + " |")
    L.append("\n## Registered comparisons (BRACE hierarchy + DenseCRF)\n")
    L.append("| ID | Comparison | VOC-Sim, VOC-sel. | COCO-Cap test, VOC-sel. | VOC-Sim, 0.55 | COCO-Cap test, 0.55 | Verdict (0.55) |")
    L.append("|---|---|---|---|---|---|---|")
    for tid, lb, x, y in (("S1", "C − B", "C", "B"), ("S2", "B − A0", "B", "A0"), ("S3", "C − A0", "C", "A0")):
        c = {(ds, mode): dci(ds, at(ds, key(x, "V3", True), mode), at(ds, key(y, "V3", True), mode))
             for mode in ("sel", "0.55") for ds in S}
        v = "holds" if (c[("coco", "0.55")][2] > 0 and c[("voc", "0.55")][1] >= -0.005) else "does not hold"
        L.append(f"| {tid} | {lb} | {c[('voc', 'sel')][0]} | {c[('coco', 'sel')][0]} | {c[('voc', '0.55')][0]} | "
                 f"{c[('coco', '0.55')][0]} | {v} |")
    L.append("\nRule: holds if the COCO-Cap test Δ at 0.55 is positive with interval above zero and the VOC-Sim Δ at 0.55 "
             "is not below −0.005.\n")
    L.append("Descriptive, seed + DenseCRF @0.55, C − B / B − A0: " + "; ".join(
        f"{NAME[ds]} {dci(ds, at(ds, key('C', 'seed', True), '0.55'), at(ds, key('B', 'seed', True), '0.55'))[0]} / "
        f"{dci(ds, at(ds, key('B', 'seed', True), '0.55'), at(ds, key('A0', 'seed', True), '0.55'))[0]}" for ds in S) + "\n")
    for ds in S:
        cl = S[ds]["cls"]
        L.append(f"Per class, {NAME[ds]}, BRACE hierarchy + DenseCRF @0.55 (A0 / B / C): " + ", ".join(
            f"{c} " + " / ".join(f"{cc.miou(at(ds, key(arm, 'V3', True), '0.55')[:, None], cl, cl == c)[0]:.2f}" for arm in ARMS)
            for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt)
    (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    M = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    sub.add_parser("split")
    p = sub.add_parser("train"); p.add_argument("--arm", choices=("B", "C"), required=True)
    p.add_argument("--steps", type=int, default=800); p.add_argument("--ckpts", default="200,400,600,800")
    p.add_argument("--n-train", type=int, default=0); p.add_argument("--tag", default=""); p.add_argument("--model", default=M)
    p = sub.add_parser("val"); p.add_argument("--ckpts", default="200,400,600,800"); p.add_argument("--n-val", type=int, default=0)
    p.add_argument("--tag", default=""); p.add_argument("--model", default=M)
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default=""); p.add_argument("--model", default=M)
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["voc", "coco"]); p.add_argument("--tag", default="")
        p.add_argument("--workers", type=int, default=16); p.add_argument("--shard", type=int, default=0)
        p.add_argument("--num-shards", type=int, default=1); p.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    {"split": split, "train": train, "val": val, "capture": capture, "score": score, "report": report}[a.cmd](a)
