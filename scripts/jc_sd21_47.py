#!/usr/bin/env python3
"""Run 47: our method (paper 2, frozen) with SD 2.1-base instead of SD 1.5, on run 44's real-photo columns (env trdp).

Only the backbone changes. Everything else is run 44's code and settings, unchanged: one noised pass per item at t = 100
(the same noise schedule; SD 2.1-base is epsilon-prediction at 512 px, 64 x 64 latent), noise seed 0, prompt
"A photograph of a {word}", the same 11 cross-attention modules (indices into the sorted attn2 names; the UNet block
layout is identical) and the same 16/32/64 self-attention modules; uniform CA seed; MK4 gate -> MK4-LR (gamma 1);
fine-only V0; min-max; fixed 0.55 (raw primary).
Profiles are REBUILT on SD 2.1 from the same photos with the same recipe and caption rules (no labels or masks):
  * the 80 COCO words: run 42's 1,600 size-20 profile photos (items_prof.json, rank < 20; they carry every negative
    assignment), dumped with jc_ov42.dump, fitted at size 20 exactly as jc_ov42.fit;
  * the 33 Context words: run 44's 2,260 profile jobs (items_prof44.json), jc_ours44.dump44 / fit44.
Implementation: jc_realvoc.MODEL, jc_ov42.WORK and jc_ours44.WORK are pointed at SD 2.1 and <CACHE>/jc_sd21_47; the
functions run unchanged (capture keeps run 44's identity checks).

prof      (CPU) <W>/ov42/items_prof.json (run 42's rank < 20 items) and <W>/ours/items_prof44.json (copy)
dump42    (GPU) jc_ov42.dump on SD 2.1                       -> <W>/ov42/dump/
fit42     (CPU) size-20 profiles, as jc_ov42.fit             -> <W>/ov42/fit.pt {"fit": {"20": ...}}
dump44    (GPU) / fit44 (CPU) jc_ours44 on SD 2.1            -> <W>/ours/dump44/, fit44.pt
capture   (GPU) jc_ours44.capture --set voc|context|coco    -> <W>/ours/cap_<set>/
score     (CPU) counts on the 0.30-0.70 grid                 -> <W>/scores_<set>/
report    (CPU) vs run 44 (SD 1.5 ours, NERVE, iSeg: counts read from run 44's complete pre-OVDiff score files)
                                                             -> runs/47_sd21_backbone/result.md
"""
from __future__ import annotations

import argparse, json, os, shutil, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
CACHE = Path(os.environ.get("BRACE_CACHE_DIR", str(ROOT / "outputs/sg_comp"))).expanduser()
W = Path(os.environ.get("SD21_47_WORK", str(CACHE / "jc_sd21_47")))          # SD21_47_WORK: separate directory for the smoke
_SD21_LOCAL = "/home/sidna/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1-base/snapshots/4e63672c03103b6c636b8fb4119ba982469b2955"
# SD21_MODEL: a local snapshot or a hub id; elsewhere the hub id at the pinned revision 4e63672c (HF_HOME cache).
SD21 = os.environ.get("SD21_MODEL", _SD21_LOCAL if os.path.isdir(_SD21_LOCAL) else "sd2-community/stable-diffusion-2-1-base")
REPORT = ROOT / "runs/47_sd21_backbone/result.md"
COLS = [("voc", "PASCAL VOC (20)"), ("context", "PASCAL Context (59)"), ("coco", "COCO-Object (80)")]


def setup():
    import jc_realvoc as R
    import jc_ov42 as O
    import jc_ours44 as U
    R.MODEL = SD21; O.WORK = W / "ov42"; U.WORK = W / "ours"
    import math, torch
    import src.segmentation.sg25_sa as SA
    if not getattr(SA._attention, "_sd21", False):
        def _attention(pair, side):
            # SD 2.1 splits the same widths into 64-dim heads (5/10/20 heads, vs SD 1.5's 8); only the frozen
            # 8-head shape check changes. Every caller imports _attention at call time and averages heads.
            q, k = pair
            if q.shape != k.shape or q.shape[1] != side * side:
                raise ValueError(f"Unexpected {side}x{side} SA Q/K: {tuple(q.shape)}, {tuple(k.shape)}")
            q, k = q.float(), k.float()
            return torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)
        _attention._sd21 = True; SA._attention = _attention
    return R, O, U


def ns(**k):
    d = dict(shard=0, num_shards=1, tag="", max_items=0, limit=0, workers=8); d.update(k); return argparse.Namespace(**d)


def prof(a):
    from jc_real44 import WORK as W44
    src = json.loads((CACHE / "jc_ov42" / "items_prof.json").read_text())
    keep = [x for x in src["items"] if x["rank"] < 20]
    assert len(keep) == 1600 and all(x["negw"] or True for x in keep)
    assert sum(len(x["negw"]) for x in src["items"] if x["rank"] >= 20) == 0          # every negative sits on a size-20 photo
    (W / "ov42").mkdir(parents=True, exist_ok=True); (W / "ours").mkdir(parents=True, exist_ok=True)
    out = dict(src, items=keep); (W / "ov42" / "items_prof.json").write_text(json.dumps(out))
    shutil.copy(W44 / "ours" / "items_prof44.json", W / "ours" / "items_prof44.json")
    print(f"profile items: run 42 size-20 {len(keep)}; run 44 Context jobs "
          f"{len(json.loads((W / 'ours' / 'items_prof44.json').read_text())['jobs'])}", flush=True)


def dump42(a):
    R, O, U = setup(); O.dump(ns(shard=a.shard, num_shards=a.num_shards, max_items=a.limit))


def fit42(a):
    """jc_ov42.fit restricted to size 20 (same code path: object profile = sum of the top-5% queries / count, 4 spherical
    k-means centroids of the pooled negatives)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    R, O, U = setup(); M = O.M
    Pj = json.loads((O.WORK / "items_prof.json").read_text())
    fs = sorted(f for f in (O.WORK / "dump").glob("p*.npz") if not f.stem.endswith(".tmp"))
    assert a.limit or len(fs) == len(Pj["items"]), (len(fs), len(Pj["items"]))
    mods = [str(x) for x in np.load(fs[0])["modules"]]; nM = len(mods)
    obj = defaultdict(list); neg = defaultdict(list)
    for f in fs:
        z = np.load(f); c = str(z["cls"]); r_ = int(z["rank"])
        for mi in range(nM):
            obj[(c, mi)].append((r_, torch.from_numpy(z[f"o_{mi:02d}"]), int(z[f"ko_{mi:02d}"])))
        for k in z.files:
            if k.startswith("n_"):
                _, mi, wk = k.split("_"); neg[(O.CLS80[int(wk)], int(mi))].append(torch.from_numpy(z[k].astype(np.float32)))
    FIT = {"mo": {}, "neg": {}}; nneg = {}
    for (c, mi), L in obj.items():
        L.sort(key=lambda x: x[0]); sub = L[:20]; FIT["mo"][(c, mods[mi])] = sum(x[1] for x in sub) / sum(x[2] for x in sub)
    for (c, mi), parts in neg.items():
        X = F.normalize(torch.cat(parts), dim=1); C4, _, _ = M.cluster(X, 4); FIT["neg"][(c, mods[mi])] = {4: C4}; nneg[(c, mi)] = int(X.shape[0])
    torch.save({"fit": {"20": FIT}, "modules": mods, "npos": Pj["npos"], "nneg_photos": Pj["nneg"], "model": SD21}, O.WORK / "fit.pt")
    print(f"fit42 (SD 2.1): {len(obj)} object pairs; {len(neg)} negative pairs; negative queries min "
          f"{min(nneg.values()) if nneg else 0}", flush=True)


def dump44(a):
    R, O, U = setup(); U.dump44(ns(shard=a.shard, num_shards=a.num_shards, limit=a.limit))


def fit44(a):
    R, O, U = setup(); U.fit44(ns(limit=a.limit))


def capture(a):
    R, O, U = setup(); U.capture(ns(set=a.set, shard=a.shard, num_shards=a.num_shards, limit=a.limit))


def score_one(job):
    import numpy as np
    s, x, o = job; o = Path(o)
    if o.exists():
        return
    import jc_ov42 as O
    import jc_score44 as S44
    z = np.load(W / "ours" / f"cap_{s}" / f"{x['id']}.npz"); truth, valid, other = S44.truth_of(s, x); hw = truth.shape
    res = {"cls": x["cls"], "img": str(x["img"])}
    for i, nm in enumerate(str(n) for n in z["names"]):
        if nm.endswith("|V0"):
            v = O.st.up(O.st.mm(z["maps"][i].astype(np.float32)), hw)
            res[f"sd21:{nm}|raw"] = O.cc.counts(v[None] >= O.THR[:, None, None], truth, valid, other)
    tmp = o.with_name(o.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, o)


def score(a):
    from multiprocessing import get_context
    from jc_real44 import WORK as W44
    X = [x for x in json.loads((W44 / "items.json").read_text())[a.set] if x["primary"]]
    X = [x for x in X if (W / "ours" / f"cap_{a.set}" / f"{x['id']}.npz").exists()]
    if a.limit: X = X[:a.limit]
    out = W / f"scores_{a.set}"; out.mkdir(parents=True, exist_ok=True)
    with get_context("fork").Pool(a.workers) as p:
        for _ in p.imap_unordered(score_one, [(a.set, x, str(out / f"{x['id']}.npz")) for x in X], chunksize=8):
            pass
    print(f"scored {len(X)} items of {a.set}", flush=True)


def report(a):
    import numpy as np
    import jc_ov42 as O
    from jc_real44 import WORK as W44, VOC20
    THR, T55 = O.THR, O.T55
    KEEP = {"ours:uL|V0|raw": "sd15:uL", "ours:u-|V0|raw": "sd15:u-", "ours:uM|V0|raw": "sd15:uM", "nerve|own": "nerve",
            "iseg|own": "iseg", "iseg|given": "iseg_given", "diffseg|0.5": "diffseg"}
    L = []; w = L.append; E = {}
    for s, n in COLS:
        X = [x for x in json.loads((W44 / "items.json").read_text())[s] if x["primary"]]
        fs = sorted(f for f in (W / f"scores_{s}").glob("*.npz") if not f.stem.endswith(".tmp"))
        if not a.tag and len(fs) != len(X):
            raise SystemExit(f"{s}: {len(fs)} scored of {len(X)}")
        Z = [np.load(f) for f in fs]; Z44 = [np.load(W44 / f"scores_{s}_pre_ovdiff" / f.name) for f in fs]
        S = {"cls": np.array([str(z["cls"]) for z in Z]), "grp": np.array([str(z["img"]) for z in Z])}
        for k in Z[0].files:
            if "|" in k:
                S[k] = np.stack([z[k] for z in Z])
        for k, nk in KEEP.items():
            if all(k in z.files for z in Z44):
                v = np.stack([z[k] for z in Z44]); S[nk] = v[:, T55] if v.ndim == 3 else v
        E[s] = (S, O.Boot(S, a.boot))
    P = lambda S, k, t=T55: S[k][:, t] if S[k].ndim == 3 else S[k]
    stat = lambda s, k, t=T55: E[s][1].stat(P(E[s][0], k, t))

    def dif(s, x, y):
        A, B = stat(s, x), stat(s, y); d = A[1] - B[1]
        return A[0] - B[0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    f3 = lambda d: f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"
    rd = lambda d: "**better**" if d[1] > 0 else "**worse**" if d[2] < 0 else "not resolved"
    w("# Run 47: our method on SD 2.1-base (`runs/47_sd21_backbone/PREREG.md`)\n")
    w("Run 44's items, protocol and frozen settings; only the backbone changes (profiles rebuilt on SD 2.1 from the same "
      f"photos). Class-macro mIoU, raw V0; 95% paired photo bootstrap ({a.boot}).\n")
    w("\n## 1. Primary contrasts: ours on SD 2.1 - ours on SD 1.5 (MK4-LR, raw 0.55)\n")
    w("| # | Column | Delta mIoU [95% CI] | Reading |\n|---|---|---|---|")
    for i, (s, n) in enumerate(COLS):
        d = dif(s, "sd21:uL|V0|raw", "sd15:uL"); w(f"| Q{i + 1} | {n} | {f3(d)} | {rd(d)} |")
    w("\n## 2. mIoU (raw 0.55)\n")
    rows = [("Ours SD 2.1, MK4-LR", "sd21:uL|V0|raw"), ("Ours SD 2.1, MK4", "sd21:uM|V0|raw"), ("Ours SD 2.1, no gate", "sd21:u-|V0|raw"),
            ("Ours SD 1.5, MK4-LR (run 44)", "sd15:uL"), ("Ours SD 1.5, no gate (run 44)", "sd15:u-"),
            ("NERVE (SD 2.1)", "nerve"), ("iSeg (SD 2.1, own labels)", "iseg"), ("iSeg, class given", "iseg_given"),
            ("DiffSegmenter (SD 1.5)", "diffseg")]
    w("| Method | " + " | ".join(n for _, n in COLS) + " |\n|---|" + "---:|" * len(COLS))
    for nm, k in rows:
        w(f"| {nm} | " + " | ".join(f"{stat(s, k)[0]:.3f}" if k in E[s][0] else "—" for s, _ in COLS) + " |")
    w("\n## 3. Secondary contrasts (raw 0.55)\n")
    w("| Contrast | " + " | ".join(n for _, n in COLS) + " |\n|---|" + "---|" * len(COLS))
    for nm, x, y in [("SD 2.1: MK4-LR - no gate", "sd21:uL|V0|raw", "sd21:u-|V0|raw"),
                     ("SD 2.1: MK4-LR - MK4", "sd21:uL|V0|raw", "sd21:uM|V0|raw"),
                     ("no gate: SD 2.1 - SD 1.5", "sd21:u-|V0|raw", "sd15:u-"),
                     ("SD 2.1 ours - NERVE", "sd21:uL|V0|raw", "nerve"), ("SD 2.1 ours - iSeg", "sd21:uL|V0|raw", "iseg"),
                     ("SD 2.1 ours - DiffSegmenter", "sd21:uL|V0|raw", "diffseg")]:
        w(f"| {nm} | " + " | ".join(f3(dif(s, x, y)) if x in E[s][0] and y in E[s][0] else "—" for s, _ in COLS) + " |")
    w("\n## 4. Subsets (MK4-LR, raw 0.55)\n")
    sub = lambda S, sel: {k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v) for k, v in S.items()}
    for s, nm, fn in (("context", "Context, 20 VOC classes", lambda c: c in VOC20), ("context", "Context, other 39", lambda c: c not in VOC20),
                      ("coco", "COCO-Object, 20 VOC classes", lambda c: c in VOC20), ("coco", "COCO-Object, other 60", lambda c: c not in VOC20)):
        S = E[s][0]; sel = np.array([fn(c) for c in S["cls"]])
        if not sel.any():
            continue
        S2 = sub(S, sel); B2 = O.Boot(S2, a.boot)
        g = lambda k: B2.stat(P(S2, k)); a1, b1 = g("sd21:uL|V0|raw"), g("sd15:uL"); d = a1[1] - b1[1]
        w(f"- **{nm}** ({int(sel.sum())} items): SD 2.1 {a1[0]:.3f}, SD 1.5 {b1[0]:.3f}, NERVE {g('nerve')[0]:.3f}; "
          f"SD 2.1 - SD 1.5 {a1[0] - b1[0]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
    w("\n## 5. Every threshold (SD 2.1, raw; descriptive: settings were developed on SD 1.5)\n")
    w("| Column | Map | " + " | ".join(f"{t:.3f}" for t in THR) + " |\n|---|---|" + "---:|" * len(THR))
    for s, n in COLS:
        for k in ("sd21:uL|V0|raw", "sd21:u-|V0|raw"):
            w(f"| {n} | {k[5:7]} | " + " | ".join(f"{stat(s, k, t)[0]:.3f}" for t in range(len(THR))) + " |")
    w("\n## 6. Per class (MK4-LR, raw 0.55)\n")
    for s, n in COLS:
        S, B = E[s]; a21 = B.miou(P(S, "sd21:uL|V0|raw"))[0]; a15 = B.miou(P(S, "sd15:uL"))[0]; ne = B.miou(P(S, "nerve"))[0]
        w(f"\n{n}:\n\n| Class | items | SD 2.1 | SD 1.5 | NERVE |\n|---|---:|---:|---:|---:|")
        for i, c in enumerate(B.cl):
            w(f"| {c} | {int((S['cls'] == c).sum())} | {a21[i]:.3f} | {a15[i]:.3f} | {ne[i]:.3f} |")
    w("\nThe paper 2 method stays frozen; this tests its backbone generality, nothing is promoted.\n")
    out = REPORT if not a.tag else REPORT.with_name(f"result_{a.tag}.md"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L)); print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prof")
    for c in ("dump42", "dump44", "capture"):
        p = sub.add_parser(c); p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
        p.add_argument("--limit", type=int, default=0)
        if c == "capture":
            p.add_argument("--set", required=True, choices=["voc", "context", "coco"])
    for c in ("fit42", "fit44"):
        sub.add_parser(c).add_argument("--limit", type=int, default=0)
    p = sub.add_parser("score"); p.add_argument("--set", required=True, choices=["voc", "context", "coco"])
    p.add_argument("--workers", type=int, default=16); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("report"); p.add_argument("--boot", type=int, default=2000); p.add_argument("--tag", default="")
    a = ap.parse_args()
    {"prof": prof, "dump42": dump42, "fit42": fit42, "dump44": dump44, "fit44": fit44, "capture": capture, "score": score,
     "report": report}[a.cmd](a)
