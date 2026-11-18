#!/usr/bin/env python3
"""runs/27_fewshot_shadow/PREREG.md: few-shot supervised shadow attention (exploratory, supervised).

Reuses runs/26's shadow branch (scripts/jc_shadow.py); the generator's attention is untouched. Rank 4, 60 steps.
  A  frozen BRACE
  B  CA copies, direct supervision of the seed            (loss on the shadow seed only)
  C  CA copies, supervision through the BRACE hierarchy   (loss on the V3 map; original SA)
  D  CA + SA (16/32/64) copies, through the hierarchy     (loss on the V3 map; shadow SA)
Train: 3 VOC-Sim images per class (60). Test: 3 COCO-Cap images per class (60). Both captured for all arms.

split    runs/27_fewshot_shadow/split.json
train    (GPU) one arm -> <BRACE_CACHE_DIR>/jc_fewshot/ckpt{tag}/{arm}.pt, runs/27_fewshot_shadow/train_{arm}{tag}.json
capture  (GPU) --set train|test: maps [4 arms, 2 maps (seed, V3), 64, 64]
score    (CPU) raw and + DenseCRF R2 counts at 0.30-0.70 (step 0.025)
report   (CPU) runs/27_fewshot_shadow/result{tag}.md
"""
from __future__ import annotations

import argparse, json, os, random, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jc_shadow as J  # noqa: E402

J.RANK = 4                                   # Low() reads the module global at construction
cc, st, nr = J.cc, J.st, J.nr
WORK = J.CACHE / "jc_fewshot"
OUT = ROOT / "runs/27_fewshot_shadow"
ARMS = ("A", "B", "C", "D")
MAPS = ("seed", "V3")
THR, T55 = J.THR, J.T55
LR, STEPS, PER_CLASS = 1e-3, 60, 3
SET_DS = {"train": "voc", "test": "coco", "all": "coco"}      # "all" = every COCO-Cap image (amendment 1)
NAME = {"train": "VOC-Sim training images (in-sample)", "test": "COCO-Cap 60 (3/class)", "all": "COCO-Cap all"}


def shadow(rt, arm):
    sh = J.Shadow(rt, "C" if arm == "D" else "B")        # CA adapters first (same init for B/C/D), then SA for D
    sh.arm = arm
    return sh.to(rt["pipe"].device)


def maps_for(rec, rt, sh):
    s = J.shadow_seed(rec, sh)
    return s, J.shadow_v3(rec, rt, s, sh if (sh is not None and sh.arm == "D") else None, grad=False)


def split(a):
    o = {}
    for nm, ds in SET_DS.items():
        by = {}
        for r in nr.rows(ds):
            by.setdefault(r["label"], []).append(r["image_id"])
        o[nm] = sorted(i for c in sorted(by) for i in sorted(by[c], key=lambda i: zlib.crc32(i.encode()))[:PER_CLASS])
        o[nm + "_classes"] = len(by)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "split.json").write_text(json.dumps(o, indent=1))
    print({k: (len(v) if isinstance(v, list) else v) for k, v in o.items()}, flush=True)


def rows_for(nm, n=0, shard=0, num_shards=1):
    R = nr.rows("coco") if nm == "all" else J.rows_for(SET_DS[nm], json.loads((OUT / "split.json").read_text())[nm])
    R = R[shard::num_shards]
    return R[:n] if n else R


def train(a):
    R = rows_for("train", a.n_train); rt = J.make_runtime(a.model); dev = rt["pipe"].device
    torch.manual_seed(0); sh = shadow(rt, a.arm)
    opt = torch.optim.AdamW(sh.parameters(), lr=LR, weight_decay=J.WD)
    ck = WORK / f"ckpt{a.tag}"; ck.mkdir(parents=True, exist_ok=True)
    order = list(range(len(R))); random.Random(0).shuffle(order); log = []
    steps = a.steps or STEPS
    print(f"[fewshot:{a.arm}] {len(R)} training images, {steps} steps, rank {J.RANK}, "
          f"{sum(p.numel() for p in sh.parameters())} parameters", flush=True)
    for step in range(1, steps + 1):
        row = R[order[(step - 1) % len(order)]]; t0 = time.perf_counter()
        rec, sample = J.generate(rt, row, "voc")
        t = J.truth64(np.asarray(sample.target_mask, bool)).to(dev)
        s = J.shadow_seed(rec, sh)
        if a.arm == "B":
            ls = J.seg_loss(s, t); lv = torch.zeros((), device=dev); loss = ls
        else:
            v = J.mm_t(J.shadow_v3(rec, rt, s, sh if a.arm == "D" else None, grad=True))
            lv = J.seg_loss(v, t); ls = J.seg_loss(s.detach(), t); loss = lv      # seed loss logged, not trained
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(sh.parameters(), J.CLIP)); opt.step()
        log.append({"step": step, "id": row["image_id"], "loss": float(loss), "l_v3": float(lv), "l_seed": float(ls),
                    "grad_norm": gn, "sec": time.perf_counter() - t0})
        print(f"[fewshot:{a.arm}] step {step} loss {float(loss):.4f} (V3 {float(lv):.4f} seed {float(ls):.4f}) "
              f"|g| {gn:.3f} {log[-1]['sec']:.1f}s", flush=True)
        del s, loss, lv, ls; J.cleanup(rt, rec, sample)
    torch.save(sh.state_dict(), ck / f"{a.arm}.pt")
    (OUT / f"train_{a.arm}{a.tag}.json").write_text(json.dumps(log, indent=0))


@torch.no_grad()
def capture(a):
    from src.segmentation import sg_combined_v2
    R = rows_for(a.set, a.max_images, a.shard, a.num_shards); ds = SET_DS[a.set]; rt = J.make_runtime(a.model)
    sh = {}
    for arm in ARMS[1:]:
        m = shadow(rt, arm); m.load_state_dict(torch.load(WORK / f"ckpt{a.tag}" / f"{arm}.pt", map_location=rt["pipe"].device))
        sh[arm] = m.eval()
    out = WORK / f"{a.set}{a.tag}"; out.mkdir(parents=True, exist_ok=True)
    for row in R:
        o = out / f"{row['image_id']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter(); rec, sample = J.generate(rt, row, ds)
        md = rt["modules"]; sa = {16: rec.multiscale_qk[md[16]], 32: rec.multiscale_qk[md[32]], 64: rec.sa_qk}
        seed = rec.seeds()["selected/A3"].float(); v3 = sg_combined_v2.aggregate(sa, seed)[0]["V3"].float()
        s0, v0 = maps_for(rec, rt, None)
        fid_s = float((s0 - seed).abs().max()); fid_v = float((J.mm_t(v0) - J.mm_t(v3)).abs().max())
        maps = [[seed, v3]] + [list(maps_for(rec, rt, sh[arm])) for arm in ARMS[1:]]
        maps = np.stack([np.stack([x.float().cpu().numpy() for x in p]) for p in maps]).astype(np.float32)
        if not np.isfinite(maps).all():
            raise ValueError(f"non-finite map for {row['image_id']}")
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, maps=maps.astype(np.float16), cls=row["label"], truth=np.asarray(sample.target_mask, bool),
                 image=np.asarray(sample.generated_image, np.uint8), fid_seed=fid_s, fid_v3=fid_v)
        os.replace(tmp, o)
        print(f"[fewshot:{a.set}] {row['image_id']} fid seed {fid_s:.1e} V3 {fid_v:.1e} {time.perf_counter() - t0:.1f}s", flush=True)
        J.cleanup(rt, rec, sample)


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
    for s in a.sets:
        sc = WORK / f"{s}{a.tag}_scores"; sc.mkdir(parents=True, exist_ok=True)
        jobs = [(str(f), str(sc / f.name)) for f in sorted((WORK / (s + a.tag)).glob("*.npz"))
                if not f.stem.endswith(".tmp")][a.shard::a.num_shards]
        print(s, len(jobs), flush=True)
        with Pool(a.workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(score_one, jobs, chunksize=1), 1):
                if n % 10 == 0 or n == len(jobs):
                    print(f"{s} {n}/{len(jobs)}", flush=True)


def _sub(S, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in S.items()}


def report(a):
    S = {s: cc.load_gen(WORK / f"{s}{a.tag}_scores") for s in ("train", "all")}
    t60 = set(json.loads((OUT / "split.json").read_text())["test"])
    S["test"] = _sub(S["all"], np.array([g in t60 for g in S["all"]["grp"]]))
    one = {s: np.ones(len(S[s]["grp"]), bool) for s in S}
    mi = lambda s, P: cc.miou(P[:, None], S[s]["cls"], one[s])[0]
    key = lambda arm, m, crf: f"{arm}|{m}{'+R2' if crf else ''}|raw"
    tsel = lambda k: int(np.argmax(cc.miou(S["train"][k], S["train"]["cls"], one["train"])))

    def at(s, k, mode):
        return S[s][k][:, T55] if mode == "0.55" else S[s][k][:, tsel(k)]

    def dci(Pa, Pb, s="all"):
        lo, hi = cc.boot(S[s], Pa, Pb, n=a.boot); d = mi(s, Pa) - mi(s, Pb)
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", d, lo

    L = ["# Few-shot supervised shadow attention (`PREREG.md`, amendment 1)\n",
         "`scripts/jc_fewshot.py`. **Exploratory, supervised, few-shot** (masks of 3 VOC-Sim images per class). Shadow "
         f"low-rank copies (rank {J.RANK}, {STEPS} steps, lr {LR:g}); the generator is untouched. All arms from one live "
         "generation per image (paired). *train-sel.* = each arm's best threshold on the 60 training images, applied "
         "unchanged to COCO-Cap. Primary test set: all COCO-Cap images (amendment 1); the registered 60-image subset is "
         f"also reported. Class-macro mIoU; Δ with paired image-bootstrap 95% interval ({a.boot}).\n",
         "## Data and checks\n"]
    for s in ("train", "all"):
        Z = [np.load(f) for f in sorted((WORK / f"{s}{a.tag}_scores").glob("*.npz")) if not f.stem.endswith(".tmp")]
        fs = max(float(z["fid_seed"]) for z in Z); fv = max(float(z["fid_v3"]) for z in Z)
        L.append(f"- {NAME[s]}: {len(Z)} images, {len(np.unique(S[s]['cls']))} classes. Zero-adapter shadow vs frozen "
                 f"BRACE: max |Δ| seed {fs:.1e}, BRACE map {fv:.1e}.")
    for arm in ARMS[1:]:
        f = OUT / f"train_{arm}{a.tag}.json"
        if f.exists():
            lg = json.loads(f.read_text()); k = max(1, len(lg) // 4)
            L.append(f"- Training {arm}: loss first {k} steps {np.mean([x['loss'] for x in lg[:k]]):.3f} → last {k} "
                     f"{np.mean([x['loss'] for x in lg[-k:]]):.3f} (seed term {np.mean([x['l_seed'] for x in lg[:k]]):.3f} → "
                     f"{np.mean([x['l_seed'] for x in lg[-k:]]):.3f}); {np.mean([x['sec'] for x in lg]):.1f} s/step.")
    lab = {"A": "A frozen BRACE", "B": "B CA, seed supervision", "C": "C CA, hierarchy supervision",
           "D": "D CA + SA, hierarchy supervision"}
    for crf in (True, False):
        L.append(f"\n## {'With DenseCRF (primary)' if crf else 'Raw (secondary)'}\n")
        L.append("| Arm | Map | Train (in-sample), own best | train-sel. thr. | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 "
                 "| COCO-Cap 60, train-sel. | COCO-Cap 60, 0.55 |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            for m, ml in (("seed", "seed"), ("V3", "BRACE hierarchy")):
                k = key(arm, m, crf)
                L.append(f"| {lab[arm]} | {ml} | {mi('train', at('train', k, 'sel')):.3f} | {THR[tsel(k)]:.3f} | "
                         + " | ".join(f"{mi(s, at(s, k, md)):.3f}" for s in ("all", "test") for md in ("sel", "0.55")) + " |")
    L.append("\n## Comparisons (BRACE hierarchy + DenseCRF)\n")
    L.append("| ID | Comparison | COCO-Cap all, train-sel. | COCO-Cap all, 0.55 | Positive (all, 0.55)? | COCO-Cap 60, 0.55 |")
    L.append("|---|---|---|---|---|---|")
    for tid, lb, x, y in (("F1", "D − C (SA copies add to hierarchy-supervised CA?)", "D", "C"),
                          ("F2", "C − B (hierarchy vs direct supervision of CA)", "C", "B"),
                          ("F3", "B − A", "B", "A"), ("F4", "C − A", "C", "A"), ("F5", "D − A", "D", "A")):
        c = {mode: dci(at("all", key(x, "V3", True), mode), at("all", key(y, "V3", True), mode)) for mode in ("sel", "0.55")}
        c60 = dci(at("test", key(x, "V3", True), "0.55"), at("test", key(y, "V3", True), "0.55"), "test")
        L.append(f"| {tid} | {lb} | {c['sel'][0]} | {c['0.55'][0]} | {'yes' if round(c['0.55'][2], 3) > 0 else 'no'} | {c60[0]} |")
    L.append("\n\"Positive\" = 95% interval above zero on all COCO-Cap at 0.55. Exploratory.\n")
    L.append("Seed + DenseCRF @0.55, COCO-Cap all, vs A: " + "; ".join(
        f"{x} {dci(at('all', key(x, 'seed', True), '0.55'), at('all', key('A', 'seed', True), '0.55'))[0]}" for x in ARMS[1:]) + "\n")
    cl = S["all"]["cls"]
    L.append("Per class, COCO-Cap all, BRACE hierarchy + DenseCRF @0.55 (A / B / C / D): " + ", ".join(
        f"{c} " + " / ".join(f"{cc.miou(at('all', key(arm, 'V3', True), '0.55')[:, None], cl, cl == c)[0]:.2f}" for arm in ARMS)
        for c in np.unique(cl)) + "\n")
    txt = "\n".join(L) + "\n"; print(txt)
    (OUT / f"result{a.tag}.md").write_text(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    M = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    sub.add_parser("split")
    p = sub.add_parser("train"); p.add_argument("--arm", choices=ARMS[1:], required=True)
    p.add_argument("--steps", type=int, default=0); p.add_argument("--n-train", type=int, default=0)
    p.add_argument("--tag", default=""); p.add_argument("--model", default=M)
    p = sub.add_parser("capture"); p.add_argument("--set", choices=("train", "test", "all"), required=True)
    p.add_argument("--max-images", type=int, default=0); p.add_argument("--tag", default=""); p.add_argument("--model", default=M)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    for nm in ("score", "report"):
        p = sub.add_parser(nm); p.add_argument("--sets", nargs="+", default=["train", "all"]); p.add_argument("--tag", default="")
        p.add_argument("--workers", type=int, default=16); p.add_argument("--boot", type=int, default=2000)
        p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    {"split": split, "train": train, "capture": capture, "score": score, "report": report}[a.cmd](a)
