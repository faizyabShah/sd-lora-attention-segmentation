#!/usr/bin/env python3
"""Size-dependent propagation depth (exploratory test on real D and on VOC-Sim).

The fine BRACE operator V0 is mean_{call, head} A_h^2 c (A_h the head-h SA operator of the frozen 64 px
module, c the A3 seed). This test generalises the depth: P_t = mean_{call, head} A_h^t c, t = 0..6
(P_0 = c, P_2 = V0 exactly). A label-free size estimate a_hat = share of the 64x64 seed with
minmax(seed) >= 0.5 chooses the depth through a per-bin rule (bins a_hat < .05, .05-.2, .2-.42, >= .42,
fixed in advance from the size-matched diagnostic); the depth for each bin and the threshold are chosen
on training folds only (grouped folds), and a rule fitted on one dataset is also applied frozen to the other.

  build-real CAPTURE OUT          per D pair and condition (renoise/n0, ddim): P_0..P_6 from stored Q/K
  build-sim  OUT [--shard i --num-shards n]   (GPU) regenerate each VOC-Sim row (native trajectory) and
                                  write P_0..P_6, V0, V3, C16 and the generated image; no attention stored
  score      MAPS_DIR SCORE_DIR   tp/fp/fn/tn of every map at 17 taus (raw) and 9 taus (+DenseCRF R2)
  report     REAL_SCORES SIM_SCORES OUT_MD
"""
from __future__ import annotations

import argparse, csv, json, math, os, sys, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
VS = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim"
CRF_SITE = ROOT / "outputs/dependencies/simplecrf_py310"
MANIFEST = ROOT / "outputs/segmentation/sg_real_atlas/manifest.json"
TMAX = 6
THR = np.round(0.30 + 0.025 * np.arange(17), 3)
THR_CRF = THR[::2]
BINS = (0.05, 0.2, 0.42)
REF = ("V0", "V3", "C16")


def depths(qk64, seed64, device):
    """qk64 [11,2,8,N,d] -> [TMAX+1, 64, 64] float32, P_t = mean over calls and heads of A_h^t c."""
    c = torch.as_tensor(np.asarray(seed64), dtype=torch.float32, device=device).reshape(-1)
    acc = torch.zeros(TMAX + 1, 4096, device=device)
    for call in range(qk64.shape[0]):
        q, k = (x.to(device).float() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), device=device).float()
                for x in (qk64[call, 0], qk64[call, 1]))
        A = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), -1)   # [8,N,N]
        y = c[None].expand(A.shape[0], -1).unsqueeze(-1)
        acc[0] += c / qk64.shape[0]
        for t in range(1, TMAX + 1):
            y = torch.bmm(A, y); acc[t] += y[..., 0].mean(0) / qk64.shape[0]
        del A, q, k
    return acc.reshape(TMAX + 1, 64, 64).cpu().numpy()


def mm(x):
    x = np.asarray(x, np.float32); lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-12)


def a_hat(seed64):
    return float((mm(seed64) >= 0.5).mean())


# ------------------------------------------------------------------ build
def build_real_one(job):
    pdir, out = Path(job[0]), Path(job[1])
    if out.exists():
        return
    torch.set_num_threads(2)
    maps = np.load(pdir / "maps.npz"); res = {}
    for cond in ("renoise/n0", "ddim"):
        tag = cond.replace("/", "_")
        qk = np.load(pdir / f"qk_{tag}.npz")["s64"]
        P = depths(qk, maps[f"{cond}/A3"], "cpu")
        v0 = maps[f"{cond}/V0"]
        res[f"{tag}/P"] = P.astype(np.float32)
        res[f"{tag}/ref"] = np.stack([maps[f"{cond}/{m}"] for m in REF]).astype(np.float32)
        res[f"{tag}/a_hat"] = a_hat(maps[f"{cond}/A3"])
        res[f"{tag}/v0_check"] = float(np.abs(mm(P[2]) - mm(v0)).max())
    tmp = out.with_name(out.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, out)


def build_real(a):
    jobs = []
    for d in sorted((a.capture / "D").glob("*/*/done.json")):
        p = d.parent; o = a.out / f"{p.parts[-2]}__{p.parts[-1]}.npz"; jobs.append((str(p), str(o)))
    a.out.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(build_real_one, jobs), 1):
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


def vocsim_rows():
    with (VS / "ovam_voc_sim.csv").open(newline="", encoding="utf-8") as h:
        r = [dict(x, attribution_prompt=x["prompt"]) for x in csv.DictReader(h) if x["annotated"].lower() == "true"]
    return sorted(r, key=lambda x: x["image_id"])


def build_sim(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    todo = vocsim_rows()[a.shard::a.num_shards]
    a.out.mkdir(parents=True, exist_ok=True)
    runtime = sg27_runtime.make_runtime(a.model)
    for row in todo:
        iid = row["image_id"]; out = a.out / f"{iid}.npz"
        if out.exists():
            continue
        with sg27_runtime.live_maps(runtime, dict(row), "voc") as (sample, sa, seed, base, previous):
            v2, _ = sg_combined_v2.aggregate(sa, seed)
            qk64 = torch.stack([torch.stack(p) for p in sa[64]])            # [11,2,8,N,d] on device
            P = depths(qk64, seed.float().cpu().numpy(), "cuda")
            ref = np.stack([v2["V0"].float().cpu().numpy(), v2["V3"].float().cpu().numpy(),
                            base["C16"].reshape(64, 64).float().cpu().numpy()])
            img = np.asarray(sample.generated_image, np.uint8).copy(); s64 = seed.float().cpu().numpy()
            del qk64
        Image.fromarray(img).save(a.out / f"{iid}.png")
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **{"generate/P": P.astype(np.float32), "generate/ref": ref.astype(np.float32),
                         "generate/a_hat": a_hat(s64), "generate/v0_check": float(np.abs(mm(P[2]) - mm(ref[0])).max())},
                 classname=row["classname"])
        os.replace(tmp, out)
        print(f"[steps] {iid} a_hat {a_hat(s64):.3f} v0 check {float(np.abs(mm(P[2]) - mm(ref[0])).max()):.2e}", flush=True)


# ------------------------------------------------------------------ score
def up(x, hw):
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None], size=hw,
                         mode="bilinear", align_corners=False)[0, 0].numpy()


def counts(stack, truth, valid):
    p = stack & valid; t = truth & valid
    tp = (p & t).sum((1, 2)); fp = (p & ~t).sum((1, 2)); fn = (~p & t[None]).sum((1, 2))
    return np.stack([tp, fp, fn, valid.sum() - tp - fp - fn], 1).astype(np.int64)


def score_one(job):
    f, out = Path(job[0]), Path(job[1])
    if out.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(CRF_SITE)
    z = np.load(f)
    if "classname" in z.files:                                            # VOC-Sim
        iid = f.stem; img = np.asarray(Image.open(f.with_suffix(".png")).convert("RGB"))
        row = next(r for r in vocsim_rows() if r["image_id"] == iid)
        truth = np.any(np.asarray(Image.open(VS / row["annotation"]).convert("RGB")) != 0, -1)
        valid = np.ones_like(truth); conds = ("generate",); cls = str(z["classname"])
    else:
        iid, k = f.stem.split("__")
        img = np.asarray(Image.open(VOC / f"JPEGImages/{iid}.jpg").convert("RGB"))
        lab = np.asarray(Image.open(VOC / f"SegmentationClass/{iid}.png")); truth, valid = lab == int(k), lab != 255
        conds = ("renoise_n0", "ddim"); cls = k
    res = {"area": float((truth & valid).sum() / valid.sum()), "cls": cls}
    for c in conds:
        maps = list(z[f"{c}/P"]) + list(z[f"{c}/ref"])                    # P0..P6, V0, V3, C16
        raw, crf = [], []
        for m in maps:
            x = up(mm(m), truth.shape)
            raw.append(counts(x[None] >= THR[:, None, None], truth, valid))
        for j in list(range(1, TMAX + 1)) + [TMAX + 2]:                 # P1..P6 and V3 (+R2)
            x = up(mm(maps[j]), truth.shape)
            ms = np.stack([apply_ovam_densecrf(img, x >= t, CRF_SITE) for t in THR_CRF]).astype(bool)
            crf.append(counts(ms, truth, valid))
        res[f"{c}/raw"] = np.stack(raw); res[f"{c}/crf"] = np.stack(crf); res[f"{c}/a_hat"] = float(z[f"{c}/a_hat"])
        res[f"{c}/v0_check"] = float(z[f"{c}/v0_check"])
    tmp = out.with_name(out.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, out)


def score(a):
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(a.out / f.name)) for f in sorted(a.maps.glob("*.npz")) if not f.stem.endswith(".tmp")]
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(score_one, jobs), 1):
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------ report
RAW_NAMES = [f"P{t}" for t in range(TMAX + 1)] + list(REF)
CRF_NAMES = [f"P{t}+R2" for t in range(1, TMAX + 1)] + ["V3+R2"]


def load_scores(d, cond):
    fs = sorted(f for f in d.glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    if cond == "generate":
        fold = np.array([zlib.crc32(f.stem.encode()) % 5 for f in fs]); grp = np.array([f.stem for f in fs])
    else:
        fo = {e["image"]: e["fold"] for e in json.loads(MANIFEST.read_text())["D"]}
        grp = np.array([f.stem.split("__")[0] for f in fs]); fold = np.array([fo[g] for g in grp])
    return {"raw": np.stack([z[f"{cond}/raw"] for z in Z]), "crf": np.stack([z[f"{cond}/crf"] for z in Z]),
            "a_hat": np.array([float(z[f"{cond}/a_hat"]) for z in Z]), "area": np.array([float(z["area"]) for z in Z]),
            "cls": np.array([str(z["cls"]) for z in Z]), "fold": fold, "grp": grp,
            "v0": np.array([float(z[f"{cond}/v0_check"]) for z in Z])}


def miou(C, cls, sel):
    """C [P, T, 4] -> class-macro mIoU per tau over the selected pairs."""
    out = []
    for c in np.unique(cls[sel]):
        x = C[sel & (cls == c)].sum(0); out.append(x[:, 0] / np.maximum(x[:, :3].sum(1), 1))
    return np.mean(out, 0)


def bin_of(a):
    return np.digitize(a, BINS)


def rule_counts(S, kind, rule):
    """Per pair counts of the depth chosen by rule[bin] (index into P maps of that kind)."""
    X = S[kind]; b = bin_of(S["a_hat"])
    return X[np.arange(len(b)), np.array([rule[i] for i in b])]


def fit_rule(S, kind, sel, choices):
    """Greedy-exact: per tau, for each bin pick the depth maximising the training-fold class-macro mIoU with
    the other bins held at their current choice (2 sweeps, start at depth 2 = V0); return best (rule, tau)."""
    X = S[kind]; b = bin_of(S["a_hat"]); best = (-1, None, None)
    for ti in range(X.shape[2]):
        rule = [2 if kind == "raw" else 1] * 4                              # start every bin at P2 (= V0)
        for _ in range(2):
            for bi in range(4):
                sc = []
                for j in range(len(choices)):
                    r = list(rule); r[bi] = j
                    C = X[np.arange(len(b)), np.array([choices[r[i]] for i in b])][:, ti:ti + 1]
                    sc.append(miou(C, S["cls"], sel)[0])
                rule[bi] = int(np.argmax(sc))
        r = [choices[i] for i in rule]
        C = X[np.arange(len(b)), np.array([r[i] for i in b])][:, ti:ti + 1]
        m = miou(C, S["cls"], sel)[0]
        if m > best[0]:
            best = (m, r, ti)
    return best


def oof_fixed(S, kind, j):
    C = S[kind][:, j]; P = np.zeros_like(C[:, 0]); taus = []
    for fo in range(5):
        tr, te = S["fold"] != fo, S["fold"] == fo
        ti = int(np.argmax(miou(C, S["cls"], tr))); P[te] = C[te, ti]; taus.append(ti)
    return miou(P[:, None], S["cls"], np.ones(len(P), bool))[0], P, taus


def oof_rule(S, kind, choices):
    P = np.zeros_like(S[kind][:, 0, 0]); rules = []
    for fo in range(5):
        tr, te = S["fold"] != fo, S["fold"] == fo
        _, r, ti = fit_rule(S, kind, tr, choices); rules.append((r, ti))
        P[te] = S[kind][np.where(te)[0], np.array([r[i] for i in bin_of(S["a_hat"][te])]), ti]
    return miou(P[:, None], S["cls"], np.ones(len(P), bool))[0], P, rules


def boot(S, Pa, Pb, n=2000):
    rng = np.random.default_rng(0); g = np.unique(S["grp"]); idx = {x: np.where(S["grp"] == x)[0] for x in g}; d = []
    for _ in range(n):
        s = np.concatenate([idx[x] for x in rng.choice(g, len(g))]); sel = np.ones(len(s), bool)
        d.append(miou(Pa[s][:, None], S["cls"][s], sel)[0] - miou(Pb[s][:, None], S["cls"][s], sel)[0])
    return np.percentile(d, [2.5, 97.5])


def report(a):
    L = []; w = L.append
    w("# Size-dependent propagation depth: real D vs VOC-Sim (exploratory)\n")
    w("P_t = mean over the 11 window calls and 8 heads of A_h^t c on the frozen 64 px BRACE module (P_2 = V0). "
      "a_hat = share of the seed with minmax ≥ 0.5 (label-free). Depth rule: one depth per a_hat bin "
      f"(< {BINS[0]}, {BINS[0]}–{BINS[1]}, {BINS[1]}–{BINS[2]}, ≥ {BINS[2]}), chosen with τ on training folds. "
      "Class-macro mIoU; OOF = τ (and rule) picked on 4 grouped folds, scored on the 5th; Δ CIs by image bootstrap.\n")
    sets = [("real D, renoise/n0", a.real, "renoise_n0"), ("real D, ddim", a.real, "ddim"), ("VOC-Sim, generate", a.sim, "generate")]
    S = {name: load_scores(d, c) for name, d, c in sets}
    fitted = {}
    for name, d, c in sets:
        s = S[name]
        w(f"## {name} ({len(s['area'])} pairs)\n")
        w(f"P_2 vs stored V0 (minmax) max abs diff: {s['v0'].max():.1e}. ρ(a_hat, true area) = "
          f"{np.corrcoef(np.argsort(np.argsort(s['a_hat'])), np.argsort(np.argsort(s['area'])))[0, 1]:.2f}; "
          "pairs per a_hat bin: " + ", ".join(str(int((bin_of(s['a_hat']) == i).sum())) for i in range(4)) + "\n")
        w("| map | OOF mIoU raw | OOF mIoU +R2 |\n|---|---:|---:|")
        res = {}
        for j, nm in enumerate(RAW_NAMES):
            m, P, _ = oof_fixed(s, "raw", j); res[nm] = P
            crf = ""
            cj = CRF_NAMES.index(f"{nm}+R2") if f"{nm}+R2" in CRF_NAMES else (CRF_NAMES.index("P2+R2") if nm == "V0" else None)
            if cj is not None:
                mc, Pc, _ = oof_fixed(s, "crf", cj); res[nm + "+R2"] = Pc; crf = f"{mc:.3f}"
            w(f"| {nm} | {m:.3f} | {crf} |")
        ch_raw = list(range(0, TMAX + 1)); ch_crf = list(range(0, TMAX))                  # crf index 0 = P1
        mr, Pr, rr = oof_rule(s, "raw", ch_raw); mc, Pc, rc = oof_rule(s, "crf", ch_crf)
        res["rule"] = Pr; res["rule+R2"] = Pc
        w(f"| **size rule** | **{mr:.3f}** | **{mc:.3f}** |\n")
        w("Rules picked per fold (depth per bin, τ): raw " + "; ".join(f"{r} @{THR[t]:.3f}" for r, t in rr)
          + " — +R2 " + "; ".join(f"{[x + 1 for x in r]} @{THR_CRF[t]:.2f}" for r, t in rc) + "\n")
        for x, y in (("rule", "V0"), ("rule+R2", "V0+R2"), ("rule+R2", "V3+R2"), ("rule", "C16")):
            lo, hi = boot(s, res[x], res[y])
            dm = miou(res[x][:, None], s["cls"], np.ones(len(s["cls"]), bool))[0] - miou(res[y][:, None], s["cls"], np.ones(len(s["cls"]), bool))[0]
            w(f"- {x} − {y}: {dm:+.3f} (95% CI {lo:+.3f} … {hi:+.3f})")
        # oracle depth per pair (headroom) at the OOF tau of P2
        X = s["raw"]; best = X[:, :, :, 0] / np.maximum(X[:, :, :, :3].sum(-1), 1)
        w(f"\nHeadroom: mean per-pair IoU at τ 0.55 — P2 {best[:, 2, 10].mean():.3f}; best depth per pair (oracle) "
          f"{best[:, :TMAX + 1, 10].max(1).mean():.3f}.\n")
        w("Best depth by true-area bin (raw, mean per-pair IoU at each depth's best global τ):")
        for lo, hi in ((0, .05), (.05, .2), (.2, .42), (.42, 1.01)):
            sel = (s["area"] >= lo) & (s["area"] < hi)
            if sel.sum() >= 5:
                v = [best[sel, j, :].mean(0).max() for j in range(TMAX + 1)]
                w(f"- area {lo:.2f}–{hi:.2f} (n={sel.sum()}): " + " ".join(f"P{j} {x:.3f}" for j, x in enumerate(v)))
        w("")
        fitted[name] = (fit_rule(s, "raw", np.ones(len(s["cls"]), bool), ch_raw), fit_rule(s, "crf", np.ones(len(s["cls"]), bool), ch_crf))
    w("## Transfer: rule fitted on all of one dataset, applied frozen to the other\n")
    w("| fitted on → applied to | rule (raw) | mIoU raw | P2 raw at same τ | rule (+R2) | mIoU +R2 | P2+R2 at same τ |\n|---|---|---:|---:|---|---:|---:|")
    for src, dst in (("real D, ddim", "VOC-Sim, generate"), ("real D, renoise/n0", "VOC-Sim, generate"),
                     ("VOC-Sim, generate", "real D, ddim"), ("VOC-Sim, generate", "real D, renoise/n0")):
        (mr_, r_, t_), (mc_, rc_, tc_) = fitted[src]; s = S[dst]; allp = np.ones(len(s["cls"]), bool)
        cr = rule_counts(s, "raw", r_)[:, t_:t_ + 1]; cc = rule_counts(s, "crf", rc_)[:, tc_:tc_ + 1]
        w(f"| {src} → {dst} | {r_} @{THR[t_]:.3f} | {miou(cr, s['cls'], allp)[0]:.3f} | "
          f"{miou(s['raw'][:, 2, t_:t_ + 1], s['cls'], allp)[0]:.3f} | {[x + 1 for x in rc_]} @{THR_CRF[tc_]:.2f} | "
          f"{miou(cc, s['cls'], allp)[0]:.3f} | {miou(s['crf'][:, 1, tc_:tc_ + 1], s['cls'], allp)[0]:.3f} |")
    a.out_md.write_text("\n".join(L) + "\n"); print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-real"); p.add_argument("capture", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("build-sim"); p.add_argument("out", type=Path); p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1); p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p = sub.add_parser("score"); p.add_argument("maps", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("report"); p.add_argument("real", type=Path); p.add_argument("sim", type=Path); p.add_argument("out_md", type=Path)
    a = ap.parse_args()
    {"build-real": build_real, "build-sim": build_sim, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
