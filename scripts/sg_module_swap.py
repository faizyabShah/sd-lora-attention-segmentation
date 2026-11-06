#!/usr/bin/env python3
"""SA module swap (exploratory, real D first-50-photo pairs and VOC-Sim).

The frozen BRACE SA modules are the last transformer of each up block (16: up_blocks.1.attentions.2,
32: up_blocks.2.attentions.2, 64: up_blocks.3.attentions.2). The real-mask SA atlas ranked them the
leakiest in their blocks on both real D and VOC-Sim. This test feeds other modules of the same block into
the unchanged frozen functions (sg_combined_v2.aggregate for V0/V3, sg27_fusion.base_paths for C16) plus
the one-step fine map P1 = mean_{call, head} A_h c:
  base   (u1.2, u2.2, u3.2)   the frozen choice (must reproduce the stored maps)
  f30    (u1.2, u2.2, u3.0)   fine module swapped
  f31    (u1.2, u2.2, u3.1)
  coarse (u1.1, u2.1, u3.2)   coarse modules swapped
  best   (u1.1, u2.1, u3.0)   best keep lift per side in the real-D atlas
Real D: renoise/n0 of the pairs whose capture stored every module's Q/K (76). VOC-Sim: regenerated live,
only the 64x64 maps and the image are written.
  build-real CAPTURE OUT | build-sim OUT [--shard --num-shards] | score MAPS OUT | report REAL SIM OUT_MD
"""
from __future__ import annotations

import argparse, json, os, sys, zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
from src.segmentation import sg_combined_v2, sg27_fusion  # noqa: E402
from src.segmentation.sg25_sa import _apply, _attention  # noqa: E402

U = lambda b, a: f"up_blocks.{b}.attentions.{a}.transformer_blocks.0.attn1"
CONFIGS = {"base": (U(1, 2), U(2, 2), U(3, 2)), "f30": (U(1, 2), U(2, 2), U(3, 0)),
           "f31": (U(1, 2), U(2, 2), U(3, 1)), "coarse": (U(1, 1), U(2, 1), U(3, 2)),
           "best": (U(1, 1), U(2, 1), U(3, 0))}
MODULES = sorted({m for c in CONFIGS.values() for m in c})
MAPS = ("V0", "V3", "C16", "P1")
CRF_MAPS = ("V0", "V3", "P1")


@torch.inference_mode()
def config_maps(qk, seed):
    """qk: module -> list of 11 (q, k) [8,N,d]; seed [64,64] tensor -> {cfg: [4,64,64]}."""
    out = {}
    for name, (m16, m32, m64) in CONFIGS.items():
        sa = {16: qk[m16], 32: qk[m32], 64: qk[m64]}
        v2, _ = sg_combined_v2.aggregate(sa, seed)
        base = sg27_fusion.base_paths(sa, seed)
        c = seed.float().reshape(-1); p1 = torch.zeros(4096, device=seed.device)
        for pair in qk[m64]:
            p1 += _apply(_attention(pair, 64), c).mean(0) / len(qk[m64])
        out[name] = torch.stack([v2["V0"].float(), v2["V3"].float(), base["C16"].reshape(64, 64).float(),
                                 p1.reshape(64, 64)]).cpu().numpy()
    return out


def build_real_one(job):
    pdir, out = Path(job[0]), Path(job[1])
    if out.exists():
        return
    torch.set_num_threads(2)
    maps = np.load(pdir / "maps.npz"); atlas = np.load(pdir / "atlas_renoise_n0.npz")
    qk = {}
    for m in MODULES:
        a = torch.from_numpy(atlas[f"qk/{m}"])                                   # [11,2,8,N,d]
        qk[m] = [(a[c, 0], a[c, 1]) for c in range(a.shape[0])]
    seed = torch.from_numpy(maps["renoise/n0/A3"]).float()
    res = config_maps(qk, seed)
    check = max(float(np.abs(st.mm(res["base"][i]) - st.mm(maps[f"renoise/n0/{m}"])).max()) for i, m in enumerate(MAPS[:3]))
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **{f"cfg/{k}": v for k, v in res.items()}, base_check=check); os.replace(tmp, out)


def build_real(a):
    first = {e["image"] for e in json.loads(st.MANIFEST.read_text())["D"] if e.get("all_qk")}
    jobs = [(str(d.parent), str(a.out / f"{d.parent.parts[-2]}__{d.parent.parts[-1]}.npz"))
            for d in sorted((a.capture / "D").glob("*/*/done.json")) if d.parent.parts[-2] in first]
    a.out.mkdir(parents=True, exist_ok=True); print(len(jobs), "pairs", flush=True)
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(build_real_one, jobs), 1):
            if n % 10 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


def build_sim(a):
    from scripts import sg27_runtime
    todo = st.vocsim_rows()[a.shard::a.num_shards]
    a.out.mkdir(parents=True, exist_ok=True)
    runtime = sg27_runtime.make_runtime(a.model)
    rc = runtime["recorder_class"]; fine = rc.selected_sa
    rc.extra_sa = tuple(m for m in MODULES if m != fine)
    for row in todo:
        iid = row["image_id"]; out = a.out / f"{iid}.npz"
        if out.exists():
            continue
        with sg27_runtime.live_maps(runtime, dict(row), "voc") as (sample, sa, seed, base, previous):
            rec = runtime["recorders"][-1]
            qk = {m: rec.multiscale_qk[m] for m in rc.extra_sa}; qk[fine] = rec.sa_qk
            if any(len(v) != 11 for v in qk.values()):
                raise ValueError(f"{iid}: SA calls {[len(v) for v in qk.values()]}")
            res = config_maps(qk, seed)
            v2, _ = sg_combined_v2.aggregate(sa, seed)
            check = max(float(np.abs(st.mm(res["base"][0]) - st.mm(v2["V0"].float().cpu().numpy())).max()),
                        float(np.abs(st.mm(res["base"][2]) - st.mm(base["C16"].reshape(64, 64).float().cpu().numpy())).max()))
            img = np.asarray(sample.generated_image, np.uint8).copy()
        Image.fromarray(img).save(a.out / f"{iid}.png")
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **{f"cfg/{k}": v for k, v in res.items()}, base_check=check, classname=row["classname"])
        os.replace(tmp, out)
        print(f"[swap] {iid} base check {check:.1e}", flush=True)


def score_one(job):
    f, out = Path(job[0]), Path(job[1])
    if out.exists():
        return
    torch.set_num_threads(1)
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    z = np.load(f)
    if "classname" in z.files:
        img = np.asarray(Image.open(f.with_suffix(".png")).convert("RGB"))
        row = next(r for r in st.vocsim_rows() if r["image_id"] == f.stem)
        truth = np.any(np.asarray(Image.open(st.VS / row["annotation"]).convert("RGB")) != 0, -1)
        valid = np.ones_like(truth); cls = str(z["classname"])
    else:
        iid, k = f.stem.split("__")
        img = np.asarray(Image.open(st.VOC / f"JPEGImages/{iid}.jpg").convert("RGB"))
        lab = np.asarray(Image.open(st.VOC / f"SegmentationClass/{iid}.png")); truth, valid = lab == int(k), lab != 255; cls = k
    res = {"area": float((truth & valid).sum() / valid.sum()), "cls": cls, "base_check": float(z["base_check"])}
    for cfg in CONFIGS:
        M = z[f"cfg/{cfg}"]
        for i, m in enumerate(MAPS):
            x = st.up(st.mm(M[i]), truth.shape)
            res[f"{cfg}/{m}/raw"] = st.counts(x[None] >= st.THR[:, None, None], truth, valid)
            if m in CRF_MAPS:
                ms = np.stack([apply_ovam_densecrf(img, x >= t, st.CRF_SITE) for t in st.THR_CRF]).astype(bool)
                res[f"{cfg}/{m}+R2/raw"] = st.counts(ms, truth, valid)
    tmp = out.with_name(out.stem + ".tmp.npz"); np.savez(tmp, **res); os.replace(tmp, out)


def score(a):
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(a.out / f.name)) for f in sorted(a.maps.glob("*.npz")) if not f.stem.endswith(".tmp")]
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(score_one, jobs), 1):
            if n % 25 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


def load(d, sim):
    fs = sorted(f for f in d.glob("*.npz") if not f.stem.endswith(".tmp")); Z = [np.load(f) for f in fs]
    if sim:
        grp = np.array([f.stem for f in fs]); fold = np.array([zlib.crc32(g.encode()) % 5 for g in grp])
    else:
        fo = {e["image"]: e["fold"] for e in json.loads(st.MANIFEST.read_text())["D"]}
        grp = np.array([f.stem.split("__")[0] for f in fs]); fold = np.array([fo[g] for g in grp])
    S = {"cls": np.array([str(z["cls"]) for z in Z]), "fold": fold, "grp": grp,
         "area": np.array([float(z["area"]) for z in Z]), "check": np.array([float(z["base_check"]) for z in Z])}
    for key in (k for k in Z[0].files if k.endswith("/raw")):
        S[key] = np.stack([z[key] for z in Z])
    return S


def report(a):
    L = []; w = L.append
    w("# SA module swap: real D (all-module pairs) vs VOC-Sim (exploratory)\n")
    w("Configs (16 / 32 / 64 px module): " + "; ".join(f"**{k}** " + " / ".join(m.replace('up_blocks.', 'u').replace('.attentions.', '.')
      .replace('.transformer_blocks.0.attn1', '') for m in v) for k, v in CONFIGS.items())
      + ". Frozen V0 / V3 / C16 functions, P1 = one fine step. Class-macro mIoU, OOF τ over grouped folds; "
      "Δ vs base for the same map with an image-bootstrap 95% CI. The module ranking came from the real-D "
      "SA atlas, which includes these real pairs; VOC-Sim is the cleaner test.\n")
    for name, d, sim in (("real D, renoise/n0", a.real, False), ("VOC-Sim, generate", a.sim, True)):
        S = load(d, sim)
        w(f"## {name} ({len(S['cls'])} pairs)\n")
        w(f"Base config vs stored frozen maps, max abs diff (minmax): {S['check'].max():.1e}\n")
        names = [m for m in MAPS] + [f"{m}+R2" for m in CRF_MAPS]
        w("| config | " + " | ".join(names) + " |\n|---|" + "---:|" * len(names))
        P = {}
        for cfg in CONFIGS:
            cells = []
            for m in names:
                X = S[f"{cfg}/{m}/raw"]
                mi, Pp, _ = st.oof_fixed({"x": X[:, None], "cls": S["cls"], "fold": S["fold"]}, "x", 0)
                P[(cfg, m)] = Pp
                if cfg == "base":
                    cells.append(f"{mi:.3f}")
                else:
                    allp = np.ones(len(Pp), bool)
                    dm = st.miou(Pp[:, None], S["cls"], allp)[0] - st.miou(P[("base", m)][:, None], S["cls"], allp)[0]
                    lo, hi = st.boot({"grp": S["grp"], "cls": S["cls"]}, Pp, P[("base", m)], n=1000)
                    cells.append(f"{mi:.3f} ({dm:+.3f}; {lo:+.3f}…{hi:+.3f})")
            w(f"| {cfg} | " + " | ".join(cells) + " |")
        w("\nBy true object area, mean per-pair IoU at each map's best global τ (V3+R2 and P1+R2):")
        for lo, hi in ((0, .05), (.05, .2), (.2, .42), (.42, 1.01)):
            sel = (S["area"] >= lo) & (S["area"] < hi)
            if sel.sum() >= 5:
                row = []
                for cfg in CONFIGS:
                    v = []
                    for m in ("V3+R2", "P1+R2"):
                        X = S[f"{cfg}/{m}/raw"][sel]; iou = X[..., 0] / np.maximum(X[..., :3].sum(-1), 1)
                        v.append(iou.mean(0).max())
                    row.append(f"{cfg} {v[0]:.3f}/{v[1]:.3f}")
                w(f"- area {lo:.2f}–{hi:.2f} (n={sel.sum()}): " + ", ".join(row))
        w("")
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
