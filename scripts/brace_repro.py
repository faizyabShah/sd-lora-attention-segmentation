#!/usr/bin/env python3
"""Reproduce frozen BRACE on the generated sets (VOC-Sim 490, COCO-Cap 707) on a new machine.

BRACE = A3 cross-attention seed -> capped 16->32->64 self-attention hierarchy (V3, one-sided cap at 16->32)
        -> min-max, bilinear up, tau 0.55 -> DenseCRF R2.  Paper: VOC-Sim 0.830, COCO-Cap 0.696 (class-macro mIoU).

check    environment, CRF, model weights, datasets (no GPU work)
capture  regenerate each image with the frozen live pipeline and score BRACE (GPU; run shards in parallel)
report   class-macro mIoU, comparison with the cluster reference (runs/00_setup_reproduce/reference_cluster.json)

Cache (small per-image maps, not committed): $SG_CACHE (default ~/sg_cache)/brace_{voc,coco}/<id>.npz
"""
from __future__ import annotations

import argparse, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import sg_steps_test as st  # noqa: E402
import sg_class_comp as cc  # noqa: E402

CACHE = Path(os.environ.get("SG_CACHE", Path.home() / "sg_cache"))
RUN = ROOT / "runs/00_setup_reproduce"
TAU = 0.55
NAME = {"voc": "VOC-Sim", "coco": "COCO-Cap"}
PAPER = {"voc": 0.830, "coco": 0.696}
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def check(a):
    ok = True
    def say(flag, msg):
        nonlocal ok; ok &= flag; print(("OK   " if flag else "FAIL ") + msg)
    import diffusers, transformers
    say(sys.version_info[:2] == (3, 10), f"python {platform.python_version()} (expected 3.10)")
    say(torch.__version__.startswith("2.6.0"), f"torch {torch.__version__} (expected 2.6.0)")
    say(diffusers.__version__ == "0.36.0", f"diffusers {diffusers.__version__} (expected 0.36.0)")
    say(transformers.__version__ == "4.44.2", f"transformers {transformers.__version__} (expected 4.44.2)")
    say(torch.cuda.is_available(), f"cuda available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    try:
        from src.segmentation.ovam_full import _import_densecrf
        _import_densecrf(st.CRF_SITE); say(True, "DenseCRF (SimpleCRF denseCRF) importable")
    except Exception as e:  # noqa: BLE001
        say(False, f"DenseCRF import: {e}")
    for ds in ("voc", "coco"):
        try:
            rows = cc.gen_rows(ds)
            miss = [r for r in rows if not (Path(st.VS if ds == "voc" else st.VS.parent / "coco_captions") / r["annotation"]).exists()]
            say(len(rows) == {"voc": 490, "coco": 707}[ds] and not miss, f"{NAME[ds]}: {len(rows)} annotated rows, {len(miss)} missing masks")
        except Exception as e:  # noqa: BLE001
            say(False, f"{NAME[ds]} dataset: {e}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, local_files_only=True); say(True, f"{MODEL} in HF cache")
    except Exception as e:  # noqa: BLE001
        say(False, f"{MODEL} not in HF cache ({type(e).__name__})")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


def capture(a):
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf
    _import_densecrf(st.CRF_SITE)
    rows = cc.gen_rows(a.dataset)[a.shard::a.num_shards]
    if a.max_images:
        rows = rows[: a.max_images]
    runtime = sg27_runtime.make_runtime(MODEL)
    out = CACHE / f"brace_{a.dataset}"; out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        iid, cls = row["image_id"], row["classname"]; o = out / f"{iid}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        with sg27_runtime.live_maps(runtime, dict(row), a.dataset) as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy(); truth = np.asarray(sample.target_mask, bool)
            seed = seed.float().clone()
            v3 = sg_combined_v2.aggregate(sa, seed)[0]["V3"].float().cpu().numpy()
        x = st.up(st.mm(v3), truth.shape)
        mask = apply_ovam_densecrf(img, x >= TAU, st.CRF_SITE).astype(bool)
        valid = np.ones_like(truth); other = np.zeros_like(truth)
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez(tmp, seed=seed.cpu().numpy().astype(np.float16), v3=v3.astype(np.float32), cls=cls,
                 brace=cc.counts(mask[None], truth, valid, other)[0],
                 v3raw=cc.counts(x[None] >= cc.THR[:, None, None], truth, valid, other),
                 iou=float((mask & truth).sum() / max((mask | truth).sum(), 1)), sec=time.perf_counter() - t0)
        os.replace(tmp, o)
        print(f"[brace:{a.dataset}] {iid} {cls} iou {float((mask & truth).sum() / max((mask | truth).sum(), 1)):.3f} "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        torch.cuda.empty_cache()


def report(a):
    ref = json.loads((RUN / "reference_cluster.json").read_text())
    L = ["# Reproduction of frozen BRACE on this machine\n",
         f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}; torch {torch.__version__}; "
         f"python {platform.python_version()}.\n",
         "| set | images | BRACE mIoU here | paper | cluster reference | mean per-image |ΔIoU| vs cluster | images with |ΔIoU| > 0.05 | s/image |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    res = {}
    for ds in ("voc", "coco"):
        fs = sorted(f for f in (CACHE / f"brace_{ds}").glob("*.npz") if not f.stem.endswith(".tmp"))
        if not fs:
            continue
        Z = [np.load(f) for f in fs]; ids = [f.stem for f in fs]
        cls = np.array([str(z["cls"]) for z in Z]); C = np.stack([z["brace"] for z in Z])
        m = cc.miou(C[:, None, :], cls, np.ones(len(Z), bool))[0]
        r = ref[ds]; d = np.array([abs(float(z["iou"]) - r["iou"][i]) for z, i in zip(Z, ids) if i in r["iou"]])
        L.append(f"| {NAME[ds]} | {len(Z)} | {m:.3f} | {PAPER[ds]:.3f} | {r['miou']:.3f} | {d.mean():.4f} | {(d > 0.05).sum()} | "
                 f"{np.mean([float(z['sec']) for z in Z]):.1f} |")
        res[ds] = {"n": len(Z), "miou": float(m), "mean_abs_diff": float(d.mean()), "n_diff_gt_0.05": int((d > 0.05).sum())}
    ok = all(abs(res[d]["miou"] - PAPER[d]) <= 0.005 for d in res) and len(res) == 2 and \
        res["voc"]["n"] == 490 and res["coco"]["n"] == 707
    L.append(f"\n**Sanity check (|mIoU − paper| ≤ 0.005 on both full sets): {'PASS' if ok else 'FAIL'}**\n")
    txt = "\n".join(L) + "\n"; print(txt)
    (RUN / "result.md").write_text(txt); (RUN / "result.json").write_text(json.dumps(res, indent=1))


def make_reference(a):
    """Run on the original cluster only: per-image BRACE IoU from the stored paper capture."""
    j = int(np.flatnonzero(np.isclose(cc.THR_CRF, TAU))[0]); out = {}
    for ds, d in (("voc", "/home/shafa/sg_comp/vocsim_scores"), ("coco", "/home/shafa/sg_comp/cococap_scores")):
        S = cc.load_gen(Path(d)); C = S["base|V3+R2|raw"][:, j]
        out[ds] = {"miou": float(cc.miou(C[:, None], S["cls"], np.ones(len(C), bool))[0]),
                   "iou": {g: float(c[0] / max(c[:3].sum(), 1)) for g, c in zip(S["grp"], C)}}
    RUN.mkdir(parents=True, exist_ok=True); (RUN / "reference_cluster.json").write_text(json.dumps(out))
    print({k: (v["miou"], len(v["iou"])) for k, v in out.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check"); sub.add_parser("report"); sub.add_parser("make-reference")
    p = sub.add_parser("capture"); p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-images", type=int)
    a = ap.parse_args()
    {"check": check, "capture": capture, "report": report, "make-reference": make_reference}[a.cmd](a)
