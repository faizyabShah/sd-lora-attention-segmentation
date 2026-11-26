#!/usr/bin/env python3
"""Frozen BRACE (A3 seed -> V3 -> tau 0.55 -> DenseCRF R2) on the generated Cityscapes-style images, LoRA loaded,
one live generation per query. Prompted classes use the caption itself as attribution prompt (VOC-Sim convention);
unprompted classes use the class word as a separate attribution prompt (COCO-Cap convention). These masks are
BRACE PREDICTIONS (pre-annotations), not ground truth.

python scripts/cs_bench_brace.py run [--shard i --nshard n] [--limit k]   GPU -> outputs/cs_bench/brace/<id>__<cls>.npz
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "runs/11_cs_bench"))
import cs_bench_gen as g  # noqa: E402
from queries import queries  # noqa: E402

OUT = ROOT / "outputs/cs_bench/brace"; DUMMY = ROOT / "outputs/cs_bench/dummy_annotation.png"


def run(a):
    import torch
    from PIL import Image
    from scripts import sg27_runtime
    from src.segmentation import sg_combined_v2
    import idea1_knockout as k1, sg_steps_test as st, ana_ops_common as oc
    oc.crf_init(); OUT.mkdir(parents=True, exist_ok=True)
    if not DUMMY.exists():
        Image.fromarray(np.zeros((512, 512, 3), np.uint8)).save(DUMMY)
    Q = queries()[: a.limit or None][a.shard::a.nshard]
    runtime = sg27_runtime.make_runtime(k1.MODEL)
    runtime["pipe"].load_lora_weights(str(g.LORA), weight_name="pytorch_lora_weights.safetensors")
    for q in Q:
        o = OUT / f"{q['image_id']}__{q['cls']}.npz"
        if o.exists():
            continue
        t0 = time.perf_counter()
        row = dict(image_id=q["image_id"], prompt=q["prompt"], classname=q["word"], seed=str(q["seed"]),
                   attribution_prompt=q["prompt"] if q["kind"] == "prompted" else q["word"],
                   num_inferece_steps="30", annotation=str(DUMMY))
        with sg27_runtime.live_maps(runtime, row, "coco") as (sample, sa, seed, _b, _p):
            img = np.asarray(sample.generated_image, np.uint8).copy()
            s64 = seed.float().clone()
            v3 = sg_combined_v2.aggregate(sa, s64)[0]["V3"].float().reshape(64, 64).cpu().numpy()
            s64 = s64.reshape(64, 64).cpu().numpy()
        ref = np.asarray(Image.open(g.OUT / f"{q['image_id']}.png").convert("RGB"), np.float32)
        diff = float(np.abs(img.astype(np.float32) - ref).mean())
        mask = oc.crf(img, st.up(st.mm(v3), (512, 512)) >= 0.55)
        tmp = o.with_name(o.stem + ".tmp.npz")
        np.savez_compressed(tmp, v3=v3.astype(np.float32), seed=s64.astype(np.float32), mask=mask, img_meandiff=diff,
                            kind=q["kind"], word=q["word"], cls=q["cls"])
        tmp.replace(o)
        print(f"[csb] {q['image_id']} {q['cls']:<10} {q['kind']:<10} area {mask.mean():.3f} |dpix| {diff:.2f} {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--shard", type=int, default=0); p.add_argument("--nshard", type=int, default=1); p.add_argument("--limit", type=int)
    a = ap.parse_args(); run(a)
