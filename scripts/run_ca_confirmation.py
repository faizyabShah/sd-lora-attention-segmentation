#!/usr/bin/env python3
"""CA-C: combined confirmation of the promoted CA changes (protocol Section 4, Section 25).

One VOC-Sim class shard per call; every image is regenerated live and released
after scoring. Uses the CA package B capture (``PackageBRecorder``: per-head CA
maps of the reference attribution, accumulated over all 31 calls) so the
arithmetic is identical to package B, and the Section 3 condition throughout.

Configurations (all calls 0-30; conditional query and key; padded softmax; all
class-word pieces):

  C/reference   all 16 modules, all 8 heads
  C/R           CA-R: without the 32x32 modules (2, 3, 10, 11, 12)
  C/L           CA-L: without modules 10, 11
  C/H           CA-H: without head 1
  C/R+H         combined candidate (CA-L is contained in CA-R)
  C/L+H         combined candidate with CA-R relaxed to CA-L (leave-CA-R-out)
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import gc
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing package B installs its recorder subclass into the live capture.
from scripts.run_ca_package_b import _RECORDERS, HEADS, WORK  # noqa: E402
import src.attention_capture.live as live  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

ALL_MODULES = tuple(range(16))
ALL_HEADS = tuple(range(HEADS))
RES32 = (2, 3, 10, 11, 12)
CA_L = (10, 11)
CA_H_DROP = (1,)
CONFIGURATIONS = {
    "C/reference": (ALL_MODULES, ALL_HEADS),
    "C/R": (tuple(m for m in ALL_MODULES if m not in RES32), ALL_HEADS),
    "C/L": (tuple(m for m in ALL_MODULES if m not in CA_L), ALL_HEADS),
    "C/H": (ALL_MODULES, tuple(h for h in ALL_HEADS if h not in CA_H_DROP)),
    "C/R+H": (tuple(m for m in ALL_MODULES if m not in RES32), tuple(h for h in ALL_HEADS if h not in CA_H_DROP)),
    "C/L+H": (tuple(m for m in ALL_MODULES if m not in CA_L), tuple(h for h in ALL_HEADS if h not in CA_H_DROP)),
}


@torch.inference_mode()
def per_head_maps(recorder, modules, resolution, device) -> torch.Tensor:
    """[16, 8, 64, 64]: call-averaged per-head CA maps, bilinear to 64x64 (Section 3)."""
    out = []
    for name in modules:
        r = resolution[name]
        heads = (recorder.ref_heads[name] / recorder.cross_calls[name]).reshape(HEADS, 1, r, r)
        if r != WORK:
            heads = F.interpolate(heads, size=(WORK, WORK), mode="bilinear", align_corners=False)
        out.append(heads[:, 0])
    return torch.stack(out).to(device)


def config_map(per_head: torch.Tensor, modules, heads) -> torch.Tensor:
    """Average selected heads within each module, then the selected modules with equal weight."""
    return per_head[list(modules)][:, list(heads)].mean(dim=1).mean(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"))
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation/controlled_attention_cv/ca_confirmation"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    classname = sorted({r["classname"] for r in rows})[args.class_shard]
    rows = [r for r in rows if r["classname"] == classname and r["annotated"].lower() == "true"][: args.max_images]
    folds = cv.read_folds(args.folds)
    (args.output_dir / "shards").mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    modules = tuple(sorted(n.removesuffix(".processor") for n in pipe.unet.attn_processors if n.endswith("attn2.processor")))
    if len(modules) != 16:
        raise RuntimeError("Expected 16 CA modules")
    names = list(CONFIGURATIONS)
    settings = {
        "package": "CA-C combined confirmation",
        "protocol": "v1.2 approved; Section 25",
        "condition": "Section 3: CA only, calls 0-30, conditional query and key, padded 77-token softmax, mean over "
                     "all class pieces, float16 per call, equal module-call weights, minmax at 64x64, bilinear to "
                     "annotation, >= threshold",
        "configurations": {k: {"modules": list(m), "heads": list(h)} for k, (m, h) in CONFIGURATIONS.items()},
        "modules": list(modules), "model": args.model, "model_dtype": "float32", "storage_dtype": "float16",
        "manifest": str(args.manifest), "folds": str(args.folds), "reference": "C/reference",
    }
    writer = cv.ShardWriter(args.output_dir / "shards" / f"class_{args.class_shard:02d}_{classname}.npz", names, folds, settings)
    print(f"{classname}: {len(rows)} annotated, {len(writer.completed)} done, {len(names)} configurations, "
          f"node={socket.gethostname()}, torch={torch.__version__}", flush=True)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        start = time.time()
        _RECORDERS.clear()
        sample = live.generate_attention_sample(pipe, row, args.manifest.parent, "float16", cross_map_kind="conditional")
        recorder = _RECORDERS[-1]
        try:
            sample_modules = tuple(sorted(n for ns in sample.modules_by_resolution.values() for n in ns))
            if sample_modules != modules or any(recorder.cross_calls[m] != len(sample.timesteps) for m in modules):
                raise RuntimeError(f"{row['image_id']}: expected all 16 CA modules on every call")
            resolution = {n: r for r, ns in sample.modules_by_resolution.items() for n in ns}
            per_head = per_head_maps(recorder, modules, resolution, args.device)
            maps = torch.stack([config_map(per_head, m, h) for m, h in CONFIGURATIONS.values()])
            counts, ap = cv.score_maps(maps, sample.target_mask, "minmax")
            writer.add(row["image_id"], classname, counts, ap)
            ious = {n: counts[i, cv.SELECTION_INDEX, 0] / max(counts[i, cv.SELECTION_INDEX, :3].sum(), 1) for i, n in enumerate(names)}
            print(f"{row['image_id']} " + " ".join(f"{n[2:]}={v:.3f}" for n, v in ious.items()) + f" ({time.time() - start:.0f}s)", flush=True)
        finally:
            sample.clear()
            recorder.ref_heads.clear()
            recorder.cells.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_images is None:
        writer.finish()


if __name__ == "__main__":
    main()
