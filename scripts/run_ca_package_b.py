#!/usr/bin/env python3
"""CA package B of the controlled protocol: CA-L (modules), CA-H (heads), CA-A (attribution).

Protocol v1.2, Sections 3 and 4, owner Claude. One VOC-Sim class shard per call.
Each image is regenerated live; per-head cross-attention maps are accumulated
over all 31 UNet calls during generation and released after scoring.

Fixed condition (Section 3): CA only; calls 0-30; every module-call observation
weighted equally; min-max on the 64x64 map; bilinear resize to the annotation;
foreground at >= 0.45 (0.35 and 0.55 and 64x64 AP recorded as diagnostics);
no SA, DenseCRF, cleanup, learned weights or token optimization.

Reference attribution (Section 2): conditional query and key, softmax over all
77 padded positions, mean over every class-name piece, rounded to float16 per
call as in the shared capture (``memory.py`` ``target_maps``).

Families written to one shard per class (configuration names):

  L/...  CA-L: all_16 (reference), non_middle_15, down_only, middle_only,
         up_only, down_middle, middle_up, single_XX, leave1_XX, leave2_XX_YY
  H/...  CA-H: all 255 nonempty head subsets, same subset in every module
         (H/0_1_2_3_4_5_6_7 is the reference)
  A/...  CA-A: query {cond, uncond} x softmax {padded, unpadded} x pieces
         {all, first, last}; A/cond_padded_all is the reference

The exploratory exhaustive layer search (65,535 module subsets, threshold 0.45
only) is written per image to ``exhaustive_raw/`` and is not committed.
Module maps of fixed figure images are written to ``figure_maps/``.

The capture is a recorder subclass defined here; shared capture code is not
modified, and self-attention tensors are not stored (CA-only package).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import gc
import itertools
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.attention_capture.live as live  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402
from src.attention_capture.memory import CompactAttentionRecorder  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

HEADS = 8
WORK = 64
CELLS = tuple(
    f"{query}_{domain}_{pieces}"
    for query in ("cond", "uncond")
    for domain in ("padded", "unpadded")
    for pieces in ("all", "first", "last")
)
REFERENCE = {"L": "L/all_16", "H": "H/" + "_".join(map(str, range(HEADS))), "A": "A/cond_padded_all"}
# Images whose per-module maps are kept for the Section 4 figures; fixed before any result.
FIGURE_IMAGES = ("ovam-voc-sim-0", "ovam-voc-sim-125", "ovam-voc-sim-250", "ovam-voc-sim-350")


class PackageBRecorder(CompactAttentionRecorder):
    """Accumulates call-summed CA maps on the device; stores nothing per step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_heads: dict[str, torch.Tensor] = {}  # module -> [8, HW] sum over calls
        self.cells: dict[str, torch.Tensor] = {}  # module -> [12, HW] head-mean sum over calls
        self.cross_calls: dict[str, int] = {}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        del value, attribution_key
        step = self.call_counts[module_name]
        self.call_counts[module_name] += 1
        self.step_tensors.setdefault(step, {})
        if kind == "cross":
            self._record_cross(module_name, query, key, attention_mask, scale)
        spatial = math.isqrt(int(query.shape[-2]))
        self.module_metadata[module_name].update(
            {"spatial_height": spatial, "spatial_width": spatial,
             "spatial_tokens": int(query.shape[-2]), "head_dim": int(query.shape[-1])}
        )

    @torch.inference_mode()
    def _record_cross(self, module_name, query, key, attention_mask, scale):
        if query.shape[0] != 2:
            raise RuntimeError("Expected a [unconditional, conditional] CFG batch")
        positions = list(self.target_positions)
        first, last = positions[:1], positions[-1:]
        k_cond = key[1].float()
        cells = []
        ref = None
        for q_index, q_name in ((1, "cond"), (0, "uncond")):
            q = query[q_index].float()
            scores = torch.matmul(q, k_cond.transpose(-1, -2)) * scale  # [heads, HW, 77]
            if attention_mask is not None:
                scores = scores + attention_mask[1].float()
            for domain in ("padded", "unpadded"):
                probs = (scores if domain == "padded" else scores[..., : self.prompt_token_count]).softmax(dim=-1)
                for pieces, selected in (("all", positions), ("first", first), ("last", last)):
                    per_head = probs[..., selected].mean(dim=-1).half().float()  # float16 storage, as memory.py
                    if q_name == "cond" and domain == "padded" and pieces == "all":
                        ref = per_head
                    cells.append(per_head.mean(dim=0))
        stacked = torch.stack(cells)  # order matches CELLS
        if module_name not in self.ref_heads:
            self.ref_heads[module_name] = torch.zeros_like(ref)
            self.cells[module_name] = torch.zeros_like(stacked)
            self.cross_calls[module_name] = 0
        self.ref_heads[module_name] += ref
        self.cells[module_name] += stacked
        self.cross_calls[module_name] += 1


live.CompactAttentionRecorder = PackageBRecorder
_RECORDERS: list[PackageBRecorder] = []
_original_init = PackageBRecorder.__init__


def _tracking_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    _RECORDERS.append(self)


PackageBRecorder.__init__ = _tracking_init


def layer_configurations(modules: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    index = {name: i for i, name in enumerate(modules)}
    down = tuple(index[m] for m in modules if m.startswith("down_blocks."))
    middle = tuple(index[m] for m in modules if m.startswith("mid_block."))
    up = tuple(index[m] for m in modules if m.startswith("up_blocks."))
    every = tuple(range(len(modules)))
    configs = {
        "L/all_16": every,
        "L/non_middle_15": down + up,
        "L/down_only": down,
        "L/middle_only": middle,
        "L/up_only": up,
        "L/down_middle": down + middle,
        "L/middle_up": middle + up,
    }
    configs.update({f"L/single_{i:02d}": (i,) for i in every})
    configs.update({f"L/leave1_{i:02d}": tuple(j for j in every if j != i) for i in every})
    configs.update({
        f"L/leave2_{a:02d}_{b:02d}": tuple(j for j in every if j not in (a, b))
        for a, b in itertools.combinations(every, 2)
    })
    return configs


def head_configurations() -> dict[str, tuple[int, ...]]:
    return {
        "H/" + "_".join(map(str, subset)): subset
        for width in range(1, HEADS + 1)
        for subset in itertools.combinations(range(HEADS), width)
    }


def configuration_names(modules: tuple[str, ...]) -> tuple[list[str], dict]:
    layers = layer_configurations(modules)
    heads = head_configurations()
    names = list(layers) + list(heads) + [f"A/{c}" for c in CELLS]
    inclusiveness = {**{k: len(v) for k, v in layers.items()}, **{k: len(v) for k, v in heads.items()}}
    return names, {"layers": layers, "heads": heads, "inclusiveness": inclusiveness}


@torch.inference_mode()
def build_maps(recorder: PackageBRecorder, modules, resolution, families, device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return candidate maps [C, 64, 64] in configuration order, plus per-module maps."""
    per_head = []  # [16, 8, 64, 64]
    cell_maps = []  # [16, 12, 64, 64]
    for name in modules:
        calls = recorder.cross_calls[name]
        r = resolution[name]
        heads = (recorder.ref_heads[name] / calls).reshape(HEADS, 1, r, r)
        cells = (recorder.cells[name] / calls).reshape(len(CELLS), 1, r, r)
        if r != WORK:
            heads = F.interpolate(heads, size=(WORK, WORK), mode="bilinear", align_corners=False)
            cells = F.interpolate(cells, size=(WORK, WORK), mode="bilinear", align_corners=False)
        per_head.append(heads[:, 0])
        cell_maps.append(cells[:, 0])
    per_head = torch.stack(per_head).to(device)
    cell_maps = torch.stack(cell_maps).to(device)
    module_maps = per_head.mean(dim=1)  # head-mean per module, [16, 64, 64]
    head_maps = per_head.mean(dim=0)  # module-mean per head, [8, 64, 64]
    maps = [module_maps[list(idx)].mean(dim=0) for idx in families["layers"].values()]
    maps += [head_maps[list(idx)].mean(dim=0) for idx in families["heads"].values()]
    maps += list(cell_maps.mean(dim=0))
    checks = {
        "max_abs_diff_L_vs_H_reference": float((module_maps.mean(0) - head_maps.mean(0)).abs().max()),
        "max_abs_diff_L_vs_A_reference": float((module_maps.mean(0) - cell_maps[:, 0].mean(0)).abs().max()),
    }
    return torch.stack(maps), module_maps, checks


@torch.inference_mode()
def exhaustive_counts(module_maps: torch.Tensor, target: np.ndarray, chunk: int = 512) -> np.ndarray:
    """tp, fp, fn, tn at 0.45 for all 65,535 nonempty module subsets (mean then min-max)."""
    device = module_maps.device
    count = module_maps.shape[0]
    masks = torch.arange(1, 1 << count, device=device)
    bits = ((masks[:, None] >> torch.arange(count, device=device)[None]) & 1).float()
    layers = module_maps.reshape(count, -1)
    truth = torch.as_tensor(target, device=device).bool()
    truth_pixels, total = int(truth.sum()), truth.numel()
    out = np.zeros((bits.shape[0], 4), dtype=np.int64)
    for start in range(0, bits.shape[0], chunk):
        block = bits[start : start + chunk] @ layers
        block = cv.normalize_batch(block, "minmax").reshape(-1, 1, WORK, WORK)
        pred = F.interpolate(block, size=truth.shape, mode="bilinear", align_corners=False)[:, 0] >= cv.SELECTION_THRESHOLD
        tp = (pred & truth).sum(dim=(1, 2))
        fp = pred.sum(dim=(1, 2)) - tp
        fn = truth_pixels - tp
        out[start : start + chunk] = torch.stack((tp, fp, fn, total - tp - fp - fn), 1).cpu().numpy()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"),
    )
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--skip-exhaustive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    classname = sorted({r["classname"] for r in rows})[args.class_shard]
    rows = [r for r in rows if r["classname"] == classname and r["annotated"].lower() == "true"][: args.max_images]
    folds = cv.read_folds(args.folds)
    stem = f"class_{args.class_shard:02d}_{classname}"
    shards = args.output_dir / "shards"
    exhaustive_dir = args.output_dir / "exhaustive_raw"
    figure_dir = args.output_dir / "figure_maps"
    for directory in (shards, exhaustive_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    modules = tuple(sorted(
        name.removesuffix(".processor") for name in pipe.unet.attn_processors if name.endswith("attn2.processor")
    ))
    if len(modules) != 16:
        raise RuntimeError(f"Expected 16 CA modules, found {len(modules)}")
    names, families = configuration_names(modules)
    settings = {
        "package": "CA package B (CA-L, CA-H, CA-A)",
        "protocol": "v1.2 approved",
        "condition": "Section 3: CA only, calls 0-30, equal module-call weights, minmax at 64x64, "
                     "bilinear to annotation, >= threshold",
        "reference_attribution": "conditional query and key, softmax over 77 padded positions, "
                                 "mean over all class pieces, float16 per call",
        "model": args.model, "model_dtype": "float32", "storage_dtype": "float16",
        "manifest": str(args.manifest), "folds": str(args.folds),
        "modules": list(modules), "cells": list(CELLS), "reference": REFERENCE,
    }
    writer = cv.ShardWriter(shards / f"{stem}.npz", names, folds, settings)
    (args.output_dir / "configurations.json").write_text(json.dumps(
        {"names": names, "inclusiveness": families["inclusiveness"], "modules": list(modules),
         "layers": {k: list(v) for k, v in families["layers"].items()}}, indent=1))
    checks_path = args.output_dir / "checks" / f"{stem}.jsonl"
    checks_path.parent.mkdir(parents=True, exist_ok=True)
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
            sample_modules = tuple(sorted(n for names_ in sample.modules_by_resolution.values() for n in names_))
            if sample_modules != modules or any(recorder.cross_calls[m] != len(sample.timesteps) for m in modules):
                raise RuntimeError(f"{row['image_id']}: expected 16 CA modules x {len(sample.timesteps)} calls")
            if recorder.target_positions[0] != recorder.ovam_target_position:
                raise RuntimeError(f"{row['image_id']}: first class piece is not OVAM's target position")
            resolution = {n: r for r, names_ in sample.modules_by_resolution.items() for n in names_}
            maps, module_maps, checks = build_maps(recorder, modules, resolution, families, args.device)
            counts, ap = cv.score_maps(maps, sample.target_mask, "minmax")
            if not args.skip_exhaustive:
                np.save(exhaustive_dir / f"{row['image_id']}.npy",
                        exhaustive_counts(module_maps, sample.target_mask).astype(np.int32))
            if row["image_id"] in FIGURE_IMAGES:
                np.savez_compressed(figure_dir / f"{row['image_id']}.npz", modules=np.array(modules),
                                    module_maps=module_maps.cpu().numpy().astype(np.float16),
                                    target=sample.target_mask, image=sample.generated_image)
            writer.add(row["image_id"], classname, counts, ap)
            with checks_path.open("a") as handle:
                handle.write(json.dumps({"image_id": row["image_id"], "calls": len(sample.timesteps),
                                         "pieces": len(recorder.target_positions), **checks}) + "\n")
            ref = names.index(REFERENCE["L"])
            iou = counts[ref, cv.SELECTION_INDEX]
            print(f"{row['image_id']} ref IoU={iou[0] / max(iou[:3].sum(), 1):.3f} "
                  f"checks={checks['max_abs_diff_L_vs_H_reference']:.1e}/{checks['max_abs_diff_L_vs_A_reference']:.1e} "
                  f"({time.time() - start:.0f}s)", flush=True)
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
