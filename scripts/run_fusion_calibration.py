#!/usr/bin/env python3
"""Fusion and calibration package (phase 3 registry Section 1.2) on one VOC-Sim class shard.

Fixed input: the frozen pair of protocol Section 11 (CA seed ``C/R``; SA module
15, all calls and heads). REF is the one-pass map ``y = mean_i(A_i c)``. Every
configuration below is built from the same live generation:

  CHECK/ca_seed     the CA seed alone
  REF/one_pass      y (min-max)
  P4/lam{25,50,75,90}  lam * y + (1 - lam) * c        (min-max inputs, min-max output)
  F3/gate           c * y
  F4/geomean        sqrt(c * y)
  N2/percentile     y, percentile 1-99 normalization
  N3/max            y / max(y)
  N4/otsu           y, per-image Otsu threshold (binary; same counts in every slot)
  R2/crf            OVAM's DenseCRF on the REF mask at each threshold of the grid

Counts are stored for the 17 thresholds 0.30-0.70 (step 0.025) of protocol
Section 37 / registry R3; ``scripts/select_propagation_cv.py`` chooses each
method parameter and threshold on the training folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_sa_package_b as sa_common  # noqa: E402
from scripts.run_streaming_voc_ablations import read_rows  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.fusion_variants import (  # noqa: E402
    gate,
    geometric_mean,
    max_normalize,
    minmax,
    otsu_threshold,
    residual_mix,
)
from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402

WORK = 64
SELECTED_SA_MODULE = 15
LAMBDAS = (0.25, 0.50, 0.75, 0.90)
THRESHOLDS = tuple(round(0.30 + 0.025 * i, 3) for i in range(17))

CONTINUOUS_MINMAX = (
    "CHECK/ca_seed", "REF/one_pass",
    *(f"P4/lam{int(round(lam * 100)):02d}" for lam in LAMBDAS),
    "F3/gate", "F4/geomean",
)
CONFIGURATIONS = (*CONTINUOUS_MINMAX, "N2/percentile", "N3/max", "N4/otsu", "R2/crf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_frozen.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"))
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/segmentation/controlled_attention_cv/fusion_calibration"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crf-site", type=Path, help="directory holding SimpleCRF's denseCRF extension")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-reference", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def one_pass(recorder, module_name: str, seed64: torch.Tensor, device: str) -> torch.Tensor:
    """REF: ``mean_i(A_i c)`` over every call and head of the frozen SA module."""
    qk_calls = recorder.sa_qk[module_name]
    seed = seed64.reshape(-1)
    total = torch.zeros_like(seed)
    for query, key in qk_calls:
        query, key = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1)
        total += torch.matmul(affinity, seed).mean(dim=0)
        del query, key, affinity
    return (total / len(qk_calls)).reshape(WORK, WORK)


def upsampled(map64: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize to the annotation, as in ``cv.score_maps``."""
    return F.interpolate(map64[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def score_sample(seed64: torch.Tensor, y: torch.Tensor, target: np.ndarray, image: np.ndarray, crf_site):
    c_n, y_n = minmax(seed64), minmax(y)
    continuous = [seed64, y, *(residual_mix(y_n, c_n, lam) for lam in LAMBDAS), gate(y_n, c_n), geometric_mean(y_n, c_n)]
    counts_a, ap_a = cv.score_maps(torch.stack(continuous), target, "minmax", thresholds=THRESHOLDS)
    counts_p, ap_p = cv.score_maps(y[None], target, "percentile_01_99", thresholds=THRESHOLDS)
    counts_m, ap_m = cv.score_maps(max_normalize(y)[None], target, "identity", thresholds=THRESHOLDS)

    shape = tuple(np.asarray(target).shape)
    y_full = upsampled(y_n, shape)
    otsu = (y_full >= otsu_threshold(y_n)).cpu().numpy()
    counts_o = cv.score_masks(otsu[None], target, thresholds=THRESHOLDS)

    counts_r = np.zeros((1, len(THRESHOLDS), 4), dtype=np.int64)
    for t_index, threshold in enumerate(THRESHOLDS):
        refined = apply_ovam_densecrf(image, (y_full >= threshold).cpu().numpy(), crf_site)
        counts_r[0, t_index] = cv.score_masks(refined[None], target, thresholds=(threshold,))[0, 0]

    counts = np.concatenate((counts_a, counts_p, counts_m, counts_o, counts_r))
    ap = np.concatenate((ap_a, ap_p, ap_m, ap_a[[CONFIGURATIONS.index("REF/one_pass")]], [np.nan]))
    return counts, ap


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    folds = cv.read_folds(args.folds)
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32,
                                                   local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    if len(ca_modules) != 16 or len(sa_modules) != 16:
        raise RuntimeError("Expected 16 CA and 16 SA modules")
    selected_module = sa_modules[SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    settings = {
        "package": "fusion_calibration_v1",
        "protocol": "v1.2; phase 3 registry Section 1.2; protocol Section 38",
        "condition": "frozen pair (CA C/R; SA module 15, all calls and heads); one non-residual pass; conditional QK",
        "ca_seed": seed_spec, "selected_sa_module_index": SELECTED_SA_MODULE, "selected_sa_module": selected_module,
        "lambdas": list(LAMBDAS), "thresholds": list(THRESHOLDS),
        "crf_parameters": "OVAM_CRF_PARAMETERS (src/segmentation/ovam_full.py), unary from the REF mask at each threshold",
        "otsu": "256 bins on the min-max REF map at 64x64; mask = bilinear REF map >= Otsu threshold",
        "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
        "manifest": str(args.manifest), "fold_file": str(args.folds), "configurations": list(CONFIGURATIONS),
    }
    shard_path = args.output_dir / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    writer = cv.ShardWriter(shard_path, CONFIGURATIONS, folds, settings, thresholds=THRESHOLDS)
    expected = len(rows)
    t45 = THRESHOLDS.index(0.45)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(pipe, row, args.manifest.parent, "float16",
                                                          cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_resolutions = {n: r for r, names in sample.modules_by_resolution.items() for n in names}
            sa_resolutions = {n: r for r, names in sample.self_modules_by_resolution.items() for n in names}
            if sa_resolutions[selected_module] != WORK:
                raise RuntimeError("Frozen SA module 15 is not 64x64")
            seed64 = sa_common.ca_seed(recorder, ca_modules, ca_resolutions, seed_spec, args.device)
            recorder.ca_calls.clear()
            y = one_pass(recorder, selected_module, seed64, args.device)
            if args.verify_reference:
                direct, _ = sa_common.propagate(recorder, sa_modules, sa_resolutions, seed64, args.device)
                difference = float((y - direct[SELECTED_SA_MODULE].mean(0)).abs().max())
                print(f"independent one-pass max abs difference={difference:.3e}", flush=True)
                if difference > 1e-4:
                    raise RuntimeError(f"Fusion/package-B reference mismatch: {difference}")
                del direct
            recorder.sa_qk.clear()
            counts, ap = score_sample(seed64, y, sample.target_mask, sample.generated_image, args.crf_site)
            writer.add(row["image_id"], classname, counts, ap)
            iou = {name: counts[i, t45, 0] / max(counts[i, t45, :3].sum(), 1) for i, name in enumerate(CONFIGURATIONS)}
            print(f"[{classname}] {row['image_id']} ({len(writer.completed)}/{expected}) "
                  + " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in iou.items() if k != "CHECK/ca_seed"), flush=True)
            del seed64, y, counts, ap
        finally:
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            sample.clear()
            del sample, recorder
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_samples is None:
        if len(writer.completed) != expected:
            raise RuntimeError(f"Shard has {len(writer.completed)} of {expected} expected images")
        writer.finish()
    print(f"Completed {len(writer.completed)} fusion/calibration samples for {classname}", flush=True)


if __name__ == "__main__":
    main()
