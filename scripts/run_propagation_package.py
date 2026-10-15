#!/usr/bin/env python3
"""Run controlled propagation variants on one VOC-Sim class shard.

The frozen CA seed and frozen SA module come from protocol Sections 35-36.
One image is generated at a time; only compact per-image metric counts remain.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_sa_package_b as sa_common  # noqa: E402
from scripts.run_streaming_voc_ablations import read_rows  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.propagation_variants import (  # noqa: E402
    power_trajectory,
    restart_trajectory,
    sparse_one_pass,
)

WORK = 64
SELECTED_SA_MODULE = 15
POWER_STEPS = (1, 2, 3, 4)
RESTART_BETAS = (0.10, 0.25, 0.50, 0.75)
RESTART_STEPS = (2, 4, 8)
SPARSE_K = (16, 32, 64, 128, 256, 512, 1024)
THRESHOLDS = tuple(round(0.30 + 0.025 * i, 3) for i in range(17))


def beta_name(beta: float) -> str:
    return f"b{int(round(beta * 100)):02d}"


CONFIGURATIONS = (
    "CHECK/ca_seed",
    "REF/one_pass",
    *(f"P1/obs_n{n}" for n in POWER_STEPS[1:]),
    *(f"W3/mean_n{n}" for n in POWER_STEPS),
    *(
        f"P2/obs_{beta_name(beta)}_n{steps}"
        for beta in RESTART_BETAS
        for steps in RESTART_STEPS
    ),
    *(
        f"P2/mean_{beta_name(beta)}_n{steps}"
        for beta in RESTART_BETAS
        for steps in RESTART_STEPS
    ),
    *(f"P6/topk_{k:04d}" for k in SPARSE_K),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument(
        "--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_frozen.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"
        ),
    )
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/controlled_attention_cv/propagation_package"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-reference", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def propagation_maps(recorder, module_name: str, seed64: torch.Tensor, device: str):
    """Construct all package maps from the same frozen module observations."""
    qk_calls = recorder.sa_qk[module_name]
    calls = len(qk_calls)
    pixels = seed64.numel()
    heads = qk_calls[0][0].shape[0]
    seed = seed64.reshape(-1)

    obs_power = torch.zeros((max(POWER_STEPS), pixels), device=device)
    obs_restart = torch.zeros(
        (max(RESTART_STEPS), len(RESTART_BETAS), pixels), device=device
    )
    obs_sparse = torch.zeros((len(SPARSE_K), pixels), device=device)
    affinity_sum = torch.zeros((pixels, pixels), device=device)

    for query, key in qk_calls:
        query, key = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]),
            dim=-1,
        )
        affinity_sum += affinity.sum(dim=0)
        obs_power += power_trajectory(affinity, seed, max(POWER_STEPS)).mean(dim=1)
        obs_restart += restart_trajectory(
            affinity, seed, RESTART_BETAS, max(RESTART_STEPS)
        ).mean(dim=1)
        obs_sparse += sparse_one_pass(affinity, seed, SPARSE_K).mean(dim=1)
        del query, key, affinity

    obs_power /= calls
    obs_restart /= calls
    obs_sparse /= calls
    mean_affinity = affinity_sum / (calls * heads)
    mean_power = power_trajectory(mean_affinity, seed, max(POWER_STEPS))
    mean_restart = restart_trajectory(
        mean_affinity, seed, RESTART_BETAS, max(RESTART_STEPS)
    )

    reference_difference = float((obs_power[0] - mean_power[0]).abs().max())
    if reference_difference > 2e-5:
        raise RuntimeError(
            f"mean(A_i c) and mean(A_i)c differ by {reference_difference:.3e} at n=1"
        )

    maps = [seed64, obs_power[0].reshape(WORK, WORK)]
    maps += [obs_power[n - 1].reshape(WORK, WORK) for n in POWER_STEPS[1:]]
    maps += [mean_power[n - 1].reshape(WORK, WORK) for n in POWER_STEPS]
    maps += [
        obs_restart[steps - 1, beta_index].reshape(WORK, WORK)
        for beta_index, _ in enumerate(RESTART_BETAS)
        for steps in RESTART_STEPS
    ]
    maps += [
        mean_restart[steps - 1, beta_index].reshape(WORK, WORK)
        for beta_index, _ in enumerate(RESTART_BETAS)
        for steps in RESTART_STEPS
    ]
    maps += [obs_sparse[index].reshape(WORK, WORK) for index in range(len(SPARSE_K))]
    if len(maps) != len(CONFIGURATIONS):
        raise RuntimeError(f"Built {len(maps)} maps for {len(CONFIGURATIONS)} names")
    return torch.stack(maps), reference_difference


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    folds = cv.read_folds(args.folds)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    if len(ca_modules) != 16 or len(sa_modules) != 16:
        raise RuntimeError("Expected 16 CA and 16 SA modules")
    selected_module = sa_modules[SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    settings = {
        "package": "controlled_propagation_package_v1",
        "protocol": "v1.2; deferred phase Section 37",
        "condition": "jointly confirmed CA seed; SA module 15, all calls and heads; conditional QK; no postprocessing; minmax once",
        "ca_seed": seed_spec,
        "selected_sa_module_index": SELECTED_SA_MODULE,
        "selected_sa_module": selected_module,
        "power_steps": list(POWER_STEPS),
        "restart_betas": list(RESTART_BETAS),
        "restart_steps": list(RESTART_STEPS),
        "sparse_topk": list(SPARSE_K),
        "thresholds": list(THRESHOLDS),
        "model": args.model,
        "model_dtype": "float32",
        "qk_storage_dtype": "float16",
        "manifest": str(args.manifest),
        "fold_file": str(args.folds),
        "configurations": list(CONFIGURATIONS),
    }
    shard_path = args.output_dir / "shards" / (
        f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    )
    writer = cv.ShardWriter(
        shard_path, CONFIGURATIONS, folds, settings, thresholds=THRESHOLDS
    )
    expected = len(rows)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        print(
            f"[{classname}] {row['image_id']} ({len(writer.completed)+1}/{expected})",
            flush=True,
        )
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(
            pipe, row, args.manifest.parent, "float16", cross_map_kind="conditional"
        )
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_resolutions = {
                name: resolution
                for resolution, names in sample.modules_by_resolution.items()
                for name in names
            }
            sa_resolutions = {
                name: resolution
                for resolution, names in sample.self_modules_by_resolution.items()
                for name in names
            }
            if sa_resolutions[selected_module] != WORK:
                raise RuntimeError("Frozen SA module 15 is not 64x64")
            seed64 = sa_common.ca_seed(
                recorder, ca_modules, ca_resolutions, seed_spec, args.device
            )
            recorder.ca_calls.clear()
            maps, arithmetic_difference = propagation_maps(
                recorder, selected_module, seed64, args.device
            )
            if args.verify_reference:
                direct, _ = sa_common.propagate(
                    recorder, sa_modules, sa_resolutions, seed64, args.device
                )
                difference = float(
                    (maps[CONFIGURATIONS.index("REF/one_pass")] - direct[SELECTED_SA_MODULE].mean(0))
                    .abs()
                    .max()
                )
                print(f"independent one-pass max abs difference={difference:.3e}", flush=True)
                if difference > 1e-4:
                    raise RuntimeError(f"Propagation/package-B reference mismatch: {difference}")
                del direct
            recorder.sa_qk.clear()
            counts, ap = cv.score_maps(
                maps, sample.target_mask, "minmax", thresholds=THRESHOLDS
            )
            writer.add(row["image_id"], classname, counts, ap)
            ref_index = CONFIGURATIONS.index("REF/one_pass")
            threshold_index = THRESHOLDS.index(0.45)
            ref = counts[ref_index, threshold_index]
            print(
                f"{row['image_id']} ref IoU={ref[0] / max(ref[:3].sum(), 1):.3f} "
                f"mean-order-check={arithmetic_difference:.2e}",
                flush=True,
            )
            del seed64, maps, counts, ap
        finally:
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            sample.clear()
            del sample, recorder
            gc.collect()
            torch.cuda.empty_cache()
    if len(writer.completed) != expected:
        raise RuntimeError(f"Shard has {len(writer.completed)} of {expected} expected images")
    writer.finish()
    print(f"Completed {expected} propagation samples for {classname}", flush=True)


if __name__ == "__main__":
    main()
