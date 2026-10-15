#!/usr/bin/env python3
"""Run controlled SA time/resolution screens on one VOC-Sim class shard."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_sa_package_b as sa_common  # noqa: E402
from scripts.run_streaming_voc_ablations import read_rows  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.controlled_ca import (  # noqa: E402
    configuration_names,
    resolution_configurations,
    time_windows,
)
from src.segmentation.controlled_sa import compose_time_resolution_maps  # noqa: E402

RESOLUTIONS = (8, 16, 32, 64)
WORK = 64


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
        default=Path("outputs/segmentation/controlled_attention_cv/sa_time_resolution"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help="recompute the inclusive SA reference through package B and compare",
    )
    return parser.parse_args()


@torch.inference_mode()
def propagate_call_resolution_sums(
    recorder: sa_common.SAPackageRecorder,
    sa_modules: tuple[str, ...],
    module_resolutions: dict[str, int],
    seed64: torch.Tensor,
    calls: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate once and retain compact sums indexed by call and resolution."""
    sums = torch.zeros((calls, len(RESOLUTIONS), WORK, WORK), device=device)
    counts = torch.zeros((calls, len(RESOLUTIONS)), dtype=torch.int64, device=device)
    resolution_index = {value: index for index, value in enumerate(RESOLUTIONS)}
    for name in sa_modules:
        resolution = module_resolutions[name]
        index = resolution_index[resolution]
        seed = (
            F.adaptive_avg_pool2d(seed64[None, None], (resolution, resolution))[0, 0]
            if resolution != WORK
            else seed64
        ).reshape(-1)
        qk_calls = recorder.sa_qk[name]
        if len(qk_calls) != calls:
            raise RuntimeError(f"{name}: expected {calls} calls, found {len(qk_calls)}")
        for call, (query, key) in enumerate(qk_calls):
            query = query.to(device).float()
            key = key.to(device).float()
            affinity = torch.softmax(
                torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]),
                dim=-1,
            )
            propagated = torch.matmul(affinity, seed).mean(dim=0).reshape(
                1, 1, resolution, resolution
            )
            if resolution != WORK:
                propagated = F.interpolate(
                    propagated, size=(WORK, WORK), mode="bilinear", align_corners=False
                )
            sums[call, index] += propagated[0, 0]
            counts[call, index] += 1
            del query, key, affinity, propagated
    return sums, counts


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_spec_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_spec_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")

    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    folds = cv.read_folds(args.folds)
    windows = time_windows()
    resolution_sets = resolution_configurations()
    names = configuration_names(windows, resolution_sets)
    settings_base = {
        "package": "controlled_sa_time_resolution_v1",
        "protocol": "v1.2",
        "condition": "Section 6: frozen CA seed; conditional SA; one non-residual pass y=A x; area downsample seed; equal observation weights; minmax once; threshold 0.45",
        "manifest": str(args.manifest),
        "fold_file": str(args.folds),
        "model": args.model,
        "model_dtype": "float32",
        "qk_storage_dtype": "float16",
        "ca_seed_file": str(args.ca_seed),
        "time_windows": {name: list(values) for name, values in windows.items()},
        "resolution_sets": {name: list(values) for name, values in resolution_sets.items()},
        "thresholds": list(cv.THRESHOLDS),
        "selection_threshold": cv.SELECTION_THRESHOLD,
        "postprocessing": "none",
    }

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    if len(ca_modules) != 16 or len(sa_modules) != 16:
        raise RuntimeError("Expected 16 CA and 16 SA modules")
    seed_spec = sa_common.resolve_seed_spec(seed_spec_json, ca_modules, 31)
    settings = {
        **settings_base,
        "ca_seed": seed_spec,
        "ca_modules": list(ca_modules),
        "sa_modules": list(sa_modules),
    }
    shard_path = args.output_dir / "shards" / (
        f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    )
    writer = cv.ShardWriter(shard_path, names, folds, settings)

    expected = len(rows)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        print(f"[{classname}] {row['image_id']} ({len(writer.completed)+1}/{expected})", flush=True)
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(
            pipe, row, args.manifest.parent, "float16", cross_map_kind="conditional"
        )
        recorder = sa_common._RECORDERS[-1]
        try:
            calls = len(sample.timesteps)
            ca_resolutions = {
                name: resolution
                for resolution, module_names in sample.modules_by_resolution.items()
                for name in module_names
            }
            sa_resolutions = {
                name: resolution
                for resolution, module_names in sample.self_modules_by_resolution.items()
                for name in module_names
            }
            seed64 = sa_common.ca_seed(
                recorder, ca_modules, ca_resolutions, seed_spec, args.device
            )
            recorder.ca_calls.clear()
            sums, counts = propagate_call_resolution_sums(
                recorder, sa_modules, sa_resolutions, seed64, calls, args.device
            )
            observed_names, maps = compose_time_resolution_maps(
                sums, counts, RESOLUTIONS, windows, resolution_sets
            )
            if observed_names != names:
                raise RuntimeError("Configuration order differs from the result writer")
            if args.verify_reference:
                reference, _ = sa_common.propagate(
                    recorder, sa_modules, sa_resolutions, seed64, args.device
                )
                difference = float(
                    (maps[names.index("time/all_00_30")] - reference.mean(dim=(0, 1)))
                    .abs()
                    .max()
                )
                print(f"inclusive reference max abs difference={difference:.3e}", flush=True)
                if difference > 1e-4:
                    raise RuntimeError(f"SA package A/B reference mismatch: {difference}")
                del reference
            recorder.sa_qk.clear()
            counts_scored, ap = cv.score_maps(maps, sample.target_mask, "minmax")
            writer.add(row["image_id"], classname, counts_scored, ap)
            del seed64, sums, counts, maps, counts_scored, ap
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
    print(f"Completed {expected} controlled SA samples for {classname}", flush=True)


if __name__ == "__main__":
    main()

