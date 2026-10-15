#!/usr/bin/env python3
"""Run the controlled SA-C combined confirmation on one VOC-Sim class shard."""

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

WORK = 64
HEADS = tuple(range(8))
SELECTED_HEADS = (1, 4, 6)
SELECTED_CALL = 30
SELECTED_RESOLUTIONS = (8, 64)
SELECTED_MODULE = 15

CONFIGURATIONS = (
    "CHECK/ca_seed", "C/reference", "C/T", "C/R", "C/L", "C/H",
    "C/T+R", "C/T+L", "C/T+H", "C/R+H", "C/L+H", "C/T+R+H", "C/T+L+H",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_frozen.json"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"))
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/segmentation/controlled_attention_cv/sa_confirmation"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-reference", action="store_true")
    return parser.parse_args()


def selected_axes(name: str) -> frozenset[str]:
    if name == "C/reference":
        return frozenset()
    if not name.startswith("C/"):
        raise ValueError(name)
    return frozenset(name.removeprefix("C/").split("+"))


@torch.inference_mode()
def confirmation_maps(recorder, sa_modules, module_resolutions, seed64, calls, device):
    """Build every unique SA-C candidate from the same propagated observations."""
    candidate_names = CONFIGURATIONS[1:]
    axes = {name: selected_axes(name) for name in candidate_names}
    sums = {name: torch.zeros((WORK, WORK), device=device) for name in candidate_names}
    counts = {name: 0 for name in candidate_names}
    for module_index, module_name in enumerate(sa_modules):
        resolution = module_resolutions[module_name]
        seed = (F.adaptive_avg_pool2d(seed64[None, None], (resolution, resolution))[0, 0]
                if resolution != WORK else seed64).reshape(-1)
        qk_calls = recorder.sa_qk[module_name]
        if len(qk_calls) != calls:
            raise RuntimeError(f"{module_name}: expected {calls} calls, found {len(qk_calls)}")
        for call, (query, key) in enumerate(qk_calls):
            query, key = query.to(device).float(), key.to(device).float()
            affinity = torch.softmax(
                torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1)
            propagated = torch.matmul(affinity, seed).reshape(len(HEADS), 1, resolution, resolution)
            if resolution != WORK:
                propagated = F.interpolate(
                    propagated, size=(WORK, WORK), mode="bilinear", align_corners=False)
            propagated = propagated[:, 0]
            for name in candidate_names:
                choice = axes[name]
                if "T" in choice and call != SELECTED_CALL:
                    continue
                if "L" in choice and module_index != SELECTED_MODULE:
                    continue
                if "L" not in choice and "R" in choice and resolution not in SELECTED_RESOLUTIONS:
                    continue
                heads = SELECTED_HEADS if "H" in choice else HEADS
                sums[name] += propagated[list(heads)].sum(dim=0)
                counts[name] += len(heads)
            del query, key, affinity, propagated
    maps = [seed64]
    for name in candidate_names:
        if counts[name] == 0:
            raise RuntimeError(f"No observations accumulated for {name}")
        maps.append(sums[name] / counts[name])
    return torch.stack(maps)


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
        args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    if len(ca_modules) != 16 or len(sa_modules) != 16:
        raise RuntimeError("Expected 16 CA and 16 SA modules")
    ca_seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    settings = {
        "package": "controlled_sa_confirmation_v1", "protocol": "v1.2 Sections 7 and 31",
        "condition": "frozen CA seed; conditional SA; one non-residual y=A x pass; area-downsample seed; equal observation weights; minmax once; threshold 0.45",
        "ca_seed": ca_seed_spec, "model": args.model, "model_dtype": "float32",
        "qk_storage_dtype": "float16", "manifest": str(args.manifest),
        "fold_file": str(args.folds), "ca_modules": list(ca_modules),
        "sa_modules": list(sa_modules), "selected_call": SELECTED_CALL,
        "selected_resolutions": list(SELECTED_RESOLUTIONS), "selected_module": SELECTED_MODULE,
        "selected_heads": list(SELECTED_HEADS), "configurations": list(CONFIGURATIONS),
        "postprocessing": "none",
    }
    shard_path = args.output_dir / "shards" / (
        f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz")
    writer = cv.ShardWriter(shard_path, CONFIGURATIONS, folds, settings)
    expected = len(rows)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        print(f"[{classname}] {row['image_id']} ({len(writer.completed)+1}/{expected})", flush=True)
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(
            pipe, row, args.manifest.parent, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            calls = len(sample.timesteps)
            ca_resolutions = {name: resolution for resolution, names in sample.modules_by_resolution.items()
                              for name in names}
            sa_resolutions = {name: resolution for resolution, names in sample.self_modules_by_resolution.items()
                              for name in names}
            if sa_resolutions[sa_modules[SELECTED_MODULE]] != 64:
                raise RuntimeError("Expected selected SA module 15 to be 64x64")
            seed64 = sa_common.ca_seed(recorder, ca_modules, ca_resolutions, ca_seed_spec, args.device)
            recorder.ca_calls.clear()
            maps = confirmation_maps(recorder, sa_modules, sa_resolutions, seed64, calls, args.device)
            if args.verify_reference:
                reference, _ = sa_common.propagate(recorder, sa_modules, sa_resolutions, seed64, args.device)
                difference = float((maps[CONFIGURATIONS.index("C/reference")] -
                                    reference.mean(dim=(0, 1))).abs().max())
                print(f"inclusive reference max abs difference={difference:.3e}", flush=True)
                if difference > 1e-4:
                    raise RuntimeError(f"SA-C/package-B reference mismatch: {difference}")
                del reference
            recorder.sa_qk.clear()
            counts, ap = cv.score_maps(maps, sample.target_mask, "minmax")
            writer.add(row["image_id"], classname, counts, ap)
            del seed64, maps, counts, ap
        finally:
            recorder.ca_calls.clear(); recorder.sa_qk.clear(); sample.clear()
            del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if len(writer.completed) != expected:
        raise RuntimeError(f"Shard has {len(writer.completed)} of {expected} expected images")
    writer.finish()
    print(f"Completed {expected} SA-C samples for {classname}", flush=True)


if __name__ == "__main__":
    main()
