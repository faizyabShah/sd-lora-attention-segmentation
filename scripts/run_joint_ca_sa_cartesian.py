#!/usr/bin/env python3
"""Run the registered joint CA x SA interaction audit on one VOC-Sim class.

The search is the full Cartesian product of the six CA-C configurations and
the twelve unique SA-C configurations, plus a no-SA control for every CA seed.
Each image is generated once.  Only compact per-image confusion counts and AP
are retained; images, CA maps, Q/K tensors, affinities, and propagated maps are
released before the next image.
"""

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
ALL_MODULES = tuple(range(16))
ALL_HEADS = tuple(range(8))
CA_RES32 = frozenset((2, 3, 10, 11, 12))
CA_LAYER_DROP = frozenset((10, 11))
CA_HEAD_DROP = frozenset((1,))

SELECTED_SA_CALL = 30
SELECTED_SA_RESOLUTIONS = frozenset((8, 64))
SELECTED_SA_MODULE = 15
SELECTED_SA_HEADS = (1, 4, 6)

# These are exactly the six configurations admitted to CA-C.
CA_CONFIGURATIONS = {
    "C/reference": (ALL_MODULES, ALL_HEADS),
    "C/R": (tuple(i for i in ALL_MODULES if i not in CA_RES32), ALL_HEADS),
    "C/L": (tuple(i for i in ALL_MODULES if i not in CA_LAYER_DROP), ALL_HEADS),
    "C/H": (ALL_MODULES, tuple(i for i in ALL_HEADS if i not in CA_HEAD_DROP)),
    "C/R+H": (
        tuple(i for i in ALL_MODULES if i not in CA_RES32),
        tuple(i for i in ALL_HEADS if i not in CA_HEAD_DROP),
    ),
    "C/L+H": (
        tuple(i for i in ALL_MODULES if i not in CA_LAYER_DROP),
        tuple(i for i in ALL_HEADS if i not in CA_HEAD_DROP),
    ),
}

# These are exactly the twelve unique configurations admitted to SA-C.  The
# excluded R+L cells are arithmetic duplicates of L because module 15 is 64x64.
SA_CONFIGURATIONS = (
    "C/reference",
    "C/T",
    "C/R",
    "C/L",
    "C/H",
    "C/T+R",
    "C/T+L",
    "C/T+H",
    "C/R+H",
    "C/L+H",
    "C/T+R+H",
    "C/T+L+H",
)
SA_CHOICES = ("none", *SA_CONFIGURATIONS)


def joint_name(ca_name: str, sa_name: str) -> str:
    return f"CA={ca_name}|SA={sa_name}"


JOINT_CONFIGURATIONS = tuple(
    joint_name(ca_name, sa_name)
    for ca_name in CA_CONFIGURATIONS
    for sa_name in SA_CHOICES
)
REFERENCE = joint_name("C/R", "C/L")


def selected_sa_axes(name: str) -> frozenset[str]:
    if name == "C/reference":
        return frozenset()
    if not name.startswith("C/"):
        raise ValueError(name)
    return frozenset(name.removeprefix("C/").split("+"))


def ca_seed_specs() -> dict[str, dict[str, object]]:
    return {
        name: {
            "name": f"joint_{name}",
            "attribution": "cond_padded_all",
            "calls": list(range(31)),
            "modules": list(modules),
            "heads": list(heads),
            "frozen": False,
        }
        for name, (modules, heads) in CA_CONFIGURATIONS.items()
    }


def _propagate_seed_batch(affinity: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """Apply each head's row-stochastic affinity to every seed.

    ``affinity`` is [heads, pixels, pixels] and ``seeds`` is [seeds, pixels].
    The result is [seeds, heads, pixels].  Keeping this operation batched makes
    the six-CA Cartesian audit require one affinity construction per SA
    observation rather than six.
    """
    if affinity.ndim != 3 or seeds.ndim != 2:
        raise ValueError("Expected affinity [H,P,P] and seeds [C,P]")
    if affinity.shape[-1] != seeds.shape[-1]:
        raise ValueError("Affinity and seed pixel counts differ")
    return torch.matmul(affinity, seeds.transpose(0, 1)).permute(2, 0, 1)


@torch.inference_mode()
def cartesian_maps(
    recorder,
    sa_modules: tuple[str, ...],
    module_resolutions: dict[str, int],
    seeds64: torch.Tensor,
    calls: int,
    device: str,
) -> torch.Tensor:
    """Return [6, 13, 64, 64] maps for the full registered Cartesian grid."""
    axes = {name: selected_sa_axes(name) for name in SA_CONFIGURATIONS}
    sums = torch.zeros(
        (len(CA_CONFIGURATIONS), len(SA_CONFIGURATIONS), WORK, WORK),
        device=device,
    )
    counts = torch.zeros(len(SA_CONFIGURATIONS), dtype=torch.int64, device=device)

    for module_index, module_name in enumerate(sa_modules):
        resolution = module_resolutions[module_name]
        native = (
            F.adaptive_avg_pool2d(seeds64[:, None], (resolution, resolution))[:, 0]
            if resolution != WORK
            else seeds64
        ).reshape(len(CA_CONFIGURATIONS), -1)
        qk_calls = recorder.sa_qk[module_name]
        if len(qk_calls) != calls:
            raise RuntimeError(
                f"{module_name}: expected {calls} calls, found {len(qk_calls)}"
            )
        for call, (query, key) in enumerate(qk_calls):
            query, key = query.to(device).float(), key.to(device).float()
            affinity = torch.softmax(
                torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]),
                dim=-1,
            )
            propagated = _propagate_seed_batch(affinity, native).reshape(
                len(CA_CONFIGURATIONS) * len(ALL_HEADS), 1, resolution, resolution
            )
            if resolution != WORK:
                propagated = F.interpolate(
                    propagated,
                    size=(WORK, WORK),
                    mode="bilinear",
                    align_corners=False,
                )
            propagated = propagated[:, 0].reshape(
                len(CA_CONFIGURATIONS), len(ALL_HEADS), WORK, WORK
            )
            for sa_index, name in enumerate(SA_CONFIGURATIONS):
                choice = axes[name]
                if "T" in choice and call != SELECTED_SA_CALL:
                    continue
                if "L" in choice and module_index != SELECTED_SA_MODULE:
                    continue
                if (
                    "L" not in choice
                    and "R" in choice
                    and resolution not in SELECTED_SA_RESOLUTIONS
                ):
                    continue
                heads = SELECTED_SA_HEADS if "H" in choice else ALL_HEADS
                sums[:, sa_index] += propagated[:, list(heads)].sum(dim=1)
                counts[sa_index] += len(heads)
            del query, key, affinity, propagated

    if bool((counts == 0).any()):
        missing = [SA_CONFIGURATIONS[i] for i in torch.where(counts == 0)[0].tolist()]
        raise RuntimeError(f"No observations accumulated for {missing}")
    propagated = sums / counts[None, :, None, None]
    return torch.cat((seeds64[:, None], propagated), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
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
        default=Path("outputs/segmentation/controlled_attention_cv/joint_ca_sa_cartesian"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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

    seed_specs = {
        name: sa_common.resolve_seed_spec(spec, ca_modules, 31)
        for name, spec in ca_seed_specs().items()
    }
    settings = {
        "package": "joint_ca_sa_cartesian_v1",
        "protocol": "v1.2; joint interaction audit Section 34",
        "scope": "full Cartesian product of 6 CA-C cells and 12 unique SA-C cells plus no-SA",
        "condition": "conditional CA/SA; CA seeds min-max normalized independently; one non-residual y=A x pass; area-downsample seeds; equal SA observation weights; final min-max once; no postprocessing",
        "reference": REFERENCE,
        "ca_configurations": seed_specs,
        "sa_configurations": list(SA_CHOICES),
        "inclusiveness": "CA retained-observation fraction + SA retained-observation fraction",
        "model": args.model,
        "model_dtype": "float32",
        "attention_storage_dtype": "float16",
        "manifest": str(args.manifest),
        "fold_file": str(args.folds),
        "ca_modules": list(ca_modules),
        "sa_modules": list(sa_modules),
        "thresholds": list(cv.THRESHOLDS),
        "postprocessing": "none",
    }
    shard_path = args.output_dir / "shards" / (
        f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    )
    writer = cv.ShardWriter(shard_path, JOINT_CONFIGURATIONS, folds, settings)
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
            calls = len(sample.timesteps)
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
            seeds64 = torch.stack(
                [
                    sa_common.ca_seed(
                        recorder, ca_modules, ca_resolutions, seed_specs[name], args.device
                    )
                    for name in CA_CONFIGURATIONS
                ]
            )
            recorder.ca_calls.clear()
            grid = cartesian_maps(
                recorder, sa_modules, sa_resolutions, seeds64, calls, args.device
            )
            maps = grid.reshape(len(JOINT_CONFIGURATIONS), WORK, WORK)
            if args.verify_reference:
                ca_index = tuple(CA_CONFIGURATIONS).index("C/R")
                independent, _ = sa_common.propagate(
                    recorder, sa_modules, sa_resolutions, seeds64[ca_index], args.device
                )
                difference = float(
                    (grid[ca_index, SA_CHOICES.index("C/reference")] - independent.mean((0, 1)))
                    .abs()
                    .max()
                )
                print(f"inclusive SA reference max abs difference={difference:.3e}", flush=True)
                if difference > 1e-4:
                    raise RuntimeError(f"Joint/inclusive SA reference mismatch: {difference}")
                del independent
            recorder.sa_qk.clear()
            counts, ap = cv.score_maps(maps, sample.target_mask, "minmax")
            writer.add(row["image_id"], classname, counts, ap)
            ref_counts = counts[JOINT_CONFIGURATIONS.index(REFERENCE), cv.SELECTION_INDEX]
            ref_iou = ref_counts[0] / max(ref_counts[:3].sum(), 1)
            print(f"{row['image_id']} sequential-reference IoU={ref_iou:.3f}", flush=True)
            del seeds64, grid, maps, counts, ap
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
    print(f"Completed {expected} joint CA x SA samples for {classname}", flush=True)


if __name__ == "__main__":
    main()
