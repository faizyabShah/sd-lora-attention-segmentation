#!/usr/bin/env python3
"""Run the registered 30-cell P3-C confirmation on one VOC-Sim class.

The package crosses five fixed propagation maps, three fixed calibration
rules, and DenseCRF off/on.  It generates and scores one image at a time and
stores only compact confusion counts and AP diagnostics.
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
from src.segmentation.fusion_variants import otsu_threshold  # noqa: E402
from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402
from src.segmentation.propagation_variants import (  # noqa: E402
    power_trajectory,
    restart_trajectory,
)

WORK = 64
SELECTED_SA_MODULE = 15
THRESHOLDS = tuple(round(0.30 + 0.025 * i, 3) for i in range(17))
PROPAGATIONS = (
    "REF/one_pass",
    "P1/obs_n2",
    "P1/obs_n3",
    "P2/obs_b10_n4",
    "W3/mean_n2",
)
CALIBRATIONS = (
    ("N0/global", "minmax"),
    ("N2/percentile", "percentile_01_99"),
    ("N4/otsu", "otsu"),
)
REFINEMENTS = ("R0/none", "R2/crf")
CONFIGURATIONS = tuple(
    f"{propagation}|{calibration}|{refinement}"
    for propagation in PROPAGATIONS
    for calibration, _ in CALIBRATIONS
    for refinement in REFINEMENTS
)
REFERENCE = "REF/one_pass|N0/global|R0/none"
MANDATORY = (
    REFERENCE,
    "P1/obs_n2|N0/global|R0/none",
    "REF/one_pass|N0/global|R2/crf",
    "P1/obs_n2|N0/global|R2/crf",
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
        default=Path("outputs/segmentation/controlled_attention_cv/phase3_confirmation"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--crf-site", type=Path, help="directory holding SimpleCRF's denseCRF extension"
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-reference", action="store_true")
    return parser.parse_args()


def verify_densecrf(crf_site: Path | None) -> str:
    """Fail before model loading if the Python-compatible CRF extension is absent."""
    if crf_site is not None:
        site = str(crf_site.resolve())
        if site not in sys.path:
            sys.path.insert(0, site)
    try:
        import denseCRF  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            f"SimpleCRF's denseCRF extension is unavailable for Python "
            f"{sys.version_info.major}.{sys.version_info.minor}; checked {crf_site}"
        ) from error
    return str(Path(denseCRF.__file__).resolve())


@torch.inference_mode()
def propagation_maps(
    recorder, module_name: str, seed64: torch.Tensor, device: str
) -> tuple[torch.Tensor, float]:
    """Build the five registered maps from identical SA observations."""
    qk_calls = recorder.sa_qk[module_name]
    calls = len(qk_calls)
    pixels = seed64.numel()
    heads = qk_calls[0][0].shape[0]
    seed = seed64.reshape(-1)
    obs_power = torch.zeros((3, pixels), device=device)
    obs_restart = torch.zeros((4, pixels), device=device)
    affinity_sum = torch.zeros((pixels, pixels), device=device)

    for query, key in qk_calls:
        query, key = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]),
            dim=-1,
        )
        affinity_sum += affinity.sum(dim=0)
        obs_power += power_trajectory(affinity, seed, 3).mean(dim=1)
        obs_restart += restart_trajectory(affinity, seed, (0.10,), 4)[:, :, 0].mean(dim=1)
        del query, key, affinity

    obs_power /= calls
    obs_restart /= calls
    mean_affinity = affinity_sum / (calls * heads)
    mean_power = power_trajectory(mean_affinity, seed, 2)
    difference = float((obs_power[0] - mean_power[0]).abs().max())
    if difference > 2e-5:
        raise RuntimeError(f"n=1 aggregation-order identity failed: {difference:.3e}")

    maps = torch.stack(
        (
            obs_power[0],
            obs_power[1],
            obs_power[2],
            obs_restart[3],
            mean_power[1],
        )
    ).reshape(len(PROPAGATIONS), WORK, WORK)
    return maps, difference


def _full_map(map64: torch.Tensor, shape: tuple[int, int], normalization: str) -> torch.Tensor:
    normalized = cv.normalize_batch(map64[None], normalization)[0]
    return F.interpolate(
        normalized[None, None], size=shape, mode="bilinear", align_corners=False
    )[0, 0]


def score_sample(
    maps: torch.Tensor,
    target: np.ndarray,
    image: np.ndarray,
    crf_site: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score all 30 registered combinations in CONFIGURATIONS order."""
    shape = tuple(np.asarray(target).shape)
    counts: list[np.ndarray] = []
    aps: list[float] = []
    for map64 in maps:
        for _, normalization in CALIBRATIONS:
            if normalization == "otsu":
                normalized64 = cv.normalize_batch(map64[None], "minmax")[0]
                full = F.interpolate(
                    normalized64[None, None], size=shape, mode="bilinear", align_corners=False
                )[0, 0]
                mask = (full >= otsu_threshold(normalized64)).cpu().numpy()
                base_counts = cv.score_masks(mask[None], target, thresholds=THRESHOLDS)[0]
                ap = float(
                    cv.average_precision_batch(
                        normalized64.reshape(1, -1), cv.target_at_size(target, WORK)
                    )[0]
                )
                crf_mask = apply_ovam_densecrf(image, mask, crf_site)
                crf_counts = cv.score_masks(crf_mask[None], target, thresholds=THRESHOLDS)[0]
            else:
                base_counts, base_ap = cv.score_maps(
                    map64[None], target, normalization, thresholds=THRESHOLDS
                )
                base_counts, ap = base_counts[0], float(base_ap[0])
                full = _full_map(map64, shape, normalization)
                crf_counts = np.zeros((len(THRESHOLDS), 4), dtype=np.int64)
                for index, threshold in enumerate(THRESHOLDS):
                    mask = (full >= threshold).cpu().numpy()
                    refined = apply_ovam_densecrf(image, mask, crf_site)
                    crf_counts[index] = cv.score_masks(
                        refined[None], target, thresholds=(threshold,)
                    )[0, 0]
            counts.extend((base_counts, crf_counts))
            aps.extend((ap, float("nan")))
    result_counts = np.stack(counts)
    result_ap = np.asarray(aps, dtype=np.float64)
    expected = (len(CONFIGURATIONS), len(THRESHOLDS), 4)
    if result_counts.shape != expected or result_ap.shape != (len(CONFIGURATIONS),):
        raise RuntimeError(
            f"P3-C score shape mismatch: {result_counts.shape}, {result_ap.shape}, expected {expected}"
        )
    return result_counts, result_ap


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if len(CONFIGURATIONS) != 30 or not set(MANDATORY).issubset(CONFIGURATIONS):
        raise RuntimeError("Registered 30-cell grid or mandatory cells are incomplete")
    densecrf_extension = verify_densecrf(args.crf_site)
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
        "package": "phase3_confirmation_v1",
        "protocol": "v1.2; registry Section 6; protocol Section 45",
        "condition": "frozen CA C/R; SA module 15; all calls and heads; conditional QK",
        "ca_seed": seed_spec,
        "selected_sa_module_index": SELECTED_SA_MODULE,
        "selected_sa_module": selected_module,
        "propagations": list(PROPAGATIONS),
        "calibrations": [name for name, _ in CALIBRATIONS],
        "refinements": list(REFINEMENTS),
        "thresholds": list(THRESHOLDS),
        "crf_parameters": "OVAM_CRF_PARAMETERS; unary from each thresholded candidate mask",
        "densecrf_extension": densecrf_extension,
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
    t45 = THRESHOLDS.index(0.45)
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
                    (maps[0] - direct[SELECTED_SA_MODULE].mean(0)).abs().max()
                )
                print(f"independent one-pass max abs difference={difference:.3e}", flush=True)
                if difference > 1e-4:
                    raise RuntimeError(f"P3-C/package-B reference mismatch: {difference}")
                del direct
            recorder.sa_qk.clear()
            counts, ap = score_sample(
                maps, sample.target_mask, sample.generated_image, args.crf_site
            )
            writer.add(row["image_id"], classname, counts, ap)
            summary = []
            for name in MANDATORY:
                index = CONFIGURATIONS.index(name)
                cell = counts[index, t45]
                summary.append(f"{name}={cell[0] / max(cell[:3].sum(), 1):.3f}")
            print(
                " ".join(summary) + f" order-check={arithmetic_difference:.2e}",
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
    if args.max_samples is None:
        if len(writer.completed) != expected:
            raise RuntimeError(f"Shard has {len(writer.completed)} of {expected} expected images")
        writer.finish()
    print(f"Completed {len(writer.completed)} P3-C samples for {classname}", flush=True)


if __name__ == "__main__":
    main()
