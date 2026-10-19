#!/usr/bin/env python3
"""Run SG8 observation-consensus propagation on one VOC or COCO class shard."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_coco_cap_transfer as transfer  # noqa: E402
import scripts.run_sa_package_b as sa_common  # noqa: E402
import scripts.run_sg1_sg3_graph_refinement as graph_common  # noqa: E402, installs DualSeedRecorder
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.graph_refinement import minmax_vector  # noqa: E402
from src.segmentation.observation_consensus import configuration_names, consensus_maps  # noqa: E402
from src.segmentation.sg_visual_audit import (  # noqa: E402
    best_thresholds, released_image, render_variant_pages,
)

WORK = 64
SPEC_PATH = ROOT / "configs/sg8_observation_consensus.json"
OUT = ROOT / "outputs/segmentation/controlled_attention_cv/sg8_observation_consensus"


class DiagnosticWriter:
    """Small, atomic and resumable sidecar aligned by image id."""

    def __init__(self, path: Path, names: tuple[str, ...], thresholds: tuple[float, ...], fields: tuple[str, ...]):
        self.path, self.names, self.thresholds, self.fields = path, names, thresholds, fields
        self.records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if tuple(data["configurations"].tolist()) != names or not np.allclose(data["thresholds"], thresholds):
                    raise RuntimeError(f"Diagnostic settings changed in {path}")
                if tuple(data["agreement_fields"].tolist()) != fields:
                    raise RuntimeError(f"Diagnostic fields changed in {path}")
                for i, image_id in enumerate(data["image_ids"].tolist()):
                    self.records[image_id] = (data["agreement"][i], data["changes"][i], data["mechanism"][i])

    def add(self, image_id: str, agreement: np.ndarray, changes: np.ndarray, mechanism: np.ndarray) -> None:
        self.records[image_id] = (agreement, changes, mechanism)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.records)
        temporary = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            image_ids=np.asarray(ids, dtype=str),
            configurations=np.asarray(self.names, dtype=str),
            thresholds=np.asarray(self.thresholds),
            agreement_fields=np.asarray(self.fields, dtype=str),
            agreement=np.stack([self.records[i][0] for i in ids]).astype(np.float32),
            changes=np.stack([self.records[i][1] for i in ids]).astype(np.int32),
            mechanism=np.stack([self.records[i][2] for i in ids]).astype(np.float64),
        )
        os.replace(temporary, self.path)


def agreement_fields(rhos: tuple[float, ...]) -> tuple[str, ...]:
    names = ["sigma_mean", "sigma_median"]
    names += [f"fraction_up_ge_{rho:.2f}" for rho in rhos]
    names += [f"fraction_down_ge_{rho:.2f}" for rho in rhos]
    names += [f"r_up_q{q}" for q in (0, 25, 50, 75, 100)]
    names += [f"r_down_q{q}" for q in (0, 25, 50, 75, 100)]
    return tuple(names)


@torch.inference_mode()
def propagated_observations(recorder, module: str, seeds64: torch.Tensor, spec: dict, device: str):
    """Return ``A_i c`` vectors without retaining any full affinity matrix."""
    seeds = minmax_vector(seeds64.reshape(2, -1).float())
    calls = recorder.sa_qk[module]
    expected = spec["sa"]
    if len(calls) != expected["calls"]:
        raise RuntimeError(f"Expected {expected['calls']} SA calls, found {len(calls)}")
    values, reference_sum = [], torch.zeros_like(seeds)
    head_count = None
    for query, key in calls:
        q, k = query.to(device).float(), key.to(device).float()
        if head_count is None:
            head_count = q.shape[0]
        elif q.shape[0] != head_count:
            raise RuntimeError("SA head count changed between calls")
        affinity = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        current = torch.matmul(affinity, seeds.T).permute(0, 2, 1)  # head, track, pixel
        reference_sum += current.sum(dim=0)
        values.append(current)
        del q, k, affinity, current
    if head_count != expected["heads"]:
        raise RuntimeError(f"Expected {expected['heads']} SA heads, found {head_count}")
    propagated = torch.cat(values, dim=0)
    if propagated.shape[0] != expected["observations"]:
        raise RuntimeError(f"Expected {expected['observations']} observations, found {propagated.shape[0]}")
    reference = reference_sum / propagated.shape[0]
    return seeds, propagated, reference


def compact_agreement(diagnostics: dict[str, torch.Tensor], rhos: tuple[float, ...]) -> np.ndarray:
    rows = []
    quantiles = torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0), device=diagnostics["sigma"].device)
    for track in range(2):
        sigma, up, down = (diagnostics[k][track] for k in ("sigma", "r_up", "r_down"))
        row = [sigma.mean(), sigma.median()]
        row += [(up >= rho).float().mean() for rho in rhos]
        row += [(down >= rho).float().mean() for rho in rhos]
        row += list(torch.quantile(up, quantiles)) + list(torch.quantile(down, quantiles))
        rows.append(torch.stack(row).cpu().numpy())
    return np.stack(rows)


@torch.inference_mode()
def mask_diagnostics(
    maps: torch.Tensor, seeds: torch.Tensor, names: tuple[str, ...], target: np.ndarray,
    thresholds: tuple[float, ...], sigma: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return threshold-wise seed changes and REF agreement/error summaries.

    ``mechanism[track, region, (sigma_sum, consensus_sum, pixels)]`` uses the
    regions added-TP, added-FP, distant-added-FP (>16 annotation pixels), and
    retained/unchanged.  This is enough to test whether leakage has weaker
    observation agreement without saving a map.
    """
    truth = torch.as_tensor(np.asarray(target), device=maps.device).bool()
    normalized = cv.normalize_batch(maps.reshape(-1, WORK, WORK), "minmax")
    full = F.interpolate(normalized[:, None], size=truth.shape, mode="bilinear", align_corners=False)[:, 0]
    seed_norm = cv.normalize_batch(seeds.reshape(2, WORK, WORK), "minmax")
    seed_full = F.interpolate(seed_norm[:, None], size=truth.shape, mode="bilinear", align_corners=False)[:, 0]
    changes = np.zeros((len(names), len(thresholds), 4), dtype=np.int64)
    track_for = np.asarray([0 if name.startswith("W/") else 1 for name in names])
    for t_index, threshold in enumerate(thresholds):
        pred = full >= threshold
        base = seed_full[torch.as_tensor(track_for, device=maps.device)] >= threshold
        added, removed = pred & ~base, base & ~pred
        changes[:, t_index, 0] = (added & truth).sum((1, 2)).cpu().numpy()
        changes[:, t_index, 1] = (added & ~truth).sum((1, 2)).cpu().numpy()
        changes[:, t_index, 2] = (removed & truth).sum((1, 2)).cpu().numpy()
        changes[:, t_index, 3] = (removed & ~truth).sum((1, 2)).cpu().numpy()

    mechanism = np.zeros((2, 4, 3), dtype=np.float64)
    t_index = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - 0.45))
    distance = ndimage.distance_transform_edt(~np.asarray(target, bool))
    sigma_full = F.interpolate(sigma[:, None].float(), size=truth.shape, mode="bilinear", align_corners=False)[:, 0]
    consensus_full = F.interpolate(torch.maximum(up, down)[:, None].float(), size=truth.shape,
                                   mode="bilinear", align_corners=False)[:, 0]
    for track, prefix in enumerate(("W", "T")):
        ref = full[names.index(f"{prefix}/REF")] >= thresholds[t_index]
        base = seed_full[track] >= thresholds[t_index]
        regions = (ref & ~base & truth, ref & ~base & ~truth,
                   ref & ~base & ~truth & torch.as_tensor(distance > 16, device=maps.device), ref == base)
        for region_index, region in enumerate(regions):
            mechanism[track, region_index] = (
                float(sigma_full[track][region].sum()),
                float(consensus_full[track][region].sum()),
                int(region.sum()),
            )
    return changes, mechanism


def save_visual(path: Path, seeds: torch.Tensor, diagnostics: dict[str, torch.Tensor], maps: torch.Tensor,
                names: tuple[str, ...]) -> None:
    """Save a fixed diagnostic panel; it does not affect any registered choice."""
    panels = []
    for track, prefix in enumerate(("W", "T")):
        values = (seeds[track], diagnostics["mean"][track], diagnostics["sigma"][track],
                  diagnostics["r_up"][track], diagnostics["r_down"][track],
                  maps[names.index(f"{prefix}/C/rho0p80")])
        row = []
        for value in values:
            flat = value.detach().float().cpu().reshape(WORK, WORK)
            low, high = flat.min(), flat.max()
            scaled = ((flat - low) / (high - low).clamp_min(1e-12) * 255).byte().numpy()
            row.append(scaled)
        panels.append(np.concatenate(row, axis=1))
    canvas = np.concatenate(panels, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").resize((WORK * 6 * 2, WORK * 2 * 2), Image.Resampling.NEAREST).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=ROOT / "configs/controlled_sa_seed_frozen.json")
    parser.add_argument("--tokens", type=Path, default=ROOT / "configs/ovam/tokens_VOC.npy")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--save-diagnostic-panels", action="store_true")
    parser.add_argument("--visual-dir", type=Path,
                        help="Render the first image and every registered cell; do not write metric shards")
    args = parser.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    ca_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not ca_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    voc = args.dataset == "voc"
    classname, rows = (transfer.voc_rows if voc else transfer.coco_rows)(args.class_shard)
    rows = rows[:1] if args.visual_dir else (rows[:args.max_samples] if args.max_samples else rows)
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    selected = sa_modules[spec["sa"]["module_index"]]
    ca_spec = sa_common.resolve_seed_spec(ca_json, ca_modules, spec["sa"]["calls"])
    released = np.load(args.tokens, allow_pickle=True).item()
    prompt = rows[0]["attribution_prompt"]
    if any(row["attribution_prompt"] != prompt for row in rows):
        raise RuntimeError(f"{classname}: attribution prompt must be constant within a class shard")
    pair, class_hidden = graph_common.hidden_states(pipe, classname, prompt, released, args.device, not voc)
    graph_common._EXTRA.update(hidden=pair, class_hidden=class_hidden, separate_class_prompt=not voc)

    taus = tuple(float(v) for v in spec["sg8_b"]["taus"])
    rhos = tuple(float(v) for v in spec["sg8_c"]["rhos"])
    thresholds = tuple(float(v) for v in spec["thresholds"])
    names = configuration_names(tuple(spec["tracks"]), taus, rhos)
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if voc else {r["image_id"]: 0 for r in rows}
    settings = {
        "package": "sg8_observation_consensus_v1", "protocol": "Section 73", "dataset": args.dataset,
        "spec": spec, "ca_seed": ca_spec, "selected_sa_module": selected, "thresholds": list(thresholds),
        "configurations": list(names), "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
    }
    base = args.output_dir / args.dataset / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    writer = None if args.visual_dir else cv.ShardWriter(base.with_suffix(".npz"), names, folds, settings, thresholds=thresholds)
    diag = None if args.visual_dir else DiagnosticWriter(base.with_suffix(".diagnostics.npz"), names, thresholds, agreement_fields(rhos))
    visual_thresholds = best_thresholds(OUT) if args.visual_dir else {}
    t45 = thresholds.index(0.45)
    for row in rows:
        if writer is not None and row["image_id"] in writer.completed:
            continue
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(pipe, row, manifest_dir, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {name: resolution for resolution, modules in sample.modules_by_resolution.items() for name in modules}
            word_seed = sa_common.ca_seed(recorder, ca_modules, ca_res, ca_spec, args.device)
            word_calls, recorder.ca_calls = recorder.ca_calls, recorder.token_calls
            token_seed = sa_common.ca_seed(recorder, ca_modules, ca_res, ca_spec, args.device)
            recorder.ca_calls = word_calls
            seeds, propagated, reference = propagated_observations(
                recorder, selected, torch.stack((word_seed, token_seed)), spec, args.device
            )
            recorder.sa_qk.clear()
            maps, diagnostics = consensus_maps(propagated, seeds, taus, rhos)
            for track, prefix in enumerate(("W", "T")):
                error = float((maps[names.index(f"{prefix}/REF")] - reference[track]).abs().max())
                if error > 2e-6:
                    raise RuntimeError(f"{prefix}/REF mismatch: max abs error {error}")
            maps64 = maps.reshape(-1, WORK, WORK)
            if args.visual_dir:
                ca_maps = [seeds[0] if name.startswith("W/") else seeds[1] for name in names]
                render_variant_pages(
                    args.visual_dir, "sg8", classname, row["image_id"],
                    released_image(row, manifest_dir), np.asarray(sample.target_mask, bool),
                    names, ca_maps, list(maps64), visual_thresholds,
                )
                print(f"[visual:{classname}] {row['image_id']} cells={len(names)}", flush=True)
                continue
            counts, ap = cv.score_maps(maps64, sample.target_mask, "minmax", thresholds=thresholds)
            agreement = compact_agreement(diagnostics, rhos)
            changes, mechanism = mask_diagnostics(
                maps64, seeds, names, np.asarray(sample.target_mask, bool), thresholds,
                diagnostics["sigma"].reshape(2, WORK, WORK), diagnostics["r_up"].reshape(2, WORK, WORK),
                diagnostics["r_down"].reshape(2, WORK, WORK),
            )
            assert diag is not None and writer is not None
            diag.add(row["image_id"], agreement, changes, mechanism)
            writer.add(row["image_id"], classname, counts, ap)
            if args.save_diagnostic_panels:
                save_visual(args.output_dir / "diagnostic_panels" / f"{row['image_id']}.png",
                            seeds, diagnostics, maps, names)
            print(f"[{args.dataset}:{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) " +
                  " ".join(f"{name}={counts[names.index(name),t45,0]/max(counts[names.index(name),t45,:3].sum(),1):.3f}"
                           for name in ("W/REF", "W/C/rho0p80", "T/REF", "T/C/rho0p80")), flush=True)
        finally:
            recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear()
            sample.clear(); del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if writer is not None and args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
