#!/usr/bin/env python3
"""Run SG9 reciprocal self-attention propagation on one dataset/class shard."""

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
from src.segmentation.reciprocal_propagation import (  # noqa: E402
    OPERATORS, configuration_names, reciprocal_affinity, row_normalize,
)
from src.segmentation.sg_visual_audit import (  # noqa: E402
    best_thresholds, released_image, render_variant_pages,
)

WORK = 64
SPEC_PATH = ROOT / "configs/sg9_reciprocal_propagation.json"
OUT = ROOT / "outputs/segmentation/controlled_attention_cv/sg9_reciprocal_propagation"


class DiagnosticWriter:
    """Atomic sidecar for compact SG9 diagnostics."""

    def __init__(self, path: Path, names: tuple[str, ...], thresholds: tuple[float, ...]):
        self.path, self.names, self.thresholds = path, names, thresholds
        self.records: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if tuple(data["configurations"].tolist()) != names or not np.allclose(data["thresholds"], thresholds):
                    raise RuntimeError(f"Diagnostic settings changed in {path}")
                for i, image_id in enumerate(data["image_ids"].tolist()):
                    self.records[image_id] = (data["changes"][i], data["spatial"][i], data["mechanism"][i])

    def add(self, image_id: str, changes: np.ndarray, spatial: np.ndarray, mechanism: np.ndarray) -> None:
        self.records[image_id] = (changes, spatial, mechanism)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(self.records)
        temporary = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez_compressed(
            temporary, image_ids=np.asarray(ids, dtype=str), configurations=np.asarray(self.names, dtype=str),
            thresholds=np.asarray(self.thresholds),
            changes=np.stack([self.records[i][0] for i in ids]).astype(np.int32),
            spatial=np.stack([self.records[i][1] for i in ids]).astype(np.int32),
            mechanism=np.stack([self.records[i][2] for i in ids]).astype(np.float64),
        )
        os.replace(temporary, self.path)


@torch.inference_mode()
def reciprocal_maps(recorder, module: str, seeds64: torch.Tensor, spec: dict, device: str,
                    collect_edges: bool = False):
    """Propagate through each call/head separately while bounding peak memory."""
    seeds = minmax_vector(seeds64.reshape(2, -1).float())
    calls = recorder.sa_qk[module]
    expected, epsilon = spec["sa"], float(spec["epsilon"])
    if len(calls) != expected["calls"]:
        raise RuntimeError(f"Expected {expected['calls']} calls, found {len(calls)}")
    sums = torch.zeros((len(OPERATORS), 2, WORK * WORK), device=device)
    asymmetry_sum = torch.zeros((2, WORK * WORK), device=device)
    cycle_sum = torch.zeros(WORK * WORK, device=device)
    edge_rows = []
    observations = 0
    for call_index, (query, key) in enumerate(calls):
        if query.shape[0] != expected["heads"]:
            raise RuntimeError(f"Expected {expected['heads']} heads, found {query.shape[0]}")
        for head in range(query.shape[0]):
            q, k = query[head].to(device).float(), key[head].to(device).float()
            affinity = torch.softmax(q @ k.transpose(0, 1) / math.sqrt(q.shape[-1]), dim=-1)
            reverse = affinity.transpose(0, 1)
            forward = affinity @ seeds.T
            sums[0] += forward.T

            product = affinity * reverse
            cycle_sum += product.sum(dim=-1)
            reciprocal = torch.sqrt(product.clamp_min(0))
            sums[1] += (row_normalize(reciprocal, epsilon) @ seeds.T).T
            del reciprocal
            sums[2] += (reciprocal_affinity(affinity, "R2_minimum", epsilon) @ seeds.T).T
            sums[3] += (reciprocal_affinity(affinity, "R3_soft_gate", epsilon) @ seeds.T).T

            asymmetry = (affinity - reverse).abs() / (affinity + reverse + epsilon)
            asymmetry_sum += (((affinity * asymmetry) @ seeds.T) / forward.clamp_min(epsilon)).T
            if collect_edges:
                values, indices = torch.topk(affinity.reshape(-1), k=2)
                for value, index in zip(values.tolist(), indices.tolist()):
                    source, target = divmod(index, affinity.shape[1])
                    backward = float(affinity[target, source])
                    edge_rows.append({"call": call_index, "head": head, "source": source, "target": target,
                                      "forward": value, "reverse": backward,
                                      "geometric": math.sqrt(max(value * backward, 0.0)),
                                      "asymmetry": abs(value - backward) / (value + backward + epsilon)})
            observations += 1
            del q, k, affinity, reverse, forward, product, asymmetry
    if observations != expected["observations"]:
        raise RuntimeError(f"Expected {expected['observations']} observations, found {observations}")
    maps_by_operator = sums / observations
    maps = torch.cat([maps_by_operator[:, track] for track in range(2)]).reshape(-1, WORK, WORK)
    return maps, seeds, asymmetry_sum.div_(observations).reshape(2, WORK, WORK), \
        cycle_sum.div_(observations).reshape(WORK, WORK), edge_rows


@torch.inference_mode()
def compact_diagnostics(maps: torch.Tensor, seeds: torch.Tensor, names: tuple[str, ...], target: np.ndarray,
                        thresholds: tuple[float, ...], asymmetry: torch.Tensor, cycle: torch.Tensor):
    truth_np = np.asarray(target, bool)
    truth = torch.as_tensor(truth_np, device=maps.device).bool()
    normalized = cv.normalize_batch(maps, "minmax")
    full = F.interpolate(normalized[:, None], size=truth.shape, mode="bilinear", align_corners=False)[:, 0]
    seed_norm = cv.normalize_batch(seeds.reshape(2, WORK, WORK), "minmax")
    seed_full = F.interpolate(seed_norm[:, None], size=truth.shape, mode="bilinear", align_corners=False)[:, 0]
    track_for = torch.tensor([0 if name.startswith("W/") else 1 for name in names], device=maps.device)
    changes = np.zeros((len(names), len(thresholds), 4), dtype=np.int64)
    spatial = np.zeros((len(names), len(thresholds), 3), dtype=np.int64)
    distance = ndimage.distance_transform_edt(~truth_np)
    for t_index, threshold in enumerate(thresholds):
        pred = full >= threshold
        base = seed_full[track_for] >= threshold
        added, removed = pred & ~base, base & ~pred
        changes[:, t_index, 0] = (added & truth).sum((1, 2)).cpu().numpy()
        changes[:, t_index, 1] = (added & ~truth).sum((1, 2)).cpu().numpy()
        changes[:, t_index, 2] = (removed & truth).sum((1, 2)).cpu().numpy()
        changes[:, t_index, 3] = (removed & ~truth).sum((1, 2)).cpu().numpy()
        pred_np, added_fp_np = pred.cpu().numpy(), (added & ~truth).cpu().numpy()
        for config in range(len(names)):
            labels, _ = ndimage.label(pred_np[config])
            touching = np.unique(labels[pred_np[config] & truth_np])
            attached = np.isin(labels, touching[touching > 0])
            new_fp = added_fp_np[config]
            spatial[config, t_index] = (int((new_fp & (distance > 16)).sum()),
                                                int((new_fp & attached).sum()),
                                                int((new_fp & ~attached).sum()))

    # Per-pixel edge diagnostics for regions changed by ordinary propagation.
    mechanism = np.zeros((2, 4, 3), dtype=np.float64)
    asym_full = F.interpolate(asymmetry[:, None], size=truth.shape, mode="bilinear", align_corners=False)[:, 0]
    cycle_full = F.interpolate(cycle[None, None], size=truth.shape, mode="bilinear", align_corners=False)[0, 0]
    t45 = thresholds.index(0.45)
    for track, prefix in enumerate(("W", "T")):
        ref = full[names.index(f"{prefix}/REF")] >= thresholds[t45]
        base = seed_full[track] >= thresholds[t45]
        regions = (ref & ~base & truth, ref & ~base & ~truth,
                   ref & ~base & ~truth & torch.as_tensor(distance > 16, device=maps.device), ref == base)
        for region_index, region in enumerate(regions):
            mechanism[track, region_index] = (float(asym_full[track][region].sum()),
                                               float(cycle_full[region].sum()), int(region.sum()))
    return changes, spatial, mechanism, full


def save_visual(path: Path, maps: torch.Tensor, seeds: torch.Tensor, names: tuple[str, ...],
                full: torch.Tensor, target: np.ndarray) -> None:
    truth = torch.as_tensor(np.asarray(target), device=maps.device).bool()
    rows = []
    for track, prefix in enumerate(("W", "T")):
        ref_i, reciprocal_i = names.index(f"{prefix}/REF"), names.index(f"{prefix}/R1_geometric")
        ref, reciprocal = maps[ref_i], maps[reciprocal_i]
        ref_mask, reciprocal_mask = full[ref_i] >= 0.45, full[reciprocal_i] >= 0.45
        removed_fp = ref_mask & ~reciprocal_mask & ~truth
        lost_tp = ref_mask & ~reciprocal_mask & truth
        binary = [F.interpolate(v[None, None].float(), size=(WORK, WORK), mode="nearest")[0, 0]
                  for v in (removed_fp, lost_tp)]
        values = (seeds[track].reshape(WORK, WORK), ref, reciprocal, reciprocal - ref, *binary)
        panels = []
        for value in values:
            value = value.detach().float().cpu()
            low, high = value.min(), value.max()
            scaled = ((value - low) / (high - low).clamp_min(1e-12) * 255).byte().numpy()
            panels.append(scaled)
        rows.append(np.concatenate(panels, axis=1))
    canvas = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").resize((WORK * 12, WORK * 4), Image.Resampling.NEAREST).save(path)


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
    parser.add_argument("--smoke-diagnostics", action="store_true")
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
    names = configuration_names(tuple(spec["tracks"]))
    thresholds = tuple(float(v) for v in spec["thresholds"])
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if voc else {r["image_id"]: 0 for r in rows}
    settings = {"package": "sg9_reciprocal_propagation_v1", "protocol": "Section 74", "dataset": args.dataset,
                "spec": spec, "ca_seed": ca_spec, "selected_sa_module": selected, "configurations": list(names),
                "thresholds": list(thresholds), "model": args.model, "model_dtype": "float32",
                "qk_storage_dtype": "float16", "computation": "one call/head affinity at a time"}
    base = args.output_dir / args.dataset / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    writer = None if args.visual_dir else cv.ShardWriter(base.with_suffix(".npz"), names, folds, settings, thresholds=thresholds)
    diag = None if args.visual_dir else DiagnosticWriter(base.with_suffix(".diagnostics.npz"), names, thresholds)
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
            maps, seeds, asymmetry, cycle, edges = reciprocal_maps(
                recorder, selected, torch.stack((word_seed, token_seed)), spec, args.device, args.smoke_diagnostics
            )
            if args.smoke_diagnostics:
                for track, raw_seed, prefix in ((0, word_seed, "W"), (1, token_seed, "T")):
                    expected = transfer.propagate(recorder, selected, raw_seed, args.device)[0]
                    observed = maps[names.index(f"{prefix}/REF")]
                    error = float((cv.normalize_batch(expected[None], "minmax")[0] -
                                   cv.normalize_batch(observed[None], "minmax")[0]).abs().max())
                    if error > 5e-5:
                        raise RuntimeError(f"{prefix}/REF mismatch against existing implementation: {error}")
            recorder.sa_qk.clear()
            if args.visual_dir:
                ca_maps = [seeds[0] if name.startswith("W/") else seeds[1] for name in names]
                render_variant_pages(
                    args.visual_dir, "sg9", classname, row["image_id"],
                    released_image(row, manifest_dir), np.asarray(sample.target_mask, bool),
                    names, ca_maps, list(maps), visual_thresholds,
                )
                print(f"[visual:{classname}] {row['image_id']} cells={len(names)}", flush=True)
                continue
            counts, ap = cv.score_maps(maps, sample.target_mask, "minmax", thresholds=thresholds)
            changes, spatial, mechanism, full = compact_diagnostics(
                maps, seeds, names, np.asarray(sample.target_mask, bool), thresholds, asymmetry, cycle
            )
            assert diag is not None and writer is not None
            diag.add(row["image_id"], changes, spatial, mechanism)
            writer.add(row["image_id"], classname, counts, ap)
            if args.smoke_diagnostics:
                diagnostic_dir = args.output_dir / "diagnostic_panels"
                save_visual(diagnostic_dir / f"{row['image_id']}.png", maps, seeds, names, full, sample.target_mask)
                (diagnostic_dir / f"{row['image_id']}_top_edges.json").write_text(
                    json.dumps(edges, indent=2) + "\n", encoding="utf-8")
            print(f"[{args.dataset}:{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) " +
                  " ".join(f"{name}={counts[names.index(name),t45,0]/max(counts[names.index(name),t45,:3].sum(),1):.3f}"
                           for name in ("W/REF", "W/R1_geometric", "T/REF", "T/R1_geometric")), flush=True)
        finally:
            recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear()
            sample.clear(); del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if writer is not None and args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
