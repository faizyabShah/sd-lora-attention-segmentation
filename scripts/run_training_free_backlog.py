#!/usr/bin/env python3
"""Run every registered, still-unrun training-free Track-A family per image.

One invocation owns one VOC-Sim or COCO-Cap class shard. The image is generated
once, all candidates are scored in memory, and every large tensor is released
before the next image. Only compact confusion-count/AP shards are persisted.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_coco_cap_transfer as transfer  # noqa: E402
import scripts.run_sa_package_b as sa_common  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation import training_free_backlog as ops  # noqa: E402

WORK = 64
SELECTED_SA_MODULE = 15
SPEC_PATH = ROOT / "configs/training_free_backlog.json"
OUT = ROOT / "outputs/segmentation/controlled_attention_cv/training_free_backlog"


def tag(value: float, scale: int = 100) -> str:
    return f"{int(round(value * scale)):03d}"


def configuration_names(spec: dict) -> tuple[str, ...]:
    f = spec["families"]
    names = ["REF/one_pass"]
    names += [f"P3/t{tag(v)}" for v in f["P3"]["t"]]
    names += [f"P5/n{v}" for v in f["P5"]["powers"]]
    names += [f"P9/s{tag(s)}_bg{int(bg)}" for s in f["P9"]["seed_thresholds"] for bg in f["P9"]["background"]]
    names += [f"F2/a{tag(v)}" for v in f["F2"]["alpha"]]
    names += [f"BG1/g{tag(v)}" for v in f["BG1"]["gamma"]]
    names += [f"W1/T{tag(v)}" for v in f["W1"]["temperature"]]
    names += [f"W2/k{v}" for v in f["W2"]["keep_heads"]]
    names += ["W4/converged", "H1/proportional"]
    names += [f"G1/tau{tag(v)}" for v in f["G1"]["kl_thresholds"]]
    names += [f"G4/s{tag(v)}" for v in f["G4"]["cosine_similarity"]]
    names += ["M1/ca_prototype"]
    names += [f"M2/k{v}" for v in f["M2"]["clusters"]]
    names += [f"R1/w{tag(v)}" for v in f["R1"]["pairwise_weight"]]
    return tuple(names)


class BacklogRecorder(sa_common.SAPackageRecorder):
    """Class-word/BOS CA maps and conditional SA Q/K, all held only in memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_calls: dict[str, list[torch.Tensor]] = {}
        if self.attribution_hidden_states is not None and transfer._PIPE is not None:
            prompt = transfer._ATTRIBUTION_PROMPT[0]
            ids = transfer._PIPE.tokenizer(
                prompt, padding="max_length", max_length=transfer._PIPE.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            ).input_ids.to(transfer._PIPE.device)
            with torch.inference_mode():
                self.attribution_hidden_states = transfer._PIPE.text_encoder(ids)[0]

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        del value
        step = self.call_counts[module_name]
        self.call_counts[module_name] += 1
        self.step_tensors.setdefault(step, {})
        if query.shape[0] != 2:
            raise RuntimeError("Expected a [unconditional, conditional] CFG batch")
        with torch.inference_mode():
            if kind == "cross":
                if attribution_key is not None:
                    if attribution_key.shape[-2] != 77:
                        raise RuntimeError(f"Expected a 77-token attribution key, got {tuple(attribution_key.shape)}")
                    key = torch.stack((key[0], attribution_key[0]))
                    attention_mask = None
                scores = query[1].float() @ key[1].float().transpose(-1, -2) * scale
                if attention_mask is not None:
                    scores = scores + attention_mask[1].float()
                probability = scores.softmax(-1)
                target = probability[..., list(self.target_positions)].mean(-1).half()
                self.ca_calls.setdefault(module_name, []).append(target)
                self.start_calls.setdefault(module_name, []).append(probability[..., 0].half())
            else:
                self.sa_qk.setdefault(module_name, []).append(
                    (query[1].detach().half().clone(), key[1].detach().half().clone())
                )
        spatial = math.isqrt(int(query.shape[-2]))
        self.module_metadata[module_name].update({
            "spatial_height": spatial, "spatial_width": spatial,
            "spatial_tokens": int(query.shape[-2]), "head_dim": int(query.shape[-1]),
        })


sa_common.live.CompactAttentionRecorder = BacklogRecorder


@torch.inference_mode()
def aggregate_ca(calls_by_module, modules, resolutions, ca_spec, device) -> torch.Tensor:
    total = torch.zeros((WORK, WORK), device=device)
    count = 0
    for index in ca_spec["modules"]:
        name, resolution = modules[index], resolutions[modules[index]]
        for call in ca_spec["calls"]:
            value = calls_by_module[name][call].float()[ca_spec["heads"]].mean(0).reshape(1, 1, resolution, resolution)
            if resolution != WORK:
                value = F.interpolate(value, (WORK, WORK), mode="bilinear", align_corners=False)
            total += value[0, 0]
            count += 1
    return ops.minmax(total / count)


@torch.inference_mode()
def selected_statistics(recorder, module: str, seed64: torch.Tensor, device: str):
    """Mean graph, all propagated observations, and query features."""
    pixels = seed64.numel()
    calls = recorder.sa_qk[module]
    graph = torch.zeros((pixels, pixels), device=device)
    observations = torch.empty((len(calls), 8, pixels), device=device)
    query_sum = None
    seed = seed64.reshape(-1).float()
    for call_index, (query16, key16) in enumerate(calls):
        query, key = query16.to(device).float(), key16.to(device).float()
        affinity = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1]), -1)
        graph += affinity.mean(0)
        observations[call_index] = affinity @ seed
        query_sum = query.clone() if query_sum is None else query_sum + query
        del query, key, affinity
    graph /= len(calls)
    return graph, observations, query_sum


@torch.inference_mode()
def mean_graph(recorder, module: str, device: str) -> torch.Tensor:
    calls = recorder.sa_qk[module]
    pixels = calls[0][0].shape[-2]
    output = torch.zeros((pixels, pixels), device=device)
    for query16, key16 in calls:
        query, key = query16.to(device).float(), key16.to(device).float()
        output += torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1]), -1).mean(0)
        del query, key
    return output / len(calls)


def resize_vector(vector: torch.Tensor, source: int, destination: int, mode: str = "bilinear") -> torch.Tensor:
    if source == destination:
        return vector.reshape(-1)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(vector.reshape(1, 1, source, source), (destination, destination), mode=mode, **kwargs)[0, 0].reshape(-1)


@torch.inference_mode()
def progressive(graphs: dict[int, torch.Tensor], seed64: torch.Tensor, threshold: float) -> torch.Tensor:
    fixed = seed64.reshape(-1).float() * (seed64.reshape(-1) >= threshold)
    x = fixed.clone()
    for resolution in sorted(graphs):
        local = resize_vector(x, WORK, resolution, "area")
        x = resize_vector(graphs[resolution] @ local, resolution, WORK)
        x = torch.maximum(ops.minmax(x), fixed)
    return x


@torch.inference_mode()
def hierarchical(graphs: dict[int, torch.Tensor], seed64: torch.Tensor) -> torch.Tensor:
    values, weights = [], []
    for resolution in sorted(graphs):
        local = resize_vector(seed64, WORK, resolution, "area")
        values.append(resize_vector(graphs[resolution] @ local, resolution, WORK))
        weights.append(resolution)
    weight = torch.tensor(weights, device=seed64.device, dtype=torch.float32)
    weight /= weight.sum()
    return sum(w * value for w, value in zip(weight, values))


@torch.inference_mode()
def evaluate(recorder, seed64: torch.Tensor, start64: torch.Tensor, sa_modules, sa_resolutions,
             target: np.ndarray, spec: dict, device: str):
    f = spec["families"]
    names = configuration_names(spec)
    selected = sa_modules[SELECTED_SA_MODULE]
    graph, observation_cube, query_sum = selected_statistics(recorder, selected, seed64, device)
    observations = observation_cube.reshape(-1, WORK * WORK)
    reference = observations.mean(0)
    background = graph @ start64.reshape(-1)
    features = ops.query_features(query_sum, len(recorder.sa_qk[selected]))

    representative = {}
    for resolution in f["P9"]["path"]:
        choices = [name for name in sa_modules if sa_resolutions[name] == resolution]
        representative[resolution] = choices[-1]
    graphs = {}
    for resolution, module in representative.items():
        graphs[resolution] = graph if module == selected else mean_graph(recorder, module, device)

    continuous: dict[str, torch.Tensor] = {"REF/one_pass": reference}
    binary: dict[str, torch.Tensor] = {}

    for value in f["P3"]["t"]:
        continuous[f"P3/t{tag(value)}"] = ops.heat_kernel(graph, seed64, float(value), int(f["P3"]["terms"]))
    for power, value in zip(f["P5"]["powers"], ops.symmetric_power(graph, seed64, tuple(f["P5"]["powers"]))):
        continuous[f"P5/n{power}"] = value
    bg_progressive = progressive(graphs, start64, 0.0)
    for threshold in f["P9"]["seed_thresholds"]:
        value = progressive(graphs, seed64, float(threshold))
        for use_background in f["P9"]["background"]:
            name = f"P9/s{tag(threshold)}_bg{int(use_background)}"
            continuous[name] = value * (1 - ops.minmax(bg_progressive)) if use_background else value
    column_mass = ops.minmax(graph.sum(0))
    for alpha in f["F2"]["alpha"]:
        gate = float(alpha) + (1 - float(alpha)) * column_mass
        continuous[f"F2/a{tag(alpha)}"] = reference * gate
    for gamma in f["BG1"]["gamma"]:
        binary[f"BG1/g{tag(gamma)}"] = reference > float(gamma) * background
    for temperature in f["W1"]["temperature"]:
        continuous[f"W1/T{tag(temperature)}"] = ops.entropy_weighted(observations, float(temperature))
    for keep in f["W2"]["keep_heads"]:
        continuous[f"W2/k{keep}"] = ops.agreement_heads(observation_cube, seed64, int(keep))
    converged, _, _ = ops.iterate_to_convergence(
        graph, seed64, float(f["W4"]["tolerance"]), int(f["W4"]["max_iterations"])
    )
    continuous["W4/converged"] = converged
    continuous["H1/proportional"] = hierarchical(graphs, seed64)

    for threshold in f["G1"]["kl_thresholds"]:
        labels = ops.diffseg_labels(
            graph, float(threshold), int(f["G1"]["anchors_per_axis"]), int(f["G1"]["merge_iterations"])
        )
        continuous[f"G1/tau{tag(threshold)}"] = ops.region_scores(labels, reference)
    for similarity in f["G4"]["cosine_similarity"]:
        binary[f"G4/s{tag(similarity)}"] = ops.seeded_region_growing(features, seed64, float(similarity))
    continuous["M1/ca_prototype"] = ops.prototype_score(features, seed64)
    for clusters in f["M2"]["clusters"]:
        continuous[f"M2/k{clusters}"] = ops.multiple_prototype_score(
            features, seed64, int(clusters), int(f["M2"]["kmeans_iterations"])
        )
    for weight in f["R1"]["pairwise_weight"]:
        continuous[f"R1/w{tag(weight)}"] = ops.sa_mean_field(
            graph, reference, float(weight), int(f["R1"]["mean_field_iterations"])
        )
    thresholds = tuple(float(v) for v in spec["thresholds"])
    continuous_names = [name for name in names if name in continuous]
    maps = torch.stack([continuous[name].reshape(WORK, WORK) for name in continuous_names])
    continuous_counts, continuous_ap = cv.score_maps(maps, target, "minmax", thresholds=thresholds, chunk=8)
    binary_names = [name for name in names if name in binary]
    if binary_names:
        masks64 = torch.stack([binary[name].reshape(WORK, WORK) for name in binary_names]).float()
        masks = F.interpolate(masks64[:, None], size=np.asarray(target).shape, mode="nearest")[:, 0].bool()
        binary_counts = cv.score_masks(masks.cpu(), target, thresholds=thresholds)
    counts = np.empty((len(names), len(thresholds), 4), dtype=np.int64)
    ap = np.full(len(names), np.nan, dtype=np.float64)
    for i, name in enumerate(continuous_names):
        destination = names.index(name); counts[destination] = continuous_counts[i]; ap[destination] = continuous_ap[i]
    for i, name in enumerate(binary_names):
        counts[names.index(name)] = binary_counts[i]
    if set(names) != set(continuous) | set(binary):
        raise RuntimeError("Not every registered configuration was evaluated")
    return counts, ap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--ca-seed", type=Path, default=ROOT / "configs/controlled_sa_seed_frozen.json")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> None:
    global_pipe = StableDiffusionPipeline
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    ca_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not ca_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    voc = args.dataset == "voc"
    classname, rows = (transfer.voc_rows if voc else transfer.coco_rows)(args.class_shard)
    if args.max_samples:
        rows = rows[: args.max_samples]
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
    pipe = global_pipe.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    transfer._PIPE = pipe
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    if len(ca_modules) != 16 or len(sa_modules) != 16:
        raise RuntimeError("Expected 16 CA and 16 SA modules")
    ca_spec = sa_common.resolve_seed_spec(ca_json, ca_modules, 31)
    names, thresholds = configuration_names(spec), tuple(float(v) for v in spec["thresholds"])
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if voc else {row["image_id"]: 0 for row in rows}
    settings = {
        "package": "training_free_backlog_v1", "protocol": spec["protocol"], "dataset": args.dataset,
        "spec": spec, "ca_seed": ca_spec, "selected_sa_module_index": SELECTED_SA_MODULE,
        "selected_sa_module": sa_modules[SELECTED_SA_MODULE], "configurations": list(names),
        "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
        "attribution": "class word only; COCO uses separate padded class-word keys",
    }
    stem = f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    path = args.output_dir / args.dataset / "shards" / f"{stem}.npz"
    writer = cv.ShardWriter(path, names, folds, settings, thresholds=thresholds)
    print(f"{args.dataset}/{classname}: {len(rows)} images, {len(writer.completed)} resumed, "
          f"{len(names)} cells, node={socket.gethostname()}", flush=True)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        started = time.time()
        transfer._ATTRIBUTION_PROMPT[0] = row["attribution_prompt"]
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(pipe, row, manifest_dir, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_resolutions = {name: resolution for resolution, modules in sample.modules_by_resolution.items() for name in modules}
            sa_resolutions = {name: resolution for resolution, modules in sample.self_modules_by_resolution.items() for name in modules}
            seed64 = aggregate_ca(recorder.ca_calls, ca_modules, ca_resolutions, ca_spec, args.device)
            start64 = aggregate_ca(recorder.start_calls, ca_modules, ca_resolutions, ca_spec, args.device)
            recorder.ca_calls.clear(); recorder.start_calls.clear()
            counts, ap = evaluate(recorder, seed64, start64, sa_modules, sa_resolutions, sample.target_mask, spec, args.device)
            writer.add(row["image_id"], classname, counts, ap)
            ref = counts[names.index("REF/one_pass"), thresholds.index(0.45)]
            print(f"{row['image_id']} ({len(writer.completed)}/{len(rows)}) ref@.45={ref[0] / max(ref[:3].sum(), 1):.3f} "
                  f"elapsed={time.time() - started:.1f}s", flush=True)
        finally:
            recorder.ca_calls.clear(); recorder.start_calls.clear(); recorder.sa_qk.clear()
            sample.clear()
            del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if args.max_samples is None:
        if len(writer.completed) != len(rows):
            raise RuntimeError(f"Shard contains {len(writer.completed)} of {len(rows)} expected images")
        writer.finish()


if __name__ == "__main__":
    main()
