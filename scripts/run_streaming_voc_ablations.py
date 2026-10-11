"""Generate VOC-Sim samples and evaluate all attention ablations in memory."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention_capture.live import generate_attention_sample
from src.segmentation.ablation_runner import (
    SegmentationSettings,
    run_cross_attention_layer_ablation,
    run_resolution_ablation,
    run_threshold_normalization_ablation,
    run_time_resolution_ablation,
    run_time_window_ablation,
)
from src.segmentation.exhaustive_layer_search import run_exhaustive_layer_search
from src.segmentation.in_memory_dataset import SingleSampleDataset
from src.segmentation.metrics import BinarySegmentationMetrics
from src.segmentation.ovam_baseline import ovam_mask
from src.segmentation.self_attention_refinement import (
    run_progressive_self_attention,
    run_self_attention_screen,
)
from src.segmentation.stage1_suite import BEST_12_INDICES, run_stage1_suite


TIME_WINDOWS = {
    "all_00_30": range(31),
    "steps_00_04": range(0, 5),
    "steps_05_09": range(5, 10),
    "steps_10_14": range(10, 15),
    "steps_15_20": range(15, 21),
    "steps_21_25": range(21, 26),
    "steps_26_30": range(26, 31),
}
RESOLUTION_CONFIGS = {
    "r" + "_r".join(map(str, values)): values
    for width in range(1, 5)
    for values in itertools.combinations((8, 16, 32, 64), width)
}
PROGRESSIVE_SEQUENCES = (
    (8, 16), (16, 32), (32, 64), (8, 16, 32), (16, 32, 64), (8, 16, 32, 64),
    (64, 32), (32, 16), (16, 8), (64, 32, 16), (32, 16, 8), (64, 32, 16, 8),
)
SUITE_NAMES = ("ovam", "basic", "stage1", "exhaustive", "sa", "progressive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation/voc_sim_full_streaming"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--model-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--exhaustive-batch-size", type=int, default=256)
    parser.add_argument("--exhaustive-summary", type=Path, default=Path("outputs/segmentation/voc_sim_100_exhaustive_layer_subsets/summary.json"))
    parser.add_argument("--fixed-stage1-results", type=Path, default=Path("outputs/segmentation/voc_sim_100_stage1_suite/results.json"))
    parser.add_argument("--suites", nargs="+", choices=SUITE_NAMES, default=SUITE_NAMES)
    return parser.parse_args()


def read_rows(path: Path, class_shard: int, max_samples: int | None) -> tuple[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    classes = sorted({row["classname"] for row in rows})
    classname = classes[class_shard]
    selected = [row for row in rows if row["classname"] == classname and row["annotated"].lower() == "true"]
    return classname, selected[:max_samples] if max_samples is not None else selected


def layer_configurations(sample) -> dict[str, tuple[str, ...]]:
    by_resolution = {name: size for size, names in sample.modules_by_resolution.items() for name in names}
    modules = tuple(sorted(by_resolution))
    down = tuple(name for name in modules if name.startswith("down_blocks."))
    mid = tuple(name for name in modules if name.startswith("mid_block."))
    up = tuple(name for name in modules if name.startswith("up_blocks."))
    output = {
        "all_layers": modules,
        "non_middle_layers": down + up,
        "down_only": down,
        "mid_only": mid,
        "up_only": up,
        "down_mid": down + mid,
        "mid_up": mid + up,
        "down_up": down + up,
    }
    output.update({f"layer_{index:02d}": (name,) for index, name in enumerate(modules)})
    return output


def threshold_layers(sample) -> dict[str, tuple[str, ...]]:
    modules = tuple(sorted(name for names in sample.modules_by_resolution.values() for name in names))
    return {
        "best_12_layers": tuple(modules[index] for index in BEST_12_INDICES),
        "all_16_layers": modules,
        "non_middle_15_layers": tuple(name for name in modules if not name.startswith("mid_block")),
    }


def _merge_leaf(state: dict, experiment: str, configuration: str, result: dict) -> None:
    destination = state.setdefault(experiment, {}).setdefault(configuration, {})
    for classname, values in result["per_class"].items():
        item = destination.setdefault(classname, {key: 0 for key in ("tp", "fp", "fn", "tn", "images")})
        for key in ("tp", "fp", "fn", "tn", "images"):
            item[key] += int(values[key])
        item["image_iou_sum"] = item.get("image_iou_sum", 0.0) + float(values["mean_image_iou"]) * int(values["images"])


def merge_results(state: dict, experiment: str, results: dict[str, dict]) -> None:
    for configuration, result in results.items():
        _merge_leaf(state, experiment, configuration, result)


def merge_nested_results(state: dict, prefix: str, groups: dict[str, dict]) -> None:
    for group, results in groups.items():
        merge_results(state, f"{prefix}/{group}", results)


def finalized(metrics: dict) -> dict:
    output: dict[str, dict] = {}
    for experiment, configs in metrics.items():
        output[experiment] = {}
        for configuration, classes in configs.items():
            per_class = {}
            for classname, value in sorted(classes.items()):
                foreground_union = value["tp"] + value["fp"] + value["fn"]
                background_union = value["tn"] + value["fp"] + value["fn"]
                per_class[classname] = {
                    **value,
                    "iou": value["tp"] / foreground_union if foreground_union else 1.0,
                    "background_iou": value["tn"] / background_union if background_union else 1.0,
                    "mean_image_iou": value["image_iou_sum"] / value["images"],
                }
            output[experiment][configuration] = {
                "miou": sum(v["iou"] for v in per_class.values()) / len(per_class),
                "mean_background_iou": sum(v["background_iou"] for v in per_class.values()) / len(per_class),
                "mean_image_iou": sum(v["image_iou_sum"] for v in classes.values()) / sum(v["images"] for v in classes.values()),
                "images": sum(v["images"] for v in classes.values()),
                "classes": len(per_class),
                "per_class": per_class,
            }
    return output


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_results(path: Path, state: dict, metadata: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"settings": metadata, "results": finalized(state["metrics"])}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate_sample(sample, tokenizer, args) -> tuple[dict, Any | None]:
    dataset = SingleSampleDataset(sample)
    ovam_settings = SegmentationSettings(threshold=0.4, normalization="ovam_max_after_resize", branch=1, working_size=64)
    calibrated_settings = SegmentationSettings(threshold=0.45, normalization="percentile_01_99", branch=1, working_size=64)
    local: dict = {}

    if "ovam" in args.suites:
        metric = BinarySegmentationMetrics()
        target = sample.load_target_mask()
        metric.update(sample.classname, ovam_mask(sample, target.shape), target)
        merge_results(local, "baseline", {"ovam_unoptimized": metric.result()})

    if "basic" in args.suites:
        # These change one OVAM aggregation axis while keeping OVAM's max
        # normalization and 0.4 threshold fixed. Excluding r8 removes the mid
        # block exactly as StableDiffusionHooker does by default.
        merge_results(local, "time_window", run_time_window_ablation(dataset, tokenizer, TIME_WINDOWS, (16, 32, 64), ovam_settings))
        merge_results(local, "resolution", run_resolution_ablation(dataset, tokenizer, RESOLUTION_CONFIGS, ovam_settings))
        merge_results(local, "time_resolution", run_time_resolution_ablation(dataset, tokenizer, TIME_WINDOWS, RESOLUTION_CONFIGS, ovam_settings))
        merge_results(local, "layers", run_cross_attention_layer_ablation(dataset, tokenizer, layer_configurations(sample), ovam_settings))
        merge_results(local, "threshold_normalization", run_threshold_normalization_ablation(
            dataset, tokenizer, threshold_layers(sample), tuple(value / 100 for value in range(10, 91, 5)),
            ("ovam_max_after_resize", "minmax", "max", "percentile_01_99"), ovam_settings,
        ))

    if "stage1" in args.suites:
        fixed_metadata = json.loads(args.fixed_stage1_results.read_text(encoding="utf-8"))["settings"]
        results, _ = run_stage1_suite(
            dataset, tokenizer, calibrated_settings, args.exhaustive_summary,
            dataset_adaptive=False, fixed_adaptive_metadata=fixed_metadata,
        )
        merge_nested_results(local, "stage1", results)

    exhaustive = None
    if "exhaustive" in args.suites:
        exhaustive = run_exhaustive_layer_search(dataset, tokenizer, SegmentationSettings(threshold=0.45, normalization="minmax", branch=1, working_size=64), args.device, args.exhaustive_batch_size)

    if "sa" in args.suites:
        for resolution in (8, 16, 32, 64):
            results, _ = run_self_attention_screen(dataset, tokenizer, resolution, args.device)
            merge_results(local, f"self_attention/r{resolution}", results)

    if "progressive" in args.suites:
        for sequence in PROGRESSIVE_SEQUENCES:
            results, _ = run_progressive_self_attention(dataset, tokenizer, sequence, args.device)
            merge_results(local, "self_attention/progressive_" + "-".join(map(str, sequence)), results)
    return local, exhaustive


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    shard_dir = args.output_dir / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = shard_dir / "checkpoint.pt"
    signature = {
        "manifest": str(args.manifest), "model": args.model, "classname": classname,
        "class_shard": args.class_shard, "suites": sorted(args.suites),
        "storage_dtype": args.storage_dtype, "model_dtype": args.model_dtype,
        "cross_map_kind": "ovam_unconditional_query_prompt_softmax",
        "exhaustive_summary": str(args.exhaustive_summary),
        "fixed_stage1_results": str(args.fixed_stage1_results),
    }
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {"signature": signature, "completed": [], "metrics": {}, "exhaustive": None}

    dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    completed = set(state["completed"])

    for row in rows:
        if row["image_id"] in completed:
            continue
        print(f"[{classname}] {row['image_id']} ({len(completed) + 1}/{len(rows)})", flush=True)
        sample = generate_attention_sample(pipe, row, args.manifest.parent, args.storage_dtype)
        try:
            local_metrics, exhaustive = evaluate_sample(sample, pipe.tokenizer, args)
            for experiment, configs in local_metrics.items():
                for configuration, classes in configs.items():
                    destination = state["metrics"].setdefault(experiment, {}).setdefault(configuration, {})
                    for class_name, values in classes.items():
                        item = destination.setdefault(class_name, {key: 0 for key in ("tp", "fp", "fn", "tn", "images")})
                        for key in ("tp", "fp", "fn", "tn", "images"):
                            item[key] += values[key]
                        item["image_iou_sum"] = item.get("image_iou_sum", 0.0) + values["image_iou_sum"]
            if exhaustive is not None:
                if state["exhaustive"] is None:
                    state["exhaustive"] = {
                        "module_names": exhaustive.module_names,
                        "tp": exhaustive.tp[:, 0], "fp": exhaustive.fp[:, 0],
                        "fn": exhaustive.fn[:, 0], "tn": exhaustive.tn[:, 0],
                        "image_iou_sum": exhaustive.image_iou_sum[:, 0],
                        "images": exhaustive.image_counts[0],
                    }
                else:
                    for key in ("tp", "fp", "fn", "tn", "image_iou_sum"):
                        # Exhaustive outputs are created under inference_mode.
                        # An out-of-place sum produces an ordinary tensor and is
                        # also safe when resuming tensors loaded from checkpoint.
                        state["exhaustive"][key] = (
                            state["exhaustive"][key]
                            + getattr(exhaustive, key)[:, 0].cpu()
                        )
                    state["exhaustive"]["images"] = (
                        state["exhaustive"]["images"]
                        + exhaustive.image_counts[0].cpu()
                    )
            state["completed"].append(row["image_id"])
            completed.add(row["image_id"])
            save_checkpoint(checkpoint_path, state)
            save_results(shard_dir / "results.json", state, {**signature, "completed": len(completed), "expected": len(rows)})
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {len(completed)} annotated samples for {classname}")


if __name__ == "__main__":
    main()
