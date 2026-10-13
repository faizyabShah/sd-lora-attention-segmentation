#!/usr/bin/env python3
"""Run controlled CA-only ablations on one full VOC-Sim class shard."""

from __future__ import annotations

import argparse
import gc
import itertools
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_streaming_voc_ablations import (
    finalized,
    merge_results,
    read_rows,
    save_checkpoint,
    save_results,
)
from src.attention_capture.live import generate_attention_sample
from src.segmentation.ablation_runner import (
    SegmentationSettings,
    run_cross_attention_head_ablation,
    run_cross_attention_layer_ablation,
    run_resolution_ablation,
    run_time_resolution_ablation,
    run_time_window_ablation,
)
from src.segmentation.exhaustive_layer_search import run_exhaustive_layer_search
from src.segmentation.in_memory_dataset import SingleSampleDataset


def time_windows() -> dict[str, tuple[int, ...]]:
    windows: dict[str, tuple[int, ...]] = {"all_00_30": tuple(range(31))}
    windows.update({f"step_{step:02d}": (step,) for step in range(31)})
    for start, end in ((0, 4), (5, 9), (10, 14), (15, 20), (21, 25), (26, 30)):
        windows[f"window_{start:02d}_{end:02d}"] = tuple(range(start, end + 1))
    for start in (5, 10, 15, 21, 26):
        windows[f"trailing_{start:02d}_30"] = tuple(range(start, 31))
    for end in (4, 9, 14, 20, 25):
        windows[f"cumulative_00_{end:02d}"] = tuple(range(end + 1))
    windows.update(
        {
            f"leave_step_{step:02d}": tuple(i for i in range(31) if i != step)
            for step in range(31)
        }
    )
    return windows


def resolution_configurations() -> dict[str, tuple[int, ...]]:
    return {
        "r" + "_r".join(map(str, values)): values
        for width in range(1, 5)
        for values in itertools.combinations((8, 16, 32, 64), width)
    }


def layer_configurations(sample) -> dict[str, tuple[str, ...]]:
    modules = tuple(
        sorted(name for names in sample.modules_by_resolution.values() for name in names)
    )
    down = tuple(name for name in modules if name.startswith("down_blocks."))
    middle = tuple(name for name in modules if name.startswith("mid_block."))
    up = tuple(name for name in modules if name.startswith("up_blocks."))
    configs = {
        "all_16": modules,
        "non_middle_15": down + up,
        "down_only": down,
        "middle_only": middle,
        "up_only": up,
        "down_middle": down + middle,
        "middle_up": middle + up,
    }
    configs.update({f"layer_{i:02d}": (name,) for i, name in enumerate(modules)})
    configs.update(
        {
            f"leave_layer_{i:02d}": tuple(
                value for index, value in enumerate(modules) if index != i
            )
            for i in range(len(modules))
        }
    )
    return configs


def head_configurations() -> dict[str, tuple[int, ...]]:
    return {
        "heads_" + "_".join(map(str, values)): values
        for width in range(1, 9)
        for values in itertools.combinations(range(8), width)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/"
            "ovam_voc_sim.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_clean_ca_ablations"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--model-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--normalization", choices=("minmax",), default="minmax")
    parser.add_argument("--exhaustive-batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def evaluate_sample(sample, tokenizer, args):
    dataset = SingleSampleDataset(sample)
    settings = SegmentationSettings(
        threshold=args.threshold,
        normalization=args.normalization,
        branch=1,
        working_size=64,
    )
    windows = time_windows()
    resolutions = resolution_configurations()
    local: dict = {}
    merge_results(
        local,
        "time",
        run_time_window_ablation(dataset, tokenizer, windows, (8, 16, 32, 64), settings),
    )
    merge_results(
        local,
        "resolution",
        run_resolution_ablation(dataset, tokenizer, resolutions, settings),
    )
    merge_results(
        local,
        "time_resolution",
        run_time_resolution_ablation(dataset, tokenizer, windows, resolutions, settings),
    )
    merge_results(
        local,
        "layers_structured",
        run_cross_attention_layer_ablation(
            dataset, tokenizer, layer_configurations(sample), settings
        ),
    )
    merge_results(
        local,
        "heads",
        run_cross_attention_head_ablation(
            dataset, tokenizer, head_configurations(), settings
        ),
    )
    exhaustive = run_exhaustive_layer_search(
        dataset,
        tokenizer,
        settings,
        args.device,
        args.exhaustive_batch_size,
    )
    return local, exhaustive


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    shard_dir = args.output_dir / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = shard_dir / "checkpoint.pt"
    signature = {
        "suite": "clean_ca_only_v1",
        "manifest": str(args.manifest),
        "model": args.model,
        "classname": classname,
        "class_shard": args.class_shard,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "token_attribution": "all tokenizer pieces of class name",
        "branch": "conditional",
        "normalization": args.normalization,
        "threshold": args.threshold,
        "working_size": 64,
        "postprocessing": "none",
        "self_attention": "none",
        "densecrf": "none",
    }
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {"signature": signature, "completed": [], "metrics": {}, "exhaustive": None}
    completed = set(state["completed"])

    dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    for row in rows:
        if row["image_id"] in completed:
            continue
        print(
            f"[{classname}] {row['image_id']} ({len(completed) + 1}/{len(rows)})",
            flush=True,
        )
        sample = generate_attention_sample(
            pipe, row, args.manifest.parent, args.storage_dtype, cross_map_kind="conditional"
        )
        try:
            local, exhaustive = evaluate_sample(sample, pipe.tokenizer, args)
            for experiment, configs in local.items():
                for configuration, classes in configs.items():
                    destination = state["metrics"].setdefault(experiment, {}).setdefault(
                        configuration, {}
                    )
                    for class_name, values in classes.items():
                        item = destination.setdefault(
                            class_name,
                            {key: 0 for key in ("tp", "fp", "fn", "tn", "images")},
                        )
                        for key in ("tp", "fp", "fn", "tn", "images"):
                            item[key] += int(values[key])
                        item["image_iou_sum"] = item.get("image_iou_sum", 0.0) + float(
                            values["image_iou_sum"]
                        )
            if state["exhaustive"] is None:
                state["exhaustive"] = {
                    "module_names": exhaustive.module_names,
                    "tp": exhaustive.tp[:, 0].cpu(),
                    "fp": exhaustive.fp[:, 0].cpu(),
                    "fn": exhaustive.fn[:, 0].cpu(),
                    "tn": exhaustive.tn[:, 0].cpu(),
                    "image_iou_sum": exhaustive.image_iou_sum[:, 0].cpu(),
                    "images": int(exhaustive.image_counts[0]),
                }
            else:
                for key in ("tp", "fp", "fn", "tn", "image_iou_sum"):
                    state["exhaustive"][key] = (
                        state["exhaustive"][key] + getattr(exhaustive, key)[:, 0].cpu()
                    )
                state["exhaustive"]["images"] += int(exhaustive.image_counts[0])
            state["completed"].append(row["image_id"])
            completed.add(row["image_id"])
            save_checkpoint(checkpoint, state)
            save_results(
                shard_dir / "results.json",
                state,
                {**signature, "completed": len(completed), "expected": len(rows)},
            )
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {len(completed)} clean CA-only samples for {classname}")


if __name__ == "__main__":
    main()
