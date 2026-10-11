"""Generate and score focused SA follow-up ablations one VOC-Sim class at a time."""

from __future__ import annotations

import argparse
import gc
import json
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
from src.segmentation.focused_self_attention import run_focused_self_attention_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_full_focused_sa_p1_p4"),
    )
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--model-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--fixed-stage1-results",
        type=Path,
        default=Path("outputs/segmentation/voc_sim_100_stage1_suite/results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    shard_dir = args.output_dir / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = shard_dir / "checkpoint.pt"
    fixed_settings = json.loads(args.fixed_stage1_results.read_text(encoding="utf-8"))["settings"]
    fixed_head_rankings = fixed_settings["head_rankings"]
    signature = {
        "suite": "focused_sa_priorities_1_to_4_v2",
        "manifest": str(args.manifest),
        "model": args.model,
        "classname": classname,
        "class_shard": args.class_shard,
        "storage_dtype": args.storage_dtype,
        "model_dtype": args.model_dtype,
        "cross_map_kind": "ovam_unconditional_query_prompt_softmax",
        "fixed_stage1_results": str(args.fixed_stage1_results),
    }
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu")
        if state["signature"] != signature:
            raise ValueError("Checkpoint settings differ from this invocation")
    else:
        state = {"signature": signature, "completed": [], "metrics": {}, "suite_metadata": None}

    dtype = torch.float16 if args.model_dtype == "float16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.detach().cpu()
    completed = set(state["completed"])

    for row in rows:
        if row["image_id"] in completed:
            continue
        print(f"[{classname}] {row['image_id']} ({len(completed) + 1}/{len(rows)})", flush=True)
        sample = generate_attention_sample(pipe, row, args.manifest.parent, args.storage_dtype)
        try:
            groups, suite_metadata = run_focused_self_attention_suite(
                sample,
                pipe.tokenizer,
                fixed_head_rankings,
                device=args.device,
                alphas_cumprod=alphas_cumprod,
            )
            for group, results in groups.items():
                merge_results(state["metrics"], f"focused/{group}", results)
            if state["suite_metadata"] is None:
                state["suite_metadata"] = suite_metadata
            elif state["suite_metadata"] != suite_metadata:
                raise RuntimeError("Focused-suite metadata changed between samples")
            state["completed"].append(row["image_id"])
            completed.add(row["image_id"])
            save_checkpoint(checkpoint_path, state)
            save_results(
                shard_dir / "results.json",
                state,
                {
                    **signature,
                    "completed": len(completed),
                    "expected": len(rows),
                    "suite_metadata": state["suite_metadata"],
                },
            )
        finally:
            sample.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()

    (shard_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {len(completed)} annotated samples for {classname}")


if __name__ == "__main__":
    main()
