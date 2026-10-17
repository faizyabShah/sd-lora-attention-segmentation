#!/usr/bin/env python3
"""Optimize one registered B2a or B2b token task with OVAM's recipe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.token_optimization import (  # noqa: E402
    binary_target, capture_optimization_queries, initial_ovam_embedding,
    optimize_ovam_embedding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("b2a", "b2b"), required=True)
    parser.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--plan", type=Path, default=Path("configs/token_optimization_b2_plan.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/segmentation/controlled_attention_cv/token_learning_b2"))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    task = plan[args.track][args.task_id]
    out = args.output_dir / args.track / "tokens" / f"{task['id']}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with np.load(out, allow_pickle=False) as saved:
            if bool(saved["complete"]):
                print(f"already complete: {out}")
                return
    recipe = plan["recipe"]
    pipe = StableDiffusionPipeline.from_pretrained(
        recipe["model"], torch_dtype=torch.float32, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.requires_safety_checker = False
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    prompt = task["prompt"]
    recorder = capture_optimization_queries(pipe, prompt, int(task["seed"]), int(recipe["steps"]))
    initial = initial_ovam_embedding(pipe, prompt)
    target = binary_target(ROOT / task["annotation"], "cuda")
    epochs = args.epochs or int(recipe["epochs"])
    result = optimize_ovam_embedding(
        pipe, recorder, initial, target, epochs=epochs,
        initial_lr=float(recipe["initial_lr"]), step_size=int(recipe["step_size"]),
        gamma=float(recipe["gamma"]),
        progress=lambda epoch, loss: print(
            f"{args.track}[{args.task_id}] epoch={epoch + 1}/{epochs} loss={loss:.6f}", flush=True
        ) if epoch % 40 == 0 or epoch + 1 == epochs else None,
    )
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, complete=np.asarray(True), embedding=result.embedding.numpy(),
        losses=result.losses, best_loss=np.asarray(result.best_loss),
        best_epoch=np.asarray(result.best_epoch), task_json=np.asarray(json.dumps(task, sort_keys=True)),
        recipe_json=np.asarray(json.dumps(recipe, sort_keys=True)),
    )
    os.replace(tmp, out)
    print(f"saved {out}; best epoch={result.best_epoch} loss={result.best_loss:.6f}")


if __name__ == "__main__":
    main()
