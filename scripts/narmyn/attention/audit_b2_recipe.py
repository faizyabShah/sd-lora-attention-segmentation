#!/usr/bin/env python3
"""Audit of the B2a optimizer against OVAM (protocol Section 64, request 1). Diagnostic only.

OVAM released, for each of its 57 VOC training images, the image, the
optimized token and the loss of every epoch (``*_losses.npy``). The loss at
epoch 0 depends only on the generation, the initial token and the attention-map
definition, not on the optimizer. For each training image this script:

  1. regenerates the image as OVAM's ``experiments/voc_simple`` scripts do
     (fp32, 30 steps, guidance 7.5, ``set_seed``) and compares it with the
     released PNG;
  2. builds OVAM's initial token (``encode_text(last word, prompt)[(0, -2)]``,
     as ``src/segmentation/token_optimization.initial_ovam_embedding``);
  3. computes the epoch-0 loss (global min-max, cross-entropy against
     [background, object]) for every combination of
       activation  linear | token_softmax  (softmax over the two tokens per head and call)
       branch      unconditional | conditional query
       middle      excluded | included
     and compares it with OVAM's recorded epoch-0 loss.

The B2 recipe (Codex, Section 63) is ``linear / unconditional / excluded``.
One JSON line per image in ``--output``.

With ``--optimize N``, N images spread over the classes are also optimized for OVAM's 800
epochs (SGD lr 100, StepLR 120 / 0.7, best pre-update state, as
``ovam/optimize/optimize.py``) with each activation (unconditional, no middle
block), and the loss curve and best token are compared with OVAM's.

    python scripts/narmyn/attention/audit_b2_recipe.py --output /path/audit.jsonl
    python scripts/narmyn/attention/audit_b2_recipe.py --output /path/optimize.jsonl --optimize 6
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention_capture.live import _install  # noqa: E402
from src.segmentation.token_optimization import binary_target, initial_ovam_embedding, seed_everything  # noqa: E402

TRAIN = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim_train"
VARIANTS = tuple(itertools.product(("linear", "token_softmax"), ("uncond", "cond"), ("no_mid", "mid")))


class QueryRecorder:
    """Every cross-attention query of both CFG branches, per module and call."""

    def __init__(self):
        self.attribution_hidden_states = None
        self.module_metadata, self.expected_modules, self.call_counts = {}, set(), {}
        self.queries: dict[str, list[torch.Tensor]] = {}

    def register_module(self, name, kind, heads, scale, processor_class):
        self.expected_modules.add(name)
        self.call_counts[name] = 0
        self.module_metadata[name] = {"kind": kind, "heads": heads, "scale": scale}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        self.call_counts[module_name] += 1
        if kind == "cross":
            self.module_metadata[module_name]["scale"] = float(scale)
            self.queries.setdefault(module_name, []).append(query.detach().float().clone())  # [2, heads, P, d]

    def flush_step(self, *args):
        pass

    def finalize(self):
        pass


def heatmap512(pipe, recorder, embedding, activation, branch="uncond", middle="no_mid") -> torch.Tensor:
    b = 0 if branch == "uncond" else 1
    maps = []
    for name, calls in recorder.queries.items():
        if middle == "no_mid" and name.startswith("mid_block."):
            continue
        module = pipe.unet.get_submodule(name)
        key = module.to_k(embedding[None])
        key = key.view(1, -1, int(module.heads), key.shape[-1] // int(module.heads)).permute(0, 2, 1, 3)[0]
        total = 0
        for q in calls:
            logits = torch.einsum("hpd,htd->htp", q[b], key) * recorder.module_metadata[name]["scale"]  # [heads, 2, P]
            total = total + (logits.softmax(dim=1) if activation == "token_softmax" else logits)
        r = math.isqrt(total.shape[-1])
        maps.append(total.sum(dim=0).reshape(1, 2, r, r))
    size = max(m.shape[-1] for m in maps)
    merged = torch.stack([m if m.shape[-1] == size else F.interpolate(m, size=(size, size), mode="bilinear") for m in maps]).sum(0)
    heatmap = F.interpolate(merged, size=(512, 512), mode="bilinear")
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())


@torch.no_grad()
def epoch0_loss(pipe, recorder, embedding, target, activation, branch, middle) -> float:
    return float(F.cross_entropy(heatmap512(pipe, recorder, embedding, activation, branch, middle), target))


def optimize(pipe, recorder, initial, target, activation, epochs=800):
    """``ovam/optimize/optimize.py`` with OVAM's VOC settings; best state recorded before each update."""
    x = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=100.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=120, gamma=0.7)
    losses, best, best_loss = [], None, float("inf")
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(heatmap512(pipe, recorder, x, activation), target)
        losses.append(float(loss))
        if losses[-1] < best_loss:
            best_loss, best = losses[-1], x.detach().clone()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return best, np.asarray(losses)


def main() -> None:
    from diffusers import StableDiffusionPipeline

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--optimize", type=int, help="optimize the first N images with each activation")
    args = parser.parse_args()
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    done = {json.loads(l)["id"] for l in args.output.open()} if args.output.exists() else set()
    losses = sorted(TRAIN.glob("tokens/*_losses.npy"))
    # --optimize: N images spread over the classes (every 57 // N-th); otherwise the first --limit.
    losses = losses[:: max(1, len(losses) // args.optimize)][: args.optimize] if args.optimize else losses[: args.limit]
    with args.output.open("a") as log:
        for path in losses:
            m = re.fullmatch(r"(sd-15_(.+)_(\d+))_\d+_losses", path.stem)
            tid, classname, seed = m.group(1), m.group(2).replace("-", " "), int(m.group(3))
            if tid in done:
                continue
            prompt = f"A photograph of a {classname}"
            recorder = QueryRecorder()
            _install(pipe, recorder)
            seed_everything(seed)  # OVAM's set_seed: random, numpy, torch, torch.cuda
            with torch.no_grad():
                image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, height=512, width=512).images[0]
            ours = np.asarray(image.convert("RGB")).astype(np.int16)
            released = np.asarray(Image.open(TRAIN / "images" / f"{tid}.png").convert("RGB")).astype(np.int16)
            diff = np.abs(ours - released)
            initial = initial_ovam_embedding(pipe, prompt).to("cuda")
            target = binary_target(TRAIN / "annotations" / f"{tid}.png", "cuda")
            ovam_l0 = float(np.load(path)[0])
            rec = {"id": tid, "prompt": prompt, "ovam_epoch0_loss": ovam_l0,
                   "image_mean_abs_diff": float(diff.mean()), "image_identical_fraction": float((diff == 0).all(-1).mean()),
                   "epoch0_loss": {"/".join(v): epoch0_loss(pipe, recorder, initial, target, *v) for v in VARIANTS}}
            if args.optimize:
                ovam_curve = np.load(path)
                ovam_token = torch.from_numpy(np.load(str(path).replace("_losses.npy", ".npy"))).float().cuda()
                rec["optimized"] = {}
                for activation in ("token_softmax", "linear"):
                    best, curve = optimize(pipe, recorder, initial, target, activation)
                    rec["optimized"][activation] = {
                        "best_loss": float(curve.min()), "best_epoch": int(curve.argmin()),
                        "ovam_best_loss": float(ovam_curve.min()), "ovam_best_epoch": int(ovam_curve.argmin()),
                        "curve_max_abs_diff": float(np.abs(curve - ovam_curve).max()),
                        "curve_median_abs_diff": float(np.median(np.abs(curve - ovam_curve))),
                        "token_cosine_to_ovam": float(F.cosine_similarity(best[1], ovam_token[1], dim=0)),
                        "sot_row_cosine_to_ovam": float(F.cosine_similarity(best[0], ovam_token[0], dim=0))}
                    print(f"  {activation}: best {curve.min():.5f}@{curve.argmin()} vs OVAM {ovam_curve.min():.5f}@{ovam_curve.argmin()}; "
                          f"curve max|diff| {rec['optimized'][activation]['curve_max_abs_diff']:.5f}; "
                          f"cos to OVAM token {rec['optimized'][activation]['token_cosine_to_ovam']:.4f}", flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            best = min(rec["epoch0_loss"].items(), key=lambda kv: abs(kv[1] - ovam_l0))
            print(f"{tid}: image |diff| {rec['image_mean_abs_diff']:.2f} identical {rec['image_identical_fraction']:.3f}; "
                  f"OVAM L0 {ovam_l0:.5f}; B2 {rec['epoch0_loss']['linear/uncond/no_mid']:.5f}; closest {best[0]} {best[1]:.5f}", flush=True)
            del recorder, image
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
