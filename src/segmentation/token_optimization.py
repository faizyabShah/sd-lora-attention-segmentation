"""OVAM-compatible text-token optimization from one annotated diffusion sample.

The implementation mirrors OVAM's released VOC optimization recipe while
remaining compatible with the Diffusers version used by this repository.  A
generation records the unconditional cross-attention queries.  Optimization
then recomputes OVAM's linear two-token maps from those fixed queries, so the
UNet is not rerun during the 800 SGD updates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.attention_capture.live import _install


OVAM_EPOCHS = 800
OVAM_INITIAL_LR = 100.0
OVAM_STEP_SIZE = 120
OVAM_GAMMA = 0.7


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OptimizationQueryRecorder:
    """Accumulate OVAM's unconditional CA queries over denoising calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.attribution_hidden_states = None
        self.module_metadata: dict[str, dict[str, Any]] = {}
        self.expected_modules: set[str] = set()
        self.call_counts: dict[str, int] = {}
        self.query_sums: dict[str, torch.Tensor] = {}

    def register_module(self, name: str, kind: str, heads: int, scale: float,
                        processor_class: str) -> None:
        self.expected_modules.add(name)
        self.call_counts[name] = 0
        self.module_metadata[name] = {
            "kind": kind, "heads": heads, "scale": scale,
            "original_processor": processor_class,
        }

    def record(self, module_name: str, kind: str, query: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               attention_mask: torch.Tensor | None, scale: float,
               attribution_key: torch.Tensor | None = None) -> None:
        del key, value, attention_mask, attribution_key
        self.call_counts[module_name] += 1
        spatial = math.isqrt(int(query.shape[-2]))
        self.module_metadata[module_name].update({
            "spatial_height": spatial, "spatial_width": spatial,
            "spatial_tokens": int(query.shape[-2]), "head_dim": int(query.shape[-1]),
            "scale": float(scale),
        })
        # OVAM's hook defaults to the unconditional CFG branch and excludes
        # the middle block. Summing here is algebraically identical to summing
        # the per-call linear attention logits in its DAAM block.
        if kind == "cross" and not module_name.startswith("mid_block."):
            q = query[0].detach().float()
            self.query_sums[module_name] = self.query_sums.get(module_name, 0) + q

    def flush_step(self, step: int, timestep: int, latent: torch.Tensor) -> None:
        del step, timestep, latent

    def finalize(self) -> None:
        missing = [n for n in self.expected_modules if self.call_counts[n] == 0]
        if missing:
            raise RuntimeError(f"Attention modules were not observed: {missing}")


def capture_optimization_queries(pipe: Any, prompt: str, seed: int,
                                 steps: int = 30) -> OptimizationQueryRecorder:
    """Generate one sample and retain only query sums needed by optimization."""
    recorder = OptimizationQueryRecorder()
    _install(pipe, recorder)
    seed_everything(seed)
    # ``inference_mode`` tensors cannot participate even as constants in the
    # later token-gradient graph. ``no_grad`` avoids a UNet graph while
    # producing ordinary tensors that autograd may save for backward.
    with torch.no_grad():
        result = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            height=512,
            width=512,
            output_type="latent",
        )
    recorder.finalize()
    del result
    return recorder


def initial_ovam_embedding(pipe: Any, prompt: str) -> torch.Tensor:
    """Reproduce OVAM's ``encode_text(...)[(0, -2)]`` initialization."""
    target_text = prompt.split(" ")[-1]
    tokens = pipe.tokenizer(prompt, padding=False, return_tensors="pt")
    target = pipe.tokenizer(target_text, padding=False, return_tensors="pt")
    with torch.no_grad():
        encoded = pipe.text_encoder(
            tokens.input_ids.to(pipe.device),
            attention_mask=tokens.attention_mask.to(pipe.device),
        )[0][0]
    target_ids = set(target.input_ids.numpy().reshape(-1).tolist())
    selected = encoded[[int(value) in target_ids for value in tokens.input_ids[0].tolist()]]
    if selected.shape[0] < 2:
        raise RuntimeError(f"OVAM initialization found fewer than two rows for {prompt!r}")
    return selected[(0, -2), :].detach().float()


def binary_target(annotation: Path, device: str) -> torch.Tensor:
    foreground = np.asarray(Image.open(annotation).convert("L")) > 0
    target = np.stack((~foreground, foreground), axis=0)[None].astype(np.float32)
    if target.shape != (1, 2, 512, 512):
        raise ValueError(f"Expected a 512x512 mask at {annotation}, got {target.shape}")
    return torch.from_numpy(target).to(device)


@dataclass(frozen=True)
class OptimizationResult:
    embedding: torch.Tensor
    best_loss: float
    losses: np.ndarray
    best_epoch: int


def ovam_linear_map(pipe: Any, recorder: OptimizationQueryRecorder,
                    embedding: torch.Tensor, expand_size: tuple[int, int] = (512, 512)) -> torch.Tensor:
    """OVAM DAAM forward: calls sum, heads sum, blocks mean, bilinear resize."""
    maps = []
    for name, query in recorder.query_sums.items():
        module = pipe.unet.get_submodule(name)
        key = module.to_k(embedding[None])
        head_dim = key.shape[-1] // int(module.heads)
        key = key.view(1, -1, int(module.heads), head_dim).permute(0, 2, 1, 3)[0]
        # [heads, pixels, d] x [heads, tokens, d] -> [tokens, H, W],
        # followed by OVAM's sum over heads.
        logits = torch.einsum("hpd,htd->htp", query, key) * float(module.scale)
        r = int(recorder.module_metadata[name]["spatial_height"])
        maps.append(logits.sum(dim=0).reshape(1, 2, r, r))
    if not maps:
        raise RuntimeError("No non-middle cross-attention queries were captured")
    latent_size = max(value.shape[-1] for value in maps)
    maps64 = [value if value.shape[-1] == latent_size else F.interpolate(
        value, size=(latent_size, latent_size), mode="bilinear", align_corners=False
    ) for value in maps]
    merged = torch.stack(maps64).mean(dim=0)
    return F.interpolate(merged, size=expand_size, mode="bilinear", align_corners=False)


def optimize_ovam_embedding(pipe: Any, recorder: OptimizationQueryRecorder,
                            initial: torch.Tensor, target: torch.Tensor,
                            epochs: int = OVAM_EPOCHS,
                            initial_lr: float = OVAM_INITIAL_LR,
                            step_size: int = OVAM_STEP_SIZE,
                            gamma: float = OVAM_GAMMA,
                            progress: Callable[[int, float], None] | None = None) -> OptimizationResult:
    """Run the released OVAM SGD schedule and retain the lowest-loss vector."""
    x = initial.detach().clone().to(pipe.device).requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    losses: list[float] = []
    best_loss = float("inf")
    best_epoch = -1
    best = x.detach().clone()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        heatmap = ovam_linear_map(pipe, recorder, x)
        low, high = heatmap.amin(), heatmap.amax()
        normalized = (heatmap - low) / (high - low).clamp_min(torch.finfo(heatmap.dtype).eps)
        loss = loss_fn(normalized, target)
        value = float(loss.detach())
        losses.append(value)
        # OVAM's callback records the state before this epoch's SGD update.
        if value < best_loss:
            best_loss, best_epoch, best = value, epoch, x.detach().clone()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if progress is not None:
            progress(epoch, value)
    return OptimizationResult(best.cpu(), best_loss, np.asarray(losses), best_epoch)

