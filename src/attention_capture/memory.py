"""Compact one-image attention capture for streaming experiments."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .recorder import _storage_dtype


class CompactAttentionRecorder:
    """RAM recorder compatible with ``RecordingAttnProcessor``.

    It keeps only per-head target cross-attention maps, exact OVAM maps, and
    conditional self-attention Q/K. Everything belongs to one generated image.
    """

    def __init__(
        self,
        target_positions: Sequence[int],
        prompt_token_count: int,
        ovam_target_position: int,
        storage_dtype: str = "float16",
    ) -> None:
        self.dtype = _storage_dtype(storage_dtype)
        self.storage_dtype = storage_dtype
        self.target_positions = tuple(int(value) for value in target_positions)
        self.prompt_token_count = int(prompt_token_count)
        self.ovam_target_position = int(ovam_target_position)
        if not self.target_positions:
            raise ValueError("At least one target prompt position is required")
        if not 0 <= self.ovam_target_position < self.prompt_token_count:
            raise ValueError("OVAM target position is outside the unpadded prompt")
        self.step_tensors: dict[int, dict[str, torch.Tensor]] = {}
        self.steps: list[dict[str, torch.Tensor]] = []
        self.module_metadata: dict[str, dict[str, Any]] = {}
        self.expected_modules: set[str] = set()
        self.call_counts: dict[str, int] = {}

    def register_module(
        self, name: str, kind: str, heads: int, scale: float, processor_class: str
    ) -> None:
        self.expected_modules.add(name)
        self.call_counts[name] = 0
        self.module_metadata[name] = {
            "kind": kind,
            "heads": heads,
            "scale": scale,
            "original_processor": processor_class,
        }

    def _cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu", dtype=self.dtype, copy=True).contiguous()

    def record(
        self,
        module_name: str,
        kind: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scale: float,
    ) -> None:
        del value
        step = self.call_counts[module_name]
        self.call_counts[module_name] += 1
        bucket = self.step_tensors.setdefault(step, {})
        conditional = 1 if query.shape[0] > 1 else 0

        if kind == "self":
            # Reproduce OVAM's released SelfAttentionHooker exactly.  Its
            # pseudo-mask pipeline applies attention to an all-ones value,
            # sums the CFG branches and feature channels, and retains one
            # spatial vector per head.  Compute this while Q/K are on device;
            # retaining the tiny result avoids materializing it again later.
            ovam_value = torch.ones_like(key)
            ovam_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                ovam_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            # The all-ones construction is constant in exact arithmetic.  Its
            # released implementation relies on tiny float32 roundoff before
            # min-max scaling, so casting this particular tensor to float16
            # would erase the map completely.
            bucket[f"{module_name}.ovam_self_map"] = (
                ovam_output.sum(dim=0)
                .sum(dim=-1)
                .detach()
                .to(device="cpu", dtype=torch.float32, copy=True)
                .contiguous()
            )
            del ovam_value, ovam_output
            # Keep a singleton branch dimension for compatibility with saved data.
            bucket[f"{module_name}.q"] = self._cpu(query[conditional : conditional + 1])
            bucket[f"{module_name}.k"] = self._cpu(key[conditional : conditional + 1])
        else:
            scores = torch.matmul(
                query[conditional].float(), key[conditional].float().transpose(-1, -2)
            ) * scale
            if attention_mask is not None:
                scores = scores + attention_mask[conditional].float()
            probabilities = scores.softmax(dim=-1)
            bucket[f"{module_name}.target_maps"] = self._cpu(
                probabilities[:, :, list(self.target_positions)].mean(dim=-1)
            )

            # OVAM uses the unconditional query and an unpadded prompt. CLIP is
            # causal, so the corresponding prefix K is identical to encoding the
            # short prompt separately, including its final EOT token.
            ovam_scores = torch.matmul(
                query[0].float(),
                key[conditional, :, : self.prompt_token_count].float().transpose(-1, -2),
            ) * scale
            bucket[f"{module_name}.ovam_target_map"] = self._cpu(
                ovam_scores.softmax(dim=-1)[:, :, self.ovam_target_position]
            )

        spatial_size = math.isqrt(int(query.shape[-2]))
        self.module_metadata[module_name].update(
            {
                "spatial_height": spatial_size,
                "spatial_width": spatial_size,
                "spatial_tokens": int(query.shape[-2]),
                "head_dim": int(query.shape[-1]),
            }
        )

    def flush_step(self, step: int, timestep: int, latent: torch.Tensor) -> None:
        del timestep, latent
        tensors = self.step_tensors.pop(step, None)
        if tensors is None:
            raise RuntimeError(f"No attention tensors captured for step {step}")
        missing = sorted(name for name in self.expected_modules if self.call_counts[name] <= step)
        if missing:
            raise RuntimeError(f"Step {step} missed attention modules: {missing}")
        if step != len(self.steps):
            raise RuntimeError(f"Nonsequential capture at step {step}")
        self.steps.append(tensors)

    def finalize(self) -> None:
        if self.step_tensors:
            raise RuntimeError(f"Unflushed steps remain: {sorted(self.step_tensors)}")

    def clear(self) -> None:
        self.step_tensors.clear()
        for step in self.steps:
            step.clear()
        self.steps.clear()

