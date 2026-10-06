"""Lossless-structure attention capture for Diffusers UNets.

The recorder stores projected, head-separated Q/K/V tensors. Cross-attention
probabilities are materialized because their text dimension is small. Full
self-attention matrices are deliberately reconstructed later from Q and K.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _storage_dtype(name: str) -> torch.dtype:
    values = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"Unsupported storage dtype: {name}") from error


class AttentionRecorder:
    """Collect one denoising step at a time and write it to safetensors."""

    def __init__(self, output_dir: Path, storage_dtype: str = "float16") -> None:
        self.output_dir = Path(output_dir)
        self.attention_dir = self.output_dir / "attention"
        self.latent_dir = self.output_dir / "latents"
        self.attention_dir.mkdir(parents=True, exist_ok=True)
        self.latent_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = _storage_dtype(storage_dtype)
        self.storage_dtype = storage_dtype
        self.step_tensors: dict[int, dict[str, torch.Tensor]] = {}
        self.static_cross_kv: dict[str, torch.Tensor] = {}
        self.static_cross_modules: set[str] = set()
        self.module_metadata: dict[str, dict[str, Any]] = {}
        self.expected_modules: set[str] = set()
        self.call_counts: dict[str, int] = {}
        self.bytes_written = 0

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

    @staticmethod
    def _cpu(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor.detach().to(device="cpu", dtype=dtype, copy=True).contiguous()

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
        step = self.call_counts[module_name]
        self.call_counts[module_name] += 1
        bucket = self.step_tensors.setdefault(step, {})
        prefix = module_name

        bucket[f"{prefix}.q"] = self._cpu(query, self.dtype)
        if kind == "self":
            bucket[f"{prefix}.k"] = self._cpu(key, self.dtype)
            bucket[f"{prefix}.v"] = self._cpu(value, self.dtype)
        elif module_name not in self.static_cross_modules:
            self.static_cross_kv[f"{prefix}.k"] = self._cpu(key, self.dtype)
            self.static_cross_kv[f"{prefix}.v"] = self._cpu(value, self.dtype)
            self.static_cross_modules.add(module_name)

        if attention_mask is not None:
            bucket[f"{prefix}.mask"] = self._cpu(attention_mask, self.dtype)

        if kind == "cross":
            # Cross attention is small enough to retain explicitly. Compute in
            # float32 for a stable probability map, then cast for storage.
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale
            if attention_mask is not None:
                scores = scores + attention_mask.float()
            probabilities = scores.softmax(dim=-1)
            bucket[f"{prefix}.probabilities"] = self._cpu(probabilities, self.dtype)

        meta = self.module_metadata[module_name]
        spatial_size = math.isqrt(int(query.shape[-2]))
        meta.update(
            {
                "query_shape": list(query.shape),
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "spatial_tokens": int(query.shape[-2]),
                "spatial_height": spatial_size if spatial_size * spatial_size == query.shape[-2] else None,
                "spatial_width": spatial_size if spatial_size * spatial_size == query.shape[-2] else None,
                "head_dim": int(query.shape[-1]),
                "cfg_batch_order": ["unconditional", "conditional"]
                if query.shape[0] == 2
                else ["conditional"],
            }
        )

    def flush_step(self, step: int, timestep: int, latent: torch.Tensor) -> None:
        tensors = self.step_tensors.pop(step, None)
        if tensors is None:
            raise RuntimeError(f"No attention tensors captured for step {step}")
        missing = sorted(name for name in self.expected_modules if self.call_counts[name] <= step)
        if missing:
            raise RuntimeError(f"Step {step} did not call {len(missing)} attention modules: {missing}")

        attention_path = self.attention_dir / f"step_{step:03d}.safetensors"
        save_file(tensors, str(attention_path), metadata={"step": str(step), "timestep": str(timestep)})
        latent_path = self.latent_dir / f"step_{step:03d}.safetensors"
        save_file(
            {"latent": self._cpu(latent, self.dtype)},
            str(latent_path),
            metadata={"step": str(step), "timestep": str(timestep)},
        )
        self.bytes_written += attention_path.stat().st_size + latent_path.stat().st_size

    def finalize(self) -> None:
        if self.step_tensors:
            raise RuntimeError(f"Unflushed attention steps: {sorted(self.step_tensors)}")
        save_file(
            self.static_cross_kv,
            str(self.attention_dir / "cross_attention_kv.safetensors"),
            metadata={"note": "K/V are constant across denoising steps for each cross-attention module"},
        )
        (self.output_dir / "attention_schema.json").write_text(
            json.dumps(
                {
                    "storage_dtype": self.storage_dtype,
                    "self_attention": "Per-step Q/K/V; reconstruct A=softmax(Q@K^T*scale).",
                    "cross_attention": "Per-step Q and probabilities; module K/V stored once.",
                    "tensor_layout": "[cfg_branch, head, query_or_key_tokens, head_dim]",
                    "modules": self.module_metadata,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class RecordingAttnProcessor:
    """Wrap an existing Diffusers attention processor without changing its output."""

    def __init__(self, original: Any, recorder: AttentionRecorder, module_name: str, kind: str) -> None:
        self.original = original
        self.recorder = recorder
        self.module_name = module_name
        self.kind = kind

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        with torch.no_grad():
            capture_hidden = hidden_states
            if attn.spatial_norm is not None:
                capture_hidden = attn.spatial_norm(capture_hidden, temb)
            if capture_hidden.ndim == 4:
                batch, channels, height, width = capture_hidden.shape
                capture_hidden = capture_hidden.view(batch, channels, height * width).transpose(1, 2)

            encoder = encoder_hidden_states
            batch_size, sequence_length, _ = (
                capture_hidden.shape if encoder is None else encoder.shape
            )
            prepared_mask = None
            if attention_mask is not None:
                prepared_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                prepared_mask = prepared_mask.view(batch_size, attn.heads, -1, prepared_mask.shape[-1])

            if attn.group_norm is not None:
                capture_hidden = attn.group_norm(capture_hidden.transpose(1, 2)).transpose(1, 2)
            query = attn.to_q(capture_hidden)
            if encoder is None:
                encoder = capture_hidden
            elif attn.norm_cross:
                encoder = attn.norm_encoder_hidden_states(encoder)
            key = attn.to_k(encoder)
            value = attn.to_v(encoder)

            head_dim = key.shape[-1] // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            self.recorder.record(
                self.module_name,
                self.kind,
                query,
                key,
                value,
                prepared_mask,
                float(attn.scale),
            )

        return self.original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )
