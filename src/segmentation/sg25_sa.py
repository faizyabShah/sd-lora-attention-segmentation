"""Locked same-call multiresolution SA propagation and depth maps for SG25."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.segmentation.sg24_ca import SG24Recorder

NAMES = ("FINE-P2", "COARSE16-P2", "COARSE32-P2", "16-to-64", "32-to-64", "64-to-16")


class SG25Recorder(SG24Recorder):
    """Reuse exact A3 construction while retaining the three frozen SA Q/K sources."""

    extra_sa: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiscale_qk = {name: [] for name in self.extra_sa}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind == "self" and module_name in self.multiscale_qk:
            step = self.call_counts[module_name]
            if 10 <= step <= 20:
                self.multiscale_qk[module_name].append(
                    (query[1].detach().half().clone(), key[1].detach().half().clone())
                )
        super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key)

    def clear(self):
        super().clear()
        for values in self.multiscale_qk.values():
            values.clear()


def _attention(pair: tuple[torch.Tensor, torch.Tensor], side: int) -> torch.Tensor:
    q, k = pair
    if q.shape != k.shape or q.shape[0] != 8 or q.shape[1] != side * side:
        raise ValueError(f"Unexpected {side}x{side} SA Q/K: {tuple(q.shape)}, {tuple(k.shape)}")
    q, k = q.float(), k.float()
    return torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1]), dim=-1)


def _apply(attention: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply eight SA heads to one shared spatial vector or eight vectors."""
    heads, pixels, _ = attention.shape
    if value.ndim == 1:
        value = value[None].expand(heads, -1)
    if value.shape != (heads, pixels):
        raise ValueError("SA input shape does not match its attention resolution")
    return torch.bmm(attention, value.unsqueeze(-1)).squeeze(-1)


def _resize(value: torch.Tensor, source: int, target: int) -> torch.Tensor:
    mode = "bilinear" if target > source else "area"
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(value.reshape(-1, 1, source, source),
                         size=(target, target), mode=mode, **kwargs).reshape(-1, target * target)


@torch.inference_mode()
def aggregate(sa_qk: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
              seed: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return six 64x64 maps and five mean fine-resolution depth maps.

    The two operators of a cross-resolution cell come from the same call.
    Averaging heads of the first operator yields one intermediate map; all
    heads of the second operator receive it equally. This avoids inventing
    a correspondence between head indices at different resolutions.
    """
    if seed.shape != (64, 64) or any(len(sa_qk.get(side, [])) != 11 for side in (16, 32, 64)):
        raise ValueError("SG25 requires one A3 seed and 11 calls at each SA scale")
    c64 = seed.float().reshape(-1)
    c16 = _resize(c64, 64, 16).reshape(-1)
    c32 = _resize(c64, 64, 32).reshape(-1)
    sums = {name: torch.zeros_like(seed, dtype=torch.float32) for name in NAMES}
    depth_sums = torch.zeros((5, 64, 64), dtype=torch.float32, device=seed.device)
    for i in range(11):
        a16 = _attention(sa_qk[16][i], 16)
        a32 = _attention(sa_qk[32][i], 32)
        a64 = _attention(sa_qk[64][i], 64)
        one16 = _apply(a16, c16)
        one32 = _apply(a32, c32)
        one64 = _apply(a64, c64)
        two16 = _apply(a16, one16)
        two32 = _apply(a32, one32)
        coarse16_to_fine = _apply(a64, _resize(one16.mean(0), 16, 64).reshape(-1))
        coarse32_to_fine = _apply(a64, _resize(one32.mean(0), 32, 64).reshape(-1))
        fine_to_coarse16 = _apply(a16, _resize(one64.mean(0), 64, 16).reshape(-1))
        sums["COARSE16-P2"] += _resize(two16.mean(0), 16, 64).reshape(64, 64) / 11
        sums["COARSE32-P2"] += _resize(two32.mean(0), 32, 64).reshape(64, 64) / 11
        sums["16-to-64"] += coarse16_to_fine.mean(0).reshape(64, 64) / 11
        sums["32-to-64"] += coarse32_to_fine.mean(0).reshape(64, 64) / 11
        sums["64-to-16"] += _resize(fine_to_coarse16.mean(0), 16, 64).reshape(64, 64) / 11
        state = one64
        depth_sums[0] += state.sum(0).reshape(64, 64)
        for depth in range(1, 5):
            state = _apply(a64, state)
            depth_sums[depth] += state.sum(0).reshape(64, 64)
        del a16, a32, a64, one16, one32, one64, state
    depth_sums /= (8 * 11)
    sums["FINE-P2"] = depth_sums[1]
    return sums, depth_sums
