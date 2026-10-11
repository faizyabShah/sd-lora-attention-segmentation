"""Saved-capture compatible wrappers for a compact in-memory capture."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch


class _TensorHandle:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def keys(self) -> list[str]:
        return list(self._tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


@dataclass
class InMemoryAttentionSample:
    sample_id: str
    classname: str
    prompt: str
    token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    timesteps: tuple[int, ...]
    modules_by_resolution: dict[int, tuple[str, ...]]
    self_modules_by_resolution: dict[int, tuple[str, ...]]
    cross_map_key: str
    target_mask: np.ndarray
    steps: list[dict[str, torch.Tensor]] = field(repr=False)

    @contextmanager
    def open_attention(self, step_index: int):
        yield _TensorHandle(self.steps[step_index])

    @contextmanager
    def open_cross_kv(self):
        yield _TensorHandle({})

    def load_target_mask(self) -> np.ndarray:
        return self.target_mask

    def clear(self) -> None:
        for step in self.steps:
            step.clear()
        self.steps.clear()


class SingleSampleDataset:
    """Minimal dataset API accepted by the existing ablation functions."""

    def __init__(self, sample: InMemoryAttentionSample) -> None:
        self.samples = [sample]

    def __iter__(self) -> Iterator[InMemoryAttentionSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return 1

    @property
    def classes(self) -> tuple[str, ...]:
        return (self.samples[0].classname,)

