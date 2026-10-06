"""Dataset access for attention tensors captured during diffusion generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CapturedSample:
    """One completed attention capture with an available target mask."""

    sample_id: str
    root: Path
    classname: str
    prompt: str
    token_ids: tuple[int, ...]
    timesteps: tuple[int, ...]
    modules_by_resolution: dict[int, tuple[str, ...]]

    @property
    def annotation_path(self) -> Path:
        return self.root / "annotation.png"

    def attention_path(self, step_index: int) -> Path:
        return self.root / "attention" / f"step_{step_index:03d}.safetensors"

    def load_target_mask(self) -> np.ndarray:
        """Return a boolean foreground mask; annotations use nonzero RGB values."""
        pixels = np.asarray(Image.open(self.annotation_path).convert("RGB"))
        return np.any(pixels != 0, axis=-1)


class CapturedAttentionDataset:
    """Discover and validate completed captured-attention samples."""

    def __init__(self, root: str | Path, require_annotation: bool = True) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Capture directory does not exist: {self.root}")
        self.samples = self._discover(require_annotation=require_annotation)
        if not self.samples:
            raise ValueError(f"No usable completed captures found under {self.root}")

    def _discover(self, require_annotation: bool) -> list[CapturedSample]:
        samples: list[CapturedSample] = []
        for sample_root in sorted(self.root.glob("ovam-voc-sim-*"), key=_sample_number):
            if not (sample_root / "COMPLETE").is_file():
                continue
            if require_annotation and not (sample_root / "annotation.png").is_file():
                continue
            metadata = json.loads((sample_root / "metadata.json").read_text(encoding="utf-8"))
            schema = json.loads((sample_root / "attention_schema.json").read_text(encoding="utf-8"))
            grouped: dict[int, list[str]] = {}
            for module_name, module in schema["modules"].items():
                if module["kind"] != "cross":
                    continue
                resolution = module.get("spatial_height")
                if resolution is not None:
                    grouped.setdefault(int(resolution), []).append(module_name)
            samples.append(
                CapturedSample(
                    sample_id=sample_root.name,
                    root=sample_root,
                    classname=metadata["classname"],
                    prompt=metadata["prompt"],
                    token_ids=tuple(metadata["token_ids"]),
                    timesteps=tuple(metadata["timesteps"]),
                    modules_by_resolution={
                        resolution: tuple(sorted(names)) for resolution, names in grouped.items()
                    },
                )
            )
        return samples

    def __iter__(self) -> Iterator[CapturedSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted({sample.classname for sample in self.samples}))


def _sample_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return 10**12
