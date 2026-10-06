"""Load captured tensors and reconstruct attention matrices on demand."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_schema(sample_dir: str | Path) -> dict:
    path = Path(sample_dir) / "attention_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(path: Path, key: str, device: str | torch.device = "cpu") -> torch.Tensor:
    with safe_open(path, framework="pt", device=str(device)) as handle:
        return handle.get_tensor(key)


def load_cross_attention(
    sample_dir: str | Path,
    step: int,
    module: str,
    branch: int | None = None,
    head: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load stored cross-attention probabilities.

    The returned layout is ``[branch, head, spatial_token, text_token]``;
    selecting a branch or head removes that dimension.
    """
    path = Path(sample_dir) / "attention" / f"step_{step:03d}.safetensors"
    tensor = _load_tensor(path, f"{module}.probabilities", device)
    if branch is not None:
        tensor = tensor[branch]
    if head is not None:
        tensor = tensor[head] if branch is not None else tensor[:, head]
    return tensor


def reconstruct_self_attention(
    sample_dir: str | Path,
    step: int,
    module: str,
    branch: int = 1,
    head: int | None = None,
    query_slice: slice | None = None,
    device: str | torch.device = "cpu",
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a self-attention probability matrix from stored Q and K.

    ``query_slice`` can restrict query rows, which is strongly recommended for
    64x64 layers. With no head selection the output is
    ``[heads, selected_queries, key_tokens]``.
    """
    root = Path(sample_dir)
    path = root / "attention" / f"step_{step:03d}.safetensors"
    q = _load_tensor(path, f"{module}.q", device)[branch]
    k = _load_tensor(path, f"{module}.k", device)[branch]
    if head is not None:
        q, k = q[head], k[head]
    if query_slice is not None:
        q = q[..., query_slice, :]
    scale = float(load_schema(root)["modules"][module]["scale"])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    return scores.softmax(dim=-1).to(output_dtype)
