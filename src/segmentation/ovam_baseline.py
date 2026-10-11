"""OVAM cross-attention heatmap before the paper's mask postprocessing.

The published OVAM pseudo-mask additionally multiplies this heatmap by a
highest-resolution self-attention fusion map and applies DenseCRF.  Functions
in this module intentionally expose only the cross-attention precursor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention_aggregation import resize_map
from .captured_dataset import AttentionSample


OVAM_THRESHOLD = 0.4
OVAM_WORKING_SIZE = 64


def _cross_modules(sample: AttentionSample) -> tuple[str, ...]:
    # StableDiffusionHooker defaults to locate_middle_block=False. Match that
    # behavior even when a general capture also contains the middle block.
    return tuple(
        sorted(
            name
            for names in sample.modules_by_resolution.values()
            for name in names
            if not name.startswith("mid_block")
        )
    )


def _prompt_token_count(sample: AttentionSample) -> int:
    # OVAM encodes without padding. Prefer the first EOT position because the
    # max-length attention mask is version dependent in CLIP tokenizers.
    try:
        return sample.token_ids.index(49407) + 1
    except ValueError:
        return int(sum(sample.attention_mask))


@torch.inference_mode()
def ovam_cross_attention_map(sample: AttentionSample) -> torch.Tensor:
    """Return OVAM's summed 64x64 CA map for the ordinary prompt embedding.

    The original code selects ``len(prompt.split())`` after also retaining SOT,
    which corresponds to the final whitespace-delimited class word for VOC-Sim.
    This preserves that behavior for the two multiword VOC classes as well.
    """
    modules = _cross_modules(sample)
    prompt_token_count = _prompt_token_count(sample)
    target_position = len(sample.prompt.split())
    total = torch.zeros((OVAM_WORKING_SIZE, OVAM_WORKING_SIZE), dtype=torch.float32)

    with sample.open_cross_kv() as cross_kv:
        for step_index in range(len(sample.timesteps)):
            with sample.open_attention(step_index) as tensors:
                keys = set(tensors.keys())
                for module_name in modules:
                    resolution = next(
                        size
                        for size, names in sample.modules_by_resolution.items()
                        if module_name in names
                    )
                    compact_key = f"{module_name}.ovam_target_map"
                    if compact_key in keys:
                        per_head = tensors.get_tensor(compact_key).float()
                    else:
                        query = tensors.get_tensor(f"{module_name}.q")[0].float()
                        key_tensor = cross_kv.get_tensor(f"{module_name}.k")
                        conditional = 1 if key_tensor.shape[0] > 1 else 0
                        key = key_tensor[conditional, :, :prompt_token_count].float()
                        scale = query.shape[-1] ** -0.5
                        probabilities = torch.matmul(query, key.transpose(-1, -2)).mul_(scale).softmax(dim=-1)
                        per_head = probabilities[:, :, target_position]
                    module_map = per_head.sum(dim=0).reshape(resolution, resolution)
                    total += resize_map(module_map, OVAM_WORKING_SIZE)
    return total


def ovam_cross_attention_mask(
    sample: AttentionSample,
    output_shape: tuple[int, int],
    threshold: float = OVAM_THRESHOLD,
) -> torch.Tensor:
    """Threshold the CA precursor without OVAM's SA fusion or DenseCRF."""
    heatmap = ovam_cross_attention_map(sample)
    resized = F.interpolate(
        heatmap[None, None], size=output_shape, mode="bilinear", align_corners=False
    )[0, 0]
    maximum = resized.max()
    normalized = resized / maximum if maximum > 0 else torch.zeros_like(resized)
    return normalized > threshold


# Backward-compatible names for saved experiment code.  These are deliberately
# aliases rather than the name of the published full OVAM pseudo-mask pipeline.
ovam_attention_map = ovam_cross_attention_map
ovam_mask = ovam_cross_attention_mask

