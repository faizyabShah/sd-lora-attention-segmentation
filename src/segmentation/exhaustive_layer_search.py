"""Memory-bounded exhaustive cross-attention layer-subset evaluation."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTokenizer

from .ablation_runner import SegmentationSettings, target_token_positions
from .attention_aggregation import load_step_module_maps, resize_map
from .captured_dataset import CapturedAttentionDataset


@dataclass
class ExhaustiveLayerResults:
    module_names: tuple[str, ...]
    classes: tuple[str, ...]
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    tn: torch.Tensor
    image_iou_sum: torch.Tensor
    image_counts: torch.Tensor

    @property
    def subset_count(self) -> int:
        return self.tp.shape[0] - 1

    def per_class_iou(self) -> torch.Tensor:
        union = self.tp + self.fp + self.fn
        return torch.where(union > 0, self.tp.double() / union, torch.ones_like(union).double())

    def miou(self) -> torch.Tensor:
        return self.per_class_iou()[1:].mean(dim=1)


def _subset_weights(module_count: int, device: torch.device) -> torch.Tensor:
    masks = torch.arange(1, 1 << module_count, dtype=torch.int64, device=device)
    bits = torch.arange(module_count, dtype=torch.int64, device=device)
    return ((masks[:, None] >> bits[None, :]) & 1).float()


@torch.inference_mode()
def run_exhaustive_layer_search(
    dataset: CapturedAttentionDataset,
    tokenizer: CLIPTokenizer,
    settings: SegmentationSettings,
    device: str = "cuda",
    batch_size: int = 256,
) -> ExhaustiveLayerResults:
    """Evaluate every nonempty subset without retaining predicted masks."""
    if settings.normalization != "minmax":
        raise ValueError("Exhaustive search currently requires minmax normalization")
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    first = dataset.samples[0]
    module_names = tuple(
        sorted(name for names in first.modules_by_resolution.values() for name in names)
    )
    module_count = len(module_names)
    subset_count = (1 << module_count) - 1
    classes = dataset.classes
    class_indices = {name: index for index, name in enumerate(classes)}
    shape = (subset_count + 1, len(classes))
    tp = torch.zeros(shape, dtype=torch.int64)
    fp = torch.zeros(shape, dtype=torch.int64)
    fn = torch.zeros(shape, dtype=torch.int64)
    tn = torch.zeros(shape, dtype=torch.int64)
    image_iou_sum = torch.zeros(shape, dtype=torch.float64)
    image_counts = torch.zeros(len(classes), dtype=torch.int64)
    weights = _subset_weights(module_count, compute_device)

    for sample in tqdm(dataset, desc="Exhaustive layer subsets"):
        positions = target_token_positions(sample, tokenizer)
        layer_sums: dict[str, torch.Tensor] = {}
        layer_counts = {name: 0 for name in module_names}
        for step_index in range(len(sample.timesteps)):
            step_maps = load_step_module_maps(
                sample, step_index, positions, module_names, branch=settings.branch
            )
            for module_name, (value, _) in step_maps.items():
                resized = resize_map(value, settings.working_size)
                if module_name not in layer_sums:
                    layer_sums[module_name] = torch.zeros_like(resized)
                layer_sums[module_name] += resized
                layer_counts[module_name] += 1

        missing = [name for name in module_names if layer_counts[name] == 0]
        if missing:
            raise ValueError(f"Missing layer maps for {sample.sample_id}: {missing}")
        layers = torch.stack(
            [layer_sums[name] / layer_counts[name] for name in module_names]
        ).to(compute_device)
        target = torch.from_numpy(sample.load_target_mask()).to(compute_device)
        truth_pixels = int(target.sum().item())
        total_pixels = target.numel()
        class_index = class_indices[sample.classname]
        image_counts[class_index] += 1

        for start in range(0, subset_count, batch_size):
            stop = min(start + batch_size, subset_count)
            maps = torch.einsum("bl,lhw->bhw", weights[start:stop], layers)
            minima = maps.amin(dim=(1, 2), keepdim=True)
            maxima = maps.amax(dim=(1, 2), keepdim=True)
            scales = maxima - minima
            maps = torch.where(scales > 0, (maps - minima) / scales, torch.zeros_like(maps))
            maps = F.interpolate(
                maps[:, None], size=target.shape, mode="bilinear", align_corners=False
            )[:, 0]
            predictions = maps >= settings.threshold
            batch_tp = (predictions & target).sum(dim=(1, 2)).cpu()
            predicted_pixels = predictions.sum(dim=(1, 2)).cpu()
            batch_fp = predicted_pixels - batch_tp
            batch_fn = truth_pixels - batch_tp
            batch_tn = total_pixels - batch_tp - batch_fp - batch_fn
            destination = slice(start + 1, stop + 1)
            tp[destination, class_index] += batch_tp
            fp[destination, class_index] += batch_fp
            fn[destination, class_index] += batch_fn
            tn[destination, class_index] += batch_tn
            union = batch_tp + batch_fp + batch_fn
            image_iou_sum[destination, class_index] += torch.where(
                union > 0, batch_tp.double() / union, torch.ones_like(union).double()
            )
            del maps, predictions

        del layers, layer_sums, target
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    del weights
    gc.collect()
    return ExhaustiveLayerResults(
        module_names=module_names,
        classes=classes,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        image_iou_sum=image_iou_sum,
        image_counts=image_counts,
    )
