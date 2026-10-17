#!/usr/bin/env python3
"""Evaluate learned B2 tokens through the frozen one-pass-SA + DenseCRF method.

VOC mode evaluates released tokens, every registered B2a 1/2/all-support
average, and the fold-specific B2b token on its held-out images. COCO mode
transfers only the released token and our B2a all-support class token. The
mask threshold (0.475) and all segmentation components are frozen from T-C.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_coco_cap_transfer as transfer  # noqa: E402
import scripts.run_sa_package_b as sa_common  # noqa: E402
from scripts.run_fusion_calibration import upsampled  # noqa: E402
from scripts.run_streaming_voc_ablations import read_rows  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.fusion_variants import minmax  # noqa: E402
from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402

THRESHOLDS = (0.475,)
VOC_CONFIGS = ("REL", "A1", "A2", "A3", "A12", "A13", "A23", "AALL", "B2B")
COCO_CONFIGS = ("REL", "AALL")
_TOKEN_NAMES: list[str] = []
_TOKEN_HIDDEN: list[torch.Tensor] = []


class LearnedTokenRecorder(transfer.TransferRecorder):
    """Capture CA for a variable list of two-row learned-token encodings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        base = self.attribution_hidden_states
        if base is None:
            raise RuntimeError("A separate attribution encoding is required")
        self.attribution_hidden_states = torch.cat((base, _TOKEN_HIDDEN[0].to(base.dtype)), dim=1)
        self.token_calls: dict[str, dict[str, list[torch.Tensor]]] = {}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind != "cross":
            return sa_common.SAPackageRecorder.record(
                self, module_name, kind, query, key, value, attention_mask, scale
            )
        if attribution_key is None:
            raise RuntimeError("Learned-token attribution keys were not projected")
        word_length = attribution_key.shape[-2] - 2 * len(_TOKEN_NAMES)
        word = torch.stack((key[0], attribution_key[0][:, :word_length]))
        sa_common.SAPackageRecorder.record(
            self, module_name, kind, query, word, value, None, scale
        )
        with torch.inference_mode():
            for index, name in enumerate(_TOKEN_NAMES):
                pair = attribution_key[0][:, word_length + 2 * index: word_length + 2 * index + 2].float()
                probs = (torch.matmul(query[1].float(), pair.transpose(-1, -2)) * scale).softmax(dim=-1)[..., 1]
                self.token_calls.setdefault(name, {}).setdefault(module_name, []).append(probs.half())


sa_common.live.CompactAttentionRecorder = LearnedTokenRecorder


def load_embedding(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as data:
        if not bool(data["complete"]):
            raise RuntimeError(f"Incomplete token: {path}")
        return torch.from_numpy(data["embedding"].astype(np.float32))


def class_tokens(pipe, classname: str, fold: int | None, plan: dict, learned: Path,
                 released: dict, device: str, mode: str) -> tuple[list[str], torch.Tensor]:
    with torch.inference_mode():
        empty = pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
        sot = pipe.text_encoder(empty)[0][0, 0].float()
    rel = torch.from_numpy(np.asarray(released[classname], dtype=np.float32))
    supports = [task for task in plan["b2a"] if task["classname"] == classname]
    vectors = [load_embedding(learned / "b2a/tokens" / f"{task['id']}.npz")[1] for task in supports]
    if len(vectors) not in (2, 3):
        raise RuntimeError(f"{classname}: expected 2 or 3 B2a vectors, found {len(vectors)}")
    if mode == "coco":
        names, values = ["REL", "AALL"], [rel, torch.stack(vectors).mean(0)]
    else:
        slots = vectors + [vectors[0]] * (3 - len(vectors))
        names = list(VOC_CONFIGS)
        values = [rel, slots[0], slots[1], slots[2],
                  torch.stack((slots[0], slots[1])).mean(0),
                  torch.stack((slots[0], slots[2])).mean(0),
                  torch.stack((slots[1], slots[2])).mean(0),
                  torch.stack(vectors).mean(0)]
        task = next(task for task in plan["b2b"]
                    if task["classname"] == classname and task["heldout_fold"] == fold)
        values.append(load_embedding(learned / "b2b/tokens" / f"{task['id']}.npz")[1])
    pairs = [torch.stack((sot.cpu(), value.cpu())) for value in values]
    return names, torch.cat(pairs, dim=0)[None].to(device)


def one_pass(recorder, module_name: str, seeds: torch.Tensor, device: str) -> torch.Tensor:
    flat = seeds.reshape(seeds.shape[0], -1).T
    out = torch.zeros_like(flat)
    for query, key in recorder.sa_qk[module_name]:
        query, key = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(query.shape[-1]), dim=-1)
        out += torch.matmul(affinity, flat).mean(dim=0)
    return (out / len(recorder.sa_qk[module_name])).T.reshape_as(seeds)


def crf_count(value: torch.Tensor, target: np.ndarray, image: np.ndarray, crf_site: Path) -> np.ndarray:
    full = upsampled(minmax(value), tuple(np.asarray(target).shape))
    mask = apply_ovam_densecrf(image, (full >= THRESHOLDS[0]).cpu().numpy(), crf_site)
    return cv.score_masks(mask[None], target, thresholds=THRESHOLDS)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/token_optimization_b2_plan.json"))
    parser.add_argument("--learned-dir", type=Path,
                        default=Path("outputs/segmentation/controlled_attention_cv/token_learning_b2"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/segmentation/controlled_attention_cv/token_learning_b2"))
    parser.add_argument("--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_frozen.json"))
    parser.add_argument("--tokens", type=Path, default=Path("configs/ovam/tokens_VOC.npy"))
    parser.add_argument("--crf-site", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    released = np.load(args.tokens, allow_pickle=True).item()
    seed_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    folds = cv.read_folds(Path("configs/voc_sim_folds.csv"))
    if args.dataset == "voc":
        classname, rows = read_rows(
            Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"),
            args.class_shard, args.max_samples,
        )
        rows = [dict(row, attribution_prompt=row["prompt"]) for row in rows]
        manifest_dir = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim"
        configurations = VOC_CONFIGS
    else:
        classname, rows = transfer.coco_rows(args.class_shard)
        rows = rows[:args.max_samples] if args.max_samples else rows
        manifest_dir = transfer.EVAL / "ovam/coco_captions"
        configurations = COCO_CONFIGS
    pipe = StableDiffusionPipeline.from_pretrained(
        plan["recipe"]["model"], torch_dtype=torch.float32, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    transfer._PIPE = pipe
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    selected_sa = sa_modules[transfer.SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    settings = {
        "package": "B2a/S4/B2b learned-token evaluation v1", "dataset": args.dataset,
        "condition": "conditional token CA; frozen CA; SA module 15 one pass; OVAM DenseCRF; threshold 0.475",
        "configurations": list(configurations), "plan": str(args.plan), "ca_seed": seed_spec,
        "selected_sa_module": selected_sa,
        "tv_note": "A3/A13/A23 repeat A1 for storage compatibility and are excluded from S4 summaries",
    }
    out = args.output_dir / f"{args.dataset}_evaluation/shards/class_{args.class_shard:02d}_{classname}.npz"
    writer_folds = folds if args.dataset == "voc" else {row["image_id"]: 0 for row in rows}
    writer = cv.ShardWriter(out, configurations, writer_folds, settings, thresholds=THRESHOLDS)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        fold = folds[row["image_id"]] if args.dataset == "voc" else None
        names, hidden = class_tokens(pipe, classname, fold, plan, args.learned_dir, released, "cuda", args.dataset)
        _TOKEN_NAMES[:] = names
        _TOKEN_HIDDEN[:] = [hidden]
        transfer._ATTRIBUTION_PROMPT[0] = row["attribution_prompt"]
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(pipe, row, manifest_dir, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {name: res for res, members in sample.modules_by_resolution.items() for name in members}
            seeds = []
            word_calls = recorder.ca_calls
            for name in names:
                recorder.ca_calls = recorder.token_calls[name]
                seeds.append(sa_common.ca_seed(recorder, ca_modules, ca_res, seed_spec, "cuda"))
            recorder.ca_calls = word_calls
            propagated = one_pass(recorder, selected_sa, torch.stack(seeds), "cuda")
            counts = np.stack([
                crf_count(value, sample.target_mask, sample.generated_image, args.crf_site)
                for value in propagated
            ])
            writer.add(row["image_id"], classname, counts,
                       np.full(len(configurations), np.nan, dtype=np.float64))
            ious = counts[:, 0, 0] / np.maximum(counts[:, 0, :3].sum(axis=1), 1)
            print(f"[{classname}] {row['image_id']} " + " ".join(
                f"{name}={iou:.3f}" for name, iou in zip(names, ious)
            ), flush=True)
        finally:
            recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear()
            sample.clear(); del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()

