#!/usr/bin/env python3
"""Run the preregistered SG1/SG3 graph screen on one VOC or COCO class."""

from __future__ import annotations

import argparse
import gc
import json
import math
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
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.graph_refinement import (  # noqa: E402
    absorbing_walk, anchor_masks, confidence_locked_trajectory, minmax_vector,
)
from src.segmentation.sg_visual_audit import (  # noqa: E402
    best_thresholds, released_image, render_variant_pages,
)

WORK = 64
THRESHOLDS = transfer.THRESHOLDS
SPEC_PATH = ROOT / "configs" / "sg1_sg3_graph_refinement.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "sg1_sg3_graph_refinement"
_EXTRA: dict[str, object] = {"hidden": None, "separate_class_prompt": False}


def parameter_cells(spec: dict) -> list[tuple[float, float, int]]:
    return [(f, b, w) for f in spec["foreground_quantiles"]
            for b in spec["background_quantiles"] for w in spec["border_widths"]]


def tag(f: float, b: float, w: int) -> str:
    return f"f{int(round(f*100)):02d}_b{int(round(b*100)):02d}_r{w}"


def configurations(spec: dict) -> tuple[str, ...]:
    names: list[str] = []
    for attribution in spec["attributions"]:
        names += [f"{attribution}/REF", f"{attribution}/P1"]
        names += [f"{attribution}/SG1/{tag(*p)}" for p in parameter_cells(spec)]
        names += [f"{attribution}/SG3/{tag(*p)}_n{n}" for p in parameter_cells(spec) for n in spec["sg3_steps"]]
    return tuple(names)


class DualSeedRecorder(sa_common.SAPackageRecorder):
    """Frozen word CA plus OVAM's released-token CA from the same generation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = _EXTRA["hidden"]
        if hidden is None:
            raise RuntimeError("Attribution hidden states were not initialized")
        parts = [hidden]
        if _EXTRA["separate_class_prompt"]:
            parts.insert(0, _EXTRA["class_hidden"])
        self.attribution_hidden_states = torch.cat(parts, dim=1)
        self.token_calls: dict[str, list[torch.Tensor]] = {}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind != "cross":
            return sa_common.SAPackageRecorder.record(
                self, module_name, kind, query, key, value, attention_mask, scale
            )
        offset = 77 if _EXTRA["separate_class_prompt"] else 0
        if offset:
            key = torch.stack((key[0], attribution_key[0][:, :77]))
            attention_mask = None
        sa_common.SAPackageRecorder.record(
            self, module_name, kind, query, key, value, attention_mask, scale
        )
        with torch.inference_mode():
            pair = attribution_key[0][:, offset:offset + 2].float()
            probability = (torch.matmul(query[1].float(), pair.transpose(-1, -2)) * scale).softmax(dim=-1)[..., 1]
            self.token_calls.setdefault(module_name, []).append(probability.half())


sa_common.live.CompactAttentionRecorder = DualSeedRecorder


def hidden_states(pipe, classname: str, prompt: str, released: dict, device: str, separate: bool):
    token_key = classname.replace(" ", "-")
    with torch.inference_mode():
        empty = pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
        sot = pipe.text_encoder(empty)[0][:, :1].float()
        learned = torch.from_numpy(np.asarray(released[token_key], dtype=np.float32)).to(device)[None, None]
        pair = torch.cat((sot, learned), dim=1)
        if not separate:
            return pair, None
        ids = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(device)
        return pair, pipe.text_encoder(ids)[0].float()


@torch.inference_mode()
def graph_maps(recorder, module: str, seeds64: torch.Tensor, spec: dict, device: str):
    seeds = minmax_vector(seeds64.reshape(2, -1).float())
    calls = recorder.sa_qk[module]
    affinity_sum = torch.zeros((WORK * WORK, WORK * WORK), device=device)
    one = torch.zeros_like(seeds)
    two = torch.zeros_like(seeds)
    heads = calls[0][0].shape[0]
    for query, key in calls:
        q, k = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        affinity_sum += affinity.sum(dim=0)
        current = torch.matmul(affinity, seeds.T).transpose(-1, -2)
        one += current.mean(dim=0)
        two += torch.matmul(affinity, current.transpose(-1, -2)).transpose(-1, -2).mean(dim=0)
        del q, k, affinity, current
    one /= len(calls)
    two /= len(calls)
    mean_affinity = affinity_sum / (len(calls) * heads)
    params = parameter_cells(spec)
    sg1, sg3 = [], []
    convergence = []
    for f, b, width in params:
        foreground, background = anchor_masks(seeds, f, b, WORK, width)
        harmonic, iterations, residual = absorbing_walk(
            mean_affinity, seeds, foreground, background,
            spec["sg1_tolerance"], spec["sg1_max_iterations"],
        )
        locked = confidence_locked_trajectory(
            mean_affinity, seeds, foreground, background, max(spec["sg3_steps"])
        )
        sg1.append(harmonic)
        sg3.append(torch.stack([locked[n - 1] for n in spec["sg3_steps"]]))
        convergence.append({"cell": tag(f, b, width), "iterations": iterations, "residual": residual})
    per_attribution = []
    sg1_stack = torch.stack(sg1)  # [parameter, attribution, pixel]
    sg3_stack = torch.stack(sg3)  # [parameter, step, attribution, pixel]
    for a in range(2):
        maps = [one[a], two[a]]
        maps += [sg1_stack[p, a] for p in range(len(params))]
        maps += [sg3_stack[p, n, a] for p in range(len(params)) for n in range(len(spec["sg3_steps"]))]
        per_attribution += maps
    return torch.stack(per_attribution).reshape(-1, WORK, WORK), convergence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=ROOT / "configs/controlled_sa_seed_frozen.json")
    parser.add_argument("--tokens", type=Path, default=ROOT / "configs/ovam/tokens_VOC.npy")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--visual-dir", type=Path,
                        help="Render the first image and every registered cell; do not write metric shards")
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    seed_spec_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_spec_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    voc = args.dataset == "voc"
    classname, rows = (transfer.voc_rows if voc else transfer.coco_rows)(args.class_shard)
    rows = rows[:1] if args.visual_dir else (rows[:args.max_samples] if args.max_samples else rows)
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    selected = sa_modules[transfer.SELECTED_SA_MODULE]
    ca_spec = sa_common.resolve_seed_spec(seed_spec_json, ca_modules, 31)
    released = np.load(args.tokens, allow_pickle=True).item()
    prompt = rows[0]["attribution_prompt"]
    if any(row["attribution_prompt"] != prompt for row in rows):
        raise RuntimeError(f"{classname}: attribution prompt must be constant within a class shard")
    pair, class_hidden = hidden_states(pipe, classname, prompt, released, args.device, not voc)
    _EXTRA.update(hidden=pair, class_hidden=class_hidden, separate_class_prompt=not voc)
    names = configurations(spec)
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if voc else {r["image_id"]: 0 for r in rows}
    settings = {
        "package": "sg1_sg3_graph_refinement_v1", "protocol": "Section 68",
        "dataset": args.dataset, "spec": spec, "ca_seed": ca_spec,
        "selected_sa_module": selected, "thresholds": list(THRESHOLDS),
        "configurations": list(names), "model": args.model,
        "model_dtype": "float32", "qk_storage_dtype": "float16",
    }
    shard = args.output_dir / args.dataset / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    writer = None if args.visual_dir else cv.ShardWriter(shard, names, folds, settings, thresholds=THRESHOLDS)
    visual_thresholds = best_thresholds(OUT) if args.visual_dir else {}
    t45 = THRESHOLDS.index(0.45)
    for row in rows:
        if writer is not None and row["image_id"] in writer.completed:
            continue
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(
            pipe, row, manifest_dir, "float16", cross_map_kind="conditional"
        )
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {n: r for r, modules in sample.modules_by_resolution.items() for n in modules}
            word_seed = sa_common.ca_seed(recorder, ca_modules, ca_res, ca_spec, args.device)
            word_calls, recorder.ca_calls = recorder.ca_calls, recorder.token_calls
            token_seed = sa_common.ca_seed(recorder, ca_modules, ca_res, ca_spec, args.device)
            recorder.ca_calls = word_calls
            maps, convergence = graph_maps(
                recorder, selected, torch.stack((word_seed, token_seed)), spec, args.device
            )
            recorder.sa_qk.clear()
            if args.visual_dir:
                ca_maps = [word_seed if name.startswith("W/") else token_seed for name in names]
                render_variant_pages(
                    args.visual_dir, "sg1_sg3", classname, row["image_id"],
                    released_image(row, manifest_dir), np.asarray(sample.target_mask, bool),
                    names, ca_maps, list(maps), visual_thresholds,
                )
                print(f"[visual:{classname}] {row['image_id']} cells={len(names)}", flush=True)
                continue
            counts, ap = cv.score_maps(maps, sample.target_mask, "minmax", thresholds=THRESHOLDS)
            writer.add(row["image_id"], classname, counts, ap)
            print(f"[{args.dataset}:{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) "
                  f"W/REF={counts[names.index('W/REF'), t45, 0] / max(counts[names.index('W/REF'), t45, :3].sum(), 1):.3f} "
                  f"T/REF={counts[names.index('T/REF'), t45, 0] / max(counts[names.index('T/REF'), t45, :3].sum(), 1):.3f} "
                  f"SG1max={max(x['iterations'] for x in convergence)}", flush=True)
        finally:
            recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear()
            sample.clear(); del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if writer is not None and args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
