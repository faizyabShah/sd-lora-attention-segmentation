#!/usr/bin/env python3
"""Run the locked token-SG3 confirmation on one VOC or COCO class shard."""

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
import scripts.run_sg1_sg3_graph_refinement as graph_screen  # noqa: E402, installs DualSeedRecorder
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.graph_refinement import (  # noqa: E402
    anchor_masks, confidence_locked_trajectory, minmax_vector,
)

WORK = 64
THRESHOLDS = transfer.THRESHOLDS
CONFIG_PATH = ROOT / "configs/token_sg3_confirmation.json"
OUT = ROOT / "outputs/segmentation/controlled_attention_cv/token_sg3_confirmation"


@torch.inference_mode()
def fixed_maps(recorder, module: str, token_seed64: torch.Tensor, spec: dict, device: str):
    seed = minmax_vector(token_seed64.reshape(1, -1).float())
    calls = recorder.sa_qk[module]
    affinity_sum = torch.zeros((WORK * WORK, WORK * WORK), device=device)
    one = torch.zeros_like(seed)
    two = torch.zeros_like(seed)
    heads = calls[0][0].shape[0]
    for query, key in calls:
        q, k = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
        affinity_sum += affinity.sum(dim=0)
        current = torch.matmul(affinity, seed.T).transpose(-1, -2)
        one += current.mean(dim=0)
        two += torch.matmul(affinity, current.transpose(-1, -2)).transpose(-1, -2).mean(dim=0)
        del q, k, affinity, current
    one /= len(calls)
    two /= len(calls)
    mean_affinity = affinity_sum / (len(calls) * heads)
    sg3 = spec["sg3"]
    foreground, background = anchor_masks(
        seed, sg3["foreground_quantile"], sg3["background_quantile"],
        WORK, sg3["border_width"],
    )
    locked = confidence_locked_trajectory(
        mean_affinity, seed, foreground, background, sg3["steps"]
    )[-1]
    return one.reshape(WORK, WORK), two.reshape(WORK, WORK), locked.reshape(WORK, WORK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=ROOT / "configs/controlled_sa_seed_frozen.json")
    parser.add_argument("--tokens", type=Path, default=ROOT / "configs/ovam/tokens_VOC.npy")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--crf-site", type=Path, required=True)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    spec = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ca_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not ca_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    voc = args.dataset == "voc"
    classname, rows = (transfer.voc_rows if voc else transfer.coco_rows)(args.class_shard)
    rows = rows[:args.max_samples] if args.max_samples else rows
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True
    ).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [name.removesuffix(".processor") for name in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(name for name in processors if name.endswith("attn2")))
    sa_modules = tuple(sorted(name for name in processors if name.endswith("attn1")))
    selected = sa_modules[transfer.SELECTED_SA_MODULE]
    ca_spec = sa_common.resolve_seed_spec(ca_json, ca_modules, 31)
    released = np.load(args.tokens, allow_pickle=True).item()
    prompt = rows[0]["attribution_prompt"]
    if any(row["attribution_prompt"] != prompt for row in rows):
        raise RuntimeError(f"{classname}: attribution prompt must be constant within a class shard")
    pair, class_hidden = graph_screen.hidden_states(
        pipe, classname, prompt, released, args.device, not voc
    )
    graph_screen._EXTRA.update(
        hidden=pair, class_hidden=class_hidden, separate_class_prompt=not voc
    )
    names = tuple(spec["configurations"])
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if voc else {r["image_id"]: 0 for r in rows}
    settings = {
        "package": "token_sg3_confirmation_v1", "protocol": "Section 72",
        "dataset": args.dataset, "spec": spec, "ca_seed": ca_spec,
        "selected_sa_module": selected, "thresholds": list(THRESHOLDS),
        "configurations": list(names), "crf_site": str(args.crf_site),
        "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
    }
    shard = args.output_dir / args.dataset / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz"
    writer = cv.ShardWriter(shard, names, folds, settings, thresholds=THRESHOLDS)
    t45 = THRESHOLDS.index(0.45)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(
            pipe, row, manifest_dir, "float16", cross_map_kind="conditional"
        )
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {n: r for r, modules in sample.modules_by_resolution.items() for n in modules}
            word_calls, recorder.ca_calls = recorder.ca_calls, recorder.token_calls
            token_seed = sa_common.ca_seed(recorder, ca_modules, ca_res, ca_spec, args.device)
            recorder.ca_calls = word_calls
            one, two, locked = fixed_maps(recorder, selected, token_seed, spec, args.device)
            recorder.sa_qk.clear()
            raw = (one, two, locked)
            raw_counts, raw_ap = cv.score_maps(
                torch.stack(raw), sample.target_mask, "minmax", thresholds=THRESHOLDS
            )
            crf_counts = [
                transfer.crf_counts(m, sample.target_mask, sample.generated_image, args.crf_site)[0]
                for m in raw
            ]
            counts = np.stack((raw_counts[0], crf_counts[0], raw_counts[1], crf_counts[1], raw_counts[2], crf_counts[2]))
            ap = np.asarray((raw_ap[0], np.nan, raw_ap[1], np.nan, raw_ap[2], np.nan))
            writer.add(row["image_id"], classname, counts, ap)
            print(f"[{args.dataset}:{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) " +
                  " ".join(f"{name}={counts[i,t45,0]/max(counts[i,t45,:3].sum(),1):.3f}" for i,name in enumerate(names)),
                  flush=True)
        finally:
            recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear()
            sample.clear(); del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
