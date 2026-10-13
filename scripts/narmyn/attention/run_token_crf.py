#!/usr/bin/env python3
"""OVAM's optimized tokens and DenseCRF on top of the earlier final method.

Exploratory experiment, run before the controlled protocol
(docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md). Every setting is frozen;
nothing is selected here. One class shard per call; each image is regenerated
live and only per-image tp/fp/fn are written.

Methods scored on the same regenerated image:

  ovam_implemented   OVAM as in src/segmentation/ovam_full.py (CA x SA, DenseCRF),
                     paper parameters per dataset
  plain_top4/*       earlier final method: class-word seed, 12 CA layers with the
                     fixed top-4 heads, U3.2, SA heads {0,1,2,4,5,6}, calls 15-30,
                     percentile 1-99, threshold 0.45, 1% cleanup
  opt_top4/*         same, with OVAM's optimized token as the CA attribution
  opt_all16/*        optimized token, all 16 CA layers x 8 heads (no CA selection)

  suffixes: ca_only (no SA), sa_iter1 / sa_iter4 (passes), +crf (OVAM's DenseCRF)

The optimized-token map uses OVAM's evaluation form: keys from [<SoT>, token],
softmax over those two tokens, unconditional query, channel 1. It is captured by
a recorder subclass defined here, so the shared capture code is not modified.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import dataclasses
import gc
import json
import re
import socket
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.attention_capture.live as live  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402
from src.attention_capture.memory import CompactAttentionRecorder  # noqa: E402
from src.segmentation.attention_aggregation import attention_to_mask  # noqa: E402
from src.segmentation.focused_self_attention import (  # noqa: E402
    build_cross_attention_layer_heads,
    build_cross_attention_seeds_from_heads,
    propagate_selected_module,
)
from src.segmentation.ovam_baseline import ovam_cross_attention_map  # noqa: E402
from src.segmentation.ovam_full import apply_ovam_densecrf, ovam_full_prediction  # noqa: E402
from src.segmentation.postprocessing import postprocess_mask  # noqa: E402

SA_MODULE = "up_blocks.3.attentions.2.transformer_blocks.0.attn1"
SA_HEADS = (0, 1, 2, 4, 5, 6)
LATE_START = 15
THRESHOLD = 0.45
SEEDS = ("plain_top4", "opt_top4", "opt_all16")
# OVAM paper parameters (SA floor alpha, threshold tau) per dataset, unoptimized.
OVAM_PARAMS = {"voc": {}, "coco": {"self_attention_minimum": 0.91, "threshold": 0.50}}
COCO_PATTERN = re.compile(
    r"^(?P<classname>.+)_sd-15_caption_(?P<caption_id>\d+)_(?P=classname)_(?P<seed>\d+)$"
)
OPT_HIDDEN: torch.Tensor | None = None  # [1, 2, 768]: <SoT> output + optimized class token


class OptTokenRecorder(CompactAttentionRecorder):
    """Store the optimized-token CA map next to the ordinary OVAM map.

    On COCO-Cap the capture already encodes the class name as a separate
    attribution prompt. The optimized pair is appended after it so that one key
    projection serves both, and the two softmaxes are computed separately.
    """

    def __init__(self, *args, attribution_hidden_states=None, **kwargs):
        self.n_plain = 0 if attribution_hidden_states is None else int(attribution_hidden_states.shape[1])
        combined = OPT_HIDDEN if attribution_hidden_states is None else torch.cat(
            (attribution_hidden_states, OPT_HIDDEN.to(attribution_hidden_states.dtype)), dim=1
        )
        super().__init__(*args, attribution_hidden_states=combined, **kwargs)

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        plain = None if attribution_key is None or not self.n_plain else attribution_key[:, :, : self.n_plain]
        super().record(module_name, kind, query, key, value, attention_mask, scale, attribution_key=plain)
        if kind == "cross" and attribution_key is not None:
            step = self.call_counts[module_name] - 1
            opt = attribution_key[0, :, self.n_plain :]
            scores = torch.matmul(query[0].float(), opt.float().transpose(-1, -2)) * scale
            self.step_tensors[step][f"{module_name}.opt_target_map"] = self._cpu(scores.softmax(dim=-1)[:, :, 1])


live.CompactAttentionRecorder = OptTokenRecorder


def evaluate(sample, tokenizer, rankings, device, crf_site, ovam_params) -> dict[str, torch.Tensor]:
    shape = sample.target_mask.shape
    image = sample.generated_image
    out: dict[str, torch.Tensor] = {}

    def crf(mask: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(apply_ovam_densecrf(image, mask.cpu().numpy().astype(bool), crf_site))

    def final(score: torch.Tensor) -> torch.Tensor:
        mask = attention_to_mask(score, shape, THRESHOLD, "percentile_01_99")
        return torch.from_numpy(postprocess_mask(mask.numpy(), "remove_fraction_0.01"))

    _, _, _, out["ovam_implemented"] = ovam_full_prediction(sample, crf_site, **ovam_params)

    ovam_raw = ovam_cross_attention_map(sample)
    seeds = {}
    for name, key in (("plain", "ovam_target_map"), ("opt", "opt_target_map")):
        view = dataclasses.replace(sample, cross_map_key=key)
        modules, resolutions, layer_heads = build_cross_attention_layer_heads(view, tokenizer)
        built = build_cross_attention_seeds_from_heads(modules, resolutions, layer_heads, rankings, ovam_raw)
        seeds[f"{name}_top4"] = built["top4_heads_percentile"]
        if name == "opt":
            seeds["opt_all16"] = built["all16_percentile"]
    matrix = torch.stack([seeds[n].flatten() for n in SEEDS], dim=1)
    propagated = propagate_selected_module(
        sample, SA_MODULE, matrix, device, track_seed_columns=tuple(range(len(SEEDS)))
    )
    steps = len(sample.timesteps)
    for column, name in enumerate(SEEDS):
        out[f"{name}/ca_only"] = attention_to_mask(seeds[name], shape, THRESHOLD, "percentile_01_99")
        for iteration in (1, 4):
            values = propagated.per_step_head_columns[iteration].reshape(steps, 8, 64, 64, len(SEEDS))[..., column]
            score = values[LATE_START:, list(SA_HEADS)].mean(dim=(0, 1))
            out[f"{name}/sa_iter{iteration}"] = final(score)
            out[f"{name}/sa_iter{iteration}+crf"] = crf(out[f"{name}/sa_iter{iteration}"])
    return out


def voc_rows(data_root: Path) -> tuple[list[dict[str, str]], Path]:
    manifest = data_root / "ovam/voc_sim/ovam_voc_sim.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle)), manifest.parent


def coco_rows(data_root: Path) -> tuple[list[dict[str, str]], Path]:
    """Same rows as scripts/build_coco_cap_manifest.py, without its image-file check."""
    coco_root = data_root / "ovam/coco_captions"
    with (data_root / "coco_captions_sampled.csv").open(newline="", encoding="utf-8") as handle:
        captions = {r["caption_id"]: r for r in csv.DictReader(handle)}
    rows = []
    for annotation in sorted((coco_root / "annotations").glob("*.png")):
        fields = COCO_PATTERN.fullmatch(annotation.stem).groupdict()
        source = captions[fields["caption_id"]]
        if source["category_name"] != fields["classname"]:
            raise ValueError(f"Class mismatch for {annotation.name}")
        rows.append({
            "image_id": f"ovam-coco-{fields['caption_id']}-{fields['classname'].replace(' ', '_')}-{fields['seed']}",
            "classname": fields["classname"],
            "prompt": source["caption"].lower(),
            "attribution_prompt": fields["classname"].lower(),
            "seed": fields["seed"],
            "num_inferece_steps": "30",
            "annotated": "True",
            "annotation": f"annotations/{annotation.name}",
            "word_included": source["word_included"],
        })
    return rows, coco_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--dataset", choices=("voc", "coco"), default="voc")
    parser.add_argument(
        "--data-root", type=Path, default=Path("datasets/ovam_experiment_with_dataset/evaluation"),
        help="OVAM release 'evaluation' folder (needs ovam/voc_sim, ovam/coco_captions/annotations, "
             "coco_captions_sampled.csv); see fetch_ovam_release_files.py",
    )
    parser.add_argument("--tokens", type=Path, default=Path("configs/ovam/tokens_VOC.npy"))
    parser.add_argument(
        "--fixed-stage1-results", type=Path,
        default=Path("outputs/segmentation/voc_sim_100_stage1_suite/results.json"),
    )
    parser.add_argument("--crf-site", type=Path, help="extra site-packages holding SimpleCRF's denseCRF")
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    global OPT_HIDDEN
    args = parse_args()
    name = "voc_sim" if args.dataset == "voc" else "coco_cap"
    output_dir = args.output_dir or Path("outputs/segmentation/narmyn_token_crf") / name
    rows, root = voc_rows(args.data_root) if args.dataset == "voc" else coco_rows(args.data_root)
    classname = sorted({r["classname"] for r in rows})[args.class_shard]
    rows = [r for r in rows if r["classname"] == classname and r["annotated"].lower() == "true"][: args.max_images]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    log_path = output_dir / f"{stem}.jsonl"
    done = {json.loads(line)["image_id"] for line in log_path.open()} if log_path.exists() else set()

    tokens = np.load(args.tokens, allow_pickle=True).item()
    rankings = json.loads(args.fixed_stage1_results.read_text())["settings"]["head_rankings"]
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        empty = pipe.tokenizer("", return_tensors="pt").input_ids.to(args.device)
        sot = pipe.text_encoder(empty)[0][:, :1]  # CLIP is causal: <SoT> output is prompt-independent
        token = torch.from_numpy(tokens[classname.replace(" ", "-")]).float().to(args.device)[None, None]
        OPT_HIDDEN = torch.cat((sot.float(), token), dim=1)

    print(f"{classname}: {len(rows)} annotated, {len(done)} done, node={socket.gethostname()}", flush=True)
    with log_path.open("a") as log:
        for row in rows:
            if row["image_id"] in done:
                continue
            start = time.time()
            sample = live.generate_attention_sample(pipe, row, root, "float16")
            try:
                preds = evaluate(sample, pipe.tokenizer, rankings, args.device, args.crf_site, OVAM_PARAMS[args.dataset])
                truth = sample.target_mask
                record = {"image_id": row["image_id"], "classname": classname,
                          "word_included": row.get("word_included"), "results": {}}
                for key, pred in preds.items():
                    p = pred.cpu().numpy().astype(bool)
                    tp, fp, fn = int((p & truth).sum()), int((p & ~truth).sum()), int((~p & truth).sum())
                    record["results"][key] = {"tp": tp, "fp": fp, "fn": fn,
                                              "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0}
                log.write(json.dumps(record) + "\n")
                log.flush()
                r = record["results"]
                print(f"{row['image_id']} ovam={r['ovam_implemented']['iou']:.3f} "
                      f"plain={r['plain_top4/sa_iter1']['iou']:.3f} opt={r['opt_top4/sa_iter1']['iou']:.3f} "
                      f"opt+crf={r['opt_top4/sa_iter1+crf']['iou']:.3f} ({time.time() - start:.0f}s)", flush=True)
            finally:
                sample.clear()
                del sample
                gc.collect()
                torch.cuda.empty_cache()
    if args.max_images is None:
        (output_dir / f"{stem}.COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
