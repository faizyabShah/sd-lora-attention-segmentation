#!/usr/bin/env python3
"""COCO-Cap transfer of the frozen pipelines (phase 3 registry Section 7), one class shard per call.

Nothing is chosen on COCO-Cap: every stage uses the configuration and mask
threshold frozen on VOC-Sim. Counts are stored for the 17 thresholds of the
phase 3 grid so the frozen threshold can be read off after P3-C; the stages are

  CA/seed      the frozen CA seed alone (protocol Section 11, CA row)
  SA/REF       CA + one-pass SA through module 15 (Section 11, SA row)
  P3/P1        two passes, mean_i(A_i^2 c)        (P3-C mandatory cell)
  P3/R2        DenseCRF on the SA/REF mask        (P3-C mandatory cell)
  P3/P1+R2     DenseCRF on the P3/P1 mask         (P3-C mandatory cell)

Attribution. The class word is often absent from a COCO caption, so, as in
OVAM's COCO-Cap evaluation, the class name is encoded as a separate
attribution prompt. To keep the frozen CA convention exactly (conditional
query, softmax over all 77 padded positions, mean over all class pieces), the
attribution prompt is encoded padded to 77 tokens, and the CA uses its keys
(``attribution_key``) instead of the caption's. ``--self-test-voc`` runs the
same path on VOC-Sim with the caption itself as the separate prompt, which
must reproduce the frozen VOC pipeline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_sa_package_b as sa_common  # noqa: E402
from scripts.run_fusion_calibration import upsampled  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.fusion_variants import minmax  # noqa: E402
from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402
from src.segmentation.propagation_variants import power_trajectory  # noqa: E402

EVAL = ROOT / "datasets" / "ovam_experiment_with_dataset" / "evaluation"
WORK = 64
SELECTED_SA_MODULE = 15
THRESHOLDS = tuple(round(0.30 + 0.025 * i, 3) for i in range(17))
CONFIGURATIONS = ("CA/seed", "SA/REF", "P3/P1", "P3/R2", "P3/P1+R2")
ANNOTATION = re.compile(r"^(?P<classname>.+)_sd-15_caption_(?P<caption_id>\d+)_(?P=classname)_(?P<seed>\d+)$")

_PIPE: StableDiffusionPipeline | None = None


class TransferRecorder(sa_common.SAPackageRecorder):
    """SA package recorder whose CA uses the separate, 77-token padded attribution prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.attribution_hidden_states is not None and _PIPE is not None:
            prompt = _ATTRIBUTION_PROMPT[0]
            ids = _PIPE.tokenizer(prompt, padding="max_length", max_length=_PIPE.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt").input_ids.to(_PIPE.device)
            with torch.inference_mode():
                self.attribution_hidden_states = _PIPE.text_encoder(ids)[0]

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind == "cross" and attribution_key is not None:
            if attribution_key.shape[-2] != 77:
                raise RuntimeError(f"Expected a 77-token attribution key, got {tuple(attribution_key.shape)}")
            # Conditional query against the attribution keys (the same keys for both CFG halves).
            key = torch.stack((key[0], attribution_key[0]))
            attention_mask = None
        super().record(module_name, kind, query, key, value, attention_mask, scale)


_ATTRIBUTION_PROMPT = [""]
sa_common.live.CompactAttentionRecorder = TransferRecorder


def coco_rows(class_shard: int) -> tuple[str, list[dict[str, str]]]:
    with (EVAL / "coco_captions_sampled.csv").open(newline="", encoding="utf-8") as handle:
        captions = {r["caption_id"]: r for r in csv.DictReader(handle)}
    rows = []
    for annotation in sorted((EVAL / "ovam/coco_captions/annotations").glob("*.png")):
        m = ANNOTATION.fullmatch(annotation.stem)
        if m is None:
            raise ValueError(f"Unexpected annotation name: {annotation.name}")
        f = m.groupdict()
        source = captions[f["caption_id"]]
        rows.append({"image_id": f"ovam-coco-{f['caption_id']}-{f['classname'].replace(' ', '_')}-{f['seed']}",
                     "classname": f["classname"], "prompt": source["caption"].lower(),
                     "attribution_prompt": f["classname"].lower(), "seed": f["seed"], "num_inferece_steps": "30",
                     "annotated": "True", "annotation": f"annotations/{annotation.name}",
                     "caption_id": f["caption_id"], "word_included": source["word_included"]})
    classname = sorted({r["classname"] for r in rows})[class_shard]
    return classname, [r for r in rows if r["classname"] == classname]


def voc_rows(class_shard: int) -> tuple[str, list[dict[str, str]]]:
    with (EVAL / "ovam/voc_sim/ovam_voc_sim.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["annotated"].lower() == "true"]
    classname = sorted({r["classname"] for r in rows})[class_shard]
    return classname, [dict(r, attribution_prompt=r["prompt"]) for r in rows if r["classname"] == classname]


@torch.inference_mode()
def propagate(recorder, module_name: str, seed64: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``mean_i(A_i c)`` and ``mean_i(A_i^2 c)`` over every call and head of the frozen SA module."""
    seed = seed64.reshape(-1)
    one = torch.zeros_like(seed)
    two = torch.zeros_like(seed)
    calls = recorder.sa_qk[module_name]
    for query, key in calls:
        query, key = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1)
        trajectory = power_trajectory(affinity, seed, 2)  # [2, heads, P]
        one += trajectory[0].mean(dim=0)
        two += trajectory[1].mean(dim=0)
        del query, key, affinity, trajectory
    return (one / len(calls)).reshape(WORK, WORK), (two / len(calls)).reshape(WORK, WORK)


def crf_counts(map64: torch.Tensor, target: np.ndarray, image: np.ndarray, crf_site) -> np.ndarray:
    full = upsampled(minmax(map64), tuple(np.asarray(target).shape))
    out = np.zeros((1, len(THRESHOLDS), 4), dtype=np.int64)
    for i, t in enumerate(THRESHOLDS):
        refined = apply_ovam_densecrf(image, (full >= t).cpu().numpy(), crf_site)
        out[0, i] = cv.score_masks(refined[None], target, thresholds=(t,))[0, 0]
    return out


def main() -> None:
    global _PIPE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_frozen.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation/coco_cap_transfer"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crf-site", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--self-test-voc", action="store_true",
                        help="VOC-Sim rows with the caption as the separate prompt (must equal the frozen pipeline)")
    args = parser.parse_args()
    seed_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    classname, rows = (voc_rows if args.self_test_voc else coco_rows)(args.class_shard)
    rows = rows[: args.max_samples] if args.max_samples else rows
    manifest_dir = EVAL / ("ovam/voc_sim" if args.self_test_voc else "ovam/coco_captions")
    _PIPE = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    _PIPE.set_progress_bar_config(disable=True)
    processors = [n.removesuffix(".processor") for n in _PIPE.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith("attn2")))
    sa_modules = tuple(sorted(n for n in processors if n.endswith("attn1")))
    selected = sa_modules[SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    settings = {
        "package": "coco_cap_transfer_v1", "protocol": "phase 3 registry Section 7; protocol Section 48",
        "dataset": "voc_sim self-test" if args.self_test_voc else "coco_cap",
        "attribution": "separate class prompt, encoded padded to 77 tokens; conditional query; all class pieces",
        "ca_seed": seed_spec, "selected_sa_module": selected, "thresholds": list(THRESHOLDS),
        "configurations": list(CONFIGURATIONS), "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
        "note": "nothing chosen on COCO-Cap; no folds (fold 0 for every image)",
    }
    out_dir = args.output_dir / ("self_test_voc" if args.self_test_voc else "coco_cap")
    writer = cv.ShardWriter(out_dir / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz",
                            CONFIGURATIONS, {r["image_id"]: 0 for r in rows}, settings, thresholds=THRESHOLDS)
    t45 = THRESHOLDS.index(0.45)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        _ATTRIBUTION_PROMPT[0] = row["attribution_prompt"]
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(_PIPE, row, manifest_dir, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {n: r for r, names in sample.modules_by_resolution.items() for n in names}
            seed64 = sa_common.ca_seed(recorder, ca_modules, ca_res, seed_spec, args.device)
            recorder.ca_calls.clear()
            one, two = propagate(recorder, selected, seed64, args.device)
            recorder.sa_qk.clear()
            counts, ap = cv.score_maps(torch.stack([seed64, one, two]), sample.target_mask, "minmax", thresholds=THRESHOLDS)
            crf = np.concatenate([crf_counts(m, sample.target_mask, sample.generated_image, args.crf_site) for m in (one, two)])
            writer.add(row["image_id"], classname, np.concatenate((counts, crf)), np.concatenate((ap, [np.nan, np.nan])))
            all_counts = np.concatenate((counts, crf))
            print(f"[{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) "
                  + " ".join(f"{n}={all_counts[i, t45, 0] / max(all_counts[i, t45, :3].sum(), 1):.3f}" for i, n in enumerate(CONFIGURATIONS)),
                  flush=True)
        finally:
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            sample.clear()
            del sample, recorder
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
