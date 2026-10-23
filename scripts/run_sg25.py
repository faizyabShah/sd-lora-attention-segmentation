#!/usr/bin/env python3
"""Resumable SG25 and Pre-SG26 class shard; generate and score one image at a time."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_sg18_temporal_windowed as sg18
from scripts import run_coco_cap_transfer as transfer
from scripts.run_sg24 import atomic_npz, mid_p2
from src.attention_capture import live
from src.segmentation import controlled_cv as cv
from src.segmentation import sg24_ca, sg25_sa
from src.segmentation.ovam_full import _import_densecrf, OVAM_CRF_PARAMETERS

BASE = ROOT / "outputs/segmentation/controlled_attention_cv/sg25"
SPEC = ROOT / "configs/sg25.json"
NAMES = tuple(name + suffix for name in sg25_sa.NAMES for suffix in ("", "+R2"))
ARRIVAL_FIELDS = ("pixels", "tp", "fp", "fp_gt16", "fp_gt64", "fp_separate")


def arrival_counts(depth_maps: torch.Tensor, truth: np.ndarray, threshold: float) -> np.ndarray:
    """First crossing among depths 1..5 after each depth's normal 64x64 minmax."""
    if depth_maps.shape != (5, 64, 64):
        raise ValueError("Expected exactly five 64x64 depth maps")
    normalized = cv.normalize_batch(depth_maps.float(), "minmax")
    full = F.interpolate(normalized[:, None], size=truth.shape, mode="bilinear", align_corners=False)[:, 0]
    masks = (full >= threshold).cpu().numpy()
    arrived = np.zeros(truth.shape, dtype=bool)
    distance = ndimage.distance_transform_edt(~truth)
    counts = np.zeros((5, len(ARRIVAL_FIELDS)), dtype=np.int64)
    for k, mask in enumerate(masks):
        first = mask & ~arrived
        arrived |= mask
        fp = first & ~truth
        labels, _ = ndimage.label(mask)
        touched = np.unique(labels[mask & truth])
        separate = fp & ~np.isin(labels, touched[touched > 0])
        counts[k] = (first.sum(), (first & truth).sum(), fp.sum(),
                     (fp & (distance > 16)).sum(), (fp & (distance > 64)).sum(),
                     separate.sum())
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--class-shard", type=int, choices=range(20), required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--verify-fine", action="store_true", help="on smoke samples, compare fine map to SG24's original operator")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site", type=Path, default=ROOT / "outputs/dependencies/simplecrf_py310")
    args = p.parse_args()
    spec = json.loads(SPEC.read_text())
    frozen = json.loads((ROOT / "configs/controlled_sa_seed_frozen.json").read_text())
    if not frozen["frozen"] or frozen["modules"] != json.loads((ROOT / "configs/sg24.json").read_text())["selected_ca_modules"]:
        raise ValueError("A3 CA module set changed")
    if spec["thresholds"] != list(transfer.THRESHOLDS):
        raise ValueError("threshold grid changed")
    classname, all_rows = (transfer.voc_rows if args.dataset == "voc" else transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    pipe, ca_modules, selected_sa, _ = sg18.make_pipe(
        args.model, "cuda", json.loads((ROOT / "configs/sg12_crf_sg14_sg15.json").read_text()), frozen)
    if selected_sa != spec["sa_modules"]["64"]:
        raise ValueError(f"fine module changed: {selected_sa}")
    all_sa = {n.removesuffix(".processor") for n in pipe.unet.attn_processors if n.endswith(".attn1.processor")}
    modules = {int(side): name for side, name in spec["sa_modules"].items()}
    if set(modules.values()) - all_sa:
        raise ValueError(f"missing SA module: {set(modules.values()) - all_sa}")
    sg25_sa.SG25Recorder.pipe = pipe
    sg25_sa.SG25Recorder.ca_modules = tuple(ca_modules[i] for i in frozen["modules"])
    sg25_sa.SG25Recorder.selected_modules = sg25_sa.SG25Recorder.ca_modules
    sg25_sa.SG25Recorder.selected_sa = selected_sa
    sg25_sa.SG25Recorder.extra_sa = (modules[16], modules[32])
    sg25_sa.SG25Recorder.experiment = "a"
    sg25_sa.SG25Recorder.enabled = ("A3",)
    recorders = []
    original_init = sg25_sa.SG25Recorder.__init__
    def tracked_init(self, *a, **k):
        original_init(self, *a, **k)
        recorders.append(self)
    sg25_sa.SG25Recorder.__init__ = tracked_init
    live.CompactAttentionRecorder = sg25_sa.SG25Recorder
    _import_densecrf(args.crf_site)
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if args.dataset == "voc" else {r["image_id"]: 0 for r in all_rows}
    out = args.output_dir or BASE
    stem = f'class_{args.class_shard:02d}_{classname.replace(" ", "_")}'
    path = out / args.dataset / "shards" / (stem + ".npz")
    side = path.with_suffix(".arrival.npz")
    settings = {"spec": spec, "dataset": args.dataset, "model": args.model,
                "ca_modules": list(sg25_sa.SG25Recorder.ca_modules),
                "sa_modules": modules, "crf": list(OVAM_CRF_PARAMETERS)}
    writer = cv.ShardWriter(path, NAMES, folds, settings, spec["thresholds"])
    arrivals = {}
    if side.exists():
        with np.load(side, allow_pickle=False) as data:
            arrivals = {str(name): counts for name, counts in zip(data["image_ids"], data["counts"])}
    if writer.completed - set(arrivals):
        raise ValueError("metric shard ahead of arrival sidecar")
    manifest = transfer.EVAL / ("ovam/voc_sim" if args.dataset == "voc" else "ovam/coco_captions")
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        from src.segmentation.attention_aggregation import find_token_subsequence
        prompt = row["attribution_prompt"]
        ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
        class_ids = pipe.tokenizer(row["classname"], add_special_tokens=False)["input_ids"]
        target = find_token_subsequence(ids, class_ids)
        sg25_sa.SG25Recorder.content = sg24_ca.content_positions(pipe.tokenizer, prompt, tuple(target))
        sg25_sa.SG25Recorder.attribution_prompt = prompt
        recorders.clear()
        sample = live.generate_attention_sample(pipe, row, manifest, "float16", cross_map_kind="conditional")
        recorder = recorders[-1]
        try:
            if len(recorder.sa_qk) != 11:
                raise ValueError("missing fine middle-window calls")
            seed = recorder.seeds()["selected/A3"]
            sa = {16: recorder.multiscale_qk[modules[16]],
                  32: recorder.multiscale_qk[modules[32]], 64: recorder.sa_qk}
            maps, depths = sg25_sa.aggregate(sa, seed)
            if args.verify_fine:
                reference = mid_p2(recorder.sa_qk, seed.reshape(1, -1))[0]
                torch.testing.assert_close(maps["FINE-P2"], reference, rtol=1e-5, atol=2e-6)
            truth = np.asarray(sample.target_mask, bool)
            image = np.asarray(sample.generated_image, np.uint8)
            counts = np.zeros((len(NAMES), len(spec["thresholds"]), 4), np.int64)
            ap = np.full(len(NAMES), np.nan)
            for j, name in enumerate(sg25_sa.NAMES):
                raw_counts, raw_ap = cv.score_maps(maps[name][None].cpu(), truth, "minmax", spec["thresholds"])
                counts[2*j] = raw_counts[0]
                ap[2*j] = raw_ap[0]
                counts[2*j+1] = transfer.crf_counts(maps[name], truth, image, args.crf_site)[0]
            arrivals[row["image_id"]] = arrival_counts(depths, truth, 0.425)
            atomic_npz(side, image_ids=np.asarray(list(arrivals)), counts=np.stack(list(arrivals.values())))
            writer.add(row["image_id"], classname, counts, ap)
            print(f"[sg25:{args.dataset}:{classname}] {row['image_id']} {len(writer.completed)}/{len(rows)}", flush=True)
        finally:
            recorder.clear(); sample.clear(); recorders.clear()
            del recorder, sample
            gc.collect(); torch.cuda.empty_cache()
    if args.max_samples is None:
        if writer.completed != {r["image_id"] for r in all_rows} or set(arrivals) != writer.completed:
            raise ValueError("incomplete SG25 class shard")
        writer.finish()


if __name__ == "__main__":
    main()
