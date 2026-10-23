#!/usr/bin/env python3
"""Resumable SG24-A or SG24-B class shard, one generated sample in memory at a time."""
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
from src.attention_capture import live
from src.segmentation import controlled_cv as cv
from src.segmentation import sg24_ca
from src.segmentation.ovam_full import _import_densecrf, OVAM_CRF_PARAMETERS

BASE = ROOT / "outputs/segmentation/controlled_attention_cv"
SPEC = ROOT / "configs/sg24.json"


def names_for(experiment: str, method: str | None = None) -> tuple[str, ...]:
    cells = ("A0", "A1", "A2", "A3", "A4") if experiment == "a" else (f"B0-{method}", f"B1-{method}")
    return tuple(f"{name}/{level}" for name in cells for level in ("CA", "FINAL"))


@torch.inference_mode()
def mid_p2(calls, seeds: torch.Tensor) -> torch.Tensor:
    if len(calls) != 11 or seeds.shape[1] != 4096:
        raise ValueError("MID-P2 requires 11 calls and 64x64 seeds")
    columns = seeds.T.float()
    total = torch.zeros_like(columns)
    for query, key in calls:
        q, k = query.float(), key.float()
        attention = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(q.shape[-1]), dim=-1)
        once = attention @ columns
        twice = attention @ once
        total += twice.sum(0)
        del q, k, attention, once, twice
    return (total / (8 * 11)).T.reshape(-1, 64, 64)


def masks_at(maps: torch.Tensor, shape, threshold=.45) -> np.ndarray:
    maps = cv.normalize_batch(maps, "minmax")
    up = F.interpolate(maps[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0]
    return (up >= threshold).cpu().numpy()


def diagnostics(masks, truth):
    distance = ndimage.distance_transform_edt(~truth)
    base = masks[0]
    values = []
    for pred in masks:
        labels, _ = ndimage.label(pred)
        touch = np.unique(labels[pred & truth])
        separate = pred & ~np.isin(labels, touch[touch > 0])
        gained, removed = pred & ~base, base & ~pred
        values.append([int((pred & truth).sum()), int((pred & ~truth).sum()), int(pred.sum()),
                       int((pred & ~truth & (distance > 16)).sum()),
                       int((pred & ~truth & (distance > 64)).sum()),
                       int((separate & ~truth).sum()), int((gained | removed).sum()),
                       int((gained & truth).sum() + (removed & ~truth).sum()),
                       int((gained & ~truth).sum() + (removed & truth).sum())])
    return np.asarray(values, np.int64)


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=("a", "b"), required=True)
    p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--class-shard", type=int, choices=range(20), required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--b-method", choices=("A3", "A4"), help="smoke only; production reads A selection")
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site", type=Path, default=ROOT / "outputs/dependencies/simplecrf_py310")
    args = p.parse_args()
    spec = json.loads(SPEC.read_text())
    frozen = json.loads((ROOT / "configs/controlled_sa_seed_frozen.json").read_text())
    if not frozen["frozen"] or frozen["modules"] != spec["selected_ca_modules"]:
        raise ValueError("frozen CA observation set has changed")
    if spec["thresholds"] != list(transfer.THRESHOLDS):
        raise ValueError("threshold grid changed")
    method = None
    if args.experiment == "b":
        path = BASE / "sg24_a" / "selection_for_b.json"
        if args.b_method:
            method = args.b_method
        else:
            if not path.is_file():
                raise FileNotFoundError(f"SG24-A VOC selection is required: {path}")
            method = json.loads(path.read_text())["automatic_method"]
    names = names_for(args.experiment, method)
    out = args.output_dir or BASE / f"sg24_{args.experiment}"
    classname, all_rows = (transfer.voc_rows if args.dataset == "voc" else transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    pipe, ca_modules, selected_sa, _ = sg18.make_pipe(args.model, "cuda", json.loads((ROOT / "configs/sg12_crf_sg14_sg15.json").read_text()), frozen)
    selected_ca = tuple(ca_modules[i] for i in spec["selected_ca_modules"])
    capture_ca = selected_ca if args.experiment == "a" else ca_modules
    if len(ca_modules) != 16 or len(selected_ca) != 11 or sg24_ca.REF_SUFFIX not in ca_modules:
        raise ValueError("unexpected SD1.5 CA module topology")
    # Patch the recorder factory in the same way as the existing streaming runners.
    sg24_ca.SG24Recorder.pipe = pipe
    sg24_ca.SG24Recorder.ca_modules = capture_ca
    sg24_ca.SG24Recorder.selected_modules = selected_ca
    sg24_ca.SG24Recorder.selected_sa = selected_sa
    sg24_ca.SG24Recorder.experiment = args.experiment
    sg24_ca.SG24Recorder.enabled = ("A0", "A1", "A2", "A3", "A4") if args.experiment == "a" else (method,)
    original_init = sg24_ca.SG24Recorder.__init__
    recorders = []
    def tracked_init(self, *a, **k):
        original_init(self, *a, **k)
        recorders.append(self)
    sg24_ca.SG24Recorder.__init__ = tracked_init
    live.CompactAttentionRecorder = sg24_ca.SG24Recorder
    _import_densecrf(args.crf_site)
    folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if args.dataset == "voc" else {r["image_id"]: 0 for r in all_rows}
    stem = f'class_{args.class_shard:02d}_{classname.replace(" ", "_")}'
    path = out / args.dataset / "shards" / (stem + ".npz")
    side = out / args.dataset / "shards" / (stem + ".diagnostics.npz")
    settings = {"experiment": args.experiment, "method": method, "spec": spec,
                "dataset": args.dataset, "model": args.model, "selected_sa_module": selected_sa,
                "ca_modules": list(capture_ca), "crf": list(OVAM_CRF_PARAMETERS)}
    writer = cv.ShardWriter(path, names, folds, settings, spec["thresholds"])
    sidecar = {}
    if side.exists():
        with np.load(side, allow_pickle=False) as data:
            for i, image_id in enumerate(data["image_ids"]):
                sidecar[str(image_id)] = (data["ca_diagnostics"][i], data["layer_weights"][i],
                                          data["content_mass"][i], data["layer_weights_b0"][i],
                                          data["content_detail"][i])
    if writer.completed - set(sidecar):
        raise ValueError("metric shard ahead of diagnostics; refusing inconsistent resume")
    manifest = transfer.EVAL / ("ovam/voc_sim" if args.dataset == "voc" else "ovam/coco_captions")
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        prompt = row["attribution_prompt"]
        ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
        class_ids = pipe.tokenizer(row["classname"], add_special_tokens=False)["input_ids"]
        from src.segmentation.attention_aggregation import find_token_subsequence
        target = find_token_subsequence(ids, class_ids)
        sg24_ca.SG24Recorder.content = sg24_ca.content_positions(pipe.tokenizer, prompt, tuple(target))
        sg24_ca.SG24Recorder.attribution_prompt = prompt
        recorders.clear()
        sample = live.generate_attention_sample(pipe, row, manifest, "float16", cross_map_kind="conditional")
        recorder = recorders[-1]
        try:
            seeds = recorder.seeds()
            keys = list(seeds)
            if args.experiment == "a":
                keys = [f"selected/{x}" for x in ("A0", "A1", "A2", "A3", "A4")]
            else:
                keys = [f"selected/{method}", f"expanded/{method}"]
            c = torch.stack([seeds[k] for k in keys])
            full = mid_p2(recorder.sa_qk, c.reshape(len(keys), -1))
            raw = torch.stack([x for pair in zip(c, full) for x in pair])
            truth = np.asarray(sample.target_mask, bool)
            image = np.asarray(sample.generated_image, np.uint8)
            counts, ap = cv.score_maps(raw.cpu(), truth, "minmax", spec["thresholds"])
            final_counts = np.stack([transfer.crf_counts(x, truth, image, args.crf_site)[0] for x in full])
            counts[1::2] = final_counts
            ap[1::2] = np.nan  # A hard-mask CRF output has no ranking score.
            weight = np.stack([w for step,label,w in recorder.weights if label == ("selected" if args.experiment == "a" else "expanded")])
            weight_b0 = np.stack([w for step,label,w in recorder.weights if label == "selected"])
            diag = diagnostics(masks_at(c, truth.shape), truth)
            fg64 = cv.target_at_size(truth).numpy()
            mass64 = recorder.content_mass_map.cpu().numpy()
            top64 = recorder.target_top_map.cpu().numpy()
            detail = np.asarray([mass64[fg64].mean() if fg64.any() else np.nan,
                                 mass64[~fg64].mean() if (~fg64).any() else np.nan,
                                 top64[fg64].mean() if fg64.any() else np.nan,
                                 top64[~fg64].mean() if (~fg64).any() else np.nan], np.float32)
            sidecar[row["image_id"]] = (diag, weight, np.asarray(recorder.content_mass, np.float32), weight_b0, detail)
            atomic_npz(side, image_ids=np.asarray(list(sidecar)),
                       ca_diagnostics=np.stack([x[0] for x in sidecar.values()]),
                       layer_weights=np.stack([x[1] for x in sidecar.values()]),
                       content_mass=np.stack([x[2] for x in sidecar.values()]),
                       layer_weights_b0=np.stack([x[3] for x in sidecar.values()]),
                       content_detail=np.stack([x[4] for x in sidecar.values()]))
            writer.add(row["image_id"], classname, counts, ap)
            print(f"[{args.experiment}:{args.dataset}:{classname}] {row['image_id']} {len(writer.completed)}/{len(rows)}", flush=True)
        finally:
            recorder.clear(); sample.clear(); recorders.clear()
            del recorder, sample
            gc.collect(); torch.cuda.empty_cache()
    if args.max_samples is None:
        if writer.completed != {r["image_id"] for r in all_rows} or set(sidecar) != writer.completed:
            raise ValueError("incomplete class shard")
        writer.finish()


if __name__ == "__main__":
    main()
