#!/usr/bin/env python3
"""Resumable SG28 VOC/COCO class shard: independent PEARL-inspired components."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_coco_cap_transfer as transfer
from scripts import sg27_runtime
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics
from src.segmentation import controlled_cv as cv
from src.segmentation import sg25_sa, sg28_pearl
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS, _import_densecrf

BASE = ROOT / "outputs/segmentation/controlled_attention_cv/sg28"
SPEC_PATH = ROOT / "configs/sg28.json"
NAMES = tuple(name+suffix for name in sg28_pearl.NAMES for suffix in ("", "+R2"))
SIDE_FIELDS = ("change", "pa", "lap", "trust_bins")


def _read_previous(path: Path) -> dict[str, tuple[np.ndarray, ...]]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["image_ids"]]
        values = {key: data[key] for key in SIDE_FIELDS}
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate SG28 diagnostic image ID")
    return {image_id: tuple(values[key][i] for key in SIDE_FIELDS)
            for i, image_id in enumerate(ids)}


def _save_sidecar(path: Path, records: dict[str, tuple[np.ndarray, ...]]) -> None:
    atomic_npz(path, image_ids=np.asarray(list(records)),
               **{key:np.stack([row[index] for row in records.values()])
                  for index,key in enumerate(SIDE_FIELDS)})


def trust_bins(trust: dict[str,np.ndarray], truth: np.ndarray,
               boundaries: tuple[float,float]) -> np.ndarray:
    gt = cv.target_at_size(truth).cpu().numpy().reshape(-1)
    output = np.zeros((3, 3, 2), np.int64)
    for source, name in enumerate(("C16", "C32", "C3")):
        value = trust[name]
        masks = (value < boundaries[0],
                 (value >= boundaries[0]) & (value < boundaries[1]),
                 value >= boundaries[1])
        for bin_id, mask in enumerate(masks):
            output[source, bin_id] = (int(mask.sum()), int((mask & gt).sum()))
    return output


def _verify_historical(dataset: str, shard: int, image_id: str,
                       counts: np.ndarray) -> None:
    path = next((ROOT / "outputs/segmentation/controlled_attention_cv/sg27" /
                 dataset / "shards").glob(f"class_{shard:02d}_*.npz"))
    with np.load(path, allow_pickle=False) as data:
        hits = np.flatnonzero(data["image_ids"] == image_id)
        if len(hits) != 1:
            raise ValueError("SG28 smoke image missing from SG27 historical shard")
        old = data["counts"][[0,1], hits[0]]
        old_names = tuple(data["configurations"][[0,1]])
    if old_names != ("S0", "S0+R2") or not np.array_equal(counts[:2], old):
        delta = int(np.abs(counts[:2].astype(np.int64)-old.astype(np.int64)).max())
        raise ValueError(f"SG28 frozen P0 counts differ from SG27 S0: max count drift {delta}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("voc", "coco"), required=True)
    p.add_argument("--class-shard", type=int, choices=range(20), required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--verify-controls", action="store_true")
    p.add_argument("--output-dir", type=Path, default=BASE)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site", type=Path, default=ROOT/"outputs/dependencies/simplecrf_py310")
    args = p.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    if tuple(spec["cells"]) != sg28_pearl.NAMES or spec["thresholds"] != list(transfer.THRESHOLDS):
        raise ValueError("SG28 locked cells or threshold grid changed")
    runtime = sg27_runtime.make_runtime(args.model)
    if tuple(runtime["spec"]["thresholds"]) != tuple(spec["thresholds"]):
        raise ValueError("SG27 capture controls differ from SG28")
    _import_densecrf(args.crf_site)
    classname, all_rows = (transfer.voc_rows if args.dataset == "voc" else transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    folds = (cv.read_folds(ROOT/"configs/voc_sim_folds.csv") if args.dataset == "voc"
             else {row["image_id"]:0 for row in all_rows})
    stem = f'class_{args.class_shard:02d}_{classname.replace(" ", "_")}'
    path = args.output_dir/args.dataset/"shards"/(stem+".npz")
    side_path = path.with_suffix(".diagnostics.npz")
    settings = {"spec":spec,"dataset":args.dataset,"model":args.model,
                "ca_modules":list(runtime["ca_modules"]),"sa_modules":runtime["modules"],
                "crf":list(OVAM_CRF_PARAMETERS)}
    writer = cv.ShardWriter(path, NAMES, folds, settings, spec["thresholds"])
    sidecars = _read_previous(side_path)
    if writer.completed-set(sidecars):
        raise ValueError("SG28 metric shard is ahead of diagnostics")
    for row in rows:
        image_id = row["image_id"]
        if image_id in writer.completed:
            continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample, sa, seed, base, sg27_maps):
            if args.verify_controls:
                previous,_ = sg25_sa.aggregate(sa,seed)
                for current,historical in (("FINE","FINE-P2"),("C16","16-to-64"),("C32","32-to-64")):
                    torch.testing.assert_close(base[current],previous[historical],rtol=1e-5,atol=2e-6)
            truth = np.asarray(sample.target_mask, bool)
            image = np.asarray(sample.generated_image, np.uint8)
            pa, pa_diagnostics = sg28_pearl.pa_maps(sa[64],seed,image_id,
                                       spec["pa"]["diagnostic_rows_per_image"])
            lap, lap_diagnostics, trust = sg28_pearl.laplacian_maps(
                sg27_maps["S0"],sg27_maps["S1"],sg27_maps["S2"],image,
                kappa=spec["laplacian"]["kappa"],tau=spec["laplacian"]["tau"],
                epsilon=spec["laplacian"]["epsilon"],rtol=spec["laplacian"]["cg_rtol"],
                maxiter=spec["laplacian"]["cg_maxiter_accuracy_cap"])
            raw = {**pa,**lap}
            new_names = sg28_pearl.NAMES[1:]
            normalized = cv.normalize_batch(torch.stack([raw[name] for name in new_names]),"minmax")
            maps = {"P0":sg27_maps["S0"],
                    **{name:normalized[i] for i,name in enumerate(new_names)}}
            scored = torch.stack([maps[name] for name in sg28_pearl.NAMES])
            raw_counts, raw_ap = cv.score_maps(scored.cpu(),truth,"minmax",spec["thresholds"])
            counts = np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap = np.full(len(NAMES),np.nan)
            counts[0::2],ap[0::2] = raw_counts,raw_ap
            for index,name in enumerate(sg28_pearl.NAMES):
                counts[2*index+1] = transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            if args.verify_controls:
                _verify_historical(args.dataset,args.class_shard,image_id,counts)
            changes = change_diagnostics(maps,list(sg28_pearl.NAMES),truth,spec["diagnostic_threshold"])
            bins = trust_bins(trust,truth,tuple(spec["laplacian"]["rho_bins"]))
            sidecars[image_id] = (changes,pa_diagnostics,lap_diagnostics,bins)
            _save_sidecar(side_path,sidecars)
            writer.add(image_id,classname,counts,ap)
            print(f"[sg28:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed!={row["image_id"] for row in all_rows} or set(sidecars)!=writer.completed:
            raise ValueError("incomplete SG28 class shard")
        writer.finish()


if __name__ == "__main__":
    main()
