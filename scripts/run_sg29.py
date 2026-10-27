#!/usr/bin/env python3
"""Resumable SG29 live VOC/COCO class shard with compact diagnostics."""
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

from scripts import run_coco_cap_transfer as transfer, sg27_runtime
from scripts.run_sg24 import atomic_npz
from scripts.run_sg27 import change_diagnostics
from src.segmentation import controlled_cv as cv, sg29_nerve
from src.segmentation.ovam_full import OVAM_CRF_PARAMETERS, _import_densecrf

BASE = ROOT / "outputs/segmentation/controlled_attention_cv/sg29"
SPEC_PATH = ROOT / "configs/sg29.json"
RAW = sg29_nerve.ALL
NAMES = tuple(n+s for n in RAW for s in ("", "+R2"))
SIDE_FIELDS = ("change", "entropy", "fused_entropy", "weights", "fused_weights",
               "head_ap", "transitions")


def read_side(path: Path) -> dict[str, tuple[np.ndarray, ...]]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        ids = [str(x) for x in data["image_ids"]]
        values = {key: data[key] for key in SIDE_FIELDS}
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate SG29 diagnostic image")
    return {image_id: tuple(values[key][i] for key in SIDE_FIELDS)
            for i, image_id in enumerate(ids)}


def save_side(path: Path, rows: dict[str, tuple[np.ndarray, ...]]) -> None:
    atomic_npz(path, image_ids=np.asarray(list(rows)),
               **{key: np.stack([row[i] for row in rows.values()])
                  for i, key in enumerate(SIDE_FIELDS)})


def historical_replay_drift(dataset: str, shard: int, image_id: str, counts: np.ndarray) -> int:
    """Compare regenerations for audit; cross-run pixel counts need not match."""
    prior = ROOT / "outputs/segmentation/controlled_attention_cv/sg28" / dataset / "shards"
    files = list(prior.glob(f"class_{shard:02d}_*.npz"))
    files = [p for p in files if not p.name.endswith(".diagnostics.npz")]
    if len(files) != 1:
        raise ValueError("SG29 baseline replay requires exactly one SG28 class shard")
    with np.load(files[0], allow_pickle=False) as data:
        found = np.flatnonzero(data["image_ids"] == image_id)
        if len(found) != 1 or tuple(data["configurations"][:2]) != ("P0", "P0+R2"):
            raise ValueError("SG29 baseline image missing from SG28")
        old = data["counts"][:2, found[0]]
    return int(np.abs(counts[:2].astype(np.int64)-old.astype(np.int64)).max())


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
    if tuple(spec["cells"]) != sg29_nerve.MAIN or tuple(spec["diagnostic_cells"]) != sg29_nerve.DIAGNOSTIC:
        raise ValueError("SG29 locked cell list changed")
    if spec["thresholds"] != list(transfer.THRESHOLDS):
        raise ValueError("SG29 threshold grid differs from historical pipeline")
    runtime = sg27_runtime.make_runtime(args.model)
    _import_densecrf(args.crf_site)
    classname, all_rows = (transfer.voc_rows if args.dataset == "voc" else transfer.coco_rows)(args.class_shard)
    rows = all_rows[:args.max_samples] if args.max_samples else all_rows
    folds = cv.read_folds(ROOT/"configs/voc_sim_folds.csv") if args.dataset == "voc" else {r["image_id"]:0 for r in all_rows}
    stem = f'class_{args.class_shard:02d}_{classname.replace(" ", "_")}'
    path = args.output_dir/args.dataset/"shards"/(stem+".npz")
    side = path.with_suffix(".diagnostics.npz")
    settings = {"spec":spec,"dataset":args.dataset,"model":args.model,
                "ca_modules":list(runtime["ca_modules"]),"sa_modules":runtime["modules"],
                "crf":list(OVAM_CRF_PARAMETERS)}
    writer = cv.ShardWriter(path,NAMES,folds,settings,spec["thresholds"])
    sidecars = read_side(side)
    if writer.completed-set(sidecars):
        raise ValueError("metric shard is ahead of SG29 diagnostics")
    for row in rows:
        image_id = row["image_id"]
        if image_id in writer.completed:
            continue
        with sg27_runtime.live_maps(runtime,row,args.dataset) as (sample,sa,seed,base,previous):
            truth = np.asarray(sample.target_mask,bool)
            image = np.asarray(sample.generated_image,np.uint8)
            raw, diag = sg29_nerve.aggregate(sa[64],seed,alpha=spec["alpha"],
                  beta=spec["beta"],length=spec["default_walk_length"],
                  concentration=spec["entropy_concentration"],
                  self_weight=spec["local_self_weight"],epsilon=spec["epsilon"],
                  truth64=cv.target_at_size(truth).to(seed.device))
            if args.verify_controls:
                torch.testing.assert_close(raw["N0"],base["FINE"],rtol=1e-5,atol=2e-6)
            norm = cv.normalize_batch(torch.stack([raw[name] for name in RAW]),"minmax")
            maps = {name:norm[i] for i,name in enumerate(RAW)}
            # Historical N0 is used exactly, after the replay assertion in smoke.
            maps["N0"] = previous["S0"]
            raw_counts,raw_ap = cv.score_maps(torch.stack([maps[n] for n in RAW]).cpu(),truth,
                                               "minmax",spec["thresholds"])
            counts = np.zeros((len(NAMES),len(spec["thresholds"]),4),np.int64)
            ap = np.full(len(NAMES),np.nan)
            counts[0::2],ap[0::2] = raw_counts,raw_ap
            for i,name in enumerate(RAW):
                counts[2*i+1] = transfer.crf_counts(maps[name],truth,image,args.crf_site)[0]
            if args.verify_controls:
                drift=historical_replay_drift(args.dataset,args.class_shard,image_id,counts)
                if drift:
                    print(f"[sg29:replay-drift] {args.dataset} {image_id} max count difference {drift}; same-generation N0 map check passed",flush=True)
            changes = change_diagnostics(maps,list(RAW),truth,spec["diagnostic_threshold"])
            sidecars[image_id] = (changes,*(diag[field].numpy() for field in SIDE_FIELDS[1:]))
            save_side(side,sidecars)
            writer.add(image_id,classname,counts,ap)
            print(f"[sg29:{args.dataset}:{classname}] {image_id} {len(writer.completed)}/{len(rows)}",flush=True)
    if args.max_samples is None:
        if writer.completed != {r["image_id"] for r in all_rows} or set(sidecars) != writer.completed:
            raise ValueError("incomplete SG29 class shard")
        writer.finish()


if __name__ == "__main__":
    main()
