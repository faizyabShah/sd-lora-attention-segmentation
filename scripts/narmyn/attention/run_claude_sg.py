#!/usr/bin/env python3
"""Runner for Claude's standalone SG families SG2 and SG4 (protocol Section 81), one class shard.

The family module (``sg2_consensus_edges`` or ``sg4_region_labels``) defines its
cells and per-image maps; this runner provides the frozen inputs (as SG10:
class word ``W`` and OVAM's released token ``T`` on the frozen CA seed, SA module
15, all calls and heads), scores every cell at the 17 phase 3 thresholds, and
writes per-image diagnostics:

  change_vs_seed / change_vs_ref   new TP / new FP (and lost TP / removed FP against the
                                   cell's reference) at threshold 0.45, annotation resolution
  family diagnostics               SG2 edge reliability by region pair; SG4 oracle-region IoU

``--visual-dir`` renders the first image of the class for every cell with the
shared ``src/segmentation/sg_visual_audit.py`` layout (no shard is written).

    python scripts/narmyn/attention/run_claude_sg.py --family sg2 --dataset voc --class-shard 0
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.narmyn.attention.run_sg10_bottleneck as base  # noqa: E402  (installs the W + T recorder)
from scripts.run_fusion_calibration import upsampled  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.fusion_variants import minmax  # noqa: E402
from src.segmentation.sg_visual_audit import best_thresholds, released_image, render_variant_pages  # noqa: E402

FAMILIES = {"sg2": "scripts.narmyn.attention.sg2_consensus_edges", "sg4": "scripts.narmyn.attention.sg4_region_labels"}
SA = base.sa_common
transfer = base.transfer


def mask(m: torch.Tensor, shape) -> np.ndarray:
    return (upsampled(minmax(m.reshape(64, 64)), shape) >= 0.45).cpu().numpy()


def changes(maps: torch.Tensor, names, fam, truth: np.ndarray) -> dict:
    masks = {n: mask(maps[i], truth.shape) for i, n in enumerate(names)}
    out = {"change_vs_seed": {}, "change_vs_ref": {}}
    for n, m in masks.items():
        s = masks[f"{n.split('/')[0]}/seed"]
        out["change_vs_seed"][n] = [int((m & ~s & truth).sum()), int((m & ~s & ~truth).sum())]
        r = masks[fam.reference_for(n)]
        out["change_vs_ref"][n] = [int((m & ~r & truth).sum()), int((m & ~r & ~truth).sum()),
                                   int((r & ~m & truth).sum()), int((r & ~m & ~truth).sum())]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--visual-dir", type=Path, help="render the first image and every cell; no shard is written")
    args = parser.parse_args()
    fam = importlib.import_module(FAMILIES[args.family])
    out_dir = args.output_dir or fam.OUT
    spec = json.loads(fam.CONFIG.read_text(encoding="utf-8"))
    voc = args.dataset == "voc"
    classname, rows = (transfer.voc_rows if voc else transfer.coco_rows)(args.class_shard)
    rows = rows[:1] if args.visual_dir else (rows[: args.max_samples] if args.max_samples else rows)
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
    pipe, ca_modules, selected, seed_spec = base.setup(args)
    prompts = {r["attribution_prompt"] for r in rows}
    if len(prompts) != 1:
        raise RuntimeError(f"{classname}: expected one attribution prompt, found {sorted(prompts)}")
    base.class_keys(pipe, classname, prompts.pop(), voc, args.device)
    names = fam.CONFIGURATIONS
    settings = {"package": f"{fam.NAME}_v1", "protocol": "protocol Section 81", "dataset": args.dataset,
                "family_spec": spec, "ca_seed": seed_spec, "selected_sa_module": selected, "thresholds": list(base.THRESHOLDS),
                "configurations": list(names), "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16"}
    stem = f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}"
    writer = diag_path = None
    done = set()
    if not args.visual_dir:
        folds = cv.read_folds(ROOT / "configs/voc_sim_folds.csv") if voc else {r["image_id"]: 0 for r in rows}
        writer = cv.ShardWriter(out_dir / args.dataset / "shards" / f"{stem}.npz", names, folds, settings, thresholds=base.THRESHOLDS)
        diag_path = out_dir / args.dataset / "diagnostics" / f"{stem}.jsonl"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["image_id"] for l in diag_path.open()} if diag_path.exists() else set()
    thresholds = best_thresholds(out_dir) if args.visual_dir else {}
    for row in rows:
        if writer is not None and row["image_id"] in writer.completed:
            continue
        SA._RECORDERS.clear()
        sample = SA.live.generate_attention_sample(pipe, row, manifest_dir, "float16", cross_map_kind="conditional")
        recorder = SA._RECORDERS[-1]
        try:
            ca_res = {n: r for r, ns in sample.modules_by_resolution.items() for n in ns}
            word_calls, seeds = recorder.ca_calls, []
            for calls in (word_calls, recorder.token_calls):
                recorder.ca_calls = calls
                seeds.append(SA.ca_seed(recorder, ca_modules, ca_res, seed_spec, args.device).reshape(-1))
            recorder.ca_calls = word_calls
            seeds = torch.stack(seeds)
            truth = np.asarray(sample.target_mask, bool)
            truth64 = F.interpolate(torch.as_tensor(truth, dtype=torch.float32)[None, None], size=(64, 64), mode="nearest")[0, 0].to(args.device)
            maps, fdiag = fam.compute(recorder, selected, seeds, truth64, args.device)
            recorder.sa_qk.clear()
            if args.visual_dir:
                ca_maps = [seeds[0] if n.startswith("W/") else seeds[1] for n in names]
                render_variant_pages(args.visual_dir, fam.NAME, classname, row["image_id"], released_image(row, manifest_dir),
                                     truth, names, ca_maps, list(maps), thresholds)
                print(f"[visual:{classname}] {row['image_id']} cells={len(names)}", flush=True)
                continue
            counts, ap = cv.score_maps(maps.reshape(-1, 64, 64), sample.target_mask, "minmax", thresholds=base.THRESHOLDS)
            diag = {"image_id": row["image_id"], "class": classname, **changes(maps, names, fam, truth)}
            if "oracle" in fdiag:  # SG4: oracle-region IoU at annotation resolution
                diag["oracle_iou"] = {}
                for name, m in fdiag["oracle"].items():
                    om = F.interpolate(m.reshape(1, 1, 64, 64), size=truth.shape, mode="nearest")[0, 0].cpu().numpy() > 0.5
                    union = (om | truth).sum()
                    diag["oracle_iou"][name] = float((om & truth).sum() / union) if union else 1.0
                diag["regions"] = fdiag["regions"]
            if "edge_reliability" in fdiag:
                diag["edge_reliability"] = fdiag["edge_reliability"]
            if row["image_id"] not in done:
                with diag_path.open("a") as log:
                    log.write(json.dumps(diag) + "\n")
            writer.add(row["image_id"], classname, counts, ap)
            print(f"[{args.family}:{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)})", flush=True)
        finally:
            recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear()
            sample.clear(); del sample, recorder
            gc.collect(); torch.cuda.empty_cache()
    if writer is not None and args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
