#!/usr/bin/env python3
"""Regenerate the images of ``story_selection.json`` and save every coarse-to-fine stage (GPU).

Same runtime as SG-combined-v3 (A3 seed, SA 16/32/64, calls 10-20, eight heads). For each
image one ``.npz`` in ``--cache`` (outside the repository) holds:

  image, gt               generated RGB (uint8) and annotation
  seed_a3, seed_a0        A3-weighted and uniform selected-module CA seeds (64x64, min-max)
  ca_8, ca_16, ca_64      unweighted mean CA of the selected modules at each resolution
  a3_weights              call-mean A3 weight per selected module (module order of the runtime)
  r16, q32, z32           call-mean coarse map Abar16 c16, uncapped 32-px proposal, capped z32
  removed32               q32 - z32 (mass removed by the cap)
  coarse16_p2             COARSE16-style map: two 16-px passes, upsampled (illustration)
  R0, V1, V3, C4          FINE-P2, uncapped hierarchy, capped hierarchy, exact KL conservation
                          (sg_combined_v1 C4), each min-max 64x64
  mask_* / crf_*          thresholded maps and their DenseCRF masks at the full-VOC thresholds

The V3 and R0 raw image IoUs are compared with the committed SG-combined-v3 counts.

    python scripts/paper/figure_story_maps.py --cache /path/to/cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import run_coco_cap_transfer as transfer, sg27_runtime  # noqa: E402
from src.segmentation import controlled_cv as cv, sg_combined_v1, sg_combined_v3  # noqa: E402
from src.segmentation.ovam_full import _import_densecrf  # noqa: E402
from src.segmentation.sg25_sa import _apply, _attention, _resize  # noqa: E402
from src.segmentation.sg_combined_v2 import cap  # noqa: E402

THRESHOLDS = {"R0": .45, "R0+R2": .425, "V1": .525, "V3": .55, "V3+R2": .55, "C4": .475}
SELECTION = ROOT / "outputs/segmentation/paper_figures/story_selection.json"
V3 = ROOT / "outputs/segmentation/controlled_attention_cv/sg_combined_v3"


def record_resolutions(cls):
    """Accumulate the unweighted per-resolution CA maps before the recorder's own flush."""
    original = cls.flush_step

    def flush_step(self, step, timestep, latent):
        sums = self.__dict__.setdefault("fig_resolution", {})
        for maps in self.pending_ca.values():
            x = maps["A3"]
            up = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)[0, 0].float()
            side = int(x.shape[-1])
            total, count = sums.get(side, (0., 0))
            sums[side] = (total + up, count + 1)
        original(self, step, timestep, latent)

    cls.flush_step = flush_step


@torch.inference_mode()
def stages(sa, seed):
    c64 = seed.float().reshape(-1)
    c16 = _resize(c64, 64, 16).reshape(-1)
    out = {k: torch.zeros(n, device=seed.device) for k, n in
           (("r16", 256), ("q32", 1024), ("z32", 1024), ("coarse16_p2", 256))}
    for call in range(11):
        h16 = _attention(sa[16][call], 16)
        h32 = _attention(sa[32][call], 32)
        r16 = _apply(h16, c16).mean(0)
        q32 = _apply(h32, _resize(r16, 16, 32).reshape(-1)).mean(0)
        z32, _ = cap(r16, q32, 16)
        out["r16"] += r16 / 11; out["q32"] += q32 / 11; out["z32"] += z32 / 11
        out["coarse16_p2"] += _apply(h16, _apply(h16, c16)).mean(0) / 11
    side = {"r16": 16, "q32": 32, "z32": 32, "coarse16_p2": 16}
    out = {k: v.reshape(side[k], side[k]) for k, v in out.items()}
    out["removed32"] = out["q32"] - out["z32"]
    return out


def counted_iou(dataset, image_id, name, threshold):
    for path in sorted((V3 / dataset / "shards").glob("class_*.npz")):
        if path.name.count(".") != 1:
            continue
        with np.load(path) as d:
            ids = [str(x) for x in d["image_ids"]]
            if image_id in ids:
                cfg = [str(x) for x in d["configurations"]]
                c = d["counts"][cfg.index(name), ids.index(image_id), list(np.round(d["thresholds"], 3)).index(threshold)]
                return float(c[0] / max(c[:3].sum(), 1))
    raise KeyError(image_id)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--crf-site", type=Path, default=ROOT / "outputs/dependencies/simplecrf_py310")
    p.add_argument("--missing-only", action="store_true", help="skip images already in the cache")
    args = p.parse_args()
    selection = json.loads(SELECTION.read_text())
    wanted = {}
    for group in selection["groups"].values():
        for item in group:
            wanted[item["image_id"]] = item
    runtime = sg27_runtime.make_runtime(args.model)
    runtime["recorder_class"].enabled = ("A0", "A3")
    record_resolutions(runtime["recorder_class"])
    _import_densecrf(args.crf_site)
    args.cache.mkdir(parents=True, exist_ok=True)
    report = {}
    for image_id, item in wanted.items():
        if args.missing_only and (args.cache / f"{image_id}.npz").exists():
            continue
        ds = item["dataset"]
        target = item["class"]
        for s in range(20):
            name, rows = (transfer.voc_rows if ds == "voc" else transfer.coco_rows)(s)
            if name.replace(" ", "-") == target.replace(" ", "-"):
                break
        row = next(r for r in rows if r["image_id"] == image_id)
        with sg27_runtime.live_maps(runtime, row, ds) as (sample, sa, seed, base, previous):
            rec = runtime["recorders"][-1]
            truth = np.asarray(sample.target_mask, bool)
            image = np.asarray(sample.generated_image, np.uint8)
            seeds = rec.seeds()
            v3, _ = sg_combined_v3.aggregate(sa, seed, seed.new_zeros((16, 16)) + _resize(seed.reshape(-1), 64, 16).reshape(16, 16))
            v1_maps, _ = sg_combined_v1.aggregate(sa, seed)
            maps = {"R0": previous["S0"], "V3": v3["R1"], "V1": v1_maps["C1"], "C4": v1_maps["C4"]}
            maps = {k: cv.normalize_batch(v[None].float(), "minmax")[0] for k, v in maps.items()}
            weights = np.stack([w for _, label, w in rec.weights if label == "selected"]).mean(0)
            res = {f"ca_{s}": (t / n).cpu().numpy() for s, (t, n) in rec.__dict__["fig_resolution"].items()}
            st = {k: v.cpu().numpy() for k, v in stages(sa, seed).items()}
            out = {"image": image, "gt": truth, "seed_a3": seeds["selected/A3"].cpu().numpy(),
                   "seed_a0": seeds["selected/A0"].cpu().numpy(), "a3_weights": weights, **res, **st,
                   **{k: v.cpu().numpy() for k, v in maps.items()}}
            shape = truth.shape
            for key, t in THRESHOLDS.items():
                base_name = key.removesuffix("+R2")
                full = transfer.upsampled(transfer.minmax(maps[base_name]), shape)
                mask = (full >= t).cpu().numpy()
                if key.endswith("+R2"):
                    out["crf_" + base_name] = transfer.apply_ovam_densecrf(image, mask, args.crf_site).astype(bool)
                else:
                    out["mask_" + key] = mask
            check = {}
            for key, cell in (("V3", "R1"), ("R0", "R0")):
                m = out["mask_" + key]
                mine = float((m & truth).sum() / max((m | truth).sum(), 1))
                check[key] = (mine, counted_iou(ds, image_id, cell, THRESHOLDS[key]))
            np.savez_compressed(args.cache / f"{image_id}.npz", **out)
        report[image_id] = check
        print(image_id, {k: tuple(round(x, 4) for x in v) for k, v in check.items()}, flush=True)
    if args.missing_only and (args.cache / "check.json").exists():
        report = json.loads((args.cache / "check.json").read_text()) | report
    (args.cache / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    worst = max(abs(a - b) for v in report.values() for a, b in v.values())
    print(f"max |regenerated - counted| image IoU: {worst:.4f}")
    if worst > .02:
        raise SystemExit("regenerated maps do not match the committed counts")


if __name__ == "__main__":
    main()
