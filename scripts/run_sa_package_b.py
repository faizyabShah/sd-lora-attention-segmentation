#!/usr/bin/env python3
"""SA package B of the controlled protocol: SA-L (modules), SA-H (heads), operator check.

Protocol v1.2, Sections 6-7, owner Claude. One VOC-Sim class shard per call;
every image is regenerated live and released after scoring.

Seed (Section 6.1): the frozen raw CA map, min-max normalized. The CA choice is
read from ``--ca-seed`` (JSON: calls, modules, heads, attribution). Until CA is
frozen in Section 11, ``configs/controlled_sa_seed_provisional.json`` (the
inclusive CA reference) is used for development runs only.

Operator (Section 6): conditional-branch SA probabilities ``A = softmax(QK^T/sqrt(d))``;
one non-residual pass ``y = A x``; for every module/call/head observation the seed
is area-downsampled to the module's resolution, propagated, and the result is
bilinearly upsampled to 64x64; observations are averaged with equal weight;
min-max once; bilinear to the annotation; >= threshold.

Families (configuration names):

  SEED/ca_seed       the CA seed alone (no SA)
  SL/...             SA-L: the CA-L families over the 16 SA (attn1) modules
                     (SL/all_16 is the SA reference: all modules, heads, calls)
  SH/...             SA-H: all 255 head subsets, same subset in every module
  OP/AT_x            operator check: y = A^T x for the SA reference

The exploratory exhaustive search over 65,535 SA module subsets (threshold 0.45)
is written per image to ``exhaustive_raw/`` and not committed.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import gc
import itertools
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.attention_capture.live as live  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402
from src.attention_capture.memory import CompactAttentionRecorder  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

HEADS = 8
WORK = 64
SA_REFERENCE = "SL/all_16"


class SAPackageRecorder(CompactAttentionRecorder):
    """Keeps per-call CA head maps (reference attribution) and conditional SA Q/K on the device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ca_calls: dict[str, list[torch.Tensor]] = {}  # module -> per call [8, HW] (float16)
        self.sa_qk: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}  # module -> per call (q, k) [8, HW, d]

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        del value, attribution_key
        step = self.call_counts[module_name]
        self.call_counts[module_name] += 1
        self.step_tensors.setdefault(step, {})
        if query.shape[0] != 2:
            raise RuntimeError("Expected a [unconditional, conditional] CFG batch")
        with torch.inference_mode():
            if kind == "cross":
                scores = torch.matmul(query[1].float(), key[1].float().transpose(-1, -2)) * scale
                if attention_mask is not None:
                    scores = scores + attention_mask[1].float()
                probs = scores.softmax(dim=-1)
                self.ca_calls.setdefault(module_name, []).append(probs[..., list(self.target_positions)].mean(dim=-1).half())
            else:
                self.sa_qk.setdefault(module_name, []).append((query[1].detach().half().clone(), key[1].detach().half().clone()))
        spatial = math.isqrt(int(query.shape[-2]))
        self.module_metadata[module_name].update(
            {"spatial_height": spatial, "spatial_width": spatial,
             "spatial_tokens": int(query.shape[-2]), "head_dim": int(query.shape[-1])}
        )


live.CompactAttentionRecorder = SAPackageRecorder
_RECORDERS: list[SAPackageRecorder] = []
_original_init = SAPackageRecorder.__init__


def _tracking_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    _RECORDERS.append(self)


SAPackageRecorder.__init__ = _tracking_init


def layer_configurations(modules: tuple[str, ...], prefix: str) -> dict[str, tuple[int, ...]]:
    index = {name: i for i, name in enumerate(modules)}
    down = tuple(index[m] for m in modules if m.startswith("down_blocks."))
    middle = tuple(index[m] for m in modules if m.startswith("mid_block."))
    up = tuple(index[m] for m in modules if m.startswith("up_blocks."))
    every = tuple(range(len(modules)))
    configs = {
        f"{prefix}/all_16": every, f"{prefix}/non_middle_15": down + up, f"{prefix}/down_only": down,
        f"{prefix}/middle_only": middle, f"{prefix}/up_only": up, f"{prefix}/down_middle": down + middle,
        f"{prefix}/middle_up": middle + up,
    }
    configs.update({f"{prefix}/single_{i:02d}": (i,) for i in every})
    configs.update({f"{prefix}/leave1_{i:02d}": tuple(j for j in every if j != i) for i in every})
    configs.update({f"{prefix}/leave2_{a:02d}_{b:02d}": tuple(j for j in every if j not in (a, b))
                    for a, b in itertools.combinations(every, 2)})
    return configs


def head_configurations(prefix: str) -> dict[str, tuple[int, ...]]:
    return {f"{prefix}/" + "_".join(map(str, s)): s for w in range(1, HEADS + 1) for s in itertools.combinations(range(HEADS), w)}


def configuration_names(sa_modules):
    layers = layer_configurations(sa_modules, "SL")
    heads = head_configurations("SH")
    names = ["SEED/ca_seed"] + list(layers) + list(heads) + ["OP/AT_x"]
    inclusiveness = {**{k: len(v) for k, v in layers.items()}, **{k: len(v) for k, v in heads.items()}}
    return names, {"layers": layers, "heads": heads, "inclusiveness": inclusiveness}


def resolve_seed_spec(spec: dict, ca_modules: tuple[str, ...], calls: int) -> dict:
    def pick(value, every):
        return list(every) if value in (None, "all") else [int(v) for v in value]
    out = {"name": spec.get("name", "unnamed"), "attribution": spec.get("attribution", "cond_padded_all"),
           "calls": pick(spec.get("calls"), range(calls)), "modules": pick(spec.get("modules"), range(len(ca_modules))),
           "heads": pick(spec.get("heads"), range(HEADS)), "frozen": bool(spec.get("frozen", False))}
    if out["attribution"] != "cond_padded_all":
        raise NotImplementedError("Only the reference CA attribution is implemented (CA-A kept it)")
    return out


@torch.inference_mode()
def ca_seed(recorder: SAPackageRecorder, ca_modules, resolution, spec, device) -> torch.Tensor:
    """Section 3 aggregation of the frozen CA choice, min-max normalized (Section 6.1)."""
    total = torch.zeros((WORK, WORK), device=device)
    count = 0
    for index in spec["modules"]:
        name = ca_modules[index]
        r = resolution[name]
        for call in spec["calls"]:
            heads = recorder.ca_calls[name][call].float()[spec["heads"]].mean(dim=0).reshape(1, 1, r, r)
            if r != WORK:
                heads = F.interpolate(heads, size=(WORK, WORK), mode="bilinear", align_corners=False)
            total += heads[0, 0]
            count += 1
    return cv.normalize_batch((total / count)[None], "minmax")[0]


@torch.inference_mode()
def propagate(recorder: SAPackageRecorder, sa_modules, resolution, seed64, device):
    """Return Y and Y^T maps [16, 8, 64, 64]: call-averaged A x and A^T x per module and head."""
    y_out, yt_out = [], []
    for name in sa_modules:
        r = resolution[name]
        seed = F.adaptive_avg_pool2d(seed64[None, None], (r, r))[0, 0].reshape(-1) if r != WORK else seed64.reshape(-1)
        y = torch.zeros((HEADS, r * r), device=device)
        yt = torch.zeros((HEADS, r * r), device=device)
        calls = recorder.sa_qk[name]
        for q, k in calls:
            q, k = q.to(device).float(), k.to(device).float()
            affinity = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
            y += torch.matmul(affinity, seed)
            yt += torch.matmul(seed, affinity)  # (A^T x) per head
            del q, k, affinity
        y = (y / len(calls)).reshape(HEADS, 1, r, r)
        yt = (yt / len(calls)).reshape(HEADS, 1, r, r)
        if r != WORK:
            y = F.interpolate(y, size=(WORK, WORK), mode="bilinear", align_corners=False)
            yt = F.interpolate(yt, size=(WORK, WORK), mode="bilinear", align_corners=False)
        y_out.append(y[:, 0])
        yt_out.append(yt[:, 0])
    return torch.stack(y_out), torch.stack(yt_out)


@torch.inference_mode()
def exhaustive_counts(module_maps: torch.Tensor, target: np.ndarray, chunk: int = 512) -> np.ndarray:
    device = module_maps.device
    count = module_maps.shape[0]
    masks = torch.arange(1, 1 << count, device=device)
    bits = ((masks[:, None] >> torch.arange(count, device=device)[None]) & 1).float()
    layers = module_maps.reshape(count, -1)
    truth = torch.as_tensor(target, device=device).bool()
    truth_pixels, total = int(truth.sum()), truth.numel()
    out = np.zeros((bits.shape[0], 4), dtype=np.int64)
    for start in range(0, bits.shape[0], chunk):
        block = cv.normalize_batch(bits[start : start + chunk] @ layers, "minmax").reshape(-1, 1, WORK, WORK)
        pred = F.interpolate(block, size=truth.shape, mode="bilinear", align_corners=False)[:, 0] >= cv.SELECTION_THRESHOLD
        tp = (pred & truth).sum(dim=(1, 2))
        fp = pred.sum(dim=(1, 2)) - tp
        fn = truth_pixels - tp
        out[start : start + chunk] = torch.stack((tp, fp, fn, total - tp - fp - fn), 1).cpu().numpy()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_provisional.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"))
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation/controlled_attention_cv/sa_layer_head"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--skip-exhaustive", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true",
                        help="permit a seed spec with frozen=false (development only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_spec = json.loads(args.ca_seed.read_text())
    if not seed_spec.get("frozen", False) and not args.allow_provisional:
        raise SystemExit(f"{args.ca_seed} is not a frozen CA choice; pass --allow-provisional for development runs")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    classname = sorted({r["classname"] for r in rows})[args.class_shard]
    rows = [r for r in rows if r["classname"] == classname and r["annotated"].lower() == "true"][: args.max_images]
    folds = cv.read_folds(args.folds)
    stem = f"class_{args.class_shard:02d}_{classname}"
    for sub in ("shards", "exhaustive_raw", "checks"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith("attn2")))
    sa_modules = tuple(sorted(n for n in processors if n.endswith("attn1")))
    if len(ca_modules) != 16 or len(sa_modules) != 16:
        raise RuntimeError("Expected 16 CA and 16 SA modules")
    names, families = configuration_names(sa_modules)
    seed = resolve_seed_spec(seed_spec, ca_modules, 31)
    settings = {
        "package": "SA package B (SA-L, SA-H, operator check)",
        "protocol": "v1.2 approved",
        "condition": "Section 6: frozen CA seed min-max; conditional SA; one pass y = A x; seed area-downsampled to the "
                     "module resolution; propagated maps bilinear to 64x64; equal weights; min-max once; >= threshold",
        "ca_seed": seed, "ca_modules": list(ca_modules), "sa_modules": list(sa_modules),
        "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
        "manifest": str(args.manifest), "folds": str(args.folds), "reference": SA_REFERENCE,
    }
    writer = cv.ShardWriter(args.output_dir / "shards" / f"{stem}.npz", names, folds, settings)
    (args.output_dir / "configurations.json").write_text(json.dumps(
        {"names": names, "inclusiveness": families["inclusiveness"], "sa_modules": list(sa_modules),
         "layers": {k: list(v) for k, v in families["layers"].items()}}, indent=1))
    print(f"{classname}: {len(rows)} annotated, {len(writer.completed)} done, {len(names)} configurations, "
          f"seed={seed['name']} (frozen={seed['frozen']}), node={socket.gethostname()}, torch={torch.__version__}", flush=True)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        start = time.time()
        _RECORDERS.clear()
        sample = live.generate_attention_sample(pipe, row, args.manifest.parent, "float16", cross_map_kind="conditional")
        recorder = _RECORDERS[-1]
        try:
            calls = len(sample.timesteps)
            if any(len(recorder.ca_calls[m]) != calls for m in ca_modules) or any(len(recorder.sa_qk[m]) != calls for m in sa_modules):
                raise RuntimeError(f"{row['image_id']}: expected {calls} calls for every module")
            res_ca = {n: r for r, ns in sample.modules_by_resolution.items() for n in ns}
            res_sa = {n: r for r, ns in sample.self_modules_by_resolution.items() for n in ns}
            seed64 = ca_seed(recorder, ca_modules, res_ca, seed, args.device)
            recorder.ca_calls.clear()
            y, yt = propagate(recorder, sa_modules, res_sa, seed64, args.device)
            recorder.sa_qk.clear()
            module_maps = y.mean(dim=1)
            head_maps = y.mean(dim=0)
            maps = [seed64]
            maps += [module_maps[list(idx)].mean(dim=0) for idx in families["layers"].values()]
            maps += [head_maps[list(idx)].mean(dim=0) for idx in families["heads"].values()]
            maps += [yt.mean(dim=(0, 1))]
            counts, ap = cv.score_maps(torch.stack(maps), sample.target_mask, "minmax")
            if not args.skip_exhaustive:
                np.save(args.output_dir / "exhaustive_raw" / f"{row['image_id']}.npy",
                        exhaustive_counts(module_maps, sample.target_mask).astype(np.int32))
            writer.add(row["image_id"], classname, counts, ap)
            check = {"image_id": row["image_id"], "calls": calls,
                     "max_abs_diff_L_vs_H_reference": float((module_maps.mean(0) - head_maps.mean(0)).abs().max())}
            with (args.output_dir / "checks" / f"{stem}.jsonl").open("a") as handle:
                handle.write(json.dumps(check) + "\n")
            iou = lambda name: (lambda c: c[0] / max(c[:3].sum(), 1))(counts[names.index(name), cv.SELECTION_INDEX])  # noqa: E731
            print(f"{row['image_id']} seed IoU={iou('SEED/ca_seed'):.3f} SA ref={iou(SA_REFERENCE):.3f} "
                  f"A^T x={iou('OP/AT_x'):.3f} ({time.time() - start:.0f}s)", flush=True)
        finally:
            sample.clear()
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            del sample
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_images is None:
        writer.finish()


if __name__ == "__main__":
    main()
