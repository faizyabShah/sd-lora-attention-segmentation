#!/usr/bin/env python3
"""Token track B1/B1b (phase 3 registry Section 2) on one VOC-Sim class shard.

OVAM's optimized class tokens replace the class word as the CA attribution;
everything else is the frozen pipeline. One live generation per image scores
every attribution and pipeline:

Attributions
  W       class word (frozen convention: conditional query, 77 padded positions, all pieces)
  TU      OVAM's released token (mean of 2-3 per-image tokens), OVAM's form:
          keys from [<SoT>, token], softmax over those two, unconditional query
  TC      the same token and keys with the conditional query
  T{k}U / T{k}C   the k-th per-image token of the class (1-shot), k = 1..3
                  (classes with two training images repeat token 1 in slot 3)

Pipelines (on each attribution's CA seed, with the frozen CA modules/calls/heads)
  seed    CA seed alone
  REF     one pass through SA module 15, mean_i(A_i c)
  P1      two passes, mean_i(A_i^2 c)
  R2      DenseCRF on the REF mask
  P1+R2   DenseCRF on the P1 mask
  (B1b tokens: REF and P1 only.)

Counts are stored for the 17 phase 3 thresholds; ``scripts/select_propagation_cv.py``
chooses, per pipeline, between W, TU and TC on the training folds.
"""

from __future__ import annotations

import argparse
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
from scripts.run_streaming_voc_ablations import read_rows  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.fusion_variants import minmax  # noqa: E402
from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402

WORK = 64
SELECTED_SA_MODULE = 15
THRESHOLDS = tuple(round(0.30 + 0.025 * i, 3) for i in range(17))
TRAIN_TOKENS = ROOT / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim_train/tokens"
TOKEN_SETS = ("TR", "T1", "T2", "T3")  # released mean, then per-image tokens
FULL = ("W", "TU", "TC")
ONE_SHOT = tuple(f"T{k}{q}" for q in "UC" for k in (1, 2, 3))
CONFIGURATIONS = (*(f"{a}/{p}" for a in FULL for p in ("seed", "REF", "P1", "R2", "P1+R2")),
                  *(f"{a}/{p}" for a in ONE_SHOT for p in ("REF", "P1")))
CRF_CELLS = tuple(f"{a}/{p}" for a in FULL for p in ("R2", "P1+R2"))

_TOKEN_HIDDEN: list[torch.Tensor] = []  # [1, 8, 768]: (<SoT>, token) for TR, T1, T2, T3


class TokenRecorder(sa_common.SAPackageRecorder):
    """Adds per-call CA maps for every token attribution (two query forms) beside the class word."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribution_hidden_states = _TOKEN_HIDDEN[0]
        self.token_calls: dict[str, dict[str, list[torch.Tensor]]] = {}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        super().record(module_name, kind, query, key, value, attention_mask, scale)
        if kind != "cross":
            return
        with torch.inference_mode():
            for j, name in enumerate(TOKEN_SETS):
                pair = attribution_key[0][:, 2 * j: 2 * j + 2].float()  # [heads, 2, d]
                for form, branch in (("U", 0), ("C", 1)):
                    probs = (torch.matmul(query[branch].float(), pair.transpose(-1, -2)) * scale).softmax(dim=-1)[..., 1]
                    self.token_calls.setdefault(f"{name}{form}", {}).setdefault(module_name, []).append(probs.half())


sa_common.live.CompactAttentionRecorder = TokenRecorder


def token_hidden(pipe, classname: str, released: dict, device: str) -> tuple[torch.Tensor, list[str]]:
    key = classname.replace(" ", "-")
    files = sorted(TRAIN_TOKENS.glob(f"sd-15_{key}_*.npy"))
    files = [f for f in files if not f.stem.endswith("_losses")]
    if not 2 <= len(files) <= 3:
        raise RuntimeError(f"{classname}: expected 2-3 per-image tokens, found {len(files)}")
    per_image = [torch.from_numpy(np.load(f, allow_pickle=False)[1].astype(np.float32)) for f in files]
    rel = torch.from_numpy(np.asarray(released[key], dtype=np.float32))
    if float((torch.stack(per_image).mean(0) - rel).abs().max()) > 1e-5:
        raise RuntimeError(f"{classname}: released token is not the mean of the per-image tokens")
    slots = per_image + [per_image[0]] * (3 - len(per_image))
    with torch.inference_mode():
        empty = pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
        sot = pipe.text_encoder(empty)[0][0, :1].float()  # CLIP is causal: <SoT> output is prompt-independent
    rows = []
    for t in [rel, *slots]:
        rows += [sot[0], t.to(device)]
    return torch.stack(rows)[None], [f.name for f in files]


def crf_counts(map64: torch.Tensor, target: np.ndarray, image: np.ndarray, crf_site) -> np.ndarray:
    """OVAM's DenseCRF on the mask at every threshold of the grid (as in the fusion package)."""
    full = upsampled(minmax(map64), tuple(np.asarray(target).shape))
    out = np.zeros((1, len(THRESHOLDS), 4), dtype=np.int64)
    for i, t in enumerate(THRESHOLDS):
        refined = apply_ovam_densecrf(image, (full >= t).cpu().numpy(), crf_site)
        out[0, i] = cv.score_masks(refined[None], target, thresholds=(t,))[0, 0]
    return out


@torch.inference_mode()
def propagate(recorder, module_name: str, seeds: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """One and two passes for several seeds at once: [S, 64, 64] each."""
    flat = seeds.reshape(seeds.shape[0], -1).T  # [P, S]
    one = torch.zeros_like(flat)
    two = torch.zeros_like(flat)
    calls = recorder.sa_qk[module_name]
    for query, key in calls:
        query, key = query.to(device).float(), key.to(device).float()
        affinity = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1]), dim=-1)
        first = torch.matmul(affinity, flat)  # [heads, P, S]
        one += first.mean(dim=0)
        two += torch.matmul(affinity, first).mean(dim=0)
        del query, key, affinity, first
    shape = (seeds.shape[0], WORK, WORK)
    return (one / len(calls)).T.reshape(shape), (two / len(calls)).T.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_frozen.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"))
    parser.add_argument("--folds", type=Path, default=Path("configs/voc_sim_folds.csv"))
    parser.add_argument("--tokens", type=Path, default=Path("configs/ovam/tokens_VOC.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation/controlled_attention_cv/tokens"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crf-site", type=Path)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    seed_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    classname, rows = read_rows(args.manifest, args.class_shard, args.max_samples)
    folds = cv.read_folds(args.folds)
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith("attn2")))
    sa_modules = tuple(sorted(n for n in processors if n.endswith("attn1")))
    selected = sa_modules[SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    released = np.load(args.tokens, allow_pickle=True).item()  # committed copy of OVAM's tokens_VOC.npy (MIT)
    hidden, token_files = token_hidden(pipe, classname, released, args.device)
    _TOKEN_HIDDEN[:] = [hidden]
    settings = {
        "package": "token_package_v1", "protocol": "phase 3 registry Section 2 (B1, B1b); protocol Section 49",
        "attributions": {"W": "class word, frozen convention", "TU": "released token, [<SoT>, token], unconditional query",
                         "TC": "released token, [<SoT>, token], conditional query",
                         "TkU/TkC": "k-th per-image token (1-shot), both query forms"},
        "per_class_token_files": "tokens_per_class.json",
        "ca_seed": seed_spec, "selected_sa_module": selected, "thresholds": list(THRESHOLDS),
        "configurations": list(CONFIGURATIONS), "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
    }
    sidecar = args.output_dir / "tokens_per_class.json"
    per_class = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    per_class[classname] = {"per_image_tokens": token_files, "token_slots_repeat_first": len(token_files) < 3}
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(per_class, indent=2, sort_keys=True) + "\n")
    writer = cv.ShardWriter(args.output_dir / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz",
                            CONFIGURATIONS, folds, settings, thresholds=THRESHOLDS)
    t45 = THRESHOLDS.index(0.45)
    seed_names = ("W", "TRU", "TRC", "T1U", "T2U", "T3U", "T1C", "T2C", "T3C")
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(pipe, row, args.manifest.parent, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {n: r for r, names in sample.modules_by_resolution.items() for n in names}
            word_calls = recorder.ca_calls
            seeds = []
            for name in seed_names:
                recorder.ca_calls = word_calls if name == "W" else recorder.token_calls[name]
                seeds.append(sa_common.ca_seed(recorder, ca_modules, ca_res, seed_spec, args.device))
            recorder.ca_calls = word_calls
            seeds = torch.stack(seeds)
            one, two = propagate(recorder, selected, seeds, args.device)
            recorder.sa_qk.clear()
            idx = {n: i for i, n in enumerate(seed_names)}
            alias = {"W": "W", "TU": "TRU", "TC": "TRC", **{a: a for a in ONE_SHOT}}
            maps, crf_maps = [], []
            for a in FULL:
                i = idx[alias[a]]
                maps += [seeds[i], one[i], two[i]]
                crf_maps += [one[i], two[i]]
            for a in ONE_SHOT:
                i = idx[alias[a]]
                maps += [one[i], two[i]]
            counts, ap = cv.score_maps(torch.stack(maps), sample.target_mask, "minmax", thresholds=THRESHOLDS)
            crf = np.concatenate([crf_counts(m, sample.target_mask, sample.generated_image, args.crf_site) for m in crf_maps])
            order, aps, c_i, k_i = [], [], 0, 0
            for name in CONFIGURATIONS:
                if name in CRF_CELLS:
                    order.append(crf[k_i]); aps.append(np.nan); k_i += 1
                else:
                    order.append(counts[c_i]); aps.append(ap[c_i]); c_i += 1
            counts_all = np.stack(order)
            writer.add(row["image_id"], classname, counts_all, np.asarray(aps))
            show = ("W/REF", "TU/REF", "TC/REF", "W/P1+R2", "TU/P1+R2", "T1U/REF")
            print(f"[{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) "
                  + " ".join(f"{n}={counts_all[CONFIGURATIONS.index(n), t45, 0] / max(counts_all[CONFIGURATIONS.index(n), t45, :3].sum(), 1):.3f}"
                             for n in show), flush=True)
        finally:
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            recorder.token_calls.clear()
            sample.clear()
            del sample, recorder
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
