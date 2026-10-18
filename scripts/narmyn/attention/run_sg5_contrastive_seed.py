#!/usr/bin/env python3
"""SG5 contrastive CA seed (protocol Section 66; ``configs/sg5_contrastive_seed.json``), one class shard.

Every attribution uses the frozen CA modules, calls and heads and the
conditional query; only the per-head map ``m`` that enters the CA average
changes. With ``l = log softmax`` over the 77 padded class-prompt keys, ``T``
the class pieces and ``log p_T = log mean_{t in T} p_t``:

  W    p_T                                   (frozen class word, reference)
  WS   log p_T - log p_SoT
  WK   p_T / (p_T + p_SoT)                   (class word in OVAM's two-key form)
  WN   log p_T - log mean_{j not in T} p_j
  WE   log p_T - log mean_{j in EoT, padding} p_j
  WY   mean of p_T over the class word and its fixed synonyms (same template)
  TC   OVAM's released token, keys [<SoT>, token]: p_tok   (few-shot reference)
  TS   log p_tok - log p_SoT

Pipelines on each seed: seed, REF (one pass, SA module 15), P1 (two passes),
R2 / P1+R2 (OVAM DenseCRF on the REF / P1 mask). Counts at the 17 phase 3
thresholds. On VOC-Sim the class keys are the generation prompt's
conditional keys (frozen convention); on COCO-Cap they come from the class
name encoded separately, padded to 77 (the transfers, Sections 48, 55).

    python scripts/narmyn/attention/run_sg5_contrastive_seed.py --dataset voc --class-shard 0 --crf-site ...
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_coco_cap_transfer as transfer  # noqa: E402  (rows; its recorder is replaced below)
import scripts.run_sa_package_b as sa_common  # noqa: E402
from scripts.run_token_package import crf_counts, propagate  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402
from src.segmentation.attention_aggregation import find_token_subsequence  # noqa: E402

THRESHOLDS = transfer.THRESHOLDS
WORD = ("W", "WS", "WK", "WN", "WE", "WY")
TOKEN = ("TC", "TS")
ATTRIBUTIONS = WORD + TOKEN
PIPELINES = ("seed", "REF", "P1", "R2", "P1+R2")
CONFIGURATIONS = tuple(f"{a}/{p}" for a in ATTRIBUTIONS for p in PIPELINES)
CONFIG = ROOT / "configs" / "sg5_contrastive_seed.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "sg5_contrastive_seed"

# Set per shard: extra keys appended after the class keys, and the synonym target positions.
_EXTRA: dict = {"hidden": None, "separate_class_prompt": False, "synonym_positions": []}


class SG5Recorder(sa_common.SAPackageRecorder):
    """Frozen class-word CA (``ca_calls``) plus per-call maps for every SG5 attribution (``sg5_calls``)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parts = [_EXTRA["hidden"]]
        if _EXTRA["separate_class_prompt"]:
            parts.insert(0, _EXTRA["class_hidden"])
        self.attribution_hidden_states = torch.cat(parts, dim=1)
        self.sg5_calls: dict[str, dict[str, list[torch.Tensor]]] = {a: {} for a in ATTRIBUTIONS if a != "W"}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind != "cross":
            sa_common.SAPackageRecorder.record(self, module_name, kind, query, key, value, attention_mask, scale)
            return
        offset = 77 if _EXTRA["separate_class_prompt"] else 0
        if offset:
            key, attention_mask = torch.stack((key[0], attribution_key[0][:, :77])), None
        sa_common.SAPackageRecorder.record(self, module_name, kind, query, key, value, attention_mask, scale)
        with torch.inference_mode():
            q = query[1].float()
            logp = (torch.matmul(q, key[1].float().transpose(-1, -2)) * scale).log_softmax(dim=-1)  # [h, HW, 77]
            targets = list(self.target_positions)
            others = [j for j in range(logp.shape[-1]) if j not in targets]
            padding = list(range(self.prompt_token_count - 1, logp.shape[-1]))  # <EoT> and the padding after it
            log_t = logp[..., targets].logsumexp(dim=-1) - math.log(len(targets))
            log_sot = logp[..., 0]
            log_others = logp[..., others].logsumexp(dim=-1) - math.log(len(others))
            log_pad = logp[..., padding].logsumexp(dim=-1) - math.log(len(padding))
            pair = attribution_key[0][:, offset:offset + 2].float()
            log_tok = (torch.matmul(q, pair.transpose(-1, -2)) * scale).log_softmax(dim=-1)
            ensemble = [log_t.exp()]
            for b, positions in enumerate(_EXTRA["synonym_positions"]):
                start = offset + 2 + 77 * b
                block = attribution_key[0][:, start:start + 77].float()
                p = (torch.matmul(q, block.transpose(-1, -2)) * scale).softmax(dim=-1)
                ensemble.append(p[..., list(positions)].mean(dim=-1))
            maps = {"WS": log_t - log_sot, "WK": torch.sigmoid(log_t - log_sot), "WN": log_t - log_others,
                    "WE": log_t - log_pad, "WY": torch.stack(ensemble).mean(dim=0),
                    "TC": log_tok[..., 1].exp(), "TS": log_tok[..., 1] - log_tok[..., 0]}
            for name, m in maps.items():
                self.sg5_calls[name].setdefault(module_name, []).append(m)


sa_common.live.CompactAttentionRecorder = SG5Recorder


def shard_keys(pipe, classname: str, attribution_prompt: str, released: dict, synonyms: dict, device: str):
    """[<SoT>, token] followed by one padded synonym prompt per synonym; synonym class-piece positions."""
    key = classname.replace(" ", "-")
    tok = pipe.tokenizer
    with torch.inference_mode():
        empty = tok("", return_tensors="pt").input_ids.to(device)
        sot = pipe.text_encoder(empty)[0][:, :1].float()  # CLIP is causal: <SoT> output is prompt-independent
        token = torch.from_numpy(np.asarray(released[key], dtype=np.float32)).to(device)[None, None]
        parts, positions = [sot, token], []
        word = classname.replace("-", " ") if classname not in attribution_prompt else classname
        if word not in attribution_prompt:
            raise RuntimeError(f"{classname!r} not found in attribution prompt {attribution_prompt!r}")
        for synonym in synonyms[key]:
            prompt = attribution_prompt.replace(word, synonym)
            ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True,
                      return_tensors="pt").input_ids.to(device)
            positions.append(find_token_subsequence(ids[0].tolist(), tok(synonym, add_special_tokens=False)["input_ids"]))
            parts.append(pipe.text_encoder(ids)[0].float())
    return torch.cat(parts, dim=1), positions


def class_hidden(pipe, prompt: str, device: str) -> torch.Tensor:
    tok = pipe.tokenizer
    ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
    with torch.inference_mode():
        return pipe.text_encoder(ids.to(device))[0]


def main() -> None:
    from diffusers import StableDiffusionPipeline

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("voc", "coco"), required=True)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=ROOT / "configs/controlled_sa_seed_frozen.json")
    parser.add_argument("--folds", type=Path, default=ROOT / "configs/voc_sim_folds.csv")
    parser.add_argument("--tokens", type=Path, default=ROOT / "configs/ovam/tokens_VOC.npy")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crf-site", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    seed_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    voc = args.dataset == "voc"
    classname, rows = (transfer.voc_rows if voc else transfer.coco_rows)(args.class_shard)
    rows = rows[: args.max_samples] if args.max_samples else rows
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if voc else "ovam/coco_captions")
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    processors = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith("attn2")))
    sa_modules = tuple(sorted(n for n in processors if n.endswith("attn1")))
    selected = sa_modules[transfer.SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    released = np.load(args.tokens, allow_pickle=True).item()  # committed copy of OVAM's tokens_VOC.npy (MIT)
    # The attribution prompt is the same for every row of a class (VOC template / COCO class name).
    prompts = {r["attribution_prompt"] for r in rows}
    if len(prompts) != 1:
        raise RuntimeError(f"{classname}: expected one attribution prompt, found {sorted(prompts)}")
    prompt = prompts.pop()
    hidden, positions = shard_keys(pipe, classname, prompt, released, spec["synonyms"], args.device)
    _EXTRA.update(hidden=hidden, synonym_positions=positions, separate_class_prompt=not voc)
    if not voc:
        _EXTRA["class_hidden"] = class_hidden(pipe, prompt, args.device)
    settings = {
        "package": "sg5_contrastive_seed_v1", "protocol": "protocol Section 66", "dataset": args.dataset,
        "attributions": spec["attributions"], "pipelines": spec["pipelines"],
        "synonyms": spec["synonyms"][classname.replace(" ", "-")], "attribution_prompt": prompt,
        "ca_seed": seed_spec, "selected_sa_module": selected, "thresholds": list(THRESHOLDS),
        "configurations": list(CONFIGURATIONS), "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
    }
    folds = cv.read_folds(args.folds) if voc else {r["image_id"]: 0 for r in rows}
    writer = cv.ShardWriter(args.output_dir / args.dataset / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz",
                            CONFIGURATIONS, folds, settings, thresholds=THRESHOLDS)
    t45 = THRESHOLDS.index(0.45)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(pipe, row, manifest_dir, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {n: r for r, names in sample.modules_by_resolution.items() for n in names}
            word_calls = recorder.ca_calls
            seeds = []
            for name in ATTRIBUTIONS:
                recorder.ca_calls = word_calls if name == "W" else recorder.sg5_calls[name]
                seeds.append(sa_common.ca_seed(recorder, ca_modules, ca_res, seed_spec, args.device))
            recorder.ca_calls = word_calls
            seeds = torch.stack(seeds)
            one, two = propagate(recorder, selected, seeds, args.device)
            recorder.sa_qk.clear()
            counts, ap = cv.score_maps(torch.cat([seeds, one, two]), sample.target_mask, "minmax", thresholds=THRESHOLDS)
            n = len(ATTRIBUTIONS)
            crf = {(i, p): crf_counts(m[i], sample.target_mask, sample.generated_image, args.crf_site)[0]
                   for i in range(n) for p, m in (("R2", one), ("P1+R2", two))}
            block = {"seed": 0, "REF": 1, "P1": 2}
            order, aps = [], []
            for i in range(n):
                for p in PIPELINES:
                    if p in block:
                        order.append(counts[block[p] * n + i]); aps.append(ap[block[p] * n + i])
                    else:
                        order.append(crf[(i, p)]); aps.append(np.nan)
            all_counts = np.stack(order)
            writer.add(row["image_id"], classname, all_counts, np.asarray(aps))
            show = ("W/REF", "WS/REF", "WK/REF", "WN/REF", "WE/REF", "WY/REF", "TC/REF", "TS/REF", "W/P1+R2")
            print(f"[{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) " + " ".join(
                f"{c}={all_counts[CONFIGURATIONS.index(c), t45, 0] / max(all_counts[CONFIGURATIONS.index(c), t45, :3].sum(), 1):.3f}"
                for c in show), flush=True)
        finally:
            recorder.ca_calls.clear()
            recorder.sa_qk.clear()
            for calls in recorder.sg5_calls.values():
                calls.clear()
            sample.clear()
            del sample, recorder
            gc.collect()
            torch.cuda.empty_cache()
    if args.max_samples is None:
        writer.finish()


if __name__ == "__main__":
    main()
