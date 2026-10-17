#!/usr/bin/env python3
"""Few-shot COCO-Cap transfer: OVAM's VOC tokens on the frozen pipelines (registry Section 7; protocol Section 55).

Same images, attribution and VOC-frozen thresholds as ``scripts/run_coco_cap_transfer.py``;
nothing is chosen on COCO-Cap. Stages (counts at the 17 phase 3 thresholds):

  W/seed, W/REF      class word (separate class prompt, padded to 77): must equal the
                     training-free transfer exactly (same environment)
  T/seed             OVAM's released VOC token, keys [<SoT>, token], conditional query
  T/REF, T/P1        one / two SA passes on the token seed
  T/R2, T/P1+R2      DenseCRF on those masks (T/R2 is the T-C few-shot pipeline, Section 52)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_coco_cap_transfer as transfer  # noqa: E402  (installs TransferRecorder; replaced below)
import scripts.run_sa_package_b as sa_common  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

THRESHOLDS = transfer.THRESHOLDS
CONFIGURATIONS = ("W/seed", "W/REF", "T/seed", "T/REF", "T/P1", "T/R2", "T/P1+R2")
_TOKEN_PAIR: list[torch.Tensor] = []  # [1, 2, 768]: (<SoT> output, token)


class TokenTransferRecorder(transfer.TransferRecorder):
    """Class-word CA from the padded class prompt (77 keys) plus token CA from [<SoT>, token] (2 keys)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.attribution_hidden_states is None:
            raise RuntimeError("The token transfer needs the separate attribution prompt")
        self.attribution_hidden_states = torch.cat(
            (self.attribution_hidden_states, _TOKEN_PAIR[0].to(self.attribution_hidden_states.dtype)), dim=1)
        self.token_calls: dict[str, list[torch.Tensor]] = {}

    def record(self, module_name, kind, query, key, value, attention_mask, scale, attribution_key=None):
        if kind == "cross":
            if attribution_key is None or attribution_key.shape[-2] != 79:
                raise RuntimeError("Expected 77 class-prompt keys followed by the two token keys")
            word = torch.stack((key[0], attribution_key[0][:, :77]))
            sa_common.SAPackageRecorder.record(self, module_name, kind, query, word, value, None, scale)
            with torch.inference_mode():
                pair = attribution_key[0][:, 77:].float()
                probs = (torch.matmul(query[1].float(), pair.transpose(-1, -2)) * scale).softmax(dim=-1)[..., 1]
                self.token_calls.setdefault(module_name, []).append(probs.half())
        else:
            sa_common.SAPackageRecorder.record(self, module_name, kind, query, key, value, attention_mask, scale)


sa_common.live.CompactAttentionRecorder = TokenTransferRecorder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-shard", type=int, choices=range(20), required=True)
    parser.add_argument("--ca-seed", type=Path, default=Path("configs/controlled_sa_seed_frozen.json"))
    parser.add_argument("--tokens", type=Path, default=Path("configs/ovam/tokens_VOC.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation/coco_cap_transfer"))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crf-site", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--self-test-voc", action="store_true")
    args = parser.parse_args()
    seed_json = json.loads(args.ca_seed.read_text(encoding="utf-8"))
    if not seed_json.get("frozen", False):
        raise SystemExit(f"{args.ca_seed} is not a frozen CA seed")
    classname, rows = (transfer.voc_rows if args.self_test_voc else transfer.coco_rows)(args.class_shard)
    rows = rows[: args.max_samples] if args.max_samples else rows
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if args.self_test_voc else "ovam/coco_captions")
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    transfer._PIPE = pipe
    processors = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith("attn2")))
    sa_modules = tuple(sorted(n for n in processors if n.endswith("attn1")))
    selected = sa_modules[transfer.SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(seed_json, ca_modules, 31)
    released = np.load(args.tokens, allow_pickle=True).item()  # committed copy of OVAM's tokens_VOC.npy (MIT)
    with torch.inference_mode():
        empty = pipe.tokenizer("", return_tensors="pt").input_ids.to(args.device)
        sot = pipe.text_encoder(empty)[0][:, :1].float()  # CLIP is causal: <SoT> output is prompt-independent
        token = torch.from_numpy(np.asarray(released[classname.replace(" ", "-")], dtype=np.float32)).to(args.device)
        _TOKEN_PAIR[:] = [torch.cat((sot, token[None, None]), dim=1)]
    settings = {
        "package": "coco_cap_token_transfer_v1", "protocol": "phase 3 registry Section 7; protocol Section 55",
        "dataset": "voc_sim self-test" if args.self_test_voc else "coco_cap",
        "attribution": {"W": "separate class prompt padded to 77; conditional query; all class pieces",
                        "T": "OVAM released VOC token; keys [<SoT>, token]; conditional query"},
        "ca_seed": seed_spec, "selected_sa_module": selected, "thresholds": list(THRESHOLDS),
        "configurations": list(CONFIGURATIONS), "model": args.model, "model_dtype": "float32", "qk_storage_dtype": "float16",
        "note": "nothing chosen on COCO-Cap; no folds (fold 0 for every image)",
    }
    out_dir = args.output_dir / ("self_test_voc_token" if args.self_test_voc else "coco_cap_token")
    writer = cv.ShardWriter(out_dir / "shards" / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.npz",
                            CONFIGURATIONS, {r["image_id"]: 0 for r in rows}, settings, thresholds=THRESHOLDS)
    t45 = THRESHOLDS.index(0.45)
    for row in rows:
        if row["image_id"] in writer.completed:
            continue
        transfer._ATTRIBUTION_PROMPT[0] = row["attribution_prompt"]
        sa_common._RECORDERS.clear()
        sample = sa_common.live.generate_attention_sample(pipe, row, manifest_dir, "float16", cross_map_kind="conditional")
        recorder = sa_common._RECORDERS[-1]
        try:
            ca_res = {n: r for r, names in sample.modules_by_resolution.items() for n in names}
            word_seed = sa_common.ca_seed(recorder, ca_modules, ca_res, seed_spec, args.device)
            word_calls, recorder.ca_calls = recorder.ca_calls, recorder.token_calls
            token_seed = sa_common.ca_seed(recorder, ca_modules, ca_res, seed_spec, args.device)
            recorder.ca_calls = word_calls
            word_one, _ = transfer.propagate(recorder, selected, word_seed, args.device)
            token_one, token_two = transfer.propagate(recorder, selected, token_seed, args.device)
            recorder.sa_qk.clear()
            counts, ap = cv.score_maps(torch.stack([word_seed, word_one, token_seed, token_one, token_two]),
                                       sample.target_mask, "minmax", thresholds=THRESHOLDS)
            crf = np.concatenate([transfer.crf_counts(m, sample.target_mask, sample.generated_image, args.crf_site)
                                  for m in (token_one, token_two)])
            all_counts = np.concatenate((counts, crf))
            writer.add(row["image_id"], classname, all_counts, np.concatenate((ap, [np.nan, np.nan])))
            print(f"[{classname}] {row['image_id']} ({len(writer.completed)}/{len(rows)}) "
                  + " ".join(f"{n}={all_counts[i, t45, 0] / max(all_counts[i, t45, :3].sum(), 1):.3f}" for i, n in enumerate(CONFIGURATIONS)),
                  flush=True)
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
