"""Offline pilot: OVAM's optimized VOC tokens as the CA seed for the final SA method.

Exploratory, run before the full live experiment (run_token_crf.py). Works on
saved disk captures (outputs/attention/voc_sim/ovam-voc-sim-*, the old capture
format with full CA probabilities and Q/K), so nothing is regenerated. The
original run used the 49 annotated captures among indices 0-49 (all aeroplane
and bicycle images).

Seeds compared (all built from the unconditional query, averaged over all 31 calls):
  plain_top4   -- the final method's seed: class word, 12 layers, fixed top-4 heads
  opt_top4     -- same aggregation, OVAM optimized token instead of the class word
  opt_all16    -- optimized token, all 16 layers x all 8 heads (no selection)
  plain_ovam15 / opt_ovam15 -- OVAM aggregation: 15 non-mid layers, heads summed

Each seed is scored CA-only and after the final SA operator (up_blocks.3.attentions.2
attn1, A^n s0, calls 15-30, heads 0,1,2,4,5,6, percentile 1-99, >= 0.45, 1% cleanup).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # do not leave __pycache__ inside the shared repo

import argparse
import csv
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.segmentation.attention_aggregation import attention_to_mask, normalize_map  # noqa: E402
from src.segmentation.focused_self_attention import propagate_selected_module  # noqa: E402
from src.segmentation.metrics import BinarySegmentationMetrics  # noqa: E402
from src.segmentation.ovam_full import apply_ovam_densecrf  # noqa: E402
from src.segmentation.postprocessing import postprocess_mask  # noqa: E402
from src.segmentation.stage1_suite import BEST_12_INDICES  # noqa: E402

CAPTURES = REPO / "outputs" / "attention" / "voc_sim"
MANIFEST = REPO / "datasets/ovam_experiment_with_dataset/evaluation/ovam/voc_sim/ovam_voc_sim.csv"
TOKENS = REPO / "configs/ovam/tokens_VOC.npy"
STAGE1 = REPO / "outputs/segmentation/voc_sim_100_stage1_suite/results.json"
UNET = next(
    (Path.home() / ".cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots").glob(
        "*/unet/diffusion_pytorch_model.safetensors"
    )
)
SA_MODULE = "up_blocks.3.attentions.2.transformer_blocks.0.attn1"
SA_HEADS = (0, 1, 2, 4, 5, 6)
LATE = 15
SEEDS = ("plain_top4", "opt_top4", "opt_all16", "plain_ovam15", "opt_ovam15")


@dataclass
class Sample:
    sample_id: str
    root: Path
    classname: str
    prompt: str
    token_ids: tuple[int, ...]
    timesteps: tuple[int, ...]
    cross_modules: tuple[str, ...]
    resolution: dict[str, int]
    self_modules_by_resolution: dict[int, tuple[str, ...]]
    target: np.ndarray

    @contextmanager
    def open_attention(self, step_index: int):
        path = self.root / "attention" / f"step_{step_index:03d}.safetensors"
        with safe_open(path, framework="pt", device="cpu") as tensors:
            yield tensors

    def load_target_mask(self) -> np.ndarray:
        return self.target


def load_samples(limit: int | None) -> list[Sample]:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = {row["image_id"]: row for row in csv.DictReader(handle)}
    samples = []
    roots = sorted(CAPTURES.glob("ovam-voc-sim-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    for root in roots:
        row = rows[root.name]
        if not (root / "COMPLETE").is_file() or row["annotated"].lower() != "true":
            continue
        meta = json.loads((root / "metadata.json").read_text())
        schema = json.loads((root / "attention_schema.json").read_text())
        cross, res, self_grouped = [], {}, {}
        for name, module in schema["modules"].items():
            r = module.get("spatial_height")
            if r is None:
                continue
            if module["kind"] == "cross":
                cross.append(name)
                res[name] = int(r)
            else:
                self_grouped.setdefault(int(r), []).append(name)
        pixels = np.asarray(Image.open(MANIFEST.parent / row["annotation"]).convert("RGB"))
        samples.append(
            Sample(
                sample_id=root.name,
                root=root,
                classname=meta["classname"],
                prompt=meta["prompt"],
                token_ids=tuple(meta["token_ids"]),
                timesteps=tuple(meta["timesteps"]),
                cross_modules=tuple(sorted(cross)),
                resolution=res,
                self_modules_by_resolution={k: tuple(sorted(v)) for k, v in self_grouped.items()},
                target=np.any(pixels != 0, axis=-1),
            )
        )
    return samples[:limit] if limit else samples


def load_to_k(modules: tuple[str, ...], device: torch.device) -> dict[str, torch.Tensor]:
    with safe_open(UNET, framework="pt", device="cpu") as weights:
        return {m: weights.get_tensor(f"{m}.to_k.weight").float().to(device) for m in modules}


@torch.inference_mode()
def ca_layer_heads(sample: Sample, to_k: dict, tokens: dict, device) -> dict[str, torch.Tensor]:
    """Return {'plain','opt'} -> [16, 8, 64, 64] CA maps averaged over all calls."""
    modules = sample.cross_modules
    if len(modules) != 16:
        raise ValueError(f"{sample.sample_id}: expected 16 CA modules, found {len(modules)}")
    prompt_len = sample.token_ids.index(49407) + 1
    position = len(sample.prompt.split())
    with safe_open(sample.root / "text_embeddings.safetensors", framework="pt", device="cpu") as t:
        sot = t.get_tensor("conditional")[0, 0].float()
    opt_hidden = torch.stack((sot, torch.from_numpy(tokens[sample.classname]).float())).to(device)

    plain_keys, opt_keys = {}, {}
    with safe_open(sample.root / "attention" / "cross_attention_kv.safetensors", framework="pt", device="cpu") as kv:
        for m in modules:
            k = kv.get_tensor(f"{m}.k")[1, :, :prompt_len].float().to(device)  # conditional, unpadded
            plain_keys[m] = k
            heads, _, dim = k.shape
            ok = (opt_hidden @ to_k[m].T).view(2, heads, dim).transpose(0, 1)  # [heads, 2, dim]
            opt_keys[m] = ok
    out = {name: torch.zeros((16, 8, 64, 64), device=device) for name in ("plain", "opt")}
    steps = len(sample.timesteps)
    for step in range(steps):
        with sample.open_attention(step) as tensors:
            for index, m in enumerate(modules):
                q = tensors.get_tensor(f"{m}.q")[0].float().to(device)  # unconditional query
                r = sample.resolution[m]
                scale = q.shape[-1] ** -0.5
                for name, keys, pos in (("plain", plain_keys, position), ("opt", opt_keys, 1)):
                    p = torch.softmax(q @ keys[m].transpose(-1, -2) * scale, dim=-1)[..., pos]
                    maps = p.reshape(-1, 1, r, r)
                    if r != 64:
                        maps = F.interpolate(maps, size=(64, 64), mode="bilinear", align_corners=False)
                    out[name][index] += maps[:, 0]
    for name in out:
        out[name] /= steps
    return out


def build_seeds(layer_heads: dict, modules: tuple[str, ...], rankings: dict) -> dict[str, torch.Tensor]:
    def top4(lh):
        maps = [lh[i, [int(h) for h in rankings[str(i)][:4]]].mean(0) for i in BEST_12_INDICES]
        return torch.stack(maps).mean(0)

    non_mid = [i for i, m in enumerate(modules) if not m.startswith("mid_block")]
    raw = {
        "plain_top4": top4(layer_heads["plain"]),
        "opt_top4": top4(layer_heads["opt"]),
        "opt_all16": layer_heads["opt"].mean(dim=(0, 1)),
        "plain_ovam15": layer_heads["plain"][non_mid].sum(dim=(0, 1)),
        "opt_ovam15": layer_heads["opt"][non_mid].sum(dim=(0, 1)),
    }
    return {k: normalize_map(v.cpu(), "percentile_01_99") for k, v in raw.items()}, raw


CRF_KEYS = (
    "plain_top4/sa_iter4", "opt_top4/sa_iter4", "opt_top4/sa_iter1", "opt_all16/sa_iter4",
    "plain_top4/sa_iter4_nocleanup", "opt_top4/sa_iter4_nocleanup",
    "plain_ovam15/ca_only_ovammax_t0.40", "opt_ovam15/ca_only_ovammax_t0.40",
)


def add_crf(sample: Sample, preds: dict, site: Path) -> None:
    """OVAM's DenseCRF (same SimpleCRF call and parameters) on the regenerated image."""
    image = np.asarray(Image.open(sample.root / "generated.png").convert("RGB"))
    for key in CRF_KEYS:
        mask = preds[key].cpu().numpy().astype(bool)
        preds[f"{key}+crf"] = torch.from_numpy(apply_ovam_densecrf(image, mask, site))


def predictions(sample: Sample, seeds: dict, raw: dict, device: str) -> dict[str, torch.Tensor]:
    shape = sample.target.shape
    out: dict[str, torch.Tensor] = {}

    def final(score, threshold=0.45, cleanup=True):
        mask = attention_to_mask(score, shape, threshold, "percentile_01_99")
        if cleanup:
            mask = torch.from_numpy(postprocess_mask(mask.numpy(), "remove_fraction_0.01"))
        return mask

    for name in SEEDS:
        out[f"{name}/ca_only_pct_t0.45"] = attention_to_mask(seeds[name], shape, 0.45, "percentile_01_99")
        out[f"{name}/ca_only_ovammax_t0.40"] = attention_to_mask(raw[name].cpu(), shape, 0.40, "ovam_max_after_resize")

    matrix = torch.stack([seeds[name].flatten() for name in SEEDS], dim=1)
    propagated = propagate_selected_module(
        sample, SA_MODULE, matrix, device, track_seed_columns=tuple(range(len(SEEDS)))
    )
    steps = len(sample.timesteps)
    for iteration in (1, 2, 3, 4):
        values = propagated.per_step_head_columns[iteration].reshape(steps, 8, 64, 64, len(SEEDS))
        for column, name in enumerate(SEEDS):
            score = values[..., column][LATE:, list(SA_HEADS)].mean(dim=(0, 1))
            out[f"{name}/sa_iter{iteration}"] = final(score)
            if iteration == 4:
                out[f"{name}/sa_iter4_nocleanup"] = final(score, cleanup=False)
                for t in (0.35, 0.40, 0.50, 0.55):
                    out[f"{name}/sa_iter4_t{t:.2f}"] = final(score, threshold=t)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=REPO / "outputs/segmentation/narmyn_token_crf/pilot_offline_49")
    parser.add_argument("--crf-site", type=Path, help="site-packages dir containing SimpleCRF's denseCRF")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    tokens = np.load(TOKENS, allow_pickle=True).item()
    rankings = json.loads(STAGE1.read_text())["settings"]["head_rankings"]
    samples = load_samples(args.limit)
    print(f"{len(samples)} annotated captures: {sorted({s.classname for s in samples})}", flush=True)
    to_k = load_to_k(samples[0].cross_modules, device)

    metrics: dict[str, BinarySegmentationMetrics] = {}
    with (args.out / "per_image.jsonl").open("w") as log:
        for number, sample in enumerate(samples, 1):
            start = time.time()
            layer_heads = ca_layer_heads(sample, to_k, tokens, device)
            seeds, raw = build_seeds(layer_heads, sample.cross_modules, rankings)
            preds = predictions(sample, seeds, raw, str(device))
            if args.crf_site:
                add_crf(sample, preds, args.crf_site)
            record = {"image_id": sample.sample_id, "classname": sample.classname, "results": {}}
            truth = sample.target
            for key, pred in preds.items():
                metrics.setdefault(key, BinarySegmentationMetrics()).update(sample.classname, pred, truth)
                p = pred.cpu().numpy().astype(bool)
                tp, fp, fn = int((p & truth).sum()), int((p & ~truth).sum()), int((~p & truth).sum())
                record["results"][key] = {"tp": tp, "fp": fp, "fn": fn, "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"[{number}/{len(samples)}] {sample.sample_id} {sample.classname} "
                  f"plain={record['results']['plain_top4/sa_iter4']['iou']:.3f} "
                  f"opt={record['results']['opt_top4/sa_iter4']['iou']:.3f} ({time.time() - start:.0f}s)", flush=True)

    summary = {key: metric.result() for key, metric in metrics.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    classes = sorted({s.classname for s in samples})
    print("\nconfiguration".ljust(44) + "".join(c[:10].rjust(11) for c in classes) + "      mIoU")
    for key in sorted(summary):
        row = summary[key]
        print(key.ljust(44) + "".join(f"{row['per_class'][c]['iou']:11.4f}" for c in classes) + f"{row['miou']:10.4f}")


if __name__ == "__main__":
    main()
