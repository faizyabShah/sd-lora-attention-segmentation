#!/usr/bin/env python3
"""AN1 part 2 (protocol Section 58): where do the pixels each SA pass adds land?

Analysis only; nothing is selected. One live generation per image, on VOC-Sim
or COCO-Cap, with the frozen CA/SA pipeline and the attribution of the
transfers (class prompt padded to 77, plus OVAM's VOC token; Sections 48, 55).
Masks are min-max maps, bilinear to the annotation, ``>= 0.45``.

For each transition between consecutive passes (class word: seed -> 1 -> 2 -> 3;
token: seed -> 1 -> 2) it counts, at annotation resolution:

  new TP   pixels that become true positives, by distance to the previous mask
  new FP   pixels that become false positives, by distance to the object, and
           whether they lie in a predicted component that touches the object
           ("bleeding") or not ("jumping")
  lost TP / lost FP   pixels that leave the mask

Distances (pixels at 512x512) are binned 1-4, 5-16, 17-64, > 64. One JSON line
per image; ``--merge`` sums them into ``summary.json``.

    python scripts/narmyn/attention/an1_spatial.py --dataset voc --class-shard 0
    python scripts/narmyn/attention/an1_spatial.py --merge
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "segmentation" / "analysis" / "an1_spatial"
BINS = (4, 16, 64)  # upper edges; the last bin is > 64
BIN_NAMES = ("1-4", "5-16", "17-64", ">64")
WORD = ("seed", "1", "2", "3")
TOKEN = ("seed", "1", "2")


def binned(distances: np.ndarray) -> list[int]:
    edges = np.digitize(distances, BINS, right=True)
    return [int((edges == i).sum()) for i in range(len(BIN_NAMES))]


def transition(prev: np.ndarray, cur: np.ndarray, truth: np.ndarray, dist_to_prev: np.ndarray, dist_to_truth: np.ndarray) -> dict:
    new_tp = cur & ~prev & truth
    new_fp = cur & ~prev & ~truth
    labels, _ = ndimage.label(cur)
    touching = np.unique(labels[cur & truth])
    attached = np.isin(labels, touching[touching > 0])
    return {"new_tp": int(new_tp.sum()), "new_fp": int(new_fp.sum()),
            "lost_tp": int((prev & ~cur & truth).sum()), "lost_fp": int((prev & ~cur & ~truth).sum()),
            "new_tp_dist_to_prev_mask": binned(dist_to_prev[new_tp]),
            "new_fp_dist_to_object": binned(dist_to_truth[new_fp]),
            "new_fp_bleeding": int((new_fp & attached).sum()), "new_fp_jumping": int((new_fp & ~attached).sum())}


def image_record(maps: dict[str, list[torch.Tensor]], truth: np.ndarray) -> dict:
    from scripts.run_fusion_calibration import upsampled
    from src.segmentation.fusion_variants import minmax
    shape = truth.shape
    dist_to_truth = ndimage.distance_transform_edt(~truth)
    rec = {"object_pixels": int(truth.sum())}
    for attribution, chain in maps.items():
        masks = [(upsampled(minmax(m), shape) >= 0.45).cpu().numpy() for m in chain]
        names = WORD if attribution == "word" else TOKEN
        rec[attribution] = {}
        for i in range(1, len(masks)):
            prev, cur = masks[i - 1], masks[i]
            dist_to_prev = ndimage.distance_transform_edt(~prev) if prev.any() else np.full(shape, 1e6)
            rec[attribution][f"{names[i - 1]}->{names[i]}"] = transition(prev, cur, truth, dist_to_prev, dist_to_truth)
        rec[attribution]["masks"] = {n: {"tp": int((m & truth).sum()), "fp": int((m & ~truth).sum()), "fn": int((~m & truth).sum())}
                                     for n, m in zip(names, masks)}
    return rec


def run(args) -> None:
    import scripts.run_coco_cap_token_transfer as tt  # installs TokenTransferRecorder
    import scripts.run_coco_cap_transfer as transfer
    import scripts.run_sa_package_b as sa_common
    from diffusers import StableDiffusionPipeline

    classname, rows = (transfer.voc_rows if args.dataset == "voc" else transfer.coco_rows)(args.class_shard)
    rows = rows[: args.max_samples] if args.max_samples else rows
    manifest_dir = transfer.EVAL / ("ovam/voc_sim" if args.dataset == "voc" else "ovam/coco_captions")
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    transfer._PIPE = pipe
    processors = [n.removesuffix(".processor") for n in pipe.unet.attn_processors]
    ca_modules = tuple(sorted(n for n in processors if n.endswith("attn2")))
    sa_modules = tuple(sorted(n for n in processors if n.endswith("attn1")))
    selected = sa_modules[transfer.SELECTED_SA_MODULE]
    seed_spec = sa_common.resolve_seed_spec(json.loads(Path("configs/controlled_sa_seed_frozen.json").read_text()), ca_modules, 31)
    released = np.load("configs/ovam/tokens_VOC.npy", allow_pickle=True).item()  # committed copy of OVAM's tokens (MIT)
    with torch.inference_mode():
        empty = pipe.tokenizer("", return_tensors="pt").input_ids.to(args.device)
        sot = pipe.text_encoder(empty)[0][:, :1].float()
        token = torch.from_numpy(np.asarray(released[classname.replace(" ", "-")], dtype=np.float32)).to(args.device)
        tt._TOKEN_PAIR[:] = [torch.cat((sot, token[None, None]), dim=1)]
    path = OUT / args.dataset / f"class_{args.class_shard:02d}_{classname.replace(' ', '_')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["image_id"] for line in path.open()} if path.exists() else set()
    with path.open("a") as log:
        for row in rows:
            if row["image_id"] in done:
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
                seeds = torch.stack([word_seed, token_seed]).reshape(2, -1).T  # [P, 2]
                powers = [seeds]
                calls = recorder.sa_qk[selected]
                acc = [torch.zeros_like(seeds) for _ in range(3)]
                for query, key in calls:
                    q, k = query.to(args.device).float(), key.to(args.device).float()
                    a = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)
                    x = seeds
                    for n in range(3):
                        x = a @ x if x.dim() == 3 else a @ x[None]
                        acc[n] += x.mean(dim=0)
                    del q, k, a, x
                powers += [p / len(calls) for p in acc]
                recorder.sa_qk.clear()
                maps = {"word": [p[:, 0].reshape(64, 64) for p in powers],
                        "token": [p[:, 1].reshape(64, 64) for p in powers[:3]]}
                rec = {"image_id": row["image_id"], "class": classname, **image_record(maps, np.asarray(sample.target_mask, bool))}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                w = rec["word"]["masks"]
                print(f"[{classname}] {row['image_id']} word IoU " + " ".join(
                    f"{n}={v['tp'] / max(v['tp'] + v['fp'] + v['fn'], 1):.3f}" for n, v in w.items()), flush=True)
            finally:
                recorder.ca_calls.clear(); recorder.sa_qk.clear(); recorder.token_calls.clear(); sample.clear()
                del sample, recorder
                gc.collect(); torch.cuda.empty_cache()


def merge() -> None:
    out = {"bins_pixels_at_annotation_resolution": list(BIN_NAMES), "threshold": 0.45, "datasets": {}}
    for ds in ("voc", "coco"):
        recs = [json.loads(l) for f in sorted((OUT / ds).glob("*.jsonl")) for l in f.open()]
        if not recs:
            continue
        d = {"images": len(recs)}
        for att in ("word", "token"):
            d[att] = {}
            for tr in [k for k in recs[0][att] if "->" in k]:
                agg = {}
                for r in recs:
                    for k, v in r[att][tr].items():
                        agg[k] = (np.add(agg[k], v).tolist() if isinstance(v, list) else agg.get(k, 0) + v) if k in agg else v
                fp = max(agg["new_fp"], 1); tp = max(agg["new_tp"], 1)
                agg["share_new_fp_jumping"] = agg["new_fp_jumping"] / fp
                agg["share_new_fp_far_gt16px"] = (agg["new_fp_dist_to_object"][2] + agg["new_fp_dist_to_object"][3]) / fp
                agg["share_new_tp_far_gt16px"] = (agg["new_tp_dist_to_prev_mask"][2] + agg["new_tp_dist_to_prev_mask"][3]) / tp
                agg["net_tp_minus_fp_per_object_pixel"] = (agg["new_tp"] - agg["lost_tp"] - agg["new_fp"] + agg["lost_fp"]) / sum(r["object_pixels"] for r in recs)
                d[att][tr] = agg
        out["datasets"][ds] = d
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for ds, d in out["datasets"].items():
        for att in ("word", "token"):
            for tr, a in d[att].items():
                print(f"{ds:4s} {att:5s} {tr:8s} new TP {a['new_tp']:>9d} (far>16px {a['share_new_tp_far_gt16px']:.2f}) | new FP {a['new_fp']:>9d} "
                      f"(jumping {a['share_new_fp_jumping']:.2f}, far>16px {a['share_new_fp_far_gt16px']:.2f}) | lost TP {a['lost_tp']:>8d} lost FP {a['lost_fp']:>8d} "
                      f"| net per object px {a['net_tp_minus_fp_per_object_pixel']:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("voc", "coco"))
    parser.add_argument("--class-shard", type=int, choices=range(20))
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    merge() if args.merge else run(args)


if __name__ == "__main__":
    main()
