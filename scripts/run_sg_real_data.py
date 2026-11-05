#!/usr/bin/env python3
"""SG-real-data: the frozen BRACE pipeline on real photos (PASCAL VOC 2012, COCO val2017).

The frozen pipeline (A3 CA seed, SA calls, capped hierarchy, DenseCRF) runs unchanged. The only new
piece is how a real photo enters the 31-call PNDM trajectory, set by --mode:
  renoise   every UNet input is the real latent noised to that call's timestep (fixed noise)
  free<t0>  same until the timestep drops below t0, then the sampler denoises freely
Prompt: "a photograph of a {class}" for each class present. Scoring: binary per (image, class),
label 255 (VOC void, COCO crowd) ignored, at original resolution; class-macro mIoU is computed in the analysis.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

ROOT = Path("/home/shafa/sd-lora-attention-segmentation")
sys.path.insert(0, str(ROOT))
from scripts import run_coco_cap_transfer as transfer, sg27_runtime  # noqa: E402
from src.segmentation import sg_combined_v2  # noqa: E402
from src.segmentation.ovam_full import _import_densecrf, apply_ovam_densecrf  # noqa: E402

VOC = ROOT / "datasets/voc2012/VOCdevkit/VOC2012"
COCO = ROOT / "datasets/coco2017"
OUT = ROOT / "outputs/segmentation/sg_real_data"
VOC_CLASSES = [transfer.voc_rows(i)[0] for i in range(20)]       # VOC label k -> VOC_CLASSES[k-1]
RAW = ("CA", "C16", "V0", "V1", "V3")
CRF = ("V0", "V3")
NAMES = RAW + tuple(n + "+R2" for n in CRF)
THR = np.asarray(transfer.THRESHOLDS)


@contextlib.contextmanager
def real_trajectory(z0, mode, gen):
    """Patch the pipeline call so its latents follow the real image (see module doc)."""
    orig = StableDiffusionPipeline.__call__
    t0 = 0 if mode == "renoise" else int(mode.removeprefix("free"))
    noise = torch.randn(z0.shape, generator=gen, device=z0.device, dtype=torch.float32)

    def noised(pipe, t):
        a = pipe.scheduler.alphas_cumprod[int(t)].to(z0.device)
        return (a.sqrt() * z0.float() + (1 - a).sqrt() * noise).to(z0.dtype)

    def patched(self, *args, latents=None, callback_on_step_end=None, num_inference_steps=50, **kw):
        self.scheduler.set_timesteps(num_inference_steps, device=z0.device)
        ts = self.scheduler.timesteps.tolist()

        def cb(p, i, t, ckw):
            out = callback_on_step_end(p, i, t, ckw) if callback_on_step_end else ckw
            if i + 1 < len(ts) and ts[i + 1] >= t0:
                out["latents"] = noised(self, ts[i + 1])
            return out
        return orig(self, *args, latents=noised(self, ts[0]), callback_on_step_end=cb,
                    num_inference_steps=num_inference_steps, **kw)

    StableDiffusionPipeline.__call__ = patched
    try:
        yield
    finally:
        StableDiffusionPipeline.__call__ = orig


def counts(full_mask_stack, truth, valid):
    """[T,H,W] bool -> [T,4] tp fp fn tn over valid pixels."""
    p = full_mask_stack & valid; t = truth & valid
    tp = (p & t).sum((1, 2)); fp = (p & ~t).sum((1, 2)); fn = (~p & t[None]).sum((1, 2))
    return np.stack([tp, fp, fn, valid.sum() - tp - fp - fn], 1).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("voc", "coco"), required=True)
    ap.add_argument("--split", choices=("train", "val"), default="val", help="VOC only; COCO uses val2017")
    ap.add_argument("--ids", type=Path, help="file of image ids (default: whole split)")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--modes", nargs="+", default=["renoise"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--crf-site", type=Path, default=ROOT / "outputs/dependencies/simplecrf_py310")
    args = ap.parse_args()
    if args.dataset == "voc":
        ids = (args.ids or VOC / f"ImageSets/Segmentation/{args.split}.txt").read_text().split()
        classes = VOC_CLASSES
        image_path = lambda i: VOC / f"JPEGImages/{i}.jpg"; label_path = lambda i: VOC / f"SegmentationClass/{i}.png"
    else:
        ids = (args.ids or OUT / "coco_val200_ids.txt").read_text().split()
        classes = (OUT / "coco_classes.txt").read_text().splitlines()  # COCO label k -> classes[k-1]
        image_path = lambda i: COCO / f"val2017/{i}.jpg"; label_path = lambda i: COCO / f"sg_real_data_labels/{i}.png"
    ids = ids[args.shard::args.num_shards]
    _import_densecrf(args.crf_site)
    runtime = sg27_runtime.make_runtime(args.model)
    pipe = runtime["pipe"]
    tmp = args.output_dir / "tmp" / f"{args.shard}"; tmp.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        out = args.output_dir / mode / f"shard_{args.shard:03d}_of_{args.num_shards:03d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = dict(np.load(out, allow_pickle=False)) if out.exists() else {}
        for iid in ids:
            image = np.asarray(Image.open(image_path(iid)).convert("RGB"))
            gt = np.asarray(Image.open(label_path(iid)))
            valid = gt != 255
            x = torch.from_numpy(np.asarray(Image.fromarray(image).resize((512, 512), Image.Resampling.BICUBIC)))
            with torch.inference_mode():
                x = x.permute(2, 0, 1)[None].to(pipe.device, pipe.vae.dtype) / 127.5 - 1
                z0 = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
            for k in sorted(set(np.unique(gt)) - {0, 255}):
                cls = classes[k - 1]; key = f"{iid}/{k}"
                if f"{key}/counts" in done:
                    continue
                truth = gt == k
                Image.fromarray(truth.astype(np.uint8) * 255).save(tmp / "gt.png")
                prompt = f"a photograph of a {cls}"
                row = {"image_id": key, "classname": cls, "prompt": prompt, "attribution_prompt": prompt,
                       "seed": "0", "num_inferece_steps": "30", "annotation": str(tmp / "gt.png")}
                gen = torch.Generator(device=pipe.device).manual_seed(zlib.crc32(iid.encode()))
                with real_trajectory(z0, mode, gen), \
                        sg27_runtime.live_maps(runtime, row, args.dataset) as (sample, sa, seed, base, previous):
                    v2, _ = sg_combined_v2.aggregate(sa, seed)
                    maps = {"CA": seed.float(), "C16": base["C16"].reshape(64, 64).float(),
                            "V0": v2["V0"], "V1": v2["V1"], "V3": v2["V3"]}
                c = np.zeros((len(NAMES), len(THR), 4), np.int64)
                thr = torch.as_tensor(THR, device=seed.device).view(-1, 1, 1)
                full = {}
                for i, n in enumerate(RAW):
                    f = transfer.upsampled(transfer.minmax(maps[n]), truth.shape)
                    full[n] = f
                    c[i] = counts((f[None] >= thr).cpu().numpy(), truth, valid)
                for j, n in enumerate(CRF):
                    masks = np.stack([apply_ovam_densecrf(image, (full[n] >= t).cpu().numpy(), args.crf_site)
                                      for t in THR]).astype(bool)
                    c[len(RAW) + j] = counts(masks, truth, valid)
                done[f"{key}/counts"] = c
                done[f"{key}/map"] = maps["V3"].cpu().numpy().astype(np.float16)
                done[f"{key}/area"] = np.asarray(truth[valid].mean())
                np.savez_compressed(out, **done)
                iou = c[:, 10, 0] / np.maximum(c[:, 10, :3].sum(1), 1)
                print(f"[real:{mode}] {key} {cls} IoU@.55 " + " ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, iou)),
                      flush=True)
    np.savetxt(args.output_dir / "names.txt", np.asarray(NAMES), fmt="%s")


if __name__ == "__main__":
    main()
