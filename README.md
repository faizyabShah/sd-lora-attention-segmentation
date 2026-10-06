# Cityscapes Stable Diffusion LoRA experiments

This repository contains the Cityscapes crop and caption pipeline, Stable Diffusion 1.5 LoRA experiments, generation, and evaluation. GPU work is submitted through Slurm. Attention-map extraction and segmentation in `src/segmentation/` are future work.

## Current experiment

`datasets/leftImg8bit/` holds the 5,000 Cityscapes images (2,975 train, 500 validation, 1,525 test). `crop_and_caption.py` makes a centered 1024×1024 crop of each image and captions it with Florence-2 Large. It writes 5,000 JPEGs and `metadata.jsonl` to `datasets/cityscapes_cropped/`. Its Slurm job is `jobs/caption_cityscapes.slurm`.

`cityscapes_workflow.py` is the single active entry point for SD 1.5 training, generation, and evaluation:

```bash
python cityscapes_workflow.py train --rank 8 --seed 1 --dry-run
python cityscapes_workflow.py train --rank 8 --seed 1
python cityscapes_workflow.py generate --rank 8 --seed 1
python cityscapes_workflow.py generate --frozen
python cityscapes_workflow.py generate --frozen --seed 1 --dry-run
python cityscapes_workflow.py generate --frozen --seed 1
python cityscapes_workflow.py evaluate --ranks 8 --seeds 1
python cityscapes_workflow.py summarize
```

Ranks are 8, 16, 32, 64, and 128; training seeds are 1–5. Training uses the local Diffusers example `train_text_to_image_lora.py`, 512 resolution, batch size 2 with accumulation 2, 5,000 steps, and saves to `outputs/lora/rank_<rank>/seed_<seed>/`. Generation uses the same ordered captions, 30 inference steps, and generation seeds 10000–14999 for LoRA runs. It saves to `outputs/lora/generated/rank_<rank>/seed_<seed>/` with `prompt_mapping.json`.

All 25 training and generation runs are already present. The metrics for this five-seed sweep have **not** been calculated. Evaluation will write one `metrics.json` per run under `outputs/lora/outputs-metrics/sd15-seeds/rank_<rank>/seed_<seed>/`. `summarize` writes mean and sample standard deviation across available training seeds for each rank. A limited validation run writes to a separate directory and omits KID.

For five **distinct frozen SD 1.5** runs, use `generate --frozen --seed 1` through `--seed 5`, or submit `jobs/generate_frozen_seeds.slurm`. Each run uses all 5,000 captions and writes to `outputs/lora/generated/frozen/seed_<seed>/`. Its generation seeds are `10000 × seed + image index`, so seed 1 uses 10000–14999 and seed 5 uses 50000–54999. The unseeded `generate --frozen` command retains the earlier `outputs/lora/cityscapes-gen-frozen/` path and image seeds 0–4999. The LoRA runs all use generation seeds 10000–14999, independently of their **training** seed.

The same operation can be called from Python: `from cityscapes_workflow import generate_images; generate_images(frozen=True, seed=1)`. Call it inside a GPU job, not on the login node.

## Slurm jobs

`jobs/train_lora_seed.slurm` maps array indices 0–24 to all rank/seed pairs. `jobs/generate_lora_1.5.slurm` resumes generation for seed 5. `jobs/generate_frozen_seeds.slurm` generates the five frozen sets. `jobs/run_evaluate.slurm` maps array indices 0–24 to metric jobs; submit a dependent summary job after they finish (details in that script). Do not run GPU commands on the login node. Slurm scripts are kept in Git; their `.out` and `.err` logs are ignored.

`jobs/evaluate_all_renyi.slurm` is the sequential 12-hour evaluation job pinned to `renyi`. It evaluates frozen seeds 1–4, then all 25 LoRA rank/seed runs, calculating FID, KID, CLIPScore, BLIP ITM/cosine, and LPIPS. Each completed run is saved under `outputs/lora/outputs-metrics/sd15-seeds/`; resubmitting the same job skips saved results. The rank summary includes a frozen row marked incomplete until seed 5 has been evaluated.

`jobs/generate_checkpoints_r8_r16_renyi.slurm` generates 500-image subsets for frozen seeds 1–5 and ranks 8/16 at checkpoints 1200, 2400, and 3600. `jobs/generate_checkpoints_r32_r64_r128_renyi.slurm` does the same checkpoint sweep for ranks 32/64/128. Both select every tenth caption (indices 0–4990), run sequentially on `renyi`, and resume by skipping existing nonempty images.

The corresponding offset-1 jobs append captions 1, 11, ..., 4991 into the same directories, producing 1,000 images per model/checkpoint/seed. `jobs/evaluate_checkpoints_renyi.slurm` evaluates those combined sets with FID, KID, CLIPScore, BLIP ITM/cosine, and LPIPS, saving per-run and aggregate results under `outputs/lora/outputs-metrics/sd15-checkpoints/`.

## Other code and historical runs

`legacy/` keeps the previous standalone SD 1.5 and SDXL scripts and their original Slurm jobs for reproducibility. Their paths refer to the repository root and they are not the current workflow. The older generic `src/`, `scripts/`, and `configs/` pipeline remains available for its own experiments; its config values do not describe the completed five-seed sweep. Existing `outputs/lora/outputs-sdxl*` and rank-only `outputs/lora/cityscapes-*` artifacts belong to earlier experiments.

## OVAM attention capture

`scripts/capture_ovam_attention.py` regenerates records from the VOC-Sim manifest and records every SD 1.5 UNet attention module at every denoising step. Self-attention is stored as head-separated Q/K/V tensors; cross-attention stores Q and explicit token probabilities, with its constant K/V tensors stored once per image. The capture also retains the initial noise, per-step latents, text embeddings, token IDs, generation settings, generated/reference images, and annotations. Outputs are resumable and written under `outputs/attention/voc_sim/`.

Submit the first 100 records with `sbatch jobs/capture_ovam_voc_sim_100.slurm`. Captures use float16 storage and are large: a run may require several hundred gigabytes. `src.attention_capture.reconstruct_self_attention` reconstructs a selected self-attention matrix from Q/K, and supports query slicing to avoid materializing a complete 64x64 affinity matrix unnecessarily.

Large datasets, generated images, model weights, and local logs are ignored by Git. Existing files on disk are retained.
