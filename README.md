# Cityscapes Stable Diffusion LoRA experiments

This repository contains the Cityscapes crop and caption pipeline, Stable Diffusion 1.5 LoRA experiments, generation, and evaluation. GPU work is submitted through Slurm. Attention-map extraction and segmentation in `src/segmentation/` are future work.

## Current experiment

`leftImg8bit/` holds the 5,000 Cityscapes images (2,975 train, 500 validation, 1,525 test). `crop_and_caption.py` makes a centered 1024×1024 crop of each image and captions it with Florence-2 Large. It writes 5,000 JPEGs and `metadata.jsonl` to `cityscapes_cropped/`. Its Slurm job is `caption_cityscapes.slurm`.

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

Ranks are 8, 16, 32, 64, and 128; training seeds are 1–5. Training uses the local Diffusers example `train_text_to_image_lora.py`, 512 resolution, batch size 2 with accumulation 2, 5,000 steps, and saves to `outputs/rank_<rank>/seed_<seed>/`. Generation uses the same ordered captions, 30 inference steps, and generation seeds 10000–14999 for LoRA runs. It saves to `outputs/generated/rank_<rank>/seed_<seed>/` with `prompt_mapping.json`.

All 25 training and generation runs are already present. The metrics for this five-seed sweep have **not** been calculated. Evaluation will write one `metrics.json` per run under `outputs-metrics/sd15-seeds/rank_<rank>/seed_<seed>/`. `summarize` writes mean and sample standard deviation across available training seeds for each rank. A limited validation run writes to a separate directory and omits KID.

For five **distinct frozen SD 1.5** runs, use `generate --frozen --seed 1` through `--seed 5`, or submit `generate_frozen_seeds.slurm`. Each run uses all 5,000 captions and writes to `outputs/generated/frozen/seed_<seed>/`. Its generation seeds are `10000 × seed + image index`, so seed 1 uses 10000–14999 and seed 5 uses 50000–54999. The unseeded `generate --frozen` command retains the earlier `outputs/cityscapes-gen-frozen/` path and image seeds 0–4999. The LoRA runs all use generation seeds 10000–14999, independently of their **training** seed.

The same operation can be called from Python: `from cityscapes_workflow import generate_images; generate_images(frozen=True, seed=1)`. Call it inside a GPU job, not on the login node.

## Slurm jobs

`train_lora_seed.slurm` maps array indices 0–24 to all rank/seed pairs. `generate_lora_1.5.slurm` resumes generation for seed 5. `generate_frozen_seeds.slurm` generates the five frozen sets. `run_evaluate.slurm` maps array indices 0–24 to metric jobs; submit a dependent summary job after they finish (details in that script). Do not run GPU commands on the login node. Slurm scripts are kept in Git; their `.out` and `.err` logs are ignored.

## Other code and historical runs

`legacy/` keeps the previous standalone SD 1.5 and SDXL scripts and their original Slurm jobs for reproducibility. Their paths refer to the repository root and they are not the current workflow. The older generic `src/`, `scripts/`, and `configs/` pipeline remains available for its own experiments; its config values do not describe the completed five-seed sweep. Existing `outputs-sdxl*` and rank-only `outputs/cityscapes-*` artifacts belong to earlier experiments.

Large datasets, generated images, model weights, and local logs are ignored by Git. Existing files on disk are retained.
