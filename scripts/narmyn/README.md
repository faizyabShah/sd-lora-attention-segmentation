# Cityscapes LoRA rank sweep (Narmyn)

Pipeline for measuring how LoRA rank affects domain adaptation of SD 1.5 to
Cityscapes. Runs standalone — it does not import from `src/`, so it sits
alongside the main codebase without coupling to it.

## Paths

Scripts read and write a data tree that is **not** in this repository:

```bash
export TRDP_ROOT=$HOME/TRDP      # default if unset: the repo root
```

`$TRDP_ROOT` holds `data/` (crops, captions) and `output/` (weights, images,
evaluation). The SLURM wrappers in `jobs/narmyn/` export it for you.

## Order

| Script | Does |
|---|---|
| `01_crop.py` | 2048x1024 -> central 1024^2 crop, with validity guards |
| `02_caption.py` | Florence-2-large detailed captions, resumable |
| `03_generate_frozen.py` | frozen SD 1.5 baseline images |
| `04_train_lora.py` | LoRA fine-tune (prepares data, launches the vendored trainer) |
| `05_sample_checkpoints.py` | per-checkpoint image grids, for inspecting drift |
| `06_generate_lora.py` | LoRA-adapted generation, paired with the baseline |
| `07_evaluate.py` | FID / KID / CLIP / BLIP / LPIPS |

`train_text_to_image_lora.py` is HuggingFace's trainer, vendored and patched to
cast LoRA params to fp32 after `unet.add_adapter`. It is **not** the same file as
the one at the repository root — that copy lacks the patch.

## Pairing

Image `i` in every generated set comes from caption line `i` with generator seed
`i`. Prompt and initial latent are therefore identical across the frozen model,
every rank, and every checkpoint, so only the adapter differs. Do not regenerate
or re-sort `metadata.jsonl` — every generated image would silently re-pair.

## Environments

| Env | Why |
|---|---|
| `trdp` | main; torch 2.11+cu130 — **renyi only** (needs driver >= 580) |
| `gen` | clone of `trdp` with torch 2.4.1+cu121 — runs on erdos/neumann |
| `caption` | clone pinned to `transformers==4.44.2` — Florence-2 breaks on 5.x |

Pass `CONDA_ENV=gen` to the job scripts to use the V100 nodes.

## Status and findings

See `docs/narmyn/WORKLOG.md` and `docs/narmyn/report.pdf`.
