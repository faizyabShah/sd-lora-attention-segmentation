# OVAM optimized tokens and DenseCRF on the earlier final method

**Owner:** Narmyn  
**Status:** exploratory, run on 2026-10-13 before
`docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md` existed. No setting was
selected from these runs, but the method they extend was itself tuned on
VOC-Sim, and its one-pass variant was informed by earlier COCO-Cap results.
Treat the numbers as evidence for the deferred phase (protocol Section 8), not
as frozen choices.

## Question

Does the earlier final method improve when its cross-attention (CA) seed comes
from OVAM's released optimized token instead of the class word, and when OVAM's
DenseCRF is added at the end?

## Data

| Item | Source |
|---|---|
| VOC-Sim | 490 annotated images (OVAM release manifest; the 10 rows without masks are excluded) |
| COCO-Cap | 707 annotated images, rows built exactly as in `scripts/build_coco_cap_manifest.py` |
| Annotations | OVAM public dataset; fetched member by member with `scripts/narmyn/attention/fetch_ovam_release_files.py` (identical to the files in the full zip) |
| Optimized tokens | `configs/ovam/tokens_VOC.npy`, unchanged from the OVAM repository (MIT). One token per class, generated with "A photograph of a ⟨class⟩" and seeds outside the evaluation sets. Each released token is the mean of 2–3 per-image tokens, each optimized on one hand-annotated image (OVAM archive `evaluation/ovam/voc_sim_train/`; see `configs/ovam/README.md`) |
| Released OVAM masks | per-image scores already committed in `outputs/segmentation/ovam_paper_baseline/` (VOC-Sim only: the release has no optimized COCO-Cap masks) |

Every image is regenerated live from its released prompt and seed with the
repository's capture code (SD 1.5, 30 steps = 31 UNet calls, guidance 7.5,
float32 model, float16 attention storage). Only per-image `tp`, `fp`, `fn` are
kept.

## Methods

All rows below share these frozen settings unless the name says otherwise:
SA module `up_blocks.3.attentions.2.transformer_blocks.0.attn1`, SA calls
15–30, SA heads {0,1,2,4,5,6}, percentile 1–99 normalization, threshold ≥ 0.45,
removal of components smaller than 1% of the image.

| Key | CA seed | SA | Post-processing |
|---|---|---|---|
| `ovam_implemented` | OVAM (`src/segmentation/ovam_full.py`): VOC α = 0.85, τ = 0.40; COCO α = 0.91, τ = 0.50 | OVAM all-ones SA fusion | DenseCRF |
| `plain_top4/*` | class word, OVAM convention (unconditional query, unpadded softmax, first piece on VOC / separate class prompt on COCO); 12 layers {0,2,5,6,7,9,10,11,12,13,14,15} with the fixed top-4 heads from `voc_sim_100_stage1_suite`; all calls | as above | as above |
| `opt_top4/*` | optimized token, same 12 layers and heads | as above | as above |
| `opt_all16/*` | optimized token, all 16 layers × 8 heads | as above | as above |

Suffixes: `ca_only` = no SA; `sa_iter1` / `sa_iter4` = 1 or 4 propagation
passes (`A x`, `A⁴ x`); `+crf` = OVAM's SimpleCRF call and parameters
`(10, 80, 13, 3, 3, 5)` on the regenerated image, starting from the binary mask.

**Optimized-token attribution.** Keys are computed by each CA layer's own
`to_k` from `[<SoT>, token]`, where `<SoT>` is the text encoder's output at
position 0 (prompt-independent because CLIP is causal). The softmax is taken
over these two tokens with the unconditional query, and the token's share
(channel 1) is the map, as in OVAM's evaluation (Supp. A.2). It is captured by
a recorder subclass in `run_token_crf.py`; the shared capture code is not
modified.

## Results

### VOC-Sim (490 images)

| Method | mIoU | Precision | Recall |
|---|---:|---:|---:|
| Implemented OVAM | 0.6945 | 0.707 | 0.898 |
| Released OVAM, unoptimized | 0.7036 | — | — |
| Released OVAM, token-optimized | **0.8247** | — | — |
| Ours, class word, 1 pass | 0.7802 | 0.873 | 0.862 |
| Ours, class word, 1 pass, + DenseCRF | 0.7957 | 0.890 | 0.861 |
| Ours, class word, 4 passes | 0.8000 | 0.869 | 0.897 |
| Ours, optimized token, 1 pass | 0.8347 | 0.875 | 0.943 |
| Ours, optimized token, 1 pass, + DenseCRF | 0.8525 | 0.895 | 0.939 |
| Ours, optimized token, 16 layers / all heads, 1 pass, + DenseCRF | **0.8553** | 0.895 | 0.943 |

### COCO-Cap (707 images)

| Method | mIoU | Precision | Recall |
|---|---:|---:|---:|
| Implemented OVAM | 0.5717 | 0.657 | 0.788 |
| OVAM token-optimized (paper Table 1; not reproducible per image) | **0.692** | — | — |
| Ours, class word, 1 pass | 0.6264 | 0.659 | 0.869 |
| Ours, class word, 1 pass, + DenseCRF | 0.6432 | 0.674 | 0.866 |
| Ours, optimized token, 1 pass | 0.6419 | 0.689 | 0.907 |
| Ours, optimized token, 1 pass, + DenseCRF | **0.6612** | 0.711 | 0.901 |
| Ours, optimized token, 16 layers / all heads, 1 pass, + DenseCRF | 0.6552 | 0.707 | 0.898 |

### Paired per-image IoU differences (bootstrap over images, 20,000 resamples, seed 20261013)

| Dataset | A | B | Mean Δ | 95% CI | A / B better |
|---|---|---|---:|---|---|
| VOC | token, 1 pass | class word, 1 pass | +0.042 | [+0.031, +0.054] | 303 / 187 |
| VOC | class word + CRF | class word | +0.019 | [+0.015, +0.022] | 363 / 127 |
| VOC | token + CRF | token | +0.022 | [+0.019, +0.026] | 374 / 116 |
| VOC | 16 layers / all heads | top-4, 12 layers (token, 1 pass) | +0.002 | [−0.001, +0.004] | 250 / 240 |
| VOC | token, 1 pass | released token-optimized OVAM | +0.001 | [−0.010, +0.011] | 203 / 287 |
| VOC | token + CRF, 1 pass | released token-optimized OVAM | +0.023 | [+0.012, +0.034] | 291 / 199 |
| COCO | token, 1 pass | class word, 1 pass | +0.013 | [+0.001, +0.025] | 327 / 380 |
| COCO | token + CRF | token | +0.035 | [+0.031, +0.040] | 564 / 142 |
| COCO | 16 layers / all heads | top-4, 12 layers (token, 1 pass) | −0.004 | [−0.007, −0.002] | 303 / 403 |
| COCO | token + CRF, 1 pass | implemented OVAM | +0.090 | [+0.072, +0.108] | 473 / 233 |

Full tables, 4-pass rows, per-class IoU and the COCO-Cap caption-word split
are in `outputs/segmentation/narmyn_token_crf/report_voc_sim.md` and
`report_coco_cap.md`.

## Findings

1. **The optimized token is the larger change on VOC-Sim** (+0.042 per image),
   mostly through recall (0.862 → 0.943). With DenseCRF the method exceeds the
   released token-optimized OVAM masks on the same 490 images.
2. **DenseCRF helps consistently** on both datasets (+0.02 to +0.035 per image).
3. **With the optimized token, CA layer/head selection matters little**: +0.002
   (not significant) on VOC-Sim, −0.004 on COCO-Cap.
4. **On COCO-Cap the token helps much less** (+0.013, and on fewer than half of
   the images), and the best result (0.661) is about 3 points below the 69.2
   OVAM reports. Why the tokens transfer less well to caption prompts in this
   pipeline is open; see `docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md` (S3, S4).

## Caveats

- The released VOC-Sim masks were made on the authors' renders; all other rows
  use our regenerated images, scored on the same annotations.
- The comparison with the paper's COCO-Cap number is mIoU against mIoU, not per
  image.
- Shards ran on V100 (`gen`: torch 2.4.1+cu121) and A100 (`trdp`: torch
  2.11.0+cu130) nodes; see `run_environment.csv` in each results folder.
  Re-running one COCO image on the other GPU type changed its per-image IoUs
  by at most 0.006.
- A re-implementation of token-optimized OVAM itself (α = 0.95, τ = 0.80) did
  not reproduce the released masks (about 0.2 IoU) and was dropped; the
  released masks are the reference.

## Reproduce

From the repository root, with an environment whose torch build runs on the
node and that provides SimpleCRF (`requirements.txt` pins `SimpleCRF==0.2.1.1`;
on numpy ≥ 2 its sources need `NPY_IN_ARRAY` replaced by `NPY_ARRAY_IN_ARRAY`
before building, or pass `--crf-site` to a directory with a built `denseCRF`):

```bash
# only if the OVAM annotations are not already under datasets/
python scripts/narmyn/attention/fetch_ovam_release_files.py

sbatch jobs/narmyn/attention/run_token_crf.slurm --dataset voc
sbatch jobs/narmyn/attention/run_token_crf.slurm --dataset coco

python scripts/narmyn/attention/summarize_token_crf.py voc_sim
python scripts/narmyn/attention/summarize_token_crf.py coco_cap
```

About 8–15 s per image per GPU. The offline 49-image pilot that preceded this
run (aeroplane and bicycle only, from saved disk captures, including a
threshold sweep and OVAM-style CA aggregation) is
`scripts/narmyn/attention/pilot_offline_token_seed.py`, with its output in
`outputs/segmentation/narmyn_token_crf/pilot_offline_49/`.

## Files

| Path | Content |
|---|---|
| `scripts/narmyn/attention/run_token_crf.py` | live runner (one class shard per call) |
| `scripts/narmyn/attention/summarize_token_crf.py` | tables and paired bootstrap tests |
| `scripts/narmyn/attention/fetch_ovam_release_files.py` | range-fetches OVAM release members |
| `scripts/narmyn/attention/pilot_offline_token_seed.py` | offline 49-image pilot |
| `jobs/narmyn/attention/run_token_crf.slurm` | 20-task array job |
| `configs/ovam/` | OVAM tokens and licence |
| `outputs/segmentation/narmyn_token_crf/{voc_sim,coco_cap}/` | per-image JSONL, COMPLETE markers, run environments |
| `outputs/segmentation/narmyn_token_crf/report_*.md` | generated summaries |
