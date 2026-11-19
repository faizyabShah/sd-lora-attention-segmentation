# JC #3 remaining comparison rows, paired with BRACE (`PREREG.md`)

Side N, exploratory. One SD 1.5 generation per item (wald V100, runs/20 corrected prompts, manifest seeds), all generation-time methods read from that trajectory; SCLIP on the generated image. BRACE = the N-side reproduction of runs/20 (`comparisons/official/`). Class-macro mIoU; paired Δ = method − BRACE, 2000-sample image bootstrap, 95% interval. *VOC-sel.* = 5-fold OOF threshold on VOC-Sim, full-VOC-Sim-best threshold applied unchanged to COCO-Cap. DenseCRF rows use a 7-step threshold window around the raw VOC-Sim optimum (see `score_f.py`). DAAM and SeeDiff need the class word in the prompt: their COCO-Cap rows and Δ use that subset, with BRACE on the same subset.

## Data checks

- VOC-Sim: 490 items; generated image bit-identical to the BRACE reproduction's: 490/490; items per method: ovam_w 490, ovam_tf 490, daam 490, seediff 490, sclip 490

- COCO-Cap: 707 items; generated image bit-identical to the BRACE reproduction's: 707/707; items per method: ovam_w 707, ovam_tf 707, daam 437, seediff 437, sclip 707


## Rows (VOC-Sim / COCO-Cap)

| Method | VOC-sel., raw | VOC-sel., DenseCRF | fixed 0.5, raw | fixed 0.5, DenseCRF | own rule, raw | own rule, DenseCRF | τ (raw / CRF) |
|---|---|---|---|---|---|---|---|
| BRACE (reproduction) | 0.818 / 0.692 | 0.831 / 0.708 | | | | | 0.550 / 0.550 |
| OVAM, class word | 0.677 / 0.532 | 0.703 / 0.568 | 0.540 / 0.481 | 0.544 / 0.496 | 0.671 / 0.519 | 0.696 / 0.546 | 0.325 / 0.325 |
| OVAM, released optimized token (few-shot, not training-free) | 0.802 / 0.629 | 0.828 / 0.664 | 0.780 / 0.636 | 0.803 / 0.664 | 0.679 / 0.488 | 0.700 / 0.507 | 0.425 / 0.425 |
| DAAM | 0.693 / 0.597 | 0.727 / 0.644 | 0.495 / 0.466 | 0.489 / 0.465 | 0.640 / 0.579 | 0.666 / 0.610 | 0.300 / 0.300 |
| SeeDiff | 0.692 / 0.634 | 0.723 / 0.665 | 0.669 / 0.641 | 0.692 / 0.670 | 0.742 / 0.676 | 0.742 / 0.679 | 0.350 / 0.325 |
| SCLIP ViT-B/16 (image only) | 0.718 / 0.606 | 0.764 / 0.652 | 0.654 / 0.496 | 0.689 / 0.516 | 0.719 / 0.583 | 0.760 / 0.620 | 0.200 / 0.200 |

## Paired Δ, method − BRACE, both at VOC-sel. (primary)

| Method | VOC-Sim raw | COCO-Cap raw | VOC-Sim DenseCRF | COCO-Cap DenseCRF |
|---|---|---|---|---|
| OVAM, class word | -0.141 [-0.154, -0.127] | -0.160 [-0.179, -0.142] | -0.128 [-0.143, -0.114] | -0.140 [-0.160, -0.120] |
| OVAM, released optimized token (few-shot, not training-free) | -0.016 [-0.028, -0.004] | -0.063 [-0.081, -0.047] | -0.003 [-0.016, +0.010] | -0.044 [-0.063, -0.026] |
| DAAM | -0.125 [-0.138, -0.113] | -0.100 [-0.122, -0.076] | -0.104 [-0.119, -0.091] | -0.067 [-0.093, -0.039] |
| SeeDiff | -0.125 [-0.138, -0.113] | -0.062 [-0.085, -0.037] | -0.108 [-0.122, -0.094] | -0.046 [-0.070, -0.018] |
| SCLIP ViT-B/16 (image only) | -0.099 [-0.115, -0.085] | -0.087 [-0.112, -0.064] | -0.067 [-0.084, -0.052] | -0.057 [-0.084, -0.031] |

BRACE on the class-word-in-prompt subsets (VOC-sel.; raw / DenseCRF): daam COCO-Cap 0.696 / 0.711; seediff COCO-Cap 0.696 / 0.711.

## Reading

- Every row is below BRACE on both benchmarks with DenseCRF at the VOC-selected threshold, except the few-shot
  OVAM optimized token on VOC-Sim, which ties (−0.003 [−0.016, +0.010]); on COCO-Cap it is −0.044 [−0.063, −0.026].
- Training-free class-word rows: SeeDiff −0.108 / −0.046, DAAM −0.104 / −0.067, OVAM −0.128 / −0.140
  (VOC-Sim / COCO-Cap, DenseCRF). SCLIP alone (CLIP, no diffusion): −0.067 / −0.057.
- With their own released binarisation, the methods move little or get worse, except SeeDiff (PAMR): 0.742 / 0.679,
  still −0.089 / −0.032 below BRACE's VOC-selected DenseCRF row (own-rule Δ not bootstrapped).
- Check against print: OVAM's own rule (fused > 0.40 + DenseCRF) gives 0.696 / 0.546 here vs 0.704 / 0.582
  published by OVAM (its Table 1; regenerated images, corrected prompts, so not identical by design).

## SeeDiff: go

Public code exists and runs: https://github.com/BAIKLAB/SeeDiff (commit 28d26f4; AAAI 2025, arXiv 2507.19808).
Its bulk script is broken (it saves the image as the mask), so the demo path (`seediff_app.py`,
`save_cross_attention`) is used, with its `aggregate_attention` and `PAMR` unchanged (PREREG Addendum 1).
SeeDiff publishes no VOC-Sim or COCO-Cap number (it reports downstream Mask2Former results on real images),
so it has no published-block row.

## Deviations (all in PREREG.md addenda, written before scoring)

1. DenseCRF on a 7-step threshold window around each method's raw VOC-Sim optimum, plus 0.5; COCO-Cap at the
   selected index and 0.5 only (DenseCRF R2 costs about 4.7 s per call on one core). Raw rows use the full grid.
2. DAAM's aggregation is reimplemented in the shared pass (its package pins diffusers 0.21.2); SeeDiff's
   AttentionStore is filled in the same pass; PAMR runs on the GPU.
3. OVAM optimized token = [text-encoder <SoT>, released `tokens_VOC.npy` token], softmax over the two (OVAM's
   `token_softmax`); OVAM VOC-Sim index `len(prompt.split())`, COCO-Cap −2 (as OVAM's own scripts).
4. DAAM and SeeDiff need the class word in the prompt: COCO-Cap rows on the 437 items where DAAM's tokenizer
   search finds it (BRACE on the same 437: 0.696 raw / 0.711 DenseCRF). SeeDiff's own match finds one more item,
   the only "potted plant" one (a 1-item class under class-macro mIoU), which is excluded to keep the subsets equal.
5. SeeDiff: SD 1.5, 30 steps, CFG 7.5 (benchmark generation) instead of SD v1-4, 50 steps, CFG 7.
6. SCLIP ran on the BRACE reproduction's stored images (CPU, `goca` env), because side F's SCLIP port does not run
   under transformers 5. The CLIP B/16 checkpoint was converted from `pytorch_model.bin` to safetensors in the HF
   cache (same tensors), because the `gen` env's transformers refuses `.bin` under torch 2.4.

## Runtime

- Shared capture (OVAM ×2, DAAM, SeeDiff from one generation): 4 jobs on wald (V100-PCIE-16GB), 6.6 GPU-hours in
  total; median 12.8 s (VOC-Sim) / 14.7 s (COCO-Cap) per image for generation plus all attention maps, plus a
  median 5.7 / 6.1 s for SeeDiff's aggregation and PAMR; peak 6.6 GiB.
- SCLIP: about 1 s per image on 8 CPU cores (4 × 4.4 min).
- Scoring: about 2 × 32 cores for 9 min (raw), 42 min (VOC-Sim DenseCRF) and 13 min (COCO-Cap DenseCRF).

## Files

`comparisons/official_f_rows/`: `PREREG.md`, `capture_f.py`, `capture.slurm`, `sclip_f.py`, `sclip.slurm`,
`score_f.py`, `score.slurm`, `report_f.py`, `result.md`, `summary.json`. Outputs (git-ignored):
`~/TRDP/external/n_side/official_f_rows/{voc,coco,sclip,scores,smoke,logs}`, `taus_voc.json`, `taus_coco.json`,
`src/` (DAAM 0.2.0, SeeDiff and OVAM sources), `deps/` (opencv-python-headless 4.10).
