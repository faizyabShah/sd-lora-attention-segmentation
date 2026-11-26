# Pre-registration: are correct object pixels more stable than BRACE's bleed under (A) image transformations and (B) noise realizations? (two diagnostics, SD1.5)

Written 2026-11-16, before any capture. Code: `scripts/equiv_noise.py`.

**Scope:**
- SD1.5 only. BRACE is frozen; this changes nothing about it.
- Both are **diagnostics**: no mask-refinement method is built unless one passes.
- Ground truth only scores.
- Nothing is tuned after the run: not the transforms, t, the seeds, the statistics, the cells or the bars.

**Relation to earlier work:**
- The re-noising **ensemble mean** (operator batch C) was null.
- What is tested here is pixelwise **variability and consistency**, not the mean, which enters the baseline.

## Images

- **Set.** All 150 VOC-Sim and 150 COCO-Cap images of the stored rich capture (`CACHE/ana_dump`).
- **Per image, from the capture:** the generated image, the ground truth, the live A3 seed, the live V3 map and the class-token positions (`tokens`).

## Capture protocol (fixed; GPU)

**Per condition c**, i.e. one version of the image plus one noise draw:
1. **Encode.** The image goes through the VAE, giving the latent z₀ (distribution mean).
2. **Noise.** For each of BRACE's 11 self-attention calls, take timestep t_i = the scheduler timesteps at PNDM steps 10–20 of 30. One noise tensor ε_c, used for all 11, gives z_t = √ᾱ z₀ + √(1−ᾱ) ε_c.
3. **UNet call.** One conditional call per t_i with the generation prompt and no CFG. This is the same single-call protocol as the earlier re-noising ensemble.
4. **Recorded per condition:**
   - **CA map** (hierarchy-free; **primary**): the class-token cross-attention probability, summed over the class-word token pieces and averaged over heads. Each layer's map is bilinearly resampled to 64 × 64, then averaged with equal weights over all cross-attention layers and the 11 calls.
   - **V3 map** (secondary): the frozen V3 hierarchy, run with the self-attention (q, k) of the 11 calls at the three frozen modules. Its seed is the live A3 seed, transformed with the image.
5. **Alignment.** Both maps are mapped back to the original coordinates. Cells that have no valid preimage are marked invalid.

**Conditions (9 per image):**
- **id_s0 … id_s5:** the original image, with noise seeds 0–5. These are Experiment B.
- **flip:** horizontal flip of the image, the noise of seed 0 and the seed.
- **shift16:** the image shifted by +16 px in both axes (reflect padding). The seed is shifted by 2 cells, and the noise of seed 0 is rolled by 2 cells. Invalid: the last 2 rows and columns after mapping back.
- **shift4:** the image shifted by +4 px in both axes (reflect padding), the seed shifted by 0.5 cell (bilinear), and the noise of seed 0 unshifted. Invalid: the last row and column.
- flip, shift16 and shift4 are Experiment A. Each is compared with id_s0, which uses the same noise draw.

**Geometry.** Each single call sees the noised latent of the unchanged image, so no reconstruction can alter object geometry. ChatGPT's geometry gate is therefore satisfied by construction. As a check, the stored image's decode is reported.

## Cells and task (scoring; CPU)

These are the same definitions as `runs/08_feat_bleed`:
- **BRACE mask M:** stored live V3, min–max, bilinear to 512, τ 0.55, R2. A cell is inside M if its pooled M is ≥ 0.5.
- **Core F:** cells with min–max live V3 ≥ 0.9 (top 5% if empty). **Clear background B:** cells with min–max live V3 ≤ 0.1, outside M.
- **Q:** cells inside M, not in F, whose ground-truth label is pure (pooled ≥ 0.75 object, ≤ 0.25 background). Label 1 = bleed (ground-truth background).

## Features

All maps are min–max normalised per condition before any statistic.

- **Baseline** (as in `runs/08_feat_bleed`): Lab distance to the means of F and B; intervening contour; distance to F; BRACE confidence (live V3); depth inside M; mask area; the matched within-image RGB classifier.
- **Mean maps**, added to the baseline for both tests: the mean over id_s0–s5 of the CA map and of the V3 map.
- **Experiment A (equivariance):**
  - **primary:** e_CA(i) = the mean over {flip, shift16, shift4} of |CA_T(i) − CA_id0(i)|, over the transforms valid at cell i;
  - **secondary:** e_V3, the same statistic on V3 maps.
- **Experiment B (noise variability):**
  - **primary:** v_CA(i) = the standard deviation over id_s0–s5 of CA(i);
  - **secondary:** v_V3, the same on V3 maps.

## Gates (fixed; decisive set COCO-Cap, where bleed is the main failure)

- **Model:** out-of-fold logistic regression, folds crc32(image_id) % 5, L2, standardised features.
- **Comparison:** pooled over the cells of Q, (baseline + mean maps) against (baseline + mean maps + the experiment's primary statistic).
- **Each experiment passes only if** ΔAUROC ≥ **+0.05** with image-bootstrap 95% CI lower bound **> 0** (2000 resamples).
- **Verdict.** A and B are judged independently. A pass only licenses a new pre-registered refinement experiment on new images.

## Reported, not deciding

- Single-feature AUROCs.
- Gates with the secondary V3 statistics, and per transform for A.
- The VOC-Sim gates.
- **Augmentation baselines** (mean per-image IoU at τ 0.55 + R2, against BRACE's mask):
  - **flip-average:** the mean of id_s0 and flip V3 maps;
  - **noise-average:** the mean of the 6 id V3 maps;
  - **reference:** the re-noised id_s0 V3 alone.
- The CA-map best-threshold IoU, as a sanity check of the protocol.
- The mean per-cell e and v, inside and outside M.

## Smoke test (2026-11-16, 2 + 2 images, before the full run)

- **Bug fixed.** The smoke test found a code bug: the attention processors were not restored after the first image, because diffusers empties the dict it is given. Fixed by passing a copy.
- **No design change.**
- **Timing:** 11.5 s per image.
- **Protocol sanity check:** the mean CA map separates object from background at cell AUROC 0.98–0.99.
- **Disclosure.** The smoke statistics (`result_smoke.md`) are not evidence.
