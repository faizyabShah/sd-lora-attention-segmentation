# Exploratory: does noise variability separate BRACE's bleed at a specific noise level? (follow-up to `runs/09_equiv_noise`)

Written 2026-11-16, **after** the pre-registered test B in `runs/09_equiv_noise` failed: COCO-Cap +0.025, CI +0.005…+0.044, bar +0.05. Code: `scripts/noise_level.py`.

**Status.** This is an exploratory search, not a confirmatory test.
- No result here changes the closed verdict of `runs/09_equiv_noise`.
- A level that reaches the bar only licenses a new pre-registered test of that single level on images not used here.

**Why.** Run 09 averaged the cross-attention (CA) maps over BRACE's 11 timesteps, t = 661…331, where the image's share of the noisy latent √ᾱ runs 0.33…0.73. A level-specific effect, especially at heavier noise than BRACE uses, could be diluted or untested.

## Protocol (fixed before capture)

- **Images:** the same 150 COCO-Cap screening images as run 09 (in-sample, disclosed). There is no VOC-Sim capture.
- **Levels:** t ∈ {101, 201, 331, 430, 529, 661, 761, 861, 941}. This is 9 levels, from light noise to almost pure noise, and includes 4 of BRACE's timesteps.
- **Per level and per noise seed 0–5:**
  - z_t = √ᾱ z₀ + √(1−ᾱ) ε_s, where the noise draw is the same as in run 09.
  - One conditional UNet call (no CFG) with the generation prompt.
  - The CA map is computed exactly as in run 09, but from **this single call**, with no averaging over timesteps.
- **Per-level statistics:**
  - mean_ca_t is the mean over seeds of the per-seed min–max CA map;
  - v_ca_t is their standard deviation over seeds.

## Scoring

- **Cells, labels and folds:** the same Q cells, bleed labels and out-of-fold logistic folds as run 09.
- **Baseline per level:** run 09's baseline (RGB, contour, distance, BRACE confidence, depth, area, RGB classifier, run 09 mean CA and mean V3) **plus mean_ca_t**. Including mean_ca_t stops level-specific variability from acting as a proxy for a level-specific mean.
- **Test per level:** that baseline + v_ca_t.
- **Reported per level:** Δ OOF AUROC with image-bootstrap 95% CI (2000 resamples); the single-feature AUROC of v_ca_t; and the object/background cell AUROC of mean_ca_t, to show where the CA signal itself collapses.
- **Bar for "promising":** the run 09 bar, ≥ +0.05 with CI lower bound > 0.
- **Multiple comparisons.** With 9 levels, a single crossing is weak evidence and is reported as such.

## Smoke test (2 images, before the full run)

- No bugs and no design change.
- Timing: 5 s per image.
- The smoke statistics are not evidence.
