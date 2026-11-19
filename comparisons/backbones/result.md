# JC #5: fine-only and BRACE in one noising pass on three backbones (side N, exploratory)

Rules fixed in `PREREG.md` before capture. Stored OVAM benchmark images (490 VOC-Sim, 707 COCO-Cap), VAE encode, noise at t = 100 (seed 0), one conditional UNet call, the runs/20 prompts and class-word query. Class-macro mIoU. *VOC-sel.* = VOC-Sim 5-fold OOF threshold; COCO-Cap gets the full-VOC-Sim-best threshold unchanged. Paired Δ: 2000-sample image bootstrap, 95% interval. **This is a different setting from the generation-time BRACE numbers (runs/20, 0.832 / 0.709) and is never mixed with them.**

## Raw (no DenseCRF)

| Backbone | Map | VOC-Sim VOC-sel. | COCO-Cap VOC-sel. | τ (full VOC) | VOC-Sim 0.55 | COCO-Cap 0.55 |
|---|---|---:|---:|---:|---:|---:|
| SD 1.5 | seed alone | 0.704 | 0.533 | 0.250 | 0.389 | 0.311 |
| SD 1.5 | fine-only | 0.784 | 0.612 | 0.400 | 0.710 | 0.611 |
| SD 1.5 | BRACE hierarchy | 0.793 | 0.650 | 0.525 | 0.792 | 0.654 |
| SD 1.5 | GoCA (reference, N run) | 0.738 | 0.610 | 0.750 | 0.694 | 0.484 |
| SD 2.1-base (community copy) | seed alone | 0.675 | 0.517 | 0.250 | 0.321 | 0.251 |
| SD 2.1-base (community copy) | fine-only | 0.749 | 0.599 | 0.400 | 0.674 | 0.608 |
| SD 2.1-base (community copy) | BRACE hierarchy | 0.803 | 0.658 | 0.550 | 0.803 | 0.658 |
| SD 2.1-base (community copy) | GoCA (reference, N run) | 0.719 | 0.586 | 0.575 | 0.720 | 0.576 |
| SDXL base 1.0 | seed alone | 0.705 | 0.450 | 0.325 | 0.441 | 0.321 |
| SDXL base 1.0 | fine-only | 0.761 | 0.506 | 0.350 | 0.624 | 0.439 |
| SDXL base 1.0 | BRACE hierarchy | 0.739 | 0.496 | 0.450 | 0.711 | 0.488 |
| SDXL base 1.0 | GoCA (reference, N run) | 0.787 | 0.602 | 0.475 | 0.771 | 0.606 |

## With DenseCRF R2 (fixed 0.55 only, Amendment A1)

| Backbone | Map | VOC-Sim 0.55 + R2 | COCO-Cap 0.55 + R2 |
|---|---|---:|---:|
| SD 1.5 | fine-only | 0.713 | 0.614 |
| SD 1.5 | BRACE hierarchy | 0.806 | 0.666 |
| SD 1.5 | GoCA + R2, VOC-sel. (τ 0.700; reference) | 0.753 | 0.599 |
| SD 2.1-base (community copy) | fine-only | 0.674 | 0.613 |
| SD 2.1-base (community copy) | BRACE hierarchy | 0.820 | 0.673 |
| SD 2.1-base (community copy) | GoCA + R2, VOC-sel. (τ 0.550; reference) | 0.740 | 0.597 |
| SDXL base 1.0 | fine-only | 0.618 | 0.436 |
| SDXL base 1.0 | BRACE hierarchy | 0.721 | 0.494 |
| SDXL base 1.0 | GoCA + R2, VOC-sel. (τ 0.450; reference) | 0.797 | 0.606 |

## Paired differences

| Backbone | Comparison | Rule | Δ VOC-Sim | Δ COCO-Cap |
|---|---|---|---:|---:|
| SD 1.5 | BRACE − fine-only | raw, VOC-sel. | +0.009 [+0.003, +0.016] | +0.038 [+0.029, +0.047] |
| SD 1.5 | BRACE − fine-only | raw, fixed 0.55 | +0.082 [+0.068, +0.096] | +0.043 [+0.026, +0.061] |
| SD 1.5 | BRACE − fine-only | 0.55 + DenseCRF | +0.094 [+0.079, +0.109] | +0.052 [+0.034, +0.071] |
| SD 1.5 | BRACE − GoCA | raw, VOC-sel. (both) | +0.055 [+0.039, +0.073] | +0.040 [+0.018, +0.062] |
| SD 1.5 | BRACE − GoCA | DenseCRF; BRACE 0.55, GoCA VOC-sel. (mixed rule) | +0.053 [+0.035, +0.073] | +0.067 [+0.046, +0.092] |
| SD 2.1-base (community copy) | BRACE − fine-only | raw, VOC-sel. | +0.054 [+0.044, +0.064] | +0.060 [+0.049, +0.071] |
| SD 2.1-base (community copy) | BRACE − fine-only | raw, fixed 0.55 | +0.129 [+0.109, +0.147] | +0.050 [+0.033, +0.066] |
| SD 2.1-base (community copy) | BRACE − fine-only | 0.55 + DenseCRF | +0.146 [+0.124, +0.166] | +0.060 [+0.042, +0.079] |
| SD 2.1-base (community copy) | BRACE − GoCA | raw, VOC-sel. (both) | +0.083 [+0.068, +0.099] | +0.072 [+0.052, +0.093] |
| SD 2.1-base (community copy) | BRACE − GoCA | DenseCRF; BRACE 0.55, GoCA VOC-sel. (mixed rule) | +0.079 [+0.063, +0.095] | +0.076 [+0.054, +0.097] |
| SDXL base 1.0 | BRACE − fine-only | raw, VOC-sel. | -0.022 [-0.030, -0.015] | -0.011 [-0.017, -0.005] |
| SDXL base 1.0 | BRACE − fine-only | raw, fixed 0.55 | +0.088 [+0.074, +0.100] | +0.049 [+0.036, +0.061] |
| SDXL base 1.0 | BRACE − fine-only | 0.55 + DenseCRF | +0.103 [+0.088, +0.118] | +0.058 [+0.045, +0.071] |
| SDXL base 1.0 | BRACE − GoCA | raw, VOC-sel. (both) | -0.048 [-0.060, -0.037] | -0.107 [-0.123, -0.090] |
| SDXL base 1.0 | BRACE − GoCA | DenseCRF; BRACE 0.55, GoCA VOC-sel. (mixed rule) | -0.076 [-0.092, -0.060] | -0.112 [-0.132, -0.089] |

BRACE − GoCA pairs use the same stored images and masks but different captures (GoCA's own code and noise draw); they are same-image, not same-capture, pairs.

## Runtime and memory per image (capture GPU)

| Backbone | GPU | UNet pass + seed, median s | fine-only step, median s | hierarchy step, median s | peak GPU memory, GiB (max) |
|---|---|---:|---:|---:|---:|
| SD 1.5 | NVIDIA A100-PCIE-40GB | 0.21 | 0.001 | 0.001 | 5.1 |
| SD 2.1-base (community copy) | NVIDIA A100-PCIE-40GB | 0.19 | 0.001 | 0.001 | 5.5 |
| SDXL base 1.0 | NVIDIA A100-PCIE-40GB | 1.49 | 0.001 | 0.001 | 15.6 |

The UNet pass + seed is shared: fine-only and BRACE come from the same call. Timings exclude model loading and DenseCRF.

## Checks

- SD 1.5, VOC-Sim: 490 items; ground-truth masks identical to the GoCA run's: True
- SD 1.5, COCO-Cap: 707 items; ground-truth masks identical to the GoCA run's: True
- SD 2.1-base (community copy), VOC-Sim: 490 items; ground-truth masks identical to the GoCA run's: True
- SD 2.1-base (community copy), COCO-Cap: 707 items; ground-truth masks identical to the GoCA run's: True
- SDXL base 1.0, VOC-Sim: 490 items; ground-truth masks identical to the GoCA run's: True
- SDXL base 1.0, COCO-Cap: 707 items; ground-truth masks identical to the GoCA run's: True

## Deviations and caveats

- **Amendment A1 (before scoring):** DenseCRF only at the frozen threshold 0.55 (4.5 s per call; the full grid
  would take about 240 CPU-hours). There are no DenseCRF VOC-sel. rows for BRACE or fine-only.
- **SD 2.1-base** is the community copy `sd2-community/stable-diffusion-2-1-base` (hash-matched to a clone of
  Stability's repo), loaded from the cached snapshot by path (revision 4e63672c; the cache has no `refs/main`).
  The user approved it on 2026-11-19; **side F has not confirmed it yet.**
- **SDXL mapping** (fixed in PREREG): all 70 CA modules, GoCA's SDXL reference layer, SA grids 32 → 64, one
  capped step. Under VOC-sel. the SDXL hierarchy loses to fine-only (−0.022 / −0.011); at the frozen 0.55 it wins.
- **Single-call re-implementation:** checked against `sg_combined_v2.aggregate` (SD 1.5, 5 images, max abs
  difference 4.8e-7 after min-max). The A3 seed is a re-implementation of `SG24Recorder` for one call; it is
  not checked against a generation-time capture, because the settings differ.
- **GoCA rows** come from the N-side GoCA runs (own code, own noise draw, t = 100): same images and masks, not
  the same capture.
- All rows are exploratory. Capture on one A100-PCIE-40GB (renyi), job 253899; scoring jobs 253913 and 253914.
