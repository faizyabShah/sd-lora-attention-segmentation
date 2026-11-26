# Equivariance (A) and noise variability (B) of SD1.5 attention at BRACE's bleed (diagnostics; `PREREG.md`)

Stored rich capture images; 9 conditions per image (6 noise seeds; flip, 16 px and 4 px shifts); BRACE's 11 attention timesteps, one conditional call each. Decisive: COCO-Cap, primary CA statistics.

## Verdicts (COCO-Cap, primary CA statistic): A equivariance **CLOSE**, B noise variability **CLOSE**

Bar: Δ OOF AUROC ≥ +0.05 with CI > 0 over baseline (RGB, contour, distance, BRACE confidence, depth, area, RGB classifier, mean CA and mean V3 maps).

| set | images | cells (bleed) | baseline | A: + e_CA Δ (CI) | B: + v_CA Δ (CI) | A: + e_V3 Δ | B: + v_V3 Δ | A per transform (flip / shift16 / shift4) |
|---|---|---|---:|---|---|---:|---:|---|
| VOC-Sim | 150 (0 excl.) | 138,987 (14.7%) | 0.778 | +0.0242 [+0.0100, +0.0371] | +0.0331 [+0.0122, +0.0528] | +0.0071 | -0.0021 | +0.0142 / -0.0005 / +0.0190 |
| COCO-Cap | 150 (0 excl.) | 105,505 (30.5%) | 0.748 | +0.0063 [-0.0008, +0.0137] | +0.0252 [+0.0047, +0.0444] | +0.0001 | -0.0031 | +0.0025 / -0.0006 / +0.0073 |

## Single-feature AUROC for bleed vs correct (higher = feature larger on bleed)

| feature | VOC-Sim | COCO-Cap |
|---|---:|---:|
| dLab_F | 0.530 | 0.515 |
| dLab_B | 0.547 | 0.471 |
| contour | 0.676 | 0.550 |
| dist_F | 0.565 | 0.643 |
| conf | 0.228 | 0.315 |
| depth | 0.388 | 0.465 |
| area | 0.517 | 0.510 |
| s_rgb | 0.487 | 0.452 |
| mean_ca | 0.396 | 0.278 |
| mean_v3 | 0.226 | 0.308 |
| e_ca | 0.614 | 0.486 |
| e_v3 | 0.680 | 0.575 |
| v_ca | 0.617 | 0.559 |
| v_v3 | 0.617 | 0.590 |
| e_ca_flip | 0.584 | 0.485 |
| e_ca_shift16 | 0.535 | 0.455 |
| e_ca_shift4 | 0.595 | 0.528 |

## Augmentation baselines (mean per-image IoU, τ 0.55 + R2) and protocol checks

| set | BRACE (live) | re-noised id_s0 V3 | flip-average | noise-average (6) | mean CA map pixel AUROC (cells) | mean e_CA in / out of M | mean v_CA in / out of M | s / image |
|---|---:|---:|---:|---:|---:|---|---|---:|
| VOC-Sim | 0.8254 | 0.8244 | 0.8250 | 0.8257 | 0.961 | 0.051 / 0.030 | 0.053 / 0.032 | 23 |
| COCO-Cap | 0.7099 | 0.6946 | 0.6913 | 0.6972 | 0.937 | 0.055 / 0.032 | 0.057 / 0.035 | 23 |

Diagnostics only; GT scores only. Augmentation rows are descriptive (not a selection).
