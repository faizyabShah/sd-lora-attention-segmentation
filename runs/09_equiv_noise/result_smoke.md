# Equivariance (A) and noise variability (B) of SD1.5 attention at BRACE's bleed (diagnostics; `PREREG.md`)

Stored rich capture images; 9 conditions per image (6 noise seeds; flip, 16 px and 4 px shifts); BRACE's 11 attention timesteps, one conditional call each. Decisive: COCO-Cap, primary CA statistics.

## Verdicts (COCO-Cap, primary CA statistic): A equivariance **CLOSE**, B noise variability **CLOSE**

Bar: Δ OOF AUROC ≥ +0.05 with CI > 0 over baseline (RGB, contour, distance, BRACE confidence, depth, area, RGB classifier, mean CA and mean V3 maps).

| set | images | cells (bleed) | baseline | A: + e_CA Δ (CI) | B: + v_CA Δ (CI) | A: + e_V3 Δ | B: + v_V3 Δ | A per transform (flip / shift16 / shift4) |
|---|---|---|---:|---|---|---:|---:|---|
| VOC-Sim | 2 (0 excl.) | 1,287 (43.4%) | 0.611 | +0.0014 [+0.0000, +0.0014] | +0.0014 [+0.0000, +0.0014] | +0.0014 | +0.0110 | +0.0096 / +0.0000 / +0.0014 |
| COCO-Cap | 2 (0 excl.) | 835 (58.3%) | 0.006 | +0.0007 [-0.0096, +0.0189] | +0.0061 [+0.0012, +0.0069] | +0.0031 | +0.0109 | +0.0025 / +0.0222 / +0.0000 |

## Single-feature AUROC for bleed vs correct (higher = feature larger on bleed)

| feature | VOC-Sim | COCO-Cap |
|---|---:|---:|
| dLab_F | 0.378 | 0.927 |
| dLab_B | 0.623 | 0.257 |
| contour | 0.625 | 0.509 |
| dist_F | 0.701 | 0.947 |
| conf | 0.056 | 0.027 |
| depth | 0.272 | 0.471 |
| area | 0.630 | 0.081 |
| s_rgb | 0.774 | 0.856 |
| mean_ca | 0.029 | 0.098 |
| mean_v3 | 0.048 | 0.010 |
| e_ca | 0.478 | 0.351 |
| e_v3 | 0.977 | 0.889 |
| v_ca | 0.379 | 0.794 |
| v_v3 | 0.642 | 0.669 |
| e_ca_flip | 0.472 | 0.223 |
| e_ca_shift16 | 0.517 | 0.480 |
| e_ca_shift4 | 0.537 | 0.630 |

## Augmentation baselines (mean per-image IoU, τ 0.55 + R2) and protocol checks

| set | BRACE (live) | re-noised id_s0 V3 | flip-average | noise-average (6) | mean CA map pixel AUROC (cells) | mean e_CA in / out of M | mean v_CA in / out of M | s / image |
|---|---:|---:|---:|---:|---:|---|---|---:|
| VOC-Sim | 0.7642 | 0.6696 | 0.7001 | 0.7228 | 0.992 | 0.057 / 0.035 | 0.046 / 0.029 | 12 |
| COCO-Cap | 0.6063 | 0.6256 | 0.6118 | 0.6099 | 0.979 | 0.044 / 0.023 | 0.056 / 0.030 | 11 |

Diagnostics only; GT scores only. Augmentation rows are descriptive (not a selection).
