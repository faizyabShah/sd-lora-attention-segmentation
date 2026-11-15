# Does UNet feature content separate BRACE's bleed from the object? (diagnostic; `PREREG.md`)

Stored rich capture, 150/150 screening images. Features `feat/up2/s15`; cells inside BRACE + R2 not in the core F. Decisive: COCO-Cap, gates 1 and 2.

## Verdict: **CLOSE** (gate 1 fail, gate 2 fail)

## Gate 1: OOF AUROC for bleed vs correct cells inside BRACE (baseline = RGB, contour, distance, BRACE confidence, depth, area, matched RGB classifier)

| set | images (excl.) | cells (bleed share) | baseline | + features | Δ | CI | + s_proto only Δ | + s_feat only Δ |
|---|---|---|---:|---:|---:|---|---:|---:|
| VOC-Sim | 8 (0) | 4,219 (18.7%) | 0.467 | 0.448 | -0.0186 | [-0.0679, +0.0044] | -0.0300 | +0.0069 |
| COCO-Cap | 8 (0) | 9,493 (19.1%) | 0.573 | 0.579 | +0.0065 | [-0.0564, +0.0893] | +0.0300 | -0.0276 |

## Gate 2: count-matched label-free corrections, mean per-image IoU gain over BRACE + R2 (95% CI)

| set | features (s_feat < 0.5) | RGB classifier | BRACE confidence | cells removed / image | feature wins |
|---|---|---|---|---:|---:|
| VOC-Sim | -0.1187 [-0.2188, -0.0308] | -0.1217 [-0.2200, -0.0350] | -0.1159 [-0.2171, -0.0274] | 134.1 | 1/8 |
| COCO-Cap | -0.1102 [-0.2139, -0.0256] | -0.1006 [-0.1879, -0.0360] | -0.0999 [-0.2272, -0.0064] | 365.0 | 2/8 |

## Reported, not deciding

**Single-feature AUROC (oriented so that higher = more bleed-like):**

| feature | VOC-Sim | COCO-Cap |
|---|---:|---:|
| dLab_F | 0.351 | 0.673 |
| dLab_B | 0.464 | 0.440 |
| contour | 0.679 | 0.539 |
| dist_F | 0.599 | 0.507 |
| conf | 0.908 | 0.690 |
| depth | 0.606 | 0.475 |
| area | 0.530 | 0.414 |
| s_rgb | 0.492 | 0.539 |
| s_proto | 0.505 | 0.437 |
| s_feat | 0.464 | 0.521 |
| rowdiv | 0.859 | 0.567 |
| consist | 0.674 | 0.527 |

**Secondary signals added to the baseline (Δ OOF AUROC, CI):**

- rowdiv, VOC-Sim: -0.0025 [-0.0455, +0.0079].
- rowdiv, COCO-Cap: +0.0400 [-0.0836, +0.1319].
- consist, VOC-Sim: -0.0075 [-0.0326, +0.0400].
- consist, COCO-Cap: -0.0658 [-0.1826, -0.0014].

Exploratory mechanism diagnostic. GT only scores; cores, classifiers and the correction are label-free.
