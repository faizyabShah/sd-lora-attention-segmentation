# Does UNet feature content separate BRACE's bleed from the object? (diagnostic; `PREREG.md`)

Stored rich capture, 150/150 screening images. Features `feat/up2/s15`; cells inside BRACE + R2 not in the core F. Decisive: COCO-Cap, gates 1 and 2.

## Verdict: **CLOSE** (gate 1 fail, gate 2 fail)

## Gate 1: OOF AUROC for bleed vs correct cells inside BRACE (baseline = RGB, contour, distance, BRACE confidence, depth, area, matched RGB classifier)

| set | images (excl.) | cells (bleed share) | baseline | + features | Δ | CI | + s_proto only Δ | + s_feat only Δ |
|---|---|---|---:|---:|---:|---|---:|---:|
| VOC-Sim | 150 (0) | 145,030 (15.2%) | 0.770 | 0.762 | -0.0081 | [-0.0164, -0.0002] | -0.0007 | -0.0051 |
| COCO-Cap | 150 (0) | 109,686 (31.1%) | 0.672 | 0.675 | +0.0034 | [-0.0180, +0.0260] | +0.0036 | +0.0034 |

## Gate 2: count-matched label-free corrections, mean per-image IoU gain over BRACE + R2 (95% CI)

| set | features (s_feat < 0.5) | RGB classifier | BRACE confidence | cells removed / image | feature wins |
|---|---|---|---|---:|---:|
| VOC-Sim | -0.1098 [-0.1363, -0.0855] | -0.1249 [-0.1513, -0.1014] | -0.0971 [-0.1236, -0.0729] | 274.2 | 26/150 |
| COCO-Cap | -0.0605 [-0.0855, -0.0367] | -0.0881 [-0.1110, -0.0661] | -0.0460 [-0.0699, -0.0227] | 239.5 | 40/150 |

## Reported, not deciding

**Single-feature AUROC (oriented so that higher = more bleed-like):**

| feature | VOC-Sim | COCO-Cap |
|---|---:|---:|
| dLab_F | 0.532 | 0.514 |
| dLab_B | 0.452 | 0.531 |
| contour | 0.675 | 0.549 |
| dist_F | 0.568 | 0.645 |
| conf | 0.771 | 0.685 |
| depth | 0.605 | 0.529 |
| area | 0.515 | 0.514 |
| s_rgb | 0.511 | 0.548 |
| s_proto | 0.573 | 0.598 |
| s_feat | 0.592 | 0.637 |
| rowdiv | 0.662 | 0.646 |
| consist | 0.591 | 0.506 |

**Secondary signals added to the baseline (Δ OOF AUROC, CI):**

- rowdiv, VOC-Sim: -0.0050 [-0.0096, -0.0007].
- rowdiv, COCO-Cap: +0.0015 [-0.0142, +0.0169].
- consist, VOC-Sim: +0.0171 [+0.0008, +0.0366].
- consist, COCO-Cap: -0.0047 [-0.0107, +0.0015].

Exploratory mechanism diagnostic. GT only scores; cores, classifiers and the correction are label-free.
