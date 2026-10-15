# Cross-validated selection

490 images, 20 classes, 13 configurations; shards complete: 20/20. Selection at threshold 0.45.

| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |
|---|---|---|---:|---|---|---|---:|---|:---:|
| SA-C | `C/reference` | `C/L` (5/5) | +0.0330 | `C/L` | fold vote and OOF guard passed | 0.7887 / 0.7556 | +0.0330 | +0.0417 [+0.0365, +0.0468] | no |

## SA-C

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `C/L` | promote | 0.7892 / 0.7580 | 0.7892 / 0.7465 |
| 1 | `C/L` | promote | 0.7825 / 0.7490 | 0.8116 / 0.7780 |
| 2 | `C/L` | promote | 0.7899 / 0.7571 | 0.7869 / 0.7518 |
| 3 | `C/L` | promote | 0.7908 / 0.7578 | 0.7825 / 0.7460 |
| 4 | `C/L` | promote | 0.7910 / 0.7559 | 0.7757 / 0.7484 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `C/L` | 0.7691 | 0.7887 | 0.7532 | 0.9560 |
| `C/L+H` | 0.7637 | 0.7871 | 0.7554 | 0.9587 |
| `C/T+L+H` | 0.7658 | 0.7825 | 0.7330 | 0.9514 |
| `C/T+L` | 0.7704 | 0.7820 | 0.7260 | 0.9474 |
| `C/T+R` | 0.7659 | 0.7749 | 0.7244 | 0.9517 |
| `C/T+H` | 0.7534 | 0.7732 | 0.7412 | 0.9563 |
| `C/T` | 0.7465 | 0.7710 | 0.7457 | 0.9551 |
| `C/T+R+H` | 0.7669 | 0.7692 | 0.7098 | 0.9512 |
| `C/R` | 0.7414 | 0.7676 | 0.7399 | 0.9542 |
| `C/R+H` | 0.7389 | 0.7656 | 0.7343 | 0.9534 |

## SA-C component effects (full 490, in-sample diagnostics)

| Component comparison | With | Without | Δ mIoU@0.45 |
|---|---|---|---:|
| T within T+L+H | `C/T+L+H` | `C/L+H` | -0.0046 |
| L within T+L+H | `C/T+L+H` | `C/T+H` | +0.0093 |
| H within T+L+H | `C/T+L+H` | `C/T+L` | +0.0005 |
| R within T+R+H | `C/T+R+H` | `C/T+H` | -0.0040 |

## Per-class IoU: selected `C/L` against `C/reference`

| Class | Reference | Selected | Δ |
|---|---:|---:|---:|
| person | 0.7218 | 0.6664 | -0.0554 |
| tv | 0.7953 | 0.7768 | -0.0184 |
| potted-plant | 0.7064 | 0.7091 | +0.0027 |
| sheep | 0.7564 | 0.7660 | +0.0096 |
| bus | 0.9064 | 0.9184 | +0.0120 |
| cat | 0.7688 | 0.7816 | +0.0127 |
| cow | 0.8055 | 0.8184 | +0.0129 |
| dining-table | 0.4511 | 0.4677 | +0.0166 |
| sofa | 0.9230 | 0.9447 | +0.0217 |
| train | 0.6470 | 0.6697 | +0.0227 |
| car | 0.8152 | 0.8420 | +0.0268 |
| dog | 0.8692 | 0.8960 | +0.0268 |
| motorbike | 0.8157 | 0.8527 | +0.0370 |
| horse | 0.8300 | 0.8696 | +0.0396 |
| bottle | 0.8126 | 0.8665 | +0.0539 |
| bird | 0.7859 | 0.8531 | +0.0673 |
| boat | 0.7638 | 0.8318 | +0.0680 |
| bicycle | 0.6788 | 0.7471 | +0.0683 |
| aeroplane | 0.7309 | 0.8450 | +0.1141 |
| chair | 0.5289 | 0.6503 | +0.1214 |
