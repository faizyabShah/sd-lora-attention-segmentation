# Cross-validated selection

490 images, 20 classes, 1279 configurations; shards complete: 20/20. Selection at threshold 0.45.

| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |
|---|---|---|---:|---|---|---|---:|---|:---:|
| SA-T | `time/all_00_30` | `time/step_30` (5/5) | +0.0153 | `time/step_30` | fold vote and OOF guard passed | 0.7710 / 0.7556 | +0.0153 | +0.0172 [+0.0138, +0.0207] | yes |
| SA-R | `resolution/r8_r16_r32_r64` | `resolution/r8_r64` (5/5) | +0.0119 | `resolution/r8_r64` | fold vote and OOF guard passed | 0.7676 / 0.7556 | +0.0119 | +0.0177 [+0.0143, +0.0211] | yes |

## SA-T

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `time/step_30` | promote | 0.7735 / 0.7580 | 0.7613 / 0.7465 |
| 1 | `time/step_30` | promote | 0.7655 / 0.7490 | 0.7893 / 0.7780 |
| 2 | `time/step_30` | promote | 0.7728 / 0.7571 | 0.7664 / 0.7518 |
| 3 | `time/step_30` | promote | 0.7720 / 0.7578 | 0.7664 / 0.7460 |
| 4 | `time/step_30` | promote | 0.7707 / 0.7559 | 0.7659 / 0.7484 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `time/step_30` | 0.7465 | 0.7710 | 0.7457 | 0.9551 |
| `time/step_29` | 0.7450 | 0.7689 | 0.7443 | 0.9545 |
| `time/window_26_30` | 0.7447 | 0.7687 | 0.7447 | 0.9545 |
| `time/trailing_26_30` | 0.7447 | 0.7687 | 0.7447 | 0.9545 |
| `time/step_28` | 0.7442 | 0.7681 | 0.7442 | 0.9543 |
| `time/trailing_21_30` | 0.7432 | 0.7675 | 0.7452 | 0.9545 |
| `time/step_27` | 0.7436 | 0.7675 | 0.7442 | 0.9542 |
| `time/step_26` | 0.7431 | 0.7671 | 0.7443 | 0.9542 |
| `time/step_25` | 0.7425 | 0.7667 | 0.7445 | 0.9542 |
| `time/step_24` | 0.7419 | 0.7664 | 0.7449 | 0.9542 |

## SA-R

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `resolution/r8_r64` | promote | 0.7691 / 0.7580 | 0.7635 / 0.7465 |
| 1 | `resolution/r8_r64` | promote | 0.7608 / 0.7490 | 0.7924 / 0.7780 |
| 2 | `resolution/r8_r64` | promote | 0.7692 / 0.7571 | 0.7635 / 0.7518 |
| 3 | `resolution/r8_r64` | promote | 0.7696 / 0.7578 | 0.7600 / 0.7460 |
| 4 | `resolution/r8_r64` | promote | 0.7690 / 0.7559 | 0.7568 / 0.7484 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `resolution/r64` | 0.7474 | 0.7678 | 0.7309 | 0.9539 |
| `resolution/r8_r64` | 0.7414 | 0.7676 | 0.7399 | 0.9542 |
| `resolution/r32_r64` | 0.7388 | 0.7627 | 0.7369 | 0.9550 |
| `resolution/r8_r32_r64` | 0.7356 | 0.7619 | 0.7399 | 0.9549 |
| `resolution/r16_r64` | 0.7199 | 0.7578 | 0.7507 | 0.9531 |
| `resolution/r16_r32_r64` | 0.7237 | 0.7568 | 0.7457 | 0.9538 |
| `resolution/r8_r16_r32_r64` | 0.7213 | 0.7556 | 0.7466 | 0.9536 |
| `resolution/r8_r16_r64` | 0.7158 | 0.7554 | 0.7516 | 0.9529 |
| `resolution/r32` | 0.7275 | 0.7532 | 0.7353 | 0.9534 |
| `resolution/r8_r32` | 0.7214 | 0.7510 | 0.7387 | 0.9528 |

## Joint SA time-resolution diagnostic

This interaction grid is diagnostic and is not eligible for automatic freezing.

Inclusive reference: `time_resolution/all_00_30__r8_r16_r32_r64` at 0.75564 mIoU.

| Rank | Configuration | mIoU@0.45 | Gain over inclusive reference |
|---:|---|---:|---:|
| 1 | `time_resolution/window_26_30__r8_r64` | 0.77518 | +0.01954 |
| 2 | `time_resolution/trailing_26_30__r8_r64` | 0.77518 | +0.01954 |
| 3 | `time_resolution/step_29__r8_r64` | 0.77511 | +0.01947 |
| 4 | `time_resolution/step_28__r8_r64` | 0.77510 | +0.01946 |
| 5 | `time_resolution/step_27__r8_r64` | 0.77493 | +0.01929 |
| 6 | `time_resolution/trailing_21_30__r8_r64` | 0.77491 | +0.01927 |
| 7 | `time_resolution/step_30__r8_r64` | 0.77490 | +0.01926 |
| 8 | `time_resolution/step_26__r8_r64` | 0.77480 | +0.01916 |
| 9 | `time_resolution/step_25__r8_r64` | 0.77458 | +0.01893 |
| 10 | `time_resolution/trailing_15_30__r8_r64` | 0.77453 | +0.01888 |
| 11 | `time_resolution/step_24__r8_r64` | 0.77438 | +0.01873 |
| 12 | `time_resolution/window_21_25__r8_r64` | 0.77432 | +0.01868 |
| 13 | `time_resolution/step_23__r8_r64` | 0.77423 | +0.01859 |
| 14 | `time_resolution/step_22__r8_r64` | 0.77413 | +0.01848 |
| 15 | `time_resolution/trailing_10_30__r8_r64` | 0.77397 | +0.01833 |
| 16 | `time_resolution/step_21__r8_r64` | 0.77386 | +0.01822 |
| 17 | `time_resolution/step_20__r8_r64` | 0.77350 | +0.01785 |
| 18 | `time_resolution/step_19__r8_r64` | 0.77337 | +0.01772 |
| 19 | `time_resolution/step_30__r16_r64` | 0.77292 | +0.01728 |
| 20 | `time_resolution/step_18__r8_r64` | 0.77280 | +0.01716 |
