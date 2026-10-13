# Cross-validated selection

490 images, 20 classes, 1279 configurations; shards complete: 20/20. Selection at threshold 0.45.

| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |
|---|---|---|---:|---|---|---|---:|---|:---:|
| CA-T | `time/all_00_30` | `time/all_00_30` (5/5) | +0.0000 | `time/all_00_30` | reference won the fold vote | 0.6661 / 0.6661 | +0.0000 | +0.0000 [+0.0000, +0.0000] | no |
| CA-R | `resolution/r8_r16_r32_r64` | `resolution/r8_r16_r64` (5/5) | +0.0296 | `resolution/r8_r16_r64` | fold vote and OOF guard passed | 0.6957 / 0.6661 | +0.0296 | +0.0211 [+0.0139, +0.0286] | no |

## CA-T

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `time/all_00_30` | no eligible candidate | 0.6650 / 0.6650 | 0.6725 / 0.6725 |
| 1 | `time/all_00_30` | no eligible candidate | 0.6618 / 0.6618 | 0.6832 / 0.6832 |
| 2 | `time/all_00_30` | no eligible candidate | 0.6651 / 0.6651 | 0.6716 / 0.6716 |
| 3 | `time/all_00_30` | no eligible candidate | 0.6667 / 0.6667 | 0.6687 / 0.6687 |
| 4 | `time/all_00_30` | no eligible candidate | 0.6725 / 0.6725 | 0.6431 / 0.6431 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `time/cumulative_00_20` | 0.6886 | 0.6752 | 0.5763 | 0.9157 |
| `time/cumulative_00_25` | 0.7001 | 0.6727 | 0.5601 | 0.9198 |
| `time/leave_step_30` | 0.7052 | 0.6678 | 0.5468 | 0.9214 |
| `time/leave_step_29` | 0.7051 | 0.6678 | 0.5471 | 0.9215 |
| `time/leave_step_28` | 0.7050 | 0.6677 | 0.5471 | 0.9216 |
| `time/leave_step_27` | 0.7050 | 0.6677 | 0.5471 | 0.9216 |
| `time/leave_step_26` | 0.7050 | 0.6677 | 0.5471 | 0.9217 |
| `time/leave_step_25` | 0.7049 | 0.6677 | 0.5471 | 0.9217 |
| `time/leave_step_21` | 0.7050 | 0.6676 | 0.5470 | 0.9218 |
| `time/leave_step_23` | 0.7050 | 0.6676 | 0.5470 | 0.9217 |

## CA-R

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `resolution/r8_r16_r64` | promote | 0.6968 / 0.6650 | 0.6907 / 0.6725 |
| 1 | `resolution/r8_r16_r64` | promote | 0.6913 / 0.6618 | 0.7118 / 0.6832 |
| 2 | `resolution/r8_r16_r64` | promote | 0.6959 / 0.6651 | 0.6973 / 0.6716 |
| 3 | `resolution/r8_r16_r64` | promote | 0.6975 / 0.6667 | 0.6872 / 0.6687 |
| 4 | `resolution/r8_r16_r64` | promote | 0.6967 / 0.6725 | 0.6872 / 0.6431 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `resolution/r8_r16_r64` | 0.7068 | 0.6957 | 0.5876 | 0.9131 |
| `resolution/r8_r16` | 0.7148 | 0.6933 | 0.5942 | 0.9171 |
| `resolution/r8_r16_r32_r64` | 0.7059 | 0.6661 | 0.5434 | 0.9218 |
| `resolution/r8_r16_r32` | 0.7082 | 0.6562 | 0.5323 | 0.9240 |
| `resolution/r16` | 0.7099 | 0.6489 | 0.5086 | 0.9059 |
| `resolution/r16_r64` | 0.6906 | 0.6449 | 0.4925 | 0.8902 |
| `resolution/r16_r32_r64` | 0.6849 | 0.6138 | 0.4590 | 0.9031 |
| `resolution/r16_r32` | 0.6877 | 0.6004 | 0.4461 | 0.9092 |
| `resolution/r8_r32_r64` | 0.6403 | 0.5632 | 0.4193 | 0.8929 |
| `resolution/r8` | 0.6021 | 0.5626 | 0.4787 | 0.8462 |

## Joint CA time-resolution diagnostic

This grid is interaction evidence only. Its in-sample maximum is not eligible for automatic freezing under protocol Section 4.

Inclusive reference: `time_resolution/all_00_30__r8_r16_r32_r64` at 0.66610 mIoU.

| Rank | Configuration | mIoU@0.45 | Gain over inclusive reference |
|---:|---|---:|---:|
| 1 | `time_resolution/cumulative_00_25__r8_r16` | 0.69680 | +0.03069 |
| 2 | `time_resolution/leave_step_17__r8_r16_r64` | 0.69678 | +0.03067 |
| 3 | `time_resolution/leave_step_18__r8_r16_r64` | 0.69675 | +0.03065 |
| 4 | `time_resolution/leave_step_16__r8_r16_r64` | 0.69674 | +0.03064 |
| 5 | `time_resolution/leave_step_19__r8_r16_r64` | 0.69670 | +0.03060 |
| 6 | `time_resolution/leave_step_15__r8_r16_r64` | 0.69666 | +0.03056 |
| 7 | `time_resolution/leave_step_20__r8_r16_r64` | 0.69666 | +0.03056 |
| 8 | `time_resolution/leave_step_14__r8_r16_r64` | 0.69658 | +0.03048 |
| 9 | `time_resolution/leave_step_21__r8_r16_r64` | 0.69655 | +0.03044 |
| 10 | `time_resolution/leave_step_13__r8_r16_r64` | 0.69640 | +0.03030 |
| 11 | `time_resolution/leave_step_22__r8_r16_r64` | 0.69639 | +0.03028 |
| 12 | `time_resolution/leave_step_23__r8_r16_r64` | 0.69627 | +0.03017 |
| 13 | `time_resolution/leave_step_12__r8_r16_r64` | 0.69626 | +0.03016 |
| 14 | `time_resolution/cumulative_00_25__r8_r16_r64` | 0.69619 | +0.03008 |
| 15 | `time_resolution/leave_step_30__r8_r16_r64` | 0.69618 | +0.03008 |
| 16 | `time_resolution/leave_step_24__r8_r16_r64` | 0.69617 | +0.03007 |
| 17 | `time_resolution/leave_step_25__r8_r16_r64` | 0.69610 | +0.03000 |
| 18 | `time_resolution/leave_step_29__r8_r16_r64` | 0.69607 | +0.02997 |
| 19 | `time_resolution/leave_step_26__r8_r16_r64` | 0.69606 | +0.02996 |
| 20 | `time_resolution/leave_step_11__r8_r16_r64` | 0.69605 | +0.02995 |
