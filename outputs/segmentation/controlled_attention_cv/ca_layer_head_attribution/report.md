# CA package B: CA-L, CA-H, CA-A

490 images, 20 classes, 426 configurations; shards complete: 20/20. Selection at threshold 0.45.

| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |
|---|---|---|---:|---|---|---|---:|---|:---:|
| CA-L | `L/all_16` | `L/leave2_10_11` (5/5) | +0.0262 | `L/leave2_10_11` | fold vote and OOF guard passed | 0.6923 / 0.6661 | +0.0262 | +0.0209 [+0.0155, +0.0265] | yes |
| CA-H | `H/0_1_2_3_4_5_6_7` | `H/0_2_3_4_5_6_7` (4/5) | +0.0080 | `H/0_2_3_4_5_6_7` | fold vote and OOF guard passed | 0.6741 / 0.6661 | +0.0104 | +0.0089 [+0.0047, +0.0131] | yes |
| CA-A | `A/cond_padded_all` | `A/cond_padded_all` (5/5) | +0.0000 | `A/cond_padded_all` | reference won the fold vote | 0.6661 / 0.6661 | +0.0000 | +0.0000 [+0.0000, +0.0000] | no |

## CA-L

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `L/leave2_10_11` | promote | 0.6922 / 0.6649 | 0.6920 / 0.6728 |
| 1 | `L/leave2_10_11` | promote | 0.6873 / 0.6618 | 0.7112 / 0.6831 |
| 2 | `L/leave2_10_11` | promote | 0.6912 / 0.6650 | 0.6980 / 0.6715 |
| 3 | `L/leave2_10_11` | promote | 0.6930 / 0.6666 | 0.6902 / 0.6686 |
| 4 | `L/leave2_10_11` | promote | 0.6978 / 0.6725 | 0.6705 / 0.6429 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `L/leave2_10_11` | 0.7036 | 0.6923 | 0.5937 | 0.9201 |
| `L/leave2_10_12` | 0.7060 | 0.6870 | 0.5832 | 0.9199 |
| `L/leave2_11_12` | 0.7016 | 0.6860 | 0.5825 | 0.9179 |
| `L/leave2_11_14` | 0.7076 | 0.6852 | 0.5771 | 0.9220 |
| `L/leave2_11_15` | 0.7062 | 0.6851 | 0.5756 | 0.9207 |
| `L/leave2_10_14` | 0.7088 | 0.6831 | 0.5743 | 0.9236 |
| `L/leave2_10_15` | 0.7070 | 0.6828 | 0.5731 | 0.9222 |
| `L/leave2_09_11` | 0.7026 | 0.6826 | 0.5725 | 0.9183 |
| `L/down_middle` | 0.6842 | 0.6819 | 0.5953 | 0.9025 |
| `L/leave2_09_10` | 0.7037 | 0.6806 | 0.5697 | 0.9200 |

## CA-H

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `H/0_2_3_4_5_6_7` | promote | 0.6751 / 0.6649 | 0.6831 / 0.6728 |
| 1 | `H/0_2_3_4_5_6_7` | promote | 0.6735 / 0.6618 | 0.6886 / 0.6831 |
| 2 | `H/0_2_3_4_5_6_7` | conditional | 0.6750 / 0.6650 | 0.6829 / 0.6715 |
| 3 | `H/0_2_3_4_5_6_7` | promote | 0.6768 / 0.6666 | 0.6792 / 0.6686 |
| 4 | `H/0_1_2_3_4_5_6_7` | no eligible candidate | 0.6725 / 0.6725 | 0.6429 / 0.6429 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `H/0_2_3_4_5_6_7` | 0.7055 | 0.6765 | 0.5629 | 0.9206 |
| `H/0_2_3_4_6_7` | 0.7110 | 0.6718 | 0.5406 | 0.9121 |
| `H/0_2_3_4_5_6` | 0.7044 | 0.6676 | 0.5514 | 0.9138 |
| `H/0_1_2_3_4_5_6_7` | 0.7057 | 0.6661 | 0.5435 | 0.9218 |
| `H/0_2_3_4_5_7` | 0.7038 | 0.6629 | 0.5377 | 0.9173 |
| `H/0_3_4_5_6_7` | 0.6928 | 0.6626 | 0.5373 | 0.9084 |
| `H/2_3_4_5_6_7` | 0.7035 | 0.6625 | 0.5405 | 0.9236 |
| `H/0_2_4_5_6_7` | 0.7011 | 0.6615 | 0.5424 | 0.9140 |
| `H/0_2_3_4_6` | 0.7095 | 0.6609 | 0.5248 | 0.9005 |
| `H/0_1_2_3_4_5_6` | 0.7044 | 0.6581 | 0.5371 | 0.9167 |

## CA-A

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `A/cond_padded_all` | no eligible candidate | 0.6649 / 0.6649 | 0.6728 / 0.6728 |
| 1 | `A/cond_padded_all` | no eligible candidate | 0.6618 / 0.6618 | 0.6831 / 0.6831 |
| 2 | `A/cond_padded_all` | no eligible candidate | 0.6650 / 0.6650 | 0.6715 / 0.6715 |
| 3 | `A/cond_padded_all` | no eligible candidate | 0.6666 / 0.6666 | 0.6686 / 0.6686 |
| 4 | `A/cond_padded_all` | no eligible candidate | 0.6725 / 0.6725 | 0.6429 / 0.6429 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `A/cond_unpadded_all` | 0.7054 | 0.6678 | 0.5480 | 0.9245 |
| `A/cond_padded_all` | 0.7057 | 0.6661 | 0.5435 | 0.9218 |
| `A/uncond_padded_all` | 0.6992 | 0.6612 | 0.5489 | 0.9222 |
| `A/uncond_unpadded_all` | 0.6986 | 0.6610 | 0.5506 | 0.9241 |
| `A/cond_unpadded_last` | 0.6998 | 0.6597 | 0.5414 | 0.9246 |
| `A/cond_padded_last` | 0.7000 | 0.6579 | 0.5365 | 0.9219 |
| `A/uncond_padded_last` | 0.6938 | 0.6541 | 0.5436 | 0.9223 |
| `A/uncond_unpadded_last` | 0.6935 | 0.6539 | 0.5454 | 0.9243 |
| `A/uncond_unpadded_first` | 0.6970 | 0.6505 | 0.5252 | 0.9127 |
| `A/cond_unpadded_first` | 0.6962 | 0.6479 | 0.5172 | 0.9107 |

## CA-H head report

| Head | Alone | Leave this head out | Folds retaining it |
|---:|---:|---:|---:|
| 0 | 0.2557 | 0.6484 | 5/5 |
| 1 | 0.0975 | 0.6765 | 1/5 |
| 2 | 0.3638 | 0.6471 | 5/5 |
| 3 | 0.3556 | 0.6549 | 5/5 |
| 4 | 0.4344 | 0.6448 | 5/5 |
| 5 | 0.4179 | 0.6516 | 5/5 |
| 6 | 0.3208 | 0.6456 | 5/5 |
| 7 | 0.3165 | 0.6581 | 5/5 |

All 8 heads: 0.6661. Frozen subset `H/0_2_3_4_5_6_7` ranks 1 of 255 in-sample; best in-sample subset `H/0_2_3_4_5_6_7` (0.6765).

Greedy backward elimination (in-sample):

| Round | Removed | Heads left | mIoU |
|---:|---:|---|---:|
| 0 | — | 0 1 2 3 4 5 6 7 | 0.6661 |
| 1 | 1 | 0 2 3 4 5 6 7 | 0.6765 |
| 2 | 5 | 0 2 3 4 6 7 | 0.6718 |
| 3 | 7 | 0 2 3 4 6 | 0.6609 |
| 4 | 0 | 2 3 4 6 | 0.6388 |
| 5 | 2 | 3 4 6 | 0.6005 |
| 6 | 6 | 3 4 | 0.5444 |
| 7 | 3 | 4 | 0.4344 |

Greedy forward selection (in-sample):

| Round | Added | Heads | mIoU |
|---:|---:|---|---:|
| 0 | 4 | 4 | 0.4344 |
| 1 | 5 | 4 5 | 0.5669 |
| 2 | 3 | 3 4 5 | 0.6080 |
| 3 | 2 | 2 3 4 5 | 0.6330 |
| 4 | 6 | 2 3 4 5 6 | 0.6564 |
| 5 | 0 | 0 2 3 4 5 6 | 0.6676 |
| 6 | 7 | 0 2 3 4 5 6 7 | 0.6765 |
| 7 | 1 | 0 1 2 3 4 5 6 7 | 0.6661 |

## Exhaustive module subsets (exploratory, not eligible for freezing)

Out-of-fold mIoU of the selection procedure 0.7070 vs all-16 reference 0.6661 (gain +0.0410); fold votes {'0x23f7': 2, '0x23ff': 3}.

| Fold | Chosen mask | Modules | Train Δ | Held-out mIoU chosen / ref |
|---:|---|---:|---:|---|
| 0 | `0x23f7` | 10 | +0.0444 | 0.7053 / 0.6728 |
| 1 | `0x23ff` | 11 | +0.0409 | 0.7221 / 0.6831 |
| 2 | `0x23ff` | 11 | +0.0411 | 0.7114 / 0.6715 |
| 3 | `0x23ff` | 11 | +0.0407 | 0.7074 / 0.6686 |
| 4 | `0x23f7` | 10 | +0.0411 | 0.6875 / 0.6429 |

In-sample top 5: `0x23f7` (10 modules, 0.7087), `0x21f7` (9 modules, 0.7081), `0x23ff` (11 modules, 0.7070), `0x23f6` (9 modules, 0.7070), `0x21ff` (10 modules, 0.7064).

## Figures

- `figures/ovam-voc-sim-0_ca_modules.png`
- `figures/ovam-voc-sim-125_ca_modules.png`
- `figures/ovam-voc-sim-250_ca_modules.png`
- `figures/ovam-voc-sim-350_ca_modules.png`
