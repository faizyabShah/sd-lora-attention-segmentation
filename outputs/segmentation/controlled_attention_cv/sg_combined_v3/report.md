# SG-combined-v3

Five cells raw and with unchanged R2, 40 class shards, 490 VOC and 707 COCO images; historical A3 seed.
VOC-only selection; COCO thresholds frozen. AP, best-threshold and GT cap-loss values are diagnostic only.

## Selection table (registered, reference R0)

| Cell | VOC forced OOF | Δ vs R0 | Fold wins | Full VOC (thr) | Frozen COCO |
|---|---:|---:|---:|---:|---:|
| R0 | 0.8098 | +0.0000 | 0/5 | 0.8114 (0.450) | 0.6599 |
| R0+R2 | 0.8230 | +0.0000 | 0/5 | 0.8244 (0.425) | 0.6623 |
| R1 | 0.8157 | +0.0059 | 5/5 | 0.8174 (0.550) | 0.6820 |
| R1+R2 | 0.8293 | +0.0063 | 5/5 | 0.8303 (0.550) | 0.6964 |
| W1 | 0.8161 | +0.0063 | 5/5 | 0.8176 (0.550) | 0.6827 |
| W1+R2 | 0.8294 | +0.0064 | 5/5 | 0.8305 (0.550) | 0.6972 |
| W2 | 0.8155 | +0.0057 | 4/5 | 0.8168 (0.550) | 0.6833 |
| W2+R2 | 0.8281 | +0.0051 | 4/5 | 0.8296 (0.550) | 0.6975 |
| W3 | 0.8116 | +0.0019 | 3/5 | 0.8134 (0.575) | 0.6792 |
| W3+R2 | 0.8261 | +0.0031 | 4/5 | 0.8261 (0.550) | 0.6828 |

Raw selector: R0 {'R0': 4, 'W2': 1}
R2 selector: R0+R2 {'R0+R2': 5}

## Mechanism table (reference R1 = v2 V3)

| Contrast | VOC OOF Δ | Fold wins | Paired CI | COCO frozen Δ | COCO best-thr Δ | COCO raw AP Δ | VOC sheep Δ |
|---|---:|---:|---|---:|---:|---:|---:|
| W1 − R1 | +0.0004 | 4/5 | [-0.0007, +0.0011] | +0.0007 | +0.0000 | +0.0003 | -0.000 |
| W1+R2 − R1+R2 | +0.0001 | 4/5 | [-0.0003, +0.0004] | +0.0007 | +0.0003 | — | +0.000 |
| W2 − R1 | -0.0002 | 3/5 | [-0.0017, +0.0009] | +0.0013 | +0.0025 | +0.0025 | -0.003 |
| W2+R2 − R1+R2 | -0.0012 | 2/5 | [-0.0029, +0.0006] | +0.0011 | +0.0025 | — | -0.004 |
| W3 − R1 | -0.0040 | 0/5 | [-0.0075, -0.0009] | -0.0028 | -0.0082 | -0.0093 | -0.012 |
| W3+R2 − R1+R2 | -0.0032 | 1/5 | [-0.0070, +0.0001] | -0.0137 | -0.0075 | — | +0.017 |
| R1 − R0 | +0.0059 | 5/5 | [+0.0006, +0.0121] | +0.0221 | +0.0173 | -0.0006 | -0.069 |
| R1+R2 − R0+R2 | +0.0063 | 5/5 | [+0.0010, +0.0115] | +0.0341 | +0.0175 | — | -0.092 |

GT cap loss at 16→32 (VOC; sheep vs non-sheep): R1 γ_GT 0.286/0.118, L_GT 0.011/0.005; W1 γ_GT 0.317/0.156, L_GT 0.013/0.006; W2 γ_GT 0.287/0.121, L_GT 0.012/0.005
W2 native coarse seed vs downsampled (VOC): {"pearson_mean": 0.9958314520972116, "pearson_min": 0.9702130556106567, "mean_abs_difference": 0.021566122008145464}
Registered decision inputs: {"W1": {"beats_R1_4of5_with_interval_above_zero": false, "voc_sheep_delta_vs_R1": {"raw": -8.927878161557601e-05, "+R2": 0.00010011185945013601}}, "W2": {"beats_R1_4of5_with_interval_above_zero": false, "voc_sheep_delta_vs_R1": {"raw": -0.003301925654178972, "+R2": -0.004044682439787284}}, "W3": {"beats_R1_4of5_with_interval_above_zero": false, "voc_sheep_delta_vs_R1": {"raw": -0.01231018819021068, "+R2": 0.016983891827416442}}}

Size bins, cap binding, per-class GT cap loss and per-class IoU are in results.json; dense maps are in the shard sidecars.
