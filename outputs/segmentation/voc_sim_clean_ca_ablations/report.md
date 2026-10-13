# Clean CA-only ablations on full VOC-Sim

Images: **490**; classes: **20**.

Fixed for every comparison: conditional CA; every tokenizer piece of the class name; equal averaging; min-max normalization; threshold 0.45; no SA, DenseCRF, learned weights, or component cleanup.

| Family | Best configuration | mIoU | Mean image IoU |
|---|---|---:|---:|
| heads | `heads_0_2_3_4_5_6_7` | 0.67641 | 0.68005 |
| layers_structured | `down_middle` | 0.68175 | 0.67752 |
| resolution | `r8_r16_r64` | 0.69568 | 0.69238 |
| time | `cumulative_00_20` | 0.67522 | 0.67500 |
| time_resolution | `cumulative_00_25__r8_r16` | 0.69681 | 0.69069 |
| layers_exhaustive | `0x23f7` ([0, 1, 2, 4, 5, 6, 7, 8, 9, 13]) | 0.70864 | --- |

All raw regular results are in `all_results.json.gz`; all 65,535 layer subsets are in `layers_exhaustive/all_subsets.csv.gz`.
