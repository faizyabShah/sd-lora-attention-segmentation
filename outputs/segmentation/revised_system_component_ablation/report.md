# Revised-system component isolation

Baseline: one SA pass and all 16 CA layers. All other final choices are unchanged.

## voc_sim

| Configuration | mIoU | Change from revised full | Impact |
|---|---:|---:|---:|
| Revised full system: one SA pass, all CA layers | 0.7824 | +0.0000 | +0.0000 |
| Without CA head filtering | 0.7625 | -0.0199 | +0.0199 |
| Without SA propagation | 0.6969 | -0.0855 | +0.0855 |
| All SA calls instead of calls 15--30 | 0.7802 | -0.0022 | +0.0022 |
| All SA heads | 0.7837 | +0.0013 | -0.0013 |
| Max normalization instead of percentile 1--99 | 0.7804 | -0.0021 | +0.0021 |
| Without 1% component cleanup | 0.7824 | -0.0001 | +0.0001 |

## coco_cap

| Configuration | mIoU | Change from revised full | Impact |
|---|---:|---:|---:|
| Revised full system: one SA pass, all CA layers | 0.6322 | +0.0000 | +0.0000 |
| Without CA head filtering | 0.6253 | -0.0069 | +0.0069 |
| Without SA propagation | 0.5862 | -0.0460 | +0.0460 |
| All SA calls instead of calls 15--30 | 0.6240 | -0.0083 | +0.0083 |
| All SA heads | 0.6274 | -0.0049 | +0.0049 |
| Max normalization instead of percentile 1--99 | 0.6147 | -0.0176 | +0.0176 |
| Without 1% component cleanup | 0.6288 | -0.0034 | +0.0034 |

