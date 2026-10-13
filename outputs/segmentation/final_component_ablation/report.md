# Final-system component isolation

Each row changes one final-system choice while holding the remaining choices fixed.
Positive impact means that removing or replacing the component lowered mIoU.

## voc_sim

| Configuration | mIoU | Change from full | Impact |
|---|---:|---:|---:|
| Full system | 0.7999 | +0.0000 | +0.0000 |
| Without CA layer filtering | 0.7975 | -0.0024 | +0.0024 |
| Without CA head filtering | 0.7832 | -0.0167 | +0.0167 |
| Without SA propagation | 0.6805 | -0.1194 | +0.1194 |
| One SA pass instead of four | 0.7801 | -0.0198 | +0.0198 |
| All SA calls instead of calls 15--30 | 0.7938 | -0.0060 | +0.0060 |
| All SA heads | 0.7940 | -0.0059 | +0.0059 |
| Max normalization instead of percentile 1--99 | 0.7176 | -0.0822 | +0.0822 |
| Without 1% component cleanup | 0.7992 | -0.0006 | +0.0006 |

## coco_cap

| Configuration | mIoU | Change from full | Impact |
|---|---:|---:|---:|
| Full system | 0.6078 | +0.0000 | +0.0000 |
| Without CA layer filtering | 0.6120 | +0.0042 | -0.0042 |
| Without CA head filtering | 0.6026 | -0.0052 | +0.0052 |
| Without SA propagation | 0.5739 | -0.0339 | +0.0339 |
| One SA pass instead of four | 0.6263 | +0.0185 | -0.0185 |
| All SA calls instead of calls 15--30 | 0.6054 | -0.0024 | +0.0024 |
| All SA heads | 0.5935 | -0.0143 | +0.0143 |
| Max normalization instead of percentile 1--99 | 0.4898 | -0.1181 | +0.1181 |
| Without 1% component cleanup | 0.6051 | -0.0027 | +0.0027 |

