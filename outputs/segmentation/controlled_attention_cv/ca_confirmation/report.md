# Cross-validated selection

490 images, 20 classes, 6 configurations; shards complete: 20/20. Selection at threshold 0.45.

| Screen | Reference | Vote winner (votes) | OOF gain | Frozen | Why | OOF mIoU (selection / reference) | In-sample Δ (frozen) | Paired Δ image IoU [95% CI] (frozen) | Calibration-dependent |
|---|---|---|---:|---|---|---|---:|---|:---:|
| CA-C | `C/reference` | `C/R` (4/5) | +0.0260 | `C/R` | fold vote and OOF guard passed | 0.6914 / 0.6653 | +0.0303 | +0.0217 [+0.0145, +0.0293] | no |

## CA-C

| Fold | Chosen | Reason | Train mIoU chosen / ref | Held-out mIoU chosen / ref |
|---:|---|---|---|---|
| 0 | `C/R` | promote | 0.6967 / 0.6644 | 0.6905 / 0.6708 |
| 1 | `C/R` | promote | 0.6912 / 0.6610 | 0.7116 / 0.6827 |
| 2 | `C/R` | promote | 0.6957 / 0.6641 | 0.6973 / 0.6714 |
| 3 | `C/R` | promote | 0.6975 / 0.6661 | 0.6868 / 0.6676 |
| 4 | `C/L` | promote | 0.6977 / 0.6715 | 0.6702 / 0.6429 |

Top configurations, full 490 (in-sample):

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| `C/R+H` | 0.7058 | 0.6960 | 0.5879 | 0.9127 |
| `C/R` | 0.7067 | 0.6956 | 0.5877 | 0.9131 |
| `C/L+H` | 0.7020 | 0.6924 | 0.5948 | 0.9190 |
| `C/L` | 0.7035 | 0.6921 | 0.5938 | 0.9201 |
| `C/H` | 0.7056 | 0.6761 | 0.5623 | 0.9205 |
| `C/reference` | 0.7057 | 0.6653 | 0.5430 | 0.9218 |
