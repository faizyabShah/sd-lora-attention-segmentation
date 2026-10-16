# Controlled propagation package

490 images, 20 classes, 12 configurations, 17 thresholds.
Method and threshold are selected on four folds and evaluated on the held-out fold.

| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Full threshold | Full mIoU | Why |
|---|---|---:|---:|---:|---|---:|---:|---|
| P4 residual mix | `REF/one_pass` | 5/5 | 0.7875 → 0.7875 | +0.0000 | `REF/one_pass` | 0.425 | 0.7888 | reference won the fold vote |
| F3/F4 gating | `REF/one_pass` | 5/5 | 0.7875 → 0.7875 | +0.0000 | `REF/one_pass` | 0.425 | 0.7888 | reference won the fold vote |
| N2-N4 calibration | `REF/one_pass` | 5/5 | 0.7875 → 0.7875 | +0.0000 | `REF/one_pass` | 0.425 | 0.7888 | reference won the fold vote |
| R2 DenseCRF | `R2/crf` | 5/5 | 0.7875 → 0.8049 | +0.0175 | `R2/crf` | 0.425 | 0.8049 | fold vote and OOF guard passed |

## P4 residual mix

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7854/0.7854 | no eligible candidate |
| 1 | `REF/one_pass` | 0.425 | 0.425 | 0.7827/0.7827 | 0.8113/0.8113 | no eligible candidate |
| 2 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7861/0.7861 | no eligible candidate |
| 3 | `REF/one_pass` | 0.425 | 0.425 | 0.7912/0.7912 | 0.7803/0.7803 | no eligible candidate |
| 4 | `REF/one_pass` | 0.450 | 0.450 | 0.7909/0.7909 | 0.7757/0.7757 | no eligible candidate |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `P4/lam90` | 0.7856 |
| `P4/lam75` | 0.7797 |
| `P4/lam50` | 0.7634 |
| `P4/lam25` | 0.7365 |

## F3/F4 gating

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7854/0.7854 | no eligible candidate |
| 1 | `REF/one_pass` | 0.425 | 0.425 | 0.7827/0.7827 | 0.8113/0.8113 | no eligible candidate |
| 2 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7861/0.7861 | no eligible candidate |
| 3 | `REF/one_pass` | 0.425 | 0.425 | 0.7912/0.7912 | 0.7803/0.7803 | no eligible candidate |
| 4 | `REF/one_pass` | 0.450 | 0.450 | 0.7909/0.7909 | 0.7757/0.7757 | no eligible candidate |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `F4/geomean` | 0.7638 |
| `F3/gate` | 0.5139 |

## N2-N4 calibration

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7854/0.7854 | no eligible candidate |
| 1 | `REF/one_pass` | 0.425 | 0.425 | 0.7827/0.7827 | 0.8113/0.8113 | no eligible candidate |
| 2 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7861/0.7861 | no eligible candidate |
| 3 | `REF/one_pass` | 0.425 | 0.425 | 0.7912/0.7912 | 0.7803/0.7803 | no eligible candidate |
| 4 | `REF/one_pass` | 0.450 | 0.450 | 0.7909/0.7909 | 0.7757/0.7757 | no eligible candidate |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `N4/otsu` | 0.7955 |
| `N2/percentile` | 0.7902 |
| `N3/max` | 0.6943 |

## R2 DenseCRF

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `R2/crf` | 0.425 | 0.425 | 0.8062/0.7902 | 0.8036/0.7854 | promote |
| 1 | `R2/crf` | 0.425 | 0.425 | 0.7982/0.7827 | 0.8327/0.8113 | promote |
| 2 | `R2/crf` | 0.425 | 0.425 | 0.8067/0.7902 | 0.8010/0.7861 | promote |
| 3 | `R2/crf` | 0.425 | 0.425 | 0.8081/0.7912 | 0.7943/0.7803 | promote |
| 4 | `R2/crf` | 0.425 | 0.450 | 0.8055/0.7909 | 0.7972/0.7757 | promote |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `R2/crf` | 0.8027 |
