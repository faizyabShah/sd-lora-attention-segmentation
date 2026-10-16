# Controlled propagation package

490 images, 20 classes, 40 configurations, 17 thresholds.
Method and threshold are selected on four folds and evaluated on the held-out fold.

| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Full threshold | Full mIoU | Why |
|---|---|---:|---:|---:|---|---:|---:|---|
| P1 observation-wise powers | `P1/obs_n2` | 4/5 | 0.7875 → 0.7991 | +0.0116 | `P1/obs_n2` | 0.450 | 0.8007 | fold vote and OOF guard passed |
| W3 average-before powers | `REF/one_pass` | 4/5 | 0.7875 → 0.7875 | -0.0000 | `REF/one_pass` | 0.425 | 0.7888 | reference won the fold vote |
| P2 observation-wise restart | `REF/one_pass` | 4/5 | 0.7875 → 0.7877 | +0.0002 | `REF/one_pass` | 0.425 | 0.7888 | reference won the fold vote |
| P2 average-before restart | `REF/one_pass` | 5/5 | 0.7875 → 0.7875 | +0.0000 | `REF/one_pass` | 0.425 | 0.7888 | reference won the fold vote |
| P6 sparse affinity | `REF/one_pass` | 5/5 | 0.7875 → 0.7875 | +0.0000 | `REF/one_pass` | 0.425 | 0.7888 | reference won the fold vote |

## P1 observation-wise powers

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `P1/obs_n2` | 0.450 | 0.425 | 0.8005/0.7902 | 0.8046/0.7855 | promote |
| 1 | `P1/obs_n2` | 0.450 | 0.425 | 0.7947/0.7827 | 0.8233/0.8113 | promote |
| 2 | `P1/obs_n2` | 0.450 | 0.425 | 0.8011/0.7902 | 0.8024/0.7861 | promote |
| 3 | `P1/obs_n2` | 0.450 | 0.425 | 0.8039/0.7912 | 0.7907/0.7805 | promote |
| 4 | `P1/obs_n3` | 0.475 | 0.450 | 0.8071/0.7910 | 0.7784/0.7757 | promote |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `P1/obs_n3` | 0.8014 |
| `P1/obs_n2` | 0.8007 |
| `P1/obs_n4` | 0.8004 |

## W3 average-before powers

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7855/0.7855 | no eligible candidate |
| 1 | `REF/one_pass` | 0.425 | 0.425 | 0.7827/0.7827 | 0.8113/0.8113 | no eligible candidate |
| 2 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7861/0.7861 | no eligible candidate |
| 3 | `W3/mean_n2` | 0.475 | 0.425 | 0.7990/0.7912 | 0.7823/0.7805 | conditional |
| 4 | `REF/one_pass` | 0.450 | 0.450 | 0.7910/0.7910 | 0.7757/0.7757 | no eligible candidate |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `W3/mean_n2` | 0.7940 |
| `W3/mean_n1` | 0.7887 |
| `W3/mean_n3` | 0.7828 |
| `W3/mean_n4` | 0.7684 |

## P2 observation-wise restart

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7855/0.7855 | no eligible candidate |
| 1 | `REF/one_pass` | 0.425 | 0.425 | 0.7827/0.7827 | 0.8113/0.8113 | no eligible candidate |
| 2 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7861/0.7861 | no eligible candidate |
| 3 | `REF/one_pass` | 0.425 | 0.425 | 0.7912/0.7912 | 0.7805/0.7805 | no eligible candidate |
| 4 | `P2/obs_b10_n4` | 0.450 | 0.450 | 0.7993/0.7910 | 0.7769/0.7757 | conditional |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `P2/obs_b10_n4` | 0.7954 |
| `P2/obs_b10_n2` | 0.7934 |
| `P2/obs_b10_n8` | 0.7922 |
| `P2/obs_b25_n4` | 0.7775 |
| `P2/obs_b25_n2` | 0.7766 |
| `P2/obs_b25_n8` | 0.7759 |
| `P2/obs_b50_n2` | 0.7470 |
| `P2/obs_b50_n4` | 0.7467 |

## P2 average-before restart

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7855/0.7855 | no eligible candidate |
| 1 | `REF/one_pass` | 0.425 | 0.425 | 0.7827/0.7827 | 0.8113/0.8113 | no eligible candidate |
| 2 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7861/0.7861 | no eligible candidate |
| 3 | `REF/one_pass` | 0.425 | 0.425 | 0.7912/0.7912 | 0.7805/0.7805 | no eligible candidate |
| 4 | `REF/one_pass` | 0.450 | 0.450 | 0.7910/0.7910 | 0.7757/0.7757 | no eligible candidate |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `P2/mean_b10_n2` | 0.7888 |
| `P2/mean_b10_n4` | 0.7825 |
| `P2/mean_b10_n8` | 0.7763 |
| `P2/mean_b25_n2` | 0.7738 |
| `P2/mean_b25_n4` | 0.7692 |
| `P2/mean_b25_n8` | 0.7656 |
| `P2/mean_b50_n2` | 0.7453 |
| `P2/mean_b50_n4` | 0.7433 |

## P6 sparse affinity

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7855/0.7855 | no eligible candidate |
| 1 | `REF/one_pass` | 0.425 | 0.425 | 0.7827/0.7827 | 0.8113/0.8113 | no eligible candidate |
| 2 | `REF/one_pass` | 0.425 | 0.425 | 0.7902/0.7902 | 0.7861/0.7861 | no eligible candidate |
| 3 | `REF/one_pass` | 0.425 | 0.425 | 0.7912/0.7912 | 0.7805/0.7805 | no eligible candidate |
| 4 | `REF/one_pass` | 0.450 | 0.450 | 0.7910/0.7910 | 0.7757/0.7757 | no eligible candidate |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `P6/topk_1024` | 0.7792 |
| `P6/topk_0512` | 0.7704 |
| `P6/topk_0256` | 0.7604 |
| `P6/topk_0128` | 0.7503 |
| `P6/topk_0064` | 0.7409 |
| `P6/topk_0032` | 0.7326 |
| `P6/topk_0016` | 0.7255 |

## Frozen-winner diagnostics

### P1 observation-wise powers: `P1/obs_n2`

Pure frozen-candidate OOF mIoU: 0.7875 → 0.7998 (+0.0123). Paired mean image-IoU +0.0143 [+0.0107, +0.0178]; 365 better, 125 worse.

| Class | Reference OOF IoU | Selected OOF IoU | Δ |
|---|---:|---:|---:|
| tv | 0.7849 | 0.7733 | -0.0116 |
| bus | 0.9122 | 0.9070 | -0.0051 |
| dining-table | 0.4576 | 0.4530 | -0.0046 |
| dog | 0.8991 | 0.8980 | -0.0011 |
| car | 0.8261 | 0.8269 | +0.0008 |
| sheep | 0.7895 | 0.7912 | +0.0018 |
| cow | 0.8305 | 0.8357 | +0.0052 |
| bird | 0.8454 | 0.8512 | +0.0058 |
| bottle | 0.8577 | 0.8636 | +0.0058 |
| horse | 0.8701 | 0.8784 | +0.0083 |
| bicycle | 0.7371 | 0.7458 | +0.0087 |
| motorbike | 0.8427 | 0.8525 | +0.0097 |
| sofa | 0.9381 | 0.9492 | +0.0112 |
| cat | 0.8080 | 0.8204 | +0.0124 |
| train | 0.6485 | 0.6668 | +0.0183 |
| potted-plant | 0.7276 | 0.7459 | +0.0183 |
| boat | 0.8178 | 0.8417 | +0.0239 |
| aeroplane | 0.8371 | 0.8670 | +0.0299 |
| person | 0.6901 | 0.7205 | +0.0304 |
| chair | 0.6301 | 0.7081 | +0.0780 |
