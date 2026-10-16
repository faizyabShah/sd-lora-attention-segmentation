# Controlled propagation package

490 images, 20 classes, 30 configurations, 17 thresholds.
Method and threshold are selected on four folds and evaluated on the held-out fold.

| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Full threshold | Full mIoU | Why |
|---|---|---:|---:|---:|---|---:|---:|---|
| P3-C mandatory confirmation | `P1/obs_n2|N0/global|R2/crf` | 5/5 | 0.7875 → 0.8135 | +0.0260 | `P1/obs_n2|N0/global|R2/crf` | 0.450 | 0.8135 | fold vote and OOF guard passed |
| P3-C expanded confirmation | `P1/obs_n2|N4/otsu|R2/crf` | 5/5 | 0.7875 → 0.8207 | +0.0332 | `P1/obs_n2|N4/otsu|R2/crf` | 0.450 | 0.8207 | fold vote and OOF guard passed |

## P3-C mandatory confirmation

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `P1/obs_n2|N0/global|R2/crf` | 0.450 | 0.425 | 0.8127/0.7902 | 0.8210/0.7855 | promote |
| 1 | `P1/obs_n2|N0/global|R2/crf` | 0.450 | 0.425 | 0.8068/0.7827 | 0.8410/0.8113 | promote |
| 2 | `P1/obs_n2|N0/global|R2/crf` | 0.450 | 0.425 | 0.8142/0.7902 | 0.8139/0.7861 | promote |
| 3 | `P1/obs_n2|N0/global|R2/crf` | 0.450 | 0.425 | 0.8179/0.7912 | 0.7994/0.7805 | promote |
| 4 | `P1/obs_n2|N0/global|R2/crf` | 0.450 | 0.450 | 0.8162/0.7910 | 0.7995/0.7757 | promote |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `P1/obs_n2|N0/global|R2/crf` | 0.8135 |
| `REF/one_pass|N0/global|R2/crf` | 0.8028 |
| `P1/obs_n2|N0/global|R0/none` | 0.8007 |
| `REF/one_pass|N0/global|R0/none` | 0.7887 |

## P3-C expanded confirmation

| Fold | Chosen | Threshold | Ref threshold | Train chosen/ref | Held-out chosen/ref | Reason |
|---:|---|---:|---:|---:|---:|---|
| 0 | `P1/obs_n2|N4/otsu|R2/crf` | 0.450 | 0.425 | 0.8211/0.7902 | 0.8217/0.7855 | promote |
| 1 | `P1/obs_n2|N4/otsu|R2/crf` | 0.450 | 0.425 | 0.8130/0.7827 | 0.8513/0.8113 | promote |
| 2 | `P1/obs_n2|N4/otsu|R2/crf` | 0.450 | 0.425 | 0.8236/0.7902 | 0.8132/0.7861 | promote |
| 3 | `P1/obs_n2|N4/otsu|R2/crf` | 0.450 | 0.425 | 0.8224/0.7912 | 0.8129/0.7805 | promote |
| 4 | `P1/obs_n2|N4/otsu|R2/crf` | 0.450 | 0.450 | 0.8231/0.7910 | 0.8047/0.7757 | promote |

Top fixed-threshold configurations:

| Configuration | mIoU@0.45 |
|---|---:|
| `P1/obs_n2|N4/otsu|R2/crf` | 0.8207 |
| `P2/obs_b10_n4|N4/otsu|R2/crf` | 0.8168 |
| `P1/obs_n3|N4/otsu|R2/crf` | 0.8168 |
| `P1/obs_n3|N2/percentile|R2/crf` | 0.8167 |
| `P1/obs_n2|N2/percentile|R2/crf` | 0.8156 |
| `P2/obs_b10_n4|N2/percentile|R2/crf` | 0.8136 |
| `P1/obs_n3|N0/global|R2/crf` | 0.8136 |
| `P1/obs_n2|N0/global|R2/crf` | 0.8135 |

## Frozen mandatory-cell diagnostics

| Configuration | Reference OOF mIoU | Candidate OOF mIoU | Gain | Paired mean image-IoU [95% CI] | Better / worse |
|---|---:|---:|---:|---:|---:|
| `P1/obs_n2|N0/global|R0/none` | 0.7875 | 0.7998 | +0.0123 | +0.0143 [+0.0107, +0.0178] | 365 / 125 |
| `REF/one_pass|N0/global|R2/crf` | 0.7875 | 0.8050 | +0.0175 | +0.0222 [+0.0185, +0.0261] | 362 / 128 |
| `P1/obs_n2|N0/global|R2/crf` | 0.7875 | 0.8135 | +0.0260 | +0.0308 [+0.0251, +0.0364] | 378 / 112 |
| `P1/obs_n2|N4/otsu|R2/crf` | 0.7875 | 0.8207 | +0.0332 | +0.0397 [+0.0317, +0.0478] | 376 / 114 |

### Incremental expanded-candidate check

Against the mandatory winner, `P1/obs_n2|N4/otsu|R2/crf` changes OOF mIoU from 0.8135 to 0.8207 (+0.0072); paired mean image-IoU +0.0089 [+0.0026, +0.0152]. It improves 12/20 classes; the worst change is `sheep` -0.0600. Therefore it is **not promoted** under R4 as an addition to the mandatory system.
