# COCO-Cap: OVAM optimized token and DenseCRF on the earlier final method

707 images, 20 classes (20/20 shards complete). Exploratory, pre-protocol; all settings frozen, nothing selected here.

| Method | mIoU | Mean image IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| `opt_all16/ca_only` | 0.6090 | 0.6062 | 0.666 | 0.886 |
| `opt_all16/sa_iter1` | 0.6358 | 0.6541 | 0.684 | 0.904 |
| `opt_all16/sa_iter1+crf` | 0.6552 | 0.6895 | 0.707 | 0.898 |
| `opt_all16/sa_iter4` | 0.6176 | 0.6447 | 0.659 | 0.907 |
| `opt_all16/sa_iter4+crf` | 0.6326 | 0.6725 | 0.677 | 0.900 |
| `opt_top4/ca_only` | 0.6181 | 0.6155 | 0.675 | 0.885 |
| `opt_top4/sa_iter1` | 0.6419 | 0.6583 | 0.689 | 0.907 |
| `opt_top4/sa_iter1+crf` | 0.6612 | 0.6935 | 0.711 | 0.901 |
| `opt_top4/sa_iter4` | 0.6221 | 0.6474 | 0.664 | 0.910 |
| `opt_top4/sa_iter4+crf` | 0.6369 | 0.6749 | 0.682 | 0.903 |
| `ovam_implemented` | 0.5717 | 0.6035 | 0.657 | 0.788 |
| `plain_top4/ca_only` | 0.5668 | 0.5699 | 0.656 | 0.776 |
| `plain_top4/sa_iter1` | 0.6264 | 0.6455 | 0.659 | 0.869 |
| `plain_top4/sa_iter1+crf` | 0.6432 | 0.6769 | 0.674 | 0.866 |
| `plain_top4/sa_iter4` | 0.6077 | 0.6387 | 0.624 | 0.899 |
| `plain_top4/sa_iter4+crf` | 0.6191 | 0.6619 | 0.636 | 0.892 |

## Paired per-image IoU differences (bootstrap over images, 20,000 resamples)

| A | B | Mean(A−B) | 95% CI | A better / B better |
|---|---|---:|---|---|
| `plain_top4/sa_iter1` | `ovam_implemented` | +0.0421 | [+0.0269, +0.0574] | 417 / 290 |
| `opt_top4/sa_iter1` | `ovam_implemented` | +0.0549 | [+0.0370, +0.0728] | 397 / 310 |
| `opt_top4/sa_iter1+crf` | `ovam_implemented` | +0.0900 | [+0.0722, +0.1081] | 473 / 233 |
| `opt_top4/sa_iter1` | `plain_top4/sa_iter1` | +0.0128 | [+0.0008, +0.0246] | 327 / 380 |
| `opt_all16/sa_iter1` | `opt_top4/sa_iter1` | -0.0043 | [-0.0070, -0.0016] | 303 / 403 |
| `plain_top4/sa_iter1+crf` | `plain_top4/sa_iter1` | +0.0314 | [+0.0269, +0.0362] | 541 / 166 |
| `opt_top4/sa_iter1+crf` | `opt_top4/sa_iter1` | +0.0352 | [+0.0308, +0.0398] | 564 / 142 |
| `opt_top4/sa_iter4` | `plain_top4/sa_iter4` | +0.0086 | [-0.0034, +0.0205] | 324 / 382 |
| `opt_top4/sa_iter4+crf` | `opt_top4/sa_iter4` | +0.0275 | [+0.0237, +0.0316] | 547 / 159 |

## Class word present in the caption?

| Method | Present | Absent |
|---|---:|---:|
| `ovam_implemented` | 0.5667 (465) | 0.5850 (242) |
| `plain_top4/sa_iter1` | 0.6271 (465) | 0.6294 (242) |
| `plain_top4/sa_iter1+crf` | 0.6439 (465) | 0.6473 (242) |
| `opt_top4/sa_iter1` | 0.6351 (465) | 0.6655 (242) |
| `opt_top4/sa_iter1+crf` | 0.6541 (465) | 0.6864 (242) |
| `opt_all16/sa_iter1+crf` | 0.6447 (465) | 0.6872 (242) |

## Pooled class IoU

| Class | ovam_implemented | plain_top4/sa_iter1 | plain_top4/sa_iter1+crf | opt_top4/sa_iter1 | opt_top4/sa_iter1+crf | opt_all16/sa_iter1+crf |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.642 | 0.667 | 0.752 | 0.738 | 0.822 | 0.813 |
| bicycle | 0.638 | 0.585 | 0.591 | 0.621 | 0.632 | 0.614 |
| bird | 0.561 | 0.739 | 0.784 | 0.726 | 0.769 | 0.779 |
| boat | 0.516 | 0.517 | 0.527 | 0.489 | 0.504 | 0.518 |
| bottle | 0.326 | 0.401 | 0.410 | 0.313 | 0.322 | 0.346 |
| bus | 0.852 | 0.885 | 0.890 | 0.881 | 0.885 | 0.891 |
| car | 0.476 | 0.451 | 0.444 | 0.654 | 0.652 | 0.647 |
| cat | 0.648 | 0.844 | 0.856 | 0.813 | 0.831 | 0.830 |
| chair | 0.282 | 0.338 | 0.345 | 0.392 | 0.404 | 0.383 |
| cow | 0.765 | 0.832 | 0.866 | 0.821 | 0.855 | 0.849 |
| dining table | 0.898 | 0.577 | 0.571 | 0.639 | 0.631 | 0.617 |
| dog | 0.568 | 0.733 | 0.746 | 0.703 | 0.718 | 0.688 |
| horse | 0.710 | 0.751 | 0.776 | 0.642 | 0.671 | 0.655 |
| motorbike | 0.652 | 0.638 | 0.648 | 0.640 | 0.656 | 0.650 |
| person | 0.195 | 0.251 | 0.252 | 0.541 | 0.564 | 0.561 |
| potted plant | 0.544 | 0.641 | 0.659 | 0.665 | 0.687 | 0.682 |
| sheep | 0.736 | 0.876 | 0.905 | 0.799 | 0.842 | 0.833 |
| sofa | 0.366 | 0.417 | 0.419 | 0.384 | 0.384 | 0.378 |
| train | 0.532 | 0.734 | 0.756 | 0.778 | 0.787 | 0.770 |
| tv | 0.528 | 0.650 | 0.670 | 0.599 | 0.607 | 0.599 |

Released-OVAM rows are the authors' masks on their original renders; every other row uses our regenerated images (same prompts and seeds) scored on the same released annotations.
