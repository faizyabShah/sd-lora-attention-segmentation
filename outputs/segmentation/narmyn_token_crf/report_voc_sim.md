# VOC-Sim: OVAM optimized token and DenseCRF on the earlier final method

490 images, 20 classes (20/20 shards complete). Exploratory, pre-protocol; all settings frozen, nothing selected here.

| Method | mIoU | Mean image IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| released OVAM (unoptimized) | 0.7036 | 0.7111 | — | — |
| released OVAM (token-optimized) | 0.8247 | 0.8251 | — | — |
| `opt_all16/ca_only` | 0.7923 | 0.7768 | 0.848 | 0.924 |
| `opt_all16/sa_iter1` | 0.8372 | 0.8275 | 0.876 | 0.946 |
| `opt_all16/sa_iter1+crf` | 0.8553 | 0.8500 | 0.895 | 0.943 |
| `opt_all16/sa_iter4` | 0.8373 | 0.8292 | 0.871 | 0.952 |
| `opt_all16/sa_iter4+crf` | 0.8525 | 0.8471 | 0.889 | 0.947 |
| `opt_top4/ca_only` | 0.7880 | 0.7749 | 0.850 | 0.914 |
| `opt_top4/sa_iter1` | 0.8347 | 0.8257 | 0.875 | 0.943 |
| `opt_top4/sa_iter1+crf` | 0.8525 | 0.8479 | 0.895 | 0.939 |
| `opt_top4/sa_iter4` | 0.8367 | 0.8281 | 0.871 | 0.951 |
| `opt_top4/sa_iter4+crf` | 0.8521 | 0.8462 | 0.889 | 0.946 |
| `ovam_implemented` | 0.6945 | 0.7027 | 0.707 | 0.898 |
| `plain_top4/ca_only` | 0.6832 | 0.6862 | 0.841 | 0.761 |
| `plain_top4/sa_iter1` | 0.7802 | 0.7833 | 0.873 | 0.862 |
| `plain_top4/sa_iter1+crf` | 0.7957 | 0.8020 | 0.890 | 0.861 |
| `plain_top4/sa_iter4` | 0.8000 | 0.8023 | 0.869 | 0.897 |
| `plain_top4/sa_iter4+crf` | 0.8143 | 0.8194 | 0.886 | 0.892 |

## Paired per-image IoU differences (bootstrap over images, 20,000 resamples)

| A | B | Mean(A−B) | 95% CI | A better / B better |
|---|---|---:|---|---|
| `plain_top4/sa_iter1` | `ovam_implemented` | +0.0806 | [+0.0659, +0.0955] | 359 / 131 |
| `opt_top4/sa_iter1` | `ovam_implemented` | +0.1230 | [+0.1088, +0.1372] | 400 / 90 |
| `opt_top4/sa_iter1+crf` | `ovam_implemented` | +0.1451 | [+0.1310, +0.1592] | 440 / 50 |
| `opt_top4/sa_iter1` | `plain_top4/sa_iter1` | +0.0424 | [+0.0313, +0.0538] | 303 / 187 |
| `opt_all16/sa_iter1` | `opt_top4/sa_iter1` | +0.0018 | [-0.0005, +0.0041] | 250 / 240 |
| `plain_top4/sa_iter1+crf` | `plain_top4/sa_iter1` | +0.0187 | [+0.0153, +0.0221] | 363 / 127 |
| `opt_top4/sa_iter1+crf` | `opt_top4/sa_iter1` | +0.0221 | [+0.0185, +0.0258] | 374 / 116 |
| `opt_top4/sa_iter4` | `plain_top4/sa_iter4` | +0.0257 | [+0.0153, +0.0365] | 263 / 227 |
| `opt_top4/sa_iter4+crf` | `opt_top4/sa_iter4` | +0.0182 | [+0.0145, +0.0220] | 360 / 130 |
| `plain_top4/sa_iter1` | `released OVAM (token-optimized)` | -0.0418 | [-0.0566, -0.0269] | 168 / 322 |
| `opt_top4/sa_iter1` | `released OVAM (token-optimized)` | +0.0006 | [-0.0095, +0.0112] | 203 / 287 |
| `opt_top4/sa_iter1+crf` | `released OVAM (token-optimized)` | +0.0228 | [+0.0121, +0.0340] | 291 / 199 |
| `opt_all16/sa_iter1+crf` | `released OVAM (token-optimized)` | +0.0249 | [+0.0146, +0.0357] | 292 / 198 |
| `opt_top4/sa_iter4+crf` | `released OVAM (token-optimized)` | +0.0211 | [+0.0092, +0.0338] | 290 / 200 |

## Pooled class IoU

| Class | rel. OVAM (unoptimized) | rel. OVAM (token-optimized) | ovam_implemented | plain_top4/sa_iter1 | plain_top4/sa_iter1+crf | opt_top4/sa_iter1 | opt_top4/sa_iter1+crf | opt_all16/sa_iter1+crf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 0.717 | 0.736 | 0.712 | 0.801 | 0.856 | 0.852 | 0.898 | 0.898 |
| bicycle | 0.606 | 0.815 | 0.592 | 0.757 | 0.767 | 0.787 | 0.798 | 0.798 |
| bird | 0.768 | 0.884 | 0.777 | 0.877 | 0.926 | 0.862 | 0.921 | 0.922 |
| boat | 0.709 | 0.838 | 0.713 | 0.845 | 0.852 | 0.786 | 0.803 | 0.810 |
| bottle | 0.803 | 0.888 | 0.789 | 0.894 | 0.921 | 0.875 | 0.914 | 0.918 |
| bus | 0.913 | 0.931 | 0.906 | 0.914 | 0.919 | 0.889 | 0.893 | 0.903 |
| car | 0.829 | 0.906 | 0.835 | 0.789 | 0.787 | 0.860 | 0.864 | 0.875 |
| cat | 0.723 | 0.931 | 0.722 | 0.679 | 0.689 | 0.860 | 0.870 | 0.869 |
| chair | 0.410 | 0.445 | 0.402 | 0.629 | 0.680 | 0.660 | 0.727 | 0.712 |
| cow | 0.898 | 0.925 | 0.897 | 0.892 | 0.907 | 0.937 | 0.960 | 0.951 |
| dining-table | 0.258 | 0.277 | 0.244 | 0.427 | 0.428 | 0.445 | 0.446 | 0.452 |
| dog | 0.775 | 0.929 | 0.753 | 0.852 | 0.863 | 0.907 | 0.923 | 0.928 |
| horse | 0.803 | 0.914 | 0.803 | 0.862 | 0.883 | 0.892 | 0.921 | 0.926 |
| motorbike | 0.780 | 0.889 | 0.774 | 0.873 | 0.869 | 0.879 | 0.876 | 0.880 |
| person | 0.561 | 0.882 | 0.569 | 0.623 | 0.636 | 0.827 | 0.833 | 0.866 |
| potted-plant | 0.671 | 0.856 | 0.574 | 0.563 | 0.568 | 0.747 | 0.749 | 0.762 |
| sheep | 0.820 | 0.940 | 0.818 | 0.839 | 0.854 | 0.917 | 0.941 | 0.951 |
| sofa | 0.744 | 0.869 | 0.712 | 0.920 | 0.924 | 0.957 | 0.964 | 0.969 |
| train | 0.586 | 0.895 | 0.596 | 0.805 | 0.820 | 0.888 | 0.881 | 0.866 |
| tv | 0.700 | 0.746 | 0.702 | 0.764 | 0.765 | 0.868 | 0.868 | 0.851 |

Released-OVAM rows are the authors' masks on their original renders; every other row uses our regenerated images (same prompts and seeds) scored on the same released annotations.
