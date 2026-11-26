# Where the draft's COCO-Cap hierarchy gain of +0.034 comes from (diagnostic)

`scripts/jc_trace_034.py`. Recomputed from the class-word maps of the `runs/15_token_ladder` capture (490 VOC-Sim, 707 COCO-Cap images, one generation per image). Fine-only = two-pass 64-px propagation; capped hierarchy = the frozen method's map. DenseCRF with the frozen settings at the original 17-threshold grid (0.30 to 0.70, step 0.025). Class-macro mIoU; Δ with image-bootstrap 95% interval (2000). Post hoc analysis of existing results; nothing is selected here.

## Thresholds

| Map | Full-VOC-Sim best threshold (this capture) | Original run | COCO-Cap best threshold (diagnostic) |
|---|---:|---:|---:|
| fine-only, raw | 0.450 | 0.450 | 0.525 |
| fine-only + DenseCRF | 0.425 | 0.425 | 0.525 |
| capped hierarchy, raw | 0.550 | 0.550 | 0.625 |
| capped hierarchy + DenseCRF | 0.550 | 0.550 | 0.600 |

## VOC-Sim: mIoU against the threshold

| threshold | fine-only raw | hierarchy raw | fine-only + DenseCRF | hierarchy + DenseCRF | Δ (+DenseCRF, same threshold) |
|---:|---:|---:|---:|---:|---:|
| 0.300 | 0.7624 | 0.7147 | 0.7846 | 0.7396 | -0.0450 |
| 0.325 | 0.7765 | 0.7299 | 0.7979 | 0.7549 | -0.0430 |
| 0.350 | 0.7874 | 0.7434 | 0.8075 | 0.7680 | -0.0395 |
| 0.375 | 0.7967 | 0.7564 | 0.8150 | 0.7799 | -0.0351 |
| 0.400 | 0.8041 | 0.7684 | 0.8203 | 0.7903 | -0.0300 |
| 0.425 | 0.8096 | 0.7802 | 0.8242 | 0.8006 | -0.0237 |
| 0.450 | 0.8114 | 0.7921 | 0.8239 | 0.8116 | -0.0124 |
| 0.475 | 0.8108 | 0.8016 | 0.8217 | 0.8196 | -0.0021 |
| 0.500 | 0.8072 | 0.8095 | 0.8158 | 0.8259 | +0.0101 |
| 0.525 | 0.8011 | 0.8142 | 0.8079 | 0.8290 | +0.0210 |
| 0.550 | 0.7911 | 0.8172 | 0.7957 | 0.8300 | +0.0343 |
| 0.575 | 0.7768 | 0.8172 | 0.7794 | 0.8279 | +0.0486 |
| 0.600 | 0.7574 | 0.8155 | 0.7576 | 0.8246 | +0.0671 |
| 0.625 | 0.7330 | 0.8106 | 0.7314 | 0.8177 | +0.0863 |
| 0.650 | 0.7048 | 0.8011 | 0.7013 | 0.8067 | +0.1054 |
| 0.675 | 0.6725 | 0.7861 | 0.6669 | 0.7893 | +0.1223 |
| 0.700 | 0.6361 | 0.7671 | 0.6281 | 0.7678 | +0.1397 |

## COCO-Cap: mIoU against the threshold

| threshold | fine-only raw | hierarchy raw | fine-only + DenseCRF | hierarchy + DenseCRF | Δ (+DenseCRF, same threshold) |
|---:|---:|---:|---:|---:|---:|
| 0.300 | 0.5501 | 0.5070 | 0.5684 | 0.5248 | -0.0436 |
| 0.325 | 0.5725 | 0.5275 | 0.5909 | 0.5464 | -0.0445 |
| 0.350 | 0.5944 | 0.5470 | 0.6128 | 0.5658 | -0.0470 |
| 0.375 | 0.6139 | 0.5665 | 0.6314 | 0.5859 | -0.0455 |
| 0.400 | 0.6307 | 0.5863 | 0.6466 | 0.6060 | -0.0406 |
| 0.425 | 0.6467 | 0.6047 | 0.6621 | 0.6234 | -0.0387 |
| 0.450 | 0.6599 | 0.6220 | 0.6736 | 0.6402 | -0.0335 |
| 0.475 | 0.6704 | 0.6391 | 0.6822 | 0.6563 | -0.0259 |
| 0.500 | 0.6765 | 0.6549 | 0.6870 | 0.6711 | -0.0159 |
| 0.525 | 0.6803 | 0.6693 | 0.6890 | 0.6845 | -0.0045 |
| 0.550 | 0.6769 | 0.6818 | 0.6841 | 0.6962 | +0.0121 |
| 0.575 | 0.6694 | 0.6907 | 0.6740 | 0.7034 | +0.0294 |
| 0.600 | 0.6601 | 0.6958 | 0.6633 | 0.7066 | +0.0434 |
| 0.625 | 0.6465 | 0.6976 | 0.6479 | 0.7063 | +0.0584 |
| 0.650 | 0.6303 | 0.6972 | 0.6301 | 0.7047 | +0.0745 |
| 0.675 | 0.6097 | 0.6916 | 0.6084 | 0.6975 | +0.0891 |
| 0.700 | 0.5838 | 0.6805 | 0.5814 | 0.6847 | +0.1032 |

## COCO-Cap: capped hierarchy minus fine-only under each convention

| Convention | hierarchy (thr) | fine-only (thr) | Δ [95% interval] | precision / recall: hierarchy; fine-only |
|---|---:|---:|---|---|
| **Draft's convention**: each map at the original run's full-VOC-Sim threshold, + DenseCRF | 0.6962 (0.550) | 0.6621 (0.425) | +0.0341 [+0.0227, +0.0450] | 0.747 / 0.885; 0.703 / 0.889 |
| Each map at this capture's full-VOC-Sim threshold, + DenseCRF | 0.6962 (0.550) | 0.6621 (0.425) | +0.0341 [+0.0227, +0.0450] | 0.747 / 0.885; 0.703 / 0.889 |
| Same threshold 0.55, + DenseCRF | 0.6962 (0.550) | 0.6841 (0.550) | +0.0121 [-0.0018, +0.0264] | 0.747 / 0.885; 0.795 / 0.804 |
| Same threshold 0.425, + DenseCRF | 0.6234 (0.425) | 0.6621 (0.425) | -0.0387 [-0.0502, -0.0265] | 0.648 / 0.921; 0.703 / 0.889 |
| Each map at its COCO-Cap best threshold, + DenseCRF (diagnostic, uses COCO-Cap labels) | 0.7066 (0.600) | 0.6890 (0.525) | +0.0176 [+0.0063, +0.0294] | 0.775 / 0.860; 0.780 / 0.827 |
| Raw, original full-VOC-Sim thresholds (0.55 / 0.45) | 0.6818 (0.550) | 0.6599 (0.450) | +0.0219 [+0.0117, +0.0321] | 0.731 / 0.889; 0.708 / 0.882 |
| Raw, same threshold 0.55 | 0.6818 (0.550) | 0.6769 (0.550) | +0.0049 [-0.0077, +0.0186] | 0.731 / 0.889; 0.783 / 0.809 |
| Raw, each at its COCO-Cap best threshold (diagnostic) | 0.6976 (0.625) | 0.6803 (0.525) | +0.0172 [+0.0074, +0.0275] | 0.776 / 0.846; 0.767 / 0.832 |

## VOC-Sim: the same conventions

| Convention | hierarchy (thr) | fine-only (thr) | Δ [95% interval] |
|---|---:|---:|---|
| **Draft's convention**: each map at the original run's full-VOC-Sim threshold, + DenseCRF | 0.8300 (0.550) | 0.8242 (0.425) | +0.0057 [-0.0009, +0.0122] |
| Each map at this capture's full-VOC-Sim threshold, + DenseCRF | 0.8300 (0.550) | 0.8242 (0.425) | +0.0057 [-0.0009, +0.0122] |
| Same threshold 0.55, + DenseCRF | 0.8300 (0.550) | 0.7957 (0.550) | +0.0343 [+0.0240, +0.0444] |
| Same threshold 0.425, + DenseCRF | 0.8006 (0.425) | 0.8242 (0.425) | -0.0237 [-0.0348, -0.0124] |
| 5-fold out-of-fold threshold per map, + DenseCRF | 0.8300 | 0.8226 | +0.0074 [+0.0009, +0.0142] |

## COCO-Cap per class, + DenseCRF: fine-only at 0.425, fine-only at 0.55, hierarchy at 0.55

| class | fine-only @0.425 | fine-only @0.55 | hierarchy @0.55 | Δ draft's convention | Δ same threshold |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.717 | 0.838 | 0.799 | +0.082 | -0.039 |
| bicycle | 0.570 | 0.611 | 0.626 | +0.056 | +0.015 |
| bird | 0.779 | 0.772 | 0.778 | -0.001 | +0.006 |
| boat | 0.563 | 0.642 | 0.586 | +0.023 | -0.056 |
| bottle | 0.496 | 0.534 | 0.504 | +0.008 | -0.030 |
| bus | 0.903 | 0.935 | 0.923 | +0.021 | -0.012 |
| car | 0.633 | 0.680 | 0.764 | +0.132 | +0.084 |
| cat | 0.855 | 0.837 | 0.870 | +0.015 | +0.033 |
| chair | 0.446 | 0.547 | 0.471 | +0.025 | -0.076 |
| cow | 0.855 | 0.814 | 0.874 | +0.018 | +0.060 |
| dining table | 0.470 | 0.314 | 0.453 | -0.018 | +0.139 |
| dog | 0.756 | 0.748 | 0.753 | -0.003 | +0.005 |
| horse | 0.674 | 0.704 | 0.717 | +0.043 | +0.014 |
| motorbike | 0.625 | 0.707 | 0.668 | +0.043 | -0.039 |
| person | 0.266 | 0.224 | 0.281 | +0.015 | +0.057 |
| potted plant | 0.697 | 0.771 | 0.743 | +0.046 | -0.027 |
| sheep | 0.874 | 0.865 | 0.874 | +0.000 | +0.009 |
| sofa | 0.662 | 0.735 | 0.699 | +0.037 | -0.037 |
| train | 0.687 | 0.730 | 0.753 | +0.066 | +0.024 |
| tv | 0.715 | 0.676 | 0.789 | +0.074 | +0.114 |
