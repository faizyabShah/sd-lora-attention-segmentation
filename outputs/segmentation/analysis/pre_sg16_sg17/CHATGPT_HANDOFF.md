# Pre-SG16 and Pre-SG17 diagnostics

Full details: `results.json`; individual region and image records: CSV files. Region AUC is exact. Pixel AUC/AP uses 1024 fixed bins and all changed pixels.
No classifier, threshold, segmentation gate, or final pipeline was selected.

## Pre-SG16: region separability

| Feature | VOC all ROC / AP | VOC ambiguous ROC / AP | VOC wrong-P1 ROC / AP | COCO all ROC / AP | COCO ambiguous ROC / AP |
|---|---:|---:|---:|---:|---:|
| p1_occupancy | 0.951 / 0.940 | 0.657 / 0.504 | 0.000 / 0.180 | 0.907 / 0.753 | 0.659 / 0.357 |
| occupancy_mean | 0.957 / 0.950 | 0.652 / 0.515 | 0.012 / 0.175 | 0.914 / 0.776 | 0.656 / 0.343 |
| ca_mean | 0.936 / 0.934 | 0.501 / 0.364 | 0.126 / 0.183 | 0.925 / 0.854 | 0.595 / 0.309 |
| p1_mean | 0.949 / 0.949 | 0.578 / 0.437 | 0.153 / 0.186 | 0.929 / 0.867 | 0.620 / 0.383 |
| obs_all_mean | 0.949 / 0.949 | 0.578 / 0.437 | 0.153 / 0.186 | 0.929 / 0.867 | 0.620 / 0.383 |
| obs_all_trim20 | 0.952 / 0.952 | 0.586 / 0.444 | 0.143 / 0.185 | 0.931 / 0.871 | 0.627 / 0.391 |
| obs_all_r_max | 0.477 / 0.500 | 0.507 / 0.376 | 0.596 / 0.379 | 0.513 / 0.319 | 0.545 / 0.258 |
| obs_top50_mean | 0.952 / 0.952 | 0.592 / 0.445 | 0.140 / 0.185 | 0.930 / 0.869 | 0.624 / 0.383 |
| obs_top50_trim20 | 0.952 / 0.952 | 0.595 / 0.450 | 0.140 / 0.185 | 0.931 / 0.870 | 0.627 / 0.387 |
| obs_top50_r_max | 0.484 / 0.506 | 0.501 / 0.363 | 0.570 / 0.353 | 0.522 / 0.324 | 0.538 / 0.251 |

Subset sizes: voc all=15680, ambiguous_p1=1603, wrong_p1=1610, priors_agree=15147, priors_disagree=533, coco all=22624, ambiguous_p1=2415, wrong_p1=3690, priors_agree=21630, priors_disagree=994

## Pre-SG17: second-pass changes

| Signal | VOC useful ROC / AP | COCO useful ROC / AP | COCO TP vs FP ROC / AP | COCO TP vs distant FP >16 ROC / AP |
|---|---:|---:|---:|---:|
| first_r_max | 0.551 / 0.670 | 0.530 / 0.413 | 0.517 / 0.324 | 0.497 / 0.353 |
| support_reliability | 0.520 / 0.668 | 0.538 / 0.456 | 0.603 / 0.420 | 0.589 / 0.451 |
| second_q_direction | 0.617 / 0.709 | 0.634 / 0.543 | 0.556 / 0.370 | 0.550 / 0.411 |
| second_median_direction | 0.582 / 0.685 | 0.629 / 0.539 | 0.544 / 0.342 | 0.539 / 0.391 |
| second_trim10_direction | 0.590 / 0.690 | 0.635 / 0.549 | 0.547 / 0.346 | 0.543 / 0.396 |
| top50_q_direction | 0.565 / 0.678 | 0.608 / 0.505 | 0.538 / 0.361 | 0.532 / 0.401 |
| top50_median_direction | 0.553 / 0.666 | 0.607 / 0.513 | 0.522 / 0.325 | 0.518 / 0.375 |
| top50_trim10_direction | 0.557 / 0.669 | 0.610 / 0.519 | 0.523 / 0.327 | 0.519 / 0.378 |

Exploratory VOC best reliability: **second_q_direction**, raw direction. This choice is for reading the diagnostic curve only.

Full endpoint tables, region per-class and area-bin AUCs, class-balanced AUCs, oracle-gain correlations, image-level correlations, changed-pixel score deciles and per-image records are in `results.json` and CSV files.
