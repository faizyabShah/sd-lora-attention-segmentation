# SG13 target-conditioned region inference

Fixed SPEC32; literal raw region scores; nearest-expanded region masks. VOC selects, COCO transfers only.
Candidate score thresholds: 0–1 by .005; legacy controls retain .30–.70 by .025. No per-image normalization of SG13 region scores.

| Screen | Vote winner | Votes | OOF REF | OOF selection | Gain | Frozen | Promising target |
|---|---|---:|---:|---:|---:|---|---|
| R1 | W/REF | 5/5 | 0.7875 | 0.7875 | +0.0000 | W/REF | False |
| R2 | W/REF | 5/5 | 0.7875 | 0.7875 | +0.0000 | W/REF | False |
| R3 | W/REF | 5/5 | 0.7875 | 0.7875 | +0.0000 | W/REF | False |
| joint | W/REF | 5/5 | 0.7875 | 0.7875 | +0.0000 | W/REF | False |

Historical pixel-count drift is reported in summary.json. These are cross-environment diagnostics; all methods and the canonical SG4 comparator use the same newly generated inputs and partition. Same-generation linear and region-score identities are enforced for every image.


| Cell | VOC threshold | VOC mIoU | VOC recall / precision | COCO mIoU | COCO recall / precision |
|---|---:|---:|---:|---:|---:|
| W/REF | 0.425 | 0.7888 | 0.9280/0.8506 | 0.6178 | 0.8940/0.6784 |
| W/P1 | 0.450 | 0.8007 | 0.9239/0.8655 | 0.6249 | 0.8991/0.6835 |
| W/SG4/SPEC32_rmean | 0.450 | 0.7888 | 0.9278/0.8500 | 0.6093 | 0.9032/0.6665 |
| W/C/rho0p80 | 0.425 | 0.7969 | 0.9224/0.8626 | 0.6213 | 0.8766/0.6920 |
| W/SG13/R1 | 0.400 | 0.7685 | 0.9044/0.8519 | 0.6012 | 0.7538/0.7533 |
| W/SG13/R2 | 0.345 | 0.7537 | 0.8937/0.8446 | 0.5924 | 0.7663/0.7268 |
| W/SG13/R3 | 0.480 | 0.7039 | 0.8832/0.7985 | 0.5431 | 0.8292/0.6248 |
| Historical selected SG2 (Section 85) | 0.450 | 0.7984 | — | 0.6234 | — |

## R1: folds

| Fold | Winner | Threshold | Held-out chosen / REF |
|---:|---|---:|---:|
| 0 | W/REF | 0.425 | 0.7855/0.7855 |
| 1 | W/REF | 0.425 | 0.8113/0.8113 |
| 2 | W/REF | 0.425 | 0.7861/0.7861 |
| 3 | W/REF | 0.425 | 0.7805/0.7805 |
| 4 | W/REF | 0.450 | 0.7757/0.7757 |
## R2: folds

| Fold | Winner | Threshold | Held-out chosen / REF |
|---:|---|---:|---:|
| 0 | W/REF | 0.425 | 0.7855/0.7855 |
| 1 | W/REF | 0.425 | 0.8113/0.8113 |
| 2 | W/REF | 0.425 | 0.7861/0.7861 |
| 3 | W/REF | 0.425 | 0.7805/0.7805 |
| 4 | W/REF | 0.450 | 0.7757/0.7757 |
## R3: folds

| Fold | Winner | Threshold | Held-out chosen / REF |
|---:|---|---:|---:|
| 0 | W/REF | 0.425 | 0.7855/0.7855 |
| 1 | W/REF | 0.425 | 0.8113/0.8113 |
| 2 | W/REF | 0.425 | 0.7861/0.7861 |
| 3 | W/REF | 0.425 | 0.7805/0.7805 |
| 4 | W/REF | 0.450 | 0.7757/0.7757 |
## joint: folds

| Fold | Winner | Threshold | Held-out chosen / REF |
|---:|---|---:|---:|
| 0 | W/REF | 0.425 | 0.7855/0.7855 |
| 1 | W/REF | 0.425 | 0.8113/0.8113 |
| 2 | W/REF | 0.425 | 0.7861/0.7861 |
| 3 | W/REF | 0.425 | 0.7805/0.7805 |
| 4 | W/REF | 0.450 | 0.7757/0.7757 |

## voc: region labelling and majority oracle

Current oracle mean-image IoU: 0.8758; historical SG4 oracle: 0.8761. These are mean-image IoUs, not class-mIoU.

| Rule | FG regions/image | Region precision | Region recall | Missed positive | Selected negative | Actual mean-image IoU | Oracle gap | Gap fraction closed vs legacy SG4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W/SG13/R1 | 17.41 | 0.8620 | 0.9063 | 760 | 1177 | 0.7673 | 0.1085 | -0.2346 |
| W/SG13/R2 | 17.17 | 0.8574 | 0.8894 | 897 | 1200 | 0.7511 | 0.1247 | -0.4182 |
| W/SG13/R3 | 17.66 | 0.8227 | 0.8775 | 994 | 1534 | 0.7018 | 0.1740 | -0.9791 |

## coco: region labelling and majority oracle

Current oracle mean-image IoU: 0.7928; historical SG4 oracle: 0.7918. These are mean-image IoUs, not class-mIoU.

| Rule | FG regions/image | Region precision | Region recall | Missed positive | Selected negative | Actual mean-image IoU | Oracle gap | Gap fraction closed vs legacy SG4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W/SG13/R1 | 9.88 | 0.7889 | 0.7781 | 1572 | 1475 | 0.5971 | 0.1957 | -0.2033 |
| W/SG13/R2 | 10.18 | 0.7621 | 0.7739 | 1602 | 1712 | 0.5913 | 0.2015 | -0.2390 |
| W/SG13/R3 | 12.79 | 0.6573 | 0.8391 | 1140 | 3099 | 0.5485 | 0.2443 | -0.5022 |

## voc: W/SG13/R1, class IoUs

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8096 | -0.0255 |
| bicycle | 0.6640 | -0.0739 |
| bird | 0.8303 | -0.0102 |
| boat | 0.8479 | +0.0349 |
| bottle | 0.8719 | +0.0152 |
| bus | 0.9091 | +0.0009 |
| car | 0.8591 | +0.0339 |
| cat | 0.7918 | -0.0214 |
| chair | 0.5715 | -0.0577 |
| cow | 0.8903 | +0.0471 |
| dining-table | 0.4260 | -0.0273 |
| dog | 0.8894 | -0.0136 |
| horse | 0.8650 | -0.0022 |
| motorbike | 0.8379 | -0.0019 |
| person | 0.3128 | -0.3925 |
| potted-plant | 0.6848 | -0.0463 |
| sheep | 0.8966 | +0.0965 |
| sofa | 0.9206 | -0.0165 |
| train | 0.6885 | +0.0450 |
| tv | 0.8021 | +0.0090 |

## voc: W/SG13/R2, class IoUs

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8000 | -0.0351 |
| bicycle | 0.6291 | -0.1088 |
| bird | 0.7654 | -0.0751 |
| boat | 0.8168 | +0.0038 |
| bottle | 0.8236 | -0.0332 |
| bus | 0.9030 | -0.0053 |
| car | 0.8269 | +0.0017 |
| cat | 0.7975 | -0.0157 |
| chair | 0.5279 | -0.1012 |
| cow | 0.8673 | +0.0241 |
| dining-table | 0.4092 | -0.0440 |
| dog | 0.8813 | -0.0217 |
| horse | 0.8594 | -0.0078 |
| motorbike | 0.8244 | -0.0154 |
| person | 0.3629 | -0.3424 |
| potted-plant | 0.6878 | -0.0433 |
| sheep | 0.8840 | +0.0839 |
| sofa | 0.8967 | -0.0405 |
| train | 0.7071 | +0.0635 |
| tv | 0.8047 | +0.0117 |

## voc: W/SG13/R3, class IoUs

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.7226 | -0.1125 |
| bicycle | 0.6067 | -0.1312 |
| bird | 0.7347 | -0.1058 |
| boat | 0.7955 | -0.0175 |
| bottle | 0.7551 | -0.1016 |
| bus | 0.8759 | -0.0324 |
| car | 0.7975 | -0.0277 |
| cat | 0.6424 | -0.1708 |
| chair | 0.4127 | -0.2164 |
| cow | 0.7835 | -0.0597 |
| dining-table | 0.4063 | -0.0470 |
| dog | 0.8208 | -0.0822 |
| horse | 0.7921 | -0.0750 |
| motorbike | 0.7763 | -0.0636 |
| person | 0.4289 | -0.2765 |
| potted-plant | 0.6778 | -0.0533 |
| sheep | 0.6764 | -0.1237 |
| sofa | 0.8890 | -0.0481 |
| train | 0.6955 | +0.0519 |
| tv | 0.7889 | -0.0042 |

## coco: W/SG13/R1, class IoUs

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.6518 | +0.0872 |
| bicycle | 0.6151 | +0.0331 |
| bird | 0.6496 | -0.0662 |
| boat | 0.5852 | +0.0678 |
| bottle | 0.5426 | +0.0836 |
| bus | 0.8968 | -0.0233 |
| car | 0.5310 | -0.0363 |
| cat | 0.6848 | -0.1445 |
| chair | 0.4988 | +0.0797 |
| cow | 0.7727 | -0.0280 |
| dining table | 0.3312 | -0.0464 |
| dog | 0.7187 | -0.0259 |
| horse | 0.6954 | +0.0593 |
| motorbike | 0.6457 | +0.0286 |
| person | 0.0299 | -0.1899 |
| potted plant | 0.5872 | -0.0562 |
| sheep | 0.8030 | -0.0429 |
| sofa | 0.5722 | -0.0182 |
| train | 0.5948 | -0.0130 |
| tv | 0.6174 | -0.0798 |

## coco: W/SG13/R2, class IoUs

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5988 | +0.0342 |
| bicycle | 0.6003 | +0.0183 |
| bird | 0.5728 | -0.1430 |
| boat | 0.5713 | +0.0539 |
| bottle | 0.5367 | +0.0777 |
| bus | 0.8990 | -0.0211 |
| car | 0.5442 | -0.0231 |
| cat | 0.6466 | -0.1827 |
| chair | 0.4913 | +0.0722 |
| cow | 0.7611 | -0.0396 |
| dining table | 0.4493 | +0.0717 |
| dog | 0.6997 | -0.0450 |
| horse | 0.6629 | +0.0269 |
| motorbike | 0.6380 | +0.0209 |
| person | 0.0407 | -0.1790 |
| potted plant | 0.5421 | -0.1013 |
| sheep | 0.7721 | -0.0739 |
| sofa | 0.5334 | -0.0570 |
| train | 0.6192 | +0.0115 |
| tv | 0.6693 | -0.0279 |

## coco: W/SG13/R3, class IoUs

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5150 | -0.0496 |
| bicycle | 0.5090 | -0.0730 |
| bird | 0.5326 | -0.1832 |
| boat | 0.5107 | -0.0067 |
| bottle | 0.4231 | -0.0359 |
| bus | 0.8822 | -0.0379 |
| car | 0.5292 | -0.0381 |
| cat | 0.6348 | -0.1946 |
| chair | 0.3546 | -0.0645 |
| cow | 0.6727 | -0.1280 |
| dining table | 0.5243 | +0.1467 |
| dog | 0.6547 | -0.0899 |
| horse | 0.5698 | -0.0662 |
| motorbike | 0.5885 | -0.0285 |
| person | 0.1799 | -0.0398 |
| potted plant | 0.5164 | -0.1270 |
| sheep | 0.6435 | -0.2025 |
| sofa | 0.5001 | -0.0903 |
| train | 0.5861 | -0.0216 |
| tv | 0.5355 | -0.1617 |

Per-image region counts and OOF diagnostics are in the CSV/JSON files. The majority oracle uses GT and is diagnostic only. Negative gap fractions are retained; no value is clipped.
