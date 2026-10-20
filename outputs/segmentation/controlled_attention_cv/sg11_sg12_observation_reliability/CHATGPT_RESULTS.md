# SG11 / SG12 / SG-Diag

AUDITED RECOVERY: historical pixel-level reproduction failed across hardware; these are internally consistent V100 comparisons. See validation_recovery and count_audit.json; independent review required.

W/REF is one-pass without DenseCRF. Existing final FTF is two passes + DenseCRF (COCO 0.6370). This screen does not replace FTF.

VOC-Sim selects; COCO-Cap transfers only. Class words, no optimized tokens.

| Family | Vote winner | Votes | OOF REF | OOF selection | Delta | Frozen | ChatGPT VOC target |
|---|---|---:|---:|---:|---:|---|---|
| SG11 | W/SG11/trim20/rho0p80 | 5/5 | 0.7873 | 0.7984 | +0.0111 | W/SG11/trim20/rho0p80 | True |
| SG12 | W/SG12/top50 | 5/5 | 0.7873 | 0.7967 | +0.0094 | W/SG12/top50 | False |

Historical SG8 W-C selected rho=.80 in 4/5 folds: OOF 0.797014 (REF 0.787508). SG8 W-A split its votes and did not freeze a robust aggregator. These historical results are context, not new selection.

## All cells and controls

| Cell | VOC threshold | VOC mIoU | Δ REF | VOC recall / precision | COCO mIoU | Δ REF | COCO recall / precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| W/REF | 0.425 | 0.7886 | +0.0000 | 0.9063 / 0.8529 | 0.6172 | +0.0000 | 0.8698 / 0.6600 |
| W/C/rho0p80 | 0.425 | 0.7967 | +0.0081 | 0.9045 / 0.8643 | 0.6205 | +0.0033 | 0.8545 / 0.6686 |
| W/A/trim20 | 0.450 | 0.7950 | +0.0063 | 0.8915 / 0.8704 | 0.6288 | +0.0116 | 0.8585 / 0.6791 |
| W/A/median | 0.425 | 0.7969 | +0.0083 | 0.9106 / 0.8580 | 0.6226 | +0.0054 | 0.8760 / 0.6600 |
| W/SG11/median/rho0p70 | 0.425 | 0.7976 | +0.0090 | 0.9116 / 0.8579 | 0.6144 | -0.0028 | 0.8714 / 0.6530 |
| W/SG11/median/rho0p80 | 0.425 | 0.7984 | +0.0097 | 0.9048 / 0.8656 | 0.6221 | +0.0049 | 0.8564 / 0.6690 |
| W/SG11/median/rho0p90 | 0.400 | 0.7912 | +0.0026 | 0.9117 / 0.8523 | 0.6155 | -0.0016 | 0.8537 / 0.6645 |
| W/SG11/trim20/rho0p70 | 0.450 | 0.7980 | +0.0094 | 0.8940 / 0.8721 | 0.6225 | +0.0053 | 0.8535 / 0.6741 |
| W/SG11/trim20/rho0p80 | 0.425 | 0.7984 | +0.0098 | 0.9051 / 0.8654 | 0.6219 | +0.0048 | 0.8559 / 0.6691 |
| W/SG11/trim20/rho0p90 | 0.400 | 0.7908 | +0.0022 | 0.9115 / 0.8520 | 0.6152 | -0.0019 | 0.8532 / 0.6645 |
| W/SG12/top50 | 0.450 | 0.7986 | +0.0099 | 0.9019 / 0.8668 | 0.6272 | +0.0100 | 0.8715 / 0.6692 |
| W/SG12/top70 | 0.425 | 0.7958 | +0.0072 | 0.9113 / 0.8570 | 0.6200 | +0.0028 | 0.8795 / 0.6571 |
| W/SG12/top80 | 0.425 | 0.7941 | +0.0055 | 0.9072 / 0.8582 | 0.6219 | +0.0047 | 0.8749 / 0.6620 |
| W/SG12/top90 | 0.425 | 0.7910 | +0.0024 | 0.9054 / 0.8561 | 0.6200 | +0.0028 | 0.8718 / 0.6623 |

## SG11: fold winners

| Fold | Winner | Threshold | Held-out chosen / REF |
|---:|---|---:|---:|
| 0 | W/SG11/trim20/rho0p80 | 0.425 | 0.7981 / 0.7854 |
| 1 | W/SG11/trim20/rho0p80 | 0.425 | 0.8152 / 0.8115 |
| 2 | W/SG11/trim20/rho0p80 | 0.425 | 0.7984 / 0.7859 |
| 3 | W/SG11/trim20/rho0p80 | 0.425 | 0.7925 / 0.7802 |
| 4 | W/SG11/trim20/rho0p80 | 0.425 | 0.7859 / 0.7752 |

## SG12: fold winners

| Fold | Winner | Threshold | Held-out chosen / REF |
|---:|---|---:|---:|
| 0 | W/SG12/top50 | 0.450 | 0.7980 / 0.7854 |
| 1 | W/SG12/top50 | 0.450 | 0.8211 / 0.8115 |
| 2 | W/SG12/top50 | 0.450 | 0.7957 / 0.7859 |
| 3 | W/SG12/top50 | 0.450 | 0.7904 / 0.7802 |
| 4 | W/SG12/top50 | 0.475 | 0.7812 / 0.7752 |

## SG-Diag

AUCs use all appreciable binary flips and 1024-bin histograms. PR-AUC is step average precision. GT enters diagnostics only. Both feature orientations are reported; these scores cannot select methods. No fitted combination or classifier.

### voc, fixed diagnostic threshold 0.425

Eligible flips: 9557571; events: {'tp_addition': 4566870, 'fp_addition': 1839738, 'correct_fg_removal': 2660910, 'incorrect_fg_removal': 490053}.

| Signal (higher predicts useful) | Addition ROC / PR | All-change ROC / PR |
|---|---:|---:|
| r_up | 0.5878 / 0.7700 | 0.4544 / 0.7570 |
| r_down | 0.4122 / 0.6743 | 0.5456 / 0.8123 |
| consensus | 0.5845 / 0.7682 | 0.6264 / 0.8247 |
| directional_consensus | 0.5878 / 0.7700 | 0.6293 / 0.8266 |
| std | 0.4363 / 0.6978 | 0.5277 / 0.7797 |
| mad | 0.4217 / 0.6754 | 0.5055 / 0.7568 |
| iqr | 0.4281 / 0.6853 | 0.5125 / 0.7643 |
| median_minus_seed | 0.5742 / 0.7558 | 0.4339 / 0.7438 |
| mean_minus_seed | 0.5643 / 0.7484 | 0.4294 / 0.7393 |
| mean_median_disagreement | 0.4804 / 0.7266 | 0.5490 / 0.7940 |
| fraction_close_to_seed | 0.4275 / 0.6883 | 0.3535 / 0.6872 |
| propagation_magnitude | 0.6448 / 0.8167 | 0.6824 / 0.8671 |
| consensus_agreement_combination | 0.5618 / 0.7662 | 0.5491 / 0.7834 |
### voc, fixed diagnostic threshold 0.45

Eligible flips: 10001770; events: {'tp_addition': 5466696, 'fp_addition': 1785433, 'correct_fg_removal': 2255857, 'incorrect_fg_removal': 493784}.

| Signal (higher predicts useful) | Addition ROC / PR | All-change ROC / PR |
|---|---:|---:|
| r_up | 0.5910 / 0.8078 | 0.5056 / 0.7941 |
| r_down | 0.4090 / 0.7164 | 0.4944 / 0.7936 |
| consensus | 0.5752 / 0.8014 | 0.5982 / 0.8249 |
| directional_consensus | 0.5910 / 0.8078 | 0.6084 / 0.8287 |
| std | 0.4513 / 0.7453 | 0.5215 / 0.7902 |
| mad | 0.4233 / 0.7179 | 0.4938 / 0.7667 |
| iqr | 0.4331 / 0.7290 | 0.5011 / 0.7735 |
| median_minus_seed | 0.5816 / 0.7974 | 0.4861 / 0.7832 |
| mean_minus_seed | 0.5704 / 0.7906 | 0.4801 / 0.7784 |
| mean_median_disagreement | 0.4988 / 0.7739 | 0.5473 / 0.8053 |
| fraction_close_to_seed | 0.4321 / 0.7376 | 0.3811 / 0.7275 |
| propagation_magnitude | 0.6439 / 0.8458 | 0.6812 / 0.8760 |
| consensus_agreement_combination | 0.5565 / 0.7968 | 0.5406 / 0.7908 |
### coco, fixed diagnostic threshold 0.425

Eligible flips: 18885554; events: {'tp_addition': 6832619, 'fp_addition': 9342392, 'correct_fg_removal': 2313983, 'incorrect_fg_removal': 396560}.

| Signal (higher predicts useful) | Addition ROC / PR | All-change ROC / PR |
|---|---:|---:|
| r_up | 0.5815 / 0.4838 | 0.4559 / 0.4785 |
| r_down | 0.4184 / 0.3671 | 0.5441 / 0.5924 |
| consensus | 0.4785 / 0.4173 | 0.5528 / 0.5447 |
| directional_consensus | 0.5815 / 0.4838 | 0.6491 / 0.6217 |
| std | 0.5957 / 0.5330 | 0.6279 / 0.6175 |
| mad | 0.5612 / 0.4833 | 0.5901 / 0.5675 |
| iqr | 0.5682 / 0.4986 | 0.5973 / 0.5795 |
| median_minus_seed | 0.6022 / 0.5206 | 0.4652 / 0.5035 |
| mean_minus_seed | 0.5852 / 0.5039 | 0.4530 / 0.4911 |
| mean_median_disagreement | 0.6004 / 0.5409 | 0.6291 / 0.6191 |
| fraction_close_to_seed | 0.4466 / 0.3879 | 0.3745 / 0.4112 |
| propagation_magnitude | 0.5901 / 0.4824 | 0.5543 / 0.5227 |
| consensus_agreement_combination | 0.5250 / 0.4398 | 0.5569 / 0.5182 |
### coco, fixed diagnostic threshold 0.45

Eligible flips: 19113793; events: {'tp_addition': 7834812, 'fp_addition': 9019735, 'correct_fg_removal': 1915640, 'incorrect_fg_removal': 343606}.

| Signal (higher predicts useful) | Addition ROC / PR | All-change ROC / PR |
|---|---:|---:|
| r_up | 0.5852 / 0.5277 | 0.4895 / 0.5216 |
| r_down | 0.4148 / 0.4036 | 0.5105 / 0.5753 |
| consensus | 0.4549 / 0.4431 | 0.5203 / 0.5479 |
| directional_consensus | 0.5852 / 0.5277 | 0.6383 / 0.6368 |
| std | 0.6126 / 0.5851 | 0.6366 / 0.6455 |
| mad | 0.5708 / 0.5306 | 0.5944 / 0.5944 |
| iqr | 0.5792 / 0.5464 | 0.6022 / 0.6063 |
| median_minus_seed | 0.6044 / 0.5612 | 0.4994 / 0.5465 |
| mean_minus_seed | 0.5847 / 0.5429 | 0.4840 / 0.5318 |
| mean_median_disagreement | 0.6152 / 0.5936 | 0.6358 / 0.6471 |
| fraction_close_to_seed | 0.4607 / 0.4368 | 0.4002 / 0.4503 |
| propagation_magnitude | 0.5761 / 0.5128 | 0.5509 / 0.5438 |
| consensus_agreement_combination | 0.5271 / 0.4814 | 0.5542 / 0.5413 |

## Class IoUs and deltas for every variant

### voc: W/REF

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8353 | +0.0000 |
| bicycle | 0.7374 | +0.0000 |
| bird | 0.8403 | +0.0000 |
| boat | 0.8125 | +0.0000 |
| bottle | 0.8561 | +0.0000 |
| bus | 0.9089 | +0.0000 |
| car | 0.8254 | +0.0000 |
| cat | 0.8132 | +0.0000 |
| chair | 0.6291 | +0.0000 |
| cow | 0.8427 | +0.0000 |
| dining-table | 0.4527 | +0.0000 |
| dog | 0.9029 | +0.0000 |
| horse | 0.8656 | +0.0000 |
| motorbike | 0.8398 | +0.0000 |
| person | 0.7057 | +0.0000 |
| potted-plant | 0.7311 | +0.0000 |
| sheep | 0.8001 | +0.0000 |
| sofa | 0.9368 | +0.0000 |
| train | 0.6433 | +0.0000 |
| tv | 0.7932 | +0.0000 |
### voc: W/C/rho0p80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8419 | +0.0066 |
| bicycle | 0.7292 | -0.0082 |
| bird | 0.8421 | +0.0017 |
| boat | 0.8441 | +0.0316 |
| bottle | 0.8646 | +0.0085 |
| bus | 0.9190 | +0.0101 |
| car | 0.8241 | -0.0013 |
| cat | 0.8274 | +0.0143 |
| chair | 0.6681 | +0.0390 |
| cow | 0.8508 | +0.0082 |
| dining-table | 0.4685 | +0.0158 |
| dog | 0.8997 | -0.0031 |
| horse | 0.8668 | +0.0013 |
| motorbike | 0.8334 | -0.0065 |
| person | 0.6952 | -0.0106 |
| potted-plant | 0.7172 | -0.0139 |
| sheep | 0.8423 | +0.0421 |
| sofa | 0.9289 | -0.0080 |
| train | 0.6795 | +0.0361 |
| tv | 0.7921 | -0.0011 |
### voc: W/A/trim20

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8552 | +0.0199 |
| bicycle | 0.7468 | +0.0093 |
| bird | 0.8583 | +0.0180 |
| boat | 0.8409 | +0.0284 |
| bottle | 0.8655 | +0.0094 |
| bus | 0.9215 | +0.0126 |
| car | 0.8438 | +0.0184 |
| cat | 0.7919 | -0.0213 |
| chair | 0.6733 | +0.0442 |
| cow | 0.8290 | -0.0137 |
| dining-table | 0.4683 | +0.0156 |
| dog | 0.8977 | -0.0052 |
| horse | 0.8719 | +0.0064 |
| motorbike | 0.8528 | +0.0129 |
| person | 0.6893 | -0.0165 |
| potted-plant | 0.7146 | -0.0165 |
| sheep | 0.7788 | -0.0213 |
| sofa | 0.9451 | +0.0083 |
| train | 0.6743 | +0.0310 |
| tv | 0.7802 | -0.0130 |
### voc: W/A/median

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8489 | +0.0136 |
| bicycle | 0.7402 | +0.0027 |
| bird | 0.8520 | +0.0117 |
| boat | 0.8286 | +0.0161 |
| bottle | 0.8611 | +0.0050 |
| bus | 0.9120 | +0.0030 |
| car | 0.8291 | +0.0037 |
| cat | 0.8202 | +0.0071 |
| chair | 0.6614 | +0.0323 |
| cow | 0.8631 | +0.0204 |
| dining-table | 0.4521 | -0.0006 |
| dog | 0.9047 | +0.0019 |
| horse | 0.8700 | +0.0045 |
| motorbike | 0.8413 | +0.0015 |
| person | 0.7245 | +0.0188 |
| potted-plant | 0.7354 | +0.0043 |
| sheep | 0.8122 | +0.0120 |
| sofa | 0.9348 | -0.0021 |
| train | 0.6521 | +0.0087 |
| tv | 0.7941 | +0.0009 |
### voc: W/SG11/median/rho0p70

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8494 | +0.0141 |
| bicycle | 0.7361 | -0.0013 |
| bird | 0.8465 | +0.0062 |
| boat | 0.8331 | +0.0206 |
| bottle | 0.8577 | +0.0016 |
| bus | 0.9153 | +0.0064 |
| car | 0.8300 | +0.0046 |
| cat | 0.8203 | +0.0072 |
| chair | 0.6709 | +0.0418 |
| cow | 0.8599 | +0.0172 |
| dining-table | 0.4489 | -0.0038 |
| dog | 0.9046 | +0.0017 |
| horse | 0.8688 | +0.0033 |
| motorbike | 0.8343 | -0.0055 |
| person | 0.7179 | +0.0122 |
| potted-plant | 0.7337 | +0.0027 |
| sheep | 0.8477 | +0.0475 |
| sofa | 0.9307 | -0.0061 |
| train | 0.6590 | +0.0156 |
| tv | 0.7875 | -0.0057 |
### voc: W/SG11/median/rho0p80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8469 | +0.0117 |
| bicycle | 0.7298 | -0.0077 |
| bird | 0.8447 | +0.0044 |
| boat | 0.8451 | +0.0326 |
| bottle | 0.8650 | +0.0089 |
| bus | 0.9189 | +0.0099 |
| car | 0.8269 | +0.0015 |
| cat | 0.8267 | +0.0136 |
| chair | 0.6777 | +0.0485 |
| cow | 0.8466 | +0.0040 |
| dining-table | 0.4683 | +0.0156 |
| dog | 0.9020 | -0.0009 |
| horse | 0.8677 | +0.0021 |
| motorbike | 0.8356 | -0.0042 |
| person | 0.6973 | -0.0084 |
| potted-plant | 0.7222 | -0.0089 |
| sheep | 0.8409 | +0.0407 |
| sofa | 0.9317 | -0.0052 |
| train | 0.6793 | +0.0360 |
| tv | 0.7938 | +0.0006 |
### voc: W/SG11/median/rho0p90

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8211 | -0.0141 |
| bicycle | 0.7116 | -0.0259 |
| bird | 0.8333 | -0.0070 |
| boat | 0.8338 | +0.0214 |
| bottle | 0.8547 | -0.0015 |
| bus | 0.9134 | +0.0045 |
| car | 0.8296 | +0.0042 |
| cat | 0.8336 | +0.0204 |
| chair | 0.6487 | +0.0196 |
| cow | 0.8655 | +0.0229 |
| dining-table | 0.4534 | +0.0007 |
| dog | 0.8933 | -0.0096 |
| horse | 0.8579 | -0.0076 |
| motorbike | 0.8158 | -0.0240 |
| person | 0.6969 | -0.0088 |
| potted-plant | 0.7081 | -0.0230 |
| sheep | 0.8748 | +0.0747 |
| sofa | 0.9220 | -0.0148 |
| train | 0.6623 | +0.0189 |
| tv | 0.7943 | +0.0010 |
### voc: W/SG11/trim20/rho0p70

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8553 | +0.0200 |
| bicycle | 0.7468 | +0.0093 |
| bird | 0.8495 | +0.0092 |
| boat | 0.8471 | +0.0346 |
| bottle | 0.8664 | +0.0103 |
| bus | 0.9242 | +0.0153 |
| car | 0.8431 | +0.0177 |
| cat | 0.7930 | -0.0202 |
| chair | 0.6880 | +0.0589 |
| cow | 0.8389 | -0.0038 |
| dining-table | 0.4688 | +0.0161 |
| dog | 0.8985 | -0.0044 |
| horse | 0.8721 | +0.0065 |
| motorbike | 0.8479 | +0.0080 |
| person | 0.6818 | -0.0239 |
| potted-plant | 0.7161 | -0.0150 |
| sheep | 0.8224 | +0.0222 |
| sofa | 0.9410 | +0.0042 |
| train | 0.6881 | +0.0448 |
| tv | 0.7714 | -0.0218 |
### voc: W/SG11/trim20/rho0p80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8464 | +0.0112 |
| bicycle | 0.7298 | -0.0076 |
| bird | 0.8455 | +0.0052 |
| boat | 0.8453 | +0.0328 |
| bottle | 0.8664 | +0.0103 |
| bus | 0.9188 | +0.0099 |
| car | 0.8261 | +0.0007 |
| cat | 0.8277 | +0.0145 |
| chair | 0.6767 | +0.0476 |
| cow | 0.8485 | +0.0058 |
| dining-table | 0.4686 | +0.0159 |
| dog | 0.9013 | -0.0016 |
| horse | 0.8678 | +0.0022 |
| motorbike | 0.8353 | -0.0046 |
| person | 0.6972 | -0.0085 |
| potted-plant | 0.7213 | -0.0098 |
| sheep | 0.8410 | +0.0408 |
| sofa | 0.9309 | -0.0059 |
| train | 0.6796 | +0.0363 |
| tv | 0.7939 | +0.0007 |
### voc: W/SG11/trim20/rho0p90

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8206 | -0.0147 |
| bicycle | 0.7117 | -0.0258 |
| bird | 0.8338 | -0.0065 |
| boat | 0.8340 | +0.0215 |
| bottle | 0.8550 | -0.0011 |
| bus | 0.9137 | +0.0048 |
| car | 0.8290 | +0.0036 |
| cat | 0.8330 | +0.0198 |
| chair | 0.6457 | +0.0166 |
| cow | 0.8655 | +0.0228 |
| dining-table | 0.4534 | +0.0007 |
| dog | 0.8927 | -0.0102 |
| horse | 0.8574 | -0.0081 |
| motorbike | 0.8152 | -0.0247 |
| person | 0.6964 | -0.0093 |
| potted-plant | 0.7065 | -0.0246 |
| sheep | 0.8754 | +0.0753 |
| sofa | 0.9213 | -0.0155 |
| train | 0.6621 | +0.0188 |
| tv | 0.7938 | +0.0005 |
### voc: W/SG12/top50

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8552 | +0.0199 |
| bicycle | 0.7453 | +0.0079 |
| bird | 0.8565 | +0.0162 |
| boat | 0.8349 | +0.0224 |
| bottle | 0.8651 | +0.0089 |
| bus | 0.9149 | +0.0060 |
| car | 0.8332 | +0.0077 |
| cat | 0.8089 | -0.0043 |
| chair | 0.6775 | +0.0483 |
| cow | 0.8456 | +0.0030 |
| dining-table | 0.4615 | +0.0088 |
| dog | 0.9021 | -0.0007 |
| horse | 0.8775 | +0.0119 |
| motorbike | 0.8501 | +0.0102 |
| person | 0.7226 | +0.0169 |
| potted-plant | 0.7340 | +0.0029 |
| sheep | 0.7845 | -0.0157 |
| sofa | 0.9477 | +0.0109 |
| train | 0.6693 | +0.0260 |
| tv | 0.7848 | -0.0084 |
### voc: W/SG12/top70

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8417 | +0.0064 |
| bicycle | 0.7406 | +0.0032 |
| bird | 0.8519 | +0.0116 |
| boat | 0.8216 | +0.0092 |
| bottle | 0.8582 | +0.0021 |
| bus | 0.9094 | +0.0005 |
| car | 0.8265 | +0.0010 |
| cat | 0.8233 | +0.0102 |
| chair | 0.6511 | +0.0220 |
| cow | 0.8572 | +0.0146 |
| dining-table | 0.4516 | -0.0011 |
| dog | 0.9063 | +0.0034 |
| horse | 0.8726 | +0.0070 |
| motorbike | 0.8423 | +0.0025 |
| person | 0.7261 | +0.0204 |
| potted-plant | 0.7404 | +0.0093 |
| sheep | 0.8090 | +0.0089 |
| sofa | 0.9395 | +0.0027 |
| train | 0.6517 | +0.0084 |
| tv | 0.7950 | +0.0018 |
### voc: W/SG12/top80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8397 | +0.0044 |
| bicycle | 0.7416 | +0.0042 |
| bird | 0.8503 | +0.0100 |
| boat | 0.8226 | +0.0102 |
| bottle | 0.8578 | +0.0017 |
| bus | 0.9127 | +0.0038 |
| car | 0.8309 | +0.0055 |
| cat | 0.8153 | +0.0021 |
| chair | 0.6441 | +0.0150 |
| cow | 0.8518 | +0.0091 |
| dining-table | 0.4543 | +0.0015 |
| dog | 0.9050 | +0.0021 |
| horse | 0.8715 | +0.0060 |
| motorbike | 0.8463 | +0.0065 |
| person | 0.7137 | +0.0080 |
| potted-plant | 0.7321 | +0.0010 |
| sheep | 0.8037 | +0.0035 |
| sofa | 0.9391 | +0.0022 |
| train | 0.6565 | +0.0132 |
| tv | 0.7935 | +0.0002 |
### voc: W/SG12/top90

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.8374 | +0.0022 |
| bicycle | 0.7393 | +0.0019 |
| bird | 0.8450 | +0.0047 |
| boat | 0.8183 | +0.0059 |
| bottle | 0.8570 | +0.0009 |
| bus | 0.9124 | +0.0035 |
| car | 0.8301 | +0.0047 |
| cat | 0.8119 | -0.0012 |
| chair | 0.6356 | +0.0064 |
| cow | 0.8444 | +0.0017 |
| dining-table | 0.4559 | +0.0032 |
| dog | 0.9035 | +0.0007 |
| horse | 0.8680 | +0.0024 |
| motorbike | 0.8438 | +0.0039 |
| person | 0.7060 | +0.0003 |
| potted-plant | 0.7300 | -0.0010 |
| sheep | 0.8004 | +0.0003 |
| sofa | 0.9379 | +0.0010 |
| train | 0.6505 | +0.0072 |
| tv | 0.7918 | -0.0014 |
### coco: W/REF

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5639 | +0.0000 |
| bicycle | 0.5813 | +0.0000 |
| bird | 0.7128 | +0.0000 |
| boat | 0.5175 | +0.0000 |
| bottle | 0.4566 | +0.0000 |
| bus | 0.9201 | +0.0000 |
| car | 0.5683 | +0.0000 |
| cat | 0.8287 | +0.0000 |
| chair | 0.4190 | +0.0000 |
| cow | 0.8006 | +0.0000 |
| dining table | 0.3742 | +0.0000 |
| dog | 0.7451 | +0.0000 |
| horse | 0.6355 | +0.0000 |
| motorbike | 0.6175 | +0.0000 |
| person | 0.2190 | +0.0000 |
| potted plant | 0.6430 | +0.0000 |
| sheep | 0.8456 | +0.0000 |
| sofa | 0.5902 | +0.0000 |
| train | 0.6075 | +0.0000 |
| tv | 0.6969 | +0.0000 |
### coco: W/C/rho0p80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5934 | +0.0294 |
| bicycle | 0.5988 | +0.0174 |
| bird | 0.7058 | -0.0071 |
| boat | 0.5487 | +0.0312 |
| bottle | 0.4830 | +0.0264 |
| bus | 0.9250 | +0.0050 |
| car | 0.5459 | -0.0224 |
| cat | 0.8341 | +0.0054 |
| chair | 0.4328 | +0.0138 |
| cow | 0.8088 | +0.0082 |
| dining table | 0.3302 | -0.0440 |
| dog | 0.7427 | -0.0024 |
| horse | 0.6476 | +0.0122 |
| motorbike | 0.6286 | +0.0111 |
| person | 0.2197 | +0.0007 |
| potted plant | 0.6505 | +0.0075 |
| sheep | 0.8574 | +0.0119 |
| sofa | 0.5681 | -0.0221 |
| train | 0.6034 | -0.0041 |
| tv | 0.6858 | -0.0112 |
### coco: W/A/trim20

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5960 | +0.0321 |
| bicycle | 0.6011 | +0.0198 |
| bird | 0.7216 | +0.0087 |
| boat | 0.5397 | +0.0222 |
| bottle | 0.4818 | +0.0253 |
| bus | 0.9280 | +0.0079 |
| car | 0.5685 | +0.0002 |
| cat | 0.8324 | +0.0037 |
| chair | 0.4338 | +0.0148 |
| cow | 0.8095 | +0.0089 |
| dining table | 0.3512 | -0.0230 |
| dog | 0.7495 | +0.0045 |
| horse | 0.6596 | +0.0241 |
| motorbike | 0.6405 | +0.0230 |
| person | 0.2277 | +0.0087 |
| potted plant | 0.6516 | +0.0086 |
| sheep | 0.8646 | +0.0190 |
| sofa | 0.6002 | +0.0100 |
| train | 0.6290 | +0.0215 |
| tv | 0.6899 | -0.0071 |
### coco: W/A/median

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5756 | +0.0117 |
| bicycle | 0.5887 | +0.0073 |
| bird | 0.7189 | +0.0060 |
| boat | 0.5203 | +0.0029 |
| bottle | 0.4705 | +0.0139 |
| bus | 0.9141 | -0.0060 |
| car | 0.5578 | -0.0105 |
| cat | 0.8411 | +0.0124 |
| chair | 0.4281 | +0.0091 |
| cow | 0.8193 | +0.0187 |
| dining table | 0.3807 | +0.0065 |
| dog | 0.7483 | +0.0032 |
| horse | 0.6483 | +0.0128 |
| motorbike | 0.6231 | +0.0055 |
| person | 0.2379 | +0.0189 |
| potted plant | 0.6365 | -0.0065 |
| sheep | 0.8640 | +0.0185 |
| sofa | 0.5793 | -0.0109 |
| train | 0.6058 | -0.0017 |
| tv | 0.6934 | -0.0035 |
### coco: W/SG11/median/rho0p70

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5713 | +0.0074 |
| bicycle | 0.5799 | -0.0014 |
| bird | 0.7115 | -0.0013 |
| boat | 0.5148 | -0.0026 |
| bottle | 0.4688 | +0.0122 |
| bus | 0.9128 | -0.0073 |
| car | 0.5503 | -0.0179 |
| cat | 0.8414 | +0.0127 |
| chair | 0.4111 | -0.0079 |
| cow | 0.8137 | +0.0131 |
| dining table | 0.3605 | -0.0137 |
| dog | 0.7461 | +0.0010 |
| horse | 0.6453 | +0.0098 |
| motorbike | 0.6165 | -0.0011 |
| person | 0.2338 | +0.0148 |
| potted plant | 0.6307 | -0.0124 |
| sheep | 0.8599 | +0.0144 |
| sofa | 0.5669 | -0.0233 |
| train | 0.5961 | -0.0114 |
| tv | 0.6561 | -0.0408 |
### coco: W/SG11/median/rho0p80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5936 | +0.0297 |
| bicycle | 0.5957 | +0.0144 |
| bird | 0.7091 | -0.0037 |
| boat | 0.5491 | +0.0317 |
| bottle | 0.4821 | +0.0255 |
| bus | 0.9233 | +0.0032 |
| car | 0.5469 | -0.0214 |
| cat | 0.8385 | +0.0098 |
| chair | 0.4337 | +0.0147 |
| cow | 0.8134 | +0.0128 |
| dining table | 0.3336 | -0.0406 |
| dog | 0.7438 | -0.0012 |
| horse | 0.6530 | +0.0175 |
| motorbike | 0.6272 | +0.0097 |
| person | 0.2234 | +0.0044 |
| potted plant | 0.6554 | +0.0124 |
| sheep | 0.8633 | +0.0177 |
| sofa | 0.5706 | -0.0196 |
| train | 0.6035 | -0.0040 |
| tv | 0.6826 | -0.0144 |
### coco: W/SG11/median/rho0p90

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5712 | +0.0072 |
| bicycle | 0.5812 | -0.0001 |
| bird | 0.7157 | +0.0029 |
| boat | 0.5310 | +0.0135 |
| bottle | 0.4825 | +0.0259 |
| bus | 0.9231 | +0.0030 |
| car | 0.5515 | -0.0168 |
| cat | 0.7852 | -0.0435 |
| chair | 0.4165 | -0.0025 |
| cow | 0.8096 | +0.0090 |
| dining table | 0.3537 | -0.0205 |
| dog | 0.7409 | -0.0041 |
| horse | 0.6374 | +0.0019 |
| motorbike | 0.6286 | +0.0111 |
| person | 0.2242 | +0.0053 |
| potted plant | 0.6297 | -0.0134 |
| sheep | 0.8477 | +0.0021 |
| sofa | 0.5704 | -0.0198 |
| train | 0.6051 | -0.0024 |
| tv | 0.7053 | +0.0084 |
### coco: W/SG11/trim20/rho0p70

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5908 | +0.0269 |
| bicycle | 0.5964 | +0.0151 |
| bird | 0.7163 | +0.0034 |
| boat | 0.5368 | +0.0194 |
| bottle | 0.4859 | +0.0293 |
| bus | 0.9243 | +0.0043 |
| car | 0.5598 | -0.0084 |
| cat | 0.8358 | +0.0071 |
| chair | 0.4224 | +0.0033 |
| cow | 0.8105 | +0.0100 |
| dining table | 0.3265 | -0.0477 |
| dog | 0.7491 | +0.0040 |
| horse | 0.6647 | +0.0292 |
| motorbike | 0.6358 | +0.0183 |
| person | 0.2245 | +0.0055 |
| potted plant | 0.6477 | +0.0047 |
| sheep | 0.8648 | +0.0193 |
| sofa | 0.5914 | +0.0012 |
| train | 0.6210 | +0.0135 |
| tv | 0.6447 | -0.0523 |
### coco: W/SG11/trim20/rho0p80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5938 | +0.0299 |
| bicycle | 0.5971 | +0.0157 |
| bird | 0.7056 | -0.0072 |
| boat | 0.5490 | +0.0316 |
| bottle | 0.4831 | +0.0265 |
| bus | 0.9242 | +0.0041 |
| car | 0.5465 | -0.0217 |
| cat | 0.8377 | +0.0090 |
| chair | 0.4341 | +0.0151 |
| cow | 0.8124 | +0.0118 |
| dining table | 0.3336 | -0.0406 |
| dog | 0.7436 | -0.0015 |
| horse | 0.6520 | +0.0165 |
| motorbike | 0.6293 | +0.0118 |
| person | 0.2213 | +0.0023 |
| potted plant | 0.6548 | +0.0118 |
| sheep | 0.8617 | +0.0161 |
| sofa | 0.5707 | -0.0195 |
| train | 0.6026 | -0.0049 |
| tv | 0.6851 | -0.0119 |
### coco: W/SG11/trim20/rho0p90

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5714 | +0.0075 |
| bicycle | 0.5822 | +0.0008 |
| bird | 0.7134 | +0.0006 |
| boat | 0.5309 | +0.0134 |
| bottle | 0.4834 | +0.0269 |
| bus | 0.9236 | +0.0035 |
| car | 0.5513 | -0.0170 |
| cat | 0.7840 | -0.0447 |
| chair | 0.4159 | -0.0032 |
| cow | 0.8081 | +0.0075 |
| dining table | 0.3554 | -0.0188 |
| dog | 0.7408 | -0.0043 |
| horse | 0.6366 | +0.0011 |
| motorbike | 0.6285 | +0.0110 |
| person | 0.2229 | +0.0040 |
| potted plant | 0.6299 | -0.0132 |
| sheep | 0.8463 | +0.0008 |
| sofa | 0.5705 | -0.0197 |
| train | 0.6054 | -0.0022 |
| tv | 0.7047 | +0.0077 |
### coco: W/SG12/top50

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5900 | +0.0261 |
| bicycle | 0.5803 | -0.0010 |
| bird | 0.7160 | +0.0031 |
| boat | 0.5301 | +0.0126 |
| bottle | 0.4730 | +0.0164 |
| bus | 0.9179 | -0.0022 |
| car | 0.5629 | -0.0054 |
| cat | 0.8387 | +0.0100 |
| chair | 0.4275 | +0.0084 |
| cow | 0.8159 | +0.0153 |
| dining table | 0.3807 | +0.0065 |
| dog | 0.7461 | +0.0010 |
| horse | 0.6540 | +0.0185 |
| motorbike | 0.6275 | +0.0100 |
| person | 0.2377 | +0.0188 |
| potted plant | 0.6605 | +0.0175 |
| sheep | 0.8677 | +0.0221 |
| sofa | 0.5985 | +0.0083 |
| train | 0.6181 | +0.0106 |
| tv | 0.7002 | +0.0033 |
### coco: W/SG12/top70

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5730 | +0.0091 |
| bicycle | 0.5752 | -0.0061 |
| bird | 0.7027 | -0.0102 |
| boat | 0.5183 | +0.0008 |
| bottle | 0.4612 | +0.0046 |
| bus | 0.9166 | -0.0035 |
| car | 0.5629 | -0.0054 |
| cat | 0.8358 | +0.0072 |
| chair | 0.4190 | -0.0000 |
| cow | 0.8131 | +0.0125 |
| dining table | 0.3945 | +0.0203 |
| dog | 0.7461 | +0.0010 |
| horse | 0.6410 | +0.0056 |
| motorbike | 0.6190 | +0.0015 |
| person | 0.2436 | +0.0246 |
| potted plant | 0.6434 | +0.0004 |
| sheep | 0.8564 | +0.0108 |
| sofa | 0.5828 | -0.0074 |
| train | 0.6052 | -0.0023 |
| tv | 0.6894 | -0.0075 |
### coco: W/SG12/top80

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5758 | +0.0119 |
| bicycle | 0.5853 | +0.0040 |
| bird | 0.7047 | -0.0082 |
| boat | 0.5233 | +0.0058 |
| bottle | 0.4642 | +0.0077 |
| bus | 0.9221 | +0.0020 |
| car | 0.5678 | -0.0005 |
| cat | 0.8327 | +0.0041 |
| chair | 0.4206 | +0.0016 |
| cow | 0.8101 | +0.0095 |
| dining table | 0.3846 | +0.0104 |
| dog | 0.7475 | +0.0024 |
| horse | 0.6457 | +0.0102 |
| motorbike | 0.6239 | +0.0064 |
| person | 0.2401 | +0.0211 |
| potted plant | 0.6430 | +0.0000 |
| sheep | 0.8565 | +0.0109 |
| sofa | 0.5869 | -0.0033 |
| train | 0.6094 | +0.0019 |
| tv | 0.6929 | -0.0040 |
### coco: W/SG12/top90

| Class | IoU | Delta REF |
|---|---:|---:|
| aeroplane | 0.5713 | +0.0074 |
| bicycle | 0.5859 | +0.0046 |
| bird | 0.7057 | -0.0071 |
| boat | 0.5230 | +0.0055 |
| bottle | 0.4595 | +0.0029 |
| bus | 0.9233 | +0.0032 |
| car | 0.5691 | +0.0008 |
| cat | 0.8285 | -0.0002 |
| chair | 0.4201 | +0.0011 |
| cow | 0.8033 | +0.0027 |
| dining table | 0.3770 | +0.0028 |
| dog | 0.7467 | +0.0017 |
| horse | 0.6412 | +0.0057 |
| motorbike | 0.6210 | +0.0035 |
| person | 0.2334 | +0.0144 |
| potted plant | 0.6430 | -0.0000 |
| sheep | 0.8519 | +0.0063 |
| sofa | 0.5900 | -0.0002 |
| train | 0.6111 | +0.0036 |
| tv | 0.6948 | -0.0021 |

Reliability by module/call/head, top-K selection frequencies, pairwise matrices and plots are in `diagnostics/`. Visual sheets are in `visuals/`.
