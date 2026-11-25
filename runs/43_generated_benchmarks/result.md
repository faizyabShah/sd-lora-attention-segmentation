# Run 43: the new pipeline on VOC-Sim and COCO-Cap, with NERVE and SCLIP in the same capture (`PREREG.md`)

Images: VOC-Sim 490, COCO-Cap 707 (word in prompt 465, not in prompt 242). One regeneration (pass 2) gives every map and the PNG for NERVE and SCLIP. Ours: uniform cross-attention seed, MK4-LR at γ = 1, fine-only propagation (V0), min-max, **fixed threshold 0.55**. Class-macro mIoU; 95% image-bootstrap intervals (2000 samples, paired).


## 1. Primary contrasts (source H, V0, raw at 0.55)

| # | Contrast | Set | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | MK4-LR − no gate | VOC-Sim | +0.019 [+0.010, +0.028] | **helps** |
| P2 | MK4-LR − no gate | COCO-Cap | +0.074 [+0.060, +0.088] | **helps** |
| P3 | MK4-LR − NERVE | VOC-Sim | -0.005 [-0.018, +0.010] | not resolved |
| P4 | MK4-LR − NERVE | COCO-Cap | +0.033 [+0.007, +0.060] | **helps** |
| P5 | MK4-LR − no gate | COCO-Cap, word not in prompt | +0.078 [+0.052, +0.100] | **helps** |

## 2. Main table (mIoU)

| Method | Read-out | VOC-Sim | COCO-Cap | COCO-Cap, word in prompt | COCO-Cap, word not in prompt |
|---|---|---:|---:|---:|---:|
| NERVE (as released, SD 2.1-base) | own binarisation | 0.805 | 0.702 | 0.738 | 0.615 |
| SCLIP (ViT-B/16, VOC21) | own prediction | 0.719 | 0.583 | 0.616 | 0.514 |
| No gate (seed + V0) | raw 0.55 | 0.782 | 0.662 | 0.673 | 0.639 |
| MK4 (H) | raw 0.55 | 0.778 | 0.695 | 0.709 | 0.665 |
| **MK4-LR (H), ours** | raw 0.55 | 0.801 | 0.736 | 0.746 | 0.717 |
| No gate (seed + V0) | + DenseCRF 0.55 | 0.786 | 0.669 | 0.679 | 0.647 |
| MK4 (H) | + DenseCRF 0.55 | 0.781 | 0.701 | 0.714 | 0.675 |
| **MK4-LR (H), ours** | + DenseCRF 0.55 | 0.805 | 0.744 | 0.753 | 0.728 |

## 3. Differences for every subset (raw 0.55 unless stated)

| Contrast | VOC-Sim | COCO-Cap | COCO-Cap, word in prompt | COCO-Cap, word not in prompt |
|---|---|---|---|---|
| MK4-LR − no gate | +0.019 [+0.010, +0.028] | +0.074 [+0.060, +0.088] | +0.073 [+0.055, +0.091] | +0.078 [+0.052, +0.100] |
| MK4-LR − MK4 | +0.023 [+0.017, +0.028] | +0.041 [+0.028, +0.054] | +0.038 [+0.020, +0.056] | +0.051 [+0.032, +0.071] |
| MK4 − no gate | -0.004 [-0.013, +0.004] | +0.033 [+0.018, +0.049] | +0.036 [+0.014, +0.058] | +0.027 [-0.004, +0.051] |
| MK4-LR − NERVE | -0.005 [-0.018, +0.010] | +0.033 [+0.007, +0.060] | +0.009 [-0.015, +0.037] | +0.102 [+0.046, +0.145] |
| MK4-LR − SCLIP | +0.082 [+0.066, +0.100] | +0.152 [+0.125, +0.181] | +0.130 [+0.109, +0.156] | +0.203 [+0.141, +0.257] |
| MK4-LR + CRF − NERVE | -0.001 [-0.015, +0.015] | +0.042 [+0.014, +0.068] | +0.016 [-0.009, +0.045] | +0.113 [+0.059, +0.158] |
| MK4-LR + CRF − no gate + CRF | +0.019 [+0.009, +0.029] | +0.076 [+0.060, +0.090] | +0.075 [+0.055, +0.094] | +0.081 [+0.054, +0.105] |
| few-shot F MK4-LR − no gate | +0.012 [+0.002, +0.021] | +0.061 [+0.047, +0.073] | +0.058 [+0.040, +0.074] | +0.068 [+0.045, +0.090] |
| few-shot F − half split H (MK4-LR) | -0.007 [-0.012, -0.002] | -0.013 [-0.022, -0.005] | -0.015 [-0.025, -0.006] | -0.010 [-0.020, +0.002] |
| VOC-Sim → COCO-Cap transfer V, MK4-LR − no gate | — | +0.060 [+0.046, +0.074] | +0.063 [+0.046, +0.080] | +0.071 [+0.044, +0.093] |
| V − H (MK4-LR) | — | -0.014 [-0.025, -0.002] | -0.010 [-0.025, +0.006] | -0.007 [-0.023, +0.007] |

## 4. Every arm (mIoU, raw 0.55)

| Seed | seed, VOC-Sim | seed, COCO-Cap | V0, VOC-Sim | V0, COCO-Cap | V3, VOC-Sim | V3, COCO-Cap |
|---|---:|---:|---:|---:|---:|---:|
| u- | 0.591 | 0.455 | 0.782 | 0.662 | 0.815 | 0.670 |
| w- | 0.613 | 0.510 | 0.792 | 0.690 | 0.819 | 0.693 |
| uHM | 0.556 | 0.472 | 0.778 | 0.695 | 0.816 | 0.703 |
| uHL | 0.597 | 0.515 | 0.801 | 0.736 | 0.828 | 0.732 |
| uFM | 0.543 | 0.453 | 0.771 | 0.685 | 0.814 | 0.687 |
| uFL | 0.586 | 0.493 | 0.794 | 0.722 | 0.824 | 0.725 |
| uVM | — | 0.439 | — | 0.683 | — | 0.681 |
| uVL | — | 0.475 | — | 0.722 | — | 0.726 |
| wHL | 0.584 | 0.524 | 0.794 | 0.736 | 0.832 | 0.738 |

Seed codes: u = uniform weights, w = SA-guided weights; - = no gate; H / F / V = profile source; M = MK4, L = MK4-LR.

Descriptive: hierarchy and SA weights added to MK4-LR (H), raw 0.55: VOC-Sim: V3 − V0 +0.027 [+0.018, +0.037], SA weights -0.006 [-0.011, -0.002]; COCO-Cap: V3 − V0 -0.004 [-0.019, +0.011], SA weights +0.001 [-0.007, +0.007].

## 5. Baselines, secondary read-outs

| Read-out | VOC-Sim | COCO-Cap | COCO-Cap, word in prompt | COCO-Cap, word not in prompt |
|---|---:|---:|---:|---:|
| NERVE own binarisation | 0.805 | 0.702 | 0.738 | 0.615 |
| NERVE without class-score gate | 0.808 | 0.717 | 0.740 | 0.655 |
| NERVE soft map, min-max 0.55 | 0.811 | 0.728 | 0.758 | 0.646 |
| SCLIP own prediction | 0.719 | 0.583 | 0.616 | 0.514 |
| SCLIP probability, min-max 0.55 | 0.633 | 0.474 | 0.520 | 0.381 |

## 6. Every threshold (V0, raw; † = 95% CI excludes 0)

| Set | Contrast | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VOC-Sim | MK4-LR − no gate | +0.027† | +0.026† | +0.026† | +0.026† | +0.026† | +0.025† | +0.022† | +0.020† | +0.020† | +0.020† | +0.019† | +0.019† | +0.019† | +0.020† | +0.024† | +0.026† | +0.028† |
| VOC-Sim | MK4-LR − MK4 | +0.008† | +0.009† | +0.011† | +0.012† | +0.015† | +0.017† | +0.017† | +0.017† | +0.017† | +0.020† | +0.023† | +0.023† | +0.022† | +0.021† | +0.023† | +0.024† | +0.025† |
| VOC-Sim | MK4-LR mIoU | 0.789 | 0.802 | 0.814 | 0.822 | 0.828 | 0.831 | 0.830 | 0.826 | 0.821 | 0.812 | 0.801 | 0.785 | 0.765 | 0.743 | 0.719 | 0.689 | 0.655 |
| COCO-Cap | MK4-LR − no gate | +0.083† | +0.084† | +0.083† | +0.082† | +0.084† | +0.083† | +0.082† | +0.079† | +0.077† | +0.077† | +0.074† | +0.072† | +0.071† | +0.069† | +0.068† | +0.068† | +0.068† |
| COCO-Cap | MK4-LR − MK4 | +0.051† | +0.049† | +0.049† | +0.048† | +0.052† | +0.053† | +0.051† | +0.048† | +0.046† | +0.044† | +0.041† | +0.038† | +0.038† | +0.041† | +0.044† | +0.046† | +0.049† |
| COCO-Cap | MK4-LR mIoU | 0.618 | 0.641 | 0.662 | 0.681 | 0.700 | 0.713 | 0.725 | 0.732 | 0.737 | 0.739 | 0.736 | 0.730 | 0.721 | 0.708 | 0.691 | 0.669 | 0.643 |
| COCO-Cap, word in prompt | MK4-LR − no gate | +0.088† | +0.087† | +0.086† | +0.084† | +0.087† | +0.087† | +0.087† | +0.083† | +0.081† | +0.079† | +0.073† | +0.070† | +0.066† | +0.062† | +0.060† | +0.060† | +0.058† |
| COCO-Cap, word in prompt | MK4-LR − MK4 | +0.049† | +0.048† | +0.050† | +0.049† | +0.054† | +0.054† | +0.055† | +0.052† | +0.049† | +0.045† | +0.038† | +0.032† | +0.030† | +0.029† | +0.027† | +0.026† | +0.026† |
| COCO-Cap, word in prompt | MK4-LR mIoU | 0.624 | 0.648 | 0.670 | 0.691 | 0.711 | 0.726 | 0.739 | 0.747 | 0.751 | 0.751 | 0.746 | 0.740 | 0.731 | 0.718 | 0.700 | 0.679 | 0.651 |
| COCO-Cap, word not in prompt | MK4-LR − no gate | +0.067† | +0.071† | +0.073† | +0.077† | +0.078† | +0.075† | +0.072† | +0.070† | +0.070† | +0.073† | +0.078† | +0.080† | +0.081† | +0.079† | +0.075† | +0.072† | +0.070† |
| COCO-Cap, word not in prompt | MK4-LR − MK4 | +0.052† | +0.051† | +0.051† | +0.052† | +0.055† | +0.055† | +0.051† | +0.049† | +0.046† | +0.048† | +0.051† | +0.055† | +0.058† | +0.063† | +0.068† | +0.071† | +0.073† |
| COCO-Cap, word not in prompt | MK4-LR mIoU | 0.607 | 0.628 | 0.649 | 0.668 | 0.683 | 0.694 | 0.703 | 0.710 | 0.715 | 0.717 | 0.717 | 0.713 | 0.704 | 0.691 | 0.673 | 0.651 | 0.627 |

## 7. Per class (raw 0.55 for ours)


VOC-Sim:

| Class | images | no gate | MK4 | MK4-LR | NERVE | SCLIP |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | 24 | 0.849 | 0.845 | 0.868 | 0.783 | 0.607 |
| bicycle | 25 | 0.695 | 0.658 | 0.711 | 0.762 | 0.645 |
| bird | 25 | 0.874 | 0.881 | 0.888 | 0.847 | 0.721 |
| boat | 25 | 0.884 | 0.840 | 0.864 | 0.809 | 0.593 |
| bottle | 25 | 0.877 | 0.797 | 0.815 | 0.767 | 0.665 |
| bus | 25 | 0.926 | 0.915 | 0.917 | 0.889 | 0.833 |
| car | 23 | 0.824 | 0.807 | 0.838 | 0.780 | 0.814 |
| cat | 25 | 0.683 | 0.694 | 0.697 | 0.902 | 0.846 |
| chair | 24 | 0.771 | 0.764 | 0.782 | 0.728 | 0.529 |
| cow | 25 | 0.791 | 0.754 | 0.801 | 0.933 | 0.823 |
| dining table | 25 | 0.539 | 0.582 | 0.638 | 0.649 | 0.555 |
| dog | 25 | 0.850 | 0.887 | 0.913 | 0.922 | 0.825 |
| horse | 23 | 0.862 | 0.872 | 0.880 | 0.887 | 0.746 |
| motorbike | 25 | 0.871 | 0.856 | 0.847 | 0.857 | 0.782 |
| person | 24 | 0.572 | 0.663 | 0.706 | 0.515 | 0.803 |
| potted plant | 25 | 0.650 | 0.667 | 0.644 | 0.810 | 0.754 |
| sheep | 25 | 0.676 | 0.746 | 0.781 | 0.919 | 0.846 |
| sofa | 25 | 0.959 | 0.940 | 0.951 | 0.912 | 0.774 |
| train | 25 | 0.787 | 0.748 | 0.830 | 0.882 | 0.643 |
| tv | 22 | 0.696 | 0.641 | 0.644 | 0.555 | 0.578 |

MK4-LR beats / trails no gate in 13 / 7 classes.

COCO-Cap:

| Class | images | no gate | MK4 | MK4-LR | NERVE | SCLIP |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | 49 | 0.740 | 0.784 | 0.832 | 0.546 | 0.410 |
| bicycle | 46 | 0.614 | 0.686 | 0.706 | 0.736 | 0.611 |
| bird | 48 | 0.747 | 0.718 | 0.789 | 0.800 | 0.619 |
| boat | 32 | 0.591 | 0.618 | 0.629 | 0.651 | 0.395 |
| bottle | 21 | 0.618 | 0.785 | 0.711 | 0.580 | 0.460 |
| bus | 37 | 0.932 | 0.909 | 0.914 | 0.834 | 0.750 |
| car | 28 | 0.563 | 0.660 | 0.733 | 0.666 | 0.679 |
| cat | 46 | 0.853 | 0.868 | 0.870 | 0.877 | 0.721 |
| chair | 20 | 0.488 | 0.520 | 0.636 | 0.695 | 0.488 |
| cow | 48 | 0.800 | 0.855 | 0.872 | 0.840 | 0.742 |
| dining table | 14 | 0.322 | 0.407 | 0.512 | 0.478 | 0.535 |
| dog | 44 | 0.792 | 0.791 | 0.806 | 0.815 | 0.721 |
| horse | 38 | 0.707 | 0.778 | 0.777 | 0.710 | 0.540 |
| motorbike | 41 | 0.715 | 0.759 | 0.763 | 0.747 | 0.640 |
| person | 52 | 0.210 | 0.324 | 0.615 | 0.707 | 0.630 |
| potted plant | 14 | 0.682 | 0.681 | 0.672 | 0.666 | 0.478 |
| sheep | 38 | 0.865 | 0.826 | 0.840 | 0.896 | 0.729 |
| sofa | 34 | 0.697 | 0.771 | 0.802 | 0.764 | 0.599 |
| train | 33 | 0.669 | 0.603 | 0.639 | 0.694 | 0.576 |
| tv | 24 | 0.632 | 0.558 | 0.597 | 0.344 | 0.345 |

MK4-LR beats / trails no gate in 15 / 5 classes.

## 8. Profiles

Centroids kept (mean over profiles) per source and module: H/0: 4.00; H/1: 4.00; H/2: 3.98; H/3: 4.00; H/4: 3.94; H/5: 4.00; H/6: 4.00; H/7: 4.00; H/8: 4.00; H/9: 4.00; H/10: 4.00; V/0: 4.00; V/1: 4.00; V/2: 4.00; V/3: 4.00; V/4: 4.00; V/5: 4.00; V/6: 4.00; V/7: 4.00; V/8: 4.00; V/9: 4.00; V/10: 4.00; F/0: 4.00; F/1: 4.00; F/2: 3.95; F/3: 3.99; F/4: 2.80; F/5: 3.94; F/6: 3.91; F/7: 3.86; F/8: 4.00; F/9: 4.00; F/10: 4.00.

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

Modules without a profile per image (mean): VOC-Sim H 0.00, VOC-Sim F 0.00, COCO-Cap H 0.00, COCO-Cap F 0.00, COCO-Cap V 0.00.

NERVE: SD sd2-community/stable-diffusion-2-1-base (4e63672c03103b6c636b8fb4119ba982469b2955), SFSA head size {'-4': 64, '-6': 64}, seed 42.

BRACE stays frozen; nothing here is a promotion.

