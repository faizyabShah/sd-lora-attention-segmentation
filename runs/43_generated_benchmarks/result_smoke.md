# Run 43: the new pipeline on VOC-Sim and COCO-Cap, with NERVE and SCLIP in the same capture (`PREREG.md`)

Images: VOC-Sim 4, COCO-Cap 4 (word in prompt 2, not in prompt 2). One regeneration (pass 2) gives every map and the PNG for NERVE and SCLIP. Ours: uniform cross-attention seed, MK4-LR at γ = 1, fine-only propagation (V0), min-max, **fixed threshold 0.55**. Class-macro mIoU; 95% image-bootstrap intervals (200 samples, paired).


## 1. Primary contrasts (source H, V0, raw at 0.55)

| # | Contrast | Set | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | MK4-LR − no gate | VOC-Sim | +0.021 [+0.008, +0.035] | **helps** |
| P2 | MK4-LR − no gate | COCO-Cap | -0.007 [-0.016, +0.001] | not resolved |
| P3 | MK4-LR − NERVE | VOC-Sim | +0.034 [+0.003, +0.065] | **helps** |
| P4 | MK4-LR − NERVE | COCO-Cap | +0.156 [+0.033, +0.452] | **helps** |
| P5 | MK4-LR − no gate | COCO-Cap, word not in prompt | -0.005 [-0.008, -0.002] | **hurts** |

## 2. Main table (mIoU)

| Method | Read-out | VOC-Sim | COCO-Cap | COCO-Cap, word in prompt | COCO-Cap, word not in prompt |
|---|---|---:|---:|---:|---:|
| NERVE (as released, SD 2.1-base) | own binarisation | 0.842 | 0.717 | 0.481 | 0.799 |
| SCLIP (ViT-B/16, VOC21) | own prediction | 0.630 | 0.622 | 0.489 | 0.581 |
| No gate (seed + V0) | raw 0.55 | 0.854 | 0.880 | 0.897 | 0.847 |
| MK4 (H) | raw 0.55 | 0.880 | 0.682 | 0.422 | 0.844 |
| **MK4-LR (H), ours** | raw 0.55 | 0.876 | 0.873 | 0.888 | 0.842 |
| No gate (seed + V0) | + DenseCRF 0.55 | 0.856 | 0.898 | 0.921 | 0.881 |
| MK4 (H) | + DenseCRF 0.55 | 0.872 | 0.681 | 0.425 | 0.871 |
| **MK4-LR (H), ours** | + DenseCRF 0.55 | 0.880 | 0.894 | 0.928 | 0.874 |

## 3. Differences for every subset (raw 0.55 unless stated)

| Contrast | VOC-Sim | COCO-Cap | COCO-Cap, word in prompt | COCO-Cap, word not in prompt |
|---|---|---|---|---|
| MK4-LR − no gate | +0.021 [+0.008, +0.035] | -0.007 [-0.016, +0.001] | -0.009 [-0.021, +0.004] | -0.005 [-0.008, -0.002] |
| MK4-LR − MK4 | -0.005 [-0.016, +0.008] | +0.190 [-0.002, +0.508] | +0.466 [+0.022, +0.557] | -0.002 [-0.013, +0.009] |
| MK4 − no gate | +0.026 [+0.003, +0.046] | -0.198 [-0.524, -0.004] | -0.475 [-0.578, -0.018] | -0.004 [-0.017, +0.010] |
| MK4-LR − NERVE | +0.034 [+0.003, +0.065] | +0.156 [+0.033, +0.452] | +0.407 [+0.086, +0.502] | +0.043 [-0.020, +0.105] |
| MK4-LR − SCLIP | +0.246 [+0.135, +0.356] | +0.251 [+0.144, +0.466] | +0.398 [+0.183, +0.474] | +0.261 [+0.105, +0.418] |
| MK4-LR + CRF − NERVE | +0.038 [+0.007, +0.068] | +0.177 [+0.035, +0.517] | +0.447 [+0.099, +0.565] | +0.074 [-0.028, +0.177] |
| MK4-LR + CRF − no gate + CRF | +0.024 [+0.002, +0.046] | -0.004 [-0.007, +0.009] | +0.006 [+0.000, +0.013] | -0.007 [-0.012, -0.002] |
| few-shot F MK4-LR − no gate | +0.021 [+0.002, +0.036] | -0.012 [-0.014, -0.002] | -0.005 [-0.008, -0.002] | -0.011 [-0.021, -0.001] |
| few-shot F − half split H (MK4-LR) | +0.000 [-0.010, +0.006] | -0.005 [-0.009, +0.010] | +0.004 [-0.006, +0.013] | -0.006 [-0.013, +0.001] |
| VOC-Sim → COCO-Cap transfer V, MK4-LR − no gate | — | -0.021 [-0.027, -0.006] | -0.019 [-0.024, -0.012] | -0.015 [-0.030, +0.001] |
| V − H (MK4-LR) | — | -0.014 [-0.019, -0.001] | -0.010 [-0.015, -0.004] | -0.010 [-0.022, +0.003] |

## 4. Every arm (mIoU, raw 0.55)

| Seed | seed, VOC-Sim | seed, COCO-Cap | V0, VOC-Sim | V0, COCO-Cap | V3, VOC-Sim | V3, COCO-Cap |
|---|---:|---:|---:|---:|---:|---:|
| u- | 0.628 | 0.662 | 0.854 | 0.880 | 0.859 | 0.871 |
| w- | 0.554 | 0.717 | 0.865 | 0.873 | 0.871 | 0.863 |
| uHM | 0.530 | 0.454 | 0.880 | 0.682 | 0.880 | 0.816 |
| uHL | 0.559 | 0.643 | 0.876 | 0.873 | 0.874 | 0.859 |
| uFM | 0.567 | 0.587 | 0.876 | 0.818 | 0.884 | 0.845 |
| uFL | 0.573 | 0.648 | 0.876 | 0.868 | 0.878 | 0.861 |
| uVM | — | 0.597 | — | 0.855 | — | 0.851 |
| uVL | — | 0.669 | — | 0.859 | — | 0.857 |
| wHL | 0.518 | 0.694 | 0.849 | 0.862 | 0.877 | 0.855 |

Seed codes: u = uniform weights, w = SA-guided weights; - = no gate; H / F / V = profile source; M = MK4, L = MK4-LR.

Descriptive: hierarchy and SA weights added to MK4-LR (H), raw 0.55: VOC-Sim: V3 − V0 -0.002 [-0.015, +0.011], SA weights -0.026 [-0.043, -0.010]; COCO-Cap: V3 − V0 -0.014 [-0.070, +0.002], SA weights -0.010 [-0.015, +0.001].

## 5. Baselines, secondary read-outs

| Read-out | VOC-Sim | COCO-Cap | COCO-Cap, word in prompt | COCO-Cap, word not in prompt |
|---|---:|---:|---:|---:|
| NERVE own binarisation | 0.842 | 0.717 | 0.481 | 0.799 |
| NERVE without class-score gate | 0.842 | 0.717 | 0.481 | 0.799 |
| NERVE soft map, min-max 0.55 | 0.672 | 0.810 | 0.832 | 0.691 |
| SCLIP own prediction | 0.630 | 0.622 | 0.489 | 0.581 |
| SCLIP probability, min-max 0.55 | 0.568 | 0.557 | 0.584 | 0.408 |

## 6. Every threshold (V0, raw; † = 95% CI excludes 0)

| Set | Contrast | 0.300 | 0.325 | 0.350 | 0.375 | 0.400 | 0.425 | 0.450 | 0.475 | 0.500 | 0.525 | 0.550 | 0.575 | 0.600 | 0.625 | 0.650 | 0.675 | 0.700 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VOC-Sim | MK4-LR − no gate | +0.019 | +0.041 | +0.055 | +0.057 | +0.046 | +0.037 | +0.042 | +0.039 | +0.038† | +0.027† | +0.021† | +0.015† | +0.006 | -0.032 | -0.062 | -0.078 | -0.076 |
| VOC-Sim | MK4-LR − MK4 | -0.022 | -0.017 | -0.017† | -0.018† | -0.018† | -0.015† | -0.006† | -0.003 | -0.002 | -0.002 | -0.005 | -0.013 | -0.038 | -0.083 | -0.105 | -0.116 | -0.114 |
| VOC-Sim | MK4-LR mIoU | 0.702 | 0.741 | 0.768 | 0.788 | 0.808 | 0.829 | 0.852 | 0.864 | 0.872 | 0.877 | 0.876 | 0.864 | 0.830 | 0.768 | 0.718 | 0.673 | 0.627 |
| COCO-Cap | MK4-LR − no gate | +0.011† | +0.009† | +0.012† | +0.012† | +0.004† | +0.001 | -0.001 | -0.004 | -0.006 | -0.006 | -0.007 | -0.008 | -0.003 | +0.004 | +0.011 | +0.015 | +0.019 |
| COCO-Cap | MK4-LR − MK4 | +0.152 | +0.136 | +0.149 | +0.168† | +0.175† | +0.188† | +0.187† | +0.182† | +0.183 | +0.186 | +0.190 | +0.196 | +0.195 | +0.187 | +0.087 | +0.037 | +0.013 |
| COCO-Cap | MK4-LR mIoU | 0.661 | 0.679 | 0.704 | 0.738 | 0.774 | 0.815 | 0.852 | 0.876 | 0.878 | 0.877 | 0.873 | 0.864 | 0.851 | 0.834 | 0.811 | 0.781 | 0.745 |
| COCO-Cap, word in prompt | MK4-LR − no gate | +0.008† | +0.007† | +0.027† | +0.031† | +0.009† | +0.000 | -0.003 | -0.005 | -0.006 | -0.008 | -0.009 | -0.006 | +0.010 | +0.032 | +0.049† | +0.060† | +0.074† |
| COCO-Cap, word in prompt | MK4-LR − MK4 | +0.384† | +0.344† | +0.372† | +0.390† | +0.403† | +0.415† | +0.426† | +0.437† | +0.447† | +0.457† | +0.466† | +0.474† | +0.478† | +0.467 | +0.248 | +0.120 | +0.052 |
| COCO-Cap, word in prompt | MK4-LR mIoU | 0.726 | 0.739 | 0.775 | 0.798 | 0.815 | 0.830 | 0.843 | 0.855 | 0.867 | 0.878 | 0.888 | 0.896 | 0.901 | 0.899 | 0.887 | 0.859 | 0.825 |
| COCO-Cap, word not in prompt | MK4-LR − no gate | +0.008 | +0.006 | +0.002 | +0.001 | +0.000 | +0.000 | -0.001 | -0.002† | -0.004† | -0.004† | -0.005† | -0.007† | -0.008† | -0.009† | -0.009† | -0.009† | -0.009† |
| COCO-Cap, word not in prompt | MK4-LR − MK4 | -0.006 | -0.003 | -0.000 | +0.011 | +0.012 | +0.020 | +0.013 | +0.004 | -0.000 | -0.002 | -0.002 | -0.001 | -0.006 | -0.013 | -0.021 | -0.022 | -0.018 |
| COCO-Cap, word not in prompt | MK4-LR mIoU | 0.662 | 0.680 | 0.697 | 0.726 | 0.759 | 0.797 | 0.831 | 0.852 | 0.852 | 0.849 | 0.842 | 0.831 | 0.815 | 0.796 | 0.771 | 0.746 | 0.716 |

## 7. Per class (raw 0.55 for ours)


VOC-Sim:

| Class | images | no gate | MK4 | MK4-LR | NERVE | SCLIP |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | 2 | 0.935 | 0.951 | 0.943 | 0.878 | 0.587 |
| bicycle | 2 | 0.774 | 0.810 | 0.809 | 0.806 | 0.673 |

MK4-LR beats / trails no gate in 2 / 0 classes.

COCO-Cap:

| Class | images | no gate | MK4 | MK4-LR | NERVE | SCLIP |
|---|---:|---:|---:|---:|---:|---:|
| aeroplane | 3 | 0.867 | 0.489 | 0.861 | 0.530 | 0.464 |
| bus | 1 | 0.893 | 0.875 | 0.885 | 0.904 | 0.780 |

MK4-LR beats / trails no gate in 0 / 2 classes.

## 8. Profiles

Centroids kept (mean over profiles) per source and module: H/0: 4.00; H/1: 4.00; H/2: 1.41; H/3: 1.39; H/4: 1.00; H/5: 1.45; H/6: 1.39; H/7: 1.32; H/8: 4.00; H/9: 3.99; H/10: 4.00; V/0: 4.00; V/1: 4.00; V/2: 3.45; V/3: 3.60; V/4: 1.00; V/5: 3.55; V/6: 3.20; V/7: 3.35; V/8: 4.00; V/9: 4.00; V/10: 4.00; F/0: 3.98; F/1: 3.98; F/2: 1.00; F/3: 1.00; F/4: 1.00; F/5: 1.00; F/6: 1.00; F/7: 1.00; F/8: 3.93; F/9: 3.78; F/10: 3.97.

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

Modules without a profile per image (mean): VOC-Sim H 0.00, VOC-Sim F 0.00, COCO-Cap H 0.00, COCO-Cap F 0.00, COCO-Cap V 0.00.

NERVE: SD sd2-community/stable-diffusion-2-1-base (4e63672c03103b6c636b8fb4119ba982469b2955), SFSA head size {'-4': 64, '-6': 64}, seed 42.

BRACE stays frozen; nothing here is a promotion.

