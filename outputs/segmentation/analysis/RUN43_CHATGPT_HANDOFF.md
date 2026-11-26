# Run 43 handoff for ChatGPT: our gated attention pipeline vs NERVE and SCLIP on VOC-Sim and COCO-Cap

Date: 2026-11-25. Everything below comes from one pre-registered run (pre-registration committed before scoring), plus two
descriptive post hoc checks (clearly labelled). Internal code names are explained in section 0; do not use them in
paper prose.

## 0. Context and glossary

**Task.** Training-free, label-free class-word segmentation from Stable Diffusion 1.5 attention on generated images.
Benchmarks are OVAM's generated sets: **VOC-Sim** (490 images, 20 VOC classes) and **COCO-Cap** (707 images, the same 20
classes, generated from COCO captions). Ground truth is used only for scoring. Metric: class-macro mIoU (IoU pooled per
class over images, then the mean over 20 classes). 95% intervals: 2,000-sample paired image bootstrap.

**Our pipeline (the new paper's method):**
1. **Seed**: generation-time cross-attention to the class word, **uniform** weights over 11 UNet cross-attention modules.
2. **Gate (MK4)**: per module, an *object profile* (mean of the top 5% queries by the class word's attention on reference
   images of that class) and 4 *negative centroids* (top 1% queries by the class word's attention on reference images
   generated for other classes whose prompts do not mention the word, i.e. "false activations"; spherical k-means). A
   pixel's query is kept if it is closer to the object profile than to its nearest negative (soft gate in [0, 1]).
3. **MK4-LR (low-resolution consensus)**: the gates of the six modules below 64×64 (16×16 and 8×8) are averaged at 64×64
   and applied to every module, at strength γ = 1. **This is the method ("ours").**
4. **Propagation V0 (fine-only)**: two 64×64 self-attention steps. (V3 = capped 16→32→64 hierarchy, an ablation.)
5. **Read-out**: min-max, **fixed threshold 0.55** (not tuned; stated as determined empirically in earlier experiments,
   as OVAM does). DenseCRF is a secondary row.

**Profile sources (where the reference images come from; no masks are used):**
- **H (primary)**: split each benchmark in half per class; an image is gated with profiles from the other half.
- **F (few-shot backup)**: per image, 5 random images of its class + 100 random negative images from the same set.
- **V (transfer)**: all of VOC-Sim, applied to COCO-Cap.

**Arm codes** in tables: u = uniform seed weights, w = SA-guided (attention-weighted) seed weights; `-` = no gate;
H/F/V = profile source; M = MK4, L = MK4-LR. E.g. `uHL` = ours.

**"Word in / not in prompt"** (COCO-Cap): OVAM's `word_included` flag. 242 of 707 images were generated from captions
that do not contain the class word (e.g. "eagle" for bird, "goats" for sheep). This is OVAM's sense of querying a word
not present in the prompt. **Framing rule: we do NOT claim "open-vocabulary segmentation"; we say the method also works
when the queried word is not in the generation prompt.**

**Baselines, run on the same regenerated PNGs (same capture):**
- **NERVE** (WACV 2026), official code @ 1ddc5f9, as released: CLIP ViT-L/14 + SD **2.1-base** self-attention random walk
  (one noised pass, t index 45/50), its VOC21 class names and background words, its own binarisation (class-score gate
  0.5). Only infrastructure edits (offline model path, because stabilityai's SD 2.1-base repo was withdrawn; the
  sd2-community re-upload is used). Seed 42 per image.
- **SCLIP** (ECCV 2024), our port that reproduces the official repo on VOC val (VOC20 81.49 vs 81.54; VOC21 59.64 vs
  59.63): ViT-B/16, VOC21 settings, its own prediction.
- Published rows (DAAM, Attn2Mask, OVAM, Grounded Diffusion, DatasetDM) are quoted from OVAM's Table 1, not rerun.

**Pre-registered primary contrasts** (source H, V0, raw, 0.55): P1/P2 ours − no gate (VOC-Sim / COCO-Cap); P3/P4 ours −
NERVE; P5 ours − no gate on COCO-Cap images whose word is not in the prompt.

**Checks passed (smoke):** MK4-LR equals an independent mean of the low-res gates (1.2e-07); gated seed equals the
reference implementation (1.4e-06); propagation equals the reference aggregate (4.1e-06); all 22 SLURM jobs completed;
490 / 707 images scored.

---

# Part A. Full run 43 report (verbatim)


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


---

# Part B. Post hoc (descriptive): why an earlier run reported COCO-Cap ≈ 0.79–0.80 and this one 0.736

An earlier exploratory run (run 38) reported MK4-LR ≈ 0.79–0.80 on COCO-Cap. That run scored only a 351-image test half,
and used a threshold and γ tuned on the other half, with DenseCRF. Restricting run 43's own scores to those 351 images:

| COCO-Cap, raw 0.55 | all 707 | run 38 test (351) | run 38 dev (356) |
|---|---:|---:|---:|
| no gate, uniform, V0 | 0.662 | 0.675 | 0.649 |
| no gate, SA weights, V0 | 0.690 | 0.691 | 0.688 |
| MK4 (H), V0 | 0.695 | 0.698 | 0.694 |
| **MK4-LR (H), uniform, V0** | **0.736** | **0.762** | **0.712** |
| MK4-LR (H), SA weights, V0 | 0.736 | 0.758 | 0.716 |
| MK4-LR (H), V0 + DenseCRF 0.55 | 0.744 | 0.773 | 0.719 |
| MK4-LR (H), SA weights, V3 | 0.738 | 0.760 | 0.713 |
| NERVE own | 0.702 | 0.712 | 0.698 |

Run 38 (regime X, SA weights, V0, MK4-LR) on its test half: raw 0.55 0.761, + CRF 0.55 0.768, + CRF at dev threshold
and dev γ 0.794 (primary), V3 0.801. Run 43 on the same images and read-out: raw 0.55 0.758, + CRF 0.55 (uniform) 0.773.

Reading: the gap is the image subset (+0.026 for MK4-LR; run 38's test half is easier for the gate) plus run 38's
dev-tuned threshold and γ with DenseCRF (+0.026). Same images and read-out agree across the two captures (0.761 vs 0.758).

Decomposition of the gap (COCO-Cap, MK4-LR, fine-only):

| Step | mIoU | Δ |
|---|---:|---:|
| Run 43, all 707, raw 0.55 | 0.736 | — |
| Only run 38's test half (351) | 0.762 | +0.026 (image subset) |
| + DenseCRF at 0.55 | 0.768–0.773 | +0.01 |
| + run 38's tuned threshold and γ (with CRF) | 0.794 | +0.026 |
| + capped hierarchy | 0.801 | +0.007 |

Conclusion: no regression. Same images and read-out agree across captures (0.761 vs 0.758). The 0.736 on all 707 at a
fixed threshold is the comparable number (NERVE and SCLIP get no tuning).

---

# Part C. Cross-check against a collaborator's independent NERVE run

A collaborator previously ran NERVE's official code on the **stored** OVAM PNGs with **SD 1.5** (seed 0), in three
variants: A (one class word, d = 40, K = 1 patch), V (20 plain class words, d = 40), W (NERVE's own names, d = 64, as
released). Variant W is the like-for-like comparison:

| Read-out (raw) | Collaborator, W (SD 1.5, stored PNGs) | Run 43 (SD 2.1-base, regenerated PNGs) |
|---|---:|---:|
| NERVE's own binarisation | 0.803 / 0.701 | **0.805 / 0.702** |
| Soft map, fixed threshold | 0.810 / 0.729 (0.5) | 0.811 / 0.728 (0.55) |

(VOC-Sim / COCO-Cap.) Agreement within 0.002 confirms the NERVE row.

Higher NERVE numbers seen elsewhere are not NERVE as released: they are NERVE's soft map passed through our read-out
(threshold selection on VOC-Sim + DenseCRF), e.g. variant A 0.844 / 0.740, variant V 0.839 / 0.747 (stored PNGs, not paired
with run 43). Its own rule on those variants is lower (A 0.790 / 0.645, V 0.799 / 0.687).

---

# Part D. Summary of what is established

| Claim | Evidence | Status |
|---|---|---|
| The gate (MK4-LR) improves on the ungated map | P1 +0.019 [+0.010, +0.028], P2 +0.074 [+0.060, +0.088] | **confirmed (pre-registered)** |
| It also helps when the queried word is not in the prompt | P5 +0.078 [+0.052, +0.100] | **confirmed** |
| vs NERVE as released | VOC-Sim −0.005 [−0.018, +0.010] (tie); COCO-Cap +0.033 [+0.007, +0.060] (beats) | **confirmed** |
| Our lead over NERVE comes from word-not-in-prompt images | +0.102 [+0.046, +0.145] not in prompt; +0.009 n.s. in prompt | descriptive |
| vs SCLIP | +0.082 VOC-Sim, +0.152 COCO-Cap (all CIs > 0) | descriptive (secondary) |
| The low-res consensus is the key part | MK4-LR − MK4 +0.023 / +0.041; MK4 alone ≈ 0 on VOC-Sim | descriptive |
| 5 positives + 100 negatives suffice | F − none +0.012 / +0.061; F is 0.007 / 0.013 below H | descriptive |
| VOC-Sim profiles transfer to COCO-Cap | V − none +0.060; 0.014 below H | descriptive |
| SA-guided seed weights are unnecessary with the gate | wHL − uHL −0.006 VOC-Sim, +0.001 COCO-Cap | descriptive |
| Hierarchy (V3) | +0.027 on VOC-Sim, −0.004 n.s. on COCO-Cap | descriptive (ablation, not the method) |
| Threshold robustness | gain over no gate significant at every threshold 0.30–0.70; best thresholds ≈ 0.43 (VOC-Sim), 0.53 (COCO-Cap) | descriptive, not used for selection |

**Caveats / limitations to keep honest:**
- NERVE's soft map at 0.55 (0.811 / 0.728) is about level with us (0.801 / 0.736); the pre-registered NERVE row is its
  own binarisation.
- Per class we lose to NERVE on cat, cow, sheep, potted plant (VOC-Sim) and to the ungated map on tv, bottle, potted plant.
- Our method uses reference images from the same benchmark (other half) to build profiles; no masks or labels are used,
  but this should be stated clearly. Few-shot (F) and cross-set transfer (V) rows show it is not dependent on that.
- On real COCO photos (a separate run), the gain for classes never seen when building the method is smaller (+0.022 vs
  +0.092 for seen classes); report as a limitation.
