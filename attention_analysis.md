# Attention Segmentation Analysis

## Executive summary

This report explains the VOC-Sim results over **490 annotated images and 20 classes**. It consolidates **78,108 evaluated configurations** (counting the exhaustive layer subsets) and adds image-level diagnostics without running another model-selection sweep.

The published, unoptimized OVAM pipeline reached **0.7036 mIoU**. The best focused seed/operator result reached **0.7922**, joint head/time tuning reached **0.7991** without morphology, and removing predicted components below 1% of the image produced the final **0.7997 mIoU**. This is an absolute gain of **0.0961** (13.7% relative).

The final method improves pooled IoU for **17/20 classes** and image IoU for **376/490 images** (0 ties). Its paired mean image-IoU gain is **+0.0912**, with a bootstrap 95% interval of **[+0.0760, +0.1064]** (20,000 image resamples; seed 20261012).

The evidence supports a coherent explanation: selected cross-attention heads give a seed that works better with the downstream operator; the last 64×64 upsampling self-attention module spreads that signal along high-resolution object structure; heads 3 and 7 reduce performance when combined with the retained set; middle-late denoising steps are more spatially aligned; and small-component removal suppresses isolated false positives.

## Experimental protocol

Each image was regenerated with its released prompt, seed, 30 inference steps, Stable Diffusion 1.5, and guidance scale 7.5. Attention was analyzed in memory. After scalar measurements and an optional visual panel were written, all Q/K tensors, affinity matrices, latents, and temporary maps were released. No raw attention tensor dataset was saved.

Qualitative panels use the released OVAM image beside attention captured from our reproduction. Earlier reproduction checks found very small pixel-level differences caused by the software/runtime stack, while composition and mask geometry remained aligned; the released annotation is the evaluation target throughout.

The published OVAM baseline is measured from the authors' released unoptimized masks. It aggregates ordinary-prompt cross-attention, multiplies it by an aggregate of the highest-resolution self-attention maps rescaled to [0.85, 1], thresholds at 0.40, and applies DenseCRF. Token optimization is excluded. Thresholding the authors' released cross-attention heatmaps alone gives 0.6616 mIoU (the paper reports 66.2 after rounding). Our regenerated cross-attention reference gives 0.6540 because the current diffusion software stack does not reproduce their stored attention exactly.

The final method uses the fixed top-four cross-attention heads in each of the previously selected 12 layers, the final 64×64 self-attention module, four propagation iterations, self-attention heads 0/1/2/4/5/6, denoising steps 15–30, percentile 1–99 normalization, threshold 0.45, and removal of connected components smaller than 1% of the image.

## What the diagnostic measures mean

- **IoU and average precision** measure spatial localization. Average precision is threshold-free and helps distinguish map quality from the chosen 0.45 cutoff.
- **Normalized entropy and effective support** measure spread. Lower values mean concentration, but concentration is useful only when it falls on the object.
- **Foreground/background gap and ROC AUC** measure how well attention ranks object pixels above background pixels.
- **Boundary gradient ratio** measures whether changes in the map concentrate around the annotated object boundary.
- **Centroid error** measures displacement between attention mass and the target center.
- **Object area, box occupancy, components, boundary fraction, and compactness** describe class geometry and let us test whether improvements are dominated by large or simple objects.

## Main performance progression

| Method | Configuration | mIoU | Absolute gain over OVAM |
|---|---|---:|---:|
| Authors' OVAM CA precursor | released heatmaps | 0.6616 | -0.0421 |
| Our regenerated CA precursor | `ovam_cross_attention_only` | 0.6540 | -0.0496 |
| Published OVAM, unoptimized | CA + SA fusion + DenseCRF | 0.7036 | 0.0000 |
| Earlier SA winner | best-12 CA, final 64×64 SA, 4 iterations | 0.7777 | 0.0741 |
| Focused winner | `top4_heads_percentile__r64__iter2__percentile_01_99__t0.45` | 0.7922 | 0.0886 |
| Joint winner, no morphology | `iter3__leave_3_7__steps_15_20__t0.40` | 0.7991 | 0.0955 |
| Final winner | `iter4__leave_3_7__trailing_15_30__t0.45__remove_fraction_0.01` | 0.7997 | 0.0961 |

The 0.7991 best result before morphology shows that almost all improvement comes from attention selection and propagation. In the matched final iteration/head/time/threshold configuration, the 1% component rule changes mIoU from 0.79910 to 0.79974, consistent with removing scattered islands rather than repairing the main object.

### Pooled class IoU against published OVAM

| Class | Published OVAM | Final | Difference |
|---|---:|---:|---:|
| aeroplane | 0.717 | 0.838 | +0.121 |
| bicycle | 0.606 | 0.760 | +0.153 |
| bird | 0.768 | 0.859 | +0.091 |
| boat | 0.709 | 0.858 | +0.149 |
| bottle | 0.803 | 0.882 | +0.079 |
| bus | 0.913 | 0.877 | -0.036 |
| car | 0.829 | 0.785 | -0.043 |
| cat | 0.723 | 0.811 | +0.089 |
| chair | 0.410 | 0.723 | +0.314 |
| cow | 0.898 | 0.915 | +0.017 |
| dining-table | 0.258 | 0.407 | +0.149 |
| dog | 0.775 | 0.887 | +0.112 |
| horse | 0.803 | 0.887 | +0.084 |
| motorbike | 0.780 | 0.857 | +0.077 |
| person | 0.561 | 0.685 | +0.124 |
| potted-plant | 0.671 | 0.603 | -0.069 |
| sheep | 0.820 | 0.873 | +0.053 |
| sofa | 0.744 | 0.928 | +0.184 |
| train | 0.586 | 0.777 | +0.190 |
| tv | 0.700 | 0.782 | +0.083 |

## Class size and shape

| Class | n | Mean area | Box area | Fill | Components | Published OVAM image IoU | Final image IoU | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 24 | 0.196 | 0.504 | 0.395 | 2.00 | 0.748 | 0.839 | +0.091 |
| bicycle | 25 | 0.335 | 0.703 | 0.492 | 4.84 | 0.615 | 0.758 | +0.144 |
| bird | 25 | 0.177 | 0.442 | 0.421 | 3.28 | 0.801 | 0.842 | +0.041 |
| boat | 25 | 0.223 | 0.398 | 0.577 | 2.08 | 0.694 | 0.834 | +0.140 |
| bottle | 25 | 0.199 | 0.287 | 0.708 | 1.16 | 0.819 | 0.877 | +0.057 |
| bus | 25 | 0.356 | 0.453 | 0.796 | 1.64 | 0.902 | 0.863 | -0.038 |
| car | 23 | 0.348 | 0.461 | 0.733 | 1.48 | 0.836 | 0.804 | -0.032 |
| cat | 25 | 0.693 | 0.873 | 0.786 | 9.80 | 0.708 | 0.822 | +0.114 |
| chair | 24 | 0.212 | 0.553 | 0.383 | 4.12 | 0.428 | 0.711 | +0.283 |
| cow | 25 | 0.441 | 0.752 | 0.572 | 2.20 | 0.899 | 0.922 | +0.023 |
| dining-table | 25 | 0.240 | 0.526 | 0.426 | 7.64 | 0.253 | 0.405 | +0.152 |
| dog | 25 | 0.406 | 0.604 | 0.650 | 1.60 | 0.773 | 0.898 | +0.125 |
| horse | 23 | 0.397 | 0.641 | 0.588 | 1.35 | 0.826 | 0.887 | +0.062 |
| motorbike | 25 | 0.367 | 0.646 | 0.572 | 4.88 | 0.778 | 0.860 | +0.082 |
| person | 24 | 0.491 | 0.719 | 0.666 | 3.46 | 0.589 | 0.716 | +0.127 |
| potted-plant | 25 | 0.526 | 0.873 | 0.596 | 8.08 | 0.660 | 0.617 | -0.044 |
| sheep | 25 | 0.497 | 0.663 | 0.698 | 2.08 | 0.846 | 0.894 | +0.049 |
| sofa | 25 | 0.503 | 0.572 | 0.865 | 1.24 | 0.764 | 0.923 | +0.159 |
| train | 25 | 0.246 | 0.430 | 0.586 | 2.12 | 0.572 | 0.772 | +0.200 |
| tv | 22 | 0.573 | 0.625 | 0.913 | 7.91 | 0.717 | 0.801 | +0.085 |

Across individual images, foreground area and improvement have Pearson **r=-0.168** (p=0.000189) and Spearman **ρ=-0.095** (p=0.0349). This correlation is descriptive: it indicates how gain varies with size, not that size causes the gain.

![Class size versus improvement](outputs/analysis/voc_sim_attention_report/figures/class_size_vs_gain.svg)

## Why cross-attention selection helps

The strongest individual CA layer by mean image IoU in this diagnostic is `mid_block.attentions.0.transformer_blocks.0.attn2` (0.569). The top-four-head seed has slightly lower entropy and changes centroid error from **0.046** to **0.045**, but its standardized foreground/background gap changes from **1.803** to **1.796**. Its advantage is therefore better downstream compatibility and threshold calibration, rather than a universal improvement in every raw-map diagnostic.

![Cross-attention layers](outputs/analysis/voc_sim_attention_report/figures/ca_layer_iou.svg)

## Why the final 64×64 self-attention module helps

Applying the same top-four CA seed, four propagation iterations, and late-step window to every SA module identifies `up_blocks.3.attentions.2.transformer_blocks.0.attn1` as the best module by mean image IoU (0.796). The selected final upsampling module is highlighted in orange in the figure. Its 64×64 token grid preserves small parts and boundaries that coarser 8×8–32×32 maps cannot represent, while its late decoder position has already integrated the global object layout from earlier U-Net stages.

For the final selected map, normalized entropy is **0.937**, effective support is **0.617**, foreground/background standardized gap is **1.976**, boundary gradient ratio is **3.952**, and centroid error is **0.037**. Across images, final-map entropy and IoU have Spearman **ρ=-0.181** (p=5.61e-05); entropy alone is therefore interpreted together with alignment metrics.

![Self-attention modules](outputs/analysis/voc_sim_attention_report/figures/sa_module_iou.svg)

## Why heads 3 and 7 are removed

Under the final module and late-step window, the best individual head is head **5** (0.784 mean image IoU). Heads 3 and 7 average **0.751**, compared with **0.767** across retained heads, but a retained head can still be weaker in isolation. Their exclusion is a joint subset result: it reflects complementarity with the retained heads rather than a simple ranking of standalone head IoUs.

| Head | IoU | AP | Entropy | FG/BG gap | Boundary ratio | Centroid error | Used? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.776 | 0.899 | 0.935 | 1.884 | 3.318 | 0.042 | yes |
| 1 | 0.783 | 0.922 | 0.938 | 1.910 | 4.010 | 0.040 | yes |
| 2 | 0.768 | 0.926 | 0.939 | 1.912 | 3.375 | 0.040 | yes |
| 3 | 0.739 | 0.918 | 0.940 | 1.854 | 3.306 | 0.042 | no |
| 4 | 0.718 | 0.902 | 0.943 | 1.796 | 3.006 | 0.045 | yes |
| 5 | 0.784 | 0.927 | 0.934 | 1.941 | 3.511 | 0.039 | yes |
| 6 | 0.776 | 0.952 | 0.940 | 1.980 | 3.910 | 0.037 | yes |
| 7 | 0.763 | 0.933 | 0.939 | 1.904 | 3.488 | 0.040 | no |

![Self-attention heads](outputs/analysis/voc_sim_attention_report/figures/sa_head_iou.svg)

## Why later denoising steps help

The mean individual-step IoU is **0.670** for steps 0–14 and **0.798** for steps 15–30. Early attention is formed while the latent is highly noisy and tends to encode coarse composition. Quality plateaus around steps 15–25 and declines slightly at the final steps. Averaging steps 15–30 reduces step-specific noise while retaining the stable middle-late geometry.

![Self-attention timesteps](outputs/analysis/voc_sim_attention_report/figures/sa_timestep_iou.svg)

## Qualitative evidence

Each panel shows the released image, annotation, OVAM cross-attention precursor, selected cross-attention seed, self-attention map before joint tuning, and final self-attention map. Panels were sampled at fixed positions within every class rather than selected by score. These panels diagnose our attention operators; they do not visualize OVAM's post-CRF mask.

- aeroplane / ovam-voc-sim-0: [outputs/analysis/voc_sim_attention_report/class_00_aeroplane/panels/ovam-voc-sim-0.png](outputs/analysis/voc_sim_attention_report/class_00_aeroplane/panels/ovam-voc-sim-0.png)
- aeroplane / ovam-voc-sim-13: [outputs/analysis/voc_sim_attention_report/class_00_aeroplane/panels/ovam-voc-sim-13.png](outputs/analysis/voc_sim_attention_report/class_00_aeroplane/panels/ovam-voc-sim-13.png)
- aeroplane / ovam-voc-sim-24: [outputs/analysis/voc_sim_attention_report/class_00_aeroplane/panels/ovam-voc-sim-24.png](outputs/analysis/voc_sim_attention_report/class_00_aeroplane/panels/ovam-voc-sim-24.png)
- bicycle / ovam-voc-sim-25: [outputs/analysis/voc_sim_attention_report/class_01_bicycle/panels/ovam-voc-sim-25.png](outputs/analysis/voc_sim_attention_report/class_01_bicycle/panels/ovam-voc-sim-25.png)
- bicycle / ovam-voc-sim-37: [outputs/analysis/voc_sim_attention_report/class_01_bicycle/panels/ovam-voc-sim-37.png](outputs/analysis/voc_sim_attention_report/class_01_bicycle/panels/ovam-voc-sim-37.png)
- bicycle / ovam-voc-sim-49: [outputs/analysis/voc_sim_attention_report/class_01_bicycle/panels/ovam-voc-sim-49.png](outputs/analysis/voc_sim_attention_report/class_01_bicycle/panels/ovam-voc-sim-49.png)
- bird / ovam-voc-sim-50: [outputs/analysis/voc_sim_attention_report/class_02_bird/panels/ovam-voc-sim-50.png](outputs/analysis/voc_sim_attention_report/class_02_bird/panels/ovam-voc-sim-50.png)
- bird / ovam-voc-sim-62: [outputs/analysis/voc_sim_attention_report/class_02_bird/panels/ovam-voc-sim-62.png](outputs/analysis/voc_sim_attention_report/class_02_bird/panels/ovam-voc-sim-62.png)
- bird / ovam-voc-sim-74: [outputs/analysis/voc_sim_attention_report/class_02_bird/panels/ovam-voc-sim-74.png](outputs/analysis/voc_sim_attention_report/class_02_bird/panels/ovam-voc-sim-74.png)
- boat / ovam-voc-sim-75: [outputs/analysis/voc_sim_attention_report/class_03_boat/panels/ovam-voc-sim-75.png](outputs/analysis/voc_sim_attention_report/class_03_boat/panels/ovam-voc-sim-75.png)
- boat / ovam-voc-sim-87: [outputs/analysis/voc_sim_attention_report/class_03_boat/panels/ovam-voc-sim-87.png](outputs/analysis/voc_sim_attention_report/class_03_boat/panels/ovam-voc-sim-87.png)
- boat / ovam-voc-sim-99: [outputs/analysis/voc_sim_attention_report/class_03_boat/panels/ovam-voc-sim-99.png](outputs/analysis/voc_sim_attention_report/class_03_boat/panels/ovam-voc-sim-99.png)
- bottle / ovam-voc-sim-100: [outputs/analysis/voc_sim_attention_report/class_04_bottle/panels/ovam-voc-sim-100.png](outputs/analysis/voc_sim_attention_report/class_04_bottle/panels/ovam-voc-sim-100.png)
- bottle / ovam-voc-sim-112: [outputs/analysis/voc_sim_attention_report/class_04_bottle/panels/ovam-voc-sim-112.png](outputs/analysis/voc_sim_attention_report/class_04_bottle/panels/ovam-voc-sim-112.png)
- bottle / ovam-voc-sim-124: [outputs/analysis/voc_sim_attention_report/class_04_bottle/panels/ovam-voc-sim-124.png](outputs/analysis/voc_sim_attention_report/class_04_bottle/panels/ovam-voc-sim-124.png)
- bus / ovam-voc-sim-125: [outputs/analysis/voc_sim_attention_report/class_05_bus/panels/ovam-voc-sim-125.png](outputs/analysis/voc_sim_attention_report/class_05_bus/panels/ovam-voc-sim-125.png)
- bus / ovam-voc-sim-137: [outputs/analysis/voc_sim_attention_report/class_05_bus/panels/ovam-voc-sim-137.png](outputs/analysis/voc_sim_attention_report/class_05_bus/panels/ovam-voc-sim-137.png)
- bus / ovam-voc-sim-149: [outputs/analysis/voc_sim_attention_report/class_05_bus/panels/ovam-voc-sim-149.png](outputs/analysis/voc_sim_attention_report/class_05_bus/panels/ovam-voc-sim-149.png)
- car / ovam-voc-sim-150: [outputs/analysis/voc_sim_attention_report/class_06_car/panels/ovam-voc-sim-150.png](outputs/analysis/voc_sim_attention_report/class_06_car/panels/ovam-voc-sim-150.png)
- car / ovam-voc-sim-162: [outputs/analysis/voc_sim_attention_report/class_06_car/panels/ovam-voc-sim-162.png](outputs/analysis/voc_sim_attention_report/class_06_car/panels/ovam-voc-sim-162.png)
- car / ovam-voc-sim-174: [outputs/analysis/voc_sim_attention_report/class_06_car/panels/ovam-voc-sim-174.png](outputs/analysis/voc_sim_attention_report/class_06_car/panels/ovam-voc-sim-174.png)
- cat / ovam-voc-sim-175: [outputs/analysis/voc_sim_attention_report/class_07_cat/panels/ovam-voc-sim-175.png](outputs/analysis/voc_sim_attention_report/class_07_cat/panels/ovam-voc-sim-175.png)
- cat / ovam-voc-sim-187: [outputs/analysis/voc_sim_attention_report/class_07_cat/panels/ovam-voc-sim-187.png](outputs/analysis/voc_sim_attention_report/class_07_cat/panels/ovam-voc-sim-187.png)
- cat / ovam-voc-sim-199: [outputs/analysis/voc_sim_attention_report/class_07_cat/panels/ovam-voc-sim-199.png](outputs/analysis/voc_sim_attention_report/class_07_cat/panels/ovam-voc-sim-199.png)
- chair / ovam-voc-sim-200: [outputs/analysis/voc_sim_attention_report/class_08_chair/panels/ovam-voc-sim-200.png](outputs/analysis/voc_sim_attention_report/class_08_chair/panels/ovam-voc-sim-200.png)
- chair / ovam-voc-sim-212: [outputs/analysis/voc_sim_attention_report/class_08_chair/panels/ovam-voc-sim-212.png](outputs/analysis/voc_sim_attention_report/class_08_chair/panels/ovam-voc-sim-212.png)
- chair / ovam-voc-sim-224: [outputs/analysis/voc_sim_attention_report/class_08_chair/panels/ovam-voc-sim-224.png](outputs/analysis/voc_sim_attention_report/class_08_chair/panels/ovam-voc-sim-224.png)
- cow / ovam-voc-sim-225: [outputs/analysis/voc_sim_attention_report/class_09_cow/panels/ovam-voc-sim-225.png](outputs/analysis/voc_sim_attention_report/class_09_cow/panels/ovam-voc-sim-225.png)
- cow / ovam-voc-sim-237: [outputs/analysis/voc_sim_attention_report/class_09_cow/panels/ovam-voc-sim-237.png](outputs/analysis/voc_sim_attention_report/class_09_cow/panels/ovam-voc-sim-237.png)
- cow / ovam-voc-sim-249: [outputs/analysis/voc_sim_attention_report/class_09_cow/panels/ovam-voc-sim-249.png](outputs/analysis/voc_sim_attention_report/class_09_cow/panels/ovam-voc-sim-249.png)
- dining-table / ovam-voc-sim-250: [outputs/analysis/voc_sim_attention_report/class_10_dining-table/panels/ovam-voc-sim-250.png](outputs/analysis/voc_sim_attention_report/class_10_dining-table/panels/ovam-voc-sim-250.png)
- dining-table / ovam-voc-sim-262: [outputs/analysis/voc_sim_attention_report/class_10_dining-table/panels/ovam-voc-sim-262.png](outputs/analysis/voc_sim_attention_report/class_10_dining-table/panels/ovam-voc-sim-262.png)
- dining-table / ovam-voc-sim-274: [outputs/analysis/voc_sim_attention_report/class_10_dining-table/panels/ovam-voc-sim-274.png](outputs/analysis/voc_sim_attention_report/class_10_dining-table/panels/ovam-voc-sim-274.png)
- dog / ovam-voc-sim-275: [outputs/analysis/voc_sim_attention_report/class_11_dog/panels/ovam-voc-sim-275.png](outputs/analysis/voc_sim_attention_report/class_11_dog/panels/ovam-voc-sim-275.png)
- dog / ovam-voc-sim-287: [outputs/analysis/voc_sim_attention_report/class_11_dog/panels/ovam-voc-sim-287.png](outputs/analysis/voc_sim_attention_report/class_11_dog/panels/ovam-voc-sim-287.png)
- dog / ovam-voc-sim-299: [outputs/analysis/voc_sim_attention_report/class_11_dog/panels/ovam-voc-sim-299.png](outputs/analysis/voc_sim_attention_report/class_11_dog/panels/ovam-voc-sim-299.png)
- horse / ovam-voc-sim-300: [outputs/analysis/voc_sim_attention_report/class_12_horse/panels/ovam-voc-sim-300.png](outputs/analysis/voc_sim_attention_report/class_12_horse/panels/ovam-voc-sim-300.png)
- horse / ovam-voc-sim-312: [outputs/analysis/voc_sim_attention_report/class_12_horse/panels/ovam-voc-sim-312.png](outputs/analysis/voc_sim_attention_report/class_12_horse/panels/ovam-voc-sim-312.png)
- horse / ovam-voc-sim-324: [outputs/analysis/voc_sim_attention_report/class_12_horse/panels/ovam-voc-sim-324.png](outputs/analysis/voc_sim_attention_report/class_12_horse/panels/ovam-voc-sim-324.png)
- motorbike / ovam-voc-sim-325: [outputs/analysis/voc_sim_attention_report/class_13_motorbike/panels/ovam-voc-sim-325.png](outputs/analysis/voc_sim_attention_report/class_13_motorbike/panels/ovam-voc-sim-325.png)
- motorbike / ovam-voc-sim-337: [outputs/analysis/voc_sim_attention_report/class_13_motorbike/panels/ovam-voc-sim-337.png](outputs/analysis/voc_sim_attention_report/class_13_motorbike/panels/ovam-voc-sim-337.png)
- motorbike / ovam-voc-sim-349: [outputs/analysis/voc_sim_attention_report/class_13_motorbike/panels/ovam-voc-sim-349.png](outputs/analysis/voc_sim_attention_report/class_13_motorbike/panels/ovam-voc-sim-349.png)
- person / ovam-voc-sim-350: [outputs/analysis/voc_sim_attention_report/class_14_person/panels/ovam-voc-sim-350.png](outputs/analysis/voc_sim_attention_report/class_14_person/panels/ovam-voc-sim-350.png)
- person / ovam-voc-sim-362: [outputs/analysis/voc_sim_attention_report/class_14_person/panels/ovam-voc-sim-362.png](outputs/analysis/voc_sim_attention_report/class_14_person/panels/ovam-voc-sim-362.png)
- person / ovam-voc-sim-374: [outputs/analysis/voc_sim_attention_report/class_14_person/panels/ovam-voc-sim-374.png](outputs/analysis/voc_sim_attention_report/class_14_person/panels/ovam-voc-sim-374.png)
- potted-plant / ovam-voc-sim-375: [outputs/analysis/voc_sim_attention_report/class_15_potted-plant/panels/ovam-voc-sim-375.png](outputs/analysis/voc_sim_attention_report/class_15_potted-plant/panels/ovam-voc-sim-375.png)
- potted-plant / ovam-voc-sim-387: [outputs/analysis/voc_sim_attention_report/class_15_potted-plant/panels/ovam-voc-sim-387.png](outputs/analysis/voc_sim_attention_report/class_15_potted-plant/panels/ovam-voc-sim-387.png)
- potted-plant / ovam-voc-sim-399: [outputs/analysis/voc_sim_attention_report/class_15_potted-plant/panels/ovam-voc-sim-399.png](outputs/analysis/voc_sim_attention_report/class_15_potted-plant/panels/ovam-voc-sim-399.png)
- sheep / ovam-voc-sim-400: [outputs/analysis/voc_sim_attention_report/class_16_sheep/panels/ovam-voc-sim-400.png](outputs/analysis/voc_sim_attention_report/class_16_sheep/panels/ovam-voc-sim-400.png)
- sheep / ovam-voc-sim-412: [outputs/analysis/voc_sim_attention_report/class_16_sheep/panels/ovam-voc-sim-412.png](outputs/analysis/voc_sim_attention_report/class_16_sheep/panels/ovam-voc-sim-412.png)
- sheep / ovam-voc-sim-424: [outputs/analysis/voc_sim_attention_report/class_16_sheep/panels/ovam-voc-sim-424.png](outputs/analysis/voc_sim_attention_report/class_16_sheep/panels/ovam-voc-sim-424.png)
- sofa / ovam-voc-sim-425: [outputs/analysis/voc_sim_attention_report/class_17_sofa/panels/ovam-voc-sim-425.png](outputs/analysis/voc_sim_attention_report/class_17_sofa/panels/ovam-voc-sim-425.png)
- sofa / ovam-voc-sim-437: [outputs/analysis/voc_sim_attention_report/class_17_sofa/panels/ovam-voc-sim-437.png](outputs/analysis/voc_sim_attention_report/class_17_sofa/panels/ovam-voc-sim-437.png)
- sofa / ovam-voc-sim-449: [outputs/analysis/voc_sim_attention_report/class_17_sofa/panels/ovam-voc-sim-449.png](outputs/analysis/voc_sim_attention_report/class_17_sofa/panels/ovam-voc-sim-449.png)
- train / ovam-voc-sim-450: [outputs/analysis/voc_sim_attention_report/class_18_train/panels/ovam-voc-sim-450.png](outputs/analysis/voc_sim_attention_report/class_18_train/panels/ovam-voc-sim-450.png)
- train / ovam-voc-sim-462: [outputs/analysis/voc_sim_attention_report/class_18_train/panels/ovam-voc-sim-462.png](outputs/analysis/voc_sim_attention_report/class_18_train/panels/ovam-voc-sim-462.png)
- train / ovam-voc-sim-474: [outputs/analysis/voc_sim_attention_report/class_18_train/panels/ovam-voc-sim-474.png](outputs/analysis/voc_sim_attention_report/class_18_train/panels/ovam-voc-sim-474.png)
- tv / ovam-voc-sim-475: [outputs/analysis/voc_sim_attention_report/class_19_tv/panels/ovam-voc-sim-475.png](outputs/analysis/voc_sim_attention_report/class_19_tv/panels/ovam-voc-sim-475.png)
- tv / ovam-voc-sim-486: [outputs/analysis/voc_sim_attention_report/class_19_tv/panels/ovam-voc-sim-486.png](outputs/analysis/voc_sim_attention_report/class_19_tv/panels/ovam-voc-sim-486.png)
- tv / ovam-voc-sim-499: [outputs/analysis/voc_sim_attention_report/class_19_tv/panels/ovam-voc-sim-499.png](outputs/analysis/voc_sim_attention_report/class_19_tv/panels/ovam-voc-sim-499.png)

## Interpretation limits

- These are exploratory ablations and explanations on VOC-Sim. Configuration selection and final reporting use the same 490 images, so the 0.7997 result needs confirmation on the other datasets already planned.
- Individual layer, head, and step profiles isolate one component at a time. They explain association with performance but do not prove a unique causal mechanism because attention components interact.
- VOC-Sim contains generated, usually centered objects and binary foreground masks. Conclusions about clutter, multiple instances, and natural-image class imbalance require external validation.
- The released images are not byte-identical to our regenerated pixels under the current software stack. Attention is captured from the seeded reproduction, while scoring uses the released spatial mask; the earlier direct checks showed aligned composition and mask geometry.
- Map entropy has no universal optimum. Very diffuse and very concentrated maps can both fail; foreground separation, boundary alignment, centroid error, and IoU provide the needed spatial context.

## Reproducible artifacts

- Compact image records: `outputs/analysis/voc_sim_attention_report/analysis_records.json.gz`
- Class geometry and performance: `outputs/analysis/voc_sim_attention_report/class_geometry_and_performance.csv`
- Selected-method statistics: `outputs/analysis/voc_sim_attention_report/selected_method_map_statistics.csv`
- CA layer statistics: `outputs/analysis/voc_sim_attention_report/cross_attention_layer_statistics.csv`
- SA module statistics: `outputs/analysis/voc_sim_attention_report/self_attention_module_statistics.csv`
- SA head statistics: `outputs/analysis/voc_sim_attention_report/winning_module_head_statistics.csv`
- SA timestep statistics: `outputs/analysis/voc_sim_attention_report/winning_module_timestep_statistics.csv`
