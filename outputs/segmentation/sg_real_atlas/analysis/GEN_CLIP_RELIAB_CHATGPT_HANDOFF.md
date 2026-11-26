# Handoff for ChatGPT: SCLIP baselines and the reliability study on generated images (2026-11-09)

## What was run (following the plan agreed earlier)
1. **SCLIP baselines** (ViT-B/16; our port reproduces the repo on VOC val, 59.64 / 81.49) on VOC-Sim (490 images) and COCO-Cap (707 images), under BRACE's class-given binary protocol:
   - SCLIP alone;
   - SCLIP + the same DenseCRF as BRACE;
   - VOC21 vocabulary and COCO-Object vocabulary.
2. **SCLIP seed → SD.** One SD1.5 capture per image (each image's own generation run, replayed exactly). The SCLIP target probability is used as the seed of BRACE's frozen self-attention paths (C16, V3+R2, P1+R2).
3. **Pre-registered reliability study** (`gen_reliab_prereg.md`, committed before any statistic was computed). Can label-free signals predict the true IoU of the frozen BRACE mask?
   - **Signals:** BRACE-internal candidates vs a CLIP-agreement baseline (prior-art style, as in the AAAI 2025 CLIP filtering).
   - **Main metric:** within-class Spearman correlation, controlling for log predicted area.
   - **Rule:** choose the BRACE-internal signal on VOC-Sim, then test it against CLIP agreement on COCO-Cap with an image-bootstrap CI.
   - **Pixel level (descriptive):** error-pixel AUROC, also within boundary-distance bins.

No training and no large-scale generation.

## Headline results
- **BRACE beats SCLIP** on both generated sets: +0.078 (CI +0.063…+0.096) on VOC-Sim and +0.088 (CI +0.059…+0.116) on COCO-Cap.
- **SCLIP seed → SD C16 matches BRACE:** −0.006 on VOC-Sim and −0.004 on COCO-Cap; both CIs include 0.
  - With V3+R2 it is slightly better: 0.845 (+0.015, CI +0.001…+0.027) on VOC-Sim and 0.713 (+0.017, CI crosses 0) on COCO-Cap.
  - So on generated images, SD self-attention propagation adds about +0.07 to +0.08 on top of CLIP.
  - This is the opposite of the standard VOC21 benchmark on real photos, where SD propagation lost to CRF (−2.00 mIoU, pre-registered FAIL). The difference is the setting here: the class is given, there is a single target, and the image's own generation run is used.
- **COCO-Cap "person" is a BRACE seed failure.** BRACE scores 0.28, SCLIP 0.64, and SCLIP seed → SD 0.61. The cross-attention seed is the weak part on that class, not the self-attention.
- **Reliability test: no difference.** BRACE's confidence signal (s_conf) scores +0.435 on COCO-Cap vs +0.381 for CLIP agreement. Δ = +0.054, CI −0.056…+0.158.
  - For filtering, CLIP agreement is as good or better: keeping the top 50% gives 0.779–0.783 vs 0.750.
  - So there is no distinct whole-mask reliability contribution.
- **Pixel level.**
  - The best BRACE map (|V3 − τ|, i.e. distance from the threshold) beats boundary distance within distance bins by only about 0.05 (0.652 vs 0.590 on VOC-Sim, 0.637 vs 0.589 on COCO-Cap).
  - On COCO-Cap, the CLIP-based and seed-based maps are better (0.687 / 0.684).
  - Under the pre-registered stopping rule, region-level uncertainty is not pursued.
- **Bug note.** The first pixel-level run had a bug: the boundary-distance map was zero everywhere. It was fixed and rerun. The image-level results and the registered test do not use this map and are unaffected.

## Questions for ChatGPT
1. Should "SCLIP seed + SD self-attention on the image's own generation run" be framed as a result, given CLIPer/NERVE (real photos, multi-class)? Our setting is class-given segmentation for synthetic data.
2. Given the null reliability result, is the generated-data story now just "BRACE > CLIP-only on synthetic images, and a CLIP seed fixes BRACE's seed failures (person)"?
3. Is a downstream test worth running? For example, train a small segmenter on BRACE vs SCLIP vs (SCLIP → SD) pseudo-masks and evaluate on real VOC val.

---

# Full report 1: SCLIP vs BRACE on generated images


BRACE = frozen V3+R2 at τ 0.55 (paper protocol). SCLIP ViT-B/16 (our port, reproduces the repo on VOC val). *bin* = SCLIP's own closed-vocabulary prediction == target class (no τ); *prob* = SCLIP's target-class probability, min-max, thresholded like BRACE (OOF = τ chosen on 4 folds, crc32(image_id) % 5). R2 = the same DenseCRF as BRACE. SCLIP seed → SD = the SCLIP VOC21 target probability used as the seed of the frozen SD paths of the image's own generation run. Δ = row − BRACE, image-bootstrap 95% CI.

## VOC-Sim (490 images)

| pipeline | mIoU | prec / rec | Δ vs BRACE @0.55 | CI |
|---|---:|---|---:|---|
| **BRACE V3+R2 @0.55 (paper)** | **0.830** | 0.890 / 0.910 | — | — |
| BRACE V3+R2 OOF | 0.830 | 0.890 / 0.910 | +0.000 | +0.000…+0.000 |
| SD fine-only P1+R2 OOF | 0.810 | 0.870 / 0.912 | -0.020 | -0.028…-0.012 |
| SCLIP VOC21 vocab, bin | 0.712 | 0.772 / 0.926 | -0.118 | -0.136…-0.103 |
| SCLIP VOC21 vocab, bin + R2 | 0.752 | 0.804 / 0.934 | -0.078 | -0.096…-0.063 |
| SCLIP VOC21 vocab, prob OOF | 0.704 | 0.813 / 0.867 | -0.126 | -0.143…-0.111 |
| SCLIP VOC21 vocab, prob + R2 OOF | 0.749 | 0.850 / 0.882 | -0.081 | -0.098…-0.065 |
| SCLIP COCO-Object vocab, bin | 0.683 | 0.786 / 0.857 | -0.148 | -0.165…-0.132 |
| SCLIP COCO-Object vocab, bin + R2 | 0.722 | 0.819 / 0.867 | -0.108 | -0.126…-0.092 |
| SCLIP COCO-Object vocab, prob OOF | 0.649 | 0.846 / 0.761 | -0.181 | -0.199…-0.163 |
| SCLIP COCO-Object vocab, prob + R2 OOF | 0.692 | 0.889 / 0.777 | -0.138 | -0.158…-0.118 |
| SCLIP seed → SD C16 OOF | 0.824 | 0.885 / 0.923 | -0.006 | -0.020…+0.007 |
| SCLIP seed → SD V3+R2 OOF | 0.845 | 0.892 / 0.934 | +0.015 | +0.001…+0.027 |
| SCLIP seed → SD P1+R2 OOF | 0.819 | 0.880 / 0.928 | -0.011 | -0.027…+0.003 |

Per class (BRACE @0.55 / best SCLIP-only = SCLIP VOC21 vocab, bin + R2 / SCLIP seed → SD C16):

aeroplane 0.88 / 0.71 / 0.86, bicycle 0.79 / 0.65 / 0.78, bird 0.92 / 0.88 / 0.88, boat 0.86 / 0.58 / 0.84, bottle 0.88 / 0.74 / 0.80, bus 0.92 / 0.87 / 0.92, car 0.85 / 0.84 / 0.85, cat 0.85 / 0.85 / 0.91, chair 0.79 / 0.58 / 0.70, cow 0.90 / 0.88 / 0.94, dining-table 0.42 / 0.49 / 0.49, dog 0.93 / 0.86 / 0.91, horse 0.90 / 0.78 / 0.90, motorbike 0.87 / 0.80 / 0.86, person 0.81 / 0.84 / 0.81, potted-plant 0.72 / 0.73 / 0.73, sheep 0.75 / 0.88 / 0.94, sofa 0.96 / 0.80 / 0.87, train 0.74 / 0.68 / 0.84, tv 0.85 / 0.61 / 0.66

Regeneration replay: BRACE V3+R2 @0.55 from the new capture = 0.830 (stored 0.830).

SCLIP runtime: 0.08 s/image for both vocabularies (GPU, excl. CRF).

## COCO-Cap (707 images)

| pipeline | mIoU | prec / rec | Δ vs BRACE @0.55 | CI |
|---|---:|---|---:|---|
| **BRACE V3+R2 @0.55 (paper)** | **0.696** | 0.747 / 0.885 | — | — |
| BRACE V3+R2 OOF | 0.707 | 0.775 / 0.860 | +0.010 | +0.006…+0.015 |
| SD fine-only P1+R2 OOF | 0.683 | 0.750 / 0.860 | -0.013 | -0.026…-0.002 |
| SCLIP VOC21 vocab, bin | 0.574 | 0.710 / 0.797 | -0.122 | -0.148…-0.096 |
| SCLIP VOC21 vocab, bin + R2 | 0.608 | 0.755 / 0.802 | -0.088 | -0.116…-0.059 |
| SCLIP VOC21 vocab, prob OOF | 0.567 | 0.759 / 0.736 | -0.130 | -0.154…-0.103 |
| SCLIP VOC21 vocab, prob + R2 OOF | 0.601 | 0.812 / 0.746 | -0.095 | -0.120…-0.066 |
| SCLIP COCO-Object vocab, bin | 0.523 | 0.745 / 0.695 | -0.174 | -0.201…-0.146 |
| SCLIP COCO-Object vocab, bin + R2 | 0.552 | 0.796 / 0.703 | -0.144 | -0.173…-0.115 |
| SCLIP COCO-Object vocab, prob OOF | 0.537 | 0.781 / 0.662 | -0.159 | -0.185…-0.131 |
| SCLIP COCO-Object vocab, prob + R2 OOF | 0.573 | 0.839 / 0.670 | -0.123 | -0.151…-0.092 |
| SCLIP seed → SD C16 OOF | 0.692 | 0.836 / 0.825 | -0.004 | -0.030…+0.022 |
| SCLIP seed → SD V3+R2 OOF | 0.713 | 0.840 / 0.848 | +0.017 | -0.008…+0.042 |
| SCLIP seed → SD P1+R2 OOF | 0.691 | 0.810 / 0.849 | -0.005 | -0.031…+0.020 |

Per class (BRACE @0.55 / best SCLIP-only = SCLIP VOC21 vocab, bin + R2 / SCLIP seed → SD C16):

aeroplane 0.80 / 0.46 / 0.68, bicycle 0.63 / 0.63 / 0.73, bird 0.78 / 0.70 / 0.80, boat 0.59 / 0.38 / 0.50, bottle 0.50 / 0.42 / 0.53, bus 0.92 / 0.78 / 0.85, car 0.76 / 0.72 / 0.73, cat 0.87 / 0.72 / 0.79, chair 0.47 / 0.53 / 0.66, cow 0.87 / 0.81 / 0.86, dining table 0.45 / 0.54 / 0.47, dog 0.75 / 0.72 / 0.75, horse 0.72 / 0.59 / 0.76, motorbike 0.67 / 0.63 / 0.68, person 0.28 / 0.64 / 0.61, potted plant 0.74 / 0.48 / 0.60, sheep 0.87 / 0.81 / 0.88, sofa 0.70 / 0.63 / 0.75, train 0.75 / 0.60 / 0.65, tv 0.79 / 0.36 / 0.57

Regeneration replay: BRACE V3+R2 @0.55 from the new capture = 0.697 (stored 0.696).

SCLIP runtime: 0.08 s/image for both vocabularies (GPU, excl. CRF).


---

# Full report 2: reliability (pre-registered)


## VOC-Sim (490 images; frozen BRACE mask class-macro IoU 0.830; 30 masks with IoU < 0.5)

Image level. ρ = Spearman with the mask's true IoU; ρ_wc = mean within-class ρ; ρ_wc|size = within-class partial ρ controlling log predicted area; AUROC_wc = detecting IoU < 0.5 with scores ranked within class; retained class-macro mIoU when keeping the top q of each class (random keep = all-data mIoU in expectation).

| signal | ρ | ρ_wc | ρ_wc\|size | AUROC_wc | keep 90% | keep 70% | keep 50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| s_area | +0.087 | +0.237 | +0.017 | 0.598 | 0.833 | 0.839 | 0.848 |
| s_seedpeak | +0.018 | -0.124 | +0.111 | 0.465 | 0.825 | 0.823 | 0.829 |
| s_clip_iou | +0.443 | +0.418 | +0.359 | 0.702 | 0.848 | 0.866 | 0.880 |
| s_clip_soft | +0.349 | +0.268 | +0.359 | 0.663 | 0.847 | 0.864 | 0.871 |
| s_conf | +0.684 | +0.549 | +0.512 | 0.707 | 0.845 | 0.867 | 0.881 |
| s_cf_iou | +0.403 | +0.261 | +0.291 | 0.609 | 0.840 | 0.850 | 0.858 |
| s_c16_iou | +0.526 | +0.414 | +0.422 | 0.671 | 0.842 | 0.861 | 0.876 |
| s_crf_iou | +0.200 | +0.232 | +0.135 | 0.590 | 0.836 | 0.843 | 0.846 |
| s_call_iou | +0.456 | +0.257 | +0.225 | 0.574 | 0.837 | 0.846 | 0.860 |
| oracle | +1.000 | +1.000 | +1.000 | 0.761 | 0.859 | 0.895 | 0.918 |

All images: 0.830.

Pixel level: which pixels of the BRACE mask are wrong? AUROC of each uncertainty map for error pixels, mean over images (40k pixels each); *strat* = the same within boundary-distance bins (<2, 2–5, 5–10, 10–20, >20 px), i.e. what a map adds beyond 'errors sit at the boundary'.

| uncertainty | AUROC | AUROC within distance bins |
|---|---:|---:|
| u_boundary | 0.878 | 0.590 |
| u_conf | 0.887 | 0.652 |
| u_cf | 0.696 | 0.559 |
| u_c16 | 0.672 | 0.575 |
| u_clip | 0.827 | 0.631 |
| u_seed | 0.815 | 0.634 |
| u_call | 0.699 | 0.549 |

## COCO-Cap (707 images; frozen BRACE mask class-macro IoU 0.696; 142 masks with IoU < 0.5)

Image level. ρ = Spearman with the mask's true IoU; ρ_wc = mean within-class ρ; ρ_wc|size = within-class partial ρ controlling log predicted area; AUROC_wc = detecting IoU < 0.5 with scores ranked within class; retained class-macro mIoU when keeping the top q of each class (random keep = all-data mIoU in expectation).

| signal | ρ | ρ_wc | ρ_wc\|size | AUROC_wc | keep 90% | keep 70% | keep 50% |
|---|---:|---:|---:|---:|---:|---:|---:|
| s_area | -0.055 | +0.123 | -0.016 | 0.495 | 0.699 | 0.701 | 0.711 |
| s_seedpeak | +0.287 | +0.003 | +0.100 | 0.552 | 0.694 | 0.697 | 0.700 |
| s_clip_iou | +0.451 | +0.433 | +0.354 | 0.733 | 0.717 | 0.751 | 0.783 |
| s_clip_soft | +0.445 | +0.351 | +0.381 | 0.734 | 0.714 | 0.758 | 0.779 |
| s_conf | +0.502 | +0.441 | +0.435 | 0.581 | 0.709 | 0.734 | 0.750 |
| s_cf_iou | +0.455 | +0.344 | +0.357 | 0.638 | 0.714 | 0.730 | 0.757 |
| s_c16_iou | +0.490 | +0.402 | +0.404 | 0.657 | 0.717 | 0.745 | 0.759 |
| s_crf_iou | +0.066 | +0.180 | +0.162 | 0.483 | 0.700 | 0.703 | 0.709 |
| s_call_iou | +0.487 | +0.392 | +0.380 | 0.649 | 0.713 | 0.733 | 0.759 |
| oracle | +1.000 | +1.000 | +1.000 | 0.892 | 0.743 | 0.815 | 0.871 |

All images: 0.696.

Pixel level: which pixels of the BRACE mask are wrong? AUROC of each uncertainty map for error pixels, mean over images (40k pixels each); *strat* = the same within boundary-distance bins (<2, 2–5, 5–10, 10–20, >20 px), i.e. what a map adds beyond 'errors sit at the boundary'.

| uncertainty | AUROC | AUROC within distance bins |
|---|---:|---:|
| u_boundary | 0.856 | 0.589 |
| u_conf | 0.860 | 0.637 |
| u_cf | 0.720 | 0.601 |
| u_c16 | 0.662 | 0.589 |
| u_clip | 0.862 | 0.687 |
| u_seed | 0.859 | 0.684 |
| u_call | 0.754 | 0.583 |

## Registered transfer test

BRACE-internal signal chosen on VOC-Sim by within-class size-controlled ρ: **s_conf** (VOC-Sim +0.512). COCO-Cap: s_conf +0.435 vs CLIP-agreement (s_clip_soft, prior-art style) +0.381.
Δ (BRACE-internal − CLIP agreement) = +0.054, image-bootstrap 95% CI -0.056…+0.158 → **no difference**.


---

# Appendix: the pre-registration


Written 2026-11-09, before any reliability statistic has been computed. BRACE (V3+R2 at τ 0.55) is frozen. Ground-truth masks are used only to evaluate the predictors, never inside them. Code: `scripts/sg_gen_clip.py reliab`.

**Target.** For each generated image (VOC-Sim 490, COCO-Cap 707), the true IoU of the frozen BRACE mask. A mask counts as low quality when its IoU is below 0.5.

**Image-level signals.** All are label-free; higher means the mask is predicted to be more reliable.

| group | signal | definition |
|---|---|---|
| baseline | s_area | predicted mask area |
| baseline | s_seedpeak | A3 seed 99th percentile − median |
| prior-art style | s_clip_iou | IoU of the BRACE mask with SCLIP's binary prediction |
| prior-art style | s_clip_soft | mean SCLIP target probability inside the mask − outside |
| BRACE-internal candidate | s_conf | −(fraction of pixels whose V3 lies within ±0.1 of τ, relative to the mask area) |
| BRACE-internal candidate | s_cf_iou | IoU of fine-only P1@τ with V3@τ (coarse/fine agreement) |
| BRACE-internal candidate | s_c16_iou | IoU of C16@τ with V3@τ |
| BRACE-internal candidate | s_crf_iou | IoU of V3@τ before and after CRF |
| BRACE-internal candidate | s_call_iou | mean pairwise IoU of the 11 per-call P1 masks@τ (diffusion-call stability) |

**Metrics.**
- Spearman ρ, overall.
- Mean within-class ρ.
- Mean within-class partial ρ controlling for log predicted area. **This is the main metric:** it removes the "easy class / large object" shortcut.
- AUROC for IoU < 0.5, using ranks computed within each class.
- Retained class-macro mIoU when keeping the top 90/70/50% of each class.

**Registered rule.**
1. On VOC-Sim, choose the BRACE-internal candidate with the highest within-class size-controlled ρ.
2. On COCO-Cap, compare that signal with s_clip_soft (CLIP agreement, the prior-art-style filter) on the same metric, with an image-bootstrap 95% CI (1000 resamples).
3. The BRACE-internal signal is "better" only if the CI is > 0.

**Pixel-level (descriptive, not pass/fail).** Which pixels of the BRACE mask are wrong? Each uncertainty map is scored by AUROC for error pixels, averaged over images. Maps:
- boundary distance;
- |V3 − τ|;
- |P1 − V3|;
- |C16 − V3|;
- |mask − SCLIP probability|;
- |mask − A3 seed|;
- the per-call std of P1.

Each map is also scored within boundary-distance bins, to measure what it adds beyond "errors sit at the boundary".

**If the BRACE-internal signal is not better than CLIP agreement,** there is no distinct whole-mask reliability contribution. Region-level uncertainty is pursued only if a BRACE map beats boundary distance within distance bins by a clear margin on both sets.
