# Pre-registration: reliability of the frozen BRACE mask on generated images

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
