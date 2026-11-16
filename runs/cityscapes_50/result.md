# Cityscapes LoRA 50: completed threshold comparison

**50 images × 14 classes × 3 methods × 3 thresholds = 6300 post-CRF binary
masks**, plus 6300 pre-CRF masks and 700 comparison sheets. Thresholds were
fixed at **0.40 / 0.55 / 0.70** before looking at masks. Methods: BRACE,
SCLIP-seeded BRACE, and early geometric seed fusion. Every method uses the
same image-anchored attention capture for an image. No frozen src/ code was
modified. The supplied LoRA and base-model revision match benchmark generation.

The committed [protocol](PREREG.md) describes the image-anchored, square-input
Cityscapes/LoRA adaptation and explicit Cityscapes SCLIP vocabulary. It is not
exact regeneration of the original batched images. Original PNGs are fixed;
masks return to their native dimensions. No pixel ground truth exists, so no
mIoU, optimal threshold or winning method is claimed. Geometric fusion remains
an exploratory comparator, not a registered promotion.

## Visual observations

- Several nearby foreground objects give useful coarse masks around 0.55.
  Raising the threshold to 0.70 can reduce spill while cutting valid object
  parts, including heads, feet, vehicle windows and bicycle structure.
- SCLIP-seeded BRACE helps the sky and some building/tree examples, but
  does not consistently improve the results. Geometric fusion can inherit
  BRACE's misplaced support as well as SCLIP's failures.
- The unchanged SCLIP area filter zeros the target seed in **6/12 confirmed
  pedestrian images**, **3/14 bicycle images** (parked or ridden), and
  **2/6 rider images**. Both SCLIP-seeded and geometric fusion are empty for
  these targets at all three thresholds. These are seed-suppression counts,
  not pixel accuracy metrics. Exact IDs are in
  [primary_class_seed_diagnostics.json](primary_class_seed_diagnostics.json).
- Rider masks often include the bicycle. Terrain is not cleanly recovered in
  the inspected park example. Normal BRACE also produces absent-class false
  positives after per-class min-max normalization.

See [VISUAL_REVIEW.md](VISUAL_REVIEW.md) for observations from every class,
including unsuccessful cases. This review covers fixed representatives and
additional diagnostic examples; it does not grade all 700 sheets.

## Validation and runtime

Selftest passed: first-image A3 parity error 6.56e-7; nonzero V3 checks had
max errors 1.79e-7 and 2.09e-7. Maximum A3 parity error across all 50 was
1.31e-6 (tolerance 1e-5). Constant-mask CRF fast paths were checked against the
unchanged R2 implementation for all-background and all-foreground masks.
Three GPU shards averaged 34.6 seconds per image for capture/propagation,
with peak allocated GPU memory 7.08 GiB. Up to six CPU workers rendered masks.
The original UNet-only adapter's text-encoder warning is expected; no inference
tracebacks were found. Count checks passed for all images, classes, methods
and thresholds. Complete binary PNG and archive integrity checks are recorded
in `validation.json` and the local bundle checksum.

## Local artifacts

- Browser: `outputs/segmentation/cityscapes_50/index.html`.
- Fixed class overview: `outputs/segmentation/cityscapes_50/class_overview_t0.55.jpg`.
- Bundle: `outputs/segmentation/cityscapes_50/cityscapes_50_brace_comparison.zip`.
- Full seeds, propagated maps, masks, image-level records and logs stay local
  under the same ignored output directory. The bundle includes source images.

Code and compact reports are committed locally; no push was requested.
