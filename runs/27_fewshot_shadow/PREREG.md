# Few-shot supervised shadow attention: direct vs hierarchy supervision, CA vs CA + SA (pre-registration)

Owner request (2026-11-18): replace runs/26 with a very small few-shot version. runs/26 was cancelled by the owner
before its test capture finished (382/507 COCO-Cap, ~290/490 VOC-Sim captured; never scored with DenseCRF; no
registered verdict exists; interim raw looks were reported in chat only). Written and committed before any training.

**Status: exploratory, supervised, few-shot** (human masks of 60 VOC-Sim images). BRACE stays frozen; nothing here is
promoted. The generator is untouched: low-rank *shadow copies* of the attention (runs/26 code, `scripts/jc_shadow.py`)
only produce the maps BRACE reads; generated images equal frozen SD.

## Data

- **Training:** 3 VOC-Sim images per class (20 classes → 60), the lowest crc32(image_id) per class.
- **Test:** 3 COCO-Cap images per class (20 classes → 60), the lowest crc32(image_id) per class. No COCO-Cap image is
  used for training or any choice. Written to `runs/27_fewshot_shadow/split.json`.

## Arms (shadow copies; q and k low-rank residuals, rank 4, B = 0 at init, same CA init in B/C/D)

| Arm | Trained | Loss | Hierarchy uses |
|---|---|---|---|
| A | nothing (frozen BRACE) | — | original SA |
| B | CA copies (11 A3 modules) | BCE + Dice on the min-max shadow seed (direct) | original SA |
| C | CA copies | BCE + Dice on the min-max shadow BRACE (V3) map, through the frozen hierarchy | original SA |
| D | CA copies + SA copies at 16, 32, 64 | as C | shadow SA |

Target = ground truth area-pooled to 64×64. AdamW, **lr 1e-3** (raised from runs/26's 3e-4 because of the much
shorter schedule; fixed before training), weight decay 0, gradient clip 1.0, batch 1, **60 steps = one pass over the
60 training images**, order shuffled with seed 0 (same order for all arms). No validation, no checkpoint selection:
the step-60 adapters are used.

## Evaluation

One live generation per image; A–D all from that generation (paired). Both the 60 training images and the 60 test
images are captured. Maps per arm: seed and BRACE (V3). Scored raw and + DenseCRF R2 at thresholds 0.30–0.70
(step 0.025). Two thresholds per arm on the test set: **fixed 0.55**, and **train-sel.** = that arm's best threshold
on its 60 training images (few-shot calibration; uses only training labels), applied unchanged to COCO-Cap test.
Training-image numbers are in-sample and reported as such. Class-macro mIoU; paired image-bootstrap 95% intervals (2000).

## Comparisons (COCO-Cap test, BRACE map + DenseCRF; 0.55 primary, train-sel. secondary)

| ID | Comparison | Question |
|---|---|---|
| F1 | D − C | do SA copies add to hierarchy-supervised CA? |
| F2 | C − B | is supervising CA through the hierarchy better than supervising the seed directly? |
| F3 | B − A | |
| F4 | C − A | |
| F5 | D − A | |

With 60 test images, intervals will be wide. A comparison is called **positive** only if its 95% interval at 0.55 is
above zero; everything else is reported as not shown. Results do not change the frozen method.

## Files

`scripts/jc_fewshot.py` (`split` → `train` per arm, GPU → `capture --set train|test`, GPU → `score`, CPU →
`report`). Checkpoints and maps under `<BRACE_CACHE_DIR>/jc_fewshot/`; report `runs/27_fewshot_shadow/result.md`.
Smoke (tag `_smoke`: 2 training images, 2 steps, 1 train + 1 test image captured) is kept.
