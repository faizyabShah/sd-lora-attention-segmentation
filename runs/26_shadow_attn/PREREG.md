# Supervised shadow attention for BRACE: CA-only vs CA + SA (pre-registration)

Owner request (2026-11-18): train LoRA-adapted *copies* of SD1.5's cross-attention (CA) and self-attention (SA) that
are used only to read out segmentation maps; the generator's own attention is untouched, so generated images are
identical to frozen SD. Two arms (CA copies only; CA + SA copies), compared with frozen BRACE on held-out images.
Owner asked for 64×64 SA copies as well as 16 and 32. Written and committed before any training or scoring.

**Status: exploratory, supervised (uses human masks of 200 COCO-Cap images).** It is not a change to BRACE, which stays
frozen and training-free; nothing here is promoted to the paper's method. COCO-Cap loses its "untuned transfer" role
for this experiment only; VOC-Sim (never trained on, never used for selection) is the cross-dataset check.

## Shadow attention

Frozen runtime and corrected prompts (runs/20, safety checker off), one live generation per image. During generation the
original attention runs unchanged; the queries/keys it computes are recorded (detached) and a shadow branch recomputes
attention from them with low-rank adapters:

    q' = q + (q A_q) B_q,   k' = k + (k A_k) B_k        (heads merged, rank 8, B = 0 at init)

SD1.5's `to_q` is square, so a low-rank update in q-space equals a LoRA on `to_q`. With B = 0 the shadow branch
reproduces frozen BRACE exactly (checked in the smoke test and on every evaluation image).

- **CA copies:** the 11 selected A3 cross-attention modules (q and k adapters each, conditional branch, all 31 calls).
  Shadow seed = A3 construction unchanged: target-token probability, head mean, SA-guided (GOCA-style) layer weights
  **computed from the original SA** (not trained), bilinear to 64, summed over calls, min-max.
- **SA copies:** the three hierarchy SA modules at 16, 32 and 64 (q and k adapters each, the 11 hierarchy calls).
- Shadow BRACE map = the V3 path of `sg_combined_v2.aggregate` reimplemented differentiably (16 SA → 32 SA →
  one-sided cap at 16→32 → 64 SA, mean over 11 calls), checked against `aggregate` at B = 0.

## Arms

| Arm | Trained | Hierarchy uses |
|---|---|---|
| A0 | nothing (frozen BRACE) | original SA |
| B  | CA copies | original SA |
| C  | CA copies + SA copies (16, 32, 64) | shadow SA |

B and C share everything else: rank 8, AdamW lr 3e-4, weight decay 0, gradient clip 1.0, batch 1 image, 800 steps,
seed 0, image order shuffled with the same seed, same loss.

**Loss (both arms):** t = ground truth area-pooled to 64×64. p_V = min-max shadow BRACE map, p_S = min-max shadow
seed (clamped to [1e-4, 1 − 1e-4]). L = [BCE(p_V, t) + Dice(p_V, t)] + 0.5·[BCE(p_S, t) + Dice(p_S, t)].
DenseCRF and the threshold are not in training.

## Data split (fixed before training)

COCO-Cap (707 images): per class, images sorted by crc32(image_id); the lowest round(200·n_c/707) go to the training
pool (largest-remainder rounding to exactly 200). Within the pool, per class sorted by crc32(image_id + "val"), the
lowest round(40·n_c/200) form the validation split (exactly 40); the other 160 train. The remaining 507 COCO-Cap
images are the **COCO-Cap test** set. All 490 VOC-Sim images are the **VOC-Sim transfer** set. Written to
`runs/26_shadow_attn/split.json`.

## Checkpoint selection (validation only)

Checkpoints at 200, 400, 600, 800 steps per arm. Selection metric: class-macro mIoU of the raw shadow BRACE map at
threshold 0.55 on the 40 validation images; highest wins, ties to the earlier step. No test image is used.

## Evaluation

One live generation per test image; A0, B and C all from that generation (same capture, paired). Maps per arm: seed
and BRACE (V3). Scored raw and + DenseCRF R2 at thresholds 0.30–0.70 (step 0.025), as runs/20/25.
Two numbers per cell: **VOC-sel.** (VOC-Sim 5-fold OOF by crc32(id) mod 5; the full-VOC-Sim best threshold applied
unchanged on COCO-Cap test) and **fixed 0.55**. Class-macro mIoU; paired image-bootstrap 95% intervals (2000).

## Registered comparisons (BRACE map + DenseCRF; 0.55 primary)

| ID | Comparison |
|---|---|
| S1 | C − B (does training SA add to training CA?) |
| S2 | B − A0 (supervised CA copies vs frozen BRACE) |
| S3 | C − A0 |

**Holds** if the COCO-Cap test Δ at 0.55 is positive with its interval above zero **and** the VOC-Sim Δ at 0.55 is
not below −0.005. Otherwise it does not hold, and is reported as such. Seed-only rows and raw rows are descriptive.

## Files

`scripts/jc_shadow.py` (`split` → `train` per arm, GPU → `val`, GPU → `capture`, GPU sharded → `score`, CPU →
`report`). Checkpoints and maps under `<BRACE_CACHE_DIR>/jc_shadow/`; report `runs/26_shadow_attn/result.md`.
Smoke (tag `_smoke`: 2 training images, 4 steps, 2 validation images, 1 VOC + 1 COCO test image) is kept.
