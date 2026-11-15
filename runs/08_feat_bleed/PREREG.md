# Pre-registration: does frozen UNet feature content separate BRACE's bleed from the true object, beyond RGB and BRACE's own confidence? (diagnostic)

Written 2026-11-15, before any statistic below has been computed. Code: `scripts/feat_bleed.py`. CPU only, on stored data.

- This is the **last diagnostic** in this family. If it fails, the feature-content direction is closed. There is no sweep over layers, steps, prototypes, clustering or classifier settings.
- BRACE stays frozen.
- Ground truth is used only for scoring. It never selects cores, trains the inference-time classifiers or sets the correction rule.

## Data

- **Images.** The stored rich capture of the class-stratified screening subsets (`CACHE/ana_dump`, 150 VOC-Sim + 150 COCO-Cap).
- **Each file holds:** the image, the ground truth, the V3 map, and conditional UNet features.
- **Primary feature map (fixed a priori):** `feat/up2/s15`, 640 channels at 64 × 64, denoising step 15.
  - It is native at BRACE's 64-cell resolution, and step 15 is the middle of BRACE's propagation window (steps 10–19).
  - `runs/analysis_cluster/part2_5.md` already reported probe results trained on ground truth for these features. That is disclosed here; no new layer or step choice is made from it.

## Cells, mask and cores

Everything is on the 64 × 64 grid (8 px cells).

- **BRACE mask M (frozen):** min–max(V3), bilinear to 512, τ 0.55, DenseCRF R2. A cell is inside M if its pooled M is ≥ 0.5.
- **Ground-truth cell labels (scoring only):** object if pooled ground truth ≥ 0.75, background if ≤ 0.25. Mixed cells are excluded from all statistics.
- **Cores (label-free):**
  - **F:** cells with min–max V3 ≥ 0.9. If there are none, the top 5% of cells.
  - **B:** cells with min–max V3 ≤ 0.1 that lie outside M.
  - Images with fewer than 5 cells in F or B are excluded and counted.
- **Task (Q):** among cells inside M and not in F, separate **bleed** (ground-truth background, label 1) from **correct** (ground-truth object, label 0).

## Per-cell scores

**Baseline (RGB, BRACE confidence, spatial cues):**
1. Lab distance from the cell's mean colour to the mean colour of F, and to B.
2. Intervening contour: the maximum Sobel gradient (Lab, σ 1) along the segment from the cell to its nearest F cell.
3. Distance to the nearest F cell, in cells.
4. BRACE confidence: min–max V3 at the cell.
5. Depth inside M: distance to M's boundary, in cells.
6. Mask area fraction of M (per image).
7. **Matched RGB classifier s_rgb.** A within-image logistic regression trained on F (1) vs B (0) cells, using each cell's mean and standard deviation of Lab (6 features). It is applied to the cell and enters as its logit.

**Feature signals (test):**
- **s_proto:** cos(f_i, mean f_F) − cos(f_i, mean f_B). Features are L2-normalised per cell; prototypes are the means of the normalised features.
- **s_feat:** an otherwise identical within-image logistic regression trained on F vs B cells. Its inputs are the first 32 per-image principal components of the standardised 640-channel features. It enters as a logit.

**Classifier settings (fixed):** L2 penalty λ = 1 on standardised inputs, Newton's method, balanced class weights. These apply to both s_rgb and s_feat.

## Gate 1: information beyond baseline (decisive set COCO-Cap)

- **Model:** out-of-fold logistic regression (folds crc32(image_id) % 5, L2, standardised features) over the cells of Q, pooled across images.
- **Comparison:** baseline vs baseline + s_proto + s_feat.
- **Pass:** ΔAUROC **≥ +0.05**, with image-bootstrap 95% CI lower bound **> 0** (2000 resamples).

## Gate 2: usable without labels

**Correction rule (fixed):** remove from M every 8 × 8 cell inside M, not in F, whose s_feat probability is below 0.5.

**Controls**, each removing the **same number of cells per image** as the feature rule:
- **RGB control:** the same rule with s_rgb, taking the lowest-probability cells.
- **Confidence control:** the cells with the lowest min–max V3 inside M, not in F.

**Pass requires all of:**
- **COCO-Cap:** mean per-image IoU gain over BRACE + R2 **> 0**, with image-bootstrap 95% CI lower bound **> 0**.
- **COCO-Cap:** mean gain **greater than** both controls' mean gains.
- **VOC-Sim guard:** mean gain **≥ −0.005**.

## Verdict

- **Continue** only if gates 1 and 2 both pass. Even then, the next step is a new pre-registered screen on images not used here, not a BRACE change.
- **Otherwise** the direction is closed.

## Reported, not deciding

- **Single-feature AUROC** for every baseline and test feature, on Q.
- **Gate 1 with s_proto alone and with s_feat alone.**
- **TRACE-style row divergence.** At 32 px, block 1 (steps 10–19), conditional: the symmetric KL between the cell's self-attention row and the mean row of F. The cell is mapped to its 32-grid parent. Reported as a single-feature AUROC and as its gain when added to the baseline.
- **Three-step consistency.** The standard deviation over steps 5, 15 and 25 of s_proto computed from `feat/up2/s{5,15,25}`. Reported the same way.
- **VOC-Sim gate 1.**
- **Bleed share and per-image counts:** the bleed share of Q, the number of included images, and the cells removed per image.
