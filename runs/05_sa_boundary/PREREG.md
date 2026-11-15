# Pre-registration: does SD self-attention carry boundary information that BRACE does not use? (diagnostic)

Written 2026-11-15, before any of these statistics has been computed. Code: `scripts/sa_boundary_diag.py`. CPU only, on stored data. This is a **diagnostic**: it decides whether a boundary mechanism is worth designing. It is not a method test. Ground truth is used only to measure.

## Data

- **Images.** The rich capture of the class-stratified screening subsets (`CACHE/ana_dump/{voc,coco}`, 150 + 150 images, `scripts/ana_capture.py`).
- **Self-attention.** Conditional, head-mean, row-stochastic, at native resolution: `sa_full/<res>_b<block>_c` for res 16 and 32.
  - **Primary:** 32 px, block 1 (steps 10–19, the window BRACE's propagation uses).
  - **Secondary:** 16 px, and blocks 0 and 2.
  - Nothing is upsampled.
- **Symmetric affinity.** w_ij = (A_ij + A_ji) / 2. The feature used is log(w + 1e−8).
- **Cell labels.** From ground truth area-pooled to the grid: object if ≥ 0.75, background if ≤ 0.25. Mixed cells are excluded from all statistics.
- **BRACE mask.** Recomputed from the stored V3 map: min–max, upsampled to 512, τ 0.55, frozen DenseCRF R2, then area-pooled to the grid.
- **BRACE core** (label-free): cells where the area-pooled min–max V3 is ≥ 0.9. If no cell qualifies, the top 5% of cells.

## RGB baseline (same edges and cells)

- **Colour:** Euclidean distance between the mean Lab colours of the two cells.
- **Intervening contour:** the maximum Sobel gradient magnitude (Lab, σ 1 at 512 px) sampled at 16 points along the segment between the two cell centres.

## Edges

Every pair of cells whose centre distance, rounded, is 1, 2 or 3 cells. "Crossing" means object–background; "within" means object–object or background–background.

## Q1: boundary discrimination (context only)

- **Per image and distance band:** the AUROC of −log w for crossing vs within edges. Bands without both kinds of edge are skipped.
- **Per image:** the mean over bands, then the mean over images, with an image-bootstrap 95% CI.
- **Reported the same way:** colour distance and intervening contour.

## Q2: value beyond RGB (context only)

- **Logistic regression** (L2, standardised features), out-of-fold by image (5 folds, crc32(image_id) % 5). At most 3000 edges per image per band, sampled with a fixed seed.
- **Two models:**
  - RGB = colour, contour, distance-band indicators;
  - RGB + SA = the same plus log w.
- **Reported:** pooled out-of-fold AUROC of each model, and ΔAUROC with an image-bootstrap 95% CI (1000 resamples).

## Q3: BRACE bleed (decisive)

- **Cells.** Inside BRACE's mask (pooled mask ≥ 0.5):
  - bleed = background cells (ground truth ≤ 0.25);
  - correct = object cells (ground truth ≥ 0.75).
- **Task.** Separate bleed (1) from correct (0).
- **Reference set R.** Two versions:
  - (a) the BRACE core, which is label-free and **decisive**;
  - (b) ground-truth object cells, a diagnostic only.
- **Features for cell i:**
  - **RGB baseline:**
    - distance from i to the nearest cell of R, in cells;
    - colour distance between i and the mean colour of R;
    - intervening contour from i to the nearest cell of R.
  - **+ SA:** the log of the mean w_ij over j in R, excluding i itself.
- **Model and output.** The same logistic regression and folds as Q2. Report the pooled out-of-fold AUROC of RGB and of RGB + SA, and ΔAUROC with an image-bootstrap 95% CI.

## Decision rule (fixed now)

Continue towards a boundary mechanism **only if**, for Q3(a) at the primary setting (32 px, block 1) on **COCO-Cap**:
- ΔAUROC (RGB + SA vs RGB) is **≥ +0.05**, **and**
- the lower bound of its 95% CI is **> 0**.

COCO-Cap is decisive because bleed is its main failure (97 of 142 failures).

**Otherwise** the direction is closed. No graph-cut or propagation variant is built or tuned.

VOC-Sim, 16 px, the other blocks, Q1, Q2 and Q3(b) are reported but do not change the decision.

## Amendment 1 (2026-11-15, after a 10-image code test, before the full run)

- **Q3(b) is degenerate as written.** With R = ground-truth object cells, every correct cell is in R (distance 0), and every bleed cell is not (distance > 0). The RGB baseline is therefore perfect by construction (AUROC 1.000 in the code test). Q3(b) now reports only the single-feature AUROC of −SA connectivity to the ground-truth object, without any model.
- **Added for transparency:** single-feature AUROCs for each Q3(a) feature (distance, colour, contour, −SA), with no model. These are needed to interpret the out-of-fold baseline, which fell below chance on the 10-image test.
- **Unchanged:** Q3(a), its features, its model and the decision rule.
