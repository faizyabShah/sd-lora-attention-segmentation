# Run 32: component factorial on real PASCAL VOC photos, one noising pass (pre-registration)

Committed before any capture or scoring of this run. BRACE stays frozen; this is a component study in a new setting.
**Exploratory.** VOC 2012 val was used once before (2026-11-09, `scripts/sg_voc21.py`, SCLIP + SD propagation),
never to tune any component tested here.

## 1. Question

How much does each component contribute, alone and stacked, on real photos?

| Component | On | Off |
|---|---|---|
| **W**: self-attention-guided layer weights | A3 weights (GoCA-style agreement) | uniform weights over the same 11 layers |
| **H**: coarse-to-fine hierarchy, no cap | 16 → 32 → 64 raw hierarchy (`sg_combined_v2` V1) | fine-only (two 64-px passes, V0) |
| **G**: 4 absent-class profiles | hard-nearest gate with K = 4 absent-class centroids (run 31 MK4) | no gate |

All 2 × 2 × 2 = 8 cells come from **one capture**, because the layer weights, the gate and the propagation are
all applied after the UNet call to recorded attention.

## 2. Setting

- **Images:** PASCAL VOC 2012 segmentation splits. **Tuning:** `train` (1,464 photos). **Evaluation:** `val`
  (1,449 photos), used only once, at the end.
- **Task (class given, as in our benchmarks):** one item per (photo, class present in the ground truth). Truth =
  that class's pixels; void (255) pixels are excluded from scoring.
- **One pass per item:**
  - photo resized to 512 × 512 (bicubic);
  - VAE encode (latent mean);
  - noise at t = 100 with a fixed noise seed 0;
  - one conditional SD1.5 UNet call with the prompt `A photograph of a {w}`.

  `{w}` is the VOC-Sim word for the class: `dining-table`, `potted-plant` and `tv` as in the VOC-Sim prompts.
  This is the same setting as JC #5 (`comparisons/backbones/single_pass.py`, SD1.5 layer mapping), with the layer
  map copied unchanged.
- **Maps:** 64 × 64, min-max normalised, bilinear to the photo size, thresholded.

## 3. Profiles (component G), two sources reported separately

| Source | Fit images (one pass each, same setting) | "Absent" means |
|---|---|---|
| **A: VOC train labels** | the train items | the class is not in the photo's ground truth (image-level labels used) |
| **B: VOC-Sim captions (label-free)** | the 490 stored VOC-Sim images, their runs/20 prompts | the caption does not mention the class |

Per class c and CA module m, exactly as run 31 (`scripts/jc_mneg.py`, functions reused):
- **object profile:** the mean q̂ at the top 5% of t̄_c on items of class c;
- **absent pool:** q̂ at the top 1% of t̄_c on items where c is absent;
- **centroids:** K = 4 by `jc_mneg.cluster`, with plain-mean centroids, small clusters dropped, and a K = 1
  fallback below 80 vectors. K = 1 (the single mean) is also fitted, for a descriptive row.
- **q̂** is the head-merged, L2-normalised CA query of that module. **t̄_c** is the head-mean attention to the
  class-word tokens of the prompt "c" alone.
- **Gate (MK4):**
  - each pixel is compared with its nearest centroid;
  - g = σ(4·(w·q̂ − (p₊+p₋)/2)/(p₊−p₋)), as in run 31;
  - g multiplies the target attention map of module m before the layer weighting;
  - a missing profile means g = 1.

**Cross-fitting on train:** source A's gate on a train item uses profiles fitted on the other train fold
(fold = crc32(image id) mod 2). On val, it uses all of train. Source B never sees VOC photos.

## 4. Thresholds and scoring

- **Grid:** 0.30–0.70, step 0.025.
- **Threshold choice:** for each map, the threshold that maximises class-macro mIoU on the train items (raw maps).
- **On val:**
  - raw mIoU at every grid value;
  - DenseCRF R2 (`apply_ovam_densecrf`) on the 12 factorial maps, at the train-selected threshold and at a fixed
    0.55.

  DenseCRF at every grid value would cost about 400 CPU-hours, so the threshold applied before DenseCRF is the one
  selected on raw train maps.
- **Metric:** class-macro mIoU over items (our benchmark metric, not the VOC21 dataset-level metric).
- **Intervals:** 95% from a 2000-sample bootstrap over photos. All items of a photo are resampled together.

## 5. Report and reading rule (fixed now)

For each profile source (A and B) and each rule (train-selected or 0.55; raw or + DenseCRF):

1. **Table:** the 8 cells, with a tick per component present.
2. **Effects of each component:**
   - **alone:** that component on and the others off, minus all off;
   - **in the full stack:** all on, minus all on without it;
   - **average effect:** the mean of the 4 paired differences that change only that component.
3. **Primary rule:** val, + DenseCRF, train-selected threshold.
   - **"Contributes"** = the average-effect interval is above zero.
   - **"Hurts"** = the interval is below zero.
   - Otherwise **"not resolved"**.

Descriptive: seed-only maps; single-profile (K = 1) gate rows for the weighted CA; the capped hierarchy (BRACE V3)
for the weighted CA without a gate (raw); per-class IoU.

W and H do not depend on the profile source, so their numbers are the same under A and B except for cells
involving G.

## 6. Checks (smoke, before the full run)

- The single-call fine-only, uncapped and capped hierarchy match `sg_combined_v2.aggregate` fed the same call 11
  times: max |Δ| < 1e-5 after min-max.
- The K = 1 hard-nearest gate equals the single-profile gate: max |Δ| < 1e-5.
- Every map is finite, and every item's prompt contains the class word.

## 7. Files

- **Script:** `scripts/jc_realvoc.py`, with subcommands `items`, `dump`, `fit`, `capture`, `score`, `select`,
  `report`.
- **Reports:** `runs/32_realvoc_factorial/result.md`.
- **Data:** `<BRACE_CACHE_DIR>/jc_realvoc/`.
- Smoke outputs are tagged `_smoke` and kept.
