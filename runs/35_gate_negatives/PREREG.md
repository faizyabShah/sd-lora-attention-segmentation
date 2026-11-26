# Run 35: better negative evidence in the absent-class gate (pre-registration)

Committed before any fit or scoring of this run. BRACE stays frozen. **Exploratory.** Builds on run 31
(`runs/31_multineg_gate/`) and run 34 (`runs/34_gate_discrim/`). It covers N4, N1, N2, G and D of the plan agreed
on 2026-11-22.

This is **stage 1 only**: CPU, on run 31's dev dumps. A full-pipeline capture is a later, separately
pre-registered run, and covers only candidates that pass here.

## 1. Setting (identical to run 34 stage 1)

- **Data:** run 31 Part B dev dumps (`<CACHE>/jc_mneg/dump_{voc,coco}/`): 253 VOC-Sim and 356 COCO-Cap dev
  images. Each holds per-pixel call-averaged CA queries, t̄ for all 20 class words, the A3 seed and the 64 × 64
  object mask.
- **Split and folds:** run 31's `split.json`.
- **Regimes:**
  - V: profiles from VOC-Sim dev only;
  - X: profiles from VOC-Sim + COCO-Cap dev.
- **Held out:** every scored image is gated with profiles that never saw it (the other dev fold; regime V on
  COCO-Cap uses all VOC-Sim dev).
- **Pools, as run 31:**
  - object pool: q̂ at the top 5% of t̄_c on images of class c;
  - absent pool: q̂ at the top 1% of t̄_c on images whose prompt doesn't mention c;
  - `jc_mneg.cluster`, K = 4.
- **Calibration, as run 31:** g = σ(4·(z − (z_o + z_n)/2)/(z_o − z_n)), where z is the score, z_o its value at the
  object profile and z_n its value at the selected negative.

## 2. Gates

| Gate | Plan item | Object side | Absent side | Score |
|---|---|---|---|---|
| NC | — | mean | 1 mean | projection on normalise(o − n) |
| **MK4** | baseline | mean | 4 centroids, nearest | projection on normalise(o − n_k) |
| **SN** | N1 self-negative | mean | 4 centroids **+ one self-negative from the scored image** | as MK4, nearest of 5 |
| **IB** | N2 cleaner object pool | **image-balanced, relative-threshold mean** | 4 centroids | as MK4 |
| **GM** | G margin score | mean | 4 centroids | **cos(q̂, o) − max_k cos(q̂, n_k)** |
| **PD** | D persistent negatives | mean | 4 centroids, **each supported by ≥ 5 effective images** | as MK4 |

### Definitions (fixed now)

**SN (self-negative).**
- For the scored image, class c and module m: s = the plain mean of q̂ over the image's **bottom 30% of t̄_c**.
- s is added as a fifth absent centroid for that image only; the nearest-negative rule is then as MK4.
- It uses only the image's own class-word attention: label-free, no fitted parameter.

**IB (image-balanced object pool).**
- Per object image, the pool is the top 5% of t̄_c **that also have t̄_c ≥ 0.5 · max t̄_c in that image** (at least
  1 pixel).
- The object profile is the **mean over images of per-image means**.
- Reason: on small objects, the fixed top 5% includes background pixels.
- The absent side is unchanged.

**GM (margin score).**
- z = cos(q̂, ô) − max_k cos(q̂, n̂_k), with ô and n̂_k normalised.
- z_o = 1 − max_j cos(ô, n̂_j).
- z_n = cos(n̂_k, ô) − 1, where k is the pixel's nearest negative.
- This is a different score from MK4's projection, not a reshaping of g.

**PD (persistent negatives).**
- After `jc_mneg.cluster(X, 4)`, compute each cluster's effective number of images n_eff = 1 / Σ_i p_i², where p_i
  is the share of the cluster's pixels from image i.
- Clusters with n_eff < 5 are dropped. All members are re-assigned to the nearest kept centroid and the centroids
  recomputed, as `jc_mneg.cluster` does for small clusters.
- If no cluster is kept, the K = 1 mean is used.

## 3. N4 diagnostics (descriptive, same job)

Computed from profiles fitted on all dev images of the regime (fits), and from the held-out gates (pixels).

1. **Centroid support (MK4 absent centroids):**
   - pixels;
   - n_eff;
   - largest single-image share;

   summarised per module and overall: median, and the fraction with n_eff < 5.
2. **What each absent cluster contains.** Per cluster, reported against the same share in the cluster's whole absent
   pool:
   - **image-label share:** the most common true label of the member images;
   - **attention share:** the most common class, excluding c, with the highest t̄ at member pixels.

   A cluster share well above its pool's share means the clusters separate confusions.
3. **Module agreement (MK4, held out):**
   - per-module AUROC (object vs far background);
   - mean pairwise correlation of g between modules at a common 64 × 64 grid (nearest upsampling);
   - the fraction of object pixels where some module has g < 0.5 while the median module has g ≥ 0.5.

   Consensus gates are **not** scored here. They will be pre-registered after these numbers.
4. **Object size:** MK4 and IB mass-weighted object retention at γ = 1, by object-area tercile, and the Spearman ρ
   between area and retention.

**Reading rule for PD (fixed now).** PD's stage-1 result is read only if diagnostic 1 shows a problem to fix: in at
least one of the two pass-rule fits (regime V and regime X, all dev), ≥ 10% of MK4 absent centroids have
n_eff < 5. Otherwise PD is reported as descriptive.

## 4. Stage 1 measurement and pass rule (identical to run 34)

- **Regions:** object, near background (≤ 6 px at 64), far background, at each module's grid.
- **Retention:** Σ w·g^γ / Σ w with γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4}.
  - Weights: pixel or mass (A3 seed; primary).
  - Averaged over modules → images → classes.
- **Matched operating range:**
  - far suppression from MK4's γ = 0.5 to γ = 2, at 5 points;
  - ΔR = mean object-retention difference vs MK4;
  - ΔS = the converse.
- **Bootstrap:** 2000 image samples.
- **Pass:** mass-weighted far background, ΔR ≥ +0.01 **and** CI above 0, on **both** VOC-Sim dev (V) and
  COCO-Cap dev (X). PD only under the reading rule above.
- **Also reported:** near background, pixel weights, regime X on VOC-Sim and regime V on COCO-Cap (descriptive),
  AUROC, per class (VOC-Sim / V and COCO-Cap / X, mass-weighted object retention at γ = 1), and **all results with
  person excluded**.

**Stopping rule (fixed now).** If no candidate here passes and the later consensus / confusion run (H, I) also has
no pass, the gate is not changed further and MK4 is written up as it is.

## 5. Checks (smoke, before the full run)

- **PD with n_min = 0** equals MK4's centroids: max |Δ| < 1e-6.
- **IB with relative threshold 0** equals MK4's object mean: max |Δ| < 1e-5. Every image has the same pixel count,
  so the mean of means equals the pixel mean.
- **SN identity:** with the self-negative replaced by a copy of an existing centroid, SN equals MK4: max |Δ| < 1e-5.
- Every gate is finite and in [0, 1].

## 6. Files and cost

- **Script:** `scripts/jc_gate35.py`.
- **Reports:**
  - `runs/35_gate_negatives/stage1.md` and `stage1.json`;
  - `runs/35_gate_negatives/diag.md` and `diag.json`.
- **Data:** `<CACHE>/jc_gate35/`.
- **Cost:** fit about 20 min; stage 1 + diagnostics about 30–40 min; one CPU job each.
- Smoke outputs are tagged `_smoke` and kept.
