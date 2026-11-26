# Run 31: query-profile gate diagnostics and a multi-negative-prototype gate (pre-registration)

This freezes `PLAN.md` (the draft sent for external review) with the reviewer's five changes (Section 8). It was
committed before any capture or scoring of this run. BRACE stays frozen. **Everything here is exploratory**:
COCO-Cap was used in runs 28–30, and VOC-Sim for the gate's development check (run 29) and class-adaptive choice
(run 30). The dev/test split only protects against selection *within this run*.

## 1. Split (`runs/31_multineg_gate/split.json`, written by `jc_mneg.py split`)

- Within each (benchmark, class), order images by `crc32(image_id)`.
  - Even rank → **dev**; odd rank → **test**.
- Within dev, the same order makes the cross-fitting folds: rank // 2 even → **dA**, odd → **dB**.
- Ground truth is used only:
  - to score;
  - to define diagnostic regions;
  - to choose thresholds and per-class strengths **on dev**.

  It never builds a profile or chooses K.

## 2. Part A: CPU diagnostics on run 30 captures (all images)

Regions at 64 px:
- object = area-pooled GT > 0.5;
- near background = within 6 px (Chebyshev) of the object;
- far background = the rest.

### A1: distributions of the module-mean NC and NO gate maps saved by run 30

- Quantiles 5 / 25 / 50 / 75 / 95 per region type.
- AUROC object vs far and object vs near: per image (Mann–Whitney, averaged) and pooled.
- Object retention (g > t) and far rejection (g ≤ t) for t ∈ {0.3, 0.5, 0.7}.
- Per class: median per-image AUROC, and retention / rejection at t = 0.5.

### A2: seed-level sharpening

- s_γ = minmax(seed_A ⊙ ĝ_NC^γ), γ ∈ {0.5, 1, 2, 4}.
- Raw seed mIoU, with the VOC-Sim OOF threshold and the VOC-Sim-selected threshold on COCO-Cap.
- Far-background share of seed mass.
- Labelled an approximation: there is no self-attention to rerun the hierarchy. The ceiling after propagation is
  run 24's oracle (+0.05 / +0.06).

## 3. Part B: GPU dump on dev images

`jc_mneg.py dump`. One live SD1.5 generation per dev image (frozen runtime, runs/20 corrected prompts, safety
checker off). Saved per image:

- **q̂ per pixel** (call-averaged, head-merged, L2-normalised; `jc_trio.qbar`) for each of BRACE's 11 CA modules,
  fp16;
- **t̄_c** for all 20 classes (class-word keys, as in run 29), fp16;
- runtime seed, GT pooled to 64 px, target class, caption-mention flags, and a seed-fidelity check (stop above
  1e-2).

## 4. Part C: CPU fits and diagnostics

### C0: profiles

Two **regimes**, each with three fits (all dev; dA; dB):

| Regime | Dev images used |
|---|---|
| **X (mixed)** | VOC-Sim dev and COCO-Cap dev |
| **V (VOC-only)** | VOC-Sim dev only |

Per class c and module m:
- **Object profile m_o:** the mean q̂ at the top 5% of t̄_c, on images of c.
- **Absent-class negative pool N:** q̂ at the top 1% of t̄_c, on images whose target ≠ c and whose caption does not
  mention c.
- **Generic pool G (per module, class-independent):** a fixed 2% random sample of all pixels of every image in the
  fit, with seed crc32(image_id).

**Clustering:**
- spherical k-means; k-means++ from seed 0, 10 restarts, 100 iterations, lowest inertia kept;
- centroid = plain mean of its member unit vectors, not renormalised, so that K = 1 gives exactly the single mean
  profile;
- a cluster with fewer than max(20, 2% of the pool) members is dropped and its members reassigned;
- a cell with fewer than 20·K vectors falls back to K = 1.

**Fitted profiles:**
- N for K ∈ {1, 2, 4, 8};
- G for K = 4;
- the all-pixel mean for NO (exact, over all pixels).

For each fit: silhouette (cosine, on at most 2,000 sampled points), inertia drop relative to K = 1, cluster sizes,
and the source classes behind each cluster.

### C1–C4: diagnostics

All on the X regime, dev images scored with cross-fitted profiles (dA with the dB fit, dB with the dA fit). Gates
are computed per module from q̂. Regions are pooled to each module's grid; the near radius is max(1, round(6·r/64)).

- **C1, are the remaining false positives clustered?**
  - **High-gated false positives (HGFP):** background pixels with NC gate > 0.5, per (class, module), at most 5,000
    sampled.
  - K-means for K ∈ {2, 4, 8}: silhouette and inertia drop.
  - **Null:** an equal-size sample of high-gated object pixels.
- **C2, are they reproducible?**
  - Split-half matched-centroid cosine for HGFP clusters at K = 4 (images split by hash; exhaustive matching over
    permutations).
  - Adjusted Rand index between the HGFP cluster assignments of pairs of 64-px modules, on pixels that are HGFP in
    both modules.
  - For the negative pools: the matched-centroid cosine between the dA and dB fits at K = 4.
- **C3, does one negative mean blend several modes?**
  - Negative-pool silhouette, sizes, cos(single mean, centroids), and source make-up per cluster.
  - Among HGFP: the fraction pushed from g > 0.5 to g < 0.5 by MK4, MG4 and MS4, relative to NC (background
    removed).
  - Among object pixels with NC g ≥ 0.5: the fraction pushed below 0.5 (object wrongly removed).
- **C1 vs C0, do the gate's prototypes cover the actual errors?**
  - For each HGFP centroid (K = 4): its maximum cosine to the C0 absent-class centroids (MK4) and to the generic
    centroids (MG4).
  - The fraction of HGFP pixels whose nearest MK4 centroid is closer than m_o.
- **C4, classes harmed by the gate.**
  - tv, train and potted plant vs person, dining table and aeroplane.
  - Object gate quantiles; the fraction of object pixels with g < 0.5; mean cos(object q̂, n) vs cos(object q̂,
    m_o).
  - The fraction of object pixels whose nearest MK4 centroid is closer than m_o, and the top source classes of
    that centroid.
- **Caveat:** "absent" uses the caption, so an unmentioned class can still be visible. This is likely for person
  and car.

**Structure present (go/no-go, fixed now):**
1. the median negative-pool silhouette at K = 4 is ≥ 0.10;
2. the median dA-vs-dB matched-centroid cosine at K = 4 is ≥ 0.90.

**Part D waits for the owner after the diagnostic report.** If the rule fails, the MK and MS arms are reported as
descriptive with "no structure".

## 5. Gates (each multiplies the class word's attention at its module in every call, before the layer weighting; gain 4)

**Single profile:** g_single(q̂; m_o, n) = σ(4·(w·q̂ − (p₊+p₋)/2)/(p₊−p₋)), with:
- w = (m_o − n)/|m_o − n|;
- p₊ = w·m_o, p₋ = w·n.

**Multi-profile:** with centroids c_1 … c_K, compute g_k = g_single(q̂; m_o, c_k) for each k.

| Arm | Definition |
|---|---|
| NC | g_single with the absent-class mean |
| NO | g_single with the all-pixel mean |
| NH | NC^0.5 |
| MK2, **MK4**, MK8 | g_{k*} with absent-class centroids, k* = argmax_k cos(q̂, c_k/\|c_k\|) (hard nearest) |
| MG4 | as MK4 with the generic centroids (reviewer's control) |
| MS4 | Σ_k α_k g_k with α = softmax(10·cos(q̂, c_k/\|c_k\|)), absent-class K = 4 (soft ablation; τ = 10 fixed, not tuned) |

**Primary K = 4, fixed**; K = 2 and 8 are ablations. Silhouette is a diagnostic only. **Smoke assertions:** K = 1
reproduces NC (max |Δ| < 1e-5), and the recomputed BRACE seed matches the runtime seed.

## 6. Part D: evaluation capture (all 1,197 images, after the owner's go)

**Arms:** A, plus {NC, NO, NH, MK2, MK4, MK8, MG4, MS4} under each regime (X and V): 17 seed arms. Maps: seed,
fine-only (V0), hierarchy (V3).

**Profiles per image:**
- test images use the regime's all-dev fit;
- dev images use the regime's other-fold fit;
- in regime V, COCO-Cap dev images use the V all-dev fit (they are never in V fits).

**Scoring:**
- raw for every map;
- DenseCRF R2 for the hierarchy maps;
- thresholds 0.30–0.70, step 0.025;
- `ProcessPoolExecutor`, restarted if a worker dies.

**Threshold and class-strength selection, per regime, on dev only:**
- **X:** the threshold maximising the mean of VOC-Sim-dev and COCO-Cap-dev mIoU.
- **V:** VOC-Sim-dev mIoU only.
- **CAg:** per class, the arm among {A, NH, NC} (in that tie order) with the highest dev IoU for that class (X: both
  benchmarks' dev images of the class pooled; V: VOC-Sim dev), each at its regime threshold.

**Comparisons** (hierarchy + DenseCRF, test halves of both benchmarks, regime thresholds; paired image bootstrap,
2000), each regime **reported separately**:

| ID | Comparison |
|---|---|
| T1 | MK4 − NC |
| T2 | MK4 − A |
| T3 | CAg − A |
| T4 | MK4 − MG4 (is modelling *spurious* attention what matters?) |

Descriptive: MS4 − MK4, MK2/MK8 − NC, NO − NC, NH − NC, every arm − A for the seed and fine-only maps (raw), fixed
0.55, best threshold on test, per class.

**Reading rule:**
- **"Promising"** = the interval is above zero on the COCO-Cap test half and the lower bound is ≥ −0.005 on the
  VOC-Sim test half.
- A claim that the gain comes from **spurious-attention modes** additionally needs T4's interval above zero on the
  COCO-Cap test half.
- Regime V is the transfer setting (VOC-Sim → COCO-Cap). Regime X is mixed-domain development and is not presented
  as transfer.

## 7. Files

`scripts/jc_mneg.py`, with subcommands `split`, `diag30`, `dump`, `fit`, `diag`, `capture`, `score`, `report`.
Reports go to `runs/31_multineg_gate/{diag30.md, diag.md, result.md}`; data to `<BRACE_CACHE_DIR>/jc_mneg/`. Smoke
outputs are tagged `_smoke` and kept.

## 8. Changes from `PLAN.md` after review

1. Added MG4, the generic multi-prototype control, and T4 (MK4 − MG4).
2. Added MS4, the soft-assignment ablation, at K = 4.
3. Primary K fixed at 4 instead of chosen by silhouette; K = 2 and 8 are ablations.
4. Added the C1-vs-C0 coverage comparison.
5. Regimes V (VOC-only development → transfer) and X (mixed development) are reported separately.
