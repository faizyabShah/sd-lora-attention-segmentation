# Runs 37 and 38: absent-class gate variants, with and without SA weights and hierarchy, on real VOC photos and on generated images (handoff for review, 2026-11-23)

**One-line answer.** On real VOC photos and on generated images alike, the two module-consensus gates (MK4-LR and
MK4-RW) beat the run 31 gate (MK4) by about +0.02 to +0.04 mIoU, and the self-negative gate (SN) is worse than MK4.
The two settings disagree on the hierarchy once LR or RW is on:
- **real photos:** the hierarchy **hurts** (−0.016);
- **generated images:** it is **neutral** (about 0).

**Status.**
- Both runs are **exploratory**. Each was pre-registered and committed before any scoring:
  - run 37: 7926e3ba2, amended in 19137fa3a before any scoring;
  - run 38: 5518d2dc8.
- The results are in commits 22a90f42f (run 37) and ee1dfe0f3 (run 38).
- Run 37 is the **third look at VOC val**: runs 32 and 33 also reported on it.
- BRACE stays frozen: A3 cross-attention seed → V3 capped 16→32→64 self-attention hierarchy → DenseCRF R2, τ 0.55.
- Inference is label-free; ground truth is used only for scoring and for fitting on train / dev.
- The full reports are in the appendix, unchanged.

---

## 0. Background

### 0.1 Base gate MK4 (run 31)

For class c and each of the 11 cross-attention (CA) modules m, MK4 compares each pixel's query q̂ with fitted
profiles.

| Term | Definition |
|---|---|
| q̂ | head-merged, L2-normalised CA query at the pixel |
| Object profile o | mean q̂ at the top 5% of t̄_c on images of c (t̄_c = attention to the class word) |
| 4 absent centroids | spherical k-means of q̂ at the top 1% of t̄_c on images where c is **absent** |
| Gate | g = σ(4·(z − (z_o + z_n)/2)/(z_o − z_n)), where n is the nearest absent centroid |

The gate multiplies module m's head-mean target attention before the seed is built.

### 0.2 Variants (run 36)

| Gate | Definition |
|---|---|
| **SN** (self-negative) | MK4's 4 centroids plus a 5th from the image itself: the mean q̂ over the image's bottom 30% of t̄_c. No labels. |
| **MK4-LR** (low-res consensus) | Each module's MK4 gate is upsampled to 64², averaged over the **six modules below 64×64** only, then area-resized back and applied at **every** module. |
| **MK4-RW** (reliability-weighted consensus) | The same, but every module gets weight w_m = clip(2·(AUROC_m − 0.5), 0, 1), where AUROC_m is how well that module's MK4 gate separates object from far background. |

Why the consensus gates: run 35 found that per-module MK4 AUROC is 0.82–0.99 at the low-resolution modules but only
0.46–0.68 at the five 64×64 modules. So gating with the noisy 64×64 gates mostly adds noise.

### 0.3 Earlier results

| Run | Setting | Finding |
|---|---|---|
| 31 | generated, regime X | BRACE 0.843 / 0.718 → MK4 0.855 / 0.762 (VOC-Sim / COCO-Cap) |
| 32–33 | real VOC val, t = 100 | SA weights about +0.02; profiles must be **fitted on real photos** (generated-image profiles give only +0.002) |
| 35 | generated dev, stage 1 | SN separated object from background far better than MK4 (ΔR +0.055 / +0.032) |
| 36 | generated, end to end, γ\* matched to MK4's far suppression | SN − MK4 **fails** (−0.019 / −0.030 in X). MK4-LR − MK4 improves: X +0.001 / +0.014 (marginal), V +0.010 / +0.032. MK4-RW improves in V only. |
| 36 (fine-only readout) | generated | The LR / RW gain is already in the seed, so it doesn't depend on the hierarchy |

Run 37 checks whether this carries over to real photos, with every tunable fitted on real train photos. Run 38 repeats
exactly the same factorial on the generated benchmarks, with everything fitted on dev.

---

## 1. Shared design (runs 37 and 38)

**Factors (2 × 2 × 5 = 20 cells per profile source or regime):**

| Factor | Off | On |
|---|---|---|
| W, SA-guided layer weights | uniform over the 11 CA modules | A3 weights (BRACE's seed) |
| H, hierarchy | fine-only (V0: two passes of 64×64 SA) | capped hierarchy V3 (BRACE) |
| G, gate | none | MK4, SN, MK4-LR, MK4-RW |

- The **"none / V3 / W on" cell is BRACE before or with DenseCRF.** It is the BRACE-equivalent baseline in every
  table.
- **Gate strength γ:** the gate is applied as g^γ, with γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}. γ is chosen per gate
  (MK4 included) × W × map × source / regime, as the γ with the best train / dev mIoU on raw maps. γ = 1 (untuned) is
  also reported.
- **Thresholds:** chosen per map on train / dev raw maps, grid 0.30–0.70. DenseCRF R2 is then applied at that
  threshold and at a fixed 0.55.
- **Primary contrasts** (fixed in advance; SA weights on, + DenseCRF, selected threshold and γ):

  | # | Contrast | Map |
  |---|---|---|
  | P1 | LR − MK4 | V3 |
  | P2 | LR − MK4 | V0 |
  | P3 | RW − MK4 | V3 |
  | P4 | RW − MK4 | V0 |
  | P5 | SN − MK4 | V3 |
  | P6 | SN − MK4 | V0 |

- **Metric:** class-macro mIoU. 95% CIs from a 2000-sample paired bootstrap. Reading: CI > 0 = helps, CI < 0 =
  hurts, otherwise not resolved.

### 1.1 Run 37: real PASCAL VOC 2012

- **Items:** 2,170 VOC train items and 2,148 VOC val items (1,449 photos). One SD1.5 pass on the photo at t = 100,
  with prompt `A photograph of a {w}`.
- **Profile sources:**
  - **C** (primary, label-free): 490 real COCO train2017 photos; "absent" means none of the 5 captions mentions c.
  - **A** (reference): VOC train labels.
- **Fitted on VOC train photos:** thresholds, γ, and RW module weights (per-module MK4 AUROC on VOC train masks,
  cross-fitted by fold).
- **No val image or mask is used for any fit.** The γ and RW fit was moved from generated to real photos **before
  scoring**, at the owner's request.
- **Bootstrap:** over photos.
- **"Holds":** helps on V3 or V0, and hurts on neither.

### 1.2 Run 38: generated VOC-Sim (490) and COCO-Cap (707)

- **Capture:** one new generation-time capture, which is BRACE's native capture. Run 31's split: dev folds and test
  halves.
- **Regimes:**

  | Regime | Profiles and fits from | COCO-Cap is |
  |---|---|---|
  | **V** | VOC-Sim dev only | untuned transfer |
  | **X** | VOC-Sim dev + COCO-Cap dev ("half of each") | in-domain |

- **Fitted on dev:** thresholds and γ. In regime X, the curve used is the mean of the two dev curves. Profiles and RW
  weights come from runs 35/36 (dev, held out).
- **Scored on the test halves.**
- **"Improves" (run 36 rule):** COCO-Cap CI lower bound > 0 **and** VOC-Sim CI lower bound ≥ −0.005.
- **"Holds":** improves on V3 or V0, and in neither map does it hurt on either benchmark.
- **Note:** DenseCRF is applied at the threshold chosen on raw dev maps (run 37's rule). Run 36 chose it on DenseCRF
  curves, so run 38's BRACE numbers differ slightly from run 36's.

---

## 2. Run 37 results: real VOC val

### 2.1 Primary contrasts (source C, SA weights on, + DenseCRF)

| # | Contrast | Map | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | MK4-LR − MK4 | V3 | +0.021 [+0.014, +0.028] | **helps** |
| P2 | MK4-LR − MK4 | V0 | +0.038 [+0.028, +0.048] | **helps** |
| P3 | MK4-RW − MK4 | V3 | +0.025 [+0.019, +0.032] | **helps** |
| P4 | MK4-RW − MK4 | V0 | +0.042 [+0.032, +0.051] | **helps** |
| P5 | SN − MK4 | V3 | −0.018 [−0.025, −0.012] | **hurts** |
| P6 | SN − MK4 | V0 | −0.021 [−0.029, −0.015] | **hurts** |

**MK4-LR and MK4-RW hold on real photos. SN does not.**

### 2.2 Full table (mIoU, + DenseCRF; seed only: raw)

| Source | SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|---|:-:|---|---:|---:|---:|---:|---:|
| C | off | V0 | 0.596 | 0.623 | 0.597 | 0.677 | 0.682 |
| C | off | V3 | 0.591 | 0.626 | 0.597 | 0.662 | 0.666 |
| C | off | seed | 0.539 | 0.556 | 0.549 | 0.619 | 0.624 |
| C | on | V0 | 0.619 | 0.646 | 0.624 | 0.684 | **0.687** |
| C | on | V3 | **0.608 (BRACE)** | 0.646 | 0.628 | 0.668 | 0.672 |
| C | on | seed | 0.558 | 0.574 | 0.574 | 0.622 | 0.626 |
| A | off | V0 | 0.596 | 0.624 | 0.597 | 0.674 | 0.678 |
| A | off | V3 | 0.591 | 0.624 | 0.597 | 0.656 | 0.658 |
| A | on | V0 | 0.619 | 0.647 | 0.636 | 0.681 | 0.683 |
| A | on | V3 | 0.608 | 0.644 | 0.632 | 0.663 | 0.666 |

Best cell: **0.687** (C, SA weights on, fine-only, MK4-RW), against BRACE's **0.608**, a gain of **+0.079**. The
label-free source C matches label-based A.

### 2.3 Component effects (source C, + DenseCRF)

| Gate | SA weights on − off, V0 | SA weights on − off, V3 | Hierarchy V3 − V0, uniform | Hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.023 [+0.016, +0.029] | +0.018 [+0.011, +0.024] | −0.005 [−0.015, +0.004] | −0.010 [−0.019, −0.002] |
| MK4 | +0.023 [+0.017, +0.028] | +0.020 [+0.015, +0.026] | +0.003 [−0.007, +0.012] | +0.001 [−0.007, +0.008] |
| SN | +0.027 | +0.031 | −0.000 | +0.004 |
| MK4-LR | +0.007 [+0.003, +0.011] | +0.006 [+0.002, +0.010] | **−0.015 [−0.023, −0.007]** | **−0.016 [−0.024, −0.008]** |
| MK4-RW | +0.006 [+0.002, +0.010] | +0.006 [+0.001, +0.010] | **−0.016 [−0.024, −0.008]** | **−0.016 [−0.023, −0.008]** |

- With LR / RW on, the hierarchy hurts.
- SA weights add only +0.006 once LR / RW is on, against about +0.02 otherwise.

### 2.4 Selected γ, robustness and gate diagnostics

- **Selected γ:**
  - MK4: 0.25–0.75 (weaker than γ = 1);
  - SN: 0.25–1;
  - LR / RW: 1.5–2 (stronger).
- **γ = 1 (untuned), C, SA weights on:**
  - LR − MK4: V0 +0.046 [+0.038, +0.054]; V3 +0.020 [+0.013, +0.026].
  - So the gain doesn't come from tuning γ.
- **Fixed 0.55 + DenseCRF (C, SA weights on):**

  | Map | none | MK4 | LR | RW |
  |---|---:|---:|---:|---:|
  | V0 | 0.615 | 0.641 | 0.666 | 0.671 |
  | V3 | 0.596 | 0.632 | 0.661 | 0.665 |

- **Real-photo RW AUROC per module (source C):**

  | Modules | Resolution | AUROC |
  |---|---|---|
  | 0 | 64×64 | 0.59 |
  | 1 | 64×64 | 0.48 |
  | 2 | 32×32 | 0.78 |
  | 3 | 32×32 | 0.85 |
  | 4–7 | 16 / 8 / 16 / 16 | 0.92–0.95 |
  | 8–10 | 64×64 | 0.43–0.49 |

  So on real photos, RW ends up close to LR.
- **Per class (C, SA weights on, V3: none → MK4 → LR → RW):**

  | Class | none | MK4 | LR | RW |
  |---|---:|---:|---:|---:|
  | **person** | 0.41 | 0.52 | 0.64 | 0.65 |
  | chair | 0.31 | 0.31 | 0.37 | 0.38 |
  | car | 0.47 | 0.52 | 0.54 | 0.54 |
  | bicycle | 0.31 | 0.35 | 0.36 | 0.38 |

  - At V0, person goes 0.36 → 0.47 → 0.67 → 0.68.
  - Flat or slightly worse: bus (0.83 → 0.84), dining table (0.46 → 0.47), bottle (0.47 → 0.46).

---

## 3. Run 38 results: generated VOC-Sim / COCO-Cap (test halves)

### 3.1 Primary contrasts (SA weights on, + DenseCRF; Δ vs MK4)

| Regime | # | Contrast | Map | VOC-Sim Δ [95% CI] | COCO-Cap Δ [95% CI] | Reading |
|---|---|---|---|---|---|---|
| X | P1 | LR − MK4 | V3 | +0.006 [−0.005, +0.016] | +0.030 [+0.014, +0.045] | does not improve (VOC-Sim lower bound −0.005, just misses) |
| X | P2 | LR − MK4 | V0 | +0.009 [−0.002, +0.020] | +0.043 [+0.027, +0.057] | **improves** |
| X | P3 | RW − MK4 | V3 | +0.008 [−0.003, +0.018] | +0.028 [+0.012, +0.043] | **improves** |
| X | P4 | RW − MK4 | V0 | +0.011 [−0.000, +0.021] | +0.044 [+0.027, +0.060] | **improves** |
| X | P5 | SN − MK4 | V3 | −0.017 [−0.029, −0.005] | −0.038 [−0.057, −0.018] | hurts |
| X | P6 | SN − MK4 | V0 | −0.019 [−0.031, −0.007] | −0.029 [−0.049, −0.011] | hurts |
| V | P1 | LR − MK4 | V3 | +0.010 [+0.005, +0.015] | +0.027 [+0.012, +0.043] | **improves** |
| V | P2 | LR − MK4 | V0 | +0.009 [+0.001, +0.017] | +0.030 [+0.015, +0.042] | **improves** |
| V | P3 | RW − MK4 | V3 | +0.008 [+0.004, +0.012] | +0.014 [+0.004, +0.025] | **improves** |
| V | P4 | RW − MK4 | V0 | +0.004 [−0.005, +0.011] | +0.031 [+0.013, +0.045] | **improves** |
| V | P5 | SN − MK4 | V3 | −0.004 [−0.012, +0.003] | −0.031 [−0.044, −0.018] | hurts (COCO-Cap) |
| V | P6 | SN − MK4 | V0 | −0.009 [−0.019, +0.001] | −0.022 [−0.045, +0.001] | not resolved |

**MK4-LR and MK4-RW hold in both regimes. SN does not hold in either.**

### 3.2 Full tables (mIoU, VOC-Sim / COCO-Cap, + DenseCRF; seed only: raw)

**Regime X** (profiles from half of each set):

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---|---|---|---|---|
| off | V0 | 0.826 / 0.671 | 0.842 / 0.719 | 0.817 / 0.697 | 0.856 / 0.784 | 0.856 / 0.788 |
| off | V3 | 0.831 / 0.706 | 0.842 / 0.728 | 0.816 / 0.716 | 0.856 / 0.791 | 0.857 / 0.792 |
| off | seed | 0.716 / 0.570 | 0.742 / 0.625 | 0.725 / 0.604 | 0.758 / 0.687 | 0.759 / 0.690 |
| on | V0 | 0.828 / 0.705 | 0.847 / 0.751 | 0.828 / 0.721 | 0.857 / 0.794 | 0.859 / 0.795 |
| on | V3 | **0.838 / 0.728 (BRACE)** | 0.851 / 0.771 | 0.834 / 0.733 | **0.857 / 0.801** | **0.859 / 0.798** |
| on | seed | 0.708 / 0.593 | 0.730 / 0.640 | 0.713 / 0.613 | 0.746 / 0.678 | 0.749 / 0.678 |

**Regime V** (VOC-Sim profiles; COCO-Cap is transfer):

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---|---|---|---|---|
| off | V0 | 0.825 / 0.650 | 0.830 / 0.702 | 0.826 / 0.663 | 0.847 / 0.734 | 0.845 / 0.727 |
| off | V3 | 0.835 / 0.698 | 0.836 / 0.701 | 0.833 / 0.692 | 0.848 / 0.751 | 0.846 / 0.753 |
| off | seed | 0.714 / 0.568 | 0.736 / 0.609 | 0.723 / 0.598 | 0.750 / 0.660 | 0.754 / 0.661 |
| on | V0 | 0.833 / 0.695 | 0.842 / 0.726 | 0.833 / 0.704 | 0.851 / 0.756 | 0.846 / 0.757 |
| on | V3 | **0.842 / 0.718 (BRACE)** | 0.845 / 0.740 | 0.840 / 0.709 | **0.854 / 0.767** | 0.853 / 0.754 |
| on | seed | 0.708 / 0.593 | 0.724 / 0.630 | 0.711 / 0.606 | 0.739 / 0.658 | 0.742 / 0.662 |

Best cell against BRACE:

| Regime | Best cell | vs BRACE |
|---|---|---|
| X | 0.857 / 0.801 | **+0.019 / +0.073** |
| V | 0.854 / 0.767 | **+0.012 / +0.049** |

### 3.3 Component effects (COCO-Cap test, + DenseCRF; VOC-Sim effects are small, mostly 0 to +0.01)

| Regime | Gate | SA weights on − off, V0 | SA weights on − off, V3 | Hierarchy V3 − V0, uniform | Hierarchy V3 − V0, SA weights |
|---|---|---|---|---|---|
| X | none | +0.034 | +0.021 | **+0.035 [+0.019, +0.053]** | **+0.023 [+0.006, +0.040]** |
| X | MK4 | +0.032 | +0.042 | +0.009 | +0.020 [+0.003, +0.037] |
| X | MK4-LR | +0.010 | +0.009 | +0.007 [−0.007, +0.022] | +0.006 [−0.009, +0.024] |
| X | MK4-RW | +0.007 | +0.006 | +0.004 | +0.003 |
| V | none | +0.045 | +0.020 | **+0.048** | **+0.023** |
| V | MK4 | +0.025 | +0.039 | −0.000 | +0.014 |
| V | MK4-LR | +0.022 | +0.016 | +0.017 | +0.011 |
| V | MK4-RW | +0.030 | +0.001 | +0.026 | −0.002 |

Without a gate, the hierarchy helps on COCO-Cap. With LR / RW, it is about neutral: **unlike real VOC, it doesn't
hurt.**

### 3.4 Selected γ, robustness, per class

- **Selected γ:**
  - SN takes **0.25 in every cell**, the smallest in the grid, so dev wants as little SN as possible;
  - MK4: 0.25–1.5;
  - LR / RW: 1.5–2 in X, 0.75–1.5 in V.
- **γ = 1 (untuned), X, SA weights on, LR − MK4:**

  | Map | VOC-Sim | COCO-Cap |
  |---|---|---|
  | V0 | +0.009 [+0.003, +0.015] | +0.039 [+0.026, +0.050] |
  | V3 | +0.007 [+0.001, +0.012] | +0.016 [+0.003, +0.032] |

- **Fixed 0.55 + DenseCRF (X, SA weights on):**

  | Map | none | LR |
  |---|---|---|
  | V3 | 0.842 / 0.718 | **0.863 / 0.800** |
  | V0 | 0.810 / 0.700 | 0.808 / 0.768 |

  Fine-only needs a well-chosen threshold; the hierarchy is robust to the threshold.
- **Per class, COCO-Cap (X, V3):**

  | Class | none | MK4 | SN | LR | RW |
  |---|---:|---:|---:|---:|---:|
  | **person** | 0.18 | 0.51 | 0.25 | **0.85** | 0.82 |
  | chair | 0.49 | 0.59 | — | 0.70 | 0.70 |
  | dining table | 0.42 | 0.52 | — | 0.54 | 0.54 |
  | bottle | 0.73 | 0.90 | — | 0.87 | 0.88 |

  Bottle is slightly worse with LR / RW than with MK4. On VOC-Sim, person goes 0.77 → 0.81 → 0.84 (none → MK4 → LR).

---

## 4. Real vs generated, side by side (SA weights on, + DenseCRF)

| Quantity | Real VOC (run 37, source C) | Generated X (VOC-Sim / COCO-Cap) | Generated V (VOC-Sim / COCO-Cap) |
|---|---|---|---|
| BRACE-equivalent (none, V3) | 0.608 | 0.838 / 0.728 | 0.842 / 0.718 |
| MK4 − none, V3 | +0.038 | +0.013 / +0.043 | +0.002 / +0.022 |
| LR − MK4, V3 | +0.021 | +0.006 / +0.030 | +0.010 / +0.027 |
| LR − MK4, V0 | +0.038 | +0.009 / +0.043 | +0.009 / +0.030 |
| RW − MK4, V3 / V0 | +0.025 / +0.042 | +0.008 / +0.028; +0.011 / +0.044 | +0.008 / +0.014; +0.004 / +0.031 |
| SN − MK4, V3 | −0.018 | −0.017 / −0.038 | −0.004 / −0.031 |
| Hierarchy V3 − V0, no gate | −0.010 | +0.009 / +0.023 | +0.009 / +0.023 |
| Hierarchy V3 − V0, with LR | **−0.016** | −0.000 / +0.006 | +0.003 / +0.011 |
| Best cell | V0 + RW: 0.687 | V3 + LR: 0.857 / 0.801 | V3 + LR: 0.854 / 0.767 |
| Person IoU, none → LR (V3) | 0.41 → 0.64 | COCO-Cap 0.18 → 0.85 | COCO-Cap 0.19 → 0.64 |

---

## 5. What is established and what is not

**Established (exploratory, but replicated in three settings: real VOC, generated X, generated V):**
1. **The consensus gates beat MK4.**
   - MK4-LR and MK4-RW improve on MK4 by about +0.02 to +0.04 on COCO-Cap and real VOC, and by about +0.01 on
     VOC-Sim.
   - This holds with and without the hierarchy, at tuned and untuned γ, and with label-free profiles.
2. **LR ≈ RW.** On real photos the RW weights put about 0 on modules 1 and 8–10 (64×64) and only about 0.17 on
   module 0, so RW is almost LR. LR is simpler and needs no mask-based weight fit.
3. **SN is worse than MK4 end to end everywhere.** This is despite its strong stage-1 discrimination (run 35). Dev
   always selects the weakest γ for it.
4. **Most of the gain is person.** The 64×64 modules' gates are near chance, so averaging the reliable low-resolution
   gates mainly cleans up classes whose word leaks widely, person above all.
5. **Overlap with SA weights and the hierarchy.** Once LR / RW is on, the SA weights and hierarchy add little. Their
   job (suppressing the noisy fine-resolution, off-object attention) overlaps with the consensus gate.

**Not established:**
1. **Whether to drop the hierarchy.**
   - It hurts with LR / RW on real photos (−0.016), is neutral on generated images, and is much more robust at a
     fixed threshold of 0.55.
   - A method change would be an owner decision; nothing is promoted.
2. **Confirmatory evidence.**
   - VOC val has been looked at three times (runs 32, 33, 37).
   - Generated test halves have been used in runs 31 and 34–38.
   - A clean confirmation needs untouched data, for example real COCO val or another real benchmark, with LR fixed in
     advance.
3. **Why the hierarchy differs between real and generated images.** One hypothesis: the single t = 100 call on real
   photos gives noisier 16 / 32 self-attention than 11 generation-time calls. That's untested.
4. **Bottle and similar small or thin classes** sometimes lose a little with LR. It isn't known whether this is
   systematic.

---

## 6. Questions for review

1. Is LR (no fitted weights) or RW (weights fitted on masks) the better candidate to carry forward?
2. Given the real-vs-generated disagreement on the hierarchy, what decides "keep V3" vs "fine-only"? For example:
   - a confirmatory real benchmark;
   - the fixed-threshold robustness;
   - a multi-call real-photo capture.
3. Which untouched dataset should be the confirmatory test, and should it be pre-registered with LR, V3 and SA weights
   fixed?
4. Is SN worth anything further (for example only on the low-resolution modules), or should it be closed? Our
   default is to close it.

---

## Appendix A: run 37 full report (`runs/37_realvoc_gates/result.md`)

Known wording bug: the SN verdict line's bracket repeats the rule text, not SN's outcome. SN hurts in both maps.

### Run 37: run 36 gates on real PASCAL VOC photos, strengths and RW weights fitted on VOC train (`PREREG.md`)

**Exploratory; third look at VOC val.** VOC 2012 val: 1449 photos, 2148 items. Fitted on VOC train (2170 items): thresholds per map, gate strength γ per gate family and map (grid [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]), RW module weights (2170 items). One SD1.5 call per item at t = 100. Class-macro mIoU; 95% photo-bootstrap intervals (2000 samples, paired).


### 1. Primary contrasts (val, + DenseCRF, train-selected threshold and γ; source C, SA weights on)

| # | Contrast | Hierarchy | Δ mIoU [95% CI] | Reading |
|---|---|---|---|---|
| P1 | MK4-LR − MK4 | capped hierarchy (V3) | +0.021 [+0.014, +0.028] | **helps** |
| P2 | MK4-LR − MK4 | fine-only (V0) | +0.038 [+0.028, +0.048] | **helps** |
| P3 | MK4-RW − MK4 | capped hierarchy (V3) | +0.025 [+0.019, +0.032] | **helps** |
| P4 | MK4-RW − MK4 | fine-only (V0) | +0.042 [+0.032, +0.051] | **helps** |
| P5 | SN − MK4 | capped hierarchy (V3) | -0.018 [-0.025, -0.012] | **hurts** |
| P6 | SN − MK4 | fine-only (V0) | -0.021 [-0.029, -0.015] | **hurts** |

- MK4-LR: **holds on real photos** (helps on V3 or V0, hurts on neither).
- MK4-RW: **holds on real photos** (helps on V3 or V0, hurts on neither).
- SN: does not hold on real photos (helps on V3 or V0, hurts on neither).

### 2. Train-selected γ

| Source | SA weights | Map | MK4 | SN | MK4-LR | MK4-RW |
|---|:-:|---|---:|---:|---:|---:|
| C |  | seed only | 0.25 | 0.25 | 1.5 | 1.5 |
| C |  | fine-only (V0) | 0.5 | 0.25 | 1.5 | 1.5 |
| C |  | capped hierarchy (V3) | 0.5 | 0.5 | 2.0 | 2.0 |
| C | ✓ | seed only | 0.25 | 1.0 | 1.5 | 1.5 |
| C | ✓ | fine-only (V0) | 0.5 | 0.25 | 1.5 | 1.5 |
| C | ✓ | capped hierarchy (V3) | 0.75 | 1.0 | 1.5 | 1.5 |
| A |  | seed only | 0.25 | 0.75 | 1.5 | 1.5 |
| A |  | fine-only (V0) | 0.5 | 0.5 | 2.0 | 2.0 |
| A |  | capped hierarchy (V3) | 0.5 | 0.5 | 2.0 | 2.0 |
| A | ✓ | seed only | 0.5 | 1.0 | 1.5 | 1.5 |
| A | ✓ | fine-only (V0) | 0.75 | 1.0 | 2.0 | 2.0 |
| A | ✓ | capped hierarchy (V3) | 1.0 | 1.5 | 2.0 | 2.0 |

### 3. Source C (real COCO photos, captions; label-free profiles)

mIoU, primary rule (seed only: raw, train threshold), train-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.596 | 0.623 | 0.597 | 0.677 | 0.682 |
|  | capped hierarchy (V3) | 0.591 | 0.626 | 0.597 | 0.662 | 0.666 |
|  | seed only | 0.539 | 0.556 | 0.549 | 0.619 | 0.624 |
| ✓ | fine-only (V0) | 0.619 | 0.646 | 0.624 | 0.684 | 0.687 |
| ✓ | capped hierarchy (V3) | 0.608 | 0.646 | 0.628 | 0.668 | 0.672 |
| ✓ | seed only | 0.558 | 0.574 | 0.574 | 0.622 | 0.626 |

Paired differences, train-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.027 [+0.017, +0.038] | -0.026 [-0.034, -0.017] | +0.054 [+0.043, +0.064] | +0.059 [+0.048, +0.069] |
|  | capped hierarchy (V3) | +0.035 [+0.026, +0.044] | -0.029 [-0.036, -0.022] | +0.036 [+0.026, +0.046] | +0.040 [+0.030, +0.050] |
|  | seed only | +0.017 [+0.011, +0.022] | -0.007 [-0.010, -0.003] | +0.064 [+0.056, +0.072] | +0.069 [+0.061, +0.076] |
| ✓ | fine-only (V0) | +0.027 [+0.019, +0.036] | -0.021 [-0.029, -0.015] | +0.038 [+0.028, +0.048] | +0.042 [+0.032, +0.051] |
| ✓ | capped hierarchy (V3) | +0.038 [+0.029, +0.047] | -0.018 [-0.025, -0.012] | +0.021 [+0.014, +0.028] | +0.025 [+0.019, +0.032] |
| ✓ | seed only | +0.016 [+0.012, +0.020] | +0.000 [-0.005, +0.005] | +0.048 [+0.042, +0.055] | +0.052 [+0.046, +0.059] |

Component effects with the gate fixed (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.023 [+0.016, +0.029] | +0.018 [+0.011, +0.024] | -0.005 [-0.015, +0.004] | -0.010 [-0.019, -0.002] |
| MK4 | +0.023 [+0.017, +0.028] | +0.020 [+0.015, +0.026] | +0.003 [-0.007, +0.012] | +0.001 [-0.007, +0.008] |
| SN | +0.027 [+0.021, +0.033] | +0.031 [+0.024, +0.038] | -0.000 [-0.010, +0.009] | +0.004 [-0.005, +0.013] |
| MK4-LR | +0.007 [+0.003, +0.011] | +0.006 [+0.002, +0.010] | -0.015 [-0.023, -0.007] | -0.016 [-0.024, -0.008] |
| MK4-RW | +0.006 [+0.002, +0.010] | +0.006 [+0.001, +0.010] | -0.016 [-0.024, -0.008] | -0.016 [-0.023, -0.008] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4 |
|---|---:|---:|---:|---:|---|
| fine-only (V0) | 0.636 | 0.629 | 0.682 | 0.686 | +0.046 [+0.038, +0.054] |
| capped hierarchy (V3) | 0.644 | 0.628 | 0.664 | 0.666 | +0.020 [+0.013, +0.026] |

### 4. Source A (VOC train labels; reference)

mIoU, primary rule (seed only: raw, train threshold), train-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.596 | 0.624 | 0.597 | 0.674 | 0.678 |
|  | capped hierarchy (V3) | 0.591 | 0.624 | 0.597 | 0.656 | 0.658 |
|  | seed only | 0.539 | 0.557 | 0.558 | 0.633 | 0.636 |
| ✓ | fine-only (V0) | 0.619 | 0.647 | 0.636 | 0.681 | 0.683 |
| ✓ | capped hierarchy (V3) | 0.608 | 0.644 | 0.632 | 0.663 | 0.666 |
| ✓ | seed only | 0.558 | 0.577 | 0.585 | 0.631 | 0.633 |

Paired differences, train-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.029 [+0.019, +0.038] | -0.027 [-0.036, -0.018] | +0.049 [+0.037, +0.061] | +0.054 [+0.042, +0.065] |
|  | capped hierarchy (V3) | +0.033 [+0.025, +0.042] | -0.026 [-0.033, -0.019] | +0.032 [+0.023, +0.042] | +0.034 [+0.025, +0.044] |
|  | seed only | +0.018 [+0.013, +0.024] | +0.000 [-0.005, +0.005] | +0.076 [+0.068, +0.083] | +0.079 [+0.071, +0.086] |
| ✓ | fine-only (V0) | +0.028 [+0.019, +0.038] | -0.011 [-0.019, -0.004] | +0.034 [+0.025, +0.043] | +0.037 [+0.027, +0.045] |
| ✓ | capped hierarchy (V3) | +0.036 [+0.026, +0.045] | -0.013 [-0.019, -0.005] | +0.019 [+0.011, +0.027] | +0.022 [+0.014, +0.030] |
| ✓ | seed only | +0.019 [+0.014, +0.025] | +0.008 [+0.003, +0.012] | +0.054 [+0.047, +0.060] | +0.056 [+0.049, +0.062] |

Component effects with the gate fixed (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.023 [+0.016, +0.029] | +0.018 [+0.011, +0.024] | -0.005 [-0.015, +0.004] | -0.010 [-0.019, -0.002] |
| MK4 | +0.023 [+0.017, +0.029] | +0.020 [+0.013, +0.028] | -0.000 [-0.010, +0.009] | -0.003 [-0.011, +0.005] |
| SN | +0.039 [+0.031, +0.046] | +0.034 [+0.026, +0.042] | +0.000 [-0.009, +0.010] | -0.004 [-0.012, +0.004] |
| MK4-LR | +0.007 [+0.003, +0.012] | +0.007 [+0.004, +0.011] | -0.018 [-0.025, -0.010] | -0.018 [-0.026, -0.010] |
| MK4-RW | +0.006 [+0.001, +0.010] | +0.008 [+0.004, +0.012] | -0.020 [-0.028, -0.012] | -0.017 [-0.025, -0.010] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4 |
|---|---:|---:|---:|---:|---|
| fine-only (V0) | 0.642 | 0.636 | 0.677 | 0.679 | +0.035 [+0.027, +0.043] |
| capped hierarchy (V3) | 0.644 | 0.629 | 0.655 | 0.657 | +0.011 [+0.004, +0.018] |

### 5. Other rules, source C, SA weights on (train-selected γ)

| Map | Gate | + CRF, train thr (primary) | + CRF, 0.55 | raw, train thr | raw, 0.55 |
|---|---|---:|---:|---:|---:|
| fine-only (V0) | none | 0.619 | 0.615 | 0.624 | 0.623 |
| fine-only (V0) | MK4 | 0.646 | 0.641 | 0.653 | 0.651 |
| fine-only (V0) | SN | 0.624 | 0.621 | 0.630 | 0.629 |
| fine-only (V0) | MK4-LR | 0.684 | 0.666 | 0.688 | 0.679 |
| fine-only (V0) | MK4-RW | 0.687 | 0.671 | 0.691 | 0.684 |
| capped hierarchy (V3) | none | 0.608 | 0.596 | 0.616 | 0.595 |
| capped hierarchy (V3) | MK4 | 0.646 | 0.632 | 0.653 | 0.632 |
| capped hierarchy (V3) | SN | 0.628 | 0.616 | 0.635 | 0.616 |
| capped hierarchy (V3) | MK4-LR | 0.668 | 0.661 | 0.674 | 0.663 |
| capped hierarchy (V3) | MK4-RW | 0.672 | 0.665 | 0.677 | 0.666 |

### 6. Per class IoU (primary rule, source C, SA weights on)

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 90 | 0.813 | 0.857 | 0.844 | 0.860 | 0.861 | 0.806 | 0.830 | 0.813 | 0.845 | 0.844 |
| bicycle | 79 | 0.311 | 0.345 | 0.345 | 0.362 | 0.376 | 0.295 | 0.336 | 0.314 | 0.377 | 0.378 |
| bird | 103 | 0.710 | 0.728 | 0.746 | 0.772 | 0.764 | 0.757 | 0.761 | 0.763 | 0.804 | 0.804 |
| boat | 72 | 0.646 | 0.726 | 0.718 | 0.742 | 0.740 | 0.660 | 0.732 | 0.675 | 0.746 | 0.749 |
| bottle | 96 | 0.465 | 0.431 | 0.402 | 0.448 | 0.459 | 0.482 | 0.428 | 0.445 | 0.502 | 0.503 |
| bus | 74 | 0.828 | 0.830 | 0.811 | 0.835 | 0.835 | 0.838 | 0.816 | 0.830 | 0.811 | 0.814 |
| car | 127 | 0.471 | 0.523 | 0.496 | 0.541 | 0.544 | 0.461 | 0.495 | 0.473 | 0.542 | 0.542 |
| cat | 119 | 0.829 | 0.856 | 0.833 | 0.874 | 0.875 | 0.846 | 0.864 | 0.845 | 0.898 | 0.900 |
| chair | 123 | 0.305 | 0.313 | 0.303 | 0.372 | 0.377 | 0.316 | 0.334 | 0.297 | 0.430 | 0.438 |
| cow | 71 | 0.827 | 0.870 | 0.868 | 0.880 | 0.879 | 0.854 | 0.874 | 0.876 | 0.891 | 0.892 |
| dining table | 75 | 0.463 | 0.451 | 0.437 | 0.472 | 0.477 | 0.476 | 0.454 | 0.462 | 0.471 | 0.475 |
| dog | 128 | 0.807 | 0.843 | 0.830 | 0.846 | 0.853 | 0.812 | 0.821 | 0.814 | 0.853 | 0.863 |
| horse | 79 | 0.765 | 0.831 | 0.819 | 0.827 | 0.840 | 0.785 | 0.856 | 0.824 | 0.881 | 0.883 |
| motorbike | 76 | 0.742 | 0.759 | 0.756 | 0.766 | 0.766 | 0.729 | 0.744 | 0.734 | 0.754 | 0.755 |
| person | 446 | 0.410 | 0.517 | 0.439 | 0.640 | 0.650 | 0.364 | 0.466 | 0.382 | 0.669 | 0.684 |
| potted plant | 85 | 0.253 | 0.288 | 0.254 | 0.294 | 0.295 | 0.294 | 0.333 | 0.282 | 0.365 | 0.370 |
| sheep | 57 | 0.745 | 0.861 | 0.857 | 0.871 | 0.872 | 0.846 | 0.885 | 0.884 | 0.879 | 0.880 |
| sofa | 90 | 0.671 | 0.709 | 0.649 | 0.735 | 0.743 | 0.640 | 0.714 | 0.640 | 0.736 | 0.749 |
| train | 84 | 0.733 | 0.756 | 0.738 | 0.776 | 0.784 | 0.720 | 0.741 | 0.725 | 0.767 | 0.772 |
| tv | 74 | 0.371 | 0.434 | 0.420 | 0.443 | 0.446 | 0.389 | 0.427 | 0.405 | 0.457 | 0.456 |

### 7. RW module weights (VOC train, all items; w = clip(2·(AUROC − 0.5), 0, 1) of the MK4 gate, object vs far)

| Source | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C AUROC | 0.587 | 0.480 | 0.783 | 0.851 | 0.950 | 0.951 | 0.928 | 0.918 | 0.488 | 0.494 | 0.434 |
| A AUROC | 0.633 | 0.487 | 0.820 | 0.898 | 0.965 | 0.971 | 0.957 | 0.915 | 0.466 | 0.469 | 0.430 |

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

Mean gate value at γ = 1 over modules and pixels (val): C/MK4 0.479, C/SN 0.287, C/MK4-LR 0.400, C/MK4-RW 0.400, A/MK4 0.492, A/SN 0.296, A/MK4-LR 0.420, A/MK4-RW 0.421.

Run 33 t = 100 (different capture, descriptive only, primary rule): all off 0.596, BRACE 0.608, full stack C 0.645 (uncapped hierarchy + MK4).


---

## Appendix B: run 38 full report (`runs/38_gen_gates_factorial/result.md`)

## Run 38: SA weights × hierarchy × gate on VOC-Sim and COCO-Cap, everything fitted on dev (`PREREG.md`)

**Exploratory.** Test halves: VOC-Sim 237, COCO-Cap 351 images. Fitted on dev (VOC-Sim 253, COCO-Cap 356): thresholds per regime and map, gate strength γ per regime × SA weights × gate × map (grid [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]). RW weights from run 36 (dev, held out). Regime V: profiles and fits from VOC-Sim dev only (COCO-Cap is untuned transfer). Regime X: VOC-Sim + COCO-Cap dev. One generation-time capture for every map. Class-macro mIoU; 95% image-bootstrap intervals (2000 samples, paired). DenseCRF at the dev threshold chosen on raw maps (run 37 rule).


### 1. Primary contrasts (test, + DenseCRF, dev-selected threshold and γ, SA weights on)

"Improves" = COCO-Cap CI lower bound > 0 and VOC-Sim CI lower bound ≥ −0.005 (run 36 rule).

| Regime | # | Contrast | Map | VOC-Sim Δ [95% CI] | COCO-Cap Δ [95% CI] | reading |
|---|---|---|---|---|---|---|
| X | P1 | MK4-LR − MK4 | capped hierarchy (V3) | +0.006 [-0.005, +0.016] (n.r.) | +0.030 [+0.014, +0.045] (helps) | does not improve |
| X | P2 | MK4-LR − MK4 | fine-only (V0) | +0.009 [-0.002, +0.020] (n.r.) | +0.043 [+0.027, +0.057] (helps) | **improves** |
| X | P3 | MK4-RW − MK4 | capped hierarchy (V3) | +0.008 [-0.003, +0.018] (n.r.) | +0.028 [+0.012, +0.043] (helps) | **improves** |
| X | P4 | MK4-RW − MK4 | fine-only (V0) | +0.011 [-0.000, +0.021] (n.r.) | +0.044 [+0.027, +0.060] (helps) | **improves** |
| X | P5 | SN − MK4 | capped hierarchy (V3) | -0.017 [-0.029, -0.005] (hurts) | -0.038 [-0.057, -0.018] (hurts) | does not improve |
| X | P6 | SN − MK4 | fine-only (V0) | -0.019 [-0.031, -0.007] (hurts) | -0.029 [-0.049, -0.011] (hurts) | does not improve |
| V | P1 | MK4-LR − MK4 | capped hierarchy (V3) | +0.010 [+0.005, +0.015] (helps) | +0.027 [+0.012, +0.043] (helps) | **improves** |
| V | P2 | MK4-LR − MK4 | fine-only (V0) | +0.009 [+0.001, +0.017] (helps) | +0.030 [+0.015, +0.042] (helps) | **improves** |
| V | P3 | MK4-RW − MK4 | capped hierarchy (V3) | +0.008 [+0.004, +0.012] (helps) | +0.014 [+0.004, +0.025] (helps) | **improves** |
| V | P4 | MK4-RW − MK4 | fine-only (V0) | +0.004 [-0.005, +0.011] (n.r.) | +0.031 [+0.013, +0.045] (helps) | **improves** |
| V | P5 | SN − MK4 | capped hierarchy (V3) | -0.004 [-0.012, +0.003] (n.r.) | -0.031 [-0.044, -0.018] (hurts) | does not improve |
| V | P6 | SN − MK4 | fine-only (V0) | -0.009 [-0.019, +0.001] (n.r.) | -0.022 [-0.045, +0.001] (n.r.) | does not improve |

- Regime X, MK4-LR: **holds** (improves on V3 or V0: yes; hurts somewhere: no).
- Regime X, MK4-RW: **holds** (improves on V3 or V0: yes; hurts somewhere: no).
- Regime X, SN: does not hold (improves on V3 or V0: no; hurts somewhere: yes).
- Regime V, MK4-LR: **holds** (improves on V3 or V0: yes; hurts somewhere: no).
- Regime V, MK4-RW: **holds** (improves on V3 or V0: yes; hurts somewhere: no).
- Regime V, SN: does not hold (improves on V3 or V0: no; hurts somewhere: yes).

### 2. Dev-selected γ

| Regime | SA weights | Map | MK4 | SN | MK4-LR | MK4-RW |
|---|:-:|---|---:|---:|---:|---:|
| X |  | seed only | 0.5 | 0.25 | 1.5 | 1.5 |
| X |  | fine-only (V0) | 0.75 | 0.25 | 1.5 | 2.0 |
| X |  | capped hierarchy (V3) | 0.75 | 0.25 | 2.0 | 2.0 |
| X | ✓ | seed only | 1.0 | 0.25 | 1.5 | 2.0 |
| X | ✓ | fine-only (V0) | 1.0 | 0.25 | 2.0 | 2.0 |
| X | ✓ | capped hierarchy (V3) | 1.5 | 0.25 | 2.0 | 2.0 |
| V |  | seed only | 0.5 | 0.25 | 1.0 | 1.0 |
| V |  | fine-only (V0) | 0.5 | 0.25 | 1.0 | 1.0 |
| V |  | capped hierarchy (V3) | 0.25 | 0.25 | 1.0 | 1.0 |
| V | ✓ | seed only | 0.75 | 0.25 | 1.0 | 1.5 |
| V | ✓ | fine-only (V0) | 1.5 | 0.25 | 1.5 | 1.5 |
| V | ✓ | capped hierarchy (V3) | 0.5 | 0.25 | 1.0 | 0.75 |

### 3. Regime X (VOC-Sim + COCO-Cap dev profiles)

mIoU, VOC-Sim / COCO-Cap test, primary rule (seed only: raw, dev threshold), dev-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.826 / 0.671 | 0.842 / 0.719 | 0.817 / 0.697 | 0.856 / 0.784 | 0.856 / 0.788 |
|  | capped hierarchy (V3) | 0.831 / 0.706 | 0.842 / 0.728 | 0.816 / 0.716 | 0.856 / 0.791 | 0.857 / 0.792 |
|  | seed only | 0.716 / 0.570 | 0.742 / 0.625 | 0.725 / 0.604 | 0.758 / 0.687 | 0.759 / 0.690 |
| ✓ | fine-only (V0) | 0.828 / 0.705 | 0.847 / 0.751 | 0.828 / 0.721 | 0.857 / 0.794 | 0.859 / 0.795 |
| ✓ | capped hierarchy (V3) | 0.838 / 0.728 | 0.851 / 0.771 | 0.834 / 0.733 | 0.857 / 0.801 | 0.859 / 0.798 |
| ✓ | seed only | 0.708 / 0.593 | 0.730 / 0.640 | 0.713 / 0.613 | 0.746 / 0.678 | 0.749 / 0.678 |

Paired differences, VOC-Sim test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.016 [+0.001, +0.030] | -0.025 [-0.040, -0.009] | +0.014 [+0.005, +0.022] | +0.014 [+0.004, +0.022] |
|  | capped hierarchy (V3) | +0.011 [+0.000, +0.019] | -0.025 [-0.038, -0.012] | +0.014 [+0.001, +0.027] | +0.015 [+0.003, +0.028] |
|  | seed only | +0.026 [+0.018, +0.034] | -0.017 [-0.026, -0.007] | +0.016 [+0.008, +0.022] | +0.017 [+0.009, +0.023] |
| ✓ | fine-only (V0) | +0.019 [+0.007, +0.031] | -0.019 [-0.031, -0.007] | +0.009 [-0.002, +0.020] | +0.011 [-0.000, +0.021] |
| ✓ | capped hierarchy (V3) | +0.013 [+0.003, +0.023] | -0.017 [-0.029, -0.005] | +0.006 [-0.005, +0.016] | +0.008 [-0.003, +0.018] |
| ✓ | seed only | +0.022 [+0.014, +0.030] | -0.017 [-0.025, -0.009] | +0.016 [+0.012, +0.020] | +0.020 [+0.015, +0.024] |

Paired differences, COCO-Cap test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.048 [+0.023, +0.072] | -0.022 [-0.041, -0.006] | +0.065 [+0.048, +0.080] | +0.069 [+0.050, +0.086] |
|  | capped hierarchy (V3) | +0.022 [+0.000, +0.046] | -0.013 [-0.030, +0.004] | +0.063 [+0.044, +0.079] | +0.064 [+0.042, +0.084] |
|  | seed only | +0.055 [+0.038, +0.070] | -0.021 [-0.030, -0.012] | +0.062 [+0.049, +0.074] | +0.065 [+0.052, +0.076] |
| ✓ | fine-only (V0) | +0.046 [+0.026, +0.067] | -0.029 [-0.049, -0.011] | +0.043 [+0.027, +0.057] | +0.044 [+0.027, +0.060] |
| ✓ | capped hierarchy (V3) | +0.043 [+0.023, +0.064] | -0.038 [-0.057, -0.018] | +0.030 [+0.014, +0.045] | +0.028 [+0.012, +0.043] |
| ✓ | seed only | +0.047 [+0.033, +0.060] | -0.026 [-0.039, -0.015] | +0.038 [+0.030, +0.045] | +0.038 [+0.030, +0.045] |

Component effects with the gate fixed, VOC-Sim test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.002 [-0.004, +0.009] | +0.007 [+0.000, +0.013] | +0.005 [-0.005, +0.015] | +0.009 [-0.000, +0.019] |
| MK4 | +0.005 [-0.002, +0.014] | +0.009 [-0.000, +0.020] | -0.001 [-0.014, +0.013] | +0.003 [-0.009, +0.016] |
| SN | +0.012 [+0.003, +0.021] | +0.018 [+0.009, +0.027] | -0.000 [-0.010, +0.011] | +0.006 [-0.004, +0.016] |
| MK4-LR | +0.001 [-0.003, +0.004] | +0.001 [-0.004, +0.006] | -0.000 [-0.008, +0.008] | -0.000 [-0.008, +0.008] |
| MK4-RW | +0.003 [-0.000, +0.007] | +0.002 [-0.003, +0.008] | +0.001 [-0.007, +0.009] | +0.000 [-0.007, +0.008] |

Component effects with the gate fixed, COCO-Cap test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.034 [+0.017, +0.050] | +0.021 [+0.005, +0.035] | +0.035 [+0.019, +0.053] | +0.023 [+0.006, +0.040] |
| MK4 | +0.032 [+0.015, +0.048] | +0.042 [+0.025, +0.057] | +0.009 [-0.004, +0.024] | +0.020 [+0.003, +0.037] |
| SN | +0.024 [+0.005, +0.042] | +0.017 [-0.003, +0.036] | +0.018 [+0.005, +0.034] | +0.012 [-0.003, +0.028] |
| MK4-LR | +0.010 [+0.001, +0.018] | +0.009 [+0.000, +0.020] | +0.007 [-0.007, +0.022] | +0.006 [-0.009, +0.024] |
| MK4-RW | +0.007 [-0.001, +0.016] | +0.006 [-0.001, +0.012] | +0.004 [-0.012, +0.022] | +0.003 [-0.012, +0.018] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule, VOC-Sim / COCO-Cap:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4, VOC-Sim | MK4-LR − MK4, COCO-Cap |
|---|---:|---:|---:|---:|---|---|
| fine-only (V0) | 0.847 / 0.751 | 0.821 / 0.730 | 0.857 / 0.790 | 0.857 / 0.786 | +0.009 [+0.003, +0.015] | +0.039 [+0.026, +0.050] |
| capped hierarchy (V3) | 0.849 / 0.775 | 0.830 / 0.746 | 0.856 / 0.792 | 0.855 / 0.790 | +0.007 [+0.001, +0.012] | +0.016 [+0.003, +0.032] |

### 4. Regime V (VOC-Sim dev profiles; COCO-Cap is transfer)

mIoU, VOC-Sim / COCO-Cap test, primary rule (seed only: raw, dev threshold), dev-selected γ:

| SA weights | Map | none | MK4 | SN | MK4-LR | MK4-RW |
|:-:|---|---:|---:|---:|---:|---:|
|  | fine-only (V0) | 0.825 / 0.650 | 0.830 / 0.702 | 0.826 / 0.663 | 0.847 / 0.734 | 0.845 / 0.727 |
|  | capped hierarchy (V3) | 0.835 / 0.698 | 0.836 / 0.701 | 0.833 / 0.692 | 0.848 / 0.751 | 0.846 / 0.753 |
|  | seed only | 0.714 / 0.568 | 0.736 / 0.609 | 0.723 / 0.598 | 0.750 / 0.660 | 0.754 / 0.661 |
| ✓ | fine-only (V0) | 0.833 / 0.695 | 0.842 / 0.726 | 0.833 / 0.704 | 0.851 / 0.756 | 0.846 / 0.757 |
| ✓ | capped hierarchy (V3) | 0.842 / 0.718 | 0.845 / 0.740 | 0.840 / 0.709 | 0.854 / 0.767 | 0.853 / 0.754 |
| ✓ | seed only | 0.708 / 0.593 | 0.724 / 0.630 | 0.711 / 0.606 | 0.739 / 0.658 | 0.742 / 0.662 |

Paired differences, VOC-Sim test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.005 [-0.004, +0.014] | -0.004 [-0.013, +0.006] | +0.017 [+0.007, +0.027] | +0.015 [+0.005, +0.025] |
|  | capped hierarchy (V3) | +0.001 [-0.003, +0.004] | -0.003 [-0.007, +0.002] | +0.012 [+0.003, +0.022] | +0.010 [+0.001, +0.019] |
|  | seed only | +0.022 [+0.014, +0.030] | -0.012 [-0.022, -0.002] | +0.015 [+0.009, +0.020] | +0.018 [+0.013, +0.023] |
| ✓ | fine-only (V0) | +0.009 [-0.001, +0.017] | -0.009 [-0.019, +0.001] | +0.009 [+0.001, +0.017] | +0.004 [-0.005, +0.011] |
| ✓ | capped hierarchy (V3) | +0.002 [-0.004, +0.007] | -0.004 [-0.012, +0.003] | +0.010 [+0.005, +0.015] | +0.008 [+0.004, +0.012] |
| ✓ | seed only | +0.017 [+0.009, +0.024] | -0.013 [-0.020, -0.005] | +0.014 [+0.011, +0.018] | +0.018 [+0.013, +0.022] |

Paired differences, COCO-Cap test, dev-selected γ:

| SA weights | Map | MK4 − none | SN − MK4 | MK4-LR − MK4 | MK4-RW − MK4 |
|:-:|---|---|---|---|---|
|  | fine-only (V0) | +0.052 [+0.029, +0.075] | -0.039 [-0.055, -0.023] | +0.032 [+0.013, +0.051] | +0.025 [+0.009, +0.041] |
|  | capped hierarchy (V3) | +0.004 [-0.009, +0.016] | -0.009 [-0.017, -0.002] | +0.050 [+0.030, +0.070] | +0.052 [+0.035, +0.068] |
|  | seed only | +0.041 [+0.025, +0.054] | -0.011 [-0.019, -0.002] | +0.050 [+0.036, +0.066] | +0.051 [+0.037, +0.066] |
| ✓ | fine-only (V0) | +0.031 [+0.009, +0.055] | -0.022 [-0.045, +0.001] | +0.030 [+0.015, +0.042] | +0.031 [+0.013, +0.045] |
| ✓ | capped hierarchy (V3) | +0.022 [+0.007, +0.038] | -0.031 [-0.044, -0.018] | +0.027 [+0.012, +0.043] | +0.014 [+0.004, +0.025] |
| ✓ | seed only | +0.037 [+0.024, +0.048] | -0.024 [-0.034, -0.014] | +0.028 [+0.020, +0.036] | +0.032 [+0.024, +0.040] |

Component effects with the gate fixed, VOC-Sim test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.008 [+0.000, +0.018] | +0.007 [-0.001, +0.017] | +0.010 [+0.002, +0.019] | +0.009 [+0.000, +0.018] |
| MK4 | +0.012 [+0.004, +0.022] | +0.009 [+0.002, +0.018] | +0.006 [-0.001, +0.015] | +0.003 [-0.005, +0.011] |
| SN | +0.007 [-0.002, +0.017] | +0.007 [-0.001, +0.015] | +0.007 [+0.000, +0.015] | +0.007 [-0.003, +0.018] |
| MK4-LR | +0.005 [+0.000, +0.008] | +0.006 [+0.002, +0.011] | +0.002 [-0.006, +0.009] | +0.003 [-0.005, +0.012] |
| MK4-RW | +0.001 [-0.004, +0.007] | +0.007 [+0.002, +0.013] | +0.001 [-0.006, +0.008] | +0.007 [-0.001, +0.015] |

Component effects with the gate fixed, COCO-Cap test (primary rule):

| Gate | SA weights (on − off), V0 | SA weights, V3 | hierarchy V3 − V0, uniform | hierarchy V3 − V0, SA weights |
|---|---|---|---|---|
| none | +0.045 [+0.027, +0.061] | +0.020 [+0.006, +0.034] | +0.048 [+0.030, +0.066] | +0.023 [+0.006, +0.042] |
| MK4 | +0.025 [+0.007, +0.042] | +0.039 [+0.024, +0.053] | -0.000 [-0.019, +0.019] | +0.014 [-0.013, +0.040] |
| SN | +0.041 [+0.025, +0.058] | +0.017 [+0.001, +0.032] | +0.029 [+0.015, +0.044] | +0.005 [-0.013, +0.023] |
| MK4-LR | +0.022 [+0.008, +0.036] | +0.016 [+0.001, +0.030] | +0.017 [+0.003, +0.032] | +0.011 [-0.006, +0.030] |
| MK4-RW | +0.030 [+0.017, +0.042] | +0.001 [-0.016, +0.017] | +0.026 [+0.011, +0.042] | -0.002 [-0.022, +0.017] |

Descriptive: every gate at γ = 1 (untuned), SA weights on, primary rule, VOC-Sim / COCO-Cap:

| Map | MK4 | SN | MK4-LR | MK4-RW | MK4-LR − MK4, VOC-Sim | MK4-LR − MK4, COCO-Cap |
|---|---:|---:|---:|---:|---|---|
| fine-only (V0) | 0.841 / 0.732 | 0.822 / 0.691 | 0.851 / 0.755 | 0.849 / 0.749 | +0.010 [+0.003, +0.016] | +0.023 [+0.012, +0.035] |
| capped hierarchy (V3) | 0.845 / 0.744 | 0.830 / 0.720 | 0.854 / 0.767 | 0.853 / 0.760 | +0.009 [+0.004, +0.014] | +0.024 [+0.009, +0.036] |

### 5. Other rules, SA weights on (dev-selected γ), VOC-Sim / COCO-Cap

| Regime | Map | Gate | + CRF, dev thr (primary) | + CRF, 0.55 | raw, dev thr | raw, 0.55 |
|---|---|---|---:|---:|---:|---:|
| X | fine-only (V0) | none | 0.828 / 0.705 | 0.810 / 0.700 | 0.816 / 0.692 | 0.801 / 0.691 |
| X | fine-only (V0) | MK4 | 0.847 / 0.751 | 0.810 / 0.735 | 0.834 / 0.738 | 0.803 / 0.728 |
| X | fine-only (V0) | SN | 0.828 / 0.721 | 0.806 / 0.718 | 0.816 / 0.709 | 0.798 / 0.710 |
| X | fine-only (V0) | MK4-LR | 0.857 / 0.794 | 0.808 / 0.768 | 0.844 / 0.778 | 0.804 / 0.761 |
| X | fine-only (V0) | MK4-RW | 0.859 / 0.795 | 0.812 / 0.771 | 0.846 / 0.778 | 0.807 / 0.763 |
| X | capped hierarchy (V3) | none | 0.838 / 0.728 | 0.842 / 0.718 | 0.825 / 0.715 | 0.826 / 0.702 |
| X | capped hierarchy (V3) | MK4 | 0.851 / 0.771 | 0.857 / 0.763 | 0.838 / 0.758 | 0.840 / 0.747 |
| X | capped hierarchy (V3) | SN | 0.834 / 0.733 | 0.841 / 0.724 | 0.822 / 0.722 | 0.824 / 0.709 |
| X | capped hierarchy (V3) | MK4-LR | 0.857 / 0.801 | 0.863 / 0.800 | 0.845 / 0.787 | 0.847 / 0.780 |
| X | capped hierarchy (V3) | MK4-RW | 0.859 / 0.798 | 0.864 / 0.796 | 0.846 / 0.785 | 0.848 / 0.776 |
| V | fine-only (V0) | none | 0.833 / 0.695 | 0.810 / 0.700 | 0.818 / 0.679 | 0.801 / 0.691 |
| V | fine-only (V0) | MK4 | 0.842 / 0.726 | 0.778 / 0.698 | 0.824 / 0.707 | 0.776 / 0.694 |
| V | fine-only (V0) | SN | 0.833 / 0.704 | 0.804 / 0.708 | 0.817 / 0.689 | 0.797 / 0.701 |
| V | fine-only (V0) | MK4-LR | 0.851 / 0.756 | 0.797 / 0.739 | 0.834 / 0.736 | 0.792 / 0.733 |
| V | fine-only (V0) | MK4-RW | 0.846 / 0.757 | 0.799 / 0.743 | 0.831 / 0.738 | 0.795 / 0.735 |
| V | capped hierarchy (V3) | none | 0.842 / 0.718 | 0.842 / 0.718 | 0.826 / 0.702 | 0.826 / 0.702 |
| V | capped hierarchy (V3) | MK4 | 0.845 / 0.740 | 0.845 / 0.740 | 0.828 / 0.724 | 0.828 / 0.724 |
| V | capped hierarchy (V3) | SN | 0.840 / 0.709 | 0.842 / 0.717 | 0.822 / 0.692 | 0.825 / 0.702 |
| V | capped hierarchy (V3) | MK4-LR | 0.854 / 0.767 | 0.854 / 0.767 | 0.838 / 0.749 | 0.838 / 0.749 |
| V | capped hierarchy (V3) | MK4-RW | 0.853 / 0.754 | 0.853 / 0.754 | 0.836 / 0.737 | 0.836 / 0.737 |

### 6. Per class IoU (primary rule, SA weights on, dev-selected γ)


Regime X, VOC-Sim test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 12 | 0.892 | 0.914 | 0.900 | 0.914 | 0.914 | 0.896 | 0.905 | 0.906 | 0.903 | 0.903 |
| bicycle | 12 | 0.785 | 0.770 | 0.786 | 0.779 | 0.779 | 0.742 | 0.738 | 0.740 | 0.754 | 0.754 |
| bird | 12 | 0.931 | 0.934 | 0.931 | 0.928 | 0.927 | 0.933 | 0.934 | 0.932 | 0.933 | 0.932 |
| boat | 12 | 0.873 | 0.880 | 0.876 | 0.880 | 0.879 | 0.862 | 0.866 | 0.868 | 0.872 | 0.872 |
| bottle | 12 | 0.887 | 0.886 | 0.887 | 0.886 | 0.887 | 0.894 | 0.897 | 0.893 | 0.904 | 0.905 |
| bus | 12 | 0.920 | 0.922 | 0.920 | 0.921 | 0.921 | 0.918 | 0.915 | 0.916 | 0.915 | 0.914 |
| car | 11 | 0.887 | 0.879 | 0.888 | 0.882 | 0.884 | 0.904 | 0.907 | 0.906 | 0.911 | 0.913 |
| cat | 12 | 0.840 | 0.823 | 0.813 | 0.819 | 0.842 | 0.793 | 0.834 | 0.773 | 0.833 | 0.860 |
| chair | 12 | 0.816 | 0.828 | 0.818 | 0.823 | 0.821 | 0.835 | 0.832 | 0.840 | 0.830 | 0.828 |
| cow | 12 | 0.892 | 0.918 | 0.914 | 0.950 | 0.950 | 0.872 | 0.922 | 0.925 | 0.951 | 0.953 |
| dining table | 12 | 0.467 | 0.534 | 0.465 | 0.542 | 0.539 | 0.551 | 0.582 | 0.543 | 0.598 | 0.597 |
| dog | 12 | 0.959 | 0.952 | 0.951 | 0.940 | 0.941 | 0.964 | 0.960 | 0.961 | 0.952 | 0.954 |
| horse | 11 | 0.812 | 0.903 | 0.817 | 0.909 | 0.924 | 0.798 | 0.853 | 0.799 | 0.921 | 0.914 |
| motorbike | 12 | 0.883 | 0.878 | 0.882 | 0.873 | 0.873 | 0.877 | 0.876 | 0.877 | 0.873 | 0.873 |
| person | 12 | 0.767 | 0.805 | 0.666 | 0.839 | 0.846 | 0.720 | 0.811 | 0.675 | 0.870 | 0.877 |
| potted plant | 12 | 0.756 | 0.767 | 0.762 | 0.743 | 0.744 | 0.720 | 0.721 | 0.713 | 0.723 | 0.732 |
| sheep | 12 | 0.794 | 0.860 | 0.809 | 0.954 | 0.954 | 0.829 | 0.956 | 0.833 | 0.951 | 0.953 |
| sofa | 12 | 0.954 | 0.954 | 0.954 | 0.954 | 0.954 | 0.951 | 0.954 | 0.960 | 0.955 | 0.956 |
| train | 12 | 0.803 | 0.845 | 0.821 | 0.864 | 0.861 | 0.782 | 0.793 | 0.805 | 0.838 | 0.832 |
| tv | 11 | 0.834 | 0.765 | 0.823 | 0.730 | 0.740 | 0.727 | 0.692 | 0.705 | 0.649 | 0.653 |

Regime X, COCO-Cap test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 24 | 0.808 | 0.833 | 0.850 | 0.891 | 0.887 | 0.737 | 0.778 | 0.786 | 0.847 | 0.832 |
| bicycle | 23 | 0.665 | 0.734 | 0.677 | 0.738 | 0.738 | 0.621 | 0.688 | 0.640 | 0.713 | 0.713 |
| bird | 24 | 0.900 | 0.876 | 0.896 | 0.891 | 0.893 | 0.880 | 0.854 | 0.873 | 0.875 | 0.877 |
| boat | 16 | 0.615 | 0.693 | 0.621 | 0.706 | 0.700 | 0.620 | 0.691 | 0.628 | 0.703 | 0.705 |
| bottle | 10 | 0.732 | 0.897 | 0.759 | 0.873 | 0.876 | 0.718 | 0.907 | 0.748 | 0.852 | 0.849 |
| bus | 18 | 0.931 | 0.915 | 0.930 | 0.921 | 0.922 | 0.939 | 0.925 | 0.936 | 0.925 | 0.927 |
| car | 14 | 0.827 | 0.824 | 0.812 | 0.851 | 0.848 | 0.688 | 0.691 | 0.687 | 0.730 | 0.717 |
| cat | 23 | 0.915 | 0.912 | 0.915 | 0.909 | 0.911 | 0.919 | 0.921 | 0.921 | 0.917 | 0.920 |
| chair | 10 | 0.488 | 0.585 | 0.492 | 0.704 | 0.699 | 0.556 | 0.547 | 0.559 | 0.694 | 0.742 |
| cow | 24 | 0.840 | 0.869 | 0.843 | 0.899 | 0.900 | 0.813 | 0.861 | 0.836 | 0.896 | 0.897 |
| dining table | 7 | 0.415 | 0.517 | 0.407 | 0.539 | 0.540 | 0.348 | 0.543 | 0.392 | 0.594 | 0.606 |
| dog | 22 | 0.804 | 0.813 | 0.810 | 0.815 | 0.814 | 0.813 | 0.830 | 0.819 | 0.833 | 0.833 |
| horse | 19 | 0.765 | 0.820 | 0.790 | 0.823 | 0.824 | 0.769 | 0.811 | 0.799 | 0.816 | 0.821 |
| motorbike | 20 | 0.734 | 0.751 | 0.738 | 0.755 | 0.755 | 0.731 | 0.740 | 0.733 | 0.743 | 0.743 |
| person | 26 | 0.183 | 0.509 | 0.246 | 0.845 | 0.824 | 0.170 | 0.336 | 0.247 | 0.852 | 0.845 |
| potted plant | 7 | 0.727 | 0.764 | 0.697 | 0.809 | 0.807 | 0.773 | 0.846 | 0.807 | 0.863 | 0.867 |
| sheep | 19 | 0.877 | 0.825 | 0.877 | 0.839 | 0.841 | 0.893 | 0.843 | 0.861 | 0.847 | 0.848 |
| sofa | 17 | 0.718 | 0.799 | 0.735 | 0.791 | 0.792 | 0.688 | 0.804 | 0.718 | 0.817 | 0.827 |
| train | 16 | 0.766 | 0.681 | 0.725 | 0.699 | 0.697 | 0.699 | 0.687 | 0.714 | 0.698 | 0.693 |
| tv | 12 | 0.840 | 0.798 | 0.843 | 0.716 | 0.703 | 0.723 | 0.714 | 0.725 | 0.668 | 0.642 |

Regime V, VOC-Sim test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 12 | 0.874 | 0.885 | 0.867 | 0.891 | 0.888 | 0.892 | 0.897 | 0.896 | 0.896 | 0.900 |
| bicycle | 12 | 0.783 | 0.784 | 0.781 | 0.782 | 0.784 | 0.746 | 0.729 | 0.752 | 0.738 | 0.749 |
| bird | 12 | 0.922 | 0.928 | 0.888 | 0.933 | 0.927 | 0.922 | 0.930 | 0.922 | 0.932 | 0.932 |
| boat | 12 | 0.854 | 0.861 | 0.847 | 0.870 | 0.867 | 0.838 | 0.837 | 0.844 | 0.855 | 0.863 |
| bottle | 12 | 0.896 | 0.896 | 0.898 | 0.896 | 0.896 | 0.902 | 0.905 | 0.902 | 0.909 | 0.907 |
| bus | 12 | 0.916 | 0.920 | 0.910 | 0.921 | 0.920 | 0.902 | 0.902 | 0.906 | 0.897 | 0.905 |
| car | 11 | 0.890 | 0.893 | 0.889 | 0.896 | 0.894 | 0.907 | 0.911 | 0.906 | 0.915 | 0.914 |
| cat | 12 | 0.880 | 0.868 | 0.882 | 0.867 | 0.877 | 0.855 | 0.881 | 0.836 | 0.862 | 0.860 |
| chair | 12 | 0.786 | 0.789 | 0.773 | 0.786 | 0.785 | 0.807 | 0.805 | 0.816 | 0.789 | 0.799 |
| cow | 12 | 0.925 | 0.896 | 0.933 | 0.909 | 0.917 | 0.915 | 0.842 | 0.921 | 0.917 | 0.892 |
| dining table | 12 | 0.436 | 0.457 | 0.428 | 0.476 | 0.464 | 0.505 | 0.548 | 0.506 | 0.532 | 0.550 |
| dog | 12 | 0.967 | 0.966 | 0.966 | 0.965 | 0.965 | 0.968 | 0.958 | 0.966 | 0.960 | 0.963 |
| horse | 11 | 0.853 | 0.875 | 0.871 | 0.907 | 0.889 | 0.829 | 0.869 | 0.822 | 0.898 | 0.872 |
| motorbike | 12 | 0.880 | 0.884 | 0.877 | 0.883 | 0.884 | 0.870 | 0.878 | 0.878 | 0.878 | 0.878 |
| person | 12 | 0.816 | 0.824 | 0.796 | 0.856 | 0.849 | 0.766 | 0.845 | 0.748 | 0.882 | 0.852 |
| potted plant | 12 | 0.816 | 0.833 | 0.846 | 0.874 | 0.876 | 0.773 | 0.817 | 0.765 | 0.815 | 0.761 |
| sheep | 12 | 0.811 | 0.819 | 0.817 | 0.829 | 0.826 | 0.845 | 0.877 | 0.843 | 0.906 | 0.869 |
| sofa | 12 | 0.949 | 0.947 | 0.946 | 0.947 | 0.946 | 0.954 | 0.953 | 0.953 | 0.949 | 0.955 |
| train | 12 | 0.733 | 0.756 | 0.726 | 0.806 | 0.790 | 0.706 | 0.682 | 0.737 | 0.750 | 0.775 |
| tv | 11 | 0.860 | 0.815 | 0.865 | 0.798 | 0.805 | 0.763 | 0.772 | 0.742 | 0.744 | 0.723 |

Regime V, COCO-Cap test:

| Class | items | none V3 | MK4 V3 | SN V3 | MK4-LR V3 | MK4-RW V3 | none V0 | MK4 V0 | SN V0 | MK4-LR V0 | MK4-RW V0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 24 | 0.740 | 0.812 | 0.728 | 0.832 | 0.819 | 0.673 | 0.699 | 0.711 | 0.753 | 0.761 |
| bicycle | 23 | 0.654 | 0.665 | 0.640 | 0.685 | 0.668 | 0.612 | 0.657 | 0.617 | 0.650 | 0.670 |
| bird | 24 | 0.897 | 0.896 | 0.874 | 0.900 | 0.898 | 0.895 | 0.871 | 0.891 | 0.892 | 0.893 |
| boat | 16 | 0.570 | 0.596 | 0.564 | 0.643 | 0.616 | 0.571 | 0.686 | 0.598 | 0.665 | 0.669 |
| bottle | 10 | 0.708 | 0.766 | 0.714 | 0.869 | 0.772 | 0.702 | 0.885 | 0.718 | 0.834 | 0.859 |
| bus | 18 | 0.924 | 0.923 | 0.920 | 0.923 | 0.923 | 0.926 | 0.917 | 0.928 | 0.916 | 0.923 |
| car | 14 | 0.834 | 0.835 | 0.819 | 0.821 | 0.834 | 0.725 | 0.605 | 0.680 | 0.697 | 0.717 |
| cat | 23 | 0.915 | 0.916 | 0.913 | 0.916 | 0.916 | 0.909 | 0.911 | 0.910 | 0.915 | 0.917 |
| chair | 10 | 0.480 | 0.497 | 0.467 | 0.516 | 0.505 | 0.485 | 0.496 | 0.473 | 0.514 | 0.561 |
| cow | 24 | 0.873 | 0.846 | 0.863 | 0.858 | 0.863 | 0.849 | 0.752 | 0.835 | 0.859 | 0.838 |
| dining table | 7 | 0.481 | 0.546 | 0.453 | 0.544 | 0.517 | 0.415 | 0.640 | 0.454 | 0.657 | 0.572 |
| dog | 22 | 0.813 | 0.822 | 0.811 | 0.829 | 0.829 | 0.808 | 0.820 | 0.811 | 0.826 | 0.831 |
| horse | 19 | 0.761 | 0.794 | 0.785 | 0.812 | 0.804 | 0.764 | 0.798 | 0.778 | 0.816 | 0.818 |
| motorbike | 20 | 0.722 | 0.730 | 0.718 | 0.738 | 0.733 | 0.713 | 0.726 | 0.722 | 0.728 | 0.735 |
| person | 26 | 0.188 | 0.368 | 0.229 | 0.636 | 0.539 | 0.164 | 0.307 | 0.203 | 0.633 | 0.639 |
| potted plant | 7 | 0.691 | 0.722 | 0.673 | 0.747 | 0.757 | 0.738 | 0.847 | 0.774 | 0.827 | 0.844 |
| sheep | 19 | 0.884 | 0.887 | 0.889 | 0.839 | 0.880 | 0.883 | 0.846 | 0.896 | 0.859 | 0.859 |
| sofa | 17 | 0.690 | 0.737 | 0.665 | 0.772 | 0.762 | 0.670 | 0.759 | 0.683 | 0.792 | 0.820 |
| train | 16 | 0.766 | 0.707 | 0.740 | 0.704 | 0.712 | 0.710 | 0.627 | 0.705 | 0.607 | 0.599 |
| tv | 12 | 0.774 | 0.734 | 0.715 | 0.760 | 0.742 | 0.690 | 0.674 | 0.700 | 0.675 | 0.610 |

### 7. RW module weights used (run 36, all dev of the regime; MK4 gate AUROC object vs far)

| Regime | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X AUROC | 0.630 | 0.464 | 0.844 | 0.935 | 0.982 | 0.988 | 0.961 | 0.966 | 0.568 | 0.604 | 0.528 |
| X weight | 0.26 | 0.00 | 0.69 | 0.87 | 0.96 | 0.98 | 0.92 | 0.93 | 0.14 | 0.21 | 0.06 |
| V AUROC | 0.650 | 0.460 | 0.902 | 0.947 | 0.987 | 0.994 | 0.970 | 0.964 | 0.594 | 0.683 | 0.593 |
| V weight | 0.30 | 0.00 | 0.80 | 0.89 | 0.97 | 0.99 | 0.94 | 0.93 | 0.19 | 0.37 | 0.19 |

Modules: 0 = down_blocks.0.attentions.0.transformer_blocks.0.attn2, 1 = down_blocks.0.attentions.1.transformer_blocks.0.attn2, 2 = down_blocks.2.attentions.0.transformer_blocks.0.attn2, 3 = down_blocks.2.attentions.1.transformer_blocks.0.attn2, 4 = mid_block.attentions.0.transformer_blocks.0.attn2, 5 = up_blocks.1.attentions.0.transformer_blocks.0.attn2, 6 = up_blocks.1.attentions.1.transformer_blocks.0.attn2, 7 = up_blocks.1.attentions.2.transformer_blocks.0.attn2, 8 = up_blocks.3.attentions.0.transformer_blocks.0.attn2, 9 = up_blocks.3.attentions.1.transformer_blocks.0.attn2, 10 = up_blocks.3.attentions.2.transformer_blocks.0.attn2

Mean gate value at γ = 1 over modules and pixels (test, VOC-Sim / COCO-Cap): X/MK4 0.541 / 0.452, X/SN 0.340 / 0.286, X/MK4-LR 0.513 / 0.380, X/MK4-RW 0.518 / 0.394, V/MK4 0.516 / 0.400, V/SN 0.331 / 0.259, V/MK4-LR 0.484 / 0.335, V/MK4-RW 0.493 / 0.354.

BRACE (W on, no gate, V3 + DenseCRF) is the 'none / capped hierarchy / ✓' cell. BRACE stays frozen; any change to the paper's method is a separate owner decision.

