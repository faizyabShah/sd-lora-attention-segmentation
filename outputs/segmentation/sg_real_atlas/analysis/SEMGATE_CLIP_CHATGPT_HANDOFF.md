# Semantic gating + CLIP track: combined handoff (2026-11-09)

All scores are class-macro IoU over (photo, class) pairs on real VOC2012 train photos, using our own per-class protocol, not the standard VOC21 benchmark. D = 200 exploration photos (307 pairs). F = 200 fresh photos (294 pairs), never used before the confirmation. SCLIP is compared without PAMR.

Contents: 1. Semantic gating (D) · 2. SCLIP port faithfulness · 3. CLIP diagnostic (D) · 4. Confirmation pre-registration · 5. Confirmation on F

---
## 1. Semantic gating (D)
## Semantic gating inside the capped hierarchy (real VOC D, exploratory)

307 pairs, 200 photos. Definitions and pass rules were fixed before the run (see `scripts/sg_sem_gate.py` docstring): gate g = (c+ε)/(c+b+2ε), c = target A3 seed, b = max of the 8 generic-word seeds (each min-max), λ = 1. `mid` gates the 16 px map that feeds A32, the V3 cap and the 16→64 C16 path; `out` multiplies the final maps by g; `shift` is `mid` with the gate rolled by (8, 8); `comp` = marg.5/mm/gen seed. OOF τ over 5 photo-grouped folds; Δ CIs photo-bootstrap 95%. Max replay |Δ| vs the stored maps: 1.3e-03.

| variant | V3+R2 OOF | C16 OOF | V3 OOF | P1+R2 OOF | V3+R2 prec / rec / FP-on-other | V3 AUROC tgt-other / tgt-bg | C16 AUROC tgt-other / tgt-bg |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.538 (τ≈0.70) | 0.553 (τ≈0.68) | 0.533 (τ≈0.72) | 0.547 (τ≈0.60) | 0.528 / 0.815 / 45% | 0.665 / 0.939 | 0.688 / 0.944 |
| comp | 0.601 (τ≈0.60) | 0.596 (τ≈0.50) | 0.597 (τ≈0.60) | 0.574 (τ≈0.40) | 0.567 / 0.829 / 49% | 0.729 / 0.943 | 0.745 / 0.945 |
| out | 0.534 (τ≈0.60) | 0.537 (τ≈0.60) | 0.524 (τ≈0.62) | 0.558 (τ≈0.50) | 0.515 / 0.798 / 41% | 0.765 / 0.941 | 0.765 / 0.942 |
| mid | 0.553 (τ≈0.65) | 0.568 (τ≈0.62) | 0.555 (τ≈0.68) | 0.547 (τ≈0.60) | 0.529 / 0.833 / 44% | 0.716 / 0.945 | 0.741 / 0.947 |
| comp+out | 0.569 (τ≈0.50) | 0.567 (τ≈0.47) | 0.564 (τ≈0.53) | 0.565 (τ≈0.30) | 0.514 / 0.835 / 42% | 0.774 / 0.945 | 0.777 / 0.945 |
| comp+mid | 0.592 (τ≈0.55) | 0.595 (τ≈0.47) | 0.592 (τ≈0.55) | 0.574 (τ≈0.40) | 0.561 / 0.824 / 48% | 0.745 / 0.945 | 0.761 / 0.946 |
| shift | 0.235 (τ≈0.55) | 0.243 (τ≈0.53) | 0.239 (τ≈0.57) | 0.547 (τ≈0.60) | 0.263 / 0.584 / 19% | 0.629 / 0.721 | 0.620 / 0.735 |

### Registered comparisons

| test | comparison | map | Δ | 95% CI | pass |
|---|---|---|---:|---|---|
| M | mid − out | V3+R2 | +0.019 | -0.005…+0.050 | no |
| M | mid − out | C16 | +0.030 | +0.014…+0.048 | yes |
| M | comp+mid − comp+out | V3+R2 | +0.024 | +0.009…+0.036 | yes |
| M | comp+mid − comp+out | C16 | +0.029 | +0.017…+0.040 | yes |
| X | mid − shift | V3+R2 | +0.318 | +0.274…+0.358 | yes |
| X | mid − shift | C16 | +0.325 | +0.283…+0.363 | yes |
| P | best gated (comp+mid C16) − comp C16 (registered) | — | -0.000 | -0.007…+0.007 | no |
| P | best gated (comp+mid C16) − comp V3+R2 | — | -0.005 | -0.022…+0.014 | no |

**Mechanism (M):** PASS. **Spatial control (X):** PASS. **Practical (P, vs comp/C16):** FAIL. **Line continues:** no — close.

### Per-class V3+R2 OOF IoU (base → comp → mid → comp+mid)

aeroplane 0.69→0.80→0.70→0.80, bicycle 0.34→0.32→0.32→0.32, bird 0.42→0.74→0.52→0.69, boat 0.62→0.71→0.69→0.72, bottle 0.21→0.20→0.21→0.20, bus 0.69→0.70→0.68→0.67, car 0.40→0.42→0.37→0.41, cat 0.88→0.92→0.87→0.90, chair 0.26→0.27→0.26→0.26, cow 0.79→0.86→0.85→0.83, dining-table 0.56→0.65→0.60→0.68, dog 0.67→0.72→0.68→0.70, horse 0.53→0.68→0.52→0.66, motorbike 0.69→0.82→0.73→0.82, person 0.33→0.34→0.33→0.33, potted-plant 0.15→0.13→0.15→0.13, sheep 0.75→0.82→0.81→0.82, sofa 0.50→0.49→0.53→0.53, train 0.79→0.83→0.75→0.81, tv 0.48→0.58→0.49→0.56

### By object size (V3+R2 OOF predictions, class-macro within bin)

| bin | n | base | comp | out | mid | comp+out | comp+mid | shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <5% | 101 | 0.140 | 0.224 | 0.203 | 0.157 | 0.242 | 0.227 | 0.047 |
| 5–20% | 104 | 0.550 | 0.603 | 0.553 | 0.548 | 0.571 | 0.591 | 0.222 |
| >20% | 102 | 0.781 | 0.772 | 0.701 | 0.790 | 0.742 | 0.766 | 0.367 |

---
## 2. SCLIP port faithfulness (VOC val; repo reference B16 VOC20 81.54 / VOC21 59.63)
```json
{
 "B16": {
  "VOC20": 81.4917447781627,
  "VOC21": 59.63612291206593,
  "VOC20_per_class": [
   93.7,
   57.1,
   90.2,
   85.1,
   79.9,
   92.2,
   87.1,
   94.0,
   42.6,
   89.0,
   67.7,
   90.4,
   88.0,
   72.8,
   85.0,
   82.1,
   87.2,
   73.9,
   94.9,
   76.9
  ],
  "n": 1449,
  "s_per_img_both_protocols": 0.1684428642872204
 },
 "L14": {
  "VOC20": 70.55754253875136,
  "VOC21": 44.31675941447429,
  "VOC20_per_class": [
   81.5,
   45.3,
   82.7,
   74.1,
   62.4,
   88.9,
   70.1,
   88.8,
   38.7,
   87.7,
   41.3,
   84.6,
   84.4,
   60.6,
   73.6,
   67.4,
   89.3,
   60.6,
   85.5,
   43.7
  ],
  "n": 1449,
  "s_per_img_both_protocols": 0.6140226721249423
 }
}
```

---
## 3. CLIP diagnostic (D)
## Dense CLIP (SCLIP ViT-B/16) vs SD attention on real VOC D (exploratory)

307 pairs, 200 photos. Pre-registered definitions and rules in `scripts/sg_clip_diag.py`. The SCLIP port reproduces the repo on VOC val (VOC20 81.49 vs 81.54, VOC21 59.64 vs 59.63). CLIP map = SCLIP's VOC21 class probability at 64×64. Class-given per-pair binary masks, class-macro IoU, OOF τ over 5 photo-grouped folds, photo-bootstrap 95% CIs. Max replay |Δ| 1.3e-03.

### Q1 — seed-level separation (before any SD propagation)

| seed | AUROC tgt vs other object | AUROC tgt vs background | AP |
|---|---:|---:|---:|
| base | 0.724 | 0.924 | 0.745 |
| comp | 0.754 | 0.911 | 0.744 |
| clip | 0.807 | 0.874 | 0.768 |
| clipcomp | 0.832 | 0.935 | 0.795 |
| fuse | 0.829 | 0.955 | 0.820 |

clip − A3 seed, tgt vs other: +0.083 (CI +0.049…+0.120)

clip − A3 seed, tgt vs background: -0.051 (CI -0.069…-0.030)

clip − A3 seed, AP: +0.023 (CI -0.003…+0.048)

By object size (seed AUROC tgt vs other / tgt vs bg):

| bin | n | A3 | CLIP |
|---|---:|---|---|
| <1% | 41 | 0.663 / 0.918 | 0.556 / 0.639 |
| 1–5% | 60 | 0.670 / 0.891 | 0.709 / 0.787 |
| 5–20% | 104 | 0.771 / 0.942 | 0.889 / 0.944 |
| >20% | 102 | 0.751 / 0.929 | 0.939 / 0.948 |

### Segmentation (OOF mIoU)

| variant | seed | C16 | V3+R2 | P1+R2 | best-map prec / rec / FP-on-other |
|---|---:|---:|---:|---:|---:|
| base | 0.488 | 0.553 | 0.538 | 0.547 | 0.562 / 0.786 / 50% (C16) |
| comp | 0.487 | 0.596 | 0.601 | 0.574 | 0.567 / 0.829 / 49% (V3+R2) |
| clip | 0.629 | 0.735 | 0.720 | 0.730 | 0.818 / 0.839 / 20% (C16) |
| clipcomp | 0.558 | 0.609 | 0.578 | 0.607 | 0.611 / 0.828 / 42% (C16) |
| fuse | 0.598 | 0.630 | 0.587 | 0.644 | 0.643 / 0.840 / 53% (P1+R2) |
| sclip | 0.639 | 0.639 | 0.673 | 0.673 | 0.825 / 0.787 / 17% (V3+R2) |

### Q2 — registered hybrid comparison

| comparison | Δ | 95% CI | pass |
|---|---:|---|---|
| best hybrid (fuse C16) − comp C16 (registered) | +0.034 | +0.007…+0.059 | yes |
| best hybrid (fuse C16) − best CLIP-only (clip C16) | -0.105 | -0.132…-0.061 | no |
| best CLIP-only − comp C16 | +0.139 | +0.091…+0.169 | — |
| clip C16 − clip seed (does SD SA help CLIP?) | +0.106 | +0.074…+0.134 | — |
| SCLIP standalone − comp C16 | +0.044 | -0.007…+0.080 | — |

**Hybrid claim:** FAIL — no hybrid is claimed.

### Per-class OOF IoU (comp C16 / SCLIP standalone / best hybrid)

aeroplane 0.75 / 0.54 / 0.78, bicycle 0.29 / 0.34 / 0.38, bird 0.74 / 0.75 / 0.52, boat 0.68 / 0.40 / 0.69, bottle 0.18 / 0.61 / 0.22, bus 0.67 / 0.81 / 0.75, car 0.41 / 0.77 / 0.56, cat 0.89 / 0.79 / 0.90, chair 0.26 / 0.41 / 0.30, cow 0.81 / 0.77 / 0.85, dining-table 0.65 / 0.66 / 0.70, dog 0.75 / 0.80 / 0.77, horse 0.81 / 0.59 / 0.81, motorbike 0.78 / 0.77 / 0.80, person 0.35 / 0.68 / 0.38, potted-plant 0.15 / 0.64 / 0.23, sheep 0.79 / 0.78 / 0.80, sofa 0.48 / 0.43 / 0.61, train 0.84 / 0.64 / 0.86, tv 0.63 / 0.60 / 0.69

### By object size (OOF IoU, class-macro within bin)

| bin | n | base V3+R2 | comp C16 | sclip seed | clip C16 | fuse C16 |
|---|---:|---:|---:|---:|---:|---:|
| <1% | 41 | 0.108 | 0.144 | 0.139 | 0.158 | 0.140 |
| 1–5% | 60 | 0.179 | 0.320 | 0.327 | 0.312 | 0.323 |
| 5–20% | 104 | 0.550 | 0.625 | 0.576 | 0.701 | 0.695 |
| >20% | 102 | 0.781 | 0.764 | 0.684 | 0.823 | 0.815 |

### Post-hoc check (not registered; computed after reading the table above)

The best CLIP-only pipeline is itself CLIP semantics + SD self-attention propagation. Against SCLIP with the same kind of refinement (DenseCRF), photo-bootstrap 95% CI:

| comparison | Δ | 95% CI |
|---|---:|---|
| clip C16 − SCLIP standalone + R2 | +0.063 | +0.028…+0.094 |
| clip C16 − SCLIP standalone | +0.096 | +0.066…+0.125 |
| clip P1+R2 − SCLIP standalone + R2 | +0.058 | +0.032…+0.082 |

Precision / recall: clip C16 0.818 / 0.839; SCLIP + R2 0.825 / 0.787. SD propagation gains recall at roughly constant precision. SCLIP's optional PAMR refinement was not tested.

---
## 4. Confirmation pre-registration
## Pre-registration: CLIP seed + SD propagation, confirmation on fresh real VOC photos (F)

Written 2026-11-08, before any F capture or score exists. Committed before the run.

**Set F.** The next 200 VOC2012 train ids in the seed-137 shuffle that produced S (`scripts/prep_sg_fresh.py`). It is disjoint from D and S by construction and has never been captured or inspected. 200 photos, 294 (photo, class) pairs, 5 photo-grouped class-stratified folds.

**Frozen candidate.** `clip C16`:
- The SCLIP (ViT-B/16, HF port reproducing the repo on VOC val) VOC21 class probability of the target, at 64×64, min-max normalised.
- It is used as the seed of the frozen C16 path through the target run's own SD1.5 DDIM self-attention.
- Thresholded at **τ = 0.575**, fitted on all of D.
- Nothing about it is changed after this note.

**Comparators.** All thresholds were fitted on D, and nothing is fitted on F:

| comparator | definition | τ |
|---|---|---|
| SCLIP standalone + R2 | SCLIP's own VOC21 prediction of the class, DenseCRF | none |
| SCLIP standalone | SCLIP's own VOC21 prediction of the class | none |
| comp C16 | generic-word SD competition (marg.5/mm/gen) → C16 | 0.50 |
| base V3+R2 | BRACE on real photos | 0.70 |

**Primary test (pass/fail).** `clip C16 @0.575` − `SCLIP standalone + R2` in class-macro IoU over F's pairs. The photo-bootstrap 95% CI (1000 resamples) must be > 0.

**Secondary tests (reported, not pass/fail).**
- `clip C16` vs SCLIP standalone, vs comp C16 and vs base V3+R2, all at the D thresholds.
- The same comparisons with τ chosen OOF within F.
- Per class and by object size (<1%, 1–5%, 5–20%, >20%).
- Seed-level AUROC (target vs other objects, target vs background) for the CLIP and A3 seeds.

**If the primary test fails,** the D result is treated as not confirmed. No variant is re-selected on F.

---
## 5. Confirmation on F
## Confirmation on fresh real VOC photos (F): CLIP seed + SD propagation

Pre-registered in `clip_confirm_prereg.md` (committed before the run). F: 294 pairs, 200 photos, never used before. All thresholds fixed on D; nothing fitted on F except the secondary OOF column.

| pipeline | τ (from D) | mIoU on F | prec / rec / FP-on-other |
|---|---:|---:|---|
| clip C16 | 0.575 | 0.733 | 0.828 / 0.862 / 13% |
| sclip R2 | — | 0.663 | 0.823 / 0.796 / 11% |
| sclip standalone | — | 0.627 | 0.772 / 0.810 / 9% |
| comp C16 | 0.50 | 0.591 | 0.600 / 0.862 / 40% |
| base V3+R2 | 0.70 | 0.534 | 0.553 / 0.801 / 38% |

**Primary (registered):** clip C16 − SCLIP + R2 = +0.069 (photo-bootstrap 95% CI +0.037…+0.108) → **CONFIRMED**.

### Secondary

| comparison | Δ | 95% CI |
|---|---:|---|
| clip C16 − sclip standalone | +0.106 | +0.075…+0.141 |
| clip C16 − comp C16 | +0.142 | +0.085…+0.172 |
| clip C16 − base V3+R2 | +0.199 | +0.132…+0.236 |

OOF within F (τ chosen on 4 F folds): clip C16 0.733, clip V3+R2 0.712, clip P1+R2 0.731, comp C16 0.599, base V3+R2 0.530, sclip V3+R2 0.663

Seed-level AUROC (tgt vs other / tgt vs bg): base 0.728 / 0.930, comp 0.736 / 0.915, clip 0.838 / 0.897

### By object size (fixed-τ IoU, class-macro within bin)

| bin | n | clip C16 | sclip R2 | comp C16 | base V3+R2 |
|---|---:|---:|---:|---:|---:|
| <1% | 25 | 0.152 | 0.144 | 0.133 | 0.035 |
| 1–5% | 58 | 0.330 | 0.269 | 0.275 | 0.261 |
| 5–20% | 121 | 0.685 | 0.580 | 0.625 | 0.568 |
| >20% | 90 | 0.857 | 0.749 | 0.752 | 0.724 |

### Per class (clip C16 / SCLIP + R2 / comp C16)

aeroplane 0.51 / 0.59 / 0.65, bicycle 0.58 / 0.55 / 0.44, bird 0.81 / 0.71 / 0.74, boat 0.70 / 0.38 / 0.36, bottle 0.66 / 0.76 / 0.31, bus 0.78 / 0.77 / 0.84, car 0.69 / 0.68 / 0.57, cat 0.93 / 0.81 / 0.90, chair 0.59 / 0.20 / 0.22, cow 0.91 / 0.90 / 0.76, dining-table 0.70 / 0.56 / 0.44, dog 0.90 / 0.90 / 0.89, horse 0.81 / 0.75 / 0.68, motorbike 0.79 / 0.68 / 0.71, person 0.58 / 0.70 / 0.44, potted-plant 0.63 / 0.73 / 0.32, sheep 0.90 / 0.82 / 0.88, sofa 0.77 / 0.57 / 0.47, train 0.74 / 0.64 / 0.81, tv 0.68 / 0.56 / 0.40

