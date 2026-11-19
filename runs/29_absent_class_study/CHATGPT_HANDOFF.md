# Absent-class calibration for training-free diffusion-attention segmentation: runs 28 and 29

Status as of 2026-11-19. This is self-contained context for discussing these results. Every number comes from
committed reports: `runs/28_trio_screen/result.md` and `runs/29_absent_class_study/result.md`.

## 1. Background: the method being modified (BRACE)

BRACE is a training-free, label-free method for class-word segmentation from Stable Diffusion 1.5 attention. Its
configuration is frozen.

1. **Generate.** SD1.5 generates the image from its prompt or caption. Queries and keys are recorded at 11
   denoising calls.
2. **Seed.** At the selected cross-attention modules, take each pixel's attention to the class word (softmax over
   the 77 text tokens, mean over heads). Weight the modules with self-attention-guided layer weights, sum over
   calls and min-max normalise to a 64×64 seed.
3. **Hierarchy (V3).** Propagate the seed through self-attention at 16×16, then 32×32, then 64×64. The 32-scale
   map is capped by the coarse 16-scale map, a one-sided cap at that step only. Averaged over the 11 calls.
4. **DenseCRF R2, then threshold** (0.55 by default) → binary mask.

"Fine-only" (V0) is the simpler alternative to step 3: two passes of 64×64 self-attention only.

Benchmarks:
- **VOC-Sim:** 490 generated images with masks, 20 VOC classes, one object class per image; the development set.
- **COCO-Cap:** 707 images generated from COCO captions with masks, cluttered scenes; normally used as untuned
  transfer.

Metric: class-macro mIoU of the binary foreground mask. Paired image bootstrap, 95% intervals, 2000 resamples.
All arms in a run come from one live generation per image, so comparisons are paired. Ground truth is used only
for scoring.

Reference values for BRACE on this corrected-data setup: 0.832–0.833 on VOC-Sim and 0.708–0.709 on COCO-Cap,
with DenseCRF at 0.55.

## 2. The idea: absent-class calibration

**Motivation.** The class word's cross-attention also fires on pixels that are not the object (for example,
"person" picks up background regions). Earlier work in this project found that the hierarchy is most hurt by
seeds with spurious mass far from the object.

**Profiles.** These are built per class c and per cross-attention module, from a separate statistics pass (a
second generation of the same prompts). The query is call-averaged, head-merged and L2-normalised per pixel. The
class word's attention is computed with keys from the class-word-only prompt.

- **Spurious profile m_s(c):** the mean query at the **top 1%** pixels of class word c's attention, over images
  where the target class ≠ c **and** the prompt or caption does not mention c (fixed synonym list). This is
  where the word fires when the class is certainly absent.
- **Object profile m_o(c):** the mean query at the **top 5%** pixels of c's attention, over images whose target
  = c.

**Gate at test time**, per module, for the query class:
- ŵ = (m_o − m_s)/|m_o − m_s|;
- p_o = ŵ·m_o, p_s = ŵ·m_s;
- g(i) = σ(4·(ŵ·q̂_i − (p_o + p_s)/2)/(p_o − p_s)), from the image's own call-averaged query q̂_i.

The class word's attention at that module is multiplied by g in every call, before the layer weighting. After
that, the BRACE pipeline is unchanged.

The constants (1%, 5%, gain 4) were fixed before run 28 and were never tuned. Inference stays label-free: it uses
only the query class and the image's own prompt or caption.

## 3. Run 28: COCO-Cap screen (exploratory)

**Design.** At the owner's request, three training-free ideas were screened on COCO-Cap only, with VOC-Sim
skipped. This breaks COCO-Cap's "untuned transfer" role for these ideas, and that is disclosed. The run was
pre-registered and committed before capture. Profiles were built from the same 707 COCO-Cap images, which makes
them **transductive**; the object profile left the test image out.

Arms, all through the BRACE hierarchy:
- A: BRACE.
- N1: absent-class-calibrated seed.
- X1: SDXL cross-attention seed. SDXL reads the SD1.5 image, noised to t = 100/300/500, caption-conditioned.
- X2: geometric mean of the BRACE seed and the SDXL seed.
- K1: class competition inside the hierarchy. Other VOC classes named in the caption are propagated together,
  and each map is multiplied by its share at every scale.
- K0: competition on the final maps only.

**Results, COCO-Cap, DenseCRF:**

| Arm | Seed @0.55 | Seed own best (thr) | Hierarchy @0.55 | Hierarchy own best (thr) | Hierarchy @0.55, competitor images (208) |
|---|---:|---:|---:|---:|---:|
| A BRACE | 0.527 | 0.629 (0.400) | 0.708 | 0.719 (0.625) | 0.688 |
| X1 SDXL seed | 0.343 | 0.444 (0.425) | 0.497 | 0.497 (0.525) | 0.467 |
| X2 √(BRACE·SDXL) | 0.559 | 0.619 (0.450) | 0.684 | 0.702 (0.650) | 0.675 |
| **N1 absent-class calibrated** | 0.511 | 0.668 (0.350) | **0.731** | **0.740 (0.625)** | 0.705 |
| K1 competition inside | 0.527 | 0.629 (0.400) | 0.663 | 0.666 (0.575) | 0.569 |
| K0 competition on final | 0.527 | 0.629 (0.400) | 0.641 | 0.663 (0.625) | 0.528 |

"Own best" is in-sample.

**Raw (no DenseCRF), hierarchy @0.55:** A 0.693, X1 0.489, X2 0.668, N1 0.717, K1 0.650, K0 0.629.

**Registered comparisons (hierarchy + DenseCRF @0.55):**

| ID | Comparison | Images | Δ [95% CI] | Promising? |
|---|---|---|---|---|
| T1 | X2 − A | 707 | −0.025 [−0.037, −0.013] | no |
| **T2** | **N1 − A** | 707 | **+0.022 [+0.011, +0.035]** | **yes** |
| T3 | K1 − A | 707 | −0.046 [−0.062, −0.028] | no |
| T3b | K1 − A | 208 with a competitor | −0.119 [−0.174, −0.085] | no |
| T4 | K1 − K0 | 208 | +0.041 [+0.006, +0.067] | yes (both far below A) |
| — | X1 − A | 707 | −0.211 [−0.236, −0.181] | no |

N1's seed alone at 0.55 was −0.016 [−0.025, −0.005] vs A. Its gain appeared only after the hierarchy.

**Per class, N1 vs A** (hierarchy + CRF @0.55):
- Up on 15 of 20 classes: person 0.28→0.40, aeroplane 0.80→0.86, horse 0.72→0.78, motorbike 0.71→0.76,
  bottle 0.57→0.63, chair 0.48→0.53, sofa 0.70→0.75.
- Down: potted plant 0.74→0.69, train 0.75→0.70, tv 0.79→0.71.

Checks: the recomputed seed matched the runtime seed to within 6.2e-4; mean gate 0.594; no module lacked a
profile.

**Verdict:** N1 was "promising", which licenses a VOC-Sim development check only. The SDXL seed and class
competition failed and are closed.

## 4. Run 29: component study on VOC-Sim and COCO-Cap

**Purpose.** Test whether the run 28 gain holds up as a component:
- on the development set;
- without transduction;
- with a control for the absent-class part;
- with and without propagation.

The run was pre-registered and committed before capture (`runs/29_absent_class_study/PREREG.md`). COCO-Cap
numbers are labelled **"screened"** because run 28 already looked at the idea there.

**Arms** (seed variants). Folds are crc32(image_id) mod 5, the same folds used for out-of-fold thresholds.

| Arm | Seed | Profiles from |
|---|---|---|
| A | BRACE seed | — |
| **NC** (primary) | calibrated | same benchmark, **other four folds only** (cross-fitted, not transductive) |
| NO (control) | gate with m_s replaced by the **mean query over all pixels** of the profile images (no absent-class set) | other folds |
| NL | calibrated, run 28 form | same benchmark, all images, own image left out of m_o (transductive) |
| NX | calibrated | the **other benchmark** (VOC-Sim → COCO-Cap; COCO-Cap → VOC-Sim) |

Each seed is propagated three ways: none, fine-only (V0) and hierarchy (V3).

Threshold rules:
- VOC-Sim: 5-fold out-of-fold (primary).
- COCO-Cap: each map's best threshold on full VOC-Sim, applied unchanged (primary).
- Both: fixed 0.55 (secondary).

**Pre-registered reading rule:**
- Development passes if P1's interval is above zero.
- The absent-class explanation is supported only if P4 (NC − NO) is above zero on VOC-Sim.
- The component goes to the owner only if both P1 and P2 are above zero.

**Checks:** recomputed vs runtime seed max |Δ| 5.3e-4 (VOC) and 6.2e-4 (COCO). No module lacked a profile.

| Mean gate | NC | NO | NL | NX |
|---|---:|---:|---:|---:|
| VOC-Sim | 0.607 | 0.399 | 0.609 | 0.648 |
| COCO-Cap | 0.589 | 0.336 | 0.594 | 0.468 |

### 4.1 VOC-Sim, DenseCRF

| Seed | Map | OOF | @0.55 | Own best (thr) |
|---|---|---:|---:|---:|
| A | seed | 0.736 | 0.628 | 0.736 (0.400) |
| NC | seed | 0.753 | 0.554 | 0.753 (0.300) |
| NO | seed | 0.757 | 0.530 | 0.757 (0.300) |
| NL | seed | 0.753 | 0.554 | 0.753 (0.300) |
| NX | seed | 0.752 | 0.590 | 0.752 (0.325) |
| A | fine-only | 0.825 | 0.796 | 0.827 (0.425) |
| NC | fine-only | 0.831 | 0.775 | 0.833 (0.400) |
| NO | fine-only | 0.831 | 0.772 | 0.831 (0.400) |
| NL | fine-only | 0.831 | 0.775 | 0.833 (0.400) |
| NX | fine-only | 0.840 | 0.790 | 0.840 (0.425) |
| **A** | **hierarchy** | **0.832** | 0.833 | 0.833 (0.550) |
| **NC** | **hierarchy** | **0.832** | 0.831 | 0.832 (0.525) |
| NO | hierarchy | 0.831 | 0.830 | 0.831 (0.525) |
| NL | hierarchy | 0.830 | 0.830 | 0.831 (0.525) |
| NX | hierarchy | 0.838 | 0.837 | 0.838 (0.525) |

### 4.2 COCO-Cap (screened), DenseCRF

| Seed | Map | VOC-Sim-selected (thr) | @0.55 | Own best (thr) |
|---|---|---:|---:|---:|
| A | seed | 0.629 (0.400) | 0.527 | 0.629 (0.400) |
| NC | seed | 0.659 (0.300) | 0.510 | 0.670 (0.350) |
| NO | seed | 0.660 (0.300) | 0.464 | 0.663 (0.325) |
| NL | seed | 0.658 (0.300) | 0.511 | 0.668 (0.350) |
| NX | seed | 0.657 (0.325) | 0.443 | 0.657 (0.325) |
| A | fine-only | 0.673 (0.425) | 0.698 | 0.702 (0.525) |
| NC | fine-only | 0.708 (0.400) | 0.724 | 0.732 (0.500) |
| NO | fine-only | 0.697 (0.400) | 0.704 | 0.716 (0.500) |
| NL | fine-only | 0.707 (0.400) | 0.720 | 0.730 (0.500) |
| NX | fine-only | 0.715 (0.425) | 0.710 | 0.725 (0.475) |
| **A** | **hierarchy** | **0.708 (0.550)** | 0.708 | 0.719 (0.600) |
| **NC** | **hierarchy** | **0.722 (0.525)** | 0.730 | 0.740 (0.600) |
| NO | hierarchy | 0.719 (0.525) | 0.725 | 0.730 (0.600) |
| NL | hierarchy | 0.723 (0.525) | 0.731 | 0.740 (0.625) |
| NX | hierarchy | 0.721 (0.525) | 0.728 | 0.733 (0.575) |

### 4.3 Raw (no DenseCRF), hierarchy map

| Seed | VOC-Sim OOF | VOC-Sim @0.55 | COCO-Cap VOC-Sim-selected | COCO-Cap @0.55 |
|---|---:|---:|---:|---:|
| A | 0.818 | 0.819 | 0.692 | 0.692 |
| NC | 0.816 | 0.818 | 0.717 | 0.717 |
| NO | 0.817 | 0.817 | 0.712 | 0.712 |
| NL | 0.817 | 0.818 | 0.717 | 0.717 |
| NX | 0.824 | 0.824 | 0.715 | 0.715 |

**Raw seed and fine-only maps:**

| Seed | VOC-Sim OOF, seed | VOC-Sim OOF, fine-only | COCO-Cap VOC-Sim-selected, seed | COCO-Cap VOC-Sim-selected, fine-only |
|---|---:|---:|---:|---:|
| A | 0.706 | 0.811 | 0.591 | 0.670 |
| NC | 0.724 | 0.817 | 0.620 | 0.704 |
| NO | 0.726 | 0.816 | 0.620 | 0.680 |
| NL | 0.724 | 0.816 | 0.619 | 0.703 |
| NX | 0.722 | 0.826 | 0.617 | 0.701 |

### 4.4 Registered comparisons (DenseCRF)

| ID | Benchmark | Comparison | Threshold | Δ [95% CI] | Above zero? |
|---|---|---|---|---|---|
| **P1 (primary)** | VOC-Sim | NC − A, hierarchy | OOF | **+0.000 [−0.005, +0.006]** | **no** |
| P1b | VOC-Sim | NC − A, hierarchy | 0.55 | −0.002 [−0.007, +0.003] | no |
| P2 | COCO-Cap | NX − A, hierarchy | VOC-Sim-selected | +0.013 [+0.001, +0.025] | yes |
| P3 | COCO-Cap | NC − A, hierarchy | VOC-Sim-selected | +0.014 [+0.003, +0.026] | yes |
| **P4** | VOC-Sim | NC − NO, hierarchy | OOF | **+0.001 [−0.001, +0.005]** | **no** |
| P4 | COCO-Cap | NC − NO, hierarchy | VOC-Sim-selected | +0.004 [−0.004, +0.012] | no |
| P5 | VOC-Sim | NC − A, seed | OOF | +0.016 [+0.010, +0.023] | yes |
| P5 | VOC-Sim | NC − A, fine-only | OOF | +0.007 [+0.001, +0.012] | yes |
| P5 | VOC-Sim | hierarchy − fine-only, A | OOF | +0.007 [+0.001, +0.013] | yes |
| P5 | VOC-Sim | hierarchy − fine-only, NC | OOF | +0.001 [−0.006, +0.007] | no |
| P5 | COCO-Cap | NC − A, seed | VOC-Sim-selected | +0.030 [+0.018, +0.042] | yes |
| P5 | COCO-Cap | NC − A, fine-only | VOC-Sim-selected | +0.036 [+0.023, +0.047] | yes |
| P5 | COCO-Cap | hierarchy − fine-only, A | VOC-Sim-selected | +0.035 [+0.024, +0.046] | yes |
| P5 | COCO-Cap | hierarchy − fine-only, NC | VOC-Sim-selected | +0.014 [+0.004, +0.025] | yes |
| P6 | VOC-Sim | NL − NC, hierarchy | OOF | −0.002 [−0.004, −0.001] | no |
| P6 | COCO-Cap | NL − NC, hierarchy | VOC-Sim-selected | +0.001 [−0.002, +0.004] | no |
| — | COCO-Cap | NC − A, hierarchy (run 28 convention) | 0.55 | +0.022 [+0.010, +0.034] | yes |
| — | COCO-Cap | NL − A, hierarchy (run 28 convention) | 0.55 | +0.022 [+0.011, +0.035] | yes |
| — | COCO-Cap | NX − A, hierarchy | 0.55 | +0.020 [+0.007, +0.035] | yes |

P5 and P6 are descriptive; P1 is the only decision test.

### 4.5 Per class, hierarchy + DenseCRF

**VOC-Sim, OOF thresholds:**

| Class | A | NC | NO | NL | NX |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.88 | 0.89 | 0.89 | 0.89 | 0.89 |
| bicycle | 0.80 | 0.78 | 0.79 | 0.78 | 0.79 |
| bird | 0.92 | 0.92 | 0.92 | 0.92 | 0.93 |
| boat | 0.85 | 0.86 | 0.86 | 0.86 | 0.85 |
| bottle | 0.87 | 0.90 | 0.91 | 0.91 | 0.91 |
| bus | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 |
| car | 0.85 | 0.86 | 0.86 | 0.85 | 0.86 |
| cat | 0.85 | 0.85 | 0.85 | 0.85 | 0.87 |
| chair | 0.79 | 0.78 | 0.78 | 0.78 | 0.78 |
| cow | 0.90 | 0.90 | 0.87 | 0.89 | 0.94 |
| dining table | 0.43 | 0.44 | 0.45 | 0.44 | 0.44 |
| dog | 0.93 | 0.93 | 0.93 | 0.93 | 0.92 |
| horse | 0.90 | 0.91 | 0.90 | 0.91 | 0.93 |
| motorbike | 0.87 | 0.87 | 0.87 | 0.87 | 0.87 |
| person | 0.81 | 0.82 | 0.83 | 0.82 | 0.84 |
| potted plant | 0.75 | 0.80 | 0.77 | 0.80 | 0.81 |
| sheep | 0.76 | 0.71 | 0.69 | 0.71 | 0.77 |
| sofa | 0.96 | 0.95 | 0.95 | 0.95 | 0.95 |
| train | 0.74 | 0.71 | 0.75 | 0.70 | 0.70 |
| tv | 0.85 | 0.82 | 0.81 | 0.82 | 0.80 |

**COCO-Cap, VOC-Sim-selected thresholds:**

| Class | A | NC | NO | NL | NX |
|---|---:|---:|---:|---:|---:|
| aeroplane | 0.80 | 0.85 | 0.85 | 0.85 | 0.84 |
| bicycle | 0.63 | 0.65 | 0.66 | 0.65 | 0.64 |
| bird | 0.78 | 0.78 | 0.78 | 0.78 | 0.78 |
| boat | 0.59 | 0.61 | 0.61 | 0.60 | 0.64 |
| bottle | 0.57 | 0.60 | 0.58 | 0.59 | 0.63 |
| bus | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 |
| car | 0.76 | 0.80 | 0.73 | 0.79 | 0.76 |
| cat | 0.92 | 0.92 | 0.92 | 0.92 | 0.91 |
| chair | 0.47 | 0.52 | 0.50 | 0.52 | 0.50 |
| cow | 0.87 | 0.90 | 0.87 | 0.90 | 0.86 |
| dining table | 0.47 | 0.54 | 0.58 | 0.55 | 0.55 |
| dog | 0.81 | 0.83 | 0.82 | 0.83 | 0.83 |
| horse | 0.72 | 0.77 | 0.75 | 0.77 | 0.77 |
| motorbike | 0.71 | 0.75 | 0.74 | 0.75 | 0.75 |
| person | 0.28 | 0.39 | 0.35 | 0.39 | 0.44 |
| potted plant | 0.74 | 0.66 | 0.67 | 0.68 | 0.69 |
| sheep | 0.87 | 0.87 | 0.86 | 0.87 | 0.89 |
| sofa | 0.70 | 0.74 | 0.74 | 0.73 | 0.71 |
| train | 0.75 | 0.67 | 0.71 | 0.67 | 0.65 |
| tv | 0.79 | 0.69 | 0.73 | 0.69 | 0.68 |

## 5. Verdict under the pre-registered rule

- **Development (P1): fail.** On VOC-Sim, calibration adds nothing after the hierarchy (+0.000).
- **Absent-class explanation (P4): not supported.** A gate that contrasts the object profile with the all-pixel
  mean, using no absent-class images, performs about the same on both benchmarks. The useful part is a generic
  "object query profile" gate, not absent-class calibration.
- **Bring to owner (P1 and P2 above zero): no.** Not adopted; BRACE stays frozen.

## 6. Observations (exploratory, post hoc; not claims)

1. **The gate helps before propagation, everywhere.** Seed: +0.016 on VOC-Sim and +0.030 on COCO-Cap. Fine-only:
   +0.007 and +0.036. On VOC-Sim the hierarchy already captures the same improvement, so the gain disappears
   after it.
2. **The gate and the hierarchy partly fix the same problem.** With the gate, the hierarchy's advantage over
   fine-only drops from +0.007 to +0.001 on VOC-Sim and from +0.035 to +0.014 on COCO-Cap. On COCO-Cap, gated
   fine-only (0.708) equals ungated BRACE (0.708).
3. **The COCO-Cap gain is smaller under the proper threshold convention.** It is +0.014 at the VOC-Sim-selected
   threshold, vs +0.022 at 0.55 (run 28). It survives with VOC-Sim-only profiles (NX +0.013). COCO-Cap was
   screened, so this stays exploratory.
4. **Transduction was not the source of the gain:** NL ≈ NC on both benchmarks.
5. **Gains are uneven across classes.**
   - COCO-Cap gains: person (0.28 → 0.39–0.44), aeroplane, horse, chair, dining table.
   - COCO-Cap losses: train, tv, potted plant (−0.05 to −0.10).
   - VOC-Sim is flat, with losses on sheep, train and tv.
6. **The gate dims the map,** so the best seed threshold falls to 0.30–0.35. That is why the seed looks worse at
   a fixed 0.55 and better at its own threshold.

## 7. Caveats

- COCO-Cap was used for screening in run 28, so none of its numbers are untuned transfer.
- Only one setting of the constants (1%, 5%, gain 4) was ever tested.
- Absent classes are found from captions with a hand-written synonym list.
- Results are from one SD1.5 generation per image on V100 and A100 SLURM nodes. Cross-GPU regenerations are
  not pixel-identical, so these numbers should not be paired with other runs' captures.

## Files

- Run 28: `scripts/jc_trio.py`, `runs/28_trio_screen/{PREREG.md, result.md}` (commits c90a7abf0, 620dce425).
- Run 29: `scripts/jc_absent.py`, `runs/29_absent_class_study/{PREREG.md, result.md}` (commits 57e45e0eb,
  03ac54e45).
