# Runs 33 and 34: handoff for review (2026-11-22)

Both runs are **exploratory**. They were pre-registered and committed before any capture or scoring
(run 33: 6a634ab64; run 34: 3b2dc70ff); the results are in dd6e92d0c. BRACE stays frozen:
- seed: cross-attention with SA-guided layer weights (A3);
- propagation: capped 16→32→64 self-attention hierarchy (V3);
- refinement: DenseCRF R2, threshold 0.55.

Inference is label-free; ground truth is used only for scoring. The full reports follow in the appendix,
unchanged.

---

## 0. Background (what came before)

### The absent-class gate (run 31)
For each class c and each cross-attention module, the gate compares the pixel's query q̂ with two profiles:
- **q̂:** the head-merged, L2-normalised, call-averaged CA query at the pixel.
- **Object profile:** mean q̂ at the top 5% of t̄_c, on images of class c. t̄_c is the class-word cross-attention.
- **Absent profiles:** q̂ at the top 1% of t̄_c on images whose prompt does **not** mention c. This is where the
  class word "leaks" when the object isn't there. These are clustered by spherical k-means.
- **Calibration:** g = σ(4·(w·q̂ − (w·o + w·n)/2)/(w·o − w·n)), with w = normalise(o − n).
- **MK4:** 4 absent centroids; each pixel uses the nearest one.
- **NC:** a single absent profile.
- **Use:** the per-module CA map for class c is multiplied by g before the seed is built.

Run 31 (generated images; test halves, VOC-Sim / COCO-Cap mIoU):

| Arm | VOC-Sim / COCO-Cap |
|---|---|
| BRACE | 0.843 / 0.718 |
| Regime X MK4 | 0.855 / 0.762 |

- **Regime V:** profiles from VOC-Sim dev only, so COCO-Cap is transfer.
- **Regime X:** profiles from VOC-Sim dev + COCO-Cap dev. This is label-free target adaptation.

### Run 32 (real PASCAL VOC 2012 val; single SD1.5 pass on the photo at t = 100)

| Component | Effect |
|---|---|
| SA-guided layer weights | +0.027 |
| Uncapped hierarchy | ≈ 0 |
| Gate, profiles from VOC labels (A) | +0.022 |
| Gate, profiles from generated VOC-Sim images with captions (B) | only +0.002 |

The open question was why B was so weak: synthetic images, the caption-based "absent" rule, or fewer fit images?

### JC #5 (earlier)
On generated images, a single pass at t = 100 costs about −0.03 / −0.04 against generation-time capture:
0.806 / 0.666 vs 0.832 / 0.709.

---

## 1. Run 33: noise levels and profile sources on real VOC photos

### Design
- **Items:** 2,170 VOC train items (thresholds and profiles) and 2,148 val items (1,449 photos).
- **Prompt:** `A photograph of a {w}`.
- **Capture:** one VAE encode and one shared noise draw. Three conditional UNet calls at **t = 100, 300, 500**.
- **Arms:** t100, t300 and t500 alone. "Multi" averages the three calls the way generation-time BRACE does:
  - the per-call weighted CA sums are averaged, then min-max;
  - propagation runs per call, then the results are averaged.
- **Profiles:** refit at each t. In multi, each call uses the profiles of its own t.
- **Factorial components:**
  - W: SA layer weights vs uniform;
  - H: uncapped hierarchy vs fine-only;
  - G: MK4 gate.

  BRACE (capped hierarchy, no gate) was also run.

**Profile sources:**

| Source | Fit images | "Absent" means | Labels? | Size |
|---|---|---|---|---|
| A | VOC train (cross-fitted on train) | not in image labels | yes | 2,170 |
| S | A cut to B's per-class counts | as A | yes | 490 |
| B | generated VOC-Sim images | prompt doesn't mention c | no | 490 |
| C | **real COCO train2017 photos**, counts matched to B | none of the 5 COCO captions mention c | no | 490 |

**What each contrast isolates:**
- **C − B:** real vs generated images.
- **S − C:** labels vs captions on real photos at equal size. It still differs in VOC vs COCO photos.
- **A − S:** fit-set size.

**Scoring:**
- class-macro mIoU over items;
- threshold grid 0.30–0.70, chosen on train;
- primary rule: val, + DenseCRF, train-selected threshold;
- 2,000-sample paired photo bootstrap.

### Key results

| # | Contrast | Δ [95% CI] | Reading |
|---|---|---|---|
| P1 | multi − t100, full stack (A) | −0.028 [−0.039, −0.017] | hurts |
| P2 | multi − t100, BRACE | −0.024 [−0.035, −0.014] | hurts |
| P3 | hierarchy, multi, A | −0.006 [−0.013, −0.001] | hurts |
| P4 | C − B | **+0.023 [+0.014, +0.032]** | helps |
| P5 | S − C | −0.003 [−0.009, +0.003] | not resolved |
| P6 | A − S | +0.004 [−0.000, +0.007] | not resolved |

**Interpretation**
1. **Several noise levels don't help on real photos.**
   - t=100 is best for every map; t=500 alone loses about 0.08.
   - The gate's own effect grows with more noise levels (+0.036 multi vs +0.022 at t100), but not enough to
     offset the lower base.
2. **The hierarchy still does not help on real photos**, at any noise level.
3. **The weak label-free gate in run 32 was due to synthetic fit images.**
   - Real COCO photos with caption-based absence (C, no labels) reach 0.645.
   - VOC labels reach 0.646 (A), or 0.642 at equal size (S).
   - The caption rule and the fit-set size don't matter much.
4. **Person shows the effect most.**

   | Map | Person IoU |
   |---|---:|
   | All off | 0.26 |
   | Source C | 0.53 |
   | Source A | 0.52 |
   | Source B | 0.33 |

   The gate hurts bottle (0.51 → 0.42–0.45) and a little on bird/sheep/bus under A, as in run 32.
5. **The t100 arm reproduces run 32** (0.596 / 0.610 / 0.646 / 0.623). That comparison is descriptive, since it is a
   different capture.

Caveat: val was already seen in run 32. Everything here was fixed beforehand, with thresholds chosen on train,
but it is a second look.

---

## 2. Run 34: better object-vs-leakage discrimination in the gate (steps 0–3 of the agreed plan)

### Design
- **Setting:** identical to run 31. VOC-Sim (490) and COCO-Cap (707), generation-time capture, run 31's
  class-stratified dev/test split, cross-fitting between dev folds, regimes V and X.

**Gates:**

| Gate | Object side | Absent side | Direction w |
|---|---|---|---|
| NC | 1 mean | 1 mean | normalise(o − n) |
| **MK4** (baseline) | 1 mean | 4 centroids | normalise(o − n) |
| MP2 / MP4 | 2 / 4 centroids of the object pool | 4 centroids | normalise(o − n), nearest object × nearest absent |
| FD4 | 1 mean | 4 centroids | normalise((o − n)/s), diagonal Fisher |
| FP2 / FP4 | 2 / 4 centroids | 4 centroids | Fisher |

The Fisher scale is s = 0.5(v_o + v_n) + 0.5·mean(v_o + v_n), fixed shrinkage, built from the per-dimension
variances of the cluster members.

**Stage 1 (CPU, run 31 dev dumps; scored images always held out from their profiles)**
- **Regions:** object, near background (≤ 6 px at 64), far background.
- **Retention** = Σw·g^γ / Σw, for γ ∈ {0.25 … 4}.
  - Weights: pixel or **mass** (the A3 seed; primary).
  - Averaged over modules → images → classes.
- **Matched operating range:**
  - far suppression from MK4's γ = 0.5 to γ = 2, at 5 points;
  - ΔR = mean object retention gain vs MK4 at matched suppression;
  - ΔS = the converse.
- **Pass rule:** mass-weighted far background; ΔR ≥ +0.01 with CI > 0, on **both** VOC-Sim dev (V) and COCO-Cap
  dev (X). FP is read only if both its parts pass.

**Step 0, Q–K diagnostic.** The profiles are refit on queries with part of each head's query removed:
- **⊥t:** the projection onto the class-word key is removed;
- **⊥P:** the projection onto the span of all non-padding prompt keys is removed.

Reading: the gate uses information beyond prompt-key logits if (AUROC⊥P − 0.5)/(AUROC − 0.5) ≥ 0.5 and the CI is
above 0.5.

**Stage 2 (GPU):**
- one new generation-time capture of all 1,197 images: arm A (no gate) plus {X, V} × 7 gates;
- maps: V3 + DenseCRF, test halves, thresholds chosen on dev;
- "improves on MK4" = COCO CI lower bound > 0 and VOC CI lower bound ≥ −0.005;
- read only for gates that passed stage 1.

### Key results

**Stage 1: no gate passes.**

| Gate | ΔR vs MK4, VOC-Sim dev (V) | ΔR vs MK4, COCO-Cap dev (X) | AUROC obj vs far (V / X) |
|---|---|---|---|
| MK4 | 0 | 0 | 0.790 / 0.752 |
| MP2 | −0.010 [−0.014, −0.007] | −0.017 [−0.020, −0.008] | 0.779 / 0.735 |
| MP4 | −0.010 [−0.014, −0.005] | −0.024 [−0.029, −0.020] | 0.776 / 0.740 |
| FD4 | +0.002 [+0.001, +0.002] | +0.000 [+0.000, +0.001] | 0.792 / 0.753 |
| FP2 | −0.011 | −0.018 | 0.779 / 0.735 |
| FP4 | −0.009 | −0.026 | 0.778 / 0.740 |

- **Several object centroids** keep more object at γ = 1, but they keep proportionally more background. Their
  discrimination (AUROC) is lower.
- **Diagonal Fisher** is a negligible change.
- **Q–K ratio** is about 0.95–1.05 in every setting, so this is "supported". But the removed prompt-key span is only
  3–12% of each head's rank, so the test is weak.
- **Removing only the class-word key (⊥t):** MK4 gets slightly worse (ΔR −0.007 to −0.015), while NC gets slightly
  better.
- **NaN entries:** ΔR shows `nan` for NC because NC's far-suppression curve does not reach MK4's matched range. Under
  the rule, a point outside a gate's curve counts as a failure. NC is not a candidate.

**Stage 2 (descriptive, since nothing passed stage 1).** Test halves, dev-selected thresholds, VOC-Sim / COCO-Cap:

| Arm | Regime X | Regime V |
|---|---|---|
| A (no gate) | 0.841 / 0.725 | 0.840 / 0.708 |
| NC | 0.847 / 0.749 | 0.841 / 0.720 |
| **MK4** | **0.854 / 0.771** | 0.844 / 0.731 |
| MP2 | 0.839 / 0.750 | 0.845 / 0.730 |
| MP4 | 0.839 / 0.761 | 0.837 / 0.726 |
| FD4 | 0.854 / 0.770 | 0.844 / 0.728 |
| FP2 | 0.844 / 0.748 | 0.845 / 0.727 |
| FP4 | 0.840 / 0.760 | 0.838 / 0.723 |

The MK4 contrasts replicate run 31 in a new capture:

| Contrast | Regime X | Regime V |
|---|---|---|
| MK4 − A | +0.013 / +0.046 (both CIs > 0) | +0.004 / +0.023 (COCO CI > 0) |
| MK4 − NC | +0.007 / +0.022 | (not resolved) |

- MP and FP lose mainly on **person** (regime X: 0.50 → 0.29–0.33), and on car, train and cow.
- MP4 helps potted plant, and in regime V helps dining table and person. None of this is read.

---

## 3. Overall conclusions and open questions

**Established (exploratory):**
- MK4 is still the best gate.
- Several object profiles hurt.
- Fisher weighting does nothing.
- Multi-timestep reading hurts on real photos.
- On real photos, **label-free profiles from real COCO photos with captions match label-based profiles.**

**Not established:**
- Whether C-style profiles help on generated benchmarks. They were only tested on real VOC.
- Why the hierarchy fails on real photos.
- Whether the gate's discrimination comes from beyond prompt keys (the test was weak).

**Not yet authorised:**
- step 4: negatives taken from pipeline failures;
- step 5: per-image adaptation of the profiles.

**Possible next steps to discuss:**
- real-photo caption profiles (source C) in the generated-image regime;
- a bigger C fit set;
- changes on the absent side only, since every object-side change failed.

---

# Appendix: full reports (unchanged)

---

## Appendix file: runs/33_realvoc_timesteps/PREREG.md

## Run 33: noise levels and profile sources on real PASCAL VOC photos (pre-registration)

Committed before any capture or scoring of this run. BRACE stays frozen. **Exploratory.** This follows run 32
(`runs/32_realvoc_factorial/`) and uses the same items, prompts, layer map, components and scoring.

**Second look at VOC val.** Run 32 already reported VOC 2012 val. Nothing in run 33 is chosen on val:
- the contrasts below are fixed now;
- every threshold is chosen on VOC train.

Run 33 val numbers are a second look at the same photos, and are labelled that way.

### 1. Questions

1. **Noise level.** Does reading attention at several noise levels beat the single pass at t = 100? With several
   noise levels, does the hierarchy start to help, as it does in generation-time BRACE?
2. **Profile source.** Run 32's label-free profiles (B: generated VOC-Sim images, absent = the prompt doesn't
   mention c) helped much less than profiles from VOC labels (A). Is that because:
   - B's images are generated;
   - "absent" is decided from captions;
   - or B has fewer fit images?

### 2. What is already known (no new run needed)

JC #5 (`comparisons/backbones/result.md`) captured the same stored VOC-Sim images in one pass at t = 100. With
SD1.5, BRACE, DenseCRF and 0.55 it scores **0.806 / 0.666** (VOC-Sim / COCO-Cap). Generation-time BRACE scores
0.832 / 0.709 (runs/20).

So on generated images, a single pass costs about −0.03 / −0.04, against a gap of about −0.2 between VOC-Sim
and real VOC. Different settings, descriptive only.

### 3. Capture (one capture for every arm)

Items, prompts, layer map and recorder are as in run 32:
- the 2,170 VOC train items and 2,148 val items;
- the prompt `A photograph of a {w}`;
- `scripts/jc_realvoc.py` functions, imported unchanged.

Per item:
- **VAE:** one encode.
- **Noise:** one noise draw ε (CPU generator, seed 0), **shared by all noise levels**.
- **UNet:** three conditional SD1.5 calls, at **t = 100, 300 and 500**.

| Arm | What is read |
|---|---|
| `t100` | the t = 100 call alone (the run 32 setting, re-captured here; not mixed with run 32) |
| `t300` | the t = 300 call alone |
| `t500` | the t = 500 call alone |
| `M` (multi) | all three calls, aggregated as generation-time BRACE does (`sg_combined_v2.aggregate`) |

How arm `M` aggregates:
- **Seed:** the per-call weighted CA sums (each call with its own A3 weights and its own gate) are averaged over
  the 3 calls, then min-max.
- **Propagation:** the seed is propagated through each call's SA (fine-only, uncapped hierarchy, capped
  hierarchy). The 3 results are averaged before min-max.

### 4. Components and profile sources

The components are as in run 32:
- **W:** A3 vs uniform layer weights.
- **H:** uncapped hierarchy V1 vs fine-only V0.
- **G:** run 31 MK4 gate, K = 4.

**Profiles are fitted separately at each noise level**, from one pass of the fit images at that t (same noise
draw). In arm `M`, each call's gate uses the profiles of its own t.

| Source | Fit images | "Absent" means | Labels? | Size |
|---|---|---|---|---|
| **A** | VOC train items (cross-fitted by fold on train; all of train for val) | not in the image-level labels | image labels | 2,170 items |
| **S** | A cut to B's size: per class, as many items as B has, seeded by crc32(item id) | as A | image labels | 490 |
| **B** | the 490 VOC-Sim generated images, runs/20 prompts | the prompt doesn't mention c | none | 490 |
| **C** | **real COCO train2017 photos**, 490 photos, per-class counts matched to B; prompt `A photograph of a {w}` | none of the photo's 5 COCO captions mention c | none (captions only) | 490 |

Selection rules:
- **C:** a photo qualifies for class c when at least one caption mentions c (`jc_trio.mentions`, the run 31/32
  synonym list). Classes are filled rarest first. Photos are drawn without replacement, seeded by
  `random.Random(33)`. COCO train2017 shares no photos with VOC.
- **S, on train:** S takes the per-class counts within the other fold (capped at what that fold has). On val, it
  takes them from all of train.

How the comparisons separate the three explanations:
- **C − B:** real vs generated images, both caption-based and both 490. They still differ in prompt and in the
  caption rule (5 captions vs the generation prompt).
- **S − C:** labels vs captions on real photos at equal size. They still differ in VOC vs COCO photos.
- **A − S:** size alone (2,170 vs 490 items, same rule).

### 5. Maps, thresholds, scoring

Per arm:
- **Seeds:** u-, w-, uA4, wA4, uB4, wB4, uC4, wC4, wS4.
- **Maps:** each seed as seed / V0 / V1, plus w-|V3 (BRACE).
- **DenseCRF R2:** the factorial maps for A, B and C, plus w-|V3 and wS4|V1. That is 18 maps per arm.

Thresholds and scoring:
- **Thresholds:** grid 0.30–0.70 in steps of 0.025, chosen per map on the train items (raw maps). On val,
  DenseCRF is applied at that threshold and at a fixed 0.55.
- **Metric:** class-macro mIoU over items.
- **Intervals:** 95% from a 2000-sample photo bootstrap. The same resamples are used for every map, so
  differences are paired.

### 6. Primary contrasts and reading rule (fixed now)

The primary rule is **val, + DenseCRF, train-selected threshold**:
- CI above 0 = **helps**;
- CI below 0 = **hurts**;
- otherwise **not resolved**.

| # | Contrast | Question |
|---|---|---|
| P1 | `M:wA4\|V1` − `t100:wA4\|V1` (full stack, source A) | Do several noise levels help? |
| P2 | `M:w-\|V3` − `t100:w-\|V3` (BRACE, one capture) | The same, for BRACE |
| P3 | Average hierarchy effect (run 32 definition) in arm `M`, source A | Does the hierarchy help with several noise levels? |
| P4 | `t100:wC4\|V1` − `t100:wB4\|V1` | Real vs generated fit images (captions both) |
| P5 | `t100:wS4\|V1` − `t100:wC4\|V1` | Labels vs captions, real photos, equal size |
| P6 | `t100:wA4\|V1` − `t100:wS4\|V1` | Number of fit images |

Secondary, also reported:
- P4–P6 in arm `M`;
- t300 and t500 against t100;
- the average W, H and G effects in every arm (sources A, B, C);
- the raw and 0.55 rules;
- per-class IoU (person especially);
- a descriptive comparison of `t100` with run 32 (a different capture, so not a paired test).

### 7. Checks (smoke, before the full run)

- At t = 100, the single-call V0 / V1 / V3 match `sg_combined_v2.aggregate` fed that call 11 times:
  max |Δ| < 1e-5 after min-max.
- Arm `M` with three identical calls equals the single-call maps: max |Δ| < 1e-5.
- Every map is finite.
- C photos: every item's class is mentioned in one of its captions, and no absent class is.
- S per-class counts equal B's (or are capped as stated).

### 8. Files

- **Script:** `scripts/jc_realvoc_t.py` (imports `scripts/jc_realvoc.py`).
- **Report:** `runs/33_realvoc_timesteps/result.md`.
- **Data:** `<BRACE_CACHE_DIR>/jc_realvoc_t/`.
- Smoke outputs are tagged `_smoke` and kept.

---

## Appendix file: runs/33_realvoc_timesteps/result.md

## Run 33: noise levels and profile sources on real PASCAL VOC photos (`PREREG.md`)

**Exploratory; second look at VOC val** (run 32 reported it first). VOC 2012 val: 1449 photos, 2148 items. Thresholds chosen on VOC train (2170 items, raw maps). One capture: three SD1.5 calls per item at t = 100, 300, 500 with one shared noise draw; arms read one call or all three. Class-macro mIoU over items; 95% photo-bootstrap intervals (2000 samples, paired).


### 1. Primary contrasts (val, + DenseCRF, train-selected threshold)

| # | Contrast | Δ mIoU [95% CI] | Reading |
|---|---|---|---|
| P1 | multi − t100, full stack (source A) | -0.028 [-0.039, -0.017] | **hurts** |
| P2 | multi − t100, BRACE (capped hierarchy, no gate) | -0.024 [-0.035, -0.014] | **hurts** |
| P3 | hierarchy, average effect, multi, source A | -0.006 [-0.013, -0.001] | **hurts** |
| P4 | C − B: real vs generated fit images (both captions), t = 100, full stack | +0.023 [+0.014, +0.032] | **helps** |
| P5 | S − C: labels vs captions (real, 490 each), t = 100, full stack | -0.003 [-0.009, +0.003] | not resolved |
| P6 | A − S: 2,170 vs 490 fit items (labels), t = 100, full stack | +0.004 [-0.000, +0.007] | not resolved |

### 2. Noise levels

| Map | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| all off (uniform, fine-only, no gate) | 0.596 | 0.578 | 0.512 | 0.573 |
| BRACE (capped hierarchy) | 0.608 | 0.603 | 0.526 | 0.584 |
| SA weights + hierarchy | 0.610 | 0.604 | 0.526 | 0.586 |
| full stack, source A | 0.646 | 0.631 | 0.549 | 0.618 |
| full stack, source S | 0.642 | 0.629 | 0.547 | 0.613 |
| full stack, source B | 0.623 | 0.618 | 0.543 | 0.601 |
| full stack, source C | 0.645 | 0.632 | 0.544 | 0.616 |

Δ vs t = 100 (primary rule):

| Map | t = 300 | t = 500 | multi |
|---|---|---|---|
| BRACE | -0.005 [-0.016, +0.004] | -0.082 [-0.097, -0.068] | -0.024 [-0.035, -0.014] |
| full stack, A | -0.014 [-0.026, -0.003] | -0.096 [-0.112, -0.082] | -0.028 [-0.039, -0.017] |
| full stack, C | -0.014 [-0.026, -0.001] | -0.101 [-0.116, -0.087] | -0.030 [-0.041, -0.019] |
| all off | -0.018 [-0.029, -0.006] | -0.083 [-0.100, -0.069] | -0.022 [-0.034, -0.012] |

Average effect of each component per noise level (primary rule):

| Component | Source | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---|---|---|---|---|
| SA-guided layer weights | A | +0.027 [+0.023, +0.031] | +0.022 [+0.018, +0.025] | +0.018 [+0.014, +0.022] | +0.018 [+0.014, +0.021] |
| hierarchy (no cap) | A | +0.002 [-0.005, +0.009] | +0.003 [-0.004, +0.009] | -0.006 [-0.013, -0.000] | -0.006 [-0.013, -0.001] |
| 4 absent-class profiles | A | +0.022 [+0.013, +0.032] | +0.029 [+0.019, +0.038] | +0.028 [+0.021, +0.035] | +0.036 [+0.027, +0.044] |
| 4 absent-class profiles | B | +0.002 [-0.008, +0.012] | +0.013 [+0.005, +0.021] | +0.015 [+0.008, +0.021] | +0.015 [+0.008, +0.023] |
| 4 absent-class profiles | C | +0.022 [+0.012, +0.032] | +0.025 [+0.016, +0.034] | +0.021 [+0.014, +0.027] | +0.030 [+0.022, +0.037] |

### 3. Profile sources

Full stack (SA weights + hierarchy + 4 profiles), primary rule:

| Source | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| no gate | 0.610 | 0.604 | 0.526 | 0.586 |
| A: VOC train labels | 0.646 | 0.631 | 0.549 | 0.618 |
| S: VOC train labels, cut to 490 | 0.642 | 0.629 | 0.547 | 0.613 |
| C: real COCO photos, captions | 0.645 | 0.632 | 0.544 | 0.616 |
| B: VOC-Sim generated, captions | 0.623 | 0.618 | 0.543 | 0.601 |

Paired source contrasts, full stack, multi (secondary):

| Contrast | Δ [95% CI] | Reading |
|---|---|---|
| C − B | +0.014 [+0.007, +0.021] | **helps** |
| S − C | -0.003 [-0.009, +0.003] | not resolved |
| A − S | +0.005 [+0.002, +0.008] | **helps** |

#### Factorial, t = 100, source A: VOC train labels

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.602 | 0.600 | 0.610 | 0.611 | 0.525 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.616 | 0.605 | 0.625 | 0.608 | 0.625 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.642 | 0.633 | 0.650 | 0.645 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.646 | 0.639 | 0.652 | 0.642 | 0.600 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.023 [+0.016, +0.029] | +0.029 [+0.023, +0.036] | +0.027 [+0.023, +0.031] |
| hierarchy (no cap) | -0.002 [-0.011, +0.007] | +0.004 [-0.005, +0.012] | +0.002 [-0.005, +0.009] |
| 4 absent-class profiles | +0.006 [-0.006, +0.019] | +0.036 [+0.026, +0.046] | +0.022 [+0.013, +0.032] |

#### Factorial, t = 100, source B: VOC-Sim generated, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.589 | 0.581 | 0.593 | 0.590 | 0.500 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.593 | 0.589 | 0.601 | 0.592 | 0.600 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.622 | 0.608 | 0.628 | 0.619 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.623 | 0.618 | 0.630 | 0.622 | 0.600 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.023 [+0.016, +0.029] | +0.030 [+0.022, +0.036] | +0.025 [+0.021, +0.029] |
| hierarchy (no cap) | -0.002 [-0.011, +0.007] | +0.000 [-0.008, +0.009] | -0.002 [-0.009, +0.006] |
| 4 absent-class profiles | -0.006 [-0.020, +0.006] | +0.013 [+0.004, +0.022] | +0.002 [-0.008, +0.012] |

#### Factorial, t = 100, source C: real COCO photos, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.596 | 0.593 | 0.599 | 0.599 | 0.525 |
|  |  | ✓ | 0.603 | 0.602 | 0.609 | 0.610 | 0.525 |
|  | ✓ |  | 0.594 | 0.582 | 0.601 | 0.583 | 0.625 |
|  | ✓ | ✓ | 0.619 | 0.606 | 0.627 | 0.607 | 0.625 |
| ✓ |  |  | 0.619 | 0.615 | 0.624 | 0.623 | 0.525 |
| ✓ |  | ✓ | 0.636 | 0.627 | 0.642 | 0.639 | 0.500 |
| ✓ | ✓ |  | 0.610 | 0.600 | 0.616 | 0.600 | 0.625 |
| ✓ | ✓ | ✓ | 0.645 | 0.637 | 0.652 | 0.639 | 0.600 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.023 [+0.016, +0.029] | +0.026 [+0.020, +0.032] | +0.024 [+0.020, +0.029] |
| hierarchy (no cap) | -0.002 [-0.011, +0.007] | +0.009 [-0.000, +0.018] | +0.004 [-0.004, +0.011] |
| 4 absent-class profiles | +0.008 [-0.006, +0.021] | +0.036 [+0.026, +0.045] | +0.022 [+0.012, +0.032] |

#### Factorial, multi (100+300+500), source A: VOC train labels

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.573 | 0.571 | 0.578 | 0.574 | 0.575 |
|  |  | ✓ | 0.608 | 0.606 | 0.617 | 0.613 | 0.575 |
|  | ✓ |  | 0.570 | 0.548 | 0.578 | 0.549 | 0.675 |
|  | ✓ | ✓ | 0.605 | 0.586 | 0.613 | 0.586 | 0.650 |
| ✓ |  |  | 0.592 | 0.591 | 0.598 | 0.595 | 0.575 |
| ✓ |  | ✓ | 0.632 | 0.631 | 0.637 | 0.638 | 0.525 |
| ✓ | ✓ |  | 0.586 | 0.562 | 0.592 | 0.562 | 0.650 |
| ✓ | ✓ | ✓ | 0.618 | 0.607 | 0.626 | 0.607 | 0.650 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.018 [+0.013, +0.024] | +0.013 [+0.008, +0.017] | +0.018 [+0.014, +0.021] |
| hierarchy (no cap) | -0.003 [-0.012, +0.005] | -0.014 [-0.021, -0.007] | -0.006 [-0.013, -0.001] |
| 4 absent-class profiles | +0.035 [+0.024, +0.046] | +0.032 [+0.023, +0.040] | +0.036 [+0.027, +0.044] |

#### Factorial, multi (100+300+500), source B: VOC-Sim generated, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.573 | 0.571 | 0.578 | 0.574 | 0.575 |
|  |  | ✓ | 0.582 | 0.582 | 0.590 | 0.590 | 0.550 |
|  | ✓ |  | 0.570 | 0.548 | 0.578 | 0.549 | 0.675 |
|  | ✓ | ✓ | 0.585 | 0.573 | 0.594 | 0.574 | 0.650 |
| ✓ |  |  | 0.592 | 0.591 | 0.598 | 0.595 | 0.575 |
| ✓ |  | ✓ | 0.613 | 0.610 | 0.619 | 0.618 | 0.525 |
| ✓ | ✓ |  | 0.586 | 0.562 | 0.592 | 0.562 | 0.650 |
| ✓ | ✓ | ✓ | 0.601 | 0.595 | 0.610 | 0.596 | 0.650 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.018 [+0.013, +0.024] | +0.016 [+0.011, +0.022] | +0.020 [+0.016, +0.024] |
| hierarchy (no cap) | -0.003 [-0.012, +0.005] | -0.012 [-0.019, -0.004] | -0.004 [-0.011, +0.002] |
| 4 absent-class profiles | +0.009 [-0.001, +0.020] | +0.015 [+0.007, +0.023] | +0.015 [+0.008, +0.023] |

#### Factorial, multi (100+300+500), source C: real COCO photos, captions

| SA weights | Hierarchy | 4 profiles | + DenseCRF, train-selected (primary) | + DenseCRF, 0.55 | raw, train-selected | raw, 0.55 | train thr |
|:-:|:-:|:-:|---:|---:|---:|---:|---:|
|  |  |  | 0.573 | 0.571 | 0.578 | 0.574 | 0.575 |
|  |  | ✓ | 0.604 | 0.603 | 0.612 | 0.609 | 0.575 |
|  | ✓ |  | 0.570 | 0.548 | 0.578 | 0.549 | 0.675 |
|  | ✓ | ✓ | 0.597 | 0.580 | 0.607 | 0.580 | 0.675 |
| ✓ |  |  | 0.592 | 0.591 | 0.598 | 0.595 | 0.575 |
| ✓ |  | ✓ | 0.623 | 0.623 | 0.630 | 0.630 | 0.550 |
| ✓ | ✓ |  | 0.586 | 0.562 | 0.592 | 0.562 | 0.650 |
| ✓ | ✓ | ✓ | 0.616 | 0.599 | 0.624 | 0.599 | 0.650 |

| Component | alone | in the stack | average |
|---|---|---|---|
| SA-guided layer weights | +0.018 [+0.013, +0.024] | +0.019 [+0.014, +0.023] | +0.018 [+0.014, +0.022] |
| hierarchy (no cap) | -0.003 [-0.012, +0.005] | -0.008 [-0.015, -0.000] | -0.006 [-0.012, +0.000] |
| 4 absent-class profiles | +0.031 [+0.020, +0.041] | +0.029 [+0.022, +0.037] | +0.030 [+0.022, +0.037] |

### 4. Descriptive

Seed only (raw, train-selected):

| Seed | t = 100 | t = 300 | t = 500 | multi (100+300+500) |
|---|---:|---:|---:|---:|
| uniform | 0.539 | 0.516 | 0.485 | 0.530 |
| SA weights | 0.558 | 0.540 | 0.502 | 0.546 |
| SA weights + gate A | 0.575 | 0.577 | 0.538 | 0.586 |
| SA weights + gate C | 0.566 | 0.566 | 0.525 | 0.574 |

Per class IoU, primary rule, full stack per source:

| Class | items | all off, t100 | A, t100 | S, t100 | C, t100 | B, t100 | A, multi | C, multi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aeroplane | 90 | 0.761 | 0.838 | 0.831 | 0.843 | 0.828 | 0.777 | 0.775 |
| bicycle | 79 | 0.308 | 0.349 | 0.341 | 0.334 | 0.342 | 0.298 | 0.288 |
| bird | 103 | 0.751 | 0.741 | 0.728 | 0.732 | 0.705 | 0.708 | 0.696 |
| boat | 72 | 0.644 | 0.732 | 0.728 | 0.728 | 0.726 | 0.679 | 0.664 |
| bottle | 96 | 0.505 | 0.419 | 0.445 | 0.435 | 0.380 | 0.427 | 0.423 |
| bus | 74 | 0.820 | 0.813 | 0.813 | 0.824 | 0.800 | 0.811 | 0.823 |
| car | 127 | 0.451 | 0.530 | 0.512 | 0.525 | 0.493 | 0.478 | 0.478 |
| cat | 119 | 0.818 | 0.843 | 0.849 | 0.852 | 0.842 | 0.819 | 0.816 |
| chair | 123 | 0.292 | 0.304 | 0.302 | 0.320 | 0.270 | 0.267 | 0.268 |
| cow | 71 | 0.855 | 0.877 | 0.876 | 0.876 | 0.841 | 0.839 | 0.839 |
| dining table | 75 | 0.415 | 0.498 | 0.485 | 0.448 | 0.522 | 0.534 | 0.513 |
| dog | 128 | 0.785 | 0.828 | 0.833 | 0.846 | 0.859 | 0.768 | 0.776 |
| horse | 79 | 0.770 | 0.830 | 0.828 | 0.830 | 0.804 | 0.763 | 0.759 |
| motorbike | 76 | 0.718 | 0.756 | 0.749 | 0.754 | 0.752 | 0.725 | 0.725 |
| person | 446 | 0.263 | 0.516 | 0.489 | 0.525 | 0.331 | 0.490 | 0.490 |
| potted plant | 85 | 0.288 | 0.324 | 0.296 | 0.288 | 0.318 | 0.312 | 0.324 |
| sheep | 57 | 0.844 | 0.834 | 0.853 | 0.858 | 0.768 | 0.771 | 0.741 |
| sofa | 90 | 0.538 | 0.709 | 0.699 | 0.710 | 0.724 | 0.670 | 0.676 |
| train | 84 | 0.706 | 0.720 | 0.740 | 0.736 | 0.703 | 0.772 | 0.781 |
| tv | 74 | 0.380 | 0.453 | 0.444 | 0.445 | 0.441 | 0.453 | 0.457 |

Run 32 (different capture, descriptive only, primary rule): all off 0.596, SA weights + hierarchy 0.610, full stack A 0.646, full stack B 0.623. Compare with the t = 100 column above.

Fit set sizes (items): {'A_all': 2170, 'A_f0': 1095, 'A_f1': 1075, 'S_all': 490, 'S_f0': 490, 'S_f1': 490, 'B_all': 490, 'C_all': 490}. Profiles with no absent-class pool fall back to g = 1.


---

## Appendix file: runs/34_gate_discrim/PREREG.md

## Run 34: better object-vs-leakage discrimination in the absent-class gate (pre-registration)

Committed before any fit, capture or scoring of this run. BRACE stays frozen. **Exploratory.** This covers steps
0–3 of the plan agreed on 2026-11-22 (Claude + ChatGPT review). It builds on run 31 (`runs/31_multineg_gate/`).

### 1. Setting (identical to run 31)

- **Benchmarks:** VOC-Sim (490) and COCO-Cap (707), generation-time capture, the runs/20 prompts.
- **Split:** run 31's `split.json` (class-stratified dev / test halves; dev folds dA / dB), unchanged.
- **Regimes:**
  - **V:** profiles from VOC-Sim dev only. COCO-Cap is transfer.
  - **X:** profiles from VOC-Sim dev + COCO-Cap dev. Label-free target-domain adaptation, not transfer.
- **Cross-fitting:** a dev image is gated with profiles from the other dev fold. Test images use profiles fitted on
  all dev images of the regime. In regime V, every COCO-Cap image uses all-dev VOC profiles.
- **Pools:** exactly as run 31 (`jc_mneg.fit`). All of these are label-free:
  - **object pool:** q̂ at the top 5% of t̄_c on images of class c;
  - **absent pool:** q̂ at the top 1% of t̄_c on images whose prompt does not mention c;
  - **clustering:** `jc_mneg.cluster` (spherical k-means, plain-mean centroids, small clusters dropped, K = 1 fallback).

### 2. Gates compared

All gates use the run 31 calibration: for a pair (object vector o, negative vector n) and a direction w,
g = σ(4·(w·q̂ − (w·o + w·n)/2) / (w·o − w·n)). Each pixel uses the pair of its nearest object centroid and its
nearest absent centroid (by cosine).

| Gate | Object side | Absent side | Direction w | Role |
|---|---|---|---|---|
| NC | 1 mean | 1 mean | normalise(o − n) | run 31 single profile |
| **MK4** | 1 mean | 4 centroids | normalise(o − n) | **primary baseline** (run 31) |
| MP2, MP4 | K_obj = 2 / 4 centroids of the object pool | 4 centroids | normalise(o − n) | step 1: several object profiles |
| FD4 | 1 mean | 4 centroids | normalise((o − n) / s) | step 2: diagonal Fisher |
| FP2, FP4 | 2 / 4 centroids | 4 centroids | normalise((o − n) / s) | step 3: combination |

How the Fisher scale s is built:
- s = 0.5·(v_o + v_n) + 0.5·mean(v_o + v_n);
- v_o and v_n are the per-dimension variances of the members of that object and absent cluster;
- the shrinkage weight 0.5 is fixed, not tuned.

### 3. Stage 1: pixel-level screen (CPU, run 31 dev dumps; no new capture)

Data: run 31 Part B dev dumps (`<CACHE>/jc_mneg/dump_{voc,coco}/`). These hold per-pixel call-averaged CA queries,
t̄, the A3 seed and the 64 × 64 object mask: 253 VOC-Sim and 356 COCO-Cap dev images. The scored image is always
held out from the profiles (cross-fold).

**Regions** at each module's grid (as run 31 C1): object, near background (within 6 px at 64), far background.

**Weights**, both reported:
- **pixel-weighted:** every pixel counts 1;
- **mass-weighted:** each pixel is weighted by the A3 seed (generation-time CA), area-resized to the module grid.
  The gate multiplies CA mass, so this is the primary weighting.

**Curves:** the gate is applied as g^γ, with γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4}. For each region,
retained = Σ w·g^γ / Σ w. Averaging order:
1. average over the 11 modules per image;
2. then over images of a class;
3. then over classes (class-balanced).

Bootstrap: 2000 samples over images.

**Matched operating range (fixed now):** far-background suppression (1 − retained far) from MK4's value at
γ = 0.5 to its value at γ = 2, at 5 evenly spaced points. At each point, every gate's object retention is linearly
interpolated on its own curve. A point outside a gate's curve counts as a failure.
- **ΔR** = mean object retention (candidate − MK4) over the 5 points.
- The converse is also reported: ΔS = mean far suppression at matched object retention, over MK4's γ = 0.5–2
  object-retention range.

**Pass rule (fixed now):** mass-weighted, far background. ΔR ≥ +0.01 **and** its 95% CI above 0, on **both** held-out
sets:
- VOC-Sim dev under regime V;
- COCO-Cap dev under regime X.

|ΔR| < 0.01 is called negligible. **FP2 / FP4 are read only if MP2 / MP4 (same K) and FD4 both pass.**

Also reported, not used for the rule:
- near background;
- pixel-weighted curves;
- regime V on COCO-Cap dev (transfer);
- regime X on VOC-Sim dev;
- pixel AUROC of g between object and far background.

#### Step 0: Q–K residual diagnostic (stage 1 data)

NC and MK4 are refitted on queries with part of each head's query removed (same pixel selection, since t̄ is
unchanged). Per CA module and head:

- **⊥t:** the projection onto the class-word key of the image's attribution prompt is removed (mean over word
  pieces).
- **⊥P:** the projection onto the span of all non-padding token keys of that prompt (BOS through EOS) is removed. The
  rank fraction of that span per head is reported. Padding keys are not removed, a stated limitation.

Then q̂ is re-normalised. Reported per setting:
- AUROC (object vs far) and ΔR vs the unprojected gate;
- the ratio (AUROC⊥P − 0.5) / (AUROC − 0.5).

Reading: "the gate uses information beyond the prompt-key logits" if that ratio is ≥ 0.5 **and** AUROC⊥P's CI is
above 0.5. Diagnostic only, never an algorithm.

### 4. Stage 2: full pipeline (GPU capture; same overnight chain)

One generation-time capture of all 1,197 images, as run 31 Part D. Arms:
- A (no gate);
- {X, V} × {NC, MK4, MP2, MP4, FD4, FP2, FP4}.

Maps: seed, fine-only, BRACE hierarchy (V3); DenseCRF R2 on V3 at the full threshold grid.

All arms are computed in the one capture so the chain needs no wait. But a candidate's stage 2 contrast **is read
only if it passed stage 1** (FP only if both its parts passed). Arms that failed stage 1 are reported as descriptive
only.

- **Primary rule (as run 31):** test halves, V3 + DenseCRF, thresholds chosen on dev (regime V: VOC-Sim dev;
  regime X: mean of both dev curves).
- **Contrast:** candidate − MK4, paired image bootstrap (2000 samples).
- **"Improves on MK4":** COCO-Cap CI lower bound > 0 **and** VOC-Sim CI lower bound ≥ −0.005 (run 31's rule), per
  regime.
- **Also reported:** fixed 0.55, candidate − A, MK4 − NC, per class.

MK4 and NC are re-captured here, so run 31 numbers are not mixed in.

### 5. Checks (smoke, before the full chain)

- MP with one object centroid and no Fisher equals `jc_mneg.g_multi` (MK4): max |Δ| < 1e-5.
- Seed fidelity of arm A < 1e-2 (as run 31).
- ⊥t / ⊥P residuals are orthogonal to the removed keys: max |cos| < 1e-4.
- Every map is finite.

### 6. Files and cost

- **Script:** `scripts/jc_gate34.py`.
- **Reports:**
  - `runs/34_gate_discrim/stage1.md` and `stage1.json`;
  - `runs/34_gate_discrim/result.md`.
- **Data:** `<CACHE>/jc_gate34/`.

| Part | Cost estimate |
|---|---|
| fit | 1–2 CPU-hours |
| stage 1 | 2–4 CPU-hours |
| capture | about 7 GPU-hours (4 shards, as run 31) |
| DenseCRF scoring | about 5 CPU-hours over 6 jobs |

Smoke outputs are tagged `_smoke` and kept.

---

## Appendix file: runs/34_gate_discrim/stage1.md

## Run 34, stage 1: object vs leakage discrimination of the gate (`PREREG.md`)

**Exploratory.** Run 31 dev dumps; every image is gated with profiles that never saw it (other dev fold; regime V on COCO-Cap uses all VOC dev). Retention = Σ w·g^γ / Σ w per region, mean over modules, then images, then classes. 95% CIs from 2000 image-bootstrap samples. Mass weight = A3 seed at the module grid.


### regime V, VOC-Sim dev (pass rule)

| Gate | weight | object kept, γ=1 | near bg kept | far bg kept | AUROC obj vs far | ΔR vs MK4 [95% CI] | ΔS vs MK4 [95% CI] |
|---|---|---:|---:|---:|---:|---|---|
| NC | mass | 0.748 | 0.581 | 0.461 | 0.790 [0.780, 0.800] | +nan [+nan, +nan] | +0.021 [+0.015, +0.026] |
| NC | pix | 0.740 | 0.543 | 0.446 | 0.790 [0.780, 0.800] | +nan [+nan, +nan] | +0.019 [+0.014, +0.024] |
| MK4 | mass | 0.661 | 0.513 | 0.397 | 0.790 [0.780, 0.799] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MK4 | pix | 0.655 | 0.479 | 0.385 | 0.790 [0.780, 0.799] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MP2 | mass | 0.688 | 0.545 | 0.437 | 0.779 [0.769, 0.787] | -0.010 [-0.014, -0.007] | -0.010 [-0.014, -0.006] |
| MP2 | pix | 0.682 | 0.513 | 0.423 | 0.779 [0.769, 0.787] | -0.011 [-0.014, -0.007] | -0.010 [-0.014, -0.006] |
| MP4 | mass | 0.706 | 0.564 | 0.456 | 0.776 [0.766, 0.784] | -0.010 [-0.014, -0.005] | -0.008 [-0.013, -0.004] |
| MP4 | pix | 0.700 | 0.531 | 0.444 | 0.776 [0.766, 0.784] | -0.011 [-0.016, -0.007] | -0.010 [-0.014, -0.006] |
| FD4 | mass | 0.668 | 0.520 | 0.403 | 0.792 [0.782, 0.801] | +0.002 [+0.001, +0.002] | +0.002 [+0.001, +0.002] |
| FD4 | pix | 0.662 | 0.486 | 0.391 | 0.792 [0.782, 0.801] | +0.002 [+0.001, +0.002] | +0.002 [+0.001, +0.002] |
| FP2 | mass | 0.692 | 0.550 | 0.442 | 0.779 [0.770, 0.787] | -0.011 [-0.014, -0.007] | -0.010 [-0.014, -0.006] |
| FP2 | pix | 0.686 | 0.518 | 0.429 | 0.779 [0.770, 0.787] | -0.011 [-0.015, -0.007] | -0.010 [-0.014, -0.006] |
| FP4 | mass | 0.710 | 0.568 | 0.460 | 0.778 [0.768, 0.786] | -0.009 [-0.013, -0.005] | -0.007 [-0.011, -0.002] |
| FP4 | pix | 0.704 | 0.535 | 0.448 | 0.778 [0.768, 0.786] | -0.011 [-0.015, -0.007] | -0.008 [-0.013, -0.004] |
| NC ⊥t | mass | 0.774 | 0.628 | 0.514 | 0.793 [0.784, 0.802] | +nan [+0.005, +0.006] | +0.006 [+0.000, +0.010] |
| NC ⊥t | pix | 0.767 | 0.593 | 0.499 | 0.793 [0.784, 0.802] | +nan [+0.004, +0.004] | +0.004 [-0.001, +0.008] |
| MK4 ⊥t | mass | 0.721 | 0.595 | 0.484 | 0.783 [0.773, 0.791] | -0.014 [-0.017, -0.011] | -0.013 [-0.017, -0.010] |
| MK4 ⊥t | pix | 0.716 | 0.562 | 0.472 | 0.783 [0.773, 0.791] | -0.014 [-0.017, -0.012] | -0.014 [-0.017, -0.011] |
| NC ⊥P | mass | 0.769 | 0.619 | 0.498 | 0.791 [0.780, 0.800] | +nan [+0.008, +0.011] | +0.012 [+0.007, +0.017] |
| NC ⊥P | pix | 0.761 | 0.583 | 0.483 | 0.791 [0.780, 0.800] | +nan [+0.008, +0.009] | +0.011 [+0.006, +0.016] |
| MK4 ⊥P | mass | 0.712 | 0.580 | 0.462 | 0.797 [0.787, 0.806] | -0.004 [-0.006, -0.001] | -0.003 [-0.005, -0.000] |
| MK4 ⊥P | pix | 0.707 | 0.546 | 0.450 | 0.797 [0.787, 0.806] | -0.004 [-0.006, -0.001] | -0.003 [-0.005, -0.000] |

Q–K diagnostic, NC: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 1.00; information beyond the prompt-key logits: **supported**.

Q–K diagnostic, MK4: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 1.02; information beyond the prompt-key logits: **supported**.

### regime X, COCO-Cap dev (pass rule)

| Gate | weight | object kept, γ=1 | near bg kept | far bg kept | AUROC obj vs far | ΔR vs MK4 [95% CI] | ΔS vs MK4 [95% CI] |
|---|---|---:|---:|---:|---:|---|---|
| NC | mass | 0.739 | 0.572 | 0.462 | 0.761 [0.752, 0.772] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| NC | pix | 0.731 | 0.537 | 0.451 | 0.761 [0.752, 0.772] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| MK4 | mass | 0.625 | 0.484 | 0.380 | 0.752 [0.745, 0.765] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MK4 | pix | 0.620 | 0.452 | 0.372 | 0.752 [0.745, 0.765] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MP2 | mass | 0.663 | 0.527 | 0.438 | 0.735 [0.729, 0.755] | -0.017 [-0.020, -0.008] | -0.016 [-0.019, -0.007] |
| MP2 | pix | 0.657 | 0.496 | 0.429 | 0.735 [0.729, 0.755] | -0.018 [-0.022, -0.009] | -0.017 [-0.020, -0.008] |
| MP4 | mass | 0.686 | 0.557 | 0.470 | 0.740 [0.733, 0.750] | -0.024 [-0.029, -0.020] | -0.022 [-0.026, -0.018] |
| MP4 | pix | 0.680 | 0.528 | 0.462 | 0.740 [0.733, 0.750] | -0.027 [-0.032, -0.023] | -0.024 [-0.028, -0.019] |
| FD4 | mass | 0.631 | 0.490 | 0.387 | 0.753 [0.746, 0.766] | +0.000 [+0.000, +0.001] | +0.001 [+0.000, +0.001] |
| FD4 | pix | 0.626 | 0.458 | 0.379 | 0.753 [0.746, 0.766] | +0.000 [+0.000, +0.001] | +0.001 [+0.000, +0.001] |
| FP2 | mass | 0.667 | 0.532 | 0.443 | 0.735 [0.729, 0.754] | -0.018 [-0.021, -0.009] | -0.017 [-0.020, -0.008] |
| FP2 | pix | 0.662 | 0.502 | 0.435 | 0.735 [0.729, 0.754] | -0.019 [-0.023, -0.011] | -0.018 [-0.021, -0.009] |
| FP4 | mass | 0.688 | 0.561 | 0.474 | 0.740 [0.733, 0.750] | -0.026 [-0.031, -0.022] | -0.023 [-0.028, -0.019] |
| FP4 | pix | 0.683 | 0.532 | 0.466 | 0.740 [0.733, 0.750] | -0.028 [-0.034, -0.024] | -0.025 [-0.029, -0.021] |
| NC ⊥t | mass | 0.765 | 0.619 | 0.519 | 0.766 [0.757, 0.775] | +nan [+nan, +nan] | +0.011 [+0.007, +0.017] |
| NC ⊥t | pix | 0.758 | 0.584 | 0.507 | 0.766 [0.757, 0.775] | +nan [+nan, +nan] | +0.009 [+0.005, +0.014] |
| MK4 ⊥t | mass | 0.688 | 0.565 | 0.467 | 0.751 [0.744, 0.760] | -0.012 [-0.013, -0.009] | -0.011 [-0.013, -0.008] |
| MK4 ⊥t | pix | 0.684 | 0.533 | 0.458 | 0.751 [0.744, 0.760] | -0.011 [-0.013, -0.008] | -0.011 [-0.013, -0.008] |
| NC ⊥P | mass | 0.746 | 0.598 | 0.491 | 0.751 [0.742, 0.764] | +nan [+nan, +nan] | +0.015 [+0.010, +0.023] |
| NC ⊥P | pix | 0.739 | 0.562 | 0.480 | 0.751 [0.742, 0.764] | +nan [+nan, +nan] | +0.013 [+0.009, +0.021] |
| MK4 ⊥P | mass | 0.652 | 0.524 | 0.421 | 0.756 [0.748, 0.770] | -0.005 [-0.007, -0.001] | -0.007 [-0.009, -0.002] |
| MK4 ⊥P | pix | 0.648 | 0.490 | 0.410 | 0.756 [0.748, 0.770] | -0.003 [-0.005, +0.001] | -0.004 [-0.006, +0.001] |

Q–K diagnostic, NC: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 0.96; information beyond the prompt-key logits: **supported**.

Q–K diagnostic, MK4: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 1.02; information beyond the prompt-key logits: **supported**.

### regime X, VOC-Sim dev (descriptive)

| Gate | weight | object kept, γ=1 | near bg kept | far bg kept | AUROC obj vs far | ΔR vs MK4 [95% CI] | ΔS vs MK4 [95% CI] |
|---|---|---:|---:|---:|---:|---|---|
| NC | mass | 0.823 | 0.653 | 0.520 | 0.775 [0.764, 0.785] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| NC | pix | 0.816 | 0.612 | 0.502 | 0.775 [0.764, 0.785] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| MK4 | mass | 0.697 | 0.545 | 0.405 | 0.779 [0.770, 0.788] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MK4 | pix | 0.692 | 0.505 | 0.391 | 0.779 [0.770, 0.788] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MP2 | mass | 0.730 | 0.581 | 0.459 | 0.766 [0.756, 0.775] | -0.017 [-0.022, -0.012] | -0.016 [-0.021, -0.010] |
| MP2 | pix | 0.724 | 0.544 | 0.444 | 0.766 [0.756, 0.775] | -0.019 [-0.024, -0.014] | -0.017 [-0.022, -0.012] |
| MP4 | mass | 0.750 | 0.611 | 0.500 | 0.755 [0.744, 0.764] | -0.036 [-0.043, -0.030] | -0.032 [-0.039, -0.026] |
| MP4 | pix | 0.744 | 0.577 | 0.487 | 0.755 [0.744, 0.764] | -0.041 [-0.047, -0.034] | -0.035 [-0.042, -0.030] |
| FD4 | mass | 0.703 | 0.552 | 0.412 | 0.781 [0.771, 0.790] | +0.001 [+0.001, +0.001] | +0.001 [+0.001, +0.002] |
| FD4 | pix | 0.699 | 0.512 | 0.398 | 0.781 [0.771, 0.790] | +0.001 [+0.001, +0.001] | +0.001 [+0.001, +0.002] |
| FP2 | mass | 0.734 | 0.587 | 0.464 | 0.766 [0.756, 0.775] | -0.018 [-0.023, -0.013] | -0.016 [-0.021, -0.011] |
| FP2 | pix | 0.729 | 0.550 | 0.450 | 0.766 [0.756, 0.775] | -0.020 [-0.025, -0.015] | -0.018 [-0.023, -0.013] |
| FP4 | mass | 0.753 | 0.615 | 0.504 | 0.755 [0.744, 0.764] | -0.038 [-0.044, -0.031] | -0.033 [-0.040, -0.027] |
| FP4 | pix | 0.747 | 0.581 | 0.491 | 0.755 [0.744, 0.764] | -0.042 [-0.047, -0.036] | -0.036 [-0.042, -0.030] |
| NC ⊥t | mass | 0.840 | 0.693 | 0.564 | 0.780 [0.770, 0.789] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| NC ⊥t | pix | 0.834 | 0.654 | 0.548 | 0.780 [0.770, 0.789] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| MK4 ⊥t | mass | 0.759 | 0.628 | 0.494 | 0.776 [0.767, 0.784] | -0.007 [-0.010, -0.005] | -0.006 [-0.010, -0.003] |
| MK4 ⊥t | pix | 0.755 | 0.589 | 0.480 | 0.776 [0.767, 0.784] | -0.008 [-0.010, -0.005] | -0.007 [-0.010, -0.003] |
| NC ⊥P | mass | 0.827 | 0.672 | 0.539 | 0.774 [0.764, 0.784] | +nan [+nan, +nan] | +nan [+nan, +nan] |
| NC ⊥P | pix | 0.820 | 0.633 | 0.522 | 0.774 [0.764, 0.784] | +nan [+nan, +nan] | +nan [+0.002, +0.009] |
| MK4 ⊥P | mass | 0.720 | 0.582 | 0.442 | 0.794 [0.785, 0.803] | -0.004 [-0.006, -0.001] | -0.005 [-0.008, -0.002] |
| MK4 ⊥P | pix | 0.716 | 0.542 | 0.427 | 0.794 [0.785, 0.803] | -0.003 [-0.006, -0.001] | -0.004 [-0.007, -0.001] |

Q–K diagnostic, NC: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 1.00; information beyond the prompt-key logits: **supported**.

Q–K diagnostic, MK4: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 1.05; information beyond the prompt-key logits: **supported**.

### regime V, COCO-Cap dev (descriptive)

| Gate | weight | object kept, γ=1 | near bg kept | far bg kept | AUROC obj vs far | ΔR vs MK4 [95% CI] | ΔS vs MK4 [95% CI] |
|---|---|---:|---:|---:|---:|---|---|
| NC | mass | 0.637 | 0.485 | 0.390 | 0.781 [0.772, 0.790] | +0.026 [+0.021, +0.031] | +0.030 [+0.022, +0.035] |
| NC | pix | 0.629 | 0.454 | 0.382 | 0.781 [0.772, 0.790] | +0.024 [+0.019, +0.030] | +0.028 [+0.020, +0.033] |
| MK4 | mass | 0.558 | 0.437 | 0.351 | 0.758 [0.751, 0.772] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MK4 | pix | 0.553 | 0.411 | 0.345 | 0.758 [0.751, 0.772] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| MP2 | mass | 0.597 | 0.469 | 0.392 | 0.757 [0.750, 0.775] | -0.002 [-0.004, +0.005] | -0.001 [-0.003, +0.006] |
| MP2 | pix | 0.591 | 0.444 | 0.385 | 0.757 [0.750, 0.775] | -0.003 [-0.005, +0.004] | -0.002 [-0.004, +0.005] |
| MP4 | mass | 0.619 | 0.494 | 0.417 | 0.762 [0.755, 0.778] | -0.004 [-0.007, +0.003] | -0.003 [-0.006, +0.004] |
| MP4 | pix | 0.613 | 0.467 | 0.410 | 0.762 [0.755, 0.778] | -0.006 [-0.009, +0.001] | -0.004 [-0.007, +0.003] |
| FD4 | mass | 0.564 | 0.443 | 0.356 | 0.761 [0.753, 0.774] | +0.001 [+0.001, +0.002] | +0.002 [+0.001, +0.002] |
| FD4 | pix | 0.559 | 0.417 | 0.350 | 0.761 [0.753, 0.774] | +0.001 [+0.001, +0.002] | +0.002 [+0.001, +0.002] |
| FP2 | mass | 0.601 | 0.474 | 0.396 | 0.758 [0.750, 0.775] | -0.002 [-0.004, +0.005] | -0.001 [-0.003, +0.006] |
| FP2 | pix | 0.595 | 0.448 | 0.389 | 0.758 [0.750, 0.775] | -0.003 [-0.005, +0.004] | -0.002 [-0.004, +0.005] |
| FP4 | mass | 0.621 | 0.497 | 0.419 | 0.763 [0.756, 0.778] | -0.004 [-0.007, +0.003] | -0.002 [-0.005, +0.004] |
| FP4 | pix | 0.615 | 0.470 | 0.412 | 0.763 [0.756, 0.778] | -0.005 [-0.009, +0.001] | -0.004 [-0.007, +0.003] |
| NC ⊥t | mass | 0.651 | 0.520 | 0.437 | 0.785 [0.775, 0.792] | +0.006 [+0.002, +0.011] | +0.009 [+0.005, +0.014] |
| NC ⊥t | pix | 0.644 | 0.491 | 0.428 | 0.785 [0.775, 0.792] | +0.005 [+0.002, +0.010] | +0.008 [+0.005, +0.013] |
| MK4 ⊥t | mass | 0.605 | 0.499 | 0.419 | 0.761 [0.752, 0.769] | -0.015 [-0.017, -0.013] | -0.015 [-0.017, -0.013] |
| MK4 ⊥t | pix | 0.601 | 0.474 | 0.412 | 0.761 [0.752, 0.769] | -0.014 [-0.017, -0.012] | -0.014 [-0.017, -0.012] |
| NC ⊥P | mass | 0.567 | 0.433 | 0.349 | 0.768 [0.760, 0.784] | +0.010 [+0.007, +0.019] | +0.011 [+0.008, +0.020] |
| NC ⊥P | pix | 0.560 | 0.403 | 0.340 | 0.768 [0.760, 0.784] | +0.010 [+0.007, +0.019] | +0.012 [+0.008, +0.021] |
| MK4 ⊥P | mass | 0.537 | 0.429 | 0.350 | 0.757 [0.749, 0.775] | -0.017 [-0.019, -0.011] | -0.019 [-0.021, -0.013] |
| MK4 ⊥P | pix | 0.532 | 0.402 | 0.342 | 0.757 [0.749, 0.775] | -0.015 [-0.017, -0.009] | -0.017 [-0.019, -0.011] |

Q–K diagnostic, NC: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 0.95; information beyond the prompt-key logits: **supported**.

Q–K diagnostic, MK4: (AUROC⊥P − 0.5)/(AUROC − 0.5) = 0.99; information beyond the prompt-key logits: **supported**.

### Pass rule (mass-weighted, far background; ΔR ≥ +0.01 and CI > 0 on VOC-Sim dev / V and COCO-Cap dev / X)

- MP2: does not pass
- MP4: does not pass
- FD4: does not pass
- FP2: does not pass (read only if its parts pass)
- FP4: does not pass (read only if its parts pass)

Rank fraction of the removed prompt-key span per head (mean over images): down0: 0.12, down1: 0.12, down2: 0.03, down3: 0.03, mid_4: 0.03, up_b5: 0.03, up_b6: 0.03, up_b7: 0.03, up_b8: 0.12, up_b9: 0.12, up_b10: 0.12


---

## Appendix file: runs/34_gate_discrim/result.md

## Run 34, stage 2: full pipeline (`PREREG.md`)

Test halves; thresholds chosen on dev. **Exploratory.** One generation-time capture for every arm. Regime V = VOC-Sim-only profiles (COCO-Cap is transfer); regime X = VOC-Sim + COCO-Cap dev profiles (label-free domain adaptation).

Test images: VOC-Sim 237, COCO-Cap 351. Stage 1 verdicts: MP2 fail, MP4 fail, FD4 fail, FP2 fail, FP4 fail. A contrast is read only for gates that passed stage 1.


### Regime X

| Arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---:|---:|---:|---:|
| A | 0.841 | 0.725 | 0.842 | 0.718 |
| X:NC | 0.847 | 0.749 | 0.849 | 0.742 |
| X:MK4 | 0.854 | 0.771 | 0.855 | 0.762 |
| X:MP2 | 0.839 | 0.750 | 0.845 | 0.745 |
| X:MP4 | 0.839 | 0.761 | 0.841 | 0.757 |
| X:FD4 | 0.854 | 0.770 | 0.854 | 0.761 |
| X:FP2 | 0.844 | 0.748 | 0.845 | 0.744 |
| X:FP4 | 0.840 | 0.760 | 0.841 | 0.755 |

Hierarchy + DenseCRF, dev-selected thresholds (0.55 in brackets column).

| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |
|---|---|---|---|---|---|---|
| MP2 − MK4 | fail | -0.015 [-0.023, -0.006] | -0.021 [-0.034, -0.008] | -0.010 [-0.017, -0.003] | -0.017 [-0.027, -0.008] | descriptive (failed stage 1) |
| MP4 − MK4 | fail | -0.015 [-0.023, -0.007] | -0.010 [-0.022, +0.003] | -0.014 [-0.022, -0.006] | -0.005 [-0.017, +0.006] | descriptive (failed stage 1) |
| FD4 − MK4 | fail | +0.000 [-0.000, +0.000] | -0.001 [-0.003, +0.000] | -0.000 [-0.001, +0.000] | -0.001 [-0.002, -0.000] | descriptive (failed stage 1) |
| FP2 − MK4 | fail | -0.010 [-0.017, -0.003] | -0.024 [-0.035, -0.012] | -0.009 [-0.016, -0.003] | -0.018 [-0.028, -0.009] | descriptive (failed stage 1) |
| FP4 − MK4 | fail | -0.014 [-0.022, -0.006] | -0.011 [-0.023, +0.001] | -0.014 [-0.022, -0.006] | -0.007 [-0.018, +0.003] | descriptive (failed stage 1) |
| MK4 − NC | — | +0.007 [+0.000, +0.013] | +0.022 [+0.006, +0.037] | +0.005 [-0.001, +0.012] | +0.019 [+0.004, +0.034] | — |
| MK4 − A | — | +0.013 [+0.004, +0.021] | +0.046 [+0.027, +0.065] | +0.012 [+0.004, +0.020] | +0.043 [+0.026, +0.062] | — |
| NC − A | — | +0.006 [-0.001, +0.012] | +0.024 [+0.004, +0.045] | +0.007 [-0.001, +0.013] | +0.024 [+0.006, +0.042] | — |

Per class, COCO-Cap test, dev-selected (A / MK4 / MP2 / MP4 / FD4 / FP2 / FP4): aeroplane 0.78 / 0.83 / 0.86 / 0.85 / 0.82 / 0.84 / 0.85, bicycle 0.66 / 0.72 / 0.73 / 0.70 / 0.72 / 0.72 / 0.70, bird 0.90 / 0.90 / 0.90 / 0.90 / 0.90 / 0.89 / 0.90, boat 0.59 / 0.65 / 0.66 / 0.64 / 0.65 / 0.65 / 0.64, bottle 0.72 / 0.91 / 0.88 / 0.91 / 0.92 / 0.89 / 0.91, bus 0.93 / 0.92 / 0.92 / 0.91 / 0.92 / 0.92 / 0.91, car 0.84 / 0.85 / 0.78 / 0.84 / 0.85 / 0.80 / 0.84, cat 0.92 / 0.92 / 0.91 / 0.92 / 0.92 / 0.92 / 0.92, chair 0.49 / 0.53 / 0.49 / 0.56 / 0.53 / 0.48 / 0.55, cow 0.85 / 0.89 / 0.84 / 0.83 / 0.89 / 0.87 / 0.83, dining table 0.46 / 0.54 / 0.54 / 0.54 / 0.54 / 0.55 / 0.54, dog 0.81 / 0.82 / 0.81 / 0.82 / 0.82 / 0.82 / 0.82, horse 0.76 / 0.81 / 0.80 / 0.82 / 0.81 / 0.80 / 0.82, motorbike 0.73 / 0.74 / 0.75 / 0.74 / 0.74 / 0.74 / 0.74, person 0.19 / 0.50 / 0.29 / 0.33 / 0.49 / 0.26 / 0.32, potted plant 0.71 / 0.76 / 0.79 / 0.85 / 0.76 / 0.76 / 0.84, sheep 0.88 / 0.84 / 0.82 / 0.81 / 0.84 / 0.83 / 0.81, sofa 0.70 / 0.79 / 0.78 / 0.79 / 0.79 / 0.77 / 0.79, train 0.77 / 0.70 / 0.67 / 0.68 / 0.70 / 0.66 / 0.68, tv 0.81 / 0.81 / 0.76 / 0.81 / 0.80 / 0.78 / 0.81


### Regime V

| Arm | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 |
|---|---:|---:|---:|---:|
| A | 0.840 | 0.708 | 0.842 | 0.718 |
| V:NC | 0.841 | 0.720 | 0.843 | 0.737 |
| V:MK4 | 0.844 | 0.731 | 0.845 | 0.744 |
| V:MP2 | 0.845 | 0.730 | 0.846 | 0.741 |
| V:MP4 | 0.837 | 0.726 | 0.843 | 0.748 |
| V:FD4 | 0.844 | 0.728 | 0.846 | 0.742 |
| V:FP2 | 0.845 | 0.727 | 0.846 | 0.739 |
| V:FP4 | 0.838 | 0.723 | 0.844 | 0.746 |

Hierarchy + DenseCRF, dev-selected thresholds (0.55 in brackets column).

| Comparison | stage 1 | VOC-Sim test | COCO-Cap test | VOC-Sim, 0.55 | COCO-Cap, 0.55 | reading |
|---|---|---|---|---|---|---|
| MP2 − MK4 | fail | +0.001 [-0.001, +0.003] | -0.001 [-0.008, +0.004] | +0.001 [-0.001, +0.003] | -0.003 [-0.009, +0.002] | descriptive (failed stage 1) |
| MP4 − MK4 | fail | -0.007 [-0.010, -0.003] | -0.005 [-0.016, +0.006] | -0.002 [-0.005, +0.001] | +0.004 [-0.006, +0.013] | descriptive (failed stage 1) |
| FD4 − MK4 | fail | +0.000 [-0.000, +0.001] | -0.003 [-0.005, -0.001] | +0.000 [+0.000, +0.001] | -0.002 [-0.005, -0.000] | descriptive (failed stage 1) |
| FP2 − MK4 | fail | +0.001 [-0.001, +0.003] | -0.004 [-0.010, +0.002] | +0.001 [-0.001, +0.004] | -0.005 [-0.010, +0.001] | descriptive (failed stage 1) |
| FP4 − MK4 | fail | -0.006 [-0.010, -0.002] | -0.008 [-0.018, +0.004] | -0.001 [-0.004, +0.002] | +0.002 [-0.007, +0.012] | descriptive (failed stage 1) |
| MK4 − NC | — | +0.003 [-0.002, +0.008] | +0.012 [-0.007, +0.030] | +0.002 [-0.002, +0.006] | +0.007 [-0.010, +0.024] | — |
| MK4 − A | — | +0.004 [-0.004, +0.012] | +0.023 [+0.005, +0.043] | +0.003 [-0.005, +0.010] | +0.026 [+0.005, +0.047] | — |
| NC − A | — | +0.001 [-0.006, +0.007] | +0.011 [-0.006, +0.029] | +0.001 [-0.007, +0.008] | +0.019 [-0.000, +0.040] | — |

Per class, COCO-Cap test, dev-selected (A / MK4 / MP2 / MP4 / FD4 / FP2 / FP4): aeroplane 0.72 / 0.75 / 0.77 / 0.75 / 0.75 / 0.77 / 0.74, bicycle 0.64 / 0.67 / 0.67 / 0.65 / 0.67 / 0.67 / 0.65, bird 0.89 / 0.88 / 0.88 / 0.87 / 0.88 / 0.88 / 0.87, boat 0.54 / 0.62 / 0.61 / 0.59 / 0.62 / 0.61 / 0.59, bottle 0.68 / 0.83 / 0.83 / 0.79 / 0.82 / 0.82 / 0.78, bus 0.92 / 0.92 / 0.91 / 0.91 / 0.92 / 0.91 / 0.91, car 0.83 / 0.78 / 0.77 / 0.77 / 0.76 / 0.77 / 0.77, cat 0.91 / 0.91 / 0.91 / 0.91 / 0.91 / 0.91 / 0.91, chair 0.47 / 0.47 / 0.48 / 0.47 / 0.47 / 0.46 / 0.47, cow 0.89 / 0.83 / 0.84 / 0.84 / 0.84 / 0.84 / 0.85, dining table 0.50 / 0.58 / 0.61 / 0.66 / 0.58 / 0.60 / 0.64, dog 0.81 / 0.82 / 0.82 / 0.81 / 0.82 / 0.82 / 0.81, horse 0.75 / 0.80 / 0.81 / 0.80 / 0.80 / 0.81 / 0.81, motorbike 0.71 / 0.73 / 0.73 / 0.72 / 0.73 / 0.73 / 0.72, person 0.19 / 0.36 / 0.34 / 0.45 / 0.35 / 0.34 / 0.42, potted plant 0.67 / 0.72 / 0.72 / 0.71 / 0.72 / 0.72 / 0.70, sheep 0.89 / 0.86 / 0.86 / 0.88 / 0.86 / 0.86 / 0.88, sofa 0.67 / 0.73 / 0.74 / 0.73 / 0.73 / 0.74 / 0.73, train 0.76 / 0.67 / 0.60 / 0.61 / 0.67 / 0.60 / 0.61, tv 0.73 / 0.67 / 0.68 / 0.61 / 0.66 / 0.66 / 0.61

