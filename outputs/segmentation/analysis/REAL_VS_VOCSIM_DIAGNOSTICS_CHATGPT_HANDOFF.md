# Handoff: real-photo vs VOC-Sim attention diagnostics (for an outside opinion)

I want your critical opinion on a set of exploratory diagnostics. Please say what you think the findings mean, which ones you don't believe, and what (if anything) is worth trying next. Be blunt; negative answers are useful.

## 1. The method being diagnosed (BRACE)

Training-free, label-free class-word segmentation from Stable Diffusion 1.5 attention. Given an image and a class word, it outputs a binary mask. No masks, no training, no tuning per image at inference.

Pipeline:

1. **Seed (A3):** class-token cross-attention (CA), guided by self-attention (SA), giving a 64×64 score map `c`.
2. **Propagation:** spread the seed through SA graphs from three frozen SA modules, one per resolution:
   - 16 px: `up_blocks.1.attentions.2` (u1.2)
   - 32 px: `up_blocks.2.attentions.2` (u2.2)
   - 64 px: `up_blocks.3.attentions.2` (u3.2)

   SA is recorded over denoising calls 10–20 (t ≈ 661→331, 11 calls), 8 heads.
   - **V0** = mean over calls and heads of A_h(A_h c), i.e. two fine steps (called P2 below).
   - **V3** = capped coarse-to-fine hierarchy using the 16/32/64 graphs (the paper's default).
   - **C16** = a 16 px path, A64 · up(A16 c16).
   - **P_t** = mean over calls and heads of A_h^t c (P0 = seed, P1 = one fine step, P2 = V0).
3. **Refine (R2):** DenseCRF on the map.
4. **Threshold:** min-max-normalise the map and threshold at τ = 0.55.

Paper numbers (V3+R2, τ 0.55): VOC-Sim 0.829 mIoU, COCO-Cap 0.696. Fine-only reference R0+R2: 0.823 / 0.662.

## 2. The two datasets in these diagnostics

**VOC-Sim, 490 images.**
- SD-generated images with prompt "A photograph of a X", one object class per image, and annotated masks.
- Objects are large: no instance is below 1% of the image and 355/490 are above 20%.
- Diagnostics were regenerated live and reproduce the paper (V3+R2 at τ 0.55 = 0.830).

**Real D, 307 pairs.**
- Real PASCAL VOC train photos (200 photos, 20 classes) with real masks, so several classes per photo, clutter and small objects.
- Captured by inverting each photo two ways:
  - **renoise/n0:** add noise at each step, single noise seed;
  - **ddim:** DDIM inversion.
- Only 76 of the 307 pairs stored every module's SA (needed for the module-swap test).

**Protocol for all numbers below.**
- Masks are used only to *measure*, never at inference.
- mIoU is class-macro.
- "OOF" means τ (and any rule) is chosen on 4 grouped folds and scored on the 5th.
- Δ CIs are image-bootstrap 95%.
- τ grid: 0.30–0.70 in 0.025 steps (CRF variants every 0.05).
- **Caveat:** everything on real D is exploratory and chosen on D.

## 3. Findings

### 3.1 The CA seed is good; single heads are worse than the pooled seed

| | real D n0 | real D ddim | VOC-Sim |
|---|---:|---:|---:|
| A3 seed AUROC (64 px) | 0.875 | 0.901 | 0.954 |
| best single (module, call, head), held out | 0.838 | 0.849 | 0.904 |
| token AUROC: SOT / class word / EOT+pads | 0.33 / 0.70 / 0.63 | 0.33 / 0.71 / 0.64 | 0.28 / 0.76 / 0.66 |

- Head rankings are very stable across photos (split-half ρ = 0.98).
- Mid-block CA is the best single CA module on real photos (0.71 AUROC per head-mean).
- EOT and padding tokens carry real class signal (0.63–0.66), and SOT is anti-correlated.
- Picking the top-K single units on held-out folds never beats the pooled seed.

### 3.2 SA "keep / leak" atlas: the frozen modules are the leakiest in their blocks

- **keep lift** = (share of a mask pixel's attention landing in the mask − area) / (1 − area); 1 is perfect.
- **leak ratio** = a non-mask pixel's in-mask attention share / area; 0 is perfect, 1 is chance.

| module | real D n0 keep lift / leak | VOC-Sim keep lift / leak |
|---|---|---|
| u3.0 (64) | 0.605 / 0.33 | 0.852 / 0.13 |
| u3.1 (64) | 0.596 / 0.34 | 0.844 / 0.13 |
| **u3.2 (64, used)** | **0.433 / 0.54** | **0.702 / 0.27** |
| u2.0 / u2.1 (32) | 0.50 / 0.41, 0.52 / 0.42 | 0.74 / 0.22, 0.75 / 0.22 |
| **u2.2 (32, used)** | **0.376 / 0.69** | **0.617 / 0.46** |
| u1.1 (16) | 0.411 / 0.66 | 0.638 / 0.51 |
| **u1.2 (16, used)** | **0.265 / 0.84** | **0.459 / 0.73** |

- The ranking is identical on both datasets.
- Keep lift rises monotonically over denoising calls (real 0.09 → 0.51; sim 0.12 → 0.67).
- On real photos it depends strongly on size: small objects 0.18, large 0.53. On VOC-Sim it is flat (0.57–0.63).

### 3.3 How many propagation steps? One or two; more hurts on real photos

AUROC / AP of y_t = A^t c (head-mean A, 64 px):

| t | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| real n0 AP | 0.620 | **0.674** | 0.666 | 0.638 | 0.583 |
| real ddim AP | 0.667 | **0.724** | 0.702 | 0.664 | 0.601 |
| VOC-Sim AP | 0.913 | 0.967 | **0.970** | 0.964 | 0.947 |

- The best depth grows with object size on both datasets.
- Real, small objects: t = 1 is best and t = 8 nearly halves AP (0.335 → 0.172).
- Real, large objects: t = 4 is best (0.919).

### 3.4 Spectral structure

- The mask's energy is concentrated in the smoothest SA eigenvectors, well above a rolled-mask control.
  - Real, 16 px mean graph: E_8 = 0.56 vs 0.24 rolled.
  - VOC-Sim: 0.78 vs 0.34.
- The oracle best eigenvector reaches AUROC 0.92–0.97. But picking an eigenvector from the seed gets only 0.80–0.83 (real) and 0.92–0.95 (sim), below the seed itself.
- Seed low-pass filtering: no gain on real photos (0.87 vs seed 0.88). A tiny gain on VOC-Sim (0.97 vs 0.955 at 32 px). It helps large objects and hurts small ones.

### 3.5 Where the errors are (very different between datasets)

| | real D (n0 V0+R2) | VOC-Sim (V3+R2) |
|---|---|---|
| precision / recall | 0.52 / 0.75 | 0.89 / 0.91 |
| FP on **another object** | **46%** | 0% (single class) |
| FP background within 8 px of boundary | 3% | 42% |
| FP background far | 51% | 58% |
| FN in the object interior (not the boundary band) | 62% | 79% |
| instances missed (<10% covered) | 20% of 520 | 0% |

- On real photos the FP pixels are mostly co-occurring objects or far background, not boundary bleed. Examples: chair → diningtable 10%, person → car / bus / train 5–6%.
- Small instances (<1% of the image) are only 49% covered.

### 3.6 Map ranking differs between real and synthetic

OOF mIoU:

| map | real D n0 | real D ddim | VOC-Sim (generate) |
|---|---:|---:|---:|
| V0+R2 | 0.495 | 0.531 | 0.816 |
| V3+R2 (paper default) | 0.489 | 0.538 | **0.830** |
| C16 | 0.519 | **0.553** | 0.802 |
| P1+R2 (one step) | 0.516 | 0.547 | 0.810 |

- Real photos favour *less* propagation (C16, P1) and ddim inversion (+0.058 for ddim/C16 over n0/V0+R2, CI +0.028…+0.088).
- VOC-Sim favours the deeper hierarchy V3.
- **Possible grid artefact:** the best τ on real photos is 0.60–0.70, often at the top edge of the grid. On VOC-Sim it is 0.45–0.55.

### 3.7 Negative results (tested, didn't work)

**(a) Per-image label-free reliability switch (V0 vs V3 per pair).**
- The per-pair oracle switch gives only 0.487 vs always-V0+R2 at 0.475 (τ 0.55, real D).
- Even oracle features using the true mask buy at most +0.007 OOF.
- Label-free features buy ≈ 0.

**(b) Size-dependent propagation depth.** A label-free area estimate â (share of the seed ≥ 0.5) picks the depth per size bin.
- â is a poor size proxy on real photos (ρ with true area 0.32, vs 0.73 on VOC-Sim).
- Real D: rule+R2 − V0+R2 = +0.020 (CI +0.002…+0.033, n0) and +0.021 (CI −0.002…+0.034, ddim).
  - The rule collapses to "P1 almost everywhere".
  - It does not beat C16.
- VOC-Sim: ties V0 and is −0.007 vs V3+R2 (CI −0.015…−0.000).
- Transfer between datasets fails, mostly because of the τ mismatch.
- Oracle per-pair depth headroom is only +0.04–0.05 IoU.

**(c) SA module swap** (replace the leaky frozen modules with the "best-keeping" ones). The Δ is vs base; VOC-Sim is the clean test.

| config (16 / 32 / 64) | VOC-Sim V0+R2 | VOC-Sim V3+R2 | real D76 V3+R2 |
|---|---|---|---|
| base u1.2 / u2.2 / u3.2 | 0.824 | 0.830 | 0.563 |
| fine → u3.0 | 0.785 (−0.040, CI excl. 0) | 0.819 (−0.011, excl. 0) | 0.587 (+0.024, −0.001…+0.042) |
| fine → u3.1 | 0.800 (−0.024) | 0.819 (−0.011) | 0.583 (+0.020, crosses 0) |
| coarse → u1.1 / u2.1 | 0.824 (same) | **0.836 (+0.005, +0.000…+0.010)** | 0.571 (+0.008, crosses 0) |
| both | 0.785 | 0.819 (−0.011) | 0.593 (+0.030, −0.008…+0.066) |

- Interpretation so far: a less-leaky fine module spreads the seed less, and on VOC-Sim that loses recall. Leakage in u3.2 seems to be doing useful propagation work.
- On real photos the swaps lean positive but are not significant. They are also measured on the very pairs the ranking came from.

## 4. What I'd like your opinion on

1. **Real vs synthetic gap.** Is the main story that real photos fail by *semantic confusion with co-occurring objects*, while VOC-Sim fails at *boundaries and interiors*?
   - If so, is SA propagation the wrong lever for real photos?
   - Would something on the CA/token side be better: EOT/pad tokens, contrasting against other class words, mid-block CA?
2. **Does the paper's claim survive?** The method (V3+R2) is best on VOC-Sim but loses to the simpler C16/P1 on real photos (inverted). Is "VOC-Sim and COCO-Cap are generated images" a sufficient scope statement, or does this undermine the method?
3. **The τ mismatch** (real 0.60–0.70 vs sim 0.45–0.55). Is a label-free per-image normalisation or threshold principled and worth testing, or is it likely to be another dead end like the per-image switch?
4. **Leakage helps.** Is "leakier SA modules propagate better" a known or plausible effect? Is it worth a sentence in the paper?
5. **Missing analyses.** Is anything here statistically shaky or misleading? For example:
   - 76 pairs for the module swap;
   - selection on D;
   - OOF with only 5 folds;
   - bootstrap over images when there are several pairs per photo.

## 5. Constraints on suggestions

- Inference must stay label-free: no masks and no per-dataset tuning on test labels.
- Frozen SD1.5, no training.
- The paper is late-stage. Prefer small, defensible changes or honest scoping over new machinery.
