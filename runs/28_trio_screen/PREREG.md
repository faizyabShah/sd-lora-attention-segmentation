# Three training-free seed/propagation ideas: COCO-Cap screen (pre-registration)

Owner request (2026-11-19): code three untested training-free ideas, smoke-test them, run overnight, and look at
COCO-Cap only for now ("leave VOC-Sim for now"). Written and committed before any image of this run is captured.

**Status: exploratory screen on COCO-Cap.** This breaks COCO-Cap's "untuned transfer" role for these three ideas,
by the owner's choice; it is disclosed. Nothing is promoted from this screen. Any idea that looks promising needs a
VOC-Sim development check and a confirmation on fresh images before any claim. BRACE stays frozen. Inference is
label-free: the query class (the image's target class, as in BRACE) and the image's own caption are inputs; masks
are used only for scoring.

## Arms (all from one live SD1.5 generation per image, frozen runtime, runs/20 corrected prompts, safety checker off)

| Arm | Seed | Propagation |
|---|---|---|
| **A** | BRACE seed (runtime A3) | BRACE V3 hierarchy |
| **X1** | SDXL cross-attention seed alone | V3 |
| **X2** | geometric mean of min-max BRACE seed and min-max SDXL seed (idea 1, primary) | V3 |
| **N1** | absent-class-calibrated BRACE seed (idea 2, primary) | V3 |
| **K1** | BRACE seed | V3 with class competition at 16, 32 and 64 (idea 3, primary) |
| **K0** | BRACE seed | V3 per class, competition only on the final maps (control for K1) |

### Idea 1: SDXL agreement seed (X1, X2)

SDXL base 1.0 (cached weights, fp16 UNet, fp32 VAE) reads the SD1.5-generated image: resized to 1024×1024, VAE mean
latent, noised to t ∈ {100, 300, 500} (one noise draw each, seed 0), one conditional UNet call per t with the
image's generation caption, no CFG. At every SDXL cross-attention module, the class word's attention is computed with
keys from the attribution prompt (as BRACE: the class word alone on COCO-Cap): softmax over the 77 tokens, mean over
the class word's tokens, mean over heads, bilinear to 64×64. Seed = uniform mean over all modules and the three
calls, min-max. X2 = min-max(√(mm(BRACE seed) · mm(SDXL seed))).

### Idea 2: absent-class calibration (N1)

Two passes. **Statistics pass** (a separate generation of the same prompts; it only defines the profiles): for every
image, every one of the 11 A3 cross-attention modules, and every class c of the 20, the class word's call-averaged
attention map t̄_c (keys from the class-word prompt) and the call-averaged head-merged query q̄ (rows L2-normalised).
- *Spurious set* for c: images whose target class ≠ c **and** whose caption does not mention c (fixed synonym list
  in the script); their top 1% pixels of t̄_c. Mean query m_s(c, module).
- *Object set* for c: images whose target class = c; their top 5% pixels of t̄_c. Mean query m_o(c, module),
  **leave-one-out** (the test image's own contribution removed).

At test time, per module: ŵ = (m_o − m_s)/|m_o − m_s|, p_o = ŵ·m_o, p_s = ŵ·m_s, gate
g(i) = σ(4·(ŵ·q̂_i − (p_o + p_s)/2)/(p_o − p_s)) from the image's call-averaged query. The class word's attention at
that module is multiplied by g in every call before the A3 weighting. Constants fixed here, not tuned.

### Idea 3: class competition inside the hierarchy (K1, K0)

Competitors = other VOC classes mentioned in the image's caption (same synonym list). Each competitor's seed is
BRACE's seed construction with its own class-word keys (same SA-guided layer weights). In each of the 11 calls, all
classes (target + competitors) are propagated through the V3 path together; after 16-SA, after the 16→32 cap, and
after 64-SA, every class's map r_c is multiplied by its share m_c/Σ m_c', m = per-class min-max at that scale.
With no competitor, K1 = K0 = A exactly. K0 applies the share once, on the final V3 maps.

## Scoring and comparisons

All 707 COCO-Cap images. Maps per arm: seed and BRACE-hierarchy map; raw and + DenseCRF R2 at thresholds 0.30–0.70
(step 0.025). Class-macro mIoU; paired image-bootstrap 95% intervals (2000). Reported at **fixed 0.55 (primary)**
and at each arm's best threshold on COCO-Cap (in-sample, descriptive).

| ID | Comparison (hierarchy map + DenseCRF, 0.55) |
|---|---|
| T1 | X2 − A |
| T2 | N1 − A |
| T3 | K1 − A (all images; also on the images with ≥ 1 competitor) |
| T4 | K1 − K0 (inside vs after; images with ≥ 1 competitor) |

Descriptive: X1 − A, K0 − A, seed rows, raw rows, per class.
**Promising** (licenses a VOC-Sim check, nothing more): Δ at 0.55 positive with its interval above zero.

## Checks

The recomputed BRACE seed from recorded queries and class-word keys must match the runtime seed (max |Δ| reported;
the job stops above 1e-2). K1/K0 with no competitors must equal A (checked per image).

## Files

`scripts/jc_trio.py` (`stats` GPU sharded → `fit` CPU → `capture` GPU sharded → `score` CPU → `report`). Data under
`<BRACE_CACHE_DIR>/jc_trio/`; report `runs/28_trio_screen/result.md`. Smoke (tag `_smoke`, a few images) is kept.
