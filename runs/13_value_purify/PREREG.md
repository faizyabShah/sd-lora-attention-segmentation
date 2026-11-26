# Pre-registration: (V) self-attention value-space compatibility and (P) semantic subspace purification of the class context (two diagnostics)

Written 2026-11-16, before any capture. Code: `scripts/replay_vp.py`. BRACE stays frozen. Ground truth only scores.

## Audit (what was run before)

- **Purification has a map-space precedent, never tested in key space.** The semantic gate (`SEMGATE_CLIP_CHATGPT_HANDOFF.md`) subtracted the 8 generic-word seeds from A3 in **map** space: mechanism PASS, practical FAIL. SG5's contrast seeds (`WS`/`WN`/`WE`, protocol §§66–68) traded precision for recall.
- **Values were never used in propagation.** SA *values* were never captured or used. Every earlier SA experiment used attention probabilities: BRACE, the cap (run 06), SCW (run 12), and the SA boundary diagnostic (run 05: +0.040, CLOSE). Run 08 used decoder features (up2, step 15): +0.003, CLOSE.
- **Not run, because already answered:** synonym ensembles (SG5 `WY`: no gain); per-image token optimisation (run 12 ROC: STOP; the B1/B2 token track uses labels).

## Data and replay (shared)

- **Images:** the 150 VOC-Sim + 150 COCO-Cap screening images (`CACHE/ana_dump`). These are development images, disclosed.
- **Replay:**
  - All 31 stored UNet inputs, conditional prompt only, on a non-wald GPU.
  - **The exact A3 seed is replayed:** the 11 frozen CA modules, A3 = head-mean class probability (77-token padded softmax, mean over class pieces), weighted per call by `affinity_similarity` of each module's paired SA against the pseudo-affinity of the reference query (`REF_SUFFIX`), summed over 31 calls, then min–max. This is the same code path as the live recorder.
  - **Fidelity, reported:** the correlation of the replay seed with the stored live seed, and of the replay V3 with the stored live V3.
- **Within-capture comparison:** every arm comes from the same replay.

## (V) Value-space compatibility: edge diagnostic (no propagation operator is built)

**Edges:**
- At the fine SA module (64 × 64), for calls 10–20, take the head-mean attention averaged over calls, A.
- For each cell i, its 16 highest-attention targets j ≠ i are edges, i.e. the links that actually carry BRACE's propagation mass.
- **Labels** (ground truth, scoring only):
  - cells are pure if pooled ground truth ≥ 0.75 (object) or ≤ 0.25 (background);
  - **crossing** (label 1) = object ↔ background; **within** (label 0) = object ↔ object;
  - background ↔ background edges are excluded.
- **Sampling:** at most 2000 edges per image, uniformly at random, with the RNG seeded by crc32(image_id).

**Features:**
- **Baseline:**
  - Lab distance between the two cells' mean colours;
  - the maximum Lab Sobel contour along the segment between them;
  - spatial distance;
  - the attention affinity, max(A_ij, A_ji);
  - the decoder feature cosine (`feat/up2/s15`, as in run 08);
  - the minimum and maximum of BRACE's confidence (min–max live V3) at the two cells.
- **Test:** the value cosine, cos(v_i, v_j) of the 320-d SA value vectors (heads concatenated) at the fine module, averaged over calls 10–20.

**Gate (decisive set COCO-Cap):**
- Out-of-fold logistic regression, folds crc32 % 5, pooled edges; baseline vs baseline + value cosine.
- **PASS:** ΔAUROC ≥ **+0.05** with image-bootstrap 95% CI lower bound **> 0** (2000 resamples).
- A pass only licenses a separately pre-registered value-aware propagation; none is built here.
- **Reported:** single-feature AUROCs, and the VOC-Sim gate.

## (P) Semantic subspace purification of the class context

**Nuisance set (fixed, registered earlier for the semantic gate):** wall, floor, ground, sky, grass, tree, building, road.
- Each word is encoded alone, padded to 77.
- Its context vector is the text-encoder hidden state, the mean over its pieces (768-d).
- N = an orthonormal basis of the 8 vectors.

**Purification:**
- For every class-piece position of the attribution context: e' = e − α N Nᵀ e, rescaled to ‖e‖ (norm-preserving).
- Only the attribution keys change; the generation conditioning and the image are untouched.
- The A3 seed is recomputed with e', then frozen V3, τ 0.55 and DenseCRF R2.

**Arms:**
- `base`: exact A3 replay, no change.
- `pur1`: α = 1. **Primary.**
- `pur0.5`: α = 0.5. Secondary.
- `rand`: 8 random Gaussian directions (seed 0), orthonormalised, α = 1, norm-preserving. **Control.**
- `meandir`: removes only the mean nuisance direction (1-d), α = 1, norm-preserving. **Control:** does any effect come from the shared common direction alone?
- `live`: the stored live seed with replayed SA (fidelity only).

**Gate (`pur1` passes only if all of these hold):**
- COCO-Cap Δ class-macro mIoU vs `base` ≥ **+0.010** with CI lower bound **> 0**.
- VOC-Sim Δ ≥ **−0.005**.
- COCO-Cap Δ(`pur1` − `rand`) with CI lower bound **> 0**.

A pass licenses a confirmation on fresh images with a live capture; it never changes BRACE.

## Fixed

Nothing is tuned after the run: not the words, α, k = 16, the edge cap, the features or the bars. There is no follow-up sweep after a negative result.

## Smoke test (2 VOC-Sim + 2 COCO-Cap images, neumann V100; before the full run)

- **The code runs at about 9 s per image.**
- **Exact-A3 replay fidelity:** r(replay seed, live seed) = 0.96–0.99, and r(V3 from replay seed, stored live V3) = 0.94–0.99.
- **Crossing edges are 2–13%** of the sampled top-16 links per image. This is expected; the AUROC is unaffected.
- No design change was made.
- The smoke numbers are not evidence.
