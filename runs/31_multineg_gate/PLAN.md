# Run 31: query-profile gate diagnostics and a multi-negative-prototype gate (implementation plan, draft for review)

Status: **draft for external review. Nothing has run.** After review, this file is frozen as `PREREG.md` and
committed before any capture or scoring. BRACE stays frozen. Everything here is exploratory.

- **COCO-Cap** was already used in runs 28–30. **VOC-Sim** was used for the gate's development check (run 29) and
  for the class-adaptive choice (run 30).
- The dev/test split below protects only against selection *within this run*. No result here is independent
  confirmation; that would need fresh annotated images.

## 0. Background and what is new

The gate in runs 29–30, per class c and per cross-attention (CA) module m of the 11 used by BRACE:

- q̂_i is pixel i's call-averaged, head-merged CA query, L2-normalised (`jc_trio.qbar`).
- **Object profile** m_o: the mean q̂ at the top 5% of the class word's call-averaged attention t̄_c, on images of
  class c.
- **Negative profile** n: the mean q̂ at the top 1% of t̄_c, on images where c is absent (target ≠ c and the caption
  does not mention c; fixed synonym list, `jc_trio.SYN`). That is the absent-class version (NC). The all-pixel
  control (NO) replaces it with the mean q̂ over all pixels.
- **Gate:**
  - w = (m_o − n)/|m_o − n|; p₊ = w·m_o; p₋ = w·n;
  - g(i) = σ(4·(w·q̂_i − (p₊+p₋)/2)/(p₊ − p₋));
  - the class word's attention at module m is multiplied by g in every call, before BRACE's layer weighting.

Run 30 found:
- object-only gates (no population negative) fail;
- absent-class and all-pixel negatives work about equally;
- per-class strength chosen on VOC-Sim helps on both benchmarks (+0.004 / +0.018);
- the gain is largest where BRACE's seed leaks most into the far background.

**Hypothesis:** a single negative mean blends several kinds of false firing. Several negative prototypes per class
and module, with each pixel compared against its nearest one, may reject more background without rejecting more
object.

### Overlap with earlier failed methods (checked in the protocol and the failure inventory)

| Earlier method | What it did | Why this is different |
|---|---|---|
| M1/M2 "query prototypes" (protocol, combined screens) | replaced the map with per-image similarity to a CA-weighted or k-means query prototype; −0.21 to −0.40 | here the CA seed and BRACE hierarchy are kept; prototypes are population-level and only *gate* CA attention softly |
| SG4 k-means on SA rows / spectral | segment the image with SA clusters, label with CA | no SA clustering; clustering is on the negative *query* pool across images |
| Run 21 layer gate, run 22 relative pixel gate | gates from layer agreement / magnitude | not query-profile contrasts; failed because they penalised magnitude |
| Run 28 K1 class competition | caption classes compete inside the hierarchy | not used; context-specific negatives (reviewer's idea 2) are close to this, so they are **not** implemented |
| SG graph-consistency work | SA-graph-based weighting | reviewer's idea 3 (SA-influence) is close to this; **not** implemented |

## 1. Data split (fixed before anything runs)

- Within each (benchmark, class), order images by `crc32(image_id)`.
  - Even rank → **dev**; odd rank → **test**.
  - This gives about 245 VOC-Sim and 354 COCO-Cap images on each side, stratified by class.
- Within dev, the same order gives two cross-fitting folds: rank // 2 even → **dA**, odd → **dB**.
- Ground-truth masks are used only to score, to define the diagnostic regions (object / near / far background),
  and to choose thresholds on dev. They are never used to build profiles, choose K, or set any gate parameter.

## 2. Part A: CPU diagnostics on the saved run 30 captures (all 1,197 images)

**Data:** `<CACHE>/jc_qgate/{voc,coco}/*.npz`:
- `gates[4,64,64]`: module-mean gate maps for NC, NO, PD, PI;
- `maps[6,3,64,64]`;
- `truth`.

**Regions** at 64×64, as in run 30:
- object = area-pooled GT > 0.5;
- near background = within 6 px (Chebyshev) of the object, excluding the object;
- far background = the rest.

### A1: gate distributions (reviewer's point 1)

For the NC and NO gate maps, on each benchmark:

- Quantiles 5 / 25 / 50 / 75 / 95 of the gate, pooled over all pixels of each region type.
- **Separation:** AUROC of the gate, object vs far background and object vs near background. Computed per image
  (Mann–Whitney) and then averaged over images, plus pooled.
- **Object retention** = fraction of object pixels with g > t, and **far rejection** = fraction of far pixels with
  g ≤ t, for t ∈ {0.3, 0.5, 0.7}.
- All of the above per class as well.

Caveat: the saved gate map is the uniform mean of the 11 module gates. The real gate acts per module and per call,
so A1 describes the average gate, not every module.

### A2: how much would a sharper single gate help (seed level only)

- s_γ = minmax(seed_A ⊙ ĝ^γ), γ ∈ {0.5, 1, 2, 4}, where ĝ is the NC module-mean gate map and seed_A is BRACE's
  seed.
- **Report:** seed mIoU (VOC-Sim OOF threshold; VOC-Sim-selected threshold on COCO-Cap) and the far-background
  share of seed mass.
- This is only an approximation at the seed: the hierarchy cannot be rerun, because run 30 saved no
  self-attention. For the ceiling after propagation we already have run 24's oracle: a perfectly clean background
  adds about +0.05 / +0.06 to BRACE's mIoU.

## 3. Part B: GPU query dump on the dev images (about 600 images)

**Script:** `scripts/jc_mneg.py dump --set voc|coco --shard i --num-shards n`.
- It uses the frozen runtime and `jc_trio`, `jc_absent` and `jc_qgate` helpers unchanged.
- One live SD1.5 generation per dev image, with the runs/20 corrected prompts and the safety checker off.

Saved per image (`<CACHE>/jc_mneg/dump_{set}/<id>.npz`):

| Field | Content | Size |
|---|---|---|
| `q_<module>` | q̂ for every pixel of each of the 11 A3 modules (5 at 64 px × 320 dims, 5 at 16 px × 1280, 1 at 8 px × 1280), fp16 | ≈ 16.6 MB |
| `t_<module>` | t̄_c for all 20 classes (class-word keys, as in run 29), fp16 | ≈ 0.9 MB |
| `seed`, `truth64`, `label`, `mention[20]`, `fid` | runtime seed, area-pooled GT at 64×64, target class, caption-mention flags, recomputed-vs-runtime seed check (stop above 1e-2) | small |

- **Total:** about 17.5 MB per image, about 10.5 GB. There is 637 GB free on /home.
- **Time:** about 15 s per image, so about 2.5 GPU-hours, split over 4 GPUs.

## 4. Part C: CPU profile fitting and query-space diagnostics on the dump

### C0: profiles (used by Parts C and D)

For each class c and module m, from dev images only:

- **Object pool O_c,m:** q̂ at the top 5% pixels of t̄_c, on dev images of class c.
- **Negative pool N_c,m:** q̂ at the top 1% pixels of t̄_c, on dev images where c is absent (target ≠ c and not
  mentioned).
- **All-pixel pool:** the mean q̂ over all pixels of the dev images (for NO).
- **Multi-negative prototypes:** spherical k-means of N_c,m for K ∈ {1, 2, 4, 8}.
  - Cosine distance on unit vectors; k-means++ from seed 0, 10 restarts, 100 iterations; lowest inertia kept.
  - Centroid n_k = the **plain mean** of its member unit vectors, not renormalised, so that K = 1 gives exactly NC's
    single profile.
  - A cluster with fewer than max(20, 2% of the pool) members is dropped and its members reassigned.
  - If |N_c,m| < 20·K, that cell falls back to K = 1.
- **Three fits:**
  - all dev (used for the test images);
  - dA only (used to score dB images);
  - dB only (used to score dA images).

  So no image is ever scored with profiles that include itself.

### C1: are the remaining false positives clustered? (reviewer's point 2)

- On dev images of class c, compute the NC gate per module from the dump (cross-fitted profiles).
- **High-gated false positives (HGFP):** background pixels (near or far) with module gate g > 0.5.
- Spherical k-means of their q̂, pooled over the dev images of class c, for K ∈ {2, 4, 8}.
- **Metrics:**
  - cosine silhouette, on at most 5,000 sampled points per cell;
  - inertia drop relative to K = 1.
- **Null comparison:** the same metrics on an equal-size sample of high-gated *object* pixels (g > 0.5, GT object).
  If the false positives are not more structured than the true positives, there is no evidence for the
  hypothesis.

### C2: are the clusters reproducible? (reviewer's point 3)

- **Across images:** split the dev images of class c into two halves by hash and cluster each half. Match the
  centroids with the Hungarian method on cosine, and report the mean matched cosine.
- **Across modules:** for the five 64-px modules, which share one pixel grid, compute the adjusted Rand index
  between the cluster assignments of the same HGFP pixels for each pair of modules.

### C3: does the single negative profile average several modes? (reviewer's point 4)

For each negative pool N_c,m and each K ∈ {2, 4, 8}:
- silhouette, inertia drop, cluster sizes, and the cosine between the single mean n and each centroid n_k;
- the **source make-up** of each cluster: the distribution of the target classes of the images that contributed
  its members.
- **The key estimate.** Among HGFP pixels, the fraction that the multi-negative gate (Section 5) would push below
  0.5 while NC leaves them above 0.5 (extra background removed). And among object pixels, the fraction it would
  newly push below 0.5 (object wrongly removed). This estimates whether the method can help before running the
  hierarchy.

### C4: classes harmed by the gate (reviewer's point 5)

Classes compared: tv, train and potted plant (harmed) vs person, dining table and aeroplane (helped). On their dev
images, per class:
- object gate quantiles, and the fraction of object pixels with g < 0.5;
- mean cos(q̂_object, n) vs mean cos(q̂_object, m_o);
- the fraction of object pixels whose nearest negative centroid is closer than m_o;
- which negative clusters absorb those pixels, and the clusters' source classes. For example: does the "tv"
  negative cluster come from sofa or living-room images?

### Go / no-go for the multi-negative gate (fixed now)

**Structure is present** if both hold, at the K chosen in Section 5:
1. the median silhouette of the negative pools over (class, module) cells is ≥ 0.10;
2. the median split-half matched-centroid cosine is ≥ 0.90.

Part D runs either way, because the owner asked for the full run. If the rule fails, the multi-negative arms are
reported as descriptive only, with the verdict "no structure; not a candidate".

## 5. The multi-negative gate (MK)

For class c and module m, with object profile m_o and negative centroids n_1 … n_K:

- For each k:
  - w_k = (m_o − n_k)/|m_o − n_k|;
  - p₊ₖ = w_k·m_o; p₋ₖ = w_k·n_k;
  - g_k(i) = σ(4·(w_k·q̂_i − (p₊ₖ + p₋ₖ)/2)/(p₊ₖ − p₋ₖ)).
- **Assignment:** k*(i) = argmax_k cos(q̂_i, n_k/|n_k|), the nearest negative mode.
- **Gate:** g(i) = g_{k*(i)}(i). It is applied exactly like NC: it multiplies the class word's attention at module m
  in every call, before the layer weighting. Gain 4, unchanged.
- **Checks:**
  - K = 1 must reproduce NC exactly (max |Δ| < 1e-5; asserted in the smoke run);
  - with all gates set to 1 the result must reproduce BRACE (seed fidelity < 1e-2, as before).
- **Why the nearest negative and not the minimum over k** (reviewer's max-similarity form). The minimum over k
  rejects a pixel if *any* negative mode explains it. That gives real object pixels more chances to be rejected,
  which is exactly the tv / train failure. The nearest mode keeps one decision per pixel. The minimum is not run.
- **Choice of K, label-free and on dev only:** K* = the K ∈ {2, 4, 8} with the highest median silhouette of the
  negative pools on all dev images. All K are reported.

## 6. Part D: GPU evaluation capture (all 1,197 images)

**Script:** `scripts/jc_mneg.py capture --set voc|coco`, with the same per-image pipeline as run 30.

**Arms (seeds):**
- A (BRACE);
- NC, NO and NH (NC at half strength, g^0.5);
- MK2, MK4, MK8.

**Profiles:** test images use the all-dev fit. dA images use the dB fit, and dB images use the dA fit.

**Saved per image:**
- maps [7 arms, 3 maps (seed, fine-only V0, hierarchy V3), 64, 64];
- module-mean gate maps for every gated arm;
- truth, the image, and the seed check.

**Scoring** (`score`, CPU):
- raw and DenseCRF R2, thresholds 0.30–0.70 in steps of 0.025, as in runs 28–30;
- `Pool(maxtasksperchild=25)` with a per-image timeout, because run 30's scoring hung when a worker died.

**Thresholds:**
- **Dev-selected (primary):** for each arm and map, the threshold that maximises the mean of VOC-Sim-dev mIoU and
  COCO-Cap-dev mIoU, applied unchanged to both test halves.
- **Also reported:** fixed 0.55, and each map's best threshold on test (uses labels; descriptive).

**Class-adaptive strength (CAg), redone on dev:** for each class, choose among {A, NH, NC} the arm with the highest
dev mIoU on that class (VOC-Sim-dev and COCO-Cap-dev images pooled, dev-selected thresholds). Ties go to A. The
choices are applied unchanged to the test halves.

**Comparisons** (test halves; hierarchy + DenseCRF; dev-selected thresholds; paired image bootstrap, 2000):

| ID | Comparison | Role |
|---|---|---|
| T1 | MK(K*) − NC | does several negative modes beat one? |
| T2 | MK(K*) − A | gain over BRACE |
| T3 | CAg − A | class-adaptive strength on fresh selection |
| — | NO − NC, NH − NC, MK2/4/8 − NC, every arm − A on seed and fine-only, per class | descriptive |

**Reading rule (fixed now):**
- **"Promising"** = the interval is above zero on the COCO-Cap test half **and** the lower bound on the VOC-Sim
  test half is ≥ −0.005.
- Both test halves have been seen in earlier runs, so even a "promising" result is exploratory.

## 7. Order of work and cost

| Step | Where | Time |
|---|---|---|
| A1, A2 | CPU, existing run 30 data | < 1 h |
| Smoke: dump + capture on 4 images per benchmark; asserts K=1 ≡ NC and the seed fidelity | GPU | < 30 min |
| B dump (≈600 dev images) | 4 GPUs | ≈ 1 h |
| C0–C4 profiles and diagnostics | CPU | ≈ 1–2 h |
| **Diagnostic report (A + C) to the owner before D** | — | — |
| D capture (1,197 images, 7 arms) | 4 GPUs | ≈ 1–1.5 h |
| D scoring (7 × 3 maps × 17 thresholds, raw + CRF) | CPU | ≈ 2–3 h |

**Files:**
- `scripts/jc_mneg.py`, with subcommands `diag30`, `dump`, `fit`, `diag`, `capture`, `score`, `report`;
- `runs/31_multineg_gate/{PREREG.md, diag.md, result.md}`;
- data under `<BRACE_CACHE_DIR>/jc_mneg/`.

**Not done here, and why:**
- **Context-specific negatives (reviewer's idea 2):** too close to the failed caption class competition.
- **SA-influence suppression (idea 3):** too close to earlier graph-consistency work.

Both can be revisited if C4 shows that the harmed classes' errors come from specific co-occurring classes.
