# Run 34: better object-vs-leakage discrimination in the absent-class gate (pre-registration)

Committed before any fit, capture or scoring of this run. BRACE stays frozen. **Exploratory.** This covers steps
0–3 of the plan agreed on 2026-11-22 (Claude + ChatGPT review). It builds on run 31 (`runs/31_multineg_gate/`).

## 1. Setting (identical to run 31)

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

## 2. Gates compared

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

## 3. Stage 1: pixel-level screen (CPU, run 31 dev dumps; no new capture)

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

### Step 0: Q–K residual diagnostic (stage 1 data)

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

## 4. Stage 2: full pipeline (GPU capture; same overnight chain)

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

## 5. Checks (smoke, before the full chain)

- MP with one object centroid and no Fisher equals `jc_mneg.g_multi` (MK4): max |Δ| < 1e-5.
- Seed fidelity of arm A < 1e-2 (as run 31).
- ⊥t / ⊥P residuals are orthogonal to the removed keys: max |cos| < 1e-4.
- Every map is finite.

## 6. Files and cost

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
