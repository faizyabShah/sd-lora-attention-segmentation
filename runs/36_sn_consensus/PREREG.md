# Run 36: self-negative gate in the full pipeline, plus module-consensus gates (pre-registration)

Committed before any scoring of this run. BRACE stays frozen. **Exploratory.** It builds on runs 31, 34 and 35:
- run 31 (`runs/31_multineg_gate/`): split, pools, MK4 and the calibration;
- run 34 (`runs/34_gate_discrim/`): the stage-1 measure and the stage-2 pipeline;
- run 35 (`runs/35_gate_negatives/`): SN passed stage 1, and N4 found that the 64 × 64 modules' gates are near chance.

There are two questions:
1. **SN, stage 2:** does the self-negative gate (run 35) improve on MK4 end to end?
2. **H, stage 1 then stage 2:** does a gate shared across modules help? It is built from the reliable low-resolution
   modules and was designed from the N4 numbers. It is applied to both MK4 and SN.

## 1. Setting (as runs 34 and 35)

- **Benchmarks:** VOC-Sim (490) and COCO-Cap (707), generation-time capture with the runs/20 prompts. Run 31's
  `split.json` (dev folds dA / dB, test halves).
- **Regimes:**
  - V: profiles from VOC-Sim dev only;
  - X: profiles from VOC-Sim + COCO-Cap dev.
- **Profiles:** run 35's fit (`<CACHE>/jc_gate35/fit.pt`): MK4 object means and 4 absent centroids. It is not
  refitted.
- **Held-out rules (as runs 34 and 35):**
  - dev images use the other fold's profiles;
  - test images use all dev;
  - regime V on COCO-Cap uses all VOC-Sim dev.

  The same rule applies to the per-module weights of RW (§2).

## 2. Gates

Base per-module gates, unchanged:
- **MK4** (run 31);
- **SN** (run 35): MK4's four absent centroids plus the image's own self-negative. The self-negative is the mean q̂
  over the bottom 30% of t̄_c, where t̄_c is attention to the class word asked for, so no label is involved.

**Consensus gates (H).** These are new. Let g_m be a base gate at module m, on that module's grid.
1. Each g_m is upsampled bilinearly to 64 × 64.
2. They are combined as ḡ = Σ_m w_m·up(g_m) / Σ_m w_m.
3. ḡ is then applied at **every** module, area-resized to that module's grid.

| Consensus | Weights w_m | Role |
|---|---|---|
| **AV** | 1 for all 11 modules | control: consensus without reliability |
| **LR** | 1 at the six modules below 64 × 64, 0 at the five 64 × 64 modules | low-resolution-only gate (N4) |
| **RW** | clip(2·(AUC_m − 0.5), 0, 1) | reliability-weighted |

- AUC_m is the base gate's class-balanced held-out AUROC at module m (object vs far background), taken from run 35's
  stage-1 rows (`<CACHE>/jc_gate35/stage1_rows.pt`).
- AUC_m is computed over the dev images of the regime's profile sets: regime V uses VOC-Sim/V; regime X uses
  VOC-Sim/X and COCO-Cap/X, pooled before class balancing.
- **Held out:** for a dev image, AUC_m comes from the other fold; for a test image and for regime V on COCO-Cap, from
  all dev of the regime.
- **This uses dev masks** to fix 11 numbers per regime and base gate, as dev thresholds already do. The scored image's
  mask is never used, and inference stays label-free.
- If Σ_m w_m = 0, the consensus is the base gate itself, unchanged. This is not expected.

Candidates: MK4-AV, MK4-LR, MK4-RW, SN-AV, SN-LR and SN-RW. Unlike run 35's "low-resolution only", which set g = 1
at the 64 × 64 modules, every module stays gated. That earlier option would leave those modules' background
untouched, so it could never match MK4's suppression.

### Operating strength (fixed now)

- The gate is applied as g^γ.
- **MK4 stays at γ = 1** (run 31).
- **Every other gate runs at γ\***: the γ at which its class-balanced, mass-weighted far-background retention on dev
  (§3 curves) equals MK4's at γ = 1.
  - Regime V: from VOC-Sim dev / V.
  - Regime X: from the mean of the VOC-Sim dev / X and COCO-Cap dev / X curves.
  - Interpolated linearly in log γ on the §3 grid; clipped to [0.25, 4].
- γ\* is fitted once per regime on all dev images, so it is not held out at the fold level. It is a single scalar.
- From run 35's curves, SN's γ\* is about 0.49 (V) and about 0.53 (X). The value used is the one this run's §3
  computes.
- **SN1** (SN at γ = 1) is also captured, as a descriptive arm.

## 3. Stage 1 for the consensus gates (CPU, run 31 dev dumps)

- **Measurement and averaging:** as run 35.
  - Retention = Σ w·g^γ / Σ w, mass weights (A3 seed) primary.
  - Mean over modules, then images, then classes.
  - γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4}; 2000 image-bootstrap samples.
  - Matched far suppression over the **base gate's** γ = 0.5 to 2 range.
  - ΔR = mean object-retention difference.
- **Gates:** MK4, SN and the six consensus gates.
- **Pass (fixed now):** mass-weighted far background, candidate vs **its own base** (MK4-x vs MK4, SN-x vs SN).
  ΔR ≥ +0.01 and CI above 0, on both VOC-Sim dev / V and COCO-Cap dev / X.
- **Reliability reading (descriptive):** RW − AV and LR − AV, matched against AV.
- **Also reported:** pixel weights, near background, AUROC, the two other settings, person excluded, and γ\* for
  every gate.

## 4. Stage 2: full pipeline (GPU capture, as run 34)

- **Capture:** one generation-time capture of all 1,197 images.
- **Arms:**
  - A (no gate);
  - {X, V} × {MK4, SN, SN1, MK4-AV, MK4-LR, MK4-RW, SN-AV, SN-LR, SN-RW}.
- **Maps:** seed, fine-only (V0) and BRACE hierarchy (V3); DenseCRF R2 on V3 at the full threshold grid.
- **Primary rule (as runs 31 and 34):**
  - test halves, V3 + DenseCRF;
  - thresholds chosen on dev (regime V: VOC-Sim dev; regime X: mean of both dev curves);
  - paired image bootstrap, 2000 samples.
- **"Improves on" its comparator:** COCO-Cap test CI lower bound > 0 **and** VOC-Sim test CI lower bound ≥ −0.005,
  per regime.

| Contrast | Read when |
|---|---|
| **SN − MK4** | always (SN passed run 35 stage 1); **primary question of this run** |
| MK4-x − MK4 | MK4-x passed §3 |
| SN-x − SN | SN-x passed §3 |
| SN1 − MK4, SN − A, MK4 − A, x − AV | descriptive |

- **Also reported:** fixed threshold 0.55, per class (COCO-Cap test), and seed-only and V0 maps (descriptive). The
  last two show whether the gain survives the hierarchy.
- **Combined reading:** if both SN − MK4 and SN-x − SN improve, the combined gate is SN-x. If only SN does, it is SN.
  This is not a promotion: BRACE stays frozen, and any change to the paper's method is a separate owner decision.

**Stopping rule (from run 35, unchanged).** If no candidate here improves end to end, the gate is not changed further
and MK4 is written up as it is.

**Follow-up not covered here.** A real-VOC check with real-COCO caption profiles (run 33 source C) for whichever gate
improves. It is pre-registered separately, only if something improves here.

## 5. Checks (smoke, before the full chain)

- AV with all w_m = 1 equals RW with all w_m set to 1: max |Δ| < 1e-6.
- At capture, SN with the self-negative replaced by a copy of an existing centroid equals MK4: max |Δ| < 1e-5.
- MK4 at γ = 1 equals `jc_mneg.g_multi` (run 31): max |Δ| < 1e-6.
- Seed fidelity of arm A < 1e-2. Every map and gate is finite, and every gate is in [0, 1].
- For smoke images that are also in the run 31 dumps, the capture-time self-negative gate is compared with the
  dump-based one. This is printed only: regeneration is not guaranteed to be pixel-identical.

## 6. Files and cost

- **Script:** `scripts/jc_gate36.py`.
- **Reports:**
  - `runs/36_sn_consensus/stage1.md` and `stage1.json` (with γ\* and the weights);
  - `runs/36_sn_consensus/result.md`.
- **Data:** `<CACHE>/jc_gate36/`.

| Part | Cost estimate (run 34 / 35 timings) |
|---|---|
| stage 1 | under 30 CPU-minutes |
| capture | about 6 GPU-hours (4 shards) |
| scoring | about 5 CPU-hours over 6 jobs |

- The whole chain takes about 4 h.
- Smoke outputs are tagged `_smoke` and kept.
