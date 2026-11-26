# Spatially adaptive cross-attention layer weighting, and signed temporal cross-attention contrast

Written and committed before any image of this run is captured or scored. Owner request (2026-11-18). Exploratory:
the direction was chosen after runs/20 results (GOCA-style layer weighting helps on COCO-Cap). No result here is
promoted into BRACE; anything that helps needs a fresh annotated set.

## Question

BRACE's seed (A3) is a weighted sum of 11 cross-attention (CA) layer maps over 31 denoising calls, with **one weight
per layer per call** (GOCA-style, from self-attention similarity). Two questions:

1. **Experiment 1 (primary).** Does a layer's reliability vary across the image, and does a training-free,
   label-free per-location reliability gate on top of the global weights improve the seed? Locations are pixels,
   regions from SD's own self-attention, or fixed grid cells (control).
2. **Experiment 2 (secondary).** Do signed differences between early and late CA (both signs) carry information
   about the seed's errors? Diagnostic first; one fixed, untuned intervention per sign is reported for the table.

Both run from **one capture** and are analysed independently. They are not combined.

## Prior overlap (protocol and runs)

- SG8 / SG11 / SG12 / SG23: per-pixel agreement, trimming and leave-one-out among **self-attention** observations,
  not CA layers.
- SG24: A2 = GOCA projected-output head weights (worse on both sets); A3 = global per-layer weights (BRACE seed). No
  per-location CA layer weighting.
- CA-T: CA call windows (single calls, disjoint/trailing/cumulative windows, leave-one-call-out) kept all calls 5/5;
  every window used positive weights only. No signed early−late contrast.
- SG18 / Pre-SG18: temporal windows of **self-attention** propagation; late calls added net FP on VOC-Sim but
  removed net FP on COCO-Cap, so no expected sign.
- Literature (searched 2026-11-18): GoCA (arXiv 2603.06178) has global layer/head aggregation and per-pixel
  *token-mass* rescaling; DiffSegmenter (2309.02773) global layer weights + SA affinity refinement; FA-Seg (2506.23323)
  fixed per-resolution weights; DAAM aggregates all timesteps. No per-location CA layer reliability found. This is
  a search, not a novelty proof.

## Data and capture

- runs/20 corrected prompts (`configs/new_reruns_prompts.json`), safety checker off, frozen runtime
  (`scripts/sg27_runtime.py`), SD 1.5 fp32, 30 steps, CFG 7.5, manifest seeds. 490 VOC-Sim, 707 COCO-Cap.
- A subclass of the frozen recorder also stores, for every call t (31) and selected CA module l (11), the
  class-word CA map A_{t,l} (mean over heads, as A3) bilinearly upsampled to 64×64, and the call's GOCA-style layer
  weights w_{t,l}. **Fidelity gate:** Σ_{t,l} w_{t,l} A_{t,l} / 31, min–max, must equal the runtime's A3 seed
  (max |Δ| ≤ 1e-5), else the job stops. Image compared with the runs/20 capture image.
- Per-layer maps saved (float16, ≈ 2.8 MB/image) so later analyses need no regeneration.

## Experiment 1: reliability gate

Label-free per image. P = mean over the 11 calls (10–20) × 8 heads of the frozen 64×64 SA module (the fine SA
operator BRACE already uses), row-stochastic, 4096×4096.

- Per-map local inconsistency: e_{t,l}(i) = |A_{t,l}(i) − (P A_{t,l})(i)| / σ_{t,l}, σ_{t,l} = median_i of the
  numerator (+1e-12).
- Pooling over a partition 𝒰 (each pixel i in unit U(i)): ē_{t,l}(i) = mean_{j ∈ U(i)} e_{t,l}(j);
  reliability r_{t,l}(i) = exp(−ē_{t,l}(i)).
- Gated seed: S_r(i) = Σ_{t,l} w_{t,l} r_{t,l}(i) A_{t,l}(i) / Σ_{t,l} w_{t,l} r_{t,l}(i). With r ≡ 1 this is BRACE's seed.

Partitions (all label-free, region counts matched between SA regions and grid):

| ID | Partition |
|---|---|
| Gpx | pixels (primary) |
| Gs16, Gs64, Gs256 | k-means (k = 16, 64, 256) on the rows of P; Euclidean, 30 iterations, deterministic init (k evenly spaced pixel rows) |
| Gg16, Gg64, Gg256 | square grid 4×4, 8×8, 16×16 cells |

Controls in the same capture:

- **BR**: BRACE seed (runtime A3). **UN**: uniform layer weights from the same stored maps (Σ A / (31·11)).
- **Shuffled** for every partition (IDs with suffix `x`): the units' reliability values are permuted among units
  of the same partition with one fixed permutation per image (crc32(id) seed), identical across (t, l). Keeps the
  value distribution and inter-layer structure; breaks the location.

## Experiment 2: temporal contrast

E, M, L = A3-weighted CA seeds over calls 0–9, 10–20, 21–30 (the protocol's fixed stages), each min–max.

- Seeds: **TE**, **TM**, **TL** (windows, references); **TD+** = mm(max(Ŝ + Ê − L̂, 0)) and
  **TD−** = mm(max(Ŝ + L̂ − Ê, 0)), Ŝ = min–max BRACE seed; coefficient fixed at 1, **not tuned**.

## Every seed through the same table

Each seed above through: seed alone, fine-only (V0), BRACE hierarchy (V3), NERVE walk (NERVE's layers on NERVE's
noised pass, as runs/20). Each cell raw and with DenseCRF R2, at the VOC-selected threshold (5-fold OOF on
VOC-Sim; full-VOC-Sim best applied unchanged to COCO-Cap) and at the fixed 0.55. COCO-Cap best threshold reported as
audit only. Class-macro mIoU, image-bootstrap 95% intervals (2000).

## Diagnostics (pixel level, 64×64, GT used only for scoring)

Pixel classes from BRACE's hierarchy output (min–max V3 of BR, raw) at 0.55 against GT (area fraction ≥ 0.5 at
64×64): predicted foreground → FP vs TP; predicted background → FN vs TN. 1024 pixels per image, uniform, fixed seed.

- Base model: logistic regression on [Ŝ, V̂3] (BRACE seed and hierarchy output, i.e. the seed and its confidence).
- Exp. 1 augmented: base + [R̄, R_A], with R̄(i) = Σ w r / Σ w and R_A(i) = Σ w r A / Σ w A (pixel reliability, Gpx).
- Exp. 2 augmented: base + [Ê, L̂] (covers both signs of the contrast). Also univariate AUC of D = Ê − L̂ and of −D.
- VOC-Sim: 5-fold OOF by image fold (crc32 mod 5). COCO-Cap: fit on all VOC-Sim, applied unchanged.
- **Diagnostic passes** for an experiment if, on VOC-Sim OOF, ΔAUC (augmented − base) ≥ 0.01 with image-bootstrap
  (2000) lower bound > 0, for FP-vs-TP or for FN-vs-TN. COCO-Cap ΔAUC reported, not used.
- Agreeing-wrong check (Exp. 1): share of FP pixels where ≥ 6 of 11 call-averaged layer maps (each min–max) are
  ≥ 0.5; mean R̄ on FP vs TP pixels.

## Selection and reading

- Within Gs and Gg, k is selected by VOC-Sim OOF mIoU of hierarchy + DenseCRF at the VOC-selected threshold (VOC
  numbers of the chosen member are in-sample).
- Each candidate (Gpx, chosen Gs, chosen Gg, TD+, TD−) is compared with BR, and with its shuffled control
  (Exp. 1), at hierarchy + DenseCRF.
- **"Helps"** (as runs/20): COCO-Cap Δ vs BR > 0 with interval above 0, VOC-Sim Δ ≥ 0, COCO-Cap best-threshold Δ > 0.
  For Exp. 1 also: Δ vs its shuffled control > 0 with interval above 0 on COCO-Cap, else "generic reweighting".
- **Regions vs grid:** chosen Gs − chosen Gg, with interval.
- A candidate is a BRACE-candidate only if its experiment's diagnostic passes **and** it helps; even then it needs
  fresh confirmation. If an improvement appears only on COCO-Cap it is exploratory; the VOC rule is not relaxed.
- Exp. 2 interventions are read only if its diagnostic passes; otherwise the direction is closed, no tuning.
- Descriptive: precision, recall, FP area, size thirds, share of images < 0.3 IoU, per class.

## Pipeline (`scripts/jc_layergate.py`)

`capture` (GPU, trdp) → `score` (CPU, sharded) → `diag` (CPU) → `report` → `runs/21_layer_gate/result.md`.
Smoke on 3 images (one corrected-prompt VOC-Sim, one corrected-prompt COCO-Cap, one unchanged COCO-Cap), tag
`_smoke`, kept. Up to 4 GPUs.
