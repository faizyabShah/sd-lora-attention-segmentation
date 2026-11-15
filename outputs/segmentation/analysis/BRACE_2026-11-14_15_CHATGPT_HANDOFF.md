# BRACE: all experiments of 14–15 November 2026 (cluster and A40), handoff for review

This file collects every result produced on 14–15 November 2026, on both machines:
- **Cluster (SLURM, V100s):** run by Claude.
- **A40 machine:** run by Codex.

Each result is labelled with its status: **pre-registered**, **exploratory** or **diagnostic**. The aim is to let a reviewer check the conclusions without the repository.

## 0. Context and fixed rules

**BRACE is frozen.** It is a training-free class-word segmenter on SD1.5 generations:
1. **Seed:** A3 cross-attention seed.
2. **Propagation:** V3, a capped 16→32→64 self-attention hierarchy. The cap is one-sided and applies only at 16→32.
3. **Mask:** DenseCRF R2 at τ 0.55.

**Reference scores** (class-macro mIoU):
- **VOC-Sim** (490 images, single-object prompts): **0.830**.
- **COCO-Cap** (707 images, COCO captions): **0.696** at τ 0.55, **0.707** with OOF τ.
- **CLIP-seeded variant:** 0.845 / 0.717.

**Protocol:**
- Inference is label-free; ground truth only scores.
- VOC-Sim is the development set. COCO-Cap is untuned transfer: nothing is selected on it.
- New experiments are pre-registered and committed before scoring.
- Compared variants come from the same capture.
- Failures are reported as failures.

**Evaluation:**
- Class-macro mIoU.
- OOF τ: τ is chosen on 4 of 5 folds (crc32(image_id) % 5).
- Image-bootstrap 95% CIs.

**Screening subsets:** class-stratified, 150 VOC-Sim + 150 COCO-Cap (`ideas_grouped.screen_ids`).

**Rich capture** (`ana_dump`) stores, for those 300 images:
- the image and ground truth;
- the seed and V3;
- self-attention at 16/32 (full matrices) and 64 (spectra and diffusion);
- cross-attention per layer;
- UNet up-block features at steps 5, 15 and 25;
- the x₀ trajectory.

---

## 1. Index

| # | experiment | machine | status | verdict |
|---|---|---|---|---|
| 1 | A40 setup and BRACE reproduction | A40 | setup | GPU verified; BRACE reproduces 0.830 / 0.707 inside run 2 |
| 2 | Idea 2: dose–response segmentation | A40 | pre-registered, full sets | **fails** (−0.55 COCO-Cap) |
| 3 | Map analysis part 1: BRACE error anatomy | cluster | exploratory | descriptive |
| 4 | Map analysis parts 2–5: SA spectra, CA, features, dynamics | cluster | exploratory | descriptive |
| 5 | Operator batch A: threshold and region operators | cluster | exploratory | nothing beats BRACE |
| 6 | Operator batches B/C: SA graph operators, re-noising ensemble | cluster | exploratory | nothing beyond noise |
| 7 | Regeneration pilot | cluster | exploratory, fixed bars | **all 4 bars fail**, closed |
| 8 | SA boundary diagnostic | cluster | pre-registered | **closed** (+0.040, CI crosses 0) |
| 9 | Failure-mechanism inventory | — | synthesis | written, then corrected |
| 10 | Cap mechanism | cluster | pre-registered | **not supported** |
| 11 | VAE-decoder ownership pilot | cluster | pre-registered | **closed** (gate A passes; B and C fail) |
| 12 | Grouped screen, ideas 1/3/4/6/7 | A40 | pre-registered, stage 1 | **all five stop** |
| 13 | Literature and overlap checks | — | desk review | see §13 |
| 14 | UNet feature-content bleed diagnostic | cluster | pre-registered | **running** at the time of writing |

---

## 2. A40 machine

### 2.1 Setup (14 Nov)

- **Hardware:** NVIDIA A40 (44.4 GiB), CUDA 11.8, GNU Screen, no SLURM.
- **Smoke test:** a CUDA matmul check and two SD1.5 denoising steps passed.
- **Reproduction:** no separate run was made. Run 2 recomputed BRACE inside the same process and reproduced the paper exactly: **0.830 VOC-Sim, 0.707 COCO-Cap OOF + R2**.

### 2.2 Idea 2: dose–response segmentation (pre-registered; all 490 + 707 images)

**Idea.** Scale the target token's cross-attention weight to 0.5× and 2× during generation, with the same seed. Map each pixel's response, in either final-latent or decoded-pixel space, starting from step 0 or step 10.

| map | VOC-Sim OOF + R2 | COCO-Cap OOF + R2 | Δ COCO vs BRACE (CI) |
|---|---:|---:|---|
| BRACE | 0.830 | 0.707 | — |
| lat_s0 (chosen on VOC) | 0.154 | 0.153 | −0.554 (−0.580…−0.527) |
| lat_s10 | 0.037 | 0.039 | |
| pix_s0 | 0.035 | 0.039 | |
| pix_s10 | 0.001 | 0.002 | |

- **Why it fails.** The maps are extremely peaky: precision is about 0.75 and recall about 0.03–0.17. The response concentrates on a few pixels; it does not cover the object.
- **Capture caveat.** On 12/490 VOC-Sim and 20/707 COCO-Cap images, the base loop differs from the live image by more than 2/255.
  - One COCO-Cap image was a safety-checker black image (127.96/255).
  - No images were excluded.
- **Cost:** about 159 s per image, 10 GiB per worker.

### 2.3 Grouped screen: ideas 1, 3, 4, 6, 7 (pre-registered two-stage design; stage 1 on 150 + 150)

**Design.** One shared generation per image. BRACE is recomputed in the same process.
- **Pass rule:** the candidate chosen on the VOC-Sim screen must beat BRACE on the COCO-Cap screen (Δ > 0).
- **Replay fidelity:** base loop vs live, mean 0.41/255.
- **Cost:** about 380 s per image.

| idea | what it is | VOC-chosen candidate | VOC-Sim (BRACE 0.819) | COCO-Cap (BRACE 0.688) | Δ COCO (CI) |
|---|---|---|---:|---:|---|
| 1 knockout / swap | Remove (`attn`) or swap (`swap`) the class token. Measure the change in the final latent (`fin`) or the summed instantaneous change in ε_cond along the base trajectory (`ins`), from step k | swap_ins_k5 | 0.634 | 0.473 | −0.215 (−0.257…−0.168) |
| 3 chrono | k-means on each pixel's 31-step x̂₀ trajectory; each cluster gets the mean of the seed | chrono_K16 | 0.716 | 0.544 | −0.144 (−0.179…−0.112) |
| 4 Jacobian | VJP of ε_cond with seed-weighted random probes, at step 10 or 20 | jac_s10 | 0.717 | 0.582 | −0.106 (−0.133…−0.077) |
| 6 optimal transport | Sinkhorn assignment of pixels to prompt tokens at 32 px; `ot_v3` uses it as the seed of V3 | ot_v3 | 0.739 | 0.583 | −0.105 (−0.150…−0.051) |
| 7 CFG contrast | Σ‖ε_c − ε_u‖ from step k | cfg_k0 | 0.631 | 0.402 | −0.286 (−0.332…−0.242) |

**Decision:** all five groups stop, and stage 2 is not run.

**Other candidates (VOC-Sim / COCO-Cap):**
- knockout `fin` maps: 0.47–0.53 / 0.32–0.35;
- `ins` maps: 0.61–0.63 / 0.47–0.48;
- chrono K4 / K8: 0.600 / 0.401 and 0.665 / 0.469;
- jac_s20: 0.710 / 0.572;
- plain `ot`: 0.642 / 0.498;
- cfg_k10: 0.615 / 0.396.

**What it shows:**
- **The causal effect of the class word is worse segmentation evidence than its attention.** Letting the knocked-out trajectory run to the end (`fin`) is worse than the instantaneous effect (`ins`), because the scene itself changes. The regeneration pilot (§3.5) found the same thing.
- **The two best candidates reuse BRACE's ingredients** (an OT seed fed into V3; seed propagation through the network Jacobian), and both are still about 0.10 below BRACE.
- **Descriptive, post hoc, about 7–8 images per class:**
  - chrono K16 beats BRACE on VOC-Sim sheep (0.92 vs 0.70), the class that broke BRACE's worst-class guard; also cow 0.93 vs 0.92.
  - On COCO-Cap dining table, every group beats BRACE (0.38–0.56 vs 0.33).
  - ot_v3 on COCO-Cap aeroplane: 0.91 vs 0.71.
  - None of this is a basis for selection.
- **COCO-Cap token rule.** On images where the caption contains the class word (63%), BRACE scores 0.720. With the label-free fallback token (37%) it scores 0.646.

---

## 3. Cluster

### 3.1 Map analysis part 1: where frozen BRACE loses (stored paper capture, 490 / 707)

**Bounds** (pooled class-macro mIoU, VOC-Sim / COCO-Cap):
- **BRACE:** 0.830 / 0.707 (OOF).
- **Oracle per-image τ + R2:** 0.902 / 0.793.
- **Pixel ranking (AUROC):** V3 0.981 / 0.927. The ranking is already good; the losses come from the threshold and the seed.

**Error anatomy:**
- **VOC-Sim:** 460/490 images have IoU ≥ 0.5. Of the failures, 20 are bleed, 8 are part-only and 2 mixed.
  - Dining table alone contributes 18 bleed failures (class IoU 0.40, recall 0.97, precision 0.41).
  - Sheep, potted plant and tv fail as part-only.
- **COCO-Cap:** 142 failures: 97 bleed, 12 wrong region, 18 mixed, 15 part-only.
  - The smallest-area quintile has IoU 0.528, with far-away false positives 8.7× the object area.

**Threshold calibration:**
- Per-image best τ: mean 0.575 (VOC-Sim) / 0.594 (COCO-Cap), sd about 0.14.
- No label-free statistic predicts it: |ρ| ≤ 0.24.
- Otsu thresholding is worse than fixed 0.55: 0.828 vs 0.835 on VOC-Sim, 0.663 vs 0.706 on COCO-Cap.

**Background-word maps don't flag BRACE's false positives:** AUROC 0.37 (VOC-Sim) and 0.49 (COCO-Cap).

### 3.2 Map analysis parts 2–5 (rich capture, 150 + 150)

**Seed vs propagation** (best-threshold IoU; then IoU at τ 0.55 + R2):

| map | VOC-Sim | COCO-Cap |
|---|---|---|
| raw A3 seed | 0.777 | 0.673 |
| BRACE (V3 from the A3 seed) | 0.892; 0.825 | 0.798; 0.710 |
| V3 from an oracle seed | 0.912; 0.887 | 0.869; 0.836 |

The seed is the bottleneck, especially on COCO-Cap.

**Self-attention spectra:**
- The best single eigenvector reaches 0.85 / 0.80 best-threshold IoU. This needs an oracle choice of both the eigenvector and the threshold.
- Spectral clustering with K = 8, labelled by the seed, reaches 0.84 at best, below V3.
- Repeated seed diffusion Aᵈs peaks at d = 2 (0.888), below V3 (0.892).

**Cross-attention:**
- The mean over all layers and heads reaches 0.821 / 0.737 best-threshold IoU.
- The stack of maps is close to rank 1: PC1 explains 62% of the variance.

**UNet decoder features (up_blocks 1 and 2, steps 5/15/25):**
- **Label-free extraction is weak:**
  - top-3 principal components: VOC-Sim 0.64–0.69, COCO-Cap 0.44–0.56;
  - k-means with K = 8, with an oracle picking the cluster: 0.65–0.74 / 0.49–0.58.
- **Probes trained on ground truth are strong:**
  - a within-image probe on up2 reaches 0.94 / 0.90 IoU. This is an upper bound: the checkerboard train/test split leaks, because features are spatially smooth;
  - a cross-image linear probe reaches pixel AUROC 0.96 (VOC-Sim test images) / 0.89 (COCO-Cap).
- **So the features contain the information, but nothing label-free extracts it.**

**Dynamics:** the x₀ settling step barely differs inside and outside the object: 25.8 vs 27.0 (VOC-Sim), 25.4 vs 26.4 (COCO-Cap).

### 3.3 Operator batch A (stored maps, all images; exploratory)

| operator | VOC-Sim Δ | COCO-Cap Δ |
|---|---:|---:|
| stability-selected τ (with CRF) | −0.011 | −0.036 |
| area-matched τ (constant fitted on VOC-Sim only) | −0.075 | −0.063 |
| connected components (keep mass ≥ 0.5 / top-1 / component at the seed peak) | 0.000 / −0.001 / 0.000 | 0.000 / −0.014 / −0.010 |
| geodesic gating | −0.075 | −0.061 |
| matting (+ R2) | −0.029 | −0.073 |

### 3.4 Operator batches B and C (150 + 150, same capture; exploratory)

Everything is within noise of BRACE or worse:

| operator | VOC-Sim Δ | COCO-Cap Δ |
|---|---:|---:|
| personalised PageRank (best) | −0.003 | −0.026 |
| fg/bg competition | −0.094 | −0.147 |
| random walker | −0.33 | −0.37 |
| graph cut + R2 | −0.018 | −0.043 |
| low-pass then V3 | +0.004 | −0.015 |
| attention sinks / temperature | −0.010 / −0.012 | −0.026 / +0.004 |
| SA-kernel CRF, w1 | +0.005 | +0.003 |
| rollout | −0.031 | −0.088 |
| re-noising ensemble (mean of 200/400) | +0.004 | +0.005 |

### 3.5 Regeneration pilot (20 images; exploratory, pass bars fixed before the run)

**Idea.** Replay the generation from step 8 while forcing the class token's cross-attention to follow a candidate mask M:
- outside M: probability 0;
- inside M: a floor at the 90% quantile;
- conditional branch only.

Then score M by D, the error of the replayed image against the original.

**Result:** **all four bars fail.**
- **Ground truth vs controls:** the ground-truth mask had lower D than shift on 15/20 images, flip 18/20, dilated by 8 px 13/20 and eroded by 8 px 12/20. The bar required 15/20 against all four.
- **Rank correlation:** the median within-image ρ(D, IoU) over the τ masks is −0.09; the bar was ≤ −0.5.
- **τ selection:** choosing τ by lowest D loses −0.029 IoU against τ 0.55, and is better on only 5/20 images.
- **Residual:** after regressing out area and intervention strength, the median ρ is −0.26; the bar was ≤ −0.3.
- **Sanity check:** the unconstrained replay reproduces the image (D ≈ 10⁻⁶).

**Mechanism.** The error localises coarsely and prefers SD's own attention support. A ground-truth mask dilated by 16 px often had the lowest D. **Attention support is wider than the object.**

### 3.6 SA boundary diagnostic (pre-registered, + amendment 1; 150 + 150; native 16/32 grids)

**Question.** Does SD self-attention affinity separate boundary-crossing cell pairs, and BRACE's bleed cells, better than RGB does?

**Amendment 1** (after a 10-image code test, before the full run): with the ground-truth object as reference, Q3(b) was degenerate, because RGB plus distance scores 1.000 by construction. It was reduced to a single-feature report.

**Primary setting: 32 px, block 1 (steps 10–19).**

| question | VOC-Sim | COCO-Cap |
|---|---|---|
| Q1: edge AUROC, SA vs colour vs contour | 0.789 vs 0.854 vs 0.815 | 0.739 vs 0.844 vs 0.805 |
| Q2: Δ OOF AUROC, RGB → RGB + SA on edges | +0.023 (+0.015…+0.032) | +0.015 (+0.008…+0.022) |
| **Q3(a): bleed vs correct cells, BRACE-core reference (decisive)** | +0.047 (+0.014…+0.082) | **+0.040 (−0.010…+0.089)** |
| Q3(b): single −SA feature, ground-truth reference (distance-confounded) | 0.896 | 0.897 |

**Decision rule:** Δ ≥ +0.05 with CI > 0 on COCO-Cap → **CLOSE**. No graph-cut or propagation variant was built.

### 3.7 Failure-mechanism inventory (`runs/FAILURE_INVENTORY.md`)

**Seed errors** dominate on COCO-Cap:
- The oracle seed is worth +0.06 / +0.13 IoU at τ 0.55 + R2 on the screening subsets (§3.2).
- What's missing is label-free selection. Every label-free selector of word, seed or threshold has failed. Among them:
  - content-token rescaling;
  - head weights;
  - caption capture (−0.004) and caption inversion (−0.033);
  - the best other caption word: an oracle gains +0.03; label-free picks gain nothing; an entropy pick loses −0.10;
  - fusion with the CLIP seed;
  - reliability gating.

**Bleed** is attention support, not a propagation bug: all propagation operators fail, and the pilot shows SD's support really is wider than the object.

**Boundaries:** R2 is the only boundary step that has ever helped.

**Per-image τ:** the oracle is worth +7 / +10 points, but no label-free signal predicts it.

**Correction (made the same day).** The inventory first called the VOC-Sim ↔ COCO-Cap trade-off "the most consistent unexploited pattern". That trade-off came from SG29 N2, SG30 S2 and SG31 S1, which help COCO-Cap and hurt VOC-Sim. **But it was measured against the old V0/S0 baseline** (COCO-Cap 0.662). BRACE's cap is itself a tighter form of propagation and already takes most of that gain without a VOC-Sim cost:

| method | VOC-Sim OOF | COCO-Cap |
|---|---:|---:|
| S0 + R2 | 0.823 | 0.662 |
| SG30 S2 + R2 | 0.813 | 0.686 |
| SG31 S1 + R2 | 0.817 | 0.698 |
| V3 + R2 (BRACE) | 0.829 | 0.697 |

Further tightening on top of V3 was tested once (SG-combined-v3): a per-head cap was null, and a temporal cap was worse.

### 3.8 Cap mechanism (pre-registered; all 490 + 707 images; stored paper capture, so V0 and V3 come from one capture)

**Question.** Does the cap help most where competing objects are present?
- **Competition score:** label-free SCLIP (VOC21) competitor coverage.
- **Outcome:** ΔIoU between V3 + R2 and V0 + R2. V0 + R2 uses τ₀ = 0.45, chosen by the full-VOC rule; the same rule gives 0.55 for V3.
- **Gate:** the recomputed counts match the stored ones exactly.
- **Primary region.** The η 0.5 score was zero on 75% of COCO-Cap images, so the fixed fallback rule switched the primary region to "SCLIP argmax is another class".

**H1 (image level):** β per standard deviation of the score, controlling for log area and class.
- COCO-Cap, primary region: −0.003 (CI −0.013…+0.006).
- Every score variant on both sets: within ±0.01.
- Class-cluster bootstrap: the same.
- **Competition does not predict the gain.**

**H2 (pixel level):** Mantel–Haenszel log odds ratio that a V0 false-positive pixel is removed by V3 when it lies on a competitor, stratified by distance from the object.
- COCO-Cap, primary region: +0.27 (CI −0.28…+0.77).
- η 0.5 region (secondary): +0.96 (CI +0.08…+1.62), but it covers only 3.4% of false positives.
- VOC-Sim at η 0.5: −2.49.

**H3 (what changes, V0 → V3):**

| set | mIoU | precision / recall (pooled) | false positives (% of ground-truth area) | true positives (% of ground-truth area) | mIoU if only false positives changed |
|---|---|---|---:|---:|---:|
| VOC-Sim | 0.824 → 0.830 | 0.895 / 0.901 → 0.890 / 0.910 | +0.7% | +0.9% | 0.823 |
| COCO-Cap | 0.673 → 0.696 | 0.724 / 0.876 → 0.747 / 0.885 | −3.4% | +0.9% | 0.691 |

- On COCO-Cap, false positives fall by about 10% both on and off competitor regions (on: 2.59M → 2.33M px; off: 11.71M → 10.53M px).

**Verdict: not supported.** The cap reduces over-expansion in general: it removes false positives on COCO-Cap and raises recall on VOC-Sim. It does not target competing objects specifically.

### 3.9 VAE-decoder ownership pilot (pre-registered, + amendment 1; 30 screening images, excluded from any confirmation set)

**Method.**
- **Latent:** the final latent is regenerated exactly. Gate 0 checks the decode against the stored image: max mean |Δ| 0.24 grey levels.
- **Perturbations:** BRACE-core cells F and clear-background cells B near the boundary band are perturbed with random directions, ε = 0.05 × the mean per-cell latent norm.
  - Every cell gets the same magnitude, and F and B have equal counts, so total energy is equal.
  - 16 draws per set, in 2 seed sets.
- **Ownership:** o(p) = e_F / (e_F + e_B).
- **Footprint control:** built from the decoder's own average single-cell influence kernel.

**Amendment 1** (after a 2-image code test): F and B were equal in count but not in distance (2.8 vs 3.6 cells from the band). The kernel decays by about e every 4 px, so they are now matched ring by ring. The test numbers were not used.

| gate | bar | result |
|---|---|---|
| A. beats the footprint (band-pixel AUROC) | ≥ +0.02, CI > 0 | **+0.105 (+0.046…+0.159), pass** |
| B. beyond RGB + footprint (OOF) | ≥ +0.05, CI > 0 | +0.004 (−0.012…+0.016), fail |
| C. fixed o > 0.5 correction in the band | ≥ 15/22 wins, mean > 0, beats the footprint rule | 2/22, −0.109 (footprint rule −0.152), fail |

- **Repeatability:** ρ 0.94–0.96 between seed sets.
- **Exclusions:** 8 of 30 images had too few matched cells (< 4 per set).
- **Within cells crossed by the true boundary:** decoder ownership AUROC 0.720, footprint 0.708, RGB colour 0.721.

**Verdict: closed.** The decoder's response depends on content, but that content information is what RGB already carries.

---

## 4. Literature and overlap checks (desk review, 15 Nov)

| direction | closest prior work | our overlap | decision |
|---|---|---|---|
| Pairwise token interactions | **DiffusionPID** (NeurIPS 2024): pixel-level redundancy and synergy between prompt tokens. **Kong et al.** (ICLR 2024): pixel-level conditional mutual information and object localisation | A40 knockout / swap | **Dropped.** The signal exists in prior work, and the A40 screen shows single-token interventions are weak segmentation evidence |
| Time-resolved causal influence | Temporal Concept Dynamics via Prompt-Conditioned Interventions (arXiv 2512.08486) | grouped screen; x₀ settling barely differs (25.8 vs 27.0) | **Dropped**, for the same reason |
| VAE-decoder attribution | Nothing direct found. Related: latent-space mask refinement (DiffDIS), mask VAEs (JoDiffusion), decoder influence for inversion | — | Tested (§3.9), **closed** |
| UNet feature content | OVDiff (ECCV 2024), FreeDA, DiffCut, DIFT | parts 2–5: supervised probes strong, label-free weak; earlier M1/M2 prototypes used attention queries, not features | **One final diagnostic** (§5) |
| Temporal feature correspondence | DIFT (temporal correspondence), Diffusion Hyperfeatures | only 3 stored steps | Secondary feature in §5 only |
| TRACE (ICLR 2026 oral, arXiv 2503.07982) | Instance emergence point plus Attention Boundary Divergence (row-vs-row divergence of opposite neighbours), distilled into an edge decoder | Our SA tests used direct affinity A_ij, not row divergence | Secondary feature in §5; reproducing it is not novel |
| BRACE seeds + SAM | Text-to-Seed (arXiv 2608.26624, 2026): SD attention seeds expanded by SAM | — | A baseline, not a contribution |
| Adaptive synthetic-data generation | What Makes Synthetic Data Effective in Image Segmentation (ICML 2026, SENSE) | — | Deferred (needs downstream training) |
| Controlled generative interventions; part-whole composition | — | regeneration pilot; SG4 | Deferred |

---

## 5. Pending: UNet feature-content bleed diagnostic (pre-registered; CPU; 150 + 150; running)

**Declared the last diagnostic in this family.** If it fails, the direction closes with no sweep.

**Data and task:**
- **Features:** `feat/up2/s15` (640 channels at 64 × 64), fixed a priori.
- **Cells:** inside BRACE + R2 and not in the core F. The task separates bleed (ground-truth background) from correct (ground-truth object).

**Baseline features:**
- Lab distance to the means of F and B;
- intervening contour;
- distance to F;
- BRACE confidence;
- depth inside the mask;
- mask area;
- a matched within-image RGB classifier trained on F vs B.

**Test features:**
- prototype cosine contrast;
- a within-image logistic regression on 32 principal components of the features, trained only on the label-free cores F vs B.

**Gates:**
- **Gate 1:** COCO-Cap Δ OOF AUROC ≥ +0.05, with CI > 0.
- **Gate 2:** the fixed correction (remove cells with feature probability < 0.5) must improve IoU over BRACE + R2 on COCO-Cap with CI > 0, and beat two **count-matched** controls (cells chosen by the RGB classifier, and by lowest BRACE confidence). The VOC-Sim guard requires a gain of at least −0.005.

**Secondary (not deciding):** TRACE-style row divergence, and consistency across steps 5/15/25.

**Code test (8 images, not evidence):** the rule removes many cells per image (134 on VOC-Sim, 365 on COCO-Cap). Result to follow: `runs/08_feat_bleed/result.md`.

---

## 6. What the two days show

1. **BRACE's ingredients are the strongest use of SD signals we have found.**
   - Causal and generation-side signals score far below BRACE's attention-plus-propagation maps: dose–response, knockout / swap, CFG contrast, Jacobians and x₀ trajectories.
   - The best of them reach about 0.72–0.74 on VOC-Sim, against 0.82.
2. **The remaining error is mostly missing information.**
   - **Seed:** the oracle is worth +0.06 / +0.13, with no label-free selector.
   - **Per-image τ:** the oracle is worth +0.07 / +0.09, with no predictor.
   - **Boundaries:** SA, the VAE decoder and regeneration add nothing usable beyond RGB + DenseCRF.
3. **Bleed comes from where SD attends.** SD's class support is wider than the object (regeneration pilot), and propagation operators cannot remove it.
4. **Paper mechanism statement.** The cap reduces over-expansion in general, not leakage onto competing objects. On COCO-Cap most of its gain is fewer false positives; on VOC-Sim it is recall.
5. **Unless §5 passes,** the agreed next step is to stop incremental BRACE diagnostics and finish the paper. The negative results go into its limitations and analysis sections.

## 7. Caveats

- Screening subsets (150 per set) have about 7–8 images per class. Class-level observations are descriptive only.
- Cross-GPU regeneration is not pixel-identical. BRACE numbers on the same screening subset differ between machines and between metrics:
  - A40: class-macro OOF + R2, 0.819 / 0.688;
  - cluster rich capture: per-image mean IoU at τ 0.55 + R2, 0.825 / 0.710.
  - Every comparison here is within one capture.
- COCO-Cap was used post hoc only, to explain existing gains (§3.8) or as a pre-specified transfer gate. Nothing was selected on it.
- The images used for the regeneration pilot (20) and the VAE pilot (30) are excluded from any future confirmation set.
- Everything is committed locally; nothing is pushed.
