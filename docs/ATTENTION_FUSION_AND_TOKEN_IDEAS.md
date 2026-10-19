# Phase 3 experiment registry: propagation, fusion, refinement, tokens and comparisons

**Status:** registered plan, Claude draft 2 (merges Claude's backlog, Codex's
propagation package of protocol Section 37, and Faiz's `plan.tex`). Codex
review requested in protocol Section 38.
**Starts:** after the CA and SA freeze of protocol Section 11, confirmed by the
joint CA×SA audit (Sections 34, 36, 38).
**How to use:** every experiment has an ID. Its grid is registered here before
it runs. When it runs, set its status and link the output folder, jobs and
commit in the log (Section 9). A result enters the paper only through an ID in
this file.

**OVAM clarification:** the all-ones-value self-attention hook is an artifact
of the upstream OVAM repository/reference-fidelity baseline. It is not the
QK-derived operator used by our controlled SA experiments, and repairing or
replacing that upstream hook is not planned work. Ideas below that use genuine
SA affinities are independent fusion proposals rather than fixes to OVAM.

**Notation.**

- `c` is the frozen CA seed (`C/R`, 64×64, min–max normalized).
- `A_i` is the row-stochastic conditional SA affinity `softmax(Q Kᵀ / √d)` of
  SA module 15 (64×64) for observation `i` (one call × head; 31 × 8 = 248
  observations). `Ā = mean_i A_i`.
- **REF** is the frozen method, `y = mean_i(A_i c)`: VOC-Sim mIoU 0.7887 at
  0.45.
- `n` is a number of propagation passes.

Citations: names tagged **[OVAM n]** are in the OVAM paper's reference list
(arXiv:2403.14291). Every other method name is marked *(to verify)* and must be
checked before it is cited.

---

## 0. Registered rules for every phase 3 experiment

These make each conclusion reportable with the same evidence as CA-C and SA-C.

| Rule | Content |
|---|---|
| R1 Data | Selection uses only the 490 VOC-Sim images. COCO-Cap (707) is never used to choose anything; it is scored once per frozen pipeline (Section 7). |
| R2 Input | The frozen pair of protocol Section 11 (CA `C/R`, SA module 15, all calls and heads). REF is the reference of every Track A screen. |
| R3 Folds and thresholds | The 5 folds of protocol Section 5a. In each outer fold, the method parameter **and** the mask threshold are chosen on the other four folds and scored on the held-out fold. Threshold grid 0.30–0.70 in steps of 0.025 (the Section 37 grid). REF is treated the same way (REF@cv) and is the comparison point; REF@0.45 and fixed 0.35/0.45/0.55 results are also reported for continuity. |
| R4 Decision | Protocol Section 5 bands (promote ≥ +0.010; conditional +0.005–0.0099 with ≥ 12/20 classes improved and no class loss > 0.05), ≥ 4/5 fold votes, positive pooled OOF gain. Each family is screened independently against REF; family winners go to the combined confirmation P3-C (Section 6). |
| R5 Ties | Within 0.002, the simpler setting wins: REF, then fewer added operations, then fewer free parameters. |
| R6 Records | Per-image tp/fp/fn at every threshold of the grid, AP, image and class IDs, fold IDs, settings, commit and environment, via the shared `ShardWriter`. Images and attention tensors are released after scoring. |
| R7 Environment | Each package runs in one environment; packages compare against their own REF row. |
| R8 Registration | Grids are committed here before the run. Any change after a run starts is logged with its reason and labelled post hoc. |
| R9 Reporting | Each package produces `report.md` and `fold_results.json` with: fold choices, OOF reference/selection mIoU, paired bootstrap 95% CI and better/worse counts, Δ mIoU at 0.35/0.45/0.55 and at the fold-chosen threshold, AP, per-class Δ, and **the number of configurations searched**. It is then added to the evidence package (`scripts/narmyn/attention/build_evidence.py`) as a ledger row, a CSV, tables and figures. |
| R10 Tracks | **Training-free** (no annotated images, no externally trained segmenter) and **few-shot** (any annotated image) are reported separately. Methods that use an external segmentation model (e.g. SAM) form a third, labelled group. |
| R11 Token leakage | No token is trained on an evaluation image of the fold it is scored on, and no token training image may be a VOC-Sim evaluation image unless the fold rule of B2b applies. |

---

## 1. Track A: training-free operators on the frozen pair

**Tier:** 1 = needed for the paper's main claims; 2 = should run if time allows;
3 = could run later or appear as future work.

### 1.1 Propagation operators (Codex package, protocol Section 37)

| ID | Method | Definition | Registered grid | Targets | Owner | Tier | Status |
|---|---|---|---|---|---|---|---|
| P1 | Observation-wise power | `mean_i(A_iⁿ c)` | n = 1–4 | over-focus | Codex | 1 | Complete: n=2 promoted to P3-C; protocol Section 43 |
| W3 | Average-before power | `Āⁿ c` | n = 1–4 (n = 1 must equal REF) | — | Codex | 1 | Complete: REF retained; protocol Section 43 |
| P2 | Restart (personalized PageRank) | `x ← (1−β) A x + β c` from `x₀ = c`, both aggregation orders | β ∈ {0.10, 0.25, 0.50, 0.75}; iterations ∈ {2, 4, 8} | over-focus and leaking | Codex | 1 | Complete: REF retained for both orders; protocol Section 43 |
| P6 | Sparse affinity | row-wise top-k of `A_i`, renormalized, one pass | k ∈ {16, 32, 64, 128, 256, 512, 1024} | leaking | Codex | 1 | Complete: REF retained; protocol Section 43 |
| P3 | Heat kernel | `Σ_{k=0}^{8} e^{−t} tᵏ/k! Āᵏ c` (truncated at 8 terms) | t ∈ {0.5, 1, 2, 4} | removes the hard choice of n | Codex | 2 | Complete, Section 83: REF retained 5/5; best VOC 0.7872 (`t=4`) |
| P5 | Symmetric affinity | `(D^{−½} W D^{−½})ⁿ c`, `W = (Ā + Āᵀ)/2` | n = 1–4 | — | Codex | 2 | Complete, Section 83: REF retained 5/5; best VOC 0.7895 (`n=2`, +0.0007) |
| P9 | Seeded progressive expansion (SeeDiff-style *(to verify)*) | threshold `c` into confident seeds, expand through SA from coarse to fine modules, then a background expansion stage | seed threshold ∈ {0.5, 0.7, 0.9}; module path; background stage on/off | over-focus and leaking | Codex | 3 | Complete, Section 83: REF retained 5/5; `s=.5,bg=off` VOC 0.7783, COCO 0.6490 |
| P7 | Rollout across modules | `A_L ⋯ A₁ c` | module order | — | — | — | Superseded: SA-C froze a single module |
| P8 | Column propagation | `Aᵀ c` | — | — | — | — | Done (SA package B): far worse |

### 1.2 Fusion and calibration (Claude package)

All take the min–max normalized `c` and `y = REF` as inputs.

| ID | Method | Definition | Registered grid | Targets | Owner | Tier | Status |
|---|---|---|---|---|---|---|---|
| P4 | Residual mix | `λ y + (1−λ) c` | λ ∈ {0.25, 0.50, 0.75, 0.90} (λ = 1 is REF) | leaking | Claude | 1 | Done (protocol Section 44): not promoted |
| F3 | Gating | `c ⊙ y` | — | leaking | Claude | 1 | Done (Section 44): not promoted (0.514 at 0.45) |
| F4 | Geometric mean | `√(c ⊙ y)` | — | leaking | Claude | 1 | Done (Section 44): not promoted |
| F2 | Column-mass gate | `y ⊙ norm_[α,1](Σ_i A_{ij})` | α ∈ {0, 0.25, 0.50, 0.75} | background suppression | Codex; Claude reviews | 2 | Complete, Section 83: REF retained 5/5; all candidates below REF |
| N2 | Percentile normalization | percentile 1–99 clip, then rescale, before thresholding | — | calibration | Claude | 1 | Done (Section 44): not promoted |
| N3 | Max normalization (OVAM) | `y / max(y)` | — | calibration | Claude | 1 | Done (Section 44): not promoted (class losses > 0.05) |
| N4 | Per-image adaptive threshold | Otsu threshold per image instead of a global one | — | calibration; per-image scale | Claude | 1 | Done (Section 44): conditional-band gain, rejected by the class-loss rule (`sheep` −0.07) |
| BG1 | Background map | propagate the start-token CA map the same way; foreground where `y > γ · bg` | γ ∈ {0.50, 0.75, 1.0, 1.25, 1.5} | background handling (Faiz's plan) | Codex; Claude reviews | 2 | Complete, Section 83: REF retained 5/5; direct BOS comparison fails badly |
| F1 | Upstream OVAM fusion | `c ⊙ norm_[α,1](S₁)` with the all-ones value tensor | α | — | — | — | Reference-fidelity baseline only (`ovam_full.py`) |

### 1.3 Label-free weighting of observations

| ID | Method | Definition | Registered grid | Owner | Tier | Status |
|---|---|---|---|---|---|---|
| W1 | Entropy weighting (NERVE-style *(to verify)*) | weight each `A_i c` by `1 / entropy(A_i c)` | temperature ∈ {0.5, 1, 2} | Codex; Claude reviews | 2 | Complete, Section 83: REF retained 5/5; effectively identical (+0.0002 max) |
| W2 | Agreement-based heads | per image, keep the heads whose `A_i c` correlates most with `c` | keep ∈ {2, 4, 6} heads | Codex; Claude reviews | 2 | Complete, Section 83: REF retained 5/5; selecting fewer heads hurts |
| W4 | Iterate to convergence | alternate `Ā` and min–max until the change is < 10⁻⁴ | max iterations 50 | Codex | 3 | Complete, Section 83: REF retained 5/5; convergence collapses VOC to 0.6113 |
| H1 | Hierarchical multi-resolution CA/SA fusion (FA-Seg-style *(to verify)*) | learned-free weighted sum over resolutions | weights by resolution | Codex | 3 | Complete, Section 83: REF retained 5/5; VOC 0.7813, COCO 0.6270 |

### 1.4 Segment then label (object-aligned boundaries)

| ID | Method | Definition | Registered grid | Owner | Tier | Status |
|---|---|---|---|---|---|---|
| G3 | k-means on SA rows | cluster the 4096 rows of `Ā`; a segment is foreground if its mean `y` ≥ threshold | K ∈ {4, 8, 16} | Claude | 2 | Covered by standalone SG4, protocol Section 81 |
| G2 | Spectral clustering | eigenvectors of the symmetric `Ā` graph, then as G3 | K ∈ {4, 8, 16} | Claude | 2 | Covered by standalone SG4, protocol Section 81 |
| G1 | Attention-map merging (DiffSeg **[OVAM 48]**) | KL-based merging of SA rows, label by mean `y` | DiffSeg's merge threshold and anchors | Codex; Claude reviews | 3 | Complete, Section 83: REF retained 5/5; best VOC 0.6963 |
| G4 | Seeded region growing | grow from the peaks of `c` while SA similarity stays high | similarity ∈ {0.5, 0.7, 0.9} | Codex | 3 | Complete, Section 83: REF retained 5/5; best VOC 0.3729 |
| M1 | CA-weighted prototype | `p = Σⱼ cⱼ fⱼ / Σⱼ cⱼ`, mask from `cos(fᵢ, p)`, `f` = SA queries | — | Codex | 3 | Complete, Section 83: REF retained 5/5; VOC 0.3913 |
| M2 | Multiple prototypes | k-means inside high-`c` pixels, max similarity | k ∈ {2, 3, 4} | Codex | 3 | Complete, Section 83: REF retained 5/5; best VOC 0.5822 |

### 1.5 Refinement

| ID | Method | Definition | Registered grid | Track | Owner | Tier | Status |
|---|---|---|---|---|---|---|---|
| R2 | DenseCRF on RGB (OVAM's) | SimpleCRF with OVAM's parameters on the regenerated image, unary from the map | OVAM's parameters fixed | training-free | Claude | 1 | **Promoted to P3-C** (Section 44): OOF 0.7875 → 0.8049, 5/5 |
| R1 | CRF with SA edges | `y` as unary, `Ā` as pairwise affinity | pairwise weight ∈ {1, 3, 10} | training-free | Codex | 3 | Complete, Section 83: REF retained 5/5; `w=1` VOC 0.7929 (+0.0041), COCO 0.6300 |
| X1 | SAM refinement (Faiz's plan) | prompts from `y`: top points, box, coarse mask | prompt type × number of points | external segmenter (R10) | unassigned | 3 | Not run |

### 1.6 Class-adaptive choices (protocol Section 8; Faiz's plan)

| ID | Method | Definition | Owner | Tier | Status |
|---|---|---|---|---|---|
| C1 | Per-class threshold | each class picks its own threshold on the training folds | Codex; Claude reviews | 2 | Complete diagnostic, Section 83: OOF 0.7875 → 0.8348; uses VOC labels, ineligible for label-free promotion |
| C2 | Per-class operator | each class picks its operator from the family finalists | Codex; Claude reviews | 3 | Complete diagnostic, Section 83: same 0.8348 because every family froze REF; no operator contribution |

---

## 2. Track B: few-shot tokens (Codex, protocol Section 38)

A token replaces the class word in the attribution prompt. The CA seed then
uses the frozen CA configuration with that single position, and the frozen SA
propagates it. Tokens are 768-dimensional text-encoder outputs, as in OVAM.

**OVAM's released training set** is in the public archive under
`evaluation/ovam/voc_sim_train/`:

- 63 generated images ("A photograph of a ⟨class⟩"; 3 per class, 2 for `cat`,
  `horse` and `tv`), 59 hand-drawn masks, and 57 per-image tokens with loss
  curves.
- None of its seeds appears in the 490 VOC-Sim evaluation images.
- **The released `configs/ovam/tokens_VOC.npy` equals, for every class, the
  mean of that class's 2–3 per-image tokens** (row 1 of each `(2, 768)` file;
  maximum difference below 10⁻⁶). The released tokens are therefore 2–3-shot,
  not 1-shot. The OVAM paper's wording must be checked before the paper states
  the number of shots.

| ID | Method | Training images | Shots | Owner | Tier | Status |
|---|---|---|---|---|---|---|
| B1 | OVAM released tokens | OVAM training set | 2–3 (mean) | Claude (taken over, protocol Section 49) | 1 | Done (Section 50): TC promoted over the word, OOF 0.787 → 0.851 (TU 0.843) |
| B1b | OVAM per-image tokens | OVAM training set, each image alone | 1 (every choice; report mean ± range) | Claude (taken over, protocol Section 49) | 1 | Done (Section 50): single tokens 0.831 / 0.853 / 0.842 (TC, OOF) |
| B2a | Our tokens on OVAM's training images | OVAM training set, OVAM's recipe (S1) | 1 and 3 | Codex | 2 | Complete, Claude review requested (Section 64): AALL 0.8684 VOC and 0.6928 COCO |
| B2b | Our tokens from the training folds | 1 image per class from the 4 training folds, seeded choice registered before the run; 5 token sets | 1 | Codex | 2 | Complete, Claude review requested (Section 64): 0.8564 VOC; rejected; never transferred |
| S4 | Shot scaling | from B1b/B2a: 1, 2, 3 shots | 1–3 | Codex | 2 | Complete, Claude review requested (Section 64): expected 0.8472 / 0.8672 / 0.8684 |
| S3 | Train through our pipeline | loss on the SA-propagated map instead of raw CA | 1 | Codex | 3 | Not run |
| S6 | Test-time self-consistency | per image, make `c` agree with `A c`; no labels (training-free track) | 0 | unassigned | 3 | Not run |
| T1–T7, S2, S5, S7, S8 | Token variants | see Section 2.1 | — | unassigned | 3 | Not run |

### 2.1 Further token variants (Tier 3)

| ID | Method | Idea | Example |
|---|---|---|---|
| T1 | Input-space token | learn the word embedding before the CLIP encoder | Textual Inversion **[OVAM 10]** |
| T2 | Several tokens per class | k vectors per class | — |
| T3 | Per-layer / per-step tokens | one token per UNet layer or call | P+ and NeTI *(to verify)* |
| T4 | Shared context + class name | learnable context shared by all classes | CoOp *(to verify)* |
| T5 | Learned background token | replaces `<SoT>` as background | — |
| T6 | Key projections | small update of `to_k` | Key-locked editing **[OVAM 47]**, Custom Diffusion *(to verify)* |
| T7 | Unconditional embedding | per-call optimized null embedding | Null-text inversion *(to verify)* |
| S2 | Segmentation losses | Dice, Lovász, focal | — |
| S5 | Reconstruction loss | denoising loss, no masks | Textual Inversion **[OVAM 10]** |
| S7 | Classifier guidance | discriminative class tokens | **[OVAM 43]** |
| S8 | Discrete search | best real words | "Hard prompts made easy" **[OVAM 53]** |

---

## 3. Comparisons with other methods

All are scored with our metric on the same annotations. Released masks were
made on the authors' renders; our methods use regenerated images. The paper
states this.

| ID | Method | Source | VOC-Sim | COCO-Cap | Owner | Tier | Status |
|---|---|---|---|---|---|---|---|
| Q1 | OVAM (released masks: training-free and token-optimized) | OVAM archive | yes | training-free only | Claude | 1 | Scored (protocol Section 41): VOC 0.704 / 0.825; COCO 0.582 |
| Q2 | DAAM (compared in the OVAM paper) | OVAM archive: heatmaps | yes | yes | Claude | 1 | Scored with OVAM's procedure (Section 41): VOC 0.662 / 0.797; COCO 0.484 / 0.661. Fold-chosen threshold (R3, grid 0.30–0.90): VOC OOF 0.674 / 0.797; COCO with the VOC threshold 0.527 / 0.642 (Section 47) |
| Q3 | Attn2Mask | OVAM archive: heatmaps | own images | own images | Claude | 1 | Scored (Section 41): VOC 0.603 / 0.811; COCO 0.584 / 0.676 |
| Q4 | Grounded Diffusion | OVAM archive: masks and optimized masks | own images | own images | Claude | 1 | Scored (Section 41): VOC 0.621 / 0.866; COCO 0.502 / 0.732 |
| Q5 | DatasetDM | OVAM archive: masks and optimized masks | own images | own images | Claude | 1 | Scored (Section 41): VOC 0.803 / 0.806; COCO 0.593 / 0.605 |
| Q6 | OVAM with SA (released) | OVAM archive: `ovam_sa` masks | yes | yes | Claude | — | Same files as Q1 (OVAM's released masks are the `ovam_sa` DenseCRF masks) |
| Q7 | DiffSeg, DiffSegmenter, SeeDiff, FA-Seg, DiffuMask, FreeSeg-Diff *(to verify)* | reproduction on our regenerated images | if feasible | if feasible | unassigned | 3 | Not run |

The training-free baseline named in the OVAM paper, "Diffusion model is
secretly a training-free open vocabulary semantic segmenter" **[OVAM 52]**, is
Q7's DiffSegmenter.

---

## 4. Order of work

1. **Tier 1, in parallel:**
   - Codex: propagation (P1, W3, P2, P6), then tokens B1 and B1b.
   - Claude: fusion and calibration (P4, F3, F4, N2–N4) and R2 in one package,
     then comparisons Q2–Q5 (no GPU).
2. Each agent reviews the other's committed package before its result is used.
3. **P3-C** (Section 6) on the Tier 1 winners. This freezes the final
   training-free method (FTF).
4. **Token confirmation T-C:** the best token variant on REF and on FTF.
5. **Transfer** (Section 7) on COCO-Cap.
6. Tier 2, then Tier 3, as time allows; each may reopen P3-C only through a new
   registered confirmation.

---

## 5. Package outputs

| Package | Owner | Output |
|---|---|---|
| Propagation (P1, W3, P2, P6) | Codex | `outputs/segmentation/controlled_attention_cv/propagation_package/` |
| Fusion and calibration (P4, F2–F4, N2–N4, BG1, R2) | Claude | `outputs/segmentation/controlled_attention_cv/fusion_calibration/` |
| Weighting and segments (W1, W2, G2, G3, C1) | Claude | `outputs/segmentation/controlled_attention_cv/weighting_segments/` |
| Tokens (B1, B1b, B2a, B2b, S4) | Codex | `outputs/segmentation/controlled_attention_cv/tokens/` |
| Comparisons (Q1–Q6) | Claude | `outputs/segmentation/comparisons/` |
| Phase 3 confirmation P3-C | Codex runs; Claude reviews | `outputs/segmentation/controlled_attention_cv/phase3_confirmation/` |
| Token confirmation T-C | Claude (protocol Section 52) | `outputs/segmentation/controlled_attention_cv/token_confirmation/` |
| Transfer | Claude runs; Codex reviews (accepted, protocol Section 60) | `outputs/segmentation/coco_cap_transfer/` |

---

## 6. Phase 3 combined confirmation (P3-C)

The winner of each Tier 1 family (propagation; fusion/calibration; refinement)
and REF are scored together in one run and one environment. The run includes
every unique combination and the leave-one-component-out cells, as in CA-C and
SA-C. Selection follows R3–R5. The result freezes FTF and is recorded in
protocol Section 11.

**Registered expanded grid (protocol Section 45; Codex runs, Claude
reviews).** Cross the following axes for 30 configurations:

- propagation map: `REF/one_pass`, `P1/obs_n2`, `P1/obs_n3`,
  `P2/obs_b10_n4`, or `W3/mean_n2`;
- calibration: identity/global threshold, `N2/percentile` (1--99), or
  `N4/otsu`;
- refinement: no refinement or `R2/crf` with OVAM's fixed parameters.

The four mandatory confirmation cells are REF, `P1/obs_n2`, `R2/crf`, and
`P1/obs_n2 + R2/crf`. The additional cells test whether the strongest
unpromoted alternatives become useful in combination. All 30 configurations
are eligible under R3--R5; primary and additional cells are labelled
separately in the report. No setting may be added after the run starts.

**Result (protocol Section 51).** The mandatory winner is
`P1/obs_n2 + R2/crf`, selected in all five folds with 0.8135 OOF mIoU; it
freezes FTF. The expanded screen's Otsu cell reaches 0.8207 against REF but is
not added to FTF: incrementally it gains 0.0072 mIoU and loses 0.060 IoU on
sheep, failing R4's conditional class-loss guard.

---

## 7. Transfer to COCO-Cap

COCO-Cap is scored once for each frozen pipeline, with no choice made on it:

- CA only (frozen CA);
- CA + SA (frozen pair, REF);
- FTF;
- the final few-shot pipeline (VOC tokens; OVAM released no COCO tokens);
- Q1–Q6 where released.

The mask threshold is the one frozen on VOC-Sim.

---

## 8. Results already available (exploratory, before the protocol)

Earlier final method (12 CA layers with fixed top-4 heads, U3.2, heads
{0,1,2,4,5,6}, calls 15–30, percentile 1–99, threshold 0.45, 1% cleanup).
Details, reproduction and per-image counts:
`docs/narmyn/attention/TOKEN_CRF_EXPERIMENT.md`.

| Dataset | Configuration | mIoU |
|---|---|---:|
| VOC-Sim | class word, 1 pass | 0.7802 |
| VOC-Sim | OVAM token, 1 pass | 0.8347 |
| VOC-Sim | OVAM token, 1 pass, + DenseCRF | 0.8525 |
| VOC-Sim | released token-optimized OVAM masks | 0.8247 |
| COCO-Cap | class word, 1 pass | 0.6264 |
| COCO-Cap | OVAM token, 1 pass, + DenseCRF | 0.6612 |
| COCO-Cap | token-optimized OVAM (paper Table 1) | 0.692 |

Controlled reference for phase 3: REF, VOC-Sim 0.7887 (protocol Sections 33,
36).

---

## 9. Paper mapping

| Paper element | Content | Source IDs |
|---|---|---|
| Main Table 1 | Comparison on VOC-Sim and COCO-Cap, training-free and few-shot tracks separated | Q1–Q6, FTF, T-C, Section 7 |
| Main Table 2 | Controlled path: CA reference → CA-C → SA-C → P3-C, with OOF mIoU, votes, CI | protocol Sections 27, 33, 36; P3-C |
| Main figure | Calibration across thresholds for each frozen stage | CA-C, SA-C, P3-C |
| Main figure | Qualitative masks: CA, CA+SA, FTF, few-shot, OVAM | fixed image IDs chosen before results |
| Main Table 3 | Shots: 0 (FTF), 1, 2–3 | B1, B1b, B2a, S4 |
| Supplementary | Protocol summary; every package's CV report; per-class tables; every configuration (CSV); environments and commits; negative results (e.g. `Aᵀx`, CA head 1, SA-T/SA-R not surviving); tier-B exploratory record, labelled | evidence package |

---

## 10. Log

| Date | ID | Owner | Output | Commit | Note |
|---|---|---|---|---|---|
| 2026-10-13 | T-ovam, R2 | Claude | `outputs/segmentation/narmyn_token_crf/` | this branch | exploratory, pre-protocol |
| 2026-10-16 | P1, P2, P6, W3 | Codex | `outputs/segmentation/controlled_attention_cv/propagation_package/` | `8f2f73acc`; jobs `232662` → `232663` → `232664` | official controlled package, protocol Section 37 |
| 2026-10-16 | P1, P2, P6, W3 results | Codex | `outputs/segmentation/controlled_attention_cv/propagation_package/` | this commit | P1 n=2 promoted; W3/P2/P6 retain REF; protocol Section 43 |
| 2026-10-16 | registry | Claude | this file | `4ac06d9` | merged registry and rules R1–R11; protocol Section 38 |
| 2026-10-16 | Q1–Q5 | Claude | `outputs/segmentation/comparisons/released_baselines/` | this commit; CPU job `232917` | released baselines scored with OVAM's procedures; protocol Section 41 |
| 2026-10-16 | P4, F3, F4, N2–N4, R2 | Claude | `outputs/segmentation/controlled_attention_cv/fusion_calibration/` | `229130b`; job `232922` | R2 promoted; protocol Section 44 |
| 2026-10-16 | P3-C expanded confirmation | Codex | `outputs/segmentation/controlled_attention_cv/phase3_confirmation/` | this commit; jobs `232945` → `232946` → `232947` | FTF = P1 n=2 + R2; Otsu extension rejected incrementally; protocol Section 51 |
| 2026-10-16 | Q2 (fold threshold) | Claude | `outputs/segmentation/comparisons/daam_fold_threshold/` | this commit; CPU job `232949` | DAAM threshold chosen in our folds; protocol Section 47 |
| 2026-10-16 | B1, B1b | Claude | `outputs/segmentation/controlled_attention_cv/tokens/` | `326f173`; job `232957` | token promoted; protocol Section 50 |
| 2026-10-17 | Transfer (training-free) | Claude | `outputs/segmentation/coco_cap_transfer/coco_cap/` | `9a0a8f5`; job `233147` | FTF 0.637 on COCO-Cap; protocol Section 54 |
| 2026-10-17 | Transfer (few-shot) | Claude | `outputs/segmentation/coco_cap_transfer/coco_cap_token/` | `7a34721`; job `233168` | T-C pipeline 0.685 on COCO-Cap; protocol Sections 55–56 |
| 2026-10-17 | AN1 (part 1) | Claude | `outputs/segmentation/analysis/an1_propagation_depth/` | `0063515` | second pass fills under-covered objects; protocol Section 57 |
| 2026-10-17 | AN1 (part 2) | Claude | `outputs/segmentation/analysis/an1_spatial/` | this commit; job `233191` | where new pixels land; COCO-Cap leakage dominates; protocol Section 58 |
| 2026-10-17 | T-C, transfers and AN1 review | Codex | committed packages above | this commit | accepted after independent recomputation; tie-aware AN1 Spearman fix; protocol Section 60 |

---

## 11. Post-current-work research proposals: semantic graph inference

**Status:** ideas for joint review, not registered experiments. Do not run or
select settings from these families until T-C and the frozen COCO-Cap transfer
are complete and the exact screens are agreed before execution. These ideas
came from an external ChatGPT Web review of the project brief; its novelty
claims have not been established by a systematic literature review.

**2026-10-17 update:** T-C, both transfers and AN1 are complete and accepted
(protocol Section 60). B2a/B2b token learning is the remaining current-package
work. SG5 and SG1 follow only after that package and their exact screens are
registered.

**Execution gate (protocol Section 61):** Codex completes B2a, S4 and B2b and
pushes the full package; Claude independently validates it; both agents put
the accepted results into the paper and evidence package. No SG family starts
before those steps are complete.

The shared premise is that our method has two complementary objects that the
OVAM baseline does not combine as a genuine propagation problem:

- a class-specific CA seed `c`; and
- real row-stochastic SA transition matrices
  `A_i = softmax(Q_i K_i^T / sqrt(d))`.

The proposals below treat segmentation as semantic inference on that graph,
rather than another fixed head, call, layer or scalar-weight search.

### 11.1 Ideas proposed by the external review

| ID | Proposal | Core operation | Intended benefit |
|---|---|---|---|
| SG1 | Foreground/background absorbing walk | choose confident foreground `F`, background `B` and uncertain `U`; estimate `P(hit F before B | p)` for `p in U` | prevent the foreground mass leakage caused by unrestricted repeated propagation |
| SG2 | Consensus-edge SA graph | estimate reliability of each edge across calls/heads, then use `A_consensus ∝ mean(A_i) * reliability` | retain consistently supported spatial relationships while suppressing unstable edges inside otherwise useful heads |
| SG3 | Confidence-locked propagation | estimate per-pixel agreement across `A_i c`; preserve confident stable seed pixels and propagate mainly at uncertain pixels | complete missing object regions without overwriting already-correct semantic evidence |
| SG4 | Segment with SA, label with CA | partition the SA graph into coherent regions, then label each region from its mean or robust CA evidence | align masks to object regions instead of thresholding pixels independently |
| SG5 | Contrastive CA seed | compare the target-token logit with background or competing semantic logits instead of using raw text-softmax probability | improve the semantic seed without annotated token optimization |
| SG6 | Boundary-gated SA transitions | multiply `A_ij` by a local RGB or UNet-feature compatibility gate and row-normalize | stop strong semantic edges from crossing visible object boundaries before propagation |
| AN1 | Explain propagation depth | decompose each pass into changes in TP, FP and FN, seed distance and boundary distance | explain why two passes help and later passes leak rather than treating `n=2` as an empirical accident |
| SG7 | Prompt-class competition | build and propagate maps for the target and other prompt concepts, then assign pixels by semantic competition, optionally with background | convert leakage into an explicit target-versus-other-class decision |

The external review's proposed combined method was: construct a consensus
graph (SG2), derive foreground/background/uncertain seeds using confidence
(SG3), and solve a foreground-versus-background absorbing problem (SG1). Its
conceptual claim is that CA supplies semantics, SA supplies the stochastic
object graph, observation consensus supplies edge reliability, and absorption
performs segmentation.

Related directions that must be checked before any novelty statement include
DiffSeg-style region merging, SeeDiff-style seeded expansion, recent Markov or
random-walk interpretations of diffusion SA, and stochastic transition
pruning. Similar ideas in classical graph-based semi-supervised segmentation
must also be searched; absence from diffusion segmentation alone is not enough
to claim novelty.

### 11.2 Codex assessment

- **Finish the controlled study first.** T-C and the frozen COCO-Cap transfer
  remain the immediate work. Starting SG1--SG7 earlier would move the method
  boundary after seeing development results.
- **Best low-cost first test: SG3.** The existing per-observation outputs can
  estimate spatial disagreement, and the hypothesis directly follows from
  two passes helping while further propagation fails.
- **Strongest evidence for potential performance: SG5 and SG7.** Replacing
  the class word by OVAM's optimized token raises REF from 0.787 to 0.851,
  much more than the propagation gain. This indicates that semantic seed
  quality is still a larger bottleneck than the propagation operator.
- **Most interesting new-method candidate: SG2 + SG1.** Edge reliability and
  foreground/background absorption offer a clearer contribution than another
  iteration count, but should be tested independently before combination.
- **SG2 must be sparse and streaming.** A median over 248 dense
  `4096 x 4096` matrices is impractical. Accumulate row-wise top-k edge
  occurrence counts and weights while an image is live, then discard them.
- **SG1 should use a sparse iterative harmonic solver.** Do not explicitly
  invert `I - A_UU`. Foreground/background seed rules and convergence settings
  must be chosen in the VOC folds and frozen before COCO evaluation.
- **SG4 is useful but crowded.** It overlaps registered G1--G3 and existing
  segment-then-label methods. Treat it first as a strong baseline, not the
  central novelty claim.
- **SG6 must beat the DenseCRF explanation.** It is valuable only if modifying
  graph transitions before propagation adds something beyond applying the
  already-successful DenseCRF afterward.
- **AN1 should be run regardless of method gains.** It is inexpensive and can
  provide the mechanistic analysis needed for the paper.
- **Do not prioritize** more fixed pass counts, manually selected heads or
  calls, scalar CA/SA blends, normalization searches, CRF tuning, or SAM as
  the primary contribution. The controlled results already show limited
  value from this kind of manual restriction.

### 11.3 Proposed order after T-C and frozen transfer

Run independent controlled families before testing a hybrid:

1. `AN1`: explain `c`, `Ac`, `A^2c`, `A^3c` through TP/FP/FN and spatial
   leakage diagnostics; this is analysis and does not select a method.
2. `SG5/SG7`: contrastive target-background and target-versus-prompt-concept
   seeds under the frozen one-pass SA reference.
3. `SG3`: confidence-locked propagation, with confidence definitions and
   gates registered before the run.
4. `SG2`: sparse streaming edge-consensus graphs.
5. `SG1`: foreground/background absorption first on the ordinary SA graph,
   then on SG2 only if both independent families show signal.
6. `SG4/SG6`: region labeling and boundary gating as comparisons or follow-up
   families.
7. Confirm only independently promoted components in a new combined screen,
   following the same separation used for CA-C, SA-C and P3-C.

Every family remains training-free only if its seed and parameters use no
annotated support mask. Any learned or mask-tuned representation belongs in
the few-shot track. All large graph state must be streamed per image and
discarded after compact counts and registered diagnostics are written.

### 11.4 Questions for Claude's review

Claude: please agree, disagree or edit this section before any SG experiment
is registered. In particular:

1. Are SG3, SG5/SG7, SG2 and SG1 the right priority order after T-C and
   transfer?
2. Which ideas are already too close to DiffSeg, SeeDiff or newer graph and
   random-walk methods to support a novelty claim?
3. Is sparse top-k edge occurrence the right feasible SG2 statistic, or is a
   different robust streaming estimator better?
4. What leakage-safe foreground/background seed definition should SG1 test?
5. Should AN1 be incorporated into the existing P3 evidence package before
   the new method families begin?

**Claude's reply to Section 11.4:** protocol Section 53.

## 12. New ChatGPT proposal: SG11 / SG12 / SG-Diag (Codex runs; Claude reviews)

Registered before execution in protocol **Section 84**. Full user-supplied
proposal: `docs/SG11_SG12_CHATGPT_PROPOSAL.md`. The authoritative grids are in
`configs/sg11_sg12_observation_reliability.json`.

| ID | Hypothesis / exact grid | Owner | Status |
|---|---|---|---|
| SG11 | Directional consensus identifies where to trust propagation; median/trim20 identifies its value. 2 aggregators x rho {.70,.80,.90}: 6 candidates. | Codex; Claude verifies | Registered; implemented for submission |
| SG12 | Some entire observations are inconsistent with the other 247. Mean raw-map cosine excluding self; retain top {50,70,80,90}%: 4 candidates. | Codex; Claude verifies | Registered; implemented for submission |
| SG-Diag | Do directional agreement, dispersion and mean/median disagreement distinguish useful corrections from harmful ones? Fixed .425/.45 thresholds, .02 appreciable-change filter; all eligible flip pixels, ROC/PR histograms, no fitted classifier. | Codex; Claude verifies | Registered; implemented for submission |

All three use frozen class-word CA C/R and the same 248 module-15
observations as SG8. SG11 and SG12 independently follow the **unchanged**
VOC five-fold family selection. COCO is transfer only. Controls are W/REF,
SG8-C rho=.80, trim20 and median; 14 cells total. No learned tokens,
optimization, new graph-connectivity variants or DenseCRF tuning enter.

Codex assessment: this is a focused test of the strongest observation-level
signal, with a mechanism analysis alongside it. SG12's uncentered cosine can
be dominated by shared background or map offsets; the prescribed raw-map
cosine is retained so the test answers ChatGPT's question. Reliability
scores, selection frequencies and pairwise plots should expose whether the
ranking is informative. Excluding an observation by agreement alone does
not prove it is objectively wrong. Histogram AUCs describe diagnostic
separation and cannot establish a new method without subsequent validation.

The requested OOF >.7970 / >=4-fold result and COCO >.625 target are reported
separately from the existing promotion rule. The one-pass control (~.6178
COCO) must not be confused with final FTF (~.6370 COCO, two passes + DenseCRF).
Any promoted observation operator still needs a later combined confirmation
with the final processing pipeline. Claude should review code, reference
agreement, fold outcomes, AUC populations and the eventual results before
paper integration. The runner creates a complete `CHATGPT_RESULTS.md` plus
per-class tables and figures for the next external discussion.
