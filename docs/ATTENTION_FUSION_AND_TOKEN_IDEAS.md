# Attention fusion and token ideas (deferred phase)

**Status:** shared backlog, Claude draft 1  
**Starts:** only after the CA and SA configurations are frozen in
`ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md` (its Section 8).  
**How to use:** each idea has an ID. When an idea is run, set its status and
link the output folder and commit. Every idea is evaluated on the frozen seed,
inside the 5 folds of protocol Section 5a, with its own threshold chosen on
the training folds, and with per-image counts saved.

Notation: `c` is the frozen CA seed (64×64, min–max normalized), `A` is the
row-stochastic SA affinity `softmax(Q Kᵀ / √d)` of the frozen SA selection,
`n` is a number of propagation passes.

Citations: names tagged **[OVAM n]** are in the OVAM paper's reference list
(arXiv:2403.14291). Everything else must be checked before it is cited.

## 1. Ways to combine self- and cross-attention

### 1.1 Multiply or gate

| ID | Method | Equation | Extra setting | Status |
|---|---|---|---|---|
| F1 | OVAM fusion (as published) | `c ⊙ norm_[α,1](S₁)`, where `S₁` uses an all-ones value tensor | α | Baseline (implemented in `ovam_full.py`) |
| F2 | OVAM fusion with a real SA signal | `c ⊙ norm_[α,1](Σᵢ Aᵢⱼ)` (column mass) | α | Not run. Isolates the effect of the all-ones value |
| F3 | Gating | `c ⊙ (A c)` | — | Not run |
| F4 | Geometric mean | `√(c ⊙ A c)` | — | Not run |

### 1.2 Propagate ("spread")

| ID | Method | Equation | Extra setting | Status |
|---|---|---|---|---|
| P1 | Power propagation (current method) | `Aⁿ c` | n | Exploratory results for n = 1–4 |
| P2 | Propagation with restart (personalized PageRank) | `x ← (1−β) A x + β c`, iterated | β, n | Not run. Targets over-focus (person) and leaking (bus) |
| P3 | Heat kernel | `Σₖ e^{−t} tᵏ/k! Aᵏ c` | t | Not run. Removes the hard choice of n |
| P4 | Residual mix | `λ Aⁿ c + (1−λ) c` | λ | Partly screened in the earlier streaming suite |
| P5 | Symmetric / spectral affinity | `(D^{−½} W D^{−½})ⁿ c`, `W = (A + Aᵀ)/2` | n | Not run |
| P6 | Sparse affinity | keep the top-k entries of each row of `A`, renormalize | k | Not run. Targets leaks into weakly similar regions |
| P7 | Rollout across modules | `A_L ⋯ A₂ A₁ c` (different SA modules in sequence) | module order | Close to the earlier progressive screen |
| P8 | Column propagation | `Aᵀ c` (normalized) | — | Operator check only (protocol Section 6) |

### 1.3 Group pixels with SA, label groups with CA

| ID | Method | Idea | Status |
|---|---|---|---|
| G1 | Attention-map merging | cluster pixels whose SA rows are similar (KL merging as in "Diffuse, attend, and segment" **[OVAM 48]**), label each segment by its mean CA | Not run |
| G2 | Spectral clustering | eigenvectors of the SA graph give segments; choose segments whose mean CA exceeds a threshold | Not run |
| G3 | k-means on SA rows | simplest form of G1/G2 | Not run |
| G4 | Seeded region growing | start at CA peaks, grow while SA similarity stays high | Not run |

These give object-aligned boundaries and do not threshold the CA map directly.

### 1.4 SA as the pairwise term of a CRF

| ID | Method | Idea | Status |
|---|---|---|---|
| R1 | Graph cut / CRF with SA edges | CA gives the unary score, SA replaces RGB similarity in the pairwise term | Not run |
| R2 | DenseCRF on RGB (OVAM's) | SimpleCRF, OVAM's parameters, on the regenerated image | Exploratory results exist (see Section 3) |

### 1.5 Prototype matching

| ID | Method | Equation | Status |
|---|---|---|---|
| M1 | CA-weighted prototype | `p = Σⱼ cⱼ fⱼ / Σⱼ cⱼ`, mask from `cos(fᵢ, p)`, `f` = SA queries or keys | Not run |
| M2 | Multiple prototypes | k-means inside high-CA pixels; max similarity over prototypes | Not run. For multi-part objects (person, bicycle) |

### 1.6 Label-free averaging over heads and steps

| ID | Method | Idea | Status |
|---|---|---|---|
| W1 | Entropy weighting | weight each head/step map by `1 / entropy` | Not run |
| W2 | Agreement-based head choice | per image, keep the SA heads whose propagated map agrees most with `c` | Not run. A label-free alternative to a fixed head subset |
| W3 | Average before or after propagation | `mean(Aⁿ) c` vs. `(mean A)ⁿ c` | Not run |
| W4 | Iterate to convergence | alternate propagation and normalization until the map stops changing | Not run |

Training-free baseline to compare against: "Diffusion model is secretly a
training-free open vocabulary semantic segmenter" **[OVAM 52]**.

**Suggested first batch:** F2, P2, P6, G1. They are cheap, label-free, and
each targets a failure already observed.

## 2. Token methods beyond OVAM's

OVAM learns one token per class in the text-encoder output space, trained with
binary cross-entropy on one annotated image per class (OVAM Sec. 4.1, Supp.
A.1). Alternatives:

### 2.1 What is learned

| ID | Method | Idea | Example |
|---|---|---|---|
| T1 | Input-space token | learn the word embedding before the CLIP encoder, so it stays context-aware | Textual Inversion **[OVAM 10]** |
| T2 | Several tokens per class | k vectors per class | — |
| T3 | Per-layer / per-step tokens | a separate token for each UNet layer or denoising step; matches our layer/step axes | P+ and NeTI (to verify) |
| T4 | Shared context + class name | learnable context shared by all classes | CoOp-style prompt learning (to verify) |
| T5 | Learned background tokens | replace `<SoT>` as background | — |
| T6 | Key projections instead of tokens | small update of `to_k` | Key-locked editing **[OVAM 47]**, Custom Diffusion (to verify) |
| T7 | Unconditional embedding | per-step optimized null embedding | Null-text inversion (to verify) |

### 2.2 Training signal

| ID | Method | Idea | Example |
|---|---|---|---|
| S1 | Mask BCE (OVAM) | current OVAM tokens | OVAM |
| S2 | Better segmentation loss | Dice, Lovász, focal | — |
| S3 | Train through our pipeline | loss on the SA-propagated output, not raw CA | New. Directly targets the COCO-Cap gap |
| S4 | Few-shot scaling | 1, 5, 10 annotated images per class | SLiMe (to verify) |
| S5 | Reconstruction loss | standard denoising loss, no masks | Textual Inversion **[OVAM 10]** |
| S6 | Self-consistency at test time | per image, make `c` agree with `A c`, or minimize map entropy; no labels | New |
| S7 | Classifier guidance | discriminative class tokens | **[OVAM 43]** |
| S8 | Discrete search | best real words | "Hard prompts made easy" **[OVAM 53]** |

**Suggested first batch:** S3 and S4 (closest to the open COCO-Cap question),
then S6 (keeps the method training-free), then T3.

## 3. Results already available

Exploratory, run before the protocol, on the earlier final method (12 CA
layers with fixed top-4 heads, U3.2, heads {0,1,2,4,5,6}, calls 15–30,
percentile 1–99, threshold 0.45, 1% cleanup). Details, reproduction and
per-image counts: `docs/narmyn/attention/TOKEN_CRF_EXPERIMENT.md`.

| Dataset | Configuration | mIoU |
|---|---|---:|
| VOC-Sim | class word, 1 pass | 0.7802 |
| VOC-Sim | OVAM token, 1 pass | 0.8347 |
| VOC-Sim | OVAM token, 1 pass, + DenseCRF | 0.8525 |
| VOC-Sim | released token-optimized OVAM masks | 0.8247 |
| COCO-Cap | class word, 1 pass | 0.6264 |
| COCO-Cap | OVAM token, 1 pass, + DenseCRF | 0.6612 |
| COCO-Cap | token-optimized OVAM (paper Table 1) | 0.692 |

## 4. Log

| Date | ID | Owner | Output | Commit | Note |
|---|---|---|---|---|---|
| 2026-10-13 | T-ovam, R2 | Claude | `outputs/segmentation/narmyn_token_crf/` | this branch | exploratory, pre-protocol |
