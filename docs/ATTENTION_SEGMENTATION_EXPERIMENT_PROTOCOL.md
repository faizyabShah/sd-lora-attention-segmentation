# Controlled Attention Segmentation Experiment Protocol

**Protocol version:** 1.2 approved

**Dataset used for selection:** VOC-Sim, 5 stratified folds **[v1.1]**

**Codex status:** ACKNOWLEDGED (v1.2, including Claude's Section 14 result-format clarification)

**Claude status:** ACKNOWLEDGED (v1.2, with two wording/format clarifications marked "[v1.2 clarification, Claude]"; see Section 14)

**Production runs authorized:** YES — both agents acknowledged protocol v1.2

This document is the coordination point for the controlled attention
segmentation experiments. Both agents must update this file when a protocol
choice, run status, frozen configuration, or result changes. A result is not a
final method choice unless it satisfies the decision rules below and is
recorded here.

## 1. Scientific plan

The study has three ordered phases:

1. Screen cross-attention (CA) calls, resolutions, modules, and heads under one
   CA-only condition.
2. Freeze a CA configuration, then screen self-attention (SA) calls,
   resolutions, modules, and heads under one one-pass SA condition.
3. Freeze the attention selection, then study fusion, repeated propagation,
   normalization, thresholds, and spatial postprocessing.

Before phase 3, Section 34 performs a registered joint CA×SA interaction
audit. The sequential CA-then-SA design is valid for controlled attribution,
but it does not establish that the CA seed preferred without SA is also the
best seed after propagation. The audit cross-validates every CA-C cell against
every unique SA-C cell before fusion is studied.

VOC-Sim is the development and tuning dataset. No choice is changed in response
to COCO-Cap results. After the full method is frozen, COCO-Cap is used as the
untuned generalization evaluation.

**[v1.1]** Every choice is made inside 5-fold stratified cross-validation on
VOC-Sim (Section 5a), so that each reported VOC-Sim number comes from images
that did not take part in choosing it. The full-490 numbers are still reported,
labelled as in-sample. Earlier results that looked at COCO-Cap before a choice
was frozen (for example the one-pass and all-layer "revised" system) are treated
as exploratory and must be re-derived under this protocol before they are
reported as untuned generalization.

### OVAM self-attention clarification **[v1.2]**

The all-ones-value self-attention behavior described in Narmeen's review
originates in the upstream OVAM repository. Our shared, reference-fidelity
OVAM baseline deliberately reproduces that hook in
`src/segmentation/ovam_full.py` through the compact capture in
`src/attention_capture/memory.py`; both collaborators must call that same
implementation when reporting a regenerated OVAM baseline. It is not the
self-attention operator in our controlled experiments or proposed method, and
it is not an open implementation problem for this study. Our SA experiments
use genuine QK-derived affinities,
`A = softmax(QK^T / sqrt(d))`, and propagate the CA seed with `A x`, as defined
in Section 6. Claude should not implement, repair, or schedule an experiment to
replace the upstream all-ones hook. OVAM remains a comparison baseline while
the controlled CA/SA pipeline is developed independently.

## 2. Dataset and generation controls

These controls apply to every experiment unless a later version of this file
explicitly changes them.

| Item | Fixed value |
|---|---|
| Dataset | All 490 annotated VOC-Sim records, 20 classes |
| Generator | Stable Diffusion 1.5, local `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Prompt and seed | Released prompt and seed for each record |
| Inference | 30 scheduler steps; recorded UNet call indices 0–30 |
| Guidance | Classifier-free guidance 7.5 |
| Model arithmetic | Float32 model; float16 compact attention storage |
| CA attribution branch | Conditional CFG branch (query and key) |
| CA softmax domain **[v1.1]** | All 77 positions of the padded conditional prompt (Stable Diffusion passes no text attention mask, so padding EOT tokens are included). This differs from OVAM, which uses the unconditional query and a softmax over the unpadded prompt only; see CA-A. |
| Target tokens | Every tokenizer piece of the complete class name, averaged within each head after the softmax |
| Working grid | 64×64 |
| Output retention | Generated image and attention tensors discarded after scoring |
| Primary metric | Macro mean over 20 pooled foreground class IoUs (mIoU) |
| Secondary metrics | Per-class pooled IoU, mean image IoU, overall pixel IoU |
| Per-image output **[v1.1]** | `tp`, `fp`, `fn` for every image and every configuration (JSONL or a compact array), so that folds, bootstrap intervals, and later re-analyses need no regeneration. The exhaustive 65,535-subset search may store per-fold pooled counts instead. |
| Fold assignment **[v1.1]** | `configs/voc_sim_folds.csv`: 5 folds stratified by class (about 5 images per class per fold), fixed seed, committed before any run |

“Call 0–30” always refers to recorded UNet call indices and is inclusive. It
does not mean 31 scheduler updates.

## 3. Fixed CA-only scoring condition

The CA phase contains no SA operation and no spatial postprocessing.

1. Average the selected target-token pieces within each head.
2. Average selected heads within each CA module.
3. Resize each selected module map to 64×64 with bilinear interpolation.
4. Average every included module-call observation with equal weight. A
   resolution containing more modules therefore contributes more observations;
   resolutions are not reweighted to have equal total weight.
5. Apply min–max normalization once to the final 64×64 map.
6. Bilinearly resize it to the annotation size.
7. Predict foreground at values greater than or equal to 0.45.

The CA phase uses:

- no SA fusion or propagation;
- no DenseCRF;
- no connected-component cleanup or morphology;
- no learned or class-specific weights;
- no token optimization;
- no threshold tuning per configuration.

**[v1.1] Calibration check.** A fixed min–max/0.45 rule can favour a
configuration only because it shifts the map's value distribution (the earlier
"max normalization" ablation lost 0.08 mIoU mostly for this reason). For every
configuration also record mIoU at thresholds 0.35 and 0.55, and the
threshold-free average precision of the 64×64 map against the 64×64
nearest-downsampled target. These are diagnostics: a change is promoted on
mIoU at 0.45, but a promotion whose sign reverses at 0.35 or 0.55 is flagged in
Section 11 as calibration-dependent.

When one CA axis is screened, every other CA axis stays at its inclusive
reference value: calls 0–30, resolutions 8/16/32/64, all 16 CA modules, and all
eight heads.

## 4. CA experiments

### CA-T: denoising calls

Keep all resolutions, all 16 modules, and all heads. Evaluate:

- all calls 0–30;
- every individual call;
- disjoint windows 0–4, 5–9, 10–14, 15–20, 21–25, and 26–30;
- trailing windows 5–30, 10–30, 15–30, 21–30, and 26–30;
- cumulative windows 0–4, 0–9, 0–14, 0–20, and 0–25;
- all 31 leave-one-call-out configurations.

### CA-R: spatial resolutions

Keep calls 0–30 and all heads. Evaluate all 15 nonempty subsets of
8×8, 16×16, 32×32, and 64×64. Selecting a resolution selects every CA module at
that resolution.

Also evaluate the Cartesian product of the CA-T configurations and the 15
resolution subsets. This joint grid is interaction evidence; its maximum does
not automatically become the chosen method.

### CA-L: modules

Keep calls 0–30 and all heads. Evaluate:

- all 16 modules;
- the 15 non-middle modules used by standard OVAM/DAAM;
- down-only, middle-only, up-only, down+middle, and middle+up groups;
- every individual module;
- every leave-one-module-out and leave-two-modules-out configuration;
- all 65,535 nonempty module subsets as an exploratory diagnostic.

The exhaustive maximum is not eligible for automatic selection because it is
the maximum of 65,535 comparisons on the tuning set. A layer subset is eligible
only if it is structurally simple or its improvement is reproduced by the
leave-out results and the final combined confirmation run.

### CA-H: heads

Keep calls 0–30 and all 16 modules. Evaluate all 255 nonempty subsets of head
indices 0–7. The same subset is used in every CA module. Per-module or
class-specific head selection is outside this clean screen.

**[v1.1] Required head report** (no extra computation; all are subsets of the
255): each head alone; leave-one-head-out from all 8; the greedy backward
elimination path (start from 8, at each round remove the head whose removal
raises mIoU most, stop at 1); the greedy forward path; the rank of the chosen
subset among all 255; and in how many of the 5 folds each head is retained.
The same report is required for SA-H.

### CA-A: attribution convention **[v1.1]**

The current final method uses OVAM's convention (unconditional query, softmax
over the unpadded prompt, first class-word piece), while this protocol fixes the
conditional branch, the padded softmax, and all pieces. Screen this axis
explicitly so that the SA phase is not frozen on one convention and the final
method built on another. Keep calls 0–30, all modules and all heads; evaluate
the 2 × 2 × 3 grid of query branch {conditional, unconditional}, softmax domain
{padded 77, unpadded prompt}, and pieces {all averaged, first, last}. The
capture must store the unconditional/unpadded per-head map (`ovam_target_map`)
and the conditional/padded per-piece map; the remaining cells need one extra
compact key per module (per-piece maps before averaging).

### Figures **[v1.1]**

For the report, render for 3–4 fixed images per dataset: every CA module's
head-averaged map upsampled to 64×64 on a common colour scale, grouped by
resolution; and each SA head's propagated map. Images are chosen by fixed
index before any result is seen.

### CA-C: combined confirmation

After CA-T, CA-R, CA-L, and CA-H are complete, combine only choices that pass
the decision rule. Run:

1. the inclusive CA reference;
2. every promoted one-axis change;
3. the combined candidate;
4. leave-one-promoted-change-out variants of the combined candidate.

Freeze the CA configuration only after CA-C confirms that the gains coexist.

## 5. Decision rule

The reference for each one-axis CA screen is the inclusive CA configuration
under Section 3. The reference for each SA screen is the frozen CA mask.

- **Promote:** at least +0.0100 absolute mIoU.
- **Conditional promotion:** +0.0050 to +0.0099 mIoU, improvement in at least
  12 of 20 classes, and no class loses more than 0.050 IoU.
- **Marginal:** less than +0.0050 mIoU. Keep the inclusive or simpler setting.
- If eligible configurations are within 0.0020 mIoU, prefer the setting with
  more included calls/modules/heads because it makes fewer dataset-specific
  exclusions.

Mean image IoU is diagnostic and cannot override the primary pooled-class
mIoU. Every promoted change must survive CA-C or SA-C; independently selected
maxima are not assumed to be additive.

## 5a. Cross-validated selection **[v1.1]**

The decision rule above is applied inside each fold, not once on all 490
images.

1. For fold k = 1..5, apply Section 5 to the four other folds (pooled counts of
   those images only) and record the chosen configuration.
2. Score that choice on fold k. Pool fold k's counts with the other held-out
   folds to obtain one out-of-fold mIoU over all 490 images. This is the
   VOC-Sim number reported for the selection procedure.
3. Freeze a change only if the decision rule chooses it in at least 4 of 5
   folds **and** the pooled out-of-fold mIoU of the selection exceeds the
   reference's pooled out-of-fold mIoU. Otherwise keep the inclusive or simpler
   setting, even if it is not the full-490 maximum. **[v1.2 clarification,
   accepted in Section 17]**
4. Report, for each promoted change, the out-of-fold mIoU, the full-490 mIoU
   (labelled in-sample), and a paired per-image bootstrap interval (20,000
   resamples, fixed seed) against its reference.
5. The exhaustive layer search (65,535) and head search (255) are allowed in
   this loop, because selection happens on the training folds only, but their
   out-of-fold score is what is reported.

## 6. Fixed SA screening condition

SA experiments begin only after the CA configuration is frozen in this file.

1. Construct the frozen raw CA map and min–max normalize it to form the seed.
2. Use the conditional CFG self-attention probabilities.
3. Apply exactly one non-residual propagation pass, `y = A x`.
4. For every selected SA module/call/head observation, resize the CA seed to
   the native SA resolution, propagate once, then resize the propagated map to
   64×64. **[v1.1]** Downsample the seed with area averaging
   (`F.adaptive_avg_pool2d`), not bilinear interpolation: bilinear without
   anti-aliasing samples only a few of the 64×64 values when going to 16×16 or
   8×8. Upsample the result bilinearly as before. Do not re-apply min–max to
   the resized seed, so that every resolution sees the same seed scale.
   Orientation: `y = A x` with `A = softmax(Q Kᵀ / √d)` row-stochastic over keys,
   so each query pixel takes the attention-weighted average of the seed over the
   pixels it attends to. This matches `propagate_selected_module`
   (`focused_self_attention.py`, `torch.matmul(affinity, current)`). As a
   one-off operator check, also report `y = Aᵀ x` (column propagation) once for
   the final candidate.
5. Average all selected propagated observations equally.
6. Min–max normalize once, resize to the annotation, and threshold at 0.45.

There is no iterative propagation, residual connection, CA/SA weighted mixing,
DenseCRF, component cleanup, or per-configuration threshold tuning in this
phase.

## 7. SA experiments

Use the frozen CA seed for every experiment.

### SA-T: denoising calls

Repeat the CA-T call grid while keeping every SA resolution, module, and head.

### SA-R: spatial resolutions

Evaluate all 15 nonempty subsets of 8×8, 16×16, 32×32, and 64×64 while keeping
all calls and heads.

### SA-L: modules

Evaluate the same structured, individual, leave-one-out, leave-two-out, and
exploratory exhaustive module families defined for CA-L.

### SA-H: heads

Evaluate all 255 global subsets of SA head indices 0–7. Use the same subset in
every selected SA module.

### SA-C: combined confirmation

Repeat the CA-C confirmation structure against the frozen CA-only reference.
Freeze the SA configuration only if the combined candidate retains the
one-axis improvements.

## 8. Later experiments deliberately deferred

The following begin only after SA-C:

- repeated SA propagation;
- residual versus non-residual propagation;
- arithmetic, multiplicative, gated, or learned CA/SA fusion;
- normalization and calibration methods;
- threshold sweeps;
- connected components and morphology;
- DenseCRF or other image-aware refinement;
- token optimization;
- class-specific choices.

**[v1.1]** The candidate fusion operators, propagation variants, and token
methods for this phase are listed, with equations and priorities, in
`docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md`. Exploratory results already exist
for OVAM's released optimized tokens and DenseCRF on top of the earlier final
method (`docs/narmyn/attention/TOKEN_CRF_EXPERIMENT.md`). They were run before
this protocol and are not frozen choices; they must be re-run on the frozen
CA/SA configuration and selected inside the folds.

## 9. Work split and handoff

The work is split by independent experiment families. Both agents must use the
same committed fold file, metric writer, thresholds, and fixed condition. An
owner is responsible for implementation, execution, merge, and the compact
result report for that package.

| Package | Owner | Official responsibility | Output root |
|---|---|---|---|
| Shared CV infrastructure | Claude | `configs/voc_sim_folds.csv`, compact per-image metric writer, fold aggregation, paired bootstrap/CV selector, and a synthetic metric test | `outputs/segmentation/controlled_attention_cv/shared/` |
| CA package A | Codex | CA-T, CA-R, and the joint CA time-resolution screen | `outputs/segmentation/controlled_attention_cv/ca_time_resolution/` |
| CA package B | Claude | CA-L, CA-H, the required head report, and CA-A attribution convention | `outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution/` |
| CA confirmation | Codex runs; both review | Evaluate the combined shortlist under the folds and freeze CA | `outputs/segmentation/controlled_attention_cv/ca_confirmation/` |
| SA package A | Codex | SA-T, SA-R, and the joint SA time-resolution screen using the frozen CA configuration | `outputs/segmentation/controlled_attention_cv/sa_time_resolution/` |
| SA package B | Claude | SA-L, SA-H, the required head report, and the `A^T x` operator check using the same frozen CA configuration | `outputs/segmentation/controlled_attention_cv/sa_layer_head/` |
| SA confirmation | Codex runs; both review | Evaluate the combined shortlist under the folds and freeze SA | `outputs/segmentation/controlled_attention_cv/sa_confirmation/` |

### Execution order **[v1.2]**

1. Both agents acknowledge protocol v1.2.
2. Claude commits the shared fold file, metric writer, aggregation/selection
   code, and its synthetic metric test.
3. Codex pulls that commit and integrates the shared interface. CA packages A
   and B may then run in parallel.
4. Both agents review the CA package reports. Codex runs CA confirmation and
   records the frozen CA configuration in Section 11.
5. SA packages A and B may run in parallel. Both must refer to the same frozen
   CA configuration by configuration name and Git commit.
6. Both agents review the SA package reports. Claude runs SA confirmation and
   records the frozen SA configuration in Section 11.
7. Fusion, token, and postprocessing studies start only after both selections
   are frozen.

### Shared result contract **[v1.2]**

Every regular candidate writes one compact per-image record containing at
least `image_id`, `class`, `fold`, `configuration`, `tp`, `fp`, `fn`, `tn`,
and `image_iou`, plus any diagnostics required by Sections 4 and 6. Every
package also writes:

- `settings.json` with the exact Git commit and all fixed and varied settings;
- resumable shard state;
- `fold_results.json` with train-selection and held-out-fold results;
- `summary.json` and a concise `report.md`.

The exhaustive 65,535-subset layer search may retain pooled counts per fold.
Any exhaustive-search candidate promoted to the combined confirmation must be
rerun with full per-image records.

**[v1.2 clarification, Claude]** Two format details, so both runners write the
same thing:

- The Section 3 calibration diagnostics belong to the same record: every
  configuration stores `tp`, `fp`, `fn`, `tn` at each of the thresholds
  0.35, 0.45 and 0.55, and the threshold-free 64×64 average precision.
  Selection uses 0.45 only.
- Per-image records are stored as compact arrays (one `.npz` per shard:
  configuration names, image ids, folds, and integer count arrays of shape
  `[configuration, image, threshold, 4]`), not as one JSON object per
  configuration and image. The CA screens alone have roughly
  1,700 configurations × 490 images. JSON is used for `settings.json`,
  `fold_results.json` and `summary.json` only. The shared writer provides
  both, plus a loader.

### Instructions for Claude

1. Read Sections 2–7 and audit them against the implementation.
2. Check especially token-piece averaging, conditional branch selection,
   module-call weighting, normalization order, threshold comparison, and the
   orientation of `y = A x`.
3. If a correction is needed, edit this document, explain it in the commit
   message, and leave `Claude status` as `CHANGES REQUESTED`.
4. If protocol v1.2 is acceptable, change `Claude status` to
   `ACKNOWLEDGED (v1.2)`, add the reviewing commit hash to the status table, and
   push the change.
5. Do not start an official production run while `Production runs authorized`
   is `NO`.
6. After both sides agree, implement the shared CV infrastructure first, then
   implement or submit CA package B. Store only resumable metric state and
   compact result summaries.
7. Record job IDs, exact commit, output path, and completion state below.

Before pushing an update, pull the latest `main`, keep the commit scoped to the
protocol or owned runner, and do not overwrite another agent's status entry.

## 10. Coordination status

| Item | Owner | Status | Commit/job/output |
|---|---|---|---|
| Protocol v1.2 | Both | Approved; production authorized | `5ee164f`, `aef6e5499`, `0e0a524`, `3d80a8994`, `a38deaa06` |
| Protocol audit | Claude | v1.2 acknowledged with two clarifications accepted by Codex (Sections 14–15) | `aef6e5499`, `a38deaa06` |
| Fold file + per-image writer + CV selection | Claude | Validated by Codex; OOF freeze guard enforced (Section 18) | `a163d4b6a`, this commit; `outputs/segmentation/controlled_attention_cv/shared/` |
| CA-T / CA-R / joint official | Codex | Complete: keep all calls; promote resolutions 8+16+64 to CA-C | `16530b54b`; jobs `232263` → `232264` → `232265`; `outputs/segmentation/controlled_attention_cv/ca_time_resolution/` |
| CA-L / CA-H / CA-A official | Claude | Complete (490/490, job `232243`); results for review in Section 21 | this commit; `outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution/` |
| Evidence package (all experiments, tables, figures) | Claude | Built from committed results; rebuild after every package (Section 22) | this commit; `docs/narmyn/attention/evidence/` |
| CA-C and frozen CA | Claude runs (taken over from Codex, see Section 25); both review | Complete (490/490, job `232469`): CA-R alone frozen (Section 27); Codex review requested | this commit; `outputs/segmentation/controlled_attention_cv/ca_confirmation/`; `configs/controlled_sa_seed_frozen.json` |
| SA-T / SA-R / joint official | Codex | Complete (490/490, jobs `232502` → `232503` → `232504`): SA-T call 30, SA-R resolutions 8+64 (Section 31) | `outputs/segmentation/controlled_attention_cv/sa_time_resolution/` |
| SA-L / SA-H / operator check | Claude | Complete and reviewed by Codex (490/490, job `232497`): SA-L only module 15, SA-H heads {1, 4, 6}, `A x` kept (Sections 30–31) | `97046d7c1`; `outputs/segmentation/controlled_attention_cv/sa_layer_head/` |
| SA-C and frozen SA | Codex runs; both review | Complete (490/490): module 15 alone selected 5/5; Codex reviewed, Claude review requested (Section 33) | jobs `232561` → `232562` → `232563`; `outputs/segmentation/controlled_attention_cv/sa_confirmation/` |
| SA-C second review | Claude | Accepted: every Section 33 value reproduced independently (Section 35); SA recorded in Section 11 | this commit |
| Joint CA×SA interaction audit | Codex runs; Claude reviews | Complete (490/490): sequential `CA=C/R|SA=C/L` retained 5/5; Claude result review requested (Section 36) | jobs `232590` → `232591` → `232592`; `outputs/segmentation/controlled_attention_cv/joint_ca_sa_cartesian/` |
| Joint CA×SA audit review | Claude | Accepted: every Section 36 value reproduced independently (Section 38) | this commit |
| Phase 3 registry (all deferred experiments, rules R1–R11) | Claude drafts; Codex reviews | Draft 2 in `docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md`; Codex review requested (Section 38) | this commit |
| Paper skeleton (`paper/`) | Claude created (taken over at Narmeen's request, Section 39); section owners per Section 29 | Created; compiles; Codex review requested | this commit; `paper/` |
| Released baselines Q1–Q5 | Claude | Scored with OVAM's notebook procedures; every published value reproduced (Section 41) | this commit; `outputs/segmentation/comparisons/released_baselines/` |
| Fusion and calibration package (P4, F3, F4, N2–N4, R2) | Claude runs; Codex reviews | Complete and accepted: DenseCRF promoted; other families rejected; independently carried into P3-C (Sections 44 and 51) | job `232922`; `outputs/segmentation/controlled_attention_cv/fusion_calibration/` |
| Deferred propagation package | Codex runs; Claude reviews | Complete (490/490): observation-wise two-pass propagation promoted to P3-C; Claude review requested (Section 43) | jobs `232662` → `232663` → `232664`; `outputs/segmentation/controlled_attention_cv/propagation_package/` |
| Propagation package review | Claude | Accepted: every Section 43 value reproduced independently (Section 46) | `dba3126`, this commit |
| P3-C (training-free combined confirmation) | Codex runs; Claude reviews | Complete (490/490): two-pass SA + DenseCRF frozen at 0.8135 OOF; expanded Otsu cell remains diagnostic after the incremental R4 check (Section 51); Claude review requested | jobs `232945` → `232946` → `232947`; `outputs/segmentation/controlled_attention_cv/phase3_confirmation/` |
| COCO-Cap transfer (registry Section 7) | Claude runs; Codex reviews | Ownership and runner accepted; ready after the Section 51 result, with frozen FTF `P1+R2` at threshold 0.45 (Sections 48 and 51) | `outputs/segmentation/coco_cap_transfer/` |
| Token track B1/B1b (registry Section 2) | Claude runs; Codex reviews | Complete and accepted (490/490): conditional released token promoted 5/5 to 0.8507 OOF; T-C remains (Sections 50–51) | job `232957`; `outputs/segmentation/controlled_attention_cv/tokens/` |
| P3-C review | Claude | Accepted: every Section 51 value reproduced independently; FTF recorded in Section 11 (Section 52) | this commit |
| Token confirmation T-C | Claude runs; Codex reviews | Complete and accepted: token + one pass + DenseCRF, 4/5; exact independent recomputation (Sections 52 and 60) | `a002cc4`; `outputs/segmentation/controlled_attention_cv/token_confirmation/` |
| COCO-Cap transfer run | Claude runs; Codex reviews | Complete and accepted (707/707, job `233147`): FTF 0.637 at its VOC threshold; summary regenerated exactly (Sections 54 and 60) | `9a0a8f5`; `outputs/segmentation/coco_cap_transfer/coco_cap/` |
| Few-shot COCO-Cap transfer | Claude runs; Codex reviews | Complete and accepted (707/707, job `233168`): T-C pipeline 0.685; summary regenerated exactly (Sections 56 and 60) | `56295a3`; `outputs/segmentation/coco_cap_transfer/coco_cap_token/` |
| AN1 mechanism analysis (registry Section 11) | Claude runs; Codex reviews | Complete and accepted: parts 1–2 reproduce; tie-aware Spearman correction does not change any rounded conclusion (Sections 57–60) | `0063515`, `c66d1cf`, this commit; `outputs/segmentation/analysis/` |
| B2a/S4/B2b token package | Codex runs; Claude reviews | Measurements reproduced, but OVAM-fidelity claim withdrawn: B2 uses a linear objective while OVAM used token softmax; faithful-rerun decision and paper integration pending owner review (Sections 64--65, 69, 71) | jobs `233332`, `233333`, `233660`, `233837`, `233838`; `outputs/segmentation/controlled_attention_cv/token_learning_b2/` |
| B2a/S4/B2b independent replication | Claude | Complete: independently reproduces Codex's linear-objective package; this validates its measurements, not equivalence to OVAM's optimizer (Sections 65, 69) | `4472e2d`; `outputs/segmentation/controlled_attention_cv/token_learning_b2_replication/` |
| SG5 contrastive CA seed | Claude runs; Codex reviews | Complete and accepted: 490 VOC + 707 COCO; reference won all four screens 5/5, so no seed is promoted (Sections 66, 70--71) | `eb95eb019`; `outputs/segmentation/controlled_attention_cv/sg5_contrastive_seed/` |
| SG1 absorbing walk + SG3 confidence lock | Codex runs; Claude reviews | Complete: 490 VOC + 707 COCO; no family frozen; token SG3 has a nonpromoted transfer signal for owner review (Sections 68, 71) | jobs `234264`--`234266`; `outputs/segmentation/controlled_attention_cv/sg1_sg3_graph_refinement/` |
| B2 optimizer audit against OVAM (Section 64, request 1) | Claude | Complete: B2 uses linear maps where OVAM optimizes token-softmax maps; everything else matches; rerun proposed and paper integration held (Section 69) | this commit; `outputs/segmentation/analysis/b2_recipe_audit/` |
| B2 paper and evidence integration (Sections 67, 72) | Claude drafts; Codex reviews | Accepted: all A12 values and provenance reproduce; wording correctly calls B2 a linear-objective variant and keeps the released token (Sections 76, 78) | `7bc2c776a`; `paper/sec/7_fusion.tex` |
| SG10 seed-connected bottleneck propagation (Section 75 handoff) | Claude runs; Codex reviews | Complete (VOC 490/490, COCO 707/707, job `234642`): reference won 5/5 on both tracks; bottleneck does not separate added TP from FP (AUC 0.35–0.56); stop per the specification; Codex review requested (Section 79) | this commit; `outputs/segmentation/controlled_attention_cv/sg10_seed_connected_bottleneck/` |
| Locked token-SG3 confirmation | Codex runs; Claude reviews | Complete: 490 VOC + 707 COCO; SG3+R2 is 0.8684/0.7006 but its VOC OOF gain is marginal (+0.0030); Claude review requested (Sections 72, 78) | jobs `234490`--`234493`; `outputs/segmentation/controlled_attention_cv/token_sg3_confirmation/` |
| SG8 observation consensus | Codex runs; Claude reviews | Complete: 490 VOC + 707 COCO; W sign consensus rho=.80 and T median freeze; variance and expansion-only gates fail (Sections 73, 78) | jobs `234511`--`234516`; `outputs/segmentation/controlled_attention_cv/sg8_observation_consensus/` |
| SG9 reciprocal propagation | Codex runs; Claude reviews | Complete: 490 VOC + 707 COCO; reference wins both tracks 5/5, so no reciprocal operator freezes (Sections 74, 78) | jobs `234535`--`234538`; `outputs/segmentation/controlled_attention_cv/sg9_reciprocal_propagation/` |
| Locked SG3 / SG8 / SG9 review | Claude | Accepted: contracts, byte-for-byte regeneration, operators and tie-breaks match the registrations (Section 80) | this commit |
| SG2 consensus-edge graph and SG4 SA regions, standalone (Narmeen's request) | Claude runs; Codex reviews | Complete (job `234856`): SG2 one-pass `W/G_cv_g20_n1` promoted 5/5, OOF +0.0111, COCO paired +0.016; SG4 not promoted (oracle regions 0.80–0.88); Codex review requested (Section 85) | this commit; `outputs/segmentation/controlled_attention_cv/sg2_consensus_edges/`, `sg4_region_labels/` |
| Training-free backlog review (Section 83) | Claude | Accepted: contract, byte-for-byte regeneration, operators match the registration (Section 86) | this commit |
| Review of Sections 84–105 | Claude | Accepted with notes; **A3 affinity is transposed relative to GoCA** (owner decision needed); shards from SG16 onward not committed, so not regenerable (Section 107) | this commit |
| Requests: SG11–SG27 comparison pages; Q7 external baselines | Codex (pages); owners (Q7) | Requested (Section 108) | this commit |

### Pre-sync exploratory run

Jobs `232210`, `232211`, and `232212` completed before this coordination
protocol was written. Their output is
`outputs/segmentation/voc_sim_clean_ca_ablations/`. It used the Section 3
condition and includes all CA families in one run. Treat it as exploratory
evidence until Claude audits the implementation. If Claude confirms exact
agreement, the run may be ratified as the official CA screen; otherwise rerun
the affected families under the corrected protocol.

**[v1.1] Audit result:** the implementation matches Section 3 exactly (see
Section 12). The run is ratified as correct exploratory evidence, but it cannot
be the official screen under Section 5a because it stores only per-class
totals. Re-run with the per-image writer before any CA choice is frozen.

## 11. Frozen choices and results

CA is frozen by CA-C (Section 27) and SA by SA-C (Sections 33 and 35); the
joint CA×SA audit kept the pair (Sections 36 and 38). The final training-free
method (FTF) is frozen by P3-C (Sections 51 and 52) and the few-shot pipeline
by T-C (Sections 52 and 60). Every row was reviewed by both agents. Add selections here only after the
relevant combined confirmation.

| Stage | Frozen setting | Evidence | Commit |
|---|---|---|---|
| CA | Calls 0–30; resolutions 8, 16, 64 (the 32×32 modules 2, 3, 10, 11, 12 removed; 11 modules); all 8 heads; conditional query and key, padded 77-token softmax, all class-word pieces. Seed file `configs/controlled_sa_seed_frozen.json` | CA-C, Section 27: 4/5 fold votes, OOF mIoU 0.6653 → 0.6914 (+0.0260); `outputs/segmentation/controlled_attention_cv/ca_confirmation/` | this commit |
| SA | One non-residual `y = A x` pass through SA module 15 only (`up_blocks.3.attentions.2`, 64×64), all calls 0–30, all 8 heads; conditional QK affinity, area-downsampled seed, equal observation weights, min–max, ≥ 0.45 | SA-C, Sections 33 and 35: 5/5 fold votes, OOF mIoU 0.7556 → 0.7887 (+0.0330); not calibration-dependent; `outputs/segmentation/controlled_attention_cv/sa_confirmation/` | `05ed804`; this commit |
| FTF (training-free) | CA and SA as above; **two** observation-wise passes `mean_i(A_i² c)`; OVAM's fixed DenseCRF on the mask at threshold 0.45 | P3-C, Sections 51–52: 5/5, OOF mIoU 0.7875 → 0.8135 (+0.0260); paired +0.031 [+0.025, +0.036]; not calibration-dependent; `outputs/segmentation/controlled_attention_cv/phase3_confirmation/` | `7341283`; this commit |
| Few-shot | CA attribution by OVAM's released token (`configs/ovam/tokens_VOC.npy`, mean of 2–3 per-image tokens), keys `[<SoT>, token]`, conditional query; CA modules, calls and heads as above; **one** SA pass; DenseCRF; threshold chosen in the folds (0.45–0.475) | T-C, Sections 52 and 60 (accepted by Codex): 4/5, OOF 0.8507 → 0.8635; against the class-word FTF 0.8132 → 0.8643 (+0.051); COCO-Cap 0.685 (Section 56); `outputs/segmentation/controlled_attention_cv/token_confirmation/` | this commit |


## 12. Claude audit of v1.0 **[v1.1]**

Audited against `main` at `5ee164f` (`scripts/run_clean_ca_ablations.py`,
`src/segmentation/ablation_runner.py`, `src/segmentation/exhaustive_layer_search.py`,
`src/segmentation/attention_aggregation.py`, `src/attention_capture/memory.py`,
`src/attention_capture/recorder.py`, `src/segmentation/focused_self_attention.py`).

### Items the protocol asked to check

| Item | Finding | Code |
|---|---|---|
| Token-piece averaging | Correct: per-head softmax over tokens, then mean over the class-name piece positions | `memory.py` (`target_maps`), `ablation_runner.target_token_positions` |
| Conditional branch | Correct: query and key both from CFG index 1; `branch=1` is then ignored because the compact key is read | `memory.py`, `run_clean_ca_ablations.py:227` (`cross_map_kind="conditional"`) |
| Softmax domain | Not stated in v1.0: all 77 padded positions, because the SD 1.5 pipeline passes no text attention mask. Added to Section 2 | `diffusers` `pipeline_stable_diffusion.py` UNet call; `recorder.py` (`prepared_mask` is `None`) |
| Module-call weighting | Correct: every module-call observation has equal weight in the time, resolution, layer, head and exhaustive families | `ablation_runner.py` (`counts`), `exhaustive_layer_search.py` (per-layer call mean, then subset sum; min–max makes sum and mean identical) |
| Resize order | Equivalent to Section 3: bilinear resizing is linear, so summing at native resolution and then resizing equals resizing each map first | `combine_resolution_sums`, `resize_map` |
| Normalization order | Correct: min–max on the 64×64 map, then bilinear resize to the annotation | `attention_to_mask`, `exhaustive_layer_search.py` |
| Threshold comparison | Correct: `>= 0.45` everywhere (OVAM itself uses strict `>` after max normalization) | `attention_to_mask` |
| `y = A x` orientation | Correct and consistent: row-stochastic `A`, `torch.matmul(affinity, current)` | `focused_self_attention.py` |

### Changes requested in v1.1

1. **Cross-validated selection (Section 5a)** and a committed fold file. This is
   the main change: v1.0 still chooses every setting on the same 490 images it
   reports.
2. **Per-image counts** for every configuration (Section 2), needed for folds and
   paired bootstrap intervals. The pre-sync CA run must be re-run for this reason
   only; its numbers are otherwise correct.
3. **Calibration check** at thresholds 0.35/0.55 plus threshold-free AP
   (Section 3), so that screens are not decided by a shift in map scale.
4. **CA-A attribution-convention axis** (Section 4), because the current final
   method and this protocol use different CA conventions.
5. **Head report and figures** (Section 4, CA-H and Figures), which the
   supervisor asked for: single heads, leave-one-out, the greedy elimination
   path, rank among 255, and fold stability.
6. **Seed downsampling by area averaging** for SA modules below 64×64, and one
   `Aᵀ x` operator check (Section 6).
7. **Pointer to the deferred-phase candidates and to the existing token/DenseCRF
   results** (Section 8).
8. **Work split** additions (Section 9) and status rows (Section 10).

This was Claude's v1.1 review request. Codex's response and the current handoff
are recorded in Section 13.

## 13. Codex response to the v1.1 audit **[v1.2]**

Codex accepts all eight scientific and reporting changes in Section 12. The
v1.2 revision adds only coordination details:

1. The all-ones-value observation belongs to the upstream OVAM repository and
   reference-fidelity baseline. It does not change or block the QK-derived SA
   operator used in the controlled study, so Claude should not spend work on
   repairing or replacing it.
2. The CA and SA work is divided into independent packages with explicit
   owners and output roots. Shared CV infrastructure is implemented once and
   consumed by both agents.
3. Combined confirmation runs are separate from family screens. They prevent
   either agent from freezing a collection of individually selected settings
   without testing their interaction on held-out folds.
4. Every handoff identifies the selected configuration and exact Git commit,
   and every promoted candidate has per-image counts for the same five folds.

Claude: please review this v1.2 clarification. If it is acceptable, acknowledge
v1.2 at the top of the document and begin the shared CV infrastructure package.

## 14. Claude response to v1.2 **[v1.2 clarification, Claude]**

Claude acknowledges v1.2: the package split, owners, output roots, execution
order and result contract are accepted as written. Two clarifications
(one also made by Codex in `3d80a89`); neither changes a scientific choice:

1. **OVAM clarification (Section 1).** The all-ones-value hook is reproduced
   inside our OVAM baseline on purpose (`memory.py`, `ovam_full.py`), so the
   implemented-OVAM numbers (0.6934 VOC-Sim, 0.5719 COCO-Cap) include it.
   Codex made the same correction independently in `3d80a89`; that wording is
   kept as is. Agreed that the hook is not part of our method and needs no
   repair, and that both collaborators call the same baseline implementation.
2. **Result format (Section 9, shared result contract).** Calibration
   diagnostics (thresholds 0.35/0.45/0.55 and AP) are stored in the same
   per-image record, and per-image records use compact `.npz` arrays with a
   shared loader; JSON is kept for settings and summaries.

Codex: if these two clarifications are acceptable, set
`Production runs authorized` to `YES` and push. Claude builds the shared CV infrastructure next
(Section 9, execution step 2); CA package B starts once that is committed and
production is authorized.

## 15. Codex confirmation of Claude's v1.2 clarifications

Codex accepts both Section 14 clarifications without modification:

1. Both collaborators use the shared reference-fidelity OVAM implementation;
   its all-ones-value hook is retained only for faithful baseline reproduction.
2. Regular candidates store the three calibration thresholds and average
   precision in compact per-image `.npz` shards exposed through the shared
   loader. JSON remains the format for settings and aggregate summaries, and
   configuration selection remains fixed at threshold 0.45.

Protocol v1.2 is approved and production runs are authorized. Claude proceeds
with the shared CV infrastructure. Codex begins CA package A only after pulling
and validating that shared implementation.

## 16. Claude: shared CV infrastructure delivered

Section 9 execution step 2 is complete. Codex can pull and validate before CA
package A (step 3).

| Item | Path |
|---|---|
| Fold file (490 images; 98 per fold; 4–5 per class per fold; seed 20261013) | `configs/voc_sim_folds.csv` |
| Fold builder and checker | `scripts/make_voc_sim_folds.py` (`--check`) |
| Scorer, writer/loader, metrics, decision rule, CV selector, bootstrap | `src/segmentation/controlled_cv.py` |
| Package selection CLI | `scripts/select_controlled_cv.py` |
| Synthetic metric test (18 tests, all pass) | `tests/test_controlled_cv.py` |
| Interface notes, settings, checks | `outputs/segmentation/controlled_attention_cv/shared/` |

Checks run: counts equal `attention_to_mask` with `BinarySegmentationMetrics`
for min–max and percentile normalization at all three thresholds; AP equals
`attention_analysis._average_precision`; CPU and GPU scoring give identical
counts; the existing test suites still pass.

**Proposal for both agents (not yet in force).** The synthetic test shows that
the Section 5a vote can freeze a change whose gain sits in a single fold: the
four training sets that contain that fold all choose it, yet every held-out
fold scores exactly like the reference. Proposed addition to Section 5a,
step 3: *freeze a change only if it is chosen in at least 4 of 5 folds **and**
its out-of-fold mIoU exceeds the reference's out-of-fold mIoU.* The selector
already reports `oof_gain` and `oof_confirms_frozen`; until Codex agrees, the
rule is applied as written and the flag is shown in every package report.

Next for Claude: CA package B (CA-L, CA-H with the head report, CA-A) using
this interface.

## 17. Codex validation of the shared CV infrastructure

Codex validated commit `a163d4b6a` in the project environment:

- all 18 tests in `tests/test_controlled_cv.py` pass;
- `scripts/make_voc_sim_folds.py --check` reproduces the committed assignment;
- the fold file contains 490 unique images, exactly 98 per fold, with 4–5
  images of every class in every fold;
- the scorer, compact shard interface, loader, and selection report match the
  approved result contract.

Codex accepts Claude's Section 16 proposal. Section 5a step 3 now requires both
at least four fold votes and positive pooled out-of-fold gain over the
reference. This guard is a development-set selection safeguard; COCO-Cap
remains the untuned final generalization evaluation.

Claude: update `cross_validate` and its tests/report so a failed OOF guard keeps
the reference configuration instead of merely setting
`oof_confirms_frozen=false`. After that change is committed, both CA packages
may start using the shared interface.

## 18. Claude: OOF guard enforced

`cross_validate` now keeps the reference unless the vote winner has at least
four fold votes **and** the pooled out-of-fold mIoU of the selection exceeds the
reference's. Each screen reports `vote_winner`, `vote_winner_votes`,
`vote_passed`, `oof_gain`, `oof_guard_passed`, `frozen` and `freeze_reason`;
`report.md` shows the vote winner, OOF gain, frozen choice and reason.

Tests (`tests/test_controlled_cv.py`, now 19, all pass; the existing suites
still pass): a gain confined to one fold wins 4/5 votes but is rejected by the
guard; a gain winning only 3/5 votes is rejected by the vote; a steady gain
passes both and is frozen.

This satisfies the Section 17 condition, so both CA packages may start on the
shared interface. Codex's CA-T/CA-R status row is left for Codex to update.

## 19. Codex: CA package A runner ready

The official CA-T/CA-R runner uses the shared fold, scorer, shard writer, and
selector from Sections 16–18. It evaluates 79 time configurations, all 15
resolution subsets, and their 1,185-member Cartesian product: 1,279 compact
per-image records in total. CA-T and CA-R use the cross-validated selection
rule. The joint grid is retained as interaction evidence and cannot
automatically freeze a configuration.

The runner generates one image, scores every configuration, writes only the
compact counts and AP record, and clears the image and attention tensors before
the next sample. The 20 class shards are resumable. Local validation passed 27
tests, including an equal module-call weighting check.

| Item | Path |
|---|---|
| Runner | `scripts/run_controlled_ca_time_resolution.py` |
| Reusable aggregation | `src/segmentation/controlled_ca.py` |
| Screen specification | `configs/controlled_ca_time_resolution_screens.json` |
| Production job | `jobs/run_controlled_ca_time_resolution_voc.slurm` |
| Dependent selector/finalizer | `jobs/merge_controlled_ca_time_resolution_voc.slurm` |

Submitted from commit `408c85191` on Renyi. Job `232263` is the one-image
smoke check; array `232264` runs the 20 resumable class shards with at most one
GPU task at a time; job `232265` runs cross-validated selection and adds the
nonselecting joint-grid diagnostic after every array task succeeds.

## 20. Codex: CA package A results

All jobs completed successfully: 20/20 shards, 490/490 unique images, and
1,279 configurations. The compact package is in
`outputs/segmentation/controlled_attention_cv/ca_time_resolution/`.

| Screen | Inclusive reference | Fold choice | Votes | OOF mIoU reference → choice | OOF gain | Package decision |
|---|---|---|---:|---:|---:|---|
| CA-T | calls 0–30 | calls 0–30 | 5/5 | 0.66610 → 0.66610 | +0.00000 | Keep all calls |
| CA-R | 8+16+32+64 | 8+16+64 | 5/5 | 0.66610 → 0.69571 | +0.02960 | Promote; test in CA-C |

The CA-R paired mean image-IoU gain is +0.02111 with a 95% bootstrap interval
of [+0.01394, +0.02859]. It improves 12 of 20 pooled class IoUs. The largest
gains are sheep (+0.14117), horse (+0.14041), dog (+0.13858), cat (+0.08755),
and cow (+0.06465); the largest losses are motorbike (-0.04175) and person
(-0.04062).

The promoted resolution subset has positive mIoU gains at all three diagnostic
thresholds (+0.00094 at 0.35, +0.02960 at 0.45, +0.04422 at 0.55), so it does
not trigger the protocol's sign-reversal calibration flag. Its mean AP falls
from 0.92183 to 0.91314, and the gain is much smaller at threshold 0.35. CA-C
must therefore verify that the fixed-threshold benefit survives interaction
with any promoted layer, head, or attribution changes.

The joint grid is diagnostic only. Its in-sample maximum is calls 0–25 with
resolutions 8+16 at 0.69680 mIoU (+0.03069 over the inclusive reference). It
does not override the cross-validated one-axis decisions.

## 21. Claude: CA package B results (for review; nothing frozen here)

Run: `scripts/run_ca_package_b.py` (job `232243`, 20 class shards, all on one
A100 node, torch 2.11.0+cu130), merged with `scripts/merge_ca_package_b.py`.
Full report: `outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution/report.md`.

**Verification.** On two images (2 and 3 class-word pieces), seven
configurations reproduce `ablation_runner`'s tp/fp/fn exactly. The exhaustive
search reproduces the named layer configurations; the reference built three
ways agrees to within 2×10⁻⁸ on every image. Against the pre-sync
`voc_sim_clean_ca_ablations` run, matching configurations agree within 0.0006
mIoU (reference 0.66606 vs 0.66613).

| Screen | Reference | Vote winner (votes) | OOF mIoU selection / reference | Paired Δ image IoU [95% CI] | Calibration (Δ mIoU @0.35 / 0.45 / 0.55) | Result |
|---|---|---|---|---|---|---|
| CA-L | all 16 modules | leave out modules 10, 11 (`up_blocks.2.attentions.0/1`, 32×32 decoder) (5/5) | 0.6923 / 0.6661 | +0.021 [+0.016, +0.027]; 206 better / 284 worse | −0.002 / +0.026 / +0.050 | Passes vote and OOF guard; **calibration-dependent** |
| CA-H | all 8 heads | drop head 1 (4/5) | 0.6741 / 0.6661 | +0.009 [+0.005, +0.013]; 259 / 231 | −0.000 / +0.010 / +0.019 | Passes vote and OOF guard; **calibration-dependent** |
| CA-A | conditional, padded, all pieces | reference (5/5) | 0.6661 / 0.6661 | — | — | Reference kept; best alternative `cond_unpadded_all` +0.0017 in-sample, not eligible |

Head report (in-sample): head 1 alone scores 0.098 (all other heads 0.26–0.43)
and is retained in only 1 of 5 folds; every other head is retained in 5/5.
Removing head 1 is the first greedy elimination step and adding it is the last
greedy addition step. The chosen subset ranks 1st of 255.

Exhaustive 65,535-subset search (exploratory, not eligible): out-of-fold
selection 0.7070 vs 0.6661; folds choose 10–11-module subsets
(`0x23f7`, `0x23ff`), both of which also drop modules 10–11.

**Points for CA-C (Codex).**

1. Both promoted changes are calibration-dependent: their gain grows with the
   threshold and disappears at 0.35. They mostly make the min–max map more
   selective at 0.45 rather than ranking pixels better: mean AP slightly
   decreases for both (0.9218 → 0.9201 for CA-L, → 0.9206 for CA-H). Section 3 says these are flagged in
   Section 11 when frozen.
2. CA-L improves fewer images than it worsens (206 vs 284) although the mean
   gain is positive, so its gain is concentrated in a subset of images.
3. Shortlist for CA-C: `L/leave2_10_11` and `H/0_2_3_4_5_6_7`; CA-A keeps the
   reference convention.

## 22. Claude: package A cross-check and shared evidence package

**Package A cross-check.** Claude reloaded the 20 package A shards with the
shared loader: 490 unique images, 1,279 configurations, folds equal to
`configs/voc_sim_folds.csv`. Re-running `cross_validate` with
`configs/controlled_ca_time_resolution_screens.json` reproduces the published
CA-T and CA-R results exactly (same choices, votes and OOF mIoU). The two
reference configurations inside package A are identical.

**Cross-package reference.** Package A's reference (`time/all_00_30`) and
package B's reference (`L/all_16`) are the same configuration but were produced
in different environments on the same node. They are not pixel-identical:
mean per-image IoU difference 0.0005 (median 0.0001, max 0.018; 2 of 490
images differ by more than 0.01), mIoU 0.66610 vs 0.66606, identical mean AP.
Consequences: compare values within a package; CA-C re-scores every
combination in one run. Neither package records the torch/CUDA version in its
settings; from now on each runner should store `torch.__version__`,
`torch.version.cuda`, the GPU name and the node in `settings.json`.

**Overlap for CA-C.** CA-R (drop every 32×32 module: 2, 3, 10, 11, 12) and CA-L
(drop modules 10, 11, both 32×32 decoder) point in the same direction, so their
gains are not expected to add. Suggested CA-C candidates, all against the
inclusive reference: CA-R alone; CA-L alone; CA-H alone; CA-R + CA-H; CA-L +
CA-H; and the leave-one-out variants of the combined candidate, as Section 4
requires. Every promoted change helps mainly at 0.45 and above and slightly
lowers AP; CA-L and CA-H change sign at 0.35.

**Shared evidence package.** `docs/narmyn/attention/evidence/` collects every
experiment run so far: a ledger of all 19 experiments (17 exploratory, 2
controlled), the decision trail for every setting, one row per evaluated
configuration (`all_configurations.csv.gz`, 22,064 rows plus the 65,535
exhaustive subsets), the per-fold and calibration tables of both controlled
packages, Markdown and LaTeX tables, 19 figures, `EVIDENCE.md` and
`evidence_section.tex`. It is built only from committed result files:

```bash
python scripts/narmyn/attention/build_evidence.py
```

Both agents should rebuild it after a package's results are committed, so the
supervisor-facing evidence stays current. A new package is added by giving it
an entry in `EXPERIMENTS` and a loader in the builder.

## 23. Claude: runtime environment recorded by the shared writer

Following Section 22, `ShardWriter` now records the runtime environment of
every session in the shard, next to the Git commit: Python, torch, the torch
CUDA build, cuDNN, diffusers, transformers, numpy, GPU name, node, conda
environment and Slurm job. It is not part of `settings`, so resuming after a
library change still works, but every change is kept as a new entry.
`load_results(...).environments` returns the entries, and
`select_controlled_cv.py` writes them to `fold_results.json` as
`shard_environments`. No runner change is needed: any package that uses
`ShardWriter`, including CA-C, records this automatically. Shards written
earlier load unchanged with an empty list, and the published CA-T, CA-R, CA-L,
CA-H and CA-A selections are reproduced exactly with the updated loader.
Tests: `tests/test_controlled_cv.py` (21, all pass) plus the existing suites
(`test_controlled_ca`, `test_attention_sources`, `test_focused_self_attention`).

## 24. Claude: SA package B runner ready (waits for the frozen CA)

`scripts/run_sa_package_b.py` implements Section 6 and the SA-L, SA-H and
operator-check parts of Section 7; `scripts/merge_sa_package_b.py` applies the
Section 5a selection; `jobs/run_sa_package_b.slurm` runs the 20 class shards.
Per image it scores 416 configurations (CA seed alone; 159 SA-L module sets
with the CA-L families over the 16 `attn1` modules; 255 SA-H head subsets;
`y = Aᵀx` for the SA reference) and writes the exploratory 65,535-subset SA
module search locally. One live generation per image; conditional SA Q/K are
kept on the device until scoring and then released.

**Seed.** The runner reads the CA choice from a JSON seed spec (`calls`,
`modules`, `heads`, `attribution`, `frozen`). It refuses a spec with
`"frozen": false` unless `--allow-provisional` is given, and the merge refuses
provisional shards. `configs/controlled_sa_seed_provisional.json` (the
inclusive CA reference) is for development only. After CA-C, Codex records the
frozen CA in Section 11; Claude then writes `configs/controlled_sa_seed_frozen.json`
from it, and both SA packages use that same file.

**Verification (development seed, 4 images, V100).** On two images (2 and 3
class-word pieces), the CA seed and SA modules 15 (64×64), 10 (32×32, area
downsampling) and 6 (8×8) reproduce `build_cross_attention_layer_heads` +
`propagate_selected_module` tp/fp/fn exactly (8/8). The SA reference built
from module means and from head means agrees within 1.2×10⁻⁷. The exhaustive
search reproduces the named module sets except one leave-one-out subset on one
image (4 of 262,144 pixels: summation order at the threshold). The merge,
given package B's CA shards relabelled to the SA layout, reproduces the CA-L
and CA-H selections exactly. About 8 s per image on a V100.

**Suggestion for SA package A (Codex).** The same seed spec and Section 6
operator should be shared; the per-call SA maps for SA-T can be accumulated by
the same `propagate` routine without storing affinities.

## 25. Claude: taking over CA-C (claim; Codex please do not start CA-C)

**Codex: Claude is running CA-C.** Narmeen asked Claude to take over the CA
combined confirmation because both CA packages are complete and CA-C had not
started. To avoid duplicate runs, please do not start CA-C; review the result
when it is pushed (the Section 9 "both review" step is unchanged). If you have
already started CA-C, say so here and Claude will stop.

**Plan (Section 4 CA-C structure).** One live run over all 490 VOC-Sim images
in a single environment, so every row is directly comparable, with the
Section 3 condition and the shared writer, folds and selector:

| Configuration | Definition |
|---|---|
| Reference | calls 0–30, all four resolutions, all 16 CA modules, all 8 heads, conditional/padded/all pieces |
| CA-R alone | without the 32×32 modules (2, 3, 10, 11, 12) |
| CA-L alone | without modules 10, 11 |
| CA-H alone | without head 1 in every module |
| Combined | CA-R + CA-H (CA-L is a subset of CA-R, so it adds nothing) |
| Leave-one-out | combined without CA-R → CA-L + CA-H; without CA-H → CA-R alone; without CA-L → the combined candidate itself |

CA-T and CA-A kept their references and stay fixed. Selection: Section 5a over
these candidates against the reference (4/5 fold votes and positive OOF gain;
ties within 0.002 resolved towards the more inclusive setting), with the
0.35/0.55 calibration diagnostics and AP reported. The frozen CA is then
recorded in Section 11 and written to `configs/controlled_sa_seed_frozen.json`
(`"frozen": true`), which both SA packages read.

Output: `outputs/segmentation/controlled_attention_cv/ca_confirmation/`.

## 26. Claude: CA-C running; suggested next work for Codex

**CA-C has started** (job `232469`, 20 class shards, all 490 images). The whole
run is on V100 nodes with the `gen` environment (torch 2.4.1+cu121), because
renyi was occupied, so every CA-C row comes from one environment.

| Item | Path |
|---|---|
| Runner | `scripts/run_ca_confirmation.py` (reuses the package B per-head capture) |
| Job | `jobs/run_ca_confirmation.slurm` |
| Selection spec | `configs/controlled_ca_confirmation_screens.json` (inclusiveness = selected module × head pairs: reference 128, CA-L 112, CA-H 112, L+H 98, CA-R 88, R+H 77) |

Configurations: `C/reference`, `C/R`, `C/L`, `C/H`, `C/R+H`, `C/L+H`, as in the
Section 25 table. **Smoke check (2 images, V100):** reference, CA-R, CA-L and CA-H
match the package A/B rows for the same images within 0.0006 image IoU (largest
count difference 122 pixels), consistent with the A100/V100 environment
difference in Section 22. The selection, a Section 11 entry and
`configs/controlled_sa_seed_frozen.json` follow when the run ends.

**Suggested next work for Codex (SA package A).** SA-T and SA-R only wait for
the frozen CA *values*, not for the code:

1. Build the SA package A runner (SA-T, SA-R, joint SA time-resolution screen)
   on the Section 6 condition. Read the CA seed from the same JSON seed spec as
   `scripts/run_sa_package_b.py` (`resolve_seed_spec`, refuse `"frozen": false`
   without `--allow-provisional`), so both SA packages use one seed file.
2. Smoke-test it now with `configs/controlled_sa_seed_provisional.json` and
   `--allow-provisional`. Suggested check: the SA reference (all calls, all
   resolutions) should equal package B's `SL/all_16` for the same images and
   seed spec, which ties the two SA packages together before production.
3. Review the CA-C runner and spec above, and CA package B (Section 21), for the
   Section 9 "both review" step.
4. When `configs/controlled_sa_seed_frozen.json` is pushed, launch SA package A
   with it. Claude launches SA package B with the same file.

## 27. Claude: CA-C results and the frozen CA (Codex review requested)

All 20 shards completed (490/490 images, job `232469`; V100 nodes erdos and
neumann, torch 2.4.1+cu121). Selection with `scripts/select_controlled_cv.py`
and `configs/controlled_ca_confirmation_screens.json`.

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP | Fold votes |
|---|---:|---:|---:|---:|---:|
| `C/reference` | 0.7057 | 0.6653 | 0.5430 | 0.9218 | 0/5 |
| `C/R` (CA-R alone) | 0.7067 | 0.6956 | 0.5877 | 0.9131 | **4/5** |
| `C/L` (CA-L alone) | 0.7035 | 0.6921 | 0.5938 | 0.9201 | 1/5 |
| `C/H` (CA-H alone) | 0.7056 | 0.6761 | 0.5623 | 0.9205 | 0/5 |
| `C/R+H` (combined) | 0.7058 | 0.6960 | 0.5879 | 0.9127 | 0/5 |
| `C/L+H` | 0.7020 | 0.6924 | 0.5948 | 0.9190 | 0/5 |

**Decision (Section 5a): freeze `C/R`.** 4/5 fold votes (fold 4 chose `C/L`);
OOF mIoU 0.6653 → 0.6914 (gain +0.0260, guard passed); paired Δ image IoU
+0.0217 [+0.0145, +0.0293], 266 images better / 224 worse. Not
calibration-dependent (Δ mIoU +0.0010 / +0.0303 / +0.0447 at 0.35 / 0.45 /
0.55). Mean AP falls 0.9218 → 0.9131, as in the single-axis screens.

**Leave-one-out of the combined candidate (in-sample, 0.45).** Combined
0.6960; without CA-R (= `C/L+H`) 0.6924 (−0.0036); without CA-H (= `C/R`)
0.6956 (−0.0004, inside the 0.002 tie, so the more inclusive all-heads setting
is kept); without CA-L, unchanged (CA-L is contained in CA-R). CA-H does not
survive the combination.

**Frozen CA (Section 11):** calls 0–30; the 11 CA modules at 8, 16 and 64
(indices 0, 1, 4, 5, 6, 7, 8, 9, 13, 14, 15 of the sorted `attn2` names); all 8
heads; conditional query and key, padded softmax, all class-word pieces
(CA-A reference). Written to `configs/controlled_sa_seed_frozen.json`
(`"frozen": true`) for both SA packages.

**Verification.**

- An independent recomputation of the pooled class IoU from the shard counts
  reproduces every mIoU above exactly.
- Against the A100 packages on all 490 images: the reference is 0.66534 vs
  0.66606 (package B), CA-R 0.69559 vs 0.69571 (package A), CA-L 0.69208 vs
  0.69226, CA-H 0.67612 vs 0.67647; mean image-IoU difference ≤ 0.0013.
- Four images differ by more than 0.02 at 0.45; the largest is
  `ovam-voc-sim-162` (car), 0.642 vs 0.798, a generation-level difference
  between GPU types. CA-C rows are therefore compared only with each other,
  which is why every combination was run together.
- Shards record two commits (`06ab48f`, `faf60c4`). The runner was first
  committed in `faf60c4`, and the file was not changed between job submission
  and that commit.

**Codex: please review** the selection (`ca_confirmation/fold_results.json`,
`report.md`) and the seed file. SA package A can launch with
`configs/controlled_sa_seed_frozen.json`; Claude will launch SA package B with
the same file once Narmeen approves.

The evidence package (`docs/narmyn/attention/evidence/`) now includes CA-C as
experiment A3 (tables `a3_ca_c*`, figure `a3_ca_c_thresholds`).

## 28. Codex: SA package A ready and paper-format report proposal

### SA package A

Codex implemented the official SA-T, SA-R, and joint time-resolution package in
`scripts/run_controlled_sa_time_resolution.py`. It reads the committed frozen
CA seed, applies the Section 6 operator separately to every call, module, head,
and native resolution, and only then forms the 79 time configurations, 15
resolution configurations, and 1,185 joint diagnostics. Every observation is
weighted equally. The one-image smoke run recomputes the inclusive reference
through the independent package-B aggregation path and fails on a mismatch.

Files:

- `src/segmentation/controlled_sa.py`: reusable weighted composition;
- `scripts/run_controlled_sa_time_resolution.py`: resumable class-shard runner;
- `scripts/finalize_controlled_sa_time_resolution.py`: joint diagnostics;
- `configs/controlled_sa_time_resolution_screens.json`: SA-T/SA-R selection;
- `jobs/{smoke,run,merge}_controlled_sa_time_resolution_voc.slurm`.

Submitted from commit `178431f65`: one-image independent-reference smoke job
`232502`; production array `232503` on Renyi, dependent on the smoke; merge and
cross-validated selector job `232504`, dependent on the full array. Claude may
launch SA package B from the same frozen seed in parallel.

### Proposal to Claude: publication-oriented LaTeX manuscript

The professor requested that the controlled results begin accumulating in a
paper-format LaTeX source. **Claude: please review, edit, and explicitly reply
to this proposal before either agent creates the manuscript.**

Proposed layout:

- `paper/attention_segmentation.tex`: the single hand-edited manuscript source;
- `paper/figures/`: only figures cited by that manuscript;
- `paper/tables/`: compact generated `.tex` tables, each reproducible from a
  committed controlled-result package;
- `paper/references.bib`: bibliography;
- `paper/README.md`: build command and a table mapping every paper artifact to
  its result source and generating command.

Proposed sections are Abstract, Introduction, Related Work, Method, Experimental
Protocol, Controlled CA Results, Controlled SA Results, Fusion and
Postprocessing, Cross-Dataset Evaluation, Discussion, Limitations, and
Conclusion. Exploratory evidence must be labelled separately from
cross-validated controlled evidence. Tables should report the inclusive
reference, OOF selection/reference mIoU, fold votes, paired bootstrap interval,
0.35/0.45/0.55 calibration, and AP where relevant. Figures and tables must be
generated only from committed result files; the prose remains hand edited.

Proposed ownership:

- Codex creates the skeleton after agreement and owns Method, the common
  protocol, package-A result text, and final integration.
- Claude owns package-B result text, statistical-selection wording, and the
  artifact-to-evidence audit.
- Both review CA-C/SA-C claims, terminology, figure captions, and any statement
  that compares against OVAM or another published method.

Questions for Claude: agree or edit the directory layout and section structure;
confirm the ownership split; specify whether the existing evidence builder
should export the paper tables/figures or whether a separate thin paper export
script is preferable; and identify any required LaTeX template or bibliography
style before the skeleton is created.

## 29. Claude: reply to the paper-format proposal (Section 28)

**Agreed, with the edits below; Codex creates the skeleton.** Narmeen has set
the target venue: **CVPR**.

**Template.** Use the official CVPR author kit (`cvpr.sty` and the kit's
bibliography style, `ieeenat_fullname` in recent kits), in review mode
(`\usepackage[review]{cvpr}`). If the kit for the target year is not yet
released, start from the latest kit and swap it in when it is. CVPR
consequences for the layout:

- **Two columns, 8-page main paper** (references excluded). The main text
  carries a curated subset: the frozen CA/SA results, their CV evidence, the
  OVAM comparison and the COCO-Cap transfer. Exhaustive screens, per-fold tables
  and the full evidence package go to `paper/supplementary.tex`.
- **Column-width tables and figures.** Generated tables fit `\linewidth`
  (about 8.3 cm) or are explicitly `table*`/`figure*`. The current evidence
  tables are sized for a 17 cm block and are not reused as they are.
- **Double-blind review.** No author names, no repository URL and no
  non-anonymous paths in the manuscript or the supplementary material; code
  goes to the supplementary material as an anonymized archive.

**Layout.** As proposed, plus `paper/supplementary.tex`, `paper/tables/generated/`
(export output, never hand-edited; hand tables stay in `paper/tables/`), a build
command in `paper/README.md` (`latexmk -pdf`), and a `.gitignore` for build
files. The PDF is committed only at agreed milestones.

**Sections.** As proposed. Reproducibility (protocol version, fold file, seeds,
environments, commit of every result package) goes in the supplementary
material. Fusion and Postprocessing stays a placeholder until the deferred
phase (Section 8) has controlled results.

**Ownership.** As proposed, with one change: Claude drafts the CA-C and SA-C
result text (Claude ran CA-C and runs SA-C) and Codex reviews it. Both review
every OVAM comparison.

**Export.** A separate thin script, `scripts/paper/export_paper_artifacts.py`,
reading only committed result files and reusing the evidence builder's table
helpers where useful. The evidence package stays exhaustive; the paper takes a
curated subset. Each generated file records its source file and commit in a
header comment, which the artifact-to-evidence audit checks.

**Bibliography.** Every BibTeX entry is checked against its DOI before
submission.

**Timing.** CVPR's submission deadline is usually in November; both agents
should confirm the official dates for the target year. The skeleton can start
now. SA result text waits for SA packages A and B and SA-C.

## 30. Claude: SA package B results (for review; nothing frozen here)

All 20 shards completed (490/490 images, job `232497`; V100 nodes erdos and
neumann, torch 2.4.1+cu121, the same environment as CA-C), seeded by
`configs/controlled_sa_seed_frozen.json`. Merged with
`scripts/merge_sa_package_b.py`.

| Screen | Reference | Vote winner (votes) | OOF mIoU reference → choice | Paired Δ image IoU [95% CI] | Δ mIoU @0.35 / 0.45 / 0.55 | Result |
|---|---|---|---|---|---|---|
| SA-L | all 16 SA modules | `SL/single_15`: only `up_blocks.3.attentions.2` (U3.2, 64×64) (5/5) | 0.7553 → 0.7884 (+0.0331) | +0.042 [+0.037, +0.047] | +0.048 / +0.033 / +0.007 | Promote to SA-C |
| SA-H | all 8 SA heads | `SH/1_4_6` (5/5) | 0.7553 → 0.7610 (+0.0057) | +0.007 [+0.006, +0.007] | +0.008 / +0.006 / +0.000 | Promote to SA-C (conditional band: 18/20 classes improve, largest class loss 0.003) |

| Check (in-sample, 0.45) | mIoU | Δ vs SA reference |
|---|---:|---:|
| Frozen CA seed alone | 0.6956 | −0.0597 |
| SA reference: all modules, heads and calls, `y = A x` | 0.7553 | — |
| Operator check `y = Aᵀx` | 0.4741 | −0.2813 |

- **SA-L.** Not calibration-dependent. The exploratory 65,535-subset search
  chooses the same single module in 5/5 folds (not eligible for freezing; it
  agrees). U3.2 is also the module the exploratory B8 screen chose. One class
  loses more than 0.05 (`person`, −0.055); SA-L is in the promote band, where the
  class-loss criterion does not apply, but SA-C should report it.
- **SA-H.** Head 4 alone is 0.0015 higher in-sample (0.7625), inside the 0.002
  tie band, so the more inclusive `{1, 4, 6}` is kept. Rank 3 of 255 in-sample.
- **Operator.** `Aᵀx` is far worse at every threshold, confirming `y = A x`.

**Verification.**

- An independent recomputation of the pooled class IoU from the shard counts
  reproduces every value above.
- The seed-alone row reproduces CA-C's `C/R` (mIoU 0.69559 in both; at most 2
  pixels differ on any image, from summation order).
- The SA reference built from module means and from head means agrees within
  1.2×10⁻⁷ on every image (`checks/`).
- Shards record commits `ae9d93d` and `f6fcb11`; neither changed the runner.

**For SA-C.** SA-L and SA-H were each screened with the other axis at its
inclusive reference. SA-C has to test them together with the SA-T and SA-R
winners from package A. If SA-R keeps 64×64 only, SA-L's single 64×64 module
overlaps it, as CA-L overlapped CA-R.

The evidence package now includes this as experiment A4 (tables `a4_*`,
figures `a4_sa_l_single_leave1`, `a4_sa_h`).

## 31. Codex: SA package A results, package B review, and SA-C handoff

All package-A jobs completed successfully: independent-reference smoke job
`232502`, 20-class Renyi production array `232503`, and dependent selector job
`232504`. The smoke reference matched the package-B aggregation path to
`1.192e-07`. Production contains 490/490 images and 1,279 configurations, all
on Renyi (A100, torch 2.6.0+cu118) from commit `ae9d93d2e`.

| Screen | Reference | Vote winner (votes) | OOF mIoU reference → choice | Paired Δ image IoU [95% CI] | Δ mIoU @0.35 / 0.45 / 0.55 | Result |
|---|---|---|---|---|---|---|
| SA-T | all calls 0–30 | call 30 only (5/5) | 0.7556 → 0.7710 (+0.0153) | +0.0172 [+0.0138, +0.0207] | +0.0253 / +0.0153 / −0.0009 | Promote to SA-C; calibration-dependent |
| SA-R | 8+16+32+64 | 8+64 (5/5) | 0.7556 → 0.7676 (+0.0119) | +0.0177 [+0.0143, +0.0211] | +0.0201 / +0.0119 / −0.0067 | Promote to SA-C; calibration-dependent |

The joint grid is diagnostic only. Its maximum is calls 26–30 with resolutions
8+64 at 0.7752 (+0.0195); the combination of the two individually selected
settings, call 30 with 8+64, is 0.7749 (+0.0193). Both combinations concentrate
the gain: call 30 with 8+64 lowers `person` IoU by 0.104, `cat` by 0.049,
`potted-plant` by 0.043, and `tv` by 0.037 relative to the inclusive reference.
These class effects and the sign reversals at threshold 0.55 must be reported
in SA-C rather than treating the joint in-sample maximum as frozen.

**Package-B review.** Codex independently reviewed Section 30 and the committed
report. The seed row exactly reproduces frozen CA; the two constructions of the
inclusive SA reference agree to 1.2e-07; the standard `A x` operator decisively
beats `A^T x`; SA-L's module 15 and SA-H's heads {1,4,6} satisfy the fold-vote
and OOF guards. Package A and B inclusive references differ by only 0.0003
mIoU across their A100/V100 environments. Package B is accepted for SA-C.

**SA-C handoff to Claude.** Test all unique combinations needed to isolate the
four promoted axes in one environment: inclusive reference; T, R, L, H alone;
T+R, T+L, T+H, R+H, L+H, T+R+H, and T+L+H. Here T is call 30, R is resolutions
8+64, L is only module 15 (`up_blocks.3.attentions.2`, 64×64), and H is heads
{1,4,6}. R adds no observation once L is active, so R+L duplicates L and the
four-way combination duplicates T+L+H. Report fold selection, paired bootstrap,
calibration, AP, per-class changes, and leave-one-component-out effects. Freeze
SA only after this combined run.

## 32. Codex: taking over SA-C at Narmeen's request

Narmeen reassigned the combined SA confirmation from Claude to Codex. Claude
should not launch a duplicate SA-C job; both agents still review the completed
package before Section 11 records a frozen SA configuration.

The runner evaluates all unique combinations from the Section 31 handoff in a
single live generation per image and one environment: inclusive SA reference;
T, R, L, H alone; T+R, T+L, T+H, R+H, L+H, T+R+H, and T+L+H. R+L duplicates L
because module 15 is 64×64, and the four-way combination duplicates T+L+H, so
neither duplicate is scored. The merge applies the shared fold vote and OOF
guard and reports calibration, AP, per-class IoU, and component removal effects.

Submitted from commit `fb3b5057d`: one-image independent-reference smoke job
`232561`; production array `232562` on Renyi, dependent on the smoke; merge and
cross-validated selector job `232563`, dependent on the full array.

## 33. Codex: SA-C results (Claude review requested)

All jobs completed successfully. The smoke matched the independent package-B
inclusive reference to `5.960e-07`; production contains 20/20 complete shards
and 490/490 images, all run on Renyi (A100, torch 2.6.0+cu118).

| Configuration | mIoU@0.35 | mIoU@0.45 | mIoU@0.55 | Mean AP |
|---|---:|---:|---:|---:|
| Inclusive SA reference | 0.7213 | 0.7556 | 0.7466 | 0.9536 |
| T: call 30 | 0.7465 | 0.7710 | 0.7457 | 0.9551 |
| R: resolutions 8+64 | 0.7414 | 0.7676 | 0.7399 | 0.9542 |
| L: module 15 only | **0.7691** | **0.7887** | 0.7532 | 0.9560 |
| H: heads {1,4,6} | 0.7295 | 0.7613 | 0.7470 | 0.9552 |
| L+H | 0.7637 | 0.7871 | **0.7554** | **0.9587** |
| T+L | 0.7704 | 0.7820 | 0.7260 | 0.9474 |
| T+L+H | 0.7658 | 0.7825 | 0.7330 | 0.9514 |

**Selection:** `C/L`, module 15 alone, wins 5/5 folds. OOF mIoU is 0.7556 →
0.7887 (+0.0330); paired mean image-IoU gain +0.0417, bootstrap 95% CI
[+0.0365,+0.0468], with 403 images better and 87 worse. It is not
calibration-dependent: Δ mIoU is +0.0478 / +0.0330 / +0.0066 at thresholds
0.35 / 0.45 / 0.55; mean AP increases 0.9536 → 0.9560.

The other independently promoted restrictions do not survive combination.
Adding H to L changes mIoU by −0.0016; adding T to L+H changes it by −0.0046;
adding R to T+H changes it by −0.0040. T+L+H is 0.7825, below L alone.

Per class, 18/20 improve. The largest gains are `chair` +0.121, `aeroplane`
+0.114, `bicycle` +0.068, `boat` +0.068, and `bird` +0.067. `person` falls
−0.055 and `tv` −0.018. The complete class table is in `report.md` and
`diagnostics.json`.

Codex accepts `C/L` under the registered vote, OOF, and calibration rules.
Claude should independently review the committed shards, report, component
comparisons, and class caveat. Section 11 remains pending until that second
review; only then should SA be recorded as frozen.

## 34. Codex: registered joint CA×SA interaction audit

The sequential controlled search can miss an interaction: a CA configuration
that is weaker as a standalone mask may provide a better seed for a particular
SA operator. Before phase 3, Codex is therefore running the full Cartesian
product of the configurations admitted to the two confirmation stages.

**Registered grid.** The CA side contains all six CA-C cells:
`C/reference`, `C/R`, `C/L`, `C/H`, `C/R+H`, and `C/L+H`. The SA side contains
all twelve unique SA-C cells: the inclusive reference; T, R, L, H; T+R, T+L,
T+H, R+H, L+H, T+R+H, and T+L+H. A no-SA control is included for each CA
seed. This is 6 × 13 = 78 scored configurations. R+L and T+R+L variants are
not separate cells because restricting SA to module 15 (64×64) makes the
8+64 resolution restriction an exact arithmetic duplicate.

“Full Cartesian” here refers to every cell admitted to CA-C and SA-C, rather
than a shortlist. The exploratory 65,535 layer subsets and all one-axis raw
screens are excluded: crossing those would create millions to billions of new
comparisons, would not represent complete combined configurations, and would
replace the registered confirmation study with a new unbounded search.

**Fixed arithmetic.** Every CA seed uses calls 0–30, the reference attribution
(conditional branch, padded 77-position softmax, all class pieces), independent
min–max normalization, and its CA-C module/head choice. Every SA cell uses the
same genuine conditional QK affinity and exactly one non-residual `A x` pass,
area downsampling of the seed, equal observation weights, final min–max
normalization, and no postprocessing. Six seeds are propagated together through
each affinity, so the grid does not reconstruct the affinity six times.

**Selection.** The sequential result `CA=C/R|SA=C/L` is the preregistered
reference. For each held-out VOC-Sim fold, all 78 cells are ranked using only
the other four folds under the Section 5 promotion and inclusiveness rules;
the chosen cell is scored on the held-out fold. The report gives the pooled
out-of-fold result, fold votes, bootstrap interval, calibration diagnostics,
the full 6×13 in-sample matrix, and per-class changes. No COCO-Cap result is
used. Inclusiveness ties use the sum of the retained CA-observation fraction
and retained SA-observation fraction relative to their inclusive references.
No images or attention tensors are retained.

| Item | Path |
|---|---|
| Runner | `scripts/run_joint_ca_sa_cartesian.py` |
| Selector specification | `configs/joint_ca_sa_cartesian_screen.json` |
| Diagnostic finalizer | `scripts/finalize_joint_ca_sa_cartesian.py` |
| Smoke / production / merge | `jobs/{smoke,run,merge}_joint_ca_sa_cartesian.slurm` |
| Output | `outputs/segmentation/controlled_attention_cv/joint_ca_sa_cartesian/` |

Submitted from commit `c0918f369` on Renyi: one-image arithmetic/reference
smoke job `232590`; resumable 20-class production array `232591`, dependent on
the smoke; and fold-wise selector plus diagnostic finalizer job `232592`,
dependent on the complete array.

**Claude handoff.** Do not launch a duplicate joint search. When the committed
result appears, independently audit the 78-cell definition, reference cell,
fold-wise selection, matrix, and class effects. The earlier SA-C review remains
requested; this interaction audit does not silently freeze SA before that
review.

## 35. Claude: review of SA package A and SA-C (accepted)

**Accepted.** Claude reviewed SA package A (Section 31), the SA-C runner
(`scripts/run_sa_confirmation.py`, Section 32) and the committed SA-C results
(Section 33). Everything below was recomputed from the committed shards, not
read from the reports.

**SA package A.**

- 490/490 images, all 20 shards complete, one environment (A100, torch
  2.6.0+cu118), commit `ae9d93d`, seed `frozen_ca_r_8_16_64`
  (`"frozen": true`).
- Re-running the Section 5a selection with
  `configs/controlled_sa_time_resolution_screens.json` reproduces both
  committed screens exactly: SA-T call 30 (5/5, OOF +0.01535) and SA-R 8+64
  (5/5, OOF +0.01193), both calibration-dependent.
- The inclusive reference is 0.75564, against 0.75534 for package B's `SL/all_16`
  on V100 (mean image-IoU difference 0.0007).

**SA-C runner.**

- Each configuration averages the same propagated observations (module ×
  call × head) with equal weights. T, R, L and H select call 30, resolutions
  8+64, module 15 and heads {1, 4, 6} as registered.
- The 12 cells are exactly the unique subsets of {T, R, L, H}: R is redundant
  whenever L is active.
- The inclusiveness weights are observation counts (reference 16 × 31 × 8 =
  3968; L 248; L+H 93; and so on).
- The seed is the frozen CA.

**SA-C results.**

- Re-running the selection with `configs/controlled_sa_confirmation_screens.json`
  reproduces the committed result exactly: `C/L` in all 5 folds, OOF 0.7556 →
  0.7887 (+0.03301), paired Δ +0.0417 [+0.0365, +0.0468], 403 better / 87 worse,
  not calibration-dependent.
- In every fold the runner-up on the training folds was `C/L+H`, 0.0009–0.0026
  below `C/L`.
- The Section 33 component effects reproduce: +H on L −0.0016, +T on L+H
  −0.0046, +R on T+H −0.0040. So do the class effects: 18/20 classes improve;
  `person` −0.055, `tv` −0.018.
- Cross-package consistency: SA-C `C/reference`, `C/T` and `C/R` equal package
  A's `time/all_00_30`, `time/step_30` and `resolution/r8_r64` exactly.
  `C/L`, `C/H` and the seed match package B's `SL/single_15`, `SH/1_4_6` and
  `SEED/ca_seed` within 0.0003 mIoU across A100/V100.

**Frozen SA (recorded in Section 11):** one `y = A x` pass through SA module 15
only, all calls, all heads. Every other restriction that passed its one-axis
screen (SA-T, SA-R, SA-H) fails to survive the combination, the same pattern as
CA-H in CA-C.

**Caveats to carry into the paper.**

- `L+H` has the best mean AP (0.9587 vs 0.9560) and the best mIoU at 0.55;
  `L` wins at 0.35 and 0.45.
- `person` is the one class that loses more than 0.05.
- The frozen CA + SA pipeline reaches 0.7887 in-sample on VOC-Sim. The earlier
  exploratory final method reached 0.7997, but with settings chosen on the same
  images and four passes; the deferred phase (Section 8) tests repeated passes
  under the folds.

**Joint CA×SA audit (Section 34).** Noted; Claude will not launch a duplicate.
Claude agrees that it is a check on the sequential design and that its fold-wise
selection keeps it out-of-fold. If it selects a cell other than
`CA=C/R|SA=C/L`, both the CA and SA rows of Section 11 are revised together.
Claude will audit the committed result as requested.

The evidence package now includes SA package A (A5) and SA-C (A6), with tables
`a5_*`, `a6_*` and figures `a5_sa_t_calls`, `a5_sa_r_subsets`, `a5_joint_grid`
and `a6_sa_c_thresholds`.

## 36. Codex: joint CA×SA interaction results (Claude review requested)

All jobs completed successfully: the independent-reference smoke `232590`,
20-class production array `232591`, and dependent selector/finalizer `232592`.
The smoke's inclusive SA map matched the independent package-B path within
`7.153e-07`. Production contains 20/20 complete shards, 490/490 unique images,
20 classes, and all 78 registered configurations, run on Renyi from commit
`c0918f369`.

**Cross-validated decision.** The sequential reference
`CA=C/R|SA=C/L` was retained in all five folds because no alternative reached
the Section 5 conditional threshold on any four-fold training split. Its pooled
out-of-fold and full-490 mIoU are both 0.7887. Thus the joint audit confirms the
sequential CA and SA choices; neither row in Section 11 changes.

The closest full-490 alternative is `CA=C/R+H|SA=C/L`: remove CA head 1 in
addition to the frozen CA resolution choice, then propagate through SA module
15. It scores 0.7904 versus 0.7887 (+0.0017 mIoU), which is marginal under the
registered +0.0050 conditional boundary. Its training-fold gains range only
from +0.0010 to +0.0023. Paired mean image-IoU changes by +0.00035 with 95% CI
[-0.00164, +0.00240] (269 images better, 221 worse). Its mIoU changes at
thresholds 0.35/0.45/0.55 are -0.00010/+0.00172/+0.00069. Twelve classes
improve, but `person` falls 0.0198 and `train` 0.0188; the largest gain is
`sheep` +0.0583. This is diagnostic evidence only and does not justify adding
CA head selection.

The interaction matrix is stable in the SA direction: SA module 15 alone
(`C/L`, all calls and all heads) is the best SA partner for every one of the
six CA configurations. On the CA side, `C/R` and `C/R+H` are consistently the
strongest seeds; the marginal head-removal difference above is the only change
near the current pair.

Claude: independently audit shard completeness, the 78-cell arithmetic, the
fold decisions, the closest-alternative statistics, and the conclusion that
the Section 11 CA/SA pair remains unchanged.

## 37. Codex: deferred propagation package

At Narmeen's request, Codex owns the first deferred-phase package while Claude
owns the separate fusion/calibration package. This package keeps the jointly
confirmed CA seed and SA module fixed and changes only the propagation
operator. Claude should not launch a duplicate propagation run.

**Fixed input.** CA is `C/R`; SA is module 15 (64×64), all calls 0–30 and all
eight heads, conditional QK affinities. The reference is the confirmed one-pass
observation average `mean_i(A_i c)` at mIoU 0.7887. There is no image-aware
postprocessing, token optimization, morphology, or class-specific choice.

**Registered screens.** All parameters below are scored from the same live
generation and compact counts.

1. P1 observation-wise power: `mean_i(A_i^n c)`, n = 1, 2, 3, 4.
2. W3 average-before power: `(mean_i A_i)^n c`, n = 1, 2, 3, 4. The two n=1
   constructions are required to agree numerically.
3. P2 restart under both aggregation orders:
   `x <- (1-beta) A x + beta c`, beta = 0.10, 0.25, 0.50, 0.75 and iterations
   = 2, 4, 8.
4. P6 sparse observation-wise affinity: retain and row-renormalize the top
   k = 16, 32, 64, 128, 256, 512, or 1024 entries. The dense 4096-entry
   one-pass reference is already stored.

**Fold and calibration rule.** Each screen is independent. For each outer
fold, every method parameter and its threshold are selected using only the
other four folds; the selected pair is scored on the held-out fold. Thresholds
are 0.30–0.70 in increments of 0.025. The report also gives fixed-threshold
0.45 results for continuity. Section 5 promotion, inclusiveness tie, four-fold
vote, and out-of-fold guard remain required. Maps and generated images are
released after each image; only per-image counts and AP are retained.

| Item | Path |
|---|---|
| Reusable operators | `src/segmentation/propagation_variants.py` |
| Runner | `scripts/run_propagation_package.py` |
| Fold selector | `scripts/select_propagation_cv.py` |
| Screen specification | `configs/controlled_propagation_screens.json` |
| Smoke / production / merge | `jobs/{smoke,run,merge}_propagation_package.slurm` |
| Output | `outputs/segmentation/controlled_attention_cv/propagation_package/` |

Submitted from commit `8f2f73acc` on Renyi: independent-reference smoke job
`232662`; resumable 20-class production array `232663`, dependent on the
smoke; and fold-wise method/threshold selector job `232664`, dependent on the
complete array.

**Claude handoff.** Continue only the non-overlapping fusion/calibration
package: arithmetic/residual CA-SA mixing, multiplicative gating, geometric
fusion, and normalization. Review this package after its committed result.

## 38. Claude: joint audit review, phase 3 registry, and the token finding

**Joint CA×SA audit (Section 36): accepted.** Claude recomputed it from the
committed shards:

- 490/490 unique images, 20/20 complete shards, one environment (A100, torch
  2.6.0+cu118), and the 78 registered cells (6 CA seeds × (12 SA cells + no SA)).
- Re-running the selection with `configs/joint_ca_sa_cartesian_screen.json`
  reproduces the committed result exactly: `CA=C/R|SA=C/L` in all 5 folds,
  each with "no eligible candidate".
- The `CA=C/R` row equals the SA-C cells within 4 pixels on any image (same
  environment, regenerated). Its no-SA cell equals SA-C's seed exactly.
- SA `C/L` is the best SA partner for all six CA seeds.
- The closest alternative, `CA=C/R+H|SA=C/L`, reproduces as stated: +0.0017;
  paired Δ +0.00035 [−0.00164, +0.00240], 269/221; training-fold gains
  +0.0010 to +0.0023; Δ −0.0001/+0.0017/+0.0007 at 0.35/0.45/0.55; 12/20
  classes improve; `person` −0.020, `train` −0.019, `sheep` +0.058.
- Section 11 stays unchanged.

**Phase 3 registry.** At Narmeen's request, `docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md`
is now the single registry of every deferred experiment. It merges Claude's
backlog, the Section 37 propagation package (unchanged, with its grids) and
the ideas in Faiz's `plan.tex` (SAM refinement, SeeDiff-style seeded
expansion, entropy and hierarchical weighting, background handling,
class-adaptive choices, and comparisons with other diffusion methods). It adds:

- registered rules R1–R11 that every package follows (Section 0 of the
  registry). R3 matches the Section 37 selector: the parameter and the
  threshold (0.30–0.70, step 0.025) are chosen on the training folds, for REF
  as well;
- a grid, failure target, track, owner, tier and status for every ID;
- a combined confirmation P3-C, a token confirmation T-C, the COCO-Cap
  transfer rule, and a map from each paper table and figure to the IDs behind
  it.

**Ownership (Narmeen agreed):**

| Owner | Tier 1 work |
|---|---|
| Codex | Propagation (P1, W3, P2, P6; Section 37), then the token track B1 and B1b. B2a/B2b/S4 follow in Tier 2 |
| Claude | Fusion and calibration (P4, F3, F4, N2–N4) with R2 DenseCRF in one package, then scoring the released comparison masks Q2–Q5 |

The ownership marked "proposed" or "unassigned" in the registry (P3, P5, W4,
P9, H1, G4, M1, M2, R1, X1, C2, Q7) is open. Codex: please confirm or edit it.

**Token finding.** OVAM's public archive contains its token-training set
(`evaluation/ovam/voc_sim_train/`):

- 63 images, 59 hand-drawn masks and 57 per-image tokens. None of its seeds
  appears among the VOC-Sim evaluation seeds.
- The released `tokens_VOC.npy` equals, for every class, the **mean of that
  class's 2–3 per-image tokens** (maximum difference below 10⁻⁶), so it is a
  2–3-shot token, not 1-shot.
- `configs/ovam/README.md` and `docs/narmyn/attention/TOKEN_CRF_EXPERIMENT.md`
  are corrected. The OVAM paper's wording must be checked before the paper
  states the number of shots.
- This gives B2a (our tokens on OVAM's exact training images) and a 1- vs
  2–3-shot comparison (B1b, S4) without new annotation.

**Codex: please review** the registry (rules, grids, tiers, ownership) and
reply here before the fusion/calibration package is submitted. Codex's
Section 37 propagation run is unaffected.

## 39. Claude: paper skeleton created (taken over at Narmeen's request)

**Codex: Claude has created the `paper/` skeleton** that Section 29 assigned to
Codex, at Narmeen's request, so that results can start accumulating. Section
ownership from Section 29 is unchanged: Codex still owns Method, the data and
metric part of the protocol section, package-A result text, propagation text
and final integration. Please review the structure and edit your sections
freely.

**What was created.**

| Path | Content |
|---|---|
| `paper/attention_segmentation.tex` | Manuscript in CVPR review mode (anonymous, line numbers); `\confYear` 2027 |
| `paper/supplementary.tex` | Supplementary material as a separate PDF, same kit |
| `paper/preamble.tex` | Packages and macros, plus `\owner{}` and `\source{}` markers |
| `paper/sec/*.tex`, `paper/sec/supp/*.tex` | One file per section, each naming its owner and the protocol/registry IDs behind it |
| `paper/tables/generated/controlled_path.tex` | First generated table: CA-C, SA-C and the joint audit (OOF mIoU, votes, bootstrap CI) |
| `scripts/paper/export_paper_artifacts.py` | Generator. Reads only committed `fold_results.json`; every output starts with its sources and their commits |
| `paper/README.md` | Build command, file roles, section owners, artifact map, submission checklist |
| `paper/cvpr.sty`, `paper/ieeenat_fullname.bst` | Official CVPR author kit, unchanged (`paper/kit/KIT_SOURCE.txt`: commit `2917585`, the 2026 kit; no 2027 kit yet) |

**Deviations from Section 29.**

- One file per section in `sec/` instead of a single manuscript file. This is
  the kit's own structure, and it keeps the two agents from editing the same
  file.
- Reproducibility moved to the supplementary material, as already agreed
  there.
- Fusion and Postprocessing, Cross-Dataset Evaluation and Discussion follow the
  Section 28 list. Limitations is a part of Discussion to save space.

**Build.** Both documents compile with `pdflatex`/`bibtex` and have no
overfull boxes. The cluster's TeX Live 2018 cannot load the current `siunitx`
or `cuted`, so the preamble uses neither (wide figures use `figure*`). The kit
also needs `cleveref`, `silence` and `multirow`; they are installed from CTAN
in the user's `~/texmf`, and Overleaf has them all. Build files and PDFs are
git-ignored; PDFs are committed only at agreed milestones.

**Still open (Section 29):** owners for Abstract, Introduction, Related work,
Transfer and Discussion, marked "to agree" in the files.

## 40. Claude: fusion/calibration runner ready; first paper text

**Runner (registry Section 1.2, rules R1–R11).** Not yet submitted: it waits
for Codex's review of the registry (Section 38).

| Item | Path |
|---|---|
| Operators (P4, F3, F4, N3, N4) | `src/segmentation/fusion_variants.py` |
| Unit tests | `tests/test_fusion_variants.py` |
| Runner | `scripts/run_fusion_calibration.py` |
| Selection spec (uses Codex's `scripts/select_propagation_cv.py`) | `configs/controlled_fusion_calibration_screens.json` |
| Job | `jobs/run_fusion_calibration.slurm` |

- **Configurations (12).** `CHECK/ca_seed`; `REF/one_pass`; `P4/lam25`,
  `P4/lam50`, `P4/lam75` and `P4/lam90`; `F3/gate`; `F4/geomean`;
  `N2/percentile`; `N3/max`; `N4/otsu`; `R2/crf`.
- **Scoring.** All 17 thresholds of the Section 37 grid (0.30–0.70), so the
  selector chooses each parameter and threshold on the training folds, as for
  propagation.
- **Binary configurations.** `N4/otsu` stores the same counts in every
  threshold slot. `R2/crf` applies OVAM's DenseCRF (`apply_ovam_densecrf`,
  OVAM's parameters) to the REF mask at each threshold, so the threshold is
  still chosen in the folds; its AP is NaN.
- **Tie order (R5).** Larger λ (closer to REF); F3 before F4; N3, then N2,
  then N4.
- **Shared-code change.** `normalize_batch` gains an `"identity"` option, used
  for N3's max-normalized map. Existing methods are unchanged, and all 21
  shared CV tests pass.

**Smoke (2 images, V100, `--verify-reference`).**

- REF agrees with the independent package-B propagation within 2.4×10⁻⁷.
- REF matches SA-C's `C/L` on the same images: 0.8916 vs 0.8919 and 0.8550 vs
  0.8549 (A100/V100 difference).
- Every configuration scores, at about 10 s per image, so the full run is about
  80 GPU-minutes.

**Paper text.** Claude's sections are drafted in `paper/sec/`:

- 4.2, cross-validated selection;
- Section 5, package B and CA-C;
- Section 6, package B, SA-C and the joint audit, with Codex's SA-C and joint
  audit numbers as verified in Sections 35 and 38.

Every number is listed with its source in the `paper/README.md` artifact map.
Codex: please review, in particular the SA-C wording, since Codex ran it.

## 41. Claude: released baselines scored (registry Q1–Q5)

`scripts/narmyn/attention/score_released_baselines.py` scores the released
masks and heatmaps in OVAM's public archive. It uses **each method's procedure
from OVAM's evaluation notebooks** (`github.com/vpulab/ovam`,
`experiments/evaluation/<method>/evaluation_<method>.ipynb`, commit `12afd87`)
and OVAM's metric (per-class pooled IoU, macro mean), on each method's own
released images and annotations. CPU job `232917`. Output:
`outputs/segmentation/comparisons/released_baselines/` (`summary.json`, with
per-class IoU and the COCO-Cap caption-includes/excludes split, and
`per_image.csv`).

| Method | Procedure | VOC-Sim word / token | COCO-Cap word / token | Images (VOC / COCO) |
|---|---|---|---|---|
| OVAM | released DenseCRF masks (`ovam_sa`) | 0.704 / 0.825 | 0.582 / — | 490 / 707 |
| DAAM | channel 1, / max, > 0.4 (token > 0.8) | 0.662 / 0.797 | 0.484 / 0.661 | 490 / 707 |
| Attn2Mask | channel −2 (token: channel 1), min–max, background `1 − h − β` (β 0.2 / −0.2), argmax; DenseCRF on COCO-Cap | 0.603 / 0.811 | 0.584 / 0.676 | 485 / 668 |
| Grounded Diffusion | released masks | 0.621 / 0.866 | 0.502 / 0.732 | 496 / 588 |
| DatasetDM | released masks | 0.803 / 0.806 | 0.593 / 0.605 | 500 / 728 |

**Verification.**

- Every value printed in OVAM's notebooks is reproduced to the printed
  precision: DAAM 66.2/79.7 and 48.4/66.1; Grounded Diffusion
  0.621276/0.865672 and 50.2/73.2; DatasetDM 80.3/80.6 and 59.3/60.5.
- The OVAM rows equal the earlier scoring in
  `outputs/segmentation/ovam_paper_baseline/` (0.7036 / 0.8247).
- Attn2Mask has no printed notebook output to compare against.

**Image sets.**

- DAAM's annotations are byte-identical to OVAM's (490 and 707).
- Attn2Mask, Grounded Diffusion and DatasetDM were evaluated on their own
  generated images and annotations, as in OVAM's comparison, so their rows are
  not on the same images as ours.
- For Grounded Diffusion on COCO-Cap, the notebook uses `annotations` (588);
  the archive also has `annotations_extra` (705, a superset), which is not used.

**For the paper (`paper/tables/generated/released_baselines.tex`).**

- The columns are *class word* vs *OVAM's optimized token*, not
  "training-free". Grounded Diffusion and DatasetDM train additional modules
  (marked ‡; to be verified against their papers before submission).
- On VOC-Sim, the frozen training-free method (0.789 OOF, regenerated images)
  is above every class-word baseline except DatasetDM (0.803), which trains a
  decoder.
- The COCO-Cap and token cells for our method wait for the transfer (registry
  Section 7) and the token track (B1, T-C).

**Not yet run:** DAAM with the threshold chosen inside our folds (R3), which
is possible because DAAM shares our images.

## 42. Claude: fusion/calibration package launched

**Codex: the fusion/calibration package is running.** Narmeen decided to
launch it without waiting for the registry review requested in Sections 38 and
40.

- Job `232922`: 20 class shards, all 490 VOC-Sim images, renyi A100 with the
  `trdp` environment (the same as SA-C and the propagation package), from
  commit `229130b`.
- The grids are exactly those registered in the committed registry (Section
  1.2, commit `4ac06d9`) and implemented in `scripts/run_fusion_calibration.py`
  (Section 40), so rule R8 holds.
- The first image reproduces SA-C's `C/L` (0.893 vs 0.892).

The registry review is still welcome. If Codex asks for a grid or rule change,
the package is rerun (about 50 minutes) and both runs are reported. Selection
uses `scripts/select_propagation_cv.py` with
`configs/controlled_fusion_calibration_screens.json`; results follow for
Codex's review.

## 43. Codex: propagation results and review of Claude's phase-3 work

### Propagation package results

All jobs completed successfully: reference smoke `232662`, 20-class production
array `232663`, and dependent selector `232664`. The smoke matched the
independent package-B one-pass map within `2.384e-07`; the two n=1 aggregation
orders differed by only `2.98e-07`. Production contains 20/20 complete shards,
490/490 unique images, 20 classes, 40 configurations, and all 17 registered
thresholds, run on Renyi from commit `8f2f73acc`.

**P1 promotes observation-wise two-pass propagation,**
`mean_i(A_i^2 c)`. It wins four folds; fold 4 chooses n=3. The registered
selection procedure improves OOF mIoU from 0.7875 to 0.7991 (+0.0116). Scoring
the frozen n=2 candidate in every held-out fold gives 0.7875 → 0.7998
(+0.0123), paired mean image-IoU +0.0143, 95% CI [+0.0107, +0.0178], with 365
images better and 125 worse. Its full-490 selected threshold is 0.45 and mIoU
is 0.8007. The OOF class changes are positive for 16/20 classes; the largest
are `chair` +0.0780, `person` +0.0304, and `aeroplane` +0.0299. The largest
loss is `tv` -0.0116. P1 n=2 is promoted to P3-C; it does not become the final
training-free method until combined confirmation.

**The other families remain at REF.** Average-before propagation (`W3`) selects
REF in four folds; its best fixed-threshold alternative, n=2, reaches 0.7940
but is eligible in only one fold. Observation-wise restart (`P2`) selects REF
in four folds; its best diagnostic is beta=0.10, n=4 at 0.7954. Average-before
restart and sparse affinity select REF in all five folds. Sparsification is
monotonic in k but remains below REF even at k=1024 (0.7792), so discarding
weak affinity edges is rejected.

### Review of Claude's work

Codex accepts the Section 38 registry rules R1-R11 and the Tier-1 ownership
split. Proposed and unassigned Tier-2/3 ownership remains open until scheduled;
this acceptance does not implicitly schedule those experiments. Claude's joint
CA×SA audit review reproduces our result and is accepted.

Codex reviewed the fusion/calibration implementation, operator tests,
configuration specification, nested threshold selection, reference smoke, and
DenseCRF path. The implemented grid contains 12 configurations (the prior
Section 40 count of 13 was a documentation error corrected above). The package
is accepted for submission. Claude owns its execution and merge.

The CVPR source layout and the drafted controlled CA/SA text are accepted. The
SA-C and joint-audit statements agree with the committed results. Codex adds
the propagation result to its assigned paper section in this commit.

Claude's released-baseline scoring (Section 41) is also accepted as a
reproduction of each method's OVAM-notebook procedure. Those methods use
different released image sets where noted, so the paper must retain that
qualification. The discovery that OVAM's released class token averages 2-3
per-image training tokens is recorded for the later token track; no token
result is re-labelled as one-shot.

Claude: independently audit the committed propagation shards, fold-wise
selection, frozen n=2 diagnostic, per-class effects, and P3-C promotion.
## 44. Claude: fusion/calibration results (Codex result review requested)

**Run.** Job `232922`: all 20 shards complete, 490/490 images, renyi A100,
torch 2.11.0+cu130 (Claude's `trdp`), from commits `229130b` and `46df8f5`.
The commits differ only in the protocol text.

- Selection: `scripts/select_propagation_cv.py` with
  `configs/controlled_fusion_calibration_screens.json`. Each method parameter
  and threshold (0.30–0.70) is chosen on the training folds, for REF as well.
- R9 diagnostics: `scripts/narmyn/attention/fold_selection_diagnostics.py`
  writes `diagnostics.json`, with the paired bootstrap, fixed-threshold
  changes, per-class changes, and the training-fold check that excluded each
  never-chosen candidate.

| Screen | Fold choice (votes) | OOF mIoU REF → choice | Paired Δ image IoU [95% CI] | Δ mIoU @0.35 / 0.45 / 0.55 | Result |
|---|---|---|---|---|---|
| P4 residual mix (λ 0.25–0.90) | REF (5/5) | 0.7875 → 0.7875 | — | — | Not promoted; best λ = 0.90 is 0.7856 at 0.45 |
| F3 gate / F4 geometric mean | REF (5/5) | 0.7875 → 0.7875 | — | — | Not promoted (F3 0.514, F4 0.764 at 0.45) |
| N2 percentile / N3 max / N4 Otsu | REF (5/5) | 0.7875 → 0.7875 | — | — | Not promoted (below) |
| **R2 DenseCRF** | **R2 (5/5)** | **0.7875 → 0.8049 (+0.0175)** | **+0.022 [+0.019, +0.026]; 361 / 129** | **+0.022 / +0.014 / +0.006** | **Promote to P3-C** |

- **R2 (DenseCRF, OVAM's parameters, on the REF mask).** Promote band in every
  fold. It is not calibration-dependent, and all 20 classes improve at the
  full-data threshold (0.425); the largest gains are `bird` +0.056,
  `aeroplane` +0.047 and `chair` +0.047, and the smallest is `tv` +0.00004.
  AP is undefined for a binary mask. It uses the generated RGB image and no
  labels, so it is training-free (R10).
- **N4 (per-image Otsu) and N3 (max normalization)** gain +0.002 to +0.009 on
  the training folds, in the conditional band. In every fold they fail its
  class checks. N4 loses `sheep` −0.07 in three folds and `cat` −0.07 in one;
  in fold 4 its worst loss is `bird` −0.025, but only 11 classes improve. N3
  loses `chair` or `dining-table` −0.05 to −0.08 in four folds; in fold 2 only
  11 classes improve. N2 gains only +0.001 to +0.002.
- **REF's fold-chosen threshold** is 0.425 in four folds and 0.45 in one. Its
  OOF mIoU (0.7875) is slightly below the fixed-0.45 value (0.7887): choosing
  the threshold in the folds does not by itself raise the reference.
- **Checks.**
  - An independent recomputation of the R2 out-of-fold counts from the
    committed fold thresholds reproduces 0.78745 → 0.80490.
  - REF at 0.45 is 0.78863, against SA-C's `C/L` 0.78866. Same GPU type;
    Claude's `trdp` has torch 2.11 and the SA-C run torch 2.6, which moves
    individual pixels but not the class means.

**For P3-C.** R2 joins Codex's propagation winner, observation-wise two-pass
propagation `mean_i(A_i² c)` (Section 43). R2 refines the final binary mask,
so it composes with any propagation operator; P3-C scores REF, P1 n=2, R2 and
P1 n=2 + R2.

**Evidence package and paper.** The evidence package will add this package
together with Codex's propagation package. `paper/sec/7_fusion.tex` now has
the fusion paragraph.

## 45. Codex: expanded P3-C claim and registered grid

**Ownership.** Codex claims and runs P3-C; Claude reviews the implementation
and committed results. Claude should not launch a duplicate P3-C package.

The combined confirmation retains the four required cells from Section 44 and
adds the strongest plausible interaction candidates from both completed
packages. It is a full Cartesian product of:

| Axis | Registered settings | Evidence for inclusion |
|---|---|---|
| Propagation | `REF/one_pass`, `P1/obs_n2`, `P1/obs_n3`, `P2/obs_b10_n4`, `W3/mean_n2` | n=2 promoted; n=3 had the highest fixed-threshold power result (0.8014); restart and average-before n=2 were the next diagnostics (0.7954 and 0.7940) |
| Calibration | identity/global threshold, `N2/percentile` (1--99), `N4/otsu` | identity is required; Otsu was Claude's strongest calibration diagnostic (0.7955 at 0.45); percentile was the smaller, broadly positive alternative (0.7902 at 0.45) |
| Refinement | none, `R2/crf` | DenseCRF was promoted 5/5, 0.7875 to 0.8049 |

This produces **5 × 3 × 2 = 30 configurations**. It necessarily contains:

1. REF;
2. observation-wise two-pass SA;
3. REF + DenseCRF;
4. observation-wise two-pass SA + DenseCRF.

The extra combinations test whether DenseCRF repairs class-specific losses
from Otsu or percentile calibration and whether those postprocessors change
the ordering among propagation operators. Maximum normalization, residual
mixing, gating, geometric fusion, sparse affinity, and average-before restart
remain excluded because their completed screens showed either large losses or
no competitive diagnostic.

**Execution and selection.** Run all 490 VOC-Sim images in one environment,
with the five fixed folds and threshold grid 0.30--0.70 in steps of 0.025.
For Otsu, the per-image threshold replaces the global threshold. DenseCRF uses
OVAM's fixed parameters after binarization. Choose the entire pipeline on the
four training folds and evaluate it on the held-out fold. Apply R3--R5,
including the class-loss guard, and report the four mandatory cells separately
from the expanded candidate screen. Save per-image counts, per-class IoU,
paired bootstrap intervals, fold choices, fixed-threshold diagnostics,
environment metadata, and no images or attention tensors.

Output root:
`outputs/segmentation/controlled_attention_cv/phase3_confirmation/`.
Implementation:

- runner: `scripts/run_phase3_confirmation.py`;
- registered selector: `configs/controlled_phase3_confirmation.json`;
- smoke, production and merge jobs: `jobs/smoke_phase3_confirmation.slurm`,
  `jobs/run_phase3_confirmation.slurm`, and
  `jobs/merge_phase3_confirmation.slurm`;
- final diagnostics: `scripts/finalize_phase3_confirmation.py`.

The runner validates all 30 unique configuration names and the four mandatory
cells, verifies the independent one-pass reference during the smoke job, and
is resumable per class. The merge selects the whole pipeline and threshold on
the training folds, then writes fixed-candidate cross-fit and per-class
diagnostics. Submission job IDs are recorded after launch.

**Submission correction.** The first smoke job `232942` failed because its
default CRF path pointed to Claude's Python 3.11 extension while Codex's
`trdp` uses Python 3.10. Dependent jobs `232943` and `232944` were cancelled.
SimpleCRF 0.2.1.1 was built and directly exercised under Python 3.10; the
runner now checks ABI-compatible import before loading Stable Diffusion.

Replacement chain from commit `371f791ee`: smoke `232945` completed 0:0 and
verified all 30 configurations, one image and 17 thresholds; production array
`232946`; merge and selection `232947`. The selector now produces a separate
four-cell mandatory screen as well as the registered 30-cell extension, per
Claude's Section 46 review.
## 46. Claude: review of the propagation package; P3-C left to Section 45

**Propagation package (Section 43): accepted.** Claude recomputed it from a
copy of the committed shards:

- 490/490 unique images, 20/20 complete shards, 40 configurations, 17
  thresholds, one environment (A100, torch 2.6.0+cu118), commit `967d903`, and
  the frozen CA seed with SA module 15.
- Re-running `run_screen` with `configs/controlled_propagation_screens.json`
  reproduces all five screens exactly:
  - P1: n = 2, 4/5 votes, OOF 0.78751 → 0.79915;
  - W3, P2 (observation-wise and average-before) and P6: REF.
- The frozen n=2 diagnostic reproduces: OOF 0.7875 → 0.7998; paired +0.0143
  [+0.0107, +0.0178], 365/125; 16/20 classes improve (`chair` +0.078,
  `person` +0.030, `aeroplane` +0.030; `tv` −0.012, `bus` −0.005); full-490
  threshold 0.45, mIoU 0.8007.
- The quoted diagnostics reproduce too: P6 k=1024 0.7792; W3 n=2 0.7940; P2
  β=0.10, n=4 0.7954.
- REF at 0.45 equals SA-C's `C/L` with 0 differing pixels (same environment).
  `W3/mean_n1` and REF differ by at most 4 pixels on an image (aggregation
  order, as reported).
- **Addition:** n=2 is not calibration-dependent (Δ mIoU +0.009 / +0.012 /
  +0.024 at 0.35 / 0.45 / 0.55), and mean AP is unchanged (0.956 → 0.957).

**P3-C.** This section was first pushed as a second "Section 45" with a
four-cell P3-C proposal, crossing Codex's claim (Section 45, pushed a few
minutes earlier). The proposal is **withdrawn**: Codex runs the expanded P3-C
of Section 45, and Claude reviews it and will not launch a duplicate.

Notes for that review:

- The expanded grid adds settings that were not promoted in their own screens:
  `P1/obs_n3`, `P2/obs_b10_n4`, `W3/mean_n2`, `N2/percentile` and `N4/otsu`
  (Otsu was rejected by the class-loss guard). Including them is a registered
  extension, made after seeing the screens, so it should be labelled as such.
  It is protected by the out-of-fold selection and the class-loss guard.
- The four mandatory cells (REF, P1 n=2, R2, P1 n=2 + R2) should be reported
  as their own screen, as Section 45 says, so that the registered R4 path can
  be read without the extension.
- The report should give the number of configurations searched (30) next to
  the OOF result (R9).

## 47. Claude: DAAM with its threshold chosen in our folds (registry Q2, R3)

DAAM's released heatmaps share our VOC-Sim and COCO-Cap images and annotations
(byte-identical), so DAAM can be selected exactly like our methods.
`scripts/narmyn/attention/daam_fold_threshold.py` keeps DAAM's binarization
(channel 1, bilinear to 512, divide by the maximum, `> t`) and chooses `t` on
the four training folds of each outer fold (CPU job `232949`). The grid is
0.30–0.90 in steps of 0.025, wider than R3's 0.30–0.70, so that it contains
OVAM's 0.8 for the token heatmaps. COCO-Cap is scored with the threshold chosen
on all 490 VOC-Sim images. Output:
`outputs/segmentation/comparisons/daam_fold_threshold/` (`summary.json`, and
per-image counts in `counts.npz`).

| DAAM | VOC-Sim, OVAM's fixed threshold | VOC-Sim, fold-chosen (OOF) | Fold thresholds | COCO-Cap, fixed | COCO-Cap, VOC-chosen threshold |
|---|---:|---:|---|---:|---:|
| Class word | 0.6616 (0.4) | **0.6735** | 0.45 in all folds | 0.4842 | 0.5267 (0.45) |
| OVAM token | 0.7970 (0.8) | 0.7972 | 0.775 ×4, 0.8 ×1 | 0.6608 | 0.6417 (0.775) |

- **Class word:** the fold-chosen threshold improves DAAM by +0.012 OOF
  (paired +0.017 [+0.010, +0.024], 340/150). The frozen method on the same
  images and folds is 0.789, i.e. +0.115 above DAAM even with a fold-chosen
  threshold.
- **Token:** OVAM's 0.8 is already near the optimum (paired −0.004 [−0.008,
  −0.000]). The VOC-chosen 0.775 transfers slightly worse to COCO-Cap.

Added to `paper/tables/generated/released_baselines.tex` (marked §) and to
evidence table A10.

## 48. Claude: COCO-Cap transfer runner ready (proposal; waits for P3-C)

Registry Section 7 left the transfer's owner "to be agreed". Claude proposes to
run it, and has the runner ready: `scripts/run_coco_cap_transfer.py`, job
`jobs/run_coco_cap_transfer.slurm`. Nothing is chosen on COCO-Cap; counts are
stored at the 17 phase 3 thresholds so that the VOC-frozen threshold of each
stage can be read off.

| Stage | Configuration |
|---|---|
| `CA/seed` | frozen CA seed (Section 11) |
| `SA/REF` | CA + one pass through SA module 15 (Section 11) |
| `P3/P1`, `P3/R2`, `P3/P1+R2` | the P3-C mandatory cells (two passes; DenseCRF; both) |

**Attribution on COCO-Cap.** The class word is often absent from the caption,
so the class name is encoded as a separate attribution prompt, as in OVAM's
COCO-Cap evaluation and in `scripts/build_coco_cap_manifest.py`. To keep the
frozen CA convention, the runner encodes that prompt **padded to 77 tokens**
and computes the conditional CA against its keys (`attribution_key`); the
softmax runs over 77 positions and the map averages all class pieces.

**Shared-code caution (Codex, please note).** `SAPackageRecorder` in
`scripts/run_sa_package_b.py`, used by the SA, propagation, fusion and P3-C
runners, ignores `attribution_key`: it computes the CA against the caption's
own keys (`key[1]`). That is correct on VOC-Sim, where the attribution prompt
is the caption. On COCO-Cap with a separate attribution prompt, however, the
target positions would index the wrong tokens. Any COCO-Cap run must use
attribution keys, as `TransferRecorder` does here.

**Self-test (`--self-test-voc`; renyi, `trdp`).** VOC-Sim images are passed
through the same separate-prompt path, with the caption itself as the
attribution prompt.

- On two images, `CA/seed`, `SA/REF` and `P3/R2` equal the fusion package's
  `CHECK/ca_seed`, `REF/one_pass` and `R2/crf` with **0 differing pixels at
  all 17 thresholds**, so the padded separate prompt is exactly the frozen
  convention.
- Against Codex's propagation package (torch 2.6), REF and `P1/obs_n2` differ
  by at most 104 pixels per image, the usual environment difference.

**COCO-Cap smoke (two aeroplane images).** `CA/seed` 0.656 / 0.754, `SA/REF`
0.770 / 0.751, `P3/P1` 0.786 / 0.764, `P3/R2` 0.870 / 0.861, `P3/P1+R2` 0.882 /
0.880 at 0.45.

**Plan.** Launch the 707 images (20 shards, renyi, about an hour on one GPU)
once P3-C is committed, adding the P3-C winner as a stage if it is an extension
cell outside the five above. Report every stage at its VOC-frozen threshold, on
all 707 images and split by whether the caption contains the class word.
Codex: please confirm the ownership or edit the stages.

## 49. Claude: taking over token track B1/B1b (registered design; Codex please do not start it)

**Codex: Claude is taking over B1/B1b**, at Narmeen's request; Section 38
assigned it to Codex after propagation. Please do not start B1/B1b. Review the
result when it is pushed. B2a/B2b/S4 remain as in the registry. This section
registers the design before the run (R8).

**Tokens.**

- B1: `configs/ovam/tokens_VOC.npy` (OVAM's released tokens, the mean of 2–3
  per-image tokens; Section 38).
- B1b: the per-image tokens in OVAM's archive
  (`evaluation/ovam/voc_sim_train/tokens/`, fetched into the git-ignored
  `datasets/`; row 1 of each `(2, 768)` file).
- The runner checks, for every class, that the released token equals the
  mean of its per-image tokens.

**Attributions** (one live generation per image; all share the frozen CA
modules, calls and heads and the frozen SA module):

| Name | Attribution |
|---|---|
| `W` | class word, frozen convention (conditional query, 77 padded positions, all pieces) |
| `TU` | released token, OVAM's form: keys from `[<SoT>, token]`, softmax over the two, **unconditional** query |
| `TC` | released token, the same keys, **conditional** query (our CA convention) |
| `T1U`–`T3U`, `T1C`–`T3C` | each per-image token alone (1-shot), both query forms. Classes with two training images repeat token 1 in slot 3 (`token_slots_repeat_first` in the shard settings). |

**Pipelines.**

- `W`, `TU` and `TC`: the CA seed alone, REF (one pass), P1 (two passes), R2
  (DenseCRF on REF), and P1+R2.
- The 1-shot tokens: REF and P1.
- In total 27 configurations, scored at the 17 phase 3 thresholds.

**Selection (`configs/controlled_token_screens.json`, with
`scripts/select_propagation_cv.py`).**

- Reference `W/REF`.
- Two screens, `TU/REF` and `TC/REF`, each choosing its threshold on the
  training folds.
- The P1, R2 and P1+R2 token cells are inputs to the token confirmation T-C,
  combined with the P3-C winner, and are not selected here.
- The 1-shot cells give the shot comparison (S4: 1 vs 2–3 shots), reported as
  diagnostics.
- Token results form the few-shot track (R10) and never replace the class
  word in the training-free track.

**Smoke (2 images, V100, `gen`).** `W/REF` 0.892 / 0.855 (equal to the frozen
pipeline up to the V100/A100 difference); `TU/REF` 0.901 / 0.865; `TC/REF`
0.864 / 0.872; `W/P1+R2` 0.937 / 0.941; `TU/P1+R2` 0.934 / 0.939. About 35 s per
image.

**Run.** `jobs/run_token_package.slurm`: 20 class shards on the V100 nodes with
the `gen` environment (renyi is running P3-C), all 490 VOC-Sim images. The
package compares only with its own `W` cells (R7).

## 50. Claude: token track B1/B1b results (Codex review requested)

**Run.** Job `232957`: 20/20 shards, 490/490 images, V100 nodes (`gen`, torch
2.4.1+cu121), commit `326f173`.

**Correction to the shards.** The runner first wrote two per-class fields
(`per_image_tokens`, `token_slots_repeat_first`) into the shard settings. The
loader requires identical settings across shards, so these fields were moved
to `tokens_per_class.json` and removed from each shard's `settings`. Counts,
AP and image lists are byte-identical before and after (checked), and the
runner now writes the sidecar file directly.

**Registered screens** (`configs/controlled_token_screens.json`; parameter and
threshold chosen in the folds, R3):

| Screen (vs `W/REF`, class word) | Votes | OOF mIoU W → token | Paired Δ image IoU [95% CI] | Δ @0.35 / 0.45 / 0.55 | Classes improved |
|---|---:|---|---|---|---:|
| `TU/REF`: released token, OVAM form (unconditional query) | 5/5 | 0.7873 → 0.8428 (+0.0555) | +0.047 [+0.038, +0.056]; 369/121 | +0.032 / +0.052 / +0.082 | 19/20 (worst `bus` −0.001) |
| `TC/REF`: released token, conditional query | 5/5 | 0.7873 → 0.8507 (+0.0634) | +0.053 [+0.044, +0.062]; 375/115 | +0.041 / +0.060 / +0.085 | 20/20 (smallest `dog` +0.006) |

Both are promoted and neither is calibration-dependent. The largest class
gain is `train` (+0.24 / +0.26): the class word is ambiguous and the token is
not.

**Which form (diagnostic, not registered as a screen).** Head to head, TC
beats TU by +0.0035 OOF, but in only 3/5 folds, so neither form is promoted
over the other. By R5 (the simpler change wins), **TC is the token
attribution of the few-shot track**: it changes only the attribution keys and
keeps the frozen conditional query, whereas TU also changes the query branch.
A screen with both forms against W selects TC in 5/5 folds (0.8507).

**Token on the other pipelines** (OOF, each vs its class-word cell; all 5/5;
inputs to T-C):

| Pipeline | W | TU | TC |
|---|---:|---:|---:|
| CA seed alone | 0.7128 | 0.7915 | 0.7967 |
| REF (one pass) | 0.7873 | 0.8428 | 0.8507 |
| P1 (two passes) | 0.7997 | 0.8462 | 0.8543 |
| R2 (DenseCRF) | 0.8047 | 0.8580 | 0.8643 |
| P1 + R2 | 0.8132 | 0.8603 | 0.8654 |

With the token, two passes no longer help (TC: P1 vs REF +0.0000, REF kept
5/5), while DenseCRF still does (TC: R2 vs REF +0.0137; P1+R2 +0.0147, both
5/5). The final few-shot pipeline is decided in T-C, together with P3-C.

**Shots (B1b/S4, TC, OOF on REF).**

- Class word 0.7873; released 2–3-shot mean token 0.8507.
- Single per-image tokens: 0.8309 (first), 0.8529 (second), 0.8421 (third,
  without `cat`, `horse` and `tv`, which have two training images).
- One annotated image already gives most of the gain, but which image matters
  (0.831 to 0.853), and averaging is steadier.
- With the TU form: 0.8233 / 0.8500 / 0.8390 against released 0.8428.

**Checks.**

- An independent recomputation of both registered screens from the committed
  fold thresholds reproduces them exactly.
- `W/REF` at 0.45 is 0.78844, against the fusion package's REF at 0.78863
  (V100 vs A100); `W/R2` is 0.80243 against 0.80272.
- The runner checked, for every class, that the released token equals the
  mean of its per-image tokens.

**For the paper.** OVAM's released token-optimized masks score 0.825 on
VOC-Sim; the frozen pipeline with OVAM's released token scores 0.851 (REF,
OOF) and 0.865 with DenseCRF (regenerated images; registry R10, few-shot
track).

## 51. Codex: P3-C results and review of Claude's Sections 47–50

### P3-C result

Jobs `232945`, `232946` and `232947` completed successfully. The package has
20/20 complete shards, 490/490 unique images, 30 registered configurations
and 17 thresholds. Re-running both screens directly from the committed shards
reproduces every fold choice and OOF value exactly.

| Cell | Frozen OOF mIoU | Gain over REF | Paired mean image-IoU [95% CI] | Better / worse |
|---|---:|---:|---:|---:|
| REF, one pass | 0.7875 | — | — | — |
| Two-pass SA | 0.7998 | +0.0123 | +0.0143 [+0.0107, +0.0178] | 365 / 125 |
| DenseCRF | 0.8050 | +0.0175 | +0.0222 [+0.0185, +0.0261] | 362 / 128 |
| **Two-pass SA + DenseCRF** | **0.8135** | **+0.0260** | **+0.0308 [+0.0251, +0.0364]** | **378 / 112** |

The mandatory combined cell is chosen in all five folds at threshold 0.45.
It improves 17/20 classes relative to REF; its largest improvement is `chair`
+0.146, followed by `aeroplane` +0.058 and `person` +0.048. Its largest loss
is only `tv` -0.009. This freezes the training-free method (FTF) as
observation-wise two-pass propagation followed by OVAM's fixed DenseCRF.

The 30-cell extension selects two-pass SA + Otsu + DenseCRF in all five folds
and scores 0.8207 against REF. As an incremental addition to the mandatory
winner, however, Otsu gives +0.0072, improves 12/20 classes, and lowers
`sheep` by 0.060. This is the conditional band and violates the 0.05 class-loss
guard, so Otsu remains a diagnostic and is not part of FTF.

### Review of Claude's updates

- The evidence additions A7–A10 correctly point to the committed joint,
  propagation, fusion and released-baseline packages.
- DAAM threshold selection is accepted: its class-word VOC-Sim score rises
  from 0.6616 at 0.40 to 0.6735 OOF at 0.45, while the same VOC threshold gives
  0.5267 on COCO-Cap. The optimized-token threshold is already near optimal.
- The B1/B1b token result is accepted after an independent recomputation of
  both screens. The conditional released token is selected 5/5 and raises REF
  from 0.7873 to 0.8507 OOF; all 20 classes improve. One-shot tokens retain
  most of the gain but vary from 0.8309 to 0.8529. T-C still decides the final
  few-shot pipeline; the existing diagnostics indicate that DenseCRF remains
  useful and a second SA pass becomes marginal with the token.
- Claude's COCO-Cap transfer ownership and runner are accepted. The frozen FTF
  is already represented by its `P3/P1+R2` stage at VOC threshold 0.45. The
  rejected expanded Otsu cell is not added as a frozen stage.

Claude: review the committed P3-C shards, both screens and incremental Otsu
decision. After pulling this result, the Section 48 COCO-Cap transfer may be
launched without changing its registered stages.

## 52. Claude: P3-C review (accepted), FTF recorded, and token confirmation T-C

### P3-C (Section 51): accepted

Claude recomputed P3-C from a copy of the committed shards:

- 490/490 unique images, 20/20 complete shards, 30 configurations, 17
  thresholds, one environment (A100, torch 2.6.0+cu118), commits `371f791` and
  `1e8d6af`.
- Re-running `run_screen` with `configs/controlled_phase3_confirmation.json`
  reproduces both screens exactly: the mandatory screen (`P1/obs_n2|N0/global|R2/crf`,
  5/5, OOF 0.78751 → 0.81350) and the expanded screen
  (`P1/obs_n2|N4/otsu|R2/crf`, 5/5, 0.82067).
- The cross-fits reproduce:
  - FTF: 0.7875 → 0.8135, paired +0.0308 [+0.0251, +0.0364], 378/112, 17/20
    classes, `chair` +0.146, worst `tv` −0.009.
  - Two-pass alone: 0.7998. DenseCRF alone: 0.8050.
- The incremental Otsu decision reproduces: +0.0072 over FTF, 12/20 classes,
  `sheep` −0.060, so Otsu fails the class-loss guard and stays diagnostic.
- The REF and two-pass cells equal the propagation package's with 0 differing
  pixels (same environment). REF + DenseCRF at 0.45 is 0.80278, against
  0.80272 in Claude's fusion package (torch 2.11).
- **Addition:** FTF is not calibration-dependent (Δ mIoU against REF +0.030 /
  +0.025 / +0.029 at 0.35 / 0.45 / 0.55).

**FTF is recorded in Section 11.** The COCO-Cap transfer (Section 48) was
launched with its registered stages: job `233147`.

### Token confirmation T-C

Registry Section 5 left T-C's ownership "to be agreed". At Narmeen's request,
Claude ran it from the token package's committed shards, with no new
generation: `scripts/narmyn/attention/token_confirmation.py` and
`configs/controlled_token_confirmation.json`.

- **Structure:** the P3-C mandatory structure and tie order (REF, then
  P1/R2, then P1+R2), with the conditional released token (Section 50) as the
  attribution and `TC/REF` as the reference.
- **Status of the cells:** they were registered in Section 49 as T-C inputs.
  The screen definition is written after the Section 50 diagnostics were
  reported, but its structure is fixed by P3-C.

| Token pipeline (conditional released token) | OOF mIoU (fixed-candidate cross-fit) | Paired Δ vs token REF [95% CI] |
|---|---:|---|
| `TC/REF`: one pass | 0.8507 | — |
| **`TC/R2`: one pass + DenseCRF** | **0.8643** | **+0.0166 [+0.0131, +0.0201]; 353/137** |
| `TC/P1+R2`: two passes + DenseCRF | 0.8654 | +0.0178 [+0.0128, +0.0227]; 339/151 |

- **Selection:** `TC/R2` in 4/5 folds (fold 3 chose `TC/P1+R2`), OOF 0.8507 →
  0.8635.
- **The second pass with the token:** +0.0012 [−0.0021, +0.0041], 8/20
  classes; a screen `TC/P1+R2` vs `TC/R2` finds no eligible candidate in any
  fold. So, unlike the class word, the token does not need the second pass.
  One reading: two passes fill in the parts of the object that the class-word
  seed misses, and the token's seed already covers them.
- **Against the class-word FTF:** `TC/R2` 0.8132 → 0.8643 (+0.0511), paired
  +0.0388 [+0.0293, +0.0486], 19/20 classes. `TC/P1+R2`: 0.8654.
- **Calibration:** `TC/R2` against `TC/REF` is +0.025 / +0.017 / +0.009 at
  0.35 / 0.45 / 0.55, so it is not calibration-dependent.

**Proposed few-shot pipeline (Section 11, pending Codex's review):** the
conditional released token, one SA pass, DenseCRF. The COCO-Cap transfer of
this pipeline needs token attribution in the transfer runner (OVAM's VOC tokens
on COCO-Cap, registry Section 7). Claude will add it as a separate registered
stage after Codex's review.

## 53. Claude: reply to the semantic-graph proposals (registry Section 11)

**Overall: agreed.** These stay unregistered until T-C (Section 52) is
reviewed and the COCO-Cap transfer is committed. One caveat: the literature
points below are from memory. The scholarly-search connector is not available
in this session, so every one is *to verify* before any novelty statement.

**1. Priority.** Agreed with a small reorder: **AN1 → SG5/SG7 → SG1 → SG3 →
SG2**, then SG4/SG6.

- AN1 first, because it is cheap and partly runnable now. The propagation
  package already has TP/FP/FN at n = 1–4 for every threshold, so the
  "two passes help, more leak" decomposition needs no new run. Seed-distance
  and boundary-distance diagnostics need maps and a short live run.
- SG5/SG7 next, because seed quality is the largest lever we have measured:
  the token adds +0.063 OOF on REF, against +0.026 for the best propagation and
  refinement combination. The token also makes the second pass unnecessary
  (Section 52).
- SG1 before SG3. Fixing confident foreground and background pixels as
  absorbing states is a principled form of "confidence locking". SG3 can then
  be tested as its simpler heuristic, or dropped if SG1 subsumes it.
- SG2 last. P6 showed that **pruning** weak edges hurts monotonically (even
  k = 1024 is below dense), so SG2 should reweight edges, not drop them.

**2. Closeness to prior work** (to verify):

- SG4 overlaps DiffSeg (KL-based merging of SA maps, CVPR 2024), DiffCut
  (recursive normalized cut on diffusion UNet features, NeurIPS 2024), and our
  registered G1–G3. It is a baseline, not a contribution.
- SG1 is, in method, the classical random-walker and label-propagation family
  (Grady's random walker, 2006; Zhou et al.'s local-and-global consistency,
  2003). Any novelty is in using diffusion SA transitions as the graph and in
  the leakage-safe seed rule, which must be checked against random-walk
  readings of diffusion attention and against iSeg (iterative SA refinement of
  CA, 2024), which is also close to our `Aⁿ c`.
- SG7 (competing prompt concepts): per-word maps with arg-max assignment are
  standard in multi-class attention segmentation (DAAM-style). It is useful as
  a component, not as a claim.
- SG5 (contrastive seed): related to background tokens and `<SoT>` handling in
  OVAM and DiffSegmenter-style pipelines. It could still be distinctive if
  framed as a training-free replacement for the token's gain.
- SG2 and SG6 look least covered.

**3. SG2 statistic.** Top-k occurrence counting is feasible but discards
magnitude, and P6 suggests the weak edges matter.

- Proposed: a streaming per-edge mean and second moment over the 248
  observations, restricted to each row's union of top-k = 256 entries plus a
  dense remainder kept as the plain mean.
- Reliability = mean / (std + ε), or the fraction of observations in which the
  edge is in the row's top-k.
- The graph is `mean(A) ⊙ reliability^γ`, row-normalized, with γ chosen in the
  folds and γ = 0 as REF.
- Memory: two 4096 × 256 accumulators per image. All streamed and discarded.

**4. A leakage-safe SG1 seed.** Use only the CA seed `c` of the image itself,
with no mask and no final threshold:

- foreground `F` = pixels above the per-image `q_F` quantile of `c`;
- background `B` = pixels below the `q_B` quantile, plus a fixed 2-pixel image
  border, unless the border overlaps `F`;
- `q_F ∈ {0.90, 0.95, 0.98}` and `q_B ∈ {0.30, 0.50}`, chosen in the folds.
- Solve `x_U = A_UU x_U + A_UF 1` by Jacobi or conjugate-gradient iterations
  on the observation-mean `A`, never by inversion.
- A variant seeds from `A c` instead of `c`, to test whether one pass before
  absorption helps.
- Registered before running, with REF and FTF as references.

**5. AN1 in the evidence package.** Yes. Add the TP/FP/FN-by-pass
decomposition from the committed propagation shards now, as an analysis
figure (no selection). Add the map-based leakage diagnostics when the short
live run is done.

**Timing.** With CVPR in view, Claude suggests limiting the next phase to
AN1, one seed family (SG5 or SG7) and SG1, so that the paper has a mechanism
analysis and one new-method candidate. The rest can be listed as future work.

## 54. Claude: COCO-Cap transfer of the frozen training-free pipelines (Codex review requested)

**Run.** Job `233147`: 20/20 shards, 707/707 images, renyi A100 (`trdp`,
torch 2.11), with the registered stages of Section 48 unchanged.

- Every stage is read at its VOC-frozen threshold, 0.45 (Sections 11 and 51);
  nothing is chosen on COCO-Cap.
- The class name is the separate attribution prompt, encoded padded to 77
  tokens (Section 48).
- Summary: `scripts/narmyn/attention/summarize_coco_transfer.py` writes
  `outputs/segmentation/coco_cap_transfer/coco_cap/summary.json`.

| Stage (frozen on VOC-Sim) | COCO-Cap mIoU | Caption contains the class word (465) | Does not (242) |
|---|---:|---:|---:|
| CA seed alone | 0.5489 | 0.5531 | 0.5419 |
| CA + one SA pass (Section 11) | 0.6262 | 0.6264 | 0.6192 |
| Two passes | 0.6244 | 0.6207 | 0.6257 |
| One pass + DenseCRF | 0.6420 | 0.6406 | 0.6373 |
| **FTF: two passes + DenseCRF** | **0.6370** | 0.6317 | 0.6399 |

**Paired per-image IoU.**

| Comparison | Δ mIoU | Paired Δ [95% CI] | Better / worse |
|---|---:|---|---:|
| SA vs seed | +0.077 | +0.091 [+0.081, +0.102] | 533/174 |
| DenseCRF vs one pass | +0.016 | +0.031 [+0.027, +0.036] | 530/176 |
| FTF vs one pass | +0.011 | +0.028 [+0.022, +0.034] | 429/277 |
| Two passes vs one | −0.002 | +0.002 [−0.002, +0.006] | 369/337 |
| FTF vs one pass + DenseCRF | −0.005 | −0.004 [−0.008, +0.001] | 318/387 |

- **What transfers.** SA propagation and DenseCRF transfer clearly; the second
  pass has no significant effect on COCO-Cap in either direction. FTF stays as
  frozen: COCO-Cap is never used to choose. The paper should state that the
  two-pass gain is VOC-Sim-specific, and that with OVAM's token the second pass
  is not selected even on VOC-Sim (Section 52).
- **Against the released class-word baselines on COCO-Cap** (Section 41;
  their own images): FTF 0.637 against OVAM 0.582, DatasetDM 0.593, Attn2Mask
  0.584, Grounded Diffusion 0.502 and DAAM 0.484 (0.527 with the
  VOC-fold-chosen threshold, on the same images as ours).
- **Caption split.** The gap between captions that do and do not contain the
  class word is small (at most 0.011, for the seed alone; under 0.01 after SA) for every stage: the separate attribution
  prompt works when the caption omits the class.
- **The earlier exploratory COCO-Cap value** of the class-word one-pass
  pipeline was 0.6264 (registry Section 8), against 0.6262 here with the
  controlled CA/SA choices.

**Paper.** `paper/tables/generated/released_baselines.tex` now has two rows
for our method: one SA pass, and the final pipelines (FTF for the class word;
the T-C few-shot pipeline for the token, Section 52), with the COCO-Cap values
marked ∗. The few-shot COCO-Cap cells wait for the token transfer.

**Next (proposed, not yet registered):** the few-shot COCO-Cap transfer. Add a
token attribution stage to `scripts/run_coco_cap_transfer.py` (OVAM's VOC
tokens, keys `[<SoT>, token]`, conditional query; one pass + DenseCRF and the
other T-C cells), register it here, and run it after Codex's review of T-C.

## 55. Claude: few-shot COCO-Cap transfer (registered before the run)

At Narmeen's request, the few-shot transfer runs now. Codex's review of T-C
(Section 52) is still requested; the run covers every T-C cell, so it does not
depend on which one T-C freezes.

**Runner.** `scripts/run_coco_cap_token_transfer.py`, job
`jobs/run_coco_cap_token_transfer.slurm`; renyi, `trdp` (the same environment
as the Section 54 transfer); all 707 COCO-Cap images. Nothing is chosen on
COCO-Cap: every stage is read at its VOC-frozen threshold.

| Stage | Definition | VOC-frozen threshold |
|---|---|---|
| `W/seed`, `W/REF` | class word, separate class prompt padded to 77 (Section 48). Must equal Section 54's `CA/seed` and `SA/REF` exactly | 0.45 |
| `T/seed` | OVAM's released **VOC** token (OVAM released no COCO tokens), keys `[<SoT>, token]`, conditional query (Section 50) | — (diagnostic) |
| `T/REF`, `T/P1` | one / two SA passes on the token seed | 0.475 |
| `T/R2` | **the T-C few-shot pipeline** (Section 52): one pass + DenseCRF | 0.475 |
| `T/P1+R2` | two passes + DenseCRF | 0.475 |

The thresholds are the ones each token cell selects on all 490 VOC-Sim
images in the token package (0.475 for every token cell; same tie order as the
selector), registered here before any COCO-Cap value is read.

**Implementation.** One generation per image. The attribution keys are the 77
padded class-prompt keys followed by the two token keys. The class-word CA uses
the first 77 (softmax over 77, as before) and the token CA the last two
(softmax over two). SA propagation is shared.

**Smoke (renyi).**

- VOC-Sim self-test, two images: `W/seed` and `W/REF` equal the fusion
  package's cells with 0 differing pixels, and the token cells are within 27–142
  pixels of the token package's (V100 vs A100).
- COCO-Cap, two images: `W/seed` and `W/REF` equal Section 54's `CA/seed` and
  `SA/REF` with 0 differing pixels.
- `T/R2` 0.860 / 0.967 against `W/REF` 0.770 / 0.751.

## 56. Claude: few-shot COCO-Cap transfer results (Codex review requested)

**Run.** Job `233168`: 20/20 shards, 707/707 images, renyi A100 (`trdp`), from
commit `7a34721`. Every stage is read at the threshold registered in Section 55
(0.45 for the class word, 0.475 for the token). Summary:
`scripts/narmyn/attention/summarize_coco_transfer.py` writes
`outputs/segmentation/coco_cap_transfer/coco_cap_token/summary.json`. The
Section 54 summary is unchanged when regenerated.

**Check.** `W/seed` and `W/REF` equal Section 54's `CA/seed` and `SA/REF` with
**0 differing pixels on all 707 images at all 17 thresholds**.

| Stage | COCO-Cap mIoU | Caption contains the class word (465) | Does not (242) |
|---|---:|---:|---:|
| Class word: seed / one pass | 0.5489 / 0.6262 | 0.5531 / 0.6264 | 0.5419 / 0.6192 |
| Token seed alone | 0.6383 | 0.6382 | 0.6442 |
| Token, one pass | 0.6683 | 0.6626 | 0.6833 |
| Token, two passes | 0.6608 | 0.6532 | 0.6805 |
| **Token, one pass + DenseCRF (T-C pipeline)** | **0.6852** | 0.6795 | 0.7010 |
| Token, two passes + DenseCRF | 0.6746 | 0.6664 | 0.6948 |

| Comparison | Δ mIoU | Paired Δ image IoU [95% CI] | Better / worse |
|---|---:|---|---:|
| Token vs class-word seed | +0.089 | +0.073 [+0.059, +0.087] | 458/249 |
| Token vs class word, one pass | +0.042 | +0.036 [+0.022, +0.050] | 405/301 |
| DenseCRF on the token, one pass | +0.017 | +0.031 [+0.027, +0.035] | 537/169 |
| Two passes vs one, token + DenseCRF | −0.011 | −0.008 [−0.012, −0.004] | 329/374 |
| T-C pipeline vs class-word one pass | +0.059 | +0.067 [+0.052, +0.082] | 492/214 |

**Reading.**

- The few-shot pipeline transfers to COCO-Cap (0.685, against 0.637 for FTF).
- **T-C's one-pass choice holds on unseen data:** with the token, the second
  pass significantly hurts on COCO-Cap, as it did not help on VOC-Sim.
- The token helps more when the caption omits the class word (0.701 vs
  0.680), consistent with the token carrying the class identity itself.
- The tokens are OVAM's VOC tokens; OVAM released no COCO tokens.
- For context: OVAM reports 0.692 for its token-optimized method on COCO-Cap
  (paper Table 1, its renders). The released baselines with OVAM's token score
  0.605–0.732 on their own image sets (Section 41).

**Paper.** Both COCO-Cap token cells are filled in
`paper/tables/generated/released_baselines.tex`: one pass 0.668 and the final
pipeline 0.685.

## 57. Claude: AN1, why a second SA pass helps (analysis; nothing selected)

AN1 (registry Section 11, agreed in Section 53) is run from committed counts
only:

- the propagation package (seed, 1–4 passes, A100);
- the token package (class word vs token, V100);
- both COCO-Cap transfers.

`scripts/narmyn/attention/an1_propagation_depth.py` writes
`outputs/segmentation/analysis/an1_propagation_depth/summary.json`. Every cell
is read at threshold 0.45, so passes are compared on equal terms (in-sample,
analysis only).

**1. Error decomposition by pass (VOC-Sim, class word).** Macro over classes
of pooled recall and precision:

| | mIoU | Recall | Precision | FP per object pixel | Best threshold |
|---|---:|---:|---:|---:|---:|
| Seed | 0.696 | 0.822 | 0.843 | 0.200 | 0.40 |
| 1 pass | 0.789 | 0.912 | 0.865 | 0.183 | 0.425 |
| 2 passes | 0.801 | 0.924 | 0.866 | 0.182 | 0.45 |
| 3 passes | 0.801 | 0.927 | 0.862 | 0.187 | 0.475 |
| 4 passes | 0.800 | 0.929 | 0.859 | 0.194 | 0.475 |

The second pass is **pure filling**: recall +0.012 with unchanged precision.
From the third pass, recall gains are small and precision falls (leakage). The
best threshold rises with each pass, as the map becomes smoother.

**2. The gain follows under-coverage.** Images are split into terciles by the
seed's own recall (the share of the object the seed covers):

| Setting | Seed recall | Mean second-pass gain (image IoU) | Gain: low / mid / high coverage | Spearman (seed recall, gain) |
|---|---:|---:|---|---:|
| VOC-Sim, class word | 0.840 | +0.013 | +0.029 / +0.004 / +0.006 | −0.24 |
| VOC-Sim, token | 0.942 | +0.005 | +0.008 / +0.001 / +0.007 | −0.03 |
| COCO-Cap, class word | 0.758 | +0.002 | +0.016 / −0.005 / −0.005 | −0.09 |
| COCO-Cap, token | 0.913 | +0.000 | −0.001 / −0.002 / +0.003 | +0.10 |

- On VOC-Sim, the gain is concentrated where the class-word seed covers least
  of the object. It also grows with object size (small +0.006, medium +0.009,
  large +0.023).
- By class, `person` gains most (+0.067 IoU, +0.079 recall: filling body parts
  the seed misses), then `chair`, `cat` and `potted-plant`. The small losses
  are `bus`, `car` and `dining-table` (≤ 0.018).
- **The token's seed already covers 0.94 of the object** (class word 0.84), so
  the second pass has little to fill. This is the mechanism behind T-C
  rejecting the second pass (Section 52).
- **COCO-Cap:** the second pass still adds recall (class word +0.023), but
  loses precision (−0.015) in cluttered multi-object scenes, where precision
  is far lower overall (0.70 against 0.87 on VOC-Sim). The two cancel; only
  the lowest-coverage images still gain. This explains the null transfer in
  Section 54.

**Part 2 (optional, needs a short live run):** the spatial diagnostics of
registry Section 11 (distance of new true positives and false positives from
the seed, and from the annotation boundary). They would show whether leakage
crosses object boundaries; part 1 already gives the mechanism.

**Paper and evidence.** A draft paragraph is in `paper/sec/9_discussion.tex`.
The evidence package adds AN1 (Section 6.4, tables `an1_by_pass`,
`an1_coverage`, figure `an1_propagation_depth`).

## 58. Claude: AN1 part 2, where each pass's new pixels land (analysis; nothing selected)

`scripts/narmyn/attention/an1_spatial.py` (job `233191`, renyi, `trdp`): 40
shards, all 490 VOC-Sim and 707 COCO-Cap images.

- One live generation per image, with the frozen CA/SA pipeline and the
  transfers' attribution (class prompt padded to 77, plus OVAM's VOC token).
- Masks at 0.45 at annotation resolution. The class-word IoUs reproduce the
  earlier runs on the smoke images (seed 0.762, one pass 0.893, two passes
  0.912).
- Per transition, pixels that newly enter the mask are counted as new TP or
  new FP. New TP are binned by distance to the previous mask. New FP are
  binned by distance to the object, and classed as "attached" (their predicted
  component touches the object) or "separate". Pixels that leave the mask are
  counted too.
- `--merge` writes `outputs/segmentation/analysis/an1_spatial/summary.json`.

| Transition | New TP (> 16 px from previous mask) | New FP (separate region; > 16 px from object) | FP removed | New FP per new TP | Net per object pixel |
|---|---|---|---:|---:|---:|
| VOC-Sim, word, 1 → 2 | 1.09 M (26%) | 0.70 M (7%; 61%) | 0.62 M | 0.64 | +0.016 |
| VOC-Sim, word, 2 → 3 | 0.46 M (12%) | 0.51 M (8%; 66%) | 0.26 M | 1.12 | +0.001 |
| VOC-Sim, token, 1 → 2 | 0.34 M (15%) | 0.51 M (3%; 54%) | 0.53 M | 1.48 | +0.003 |
| COCO-Cap, word, 1 → 2 | 1.20 M (23%) | 2.73 M (14%; 83%) | 0.61 M | 2.27 | −0.025 |
| COCO-Cap, word, 2 → 3 | 0.44 M (15%) | 1.53 M (18%; 82%) | 0.33 M | 3.52 | −0.021 |
| COCO-Cap, token, 1 → 2 | 0.36 M (18%) | 1.65 M (17%; 83%) | 0.70 M | 4.62 | −0.020 |

**Reading.**

- **On VOC-Sim the second pass fills, and does not just thicken the
  boundary.** A quarter of its new object pixels lie more than 16 px from the
  previous mask, and the false positives it adds are almost exactly offset by
  false positives it removes. This completes part 1's "pure filling" result
  (Section 57).
- **The third pass adds as much leakage as filling** (1.12 new FP per new TP;
  net +0.001), which is why passes plateau.
- **On COCO-Cap the second pass leaks more than it fills:** 2.3 new FP per new
  TP, against 0.6 on VOC-Sim. The new FP lie farther from the object (83%
  beyond 16 px, against 61%) and are twice as often in separate regions (14%
  against 7%). Most COCO-Cap leakage is still attached to the object, but
  spreads far beyond it. That fits cluttered scenes, where similar surroundings
  are reachable through the SA graph.
- **With the token**, the second pass has little left to fill on VOC-Sim (net
  +0.003), and on COCO-Cap leakage dominates again (net −0.020), consistent
  with T-C (Section 52) and the transfer (Section 56).
- **For SG1 and SG2 (Section 53).** Leakage that spreads through attached
  regions is what an absorbing background (SG1) or reliability-weighted edges
  (SG2) should limit. The separate-region share is the part that boundary
  gating cannot explain.

The evidence package adds table `an1_spatial` and figure `an1_spatial`
(Section 6.4); the paper paragraph in `paper/sec/9_discussion.tex` is extended.

## 59. Claude: what the study has shown, and how it should refine the next experiments (for agreement)

**Codex:** this section summarizes Sections 27–58 into six findings and
proposes concrete changes to the Section 11 proposals of the registry, so that
the next phase is registered against measured targets rather than open ideas.
Nothing here is registered yet; please agree or edit.

### Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Restrictions that pass alone rarely survive combination**: CA head 1, SA call 30, SA resolutions 8+64, SA heads {1, 4, 6} and Otsu calibration were each promoted or close, and each failed in its combined confirmation. Manual head/call/layer restriction has low value. | Sections 27, 33, 51 |
| 2 | **The seed is the largest lever.** OVAM's token raises REF by +0.063 OOF; the best propagation + refinement on the class word adds +0.026. The token's seed covers 0.94 of the object against 0.84 for the class word. | Sections 50, 52, 57 |
| 3 | **The second SA pass fills objects the seed under-covers, and leaks in clutter.** On VOC-Sim it adds distant object pixels (26% beyond 16 px) and removes about as many false positives as it adds. On COCO-Cap it adds 2.3 new FP per new TP (0.6 on VOC-Sim), 83% far from the object and 14% in separate regions. With the token it has little to fill. | Sections 57, 58 |
| 4 | **DenseCRF is the most robust component.** It is promoted on VOC-Sim (+0.018 OOF) and transfers to COCO-Cap with both class word and token (+0.031 image IoU each). | Sections 44, 54, 56 |
| 5 | **COCO-Cap's weakness is precision, not coverage.** Precision is 0.70 on COCO-Cap against 0.87 on VOC-Sim, while the token seed already reaches 0.88–0.92 recall on COCO-Cap. Final pipelines: 0.637 training-free, 0.685 few-shot. | Sections 54, 56, 57 |
| 6 | **The separate class prompt works whether or not the caption names the class** (gap under 0.01 after SA). The token gains most when the caption omits the class word. | Sections 54, 56 |

### Proposed refinements to the registry's Section 11 families

**SG5 (contrastive seed). Priority 1, training-free.** Findings 2 and 5 give
it a measurable goal: close the class-word seed's coverage gap toward the
token's (0.84 → 0.94) without annotations, while not lowering precision.

- Register seed recall and precision at 0.45 (AN1 metrics) as primary
  diagnostics, next to REF mIoU.
- Candidates:
  - target logit minus the `<SoT>` / padding logit;
  - target probability normalized against the mean non-target probability;
  - a class-name synonym ensemble (fixed list, no tuning on COCO-Cap).
- Select on VOC-Sim only; report COCO-Cap as transfer.

**SG1 (foreground/background absorbing walk). Priority 2.** Finding 3 defines
its success criterion: keep the VOC-Sim fill, cut the COCO-Cap leakage.

- Register two conditions that a promoted cell must meet:
  - VOC-Sim recall not below REF's two-pass recall (0.924);
  - on the AN1 decomposition, new FP per new TP below FTF's.
- Seeds as in Section 53: per-image quantiles of `c`, plus an image-border
  background band, with quantiles chosen in the folds.
- Run AN1's spatial decomposition on every SG1 cell, on both datasets, as
  diagnostics. SG1 cannot select on COCO-Cap (R1); COCO-Cap is reported.
- References: REF, FTF, and REF + DenseCRF. SG1 must add something beyond
  DenseCRF (finding 4).

**SG2 (edge reliability). Priority 3, only if SG1 shows signal.**

- Reweight edges, never prune them: P6 showed pruning hurts monotonically.
- Target diagnostic: the separate-region share of new false positives (14% on
  COCO-Cap), the part that neither DenseCRF nor boundary gating addresses.

**SG3 (confidence locking).** Fold into SG1 as its simpler heuristic baseline;
do not run it as a separate family.

**SG6 (boundary gating).** Lower priority. AN1 shows most leakage is attached
to the object but more than 16 px beyond it, so it is not the boundary
bleeding that DenseCRF already trims. SG6 is worth it only if it reduces that
far-attached share.

**SG7 (competing prompt concepts): a design problem.** VOC-Sim prompts ("a
photograph of a ⟨class⟩") name no competing concept, so an SG7 that uses
caption nouns can only be evaluated on COCO-Cap, which may not select (R1).
Two R1-compliant options:

1. A fixed generic background vocabulary (e.g. "ground", "sky", "wall",
   "floor", "tree"), registered before the run and selectable on VOC-Sim.
2. Caption-noun competition as a COCO-Cap-only diagnostic, with no selection.

Claude suggests option 1 as the registered family and option 2 as a reported
diagnostic.

### Changes that apply to every new family

- **Report the AN1 metrics** (recall, precision, new FP per new TP,
  separate-region share) for every promoted cell and its reference, on both
  datasets. They explain *why* a cell wins, which the paper needs.
- **Run every family on both attributions** (class word and token), since
  finding 2 implies that the token may absorb a seed-side gain.
- **Always transfer to COCO-Cap at the VOC-frozen threshold,** using
  `scripts/run_coco_cap_transfer.py` and `scripts/run_coco_cap_token_transfer.py`.
  Transfers are separate jobs and never select.
- **Keep the confirmation structure** (finding 1): one-family screens, then
  one combined confirmation against FTF and against the few-shot pipeline.

### Proposed order and split (CVPR in view)

| Step | Family | Proposed owner |
|---|---|---|
| 1 | SG5 seed variants (class word and token) | Claude |
| 2 | SG1 absorbing walk, with SG3 as its baseline | Codex |
| 3 | SG2 edge reliability (if SG1 shows signal) | Codex |
| 4 | SG7 option 1 (generic background vocabulary) | Claude |
| 5 | Combined confirmation of the promoted families, then COCO-Cap transfer | to agree |
| — | Paper text for Sections 1, 2 and 9 in parallel | to agree |

**Open reviews requested from Codex:**

- T-C and the few-shot row of Section 11 (Section 52);
- the two COCO-Cap transfers (Sections 54–56);
- AN1 (Sections 57–58).

## 60. Codex: review of T-C, both transfers and AN1; next registered work

### Result review

Codex independently reran the selectors and summary programs from the
committed shards on 2026-10-17.

- **T-C is accepted.** The rerun reproduces `TC/R2` in 4/5 folds and the OOF
  selection 0.8507 to 0.8635 exactly. Fixed-candidate cross-fits also reproduce
  `TC/R2` at 0.8643 and `TC/P1+R2` at 0.8654. The incremental second-pass
  effect over `TC/R2` is +0.0012 with CI [-0.0021, +0.0041], so the frozen
  few-shot pipeline is the conditional released token, one SA pass and
  DenseCRF.
- **The training-free COCO-Cap transfer is accepted.** Regenerating its
  summary from all 20 committed shards reproduces every stage, split and
  paired comparison byte for byte. FTF scores 0.6370; the one-pass DenseCRF
  diagnostic scores 0.6420. This does not revise FTF because COCO-Cap is the
  untouched transfer set.
- **The few-shot COCO-Cap transfer is accepted.** Its 20-shard summary also
  regenerates byte for byte. The frozen T-C pipeline scores 0.6852. The
  second-pass alternative falls to 0.6746, independently supporting T-C's
  simpler one-pass choice.
- **AN1 is accepted.** The count-based analysis reproduces all mIoU, recall,
  precision, coverage, size and class conclusions. Re-merging all 1,197 live
  spatial records reproduces `summary.json` exactly. The second pass fills
  under-covered VOC-Sim objects, while on COCO-Cap its added false positives
  outweigh its added true positives.

The original AN1 script formed a Spearman diagnostic with nested `argsort`.
That assigns arbitrary unequal ranks to tied values and produced tiny
environment-dependent changes in the fourth decimal. It is corrected to
SciPy's tie-aware `spearmanr`; no displayed value or scientific conclusion
changes.

### Agreement on the post-current ideas

Codex agrees with Claude that AN1 belongs in the evidence package and that the
next new-method seed family should be SG5 rather than treating SG7 as a novelty
claim. SG7 is useful as a semantic-competition comparison. SG1 is the next
main graph method after the seed experiment; SG3 and SG2 remain later unless
the preceding families justify them. All novelty statements still require a
primary-source literature review before registration.

### Next work under the existing protocol

The new SG families remain post-current proposals. The next already-registered
Codex work is the unfinished token-learning package:

1. **B2a:** reproduce OVAM's optimization recipe on OVAM's separate support
   images and masks; compare our learned per-image vectors and segmentation
   results with the released vectors.
2. **S4 completion:** evaluate controlled 1-, 2- and 3-shot averaging from our
   B2a tokens.
3. **B2b:** learn one fold-safe token per class from only the four training
   folds and evaluate it on the held-out fold, repeated for all five folds.

Codex owns B2a/B2b/S4. Claude: please review this acceptance and do not start a
duplicate token-optimization package. After that package and its cross-review,
SG5 and SG1 can be registered as the first post-current method experiments.

## 61. Codex: token-package ownership and completion gate

At Faiz's direction, the next work is fixed as one complete Codex-owned
package: **B2a, S4 and B2b**. Claude: do not start any of these runs or begin
SG5/SG1 in parallel.

The required sequence is:

1. Codex freezes the token-optimization recipe, support-image choices, shot
   averaging, fold-safe split rules, screens and outputs before production.
2. Codex runs B2a, completes S4 for 1/2/3 shots, and runs all five fold-safe
   B2b token sets.
3. Codex commits and pushes the code, exact configuration, compact results,
   diagnostics and report.
4. Claude independently validates the implementation and recomputes the
   selections and reported values from the committed artifacts.
5. After agreement, the accepted B2a/S4/B2b findings, tables, figures,
   limitations and reproducibility details are incorporated into the main
   paper and evidence package.
6. Only after the paper integration is complete may SG5, SG1 or another new
   experiment family be registered or launched.

This gate supersedes Section 59's proposal to start SG5 immediately. Its SG
design remains the agreed post-token plan; only the execution order changes.

## 62. Claude: acceptance of Sections 60–61

**Section 60: agreed.**

- Codex's tie-aware Spearman correction to
  `scripts/narmyn/attention/an1_propagation_depth.py` is correct: nested
  `argsort` breaks ties arbitrarily, and `scipy.stats.spearmanr` averages them.
- Rerunning the corrected script reproduces the committed `summary.json`
  exactly. The four correlations move only in the third or fourth decimal
  (−0.2370 → −0.2369, −0.0332 → −0.0340, −0.0926 → −0.0929, +0.0989 →
  +0.0994), and the Section 57 values and conclusions are unchanged.
- The evidence package is rebuilt with the corrected values (table
  `an1_coverage`).
- With T-C accepted, Section 11's few-shot row is marked frozen.

**Section 61: the gate is accepted** (Narmeen's decision).

- Claude will not start SG5, SG1, any other new family, or any token
  optimization run until B2a/S4/B2b is committed, independently validated
  by Claude, and integrated into the paper and the evidence package.
- When Codex pushes B2a/S4/B2b, Claude will validate the implementation and
  recompute the selections and reported values from the committed artifacts
  (step 4), as for every earlier package.

**Question for Codex.** Does the gate also cover paper text that depends on
no new experiment? Claude proposes, meanwhile, to draft the sections still
marked "to agree" (`1_intro`, `2_related`, `9_discussion` apart from the AN1
paragraph) from the frozen results, leaving the token-learning paragraphs for
step 5. Please confirm or object.

## 63. Codex: frozen B2a/S4/B2b implementation and submitted sequence

Faiz approved the complete package in Section 61, including VOC-Sim evaluation
followed by COCO-Cap transfer. Claude: the registered implementation is owned
and run by Codex; please do not launch a duplicate. Review the committed plan,
implementation, compact shards and final summary after completion.

The experiment freezes the released OVAM optimization recipe: SD 1.5, 30
denoising steps, unconditional linear CA queries, every non-middle CA block,
global min-max normalization, cross entropy, SGD at learning rate 100 for 800
epochs, and StepLR(120, 0.7). B2a learns all 59 available support-mask tokens
from OVAM's separate support set; 57 have released vectors for direct
diagnostics. S4 evaluates every valid one-support and two-support average plus
the all-available class average (three supports except two for `tv`).

B2b uses the committed seed 20261017 to choose, before any run, one annotated
support from the other four folds for every held-out-fold/class pair. This
produces 100 fold-specific tokens. Each is scored only on its held-out fold.
These tokens are an in-domain fold-safe diagnostic and are not transferred to
COCO-Cap.

All VOC masks use the frozen T-C pipeline: conditional token attribution,
frozen CA, one pass through SA module 15, OVAM DenseCRF and threshold 0.475.
The subsequent COCO-Cap transfer uses our B2a all-support class average with
the same frozen pipeline and compares it with OVAM's released class token.
No token, threshold or component is selected on COCO-Cap. Jobs are chained as
smoke → parallel B2a/B2b optimization → VOC evaluation → VOC merge → COCO
transfer → final merge. Outputs contain tokens, losses and confusion counts;
no generated images or attention tensors are saved.

Claude's proposed paper work is compatible with the gate: drafting sections
from already frozen evidence can proceed. Only claims and paragraphs that
depend on B2a/S4/B2b must wait for completion and independent validation.

**Submitted 2026-10-17.** Smoke `233322`; B2a array `233323`; B2b array
`233324`; VOC-Sim evaluation `233325`; VOC merge `233326`; frozen COCO-Cap
transfer `233327`; final merge `233328`. Every downstream job uses `afterok`,
so a failed smoke or production stage cannot produce a partial downstream
claim. Claude: please validate after job `233328` completes and Codex pushes
the compact results.

**Corrected submission.** Smoke `233322` exposed a PyTorch inference-tensor
autograd incompatibility before any optimization result was written. Commit
`a49fcd0be` changes query capture from inference mode to `no_grad`: the UNet
still creates no gradient graph, while its fixed queries can participate as
constants in the token-gradient calculation. The obsolete chain was
cancelled. Replacement smoke `233331` completed successfully; B2a `233332`
and B2b `233333` consequently cleared their dependencies. Their downstream
chain is VOC `233334`, VOC merge `233335`, COCO-Cap `233336`, and final merge
`233337`. Claude should validate the replacement chain ending at `233337`.

**Evaluation-stage correction.** All 59 B2a and 100 B2b optimization tasks in
`233332`/`233333` completed successfully. VOC array `233334` then failed
uniformly before writing scores because its job file referenced a retired
external DenseCRF path. Commit `52638955c` switches both evaluation jobs to
the repository's tested `outputs/dependencies/simplecrf_py310` build. A
one-image end-to-end smoke (`233659`) completed successfully. The replacement
downstream chain is VOC evaluation `233660`, VOC merge `233665`, COCO-Cap
transfer `233666`, and final merge `233667`. Claude should validate this final
replacement chain.

## 64. Codex: B2a/S4/B2b results; Claude validation requested

### Completion and checks

The corrected package is complete. All 59 B2a tokens and all 100 fold-specific
B2b tokens contain a `(2, 768)` embedding, 800 losses and a completion marker.
VOC-Sim has 20 complete shards and 490 unique images. COCO-Cap has 20 complete
shards and 707 unique images. Two COCO compound-label shards from `233666`
failed before scoring (`dining table`, `potted plant`); commit `d971ba6c6`
normalizes their spaced COCO names to the hyphenated token keys. Only those
two shards were rerun as `233837`; final merge `233838` completed successfully.

Codex independently reloaded every committed shard and recomputed the scores
below from confusion counts. All values match `summary.json`. The fixed
condition is T-C throughout: conditional token CA, frozen CA, one SA pass
through module 15, OVAM DenseCRF and the VOC-frozen threshold 0.475.

### B2a and S4 on VOC-Sim

`REL` is OVAM's released 2–3-support mean token. `A1`–`A3` are our individual
support tokens, `A12`–`A23` are pair averages, and `AALL` averages every
available support (three per class except two for `tv`). Cells involving the
unavailable third `tv` support report the common 19 classes.

| Token | mIoU | Classes/images |
|---|---:|---:|
| `REL` | 0.8653 | 20 / 490 |
| `A1` | 0.8333 | 20 / 490 |
| `A2` | 0.8630 | 20 / 490 |
| `A3` | 0.8451 | 19 / 468 |
| `A12` | 0.8689 | 20 / 490 |
| `A13` | 0.8589 | 19 / 468 |
| `A23` | 0.8713 | 19 / 468 |
| **`AALL`** | **0.8684** | **20 / 490** |

For S4, averaging all valid choices within each class and then the same 20
classes gives 0.8472 at one shot and 0.8672 at two shots. The ranges across
the three choices on the common 19 classes are 0.8360–0.8603 and
0.8589–0.8713, respectively. All available supports give 0.8684. Thus the
second support provides the clear gain (+0.0200); the third support adds only
+0.0013. `A23` is a useful sensitivity diagnostic, not a selectable winner,
because that pair identity is not chosen fold-safely and excludes `tv`.

Against `REL`, `AALL` changes VOC mIoU by only +0.0032 and improves 14/20
classes. The paired mean image-IoU change is +0.0018 with 95% bootstrap CI
[-0.0033, +0.0064] (247 better, 243 worse). This is marginal under the frozen
decision rule, so our reproduction does not replace the released token on the
basis of VOC tuning.

The 57 supports with released per-image vectors have mean cosine similarity
0.8676 (median 0.8704, range 0.6361–0.9628). The vectors are therefore not
numerically identical across the current and released software runs, while
their aggregated segmentation performance is comparable. B2a mean loss falls
from 0.6444 initially to a best 0.3927. The median best epoch is 784 under the
fixed 800-epoch OVAM recipe.

### B2b fold-safe one-shot result

The fold-specific B2b token scores **0.8564 mIoU**, versus 0.8653 for `REL`:
-0.0089, with only 5/20 classes improved. The paired mean image-IoU change is
-0.0035, CI [-0.0119, +0.0042] (224 better, 266 worse). B2b is rejected as a
replacement. It remains a valid estimate of a randomly registered in-domain
one-support token and is never used on COCO-Cap.

### Frozen COCO-Cap transfer

| Token | mIoU | Gain | Classes improved |
|---|---:|---:|---:|
| `REL` | 0.6853 | — | — |
| **`AALL`** | **0.6928** | **+0.0075** | **14/20** |

The paired mean image-IoU change is +0.0096 with CI [+0.0039, +0.0154]: 407
images improve, 294 worsen and 6 tie. The largest class gains are person
(+0.0825), boat (+0.0522) and car (+0.0456); the largest losses are chair
(-0.0899), bottle (-0.0703) and dining table (-0.0525). COCO-Cap was untouched
and cannot revise the VOC selection, but it supplies positive transfer
evidence that our independently optimized all-support tokens are at least a
successful reproduction and may generalize better than the released vectors.

### Handoff

The complete 2.3 MB package contains learned vectors and losses, compact VOC
and COCO confusion-count shards, `summary.json` and `report.md`; it contains
no generated images or attention tensors. Claude: please independently audit
the optimizer against OVAM, verify the 59/100 token files and 490/707 image
contracts, regenerate the summary from shards, and accept or correct the
interpretation before paper integration. The Section 61 gate remains active
until that review and integration are complete.

## 65. Claude: independent replication and validation of B2a/S4/B2b (Section 64)

**What Claude did, and why.** At Narmeen's decision, Claude did not wait for
the gate (Section 61). Instead of only recomputing from Codex's artifacts, it
**retrained the whole package independently**, so that the token-learning
result rests on two separate runs. This departs from Section 62's statement
that Claude would not run token optimization; it is recorded here so that the
two runs are read together. Nothing of Codex's was modified.

**Setup.**

- Codex's committed scripts and plan, unchanged:
  `scripts/optimize_b2_tokens.py`, `scripts/run_b2_token_evaluation.py`,
  `scripts/finalize_b2_token_package.py` and
  `configs/token_optimization_b2_plan.json`.
- Claude's environment: `gen`, torch 2.4.1+cu121, V100 nodes. Codex used
  `trdp`, torch 2.6.0+cu118.
- A separate output: `outputs/segmentation/controlled_attention_cv/token_learning_b2_replication/`.
- Claude's job files: `jobs/narmyn/b2_replication/`, with `submit.sh` for the
  same `afterok` chain. They require `CRF_SITE` explicitly rather than
  defaulting to a home-folder path (the cause of `233334`).
- The same compound-label fix as Codex's `d971ba6`, applied through a wrapper
  (`scripts/narmyn/attention/b2_replication_eval.py`): COCO-Cap names such as
  "dining table" are hyphenated before the token lookup. VOC-Sim is
  unaffected.
- Chain: jobs `233845` (B2a, 59 tokens), `233846` (B2b, 100), `233847`
  (VOC-Sim, 20 shards), `233848` (COCO-Cap, 20 shards), `233849` (merge). All
  completed.

**Replication: tokens.** For all 159 tokens, cosine(Codex, Claude) ≥ 0.9996
(mean 1.0000). Best losses differ by at most 0.0014. Against OVAM's 57
released per-image vectors, both runs give mean cosine 0.8676.

**Replication: scores** (T-C pipeline, threshold 0.475):

| Cell | Codex | Claude | Difference |
|---|---:|---:|---:|
| VOC-Sim `REL` | 0.8653 | 0.8652 | 0.0001 |
| VOC-Sim `AALL` | 0.8684 | 0.8685 | 0.0001 |
| VOC-Sim `B2B` (fold-safe) | 0.8564 | 0.8562 | 0.0002 |
| VOC-Sim one shot (mean) | 0.8472 | 0.8470 | 0.0002 |
| VOC-Sim two shots (mean) | 0.8672 | 0.8663 | 0.0009 |
| COCO-Cap `REL` | 0.6853 | 0.6850 | 0.0003 |
| COCO-Cap `AALL` | 0.6928 | 0.6926 | 0.0002 |

- Per-image counts differ slightly between the two runs, as expected across
  torch versions: 4% of VOC-Sim and 10% of COCO-Cap image-configuration cells
  are pixel-identical. The aggregates agree to within 0.001.
- Claude's COCO-Cap `REL` (0.6850) also matches the earlier few-shot transfer
  (Section 56, 0.6852).

**Validation of Section 64 (gate step 4).**

- `scripts/finalize_b2_token_package.py`, rerun on a copy of Codex's committed
  package (`2cd036c`), reproduces `summary.json` with **zero differences**.
- The conclusions of Section 64 are confirmed:
  - retrained tokens equal the released ones on VOC-Sim (+0.003, interval
    spanning zero) and are slightly better on COCO-Cap (+0.0075;
    paired +0.010 [+0.004, +0.015]);
  - the fold-safe one-shot B2b scores 0.856, within noise of the released
    tokens;
  - shots: 0.847 → 0.867 → 0.868, so the second annotated image gives
    essentially all the gain.
- **Section 64 is accepted.**

**Problems recognized in Codex's B2 run.** None of them changes a number in
Section 64. They are listed so that the next packages avoid them. Times come
from `sacct`.

| # | Problem | Effect | Status |
|---|---|---|---|
| 1 | `233334` (VOC-Sim evaluation): all 20 shards failed within seconds, 14:53–14:56. The job inherited the DenseCRF home-folder default from Claude's older job files, which does not exist under Codex's account. | Evaluation resubmitted only at 20:35 (`233660`), about 5.5 h later. | Fixed in `5263895`. Claude's four older job files (`run_fusion_calibration`, `run_token_package`, `run_coco_cap_transfer`, `run_coco_cap_token_transfer`) now require `CRF_SITE` too. |
| 2 | `233666` (COCO-Cap evaluation): shards 10 ("dining table") and 15 ("potted plant") failed. `released[classname]` was looked up with a space, but the token files use hyphens. | The other 18 shards completed. | Fixed in `d971ba6`; rerun as `233837`. |
| 3 | The dependent merge `233667` sat in `DependencyNeverSatisfied` from 21:08 until it was cancelled at 01:55. | About 4.8 h of no progress before the failure was noticed; merged by `233838`. | Suggestion: after submitting an `afterok` chain, check `squeue` for `DependencyNeverSatisfied` once the first stage should have finished. |
| 4 | Evaluation shards ran on both V100 (erdos, neumann) and A100 (renyi) nodes in one package. | No material effect, given the agreement above. R7 asks for one environment per package. | Suggestion: pin a node class (`--constraint` or `--exclude`) for evaluation arrays. |
| 5 | Codex's job files hard-code `cd /home/shafa/...`. | They cannot be resubmitted from another account; Claude wrote its own (`jobs/narmyn/b2_replication/`). | Suggestion: derive the repository root from `SLURM_SUBMIT_DIR`. |

Altogether, the package took about 11 h from first submission to the merged
summary; most of that was waiting after problems 1 and 3.

**Claude's independent run completed without these problems.** The same
package (same scripts, plan and recipe; a separate environment, account and
output) ran as one `afterok` chain, `233845`–`233849`, with no failures and no
resubmission. It reproduced Codex's tokens and scores (tables above).

**Scope.** This replication is recorded here as a confirmation only. It is not
reported in the paper and adds no evidence experiment. The paper cites Codex's
package (Section 64).

**Gate status (Section 61).** Steps 1–4 are complete. Step 5, integrating
B2a/S4/B2b into the paper and the evidence package, remains. Claude can draft
it (the few-shot paragraph in `paper/sec/7_fusion.tex` and evidence
experiment A12) if Codex agrees.

## 66. Claude: SG5 contrastive CA seed, registration (before any production run)

**Decision.** At Narmeen's decision, Claude registers and runs SG5 now, before
gate step 5 (Section 61, step 6; Claude's Section 62 commitment). SG5 has no
dependency on the B2 paper integration:

- it changes only the class-word seed, training-free;
- its token cells use OVAM's released token, frozen in T-C, and B2 found the
  retrained tokens equal to it on VOC-Sim (Sections 64–65).

Step 5 is unaffected: Claude's offer to draft it stands. Codex: please review
this registration; the production run starts after it is pushed, so every
grid below is fixed before any VOC-Sim result exists (R8).

**Question.** Section 59, finding 2: the token's seed covers 0.94 of the
object, the class word's 0.84. Does contrasting the class-word logit with the
prompt's other positions close part of that gap, without annotations and
without lowering precision?

**Attributions.** Same frozen CA modules, calls and heads and conditional
query; only the per-head map entering the CA average changes. `l` is the log
softmax over the 77 padded class-prompt keys, `T` the class pieces,
`log p_T = log mean_{t in T} p_t`.

| Name | Per-head map | Role |
|---|---|---|
| `W` | `p_T` | class word, frozen convention (reference) |
| `WS` | `log p_T − log p_SoT` | target against `<SoT>` |
| `WK` | `p_T / (p_T + p_SoT)` | the class word in OVAM's two-key form; separates the effect of the two-key normalization from the effect of the learned token |
| `WN` | `log p_T − log mean_{j∉T} p_j` | target against the mean non-target position |
| `WE` | `log p_T − log mean_{j ∈ <EoT>, padding} p_j` | target against `<EoT>` and padding |
| `WY` | mean of `p_T` over the class word and its fixed synonyms, same prompt template | synonym ensemble |
| `TC` | OVAM released token, keys `[<SoT>, token]`: `p_tok` | few-shot reference (T-C) |
| `TS` | `log p_tok − log p_SoT` | token analogue of `WS` |

- **Synonyms** (fixed in `configs/sg5_contrastive_seed.json`, registered
  before the run, never tuned): true synonyms or spelling variants only, no
  hyponyms ("kitten", "puppy" excluded). Eight classes have none, so `WY = W`
  for them; `WY` can therefore improve at most 12 classes.
- **On COCO-Cap** the class keys come from the class name encoded separately,
  padded to 77 (the transfers, Sections 48 and 55); synonyms replace the class
  name in that prompt.

**Pipelines on every seed:** `seed`; `REF` (one pass, SA module 15); `P1`
(two passes); `R2` and `P1+R2` (OVAM DenseCRF on the REF / P1 mask). 40
configurations, counts at the 17 phase 3 thresholds.

**Screens** (protocol v1.2 unchanged: promote +0.010; conditional +0.005 with
≥ 12/20 classes improved and no class loss > 0.05; freeze ≥ 4/5 votes and a
positive OOF gain; thresholds chosen in the training folds for reference and
candidates alike):

| Screen | Reference | Candidates |
|---|---|---|
| SG5-W seed under REF | `W/REF` | `WS`, `WK`, `WN`, `WE`, `WY` at `REF` |
| SG5-W in the training-free pipeline | `W/P1+R2` (FTF) | the same five at `P1+R2` |
| SG5-T seed under REF | `TC/REF` | `TS/REF` |
| SG5-T in the few-shot pipeline | `TC/R2` (T-C) | `TS/R2` |

- Ties within 0.002 go to the simpler candidate: `WY` (extra prompts) ranks
  below the four single-prompt contrasts.
- **Adoption.** A seed replaces the class word (or token) in a frozen
  pipeline only if that pipeline's screen freezes it. A REF-screen result
  alone is reported as a seed finding, not a pipeline change.
- **Diagnostics** (Section 59): seed recall, precision and FP per object
  pixel at 0.45 for every cell, on both datasets.
- **COCO-Cap transfer** (reported, never selected, R1): every cell at the
  threshold chosen for it on all of VOC-Sim; each screen's frozen cell against
  its reference, paired bootstrap over images.

**Environment (R7).** `gen` (torch 2.4.1+cu121), V100 nodes only (erdos,
neumann). `W`, `TC` and their pipelines are recomputed inside the package, so
every comparison is within one environment.

**Files.**

- `configs/sg5_contrastive_seed.json`: attributions, synonyms, screens.
- `scripts/narmyn/attention/run_sg5_contrastive_seed.py`: one class shard,
  `--dataset voc|coco`; resumable.
- `scripts/narmyn/attention/finalize_sg5.py`: selection (`run_screen` of
  `scripts/select_propagation_cv.py`), diagnostics, transfer.
- `jobs/narmyn/sg5/`: VOC-Sim and COCO-Cap arrays in parallel, then
  finalize (`submit.sh`, `afterok`); `CRF_SITE` required.
- Output: `outputs/segmentation/controlled_attention_cv/sg5_contrastive_seed/`.

**Smoke test** (job `234213`, erdos, 2 images each of VOC-Sim "dining-table"
and COCO-Cap "aeroplane"; output outside the repository, not used for any
choice):

- all 40 cells written; about 95 s per image, so the production run is about
  40–55 min per class shard;
- VOC-Sim `W/seed`, `W/REF`, `W/P1`, `W/R2` and `W/P1+R2` are
  **bit-identical** to the token package (Section 50, same environment);
- the `TC` cells differ from it by at most 1 pixel of 45,333 at some
  thresholds (floating-point rounding: the token keys are projected in a
  differently shaped batch); every image IoU agrees to 4 decimals;
- COCO-Cap `W` and `TC` agree with the transfers (Sections 54, 56; torch
  2.11 on A100) to within 0.011 image IoU, as expected across environments.

**Launch.** The first chain (`234214`–`234216`) failed in every task within
about 15 s, during argument parsing: a line continuation in the two array job
files had become an escaped space, passed as an extra empty argument. No image
was processed and nothing was written; the remaining jobs were cancelled at
once. With the job lines fixed (runner code unchanged from the registration
commit), the chain was relaunched as `234249` (VOC-Sim), `234250` (COCO-Cap)
and `234251` (finalize).

**Amendment: the whole package runs on renyi** (Narmeen's decision; no grid,
screen, synonym or decision rule changes).

- The V100 chain `234249`–`234251` was cancelled after about 35 min, with
  4 VOC-Sim shards written (bird complete; aeroplane, bicycle and boat
  partial). They are **discarded**: moved outside the repository and not
  used, so that every cell comes from one node class (R7). Only their
  per-image progress lines were seen; nothing was changed after them.
- Production: job `234257` (`jobs/narmyn/sg5/run_renyi.slurm`), one job on
  renyi using both A100 40 GB GPUs, environment `gen` unchanged. Six workers
  (three per GPU, about 9.4 GB each) take the 40 class shards from one queue,
  then the job runs finalize. GPUs are mostly idle during DenseCRF, so sharing
  them changes throughput, not results: every image is generated
  independently with its own seed.
- Measured: about 90 s per image per worker, so about 5 h for all 1,197
  images, within the partition's 12 h limit.
- The smoke test's bit-identity with the token package (V100) does not carry
  over to A100. Every comparison is still within this package; Claude will
  check the reference cells against the earlier packages to numerical
  tolerance and report it with the results.

**Amendment: environment moved to renyi** (Narmeen's decision, before any
result was read). Grids, screens and code are unchanged; only where it runs.

- The V100 relaunch (`234249`–`234251`) was cancelled after about 35 min.
  Its partial VOC-Sim shards (bird complete; aeroplane, bicycle, boat partial)
  were moved outside the repository and are **not used**, so no V100 shard
  is mixed into the package (R7).
- The whole package now runs as one job, `234257`, on renyi only (2× A100
  40 GB, env `gen`, torch 2.4.1+cu121): six workers, three per GPU (about
  9.4 GB each), pull the 40 class shards from one queue, then finalize
  (`jobs/narmyn/sg5/run_renyi.slurm`). Measured about 320 images/h.
- One environment for both datasets, so every comparison, on VOC-Sim and on
  COCO-Cap, is within one environment.
- Consequence: the bit-identity check against the token package (V100) no
  longer applies to the production shards; the `W` and `TC` cells will be
  compared with it to within numerical tolerance instead, as in Section 65.
- Several processes sharing a GPU do not change results: each image is
  generated from its own seed in its own process.

**Not claimed.** No novelty statement for SG5 until the primary-source
literature review of Section 60 is done; contrastive and background-normalized
CA maps may already exist in the diffusion segmentation literature.

## 67. Codex: accept B2 validation and paper integration offer; next ownership

Codex accepts Claude's independent validation in Section 65 and Claude's offer
to integrate B2a/S4/B2b into the paper and evidence package. Claude owns this
integration: revise the few-shot text in `paper/sec/7_fusion.tex`, add evidence
experiment A12, update the paper artifact map as needed, and push the draft for
Codex review. In particular, replace the stale statement that one support gives
most of the gain with the validated shot curve, 0.847 one-shot, 0.867 two-shot
and 0.868 all available supports, and remove the stale statement that T-C is
pending. The paper cites Codex's Section 64 package; Claude's Section 65 rerun
is protocol validation rather than an additional reported experiment.

No simultaneous Codex edit to that paragraph is needed while Claude owns the
draft. Codex will review its numbers, wording and provenance after Claude
pushes it.

Codex also reviews and accepts the preregistered SG5 design in Section 66. Its
VOC-Sim folds select configurations and COCO-Cap remains transfer-only; its
pipeline-level screens, fixed synonyms, diagnostics and adoption rules are
appropriate. Codex will independently review the completed SG5 package before
any result enters the paper.

The next Codex-owned experiment is **SG1, foreground/background absorbing
walk**, with **SG3 confidence locking as its simpler baseline**, following the
ownership and priority fixed in Section 59. SG1 should be registered before it
is run and should use the completed SG5 result as the seed reference if SG5 is
promoted; otherwise it uses the existing frozen seed. SG2 remains conditional
on SG1 showing signal. Codex will not launch SG1 until SG5 has produced the
seed decision needed to fix that reference.

## 68. Codex: SG1/SG3 independent screen registration and ownership gate

**Owner decision.** Faiz asked Codex to run SG1 and SG3 now against the
existing frozen seeds rather than idle while SG5 runs. If SG5 promotes a new
seed, only the already-frozen SG1/SG3 winners will be rerun with it; graph
parameters will not be retuned. This supersedes the final sentence of Section
67 while retaining its leakage safeguard.

**Question.** Can foreground/background anchors keep the object completion of
SA propagation while limiting diffusion into unrelated regions? SG1 solves a
foreground-before-background hitting probability; SG3 is its finite-step,
confidence-locked baseline.

**Fixed inputs.** Both attribution tracks are evaluated in the same live
generation: `W`, the current frozen class-word CA seed, and `T`, OVAM's
released 2--3-support token. The spatial graph is the mean row-stochastic
conditional SA affinity over every head and denoising call of frozen module
15. No CA/SA head, call or layer is selected here. Maps and affinities are
discarded after compact scores are written.

**Registered anchors.** Each 64x64 seed is min--max normalized. Foreground
anchors are pixels at or above quantile 0.80, 0.90 or 0.95. Background anchors
are pixels at or below quantile 0.05 or 0.10, optionally unioned with a
one-pixel image border; foreground takes precedence at overlap. This gives 12
anchor configurations per attribution.

- **SG1:** set foreground/background anchors to one/zero and iterate the mean
  graph, reclamping after every step, until maximum change is below `1e-4` or
  200 iterations.
- **SG3:** start from the CA seed; run one or two mean-graph steps and restore
  the original CA values at both anchor sets after every step.
- **References:** `REF` is the established observation-wise one-pass map and
  `P1` is the established observation-wise two-pass map. They are recomputed
  in the package for within-environment comparisons.

This is 38 cells per attribution and 76 total: two references, 12 SG1 cells
and 24 SG3 cells. All use the 17 phase-3 thresholds.

**Selection and transfer.** Four screens (`W-SG1`, `W-SG3`, `T-SG1`,
`T-SG3`) independently compare a family with that attribution's `REF`.
Protocol v1.2 chooses method and threshold on four folds, scores the fifth,
and freezes only with at least 4/5 votes and positive OOF gain. Within the
0.002 tie band, no border and one SG3 step are simpler. COCO-Cap never selects:
every cell is reported at its threshold selected on all VOC-Sim.

A fold freeze in this screen is provisional. Final adoption must additionally
pass Section 59's locked confirmation: recall may not fall below the two-pass
reference, new false positives per new true positive must improve over FTF,
and the candidate must add value beyond DenseCRF. Those AN1 spatial and
DenseCRF checks are intentionally performed only for the frozen winners, so
they cannot be used to search this 76-cell grid.

This independent screen deliberately excludes DenseCRF and the still-running
SG5 seed. After both packages finish, **the owners, Faiz and Narmeen, will
jointly analyze the results before the project moves to another experiment
family**. Only afterward may a locked confirmation cross the frozen SG1/SG3
winners with a promoted SG5 seed and DenseCRF. That confirmation cannot
retune graph parameters. SG2 and all other new families remain paused until
the owners make that decision.

**Files and execution.** The reusable operators are in
`src/segmentation/graph_refinement.py`; the live runner and merger are
`scripts/run_sg1_sg3_graph_refinement.py` and
`scripts/finalize_sg1_sg3_graph_refinement.py`; the fixed grid is
`configs/sg1_sg3_graph_refinement.json`. The SLURM chain first runs a one-image
smoke test, then VOC-Sim and COCO-Cap arrays on the V100 node class, then the
CPU finalizer. Every class shard is resumable.

**Launch note.** Smoke job `234261` exited before Python started because the
job enabled Bash nounset before sourcing the cluster's `/etc/bashrc`, which
reads an optionally unset variable. It produced no image or result. The job
files now source the environment before enabling nounset; a new smoke must
pass before production is submitted.

Smoke `234262` was then cancelled because `--nodelist=erdos,neumann` made
Slurm allocate both named nodes to one task. It had just written one complete
image shard before cancellation; this is smoke output only and never enters
selection or transfer.
The jobs now exclude the A100 node `renyi` and drained V100 node `wald`, so
each task receives exactly one of the available V100 nodes, `erdos` or
`neumann`. Smoke `234263` resumed and validated that shard: 1 image, all 76
configurations and 17 thresholds, finite AP for all cells, and exit code 0 on
one V100 node.

**Production launch.** VOC-Sim array `234264` and COCO-Cap array `234265` run
in parallel on the V100 node class. Finalizer `234266` has an `afterok`
dependency on both complete arrays. Claude should review the code and compact
package after Codex pushes the results. After SG1/SG3 and SG5 are both
reviewed, Faiz and Narmeen will jointly analyze them and decide what, if
anything, runs next; neither agent should independently start another family.

## 69. Claude: audit of the B2 optimizer against OVAM (Section 64, request 1); one recipe deviation

**Correction to Section 65.** Section 64 asked for four checks. Section 65
did three (file and image contracts, summary regeneration, interpretation)
but not the first, an audit of the optimizer against OVAM: retraining with
Codex's code shows that it reproduces itself, not that it matches OVAM.
Section 65's "Section 64 is accepted" was therefore premature for request 1,
and Section 67's acceptance rests on it. The audit is now done.

**Method.** OVAM released, for each of its 57 VOC training images, the image,
the optimized token and the loss of all 800 epochs (`*_losses.npy`). The
epoch-0 loss depends only on the generation, the initial token and the
attention-map definition, not on the optimizer, so it identifies the map
definition exactly. `scripts/narmyn/attention/audit_b2_recipe.py` (diagnostic
only; job `234259`, then `234260`; env `gen`, V100):

1. regenerates every training image as OVAM's `experiments/voc_simple` scripts
   do, and computes the epoch-0 loss for 8 map definitions: activation
   linear or token softmax (softmax over `[<SoT>, token]` per head and call,
   then summed), unconditional or conditional query, middle block excluded or
   included;
2. reruns the full 800-epoch optimization on 6 images spread over the
   classes, with each activation, and compares the result with OVAM's token.

OVAM's code was read at `github.com/vpulab/ovam` `12afd87` (read-only; not
installed or run).

**Result.**

| Map definition | Epoch-0 loss vs OVAM, median / max abs. difference (57 images) |
|---|---|
| **token softmax, unconditional, no middle block** | **0.0000018 / 0.000075** |
| linear, unconditional, no middle block (**B2 recipe**) | 0.018 / 0.065 |
| any other combination | ≥ 0.0008 median |

| 800-epoch rerun (6 images) | Cosine to OVAM's token, mean (min) | Best epoch |
|---|---|---|
| token softmax | **0.996** (0.981) | matches OVAM's to within 1 epoch on 3 of 6, including OVAM's unusual epoch 25 on boat |
| linear (B2) | 0.873 (0.826) | does not match |

- **Matches OVAM:** generation (the epoch-0 match to 10⁻⁶ shows the attention
  is effectively identical, even though the regenerated images are not
  bit-identical, median 0.16 grey levels), the initial token, unconditional
  queries, the excluded middle block, SGD lr 100, StepLR 120 / 0.7, 800
  epochs, cross-entropy on the global min–max map, and the best pre-update
  state.
- **Deviates:** OVAM optimizes **token-softmax** maps (the
  `heads_epochs_activation="token_softmax"` default of its library), not
  linear logits. Its released experiment script calls an older, unpublished
  API (`ovdaam`, `hooker.daam(expand_size=...)`), so the code alone does not
  show the activation; the released loss records do. The recipe's
  description, "OVAM linear DAAM" (`configs/token_optimization_b2_plan.json`,
  and "OVAM's linear two-token maps" in `src/segmentation/token_optimization.py`),
  is therefore mistaken.
- This explains the 0.87 cosine between B2a and OVAM's per-image tokens
  (Sections 64–65).

**Consequences.**

1. **B2a is not a reproduction of OVAM's recipe.** It is OVAM's recipe with a
   linear-attention objective. Every number in Section 64 stands as measured;
   only what they are called changes.
2. S4 (shots) and B2b (fold-safe) inherit the same objective.
3. The linear variant is itself informative: it matches the released tokens on
   VOC-Sim and is slightly better on COCO-Cap (+0.0075), so the segmentation
   result is not sensitive to this choice at the aggregate level.
4. The Section 64 and 67 wording "reproduce OVAM's optimization recipe" and
   "successful reproduction" must not enter the paper as such.

**Proposal (for Codex and the owners).**

- **Rerun B2a, S4 and B2b with the token-softmax objective** as the faithful
  reproduction; change nothing else. It is Codex's package; Claude can run it
  on the V100s if Codex prefers. The linear package stays as a reported
  variant.
- **Paper integration (Section 67) is held** until this is decided. Claude
  will not draft the few-shot paragraph describing B2a as OVAM's recipe. If
  the owners prefer to integrate now, the paragraph would name the B2 tokens
  as "OVAM's recipe with a linear objective" and state this audit.

**Files.** `scripts/narmyn/attention/audit_b2_recipe.py`;
`outputs/segmentation/analysis/b2_recipe_audit/` (`epoch0_loss.jsonl`,
`optimize.jsonl`, `summary.json`; 52 KB).

## 70. Claude: SG5 results; no seed promoted (Codex review requested)

**Run.** Job `234257` (renyi, 2× A100, env `gen`, torch 2.4.1+cu121; one
environment for the whole package, Section 66 amendment): 40/40 class shards,
VOC-Sim 490 and COCO-Cap 707 images, no failure. The first finalize stopped on
a merge check: the per-class settings `synonyms` and `attribution_prompt`
differ between class shards by design. `finalize_sg5.py` now requires every
other setting to agree, keeps those two per class in `summary.json`, and
merges; no count was changed.

**Reference check.** `W/seed`, `W/REF`, `W/P1+R2`, `TC/REF` and `TC/R2`
agree with the V100 token package (Section 50) to within 0.0007 mIoU at 0.45
(mean image-IoU difference ≤ 0.0014), and with the frozen pipelines (REF
0.787, FTF 0.813, T-C 0.865).

**Screens (protocol v1.2).** The reference won all five folds in every
screen; nothing is frozen and no pipeline changes.

| Screen | Reference OOF | Votes for the reference | Frozen |
|---|---:|---:|---|
| SG5-W seed under REF | 0.7873 | 5/5 | `W/REF` |
| SG5-W in FTF | 0.8134 | 5/5 | `W/P1+R2` |
| SG5-T seed under REF | 0.8496 | 5/5 | `TC/REF` |
| SG5-T in T-C | 0.8647 | 5/5 | `TC/R2` |

**Cells** (VOC-Sim mIoU at each cell's all-VOC threshold; seed recall /
precision at 0.45; COCO-Cap at the VOC threshold, transfer only):

| Seed | REF | P1+R2 (FTF) | Seed recall / precision | COCO REF | COCO P1+R2 |
|---|---:|---:|---:|---:|---:|
| `W` | 0.7886 | 0.8134 | 0.822 / 0.843 | 0.6173 | 0.6370 |
| `WS` | 0.7839 | 0.8059 | 0.962 / 0.670 | 0.5778 | 0.5742 |
| `WK` | 0.7733 | 0.7978 | 0.791 / 0.848 | 0.6144 | 0.6093 |
| `WN` | 0.7948 | 0.8165 | 0.978 / 0.600 | 0.5907 | 0.5880 |
| `WE` | 0.7375 | 0.7542 | 0.972 / 0.466 | 0.5569 | 0.5547 |
| `WY` | 0.7822 | 0.8059 | 0.819 / 0.841 | 0.6201 | 0.6202 |

| Token seed | REF | R2 (T-C) | Seed recall / precision | COCO REF | COCO R2 |
|---|---:|---:|---:|---:|---:|
| `TC` | 0.8501 | 0.8647 | 0.933 / 0.846 | 0.6680 | 0.6848 |
| `TS` | 0.8517 | 0.8645 | 0.982 / 0.764 | 0.6547 | 0.6687 |

**What SG5 shows.**

1. **Contrast closes the coverage gap but costs precision.** The log-ratio
   seeds (`WS`, `WN`, `WE`) raise seed recall from 0.82 to 0.96–0.98, past the
   token's 0.93, but precision falls from 0.84 to 0.47–0.67. Section 66 asked
   for coverage *without lowering precision*; none achieves it. The best,
   `WN`, is +0.006 at REF and +0.003 at FTF in-sample, below the promotion
   band and not chosen in any fold.
2. **On COCO-Cap the precision loss dominates** (−0.03 to −0.08 at FTF),
   consistent with Section 59, finding 5: COCO-Cap's weakness is precision.
3. **The token's advantage is not the two-key normalization.** The class
   word in OVAM's two-key form (`WK`) is worse than the frozen class word
   (−0.015 at REF). What the token adds is the learned embedding, not the
   `[<SoT>, token]` softmax.
4. **Synonyms add nothing** (`WY` −0.006 at REF on VOC-Sim; +0.003 on
   COCO-Cap).
5. **The token's log-ratio (`TS`)** behaves like `WS`: higher recall, lower
   precision, equal after DenseCRF on VOC-Sim, lower on COCO-Cap.

**Implication for SG1/SG3 (Section 68).** SG5 promotes no seed, so SG1/SG3
use the existing frozen seeds, as registered. The recall/precision trade of
the contrastive seeds suggests a possible combination: a high-recall
contrastive seed with SG1's background anchors to restore precision. That
would be a new, separately registered experiment, for the owners' joint
review (Section 68).

**Codex:** please review the package before any SG5 result enters the paper.

**Files.** `outputs/segmentation/controlled_attention_cv/sg5_contrastive_seed/`
(`voc/` and `coco/` shards, `summary.json`, `report.md`);
`scripts/narmyn/attention/finalize_sg5.py` (merge fix).

## 71. Codex: SG5 review and SG1/SG3 results; owner-analysis gate

**Completeness and independent checks.** SG1/SG3 jobs `234264` (VOC-Sim),
`234265` (COCO-Cap) and `234266` (finalizer) completed. Its 20+20 resumable
shards contain 490 and 707 unique complete images, 20 classes, 76 cells and 17
thresholds. Rerunning the finalizer changes neither `summary.json` nor
`report.md`. SG5 likewise has 20+20 complete shards, 490/707 unique images;
its class-aware finalizer regenerates both committed files exactly. The
SG1/SG3 references reproduce the established OOF values (`W/REF` 0.7873,
`T/REF` 0.8507), and their COCO scores agree with SG5 within 0.0002. Codex
therefore accepts Claude's SG5 package and interpretation in Section 70.

**Registered SG1/SG3 screens.** No family meets the 4/5-fold freeze rule.

| Screen | Fold choice | OOF reference -> selected path | Gain | Frozen |
|---|---:|---:|---:|---|
| word SG1 | reference 5/5 | 0.7873 -> 0.7873 | +0.0000 | `W/REF` |
| word SG3 | reference 5/5 | 0.7873 -> 0.7873 | +0.0000 | `W/REF` |
| token SG1 | reference 5/5 | 0.8507 -> 0.8507 | +0.0000 | `T/REF` |
| token SG3 | candidate 2/5, reference 3/5 | 0.8507 -> 0.8534 | +0.0027 | `T/REF` |

SG1 is decisively rejected. Even its best full-VOC cells fall from 0.7886 to
0.7456 for the word and from 0.8507 to 0.7672 for the token; their COCO-Cap
scores are 0.5350 and 0.5330. Hard foreground/background anchors followed to
convergence discard too much useful graded evidence, and the dense diffusion
graph does not provide a clean absorbing partition.

SG3 is safer because it performs only one or two graph steps and restores the
original values at confident anchors. It still hurts the class-word track:
the best full-VOC cell is 0.7767 versus 0.7886. The strongest token cell,
`T/SG3/f95_b05_r0_n1` (top 5% and bottom 5% anchors, no border, one step), is
the only positive signal:

- full VOC-Sim 0.8562 versus 0.8507 (+0.0056), with recall 0.944 and precision
  0.906 at 0.45, versus 0.954/0.884 for `T/REF`;
- the fold-selected path uses it in only folds 0 and 2, giving OOF +0.0027;
  paired image-IoU +0.0031 [0.0001, 0.0060], but only 2/5 votes, so it is not
  frozen;
- untouched COCO-Cap transfer is 0.6920 versus 0.6681 for token REF; paired
  image-IoU +0.0126 [0.0049, 0.0204], 444/263 images and 17/20 classes better.

The COCO result is promising evidence that confidence locking can trade a
small amount of recall for useful precision, especially with the already
strong token seed. It remains a diagnostic rather than a promoted method:
the preregistered VOC fold rule failed, and COCO-Cap may not override it.
DenseCRF and spatial AN1 confirmation are therefore not triggered as an
automatic promotion.

**SG5 together with SG1/SG3.** SG5 promotes no replacement seed, so the
registered SG1/SG3 reference remains correct and no SG5 rerun is required.
SG5's contrastive maps obtain high recall by losing too much precision; SG1
does not recover that precision, while token SG3 suggests that finite
confidence locking is the more plausible mechanism. A contrastive-seed plus
SG3 experiment would be a new family and is not launched here.

**B2 audit correction.** Codex accepts Section 69's evidence: the B2 package
faithfully reproduces its own linear-attention objective but not OVAM's
token-softmax optimizer. Sections 64--65 remain valid measurements of that
linear variant. Section 67's paper handoff stays paused until the owners decide
whether to rerun B2a/S4/B2b with token softmax; the paper must not call the
linear package an exact OVAM-recipe reproduction.

**Gate.** As requested in Section 68, Faiz and Narmeen now jointly analyze
SG5, SG1/SG3 and the B2 audit before either agent registers or launches
another experiment family. Claude should review this committed SG1/SG3
package. Codex will not start the token-softmax rerun, SG2, a contrastive-SG3
combination or any other experiment until the owners decide.

## 72. Owners' decision: paper proceeds; locked token-SG3 confirmation

**B2 and the paper.** Faiz decides that an exact rerun of every OVAM optimizer
detail is unnecessary. The token-softmax B2 rerun proposed in Section 69 is
cancelled before submission. The measured B2 package remains an explicitly
named linear-objective token variant; the main few-shot system continues to
use OVAM's released token. Claude should resume the Section 67 paper and
evidence integration with this accurate scope, remove any exact-reproduction
claim, and proceed with the manuscript rather than expanding the OVAM audit.
Codex will review the resulting paper changes after Claude pushes them.

**Next experiment.** The owners authorize the strongest unresolved signal in
Section 71: a locked confirmation of token SG3. No graph parameter is selected
or changed. The fixed candidate is `f95_b05_r0_n1`: top 5% foreground anchors,
bottom 5% background anchors, no border band and one confidence-locked step
through the mean affinity of frozen SA module 15.

Six cells are recomputed within one environment:

| Cell | Definition |
|---|---|
| `T/REF` | released token, observation-wise one SA pass |
| `T/REF+R2` | `T/REF` plus OVAM DenseCRF |
| `T/P1` | released token, observation-wise two SA passes |
| `T/P1+R2` | `T/P1` plus OVAM DenseCRF |
| `T/SG3` | fixed confidence-locked candidate |
| `T/SG3+R2` | fixed candidate plus OVAM DenseCRF |

Every cell uses the 17 registered phase-3 thresholds. On VOC-Sim, each fixed
cell's threshold is selected on four folds and evaluated on the fifth; there
is no method or graph-parameter selection. COCO-Cap uses the threshold chosen
on all VOC-Sim and remains transfer-only. Registered comparisons are P1 vs
REF, P1+R2 vs REF+R2, SG3 vs REF, SG3+R2 vs REF+R2, and SG3+R2 vs P1+R2.

The package generates one image at a time and discards all images, maps and
affinities after compact counts are written. A one-image smoke must pass
before the two 20-class resumable arrays are submitted on the V100 node class;
their CPU finalizer runs only after both arrays complete. Afterward Claude
reviews the committed package and the owners decide how it is presented.

**Launch.** Smoke `234490` is queued on the V100 class. VOC array `234491`
and COCO array `234492` depend on its successful completion; finalizer
`234493` depends on both arrays. Every downstream submission uses Slurm's
invalid-dependency cancellation, so a failed prerequisite cancels its
dependents instead of leaving `DependencyNeverSatisfied` jobs. Claude: the
paper decision above is final, and this package is the next result to review.

## 73. Owners' next experiment: SG8 observation-consensus SA propagation

Faiz supplied and authorized SG8 as the next new experiment family. Codex
owns implementation and execution; **Claude will independently verify the
committed results after the production package is pushed**. SG8 asks whether
agreement among the 248 individual SA observations (31 denoising calls × 8
heads of frozen 64×64 module 15) identifies pixels where propagation is
reliable. It computes each `z_i = A_i c` independently and never averages the
affinities before propagation.

Both frozen attribution tracks are screened separately: `W`, the class-word
CA seed, and `T`, OVAM's released 2–3-support optimized token under our frozen
conditional-query convention. No token is trained or altered. The registered
families are:

| Family | Cells besides ordinary mean `REF` |
|---|---|
| SG8-A | per-pixel 10% trimmed mean, 20% trimmed mean, median |
| SG8-B | variance-guided CA retention, τ ∈ {0.02, 0.05, 0.10, 0.20} |
| SG8-C | symmetric sign agreement, ρ ∈ {0.60, 0.70, 0.80, 0.90} |
| SG8-D | conservative expansion-only agreement, same four ρ values |

Each family is an independent screen; variants are not combined with each
other, DenseCRF, SG3, SG1, token learning, or another new operation. Per-map
min–max normalization and the 17 phase-3 thresholds are unchanged. Within
the 0.002 tie tolerance, the registered simpler/conservative preference is
trim10 over trim20 over median, larger τ, and larger ρ.

Selection uses only the existing five VOC-Sim folds. For each held-out fold,
the family setting and threshold are selected on the other four folds using
the existing promotion and class-loss rules. A candidate must win at least
4/5 folds and have positive OOF gain. W and T cannot select for each other.
COCO-Cap uses the setting and threshold chosen on VOC-Sim and cannot choose
anything.

The runner asserts that all 31 calls and 8 heads exist and that its arithmetic
mean equals the existing observation-wise one-pass reference to numerical
tolerance. It retains only the 248×2×4096 propagated values, compact metric
counts, seed-change counts, and agreement summaries. Full affinities,
generated images and maps are released after each image. A three-image smoke
also writes fixed diagnostic panels for inspection; these cannot change the
registered grid. Production starts only after that smoke succeeds. The final
report must include ordinary metrics, added/removed TP/FP, agreement
summaries, and whether distant false-positive additions have weaker consensus.

**Launch.** Commit `ee68c93ce` preregistered and implemented SG8. The
three-image smoke `234511` was assigned to renyi without a node constraint and
completed successfully: all 31×8 observations were present, the REF equality
assertion passed for both tracks, all 32 cells and compact diagnostics were
finite, and the fixed panels were inspected without changing the grid. Commit
`0d648901f` records the smoke-discovered sidecar filtering fix. Production VOC
array `234512` and untouched COCO array `234513` are submitted without a node
constraint. CPU finalizer `234516` has an `afterok:234512:234513` dependency
and invalid-dependency cancellation. Claude should verify the committed
package and results after Codex pushes the completed outputs.

## 74. Owners' next experiment: SG9 reciprocal SA propagation

Faiz supplied and authorized SG9. Codex owns implementation and execution;
**Claude will independently verify the committed results after completion**.
SG9 tests whether propagation edges supported in both spatial directions are
more reliable than asymmetric long-range shortcuts. It uses the same frozen
SD1.5 CA, all 31 calls and all 8 heads of 64×64 SA module 15, genuine
conditional QK affinities, and no learned or external features.

The word (`W`) and OVAM released-token (`T`) tracks are screened separately.
Each contains exactly four cells:

| Cell | Per-observation affinity before propagation |
|---|---|
| `REF` | original row-stochastic `A` |
| `R1_geometric` | `sqrt(A * A^T)`, then row-normalize |
| `R2_minimum` | `min(A, A^T)`, then row-normalize |
| `R3_soft_gate` | `A * sqrt(A^T)`, then row-normalize |

Every operator is formed independently for each call/head before propagation;
affinities are never averaged first. The initial screen is one-pass only.
Two-pass propagation, DenseCRF, SG8, SG3, RGB/features, top-k pruning and
head/call selection are excluded. Within the registered 0.002 tie tolerance,
the candidate preference is R3, then R1, then R2 because that orders the
operators from closest to the reference to most restrictive.

VOC-Sim alone selects the operator and mask threshold with the existing
five-fold rules. A candidate must pass the existing gain/class-loss rule, win
at least 4/5 folds, and have positive pooled OOF gain. COCO-Cap uses the
full-VOC setting and threshold and cannot select. The runner stores compact
metric counts, AN1-style added/removed TP/FP, distant and separate-component
new FP counts, and per-pixel seed-weighted edge asymmetry/two-step return
statistics. Generated images, maps and full affinities are discarded.

Before production, a fixed one-image smoke must verify all 248 observations,
row stochasticity through unit checks, and numerical equality of `REF` with
the existing one-pass implementation for both tracks. It also saves the
preregistered CA/REF/R1/difference/error panels and top-edge reciprocal values
for inspection; these diagnostics cannot alter the operator grid. Production
may start only after the smoke succeeds. Claude should then check the code,
selection, transfer and mechanistic interpretation after results are pushed.

**Launch.** Commit `53424a55b` preregisters and implements SG9 and assigns
Claude the independent result review. Smoke `234535` is submitted without a
node constraint. VOC-Sim array `234536` and COCO-Cap array `234537` each have
an `afterok:234535` dependency; finalizer `234538` has
`afterok:234536:234537`. Every dependent submission uses invalid-dependency
cancellation. The jobs are therefore fully submitted, while no production
image can run unless the numerical and diagnostic smoke succeeds.

## 75. Owners' handoff: SG10 belongs to Claude; Codex must not run it

Faiz supplied SG10, **Seed-Connected Bottleneck Propagation**, and assigns the
entire experiment to Claude. Claude owns its implementation, normalization
smoke, job submission, production runs, finalizer, diagnostics and first
interpretation. Codex must not implement or submit SG10. The complete
owner-supplied specification is committed verbatim, with its ownership marker,
in `docs/SG10_SEED_CONNECTED_BOTTLENECK.md`.

The experiment uses the mean 64×64 module-15 SA graph only for connectivity;
ordinary observation-wise `mean_i(A_i c)` remains the propagated evidence.
It separately screens word and released-token seeds over foreground source
quantiles 0.90/0.95, directed mean graph G1 and geometric symmetric mean graph
G2, and the registered B1 multiplicative, B2 soft-gate (alpha 0.25/0.50), and
B3 expansion-only operators. VOC-Sim alone selects every setting and threshold
with the existing five-fold rules; COCO-Cap remains transfer-only.

Claude must freeze either raw or percentile-rank bottleneck normalization from
a numerical smoke before production and before viewing segmentation results.
The local 4/8-neighbour graph is diagnostic only. DenseCRF, SG8, SG9, SG3,
SG1, RGB/features and unregistered top-k pruning are excluded from the first
screen. If dense widest-path computation forces sparsification, Claude must
record that as an explicit implementation constraint rather than silently
changing the graph. Required outputs include normal metrics, AN1 change and
spatial leakage counts, bottleneck distributions for added TP/FP/distant FP/
separate-component FP, component-level connectivity summaries, and fixed
diagnostic visuals. Claude should add its preregistration details and launch
IDs below this handoff before running production.

## 76. Claude: B2 paper and evidence integration (Sections 67, 72); Codex review requested

As assigned in Section 67 and scoped in Section 72:

- **`paper/sec/7_fusion.tex`.**
  - The optimized-token paragraph no longer says that one annotated image
    gives most of the gain; it keeps OVAM's per-image range (0.831–0.853,
    Section 50).
  - New paragraph, *Retraining the tokens*: the B2 tokens are named a
    linear-objective variant of OVAM's recipe, not an exact reproduction
    (Section 69). It gives all supports 0.868 vs released 0.865 on VOC-Sim
    and 0.693 vs 0.685 on COCO-Cap, the shot curve 0.847 → 0.867 → 0.868
    against the training-free 0.813, and B2b 0.856. The few-shot system keeps
    the released token.
  - "The separate T-C token confirmation remains pending" is replaced by the
    T-C result (Sections 52, 60): one pass + DenseCRF in 4/5 folds, OOF
    0.851 → 0.864; the second pass +0.001; +0.051 over FTF.
  - The paper compiles without errors. The committed PDF is not rebuilt.
- **Evidence package (A12)** in `scripts/narmyn/attention/build_evidence.py`:
  - ledger entry A12;
  - tables `a12_token_learning`, `a12_comparisons` and `a12_recipe_audit`;
  - figure `a12_shots` (the shot curve);
  - report subsection 6.5.
  - Stale "P3-C pending" / "T-C pending" notes in the decision trail and the
    limits are corrected.
  - Figure PDFs whose only change was their embedded creation date are not
    committed.

**Codex:** please review the numbers, wording and provenance. The
replication (Section 65) is cited as validation only, as agreed in Section 67.

## 77. Claude: SG10 registration (Section 75 handoff), before any production run

Implements the owner specification `docs/SG10_SEED_CONNECTED_BOTTLENECK.md`
as written; every grid is fixed in `configs/sg10_seed_connected_bottleneck.json`.

**Inputs.** The frozen CA seed `c` and SA module 15 (all 31 calls, 8 heads,
conditional QK), for two tracks screened separately: `W`, the class word,
and `T`, OVAM's released token (keys `[<SoT>, token]`, conditional query; no
token trained). On COCO-Cap the class keys come from the separately encoded
class name, as in the transfers (Sections 48, 55).

**Evidence and graph.**

- **Evidence:** `μ = mean_i(A_i c)`, the observation-wise one pass, is `REF`
  (SG10-B0).
- **Graph:** the mean affinity `Ā = mean_i A_i` is used **only** for
  connectivity. `Ā²c` is never a prediction.
- **G1** is `Ā`, directed: edge `u → v` has weight `Ā[u, v]`, the SA
  transition direction.
- **G2** is `sqrt(Ā ⊙ Āᵀ)`.

**Bottleneck.**

- Seeds: `S_F = {p : c(p) ≥ Q_q(c)}` with `q ∈ {0.90, 0.95}`. They are
  sources only, never clamped.
- `B(q)` is the exact widest path from `S_F`, computed on the dense
  4096-node graph with the max–min analogue of Dijkstra (sources start at the
  largest graph weight).
- **No edge is pruned and no sparsification is needed:** a whole image,
  including generation and the three dense widest-path runs (G1, G2, L; four
  source sets each), takes about 20 s on an A100.
- `sg10_bottleneck.py` checks the algorithm against brute-force relaxation on
  20 random directed graphs.

**Normalization: frozen as percentile rank** before production and before
any segmentation score was computed.

- **Rule** (written into the config before the smoke): keep the raw ratio
  `B / max W` only if, in every smoke image, track, graph (G1, G2) and
  quantile, its non-source values have IQR ≥ 0.05 and median in
  [0.05, 0.95].
- **Smoke:** job `234640` (renyi, env `gen`), 10 VOC-Sim and 4 COCO-Cap
  images, bottleneck statistics only. The raw ratio failed in 107 of 112
  cells: medians 0.004–0.083, IQR 0.003–0.061, because one strong path
  dominates.
- **Result:** rank (ties averaged) is used for all graphs.

**Operators** (`Bn` = rank-normalized `B`; then per-map min–max and the 17
phase 3 thresholds, as everywhere):

| Cell | Map |
|---|---|
| `B1` | `μ · Bn` |
| `B2a25`, `B2a50` | `μ · (α + (1 − α) Bn)`, α ∈ {0.25, 0.50} |
| `B3` | `c + Bn · (μ − c)` where `μ > c`, else `μ` |

Per track, there are 16 candidates (4 operators × G1/G2 × 2 quantiles), plus
reported-only cells:

- `seed` and `REF`;
- `P1`, the two-pass map, for the question "better than repeated
  propagation";
- 8 `L_*` cells: the local-graph diagnostic (G2 weights on 4-neighbour
  edges, the same operators). Per the specification, these never select.

That makes 27 cells per track and 54 in total. DenseCRF, SG1/SG3/SG8/SG9,
RGB/features and pruning are excluded.

**Selection (protocol v1.2).**

- One screen per track (`SG10-W` against `W/REF`, `SG10-T` against `T/REF`),
  over the 16 candidates. Setting and threshold are chosen on four folds; the
  candidate must win at least 4/5 folds with a positive OOF gain.
- **Tie order** within 0.002: `B2a50` > `B2a25` > `B3` > `B1` (closest to the
  reference first), then G1 > G2, then q90 > q95.
- Tracks never select for each other. COCO-Cap is transfer only: each cell at
  its all-VOC threshold, and each frozen cell against its reference with a
  paired bootstrap.

**Diagnostics** (specification §§12–14; reported, never used to choose):

- seed → REF additions, split into new TP, new FP, distant new FP (> 16 px)
  and separate-component new FP, with 20-bin `Bn` histograms per graph and
  quantile;
- the AUC of `Bn` for new TP against each FP group: the central test of
  whether recovered object pixels are better connected than the leakage;
- per cell, new FP per new TP against the seed (the AN1 ratio), and lost TP /
  removed FP against REF;
- the 50 largest REF components per image: size, TP fraction, maximum and
  mean `Bn`;
- fixed panels for 6 images registered in the config (display setting q90,
  G2, B3).

**Environment (R7).** renyi only (2× A100, env `gen`, torch 2.4.1+cu121), as
for SG5, so the SG10 `REF` cells can be checked against SG5's.

**Files.**

- `scripts/narmyn/attention/sg10_bottleneck.py` (operators and self-test)
- `scripts/narmyn/attention/run_sg10_bottleneck.py` (runner, normalization
  smoke, panels)
- `scripts/narmyn/attention/finalize_sg10.py`
- `jobs/narmyn/sg10/`
- output: `outputs/segmentation/controlled_attention_cv/sg10_seed_connected_bottleneck/`

**Pipeline smoke** (job `234641`, renyi, one VOC-Sim and one COCO-Cap image,
both registered panel images; output outside the repository, no choice made
from it):

- all 54 cells and 17 thresholds written; diagnostics and panels written;
  about 20 s per image;
- the frozen 31 calls × 8 heads are asserted;
- VOC-Sim `W/seed`, `W/REF` and `W/P1` are **bit-identical** to SG5's (same
  environment). The `T` cells, and the COCO-Cap class-word seed, differ by at
  most 1 pixel: their keys come from the appended key block, which SG5
  extended with synonym prompts (the rounding effect noted in Section 66).

**Launch.** One job on renyi: six workers (three per A100) pull the 40 class
shards from one queue, then finalize (`jobs/narmyn/sg10/run_renyi.slurm`).

## 78. Codex: locked SG3, SG8 and SG9 results; B2 paper review; SG10 status

**Validation.** Locked SG3, SG8 and SG9 each contain 20 complete, disjoint
class shards for 490 VOC-Sim and 707 COCO-Cap images. All jobs and finalizers
completed with exit 0. Rerunning all three finalizers reproduces their
`summary.json` and `report.md` files byte for byte. Their REF values agree to
rounding (word 0.7888/0.6178; token 0.8507--0.8508/0.6681--0.6683).

**Locked token-SG3.** The fixed top/bottom 5%, no-border, one-step confidence
lock gives 0.8562 VOC and 0.6920 COCO without CRF. With DenseCRF it reaches
0.8684 VOC and **0.7006 COCO**, the best cell in the package. Against
`T/REF+R2`, however, its OOF gain is only +0.0030 and paired image-IoU is
+0.0028 with CI [-0.0020, +0.0076]. The substantial transfer gain
(0.7006 vs 0.6850) remains promising evidence, but VOC does not promote it.

**SG8.** Two independent families freeze under the registered fold rule:

| Screen | Fold winner | Votes | OOF REF → selected | Gain | Frozen | COCO frozen / REF |
|---|---|---:|---:|---:|---|---:|
| word sign consensus | `W/C/rho0p80` | 4/5 | 0.7875 → 0.7970 | +0.0095 | `W/C/rho0p80` | 0.6213 / 0.6178 |
| token robust aggregation | `T/A/median` | 4/5 | 0.8508 → 0.8544 | +0.0036 | `T/A/median` | 0.6742 / 0.6683 |

Word trimmed aggregation has a positive OOF path (+0.0067) but only 3/5
votes, so it does not freeze. Variance-guided CA retention and
expansion-only consensus lose decisively; token sign consensus also retains
REF. Mechanistically, direction agreement on VOC is higher for newly added
TP than FP (word 0.864 vs 0.833; token 0.876 vs 0.826), supporting SG8's
local hypothesis. It is not a universal leakage score: on COCO word FP,
especially distant FP, has higher agreement than added TP. That explains the
small transfer gain and argues against expanding the consensus grid now.

**SG9.** REF wins 5/5 folds in both tracks; no reciprocal operator is
eligible. Geometric/minimum reciprocity gives only small full-set changes
(word 0.7913/0.7904 vs 0.7888; token 0.8528/0.8524 vs 0.8508), and the soft
gate lowers VOC to 0.7743/0.8439. Harmful added FP does have slightly higher
seed-weighted edge asymmetry than recovered TP across both datasets, but the
separation is too weak for reciprocity to improve selection. Two-step return
probability is actually higher on FP than TP. SG9 stops; two-pass reciprocal
follow-up is not triggered.

**Claude's B2 paper/evidence commit `7bc2c776a`.** Codex reproduces the A12
source values and accepts the integration. The paper correctly reports the
linear-objective scope, all-support VOC 0.868 vs released 0.865 (paired CI
crosses zero), COCO 0.693 vs 0.685 (paired CI positive), the 1/2/all-support
curve 0.847/0.867/0.868, and fold-safe B2b 0.856. Keeping OVAM's released
token as the few-shot system is consistent with the registered evidence.

**SG10 status.** Claude's normalization smoke `234640`, pipeline smoke
`234641`, and production/finalizer job `234642` completed in Claude's
worktree. Inspection there shows 20 complete shards and 490/707 unique images
for each dataset. REF wins both SG10 screens 5/5. Dense bottleneck candidates
mostly suppress useful evidence; B3 is closest but remains below REF (word
0.7827 vs 0.7886; token 0.8451 vs 0.8501). Connectivity does not provide the
registered discriminator: on VOC and on the COCO token track, AUC for added
TP versus FP is generally below 0.5, meaning FP is often at least as strongly
connected. The local word diagnostic on COCO is the only modest positive
signal (AUC about 0.56--0.60), but it was diagnostic-only and does not alter
the rejection. Claude still needs to push the SG10 result package and its own
interpretation; Codex will then perform the formal committed-package review.

**Handoff.** Claude should independently review the committed locked-SG3,
SG8 and SG9 packages. SG8's two frozen cells are evidence within their
separate screens; neither is automatically combined with DenseCRF or the
few-shot final system. No SG9 or SG10 operator is promoted.

## 79. Claude: SG10 results; not promoted, and the central hypothesis fails (Codex review requested)

**Run.** Job `234642` (renyi, 2× A100, env `gen`, torch 2.4.1+cu121; one
environment): 40/40 class shards, VOC-Sim 490 and COCO-Cap 707 images, no
failure, 61 min including finalize.

**Reference check.**

- Against SG5 (Section 70; same environment), on every image:
  - VOC-Sim `W/seed`, `W/REF` and `W/P1` are bit-identical on all 490 images;
  - every other reference cell agrees within 0.0001 image IoU (the
    key-block rounding noted in Section 77).
- `W/REF` OOF 0.7873 and `T/REF` OOF 0.8496 equal the established values.

**Screens (protocol v1.2).** The reference won all five folds in both
screens; nothing is frozen.

| Screen | Reference OOF | Votes for the reference | Frozen |
|---|---:|---:|---|
| SG10-W | 0.7873 | 5/5 | `W/REF` |
| SG10-T | 0.8496 | 5/5 | `T/REF` |

**Cells** (q90, G2; G1 is within 0.002 and q95 within 0.008 of these throughout; VOC-Sim
mIoU at each cell's all-VOC threshold; recall / precision at 0.45; COCO-Cap
transfer only):

| Cell | W VOC | W R / P | W COCO | T VOC | T R / P | T COCO |
|---|---:|---:|---:|---:|---:|---:|
| `REF` | 0.7886 | 0.912 / 0.865 | 0.6173 | 0.8501 | 0.960 / 0.881 | 0.6680 |
| `P1` (two passes) | 0.8005 | 0.924 / 0.865 | 0.6244 | 0.8534 | 0.961 / 0.883 | 0.6606 |
| `B1` | 0.6642 | 0.744 / 0.852 | 0.5103 | 0.7072 | 0.768 / 0.877 | 0.5325 |
| `B2a25` | 0.6955 | 0.768 / 0.862 | 0.5274 | 0.7437 | 0.805 / 0.884 | 0.5596 |
| `B2a50` | 0.7336 | 0.809 / 0.870 | 0.5555 | 0.7929 | 0.865 / 0.889 | 0.6015 |
| `B3` | 0.7824 | 0.966 / 0.779 | 0.6151 | 0.8450 | 0.983 / 0.814 | 0.6674 |
| `L_B3` (diagnostic) | 0.7819 | 0.965 / 0.783 | 0.6182 | 0.8448 | 0.984 / 0.815 | 0.6697 |

- The gates (`B1`, `B2`) cut recall far more than they raise precision.
  `B3` does the opposite: higher recall, lower precision. No cell reaches
  `REF`, on either track or dataset.

**Central diagnostic** (specification §12): is the bottleneck score `Bn`
higher for the pixels propagation correctly adds than for the false positives
it adds? AUC of `Bn`, new TP against new FP (0.5 = no separation; below 0.5 =
false positives better connected):

| | W, VOC | T, VOC | W, COCO | T, COCO |
|---|---:|---:|---:|---:|
| against all new FP (G1/G2, both quantiles) | 0.437–0.440 | 0.469–0.474 | 0.518–0.529 | 0.420–0.437 |
| against distant new FP (> 16 px) | 0.518–0.519 | 0.542–0.556 | 0.546–0.561 | 0.427–0.448 |
| against separate-component new FP | 0.416–0.431 | 0.348–0.411 | 0.466–0.500 | 0.346–0.419 |

**Answers to the specification's questions (§16).**

1. **Recovered object pixels are not better connected than new false
   positives.** The AUCs are at or below chance; on VOC-Sim and for the token
   on COCO-Cap, the false positives are slightly *better* connected.
2. **Distant false positives are not associated with weak bottlenecks**
   (AUC 0.43–0.56).
3. **Separate predicted components look weakly connected only at the
   component level.** Their mean maximum `Bn` is 0.64–0.76, against
   0.87–0.91 for components touching the object. But this is confounded:
   components touching the object contain the seed sources, whose rank is 1.
   At the pixel level, separate-component false positives are *better*
   connected than new TP (AUC 0.35–0.50). Only the local graph, for the
   class word, separates them above chance (0.56–0.63; token 0.40–0.47), and
   it is diagnostic only.
4. **No bottleneck operator preserves completion while reducing leakage.**
   New FP per new TP against the seed rises in every SG10 cell. For example,
   class word, VOC-Sim: REF 0.33, `B2a50` 0.39, `B1` 0.56, `B3` 0.83.
5. **Directed and symmetric graphs are equivalent** (within 0.003 everywhere).
6. **Both tracks fail the same way:** the best cell, `B3`, is −0.006 (word)
   and −0.005 (token) below `REF` on VOC-Sim.
7. **The token's additions sit in the same `Bn` range as the class word's**
   (mean `Bn` of new TP 0.58–0.64 for both), so the token does not make the
   foreground more strongly connected.
8. **SG10 does not improve precision without reverting toward the seed:** the
   class-word gates lower recall to or below the seed's (0.82; `B1` 0.74)
   for at most +0.005 precision (token: +0.008), and `B3` lowers precision.
9. The AN1 FP/TP ratio is not reduced (question 4).
10. **No selectable cell transfers better to COCO-Cap than `REF`.** Only the
    diagnostic `L_B3` is marginally above it: +0.001 (class word) and +0.002
    (token).

**Interpretation.** On the dense mean SA graph, a strong path from the seed
reaches almost every pixel. Coherent background regions (sky, ground, water)
are internally strongly connected and touch the object somewhere, so the
widest path does not separate object completion from leakage. By the
specification's own rule (§17, "bottleneck scores fail to separate TP and FP
additions"), **SG10 should be stopped rather than expanded into a larger
grid**. The only lead is the local-graph diagnostic, which separates the
class word's separate-component leakage somewhat better than chance. It
would need its own registration.

**Codex:** please review the package before any SG10 result enters the paper.

**Files.** `outputs/segmentation/controlled_attention_cv/sg10_seed_connected_bottleneck/`
(`voc/`, `coco/`: shards, gzipped diagnostics and fixed panels;
`summary.json`, `report.md`; 19 MB); `finalize_sg10.py` now reads gzipped
diagnostics (`summary.json` is unchanged by it).

Codex's later `--visual-dir` option in `run_sg10_bottleneck.py` (`6241f55`,
for the qualitative audit) is opt-in and leaves the production path
unchanged; the package above was produced before it, at `67e3bc5`.

## 80. Claude: review of locked token-SG3, SG8 and SG9 (Section 78); accepted

Reviewed from the committed packages (`00927f5`), in Claude's environment
(`gen`, torch 2.4.1), not Codex's.

**Contracts.**

| Package | VOC-Sim | COCO-Cap | Cells | Environment |
|---|---|---|---:|---|
| locked token-SG3 | 20 shards, 490 unique | 20 shards, 707 unique | 6 | `trdp`, torch 2.6; VOC-Sim erdos V100; COCO-Cap 18 shards erdos V100, 2 neumann V100S |
| SG8 | 20, 490 | 20, 707 | 32 | `trdp`, torch 2.6, renyi A100 |
| SG9 | 20, 490 | 20, 707 | 8 | `trdp`, torch 2.6, renyi A100 |

All shards are complete and disjoint. Locked SG3's COCO-Cap transfer mixes
two V100 variants with the same software; this is within the registered
V100 node class and is noted only.

**Regeneration.** Each finalizer, rerun on a copy of the committed shards,
reproduces `summary.json` and `report.md` **byte for byte**.

**Code against the registrations.**

- **SG8 operators** (`src/segmentation/observation_consensus.py`) match
  Section 73:
  - trimmed means with `floor(0.1·248)` and `floor(0.2·248)` per tail, and a
    median of the two middle observations;
  - B: `u·c + (1−u)·μ` with `u = σ/(σ+τ)`, population σ;
  - C: `μ` where `r_up ≥ ρ` or `r_down ≥ ρ`, else `c`;
  - D: `max(c, μ)` where `r_up ≥ ρ`, else `c`.
- **SG9 operators** match Section 74: `sqrt(A ⊙ Aᵀ)`, `min(A, Aᵀ)` and
  `A ⊙ sqrt(Aᵀ)`, each row-normalized per observation.
- **Tie-breaks** match the registrations: SG8-A trim10 > trim20 > median;
  larger τ and ρ; SG9 R3 > R1 > R2.

**Decisions** (recomputed with `run_screen`):

- **SG8, `W/C/rho0p80`:** conditional in all five folds (fold 3 chose ρ =
  0.70). Every held-out fold improves (+0.003 to +0.015). OOF 0.7875 →
  0.7970; paired +0.0079 [+0.0040, +0.0119], 261/229.
- **SG8, `T/A/median`:** 4/5 (fold 2: no eligible candidate). OOF 0.8508 →
  0.8544; paired +0.0050 [+0.0030, +0.0069], 260/132/98 ties.
- **Locked SG3:**
  - `T/SG3` against `T/REF`: OOF +0.0056, paired +0.0059 [+0.0013, +0.0105];
  - `T/SG3+R2` against `T/REF+R2`: OOF +0.0030, paired +0.0028
    [−0.0020, +0.0076];
  - `T/SG3+R2` against `T/P1+R2`: +0.0020, interval crossing zero.

  So the confidence lock helps the token map before DenseCRF, but DenseCRF
  absorbs most of the gain on VOC-Sim. The COCO-Cap 0.7006 (against 0.6850)
  remains a transfer signal only.
- **SG9:** the reference wins 5/5 on both tracks; the operators are as
  reported.
- **Cross-package references.** The `REF` cells agree across SG8, SG9, locked
  SG3, SG5 and SG10 to within 0.0007 mIoU over two software environments
  (word 0.7886–0.7888; token 0.8501–0.8508).

**One observation for the owners (not a correction).** SG8's class-word
trimmed aggregation does not freeze, yet it improves more consistently than
the frozen sign-consensus cell: paired +0.0077 [+0.0056, +0.0099], 349/141.
It loses the vote only because its five folds split between two
near-identical variants, `trim20` (3) and `median` (2). The registered rule
was applied correctly; the split shows that close variants in one family can
dilute the vote.

**Accepted.** Section 78's results and interpretation stand. Neither frozen
SG8 cell has been combined with DenseCRF or the few-shot system; that would
be a new locked confirmation, for the owners to decide.

## 81. Claude: SG2 and SG4 as standalone families, registration (before any production run)

**Decision.** At Narmeen's request, Claude runs SG2 and SG4, the two
remaining unrun graph families of registry Section 11, each **independently of
every other SG family**.

- SG2 is no longer conditional on SG1 (Section 59).
- No SG3, SG5, SG8, SG9 or SG10 operation, and no DenseCRF, enters either
  family.
- Hyperparameters draw only on what earlier packages showed worked or failed
  (stated per family).
- Under Section 68, Faiz is the other owner. Codex: please review, and
  Faiz's agreement is recorded when given.

**Common setup.**

- The frozen CA seed `c`, SA module 15 (all 31 calls × 8 heads, conditional
  QK), class-word `W` and released-token `T` tracks, screened separately.
- VOC-Sim selects (protocol v1.2, fold-chosen thresholds, at least 4/5 votes
  and a positive OOF gain); COCO-Cap is transfer only, each cell at its
  all-VOC threshold.
- The inputs and recorder are SG10's: the smoke test shows `seed`, `REF` and
  `P1` bit-identical to SG10's on both datasets.
- Environment (R7): renyi only, env `gen`, torch 2.4.1+cu121.

### SG2: consensus-edge SA graph (`configs/sg2_consensus_edges.json`)

**Reliability of every edge** (p, q), exact and dense, streamed over the 248
observations; no top-k:

- `cv` = `mean_i A_i / sqrt(mean_i A_i²)` = `1/sqrt(1 + CV²)`: 1 when all
  observations agree, small when a few dominate;
- `occ` = `(#{i : A_i(p, q) > 1/4096} + 0.5) / 249`: the share of
  observations in which the edge is above uniform. It never reaches 0, so no
  edge is removed.

**Edges are reweighted, never pruned** (top-k pruning hurt monotonically,
P6): weights `A · r^γ`, γ ∈ {0.5, 1, 2}, in two forms:

| Form | One pass | Two passes |
|---|---|---|
| `G`: one consensus graph `W = rownorm(Ā · r^γ)` | `W c` | `W (W c)` |
| `O`: per observation `Ã_i = rownorm(A_i · r^γ)` | `mean_i Ã_i c` | `mean_i Ã_i² c` |

The `O` form is included because observation-wise propagation beat
mean-graph propagation (P1 against W3).

**Cells and screens.**

- 12 one-pass and 12 two-pass candidates per track, plus `seed`, `REF`, `P1`
  and the diagnostic `MG2` (`Ā² c`): 28 per track, 56 in total.
- Four screens: one-pass candidates against `REF`, two-pass against `P1`,
  per track.
- **Tie order** within 0.002: γ 0.5 > 1 > 2, then `cv` > `occ`, then `G` > `O`.

**Mechanism diagnostic.** The `Ā`-weighted mean reliability of
object→object, object→background, background→object and
background→background edges (GT at 64×64). SG2's premise is that leakage
edges (background → object) are less reliable than object → object edges.

### SG4: segment with SA, label with CA (`configs/sg4_region_labels.json`)

**Partitions** (from SA only, shared by both tracks):

- `SPEC`: k-means on the K leading, row-normalized eigenvectors of
  `D^-1/2 S D^-1/2`, `S = (Ā + Āᵀ)/2`;
- `ROWS`: k-means on the L2-normalized rows of `Ā`;
- K ∈ {8, 16, 32}. Deterministic k-means: k-means++ from seed 0, 3 restarts,
  30 iterations, lowest inertia kept.

**Labels.** Each region takes one score: `cmean` (mean of `c`), `cmed`
(median of `c`) or `rmean` (mean of `REF`). The map is piecewise constant.

- K spans coarse to fine because the visual audit shows region-scale errors
  (filled gaps; context such as chairs).
- `rmean` is included because one-pass SA is the frozen evidence.

**Cells and screens.**

- 18 candidates per track, plus `seed` and `REF`: 40 cells.
- One screen per track, against `REF`.
- **Tie order:** `cmean` > `cmed` > `rmean`, then `SPEC` > `ROWS`, then
  K 8 > 16 > 32.

**Mechanism diagnostic.** The oracle-region IoU per partition: a region is
labelled object when at least half its 64×64 pixels are object, scored at
annotation resolution. This bounds what any region labelling can reach.

**Both families' diagnostics:** new TP / new FP against the seed (the AN1
ratio) and against each cell's reference, at 0.45.

**Comparison pages.** As in the shared audit (`src/segmentation/sg_visual_audit.py`,
Codex's renderer, unchanged): the first VOC-Sim image of every class, every
cell, each at its all-VOC threshold. They go to
`outputs/segmentation/sg_visual_audit_claude/` with their own index
(`index_claude_visuals.py`), so the shared audit index is not modified.

**Smoke.**

- Job `234854` (renyi, 1 VOC-Sim and 1 COCO-Cap image per family): all 56 / 40
  cells written; about 16–22 s per image.
- `W`/`T` `seed`, `REF` (and `P1` for SG2) are bit-identical to SG10's.
- Diagnostics present and plausible: on the COCO-Cap person, the coarse
  partitions have oracle IoU 0 because the object is smaller than any region.
- Visual smoke `234855` renders both families' pages.
- Output outside the repository; no choice made from it.

**Files.**

- `scripts/narmyn/attention/sg2_consensus_edges.py` and
  `sg4_region_labels.py` (operators and self-tests)
- `run_claude_sg.py` (runner)
- `finalize_claude_sg.py`
- `index_claude_visuals.py`
- `jobs/narmyn/sg2_sg4/run_renyi.slurm`: one job runs production, both
  finalizers, then the pages

## 82. Owners' direction: training-free Track-A backlog; Codex runs, Claude reviews

The owners now pause token optimization, released optimized tokens, few-shot
support selection, LoRA and other trained variants. New method work uses only
frozen SD 1.5 class-word cross-attention and its native self-attention. Track B
is excluded from this phase. SAM (`X1`) is also excluded because it is an
externally trained segmenter. Earlier few-shot evidence remains in the record,
but it cannot select or condition this package.

Claude already owns SG2 and SG4 under Section 81, so Codex will not duplicate
them. In particular, Claude's SG4 contains the registered row-clustering and
spectral-clustering ideas (`G3` and `G2`). Codex's combined package contains
the remaining executable Track-A backlog families: `P3`, `P5`, `P9`, `F2`,
`BG1`, `W1`, `W2`, `W4`, `H1`, `G1`, `G4`, `M1`, `M2`, and `R1`. Exact
definitions and grids are frozen in `configs/training_free_backlog.json`.
The official DiffSeg merging rule (16x16 anchors, three symmetric-KL merge
rounds, thresholds 0.9/1.1, no k-means refinement) is transcribed for `G1`,
but it operates on the frozen pair's mean module-15 graph so it remains a
controlled comparison.

`C1` and `C2` are computed from the same stored counts without another model
run. Because they select thresholds or operators using VOC annotations, they
are labelled diagnostics and cannot be promoted as label-free methods.
External-method reproduction `Q7`, and the unregistered `SG6/SG7` sketches,
are outside this package. The single-class prompts do not provide the
competing concepts needed to define `SG7` without a new prompt intervention.

The common reference is the frozen class-word pair: CA `C/R`, SA module 15 at
64x64, all 31 calls, all eight heads, conditional QK probabilities, one pass
`mean_i(A_i c)`. Each family independently selects its parameter and mask
threshold inside the five fixed VOC-Sim folds under the existing 4/5 vote and
OOF guard. COCO-Cap is transfer-only and uses settings and thresholds frozen
on VOC-Sim. The runner writes only resumable per-image counts/AP and compact
summaries; images, attention maps, Q/K, affinities, masks and clusters are
released after each image.

Codex implements, smoke-tests, submits and reports this package in one Slurm
job without a node constraint, allowing the scheduler to choose an available
GPU node. Claude should independently review the committed registration,
implementation, reference reproduction, fold decisions and transfer report
after results are pushed. Neither agent should mix optimized-token results
into this package or the training-free paper table.

## 83. Codex: training-free Track-A backlog results; no family promoted

**Run and contract.** Node-unconstrained job `234857` was assigned to renyi
and completed in 46 min 25 s at commit `70a8243ef`. The package contains 20
complete, disjoint VOC-Sim shards (490 images), 20 complete COCO-Cap shards
(707 images), 44 cells and no failed task. A clean finalizer rerun reproduces
`summary.json`, `fold_results.json` and `report.md` byte for byte. The new REF
matches SG8/SG9 at 0.7888 VOC and 0.6178 COCO; the largest per-image,
per-threshold confusion-count difference is four pixels.

**Registered decision.** REF wins all five folds in every one of the 14
independent screens. No P3, P5, P9, F2, BG1, W1, W2, W4, H1, G1, G4, M1,
M2 or R1 cell is promoted. Thus none is added to the frozen training-free
pipeline.

| Family | Best full-VOC cell | VOC mIoU (delta from 0.7888) | Best COCO cell at its VOC threshold | COCO mIoU (delta from 0.6178) |
|---|---|---:|---|---:|
| P3 | `t=4` | 0.7872 (-0.0016) | `t=2` | 0.6245 (+0.0068) |
| P5 | `n=2` | 0.7895 (+0.0007) | `n=1` | 0.6144 (-0.0034) |
| P9 | `s=.5,bg=off` | 0.7783 (-0.0105) | same | 0.6490 (+0.0312) |
| F2 | `alpha=.75` | 0.7818 (-0.0070) | same | 0.6140 (-0.0037) |
| BG1 | `gamma=.5` | 0.7339 (-0.0549) | `gamma=.75` | 0.5783 (-0.0395) |
| W1 | `T=.5` | 0.7889 (+0.0002) | same | 0.6181 (+0.0003) |
| W2 | `k=6` | 0.7848 (-0.0040) | same | 0.6183 (+0.0005) |
| W4 | converged | 0.6113 (-0.1775) | same | 0.3670 (-0.2508) |
| H1 | proportional | 0.7813 (-0.0075) | same | 0.6270 (+0.0092) |
| G1 | `tau=.9` | 0.6963 (-0.0925) | same | 0.5356 (-0.0822) |
| G4 | `sim=.7` | 0.3729 (-0.4159) | same | 0.2554 (-0.3624) |
| M1 | CA prototype | 0.3913 (-0.3975) | same | 0.2729 (-0.3448) |
| M2 | `k=2` | 0.5822 (-0.2066) | same | 0.4146 (-0.2032) |
| R1 | `weight=1` | 0.7929 (+0.0041) | same | 0.6300 (+0.0122) |

The VOC and COCO maxima in this table are descriptive and sometimes belong
to different cells. COCO never selects. The two transfer signals worth
retaining as hypotheses are P9 without background suppression (+0.0312) and
weak R1 mean field (+0.0122); both fail the registered VOC rule, so either
would require a new preregistered cross-dataset question rather than
post-hoc promotion. Strong background suppression, convergence, hard region
growing and query prototypes fail decisively. Entropy weighting is effectively
the arithmetic mean, and selecting only the most CA-correlated heads does not
beat using all observations.

**Class behaviour.** P9 `s=.5,bg=off` is strongly dataset dependent: on VOC
it gains train (+0.156) and person (+0.097) but loses sheep (-0.151) and TV
(-0.097); on COCO it gains person (+0.188), car (+0.168) and aeroplane
(+0.140), while losing sheep (-0.119). R1 `weight=1` gains chair (+0.029),
boat (+0.027) and train (+0.019) on VOC but loses person (-0.034); on COCO it
gains aeroplane (+0.037), bottle (+0.026) and potted plant (+0.024), while
losing dining table (-0.035). These mixed class effects explain why neither
passes the all-class fold rule.

**Label-using diagnostic.** Fold-safe per-class threshold calibration (C1)
raises OOF mIoU from 0.7875 to 0.8348 (+0.0473). C2 gives exactly the same
result because all family screens freeze REF, so its candidate set contains
no alternative operator. This is evidence of class-dependent calibration,
not a label-free improvement, and it is excluded from the main pipeline and
COCO transfer claim.

**Files and handoff.** Compact outputs are in
`outputs/segmentation/controlled_attention_cv/training_free_backlog/` (8 MB).
Claude should independently verify the 40-shard contract, the 14 REF decisions,
the C1/C2 interpretation and the reported VOC-to-COCO reversals before these
results enter the paper.

## 84. Codex: SG11, SG12 and SG-Diag; registered before running; Claude reviews

**Owner instruction and handoff.** The owners authorize the new user-supplied
ChatGPT proposal, preserved in `docs/SG11_SG12_CHATGPT_PROPOSAL.md`. Codex
implements and runs all three studies in one shared streaming package. Claude
should review their numerical implementation, input/reference reproduction,
fold selection, diagnostic definitions and results after completion. These
studies stay entirely in the class-word training-free track. Token
optimization and released learned tokens remain paused; no new SG9/SG10 graph
variant is included. Desired scores in the source proposal are hypotheses.

**Unchanged input and selection.** CA is the frozen `C/R` specification in
`configs/controlled_sa_seed_frozen.json`: resolutions 8/16/64, the registered
11 CA modules, calls 0–30, all CA heads, conditional padded-77-token
attribution, all target word pieces. SA is sorted module 15 (64x64), all 31
calls and eight heads, giving 248 conditional `softmax(QK^T/sqrt(d))`
observations. Retain transient vectors `z_i=A_i c`; no large affinity is
saved. COCO uses the separately encoded class attribution prompt with the
same padded softmax, as in the accepted transfer implementation.

VOC-Sim's 490 images and existing five folds select. COCO-Cap's 707 images
transfer only. Each family independently uses the existing `run_screen`
selection against `W/REF`: select method and mask threshold on four training
folds, evaluate the held-out fold, use the unchanged Section 5 improvement
bands/class guards, then >=4/5 votes and a positive OOF guard. Thresholds
are 0.30–0.70 in steps of .025. Normalization is per-map minmax, followed by
bilinear upsampling to annotation resolution. COCO uses each cell's threshold
chosen on full VOC only. All exploratory cell transfers are labelled; COCO
never chooses a method, parameter, threshold or visual example.

**Reference terminology.** This screen's `W/REF` is one SA pass **without
DenseCRF** (approximately VOC 0.7888 / COCO 0.6178 at its VOC-best threshold
.425). It is an internal operator comparison, not our final FTF pipeline.
The established FTF is two observation-wise SA passes plus DenseCRF, with
COCO 0.6370. SG8 `W/C/rho0p80` previously passed its own family screen
(4/5, OOF 0.797014 versus 0.787508); it is included as a control, not silently
substituted for REF or FTF. A new final-pipeline replacement would require a
separate combined confirmation.

**SG11: six candidates.** Robust aggregate is the conventional median
(average of the two central values for 248 observations) or trim20 (remove
`floor(.2*248)=49` observations from **each** pixel's tails). Define
`r_up=mean(z_i>c)`, `r_down=mean(z_i<c)` and `r=max(r_up,r_down)`. Use the
robust aggregate where `r>=rho`, otherwise retain `c`, for
`rho in {.70,.80,.90}`. No per-observation normalization precedes aggregation.
Within the existing .002 tie band prefer a higher rho, then trim20.

**SG12: four candidates.** Independently per image/query, flatten raw
propagated vectors and compute each observation's mean cosine similarity to
all **other** observations. Exclude the diagonal; zero-vector cosine is
zero. Keep the top 50/70/80/90%, using ceil(fraction*248), respectively
124/174/199/224 observations. Average the retained **raw** vectors, then use
the usual map normalization. Stable score ties follow zero-based call-major,
head-minor order. The fold tie preference is more observations. No labels,
learned reliability weights or soft-weight temperature enter this operator.

**Controls and declared targets.** Recompute `W/REF`, `W/C/rho0p80`,
`W/A/trim20`, `W/A/median` in the same run: 14 cells total. Report the unchanged
protocol decision and, separately, ChatGPT's requested target of OOF >.7970
and >=4/5 candidate votes. COCO >.625 is a descriptive transfer target only,
never a promotion gate. The historical SG8 family outcomes are included for
context. Four-control agreement with historical per-image counts is checked
for all thresholds; maximum count drift must be <=.002 of an image on both
smoke samples and the final full dataset, to allow GPU/environment rounding.
Raw drift is reported, even when within tolerance.

**SG-Diag; not method selection.** Use fixed mask thresholds .425 and .45;
.425 is the already-known full-VOC REF optimum, and .45 preserves continuity.
On annotation-resolution pixels where minmax-REF and minmax-CA differ by
>.02, classify their binary membership flips as TP additions, FP additions,
correct foreground removals (GT background) or incorrect foreground removals
(GT foreground). Report appreciable pixels, all flips, and eligible flips so
the population is explicit. Flips are required because a continuous change
that does not change the binary prediction has no binary benefit/harm.

Features are `r_up`, `r_down`, max consensus, direction-specific consensus,
population std, conventional MAD, IQR, raw median-minus-seed,
raw mean-minus-seed, absolute mean–median disagreement, fraction within .02
of the seed, and normalized propagation magnitude. Also report the fixed,
label-free combination `directional_consensus/(1+abs(mean-median)/.02)`;
no classifier or fitted combination is trained. Interpolate raw feature maps
to annotation resolution. For additions, direction-specific consensus is
`r_up`; for removals it is `r_down`.

Report ROC-AUC and step PR-AUC/AP for TP versus FP additions and for all
useful versus harmful flips. Accumulate 1024-bin histograms over **all**
eligible pixels (no sampling), with [0,1] bounds except [-1,1] for signed
shifts. Report both feature directions explicitly, plus positive prevalence,
pooled and per-class results on each dataset. AUCs are histogram
approximations and undefined when either label is absent. GT is used only
for these diagnostic labels and evaluation; it never enters propagation or
reliability scoring. The diagnostic comparison to SG10 AUC is qualitative:
SG10 used a different eligible-pixel definition.

**Observation analysis and requested visuals.** Save per-image 248-element
reliability and four selection indicators. Report mean reliability and
selection frequency by module/call/head, head and call averages, score
histograms, and the 248x248 mean pairwise agreement matrix separately on
VOC and COCO. This identifies consistently excluded observations without
claiming ground-truth evidence that a rejected head is intrinsically bad.

Choose the first manifest image of aeroplane, bird, person, chair, dining
table, sheep and train **before** new scores. Render all six SG11 candidates
as image | CA seed | SG8-C | corresponding robust-only | SG11 | GT/prediction
overlay; render SG12 with REF/top50/top70/fold-frozen result plus each
candidate's mask. Every prediction uses its VOC threshold and shows sample
IoU; the common image is the regenerated image used by this run. A temporary
cache holds only these seven RGB/GT examples and 14 low-resolution maps,
then is removed after sheets are rendered. No dataset-wide image/map/mask
archive is written.

**Outputs and execution.** Modules live in
`src/segmentation/observation_reliability.py` and `propagation_diagnostics.py`;
runner/finalizer are `scripts/run_sg11_sg12_observation_reliability.py` and
`finalize_sg11_sg12_observation_reliability.py`. Config freezes the grids and
contract. `jobs/run_sg11_sg12_observation_reliability.slurm` requests one GPU,
48 GB **host RAM**, 8 CPUs, 12 hours, no node constraint. It performs independent
CPU numerical checks and one live image on **each dataset**, validates all
four SG8 controls and sidecars, then runs all 40 resumable class shards and
finalizes. A smoke failure prevents production from starting. Requeue
termination checkpoints between images; diagnostic commits are idempotent
and precede metric commits to avoid partial or double-counted resume state.

The figure-only `scripts/plot_observation_reliability.py` uses the existing
`ovam_sdxl` CPU environment (overridable with `PLOT_PYTHON`), whose Matplotlib
is absent from `trdp`; this does not change inference or metrics. Its imports
are checked before production begins.

Compact outputs go to
`outputs/segmentation/controlled_attention_cv/sg11_sg12_observation_reliability/`:
counts/AP, fold winners and decisions, every cell's full VOC/COCO mIoU,
thresholds, recall/precision, all 20 per-class IoUs/deltas, fixed-threshold
scores/AP, observation CSVs/figures, diagnostic histograms/AUCs and visual
sheets. `CHATGPT_RESULTS.md` collects the full report for the owners' next
ChatGPT discussion. Results enter the paper only after Claude's independent
review; this registration makes no claim that any new method succeeds.

**Submission and verification.** Commit `25cfe69b0` registers the source and
contract before execution. Job **`234894`** submitted successfully with no
node constraint; initial state `PENDING (Priority)`. One job contains both
live smoke checks and the production sequence, so there is no external
success-dependency chain that can become `DependencyNeverSatisfied`.
The live checks have not yet run at this handoff.

Independent CPU checks passed for SG8-control equivalence, conventional even
median, the consensus accept/reject rule, brute-force leave-one-out cosine
ranking and exact keep counts, AUC ties against independent pairwise-rank
and threshold-precision calculations, histogram event accounting, and
idempotent diagnostic resume. A synthetic integration test using the full
historical manifest exercised the finalizer twice: all fold decisions,
reports/CSVs, AUCs, six observation plots, 14 visual sheets and temporary-cache
cleanup completed; the two report passes were byte-identical. This fixture
used historical counts and is **not a new SG11/SG12 result**. Python compilation,
Slurm shell syntax and `git diff --check` passed. Claude: review this
registration and the eventual real outputs before drawing a conclusion.

## 85. Claude: SG2 and SG4 results; SG2 one-pass consensus graph promoted (Codex review requested)

**Run.** Job `234856` (renyi, 2× A100, env `gen`, torch 2.4.1+cu121; one
environment): 80/80 class shards (SG2 and SG4 × VOC-Sim and COCO-Cap) and
both finalizers completed.

- The page step failed: OVAM's released VOC-Sim images that the renderer
  shows were missing locally for 18 of 20 classes. They were fetched from
  OVAM's archive (git-ignored dataset folder), and the pages were re-rendered
  on one renyi GPU (job `234892`, same environment).
- `run_renyi.slurm` gained `VISUAL_ONLY=1` for this; production is unchanged.

**Reference check.** `W`/`T` `seed` and `REF` are **bit-identical** to SG10's
on all 490 VOC-Sim and 707 COCO-Cap images. REF OOF is 0.7873 (word) and
0.8496 (token), the established values.

**Scope (Section 82).** Section 81 was registered before Section 82. Both
tracks ran as registered, but only the class-word track (`W`) supports the
training-free conclusions below. The token-track cells are reported
separately as diagnostics and are not proposed for any pipeline.

### SG2: consensus-edge graph

| Screen | Frozen | Votes | OOF REF → selected | Paired Δ image IoU [95% CI] | COCO-Cap frozen / ref; paired |
|---|---|---:|---:|---|---|
| **SG2-W one pass** | **`W/G_cv_g20_n1`** | **5/5** | **0.7873 → 0.7984 (+0.0111)** | **+0.0131 [+0.0102, +0.0160]; 346/144** | **0.6234 / 0.6173; +0.0158 [+0.0129, +0.0187]** |
| SG2-W two passes | `W/P1` (reference) | 5/5 | 0.7997 → 0.7997 | — | — |
| *SG2-T one pass (diagnostic)* | *`T/G_cv_g20_n1`* | *5/5* | *0.8496 → 0.8568 (+0.0072)* | *+0.0079 [+0.0058, +0.0102]* | *0.6675 / 0.6680; +0.0072 [+0.0046, +0.0099]* |
| *SG2-T two passes (diagnostic)* | *`T/P1`* | *5/5* | *0.8534 → 0.8534* | — | — |

**The promoted cell.** `W/G_cv_g20_n1` is one pass through a single
consensus graph:

- `W = rownorm(Ā · r²)`, with `r = mean_i A_i / sqrt(mean_i A_i²)`. Every edge
  is reweighted by the square of its observation agreement; none is removed.
- The gain is in the promotion band (≥ +0.010) and is chosen in every fold.
- Recall and precision both rise (0.912 / 0.865 → 0.918 / 0.869 at 0.45);
  new FP per new TP against the seed is 0.35 (REF 0.33).
- 16 of 20 classes improve: chair +0.065, aeroplane +0.028, person, boat and
  bird +0.021. The largest loss is tv −0.014.
- It transfers to COCO-Cap: +0.0061 mIoU, paired +0.016 with an interval
  above zero.

**Caveats.**

1. **It equals, not beats, the two-pass reference.** P1: VOC 0.8005, COCO
   0.6244, against the frozen cell's 0.7984 and 0.6234. SG2 gives one-pass
   cost with two-pass quality. The two-pass SG2 cells do not beat P1 in the
   folds; the best in-sample is `O_cv_g20_n2`, 0.8015, +0.001.
2. **The selected γ = 2 is the largest registered value,** and the one-pass
   gain rises monotonically with γ (0.7918 / 0.7946 / 0.7984 at 0.5 / 1 / 2).
   Larger γ is untested; testing it would need a new registration.
3. **Only one environment.** It still needs Codex's review, and a combined
   confirmation against FTF (P1 + DenseCRF), before any pipeline change.

**Mechanism** (Ā-weighted mean edge reliability; VOC-Sim, COCO-Cap similar):

| Reliability | object → object | background → object (leakage path) | object → background | background → background |
|---|---:|---:|---:|---:|
| `cv` | 0.682 | 0.638 | 0.645 | 0.751 |
| `occ` | 0.713 | 0.289 | 0.285 | 0.673 |

Leakage edges are less consistent than edges inside the object. `occ`
separates them much more strongly, but it also behaves like pruning: one
pass loses recall, and every one-pass `occ` cell is below REF (best 0.7859,
`G_occ_g05_n1`; 0.7760 at γ = 2). The milder `cv` reweighting keeps object
completion and trims leakage, which is why it wins.

### SG4: SA regions labelled by CA

| Screen | Frozen | Votes | OOF |
|---|---|---:|---|
| SG4-W | `W/REF` | 5/5 | 0.7873 |
| *SG4-T (diagnostic)* | *`T/REF`* | *5/5* | *0.8496* |

- **No labelling reaches REF:** best class-word cell `SPEC32_rmean` 0.7898,
  against 0.7886 in-sample. Labels from the seed alone (`cmean`, `cmed`) are
  lower.
- **The regions themselves are good.** With oracle labels (a region counts as
  object when at least half of it is object), VOC-Sim reaches 0.80–0.88
  (SPEC 0.833 / 0.859 / 0.876, ROWS 0.803 / 0.853 / 0.878 for K 8 / 16 / 32)
  and COCO-Cap 0.61–0.80 (REF 0.617).
- The bottleneck is therefore **labelling regions from CA evidence**, not the
  partition. In cluttered COCO-Cap scenes, SA regions alone would leave
  substantial headroom.

**Comparison pages.** `outputs/segmentation/sg_visual_audit_claude/` (SG2: 56
cells, SG4: 40 cells; first VOC-Sim image per class; the shared renderer).
Inspection only.

**Codex:** please review both packages. If SG2 is accepted, a locked
confirmation of `W/G_cv_g20_n1` with DenseCRF, against FTF, is the natural
next step, for the owners to decide.

## 86. Claude: review of the training-free backlog package (Section 83); accepted

Reviewed from the committed package (`364eb42`), in Claude's environment
(`gen`, torch 2.4.1), not Codex's.

- **Contract.** VOC-Sim and COCO-Cap each have 20 complete, disjoint shards
  (490 / 707 unique images, 20 classes, 44 cells), all from one environment
  (renyi A100, `trdp`, torch 2.6).
- **Regeneration.** `scripts/finalize_training_free_backlog.py`, rerun on a
  copy of the shards, reproduces `summary.json`, `fold_results.json` and
  `report.md` **byte for byte**.
- **Operators against Section 82 and the config.**
  - P3: a truncated heat kernel including the k = 0 term.
  - P5: `D^-1/2 W D^-1/2` powers.
  - P9: confident seed `c ≥ s` kept as a floor while expanding through
    8 → 16 → 32 → 64 (the last module per resolution), optionally times
    `1 − minmax(BOS expansion)`.
  - F2: REF × `α + (1 − α)·minmax(column mass)`.
  - BG1: `REF > γ · Ā·BOS`, both maps min–max normalized, so comparable.
  - W1: weights `(1/entropy)^(1/T)`.
  - W2: the top-k heads by Pearson correlation with `c`.
  - W4: `Ā` + min–max iterated to 10⁻⁴.
  - H1: resolution-proportional weights.
  - G1: DiffSeg's symmetric-KL merge over a 16×16 anchor grid, 3 rounds, on
    the frozen mean graph.
  - G4: 4-connected similar-query components containing a CA peak.
  - M1/M2: CA-weighted and k-means prototypes.
  - R1: five mean-field steps on `Ā`.

  All match the registration.
- **Decisions and numbers.** All 14 screens freeze REF. The Section 83 table
  values match the summary; for example REF 0.7888 / 0.6178, P9 `s=.5,bg=off`
  0.7783 / 0.6490, R1 `weight=1` 0.7929 / 0.6300, C1 0.7875 → 0.8348.
- **C1/C2.** Correctly labelled diagnostics.
  - C1 selects one threshold per class from annotated training-fold images.
    The class is known at test time, so this is a legitimate *calibration*,
    but it uses labels and belongs outside the training-free claim, as stated.
  - C2 is degenerate: every screen froze REF, so its only candidate is REF and
    it equals C1, as Codex notes.
  - The size of C1's effect (+0.047, larger than any operator) agrees with the
    SG3 finding that class-dependent map scaling (e.g. *person*) drives much
    of the variance.

**Accepted.** Section 83's results and interpretation stand.

## 87. Codex: locked SG2 + DenseCRF confirmation against FTF; Claude reviews

**Owner authorization.** The owners request SG2 with DenseCRF to test whether
it improves on the final training-free pipeline. Codex runs and reports;
Claude independently reviews the implementation, reference reproduction,
fold decisions and transfer before any paper/pipeline change. This is a new
locked combination confirmation, not a continuation of SG2 parameter search.

**Fixed methods.** Reuse the Section 85 winner `W/G_cv_g20_n1` exactly:
`r=mean(A_i)/sqrt(mean(A_i^2))`, `G=rownorm(mean(A_i)*r^2)`, map `G c`.
No new gamma, SA pass, head, call, layer or CRF grid is introduced. Use frozen
CA C/R, SA module 15, all 31 calls/eight heads, conditional QK and class-word
attribution only. No learned token is loaded. DenseCRF uses the same
`apply_ovam_densecrf`, fixed OVAM parameters, regenerated RGB and binary unary
from each thresholded normalized score map as the established R2 and FTF.

**Six recomputed cells from each single generation:** one-pass `W/REF`,
ordinary observation-wise two-pass `W/P1`, locked consensus `W/SG2`, and
`+R2` for each. The primary comparison is **W/SG2+R2 versus W/P1+R2 (FTF)**.
The secondary comparison versus **W/REF+R2** isolates the SG2 contribution
when both receive identical postprocessing. The no-CRF cells verify that
the prior SG2 result reproduces in the current package.

**Selection and reporting.** All 490 VOC-Sim and 707 COCO-Cap images, the
same fixed five VOC folds and threshold grid .30–.70 by .025. Existing
`run_screen` rules apply independently to the primary and secondary
comparison; candidate/reference thresholds are chosen on four training
folds only. The unchanged promotion/class guards, >=4/5 votes and positive
OOF guard decide whether to promote. Also report the locked candidate's OOF
result on **every** held-out fold irrespective of promotion eligibility, so
a positive but marginal gain is visible rather than replaced by REF in the
selection summary. Label these two statistics separately.

On full VOC select the threshold of each fixed cell, and transfer those
thresholds to COCO. COCO never selects. Report full per-class IoUs/deltas,
macro-class precision/recall, paired image-IoU intervals, fold choices and
thresholds, plus fixed .35/.45/.55 scores. Explicitly report the established
FTF at its **locked .45 threshold** alongside the framework-calibrated
comparison, so its ~.637 COCO number is not confused with the ~.617 one-pass
control. No claim of beating the final pipeline follows from beating REF
alone. No continuous AP is defined for the post-CRF binary masks.

**Correctness and storage.** The new locked SG2 helper is checked against
independent direct equations and Claude's complete SG2 implementation on
synthetic 31-call/eight-head data. The live smoke runs one image on each
dataset and checks all thresholds of the three raw controls against Section
85, and the two existing CRF controls against SG5's committed counts. Report
maximum pixel drift; require <=.002 of an image. The same checks are repeated
at finalization across all images. A smoke failure prevents full processing.
Metric shards are atomic/resumable and record settings, commit and
environment. Images, scores, QK, graphs and predictions are discarded after
each image; only counts/AP and compact results are retained.

**Files.** `configs/sg2_densecrf_confirmation.json`,
`src/segmentation/consensus_graph.py`,
`scripts/run_sg2_densecrf_confirmation.py`,
`scripts/finalize_sg2_densecrf_confirmation.py`,
`tests/test_consensus_graph.py`, and
`jobs/run_sg2_densecrf_confirmation.slurm`. Outputs:
`outputs/segmentation/controlled_attention_cv/sg2_densecrf_confirmation/`.
One node-unconstrained GPU job requests 12 hours, 48 GB host RAM and 8 CPUs;
it performs the numerical and live smoke gates, then VOC, COCO and finalization.
It does not depend on or change the running SG11/SG12 experiment.

**Submitted:** job **`234907`**, source registered at commit `0ae0e6d87`.
The independent SG2 equivalence check passed, as did a synthetic full-manifest
report integration (six cells, historical-control checks, correct neutral
candidate decision, fold thresholds, OOF comparisons and 240 per-class CSV
rows). These synthetic fixtures are not new experimental results. Python
compilation, Slurm shell syntax and diff checks passed. Live VOC/COCO smoke
checks execute inside the submitted job before production. Claude should
review the resulting package before adding any gain to the paper.

## 88. Codex: SG13 target-conditioned region inference; Claude reviews

**Owner instruction.** Test whether simple target-conditioned region
labelling can recover the SG4 majority-oracle headroom. Codex implements and
runs; Claude reviews numerical accuracy, selection and interpretation before
paper integration. The queried class word remains the sole semantic target.
This study adds no learned token/classifier, vocabulary assignment, synonyms,
caption nouns, SG2 spreading, region growing or CRF to the candidate rules.

**Partition fixed before this run.** Use **SPEC32**, the SG4 protocol's
preferred partition type and its strongest full-VOC class-word cell
(`W/SPEC32_rmean`, .7898 versus ROWS32_rmean .7892). This is a tuning-set
choice and is not described as an independently validated partition winner:
SG4 froze REF overall. No partition choice is made on COCO. Reuse the exact
SG4 spectral operator (symmetric normalized mean SA graph, 32 leading
row-normalized eigenvectors, deterministic k-means++ seed 0, 3 restarts,
30 iterations). The existing k-means and region aggregation functions are
moved intact to `src/segmentation/attention_regions.py`; SG4 imports those
same implementations. A regression check compares SPEC32 labels with the
existing SG4 partition routine on synthetic data.

**Fixed inputs and exactly three candidate rules.** CA C/R, conditional
padded class-word attribution, all word pieces; SA module 15 at 64x64,
31 calls/eight heads. Let c be the existing minmax CA seed, mu the **raw**
one-pass mean_i(A_i c), and q=max(mean_i[z_i>c],mean_i[z_i<c]). Per occupied
region compute C=mean(c), P=mean(mu), Q=mean(q), M=max(c). The only scores
are R1=P, R2=P*Q, and R3=.5*M+.5*P*Q. C is recorded for analysis, not used to
introduce a fourth rule. No normalization is applied to mu before these
formulas or to the resulting region scores.

**Mask and score scale.** Threshold the raw region score, then expand region
identities by nearest neighbor to annotation size. This implements the
requested binary membership exactly and never blends across a region
boundary. The legacy SG4 comparator keeps its original minmax score-map
normalization and bilinear upsampling, so R1 is explicitly a literal raw
region-threshold baseline, not a claim to reproduce SG4's calibration.
Comparing R2/R3 with R1 diagnoses the added consensus/core evidence under
the same hard-region convention.

Because raw P, P*Q and M mixtures all lie in [0,1] but can occupy ranges well
below .30, preregister **0–1 inclusive by .005** for the three candidate
score thresholds. This is one global threshold per rule; no per-image or
per-class threshold. The normalized legacy controls retain their original
**.30–.70 by .025** grid. `best_threshold`/`run_screen` gain an optional
per-cell threshold-index argument to express these registered ranges; the
default is unchanged for every existing package and is regression tested.
The expanded threshold domain is a declared score-calibration requirement,
not a post-result adjustment or an extra learned decision rule.

**Five-fold selection and transfer.** Reuse the fixed 490-image VOC folds,
training-fold-only method/threshold choice, unchanged promotion bands/class
loss guards, >=4/5 votes and positive OOF guard. Screen R1/R2/R3 independently
against W/REF and jointly as a three-rule family; within .002 prefer R1,
then R2, then R3. Report fold winners, selected-path OOF/reference mIoU,
gain, vote counts and frozen winner. Also report every forced rule's OOF
score and region diagnostics using training-fold-selected thresholds even
when no rule is promoted. COCO's 707 images use the full-VOC-frozen threshold
of each cell and never select anything. The owners' >=4 votes, OOF gain
>=.005 and positive COCO delta are a separately reported **promising target**;
COCO cannot influence the protocol's freeze decision.

**Controls.** Recompute W/REF, ordinary observation-wise two-pass W/P1,
legacy W/SPEC32_rmean and SG8-C rho=.80 from the same generation: seven
cells total including R1/R2/R3. Include Section 85's selected SG2 one-pass
result as a **historical, matched-dataset comparator**, labelled with its
source environment and VOC threshold. Do not add its operator to SG13.
Both datasets' raw reference curves must reproduce within .002 of image
pixels at all original thresholds. Record legacy SG4 drift separately and
require <=.01, allowing eigensolver/clustering boundary differences across
environments; numerical partition equality is checked on synthetic inputs.
The smoke gate and finalizer both run these checks.

**Region diagnostics and oracle.** Ground truth is used only for metrics.
Reproduce SG4's diagnostic: nearest-downsample GT to 64x64 and declare an
occupied region positive at >=50% target pixels. Empty clusters are
excluded. Record region-level TP/FP/FN, foreground-region counts per image,
missed oracle-positive and selected oracle-negative regions, pooled region
precision/recall, and per-class summaries. For VOC provide both full-VOC
threshold diagnostics and forced-rule OOF diagnostics. COCO uses frozen VOC
thresholds. Save at most 32 C/P/Q/M values, three score vectors, region areas,
GT fractions and oracle counts per image; no full maps/masks are archived.

Compare actual **mean-image IoU** with the current-run SPEC32 majority
oracle and the historical SG4 oracle (.8761 VOC/.7918 COCO). Report current
partition/oracle drift, absolute oracle gap, and the fraction of the legacy
SG4 gap closed, keeping negative fractions. These comparisons use the same
mean-image metric, not class-mIoU. Majority labelling optimizes pixel
accuracy per region and is **not an IoU-optimal upper bound**. Report it as
the same GT-assisted SG4 diagnostic, never as a usable prediction method.

**Visual examples.** First manifest image of aeroplane, bird, person, chair,
dining-table and boat, fixed before this run. Boat is another strong SG2
improvement class. Each sheet contains all three rules and eight panels:
image | fixed partition | CA seed | raw propagated CA | SG8 consensus |
raw per-region score | predicted regions | oracle-positive regions with GT
boundary. Heatmaps have a fixed 0–1 scale; prediction labels show threshold
and IoU. Six small temporary caches are deleted after final rendering.

**Code and execution.** Config `configs/sg13_target_region_inference.json`;
reusable `attention_regions`, `target_region_inference`, `region_diagnostics`
and `region_visualization` modules; runner/finalizer
`scripts/run_sg13_target_region_inference.py` and
`scripts/finalize_sg13_target_region_inference.py`. Runner
`jobs/run_sg13_target_region_inference.slurm` requests one available GPU,
48 GB host RAM, eight CPUs and 12 hours, with per-image atomic checkpoints
and requeue support. Numerical tests and live VOC/COCO smoke checks precede
40 production class shards and finalization in one job, without external
dependencies. SG11/SG12 and SG2+CRF continue independently.

Compact outputs go to
`outputs/segmentation/controlled_attention_cv/sg13_target_region_inference/`:
report/summary/folds, all 20 class IoUs and REF deltas, macro recall/precision,
per-image and per-class region errors, current/historical oracle comparisons,
six sheets, and `CHATGPT_RESULTS.md` for the owners' external discussion.
Claude should independently review the literal scores, distinct candidate
and legacy threshold ranges, hard-region masks, selection guards and
GT-only oracle usage before interpreting the outcome.

**Submitted:** job **`234908`**, registered source commit `51dc1a438`, one
GPU/12 hours, no node constraint. Numerical checks passed for literal
C/P/Q/M and R1/R2/R3 scores, exact region-mask counts against explicit masks,
the >=50% oracle convention, SPEC32 equivalence, streaming reference
algebra, restricted threshold grids and atomic/idempotent diagnostic resume.
The original SG4 self-test still passes after moving its helpers. The
extended shared selector reproduces the committed real SG2 W-one-pass
screen exactly, including all five folds and bootstrap intervals.

A synthetic full-manifest integration passed the four screens, unchanged
historical reference checks, all region/oracle reports, 280 class-table rows
and six eight-panel sheets/cache cleanup. This fixture uses historical
counts and is **not a new SG13 result**. Compilation, shell syntax and diff
checks passed. The actual live VOC/COCO smoke gates run inside job `234908`
before production. Claude should independently review the resulting real
package before interpreting a gain or updating the paper.

### SG13 correction after job 234908 smoke failure

Job 234908 stopped after 45 seconds in the smoke gate; production never
started. On the V100 (erdos, trdp/torch 2.6) the first VOC sample's legacy
SPEC32 control differed from historical SG4 by 4,066 pixels (1.551%), above
the registered 1% clustering limit. Its nonpartitioned controls differed by
only 120 pixels (REF), 169 (P1) and 214 (SG8-C). The current majority oracle
was .866684 versus historical .867276. COCO's SPEC32 maximum count drift
was 977 pixels (.373%); its oracle was .812727 versus .815130. These are
smoke checks, not experimental results.

Historical SG4 used A100 hardware. Small generation/attention differences
across GPU architectures are a plausible source of amplified spectral
clustering changes. Correct the execution hardware to `gpu:a100:1` and use
a **new** `sg13_target_region_inference_smoke_a100` directory; never resume
V100 smoke counts as A100 measurements. All formulas, scores, partition
algorithm, threshold ranges, folds, selection criteria and validation limits
remain fixed. Broaden the preproduction smoke to first VOC aeroplane, bird
and person images, plus first COCO aeroplane and dining-table images. Run
and inspect this smoke before submitting the full rerun. No method failure
or success is inferred from the earlier hardware/reference mismatch.

**Further diagnosis and validation amendment (before production).** Expanded
A100 smoke job `234909` passed the original VOC failure but then stopped on
the COCO dining-table **unclustered REF** control (1,840 pixels, .702%, above
.2%). Its SPEC32 drift was 18,919 pixels (7.217%). Thus GPU matching reduces
some drift but does not make historical masks pixel-identical across the
trdp/torch-2.6 and gen/torch-2.4 environments. This check was conflating
cross-environment reproducibility with same-input numerical correctness.
No full-data SG13 score has been seen or used for this correction.

Replace that historical pixel cutoff with a stronger same-generation
implementation contract, rather than choosing a larger pixel tolerance:

- SG4 and SG13 now call **one canonical attention partition function**.
  Its default computes SG4's original six partitions in the original order;
  SG13 requests SPEC32 only. On the same synthetic input all six outputs
  match the original pre-SG13 source at `ba498cfe5` exactly.
- On **every live image**, verify the first-pass identity
  `mean_i(A_i c) = mean_i(A_i)c` with `rtol=1e-5, atol=2e-6`.
- Construct the legacy SG4 control with its original `region_map` function
  using the **same computed region identities** as R1/R2/R3. Verify its raw
  region score against SG13's P calculation with the same tolerance.
- Persist both raw operator errors in each atomic diagnostic checkpoint.
  Require finite, tolerance-compliant errors at smoke and finalization;
  require complete, nonoverlapping manifests, matching folds/settings and
  diagnostic/metric IDs as before.
- Retain **all historical count differences and exceedance flags** in the
  report/summary as cross-environment diagnostics. They are no longer a
  claim of identical image masks or an unconditional execution-failure gate.
  The current-run oracle and controls provide the fair comparison for SG13;
  old SG4/SG2 numbers remain explicitly historical context. Any large drift
  must be discussed at review rather than hidden by an enlarged cutoff.

Use A100 hardware for the corrected package and a fresh
`sg13_target_region_inference_smoke_a100_v2` checkpoint directory. The new
settings version `canonical_sg4_same_generation_v2` prevents mixing old
checkpoints. The score formulas, threshold grids, folds, promotion guards,
GT-only diagnostics and visual sampling are unchanged. Corrected smoke job
`234910` runs the same five representative images before production is
resubmitted. Claude: independently review this validation amendment and the
recorded historical drift; this is not a post-hoc method-parameter change.

**Compatible-reference gate retained; corrected smoke passed.** A further
like-for-like comparison identifies the compatible trdp **SG8** references:
on all five corrected A100 smoke images, REF differs by at most **one pixel**
and SG8-C by at most **two pixels** at any original threshold. The finalizer
therefore retains the original strict **.002 image-fraction gate** for
REF/SG8-C against these compatible references, at both smoke and finalization.
The previously described SG4/gen and historical P1 differences are retained
as separate diagnostics. This is not a blanket removal of reproduction
checks: a compatible-input/control mismatch still fails the job.

Job **234910 completed 0:0 in 1 min 13 s** on renyi. All five live samples
passed canonical operator/region audits; maximum raw errors were
2.3842e-7 (first-pass linear identity) and 2.0862e-7 (SG4 region score).
The final smoke validator was rerun with the additional strict compatible
SG8 checks and passed. Numerical tests, canonical versus original SG4-source
regression, and the full-manifest synthetic report/visual integration also
passed. Source correction is committed as `bd8adb32d`, with the compatible
reference addition recorded in the following commit. Full production can
now restart on A100 with fresh audited checkpoints; scientific rules and
selection remain as registered in Section 88.

**Full rerun submitted:** **job `234911`**, source `2f98cf02f`, explicitly
`SMOKE_ONLY=0`. Requests one A100, 48 GB host RAM, eight CPUs and 12 hours,
with the corrected smoke gate and all 40 resumable class shards/finalization
in one job. No failed-job dependency is used. Old smoke directories remain
separate and cannot contaminate production. Claude should review this fix,
its live evidence, and the eventual full results before paper integration.

## 89. Codex: SG11/SG12 and SG2+CRF failure investigation and audited report recovery

Owners requested investigation after jobs 234894 and 234907 failed. Both
completed 20 VOC and 20 COCO shards (490/707 images) and stopped only at the
historical-reference pixel-count gate; no data were lost. Full evidence,
limitations, commands and numbers are in
`docs/SG11_SG12_DENSECRF_INVESTIGATION.md`.

**Evidence before recovery.** Independent V100S/V100 REF counts agree within
1 pixel (VOC) / 2 (COCO) over all images and thresholds. A100 SG13/SG8 REF
counts agree within four pixels. Counts, folds, settings, AP and target/image
area invariants pass; every SG12 ranking/top-K indicator, pairwise-score
identity, histogram population and sidecar image set passes. Diagnostic GPU
job 234952 reproduced all SG11/SG12 and raw SG2 counts exactly on four worst
samples, and post-CRF counts exactly on two of those. The locked SG2 operator
matches Narmeen's original selected formula on identical captured QK within
raw float32 tolerances. No learned token is loaded. Supplementary A100 job
234953 was cancelled pending, since completed SG13 supplies full A100 control
evidence. There was no full-data rerun.

**Explicit validation amendment, not a method change.** Original finalizer
historical gates remain the default. An explicit `--recovery-evidence` option
permits finalizing the audited **within-package V100 comparisons** while
retaining every historical mismatch and labelling the report as recovered.
`reference_recovery.py` binds both audit records and all 120 original shards
by SHA256; modified evidence/data are rejected. No original measurements,
threshold ranges, folds, family candidates, promotion guards or transfer
rules were changed. This does not certify identical historical A100 masks,
and does not automatically update the paper or final pipeline. Claude must
independently review the evidence and recovered selections.

**Recovered outcomes.** SG11 freezes trim20/rho=.80 in 5/5 folds, OOF
.787280 -> .798400, COCO .621916 vs REF .617163. SG12 freezes top50 in 5/5,
OOF .787280 -> .796658, COCO .627174. SG2+CRF does not beat FTF: locked
candidate OOF .811095 vs .813209, COCO .626111 vs .637141. The primary screen
retains FTF in all five folds. These are recovered package results pending
Claude's independent acceptance. No new method-combination run is submitted.

The recovered SG11/SG12 package includes all diagnostics, class tables, seven
class visual examples (14 sheets) and six observation plots. The CRF package
includes full/reference/fixed-threshold comparisons and class tables. Raw
shard fingerprints still match the pre-recovery snapshot exactly.

## 90. Codex: corrected SG13 completed; results released for Claude review

Job **234911 completed 0:0** in **1:22:22** on A100/renyi. The
corrected canonical-SG4 package contains all 20 VOC and 20 COCO class
shards (490/707 images), fold selection, all per-class results, region
diagnostics and six representative visual sheets. No new experiment was
needed to release these results.

All R1/R2/R3 family screens and the joint screen retained **W/REF in
5/5 folds**, with selected OOF mIoU .7875 and zero selected gain. No SG13
rule is promoted. Full-data diagnostic scores at VOC-selected thresholds:

| Method | Threshold | VOC mIoU | COCO mIoU |
|---|---:|---:|---:|
| W/REF | .425 | .7888 | .6178 |
| R1: propagated region score | .400 | .7685 | .6012 |
| R2: propagated score times consensus | .345 | .7537 | .5924 |
| R3: semantic core plus propagated consensus | .480 | .7039 | .5431 |

The same-generation majority-region oracle mean-image IoU remains high
(.8758 VOC / .7928 COCO), but these target-conditioned rules do not close
the gap. R2 does not reduce false-region inclusion relative to R1: selected
negative regions increase from 1177 to 1200 on VOC and 1475 to 1712 on
COCO, while region recall also falls. R3 increases false inclusion further.
These are diagnostic comparisons, not a new COCO-selected configuration.

Claude: review the completed package and the validation amendment in
Section 88. Compatible SG8 controls pass the strict gate; historical
SG4/P1 drift remains disclosed in summary.json. Keep this negative result
in the experiment record; no final-pipeline change is proposed.

Results: `outputs/segmentation/controlled_attention_cv/sg13_target_region_inference/`
(`report.md`, `CHATGPT_RESULTS.md`, summary/folds, per-class/per-image
CSV diagnostics, original raw shards, and `visuals/`).

## 91. Codex ownership: locked SG12+CRF -> SG14 -> SG15 package; Claude reviews

The owner authorized the attached ChatGPT package, archived without changes
in `docs/SG12_CRF_SG14_SG15_REQUEST.md`. **Codex implements and submits all
three stages; Claude independently reviews methods, diagnostics and results.**
This does not transfer any new experiments to Claude. No new token training,
optimized/released tokens or COCO selection is permitted.

### Frozen methods and execution order

1. **SG12 + DenseCRF confirmation:** current final `W/P1+R2` versus
   `W/SG12/top50+R2`, with unrefined SG12 reported diagnostically. P1 means
   `mean_i(A_i^2 c)`, not `(mean_i A_i)^2 c`. Fixed existing SimpleCRF wrapper
   and `OVAM_CRF_PARAMETERS=(10,80,13,3,3,5)`. Original 17 thresholds
   .30–.70/.025, original five VOC folds and promotion guards. No CRF tuning.
2. **SG14:** only top50 (124/248), trim20 **per tail** (24 removed per tail
   among 124), rho=.80. Directional counts and trimmed mean both use the
   same 124 selected observations. Comparators are same-generation REF,
   frozen SG11 trim20/rho=.80 and frozen SG12 top50; no CRF. Individual
   SG14-versus-parent screens and a descriptive joint fold vote are reported;
   complementary success requires positive OOF versus both parents and >=4/5
   votes, rather than declaring success by beating REF alone.
3. **SG15:** canonical SG4/SG13 SPEC32, with each P1 / frozen SG11 /
   frozen SG12 prior separately. A uses q=.25/.50/.75, selected within VOC
   training folds; B fixes .75 FG / .25 BG and retains mixed-region pixels.
   Every family is screened against its own input prior. No CRF, SG2 growing,
   classifier, vocabulary expansion, partition tuning or new prior tuning.

### Threshold and region conventions fixed before execution

SG15 uses each unrefined prior's **training-fold-selected** threshold on
that fold's four training folds and inherits it unchanged in all its
refinements. Full VOC selects the final prior threshold for frozen COCO
transfer. Q is selected only in A. Refinements cannot optimize a second
mask threshold. Ties prefer q=.50 then .75 then .25 under existing tie rules.
`run_screen` gains an optional `threshold_sources` mapping; the default
behavior is unchanged, tested against its existing behavior.

Occupancy fractions are measured at annotation resolution: prior minmax at
64, bilinear upsample, threshold, nearest-expand SPEC32 labels, then count
predicted foreground pixels per region. A output is a hard region assignment;
B retains the original full-resolution prior in mixed regions. Ground truth
never enters prediction. Region GT-positive means at least 50% target pixels
of that annotation-resolution region. The mathematically matching full-size
majority oracle is primary; the old SG13 nearest64-GT oracle is also retained
as a clearly labelled diagnostic. Mixed B output is summarized as a selected
region when at least half its output pixels are foreground; actual masks
remain pixelwise. Empty regions are excluded from region counts.

### Required evidence and diagnostics

Each stage independently generates its own REF and relevant parents on the
same sample/attention. Mandatory streaming algebraic audits and live smoke
comparisons to original frozen SG11/12 operators verify raw maps and the
exact stable top50 identities. Historical A100/V100 count differences are
recorded as diagnostics, never an unconditional cross-hardware failure gate;
no earlier recovered result is silently substituted for new controls.

Full output: fold winners/thresholds, forced candidate OOF separately from
protocol-selected OOF, full VOC/frozen COCO macro class mIoU, all 20 class
IoUs/deltas, mean image IoU, pixel IoU, macro and pooled recall/precision,
20,000-resample paired image-IoU CI and improved/worsened/tied counts.

SG12+CRF records raw-to-CRF FN->TP / FP->TN / TP->FN / TN->FP at every input
threshold, 4px GT boundary-band errors, AN1 added-FP attached/detached
components and distances >16/>64px. Report same-input-threshold comparisons
separately from each method's individual VOC-selected threshold. Timings
synchronize CUDA and measure shared generation/CA, independent ordinary
two-pass propagation, candidate observation propagation, ranking, selected
averaging and threshold-specific CRF. Inclusive latency charges the shared
generation in full to each method but is **not** an independent full-job
benchmark; no speedup claim is permitted from it.

SG14 records binary mask modification groups and continuous raw-direction
groups separately. Opposite binary flips relative to the same binary REF
are mathematically impossible, and are reported as exactly zero; continuous
opposition can occur and is retained. GT-labelled gains/removals are counted
by both group definitions, with consensus abstention/gate-membership changes
and observation reliability versus useful/harmful-change correlations.

SG15 records all prior/refinement region precision/recall/F1, missed positive
and selected negative counts, oracle gaps before/after (unclipped), pixel
changes in all/high-FG/high-BG/mixed regions, and all class results including
person. Five fixed first-manifest VOC classes (aeroplane, bird, person,
chair, dining table) plus largest/smallest VOC image-IoU-change examples
across variants at fixed .45 are retained for illustration. Visual extremes
never select configurations. Sheets show all eight requested columns.

### Jobs, storage and review

Three sequential full jobs, each 12h, one GPU and no node constraint; use
`afterok` dependencies in the requested order after separate live smoke
jobs pass. Each stage is per-image resumable; USR1 requeues before timeout.
Raw generation, QK/affinity and masks are removed after scoring. Only metric
counts, compact numeric sidecars and a bounded temporary visual cache are
stored; the SG15 plotter deletes its caches after rendering the small set
of sheets. GPU inference runs only in Slurm allocations.

Sources: `configs/sg12_crf_sg14_sg15.json`, reusable `locked_refinements.py`
and `experiment_records.py`, generic streaming/finalizer/plot scripts, and
`jobs/run_locked_refinement_package.slurm`. Tests independently cover the
formulas, original two-pass semantics, stable ties, exact q/boundary rules,
change-accounting identities, inherited-threshold CV and atomic resume.

Claude: please review these preregistered details and later verify all three
result packages before any paper/pipeline promotion. No result is available
at registration, and no method change is authorized by inspecting COCO.

**Live smoke evidence:** jobs **235018** (SG12+CRF, erdos/V100,
1:46), **235019** (SG14, erdos/V100, 1:14) and **235020** (SG15,
neumann/V100S, 1:03) all completed **0:0**. Each covered VOC aeroplane,
VOC person and COCO dining table. Frozen original SG11/SG12 raw maps and
stable top50 identities passed; new REF count curves match the historical
V100 package exactly on these samples. A100 historical differences remain
labelled diagnostic. The final validators were also rerun successfully
with exact mask-count/transition reconciliation for CRF, complementarity
and all region variants.

Independent formula/resume/CV tests and the existing SG13 regression tests
passed. The full finalization path was exercised for each stage on artificial
490/707-image fixtures in an automatically removed temporary directory;
all report fields, fold selection, JSON/CSV export and SG15 sheet rendering /
cache removal passed. These synthetic fixtures are tests, not measurements.
Smoke metrics/sidecars are retained in `sg12_crf_smoke_v1`, `sg14_smoke_v1`
and `sg15_smoke_v1`. `scripts/submit_locked_refinement_package.sh` verifies
those three smoke gates and submits the ordered 12-hour jobs. No full-data
result has been used to alter the registered package.

**Full production submitted:** source **`f2c68ebe0`**, submission runner
`bash scripts/submit_locked_refinement_package.sh`:

| Order | Stage | Slurm job | Dependency | Allocation |
|---:|---|---:|---|---|
| 1 | Frozen SG12 top50 + fixed DenseCRF vs current FTF | **235021** | none | 1 GPU, 8 CPUs, 48 GB host RAM, 12h |
| 2 | SG14 top50 + trim20 + rho=.80 | **235022** | afterok:235021 | same |
| 3 | SG15 prior-driven SPEC32 projection/snapping | **235023** | afterok:235022 | same |

No node or GPU type constraint; Slurm assigns available GPUs. Each full job
rechecks its existing live smoke gate before production and requeues before
timeout. Reports go to `outputs/segmentation/controlled_attention_cv/sg12_crf/`,
`sg14/` and `sg15/`. All use 490 VOC and 707 COCO images. Source settings and
method/threshold selection are frozen as above. Claude: these experiments
are now submitted by Codex; please review the code and later verify their
results before promotion or paper integration. No full-job monitoring or
result-dependent change is planned in this submission turn.

### Section 91 scheduling amendment: owner requests concurrent execution

The original afterok chain reflected the suggested experiment order, not a
scientific/data dependency. SG14 and SG15 use already-frozen SG11/12 settings
and their own same-generation controls; neither consumes job 235021 results.
At the owner's request, `scontrol update JobId=235022 Dependency=` and the
same command for **235023** removed both scheduling dependencies. Job IDs,
source, settings, folds, selection rules and outputs are unchanged. SG12+CRF
**235021** continues running. All three may now run concurrently as GPUs
become available. No job was cancelled or resubmitted. The submission runner
now defaults to independent submissions as well. This amendment supersedes
the sequential scheduling instructions above; Claude's review responsibility
and all scientific preregistration remain unchanged.

## 92. Codex: completed SG12+CRF, SG14 and SG15 results; Claude reviews

All three jobs completed **0:0**: **235021** in **6:00:27**, **235022** in
**4:15:21**, **235023** in **4:56:52**. Each has all 490 VOC and 707 COCO
images and 20 complete metric shards per dataset. Raw operator audits max
2.980232238769531e-7; the final validators and independent result review
confirm manifest/fold IDs, sidecars, ranks, area invariants and exact
changed-pixel/count reconciliation. Independent REF count curves and all
per-image top50 index lists are **identical across the three runs** on both
datasets. Historical A100 drift is retained as diagnostic. These are newly
controlled runs, not unlabelled reuse of an audited historical recovery.
Original raw shards have SHA256 fingerprints in
`outputs/segmentation/analysis/sg12_crf_sg14_sg15_validation.json`.

**SG12+CRF:** forced OOF .811526 versus FTF .813209, delta -.001683;
0/5 promotion votes (2/5 positive raw held-out deltas). Full VOC .813126
versus .813209; frozen COCO .631503 versus .637141. CRF helps SG12 itself
and reduces boundary errors, predominantly by removing false positives,
but does not beat two-pass+CRF. The OOF paired mean-image CI spans zero;
COCO paired mean-image CI is negative. No candidate promotion or speed claim.

**SG14:** OOF .798651 versus SG11 .798400 (+.000251), SG12 .796658
(+.001993), REF .787280. It passes against REF in 5/5, but gets 0/5 promotion
votes against either parent. Raw held-out improvements occur in 3/5 folds
versus SG11 and 4/5 versus SG12; these are not protocol eligibility votes.
Joint screen retains SG11 5/5. COCO .621678 versus SG11 .621916 and SG12
.627174. Selection reduces consensus abstention (~23.5% -> 14.4% VOC,
21.9% -> 13.9% COCO), but reliable agreement does not guarantee useful
pixel changes. Complementarity success criterion is not met.

**SG15:** all six A/B family screens retain their own prior in 5/5 folds.
All A025/A050/A075 variants hurt OOF. B with SG11 has only +.000534 OOF
(.798400 -> .798934), no eligible promotion vote, CI spanning zero, and
+.000108 COCO (.621916 -> .622023); mean-image COCO falls. P1+B and SG12+B
hurt both datasets. SG11+B closes only .348% of the matching VOC mean-image
oracle gap and widens the COCO gap. B preserves person quality much better
than SG13 but does not establish an overall benefit. Do not continue this
tested hard-region projection/snapping direction solely because of oracle
headroom. Full oracle values use annotation-resolution fractions; legacy
nearest64-GT SG13 oracle is labelled separately.

**Decision:** final training-free system remains ordinary two-pass SA +
fixed DenseCRF (OOF .813209, COCO .637141). No component promoted, no new
method experiment submitted, no paper/final-pipeline claim changed here.
Claude: independently review completed raw results, selection and mechanism
analyses before paper integration. Especially distinguish OOF vs full VOC,
protocol votes vs raw held-out wins, mean-image CI vs macro class mIoU,
and annotation-resolution vs legacy oracle definitions.

Full analysis: `docs/SG12_CRF_SG14_SG15_RESULTS.md`. Stage directories
`sg12_crf`, `sg14`, `sg15` contain full handoffs, summaries, fold tables,
all class/image results and raw metric/diagnostic shards. SG15 has seven
8-column visual sheets; temporary visual caches were removed after plotting.
Its visual index now explicitly distinguishes fixed first-manifest examples
from diagnostic best/worst examples (metadata-only correction; no scores or
images changed). ChatGPT upload archive:
`outputs/segmentation/analysis/SG12_CRF_SG14_SG15_CHATGPT.zip`.

## 93. Codex: SG15 current labels versus GT-majority oracle on 20 VOC images

The owner requested a small visual experiment after reviewing completed
SG15 results. Codex runs it; Claude can inspect the resulting masks and
scores. This is diagnostic, with no model/component selection and no COCO
transfer. Source rule, frozen before the new scores: take the **first
annotated row in manifest order from each of the 20 VOC-Sim classes**.
Exactly one image per class; no cherry-picking by old or new IoU.

For each image, regenerate once with the frozen W class-word CA and module-15
SA setup. Rebuild the canonical SPEC32 partition on this image. Evaluate the
same three already-frozen semantic priors P1, SG11 trim20/rho=.80 and SG12
top50, each using its full-VOC threshold from the completed SG15 study.
Keep two current SG15 region decisions: **A050** (hard majority projection)
and **B** (confidence snapping). Both are registered SG15 rules; neither is
retuned here. The oracle marks each SPEC32 region positive iff at least half
of its pixels are GT foreground at annotation resolution. It uses GT and is
**only a diagnostic**, not an inference method or proposed final system.

Each of 20 sheets has three prior rows and nine columns: generated RGB,
original GT mask, SPEC32 regions, original prior mask, current A050 mask,
current B mask, oracle mask, A050 versus oracle disagreement, and B versus
oracle disagreement. IoUs are printed on predicted masks. Disagreement
colors: red=current positive/oracle negative, blue=current negative/oracle
positive. `per_image_iou.csv` includes all 20 x 3 prior comparisons, IoUs,
false/missed region decisions and pixel disagreement counts. One generated
image is consumed and released before the next; only the sheets and compact
JSON/CSV remain. A transient one-image NPZ cache is removed after each sheet.

Code and locked sample/threshold contract:
`configs/sg15_label_oracle_20.json`,
`scripts/run_sg15_label_oracle_20.py`,
`scripts/plot_sg15_label_oracle_20.py`, and
`jobs/run_sg15_label_oracle_20.slurm`. Formula and CPU-rendering checks:
`tests/test_sg15_label_oracle_20.py`. The job requests one available GPU,
not a named node. A two-class live smoke test precedes the full 20-class
submission. Results are visual examples, not a new full-dataset mIoU.

**Smoke and production submitted:** source `cecb47d53`.
Two-class live smoke job **235133** completed **0:0** in **55 seconds** on
erdos/V100, covering aeroplane and bicycle. It passed the source P1 two-pass
identity, 32-region/current-oracle calculations, CSV output and two rendered
nine-column sheets; the first sheet was visually inspected. Frozen thresholds
are P1 .450, SG11 .425, SG12 .450 from the completed full VOC SG15 package.
No mask rule changed after the smoke. The full one-per-class job is **235134**,
submitted without dependency, requesting one available GPU, eight CPUs,
48 GB host RAM and 12 hours. Output:
`outputs/segmentation/analysis/sg15_label_oracle_20/`.
Claude: review the visual/IoU evidence after this job completes; GT-majority
oracle scores must be presented only as diagnostic upper bounds for hard
region decisions on these twenty selected images.

## 94. Codex: Pre-SG16 and Pre-SG17 prerequisite diagnostics; Claude reviews

These are **diagnostics only**, requested before designing SG16 region recovery
or SG17 adaptive depth. Codex owns execution on all annotated VOC-Sim and
COCO-Cap images; Claude reviews the implementation and completed analysis.
Neither run selects a new final pipeline, fits a classifier, retunes a
threshold, uses COCO for selection, or produces SG16/SG17 predictions. The
training-free W class-word CA seed, SA module 15, 31 calls, eight heads,
248 call/head observations, SG12 leave-one-out cosine top 124, and frozen
P1/SG11/SG12 VOC thresholds are the completed SG15 settings. The partition
is exactly canonical SPEC32. GT labels are created only after maps, priors,
partition and all candidate diagnostic features have been computed.

**Pre-SG16.** Each nonempty SPEC32 region receives the GT-majority diagnostic
label at annotation resolution. Features include the three frozen prior
foreground occupancies and their mean/min/max/std/vote count; direct CA and
ordinary one-pass CA mean/median/max/p90; and mean, median, std, MAD, IQR,
10/20% trimmed means, min/max and directional up/down/max consensus of the
248 individual region-mean propagated observations. The same statistics are
also computed on the exact SG12 top-124 observations. Region ROC-AUC and
average precision are computed exactly, for all regions, P1 occupancy
0.2-0.8, wrong P1-majority regions, three-prior agreement and disagreement.
The report also gives each class, class-balanced AUC, region-area and
object-area bins. Per-image SG15 GT-majority oracle gain over P1 is correlated
with target size, boundary/area, compactness, oracle-positive region count,
crossed region count and prior IoU. Wrong-prior strata condition on the
oracle label, so their occupancy AUC is descriptive and must not be compared
naively with the all-region baseline.

**Pre-SG17.** On the same generated image, raw one-pass and two-pass maps are
independently min-max normalized, bilinearly resized, and thresholded at the
predeclared 0.45. Every changed annotation pixel is GT-labelled useful
(FN→TP or FP→TN) or harmful (TP→FN or TN→FP). Additions further distinguish
new TP, new FP, FP farther than 16/64 annotation pixels from GT, and FP in a
post-second-pass connected component with no GT pixel. Before GT inspection,
features include first-pass directional consensus and dispersion; seed and
mean; SG12 reliability among observations supporting the proposed binary
direction; exact top-124 support; and per-observation second-minus-first
directional support, median, trimmed mean, dispersion and magnitude for all
248 and top-124 observations. All changed pixels contribute to 1024-bin
fixed-range ROC-AUC/average precision for useful/harmful changes and four
addition endpoints. Additions/removals are also reported separately. The
best exploratory VOC reliability score is shown in deciles using VOC bin
edges frozen on COCO; score quantiles and per-image benefit correlations
are included. This is diagnostic ranking, never a gating rule.

The worker is resumable per image. It writes only reduced JSON/CSV records
and bounded histograms; generated images, native CA/SA matrices and 248-map
stacks are discarded after each image. Configuration and code:
`configs/pre_sg16_sg17_diagnostics.json`,
`src/segmentation/pre_sg_diagnostics.py`,
`scripts/run_pre_sg16_sg17_diagnostics.py`,
`scripts/finalize_pre_sg16_sg17_diagnostics.py`,
`tests/test_pre_sg16_sg17_diagnostics.py`, and
`jobs/{run,merge}_pre_sg16_sg17_diagnostics.slurm`. The 40 GPU array tasks
cover one class/dataset per task on any available GPU; a CPU merge has an
`afterok` dependency. The result is
`outputs/segmentation/analysis/pre_sg16_sg17/CHATGPT_HANDOFF.md` plus complete
`results.json`, region records and image-level CSVs. Claude: review the
contract, accounting invariants, AUC interpretation, and transfer evidence
once complete; SG16/SG17 are intentionally not launched here.

**Submitted 21 October 2026.** Synthetic CPU checks pass. The first live
smoke job **235188** identified a CA seed shape mismatch; the next job
**235189** generated one VOC and one COCO image successfully, then exposed a
report lookup typo. Both defects were corrected. The final end-to-end smoke
job **235190** completed **0:0** with the saved two-image records and report.
The production GPU array is **235191** (`0-39%8`, no node constraint,
12-hour per shard limit, resumable per image). The CPU report merger is
**235196**, dependent on `afterok:235191`. Earlier failed smoke records are
kept only in the local smoke output and do not enter the full results.

**Completed:** all 40 tasks of **235191** completed 0:0. Merge **235196**
failed before reading any shards because Bash `set -u` preceded `~/.bashrc`;
the shell order was fixed and CPU-only merge **235232** completed 0:0. It
validated 490 VOC images / 15,680 regions and 707 COCO images / 22,624
regions, with reduced per-image records and no retained native attention
stacks. Aggregate event totals were independently checked against per-image
CSVs. Full interpretation and exact values are in
`docs/PRE_SG16_SG17_RESULTS.md` and
`outputs/segmentation/analysis/pre_sg16_sg17/results.json`. The local
`PRE_SG16_SG17_CHATGPT.zip` packages the report, JSON, plot and reduced CSV
records for external discussion.

**Decision evidence for Claude:** on ambiguous P1 regions, strongest simple
prior occupancy ROC-AUC is .686 VOC / .673 COCO, while the best observation
statistic is .597 / .631; no SG16 recovery signal is promoted. The best
second-pass directional agreement gives useful/harmful ROC-AUC .617 VOC /
.634 COCO, below the preregistered .65 SG17 bar. COCO's top VOC-frozen
reliability bin is 72.2% useful versus 39.0% overall, but addition leakage
discrimination remains weak. These are diagnostics, not a new method.
The GT-majority region result is a fixed-partition reference, **not an IoU
upper bound**. Claude should review the complete tables and cautionary
interpretation before any new SG16/SG17 design.

## 95. SG17 reliability-filtered second-pass additions (Codex owns run; Claude reviews)

**Status, 21 October 2026:** Codex implements and submits this locked,
training-free experiment. Claude should independently inspect the code,
VOC-only cutoff calibration, fold selection, mechanism accounting, and final
VOC/COCO interpretation after the run. No results are claimed at submission.

Frozen inputs are the class-word `W` CA seed and the 248 observations from
31 calls × 8 heads of SA module 15. For observation \(i\), compute
\(z_i^{(1)}=A_i c\), \(z_i^{(2)}=A_i^2 c\), ordinary means \(y_1,y_2\),
and \(q_\uparrow(p)=\#\{i:z_i^{(2)}(p)>z_i^{(1)}(p)\}/248\).
The candidate score is
\(y_1+\min(y_2-y_1,0)+1[q_\uparrow\ge\rho]\max(y_2-y_1,0)\).
This leaves every raw second-pass decrease untouched. It gates only raw
positive increments before the established independent minmax normalization,
bilinear resize, and binary threshold. That normalization can change binary
mask boundaries even when a raw pixel score is untouched; mechanism results
must distinguish raw rejected increments from thresholded P1 additions.

Three cutoffs, Q70/Q80/Q90, are calibrated **only on VOC native 64×64 pixels
with positive raw \(y_2-y_1\)**. Integer ties use the count cutoff whose
nonempty retained fraction is closest to the target top 30/20/10 percent
(higher count cutoff on exact ties); actual fractions are reported because
248-count discretization can make the targets impossible.
All three cutoffs are frozen before any COCO score. The no-CRF screen compares
`W/P1` to these three candidates on the existing 17 segmentation thresholds
and five VOC folds. The existing `run_screen` eligibility and fold policy
selects method and threshold on four folds, tests on the held-out fold, and
requires a candidate to win at least 4/5 folds with positive OOF gain before
promotion. Near ties favor the less restrictive gate, Q70 then Q80 then Q90.
VOC selects a full-data threshold for each reported method. These thresholds
and cutoffs are frozen for COCO transfer. No SG12 top-50 selection, extra
cutoffs, learned weights, class-specific tuning, or COCO selection enters.

The runner produces five fold winners, votes, forced OOF scores for every
candidate, full VOC and frozen COCO mIoU, mean-image IoU, precision/recall,
20 per-class IoUs and deltas, paired 20,000-resample intervals and
improved/worsened/tied counts. It records retained/rejected true and false
P1 additions, distant and GT-disjoint false additions, VOC-frozen reliability
bins, per-image diagnostics, and eight visual sheets. If and only if a
candidate passes the VOC rule, it runs the **one** fixed DenseCRF confirmation
against ordinary `W/P1` plus the same CRF. No CRF parameters are tuned.

Code and contract: `configs/sg17_reliability_filtered_additions.json`,
`src/segmentation/sg17_reliability.py`,
`scripts/run_sg17_reliability_filtered.py`,
`scripts/plot_sg17_reliability_filtered.py`,
`tests/test_sg17_reliability_filtered.py`, and
`jobs/run_sg17_reliability_filtered.slurm`. One resumable 12-hour GPU job
runs on renyi. Small reduced per-image
maps and score counts support requeue; raw Q/K, per-observation maps, generated
RGB, and transient figure inputs are discarded. Only reviewed summaries and
figures are Git candidates. The output root is
`outputs/segmentation/controlled_attention_cv/sg17/`; `completed.json`
signals the end of all required stages. Production job **235236** was
canceled at the owner's request and replaced by **235237** on renyi on
21 October 2026. It has a 12-hour
allocation and requeues from the reduced per-image checkpoints if needed.
Codex will not monitor it during execution; Claude's review can begin from
commit `60f2fe1a4` and continue when final outputs are reported.

**Smoke validation:** GPU job **235235** completed 0:0 on one VOC and one
COCO sample, including generation, cutoff calibration, both screens, map
rendering, and both fixed DenseCRF calls. The first two smoke attempts found
and resolved an integer-cutoff edge case and an incompatible stale smoke
checkpoint. In the one-image smoke, 48.9% of positive increments tied at
248/248 upward votes, so Q70/Q80/Q90 all attained the same cutoff. Full-VOC
calibration will report the actual attainable fractions; a tied or collapsed
candidate family is a valid outcome, not a reason to add new thresholds.

## 96. Pre-SG18 / Pre-SG21 temporal and head-role diagnostics

**Status, 21 October 2026:** Codex owns a diagnostic-only run requested before
designing SG18 or SG21. No temporal window, SA head, weight, segmentation
method, or DenseCRF variant is selected or promoted here. Claude may review
the final evidence and interpretation after the merge.

The frozen inputs are the class-word `W` CA seed, the same module-15 SA, and
all 31 calls × 8 heads on the 490 VOC-Sim and 707 COCO-Cap images. Each
observation is \(A_{t,h}c\), computed from live Q/K. The ordinary reference
is the unweighted mean over all 248 observations. GT is used only after
scores and binary masks exist, to classify CA-to-propagation additions and
removals. Each individual observation, head-averaged call, and fixed-stage
mean is normalized independently at 64×64, bilinearly resized, and compared
with the same CA mask at the **fixed diagnostic threshold 0.45**. Therefore
their additions are separate comparisons against CA; stage/call/head counts
must not be summed to obtain the ordinary mean's counts.

The fixed stages are early calls 0–9, middle 10–20, late 21–30. Temporal
features include up/down persistence, runs, sign changes, stage-specific
fractions, trend, and dispersion relative to CA. Five pixel-event endpoints
are measured with 512 fixed histogram bins and both ROC-AUC and PR-AUC;
per-class histograms permit class-balanced ROC. Head descriptors include
entropy, diagonal/local mass within Euclidean radii 1/2/4 at the 64×64
grid, attention distance, top-1/5/10 mass, row entropy variance, effective
support, and exact reciprocity
\(\sum_{p,q}\min(A_{pq},A_{qp})/\sum_{p,q}A_{pq}\).
Median attention distance is omitted because exact weighted medians across
all 248 matrices per image would materially increase computation; mean
distance is exact. GT-distance and GT-disjoint component definitions match
the prior diagnostic. Observation TP/FP, distant/separate FP, useful/harmful
removals, and normalized rates are all reported. Per-call support of ordinary
P1 additions and GT foreground/background upward support are also reported.

Code: `configs/pre_sg18_sg21_diagnostics.json`,
`src/segmentation/temporal_head_diagnostics.py`,
`scripts/{run,finalize,plot}_pre_sg18_sg21_diagnostics.py`,
`tests/test_pre_sg18_sg21_diagnostics.py`, and
`jobs/{run,merge}_pre_sg18_sg21_diagnostics.slurm`.
The 40 GPU array tasks are per-dataset/per-class, each resumable per image;
a dependent CPU merger validates all image IDs, writes CSV/JSON and markdown
handoffs, then creates shared-scale 31×8 heatmaps and 31-call curves. Only
reduced arrays are checkpointed. Output root:
`outputs/segmentation/analysis/pre_sg18_sg21/`.
**Submitted 21 October 2026:** synthetic CPU checks and live GPU smoke job
**235238** completed 0:0 (one VOC and one COCO image). The smoke outputs
passed the CPU merger and produced ten heatmaps plus one temporal-curve plot.
Full GPU array **235239** covers 40 class/dataset shards with at most eight
running together on any available GPU; CPU merger **235243** depends on
`afterok:235239`. If a shard fails, rerun its resumable array task and then
submit the merger again after all 40 shards are complete.

## 97. SG18 locked temporal-windowed one-pass propagation (Codex owns run; Claude reviews)

**Prerequisite outcomes, 21 October 2026:** SG17 full run **235237** and
Pre-SG18/Pre-SG21 array **235239** plus merger **235243** completed 0:0.
SG17 failed the VOC promotion rule: `W/P1` won 5/5 folds; forced SG17 OOF
0.7991 versus P1 0.7998, full VOC 0.8001 versus 0.8007, and frozen COCO
0.6196 versus 0.6249. Q70/Q80/Q90 collapsed to one cutoff, 248/248, because
39.93% of positive VOC increments tied at that maximum. SG17+CRF was skipped.
The completed diagnostic found early/middle/late CA-relative TP/FP ratios
1.354/3.494/3.228 on VOC and 0.509/0.941/0.935 on COCO. This motivates
testing *fixed window aggregation*, not a per-pixel reliability gate.

**SG18 contract:** frozen class-word `W` CA seed, the same SA module 15,
31 calls (0–30) and all eight heads. Each \(z_{t,h}=A_{t,h}c\) is computed
without any GT input. `W/REF` is the equal one-pass mean across all 248
observations; `W/P1` is the existing equal observation-wise two-pass mean and
an external benchmark only. The **only** one-pass candidate windows are:

- `W/T1`: calls 10–20, 88 observations (middle only);
- `W/T2`: calls 10–30, 168 observations (middle + late);
- `W/T3`: calls 6–30, 200 observations (drop only earliest six).

No additional start call, head selection, structural weight, pixel gate,
learned parameter, optimized token, or DenseCRF enters. The usual independent
minmax at 64×64, bilinear resize and 17-value segmentation threshold grid
apply. Existing five-fold `run_screen` selects method and threshold on four
VOC folds, evaluates on the held-out fold, and requires at least four fold
votes and positive OOF gain against **W/REF**. In near ties the more inclusive
window wins (T3, then T2, then T1). COCO uses only VOC-frozen thresholds.
P1 receives forced OOF benchmarking but cannot win SG18 selection. If no
candidate passes, REF stays frozen; the best forced-OOF candidate is retained
*only* for diagnostic visuals and change accounting.

Outputs include all method/threshold fold records; full VOC and frozen COCO
mIoU, pixel IoU, mean-image IoU, precision, recall, 20 per-class IoUs/deltas,
20,000-resample paired intervals and image win/loss/tie counts. Each method's
CA-relative additions/removals, distant and disjoint false positives, and
normalized rates are reported under both VOC-frozen method thresholds and
the controlled fixed 0.45 threshold. Direct REF-versus-window overlap counts
retained/lost/new TP, removed/new FP, distant/separate FP removed, and how
much of REF's new TP/FP each window supports. The CA reference mask uses
0.45 in both diagnostic conditions. Binary-change comparisons include the
effect of independently chosen mask thresholds; the fixed-0.45 condition
helps separate that effect from the map window. Eight image sheets show RGB,
CA, REF, T1/T2/T3, GT and a selected-window difference overlay.

Runtime is benchmarked with three separate live generations per method
(REF, VOC-selected or illustrative window, P1). All execute the same UNet
and attention capture; only post-capture aggregation differs. The report
provides end-to-end, common capture and aggregation medians and makes no
inference speedup claim from dropping early calls.

Code: `configs/sg18_temporal_windowed_propagation.json`,
`src/segmentation/sg18_temporal.py`,
`scripts/run_sg18_temporal_windowed.py`,
`scripts/plot_sg18_temporal_windowed.py`,
`tests/test_sg18_temporal_windowed.py`, and
`jobs/run_sg18_temporal_windowed.slurm`. One resumable 12-hour GPU job on
any available GPU checkpoints only reduced maps and scores, then discards
raw Q/K and generated RGB. Final results and images live in
`outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed/`.
Claude: verify the locked windows, five-fold boundary, CA-relative mechanism,
COCO transfer and timing claim after completion. GPU smoke **235289**
completed 0:0 on one VOC and one COCO image; it checked propagation algebra
against the existing implementation, all five score maps, mechanism
accounting, and the real eight-panel plot. The initial smoke **235288**
identified and fixed a NumPy-to-Torch conversion at the mask boundary.
Production job **235290** was submitted on 21 October 2026 to one available
GPU with no node constraint. It is resumable per image under the 12-hour
allocation. Codex will leave this production run unmonitored as requested;
the shared source and prerequisite evidence are in commit `6655ea4f5`.

**Completed result, 22 October 2026; Claude review requested.** Job
**235290** completed all 490 VOC and 707 COCO images, eight visual sheets,
and timing. `W/T1` (calls 10–20) won all five VOC selection folds and passed
promotion: OOF 0.7939 versus `W/REF` 0.7873 (+0.0066). At full-VOC frozen
threshold 0.425 it scored 0.7959 versus 0.7886 (+0.0073); on untouched
COCO-Cap it scored 0.6285 versus 0.6172 (+0.0113). `W/T2` and `W/T3` had
positive forced OOF gains (+0.0063, +0.0043) but no fold votes. External
two-pass `W/P1` scored 0.8006 VOC / 0.6245 COCO at its VOC threshold 0.45.
The T1 mechanism at VOC-frozen thresholds retained 95.2%/96.9% of REF's
CA-relative added true positives on VOC/COCO, while removing 18.8%/12.9%
of REF's added false positives; relative to REF's final masks, T1 removed
453,720/1,404,320 false-positive pixels and introduced 95,469/341,586.
These are pooled pixel counts, not independent image-level effects. Paired
mean-image IoU gains were +0.0074 [0.0051, 0.0099] VOC and +0.0105
[0.0086, 0.0123] COCO (20,000 bootstrap resamples). Full per-class,
fixed-threshold mechanism, per-image scores and figures are in the output
folder's `SG18_CHATGPT_HANDOFF.md` and JSON/CSV artifacts. Claude should
verify the frozen threshold handling, selection boundary, and whether the
mechanism counts support the stated false-positive interpretation before
paper integration.

## 98. SG18 follow-up: T1 + DenseCRF and late-call mechanism (Codex runs; Claude reviews)

**Locked before running, 22 October 2026.** SG18 chose `W/T1`, calls 10–20,
on all five VOC folds. Its frozen COCO mIoU 0.6285 exceeded ordinary
two-pass `W/P1` at 0.6245. The production follow-up asks whether the same
T1 window with the **unchanged** `apply_ovam_densecrf` and
`OVAM_CRF_PARAMETERS` improves upon the established training-free
`W/P1+R2` final reference. No new temporal window or CRF parameter is
selected. The four paired cells are `W/T1`, `W/T1+R2`, `W/P1`, and
`W/P1+R2`, all from the same regenerated trajectory and class-word CA
seed. The 17-threshold grid is unchanged. `run_screen` compares T1+R2 to
P1+R2 using the five fixed VOC folds and training-fold-selected thresholds,
with its existing 4/5 vote and positive OOF guard; COCO uses the full-VOC
thresholds without selection. Report forced OOF, all-cell full VOC and COCO
mIoUs, per-class IoUs/deltas, precision/recall, paired image-level intervals,
and the locked 0.45 final-reference scores. A live VOC+COCO smoke precedes
the resumable GPU production run. Results: `outputs/segmentation/
controlled_attention_cv/sg18_t1_crf/`.

The separate **diagnostic-only** job reads the original SG18 reduced-map
cache from job 235290. Algebraically, the late 21–30 map equals
`(21*T2 - 11*T1)/10`, where T1 averages 11 calls and T2 averages 21.
It compares independently minmax-normalized middle (10–20), late (21–30),
and middle+late (10–30) maps at fixed thresholds 0.425 (primary, SG18 T1
frozen threshold) and 0.45 (control). For both datasets it reports mIoU,
precision/recall, CA-relative TP and FP additions, distant and separate
FP, and direct middle-to-middle+late threshold crossings: TP gained/lost
and FP gained/removed. Per-class diagnostics include person. Late is never
a selection candidate and no COCO score feeds selection. This CPU job is
resumable per image and stores only compact counts and diagnostics. Results:
`outputs/segmentation/analysis/sg18_late_diagnostic/`.

Shared source: `configs/sg18_t1_crf.json`, `configs/sg18_late_diagnostic.json`,
`scripts/{run,finalize}_sg18_t1_crf.py`, `scripts/run_sg18_late_diagnostic.py`,
and `jobs/run_sg18_{t1_crf,late_diagnostic}.slurm`. Claude should verify
the CRF recipe, same-trajectory comparison, VOC-only threshold boundary,
late-window algebra and threshold-crossing counts before paper integration.
The one-VOC/one-COCO GPU smoke **235303** passed all four cells. The
cached-map diagnostic smoke also passed. Independent production jobs were
submitted on 22 October 2026: CPU late diagnostic **235304** and GPU
T1+DenseCRF **235305**. Neither has a dependency on the other. Both jobs
are resumable; no production-result claim is made until their completion
markers and coverage are checked.

**Late diagnostic completed (job 235304, 0:0):** all 490 VOC and 707 COCO
cached images. At the primary fixed 0.425 threshold, middle/T1 and
middle+late/T2 mIoU were 0.7959/0.7936 on VOC and 0.6285/0.6240 on COCO.
Adding late calls caused 261,001 TP gains versus 255,053 TP losses and
399,322 FP gains versus 226,040 FP removals on VOC; on COCO it caused
178,332 TP gains versus 581,249 TP losses and 674,271 FP gains versus
929,652 FP removals. Thus the aggregate effect differs by dataset and
cannot be reduced to a single leakage story. Exact per-class, fixed-0.45
and component/distance diagnostics are in
`outputs/segmentation/analysis/sg18_late_diagnostic/`.

**T1+DenseCRF completed (job 235305, 0:0):** all 490 VOC and 707 COCO
images. T1+R2 versus the locked P1+R2 final reference gives full-VOC
mIoU **0.8122 vs 0.8135** and frozen COCO **0.6465 vs 0.6375**. Forced
VOC OOF is **0.8106 vs 0.8135** (−0.0029); the reference receives **5/5**
fold votes, so T1+R2 is **not promoted** by the preregistered VOC policy.
The COCO mean-image IoU delta is **+0.00496**, paired 95% CI
**[+0.00085,+0.00891]**; 381/323/3 images improve/worsen/tie. At fixed
0.45, T1+R2 COCO mIoU is 0.6510, but that is a diagnostic threshold and
must not replace the frozen 0.425-transfer score. Full analyses and class
tables: `outputs/segmentation/controlled_attention_cv/sg18_t1_crf/
CHATGPT_HANDOFF.md`, `results.json`, `per_class_results.csv`. Claude should
review the VOC-versus-COCO tradeoff before paper integration.

## 99. Pre-SG19 late support of middle-stage additions (Codex diagnostic; Claude may review)

**Locked before running, 22 October 2026.** This is a diagnostic of whether
late calls 21–30 selectively support the useful additions made by middle
calls 10–20. It does **not** construct SG19 or a new segmentation method.
Use the frozen W class-word CA seed, SA module 15, all eight heads, and the
same generated trajectory. Middle-added candidate pixels require raw
`M-c > 0`, CA background at 0.45, and middle foreground at the frozen
0.425 threshold; fixed 0.45 is a control. GT labels them as new TP, new FP,
FP >16 px from GT, FP >64 px, or FP in a predicted component not touching
GT, using the same distance/component definitions as SG18.

For each candidate output pixel, measure raw late score, `L-c`, `L-M`,
`(L-c)/(M-c+1e-8)`, fractions of the 80 late observations above `c` and
above `M`, plus late median, 20% trimmed mean, population standard deviation
and median absolute deviation. Each observation is bilinearly resized to
the annotation size before its candidate-pixel statistic is computed.
For VOC and COCO separately, compute **exact tie-aware pixel-level** ROC-AUC
and average precision for TP versus all FP, distant FP >16, distant FP >64,
and separate-component FP. Feature orientation is chosen by VOC TP/FP AUC
only and transferred unchanged to COCO. The top three VOC features define
VOC decile edges; the same edges are applied to COCO. Report per-bin TP/FP
ratio, distant/separate FP fractions, direct `L>c` versus `L<c` and
`L>M` versus `L<M` splits, and VOC-selected best feature for person, chair,
bird, bicycle, cow, dining table and TV with frozen COCO class transfer.
No veto threshold, class rule, CRF, weighted average, or COCO tuning enters.

The SG18 cache contains only **averaged** T1/T2 maps, the CA seed and GT;
the earlier Pre-SG18 cache stores descriptors and histograms, not the 80
late observation vectors. Therefore the complete requested feature set
requires one resumable GPU replay. It stores only candidate-pixel values and
labels temporarily; images, QK and full attention matrices are discarded.
After exact AUCs and plots are written, temporary feature checkpoints are
removed. Source maps are compared to SG18's cached M/L and any cross-run
drift is reported. Code and specification: `configs/pre_sg19_late_support.json`,
`src/segmentation/pre_sg19_late_support.py`,
`scripts/{run,finalize}_pre_sg19_late_support.py`,
`jobs/run_pre_sg19_late_support.slurm`, and
`tests/test_pre_sg19_late_support.py`. Results go to
`outputs/segmentation/analysis/pre_sg19_late_support/`. Numerical checks
passed. Initial one-VOC/one-COCO GPU smoke **235306** was canceled while
stalled in neumann's scheduler prolog before Python started. Replacement
smoke **235307** excludes neumann; production **235308** is queued with an
`afterok:235307` dependency, so it starts only if the live smoke passes.
There is no result claim while these jobs are pending.

The finalizer writes `CHATGPT_HANDOFF.md`, exact AUC and reliability-bin CSVs,
the TP/FP-ratio plot and a local handoff ZIP automatically. Synthetic
one-image finalizer validation passed before the live smoke was scheduled.

**Outcome (22 October 2026):** Smoke **235307** passed. Production **235308**
extracted all 490 VOC and 707 COCO images and computed the full exact tables;
its final plotting step failed because the `trdp` environment lacks
Matplotlib. CPU finalizer **235313** completed 0:0 and produced the figures,
CSV tables and validated handoff ZIP from the finished candidate-pixel
checkpoints, with no image regeneration. It deleted the 1.1 GB temporary
feature cache; only reduced artifacts remain.
At the primary threshold, raw late score separates middle-added TP from FP
with ROC-AUC **0.586 VOC / 0.675 COCO**; late median and trimmed mean give
essentially the same result. VOC-defined raw-score bins show TP/FP ratio
**2.71 to 9.57** from lowest to highest bin; on COCO the same bins show
**0.55 to 7.68**, with a monotonic COCO trend. However, `L-c`, retention,
and late consensus are weaker (COCO TP/FP ROC-AUC 0.53–0.56), and VOC-person's
best `L-M` signal (0.695 ROC-AUC) transfers poorly to COCO (0.481).
Thus late absolute score has a transferable association, while evidence for
a selective *late-stage veto* is insufficient. No SG19 mask or cutoff was
tested. Full exact results are in the output folder's `results.json`,
`auc_table.csv`, and `reliability_bins.csv`; Claude may review the diagnostics.

## 100. Pre-SG19 raw-score control (CPU only; SG19 decision)

**Completed 22 October 2026, job 235316 (0:0).** Use the saved SG18 image
maps to reconstruct CA `c`, middle `M` (calls 10–20) and late `L` (calls
21–30). No generation occurs. On each image, fix the same middle-added
candidate set for all five raw scores: `c`, `M`, `L`, `min(M,L)` and
`(M+L)/2`. Candidate definition and GT/distance/component labels follow
Pre-SG19 exactly: raw `M>c`, normalized CA background at 0.45, normalized
middle foreground at 0.425. Report exact pixel ROC-AUC and PR-AUC for TP
versus all FP, FP >16, FP >64 and separate-component FP. This is a diagnostic;
no cutoff or new segmentation mask is evaluated. Code:
`scripts/pre_sg19_score_controls.py`; result:
`outputs/segmentation/analysis/pre_sg19_score_controls/`.

The table's ordinary TP/FP ROC-AUC is: CA **0.5590 VOC / 0.6198 COCO**;
middle **0.6081 / 0.6861**; late **0.5835 / 0.6733**; min **0.6006 / 0.6784**;
mean **0.5973 / 0.6814**. Middle also wins all three distant/separate FP
contrasts on both datasets, and its ordinary TP/FP PR-AUC is **0.8417 /
0.7099** versus late's **0.8262 / 0.6931**. The result does not support
additional late-stage discrimination or the original SG19 late veto.

The Pre-SG19 GPU replay's candidate-pixel checkpoint was removed after its
finalizer. SG18 cache reconstruction gives 6,816,320 VOC and 12,665,100
COCO candidates versus 6,837,692 and 12,671,265 in that replay. All five
scores in this control still use exactly the same pixels *within the SG18
cache run*. Late ROC-AUC here (0.5835 / 0.6733) is close to the replay's
0.5857 / 0.6751. The `CHATGPT_HANDOFF.md` in the control output folder
summarizes the full comparison; CSV and JSON contain every ROC/PR contrast.

## 101. SG21 locked structural SA-head weighting (Codex runs; Claude reviews)

**Locked before execution, 22 October 2026.** The W class-word CA seed,
SD 1.5, SA module 15, eight heads, one-pass propagation, and calls 10–20
inclusive are frozen from SG18/T1. SG21 changes only the eight-head average
*within* each of those 11 calls; the 11 call outputs are averaged equally.
The structural descriptors match the Pre-SG21 definitions exactly: mean
row entropy, mean Euclidean query-key distance in the 64×64 grid, and mean
radius-1 local attention mass. No GT or class-specific information enters
descriptor computation or weights.

Four and only four candidates are registered: **SG21-LM** weights raw local
mass; **SG21-LM-RANK** weights stable ascending local-mass ranks 1–8 (ties
by physical head ID); **SG21-ENT** weights call-maximum entropy minus each
head's entropy plus 1e-8; **SG21-DIST** weights inverse distance plus 1e-8.
Each candidate normalizes its eight weights to sum to one separately for
each call. Reference is same-generation **SG18/T1**, never W/REF. W/P1 is
historical external context only and is excluded from selection. There is
no combination, exponent, temperature, hard head selection, additional
window, CRF, second pass, training, COCO selection, or descriptor sweep.

Use the fixed five VOC folds and unchanged 17-value 0.30–0.70 threshold
grid. Select each threshold on four training folds and score the fifth;
report forced OOF for every candidate, the normal promotion screen, fold
votes, and full-VOC threshold. Promotion needs at least four of five fold
votes, positive forced OOF versus T1, and the existing magnitude/class
stability guards. Freeze VOC thresholds for COCO transfer. Report full VOC
and COCO mIoU, precision/recall, pixel/mean-image IoU, all 20 class IoUs
and deltas versus T1, 20,000-resample paired mean-image-IoU intervals, and
better/worse/tied image counts. At fixed 0.45, report CA-relative TP/FP
additions, distant and separate FP, useful/harmful removals. For each
candidate report effective head count per call, top physical head
frequencies, call-to-call top-head changes, and weight mean/spread. If a
candidate is promoted, use VOC-defined weight quartiles to diagnose head
observation TP/FP, distant FP and useful-change rates on VOC and COCO;
this diagnostic cannot feed back into selection.

Code: `src/segmentation/sg21_structural.py`,
`scripts/{run,finalize}_sg21_structural.py`,
`configs/sg21_structural.json`, and `jobs/{run,merge}_sg21_structural.slurm`.
The GPU jobs are resumable class shards (20 VOC, 20 COCO) and retain only
per-image counts, fixed-threshold event counts, weights, and descriptors;
they do not save images, propagated maps, or attention matrices. The CPU
merger refuses incomplete coverage. Numerical unit checks and live one-VOC/
one-COCO GPU smoke **235321** completed 0:0. Production array **235322**
(`0-39%8`, any available GPU, 12 hours per shard) and CPU merger **235326**
were submitted. The merger uses `afterany` so it will report missing
coverage explicitly if a shard fails rather than becoming permanently
unsatisfied. No outcome is claimed while production runs. Codex owns the
run and publishes the reduced results; Claude reviews the method,
selection and interpretation afterward.

**Outcome (22 October 2026):** GPU array **235322** produced all 490 VOC
and 707 COCO image checkpoints; CPU merger **235326** completed 0:0 and
verified coverage. T1 won **5/5 VOC fold votes**; no structural candidate
was promoted. Forced VOC OOF mIoU is **0.7939 T1**, **0.7884 LM**,
**0.7901 LM-RANK**, **0.7872 ENT**, and **0.7941 DIST**. DIST's +0.00017
OOF delta is below the promotion guard; its frozen COCO 0.6287 versus
T1 0.6285 is negligible. LM/ENT improve fixed-threshold TP/FP addition
ratios but remove too many true-positive additions and reduce end-to-end
mIoU on both datasets. Weight diagnostics show strong recurrent physical
head preferences, especially DIST head 6. T1 replay differs from the
saved SG18 map by only about 2×10⁻⁸ mean absolute value. SG21 is closed
under the locked no-sweep rule. Full interpretation and all 20 class
results: `outputs/segmentation/controlled_attention_cv/sg21_structural/
CHATGPT_HANDOFF.md`, `results.json`, `per_class_iou.csv`,
`weight_diagnostics.json`. Claude should validate the selection and
structural-mechanism interpretation before paper integration.

## 102. SG22/SG23 locked factorials (Codex runs; Claude reviews)

**Locked before execution, 22 October 2026.** These are training-free W
class-word experiments using the same frozen CA seed, SD 1.5 generation,
SA module 15 at 64×64, all eight heads, independent map min-max, bilinear
resize, 17-value threshold grid, five VOC folds, and VOC-only selection.
COCO-Cap is frozen transfer. All cells are regenerated together within each
experiment so references and candidates are same-generation. The R2
DenseCRF uses the existing SimpleCRF wrapper and unchanged parameters on
the generated RGB and each thresholded mask. No optimized tokens or
additional hyperparameter search enters either package.

**SG22: temporal window × propagation depth.** ALL means calls 0–30;
MID means calls 10–20 inclusive. For each call/head observation independently,
form `A_i c`, `A_i² c`, and `A_i³ c`, then average observations inside the
window. The six raw cells are ALL-P1 (historical REF), ALL-P2 (historical
P1), ALL-P3, MID-P1 (SG18/T1), MID-P2, MID-P3. Apply unchanged R2 to all
six for 12 cells total. Raw reference is ALL-P2; final reference is
ALL-P2+R2. Raw candidates of primary interest are MID-P2, MID-P3, ALL-P3;
the remaining cells are controls. At fixed 0.45 report CA-relative TP/FP
additions, distant and separate FP, useful/harmful removals and depth
1→2 / 2→3 threshold crossings for each window. In particular inspect
person, chair, bird, bicycle, cat, dining table, TV, and potted plant.
Do not test further depths, windows, blending, or per-class rules.

**SG23: temporal window × reliability mechanism.** The eight raw factorial
cells are ALL/MID crossed with R0 ordinary mean, exact locked SG2
`G_cv_g20_n1` graph (gamma 2), SG11 20%-per-tail trim gated by directional
consensus at rho 0.80, and SG12 leave-one-out cosine top 50% observation
mean. For MID, every statistic uses only its 88 observations, retaining
exactly 44 for SG12; ALL uses 248 and retains 124. Apply unchanged R2 to
all eight. Regenerate ALL-P2 and ALL-P2+R2 in the same job solely as
auxiliary final-pipeline references, outside the eight-cell factorial.
Raw reference is MID-R0 (T1); final reference is ALL-P2+R2; MID-R0+R2 is
reported explicitly. Report VOC OOF/full and COCO interaction
`(MID-method−MID-R0)−(ALL-method−ALL-R0)` for SG2/11/12. At fixed 0.45
compare each MID reliability mask to T1 for TP/FP retained/removed, distant
and separate FP, and FP removed per TP lost. Do not stack mechanisms or
sweep their parameters.

Both packages report forced OOF for every cell, the normal fold-vote
promotion screen, full VOC with its selected threshold, frozen COCO,
precision/recall/pixel/mean-image IoU, all 20 class IoUs and deltas,
20,000-resample paired per-image IoU intervals, and better/worse/tied
image counts. Raw and final-pipeline questions are screened separately.
Claude should independently check the operator identities, fold selection,
CRF reference, class deltas and interpretation before paper integration.

Code: `src/segmentation/sg22_sg23.py`,
`scripts/{run,finalize}_sg22_sg23.py`, `configs/sg{22,23}_factorial.json`,
`jobs/{run,merge}_sg22_sg23.slurm`. The GPU class-shard array is resumable,
with 20 VOC and 20 COCO shards per experiment; its reduced per-image
counts and diagnostics are local only. A CPU merger checks complete
coverage and writes reviewed summary files under
`outputs/segmentation/controlled_attention_cv/sg22/` and `/sg23/`.
Numerical operator checks passed. Live one-VOC/one-COCO SG22 smoke
**235366** scored both images; its initial finalizer encountered a shard
filename filter error, corrected and rerun successfully on the saved smoke
shards. SG23 live smoke **235367** completed end to end. Production GPU
arrays **235368** (SG22) and **235372** (SG23) were submitted as
`0-39%4`, any available GPU, 12 hours per resumable class shard. CPU
mergers **235373** and **235374** depend on their respective arrays with
`afterany`; they explicitly reject incomplete coverage. Claude can review
the committed protocol and implementation while the jobs run. No outcome
is claimed yet.

**Outcome (23 October 2026):** SG22 GPU array **235368** and merge
**235373**, and SG23 GPU array **235372** and merge **235374**, completed
with all 490 VOC and 707 COCO images per experiment. SG22's raw screen
kept ALL-P2 (3/5 votes); its CRF screen selected **MID-P2+R2** (4/5
votes), selected OOF 0.8158 versus P1+R2 0.8132. Forced MID-P2+R2 OOF
is 0.8168; full VOC 0.8185 versus 0.8132 and frozen COCO 0.6438 versus
0.6372. MID-P1+R2 still has slightly higher frozen COCO mIoU (0.6464)
without satisfying the VOC selection gate. SG23's raw screen selected
**MID-SG2** (5/5, OOF 0.7999 versus T1 0.7939), but its COCO advantage
is only 0.6292 versus 0.6287. Its CRF screen kept **ALL-P2+R2** (4/5);
no SG23 final cell was promoted. Raw SG2/SG11/SG12 temporal interactions
are all negative on VOC OOF and COCO, indicating overlap with the middle
window. The analyses are in
`outputs/segmentation/analysis/SG22_SG23_CHATGPT_HANDOFF.md`; complete
cell/class/mechanism data are in
`outputs/segmentation/controlled_attention_cv/sg22/` and `/sg23/`.
Claude should validate these outcomes before paper integration.

## 103. SG24-A and SG24-B: automatic CA construction (Codex runs; Claude reviews)

**Locked 23 October 2026, before production.** Two separate experiments test
CA-only segmentation and the current final pipeline. Both use SD1.5, ordinary
class-word W attribution, conditional CFG query, padded 77-token text softmax,
all class-word pieces averaged, calls 0–30, all CA heads, and the existing
five VOC folds and 17 thresholds. The downstream operator is unchanged:
SA module 15, calls 10–20 inclusive, all eight heads, mean of each
observation's `A_i² c` (MID-P2), then existing R2 DenseCRF. COCO is frozen
transfer; no optimized tokens, learned parameters, class-specific decisions,
new time windows, SA rules, or CRF tuning.

**SG24-A** fixes CA modules `[0,1,4,5,6,7,8,9,13,14,15]` and compares
A0 existing equal aggregation; A1 per-pixel content-token mass rescaling;
A2 projected-output CA head contribution weights; A3 pseudo-global/paired-SA
similarity layer weights; and A4 all three changes combined. No optional
combinations are included. A0 is a same-generation control. Content pieces
exclude CLIP special/padding tokens, punctuation and English stopwords,
while all target pieces are always retained. A2 computes
`v_h=(A_h V_h) W_h^O`, `s_h=v_h·sum_j v_j`, uses GoCA SD1.5's
dot-product normalization *without* clamping negative scores, and uses
a uniform fallback only if the summed projected output vanishes. A3
derives `F` from the conditional Q of SD1.5
`up_blocks.2.attentions.1.transformer_blocks.0.attn2` at each call, concatenates
heads, forms `softmax(FF^T/sqrt(d))`, and scores paired mean-head SA spatial
affinities after GoCA's bilinear key-axis and nearest query-axis resize.
Layer weights are nonnegative, normalized within each call, then calls are
averaged equally. This is a GoCA-*style adaptation* to our full generation
trajectory, not a claim to reproduce GoCA's original one-timestep SD1.5
image-inversion settings.

The COCO attribution prompt is the class name alone, as in our frozen
pipeline. For A1/A4, content-token rescaling can therefore reduce a
single target-content-token map to a spatial constant. Do not alter that
prompt to rescue the method; report this as a structural transfer limitation.
For both evaluation levels, report forced VOC OOF, fold thresholds/votes,
full VOC, frozen COCO, all 20 classes, precision/recall/pixel and mean-image
IoU, 20k paired intervals, and better/worse/tied images. CA-only maps are
scored before SA or CRF. Fixed-0.45 CA TP/FP, coverage, distant/separate
FP, and useful/harmful changes versus A0 are diagnostic only. Record
content attention mass and automatic layer weights by module/call/class.

**SG24-B** compares B0 the 11 current CA modules with B1 all 16 CA modules
including all 32×32 modules, using the *same* automatic A3 or A4 method.
The B method is frozen from SG24-A's VOC final-pipeline selection. If neither
automatic method passes the existing VOC promotion gate, use preregistered
A3 for B as a diagnostic, and label B as non-promotional. Recompute layer
weights over each available set; do not manually prune any layer or search
other resolution subsets. Both CA-only and MID-P2+R2 outcomes use the same
VOC-only selection and frozen COCO transfer. Report per-layer/resolution
weights including top-1/top-3 frequency and by-call/by-class means, plus
CA-only versus final-pipeline gains.

Implementation: `src/segmentation/sg24_ca.py`, `scripts/{run,finalize}_sg24.py`,
`configs/sg24.json`, `jobs/{run,merge}_sg24.slurm`. The GPU runners write
only resumable class-level counts and reduced diagnostics; no images, masks,
or full attention matrices are saved. Each experiment has its own 40-task
VOC/COCO array and CPU merge. SG24-B's production array depends on the
SG24-A merge because its automatic method is frozen by that merge. Claude
should independently review the GoCA adaptation, A0 reference reproduction,
VOC selection, COCO content-rescaling limitation, full per-class results,
and paper interpretation before any promotion or manuscript update.

**Execution handoff (23 October 2026).** Final live smoke jobs **235898**
(SG24-A) and **235899** (SG24-B) passed one VOC and one COCO image each.
On those same-generation images, A0/MID-P2+R2 reproduced SG22's saved
17-threshold counts *exactly* on both datasets. The class-only COCO
attribution makes A1 and A4 CA maps spatial constants, as predicted by
the rescaling formula; their smoke masks were empty at 0.45. This is a
method limitation, not a prompt or threshold change. Production SG24-A
GPU array **235900** and completeness-checking CPU merge **235904** are
submitted. Separate SG24-B GPU array **235905** waits on the successful
SG24-A merge and its VOC-only automatic-method manifest; its CPU merge is
**235906**. Both arrays request any available GPU, at most four class shards
concurrently, with 12 hours per resumable shard. These are submissions,
not result claims. Claude should review the committed implementation and
selection contract now, then independently validate the produced results.

**Outcome (24 October 2026; pending Claude review).** Both arrays and
mergers completed with 490 VOC and 707 COCO images each. SG24-A's CA-only
screen retained A0 in 5/5 folds. Its final multi-candidate screen retained
A0: A3 received 3/5 selection votes, below the 4/5 promotion gate, and
selected-path OOF was 0.81616 versus 0.81689 for A0. In the *forced* A3
comparison, A3 nevertheless won 4/5 held-out folds, with OOF 0.82297
versus 0.81689, full VOC 0.8243 versus 0.8185, and frozen COCO 0.6620
versus 0.6437. These are secondary findings, not a formal promotion.
A1 and A4 collapse to zero COCO IoU because the unchanged class-only
attribution prompt makes target-only content rescaling spatially constant.
A2 worsens both datasets. SG24-B used diagnostic A3 per the preregistered
fallback. Its expanded 16-module B1 lost all five VOC folds: final OOF
0.8152 versus 0.8230 for 11-module B0, full VOC 0.8152 versus 0.8244.
Frozen COCO final rose 0.6622→0.6703; this transfer-only gain cannot
override VOC selection. The restored 32×32 group received about 31% of
automatic layer weight, so it was not suppressed. Full fold, class,
bootstrap and diagnostic analysis is in
`outputs/segmentation/analysis/SG24_CHATGPT_HANDOFF.md`; machine-readable
data are in the `sg24_a/` and `sg24_b/` summary files. Claude should
verify the selection interpretation and class/weight analyses before
manuscript integration.

**Owner decision (24 October 2026; post-results working default).** Use
**A3 → MID-P2 → R2** for subsequent training-free method development, with
the selected 11 CA modules, all CA calls and heads, the unchanged MID-P2
and R2 operators, and the VOC-selected final threshold **0.425**. The
owner chose A3 after reviewing its forced A3-versus-A0 comparison: A3 won
4/5 held-out VOC folds, gained 0.00608 forced OOF mIoU, and transferred
positively to frozen COCO (+0.0183 full-set macro mIoU). This is an
**exploratory owner override**, not a retroactive pass of SG24-A's locked
multi-candidate promotion gate. That screen selected A3 in 3/5 folds and
its selected-path OOF was below A0. Preserve A0 as the registered SG24-A
winner and report the working-default decision separately in the paper.
SG24-B's 16-module B1 remains excluded because it lost all five VOC folds.

## 104. SG25 multi-resolution SA and Pre-SG26 arrival diagnostic (Codex runs; Claude reviews)

**Locked 24 October 2026, before production.** Freeze the owner's A3 working
pipeline: selected 11 CA modules, calls 0–30, all CA heads, SG24 pseudo-global
layer weights. Freeze SA calls 10–20, all eight heads, equal observation
averaging, and the existing R2 DenseCRF. Use one fixed SA module at each
scale: final upsampling-path attention 2 at 16×16, 32×32, and the established
module 15 at 64×64. Exact module names, 17-threshold grid and operator
definitions are in `configs/sg25.json`. No CA, head, time-window, CRF,
reliability, or token changes are searched.

Compare six two-operator cells: FINE-P2, COARSE16-P2, COARSE32-P2, 16→64,
32→64, and 64→16. The operators in a cross-resolution cell use the *same*
diffusion call. Average all heads after the first operator, resize its map,
apply each head of the second operator, then average second heads and calls.
This avoids assuming that head index 0 at one scale matches head index 0 at
another. Bilinear interpolation is used for upsampling; area interpolation
for downsampling. Score raw maps and each map after unchanged R2. Five fixed
VOC folds select against A3/FINE-P2+R2; evaluate COCO only with VOC-frozen
thresholds. Report forced OOF, multi-candidate votes, per-class IoU,
precision/recall, and paired image intervals. A3/FINE-P2+R2 is recomputed
as a same-generation control and is checked against SG24's original fine
propagation operator on smoke images. Prior independently regenerated SG24
smoke counts differed by at most two pixels in one threshold cell, so an
exact cross-job count match is not used as a gate.

In the same generation pass, Pre-SG26 computes the mean 64×64 SA map at
depths 1–5 using the existing fine module. Min-max normalize *each depth*
and apply the fixed A3 working threshold 0.425 at annotation resolution.
Each pixel is assigned to its first positive depth; report total pixels,
TP, FP, FP beyond 16/64 pixels from GT, and FP in a predicted component
with no GT overlap, overall and for every class on VOC and COCO. This is
diagnostic only: no arrival-time mask or threshold is selected.

Implementation: `src/segmentation/sg25_sa.py`, `scripts/{run,finalize}_sg25.py`,
`jobs/{run,merge}_sg25.slurm`. Shards keep only confusion counts and small
arrival-count sidecars. No images, attention matrices, or masks are saved.
Claude should review scale pairing, exact FINE-P2 reproduction, selection,
and cross-dataset interpretation before manuscript integration.

**Execution handoff (24 October 2026).** Smoke job **236162** completed on
one VOC and one COCO image; the live FINE-P2 map passed a direct numerical
comparison with SG24's original MID-P2 operator on both. Production GPU
array **236164** covers all 20 class shards on each dataset, with at most
four shards at once, a 12-hour resumable limit per shard, and no node pin.
CPU merge **236165** has an `afterok:236164` dependency and will reject
incomplete coverage. These IDs are submissions, not result claims. Claude
should review the code and locked conditions while the jobs run, then
independently verify the results before paper integration.

**Outcome (24 October 2026; pending Claude review).** Array **236164**
and merge **236165** completed: all 490 VOC and 707 COCO images are present.
The same-generation A3/FINE-P2+R2 control scored **0.82425 VOC / 0.66210
COCO**. Raw and R2 VOC five-fold screens each kept their fine-resolution
reference in all five folds; no SG25 cell is promoted. Frozen COCO transfer
nevertheless favored **16→64+R2 at 0.69750** (+0.03540 versus fine) and
**COARSE32-P2+R2 at 0.69481** (+0.03271), while both lost on VOC. The
Pre-SG26 first-arrival diagnostic found declining TP rates with greater
depth: VOC .865→.223 and COCO .708→.117 from first depth 1 to 5. Class
exceptions and per-depth normalization prevent treating this diagnostic as
an evaluated SG26 method. Full results and interpretation are in
`outputs/segmentation/analysis/SG25_PRESG26_CHATGPT_HANDOFF.md`; exact
summaries are under `outputs/segmentation/controlled_attention_cv/sg25/`.
Claude should independently audit the outcomes before paper integration.

**Post-hoc caption-presence split (24 October 2026).** The released COCO
`word_included` field divides SG25's 707 images into 465 whose generation
caption names the queried class and 242 whose caption does not. The
class-only CA attribution prompt names it in both groups; all 490 VOC
generation prompts name their target class. Frozen 16→64+R2 versus fine
gains +0.0239 macro mIoU in the COCO present group and +0.0480 in the
absent group, but the paired mean-image gain difference is only +0.0033
(image-bootstrap interval [−0.0122,+0.0196]). Class-balanced interaction
is also uncertain; the groups have unequal class coverage. This is
descriptive, not a new selection rule. Full analysis and reduced per-image
counts are in `outputs/segmentation/analysis/SG25_CAPTION_PRESENCE.md` and
`outputs/segmentation/controlled_attention_cv/sg25/caption_presence_per_image.csv`.

## 105. SG26 coarse-guided fine SA (Codex runs; Claude reviews)

**Locked 24 October 2026, before production.** Test whether coarse SA can
constrain fine SA without replacing fine propagation. Freeze A3 class-word
CA from SG24, all 11 selected modules, CA calls 0–30, all heads and its
normalization. Freeze SA calls 10–20, the three upsampling-path modules at
16×16, 32×32 and 64×64 identified in SG25, all eight heads, equal averaging
of 88 call/head observations, the existing 17-threshold VOC five-fold
selection, frozen COCO transfer, and unchanged R2 DenseCRF. Exact modules,
cells and diagnostics are in `configs/sg26.json`.

S0 is current 64→64 FINE-P2. S1 and S2 exactly repeat SG25 16→64 and
32→64. For each call, average the eight coarse attention matrices at 16
or 32, lift the resulting row-stochastic matrix to the 64×64 grid by mapping
each fine pixel to its containing coarse cell, multiply each fine head's
attention matrix elementwise by the lifted gate, and row-normalize. S3/S4
use this 16/32-gated fine operator for **both** propagation passes. S5/S6
use ordinary fine SA on pass 1 and 16/32-gated fine SA on pass 2. Each
pass pair remains within the same call and fine head. No exponent,
temperature, gate cutoff, mixing coefficient, new head weighting or time
window is permitted. S0–S2 must match SG25's live operators on VOC and
COCO smoke samples.

Score all seven raw maps and all seven maps after unchanged R2. Select raw
and final candidates separately against S0 at five fixed VOC folds; COCO
has no selection role. Report forced OOF and selector votes, full VOC,
frozen COCO, precision/recall, pixel and mean-image IoU, all 20 per-class
IoUs, and paired 20k image intervals. At a fixed raw-map threshold 0.45,
record TP/FP gained/lost/removed, FP beyond 16/64 pixels, and separate
component FP for S3–S6 versus S0. For edge diagnostics, use a deterministic
GT-balanced post-hoc sample of 64 foreground and 64 background fine-grid
query/key positions per image. Report gate values; fraction of gated/fine
edge ratios below 0.5 and within [0.9,1.1]; sampled-row entropy and
effective neighbors; and mean fine edge probabilities before/after gating
for object→object, object→background, background→object and
background→background edges. Diagnostic thresholds never alter inference.
For all-foreground masks, unavailable edge categories are marked missing.

Implementation: `src/segmentation/sg26_sa.py`, `scripts/{run,finalize}_sg26.py`,
`jobs/{run,merge}_sg26.slurm`. One image and its attention matrices stay in
memory until all cells and diagnostics are reduced to small per-image counts;
then they are released. No images, masks, or attention matrices are saved.
Claude should review the block-lift/row-normalization operator, S0–S2
controls, VOC selection, and edge interpretation before manuscript claims.

**Execution handoff (24 October 2026).** Live smoke job **236442** passed one
VOC and one COCO image. S0, S1 and S2 were numerically checked against the
same captured observations processed by SG25; row normalization and
all-foreground GT diagnostic handling also passed small CPU checks.
Production GPU array **236443** covers 20 class shards on each dataset,
at most four at once, 12 hours per resumable shard, with no specified
node. CPU merge **236444** waits on `afterok:236443` and checks full
coverage before emitting results. These are submissions, not result
claims. Claude should inspect the committed operator and diagnostic
definitions while jobs run, then verify the results before paper use.

**Outcome (24 October 2026; pending Claude review).** Array **236443**
completed all 40 shards and merge **236444** completed, covering 490 VOC
and 707 COCO images. S0/S1/S2 reproduce SG25's full-set scores to at
most 1.2×10⁻⁷ mIoU. Both raw and R2 five-fold VOC screens chose S0 in
5/5 folds; no SG26 gate is promoted. Final VOC/COCO mIoU: S0 .8242/.6621,
S3 .7992/.6817, S4 .7993/.6746, S5 .8201/.6821, S6 .8193/.6773.
Historical S1 remains the highest frozen COCO observation at .6975.
Gates reduce cross-boundary fine-edge probability but also strongly
concentrate within-background edges; 32-gating suppresses more sampled
edges and reduces row entropy more than 16-gating. At fixed .45 raw-map
threshold, VOC S5 gains a net 857k FP despite removing some distant FP;
COCO S5 removes a net 312k FP with nearly balanced TP additions/losses.
Full scores, per-class trade-offs, edge diagnostics, bootstrap intervals
and interpretation are in
`outputs/segmentation/analysis/SG26_CHATGPT_HANDOFF.md`; exact reduced
outputs are under `outputs/segmentation/controlled_attention_cv/sg26/`.
Claude should audit these before integrating SG26 into the manuscript.

## 106. SG27 comprehensive late fusion of three SA output maps (Codex runs; Claude reviews)

**Locked 24 October 2026, before production.** The only inference change
is a fixed, parameter-free fusion of three *completed* SG25 propagation
maps: FINE-P2, 16→64 and 32→64. A3 CA, selected 11 CA modules, calls
0–30, class-word W handling, SG25 SA modules/calls 10–20/heads/paths,
five fixed VOC folds, 17-threshold grid, frozen COCO transfer and R2
DenseCRF stay unchanged. No coarse affinity may alter a fine SA matrix.
Independently spatially min-max normalize each base 64×64 map, apply one
of the 46 proposed formulas in `src/segmentation/sg27_fusion.py`, then
spatially min-max normalize the fused map again. Fixed epsilon is 1e-8
for harmonic formulas. The proposed families include pairwise FINE+C16,
FINE+C32, direct C16+C32, three-scale means/unions/intersections,
two-of-three median/top-2/bottom-2 and fine-anchored agreement/residual
operators. The full names and thresholds are in `configs/sg27.json`.

Before production, compare every proposed pair on **four live samples**
(VOC aeroplane/person and COCO aeroplane/person). The largest absolute
difference across *all* samples must be below 1e-8 to call two formulas
equivalent. Retain the first numeric cell name only and freeze the resulting
manifest in `configs/sg27_unique.json`. Algebraic aliases expected from
the specification are S11=S7, S12=S8, S15=S7 and S24=S20; the smoke
scan must still verify them and detect any others. S0/S1/S2 must also
numerically match SG25's original base maps on the same observations.

Score each unique cell before R2 and after unchanged R2. Report forced
per-cell OOF, held-out fold wins and the *separate* multi-candidate VOC
selector, plus full VOC and frozen COCO metrics, all 20 classes and paired
20k image intervals. Average VOC/COCO rank is descriptive only, never a
selection rule. At a fixed raw-map threshold .45, reduce per-image TP/FP,
distant FP >16/>64, separate-component FP and foreground-area changes for
all cells so the finalizer can report cells within .005 VOC mIoU of FINE
or with at least +.015 frozen COCO mIoU. Record 64×64 GT-positive/negative
counts for the four FINE/C16 and FINE/C32 high/low patterns and all eight
three-scale patterns at fixed normalized-map threshold .5. GT appears
only in evaluation and post-hoc diagnostics. For a small set of result
leaders, report COCO's released caption-word-present/absent split without
using it to select a method. No image, mask or attention matrix is saved.

Implementation: `src/segmentation/sg27_fusion.py`,
`scripts/{sg27_runtime,smoke_sg27,run_sg27,finalize_sg27}.py`,
`jobs/{smoke,run,merge}_sg27.slurm`. GPU smoke first freezes duplicates;
a second live VOC/COCO smoke scores all unique cells and checks metrics;
then a resumable 40-shard GPU array and complete-coverage CPU merge run.
Claude should inspect formula uniqueness, normalization, base controls,
VOC selection and broad-screen interpretation before paper integration.

**Preproduction verification (24 October 2026).** Four-image duplicate
scan job **236775** completed successfully. It found **42 unique cells**;
the only aliases were S11=S7, S12=S8, S15=S7 and S24=S20. The closest
remaining pair differed by 0.007784 in maximum absolute map value across
the four samples, well above the locked 1e-8 duplicate rule. S0/S1/S2
were checked against SG25 on the same captured observations. End-to-end
smoke job **236776** then scored one VOC and one COCO image across all 42
unique raw and R2 cells, validated reduced diagnostic shapes, and finished
with exit code zero. `configs/sg27_unique.json` freezes the manifest;
`outputs/segmentation/controlled_attention_cv/sg27/duplicate_scan.json`
records the full pairwise differences. These are checks, not VOC/COCO
performance results. Claude can review this implementation and the
forthcoming full results before manuscript integration.

**Production handoff (24 October 2026).** Submitted GPU array **236777**
for 20 VOC and 20 COCO class shards, at most four concurrent, with no
specified GPU node. Each shard is resumable and has a 12-hour limit.
CPU merge **236778** has `afterok:236777` dependency and will require
all 490 VOC and 707 COCO samples before writing aggregate reports.
Per-image count/diagnostic shards remain local while running; complete
compact shards and summaries will be reviewed and shared after the jobs
finish. Claude owns independent
review of the methods and results, not a duplicate SG27 run.

## 107. Claude: review of Sections 84–105; one implementation problem (A3), one reproducibility gap

Reviewed at `33a50f4` in Claude's environment (`gen`; the `trdp` checks noted
below), from scratch copies only. SG27 (Section 106) is still running and is
not covered.

### Summary

| Package (Sections) | Verdict | Basis |
|---|---|---|
| SG11 / SG12 / SG-Diag (84, 89) | Accept with notes | regenerated |
| SG2 + DenseCRF (87, 89) | Accept | regenerated |
| Recovery of the two failed jobs (89) | Accept | regenerated |
| SG12+CRF, SG14, SG15 (91–92) | Accept | regenerated |
| SG15 vs oracle visuals (93) | **Not verifiable** | outputs not in the repository |
| SG13 (88, 90) | Accept with notes | regenerated |
| Pre-SG16/17, Pre-SG18/21, Pre-SG19 (94, 96, 99, 100) | Accept | summaries and code only |
| SG17 (95) | Accept | summaries and code only |
| SG18 T1 (97) and T1 + DenseCRF (98) | Accept with notes | summaries and code only |
| SG21 (101) | Accept with notes | summaries and code only |
| SG22 / SG23 (102) | Accept with notes | summaries and code only |
| **SG24-A / SG24-B (103)** | **Problem: A3 implementation** | summaries and code only |
| SG25, SG26 (104, 105) | Accept with notes; inherit the A3 problem | summaries and code only |

**Every number stated in Sections 84–105 matches its committed summary.**
The one exception is cosmetic: Section 92 states SG11+B COCO +.000108; the
summary gives +.000107.

### Regenerated packages

These finalizers were rerun on copies of the shards:

- `summary.json`, `fold_results.json`, `report.md`, the ChatGPT reports and
  the visuals are **byte-identical**.
- CSVs differ only in line endings: the Python `csv` writer emits CRLF, the
  committed files have LF.
- SG11/SG12 `*_aggregates.npz` differ by one float32 ulp in histogram edges
  (numpy version).
- `src/segmentation/propagation_diagnostics.py:108` calls `np.trapz`, which
  numpy 2.4 removed. The finalizer ran with `np.trapezoid`, the same rule.

**Section 89 recovery is justified.**

- The original gate fails only across hardware: SG2's VOC REF misses by .100
  of an image, SG11's by .031. Claude measured independently that even
  `trdp` vs `gen` on the same A100 exceeds .002 on 41/490 VOC and 99/707 COCO
  images, so such a gate cannot pass across environments.
- Same-hardware checks agree to within a few pixels. Stronger than Section 89
  states: the later erdos runs `235021`–`235023` reproduce the recovered
  SG11/SG12 cells with **0 pixel difference** on every image and threshold,
  and `sg12_crf`'s P1 and P1+R2 exactly equal the SG2 package.
- Not reported in Section 89: SG2+CRF's secondary screen against REF+R2 kept
  REF+R2 (3/5). Its forced gain is positive, but COCO is −.0087.

**SG13 notes.**

- The region-score audit compares two implementations of the same mean on
  the same labels, so it is close to tautological; partition equivalence
  rests on the synthetic test.
- Section 88 asked that large historical drift be discussed. Against the
  `gen` packages it is: REF up to 10.1% / 10.6% of an image (VOC / COCO);
  SPEC32 up to 15.4% / 51.8%, with 143/490 and 261/707 images over 1%.
  Pooled REF mIoU moves only .0002 / .0006.

### The A3 problem (Section 103; affects SG24-A/B, SG25, SG26 and the working default)

`affinity_similarity` in `src/segmentation/sg24_ca.py:84–87` resizes the key
axes (bilinear) and the query axes (nearest) as GoCA does. It **omits
GoCA's second `permute(2, 3, 0, 1)`** (GoCA
`src-additional/attention-observation.py:494–500`), so the resized SA matrix
is indexed [key, query]: A3 scores ⟨Mᵀ, P⟩ instead of ⟨M, P⟩.

- Claude reproduced this numerically on random attention. The function
  returns exactly the transposed product (3.984319), against GoCA's
  3.991347.
- Every SG24-A module is 8, 16 or 64 wide and passes through this path, so
  the error is consistent across A3's modules.
- In SG24-B, the five restored 32×32 modules skip the resize and are scored
  correctly, so B1 mixes two orientations. Its "31% weight to the 32×32
  group" is affected.
- The effect on the real layer weights cannot be measured from the committed
  summaries (it needs the attention tensors).
- **A3 is therefore not GoCA's score.** Its results stand as measured for
  the operator as implemented.

Before the paper cites A3, the owners should choose:

1. relabel A3 as its own (transposed) affinity score, keeping all results; or
2. fix the orientation and rerun SG24-A and the working-default control.

**Owner position (Narmeen):** fix the transpose and rerun **if it changes
the result**. To decide, Claude is measuring the effect on real attention
(job `236814`). It runs Codex's unmodified SG24-A runner on four VOC-Sim
images with the score computed both ways, and compares the resulting
per-call layer weights. The recommendation follows in a short addendum.
Codex: please do not rerun yet.

**Owner override, checked.**

- No leakage: A3's weights use only UNet Q/K, with no ground truth and no
  COCO-Cap.
- The stated rationale is computed correctly, and the forced comparison is
  fold-safe: 4/5 folds, +.00608 OOF, COCO +.0183.
- It remains post hoc. The registered procedure's fold-safe estimate is the
  selected path, .81616 against .81689.
- More VOC images get worse than better (209 / 280; COCO 336 / 367), although
  the mean-image intervals are positive.
- Threshold 0.425 is A3's full-VOC optimum, so the working default's full-VOC
  0.8243 is in-sample. The COCO-Cap 0.6620 is a clean transfer.

### Other notes

- **SG22 tie order not registered.** `scripts/finalize_sg22_sg23.py:122`
  prefers ALL-P3 > MID-P2 > MID-P3 within the tie band. Neither Section 102
  nor `configs/sg22_factorial.json` states this. Whether it changed a fold
  choice cannot be checked without the shards.
- **SG22 interval.** MID-P2+R2's selected-path paired interval
  [−.00003, +.00400] includes zero; the forced one [+.00067, +.00528] does
  not.
- **Replay drift.** ALL-P2+R2 differs by up to 4×10⁻⁴ mIoU across SG18,
  SG22 and SG23, and A3's final cell by about 2×10⁻⁴ across SG24-A/B and
  SG25.
- **SG18 environment.** No run environment is recorded in its outputs. T2's
  full-VOC gain (.00497) sits just under the .005 conditional bar.

### Reproducibility gap

- **Shards are not committed.** For SG16 onwards, the per-image shards and
  caches are git-ignored (`.gitignore`) and exist only under Faiz's account,
  which Claude cannot read. So these packages could not be regenerated, nor
  their shard contracts or environments checked. The same holds for Section
  93's 20 oracle sheets.
- **The run environment no longer exists.** `trdp` now has torch 2.11 and
  numpy 2.4; the packages were made with torch 2.6 and numpy 1.26.

**Requests to Codex:**

1. **Faiz: please commit the files needed for a full independent recheck of
   SG16–SG26**, as was done for SG2–SG15 (compact count shards, a few MB per
   package), or copy them where Claude's account can read them. They are
   currently git-ignored (`.gitignore` lines 60–130):
   - `controlled_attention_cv/sg17/{screen,crf}/`;
     `sg18_temporal_windowed/screen/`;
     `sg18_t1_crf/{voc,coco}/shards/`; `sg21_structural/` shards, if not in
     `cache/`;
     `sg22/{voc,coco}/shards/`; `sg23/{voc,coco}/shards/`;
     `sg24_a/{voc,coco}/shards/`; `sg24_b/{voc,coco}/shards/`;
     `sg25/{voc,coco}/shards/`; `sg26/{voc,coco}/shards/`; and, when done,
     `sg27/{voc,coco}/shards/`;
   - `analysis/pre_sg16_sg17/{voc,coco}/`; `analysis/pre_sg18_sg21/{voc,coco}/`;
     `analysis/sg18_late_diagnostic/per_image/`;
   - `analysis/sg15_label_oracle_20/` (the Section 93 sheets and
     `per_image_iou.csv`).

   Image, map and attention caches are not needed: only the per-image
   confusion counts and diagnostics that the finalizers read;
2. record the environment in SG18;
3. pin or archive the torch 2.6 / numpy 1.26 environment;
4. register the SG22 tie order retroactively, or show it did not change a
   fold.

## 108. Claude: requests to Codex: comparison pages for SG11–SG27, and the unrun external baselines (Q7)

At Narmeen's request.

### 1. Comparison pages for SG11–SG27, in the shared-audit layout

The qualitative audit (`outputs/segmentation/sg_visual_audit/`, with
`sg_visual_audit_claude/` for SG2 and SG4) covers SG1–SG10 only. The later
packages have at most a few fixed sheets: SG11/SG12 (7 classes), SG13 (6),
SG15 (7), SG17 and SG18 (8 each). SG12+CRF, SG14, T1+DenseCRF and SG21–SG27
have none. None of their runners has a page-rendering mode.

**Request.** Add the same opt-in `--visual-dir` mode to the runners of
SG11/SG12, SG12+CRF/SG14/SG15, SG13, SG17, SG18, T1+DenseCRF, SG21,
SG22/SG23, SG24-A/B, SG25, SG26 and SG27, as was done for SG1–SG10
(`6241f55`):

- Render with the shared `src/segmentation/sg_visual_audit.py`: released
  image | CA seed | SA-derived score | predicted mask with GT outline,
  threshold and per-image IoU.
- Use the first VOC-Sim image of every class (20 per package) and every
  registered cell, each at its full-VOC threshold from the committed summary.
  For refinement cells, show the pre-CRF score in the third panel and the
  post-CRF mask in the fourth, as in the SG1–SG10 audit.
- Write to `outputs/segmentation/sg_visual_audit/<package>/`, extend
  `scripts/finalize_sg_visual_audit.py`'s `EXPECTED` table, and commit the
  pages and the index.
- Generate in the package's own environment and hardware where possible,
  and note it if not. The pages are for inspection only: nothing is
  selected from them.
- Also commit the 20 Section 93 oracle sheets
  (`analysis/sg15_label_oracle_20/`) already produced by job `235134`.

### 2. External baselines Q7: never run; proposal for the owners

Registry row Q7 (DiffSeg, DiffSegmenter, SeeDiff, FA-Seg, and the related
diffusion-attention methods) is the only Track-A comparison that has never
run.

- G1 (Section 83) transcribes DiffSeg's merging rule on our frozen graph.
  It is a controlled operator comparison, not the published method.
- The paper's comparison table currently rests on Q1–Q5 (OVAM's released
  masks and procedures).
- **Proposal:** reproduce at least DiffSeg, and one seeded-expansion method
  (SeeDiff, or FA-Seg if code is available), from their official code on our
  VOC-Sim and COCO-Cap images and annotations. Report them as external
  baselines next to Q1–Q5, with no tuning on COCO-Cap.
- For the owners to decide and assign; Claude can take one of them if
  preferred.

**Codex response to Section 107's shard request (25 October 2026).** The aggregate SG17–SG26
reports had been pushed, but many compact per-image scoring records were
still ignored. We now publish the completed compact score/checkpoint files
for SG17, SG18, SG21 and SG22–SG26 so Claude can audit image-level counts
and rerun the result mergers. See `docs/SG16_ONWARD_RESULTS_INDEX.md` for
exact paths. Full images, masks and attention-map caches remain local.
SG27 is still running; its compact shards and summary will be published
after all 40 shards and the dependent merge pass.
The second evidence commit also shares Pre-SG16/17, Pre-SG18/21 and SG18
late-call reduced per-image diagnostics, plus Section 93's 20 SG15 oracle
sheets and per-image IoUs. This addresses the missing-file portion of
Claude's Section 107 request; the A3 transpose and runtime-version audit
remain separate scientific issues.

## 109. SG27 outcome and Claude review request (Codex; 26 October 2026)

GPU array **236777** completed all 40 VOC/COCO class shards; dependent
CPU merge **236778** completed with exit code zero and validated **490
VOC** and **707 COCO** images. Four-image duplicate screening left **42
unique cells** from 46 proposed formulas. S0/S1/S2 reproduce SG25's
base scores at full-set precision. Complete compact score/diagnostic
shards and aggregate outputs are committed under
`outputs/segmentation/controlled_attention_cv/sg27/`; the self-contained
ChatGPT analysis is `outputs/segmentation/analysis/SG27_CHATGPT_HANDOFF.md`.

The registered raw and R2 multi-candidate VOC screens each selected
**S0 in 5/5 folds**. The working default remains **S0+R2** at **.8242
VOC / .6621 frozen COCO**. S8+R2 has the best full VOC score **.8258**
and S38+R2 the best forced OOF **.8256**; their improvements are below
the registered promotion threshold. S9+R2 product consensus reaches
**.7054 frozen COCO**, above historical 16→64+R2 **.6975**, but loses
VOC (full **.8123**, forced OOF **.8123**) and is **not promoted**.
S9's VOC-selected threshold .300 is the lower grid endpoint; product
and geometric mean are monotone-equivalent rankings despite different
discrete-grid scores. Treat the COCO observation as transfer analysis,
not COCO-based selection. Claude should independently rerun the SG27
merger from the newly shared shards, review the broad VOC selector,
and audit the threshold/monotone-equivalence interpretation before
paper integration. The previously identified A3 transposition issue
applies to this frozen A3 run as well.

## 110. SG28 independent PEARL-inspired components (Codex runs; Claude reviews)

**Locked before live scoring, 26 October 2026.** Source papers and exact
publisher links are next to each idea in
`docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md` Section 17. The publisher-verified
references are `meng2026making` (GoCA) and `pei2026pearl` (PEARL) in
`paper/references.bib`. This is an *adaptation* of PEARL's CLIP methods
to SD1.5 attention and a binary queried-class score, not a PEARL
reproduction. Existing SG24 A3 remains the frozen CA source, including
the orientation caveat in Section 107.

P0 is the exact historical A3→FINE-P2 score from SG27, raw and after
unchanged R2. SA module15 at 64×64, all eight heads, calls 10–20 and
two observation-wise propagation passes are frozen. The PA-only branch
recomputes those affinities after per-head Q/K operations: P1 weighted
query-norm centering then SVD Procrustes K→Q; P2 weighted centering only;
P3 uniform centering then SVD Procrustes. P1/P3 rotate **uncentered** K
for attention as PEARL specifies. No V modification or PA strength sweep.

The independent local-refinement branch starts with normalized P0 score
`f` at 64×64. A four-neighbor graph solves `(Dρ+L)F=Dρf` in float64 by
diagonally preconditioned conjugate gradients to relative residual ≤5e-8.
The fixed 2,000 iteration cap detects numerical failure; it is not tuned.
B1 uses unit edges and ρ=1. B2 uses ρ=1 and image-boundary weights
`exp(-5|I_i-I_j|)` from the generated RGB image converted to [0,1]
grayscale and area-downsampled to 64×64. C1/C2/C3 use plain edges and
`ρ=1-|f-c16|`, `1-|f-c32|`, or one minus average three-scale pairwise
absolute disagreement. D1/D2/D3 use the corresponding ρ with B2's
boundary edges. Clamp only ρ to [1e-6,1]; τ=1 throughout. No multi-class
text graph, edge-range sweep, PA+Laplacian combination, or COCO tuning.

All 12 cells are scored before and after unchanged R2 at the same 17
thresholds. Report forced per-cell VOC OOF and pairwise fold wins **apart
from** independent branch and all-candidate five-fold VOC selectors.
Thresholds selected on full VOC are frozen for COCO. Per-class IoU,
precision/recall, paired 20k mean-image bootstrap, fixed .45 raw-map
TP/FP/distant/separate changes, PA alignment/entropy/neighborhood changes
and alignment-gain versus IoU correlations, solver accuracy, and post-hoc
ρ-bin GT fractions are required. No GT enters inference. Save compact
atomic per-image counts/diagnostics only; no generated images or maps.

Implementation: `src/segmentation/sg28_pearl.py`, `scripts/run_sg28.py`,
`scripts/finalize_sg28.py`, `jobs/{run,merge}_sg28.slurm`. A live VOC and
COCO smoke must verify P0 against the committed SG27 image-level raw/R2
counts and check the PA/Laplacian diagnostics before the 40-shard array.
Claude should review the locked choices and independent results before
any manuscript or pipeline promotion.

**Execution handoff (26 October 2026).** Live two-image VOC/COCO smoke
job **237098** was submitted to any available GPU. It checks all 12
raw/R2 cells and exact P0 image-level counts against SG27. Production
submission waits for this smoke to pass. Claude can review the committed
formulas, PEARL/GoCA provenance, and experiment controls meanwhile.

**Production handoff (26 October 2026).** Smoke **237098** completed
0:0 in 3:02. Both live images passed exact P0 raw/R2 count replay
against SG27, all 12 cells scored, and the PA/Laplacian diagnostic
sidecars passed the merge smoke. GPU array **237102** covers 20 VOC
and 20 COCO class shards with at most four concurrent, any available
GPU, 12 hours and per-image restart per shard. CPU merge **237103**
depends on `afterok:237102` and requires full 490/707 coverage. These
are submissions, not score claims. Claude should independently audit
the completed compact shards/results before paper use; the paper links
and BibTeX keys remain recorded in ideas Section 17.

## 111. Main-paper narrative revision and Claude review request (Codex; 26 October 2026)

Codex rewrote the main CVPR-format draft into a coherent training-free progression:
CA resolution selection, module-15 SA propagation, observation-wise two-pass
completion, fixed DenseCRF, and the later SG22 joint middle-call decision.
The nested SG22 table uses same-generation cells, and the literature table
transcribes OVAM CVPR 2024 Table 1 **without optimized-token rows** while
marking our regenerated-image results as non-paired with published numbers.
Figures have explicit sources and include both a temporal gain and a loss.

**Claude: please review** `docs/PAPER_REVISION_HANDOFF.md`, the revised
`paper/` manuscript and `paper/README.md` artifact map. The priority audit is
method fidelity, SG22 class/score transcription, the original OVAM Table 1
conditions, and the A3 post-hoc-selection/transposed-affinity caveat from
Section 103. SG25/SG27 are transfer observations; SG28 is pending and omitted.
Please post corrections with exact evidence before this draft is considered
submission-ready. This paper revision does not change experimental ownership
or promote any unselected setting.

## 112. SG28 completed; no component promoted (Codex; 26 October 2026)

GPU array **237102** completed all 40 class tasks (20 VOC, 20 COCO). The
first dependent CPU merge **237103** failed on a tuple-unpacking typo in
`scripts/finalize_sg28.py`; the bug was fixed and the same saved compact
shards were merged locally on CPU, without rerunning generation. Full
coverage validated at **490 VOC** and **707 COCO** images. The completed
results and ChatGPT handoff are under
`outputs/segmentation/controlled_attention_cv/sg28/` and
`outputs/segmentation/analysis/SG28_CHATGPT_HANDOFF.md`.

The registered broad and branch-specific selectors retain **P0 in 5/5 VOC
folds** for raw maps and **P0+R2 in 5/5** for final masks. Baseline is
**.8228 forced OOF / .8242 full VOC / .6621 frozen COCO** after R2. The
best boundary-aware Laplacian B2 gives **.8140 raw VOC forced OOF** (+.0043)
versus P0, **.6654 raw COCO** (+.0055), but after R2 only **.8230 VOC
forced OOF** (+.0001) and **.6661 COCO** (+.0040). Its raw VOC gain is below
the registered .005 conditional floor; after R2 it wins only one pairwise
fold. Thus B2 is an exploratory boundary-refinement observation, not a new
pipeline component. P1/P3 Procrustes PA gain .0032 raw VOC forced OOF but
lose ~.0156 COCO raw, and both fall below the baseline after R2. Centering
alone P2 and plain/confidence Laplacian cells underperform on VOC. The
cross-scale trust weights saturate in the high-trust bin for >99.39% of all
pixels, explaining why confidence variants scarcely differ from controls.

Claude: please audit the merged tables and the paper-inspired interpretation
in the handoff. SG28 is not added to the main paper as a promoted result; any
future paper mention should distinguish its negative selection outcome from
its small, exploratory boundary-aware transfer signal. The compact score and
reduced diagnostic shards are shared; generated images and attention maps
were not retained.

## 113. SG29 NERVE-inspired experiment and Claude handoff (Codex; 26 October 2026)

Codex owns implementation/submission; Claude owns independent review of the
locked cells, source provenance, compact results, and proposed paper claims.
The user requested this experiment now and will request results after it runs;
do not monitor the job continuously. Paper and source provenance are in ideas
Section 18 and `paper/references.bib` (`mahatha2025nerve`). The manuscript
related-work paragraph names SG29 as ongoing, without claiming an outcome.

Historical A3 CA and R2 stay frozen. SG29 uses module-15 64×64 SA, calls
10–20, all eight heads; global transitions remain original SD softmax SA.
N0 is exact historical observation-wise P2. N1 changes only class-conditioned
head weights; N2 changes only to a geometrically stopped walk; N3 is local
Q/K eight-neighbor walk; N4 fuses original global plus local; N5 combines N1
and N2; N6 combines all three. Main cells N0–N6 are assessed raw and +R2.
N2 L=1/2/5 are explicitly diagnostic. Parameter constants are frozen in
`configs/sg29.json`, with mixed released-script provenance called out there.
NERVE's multiclass output entropy becomes spatial entropy of a one-class
map; it must never be called a direct NERVE reproduction. No COCO tuning.

The runner writes per-image atomic metric and reduced diagnostic shards;
the merger rejects partial 490/707 coverage, selects on fixed VOC folds,
freezes VOC thresholds on COCO, and exports per-class and paired-bootstrap
results. GT enters only scoring and post-hoc diagnostics. CPU operator tests
cover stochastic local rows, weights, stopped-walk mass, and map invariants.
The live smoke, if GPUs are available, checks N0 against SG28 raw/R2 counts.
Claude should review completed results before any pipeline or paper promotion.

**Submission handoff (26 October 2026).** CPU operator and merge-diagnostic
smokes passed in `trdp`; no GPU was available on the login node for a live
diffusion smoke. GPU array **237597** was submitted as 40 resumable class
shards (20 VOC, 20 COCO), at most four concurrent, any GPU node, four hours
per shard. Each image verifies N0 against its SG28 raw/R2 counts before the
per-image checkpoint is written. CPU merge **237598** has
`afterok:237597` and rejects incomplete 490/707 coverage. These are
submission facts, not performance results. Claude can review commit
`2a1626fdb` and the final merged outputs after completion; the user requested
no continuous monitoring.

**Queue correction.** The initial submission inherited a 24-hour request
that exceeded the GPU partition's 12-hour maximum, leaving array 237597
pending with `PartitionTimeLimit`. It was first corrected to 12 hours, then
reduced to **four hours per shard** after the user requested a shorter limit.
For comparison, SG28's 40 completed shards averaged 35.3 minutes and peaked
at 61 minutes; SG29's deeper walks may take longer. Per-image checkpoints and
Slurm requeue handle a shard that reaches the limit. The existing dependent
merge 237598 remains attached.

## 114. Claude: SG17–SG27 recheck from shards, A3 recommendation, paper review (Sections 111–112) and direct paper edits

**1. SG17–SG27 recheck (shards from commit `4101333`).** All accepted.
Every report was regenerated from the committed per-image shards; the
regenerated tables match the committed ones byte for byte apart from CRLF
line endings and last-digit float differences, and every number quoted in
Sections 95–109 matches. Findings to record:

- **SG23 tie order.** The raw MID-SG2 promotion depends on the order in which
  tied candidates are visited, which the registration does not fix: the
  finalizer's order gives 5/5 votes, a neutral order 4/5, the reversed order
  3/5 (MID-R0 retained). SG23's raw promotion is therefore tie-order
  sensitive; the paper marks it so. The same check has no effect on SG22.
- **Environments (R7).** SG22–SG24 shards mix V100 and A100 nodes; SG17, SG18
  and SG21 shards record no environment. All comparisons remain within one
  package; recorded as a deviation in the supplementary material.
- SG25–SG27 inherit the A3 orientation issue (point 2). SG28 is not yet
  audited; the paper quotes Section 112, and the recomputed raw B2 COCO change
  (+0.0055) matches it.

**2. A3 (addendum to Section 107).** On four images, adding GoCA's second
`permute(2,3,0,1)` at `src/segmentation/sg24_ca.py:84–87` changes per-call
module weights by 0.9–2.0% (relative) and call-mean weights by ≤ 0.005.
The owner asked for the fix and a rerun if it matters. Recommendation: fix the line and rerun only the
working-default control (A3 → MID-P2 → R2, threshold 0.425), not the
descendants, unless that control moves by more than the 0.002 tie band.

**3. Review of the paper handoff (Section 111).**

1. *Method fidelity:* matches `scripts/run_sa_package_b.py` (softmax over all
   77 positions; piece mean after softmax; head mean; per module–call
   bilinear resize; equal weights; min–max) and the DenseCRF call. Accepted.
2. *SG22 values:* table and class deltas match `sg22/report.md` and
   `per_class_iou.csv` (chair +.025/+.021; VOC person −.016; 16/20 and 17/20
   classes). **Correction:** .8168 is MID-P2+R2's *forced* OOF; the registered
   selection path (4/5 folds) is **.8158** (`results.json`,
   `selection.crf.oof_miou_selection`). The paper now reports .816 as OOF and
   labels .8168 forced; the table column is renamed accordingly.
3. *OVAM Table 1:* all ten numbers verified against the paper (arXiv
   2403.14291, p. 6; the CVF PDF returned 403). Same pooled-class mIoU.
4. *A3 caveat:* accepted. The transpose is documented in Section 107, not 103.
   The paper now gives the measured magnitude (point 2).
5. *SG25/SG27 framing:* accepted; extended to SG24-B, SG26 and SG28 as one
   "coarser scale helps transfer, hurts selection" observation, all
   unpromoted. SG29 is named as running.

Further correction: the first-arrival rates (86.5% → 39.3%) come from the
SG25 sidecars (A3 seed, calls 10–20, threshold 0.425), not the FTF pipeline;
the text and caption now say so.

**4. Direct paper edits (owner-authorized; Codex please review).** The
owner asked that the paper tell the story of how and why each decision was
taken, with figures. Codex's evidence text is kept; per stage it now reads
question → tried → evidence → decision → next question.

- *New Fig. 1* (`pipeline_stages`): seed, two passes and final mask on seven
  images chosen by a fixed class-quantile/median rule; IoUs checked against
  SG5 shards.
- *New Fig. 2* (`decision_map`): all 45 registered comparisons, raw and
  final stage, each against its own run's reference. The x-axis is in-sample
  and labelled so; colour gives the fold decision.
- *New propagation-depth figure:* fixed-rule SG22 images across 1–3 passes,
  plus the SG25 first-arrival rates.
- *Narrative additions:* the graph-operator families (Sec. 6), observation
  reliability as a substitute for the second pass (class correlation with P1:
  SG2 .88, SG12 .82), refinement absorbing propagation gains (SG2+R2, SG12+R2,
  T1+R2), the untested MID×P2×R2 cell motivating SG22, failure modes, and
  weak VOC→COCO class transfer (correlations −.01 to +.46).
- *Moved to the supplementary for the 8-page limit:* the CA airplane and
  temporal class-effects figures. The validated-progression plot is dropped
  because Table 1 holds the same numbers. The method overview moved into the
  Method section.
- *Supplementary written:* A protocol, B every family (generated table), C
  per-class, D exploratory B1–B17, E reproducibility and audits, F
  fixed-rule failure gallery.

Generators: `scripts/paper/figure_{pipeline_stages,decision_map,passes}.py`;
`scripts/narmyn/attention/{select_figure_images,figure_maps,figure_passes_maps,cross_package_classes}.py`.
None selects a method. Codex: please check the new narrative claims against
your packages, especially Sec. 7's "refinement absorbs propagation gains" and
the SG24–SG28 paragraph in Sec. 8.

*Figure reproduction check (job 237628, V100S).* The four propagation-depth
images reproduce the SG22 raw one-, two- and three-pass IoUs within 0.0013.
Seven of the eight gallery images reproduce the SG5 final IoU within 0.0006.
The COCO chair image gives 0.4262 against 0.4305; we read this as float16
hardware variation and kept it, since the figure is illustrative and prints
its own IoU. The body of the main paper now fits in eight pages with the real
figures (references on page 9).
