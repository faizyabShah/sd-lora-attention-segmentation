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
| SA confirmation | Claude runs; both review | Evaluate the combined shortlist under the folds and freeze SA | `outputs/segmentation/controlled_attention_cv/sa_confirmation/` |

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
| CA-C and frozen CA | Claude runs (taken over from Codex, see Section 25); both review | In progress (claimed) | `outputs/segmentation/controlled_attention_cv/ca_confirmation/` |
| SA-T / SA-R / joint official | Codex | Blocked on frozen CA | `outputs/segmentation/controlled_attention_cv/sa_time_resolution/` |
| SA-L / SA-H / operator check | Claude | Runner and merge ready and verified (Section 24); blocked on frozen CA | `outputs/segmentation/controlled_attention_cv/sa_layer_head/` |
| SA-C and frozen SA | Claude runs; both review | Not started | `outputs/segmentation/controlled_attention_cv/sa_confirmation/` |

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

No CA or SA choice is frozen yet. Add selections here only after the relevant
combined confirmation.

| Stage | Frozen setting | Evidence | Commit |
|---|---|---|---|
| CA | Pending | — | — |
| SA | Pending | — | — |


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
