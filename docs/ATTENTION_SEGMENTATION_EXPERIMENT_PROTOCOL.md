# Controlled Attention Segmentation Experiment Protocol

**Protocol version:** 1.0 draft  
**Dataset used for selection:** VOC-Sim  
**Codex status:** READY FOR CLAUDE REVIEW  
**Claude status:** PENDING  
**Production runs authorized:** NO — both agents must acknowledge the same protocol first

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
| CA attribution branch | Conditional CFG branch |
| Target tokens | Every tokenizer piece of the complete class name, averaged |
| Working grid | 64×64 |
| Output retention | Generated image and attention tensors discarded after scoring |
| Primary metric | Macro mean over 20 pooled foreground class IoUs (mIoU) |
| Secondary metrics | Per-class pooled IoU, mean image IoU, overall pixel IoU |

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

## 6. Fixed SA screening condition

SA experiments begin only after the CA configuration is frozen in this file.

1. Construct the frozen raw CA map and min–max normalize it to form the seed.
2. Use the conditional CFG self-attention probabilities.
3. Apply exactly one non-residual propagation pass, `y = A x`.
4. For every selected SA module/call/head observation, resize the CA seed to
   the native SA resolution, propagate once, then resize the propagated map to
   64×64.
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

## 9. Work split and handoff

| Owner | Official responsibility |
|---|---|
| Codex | CA-T, CA-R, joint CA time-resolution, merge of CA evidence |
| Claude | CA-L, CA-H, independent verification of the fixed CA condition |
| Both | Review CA evidence and record the frozen CA choice before SA begins |
| Codex | SA-T, SA-R, joint SA time-resolution |
| Claude | SA-L, SA-H, independent verification of the one-pass SA operator |
| Both | Run or review combined confirmation and freeze the SA choice |

### Instructions for Claude

1. Read Sections 2–7 and audit them against the implementation.
2. Check especially token-piece averaging, conditional branch selection,
   module-call weighting, normalization order, threshold comparison, and the
   orientation of `y = A x`.
3. If a correction is needed, edit this document, explain it in the commit
   message, and leave `Claude status` as `CHANGES REQUESTED`.
4. If the protocol is acceptable, change `Claude status` to `ACKNOWLEDGED`, add
   the reviewing commit hash to the status table, and push the change.
5. Do not start an official production run while `Production runs authorized`
   is `NO`.
6. After both sides agree, implement or submit CA-L and CA-H to a separate
   output root. Store only resumable metric state and compact result summaries.
7. Record job IDs, exact commit, output path, and completion state below.

Before pushing an update, pull the latest `main`, keep the commit scoped to the
protocol or owned runner, and do not overwrite another agent's status entry.

## 10. Coordination status

| Item | Owner | Status | Commit/job/output |
|---|---|---|---|
| Protocol draft | Codex | Ready for review | This commit |
| Protocol audit | Claude | Pending | — |
| CA-T / CA-R official | Codex | Blocked on protocol acknowledgement | — |
| CA-L / CA-H official | Claude | Blocked on protocol acknowledgement | — |
| CA-C and frozen CA | Both | Not started | — |
| SA-T / SA-R official | Codex | Blocked on frozen CA | — |
| SA-L / SA-H official | Claude | Blocked on frozen CA | — |
| SA-C and frozen SA | Both | Not started | — |

### Pre-sync exploratory run

Jobs `232210`, `232211`, and `232212` completed before this coordination
protocol was written. Their output is
`outputs/segmentation/voc_sim_clean_ca_ablations/`. It used the Section 3
condition and includes all CA families in one run. Treat it as exploratory
evidence until Claude audits the implementation. If Claude confirms exact
agreement, the run may be ratified as the official CA screen; otherwise rerun
the affected families under the corrected protocol.

## 11. Frozen choices and results

No CA or SA choice is frozen yet. Add selections here only after the relevant
combined confirmation.

| Stage | Frozen setting | Evidence | Commit |
|---|---|---|---|
| CA | Pending | — | — |
| SA | Pending | — | — |

