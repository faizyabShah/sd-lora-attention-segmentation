# JC review workplan

**Purpose.** Shared working document for answering Juan Carlos's review of the
BRACE draft (email of November 2026, comments #1–#9) before the meeting, and for
the paper revision that follows. Both sides work from this file and audit each
other's work here.

**Sides.**

- **F** = Faizyab's agents (Claude on the SLURM cluster and Codex on the A40
  machine, counted as one side).
- **N** = Narmeen's agent (Claude under Narmeen's account).

**Status of this file.** Sections 1–5 are the agreed plan and change only by an
explicit amendment entry in Section 7. Section 6 is the status board, edited in
place by the owner of each line. Section 7 is the append-only log.

This file replaces `docs/ATTENTION_FUSION_AND_TOKEN_IDEAS.md` (last edited
29 October 2026) as the place where work is split. The scientific protocol in
`docs/ATTENTION_SEGMENTATION_EXPERIMENT_PROTOCOL.md` and the rules in
`AGENTS.md` still apply in full.

---

## 1. What JC asked for

| # | Comment, in short |
|---|---|
| 1 | Frame the contribution around cluttered scenes (COCO-Cap, hierarchy +0.034), not "+12.5 points over OVAM". The two new components add +0.006 each on VOC-Sim; most of the gain is configuration. Handle carefully that both components were adopted after the selection rule kept the simpler reference. |
| 2 | A table mapping each of the four contributions (pipeline, controlled ablations, oracle threshold analysis, fold-safe protocol) to its section, table, figure and key number. |
| 3 | A paired comparison on the same images next to the published numbers: our OVAM baseline (plus optimized tokens, flagged as another setting), GoCA, and NERVE if possible. PEARL cited for positioning only. |
| 4 | BRACE seeded with OVAM's optimized tokens, as a separate "optimized token" block. Check which images the tokens were optimized on. State that this block is not training-free. |
| 5 | Fine-only, BRACE and GoCA on SD 1.5, SD 2.1 and SDXL, on the existing images, adding noise and reading attention in one denoising pass. SD3/FLUX declared out of scope. |
| 6 | Consistency: one estimate throughout (0.829 vs 0.830, 0.696 vs 0.697); no raw and DenseCRF values in one comparison; "Eq. ??" in Fig. 2; citations for Attn2Mask, Grounded Diffusion, DatasetDM; supplementary pointers. |
| 7 | Table 1: COCO-Cap values for the first three rows, a COCO-Cap Δ column, folds won per row, explicit reference for every Δ, and the hidden .691 → .789 jump made visible. |
| 8 | Rename Sec. 5 "Experimental results", move Sec. 6 into it; hardware and software details and the per-image runtime and memory cost in Sec. 4. |
| 9 | Target ICPR 2027: paper registration 1 March 2027 (author list fixed), submission 8 March 2027. |

Also promised in our reply: a slide deck with the possible actions per comment,
and the SCLIP-seeded result as an extra discussion item.

---

## 2. Division of work

The split is **our method (F) versus comparison methods and manuscript (N)**.
Whoever produces a number also writes its table, caption and slide.

| JC item | F: Faizyab's agents | N: Narmeen's agent |
|---|---|---|
| **#1 Claim** | Gain breakdown on VOC-Sim and COCO-Cap (table and slide) | Reframed abstract and introduction around cluttered scenes, from numbers already in the draft |
| **#2 Mapping** | — | Contribution-to-evidence table, from the current draft |
| **#3 Reruns** | OVAM class word; OVAM optimized tokens (few-shot block); DAAM; SeeDiff; SCLIP + DenseCRF; BRACE rows | GoCA; NERVE; DiffSegmenter; SPARK |
| **#3 Published block** | — | Attn2Mask, Grounded Diffusion, DatasetDM numbers and citations; which benchmarks GoCA reports; PEARL positioning |
| **#4 Optimized tokens** | BRACE seeded with OVAM's released tokens | Token provenance: which images the tokens were optimized on |
| **#5 Backbones** | Fine-only and BRACE on SD 1.5, SD 2.1, SDXL | GoCA on SD 1.5, SD 2.1, SDXL |
| **#6 Consistency** | — | All of it |
| **#7 Table 1** | Whole item: missing COCO-Cap values, COCO-Cap Δ, folds won, explicit Δ references, new table and caption | — |
| **#8 Structure and cost** | Runtime and memory per image; hardware and software paragraph | Section restructuring |
| **#9 Venue** | — | ICPR 2027 dates, page limit, template and anonymity rules |
| **Slides** | #4, #5, #7, cost, SCLIP-seeded result, own #3 rows | Deck skeleton; #1, #2, #6, #8, #9, own #3 rows |
| **Audit of the other side** | Traces every number in N's tables and text to a committed result | Audits F's experiment entries |

### 2.1 Why the split looks like this

- **No shared code or files.** F does not touch GoCA, NERVE, DiffSegmenter,
  SPARK or the manuscript body. N does not touch the BRACE pipeline.
- **OVAM sits with the optimized-token experiment.** F needs the released tokens
  for #4 anyway, so one download and one harness serve #3 and #4.
- **GoCA sits with its backbone runs.** One owner ports it once across the three
  backbones.
- **The leakage check is independent.** The side that runs the optimized-token
  experiment is not the side that certifies the tokens did not see our
  evaluation images.
- **N's writing items need only the current draft.** #1, #2, #6, #8 and #9 can
  start immediately and wait for no new run.

### 2.2 Details per item

**#1 Claim.**
- F: one table per benchmark listing each step from the OVAM-style baseline to
  BRACE with its paired Δ, split into "configuration" (CA scales, SA passes,
  call window, DenseCRF) and "new components" (SA-guided layer weighting, capped
  hierarchy). Uses existing committed results; no new method runs. Reuses the
  #7 numbers.
- N: draft abstract, introduction and contribution list around the
  cluttered-scene claim. The text must state plainly that the registered
  selection rule kept the simpler reference and that the final method is an
  owner override (protocol Section 135), not a registered promotion.

**#2 Mapping.**
- N: one row per contribution with section, table, figure, key number, and
  whether the support is in the main paper or only in the supplementary. Add a
  list of results that support no contribution (candidates to shorten or move).

**#3 Comparison methods.**
- Every method is rerun **from its own public repository** as a method to
  compare with, on the shared benchmark. Our older pipeline-specific
  adaptations (for example the SG29 NERVE-inspired and SG31 SeeDiff-inspired
  studies) are **not** reused as comparison rows.
- First step per method is a **go / no-go** entry: does public code exist, does
  it run on SD 1.5 with a class-word query, and can it produce a per-class soft
  map or mask on our images? A no-go moves the method to the published block
  with a one-line reason.
- F methods read attention **during generation**. They count as "same images"
  only if the rerun reproduces our prompts and seeds; otherwise the row is
  flagged as "same prompts, regenerated".
- N methods take a **finished image** and noise it; they run directly on the
  stored benchmark images.
- N also owns the published block: numbers exactly as printed, with table and
  page of the source, marked "not paired".

**#4 Optimized tokens.**
- F: run frozen BRACE with OVAM's released optimized token replacing the class
  word in the cross-attention seed; nothing else changes. Report next to the
  class-word block, same capture for both blocks. Pre-register before scoring.
- N: find which images each released token was optimized on, and whether any
  overlap the VOC-Sim or COCO-Cap evaluation images. Report the overlap per
  class.
- Starting position, for the record: optimized tokens gave +0.051 on an older,
  simpler pipeline. BRACE itself has never been run with them. No result is
  expected in either direction.
- If overlap is found, F reports #4 with the caveat and reruns on the clean
  subset.

**#5 Backbones.**
- Setting: stored benchmark image → encode → add noise at a fixed timestep →
  one denoising pass → read attention. SD 1.5 is rerun in this setting as the
  reference; it is a different setting from the generation-time results and is
  never mixed with them in one comparison.
- F: fine-only and BRACE. SD 2.1 first, SDXL second. SDXL has two attention
  scales, so the hierarchy has one step; the mapping of the frozen method onto
  each backbone is written down and committed before scoring.
- N: GoCA on the same three backbones, same images, same noise seed.
- Pre-register: timestep, noise seed, layer mapping and threshold rule.
- SD3 / FLUX: out of scope; N writes the scoping sentence.

**#6 Consistency.**
- N: every item JC listed, plus a list of each numeric claim in the paper with
  its source file. F audits that list.

**#7 Table 1.**
- F: all rows on both benchmarks from one capture; each Δ with its explicit
  reference row; folds won (x/5) per row; the SA-propagation jump shown as its
  own row. If a row cannot be produced in the same capture, it is marked so and
  not given a paired Δ.

**#8 Structure and cost.**
- F: wall-clock time and peak GPU memory per image for BRACE and for fine-only,
  on a named GPU, measured on a fixed sample; hardware, driver and library
  versions.
- N: restructuring of Sections 4–6 as JC described.

**#9 Venue.**
- N: confirm ICPR 2027 dates, page limit, template and anonymity rules from the
  conference site; note what the current CVPR-style layout must change.

---

## 3. Day-one contract

Fixed once so that neither side waits for the other. F drafts it as Entry 1;
N reviews it; it changes afterwards only by an amendment entry.

| Item | Rule |
|---|---|
| **Images** | The VOC-Sim and COCO-Cap evaluation images, masks and manifests from the public OVAM archive (`ovam_experiment_with_dataset.zip`; download and extraction in `docs/NEW_MACHINE_SETUP.md`, "Datasets"). Both sides can fetch it; nothing needs to be handed over. |
| **Image list** | A committed manifest per benchmark (image id, class, prompt, seed where known), produced in Entry 1. Every method is scored on exactly this list. |
| **Query** | The class word only. VOC-Sim and COCO-Cap class-name conventions as in the manifest. No per-class prompt tuning for any method. |
| **Output** | One map or mask per (image, class) at the image resolution used for scoring, saved so the other side can rescore it. |
| **Metric** | Class-macro mIoU of the binary foreground mask, with a 2000-sample image bootstrap for paired differences. One shared scoring script, named in Entry 1; both sides use it unchanged. |
| **Thresholds** | Development on VOC-Sim only. COCO-Cap is untuned transfer: a threshold chosen on VOC-Sim is applied unchanged. A method's published default binarisation is reported as well when it has one. |
| **DenseCRF** | Reported with and without, using the frozen R2 settings for every method. Raw and DenseCRF values never share a comparison. |
| **Noise (image-input methods)** | One fixed noise seed and timestep per backbone, stated in Entry 1. |
| **Label use** | Ground truth is for scoring only, for every method. |

---

## 4. Working rules

1. **Append, don't rewrite.** New work goes in a new numbered entry in
   Section 7. Earlier entries are corrected by a later entry, not edited.
2. **Review does not gate work.** The other side's check is required before a
   result goes on a slide or into the paper, not before the next run starts.
3. **Pre-register before scoring.** #4 and #5 and any comparison that involves a
   choice (threshold, layer mapping, timestep) get their rule committed first.
4. **Same capture for paired claims.** Variants compared with a paired Δ come
   from one capture on one machine. Rows from different machines are joined in
   a table only as unpaired rows, and labelled so.
5. **Report failures.** A method that does not run, a test that fails, a result
   that goes the wrong way: all get an entry. Label every result as
   exploratory, in-sample, OOF, fixed-threshold or confirmatory.
6. **The frozen method stays frozen.** Nothing here changes BRACE (A3 seed, V3
   hierarchy, DenseCRF R2, threshold 0.55).
7. **Committed results only.** N cannot read F's git-ignored files. Anything N
   must check has to be in a committed file, cited by path and commit.
8. **No deletions** without the owner's explicit yes. Commit locally; push when
   the owner asks.
9. **Manuscript text** avoids internal names (A3, V3, C16, SGxx). This file may
   use them.

---

## 5. Order of work for the meeting

The meeting is expected within two weeks of JC's email. Items are ordered by
what can be shown, not by comment number.

| Priority | F | N |
|---|---|---|
| **First** | Entry 1 (contract and manifests); #7 Table 1 numbers; #4 pre-registration and run | #2 mapping table; #6 consistency list; go / no-go for GoCA |
| **Second** | #5 on SD 2.1; OVAM class-word and optimized-token reruns; #8 cost | GoCA on SD 1.5; #1 reframing draft; token provenance |
| **Third** | #5 on SDXL; DAAM, SeeDiff, SCLIP rows | GoCA on SD 2.1 and SDXL; NERVE, DiffSegmenter, SPARK; published block |
| **Last** | Own slides | Deck skeleton and own slides; #8 restructuring; #9 |

Anything not finished by the meeting goes on its slide as "planned", with what
is already known.

---

## 6. Status board

Edited in place by the owner. States: `not started`, `in progress`,
`done, awaiting review`, `reviewed`, `no-go`, `blocked (reason)`.

| Item | Owner | State | Latest entry |
|---|---|---|---|
| Contract and manifests | F | not started | — |
| #1 gain breakdown | F | not started | — |
| #1 reframing text | N | not started | — |
| #2 mapping table | N | not started | — |
| #3 OVAM class word | F | not started | — |
| #3 OVAM optimized tokens | F | not started | — |
| #3 DAAM | F | not started | — |
| #3 SeeDiff | F | not started | — |
| #3 SCLIP + DenseCRF | F | not started | — |
| #3 GoCA | N | not started | — |
| #3 NERVE | N | not started | — |
| #3 DiffSegmenter | N | not started | — |
| #3 SPARK | N | not started | — |
| #3 published block | N | not started | — |
| #4 BRACE with optimized tokens | F | not started | — |
| #4 token provenance | N | not started | — |
| #5 fine-only and BRACE, SD 1.5 / 2.1 / XL | F | not started | — |
| #5 GoCA, SD 1.5 / 2.1 / XL | N | not started | — |
| #6 consistency | N | not started | — |
| #7 Table 1 | F | not started | — |
| #8 cost and hardware | F | not started | — |
| #8 restructuring | N | not started | — |
| #9 venue | N | not started | — |
| Slides | F and N | not started | — |

---

## 7. Log

Append entries below, numbered in order. Use this template.

```
### Entry <n> — <side F or N>, <agent>, <date>
**Item:** <JC item and line of the status board>
**Kind:** plan | pre-registration | go/no-go | result | text | amendment | correction
**What was done:** <method, repository and commit of external code, command or job>
**Where:** <committed paths; commit hash>
**Result:** <numbers with their label: exploratory / in-sample / OOF /
fixed-threshold / confirmatory; failures included>
**Open points:** <what is not done or not verified>

**Review — <other side>, <date>:** <checked / objections / requested changes>
```

Rules for the review line: the reviewer states what was checked (recomputed,
read, or traced to a file) and what was not. An objection is answered in a new
entry, not by editing the reviewed one.

<!-- Entries start here. -->
