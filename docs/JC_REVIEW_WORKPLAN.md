# JC review workplan

**Purpose.** Shared working document for answering Juan Carlos's review of the
BRACE draft (email of November 2026, comments #1–#9) before the meeting, and for
the paper revision that follows. Both sides work from this file and audit each
other's work here.

**Sides.**

- **F** = Faizyab's agents (Claude on the SLURM cluster and Codex on the A40
  machine, counted as one side).
- **N** = Narmeen's agent (Claude under Narmeen's account).

**Status of this file.** Sections 1–6 are the agreed plan and change only by an
explicit amendment entry in Section 8. Section 7 is the status board, edited in
place by the owner of each line. Section 8 is the append-only log.

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
   Section 8. Earlier entries are corrected by a later entry, not edited.
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
10. **Every result also goes into the slide deck.** An entry that reports a
    result, a table, a figure or a decision is not complete until its slide is
    added (Section 6). The entry names the slide file it touched.

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

## 6. Slide deck for the meeting

JC asked for a short slide deck with, for each comment, the possible actions.
The agents build it as the work proceeds: **whatever is done for this workplan
is added to the deck in the same commit as its log entry.**

### 6.1 Where it lives

| Path | Content |
|---|---|
| `slides/jc_meeting/main.tex` | Deck skeleton: title slide and the list of section files. Changed only to add a new section file. |
| `slides/jc_meeting/style.tex` | The one style definition. Not changed per slide. |
| `slides/jc_meeting/sections/c<k>_<side>.tex` | Slides for comment `k` by side `F` or `N`. Each side edits only its own files, so there are no merge conflicts. |
| `slides/jc_meeting/sections/extra_sclip_F.tex` | The SCLIP-seeded result. |
| `slides/jc_meeting/figures/` | Images used in the deck, committed (keep each under about 1 MB). |
| `slides/jc_meeting/main.pdf` | Compiled deck, committed after each change so the other side can read it without building. |

Build with `pdflatex main.tex` (twice) inside `slides/jc_meeting/`, on a compute
node, not the login node. The deck is LaTeX Beamer, 16:9. Only packages in
`style.tex` are used (`multirow` is not installed on the cluster).

### 6.2 Appearance rules

- **White background, black text.** No coloured themes, no coloured text, no
  logos, no decoration. Colour appears only inside figures where it carries
  meaning (masks, overlays, plot series).
- **Serif font.** Palatino for text and maths, set once in `style.tex`.
- **Fixed sizes.** Frame title `\Large` bold; body `\normalsize`; tables
  `\small`; captions and source lines `\footnotesize`. Nothing smaller, and no
  per-slide size changes. If content does not fit, split the slide.
- **Tables.** `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`), no
  vertical lines, inside the `slidetable` environment. Numbers right-aligned
  with the same number of decimals in a column (three for mIoU). At most about
  8 rows and 6 columns per slide. The row or number the slide is about is in
  bold. Units and the benchmark are in the header.
- **Images.** Inserted with `\fig{file}` so they are centred, keep their aspect
  ratio and never overflow the slide. Vector (PDF) for plots; PNG or JPG for
  photos and masks. Every figure has axis labels or a one-line caption that
  says what is shown. Side-by-side images use equal widths.
- **Layout.** One message per slide, stated in the frame title as a full
  sentence where possible. At most five bullets, no nested bullets beyond one
  level, no full paragraphs.

### 6.3 Content rules

- **Structure per comment:** (1) the comment in one line, (2) what we did or
  found, (3) the possible actions and the one we recommend.
- **Every number has a source line** (`\src{...}`) naming the committed file or
  the log entry it comes from, and a status tag (`\status{done}`,
  `\status{running}` or `\status{planned}`).
- **Label the kind of result** where it matters: exploratory, in-sample, OOF,
  fixed-threshold or confirmatory. Paired and unpaired rows are never mixed
  without a label. Failures and negative results get a slide like any other.
- **No internal names** on slides (A3, V3, C16, SGxx): JC reads the deck.
- **Review:** a slide counts as final once the entry behind it has its review
  line. Until then its status tag stays `running` or carries `unreviewed`.

---

## 7. Status board

Edited in place by the owner. States: `not started`, `in progress`,
`done, awaiting review`, `reviewed`, `no-go`, `blocked (reason)`.

| Item | Owner | State | Latest entry |
|---|---|---|---|
| Contract and manifests | F | not started | — |
| #1 gain breakdown | F | in progress: step-by-step numbers in Entry 8; configuration vs new-component split not yet written | Entry 8 |
| #1 reframing text | N | done, awaiting review (draft; nothing in `paper/` changed) | Entry 13 |
| #2 mapping table | N | done, awaiting review | Entry 7 |
| #3 OVAM class word | F (run by N on 2026-11-19 at Narmeen's request) | done, awaiting review | Entry 21 |
| #3 OVAM optimized tokens | F (run by N on 2026-11-19 at Narmeen's request) | done, awaiting review | Entry 21 |
| #3 DAAM | F (run by N on 2026-11-19 at Narmeen's request) | done, awaiting review | Entry 21 |
| #3 SeeDiff | F (run by N on 2026-11-19 at Narmeen's request) | done, awaiting review | Entry 21 |
| #3 SCLIP + DenseCRF | F (run by N on 2026-11-19 at Narmeen's request) | done, awaiting review | Entry 21 |
| #3 GoCA | N | done, awaiting review (official paired rows, SD 1.5 and SDXL) | Entries 2–4, 11 |
| #3 NERVE | N | done, awaiting review (official paired rows for A, V, W; A is the class-word row) | Entries 2–4, 11 |
| #3 seed × propagation swap, BRACE and NERVE (cross-side diagnostic; does not replace N's NERVE row) | F | reviewed with issues (Entry 10); corrected, awaiting review | Entries 10, 16 |
| #3 DiffSegmenter | N | done, awaiting review (official paired class-word row) | Entries 2–4, 11 |
| #3 SPARK | N | no-go | Entry 2 |
| #3 published block | N | done, awaiting review | Entry 12 |
| #4 BRACE with optimized tokens | F | done, awaiting review | Entry 8 |
| #4 token provenance | N | done, awaiting review | Entry 5 |
| #5 fine-only and BRACE, SD 1.5 / 2.1 / XL | F (run by N on 2026-11-19 at Narmeen's request) | done, awaiting review | Entry 15 |
| #5 GoCA, SD 1.5 / 2.1 / XL | N | done, awaiting review (SD 2.1-base on a hash-checked community copy) | Entries 3–4, 12, 14 |
| #6 consistency | N | done, awaiting review | Entry 6 |
| #7 Table 1 | F | numbers done (Entry 8; full ladder on both benchmarks by N, Entry 19, at Narmeen's request); paper table and caption not yet rewritten (F) | Entries 8, 19 |
| #8 cost and hardware | F (measured by N on 2026-11-19 at Narmeen's request) | done, awaiting review; paragraph for Sec. 4 ready | Entry 18 |
| #8 restructuring | N | done, awaiting review (proposed outline; applied after #7) | Entry 13 |
| #9 venue | N | done, awaiting review | Entry 12 |
| Slide deck: blank skeleton | F | done | `slides/jc_meeting/` |
| Slides: own sections | F and N | in progress | Entries 2–8 |

---

## 8. Log

Append entries below, numbered in order. Use this template.

```
### Entry <n> — <side F or N>, <agent>, <date>
**Item:** <JC item and line of the status board>
**Kind:** plan | pre-registration | go/no-go | result | text | amendment | correction
**What was done:** <method, repository and commit of external code, command or job>
**Where:** <committed paths; commit hash>
**Result:** <numbers with their label: exploratory / in-sample / OOF /
fixed-threshold / confirmatory; failures included>
**Slides:** <slide file changed, or "none" with the reason>
**Open points:** <what is not done or not verified>

**Review — <other side>, <date>:** <checked / objections / requested changes>
```

Rules for the review line: the reviewer states what was checked (recomputed,
read, or traced to a file) and what was not. An objection is answered in a new
entry, not by editing the reviewed one.

<!-- Entries start here. -->

*Entry 1 is reserved for F's day-one contract (Section 3). N's entries start at Entry 2.*

### Entry 2 — N (Narmeen), 2026-11-16
**Item:** #3 GoCA, NERVE, DiffSegmenter, SPARK (go / no-go)
**Kind:** go/no-go
**What was done:** Cloned each method's public repository and ran a 12-item smoke test (8 VOC-Sim classes, 4 COCO-Cap images) on SD 1.5 with the class word as the only query, using each repository's own settings. Repositories and commits are in `comparisons/README.md`.
**Where:** `comparisons/`; this commit.
**Result:**
- **GoCA (CVPR 2026): go.** Code unchanged, SD 1.5 config as shipped (t=100). Prompt "<class>, background", which is what GoCA's own captioner builds for one entity; the class word alone makes its sum-1 rescaling 0/0. The output is GoCA's per-class soft map after aggregation, rescaling and affinity, before its multi-class argmax.
- **NERVE (WACV 2026): go.** NERVE's own `Pipeline` with its VOC21 ViT-L/14 config, driven without mmseg. With a one-class vocabulary its entropy head weighting divides by log2(1) = 0, so a one-line guard is needed (deviation D1). SD 1.5 instead of the paper's SD 2.1.
- **DiffSegmenter (TIP 2025): go.** Code unchanged. Prompt "a photograph of {class}++". The published method appends a BLIP-large caption; that variant is not run (image-derived text breaks the class-word rule, and the download is pending a decision).
- **SPARK (2026): no-go.** No public code (project page 404, no repository in the paper), and the method is unsupervised: it takes no class query. It moves to the published block.
**Slides:** `slides/jc_meeting/sections/c3_N.tex`
**Open points:** Smoke numbers are not results and are not repeated here.

**Review — F, <date>:** pending

### Entry 3 — N (Narmeen), 2026-11-16
**Item:** #3 NERVE; #5 GoCA, SD 1.5 / 2.1 / XL
**Kind:** pre-registration
**What was done:** Fixed the settings of the full runs before running them. Both notes were written before any full-run output existed and have no amendments.
**Where:** `comparisons/goca/PREREG_backbones.md`, `comparisons/nerve/PREREG_sd15.md`; this commit.
**Result:**
- **GoCA SDXL:** GoCA's `config_model_xl.py` unchanged; 512→1024 bicubic upscale; fp16 VAE with `force_upcast` (no NaN occurred, so no fallback). **SD 2.1: blocked**, because `stabilityai/stable-diffusion-2-1-base` is no longer available on Hugging Face; the planned config and the "!"-padding fix are written down. Using a community mirror needs agreement between F and N, since F's #5 needs the same weights.
- **NERVE:** three variants, all run. A = {class word} + NERVE's background words, head size 40, with the K=1 guard (D1). V = the 20 benchmark class words, head size 40. W = NERVE's released `cls_voc21` names, head size 64 (as released). Timestep index 45 (t=101), seed 0. Deviations D1 (K=1 guard), D2 (head size 64 → 40 on SD 1.5, A and V), D3 (offline model id).
- **All methods:** noise seed 0; thresholds and DenseCRF follow Entry 1's shared scorer once it exists.
**Slides:** `slides/jc_meeting/sections/c3_N.tex`, `c5_N.tex`
**Open points:** Which NERVE variant is the official row (A: class word only, as for every other method; V: native vocabulary) is undecided.

**Review — F, <date>:** pending

### Entry 4 — N (Narmeen), 2026-11-16
**Item:** #3 GoCA, NERVE, DiffSegmenter; #5 GoCA SD 1.5 and SDXL
**Kind:** result
**What was done:** Full runs on the 490 VOC-Sim and 707 COCO-Cap items of `comparisons/manifests/` (stored OVAM benchmark images), on one A100-40GB (node renyi). SLURM jobs 253155 (GoCA SD 1.5), 253160 (GoCA SDXL), 253159 (DiffSegmenter), 253163 (NERVE A, V, W). Every map was rescored independently by `comparisons/verify_runs.py`.
**Where:** `outputs/segmentation/comparisons/n_side/<run>/{sanity_summary.json, per_item_counts.csv}`; this commit.
**Result:** fixed threshold 0.5 on the min-max map, no DenseCRF, class-macro IoU. **Sanity, fixed-threshold; not the official comparison.**

| Run | VOC-Sim | COCO-Cap | s / image (A100) | Peak GPU memory |
|---|---:|---:|---:|---:|
| GoCA, SD 1.5 | 0.671 | 0.452 | 0.14 | 4.4 GB |
| GoCA, SDXL | 0.788 | 0.607 | 0.66 | 13.3 GB |
| DiffSegmenter, SD 1.5 | 0.713 | 0.653 | 0.15 | 8.2 GB |
| NERVE A, SD 1.5 | 0.830 | 0.729 | 0.30 | 11.7 GB |
| NERVE V, SD 1.5 | 0.824 | 0.743 | 0.30 | 11.7 GB |
| NERVE W, SD 1.5 | 0.810 | 0.729 | 0.30 | 11.7 GB |
| Centre-prior control (no model) | 0.318 | 0.225 | — | — |

- All 1197 items per run: no missing, NaN, constant or duplicate maps; no image above 0.995 IoU (no sign of label leakage); median per-image AUROC 0.993–0.998.
- NERVE also uses CLIP ViT-L/14, a second model; it belongs in a separately labelled block next to the SD-only methods.
- NERVE's own binarisation and DiffSegmenter's own threshold rule are in the summaries. GoCA's per-class maps and background channel are saved for its own rule.
- Not the same setting as BRACE's numbers (generation-time attention, tuned threshold, DenseCRF): no paired claim is made.
**Slides:** `slides/jc_meeting/sections/c3_N.tex`, `c5_N.tex`
**Open points:** Official scoring with Entry 1's scorer, thresholds chosen on VOC-Sim, and with/without DenseCRF. The soft maps are local to N (0.7–2.4 GB per run) and can be shared for rescoring. Possible annotation gaps found while checking NERVE overlays (`ovam-voc-sim-50` bird tail, `ovam-coco-687317-cow-478200312` second cow) are listed in `comparisons/nerve/` and were not changed.

**Review — F, <date>:** pending

### Entry 5 — N (Narmeen), 2026-11-16
**Item:** #4 token provenance
**Kind:** result
**What was done:** Matched OVAM's 57 released per-image tokens with the VOC-Sim and COCO-Cap evaluation images by class and generation seed, plus a pixel comparison of the locally held images.
**Where:** `comparisons/notes/entry_4_token_provenance.md`; this commit.
**Result:** The tokens were optimized on 57 separate annotated generated images (2–3 per class) in `voc_sim_train`. **No overlap with any VOC-Sim (500) or COCO-Cap (707) evaluation image.** The #4 block is therefore few-shot (not training-free) but has no evaluation leakage; no clean-subset rerun is needed.
**Slides:** `slides/jc_meeting/sections/c4_N.tex`
**Open points:** none.

**Review — F, <date>:** pending

### Entry 6 — N (Narmeen), 2026-11-16
**Item:** #6 consistency
**Kind:** text
**What was done:** Traced every numeric claim in the main paper and the current supplementary to a committed source, and checked each of JC's #6 items.
**Where:** `comparisons/notes/entry_6_consistency.md`; this commit.
**Result:** 162 claims: 137 traced and matching, 22 differing, 3 not traced. Main fixes: use the v3 run (0.829 / 0.696) throughout; separate raw and DenseCRF values in every comparison; label 0.829 as forced OOF; correct the "objects that fill the frame" failure statement (the record says sheep is the exception); replace ten "(supplementary material)" pointers; rebuild both PDFs (the committed main PDF predates commit 2a954a2). "Eq. ??" could not be reproduced from any committed source.
**Slides:** `slides/jc_meeting/sections/c6_N.tex`
**Open points:** Which PDF JC read. The three proposed citations must be checked against DOI/arXiv before entering `references.bib`.

**Review — F, <date>:** pending

### Entry 7 — N (Narmeen), 2026-11-16
**Item:** #2 mapping table
**Kind:** text
**What was done:** Mapped each of the four contributions in `paper/sec/1_intro.tex` to its sections, tables, figures, key numbers and sources.
**Where:** `comparisons/notes/entry_2_mapping.md`; this commit.
**Result:** Contributions 1–3 are supported in the main paper. Contribution 4 (fold-safe protocol) has no main-paper table or figure, and the registered selector's own outcome (fine-only kept) is not printed. Thirteen results support no contribution and are candidates to shorten or move.
**Slides:** `slides/jc_meeting/sections/c2_N.tex`
**Open points:** none.

**Review — F, <date>:** pending

### Entry 8 — F (Faizyab's Claude), 2026-11-16
**Item:** #4 BRACE with optimized tokens; #7 Table 1 numbers; #1 gain breakdown (numbers only)
**Kind:** pre-registration and result
**What was done:** One live generation per image with the frozen runtime on this cluster (V100 nodes), 490 VOC-Sim and 707 COCO-Cap images. Each generation gives two query columns, class word (frozen attribution) and OVAM's released VOC-Sim token (`configs/ovam/tokens_VOC.npy`; keys `[<SoT>, token]`, conditional query, the form fixed in protocol Section 50), and every pipeline step for both. The layer weights are text-independent, so both columns share them. Jobs 253146–253153. Rules were committed before scoring (95288dc09); one later code change guards empty folds in the report and does not touch scoring (95381a493).
**Where:** `runs/15_token_ladder/PREREG.md`, `runs/15_token_ladder/result.md`, `scripts/jc_ladder.py`; result commit 812405e1c. Per-image maps are in the ignored cache (`jc_ladder/`), not in git.
**Result (fixed threshold 0.55, paired, class-macro mIoU, 95% image-bootstrap intervals):**
- Reproduction: class-word full pipeline 0.830 VOC-Sim / 0.696 COCO-Cap. In-sample for the method design.
- Optimized token, full pipeline: 0.861 / 0.681. Token minus word: +0.031 [+0.022, +0.039] on VOC-Sim, −0.015 [−0.030, +0.000] on COCO-Cap. Few-shot, not training-free.
- Registered comparisons with the token, all three **do not hold**: capped hierarchy vs fine-only (both DenseCRF) +0.010 / −0.017 [−0.028, −0.007]; guided vs uniform layer weights +0.003 / +0.004 [−0.002, +0.011]; cap vs no cap +0.012 / −0.002 [−0.006, +0.003].
- Same comparisons with the class word (rerun, reference): hierarchy +0.034 / +0.012 [−0.002, +0.026]; layer weights +0.004 / +0.024 [+0.013, +0.036]; cap +0.010 / −0.005 [−0.011, −0.001].
- Best token cell on COCO-Cap is fine-only + DenseCRF, 0.698.
- Seed alone: the token raises it by about +0.18 on both benchmarks (0.588 → 0.766, 0.444 → 0.625, uniform layers).
- Regeneration: 238/490 and 243/707 images bit-identical to the stored capture; the rest differ by GPU nondeterminism (class-word maps: median max |Δ| 8e-4 / 3e-3).
**Slides:** `slides/jc_meeting/sections/c4_F.tex` (4 slides), `slides/jc_meeting/sections/c7_F.tex` (2 slides).
**Open points:**
- The draft's COCO-Cap hierarchy gain of +0.034 is not reproduced by this rerun (+0.012 at 0.55, +0.022 with the VOC-Sim-selected threshold). Where the +0.034 comes from has not been traced.
- The COCO-Cap loss with the token is concentrated in a few classes (sofa 0.70 → 0.34, car 0.76 → 0.66; person rises 0.28 → 0.62). Not investigated.
- Paper Table 1 and its caption are not rewritten yet. The configuration vs new-component split for #1 is not written yet.
- Entry 1 (contract) was dropped as a separate task: both sides have the same benchmark data. The scoring rules of Section 3 stand.

**Review — N, 2026-11-19:** confirmed with notes. I traced every number to `runs/15_token_ladder/result.md` and
found no mismatch. The PREREG (`95288dc`, 18:41) precedes every job: smoke at 18:41:40, full jobs 253146–253153 at
18:45:18 per `sacct`, scoring 20:41–21:58. The only later change (`95381a4`) guards empty folds in the report and
does not touch scoring (diff read). The token verdict ("none of H, L, C holds") is the registered rule's verdict.
Notes: (1) under the same rule the class-word rerun also fails H (+0.012 [−0.002, +0.026]) and C (COCO-Cap −0.005
[−0.011, −0.001]); only L holds, and the entry should say so. (2) At 0.55, fine-only is scored away from its
VOC-Sim optimum, so the class-word VOC-Sim H of +0.034 is partly threshold, the mirror of Entry 9. (3) "Best token
cell = fine-only + DenseCRF" holds at 0.55 only. (4) runs/15's DenseCRF grid is 0.05, so its VOC-sel. values are
not comparable with runs/17/20. (5) The PREREG asked for OVAM's published token scores as an unpaired reference;
they are missing. (6) The capture predates the prompt correction (runs/19 addendum), so these numbers wait for
runs/25 before going into the paper. Not checked: the bootstrap intervals and per-image maps (not committed).

### Entry 9 — F (Faizyab's Claude), 2026-11-16
**Item:** #7 Table 1; #1 claim (source of the draft's COCO-Cap hierarchy gain, +0.034)
**Kind:** result (diagnostic, post hoc; nothing selected)
**What was done:** Traced the +0.034 to its source and recomputed it from the Entry 8 capture with DenseCRF on the original 17-threshold grid (0.30–0.70, step 0.025). Jobs 253170–253171.
**Where:** `runs/16_trace_034/result.md`, `scripts/jc_trace_034.py`; this commit. Source of the original number: `outputs/segmentation/controlled_attention_cv/sg_combined_v3/report.md` and `results.json`, protocol Section 135, `configs/owner_default_v3.json`.
**Result:**
- **Source.** +0.0341 = capped hierarchy + DenseCRF at threshold 0.55 (0.6964) minus fine-only + DenseCRF at threshold 0.425 (0.6623). Each map is scored at its own full-VOC-Sim-best threshold, frozen for COCO-Cap. The paper's hierarchy table caption says "COCO-Cap: VOC-frozen threshold" but does not say the two thresholds differ.
- **Reproduced.** Same convention on the new capture: +0.0341 [+0.0227, +0.0450] (0.6962 vs 0.6621). The full-VOC-Sim-best thresholds are again 0.55 and 0.425.
- **Other conventions, COCO-Cap, with DenseCRF.** Same threshold 0.55: +0.0121 [−0.0018, +0.0264]. Same threshold 0.425: −0.0387 [−0.0502, −0.0265]. Each map at its COCO-Cap-best threshold (0.600 / 0.525; uses COCO-Cap labels, diagnostic only): +0.0176 [+0.0063, +0.0294]. The original run's own report already listed "COCO best-thr Δ +0.0175" next to the +0.0341.
- **Raw (no DenseCRF), COCO-Cap.** VOC-Sim thresholds 0.55 / 0.45: +0.0219 [+0.0117, +0.0321]; same threshold 0.55: +0.0049 [−0.0077, +0.0186]; each at its COCO-Cap best: +0.0172 [+0.0074, +0.0275].
- **VOC-Sim.** Each map at its own threshold: +0.0057 [−0.0009, +0.0122]; 5-fold out-of-fold threshold per map: +0.0074 [+0.0009, +0.0142].
- **Why the conventions differ.** The fine-only map's best threshold moves from 0.425 on VOC-Sim to 0.525 on COCO-Cap; the hierarchy's moves from 0.55 to 0.60. Fine-only at its VOC-Sim threshold is over-inclusive on COCO-Cap (precision 0.703). So the +0.034 is part localisation (+0.018 at each map's best threshold) and part threshold transfer.
- **Per class (COCO-Cap).** Under the draft's convention the largest gains are car +0.132, aeroplane +0.082, tv +0.074. At the same threshold the hierarchy loses on 8 of 20 classes (chair −0.076, boat −0.056) and gains most on dining table +0.139 and tv +0.114.
**Slides:** `slides/jc_meeting/sections/c7_F.tex` (one bullet rewritten, one slide added).
**Open points:** The paper text is not changed. Proposed wording for the claim: the hierarchy's map transfers to cluttered scenes at its VOC-Sim threshold, where the fine-only map does not; about half of the +0.034 remains when each map is given its best threshold. This agrees with protocol Section 134 ("cross-image calibration, not better localisation").

**Review — N, 2026-11-19:** confirmed with notes. I traced every number to `runs/16_trace_034/result.md` and found
no mismatch, and I recomputed the source from the committed counts of the original run (`sg_combined_v3` shards).
That gives thresholds 0.425/0.55, +0.0341 [+0.0224, +0.0454] (0.6964 − 0.6623), +0.0123 [−0.0026, +0.0265] at a
common 0.55, and +0.0175 at each map's COCO-best, so the two-threshold attribution holds on the original run too.
The work is post hoc and labelled so, with no selection. Notes: (1) calling the +0.018 at each map's best threshold
"localisation" contradicts Section 134 and the paper's per-image oracle (tie, 0.896 vs 0.895), which the entry says
it agrees with. Both parts are cross-image calibration: about +0.018 at a COCO-chosen global threshold and about
+0.016 from transferring fine-only's VOC-Sim threshold. (2) The same convention produces the paper's "DenseCRF gains
six times more on capped maps" (+0.002 vs +0.014). At matched thresholds DenseCRF adds +0.014 [+0.012, +0.016] to
fine-only at 0.45, against +0.014 for the capped map at 0.55 (N's recomputation, post hoc). (3) runs/20 reproduces
the pattern on the corrected data: +0.036 at the VOC-selected thresholds, +0.010 [−0.004, +0.026] at 0.55, +0.017 at
COCO-best. Not checked: runs/16's own intervals (no per-image counts committed).

### Entry 10 — F (Faizyab's Claude), 2026-11-17
**Item:** #3 comparison methods (cross-side diagnostic: where NERVE's lead over BRACE comes from)
**Kind:** result (pre-registered: `runs/17_seed_walk_swap/PREREG.md`, commit 6a445fc0e, before scoring)
**What was done:** Swapped seed and propagation between the two methods on all 490 VOC-Sim and 707 COCO-Cap images. Seeds: BRACE's cross-attention seed (live generation) and NERVE's CLIP ViT-L/14 seed (stored image; {class word} + NERVE's VOC background words, N's variant A). Propagations: none, BRACE fine-only, BRACE capped hierarchy (frozen code), NERVE's released refinement (N's patched copy, unchanged). Jobs 253191–253203. This uses N's NERVE setup as is and does not replace N's official NERVE row.
**Where:** `runs/17_seed_walk_swap/result.md`, `scripts/jc_swap.py`, `scripts/jc_swap_nerve.py`, `jobs/nerve_gpu.slurm`; this commit.
**Result (primary: threshold selected on VOC-Sim per cell, with DenseCRF; class-macro mIoU):**

| Propagation | VOC-Sim, CA seed | VOC-Sim, CLIP seed | COCO-Cap, CA seed | COCO-Cap, CLIP seed |
|---|---:|---:|---:|---:|
| none | 0.735 | 0.759 | 0.619 | 0.693 |
| BRACE fine-only | 0.823 | 0.830 | 0.662 | 0.704 |
| BRACE capped hierarchy | 0.830 | 0.836 | 0.696 | 0.704 |
| NERVE walk | 0.828 | 0.842 | 0.678 | 0.737 |

- **Registered contrasts, COCO-Cap.** Walk − hierarchy on the CLIP seed: +0.033 [+0.019, +0.049]. Walk − hierarchy on the CA seed: −0.019 [−0.031, −0.005]. CLIP − CA seed under the hierarchy: +0.008 [−0.012, +0.029]. CLIP − CA seed under the walk: +0.060 [+0.035, +0.083]. Interaction: +0.052 [+0.034, +0.070].
- **Registered contrasts, VOC-Sim.** +0.006 [−0.002, +0.015]; −0.002 [−0.013, +0.008]; +0.006 [−0.007, +0.018]; +0.014 [+0.001, +0.029]; interaction +0.008 [−0.004, +0.021].
- **Reading rule: mixed.** Neither "the seed explains it" nor "the walk explains it" holds. NERVE's COCO-Cap lead (0.737 vs 0.696) appears only when its walk is applied to its CLIP seed. The hierarchy removes the seed gap (+0.074 for seeds alone, +0.008 after the hierarchy); the walk is worse than the hierarchy on the CA seed.
- **Per class, COCO-Cap.** CLIP seed: person 0.28 → 0.64 (hierarchy) / 0.71 (walk), chair 0.47 → 0.59 / 0.68; tv 0.79 → 0.26 / 0.25, aeroplane 0.80 → 0.67 / 0.64.
- **Fidelity.** BRACE cell at 0.55 with DenseCRF: 0.830 / 0.696 (paper: same). NERVE cell at N's convention (0.5, raw): 0.8302 / 0.7295, equal to N's committed counts to four decimals; per-image counts identical on 40/490 and 94/707 images, max |ΔIoU| 0.0016 / 0.0021. The residual is not explained (NERVE runs CLIP in half precision and zeroes seed values below 20% of the maximum; both NERVE stages ran on the A100 node as N's did).
- **Sensitivity.** COCO-Cap, bit-identical regenerated images only (243): same signs; walk − hierarchy on the CA seed becomes −0.013 [−0.035, +0.008]. Raw (no DenseCRF): same signs and conclusions.
**Caveats:** (1) The CLIP seed and NERVE's walk read the stored image; BRACE's attention comes from the regeneration, which is bit-identical to the stored image on 238/490 and 243/707 images. (2) Both benchmarks were used to develop BRACE, not NERVE. (3) The CA seed is min–max normalised; NERVE's walk was designed for CLIP probabilities and none of its constants were retuned for the CA seed. (4) CLIP-seed cells use a second model and are not SD-only. (5) Two code changes after the pre-registration, neither in scoring: an empty-subset guard in the report and an output-folder suffix for the hardware check (commit 150586fa3).
**Slides:** `slides/jc_meeting/sections/c3_F.tex` (two slides).
**Open points:** N to confirm that the NERVE cell here matches the intended variant A. Whether the paper states this result (e.g. one sentence: with the same CLIP seed, our hierarchy and NERVE's walk differ by +0.033 on COCO-Cap in NERVE's favour; with the SD-only seed, by 0.019 in ours) is for the owners.

**Review — N, 2026-11-19:** issues (numbers and rule confirmed; one caveat wrong; headline superseded). I traced
every number to `runs/17_seed_walk_swap/result.md` and found no mismatch. The PREREG (`6a445fc`, 22:55) precedes
the full jobs 253191–253203 (submitted 23:05:16 per `sacct`, scoring 23:59–02:05). The only change (`150586f`) is a
report guard and an output suffix, with no scoring change (diff read). "Mixed" is the registered rule's outcome. The
NERVE cell is N's variant A (class word + NERVE's 25 background words, d = 40, K=1 guard; checked in
`jc_swap_nerve.py` against `PREREG_sd15.md`). Its 0.8302/0.7295 equals N's `sanity_summary.json` and N's recomputed
counts (0.83020/0.72946) to four decimals; the per-image residuals were not checked (not committed). Issues: (1)
caveat (1) and the "bit-identical" sensitivity compare the regeneration with the stored paper capture
(`jc_swap.py:65`), not with the official PNG that the CLIP seed, the walk and the annotation use. Those differ on
101 COCO-Cap images (runs/19 addendum), so the subset does not control the mismatch. (2) The 0.737 vs 0.696 lead is
superseded. On corrected prompts (runs/20) NERVE − BRACE is +0.027 [+0.002, +0.054] on COCO-Cap and +0.011
[−0.003, +0.026] on VOC-Sim. The swap stays mixed (P-CLIP +0.016 [+0.004, +0.030], P-CA −0.033 [−0.045, −0.020],
S-H +0.010 [−0.011, +0.034], S-W +0.060 [+0.037, +0.084]). (3) `c3_F.tex` needs the corrected numbers; a correction
entry is requested.

### Entry 11 — N (Narmeen), 2026-11-19
**Item:** #3 GoCA, NERVE, DiffSegmenter: official rows paired with BRACE
**Kind:** result
**What was done:** BRACE regenerated with side F's committed modules imported unchanged (`scripts/jc_newrerun.rows` with the runs/20 corrected prompts, `sg27_runtime.live_maps`, `sg_combined_v2.aggregate`), safety checker off, job 253750 (wald V100). N methods scored from their stored soft maps (jobs 253753, 253757, 253803). Metric, OOF and bootstrap code reused unchanged from `scripts/sg_class_comp.py`; DenseCRF R2 = `src/segmentation/ovam_full.apply_ovam_densecrf`. Threshold grid: runs/20's 0.30–0.70 extended to 0.05–0.95 (27 values), because GoCA SD 1.5 peaks above 0.70; BRACE's numbers are the same on either grid.
**Where:** `comparisons/official/` (`brace_repro.py`, `score_n.py`, `report.py`, `result.md`, `summary.json`, `per_item_counts.csv`); this commit.
**Result (exploratory; BRACE's VOC-Sim numbers are in-sample for its design):**
- BRACE reproduction of runs/20: DenseCRF 0.831 / 0.708 VOC-sel. (runs/20: 0.832 / 0.709); raw 0.818 / 0.692 (0.818 / 0.693); fine-only and seed-alone cells within 0.001. Ground-truth masks identical on 1197/1197 items.
- Paired Δ, method − BRACE, DenseCRF, VOC-sel., 2000-sample image bootstrap:

| Method | Δ VOC-Sim | Δ COCO-Cap |
|---|---|---|
| GoCA SD 1.5 | −0.078 [−0.095, −0.060] | −0.110 [−0.134, −0.084] |
| GoCA SDXL | −0.034 [−0.048, −0.021] | −0.102 [−0.124, −0.078] |
| DiffSegmenter | −0.036 [−0.049, −0.023] | −0.020 [−0.041, +0.001] |
| NERVE A (class word) | +0.012 [−0.002, +0.028] | +0.032 [+0.007, +0.060] |
| NERVE V (20-class vocabulary) | +0.008 [−0.007, +0.023] | +0.039 [+0.013, +0.068] |
| NERVE W (NERVE's names) | −0.002 [−0.017, +0.012] | +0.022 [−0.005, +0.052] |

- Raw maps give the same signs. With its own released rule (image-score gate) NERVE A is 0.812 / 0.663 (DenseCRF), below BRACE. DiffSegmenter's own evaluation sweeps its background score on the evaluated set, so its own-rule row is an in-sample audit only.
- Sensitivity: restricting to items whose regenerated image is within 1 or 2 grey levels of the stored PNG keeps every sign.
**Slides:** `slides/jc_meeting/sections/c3_N.tex` (three frames added; sanity frame set to done).
**Open points:** The pairs are not same-setting: BRACE reads attention during generation, the N methods segment the stored image; regeneration differs from the stored PNG by under one grey level on 431/490 and 583/707 items. NERVE uses a second model (CLIP ViT-L/14). Two workarounds outside the repo, neither touching scoring: SimpleCRF 0.2.1.1 built with a numpy-2 define; the COCO-Cap CSV rebuilt with the committed `scripts/build_coco_cap_manifest.py`.

**Review — F, <date>:** pending

### Entry 12 — N (Narmeen), 2026-11-19
**Item:** #3 published block; #9 venue; #5 SD 2.1 weights
**Kind:** result
**What was done:** Read each paper and the ICPR 2027 pages; rescored Attn2Mask's released masks; compiled the current paper with the ICPR `llncs` template in a scratch copy (not committed). Checked the SD 2.1-base community copy against independent copies.
**Where:** `comparisons/notes/entry_venue_and_published_block.md`; this commit.
**Result:**
- Published block: none of DAAM, Attn2Mask, Grounded Diffusion, DatasetDM publishes VOC-Sim or COCO-Cap numbers; the draft's Table 5 rows are OVAM's runs of them (the caption is right, supplement line 434 is not). GoCA, NERVE, DiffSegmenter, SPARK and PEARL report only real-image benchmarks, so they enter only through paired reruns. Rescoring Attn2Mask's released masks gives 0.603 on VOC-Sim against OVAM's printed 68.7 (to be checked). BibTeX for the five papers is in the note.
- Venue: ICPR 2027 (ICPRv) is online only (October 2027), Springer LNCS one column, 15 pages plus references, double-blind, with mandatory self-review pages including an LLM-use disclosure; arXiv allowed. Registration 1 March 2027 and submission 8 March 2027 (AoE) confirmed (the site marks dates tentative). The unchanged text fills about 13.5 of the 15 pages before JC's additions; Table 3 is about 6.5 cm too wide for LNCS.
- SD 2.1: `sd2-community/stable-diffusion-2-1-base` has unet / text_encoder / vae safetensors with the same sha256 as `kaseris/stable-diffusion-2-1-base` (a clone of Stability's repository), and its `v2-1_512-ema-pruned.ckpt` sha256 (88ecb782…) equals the LFS pointer in Stability's original commit. Narmeen approved this copy for every method on 2026-11-19; side F has not confirmed yet.
**Slides:** `slides/jc_meeting/sections/c9_N.tex` (two frames); `c3_N.tex` caveats frame.
**Open points:** The self-review page count (one or two) differs between the ICPR sources; follow the two-page template. The Attn2Mask gap is not explained yet.

**Review — F, <date>:** pending

### Entry 13 — N (Narmeen), 2026-11-19
**Item:** #1 reframing text; #8 restructuring
**Kind:** text
**What was done:** Drafted the reframed abstract, contribution list and owner-override sentence, a claim-by-claim table, and the proposed section outline, from Entries 8–10, runs/20 and the protocol log (Section 135). Nothing in `paper/` is changed.
**Where:** `comparisons/notes/entry_1_8_reframing_and_structure.md`; `comparisons/notes/audit_F_entries_8_10.md` (N's audit behind the review lines of Entries 8–10); this commit.
**Result:**
- The protocol log shows the override that adopted the hierarchy was made after COCO-Cap was scored and cites its +0.0341, so COCO-Cap is not untouched for that one decision; the draft says so.
- The +0.034 is restated as cross-image calibration (Entry 9 and N's review of it); "DenseCRF gains six times more on capped maps" is dropped (same two-threshold artefact; about +0.014 for both maps at matched thresholds); "best training-free method" is narrowed to diffusion-only, with CLIP-seeded methods in a separate block.
- #8: comparison moves to the front of a renamed Sec. 5 "Experimental results"; hardware, runtime and memory go into Sec. 4. To be applied in one commit after F's Table 1 rewrite (#7).
**Slides:** `slides/jc_meeting/sections/c1_N.tex` (five frames), `c8_N.tex` (one frame).
**Open points:** The abstract's headline numbers follow the owners' choice between the original and the corrected capture (runs/19–20).

**Review — F, <date>:** pending

### Entry 14 — N (Narmeen), 2026-11-19
**Item:** #5 GoCA on SD 2.1-base
**Kind:** result
**What was done:** GoCA's `config_model_15.py` copied to `config_model_21.py` with only `name` and `version` changed; weights `sd2-community/stable-diffusion-2-1-base` revision `4e63672c`, fp32 safetensors, sha256 recomputed after download (Entry 12). Planned pad-token fix in the N adapter; layer-key check (81 of 81 returned). Full run job 253878 (erdos V100), then the official scoring of Entry 11 (jobs 253879, 253900).
**Where:** `comparisons/goca/` (`PREREG_backbones.md` amendment of 2026-11-19, `configs/config_model_21.py`, `goca_full.py`, `goca_classword.py`, `goca_sd21_smoke.slurm`); `comparisons/official/` (regenerated; rows of the other methods unchanged); this commit.
**Result (exploratory):** fixed 0.5 raw 0.710 / 0.546 (SD 1.5 0.671 / 0.452, SDXL 0.788 / 0.607). VOC-sel. DenseCRF 0.740 / 0.597. Paired Δ vs BRACE, DenseCRF, VOC-sel.: −0.091 [−0.110, −0.072] / −0.111 [−0.134, −0.086]; raw −0.098 [−0.117, −0.080] / −0.107 [−0.129, −0.084]. 0.20 s per image, peak 3.8 GB (V100; the SD 1.5 and SDXL timings were on an A100).
**Slides:** `slides/jc_meeting/sections/c5_N.tex`.
**Open points:** Deviations, all in the amendment: `--model-path` added to `goca_full.py` because GoCA's loader hard-codes the removed repository (no vendored code changed); the Euler scheduler that GoCA's loader builds for 2.1 gives timestep 99 (SD 1.5: 101). Side F's #5 runs should use the same revision.

**Review — F, <date>:** pending

### Entry 15 — N (Narmeen), 2026-11-19
**Item:** #5 fine-only and BRACE on SD 1.5, SD 2.1-base and SDXL (F's item, run by N at Narmeen's request so that it is ready for the meeting)
**Kind:** result
**What was done:** JC's setting: stored benchmark images, VAE encode, noise at t = 100 (seed 0), one conditional UNet call with the runs/20 prompts and class-word query. Seed, fine-only and hierarchy re-implemented for one call with F's helper functions imported unchanged; on SD 1.5 the single-call hierarchy equals `sg_combined_v2.aggregate` to 4.8e-7. Per-backbone layer mapping fixed in the pre-registration before any capture. Renyi A100.
**Where:** `comparisons/backbones/` (`PREREG.md` with Amendment A1, `single_pass.py`, `run.slurm`, `result.md`, `summary.json`); this commit.
**Result (exploratory; a different setting from the generation-time 0.832 / 0.709, never mixed with it):** BRACE − fine-only, raw, VOC-sel.: SD 1.5 +0.009 [+0.003, +0.016] / +0.038 [+0.029, +0.047]; SD 2.1-base +0.054 [+0.044, +0.064] / +0.060 [+0.049, +0.071]; **SDXL −0.022 [−0.030, −0.015] / −0.011 [−0.017, −0.005]**. At fixed 0.55 (raw or + DenseCRF) BRACE wins on all three, but that scores fine-only away from its best threshold. BRACE − GoCA, raw, VOC-sel.: SD 1.5 +0.055 / +0.040; SD 2.1-base +0.083 / +0.072; **SDXL −0.048 / −0.107**. Runtime per image (UNet pass + seed): 0.21 s / 0.19 s / 1.49 s, peak 5.1 / 5.5 / 15.6 GiB; DenseCRF 4.5 s per call on one CPU core.
**Slides:** `slides/jc_meeting/sections/c5_N.tex` (two frames; F's `c5_F.tex` not touched).
**Open points:** Amendment A1 (before scoring): DenseCRF only at the frozen 0.55 because the full grid would need about 240 CPU-hours. SDXL mapping: all 70 CA modules, GoCA's SDXL reference layer, SA grids 32 → 64 (one-step hierarchy). SD 2.1-base copy approved by Narmeen; F has not confirmed it. BRACE − GoCA pairs are same-image, not same-capture.

**Review — F, <date>:** pending

### Entry 16 — F (Faizyab's Claude), 2026-11-19
**Item:** #3 seed × propagation swap, BRACE and NERVE (correction of Entry 10, answering N's review of it)
**Kind:** correction
**What was done:**
- Accepted both issues in N's review of Entry 10.
- Recomputed the swap contrasts from the committed-design runs/20 counts (CPU job; same scorer, OOF and bootstrap
  code as Entry 10). The cells come from runs/20's single capture with corrected prompts and the safety checker
  off. There, the CLIP seed and NERVE's walk read **our own regenerated image**, so every cell of a comparison sees
  the same image, the one the mask is scored against. NERVE's noised pass is ported and verified equal to NERVE's
  code (runs/20 PREREG, step 4).
- No new capture.

**Where:** `runs/20_new_reruns/{PREREG.md, result.md}` (8baa528ad); `runs/19_source_rule/` addendum (5520bd310,
bb10eb8fd); this commit.

**Corrections to Entry 10:**
1. **Caveat (1) and the "bit-identical" sensitivity are withdrawn.**
   - `jc_swap.py:65` compared the regeneration with the stored paper capture, not with the official OVAM PNG that
     the CLIP seed, the walk and the annotation use.
   - The stored captures themselves differ from the official PNG on 101/707 COCO-Cap and 55/490 VOC-Sim images
     (runs/19 addendum). So the 238/243-image subsets did not control the mismatch they were meant to control.
   - The same conflation was in the runs/17 PREREG text; it is a design-text error, not a change after
     registration.
2. **The headline "NERVE leads 0.737 vs 0.696 on COCO-Cap" is superseded.**
   - The mismatch inflated it. On the 528 COCO-Cap images matched to the official PNG, the two tie: −0.001
     [−0.028, +0.043] (runs/19 addendum, post hoc).
   - On the prompt-corrected rerun (runs/20), NERVE − BRACE is +0.011 [−0.003, +0.026] on VOC-Sim and **+0.027
     [+0.002, +0.054]** on COCO-Cap (DenseCRF, VOC-sel.).

**Result (corrected swap, runs/20 capture; DenseCRF; VOC-Sim OOF threshold, COCO-Cap full-VOC-Sim threshold):**

| Propagation | VOC-Sim, CA seed | VOC-Sim, CLIP seed | COCO-Cap, CA seed | COCO-Cap, CLIP seed |
|---|---:|---:|---:|---:|
| none | 0.736 | 0.761 | 0.629 | 0.692 |
| BRACE fine-only | 0.824 | 0.831 | 0.673 | 0.713 |
| BRACE capped hierarchy | 0.832 | 0.840 | 0.709 | 0.719 |
| NERVE walk | 0.831 | 0.843 | 0.675 | 0.735 |

| Contrast (Entry 10 name) | VOC-Sim | COCO-Cap |
|---|---|---|
| Walk − hierarchy, CLIP seed (P-CLIP) | +0.003 [−0.005, +0.012] | +0.016 [+0.004, +0.030] |
| Walk − hierarchy, CA seed (P-CA) | −0.001 [−0.011, +0.009] | −0.033 [−0.045, −0.020] |
| CLIP − CA seed, hierarchy (S-H) | +0.008 [−0.006, +0.020] | +0.010 [−0.011, +0.034] |
| CLIP − CA seed, walk (S-W) | +0.012 [−0.001, +0.027] | +0.060 [+0.037, +0.084] |
| Interaction | +0.004 [−0.008, +0.017] | +0.049 [+0.033, +0.065] |

- These match N's recomputation in `comparisons/notes/audit_F_entries_8_10.md` to three decimals. The VOC-Sim
  column and the interaction interval are new here.
- **Reading under the registered rule: still mixed.** S-H crosses zero, so the seed alone does not explain the
  lead; P-CA is negative, so the walk alone does not explain it either.
- NERVE's remaining COCO-Cap lead (+0.027) appears only when its walk runs on its CLIP seed. On VOC-Sim, no
  contrast excludes zero.
- **Labels:** exploratory. Both benchmarks were used to develop BRACE, not NERVE. The CLIP-seed cells use a second
  model and are not SD-only.

**Slides:** `slides/jc_meeting/sections/c3_F.tex`. Both frames now carry the runs/20 numbers and one line on the
correction. `main.pdf` is not rebuilt in this commit; the owner rebuilds the deck.

**Open points:**
- Entries 8 and 9 use the pre-correction capture as well. N's reviews of them stand (numbers trace), but their
  values differ slightly from the corrected data (BRACE 0.830/0.696 → 0.832/0.709). Table 1 (#7) will be rebuilt
  on the corrected capture.

**Review — N, <date>:** pending

### Entry 17 — N (Narmeen), 2026-11-19
**Item:** coordination: F items run by N today
**Kind:** plan
**What was done:** At Narmeen's request, side N is running the remaining F items so that every JC experiment is finished before the meeting: #3 OVAM class word, OVAM optimized tokens, DAAM, SeeDiff, SCLIP + DenseCRF (paired rows); #4 runs/25 on the corrected data (full run per its PREREG, outputs outside `runs/`; F committed its own full runs/25 result in a64aff6 while this was running, so N's copy is reported only as an independent reproduction); #7 the COCO-Cap capture for Table 1's first rows and a draft table; #8 runtime and memory. #5 fine-only and BRACE is already done (Entry 15).
**Where:** `comparisons/notes/handoff_to_F_agent_2026-11-19.md` (full list, output paths, SD 2.1 revision); this commit.
**Result:** none yet; each run gets its own entry when it finishes.
**Slides:** none (coordination only).
**Open points:** **Side F: please do not start these runs.** If one is already running on your side, say so in a new entry and N will stop its copy. Table 1 in `paper/` stays F's.

**Review — F, <date>:** pending

### Entry 18 — N (Narmeen), 2026-11-19
**Item:** #8 cost and hardware (F's item, measured by N at Narmeen's request)
**Kind:** result
**What was done:** Per-image runtime and peak GPU memory of BRACE and fine-only, split by stage, with F's frozen runtime imported unchanged; 100 benchmark images (50 VOC-Sim + 50 COCO-Cap), SD 1.5 float32, 30 steps, CFG 7.5, 512 px, batch 1, warm-up discarded; one A100 (renyi) and one V100 (erdos). Checks: recorded image bit-identical to plain generation, and stage-wise maps equal `aggregate()`'s V0/V3, on all 100 images.
**Where:** `comparisons/cost/` (`measure_cost.py`, `cost.slurm`, `summarize.py`, `result.json`, `result.md` with a paragraph for Sec. 4); this commit.
**Result (median per image):** A100: generation 2.66 s, with attention recording 6.94 s, maps 0.034 s (fine-only) / 0.038 s (BRACE), DenseCRF 5.92 s on one CPU core, total 12.9 s (7.0 s GPU). V100: 12.11 / 12.02 / 0.080 / 0.098 / 5.07 s, total 17.2 s (12.1 s GPU). Peak GPU memory 5.14 GiB on both. The hierarchy adds no measurable time or memory over fine-only.
**Slides:** response deck (`comparisons/response/slides/main.tex`); `c8_N.tex` keeps the structure frame.
**Open points:** the official wald capture (job 253750) ran at 17.3 s for generation + recording + maps, against 12.4 s on erdos for the same stages (same GPU model, bit-identical outputs); the cause is not established, so 17.3 s is an upper range. Software versions and hardware are in `result.md`.

**Review — F, <date>:** pending

### Entry 19 — N (Narmeen), 2026-11-19
**Item:** #7 Table 1 numbers on both benchmarks (F's item; N ran the capture at Narmeen's request)
**Kind:** pre-registration and result
**What was done:** No committed run contains the paper's first three rows, so a new ladder was pre-registered (`comparisons/drafts/table1_ladder/PREREG.md`) and run: all nine rows of the current Table 1 from one live generation per image (frozen runtime, runs/20 corrected prompts, safety checker off), 490 VOC-Sim and 707 COCO-Cap, renyi A100s (job 253917). Paired Δ against an explicit reference row, 95% image bootstrap; VOC-sel. primary, fixed 0.45 and 0.55 also reported. COCO-Cap DenseCRF was computed only at the thresholds the report reads (`comparisons/drafts/fast_score.py`); VOC-Sim on the full grid.
**Where:** `comparisons/drafts/table1_ladder/` (`PREREG.md`, `t1_ladder.py`, `capture.slurm`, `score.slurm`, `make_tex.py`, `result.md`, `result.json`); LaTeX `comparisons/drafts/table1_jc7.tex` (fits the LNCS width); this commit.
**Result (exploratory; VOC-sel.; VOC-Sim / COCO-Cap):** row 1 .707 / .574; row 3 (one SA pass, 64 px) .790 / .629, +.076 [+.069, +.085] / +.057 [+.045, +.069] vs row 2, the largest step; row 7 (fine-only reference) .825 / .673; row 8 (BRACE) .832 / .708, +.007 [+.001, +.013] / +.036 [+.024, +.046] vs row 7 (different thresholds, 0.425 vs 0.55; at a shared 0.55, +.010 [−.005, +.026]). Reproduces the paper's VOC-Sim screens at 0.45 (.667 / .696 / (.757) / .790 / .803 / .816 / .821 vs .665 / .691 / (.756) / .789 / .800 / .813 / .817) and F's runs/25 rows 7–8 at 0.55 to 0.001.
**Slides:** response deck (`comparisons/response/slides/main.tex`); F's `c7_F.tex` not touched.
**Open points:** A100 images are not bit-identical to the V100 captures. The table and caption in `paper/` stay F's.

**Review — F, <date>:** pending

### Entry 20 — N (Narmeen), 2026-11-19
**Item:** #4 runs/25 on the corrected data: independent reproduction of F's a64aff6
**Kind:** result
**What was done:** F's pre-registered `scripts/jc_ladder_nr.py` run unchanged through a thin N wrapper on all 1,197 images (2× V100, erdos; job 253886), scored with the same rules.
**Where:** `comparisons/token_ladder_nr/` (`run_nr.py`, `capture.slurm`, `score.slurm`, `fidelity.py`, `result.md`, `result_n_full.md`); this commit.
**Result:** 172 of 181 three-decimal cells identical to F's `runs/25_token_ladder_nr/result.md`, 9 differ by 0.001 (all COCO-Cap, none a primary DenseCRF stage cell); the verdicts agree (F holds for both queries, H fails for both). BRACE word / token at 0.55: 0.832 / 0.864 VOC-Sim, 0.708 / 0.693 COCO-Cap. F's result is the official one.
**Slides:** none (F's `c4_F.tex` carries the result).
**Open points:** F's data-check lines read F's cache, which N cannot read; replaced by a note in N's copy.

**Review — F, <date>:** pending

### Entry 21 — N (Narmeen), 2026-11-19
**Item:** #3 OVAM class word, OVAM optimized tokens, DAAM, SeeDiff, SCLIP + DenseCRF (F's rows, run by N at Narmeen's request), paired with BRACE
**Kind:** pre-registration and result
**What was done:** One shared SD 1.5 generation per image (same prompts and seeds as the BRACE reproduction of Entry 11; wald V100s, 4 jobs) records OVAM's class-word and optimized-token maps, DAAM and SeeDiff (code github.com/BAIKLAB/SeeDiff, commit 28d26f4; `aggregate_attention` and PAMR imported unchanged); SCLIP ViT-B/16 on BRACE's stored images (CPU). Scored with the Entry 11 code (VOC-sel. primary, fixed 0.5, each method's own rule; raw and DenseCRF; paired Δ vs BRACE, 2000-sample image bootstrap). Pre-registered in `comparisons/official_f_rows/PREREG.md` (addenda before scoring).
**Where:** `comparisons/official_f_rows/` (`PREREG.md`, `capture_f.py`, `capture.slurm`, `sclip_f.py`, `sclip.slurm`, `score_f.py`, `score.slurm`, `report_f.py`, `result.md`, `summary.json`); this commit.
**Result (exploratory; DenseCRF, VOC-sel.; VOC-Sim / COCO-Cap; Δ = method − BRACE):**
- Images bit-identical to the BRACE reproduction on 1197/1197.
- OVAM, class word: 0.703 / 0.568; Δ −0.128 [−0.143, −0.114] / −0.140 [−0.160, −0.120]. OVAM's own rule gives 0.696 / 0.546 (published 0.704 / 0.582).
- OVAM, optimized token (few-shot): 0.828 / 0.664; Δ −0.003 [−0.016, +0.010] / −0.044 [−0.063, −0.026].
- DAAM: 0.727 / 0.644; Δ −0.104 [−0.119, −0.091] / −0.067 [−0.093, −0.039]. SeeDiff: 0.723 / 0.665; Δ −0.108 [−0.122, −0.094] / −0.046 [−0.070, −0.018]. Both need the class word in the caption, so on COCO-Cap they are scored on those 437 images, with BRACE on the same 437 (0.711).
- SCLIP + DenseCRF: 0.764 / 0.652; Δ −0.067 [−0.084, −0.052] / −0.057 [−0.084, −0.031].
- SeeDiff: go (public code; no published numbers on these benchmarks).
**Slides:** response deck (`comparisons/response/slides/main.tex`, comment-3 and comment-4 slides).
**Open points:** DenseCRF on a reduced threshold grid (VOC-Sim: a 7-threshold window around each raw optimum plus 0.5; COCO-Cap: the selected threshold and 0.5), raw on the full grid. DAAM reimplemented inside the shared pass (its package pins an older diffusers, which would change the images). SeeDiff on SD 1.5 at 30 steps and CFG 7.5 (benchmark generation), not its own v1-4 settings; its bulk script saves the image as the mask, so its demo path was used. SCLIP run in a separate environment on CPU.

**Review — F, <date>:** pending
