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
| #1 reframing text | N | not started | — |
| #2 mapping table | N | done, awaiting review | Entry 7 |
| #3 OVAM class word | F | not started | — |
| #3 OVAM optimized tokens | F | not started | — |
| #3 DAAM | F | not started | — |
| #3 SeeDiff | F | not started | — |
| #3 SCLIP + DenseCRF | F | not started | — |
| #3 GoCA | N | done, awaiting review (sanity only; official scoring waits for Entry 1) | Entries 2–4 |
| #3 NERVE | N | done, awaiting review (sanity only; official row undecided: A or V) | Entries 2–4 |
| #3 DiffSegmenter | N | done, awaiting review (sanity only; class-word row) | Entries 2–4 |
| #3 SPARK | N | no-go | Entry 2 |
| #3 published block | N | not started | — |
| #4 BRACE with optimized tokens | F | done, awaiting review | Entry 8 |
| #4 token provenance | N | done, awaiting review | Entry 5 |
| #5 fine-only and BRACE, SD 1.5 / 2.1 / XL | F | not started | — |
| #5 GoCA, SD 1.5 / 2.1 / XL | N | in progress: SD 1.5 and SDXL done; SD 2.1 blocked (official weights removed) | Entries 3–4 |
| #6 consistency | N | done, awaiting review | Entry 6 |
| #7 Table 1 | F | in progress: numbers done (Entry 8); paper table and caption not yet rewritten | Entry 8 |
| #8 cost and hardware | F | not started | — |
| #8 restructuring | N | not started | — |
| #9 venue | N | not started | — |
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

**Review — N, <date>:** pending

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

**Review — N, <date>:** pending

