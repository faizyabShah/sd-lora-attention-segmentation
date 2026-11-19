# Handoff from side N to side F's agent, 2026-11-19

**Read this before starting any JC-review run.** Narmeen asked side N to finish every experiment JC asked for
before the meeting, so several F items from `docs/JC_REVIEW_WORKPLAN.md` (Section 2) are being run by side N
today. To avoid duplicate runs, **do not start the runs listed in Sections 2 and 3 below.** If you already have one
of them running, add a workplan entry saying so; side N will then stop its own copy.

Side N edits none of F's files. All N code and outputs are in `comparisons/` (repo) and in N's git-ignored folder.
F's committed modules are imported unchanged.

## 1. Done and pushed (commit `d239b2b`; workplan Entries 11–15; coordination entry: Entry 17)

| JC # | What | Where | Entry |
|---|---|---|---|
| #3 | Official rows for GoCA (SD 1.5, SDXL, SD 2.1-base), NERVE (variants A, V, W) and DiffSegmenter, paired with a BRACE reproduction of runs/20 (matches to 0.001) | `comparisons/official/` | 11, 14 |
| #3, #9 | Published block (none of DAAM, Attn2Mask, Grounded Diffusion, DatasetDM publish VOC-Sim/COCO-Cap numbers); ICPR 2027 rules and page fit | `comparisons/notes/entry_venue_and_published_block.md` | 12 |
| #1, #8 | Reframing draft (abstract, contributions, override sentence) and section outline; `paper/` unchanged | `comparisons/notes/entry_1_8_reframing_and_structure.md` | 13 |
| #5 | GoCA on SD 2.1-base | `comparisons/goca/`, `comparisons/official/` | 14 |
| #5 (F item) | Fine-only and BRACE on SD 1.5, SD 2.1-base and SDXL, in JC's setting: stored images, noise at t = 100 (seed 0), one UNet call; pre-registered before capture. The hierarchy holds on SD 1.5 and SD 2.1 and loses on SDXL (raw, VOC-sel.: −0.022 / −0.011); GoCA beats BRACE on SDXL | `comparisons/backbones/` | 15 |
| review | Review lines for Entries 8, 9, 10 (in their reserved slots). Entry 10 asks for a correction entry for `c3_F.tex` | `docs/JC_REVIEW_WORKPLAN.md` | — |

**SD 2.1 weights.** Use `sd2-community/stable-diffusion-2-1-base`, revision `4e63672c03103b6c636b8fb4119ba982469b2955`
(512 px, epsilon prediction), for any SD 2.1 run, so that every SD 2.1 row shares one set of weights. Its unet,
text_encoder and vae safetensors have the same sha256 as `kaseris/stable-diffusion-2-1-base`, a clone of Stability's
repository. Its `v2-1_512-ema-pruned.ckpt` (sha256 `88ecb782…0f36`) matches the LFS pointer in Stability's original
commit. Narmeen approved this copy; please confirm in a workplan entry.

## 2. Running now on side N (do not start these)

| JC # | Run | N output (repo) | Status board row |
|---|---|---|---|
| #3 | OVAM with the class word; OVAM with its released optimized tokens (few-shot block); DAAM; SeeDiff (a documented no-go if no code is released); SCLIP + DenseCRF as a standalone row; all paired with BRACE under the Section 3 contract | `comparisons/official_f_rows/` | #3 OVAM class word, OVAM optimized tokens, DAAM, SeeDiff, SCLIP + DenseCRF |
| #4 | runs/25 stage ladder on the corrected data, run in full as `runs/25_token_ladder_nr/PREREG.md` specifies, outputs in N's folder. **F's own full result (a64aff6) is the official one; N's copy is an independent reproduction only.** | `comparisons/token_ladder_nr/` | — |
| #7 | A fresh capture for the COCO-Cap values of Table 1's first rows, plus a **draft** Table 1 in LaTeX with every number traced to its committed source | `comparisons/drafts/table1_jc7.tex` | #7 Table 1 (numbers) |
| #8 | Runtime and peak GPU memory per image for BRACE and fine-only (A100 and V100, split by stage), plus a hardware and software paragraph for Sec. 4 | `comparisons/cost/` | #8 cost and hardware |

Each will be pushed today with its own workplan entry and slide frames in N's slide files (`c3_N`, `c4_N`, `c5_N`,
`c8_N`). F's slide files are not touched.

## 3. Still F's

- Putting Table 1 into `paper/` (#7), starting from N's draft, plus the paper text that follows from these results.
- A correction entry for `slides/jc_meeting/sections/c3_F.tex` (see N's review of Entry 10).
- Confirming the SD 2.1-base copy above.
- Review lines for N's Entries 2–7 and 11–15, and for the entries that follow.
