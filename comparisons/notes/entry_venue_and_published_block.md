### Entry — N (Narmeen), 2026-11-19
**Items:** #9 venue (ICPR 2027) and #3 published block
**Kind:** desk research (primary sources only), no new runs
**Where:** this file. Source PDFs were downloaded to a session scratch area and are not committed. Every number below can be checked at the cited URL, table and page.

---

## Part 1 — #9 Venue: ICPR 2027

### 1.1 What ICPR 2027 is

- The 2027 edition is **ICPRv 2027**, "the first VIRTUAL 29th Int. Conf. on Pattern Recognition 2027", run by IAPR. It is **fully online**, on **4, 5, 6 and 14, 15 October 2027**. It sits between ICPR 2026 (28th, Lyon) and ICPR 2028 (30th, Sydney). Sources: https://www.icpr2027.com/ (`https://iapr.org/icpr-2027/` redirects there).
- Proceedings: **Springer LNCS**. The site says: "Papers will have the same paper length, reviewing, and quality as the in-person ICPRs."
- There are no tracks. Papers are presented in two-hour thematic sessions, and authors pick a preferred session topic when they submit.
- A 2027 call is published, so ICPR 2026 rules are not needed as a baseline.

### 1.2 Dates (https://www.icpr2027.com/web/dates; the page says "All dates are tentative")

| Milestone | Date |
|---|---|
| Paper registration (author list and order fixed) | **1 March 2027, AoE** |
| Paper submission | **8 March 2027, AoE** |
| Reviews to authors | 10 May 2027 |
| Author rebuttal | 17 May 2027 (provisional) |
| Author notification | 14 June 2027 |
| Camera-ready | 28 June 2027 |
| Conference site open to delegates | 27 September 2027 |
| Conference | 4–6 and 14–15 October 2027 |

The two dates JC gave (registration 1 March, submission 8 March) are **confirmed**.

**Superseded source, do not use.** The first CFP PDF, dated 18 April 2026 (https://homepages.inf.ed.ac.uk/rbf/CFPICPRv27vn1.pdf), gave different terms:
- submission on 1 March, reviews 9 April, rebuttal 16 April, notification 24 May, camera-ready 7 June;
- **single-blind** review, with names and affiliations in the manuscript;
- short papers excluded from the proceedings.

The current website overrides all of this. It says double-blind, gives the dates above, and puts short papers in the proceedings. Anyone who finds the old PDF should not follow it.

### 1.3 Submission rules
Sources: https://www.icpr2027.com/web/instructions, https://www.icpr2027.com/web/submitpaper, https://www.icpr2027.com/web/cfp, and the official long-paper template (`samplepaper.tex`, Springer llncs v2.25, 2026/09/03, linked from the submit page).

| Topic | Rule |
|---|---|
| Paper types | **Long: up to 15 pages** in LNCS format, **excluding references**. Short: up to 8 pages excluding references. Short papers are for "Blue Sky, Application, and Resource" work. Both types go into the proceedings. |
| What counts | "All title and abstract material, sections, figures, tables, equations, and other main-paper content count toward the limit. The references may extend beyond the 15 pages." (template) |
| Mandatory self-review | Must come **after the references** in the same PDF. It does not count toward the limit. The template and the instructions page say **exactly two pages**: Q1–5 on the first page and Q6 (LLM usage) on the second. The CFP page and a template comment still say "one additional last page". The sources disagree, so follow the two-page template and ask icpr27pc@iapr.org if unsure. Q1 is the main contribution, Q2 the difference from the closest work, Q3 the evidence, Q4 the limitations, Q5 the importance. Q6 asks for a detailed LLM-usage disclosure: uses, model name and version, customisation, human validation, and whether prompts are shared. Leaving this section out can lead to desk rejection. |
| Template | Springer **`llncs` class**, `\documentclass[runningheads]{llncs}`, single column, with `splncs04.bst` and `newtxtext`/`newtxmath` fonts. The template says: "do not alter the prescribed margins, fonts, spacing, or page geometry". LaTeX zip and Overleaf links are on https://www.icpr2027.com/web/submitpaper. |
| Review | **Double-blind.** The manuscript, supplement and metadata must not reveal identities. This covers acknowledgements, funding, figures, links and PDF metadata. Self-citations are written in the third person and are not blinded. Unpublished concurrent work must be cited and supplied as an anonymized supplement, with the differences explained. Non-compliance may lead to rejection without review. |
| Supplementary | Optional. One PDF, RAR or ZIP file, at most **250 MB**, anonymized. Reviewers are **not required** to read it, and decisions rest mainly on the main paper. It must not be used to get around the page limit. If the paper is accepted, the supplement is published on SpringerLink but is not part of the proceedings volume. |
| Main PDF | One PDF, at most 50 MB, submitted through **Microsoft CMT** (https://cmt3.research.microsoft.com/User/Login?ReturnUrl=%2FICPRv2027). Reviewers are assigned with TPMS. |
| Links | No external links that add content. Links to anonymous code repositories, public datasets and published software are allowed if they keep the paper anonymous. |
| arXiv / preprints | **Allowed** before and during review. A non-peer-reviewed preprint does not count as prior publication. Do not cite your own repository or project page if that would break anonymity. No press releases or media coverage while under review. |
| Dual submission | Not allowed: the paper must not be under review elsewhere in substantially similar form, or previously published. |
| Author list | **Final at registration (1 March 2027).** Later changes happen only in exceptional cases, with written consent from all co-authors and the corresponding author's department chair. |
| Other obligations | The corresponding author nominates reviewers from the author list, and authors are expected to review. At least one author must register and present virtually, or the paper may be removed from the proceedings. LLM use in the method or results must be disclosed. Details of a "PAT paper review LLM service" will appear after 11 January 2027. Accepted papers can apply for an RRPR reproducibility badge. |

### 1.4 What the current manuscript must change

The current paper is `paper/attention_segmentation.tex`, built with `cvpr.sty` in review mode, 10 pt, two columns, letter paper. Its PDF (`paper/attention_segmentation.pdf`) has **9 pages**:
- the main text, with Table 5 and Fig. 6, ends in the right column of **page 8**;
- the references take the rest of page 8 and all of page 9;
- so the main paper is about 7.7 CVPR pages, against CVPR's 8-page limit.

**Trial conversion.** I compiled the current section files unchanged with the ICPRv 2027 `llncs.cls` in a scratch copy, which was not committed. The repository's TeX Live has no `newtx`, so I used `mathptmx` (Times) in its place, and I added `\eg`/`\ie` definitions. Result:
- the body text ends on LNCS page 11;
- the floats run on to **page 14**, which is about half full;
- the references take about 1 page;
- so the main content is **about 13.5 of the 15 allowed pages**, *before* adding anything JC asked for.

Changes, in order of effort:

1. **Class and style.**
   - Replace `\usepackage[review]{cvpr}` and the two-column article with `\documentclass[runningheads]{llncs}`.
   - Remove `\paperID`, `\confName`, the CVPR header and line numbers, and the `pagebackref` bibliography back-references.
   - Change `ieeenat_fullname` to `splncs04`.
   - Add `\titlerunning`, `\authorrunning`, `\institute{Anonymous Institution}` and `\keywords{...}`, which the LNCS template requires.
   - The build machine needs `newtx`, or the paper is built on Overleaf.
2. **Captions.** LNCS puts table captions **above** the table. All five tables currently place `\caption` below.
3. **Float widths.**
   - In single column, `\columnwidth` equals `\textwidth`, which is 12.2 cm, so Figs. 4, 5 and 6 (`cap_mechanism`, `crf_effect`, `threshold_curves`) grow to full width and use more space. Set them to about 0.6–0.7 `\textwidth` or put them side by side.
   - `figure*` figures shrink, because the CVPR text width is 17.5 cm. Fig. 2 (`pipeline_c2f`, at `.8\textwidth`) becomes about 9.8 cm wide and its small panels may be hard to read. Fig. 1 (`method_overview`) drops from 17.5 cm to 12.2 cm.
   - **Table 3 (`tab_hierarchy`) is about 6.5 cm (185 pt) too wide** in LNCS and must be restructured, for example by splitting the VOC and COCO columns or transposing it.
4. **Page budget after JC's comments.** Only about 1.5 pages are left. JC's items add more:
   - #2 the contribution-to-evidence table;
   - #3 the paired comparison table, which is larger than the current Table 5;
   - #4 the optimized-token block;
   - #7 Table 1 with a COCO-Cap Δ column and folds won;
   - #8 the hardware, runtime and memory paragraph;
   - #5 the backbone block and the SD3/FLUX scoping sentence.

   Likely cuts or moves to the supplement:
   - **Table 4** (`tab_refine`, what DenseCRF adds) shrinks to one sentence or one supplement table;
   - **Fig. 5** (`crf_effect`) goes to the supplement, since it is qualitative and the claim is made by numbers;
   - **Fig. 3** (`scale_profiles`) or **Fig. 6** (`threshold_curves`) is kept, not both;
   - **Table 2** (`tab_seed`) loses its top half (the per-resolution CA screen), which moves to the supplement;
   - the long figure and table captions are shortened. They are written in CVPR style, and in single column each caption line holds fewer words, so captions cost more space.

   The self-review pages do not count toward the limit, but they must be written.
5. **Anonymity.**
   - The current draft is already anonymous ("Anonymous CVPR submission", and no URLs in `sec/` or `tables/`).
   - The author name becomes "Anonymous Authors", and the institution "Anonymous Institution".
   - Check the PDF metadata (pdfauthor, and the hyperref `pdfinfo`) and the supplementary zip, including file paths and figure metadata.
   - Self-citations, if any, are written in the third person.
   - No acknowledgements or funding statement in the review version.
6. **Supplementary.** It can stay as one anonymized PDF, as it is now. Reviewers need not read it, so every claim must be supported in the main paper, and the pointers to the supplement must not carry essential evidence. This matters for JC #2, the main-paper versus supplement column.
7. **Abstract.** It is 230 words, within the 150–250 the template recommends. The template asks for no special characters, references or line breaks in the abstract. The current abstract uses `\emph` and `\%`; check that they render.
8. **Registration logistics.** The full author list must be entered in CMT **by 1 March 2027**. Every author needs a CMT profile with conflicts declared, and authors may be asked to review.

---

## Part 2 — #3 Published block

### 2.1 Key fact about the current Table 5

The Table 5 rows for **DAAM, Attn2Mask, Grounded Diffusion and DatasetDM are not numbers those papers published.** They are **OVAM's own runs** of those methods on OVAM's VOC-sim and COCO-cap (OVAM, CVPR 2024, **Table 1, p. 9247**), with these settings:
- **DAAM**: the official code on SD 1.5, 30 steps, τ = 0.4. For COCO-cap images whose prompt lacks the class word, OVAM switched it to OVAM's attention.
- **Attn2Mask**: OVAM's own **reimplementation**, because no code had been released. SD 1.5, 100 steps, CA at t = 50, τ = 0.5, then dCRF.
- **Grounded Diffusion** and **DatasetDM**: the authors' released VOC-class weights.

Source: OVAM supplementary A.3, p. 13 of the arXiv PDF (current version), https://arxiv.org/pdf/2403.14291.

None of those four papers reports VOC-sim or COCO-cap numbers. The Table 5 caption ("published in OVAM, Table 1") is therefore correct. But `paper/supplementary.tex` l. 434, "Table 5 of the main paper quotes the class-word results reported by each method", is **inaccurate**. It should say "reported by OVAM (Table 1) for each method". This is a JC #6 item.

### 2.2 Numbers exactly as printed

The metric is mIoU in %. "Gen." means generated images and "Real" means real images.

| Method | Source table, page | Benchmark (images) | Number(s) as printed | Metric | Backbone | Supervision | Comparable to our VOC-Sim/COCO-Cap? |
|---|---|---|---|---|---|---|---|
| **OVAM** (w/o token opt.) | OVAM Table 1, p. 9247 | VOC-sim / COCO-cap (Gen.) | **70.4 / 58.2** | mIoU = (1/20) Σ_c IoU_c, with IoU_c = TP/(TP+FP+FN) per class (p. 9247) | SD 1.5 | training-free, class word; SA refinement + dCRF, τ = 0.4, α = 0.85 | **Yes** (same benchmark; we regenerate the images from the released prompt and seed) |
| OVAM + token opt. | OVAM Table 1, p. 9247 | VOC-sim / COCO-cap | 82.5 / 69.2 | same | SD 1.5 | **few-shot** (token optimized on annotated images; see Entry 5) | Yes, but a different setting (not training-free) |
| **DAAM** | OVAM Table 1, p. 9247 (OVAM's run) | VOC-sim / COCO-cap | **66.2 / 48.4** | same | SD 1.5 (OVAM's run) | training-free; thresholded CA, **no dCRF mentioned** | Yes (OVAM's run, not DAAM's paper) |
| DAAM (own paper) | DAAM ACL 2023, Table 1, p. 5647 | COCO-Gen / Unreal-Gen (Gen., 500 prompt–image pairs each, hand-segmented nouns) | DAAM-0.4: COCO-Gen **62.8** (mIoU^80) / **58.8** (mIoU^∞); Unreal-Gen **64.8** / **62.2**. DAAM-0.3: 62.7 / 57.0 / 64.7 / 62.6. DAAM-0.5: 59.6 / 55.9 / 60.0 / 57.1 | mIoU "over the prediction–ground truth mask pairs"; ^80 = only the 80 COCO classes, ^∞ = all nouns | **SD 2.0-base**, 30 DPM steps | training-free | **No.** Different benchmark, backbone and metric. Note: the arXiv version's Table 1 has *different* numbers (100 pairs, e.g. DAAM-0.4 64.8/60.7); cite the ACL version. |
| **Attn2Mask** | OVAM Table 1, p. 9247 (OVAM's reimplementation) | VOC-sim / COCO-cap | **68.7 / 55.0** | OVAM mIoU | SD 1.5 (reimpl.) | training-free + dCRF | Yes, but a reimplementation |
| Attn2Mask (own paper) | Yoshihashi et al., ACCV 2024, Table 1 (PDF p. 10 = proc. p. 2309) and Table 2 (p. 11 = 2310) | **PASCAL VOC 2012 val (Real)** | **62.2** (DeepLabv3+ ResNet-50), **71.0** (Swin-B), trained only on synthetic images and pseudo-masks | standard VOC mIoU (21 classes incl. background) | SD **1.4** generator; the reported number is a *trained segmenter* | the pseudo-masks are training-free, **but the number is from a trained segmenter** (weakly supervised synthetic training) | **No.** A downstream segmenter on real images, not pseudo-mask quality. |
| **Grounded Diffusion** | OVAM Table 1, p. 9247 (OVAM's run) | VOC-sim / COCO-cap | **62.1 / 50.2** | OVAM mIoU | SD 1.5 | **trained** grounding module (VOC-class weights) | Yes (OVAM's run) |
| Grounded Diffusion (own paper) | Li et al., ICCV 2023, Table 1, p. 7671 | **PASCAL-sim / COCO-sim** (Gen.; their own benchmarks; ground truth from an off-the-shelf **Mask R-CNN**, not human) | Ours, Average over 3 splits. PASCAL-sim: one object seen **90.30**, unseen **83.08**; two objects seen 79.10, seen+unseen 67.86, unseen 55.74. COCO-sim: 83.68, 73.21, 64.16, 56.31, 44.40. (DAAM row, same table, PASCAL-sim average: 64.84, 62.90, 48.21, 49.19, 50.95) | category-wise mIoU, (1/C) Σ IoU_c | Stable Diffusion (version not stated in the main paper) + trained grounding module | **trained** on seen categories (15 of 20 VOC classes, 65 of 80 COCO) | **No.** Different generated sets, detector pseudo-ground truth, and a seen/unseen split. OVAM's VOC-sim "partially replicat[es] experiments from [GD]" but is not the same set. |
| Grounded Diffusion (own paper, real) | Li et al., ICCV 2023, Table 2, p. 7673 | PASCAL VOC / COCO **(Real)** zero-shot, MaskFormer R101 trained on synthetic data | Model-C (synthetic only): VOC seen 69.5 / unseen 63.2 / harmonic 66.2; COCO 38.3 / 38.1 / 38.2 | mIoU | — | trained segmenter | No |
| **DatasetDM** | OVAM Table 1, p. 9247 (OVAM's run) | VOC-sim / COCO-cap | **80.3 / 59.3** | OVAM mIoU | SD 1.5 + P-decoder (authors' VOC weights) | **trained** decoder (around 100 labelled real images) | Yes (OVAM's run) |
| DatasetDM (own paper) | Wu et al., NeurIPS 2023, Table 4, PDF p. 9 (proc. pp. 54683–54695) | **PASCAL VOC 2012 val (Real)** | Synthetic only (40k images, R:100): **60.3** (R50), **73.7** (Swin-B); with 100 real images added: 66.1 (R50), 78.5 (Swin-B) | VOC mIoU | SD V1 + Mask2Former-style P-decoder; reported number from a downstream Mask2Former | **trained** (P-decoder on about 100 labelled images, then a trained segmenter) | No |
| DatasetDM (own paper, zero-shot) | same, Table 7, p. 9 | VOC 2012 (Real), 15 seen / 5 unseen | seen 78.8 / unseen 60.5 / harmonic 68.4 | mIoU | — | trained | No |
| **GoCA** | Meng et al., CVPR 2026, Table 1, p. 35866 | **VOC 2012, Pascal-Context-59, COCO-Object, Cityscapes, ADE20K (all Real)** | Ours SD v1.5: **60.7 / 40.4 / 39.2 / 16.1 / 22.0**; SDXL 65.6 / 42.3 / 44.3 / 21.2 / 23.2; PixArt-Σ 63.6 / 43.2 / 39.8 / 22.6 / 23.8; Flux 70.7 / 51.1 / 48.1 / 27.1 / 29.3. Baseline SD v1.5 (manual layer weights): 51.1 / 35.4 / 36.9 / 18.4 / 21.0. Vanilla SD v1.5: 44.3 / 32.3 / 32.3 / 11.8 / 18.0 | mIoU over classes | SD 1.5, SDXL, PixArt-Σ, Flux | training-free; **prompts modulated with GPT-4o** to name the objects present ("common practice to rely on external object detectors to modulate prompts") | **No.** Real images, multi-class semantic segmentation, and a prompt listing the classes present. |
| **NERVE** | Mahatha et al., WACV 2026, Table 1, p. 3249 | V21, PC60, C-Obj, V20, ADE, PC59, C-Stf (all Real) | **69.7 / 37.7 / 43.3 / 90.1 / 24.0 / 43.4 / 28.8**, avg **48.1**, no post-processing | mIoU | CLIP ViT-L/14 (cross-attention to text) + **SD 2.1** self-attention, 336 px | training-free | **No.** Real images. The text on p. 3248 gives 69.4 on V21 and 37.6 on PC60, which differs from the table; quote the table. |
| NERVE (pseudo-masks) | same, Table 2, p. 3249 | PASCAL VOC 2012 / MS-COCO **train sets** (Real) | "NERVE (for iSeg)": **76.2 / 45.7** | mIoU | as above | training-free | No |
| **DiffSegmenter** | Wang et al., arXiv v3, Table 1, p. 6 (the TIP 2025 version could not be accessed) | VOC / Context / COCO-Object (Real), zero-shot OVSS | **60.1 / 27.5 / 37.9** | mIoU | **SD 1.5**, plus BLIP captions and CLIP for class selection | training-free | No. Real images. GoCA's Table 1 prints the same three numbers. |
| DiffSegmenter (pseudo-masks) | same, Table 2, p. 6 | VOC 2012 **train**, image-level labels given (Real) | **70.5** | mIoU | as above | uses image-level labels | No |
| DiffSegmenter (WSSS) | same, Table 3, p. 6 | VOC 2012 val / test (Real), DeepLab R101 trained on its masks | 69.1 / 68.6 | mIoU | trained segmenter | trained | No |
| **SPARK** | Mahatha et al., arXiv 2602.00516 v1, Table 1, p. 6 | VOC / Context / COCO-Object / COCO-Stuff-27 / Cityscapes / ADE20K (Real) | **66.9 / 57.7 / 42.7 / 52.0 / 33.9 / 48.0** | mIoU, after matching clusters to classes | **SSD-1B** (distilled SDXL), empty prompt, t = 50, 1024 px, with **PAMR** | training-free, **unsupervised** (class-agnostic clustering, no text query) | **No.** It is a different task (unsupervised segmentation), with no class-word query and real images. |
| **PEARL** | Pei et al., CVPR 2026, Table 1, p. 17931 | V21 / PC60 / Object / V20 / PC59 / Stuff / City / ADE (Real) | **64.1 / 35.1 / 37.3 / 86.9 / 38.6 / 26.3 / 37.6 / 19.4**, avg **43.2** | mIoU, no DenseCRF or PAMR | **CLIP ViT-B/16, no diffusion model** | training-free | **No.** It is CLIP-based and uses real images. |

### 2.3 Not directly comparable: summary flags

1. **Only OVAM Table 1 uses our benchmarks.** These are its own OVAM rows and its runs of DAAM, Attn2Mask, GD and DatasetDM. Everything else is on different benchmarks, so it cannot go in a "same benchmark" block:
   - real-image OVSS benchmarks: GoCA, NERVE, DiffSegmenter, PEARL, SPARK;
   - downstream segmenters trained on real-image benchmarks: Attn2Mask, DatasetDM, GD Table 2;
   - other generated sets: DAAM COCO-Gen, GD PASCAL-sim and COCO-sim.
2. **Raw and DenseCRF values mixed inside OVAM Table 1.** OVAM and Attn2Mask (OVAM's reimplementation) use dCRF. DAAM, as OVAM ran it, is thresholded with no dCRF mentioned. GD and DatasetDM use their own heads. The day-one contract says "raw and DenseCRF values never share a comparison". OVAM's Table 1 breaks this rule by construction, so the caption should say so, or the block should be labelled "as printed, mixed refinement".
3. **Attn2Mask rescoring disagrees with OVAM's print.** OVAM prints Attn2Mask at 68.7 / 55.0. Supplement Table `tab:s_released` (rescored from released masks with OVAM's procedure) gives **.603 / .584**, on 485 and 668 images. The VOC-Sim gap is 8.4 points. OVAM, DAAM, GD and DatasetDM reproduce OVAM's print exactly on both benchmarks. **This should be checked and explained before submission**; it may be in F's audit scope.
4. **The OVAM token-opt row is few-shot.** OVAM's text says "a single token using only one image" per class. The released tokens are means of 2–3 per-image tokens (Entry 5). Either way the row is not training-free.
5. **Metric definitions differ.**
   - OVAM, GD and our paper: class IoU pooled over all images of the class, then averaged over classes. These match (`sec/4_protocol.tex` l. 4).
   - DAAM: averages over prediction–ground-truth mask pairs, with two class-set variants.
   - Real-image OVSS papers: dataset-level semantic mIoU, with (V21) or without (V20) a background class.
6. **Backbones differ.**
   - SD 1.5: OVAM, DiffSegmenter, GoCA SD 1.5 row.
   - SD 1.4: Attn2Mask's own paper.
   - SD 2.0-base: DAAM's own paper.
   - SD 2.1: NERVE.
   - SSD-1B: SPARK.
   - CLIP only: PEARL.
7. **Prompting differs.** GoCA uses GPT-4o-modulated prompts that list the objects present. DiffSegmenter uses BLIP captions. Neither uses the class word alone, so neither matches our class-word query, even before the image domain is considered.

### 2.4 Which benchmarks GoCA reports
GoCA (CVPR 2026) reports:
- **Table 1, p. 35866**: segmentation on **real-image** semantic segmentation benchmarks only — Pascal VOC 2012, Pascal-Context-59, COCO-Object, Cityscapes and ADE20K — with SD 1.5, SDXL, PixArt-Σ and Flux;
- **Table 2, p. 35867**: an ablation on Pascal VOC 2012 (SD 1.5 and SDXL);
- **Table 3, p. 35868**: *generation* quality (FID and CLIP score on COCO-30k) when GoCA is plugged into S-CFG.

It reports **nothing on VOC-sim, COCO-cap or any generated-image segmentation set**. GoCA can only appear in our paper as a **rerun** (N's #3 rerun, already under way in `comparisons/goca/`). It cannot appear as a published number on our benchmarks.

### 2.5 PEARL positioning (suggested text)
> PEARL [Pei et al., CVPR 2026] is a training-free open-vocabulary segmenter built on CLIP, not on a diffusion model. It aligns keys to queries inside CLIP's last self-attention block and refines logits with a text-aware Laplacian propagation. It is evaluated on real-image benchmarks (64.1 mIoU on VOC with background, 43.2 average over eight benchmarks). We cite it as evidence that "align, then propagate within an affinity graph" also helps outside diffusion models. We do not compare numbers, because it neither generates nor reads attention from generated images.

Recommended position: related work only, and no row in any table. This matches JC #3 ("PEARL cited for positioning only").

### 2.6 BibTeX (official sources; keys follow `paper/references.bib` style)

These entries are **missing** from `paper/references.bib`. Table 5 currently cites no key for them (JC #6).

```bibtex
@inproceedings{yoshihashi2024attn2mask,
  author    = {Yoshihashi, Ryota and Otsuka, Yuya and Doi, Kenji and Tanaka, Tomohiro and Kataoka, Hirokatsu},
  title     = {Exploring Limits of Diffusion-Synthetic Training with Weakly Supervised Semantic Segmentation},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2024},
  pages     = {2300--2318},
  note      = {arXiv:2309.01369 (v1 titled ``Attention as Annotation: Generating Images and Pseudo-masks for Weakly Supervised Semantic Segmentation with Diffusion''), which is the version OVAM cites},
  url       = {https://openaccess.thecvf.com/content/ACCV2024/html/Yoshihashi_Exploring_Limits_of_Diffusion-Synthetic_Training_with_Weakly_Supervised_Semantic_Segmentation_ACCV_2024_paper.html}
}

@inproceedings{li2023groundeddiffusion,
  author    = {Li, Ziyi and Zhou, Qinye and Zhang, Xiaoyun and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  title     = {Open-vocabulary Object Segmentation with Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023},
  pages     = {7667--7676},
  url       = {https://openaccess.thecvf.com/content/ICCV2023/html/Li_Open-vocabulary_Object_Segmentation_with_Diffusion_Models_ICCV_2023_paper.html}
}

@inproceedings{wu2023datasetdm,
  author    = {Wu, Weijia and Zhao, Yuzhong and Chen, Hao and Gu, Yuchao and Zhao, Rui and He, Yefei and Zhou, Hong and Shou, Mike Zheng and Shen, Chunhua},
  title     = {{DatasetDM}: Synthesizing Data with Perception Annotations Using Diffusion Models},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  pages     = {54683--54695},
  year      = {2023},
  doi       = {10.52202/075280-2383},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/ab6e7ad2354f350b451b5a8e14d04f51-Abstract.html}
}

@article{wang2025diffsegmenter,
  author  = {Wang, Jinglong and Li, Xiawei and Zhang, Jing and Xu, Qingyuan and Zhou, Qin and Yu, Qian and Sheng, Lu and Xu, Dong},
  title   = {Diffusion Model is Secretly a Training-Free Open Vocabulary Semantic Segmenter},
  journal = {IEEE Transactions on Image Processing},
  volume  = {34},
  pages   = {1895--1907},
  year    = {2025},
  doi     = {10.1109/TIP.2025.3551648}
}
```

**Replace** the current arXiv entry `mahatha2025nerve` with the WACV version. The official CVF author string is "Mahatha, Kunal", not "Mahatha B, Kunal":

```bibtex
@inproceedings{mahatha2026nerve,
  author    = {Mahatha, Kunal and Dolz, Jose and Desrosiers, Christian},
  title     = {{NERVE}: Neighbourhood \& Entropy-Guided Random-Walk for Training Free Open-Vocabulary Segmentation},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026},
  pages     = {3243--3253},
  url       = {https://openaccess.thecvf.com/content/WACV2026/html/Mahatha_NERVE_Neighbourhood__Entropy-Guided_Random-Walk_for_Training_Free_Open-Vocabulary_Segmentation_WACV_2026_paper.html}
}
```

These existing entries were checked against the official pages and are correct as they stand:
- `marcosmanchon2024ovam`: CVPR 2024, pp. 9242–9252;
- `tang2023daam`: ACL 2023, pp. 5644–5659;
- `meng2026making` (GoCA): CVPR 2026, pp. 35861–35871;
- `pei2026pearl`: CVPR 2026, pp. 17927–17937.

`mahatha2026spark` stays an arXiv entry. No venue was found (see 2.7).

### 2.7 Could not verify
- **DiffSegmenter, TIP 2025 version.** The IEEE Xplore full text was not accessible. The numbers, table numbers and pages above come from **arXiv v3** (https://arxiv.org/abs/2309.02773). The bibliographic data comes from Crossref (DOI 10.1109/TIP.2025.3551648). The TIP version may number its tables differently. The same three Table 1 numbers appear in GoCA's Table 1, which supports them. NERVE's Table 2 cites DiffSegmenter as "PAMI'25", which is wrong; the venue is TIP.
- **SPARK venue.** arXiv 2602.00516 v1 (31 January 2026) uses the ICML 2026 template, but I found no acceptance at ICML 2026 or elsewhere. Treat it as a preprint.
- **Grounded Diffusion SD version.** The main paper says "pre-trained Stable Diffusion" without a version. Not checked in the GD supplementary material.
- **ICPR 2027 self-review length.** The sources say both one and two pages (see 1.3). The template and instructions page say two.
- **ICPR 2027 dates** are labelled "tentative" by the organizers. Re-check https://www.icpr2027.com/web/dates in February 2027.
- **The trial LNCS page count** (about 13.5 pages) used Times through `mathptmx` instead of `newtx`, and unchanged float sizes. Real numbers will move by a fraction of a page.

**Primary sources used:**
- ICPR 2027: https://www.icpr2027.com/, https://www.icpr2027.com/web/dates, https://www.icpr2027.com/web/instructions, https://www.icpr2027.com/web/cfp, https://www.icpr2027.com/web/submitpaper, and the long-paper template zip linked there.
- OVAM: https://openaccess.thecvf.com/content/CVPR2024/papers/Marcos-Manchon_Open-Vocabulary_Attention_Maps_with_Token_Optimization_for_Semantic_Segmentation_in_CVPR_2024_paper.pdf, plus https://arxiv.org/abs/2403.14291 for the supplementary.
- DAAM: https://aclanthology.org/2023.acl-long.310.pdf
- Attn2Mask: https://openaccess.thecvf.com/content/ACCV2024/papers/Yoshihashi_Exploring_Limits_of_Diffusion-Synthetic_Training_with_Weakly_Supervised_Semantic_Segmentation_ACCV_2024_paper.pdf
- Grounded Diffusion: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Open-vocabulary_Object_Segmentation_with_Diffusion_Models_ICCV_2023_paper.pdf
- DatasetDM: https://proceedings.neurips.cc/paper_files/paper/2023/file/ab6e7ad2354f350b451b5a8e14d04f51-Paper-Conference.pdf
- GoCA: https://openaccess.thecvf.com/content/CVPR2026/papers/Meng_Making_Training-Free_Diffusion_Segmentors_Scale_with_the_Generative_Power_CVPR_2026_paper.pdf
- NERVE: https://openaccess.thecvf.com/content/WACV2026/papers/Mahatha_NERVE_Neighbourhood__Entropy-Guided_Random-Walk_for_Training_Free_Open-Vocabulary_Segmentation_WACV_2026_paper.pdf
- DiffSegmenter: https://arxiv.org/abs/2309.02773
- SPARK: https://arxiv.org/abs/2602.00516
- PEARL: https://openaccess.thecvf.com/content/CVPR2026/papers/Pei_PEARL_Geometry_Aligns_Semantics_for_Training-Free_Open-Vocabulary_Semantic_Segmentation_CVPR_2026_paper.pdf

**Slides:** #9 and the #3 published-block slide can be built directly from Parts 1.2–1.4 and 2.2–2.5.
**Open points:**
- Fix `supplementary.tex` l. 434 wording (JC #6).
- Explain the Attn2Mask gap: printed .687 against rescored .603.
- Add the five BibTeX entries.
- Decide which floats move to the supplement under the 15-page LNCS budget.
