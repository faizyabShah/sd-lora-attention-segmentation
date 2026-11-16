### Entry 6 — N (Narmeen), 2026-11-16
**Item:** #6 Consistency — status board line "#6 consistency"
**Kind:** text
**What was done:** Read every numeric claim in the main paper (`paper/sec/0_abstract.tex`–`7_discussion.tex`, `paper/tables/tab_*.tex`) and the current supplementary (`paper/supplementary.tex`) at shared-repo commit `f14bd19`. Traced each claim to a committed result file. Where no report states a number, it was recomputed. DenseCRF image counts and COCO classes improved came from the `sg_combined_v3` count shards. Per-class values came from `sg_combined_v3/per_class_iou.csv`. Layer-weight shares came from `sg24_a/results.json`. Per-class correlations came from `analysis/cross_package_classes/summary.json`. Torch and GPU versions came from the shard `environments` fields. Labels and refs were checked in the `.tex` and in every committed version of `attention_segmentation.pdf`. Nothing in the shared repo was edited. The scripts used are in this session's scratchpad (not committed).
**Where:** `comparisons/notes/entry_6_consistency.md`. Paths below are relative to the shared repo root. "pN" is the page of the committed PDF.
**Result:** Part 1 covers JC's five #6 items with fixes. Part 2 is the full claims table. Part 3 lists further inconsistencies. Summary counts are at the end of Part 2.
**Slides:** none yet. Planned for `slides/jc_meeting/sections/c6_N.tex`: one slide on the 0.829/0.830 decision, one on raw/CRF mixing and the top fixes.
**Open points:** (1) "Eq. ??" cannot be reproduced from any committed source or PDF (item 1.3); we need to know which PDF JC read. (2) The three missing citations are proposed from memory and must be checked against DOI/arXiv before entering `references.bib` (its header rule). (3) Three numbers are not traced (Part 2). (4) F audits this list (workplan Section 2.2 #6).

**Review — F, <date>:** pending

Notation: main-paper objects as printed (Table 3, Fig. 2, Sec. 5.3). Supplementary objects carry an "S" (Tab. S5 = "Table 5" in `supplementary.pdf`). Run names are used only here: **v1/v2/v3** = `controlled_attention_cv/sg_combined_v1|v2|v3`. **R0(+R2)** = fine-only (+DenseCRF). **R1+R2 / V3+R2** = BRACE.

---

## Part 1 — JC's #6 items

### 1.1 One estimate throughout: 0.829 vs 0.830, 0.696 vs 0.697

**Where each value comes from.** The same pipeline was generated and scored three times. Protocol Section 135 states that v3 R1 is identical code to v2 V3. Floating-point differences in captured attention move the numbers slightly between generations.

| Run (protocol Section, jobs) | File and row | VOC forced OOF | COCO frozen | Fine-only reference (R0+R2) in the same run |
|---|---|---:|---:|---:|
| **v2** (Section 127; 238514) | `sg_combined_v2/report.md` row `V3+R2` | **0.8304** | **0.6966** | 0.8231 / 0.6622 |
| **v3** (Section 132; 238580→238581) | `sg_combined_v3/report.md` row `R1+R2` (line 13) | **0.8293** | **0.6964** | 0.8230 / 0.6623 |
| SG-A3G (Section 131; 238557→238558) | `sg_a3g/report.md` row `V3+R2` | 0.8292 | 0.6967 | 0.8228 / 0.6623 |

v3 also gives the in-sample full-VOC value **0.8303** (threshold 0.55; `report.md` line 13). That value prints as ".830" in Table 4.

**Every occurrence:**

| Location (PDF page) | Printed | Run / quantity |
|---|---|---|
| `sec/0_abstract.tex:2` (p1) | 0.829, 0.696 | v3, forced OOF / frozen |
| `sec/1_intro.tex:7` (p1) | 0.829, 0.696 | v3 |
| `tables/tab_progression.tex:15` (Table 1, p4) | .829, .696 | v3 |
| `sec/5_analysis.tex:35` (Sec. 5.3, p5) | **0.830**, **0.697** | **v2** |
| `tables/tab_hierarchy.tex:20` (Table 3, p7) | **.8304**, **.6966** | **v2** |
| `tables/tab_hierarchy.tex:21` (Table 3, second cap) | .8290, .6964 | v2 V4 (different variant; its .6964 equals v3's BRACE COCO) |
| `tables/tab_refine.tex:12` (Table 4, p7) | **.830**, .696 | v3, but **in-sample** full VOC (0.8303), not OOF |
| `tables/tab_published.tex:19` (Table 5, p8) | .829, .696 | v3 |
| `sec/6_comparison.tex:3` (p7) | 0.125, 0.114 | derived from .829/.696 |
| `supplementary.tex:59–60` (Tab. S1, p1) | .8292/.6967; .8304/.6961 (GoCA orientation) | SG-A3G (third generation) |
| `supplementary.tex:199` (Tab. S5, p5) | .8293, .6964 | v3 |
| `supplementary.tex:253, 256` (Tab. S6, p5) | 83.0, 69.6 (in-sample); caption "0.823 and 0.829" | v3 |
| `supplementary.tex:279` (Tab. S7, p6) | .8293, .6964 | v3 |
| `supplementary.tex:452` (Tab. S11, p10) | .829, .696 | v3 |
| (coincidence) `tables/tab_seed.tex:15` (Table 2 top) | **.696** in bold | CA-only 8+16+64 px, in-sample at 0.45 (`a2_ca_r.tex` 0.6957). Unrelated, but a second bold ".696" in the paper |

**Recommendation: canonical = v3 R1+R2, printed as 0.829 (forced OOF) and 0.696 (frozen).** Reasons:
1. The owner override that makes this the paper's method (protocol Section 135) cites the v3 run as its evidence ("SG-combined-v3 run, R1+R2 vs R0+R2": 0.8293, +0.0063, CI [+0.0010, +0.0115], COCO +0.0341). Its config is `configs/owner_default_v3.json`.
2. v3 is already the source of Table 1's last row, Table 4, Fig. 6 (`scripts/paper/figure_coarse_to_fine.py:97–117`), all qualitative figures (`outputs/segmentation/paper_figures/story_selection.json` "rule": "V3+R2 = SG-combined-v3 R1+R2"), and Tab. S5, S6, S7 and S11. Only Sec. 5.3, Table 3 and the left and middle panels of Fig. 3 use v2. Changing those is the smallest edit.
3. `paper/README.md:13` already names v3 as the headline.

**Proposed fixes:**
- `sec/5_analysis.tex:35`, replace
  "The one-sided cap is the only variant that beats both fine-only propagation and the plain hierarchy in every fold: 0.830 after DenseCRF, $+0.012$ over the plain hierarchy (CI $[+0.006,+0.013]$), with 0.697 on COCO-Cap."
  with
  "The one-sided cap is the only variant that beats both fine-only propagation and the plain hierarchy in every fold. After DenseCRF it gains $+0.012$ mIoU over the plain hierarchy (paired mean image IoU $+0.009$, 95\% CI $[+0.006,+0.013]$) and keeps the COCO-Cap gain (Table~\ref{tab:hierarchy})."
  This avoids a second absolute value and also fixes the CI mismatch (item 3.6).
- `tables/tab_hierarchy.tex:25`, caption. Replace "All rows share one generation of the 490 VOC-Sim and 707 COCO-Cap images, the same seed and SA calls 10--20." with
  "Rows come from two generations of the 490 VOC-Sim and 707 COCO-Cap images with identical settings (the three two-sided rows without completion from one, the other rows from the other; fine-only reference .8228 and .8231). Within a generation, every row has the same seed and SA calls 10--20, and wins are counted against that generation's fine-only row. The cap row is a separate generation of the method reported in Tables~\ref{tab:progression}, \ref{tab:refine} and \ref{tab:published} (.829 / .696); three generations of the method give .8292--.8304 VOC and .6964--.6967 COCO-Cap."
  Alternative (F's side, new run): regenerate the two-sided rows inside the v3 capture. Then Table 3 is one run and the cap row prints .8293/.6964.
- `tables/tab_refine.tex:15`, caption. Replace "Full-VOC mIoU" with "Full-VOC mIoU (in-sample: thresholds chosen on all of VOC-Sim; the forced out-of-fold values are .823 and .829, Table~\ref{tab:progression})".
- `tables/tab_progression.tex:18`, caption. Replace "Runs differ by at most 0.0003 on a shared reference." with "Runs differ by at most 0.0003 on the shared fine-only reference and by at most 0.0012 on the final method (.8292--.8304 across three generations)."
- `supplementary.tex:63`, Tab. S1 caption. Add "This rerun is a third generation of the method (.8292 vs .8293 in the main paper)."

### 1.2 No raw and DenseCRF values in one comparison

| Location (page) | What is mixed | Proposed fix |
|---|---|---|
| `sec/5_analysis.tex:24` (Sec. 5.2, p4–5 in `.tex`; the stale PDF has older numbers, item 3.1) | "this already beats fine-only propagation (0.693 vs. 0.662 **after DenseCRF**, Table 3). On VOC-Sim, however, it falls short (0.806 vs. 0.810 OOF)". The VOC pair is **raw** (v2 V1 .8061 vs V0 .8098). The COCO pair is after CRF (V1+R2 .6925 vs V0+R2 .6622) | Use one stage throughout. After DenseCRF: "(COCO-Cap 0.693 vs.\ 0.662) ... it falls short (0.818 vs.\ 0.823 forced OOF), both after DenseCRF (Table~\ref{tab:hierarchy})". Values from v2 V1+R2 .8184 and V0+R2 .8231. Or raw throughout: COCO 0.678 vs 0.660 (V1 .6783, V0 .6598), VOC 0.806 vs 0.810. Prefer after-CRF: it is the metric Table 3 counts wins on |
| `sec/0_abstract.tex:2` and `sec/5_analysis.tex:35` (Sec. 5.3), `tables/tab_hierarchy.tex:25` | "costs up to five points" / "0.759 raw, −5 points" / "loses up to 5 VOC points" are **raw** (v1 C4 −0.0505). They sit in a paragraph and table whose other claims are after DenseCRF (0.830, wins). After DenseCRF the loss is −3.1 points (C4+R2 .7917 vs .8228, Δ −0.0311) | Abstract: "costs up to three points after DenseCRF (five before)". Sec. 5.3: "is far worse (0.792 after DenseCRF, $-3$ points; $-5$ before)". Table 3 caption: "loses up to 3 VOC points after DenseCRF (5 raw)" |
| `tables/tab_hierarchy.tex:7` (Table 3) | Columns are "VOC raw", "VOC + CRF", "COCO + CRF": a raw VOC column without a raw COCO column | Either drop "VOC raw" and move it to Tab. S5, or add "COCO raw" from the same reports (v1 C*, v2 V* rows) so each stage has both datasets |
| `tables/tab_published.tex` (Table 5, p8) and `supplementary.tex:445–452` (Tab. S11) | OVAM's published and released values use DenseCRF (released `ovam_sa` masks, Section 41). Attn2Mask uses DenseCRF on COCO-Cap only. DAAM is raw (threshold 0.4). Grounded Diffusion and DatasetDM are released masks. Our rows are after DenseCRF. Raw and CRF rows sit in one comparison without a marker | Add a "DenseCRF" column (✓/–, per protocol Section 41's procedure column) and our raw row: forced OOF .816 / frozen .682 (v3 R1: 0.8157 / 0.6820). This matches the workplan Section 3 contract ("reported with and without") |
| `sec/5_analysis.tex:37` (Sec. 5.3) and Fig. 4 | "COCO cat: IoU 0.58, against 0.81 with the cap" are **raw-map** IoUs. `story_selection.json` rule: "cap contrast = SG-combined-v2 V3 (0.55) vs V1 ... raw". The paragraph's numbers are otherwise after CRF | Say "(raw maps; COCO cat: IoU 0.58, against 0.81 with the cap)" and add "raw maps, before DenseCRF" to the Fig. 4 caption |
| `sec/5_analysis.tex:63` (Sec. 5.5) | Ranking evidence is raw AP (−0.0006). The oracle tie is after DenseCRF (0.896 vs 0.895) | Give both stages for the oracle: "(0.887 vs.\ 0.884 raw, 0.896 vs.\ 0.895 after DenseCRF)" from `sg_threshold/report.md` "Ceiling" |
| `tables/tab_progression.tex` (Table 1) | Rows 1–4 have no DenseCRF and rows 5–8 do. The Δ of row 5 is the DenseCRF step itself. This is a progression, so acceptable, but rows 1–3 also use a fixed threshold of 0.45 | Leave the stage mix (it is the axis of the table). Table 1 is F's item #7 |

Acceptable as is (the comparison *is* raw vs CRF, and columns are labelled): Table 4, Fig. 6, Tab. S4, Tab. S5 columns, Tab. S8 blocks.

### 1.3 "Eq. ??" in Fig. 2

- **Current source.** `sec/5_analysis.tex:8`, the Fig. 2 caption, reads "CA seed $c$ (Eq.~\ref{eq:weights})". `eq:weights` is defined at `sec/3_method.tex:36`, and the committed PDF prints "Eq. 3" (p5).
- **History.** Before commit `7c8ad5d` (31 Oct) the caption read `\ref{eq:seed}`, and `eq:seed` was then defined in `sec/3_method.tex`. A check of every commit since 20 Oct that touched `paper/sec`, `paper/tables` or the main `.tex` found **no `\ref` without its `\label`**. A text dump of every committed `attention_segmentation.pdf` found **no "??"**. No committed PDF has the "??" JC reported.
- **Likely cause.** JC read a PDF built outside the repo (an Overleaf or email copy), from one LaTeX pass or with a stale `.aux` around the 31 Oct label rename (`eq:seed` → `eq:weights`, same commit).
- **Fix.** (a) Ask which PDF JC read, then rebuild both PDFs with the full sequence (`pdflatex`, `bibtex`, `pdflatex`, `pdflatex`, or `latexmk -pdf`) and commit them. (b) Make the caption self-contained so it survives renumbering: "CA seed $c$ (Sec.~\ref{sec:method_weight}, Eq.~\eqref{eq:weights})". (c) Add a check before sending: `gs -q -dNOPAUSE -dBATCH -sDEVICE=txtwrite -sOutputFile=- attention_segmentation.pdf | grep -c '??'` must print 0.

### 1.4 Citations missing for Attn2Mask, Grounded Diffusion, DatasetDM

Occurrences without `\cite`:
- `tables/tab_published.tex:10` (Attn2Mask), `:14` (Grounded Diffusion), `:15` (DatasetDM). Table 5, p8.
- `sec/6_comparison.tex:3`: "It also exceeds DatasetDM, which trains an extra decoder".
- `supplementary.tex:434` ("Grounded Diffusion and DatasetDM train a module or decoder") and `:448–450` (Tab. S11 rows).
- `paper/references.bib` has no entry for any of the three. Sec. 2 (Related work) does not mention them.

Proposed entries. These are from memory: check each against its DOI or arXiv page before adding, per the `references.bib` header rule and protocol Section 29. Also confirm they are the works OVAM's Table 1 cites.

```bibtex
@misc{yoshihashi2023attention,      % Attn2Mask
  author = {Yoshihashi, Ryota and Otsuka, Yuya and Doi, Kenji and Tanaka, Tomohiro and Kataoka, Hirokatsu},
  title  = {Attention as Annotation: Generating Images and Pseudo-masks for Weakly Supervised Semantic Segmentation with Diffusion},
  year   = {2023}, eprint = {2309.01369}, archivePrefix = {arXiv}
}
@inproceedings{li2023open,          % Grounded Diffusion
  author    = {Li, Ziyi and Zhou, Qinye and Zhang, Xiaoyun and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  title     = {Open-vocabulary Object Segmentation with Diffusion Models},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023}, pages = {7667--7676}
}
@inproceedings{wu2023datasetdm,     % DatasetDM
  author    = {Wu, Weijia and Zhao, Yuzhong and Chen, Hao and Gu, Yuchao and Zhao, Rui and He, Yefei and Zhou, Hong and Shou, Mike Zheng and Shen, Chunhua},
  title     = {{DatasetDM}: Synthesizing Data with Perception Annotations Using Diffusion Models},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023}
}
```

Text changes: `tab_published.tex:10` `Attn2Mask~\cite{yoshihashi2023attention}`, `:14` `Grounded Diffusion~\cite{li2023open}`, `:15` `DatasetDM~\cite{wu2023datasetdm}`. Same for the first mention in `sec/6_comparison.tex:3` and in `supplementary.tex:434`. This overlaps N's #3 "published block" item, so do it once there.

### 1.5 Supplementary pointers

**Main → supplementary.** Ten pointers say only "(supplementary material)", with no section, table or figure:

| Location | Current | Proposed pointer (S-numbering, see below) |
|---|---|---|
| `sec/2_related.tex:7` | "(supplementary material)" | "(Sec.~S7, Tab.~S9)" |
| `sec/3_method.tex:32` (footnote) | "(supplementary material)" | "(Tab.~S1)" |
| `sec/3_method.tex:50` | "(supplementary material)" | "(Sec.~S2, Tab.~S2)" |
| `sec/3_method.tex:57` | "(proof in the supplementary material)" | "(proof in Sec.~S1)" |
| `sec/4_protocol.tex:10` | "The supplementary material reports the registered outcomes in full." | "Section~S8 and Tab.~S10 report the registered outcomes in full." |
| `sec/5_analysis.tex:16` | "(see the supplementary seed gallery)" | "(Fig.~S4)". There is no "seed gallery" in the current supplementary; the old one had it |
| `sec/5_analysis.tex:18` | "(supplementary material)" | "(Fig.~S1)" |
| `sec/5_analysis.tex:65` | "(supplementary material)" | "(Sec.~S5, Tab.~S7)" |
| `sec/6_comparison.tex:3` | "are in the supplementary material" | "are in Sec.~S9 (Tab.~S11)" |
| `sec/7_discussion.tex:4` | "(supplementary material)" | "(Sec.~S7, Tab.~S8)" |

**Supplementary → main.** Fifteen hard-coded references. All are correct for today's numbering, but any restructuring will break them silently. #8 moves Sec. 6 into Sec. 5, which would change "Table 5" and possibly "Section 5.5". The references are `supplementary.tex:35` (Table 2), `:38` (Eq. 3), `:73` (Eq. 5), `:131` (Fig. 2, Table 2), `:136` (Fig. 2), `:162` (Fig. 3), `:184` (Table 3), `:205` (Table 3), `:265` (Section 5.5), `:315`, `:320`, `:330` (Fig. 2), `:360` (Sec. 3.3), `:434` (Table 5).

**Fix.**
1. In `supplementary.tex`, after `\begin{document}`: `\renewcommand{\thesection}{S\arabic{section}}`, and the same for `\thetable`, `\thefigure` and `\theequation`. Today the supplementary prints "Table 1", "Fig. 2", "Sec. 5" and "Eq. (1)", which collide with the main paper's own numbers. For example, `supplementary.tex:286` says "loses mainly on sheep (Sec. 5)", meaning supplementary Sec. 5, right after citing main Sec. 5.5.
2. Load `xr-hyper` in `preamble.tex` (before `hyperref`, which the preamble already precedes). Then add `\externaldocument[S-]{supplementary}` in the main file and `\externaldocument[M-]{attention_segmentation}` in the supplementary. Replace each hard-coded number with `\ref{M-tab:seed}` and similar, and the main-paper pointers above with `\ref{S-sec:s_alternatives}` and similar. Overleaf needs a `latexmkrc` for `xr`. Check that ICPR (#9) allows cross-PDF references.

---

## Part 2 — Every numeric claim and its source

Status values: **traced, matches** / **traced, differs (…)** / **not traced**. "Matches" means equal at the printed precision. Labels: forced OOF, OOF (selection procedure), frozen, in-sample, published, diagnostic (uses GT or a non-selection measure), descriptive, fixed fact/setting. Pure restatements of the same number in the same paragraph are merged into one row. Table rows are one row each; their cell count is given.

### Main paper

| # | Location | Printed | Measures | Label | Source (file; key/row; protocol Section) | Status |
|---|---|---|---|---|---|---|
| M1 | `0_abstract.tex:2` | 16, 32, 64 pixels | SA scales | setting | v3 shard `settings.sa_modules`; Sec. 135 | traced, matches |
| M2 | `0_abstract.tex:2` | 13% | proposal mass removed in background (VOC) | diagnostic | `sg_combined_v3/results.json` `diagnostics.voc.gt_cap_loss_16_to_32.R1` L_background 0.132; Sec. 132 | traced, matches |
| M3 | `0_abstract.tex:2` | 0.5% | same, inside object | diagnostic | same, L_GT 0.0052 | traced, matches |
| M4 | `0_abstract.tex:2` | "up to five points" | exact conservation (both transitions) vs fine-only, VOC | forced OOF, **raw** | `sg_combined_v1/report.md` C4 Δ −0.0505; Sec. 123 | traced, matches (raw only; −3.1 after CRF, item 1.2) |
| M5 | `0_abstract.tex:2` | "six times more" | CRF gain on COCO, capped vs fine-only | frozen | v3 `cells_vs_R0.coco`: R1→R1+R2 +0.0145, R0→R0+R2 +0.0024 | traced, matches |
| M6 | `0_abstract.tex:2` | five folds | fold count | setting | Sec. 5a; `configs/voc_sim_folds.csv` | traced, matches |
| M7 | `0_abstract.tex:2` | 0.829 "out-of-fold" | BRACE VOC mIoU | forced OOF | `sg_combined_v3/report.md:13` R1+R2 0.8293 | traced, differs (label: forced OOF; the registered selection procedure's OOF is 0.8230, Sec. 5a.2 and 135) |
| M8 | `0_abstract.tex:2` | 0.696 | BRACE COCO mIoU | frozen | same row, 0.6964 | traced, matches |
| M9 | `0_abstract.tex:2` | 0.704, 0.582 | OVAM class word | published | OVAM Table 1 per Sec. 114 item 3; `comparisons/released_baselines/summary.json` 0.7036 / 0.5823 (Sec. 41) | traced, matches |
| M10 | `1_intro.tex:5` | 31 calls, 4 scales, 16 layers per type, 8 heads | SD 1.5 structure | fact | Sec. 2 (calls 0–30); v3 spec | traced, matches |
| M11 | `1_intro.tex:7` | 0.829, 0.696, 0.704, 0.582 | as M7–M9 | as M7–M9 | as M7–M9 | traced, differs (0.829 labelled "out-of-fold", as M7) |
| M12 | `1_intro.tex:20` (Fig. 1 cap.) | 16-, 32-, 64-px; 2×2; calls 10–20 | pipeline | setting | Sec. 135 pipeline | traced, matches |
| M13 | `3_method.tex:7` | 512×512, 64×64, 30 steps, 64/32/16/8 px | generation | fact | Sec. 2 table | traced, matches |
| M14 | `3_method.tex:17` | eight heads | heads | fact | v3 spec `sa` "eight heads" | traced, matches |
| M15 | `3_method.tex:26` | 16 CA layers (5/5/5/1); 11 used | CA layers | setting | v3 spec `ca` (64-px 0,1,13,14,15; 16-px 4,5,7,8,9; 8-px 6) | traced, matches |
| M16 | `3_method.tex:30` | five of eleven votes | uniform share of 64-px layers | fact | as M15 | traced, matches |
| M17 | `3_method.tex:32` | 32 pixels | reference block of the weighting | setting | `sg24_a/results.json` spec `ref_feature` `up_blocks.2.attentions.1…attn2` | traced, matches |
| M18 | `3_method.tex:32` (fn) | 0.99999 | Pearson, two orientations of the seed | diagnostic | `sg_a3g/report.md:31` seed_pearson_mean 0.999993; Sec. 131 | traced, matches |
| M19 | `3_method.tex:38` | "the 31 steps" | seed averaging | setting | Sec. 2: 30 scheduler steps, 31 UNet calls | traced, differs (should be "31 calls"; `3_method.tex:7` says 30 steps) |
| M20 | `3_method.tex:38` | 84% | weight on 8/16-px layers | descriptive | `sg24_a/results.json` `layer_weights.voc.A`: 0.243 + 0.597 | traced, matches |
| M21 | `3_method.tex:50` | steps 10–20 | SA window | setting | `sg22` MID (Sec. 102); Sec. 135 | traced, matches |
| M22 | `3_method.tex:61` | τ = 0.55 | global threshold | in-sample (also chosen in all 5 folds) | v3 report full-VOC thr 0.550; v2 `contrasts` folds candidate_threshold 0.55 ×5 | traced, matches |
| M23 | `3_method.tex:66` | 10; 80, 13; 3; 3; five iterations | DenseCRF parameters | setting | v3 shard `settings.crf` [10.0, 80, 13, 3.0, 3, 5.0] | traced, matches |
| M24 | `4_protocol.tex:4` | 490, 20 classes, 707 | benchmark sizes | fact | `released_baselines/summary.json` images 490 / 707 | traced, matches |
| M25 | `4_protocol.tex:4` | 30 steps, guidance 7.5 | generation | setting | Sec. 2 | traced, matches |
| M26 | `4_protocol.tex:7` | five folds; 0.010; 0.005; 12 of 20; 0.05; 4 of 5 | promotion rule | protocol | Sec. 5, 5a | traced, matches (omits 5a.3's positive-pooled-OOF condition) |
| M27 | `4_protocol.tex:7` | 95%, 20,000 resamples | bootstrap | protocol | Sec. 5a.4 | traced, matches (workplan Sec. 3 contract says 2000; item 3.20) |
| M28 | `4_protocol.tex:10` | 4/5 | SA-guided weights, folds won | forced OOF | `sg24_a/results.json` `forced_oof["A3/FINAL"].fold_wins` 4 | traced, matches |
| M29 | `4_protocol.tex:10` | 5/5 | cap vs both references | forced OOF | v2 report V3+R2 5/5; `contrasts["V3+R2_vs_V1+R2"].fold_wins` 5 | traced, matches |
| M30 | `4_protocol.tex:10` | 0.05 guard | class guard | protocol | Sec. 127 (worst class −0.057 to −0.104) | traced, matches |
| M31 | `5_analysis.tex:8` (Fig. 2 cap.) | best / upper-third / lower-third / worst; "median-IoU image" | column selection rule | fixed rule | `paper_figures/story_selection.json` `groups.stages` | traced, differs (COCO person column has image_quantile 0.75, IoU 0.6316; the median image is 0.2546) |
| M32 | `tab_progression.tex:8` | .665 | CA, all scales | OOF, fixed thr 0.45 | `ca_confirmation/report.md:7` 0.6653; Sec. 27 | traced, matches |
| M33 | `tab_progression.tex:9` | .691, +.026 | no 32-px CA | OOF | same, 0.6914, OOF gain +0.0260 | traced, matches |
| M34 | `tab_progression.tex:10` | .789, +.033ᵃ (vs .756) | one 64-px SA layer | OOF, thr 0.45 | `sa_confirmation/report.md:7` 0.7887 / 0.7556, +0.0330; Sec. 33 | traced, matches (Δ is not against the row above; JC #7) |
| M35 | `tab_progression.tex:11` | .800, +.012, .625 | two passes | forced OOF / frozen | `sg22/report.md` ALL-P2 0.7997 vs ALL-P1 0.7873, COCO 0.6246; Sec. 102 | traced, matches (Δ against 0.7873 of sg22, not the .789 above) |
| M36 | `tab_progression.tex:12` | .813, +.014, .637 | + DenseCRF | forced OOF / frozen | sg22 ALL-P2+R2 0.8132 (+0.0135), 0.6372 | traced, matches |
| M37 | `tab_progression.tex:13` | .817, +.004, .644 | calls 10–20 | forced OOF / frozen | sg22 MID-P2+R2 0.8168 (+0.0036), 0.6438 | traced, matches (forced OOF; registered path 0.8158, Sec. 114 item 2; caption says "OOF") |
| M38 | `tab_progression.tex:14` | .823, +.006, .662 | SA-guided weights | forced OOF / frozen | `sg24_a/report.md` A3/FINAL 0.8230 (+0.0061), 0.6620 | traced, matches |
| M39 | `tab_progression.tex:15` | .829, +.006, .696 | coarse-to-fine + cap | forced OOF / frozen | v3 R1+R2 0.8293 (+0.0063), 0.6964 | traced, matches |
| M40 | `tab_progression.tex:18` (cap.) | 0.45; .756; "at most 0.0003" | run-to-run spread | — | fine-only: v1 .8228, v2 .8231, v3 .8230, sg25 .8228; sg22 .8132 vs P3-C .8135 | traced, differs (holds for the reference; the method spreads 0.0012, item 1.1) |
| M41 | `5_analysis.tex:16` | 0.17 | 64-px CA alone, mIoU@0.45 | in-sample | `paper/evidence/tables/a2_ca_r.tex` row 64: 0.1734; Sec. 20 | traced, matches |
| M42 | `5_analysis.tex:16` | 0.665 → 0.691, 4/5 | drop 32-px CA | OOF | as M32–M33 | traced, matches |
| M43 | `5_analysis.tex:16` | paired +0.022, CI [+0.015, +0.029] | same | paired mean-image IoU | `ca_confirmation/report.md:7` +0.0217 [+0.0145, +0.0293] | traced, matches (a different statistic from Table 1's +.026; item 3.6) |
| M44 | `5_analysis.tex:18` | five of eleven; 84% | | | as M16, M20 | traced, matches |
| M45 | `5_analysis.tex:18` | 0.714 → 0.705 | CA-only, uniform vs SA-guided | forced OOF | `sg24_a/report.md` A0/CA 0.7135, A3/CA 0.7049 | traced, matches |
| M46 | `5_analysis.tex:18` | 0.817 → 0.823, 4/5 | final mask | forced OOF | A0/FINAL 0.8169, A3/FINAL 0.8230, wins 4 | traced, matches |
| M47 | `5_analysis.tex:18` | 0.644 → 0.662 | COCO | frozen | 0.6437 → 0.6620 | traced, matches |
| M48 | `tab_seed.tex:9–15` (Table 2 top, 14 cells) | .563/.846 … .696/.913 | CA-only mIoU@.45 and AP per resolution subset | in-sample, descriptive | `a2_ca_r.tex` rows 8, 16, 32, 64, 8+16+32+64, 8+16, 8+16+64; Sec. 20 | traced, matches (14/14) |
| M49 | `tab_seed.tex:22–23` (6 cells) | .714/.817/.644; .705/.823/.662 | weighting | forced OOF / frozen | `sg24_a/report.md` | traced, matches |
| M50 | `tab_seed.tex:26` (cap.) | 0.45; 0.665 → 0.691 (4/5) | | OOF | CA-C | traced, matches (the same configurations print .666 and .696 in the top table, from a different run and statistic; item 3.17) |
| M51 | `5_analysis.tex:24` | 0.72 | 16-px SA alone, peak | raw, forced OOF | `sg25/report.md` COARSE16-P2 0.7225 (full 0.7231); Sec. 104 | traced, matches |
| M52 | `5_analysis.tex:24` | τ = 0.45 | fine SA best threshold | in-sample | v2 report V0 full thr 0.450 | traced, matches |
| M53 | `5_analysis.tex:24` | 0.693 vs 0.662 after DenseCRF | plain hierarchy vs fine-only, COCO | frozen | v2 V1+R2 0.6925, V0+R2 0.6622 | traced, matches |
| M54 | `5_analysis.tex:24` | 0.806 vs 0.810 OOF | same, VOC | forced OOF, **raw** | v2 V1 0.8061, V0 0.8098 | traced, matches (raw beside CRF in one comparison; item 1.2) |
| M55 | `5_analysis.tex:29` (Fig. 3 cap.) | 13%, 0.5% | as M2–M3 | diagnostic | v3 diagnostics | traced, matches |
| M56 | `5_analysis.tex:35` | 0.759 raw, −5 points | conservation, both transitions | forced OOF, raw | v1 C4 0.7592, −0.0505 | traced, matches |
| M57 | `5_analysis.tex:35` | 0.830 | cap after CRF | forced OOF | v2 V3+R2 0.8304 | traced, differs (canonical v3 0.8293 → 0.829; item 1.1) |
| M58 | `5_analysis.tex:35` | +0.012 | cap − plain hierarchy, after CRF | forced OOF Δ mIoU | v2 `contrasts["V3+R2_vs_V1+R2"].delta` 0.01194 | traced, matches |
| M59 | `5_analysis.tex:35` | CI [+0.006, +0.013] | same | paired mean-image IoU | same, `paired_mean_image_iou` mean 0.00904, CI [0.00570, 0.01271] | traced, differs (the CI belongs to +0.009 mean-image, not to +0.012 mIoU) |
| M60 | `5_analysis.tex:35` | 0.697 | cap, COCO | frozen | v2 V3+R2 0.6966 | traced, differs (canonical v3 0.6964 → 0.696) |
| M61 | `5_analysis.tex:37` | 58%, 7.6% | binding share; removed mass | diagnostic | Sec. 127 (v2) | traced, matches |
| M62 | `5_analysis.tex:37` | 13%, 0.5%, at most 1.6% | removed mass by class | diagnostic | Sec. 132 (cat 1.6%) | traced, matches |
| M63 | `5_analysis.tex:37` | 0.81 (COCO cat, cap) | image IoU, raw map | diagnostic | `story_selection.json` `groups.cap[1].v3_iou` 0.8103 | traced, matches |
| M64 | `5_analysis.tex:37` | 0.58 (COCO cat, conservation) | image IoU, raw map | diagnostic | appears only in the figure; not in `story_selection.json` | not traced |
| M65 | `tab_hierarchy.tex:10` (4 cells) | –, .8098, .8231, .6622 | fine only | forced OOF / frozen | v2 V0, V0+R2 | traced, matches |
| M66 | `tab_hierarchy.tex:11` | 1/5, .8061, .8184, .6925 | plain hierarchy | | v2 V1, V1+R2 | traced, matches |
| M67 | `tab_hierarchy.tex:14` | 2/5, .8034, .8185, .7038 | conservation 16→32 | | **v1** C2, C2+R2 | traced, matches (from v1, not the v2 run of the other rows) |
| M68 | `tab_hierarchy.tex:15` | 0/5, .7894, .8135, .7022 | conservation 32→64 | | **v1** C3 | traced, matches (v1) |
| M69 | `tab_hierarchy.tex:16` | 0/5, .7592, .7917, .6851 | conservation both | | **v1** C4 | traced, matches (v1) |
| M70 | `tab_hierarchy.tex:17` | 2/5, .8037, .8204, .7063 | conservation + completion | | v2 V2 (v1 C6 is .8035/.8201/.7063) | traced, matches |
| M71 | `tab_hierarchy.tex:20` | 5/5, .8157, .8304, .6966 | cap (ours) | | v2 V3, V3+R2 | traced, matches (differs from canonical v3 .8293/.6964) |
| M72 | `tab_hierarchy.tex:21` | 4/5, .8151, .8290, .6964 | cap at both transitions | | v2 V4 | traced, matches |
| M73 | `tab_hierarchy.tex:22` | 3/5, .8131, .8209, .6647 | cap + completion | | v2 V5 | traced, matches |
| M74 | `tab_hierarchy.tex:25` (cap.) | 490 / 707; calls 10–20; "one generation" | | | v1 and v2 are separate generations (fine-only .8228 vs .8231; Sec. 123, 127) | traced, differs (two generations; item 3.3) |
| M75 | `tab_hierarchy.tex:25` (cap.) | up to 5 VOC points; +0.012, CI [+0.006, +0.013] | | | as M56, M58, M59 | traced, differs (raw "5 points" beside CRF claims; CI statistic as M59) |
| M76 | `5_analysis.tex:44` (Fig. 4 cap.) | 95th, 5th percentile | selection rule | fixed rule | `story_selection.json` `groups.cap` quantiles 0.95, 0.95, 0.05 | traced, matches |
| M77 | `5_analysis.tex:50` | 278 better, 97 worse | VOC images, CRF on capped map | in-sample thr | recomputed from `sg_combined_v3/voc/shards` (R1 vs R1+R2 at 0.55, \|Δ\| > 0.005) | traced, matches (no committed report states it) |
| M78 | `5_analysis.tex:50` | 0.013 | VOC gain from CRF, capped | in-sample | v3 `cells_vs_R0.voc` R1 0.8174 → R1+R2 0.8303 | traced, matches |
| M79 | `5_analysis.tex:50` | +0.002; 243 worse | CRF on fine-only, COCO | frozen | v3 R0 0.6599 → R0+R2 0.6623; recomputed 354/243 | traced, matches |
| M80 | `5_analysis.tex:50` | six times; +0.014; 18 of 20 | CRF on capped, COCO | frozen | v3 R1 0.6820 → R1+R2 0.6964; recomputed 18/20 classes | traced, matches |
| M81 | `5_analysis.tex:50` | 427 better, 105 worse | COCO images | frozen | recomputed from `sg_combined_v3/coco/shards` | traced, matches |
| M82 | `tab_refine.tex:9` (4 cells) | .811, .880/.904, .660, .708/.882 | fine-only raw | in-sample VOC / frozen | v3 `cells_vs_R0` R0 | traced, matches |
| M83 | `tab_refine.tex:10` (6 cells) | .824, .883/.916, 291/132, .662, .703/.889, 354/243 | fine-only + CRF | | R0+R2; counts recomputed | traced, matches |
| M84 | `tab_refine.tex:11` (4 cells) | .817, .875/.915, .682, .731/.889 | capped raw | | R1 | traced, matches |
| M85 | `tab_refine.tex:12` (6 cells) | .830, .890/.910, 278/97, .696, .747/.886, 427/105 | capped + CRF | **in-sample** VOC | R1+R2 (0.8303) | traced, matches (in-sample .830 reads like the forced-OOF .830 of Table 3; label it, item 1.1) |
| M86 | `tab_refine.tex:15` (cap.) | 0.005; +0.002; +0.014; 18 of 20 | | | as above | traced, matches |
| M87 | `5_analysis.tex:57` (Fig. 5 cap.) | 90th, 10th percentile | selection rule | fixed rule | `story_selection.json` `groups.crf` 0.9, 0.9, 0.1 | traced, matches |
| M88 | `5_analysis.tex:63` | −0.0006 | COCO raw AP, capped − fine | diagnostic | `sg_combined_v3/report.md:34` | traced, matches |
| M89 | `5_analysis.tex:63` | 0.896 vs 0.895 | per-image oracle after CRF | diagnostic (GT) | `sg_threshold/report.md` Ceiling: 0.8961 / 0.8947; Sec. 134 | traced, matches |
| M90 | `5_analysis.tex:63` | 0.425 → 0.55 | best global threshold | in-sample | v3 report R0+R2 0.425, R1+R2 0.550 | traced, matches |
| M91 | `5_analysis.tex:65` | 0.35 | sheep's preferred threshold | diagnostic | Sec. 132: R0 best mean-image IoU 0.879 at 0.35 | traced, matches (Sec. 128 on v2 gives 0.30 by class IoU) |
| M92 | `5_analysis.tex:65` | 0.55; falls by 0.09 | sheep IoU at shared threshold | in-sample | v3 report sheep Δ −0.092 | traced, matches |
| M93 | `5_analysis.tex:65` | 1.1% of sheep pixels | cap loss inside sheep | diagnostic | Sec. 132: L_GT sheep 0.0114 | traced, differs (it is 1.1% of the proposal *mass* inside sheep GT, not of pixels) |
| M94 | `5_analysis.tex:65` | six rules; "None ... improved" | per-image thresholds | forced OOF | `sg_threshold/report.md` R1.T6+R2 +0.0004 (2/5) vs R1.T0+R2 | traced, differs (one rule gains +0.0004 in 2/5 folds; say "none improved by more than 0.001") |
| M95 | `5_analysis.tex:70` (Fig. 6 cap.) | 0.425 and 0.55 | thresholds | in-sample | as M90 | traced, matches |
| M96 | `6_comparison.tex:3` | 0.125, 0.114 | margin over OVAM | ours forced OOF / frozen vs published | .829 − .704; .696 − .582 | traced, matches |
| M97 | `6_comparison.tex:3` | "exceeds DatasetDM on both" | | | .829 > .803; .696 > .593 | traced, matches |
| M98 | `6_comparison.tex:3` | "Most of the COCO-Cap margin over our fine-only pipeline comes from the capped hierarchy" | share of margin | | .696 − .662 = .034 of the .114 over OVAM | traced, differs (the margin over fine-only *is* the hierarchy by definition; of the margin over OVAM it is 0.034 of 0.114, not most; item 3.11) |
| M99 | `tab_published.tex:9` | .662 / .484 | DAAM | published | Sec. 114; `released_baselines/summary.json` 0.6616 / 0.4842 | traced, matches |
| M100 | `tab_published.tex:10` | .687 / .550 | Attn2Mask | published | Sec. 114 item 3 (checked against arXiv 2403.14291 p. 6); `docs/PAPER_REVISION_HANDOFF.md:11`. Released masks rescore to .603 / .584 (Sec. 41) | traced, matches (only through the Sec. 114 record; the OVAM PDF is not committed) |
| M101 | `tab_published.tex:11` | .704 / .582 | OVAM | published | as M9 | traced, matches |
| M102 | `tab_published.tex:14` | .621 / .502 | Grounded Diffusion | published | `summary.json` 0.6213 / 0.5015 | traced, matches |
| M103 | `tab_published.tex:15` | .803 / .593 | DatasetDM | published | `summary.json` 0.8027 / 0.5928 | traced, matches |
| M104 | `tab_published.tex:18` | .823 / .662 | fine-only + CRF | forced OOF / frozen | `sg24_a` A3/FINAL 0.8230 / 0.6620 (= v3 R0+R2 0.8230 / 0.6623) | traced, matches |
| M105 | `tab_published.tex:19` | .829 / .696 | BRACE | forced OOF / frozen | v3 R1+R2 | traced, matches |
| M106 | `7_discussion.tex:4` | 45 comparisons | registered operator families | — | `paper/tables/generated/decision_map_families.tex` (34 + 11 rows) | traced, matches (its examples are mostly outside the 45; item 3.9) |
| M107 | `7_discussion.tex:7` | 0.28 | COCO person IoU | frozen | `sg_combined_v3/per_class_iou.csv` coco R1+R2 person 0.2819 | traced, matches |
| M108 | `7_discussion.tex:7` | 4/5 and 5/5 | adopted components' folds | forced OOF | as M28–M29 | traced, matches |

### Supplementary (`supplementary.tex`)

| # | Location | Printed | Measures | Label | Source | Status |
|---|---|---|---|---|---|---|
| S1 | `:32` | 30 steps; 7.5; float32 / float16; 31 calls (0–30); 77 positions; 64×64 | generation | setting | Protocol Sec. 2 table | traced, matches |
| S2 | `:35` | 11 of 16; indices 0,1,4,5,6,7,8,9,13,14,15; excluded 2,3,10,11,12; three SA modules; eight heads; calls 10–20 | layers | setting | v3 shard `settings.ca_modules` / `sa_modules` | traced, matches |
| S3 | `:38` | a quarter; 60%; "**six** 64-pixel layers"; 16% | weight shares | descriptive | `sg24_a/results.json` `layer_weights.voc.A`: 0.243 / 0.597 / 0.160; five 64-px layers (indices 0,1,13,14,15) | traced, differs ("six" → "five") |
| S4 | `:43` (Fig. S1 cap.) | 84% | | descriptive | as M20 | traced, matches |
| S5 | `:48` | 0.99999; 0.0008 | seed agreement | diagnostic | `sg_a3g/report.md:31` | traced, matches |
| S6 | `:57–60` (Tab. S1, 12 cells) | .8228/–/.6623; .8228/3/5/.6617; .8292/5/5/.6967; .8304/5/5/.6961 | orientation | forced OOF / frozen | `sg_a3g/report.md` V0+R2, G0+R2, V3+R2, G3+R2 | traced, matches |
| S7 | `:63` (cap.) | −0.0000 [−0.0001, +0.0001]; +0.0075 (5/5) [+0.0020, +0.0128] | contrasts | Δ forced OOF; CI on mean-image IoU | `sg_a3g/report.md:20,24` | traced, matches (CI statistic as M59) |
| S8 | `:76` | (10, 80, 13, 3, 3, 5) | DenseCRF | setting | as M23 | traced, matches |
| S9 | `:79` | 11 CA layers at every call; 3 SA layers at 11 calls; 4096×4096 | cost | fact | v3 spec | traced, matches |
| S10 | `:87` | 0.756 → 0.789; all five folds; +0.042 [+0.037, +0.047] | one SA layer | OOF (thr .45); mean-image CI | `sa_confirmation/report.md:7`; Sec. 33 | traced, matches |
| S11 | `:87` | six CA seeds × twelve SA settings | joint audit | — | Sec. 38 (78 cells = 6 × (12 + 1)); `joint_ca_sa_cartesian` | traced, matches |
| S12 | `:90` | 0.788 → 0.800; chair +0.078; person +0.030 | second pass | forced OOF | Sec. 43: 0.7875 → 0.7998; chair +0.0780, person +0.0304 | traced, matches |
| S13 | `:90` | 86.5% (70.8%); 39.3% (14.0%) | first-arrival TP rate | diagnostic | `sg25/report.md` first-arrival, depths 1 and 3 | traced, differs (from the sg25 run: weighted seed, calls 10–20, thr 0.425. The paragraph describes the uniform-seed, all-calls reference. Sec. 114 asked for this to be stated) |
| S14 | `:95` (Fig. S2 cap.) | 90th / 10th percentile | selection | fixed rule | `paper_figures/passes_selection.json` (q0.90, q0.10) | traced, matches |
| S15 | `:100` | all five folds; four of five | DenseCRF; call window | OOF | Sec. 44 (R2 5/5); `sg22/report.md:4` (MID-P2+R2 4 votes) | traced, matches |
| S16 | `:109–112` (Tab. S2, 8 cells) | .7873/.6172; .7997/.6246; .8132/.6372; .8168/.6438 | fine-only reference | forced OOF / frozen | `sg22/report.md` ALL-P1, ALL-P2, ALL-P2+R2, MID-P2+R2 | traced, matches |
| S17 | `:131` | 0.17 | 64-px CA alone | in-sample | as M41 | traced, matches |
| S18 | `:150–157` (Tab. S3, 15 cells + cap.) | uniform .7135/.8169/.6437 … combined .6288/.8004/.0000; 4/5 | layer weightings | forced OOF / frozen | `sg24_a/report.md` A0–A4 | traced, matches |
| S19 | `:171–176` (Tab. S4, 24 cells) | 64 only .8097/.8228/.6599/.6621 … 64→16 .7505/.7825/.6403/.6705 | SA scales | forced OOF / frozen | `sg25/report.md` | traced, matches |
| S20 | `:184` | 58% / 65%; 7.6% / 8.3%; 0.84 | cap binding | diagnostic | Sec. 127 (factor 0.84–0.85) | traced, matches |
| S21 | `:193–202` (Tab. S5, 30 cells) | C5 .7926/.8149/.7059 … W3 .8116/.8261/.6828 | hierarchy variants | forced OOF / frozen | v1 C5, C7, C8, C9; v2 V6; v3 R1, W1, W2, W3 | traced, matches |
| S22 | `:205` (cap.) | "two runs"; "agree ... to 0.0003" | | | the rows come from **three** runs (v1, v2, v3); references .8228 / .8231 / .8230 | traced, differs (three runs; "ours" here is .8293 vs Table 3's .8304) |
| S23 | `:205` (cap.) | +0.0001; −0.0012 (2/5); −0.0032 (1/5); 7.7% vs 7.6%; 0.996; −0.016 | | forced OOF; diagnostic | `sg_combined_v3/report.md:29,31,33,38`; Sec. 132 | traced, matches |
| S24 | `:210` | 375 of 490; 278; 427 of 707; 18 of 20 | CRF effect | in-sample / frozen | recomputed from v3 shards | traced, matches |
| S25 | `:222` | VOC gains train, tv, person, bicycle, bottle; losses sheep, dining table | per class | in-sample | v3 `per_class_iou.csv` | traced, matches |
| S26 | `:222` | "16 of 20" COCO classes improved | | frozen | `per_class_iou.csv`: 17 improve strictly (sheep +0.003 points) | traced, differs (17) |
| S27 | `:222` | +13.0, +8.2, +7.4, +6.6; ≤ 1.6 loss; 28.2 | | frozen | same (dining table −1.55) | traced, matches |
| S28 | `:232–253` (Tab. S6, 126 cells) | class IoU ×100 and means 82.4 / 81.7 / 83.0 / 66.2 / 68.2 / 69.6 | per class | in-sample / frozen | v3 `per_class_iou.csv` R0+R2, R1, R1+R2 | traced, matches (126/126) |
| S29 | `:256` (cap.) | 0.425; 0.55; 0.823 and 0.829 | | | v3 | traced, matches |
| S30 | `:265` | six rules; 17-point grid; Otsu 4 folds; −0.021; 0.896 / 0.895 | per-image thresholds | forced OOF; diagnostic | `sg_threshold/report.md` (R0.T1+R2 COCO 0.6411 vs 0.6623); Sec. 134 | traced, matches |
| S31 | `:274–285` (Tab. S7, 55 cells) | R0.T0 .8230 … R1.T6 .8297 / +.0004 (2/5) / −.090 / .6586 | | forced OOF / frozen | `sg_threshold/report.md` +R2 rows | traced, matches |
| S32 | `:288` (cap.) | 20–87% | rules hitting the grid edge | diagnostic | same, "Edge %" (64 / 87 / 53 / 20) | traced, matches |
| S33 | `:293` | 294,665; 12,173; 79–96% | sheep TP/FP change | in-sample | Sec. 128 (computed on **v2**, V3+R2 vs V0+R2) | traced, matches (from v2, while the rest of Sec. S5 uses v3) |
| S34 | `:293` | "every threshold above 0.45"; "twice"; > 75% | sheep curves; size bins | diagnostic | Sec. 132 (FP/GT 0.205 vs 0.104 at 0.30; 0.125 vs 0.032 at 0.425; ≥ 0.75 bin) | traced, matches ("twice" holds at 0.30; about ×4 at 0.425) |
| S35 | `:298` (Fig. S6 cap.) | 25 images | | fact | Sec. 132 | traced, matches |
| S36 | `:312` | largest difference 0.0054 | regenerated vs stored image IoU | check | the figure job cache is outside the repo (`paper/README.md`); no committed record | not traced |
| S37 | `:325` | bottle mask misses 40% | area | diagnostic | none found | not traced |
| S38 | `:325` | 0.76, 0.81, 0.92 | median image IoU | in-sample | `story_selection.json` `failures[1..3]` 0.7569, 0.8064, 0.9219 | traced, matches |
| S39 | `:330` (Fig. S9 cap.) | four weakest classes per dataset | | | `story_selection.json` `failures`; Tab. S6 | traced, matches |
| S40 | `:337` | 45; about one point | | in-sample | `decision_map_families.tex` | traced, matches |
| S41 | `:337` | correlation 0.82–0.88 | per-class gains vs second pass, for the reliability operators | descriptive | `analysis/cross_package_classes/summary.json` (recomputed: SG2 0.88, SG12 0.82, SG11 0.44, SG8 0.34) | traced, differs (holds for two of the four) |
| S42 | `:347–363` (Tab. S8, 15 rows) | family counts and Δ ranges | | in-sample | `decision_map_families.tex` | traced, matches (counts sum to 45; ranges match) |
| S43 | `:371` | 7.998 of 8; three of five | | | Sec. 115; `sg33/report.md` M2+R2 3/5 | traced, matches |
| S44 | `:380–390` (Tab. S9, 11 rows) | +.0061 (4/5) / +.0183 … −.0123 (0/5) / +.0353 | adaptations | forced OOF / frozen | `sg24_a`; `sg28` B2+R2, P1+R2; `sg29` N1+R2, N2+R2; `sg30` S2+R2; `sg31` S3/S4+R2, S6+R2; `sg32` S7+R2; `sg33` M2+R2, H3+R2 | traced, matches |
| S45 | `:402` | five folds; about five per class per fold; grid 0.30–0.70 | protocol | setting | Sec. 5a; 17 shard thresholds | traced, matches |
| S46 | `:405` | 0.010; 0.005–0.010; 12 of 20; 0.05; 4 of 5; positive pooled OOF | rule | protocol | Sec. 5, 5a | traced, matches |
| S47 | `:405` | "Within 0.002, the simpler setting wins" | tie rule | protocol | Sec. 5: within 0.0020, prefer the setting with *more* calls/modules/heads | traced, differs |
| S48 | `:408` | three of five; 0.05 | selector for weights; guard | OOF | `sg24_a/results.json` `selection.FINAL.selection_counts` {A3: 3, A0: 2} | traced, matches |
| S49 | `:417–423` (Tab. S10, rows 1–7) | 0.665→0.691 [+0.015, +0.029]; 5/5 0.756→0.789; 0.788→0.800, 5/5 (0.813); 5/5, all 20 classes; 4/5; 4/5, +0.006, COCO +0.018, 3/5 | component status | OOF / forced OOF | CA-C; SA-C; Sec. 43, 51; Sec. 44; sg22; sg24_a | traced, matches |
| S50 | `:424` (Tab. S10, row 8) | 5/5; +0.006; COCO +0.034; "sheep lost 0.057–0.104 in the training folds" | hierarchy status | forced OOF | v3 (+0.0063, +0.0341); Sec. 127: 0.057–0.104 is the per-fold *worst class*; sheep is named only on full VOC | traced, differs (wording) |
| S51 | `:434` | "Table 5 ... quotes the class-word results reported by each method" | | | Table 5 transcribes OVAM's Table 1 | traced, differs (they are OVAM's reported numbers; the Attn2Mask gap, .687 published vs .603 rescored, is not explained) |
| S52 | `:445–452` (Tab. S11, 35 cells) | OVAM 490/.704/.825/707/.582 … BRACE .829/.696 | released masks | published-mask rescoring; ours forced OOF / frozen | `released_baselines/summary.json`; `daam_fold_threshold/summary.json` (.6735, .797, .5267, .6417); v3 | traced, matches |
| S53 | `:467` | "PyTorch 2.4 or 2.6"; "V100 or A100" | environments | fact | shard `environments`: 2.4.1 (CA-C), 2.6.0 (most runs), **2.11.0** (`fusion_calibration`, the DenseCRF screen of Sec. 44); GPUs V100, V100S, A100 | traced, differs (2.11 missing) |
| S54 | `:467` | 0.8132–0.8135 | fine-only across runs | forced OOF | sg22 ALL-P2+R2 0.8132; P3-C 0.8135 (Sec. 51) | traced, matches |

### Counts

| | Rows | Traced, matches | Traced, differs | Not traced |
|---|---:|---:|---:|---:|
| Main paper | 108 | 94 | 13 | 1 |
| Supplementary | 54 | 43 | 9 | 2 |
| **Total** | **162** | **137** | **22** | **3** |

Rows that group table cells cover about 330 further individual cells in the main paper and supplementary tables. Every cell in those rows matched its source, apart from the issues named in the row. Of the 22 "differs" rows, 11 are a wrong value, count, rule or attribution: M19, M31, M57, M60, M98, S3, S22, S26, S41, S47, S53. The other 11 are labels, wording, CI statistics or run notes: M7, M11, M40, M59, M74, M75, M93, M94, S13, S50, S51. The 3 not traced: M64 (conservation IoU 0.58 in Fig. 4), S36 (0.0054), S37 (40% of the bottle area).

---

## Part 3 — Other inconsistencies

1. **The committed main PDF is stale.** `paper/attention_segmentation.pdf` was last committed in `4c4ce1a` (31 Oct). Sec. 5.2 and Fig. 3 changed in `2a954a2` (4 Nov). Sec. 5.2 in the PDF reads "0.73 at τ=0.70 … against 0.61", "0.723 OOF", "(0.687 vs. 0.660 frozen)", "(0.802 vs. 0.810 OOF)". The `.tex` reads "0.72", "τ=0.45", "0.693 vs. 0.662 after DenseCRF", "0.806 vs. 0.810". Fig. 3 in the PDF is a precision–recall plot, but the `.tex` caption describes mIoU against threshold. **Fix:** rebuild and commit both PDFs before anything goes to JC. Say which PDF the reply refers to.
2. **The "frame-filling objects" failure claim contradicts the evidence.** The abstract (`0_abstract.tex:2`: "its one systematic failure, objects that fill the frame") and Discussion (`7_discussion.tex:7`: "frame-filling objects lose under one global threshold") disagree with the supplementary (`supplementary.tex:293`: "objects that fill more than 75% of the frame gain from the cap at every threshold; sheep is the exception") and with protocol Section 132 ("not a general large-object failure"; the ≥ 0.75 bin gains at every threshold). **Fix:** abstract "…which also explains its one systematic failure: close-ups of sheep, whose surroundings self-attention groups with the animal." Discussion: "and frame-filling sheep close-ups lose under one global threshold".
3. **"One run" claims that do not hold.** `sec/4_protocol.tex:7` says "Every comparison in Secs. 5.1–5.5 is between cells of one run". `sec/5_analysis.tex:35` says "Table 3 answers this in one run", and the Table 3 caption says "All rows share one generation". But Table 3 joins v1 (three conservation rows) and v2. Fig. 3's left and middle panels join sg25 (coarse-only curve) and v2 (the other three curves; `scripts/paper/figure_coarse_to_fine.py:63–68`). Table 2 top (CA-R run) and its caption (CA-C run) are different runs. Paired Δs are within one run in every case. **Fix:** Sec. 4: "Every paired difference in Secs. 5.1–5.5 is between cells of one run …; tables that join runs say so." Add run notes to the Table 3 caption (item 1.1) and the Fig. 3 caption ("the coarse-only curve is from a separate generation with the same seed and SA calls").
4. **Fig. 2 caption, "each shown by its median-IoU image".** The COCO-Cap person column is the 75th-percentile image (IoU 0.63). The median is 0.25 and appears in Fig. S9 (`story_selection.json` `stages[6].image_quantile` 0.75). **Fix:** "…each shown by its median-IoU image (for the worst COCO-Cap class, its 75th-percentile image, so that the object is visible; the median is in Fig.~S9)". Or regenerate with the median.
5. **"Out-of-fold" vs "forced out-of-fold".** The abstract, intro and Table 1 caption ("VOC: OOF mIoU") call forced-OOF numbers OOF. Rows .817, .823 and .829 are forced OOF of variants the registered selector did not keep. Protocol Section 5a.2 defines the reported OOF as that of the selection procedure. **Fix:** "forced out-of-fold" in the abstract and intro, a one-line definition in Sec. 4 (already there: "forced OOF scores a named variant in every fold"), and Table 1 caption "VOC: forced OOF mIoU (rows 1–3: OOF of the selection procedure)". This also serves JC #1.
6. **CIs attached to a different statistic.** Paired CIs are bootstrap intervals of the mean *image* IoU difference. The adjacent Δ is a class-mIoU difference. Protocol Section 120: "Mean-image bootstrap intervals must not be described as intervals on 20-class macro mIoU." Cases: `5_analysis.tex:35` and the Table 3 caption (+0.012 mIoU with a CI around +0.009 mean-image); `5_analysis.tex:16` (+0.022 mean-image beside Table 1's +.026 mIoU for the same step); Tab. S1 caption; Tab. S10 row 1. **Fix:** name the statistic each time ("paired mean image IoU $+0.009$, 95\% CI …"), and add one sentence in Sec. 4: "intervals are for the mean per-image IoU difference".
7. **Two operators share the name "16 to 64".** In the main text (`5_analysis.tex:24`, Fig. 3 "coarse → fine") it means 16 → 32 → 64 (v2 V1). In Tab. S4 ("16 → 64", .8020) it is the direct 16 → 64 of sg25. `supplementary.tex:162` also claims Tab. S4 is "the run behind Fig. 3 of the main paper", which is true only for the coarse-only curve. **Fix:** rename the Tab. S4 row "16 → 64 (skipping 32)" and change `:162` to "the run behind the coarse-only curve of Fig. 3".
8. **Table 1 caption "Runs differ by at most 0.0003 on a shared reference".** True for the fine-only reference. The final method varies by 0.0012 across runs (item 1.1).
9. **Discussion examples outside the "45".** `7_discussion.tex:4` says "45 comparisons of alternative operators, such as random walks, graph sharpening and re-seeding between scales". The 45 are the families of Tab. S8 / `decision_map_families.tex`. The NERVE walk, SPARK sharpening and SeeDiff re-seeding are in Tab. S9, which is not part of the 45. Tab. S8's caption counts "families", while the text says "comparisons". **Fix:** "45 registered families of alternative operators, such as absorbing walks, consensus graphs and region labelling (Sec.~S7, Tab.~S8), and adaptations of random-walk, sharpening and re-seeding methods (Tab.~S9)".
10. **"31 steps"** (`3_method.tex:38`) vs "30 denoising steps" (`3_method.tex:7`, `4_protocol.tex:4`). Replace with "the 31 denoising calls".
11. **Sec. 6 margin sentence** (`6_comparison.tex:3`). **Fix:** "Of the 0.114 COCO-Cap margin over OVAM, the capped hierarchy contributes 0.034 (Table~\ref{tab:progression}); the rest comes from configuration and the layer weighting." This also feeds JC #1.
12. **Supplementary wording and value errors** (each in Part 2): "six 64-pixel layers" → five (`:38`); "16 of 20" → 17 (`:222`); "0.82–0.88" holds for two of the four operators (`:337`); "simpler setting wins" → "more inclusive setting wins" (`:405`); "two runs" → three (`:205`); "PyTorch 2.4 or 2.6" → "2.4, 2.6 or 2.11" (`:467`); "reported by each method" → "reported in OVAM's Table 1" (`:434`). Add one sentence on why Attn2Mask's published .687 / .550 differs from our rescoring of its released masks (.603 / .584): different images (485 / 668) and a procedure without printed notebook output (Section 41).
13. **Supplementary numbering collides with the main paper.** Both print "Table 1", "Fig. 2", "Sec. 5" and "Eq. (1)". Fix in item 1.5.
14. **Internal names.** None in the main `.tex`, the main PDF or the main-paper figures. The supplementary text uses only T0–T6, which it defines. Two **supplementary figures** carry internal names: Fig. S1 axis label "mean A3 weight" (`figures/a3_weights.pdf`; `scripts/paper/figure_coarse_to_fine.py:132` `set_xlabel("mean A3 weight")`), and Fig. S2 panel title "All images (SG25)" (`figures/propagation_depth.pdf`; `scripts/paper/figure_passes.py:117`). **Fix:** "mean layer weight" and "All images". Rule 9.
15. **Broken `\ref`.** None in the current main `.tex` (34 labels, every `\ref` / `\eqref` resolves) or the supplementary (31 labels, all resolve). No committed PDF contains "??" (item 1.3).
16. **Orphan files that diverge.** `paper/tables/supp/a3g.tex` and `paper/tables/supp/hierarchy_all.tex` are not `\input` by either build. They hold older copies of Tab. S1 and Tab. S5 with internal names (R1, V3, W1, C5–C9) and an extra contrast ("GoCA − ours, coarse-to-fine: +0.0012 (4/5)") that the current Tab. S1 lacks. `paper/README.md:43` and `:93` still name `sec/7_fusion.tex` and `sec/8_transfer.tex`, which no longer exist. **Fix:** delete or archive with the owner's yes (rule 8), and update the README.
17. **Table 2 prints two values for the same configurations.** The top table has all scales .666 and 8+16+64 .696 (in-sample at 0.45, CA-R run). The caption has 0.665 → 0.691 (OOF, CA-C run). **Fix:** add "(top: in-sample, CA package A; caption: out-of-fold confirmation run)" or quote the top-table values in the caption.
18. **Sheep analysis mixes runs and metrics.** The supplementary's 294,665 / 12,173 px and 79–96% come from v2 (Section 128). The curves, thresholds and the 25-image analysis come from v3 (Section 132). The main text gives sheep's preferred threshold as 0.35 (v3, mean-image IoU). Section 128 gives 0.30 (v2, class IoU). **Fix:** recompute the TP/FP numbers on v3, or label them "(separate generation)".
19. **The supplementary's first-arrival rates** (`:90`) come from the sg25 run (weighted seed, calls 10–20, threshold 0.425), not from the fine-only reference the paragraph describes. Protocol Section 114 already asked for this to be said.
20. **The workplan contract and the paper disagree on bootstrap size.** The workplan Section 3 says "2000-sample image bootstrap". The paper and protocol Section 5a say 20,000. For F's Entry 1: use one, and state it.
21. **`5_analysis.tex:16` "(see the supplementary seed gallery)".** The current supplementary has no gallery; the figure is Fig. S4, "CA by resolution".
22. **JC #1 overlap.** The abstract sentence "Selecting every choice on five class-stratified folds of VOC-Sim, the pipeline reaches 0.829 out-of-fold mIoU" reads as if the registered selection produced 0.829. Sec. 4 "Selection status" and Sec. 7 say otherwise. Rewording belongs to N's #1 text, but the number's label (item 5) should change in the same edit.

### Top five fixes, in order

1. **One canonical estimate (item 1.1).** Use v3: 0.829 forced OOF / 0.696 frozen. Remove the absolute 0.830/0.697 from Sec. 5.3. Label Table 3's cap row as a separate generation and Table 4's .830 as in-sample.
2. **Raw vs DenseCRF (item 1.2).** Fix the Sec. 5.2 sentence (0.818 vs 0.823 after CRF). Give "five points" its stage, or use −3 after CRF. Add a DenseCRF column and our raw row to Table 5 and Tab. S11.
3. **Labels and statistics (Part 3, items 5–6).** "Forced out-of-fold" for 0.829, 0.823 and 0.817. Name the statistic behind each CI (the +0.012 with CI [+0.006, +0.013] case).
4. **Claims contradicted by the record (Part 3, items 2–4).** The "frame-filling objects" failure. "One run" for Table 3, Sec. 4 and Fig. 3. Fig. 2's "median-IoU image".
5. **Citations, pointers and build (items 1.3–1.5, Part 3 items 1 and 13).** Add and verify the three bib entries. Use S-numbering plus `xr` for every cross-document reference. Rebuild and commit both PDFs (the main PDF is stale), and check for "??" before sending.
