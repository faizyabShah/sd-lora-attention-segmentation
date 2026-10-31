# Paper: CVPR submission source

Layout agreed in protocol Sections 28–29, created in Section 39.

## Current narrative draft (31 October 2026)

The main paper is told as a method story: **seed, envelope, detail, boundary**.
- A3-weighted class-word CA seed (8/16/64 px, no 32-px modules).
- Coarse-to-fine SA (16 -> 32 -> 64 px, calls 10-20) with a one-sided mass cap at 16 -> 32.
- Fixed OVAM DenseCRF at threshold 0.55.

Headline: VOC-Sim 0.829 forced OOF / COCO-Cap 0.696 frozen (SG-combined-v3 R1+R2).

The A3 weights and the capped hierarchy are **owner decisions**: protocol Sections 103 and 135. The paper labels them in Sec. 4 and supplementary Sec. 5, and reports the registered outcomes. The earlier registered fine-only pipeline (0.813 / 0.637; MID-P2+R2 0.816 / 0.644) is now supplementary Sec. 2. The negative record (decision map, 45 families) is in the supplementary.

The published comparison transcribes the **class-word rows of OVAM Table 1**; optimized-token rows are excluded.

Figures (fixed-rule image choice; regenerated maps are checked against the committed v3 counts):

| Figure | Generator |
|---|---|
| Fig. 1 `method_overview`, Fig. 2 `pipeline_c2f`, Fig. 4 `cap_mechanism`, Fig. 5 `crf_effect`, supp. `supp_pipeline_c2f`, `supp_ca_seeds`, `supp_sheep_case` | `scripts/paper/select_story_images.py` (CPU) -> `scripts/paper/figure_story_maps.py` (GPU, `jobs/figure_story_maps.slurm`, cache outside the repo) -> `scripts/paper/figure_story_qualitative.py` |
| Fig. 3 `scale_profiles`, Fig. 6 `threshold_curves`, supp. `a3_weights`, `supp_sheep_curves`, `supp_crf_classes` | `scripts/paper/figure_coarse_to_fine.py` (CPU, from committed shards and results) |
| supp. `decision_map`, `propagation_depth`, `temporal_tradeoff`, gallery, class deltas | the earlier generators listed in the artifact map below |
| supp. Sec. G experiment record | `scripts/paper/supp_experiment_record.py` (now including SG-A3G, SG-combined-v3, SG-threshold) |

Plotting uses `MPLCONFIGDIR=/tmp ~/miniconda3/envs/lab1/bin/python`; the record generator and the GPU job use `trdp`.

## Build

```bash
cd paper
pdflatex attention_segmentation && bibtex attention_segmentation \
  && pdflatex attention_segmentation && pdflatex attention_segmentation
# or: latexmk -pdf attention_segmentation.tex   (Overleaf does this)
# supplementary: the same with supplementary.tex
```

Regenerate the tables first: `python scripts/paper/export_paper_artifacts.py`
from the repository root.

The two current main-text tables are authored directly in
`sec/7_fusion.tex` (same-run SG22 nested steps) and `sec/8_transfer.tex`
(published OVAM Table 1 alongside clearly marked regenerated-image scores).
The generated tables below remain archived for the older draft and supplement.

The cluster has TeX Live 2018. The preamble therefore avoids `siunitx` and
`cuted` (use `figure*` for wide figures). The kit also needs `cleveref`,
`silence` and `multirow`, which the cluster lacks; install them from CTAN into
`~/texmf/tex/latex/` (as for the first build) or build on Overleaf.

## Files

| Path | Content | Edited by hand? |
|---|---|---|
| `attention_segmentation.tex` | manuscript: title, review mode, section order | yes |
| `supplementary.tex` | supplementary material (separate PDF) | yes |
| `preamble.tex` | packages and macros, before `hyperref` | yes |
| `sec/*.tex`, `sec/supp/*.tex` | manuscript and supplementary sections | yes, after shared review |
| `tables/generated/*.tex` | tables from `scripts/paper/export_paper_artifacts.py` | **no** |
| `tables/*.tex` | hand-written tables | yes |
| `figures/` | only figures cited by the manuscript | — |
| `references.bib` | bibliography; every entry checked against its DOI | yes |
| `cvpr.sty`, `ieeenat_fullname.bst` | CVPR author kit, unchanged (`kit/KIT_SOURCE.txt`) | no |

## Shared review

The original section assignments in protocol Section 29 produced the first
draft. Codex integrated the present narrative draft across the main-text
sections at the owner's request. Claude's precise audit requests are in
`docs/PAPER_REVISION_HANDOFF.md`; the supplementary source remains in its
earlier state. This manuscript is a working draft awaiting shared review, not a frozen
submission.

## Artifact map

Every number, table and figure in the paper must appear here with its source.
The audit checks each row against the committed file and the header of the
generated table.

| Paper artifact | Generated file | Source | Generator |
|---|---|---|---|
| Table: controlled selection path | `tables/generated/controlled_path.tex` | `outputs/segmentation/controlled_attention_cv/{ca_confirmation,sa_confirmation,joint_ca_sa_cartesian}/fold_results.json` | `scripts/paper/export_paper_artifacts.py` |
| Table: released baselines and the frozen method | `tables/generated/released_baselines.tex` | `outputs/segmentation/comparisons/released_baselines/summary.json`; `outputs/segmentation/controlled_attention_cv/sa_confirmation/fold_results.json` | `scripts/paper/export_paper_artifacts.py` (scores: `scripts/narmyn/attention/score_released_baselines.py`) |
| Sec. 4.2 selection constants (5 folds of 98, seed, 20,000 resamples, bands) | — | `src/segmentation/controlled_cv.py`, `configs/voc_sim_folds.csv`, protocol Sections 5, 5a | hand-written |
| Sec. 5 package-B numbers (CA-L, CA-H, CA-A) | — | `outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution/{fold_results,head_report}.json`; protocol Section 21 | hand-written |
| Sec. 5 CA-C numbers | — | `outputs/segmentation/controlled_attention_cv/ca_confirmation/fold_results.json`; protocol Section 27 | hand-written |
| Sec. 6 package-B numbers (SA-L, SA-H, `Aᵀx`, seed alone) | — | `outputs/segmentation/controlled_attention_cv/sa_layer_head/fold_results.json`; protocol Section 30 | hand-written |
| Sec. 6 SA-C and joint-audit numbers | — | `outputs/segmentation/controlled_attention_cv/{sa_confirmation,joint_ca_sa_cartesian}/`; protocol Sections 33, 36 | hand-written |
| Sec. 7 propagation numbers | — | `outputs/segmentation/controlled_attention_cv/propagation_package/fold_results.json`, `diagnostics.json`, and `report.md`; protocol Section 43 | hand-written |
| Sec. 7 P3 confirmation numbers | — | `outputs/segmentation/controlled_attention_cv/phase3_confirmation/fold_results.json`, `diagnostics.json`, and `report.md`; protocol Section 51 | hand-written |
| Sec. 7 token numbers | — | `outputs/segmentation/controlled_attention_cv/tokens/fold_results.json`, `diagnostics.json`, and `report.md`; protocol Section 50 | hand-written |
| Main-text nested component table and stage figure | `sec/7_fusion.tex`, `figures/validated_progression.pdf` | `outputs/segmentation/controlled_attention_cv/sg22/report.md`, `results.json` | hand-written table; `scripts/paper/build_story_figures.py` |
| CA-resolution visual | `figures/ca_airplane.pdf` | `outputs/segmentation/controlled_attention_cv/ca_layer_head_attribution/figures/ovam-voc-sim-0_ca_modules.png` | `scripts/paper/build_story_figures.py` |
| Opposing raw temporal examples | `figures/temporal_tradeoff.pdf` | `outputs/segmentation/controlled_attention_cv/sg18_temporal_windowed/visuals/{06_largest_gain,07_largest_loss}.png` | `scripts/paper/build_story_figures.py` |
| Per-class temporal effect | `figures/temporal_class_effects.pdf` | `outputs/segmentation/controlled_attention_cv/sg22/per_class_iou.csv` | `scripts/paper/build_story_figures.py` |
| Published class-word comparison | `tables/tab_published.tex` | OVAM CVPR 2024 Table 1; `sg_combined_v3/results.json` | hand-written, source-checked |
| Tab. 1 progression | `tables/tab_progression.tex` | `ca_confirmation`, `sa_confirmation`, `sg22`, `sg24_a`, `sg_combined_v3` reports | hand-written |
| Tab. 2 seed | `tables/tab_seed.tex` | `paper/evidence/tables/a2_ca_r.tex`; `ca_confirmation`; `sg24_a/report.md` | hand-written |
| Tab. 3 hierarchy | `tables/tab_hierarchy.tex` | `sg_combined_v1/report.md`, `sg_combined_v2/{report.md,results.json}` | hand-written |
| Tab. 4 DenseCRF | `tables/tab_refine.tex` | `sg_combined_v3/results.json` (cells_vs_R0) and v3 count shards | hand-written |
| Supp. tables | `tables/supp/*.tex`, `sec/supp/S*.tex` | `sg_a3g`, `sg25`, `sg24_a`, `sg_combined_v1/v2/v3`, `sg_threshold` reports | hand-written |

## Before submission

- Disable `\TODO`, `\todo`, `\owner` and `\source` in `preamble.tex`.
- Paper ID and `\confYear`; the kit for the submission year.
- No names, affiliations, repository URLs or non-anonymous paths in the PDF or
  the supplementary material; code goes in an anonymized archive.
- 8 pages without references.
