# Paper: CVPR submission source

Layout agreed in protocol Sections 28–29, created in Section 39.

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
| `sec/*.tex`, `sec/supp/*.tex` | one file per section; owner named in the file | yes, by its owner |
| `tables/generated/*.tex` | tables from `scripts/paper/export_paper_artifacts.py` | **no** |
| `tables/*.tex` | hand-written tables | yes |
| `figures/` | only figures cited by the manuscript | — |
| `references.bib` | bibliography; every entry checked against its DOI | yes |
| `cvpr.sty`, `ieeenat_fullname.bst` | CVPR author kit, unchanged (`kit/KIT_SOURCE.txt`) | no |

## Section owners (protocol Section 29)

| Section | Owner |
|---|---|
| Method; Experimental protocol (data and metric); package-A result text; final integration | Codex |
| Cross-validated selection wording; package-B result text; CA-C and SA-C text; artifact-to-evidence audit; supplementary C–E | Claude |
| Propagation (Section 7) | Codex |
| Fusion and calibration (Section 7) | Claude |
| Abstract, introduction, related work, transfer, discussion | to agree |

## Artifact map

Every number, table and figure in the paper must appear here with its source.
The audit checks each row against the committed file and the header of the
generated table.

| Paper artifact | Generated file | Source | Generator |
|---|---|---|---|
| Table: controlled selection path | `tables/generated/controlled_path.tex` | `outputs/segmentation/controlled_attention_cv/{ca_confirmation,sa_confirmation,joint_ca_sa_cartesian}/fold_results.json` | `scripts/paper/export_paper_artifacts.py` |

## Before submission

- Disable `\TODO`, `\todo`, `\owner` and `\source` in `preamble.tex`.
- Paper ID and `\confYear`; the kit for the submission year.
- No names, affiliations, repository URLs or non-anonymous paths in the PDF or
  the supplementary material; code goes in an anonymized archive.
- 8 pages without references.
