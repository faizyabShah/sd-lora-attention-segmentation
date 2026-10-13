# Attention segmentation report package

This directory is self contained for sharing with a collaborator or professor.

- `attention_segmentation_report.tex`: editable LaTeX source.
- `attention_segmentation_report.pdf`: compiled 12 page report.
- `figures/`: every plot and qualitative panel used by the report. Quantitative plots are provided as both PDF and PNG; qualitative panels are PNG files.

Compile from this directory with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error attention_segmentation_report.tex
pdflatex -interaction=nonstopmode -halt-on-error attention_segmentation_report.tex
```

The report needs a standard LaTeX installation with `graphicx`, `booktabs`, `caption`, `subcaption`,
`float`, `microtype`, `enumitem`, and `hyperref`.
