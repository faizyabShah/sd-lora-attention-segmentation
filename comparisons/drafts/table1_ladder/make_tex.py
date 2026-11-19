#!/usr/bin/env python3
"""Write comparisons/drafts/table1_jc7.tex (JC #7 draft of Table 1) from table1_ladder/result.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
J = json.loads((HERE / "result.json").read_text())
R = {r["key"]: r for r in J["rows"]}
NUM = {r["key"]: r["label"].split()[0] for r in J["rows"]}

ROWS = [  # key, printed step (LaTeX)
    ("seed16|raw", "CA seed, all 16 CA modules"),
    ("seed11|raw", "$-$ five 32-px CA modules"),
    ("p1_16sa|raw", "\\quad(1 SA pass, 16 modules)"),
    ("p1_m15|raw", "+ 1 SA pass, 64-px module"),
    ("p2_m15|raw", "+ 2nd SA pass"),
    ("p2_m15|crf", "+ DenseCRF"),
    ("v0_a0|crf", "+ SA calls 10--20"),
    ("v0_a3|crf", "+ SA-guided CA weights"),
    ("v3_a3|crf", "+ hierarchy, cap (BRACE)"),
]


def d(c, bold_if_excludes_zero=True):
    x = f"{c['d']:+.3f}".replace("0.", ".", 1)
    if bold_if_excludes_zero and (c["lo"] > 0 or c["hi"] < 0):
        return "$\\mathbf{" + x + "}$"
    return "$" + x + "$"


def v(x):
    return f"{x:.3f}"[1:] if x < 1 else f"{x:.3f}"


lines = [
    "% JC #7 draft of Table 1 (side N). Not in paper/; for the owners to adopt or edit.",
    "% Source of every number: comparisons/drafts/table1_ladder/result.json (result.md), one live generation per image,",
    "% frozen runtime, runs/20 corrected prompts, 490 VOC-Sim + 707 COCO-Cap; script t1_ladder.py; PREREG.md in that folder.",
    "% Paper's current values for comparison (paper/tables/tab_progression.tex): .665 .691 (.756) .789 .800 .813 .817 .823 .829 / COCO .625 .637 .644 .662 .696.",
    "% Cross-check with side F's committed runs/25 (a64aff6, V100, same runtime): fine-only + DenseCRF @0.55 0.796 / 0.698 and",
    "% BRACE @0.55 0.832 / 0.708; here (A100 capture) 0.796 / 0.698 and 0.833 / 0.708. VOC-sel. BRACE: F 0.831 / 0.708, here 0.832 / 0.708.",
    "% Reproduction of the paper's VOC-Sim screens at their fixed threshold 0.45 (result.md, fixed-threshold table):",
    "% rows 1, 2, (3a), 3, 4, 5, 6 = .667 .696 (.757) .790 .803 .816 .821 vs paper .665 .691 (.756) .789 .800 .813 .817.",
    "% Bold = 95% paired image-bootstrap interval excludes zero (2000 resamples); all intervals in result.md.",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{\\textbf{The pipeline, one decision at a time, on both benchmarks.} Every row comes from one generation per image "
    "and adds one decision to the row named in the \\emph{vs} column; $\\Delta$ is the paired difference to that row (bold: 95\\% "
    "image-bootstrap interval excludes zero). Threshold convention: chosen on VOC-Sim (5-fold out-of-fold), the full-VOC-Sim "
    "threshold applied unchanged to COCO-Cap, so consecutive rows may use different thresholds (row~8 uses 0.55, row~7 0.425). "
    "Folds: VOC-Sim held-out folds won (of 5). Row~7 is the fine-only reference kept by the registered rule. The last three columns use one shared threshold, 0.55, for every row; "
    "there row~8 gains $+0.010$ on COCO-Cap over row~7, against $+0.036$ with each row at its own VOC-Sim threshold. The indented row is the alternative to row~3 (one pass through all 16 SA modules); row~3 builds on row~2. "
    "VOC-Sim values are in-sample for the method design.}",
    "\\label{tab:progression}",
    "\\footnotesize\\setlength{\\tabcolsep}{2pt}",
    "\\begin{tabular}{@{}rlr rrr rr rrr@{}}",
    "\\toprule",
    " & & & \\multicolumn{3}{c}{VOC-Sim} & \\multicolumn{2}{c}{COCO-Cap} & \\multicolumn{3}{c}{shared threshold 0.55} \\\\",
    "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}",
    " & Step & vs & mIoU & $\\Delta$ & folds & mIoU & $\\Delta$ & VOC & COCO & $\\Delta$ COCO \\\\",
    "\\midrule",
]
for key, lab in ROWS:
    r = R[key]; n = NUM[key]; prev = NUM.get(r["prev"], "") if r["prev"] else "---"
    if r["prev"]:
        cells = [v(r["voc_sel"]), d(r["d_voc_sel"]), f"{r['folds_won']}/5", v(r["coco_sel"]), d(r["d_coco_sel"])]
    else:
        cells = [v(r["voc_sel"]), "---", "---", v(r["coco_sel"]), "---"]
    nn = "" if n == "3a" else n
    d55 = d(r["d_coco_0.55"]) if r["prev"] else "---"
    lines.append(f"{nn} & {lab} & {prev} & " + " & ".join(cells) + f" & {v(r['voc_0.55'])} & {v(r['coco_0.55'])} & {d55} \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
out = HERE.parent / "table1_jc7.tex"
out.write_text("\n".join(lines) + "\n")
print(out.read_text())
