#!/usr/bin/env python3
"""Supplementary "complete experiment record": every registered decision and every evaluated cell.

Nothing is re-run and nothing is selected here; every number comes from a committed result file.

  record_screens.tex     one row per registered fold decision (113 screens), from the decision records
                         (fold_results.json / results.json / summary.json) of each package
  record_packages.tex    one subsection per package: description (from record_glossary.json), then a
                         table of every cell computed from the committed shards (or results.json)
  record_ledger.tex      one row per experiment: the evidence ledger (B1-B17, A1-A12, AN1) extended
                         with every later package

Cell columns: the cell's best full-VOC threshold t* on the 17-point grid (three points for the early
fixed-threshold screens), VOC-Sim mIoU there (in-sample), its change against the package reference,
a threshold-only out-of-fold mIoU (t* chosen on four folds, scored on the fifth, pooled), and COCO-Cap
mIoU at the VOC t* (frozen transfer). Writes ``paper/tables/generated/record/*.tex``.

    python scripts/paper/supp_experiment_record.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/paper"))
from figure_decision_map import STANDARD, load  # noqa: E402

B = ROOT / "outputs/segmentation/controlled_attention_cv"
OUT = ROOT / "paper/tables/generated/record"
GLOSSARY = ROOT / "outputs/segmentation/paper_figures/record_glossary.json"
LEDGER = ROOT / "docs/narmyn/attention/evidence/ledger.csv"

# package directory, record id, stage, results section(s); the early CA/SA screens (A1-A5) have
# hundreds of cells and are tabulated by the evidence package instead of cell by cell
PACKAGES = [
    ("ca_layer_head_attribution", "A1", "ca", "21"), ("ca_time_resolution", "A2", "ca", "20"),
    ("ca_confirmation", "A3", "ca", "27"), ("sa_layer_head", "A4", "sa", "30"),
    ("sa_time_resolution", "A5", "sa", "31"), ("sa_confirmation", "A6", "sa", "33"),
    ("joint_ca_sa_cartesian", "A7", "sa", "36"), ("propagation_package", "A8", "prop", "43"),
    ("fusion_calibration", "A9", "prop", "44"), ("phase3_confirmation", "P3-C", "prop", "51"),
    ("tokens", "A11", "prop", "50"), ("token_confirmation", "T-C", "prop", "52"),
    ("sg5_contrastive_seed", "SG5", "graph", "70"), ("sg1_sg3_graph_refinement", "SG1/SG3", "graph", "71"),
    ("token_sg3_confirmation", "SG3-T", "graph", "78"), ("sg8_observation_consensus", "SG8", "graph", "78"),
    ("sg9_reciprocal_propagation", "SG9", "graph", "78"), ("sg10_seed_connected_bottleneck", "SG10", "graph", "79"),
    ("sg2_consensus_edges", "SG2", "graph", "85"), ("sg4_region_labels", "SG4", "graph", "85"),
    ("training_free_backlog", "BL", "graph", "83"), ("sg11_sg12_observation_reliability", "SG11/SG12", "graph", "89"),
    ("sg2_densecrf_confirmation", "SG2+R2", "graph", "89"), ("sg13_target_region_inference", "SG13", "graph", "90"),
    ("sg12_crf", "SG12+R2", "graph", "92"), ("sg14", "SG14", "graph", "92"), ("sg15", "SG15", "graph", "92"),
    ("sg17", "SG17", "time", "95"), ("sg18_temporal_windowed", "SG18", "time", "97"),
    ("sg18_t1_crf", "SG18-T1+R2", "time", "98"), ("sg21_structural", "SG21", "time", "101"),
    ("sg22", "SG22", "time", "102"), ("sg23", "SG23", "time", "102"), ("sg24_a", "SG24-A", "time", "103"),
    ("sg24_b", "SG24-B", "time", "103"), ("sg25", "SG25", "time", "104"), ("sg26", "SG26", "time", "105"),
    ("sg27", "SG27", "time", "109"), ("sg28", "SG28", "time", "112"),
    ("sg29", "SG29", "late", "115"), ("sg30", "SG30", "late", "118"), ("sg31", "SG31", "late", "119, 120"),
    ("sg32", "SG32", "late", "121"), ("sg33", "SG33", "late", "122"),
    ("sg_combined_v1", "SG-combined-v1", "late", "123"), ("sg_combined_c10", "SG-combined C10", "late", "124, 127"),
    ("sg_combined_v2", "SG-combined-v2", "late", "126, 127"),
]
STAGES = {"ca": "Cross-attention seed", "sa": "Self-attention module",
          "prop": "Propagation, refinement and tokens", "graph": "Graph, seed and observation operators",
          "time": "Temporal, structural, CA-weighting and multi-scale studies",
          "late": "Paper-inspired propagation, hierarchy and combined studies"}
CELL_LIMIT = 100  # packages with more cells are summarized by the evidence tables


def tex(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
            .replace("_", "\\_").replace("$", "\\$").replace("~", "\\textasciitilde{}").replace("^", "\\^{}")
            .replace("->", "$\\rightarrow$").replace(">=", "$\\geq$").replace("<=", "$\\leq$")
            .replace("<", "$<$").replace(">", "$>$").replace("|", "$|$"))


def brk(s: str) -> str:
    """Escaped text that may break after / _ - + = (for long cell and screen names)."""
    out = tex(s)
    for ch in ("/", "-", "+", "="):
        out = out.replace(ch, ch + "\\allowbreak{}")
    return out.replace("\\_", "\\_\\allowbreak{}")


def tt(s: str) -> str:
    return "\\texttt{" + brk(s) + "}"


# --------------------------------------------------------------------------- decision records
def records(pkg: str) -> dict[str, dict]:
    def find(x, path=""):
        if isinstance(x, dict):
            if "reference" in x and "frozen" in x:
                yield path, x
            else:
                for k, v in x.items():
                    yield from find(v, f"{path}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from find(v, f"{path}[{i}]")
    seen: dict[str, dict] = {}
    for name in ("fold_results.json", "results.json", "summary.json"):
        p = B / pkg / name
        if not p.exists():
            continue
        for path, r in find(json.loads(p.read_text())):
            if any(w in path.lower() for w in ("transfer", "historical")):
                continue
            key = path.replace("/screens", "").replace("/selection", "").strip("/") or "screen"
            seen.setdefault(key, r)
    return seen


def oof_pair(r: dict) -> tuple[float, float]:
    for a, b in (("out_of_fold_miou_reference", "out_of_fold_miou_selection"), ("oof_miou_reference", "oof_miou_selection")):
        if a in r:
            return r[a], r[b]
    return float("nan"), float("nan")


def outcome(r: dict) -> str:
    reason = r.get("freeze_reason", "")
    if r["frozen"] != r["reference"]:
        return "promoted"
    if "fewer than" in reason:
        return "kept (votes)"
    if "OOF guard" in reason:
        return "kept (OOF guard)"
    return "kept"


def picks(r: dict) -> str:
    c = r.get("selection_counts", {})
    order = sorted(c.items(), key=lambda kv: (-kv[1], kv[0] != r["reference"]))
    return "; ".join(("ref" if k == r["reference"] else tt(k)) + f" {v}" for k, v in order)


def write_screens(meta: dict) -> int:
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit.",
             "{\\scriptsize\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{longtable}{@{}l>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{2.7cm}r"
             ">{\\raggedright\\arraybackslash}p{3.6cm}rrrl@{}}",
             "\\caption{\\textbf{Every registered fold decision.} One row per screen, in protocol order. Ref.: the "
             "screen's reference cell; Cand.: number of candidates; Fold picks: the cell chosen on each set of four "
             "training folds (ref = reference); OOF: pooled out-of-fold VOC-Sim mIoU of the reference and of the "
             "selection procedure. A change is promoted only if one candidate wins at least four of five folds and "
             "the OOF guard passes. Rows without OOF values come from screens whose record keeps only the votes.}"
             "\\label{tab:record_screens}\\\\",
             "\\toprule", "Pkg. (Sec.) & Screen & Ref. & Cand. & Fold picks & OOF ref & OOF sel. & Gain & Outcome \\\\",
             "\\midrule", "\\endfirsthead", "\\toprule",
             "Pkg. (Sec.) & Screen & Ref. & Cand. & Fold picks & OOF ref & OOF sel. & Gain & Outcome \\\\",
             "\\midrule", "\\endhead"]
    n = 0
    stage = None
    for pkg, rid, st, sec in PACKAGES:
        recs = records(pkg)
        if not recs:
            continue
        if st != stage:
            stage = st
            lines.append(f"\\midrule\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{STAGES[st]}}}}}\\\\")
        for key, r in recs.items():
            ref, sel = oof_pair(r)
            nums = "-- & -- & --" if np.isnan(ref) else f"{ref:.4f} & {sel:.4f} & {100 * (sel - ref):+.2f}"
            lines.append(f"{tex(rid)} ({sec}) & {brk(key)} & {tt(r['reference'])} & {len(r['candidates'])} & "
                         f"{picks(r)} & {nums} & {outcome(r)} \\\\")
            meta.setdefault(pkg, []).append(r)
            n += 1
    lines += ["\\bottomrule", "\\end{longtable}}"]
    (OUT / "record_screens.tex").write_text("\n".join(lines) + "\n")
    return n


# --------------------------------------------------------------------------- cells
def class_miou(counts: np.ndarray, cls: np.ndarray, uniq: list) -> np.ndarray:
    """counts [N, T, 4] -> class-mean IoU per threshold [T]."""
    per = np.stack([counts[cls == k].sum(0) for k in uniq])  # [K, T, 4]
    tp, fp, fn = per[..., 0], per[..., 1], per[..., 2]
    return np.where(tp + fp + fn > 0, tp / np.maximum(tp + fp + fn, 1), 1.0).mean(0)


def pick(scores: np.ndarray, grid: list[int], thr: tuple) -> int:
    return max(grid, key=lambda k: (scores[k], -abs(thr[k] - 0.45), -thr[k]))


def shard_cells(pkg: str) -> tuple[list[dict], bool] | None:
    split = (B / pkg / "voc" / "shards").exists()
    if not split and not (B / pkg / "shards").exists():
        return None
    voc = load(pkg, "voc") if split else _load_flat(pkg)
    coco = load(pkg, "coco") if split and (B / pkg / "coco" / "shards").exists() else None
    if len(voc.configurations) > CELL_LIMIT:
        return None
    thr = voc.thresholds
    vc = np.asarray(voc.classes)
    vu = sorted(set(vc))
    folds = np.asarray(voc.folds)
    if coco is not None:
        cc = np.asarray(coco.classes)
        cu = sorted(set(cc))
    rows = []
    for i, name in enumerate(voc.configurations):
        grid = list(range(len(thr)))
        if len(thr) > 17 and not name.startswith("W/SG13/"):
            grid = [k for k, t in enumerate(thr) if round(t, 3) in STANDARD]
        full = class_miou(voc.counts[i], vc, vu)
        best = pick(full, grid, thr)
        held = np.zeros((voc.counts.shape[1], 4), dtype=np.int64)
        for f in sorted(set(folds)):
            tr = folds != f
            k = pick(class_miou(voc.counts[i, tr], vc[tr], vu), grid, thr)
            held[folds == f] = voc.counts[i, folds == f, k]
        oof = class_miou(held[:, None, :], vc, vu)[0]
        row = {"cell": name, "t": thr[best], "voc": full[best], "oof": oof}
        if coco is not None:
            row["coco"] = class_miou(coco.counts[coco.index(name)], cc, cu)[best]
        rows.append(row)
    return rows, coco is not None


def _load_flat(pkg: str):
    from src.segmentation import controlled_cv as cv
    paths = [p for p in sorted((B / pkg / "shards").glob("*.npz")) if ".diagnostics." not in p.name]
    return cv.load_results(paths)


def json_cells(pkg: str) -> tuple[list[dict], bool] | None:
    p = B / pkg / "results.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text()).get("metrics")
    if not m or "voc" not in m:
        return None
    rows = []
    for name, v in m["voc"].items():
        row = {"cell": name, "t": v.get("threshold", float("nan")), "voc": v["miou"], "oof": float("nan")}
        if "coco" in m and name in m["coco"]:
            row["coco"] = m["coco"][name]["miou"]
        rows.append(row)
    return rows, "coco" in m


def cell_table(pkg: str, rid: str, rows: list[dict], has_coco: bool, recs: list[dict], frozen: set[str]) -> list[str]:
    """Each cell is compared with the reference of the screen that lists it (raw and final screens differ)."""
    by_name = {r["cell"]: r for r in rows}
    refs = {r["reference"] for r in recs}
    own = {}
    for rec in recs:
        for c in [rec["reference"], *rec["candidates"]]:
            own.setdefault(c, rec["reference"])
    first = recs[0]["reference"] if recs else ""
    cols = "@{}>{\\raggedright\\arraybackslash}p{5.2cm}rrrr" + ("rr" if has_coco else "") + "@{}"
    head = "Cell & $t^*$ & VOC & $\\Delta$ & OOF" + (" & COCO & $\\Delta$" if has_coco else "") + " \\\\"
    ncol = 7 if has_coco else 5
    out = ["{\\scriptsize\\setlength{\\tabcolsep}{4pt}", f"\\begin{{longtable}}{{{cols}}}",
           f"\\caption{{\\textbf{{{tex(rid)}: every cell}} ({len(rows)} cells). Screen references "
           f"({', '.join(chr(92) + 'texttt{' + tex(r) + '}' for r in sorted(refs)) or 'none recorded'}) in bold; promoted cell marked $\\star$. "
           "$\\Delta$ in mIoU points against the reference of the screen that lists the cell.}"
           f"\\label{{tab:record_{re.sub(r'[^a-z0-9]', '', pkg)}}}\\\\",
           "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead",
           f"\\midrule\\multicolumn{{{ncol}}}{{r}}{{\\emph{{continued}}}}\\\\", "\\endfoot", "\\bottomrule", "\\endlastfoot"]
    for r in rows:
        name = tt(r["cell"]) + ("$^\\star$" if r["cell"] in frozen else "")
        if r["cell"] in refs:
            name = "\\textbf{" + name + "}"
        r0 = by_name.get(own.get(r["cell"], first))
        dv = "--" if r0 is None else f"{100 * (r['voc'] - r0['voc']):+.2f}"
        oof = "--" if np.isnan(r["oof"]) else f"{r['oof']:.4f}"
        t = "--" if np.isnan(r["t"]) else f"{r['t']:.3f}"
        line = f"{name} & {t} & {r['voc']:.4f} & {dv} & {oof}"
        if has_coco:
            c = r.get("coco", float("nan"))
            dc = "--" if r0 is None or np.isnan(c) else f"{100 * (c - r0.get('coco', np.nan)):+.2f}"
            line += f" & {'--' if np.isnan(c) else f'{c:.4f}'} & {dc}"
        out.append(line + " \\\\")
    out += ["\\end{longtable}}"]
    return out


def write_packages(meta: dict, glossary: dict) -> list[dict]:
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit."]
    ledger = []
    stage = None
    for pkg, rid, st, sec in PACKAGES:
        g = glossary.get(pkg, {})
        recs = meta.get(pkg, [])
        got = shard_cells(pkg) or json_cells(pkg)
        if got is None and not g:
            continue
        if st != stage:
            stage = st
            lines += [f"\\subsubsection{{{STAGES[st]}}}"]
        title = g.get("title", rid)
        lines += [f"\\paragraph{{{tex(rid)}: {tex(title)} (Sec.~{tex(g.get('sections', sec))}).}}"]
        for k in ("what", "names", "reference", "outcome"):
            if g.get(k):
                label = {"what": "", "names": "\\emph{Cell names.} ", "reference": "\\emph{Reference.} ",
                         "outcome": "\\emph{Decision.} "}[k]
                lines.append(label + tex(g[k]))
                lines.append("")
        frozen = {r["frozen"] for r in recs if r["frozen"] != r["reference"]}
        n = 0
        if got is not None:
            rows, has_coco = got
            n = len(rows)
            lines += cell_table(pkg, rid, rows, has_coco, recs, frozen)
        else:
            lines.append("\\emph{Cells are tabulated in the evidence tables (Sec.~\\ref{sec:supp_record_early}).}")
        lines.append("")
        ledger.append({"id": rid, "title": title, "section": g.get("sections", sec), "cells": n or "",
                       "screens": len(recs), "promoted": ", ".join(sorted(frozen)) or "none"})
    (OUT / "record_packages.tex").write_text("\n".join(lines) + "\n")
    return ledger


def write_ledger(rows: list[dict]) -> None:
    early = list(csv.DictReader(LEDGER.open()))
    done = {r["id"] for r in early}
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit.",
             "{\\scriptsize\\setlength{\\tabcolsep}{3pt}",
             "\\begin{longtable}{@{}l>{\\raggedright\\arraybackslash}p{5.0cm}>{\\raggedright\\arraybackslash}p{3.3cm}r"
             ">{\\raggedright\\arraybackslash}p{5.4cm}@{}}",
             "\\caption{\\textbf{Ledger of every experiment.} B: exploratory, before the fold protocol (in-sample, never "
             "used as evidence). A, P3-C, T-C: controlled screens of the first three phases. SG, BL: later registered "
             "packages. Sec.: protocol section with the result. Best/decision: for B rows, the best in-sample "
             "configuration and mIoU; for controlled rows, the promoted cell(s) or ``none''.}\\label{tab:record_ledger}\\\\",
             "\\toprule", "ID & Experiment & Status & Cells & Best / decision \\\\", "\\midrule", "\\endfirsthead",
             "\\toprule", "ID & Experiment & Status & Cells & Best / decision \\\\", "\\midrule", "\\endhead"]
    for r in early:
        lines.append(f"{tex(r['id'])} & {tex(r['experiment'])} & {tex(r['status'])} & {tex(r['configurations'])} & "
                     f"{tex(r['decision_or_role'])} \\\\")
    for r in rows:
        if r["id"] in done:
            continue
        lines.append(f"{tex(r['id'])} & {tex(r['title'])} (Sec.~{tex(r['section'])}) & controlled, 5-fold & {r['cells']} & "
                     f"promoted: {', '.join(tt(x) for x in r['promoted'].split(', ')) if r['promoted'] != 'none' else 'none'} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}}"]
    (OUT / "record_ledger.tex").write_text("\n".join(lines) + "\n")


def write_classes() -> None:
    """Per-class IoU of the main pipelines at the frozen threshold 0.45, both datasets (same-run cells)."""
    cols = [("sg22", "ALL-P1", "1 pass"), ("sg22", "ALL-P2", "2 passes"), ("sg22", "ALL-P2+R2", "FTF"),
            ("sg22", "MID-P2+R2", "MID"), ("sg24_a", "A3/FINAL", "A3")]
    data, best = {}, {}
    for pkg, cell, _ in cols:
        r = load(pkg, "voc")
        best[cell] = pick(class_miou(r.counts[r.index(cell)], np.asarray(r.classes), sorted(set(r.classes))),
                          list(range(len(r.thresholds))), r.thresholds)
    for ds in ("voc", "coco"):
        for pkg, cell, _ in cols:
            r = load(pkg, ds)
            cls = np.asarray(r.classes)
            c = r.counts[r.index(cell), :, best[cell]]
            data[ds, cell] = {re.sub("[-_]", " ", str(k)): c[cls == k, 0].sum() / max(c[cls == k, :3].sum(), 1) for k in sorted(set(cls))}
    classes = sorted(data["voc", "ALL-P1"])
    head = "Class & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{d}}}" for d in ("VOC-Sim", "COCO-Cap")) + " \\\\"
    sub = " & " + " & ".join(c for _ in range(2) for _, _, c in cols) + " \\\\"
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit.",
             "\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{tabular}{@{}l" + "r" * 10 + "@{}}", "\\toprule", head,
             "\\cmidrule(lr){2-6}\\cmidrule(l){7-11}", sub, "\\midrule"]
    for k in classes + ["mean"]:
        vals = []
        for ds in ("voc", "coco"):
            for _, cell, _ in cols:
                d = data[ds, cell]
                v = np.mean(list(d.values())) if k == "mean" else d[k]
                vals.append(f"{100 * v:.1f}")
        if k == "mean":
            lines.append("\\midrule")
        lines.append(f"{tex(k)} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Per-class IoU of the main pipelines} (IoU $\\times 100$), each at its full-VOC threshold "
              f"({', '.join(f'{c} {STANDARD[best[k]]:.3f}' for _, k, c in cols)}). "
              "1 pass, 2 passes, FTF (two passes + DenseCRF) and MID (FTF with SA calls 10--20) are cells of the SG22 run; "
              "A3 is the SG24-A working default (owner override; not promoted by its selector). COCO-Cap uses the VOC "
              "threshold; nothing is chosen on it.}\\label{tab:record_classes}", "\\end{table}"]
    (OUT / "record_classes.tex").write_text("\n".join(lines) + "\n")


def write_transfer() -> None:
    """Frozen COCO-Cap transfer by stage (Secs. 54, 56) and DAAM with an in-fold threshold (Sec. 47)."""
    seg = ROOT / "outputs/segmentation"
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit.",
             "\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
             "Track & Stage & $t$ & mIoU & caption has class & caption lacks class & mean image IoU \\\\", "\\midrule"]
    for track, sub in (("class word (Sec.~54)", "coco_cap"), ("few-shot token (Sec.~56)", "coco_cap_token")):
        d = json.loads((seg / "coco_cap_transfer" / sub / "summary.json").read_text())
        for stage, v in d["stages"].items():
            lines.append(f"{track} & {tt(stage)} & {v['threshold']:.3f} & {v['miou_all']:.4f} & "
                         f"{v['miou_caption_includes_class']:.4f} & {v['miou_caption_excludes_class']:.4f} & "
                         f"{v['mean_image_iou']:.4f} \\\\")
            track = ""
        lines.append("\\midrule")
    n_in, n_out = d["caption_includes_class"], d["caption_excludes_class"]
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}",
              f"\\caption{{\\textbf{{Frozen COCO-Cap transfer by pipeline stage}} (707 images; the caption names the class "
              f"in {n_in} and not in {n_out}). Thresholds frozen on VOC-Sim; nothing is chosen on COCO-Cap. "
              "\\texttt{CA/seed}: cross-attention seed; \\texttt{REF}: one SA pass; \\texttt{P1}: two passes; "
              "\\texttt{R2}: DenseCRF. \\texttt{W/}: class word; \\texttt{T/}: OVAM's released token.}"
              "\\label{tab:record_transfer}", "\\end{table}", ""]
    d = json.loads((seg / "comparisons/daam_fold_threshold/summary.json").read_text())
    lines += ["\\begin{table}[H]\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}",
              "\\begin{tabular}{@{}llrrlrl@{}}", "\\toprule",
              "DAAM variant & Data & OVAM $t$ & mIoU at OVAM $t$ & our $t$ & mIoU at our $t$ & paired $\\Delta$ image IoU [95\\% CI] \\\\",
              "\\midrule"]
    for var, v in d["variants"].items():
        t_ovam = v["voc_sim"]["ovam_fixed_threshold"]
        for ds, x in v.items():
            b = x.get("paired_bootstrap_fold_chosen_vs_fixed", {})
            ci = f"{b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]" if b else "--"
            if "oof_miou_fold_chosen" in x:
                per_fold = ", ".join(f"{f['threshold']:.2f}" for f in x["folds"])
                ours = f"folds: {per_fold} & {x['oof_miou_fold_chosen']:.4f} (OOF)"
            else:
                ours = f"VOC {x['threshold_from_voc']:.3f} & {x['miou_threshold_from_voc']:.4f} (frozen)"
            lines.append(f"{tex(var.replace('_', ' '))} & {tex(ds.replace('_', '-'))} & {t_ovam:.2f} & "
                         f"{x['miou_ovam_fixed']:.4f} & {ours} & {ci} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"\\caption{{\\textbf{{DAAM with its threshold chosen in our folds}} (Sec.~47; grid {tex(d['grid_note'])}). "
              "On COCO-Cap the threshold is the one chosen on VOC-Sim.}\\label{tab:record_daam}", "\\end{table}"]
    (OUT / "record_transfer.tex").write_text("\n".join(lines) + "\n")


def write_diagnostics(glossary: dict) -> None:
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit."]
    for key, g in glossary.items():
        if not g.get("key_results"):
            continue
        lines += [f"\\paragraph{{{tex(g['title'])} (Sec.~{tex(g.get('sections', ''))}).}} {tex(g.get('what', ''))}", "",
                  "\\begin{table}[H]\\centering\\scriptsize", "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{10cm}l@{}}",
                  "\\toprule"]
        for label, value in g["key_results"]:
            lines.append(f"{tex(label)} & {tex(value)} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", f"\\emph{{Finding.}} {tex(g.get('outcome', ''))}", ""]
    (OUT / "record_diagnostics.tex").write_text("\n".join(lines) + "\n")


def write_calibration(glossary: dict) -> None:
    """The Sec. 125 calibration/AP audit: its requested-cells table, as committed."""
    g = glossary.get("calibration_ap_audit", {})
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit."]
    rep = ROOT / "outputs/segmentation/analysis/calibration_ap_audit/report.md"
    if not g or not rep.exists():
        (OUT / "record_calibration.tex").write_text("\n".join(lines) + "\n")
        return
    rows, on = [], False
    for line in rep.read_text().splitlines():
        if line.startswith("| Package |"):
            on = True
            continue
        if on and not line.startswith("|"):
            break
        if on and not line.startswith("|---"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    lines += [f"\\paragraph{{{tex(g['title'])} (Sec.~{tex(g['sections'])}).}} {tex(g['what'])}", "",
              f"\\emph{{Columns.}} {tex(g['names'])}", "",
              "{\\scriptsize\\setlength{\\tabcolsep}{3pt}",
              "\\begin{longtable}{@{}llrrlrll@{}}",
              f"\\caption{{\\textbf{{Calibration and ranking audit}} ({len(rows)} cells). Every $\\Delta$ is against the same "
              "package's same-stage reference. Best-threshold columns use each dataset's labels and are diagnostics, never "
              "results; AP is undefined for binary R2 masks.}\\label{tab:record_calibration}\\\\",
              "\\toprule Package & Cell & VOC OOF $\\Delta$ & COCO frozen $\\Delta$ & COCO best-thr $\\Delta$ (thr) & "
              "VOC best-thr $\\Delta$ & COCO AP $\\Delta$ [95\\%] & VOC AP $\\Delta$ [95\\%] \\\\ \\midrule\\endfirsthead",
              "\\toprule Package & Cell & VOC OOF $\\Delta$ & COCO frozen $\\Delta$ & COCO best-thr $\\Delta$ (thr) & "
              "VOC best-thr $\\Delta$ & COCO AP $\\Delta$ [95\\%] & VOC AP $\\Delta$ [95\\%] \\\\ \\midrule\\endhead",
              "\\bottomrule\\endfoot"]
    for r in rows:
        lines.append(" & ".join(tt(r[0]) if i == 0 else tex(c) for i, c in enumerate(r[:8])) + " \\\\")
    lines += ["\\end{longtable}}", f"\\emph{{Finding.}} {tex(g['outcome'])}", ""]
    (OUT / "record_calibration.tex").write_text("\n".join(lines) + "\n")


def write_pending(glossary: dict) -> None:
    """Packages registered and submitted whose results are not yet merged."""
    lines = ["% Generated by scripts/paper/supp_experiment_record.py; do not edit."]
    for g in glossary.values():
        if g.get("pending"):
            lines += [f"\\paragraph{{{tex(g['title'])} (Sec.~{tex(g['sections'])}).}} {tex(g['what'])}", "",
                      f"\\emph{{Cell names.}} {tex(g['names'])}", "", f"\\emph{{Reference.}} {tex(g['reference'])}", "",
                      f"\\emph{{Status.}} {tex(g['outcome'])}", ""]
    (OUT / "record_pending.tex").write_text("\n".join(lines) + "\n")


def copy_evidence() -> None:
    """The early-phase evidence tables and figures (scripts/narmyn/attention/build_evidence.py), unchanged."""
    import shutil
    src = ROOT / "docs/narmyn/attention/evidence"
    for sub, pattern in (("tables", "*.tex"), ("figures", "*.pdf"), ("figures", "*_ca_modules.png")):
        dst = ROOT / "paper/evidence" / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in (src / sub).glob(pattern):
            shutil.copy2(f, dst / f.name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    glossary = json.loads(GLOSSARY.read_text()) if GLOSSARY.exists() else {}
    meta: dict = {}
    n = write_screens(meta)
    rows = write_packages(meta, glossary)
    write_ledger(rows)
    write_classes()
    write_transfer()
    write_diagnostics(glossary)
    write_calibration(glossary)
    write_pending(glossary)
    copy_evidence()
    print(f"{n} screens, {len(rows)} packages -> {OUT}")


if __name__ == "__main__":
    main()
