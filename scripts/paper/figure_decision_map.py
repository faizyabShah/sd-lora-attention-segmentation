#!/usr/bin/env python3
"""Paper figure: every training-free experiment family on one map (descriptive; nothing is selected here).

One point per registered family, class-word track only. Each point compares a candidate
with the **reference of its own package** (same run, same images), so no cross-package
or cross-environment comparison enters.

  x = change in VOC-Sim mIoU, each cell at its own full-VOC threshold (in-sample, descriptive)
  y = change in COCO-Cap mIoU, each cell at that VOC threshold (frozen transfer)

Plotted cell: the candidate the registered fold rule froze, if any (filled marker);
otherwise the family's best candidate on full VOC (open marker). The fold decision itself
(out-of-fold, protocol v1.2) is what the marker records; the x position is in-sample.

Two panels: (a) operators on the propagated map, against one-pass (or the stated) reference;
(b) full pipelines with DenseCRF, against that package's final pipeline.

Writes ``paper/figures/decision_map.{pdf,png}`` and ``outputs/segmentation/paper_figures/decision_map.csv``.

    python scripts/paper/figure_decision_map.py
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
from src.segmentation import controlled_cv as cv  # noqa: E402

B = ROOT / "outputs" / "segmentation" / "controlled_attention_cv"
LIMITS = {"a": ((-5.0, 2.2), (-3.5, 3.5)), "b": ((-1.2, 0.9), (-1.4, 2.2))}  # mIoU points
# short tag and label offset (points) for every family that is labelled
TAGS = {"P1 two passes": ("P1", (4, 0)), "R2 DenseCRF": ("R2", (4, 1)), "SG8 observation consensus": ("SG8", (6, -9)),
        "SG2 consensus graph": ("SG2", (5, -3)), "SG11 robust consensus": ("SG11", (4, -7)),
        "SG12 observation filtering": ("SG12", (4, 2)), "SG18 middle calls": ("SG18", (-12, 5)),
        "SG23 middle-call SG2 (vs MID one-pass)": ("SG23", (2, -10)), "SG13 region inference": ("SG13", (4, 1)),
        "SG5 contrastive seed": ("SG5", (4, 1)), "SG3 confidence lock": ("SG3", (-15, 2)), "P9 (backlog)": ("P9", (4, -2)),
        "R1 (backlog)": ("R1", (-12, 2)), "H1 (backlog)": ("H1", (-12, 2)), "P3 (backlog)": ("P3", (-14, -4)), "P5 (backlog)": ("P5", (4, -3)),
        "SG15 refinement (vs SG11)": ("SG15", (4, -3)), "SG4 SA regions": ("SG4", (4, -2)),
        "SG2 + CRF (vs FTF)": ("SG2", (4, -2)), "SG12 + CRF (vs FTF)": ("SG12", (4, -2)), "T1 + CRF (vs FTF)": ("SG18 T1", (-27, -3)),
        "SG22 middle calls + CRF (vs FTF)": ("SG22 MID", (4, -6)), "SG23 middle-call SG2/11/12 + CRF": ("SG23", (4, -6)),
        "SG24-A CA weighting (A3)": ("SG24 A3", (-30, 4)), "SG24-B 16 CA modules": ("SG24-B", (4, -2)),
        "SG25 multi-resolution SA": ("SG25", (-6, 5)), "SG26 coarse-guided SA": ("SG26", (4, -6)), "SG27 late fusion": ("SG27", (4, 1)),
        "SG28 PEARL-inspired (vs A3 P2)": ("SG28", (-22, 3)), "SG28 PEARL-inspired + CRF": ("SG28", (4, -2))}
STANDARD = tuple(round(0.30 + 0.025 * i, 3) for i in range(17))

# label, panel, package, reference, candidate regex, frozen cell (or None), status, protocol section
FAMILIES = [
    ("P1 two passes", "a", "sg2_densecrf_confirmation", "W/REF", r"^W/P1$", "W/P1", "promoted", "43"),
    ("R2 DenseCRF", "a", "sg2_densecrf_confirmation", "W/REF", r"^W/REF\+R2$", "W/REF+R2", "promoted", "44"),
    ("SG1 absorbing walk", "a", "sg1_sg3_graph_refinement", "W/REF", r"^W/SG1/", None, "rejected", "71"),
    ("SG3 confidence lock", "a", "sg1_sg3_graph_refinement", "W/REF", r"^W/SG3/", None, "rejected", "71"),
    ("SG5 contrastive seed", "a", "sg5_contrastive_seed", "W/REF", r"^W[SKNEY]/REF$", None, "rejected", "70"),
    ("SG8 observation consensus", "a", "sg8_observation_consensus", "W/REF", r"^W/[ABCD]/", "W/C/rho0p80", "promoted", "78"),
    ("SG9 reciprocal links", "a", "sg9_reciprocal_propagation", "W/REF", r"^W/R\d", None, "rejected", "78"),
    ("SG10 bottleneck", "a", "sg10_seed_connected_bottleneck", "W/REF", r"^W/B", None, "rejected", "79"),
    ("SG2 consensus graph", "a", "sg2_consensus_edges", "W/REF", r"^W/[GO]_.*_n1$", "W/G_cv_g20_n1", "promoted", "85"),
    ("SG4 SA regions", "a", "sg4_region_labels", "W/REF", r"^W/(SPEC|ROWS)", None, "rejected", "85"),
    ("SG11 robust consensus", "a", "sg11_sg12_observation_reliability", "W/REF", r"^W/SG11/", "W/SG11/trim20/rho0p80", "promoted", "89"),
    ("SG12 observation filtering", "a", "sg11_sg12_observation_reliability", "W/REF", r"^W/SG12/", "W/SG12/top50", "promoted", "89"),
    ("SG13 region inference", "a", "sg13_target_region_inference", "W/REF", r"^W/SG13/", None, "rejected", "90"),
    ("SG14 combined consensus (vs SG11)", "a", "sg14", "W/SG11/trim20/rho0p80", r"^W/SG14$", None, "rejected", "92"),
    ("SG15 refinement (vs SG11)", "a", "sg15", "W/SG11/trim20/rho0p80", r"^W/SG11/trim20/rho0p80/SG15", None, "rejected", "92"),
    ("SG17 filtered 2nd pass (vs P1)", "a", "json:sg17", "W/P1", r"^W/SG17", None, "rejected", "97"),
    ("SG18 middle calls", "a", "json:sg18_temporal_windowed", "W/REF", r"^W/T\d$", "W/T1", "promoted", "97"),
    ("SG21 head weighting (vs T1)", "a", "json:sg21_structural", "SG18/T1", r"^SG21-", None, "rejected", "101"),
    ("SG23 middle-call SG2 (vs MID one-pass)", "a", "sg23", "MID-R0", r"^MID-SG(2|11|12)$", "MID-SG2", "borderline", "102"),
    ("SG28 PEARL-inspired (vs A3 P2)", "a", "sg28", "P0", r"^[PBCD][1-3]$", None, "rejected", "112"),
    ("SG2 + CRF (vs FTF)", "b", "sg2_densecrf_confirmation", "W/P1+R2", r"^W/SG2\+R2$", None, "rejected", "89"),
    ("SG12 + CRF (vs FTF)", "b", "sg12_crf", "W/P1+R2", r"^W/SG12/top50\+R2$", None, "rejected", "92"),
    ("T1 + CRF (vs FTF)", "b", "sg18_t1_crf", "W/P1+R2", r"^W/T1\+R2$", None, "rejected", "98"),
    ("SG22 middle calls + CRF (vs FTF)", "b", "sg22", "ALL-P2+R2", r"^(MID|ALL)-P[23]\+R2$", "MID-P2+R2", "promoted", "102"),
    ("SG23 middle-call SG2/11/12 + CRF", "b", "sg23", "ALL-P2+R2", r"^(MID|ALL)-SG(2|11|12)\+R2$", None, "rejected", "102"),
    ("SG24-A CA weighting (A3)", "b", "sg24_a", "A0/FINAL", r"^A[1234]/FINAL$", "A3/FINAL", "override", "103"),
    ("SG24-B 16 CA modules", "b", "sg24_b", "B0-A3/FINAL", r"^B1-A3/FINAL$", None, "rejected", "103"),
    ("SG25 multi-resolution SA", "b", "sg25", "FINE-P2+R2", r"\+R2$", None, "rejected", "104"),
    ("SG26 coarse-guided SA", "b", "sg26", "S0+R2", r"^S[3-6]\+R2$", None, "rejected", "105"),
    ("SG28 PEARL-inspired + CRF", "b", "sg28", "P0+R2", r"^[PBCD][1-3]\+R2$", None, "rejected", "112"),
    ("SG27 late fusion", "b", "sg27", "S0+R2", r"^S([3-9]|\d\d)\+R2$", None, "rejected", "109"),
]
BACKLOG = ["P3", "P5", "P9", "F2", "BG1", "W1", "W2", "W4", "H1", "G1", "G4", "M1", "M2", "R1"]
for fam in BACKLOG:  # Section 83: 14 registered families, all against REF/one_pass, none promoted
    FAMILIES.insert(13, (f"{fam} (backlog)", "a", "training_free_backlog", "REF/one_pass", rf"^{fam}/", None, "rejected", "83"))


def load(package: str, ds: str) -> cv.Results:
    paths = [p for p in sorted((B / package / ds / "shards").glob("*.npz"))
             if ".diagnostics." not in p.name and ".arrival." not in p.name and ".regions." not in p.name]
    try:
        return cv.load_results(paths)
    except ValueError as e:  # SG5 shards store per-class prompts in settings; configurations/thresholds are still checked
        if "settings differ" not in str(e):
            raise
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    for part in parts[1:]:
        assert list(part["configurations"]) == list(parts[0]["configurations"])
        assert list(part["thresholds"]) == list(parts[0]["thresholds"])
    return cv.Results(configurations=tuple(map(str, parts[0]["configurations"])), thresholds=tuple(map(float, parts[0]["thresholds"])),
                      image_ids=tuple(str(v) for q in parts for v in q["image_ids"]), classes=tuple(str(v) for q in parts for v in q["classes"]),
                      folds=np.concatenate([q["folds"].astype(np.int64) for q in parts]),
                      counts=np.concatenate([q["counts"].astype(np.int64) for q in parts], axis=1),
                      ap=np.concatenate([q["ap"].astype(np.float64) for q in parts], axis=1), settings={})


def pooled(c: np.ndarray, classes: np.ndarray) -> float:
    return float(np.mean([c[classes == k, 0].sum() / c[classes == k, :3].sum() for k in sorted(set(classes))]))


def cell_scores(voc: cv.Results, coco: cv.Results, name: str) -> tuple[float, float, float]:
    """Full-VOC best threshold (standard 17-point grid unless the cell is an SG13 candidate), VOC and COCO mIoU there."""
    i = voc.index(name)
    grid = range(len(voc.thresholds))
    if len(voc.thresholds) > 17 and not name.startswith("W/SG13/"):
        grid = [k for k, t in enumerate(voc.thresholds) if round(t, 3) in STANDARD]
    vc, cc = np.asarray(voc.classes), np.asarray(coco.classes)
    scores = {k: pooled(voc.counts[i, :, k], vc) for k in grid}
    best = max(scores, key=lambda k: (scores[k], -abs(voc.thresholds[k] - 0.45), -voc.thresholds[k]))
    return scores[best], pooled(coco.counts[coco.index(name), :, best], cc), voc.thresholds[best]


def json_scores(package: str, name: str) -> tuple[float, float, float]:
    d = json.loads((B / package / "results.json").read_text())["metrics"]
    return d["voc"][name]["miou"], d["coco"][name]["miou"], d["voc"][name].get("threshold", float("nan"))


def write_table(rows: list[dict]) -> None:
    """Supplementary table: every plotted family, its reference, cell and both deltas (points)."""
    tt = lambda x: "\\texttt{" + x.replace("_", "\\_").replace("%", "\\%") + "}"  # noqa: E731
    status = {"promoted": "promoted", "borderline": "promoted$^{a}$", "override": "owner$^{b}$", "rejected": "--"}
    lines = ["% Generated by scripts/paper/figure_decision_map.py; do not edit.",
             "\\begin{table*}[t]", "\\centering", "\\scriptsize\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lrllrrrl@{}}", "\\toprule",
             "Family & Sec. & Reference & Plotted cell & Cand. & $\\Delta$VOC & $\\Delta$COCO & Decision \\\\"]
    for panel, title in (("a", "(a) Operators on the propagated map"), ("b", "(b) Complete pipelines with DenseCRF")):
        lines += ["\\midrule", f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\"]
        for r in (r for r in rows if r["panel"] == panel):
            fam = r["family"].replace("%", "\\%")
            lines.append(f"{fam} & {r['protocol_section']} & {tt(r['reference'])} & {tt(r['plotted_cell'])} & "
                         f"{r['candidates']} & {100 * r['voc_delta']:+.2f} & {100 * r['coco_delta']:+.2f} & {status[r['status']]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{\\textbf{Every registered family in Fig.~2.} Changes in mIoU points against the reference cell of the same run; "
              "VOC at each cell's full-VOC threshold (in-sample), COCO at that threshold (frozen transfer). The plotted cell is the "
              "promoted one, or else the best of the family's candidates (Cand.) on full VOC. Sec.: protocol section with the "
              "registration and result. $^{a}$Promotion depends on the unregistered tie order (Sec.~\\ref{sec:supp_reproducibility}). "
              "$^{b}$Not promoted by the registered selector; retained by owner decision.}",
              "\\label{tab:supp_families}", "\\end{table*}"]
    out = ROOT / "paper/tables/generated/decision_map_families.tex"
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = []
    for label, panel, package, ref, pattern, frozen, status, section in FAMILIES:
        if package.startswith("json:"):
            pkg = package[5:]
            names = list(json.loads((B / pkg / "results.json").read_text())["metrics"]["voc"])
            score = lambda n: json_scores(pkg, n)  # noqa: E731
        else:
            voc, coco = load(package, "voc"), load(package, "coco")
            names = list(voc.configurations)
            score = lambda n, v=voc, c=coco: cell_scores(v, c, n)  # noqa: E731
        cands = [n for n in names if re.search(pattern, n) and n != ref]
        if not cands:
            raise SystemExit(f"{label}: no candidates match {pattern}")
        rv, rc, rt = score(ref)
        if frozen:
            pick = frozen
        else:
            pick = max(cands, key=lambda n: score(n)[0])
        cv_, cc_, ct = score(pick)
        rows.append({"family": label, "panel": panel, "status": status, "protocol_section": section,
                     "package": package.replace("json:", ""), "reference": ref, "plotted_cell": pick,
                     "candidates": len(cands), "voc_ref": round(rv, 4), "voc_cell": round(cv_, 4), "voc_delta": round(cv_ - rv, 4),
                     "coco_ref": round(rc, 4), "coco_cell": round(cc_, 4), "coco_delta": round(cc_ - rc, 4)})
    out_csv = ROOT / "outputs/segmentation/paper_figures/decision_map.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    write_table(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"promoted": dict(marker="o", facecolor="#2a78d6", edgecolor="#2a78d6"),
             "borderline": dict(marker="o", facecolor="#9fc3ee", edgecolor="#2a78d6"),
             "override": dict(marker="*", facecolor="#eda100", edgecolor="#a06d00"),
             "rejected": dict(marker="o", facecolor="white", edgecolor="#898781")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    titles = {"a": "(a) Operators on the propagated map", "b": "(b) Full pipelines with DenseCRF"}
    for ax, panel in zip(axes, ("a", "b")):
        sub = [r for r in rows if r["panel"] == panel]
        ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0); ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=0)
        (x0, x1), (y0, y1) = LIMITS[panel]
        off = []
        for r in sub:
            s = style[r["status"]]
            x, y = 100 * r["voc_delta"], 100 * r["coco_delta"]
            if x < x0 or y < y0:  # destructive variants: drawn on the lower-left edge, values listed in the corner
                off.append(r)
                ax.scatter(max(x, x0 + 0.08), max(y, y0 + 0.08), s=20, marker="v", facecolors="white",
                           edgecolors=s["edgecolor"], linewidths=0.8, zorder=3, clip_on=False)
                continue
            ax.scatter(x, y, s=80 if s["marker"] == "*" else 26, marker=s["marker"],
                       facecolors=s["facecolor"], edgecolors=s["edgecolor"], linewidths=0.9, zorder=3)
            if r["family"] in TAGS:
                tag, dxy = TAGS[r["family"]]
                ax.annotate(tag, (x, y), fontsize=5.4, xytext=dxy, textcoords="offset points", color="#0b0b0b")
        if off:
            txt = "off scale (ΔVOC, ΔCOCO):\n" + "\n".join(f"{r['family'].split(' (')[0]}  ({100 * r['voc_delta']:+.1f}, {100 * r['coco_delta']:+.1f})" for r in off)
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=4.8, ha="left", va="top", color="#52514a")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(titles[panel], fontsize=7.5)
        ax.set_xlabel("Δ VOC-Sim mIoU, points (full set, in-sample)", fontsize=6.5)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Δ COCO-Cap mIoU, points (frozen transfer)", fontsize=6.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle="", marker=style[k]["marker"], markerfacecolor=style[k]["facecolor"],
                      markeredgecolor=style[k]["edgecolor"], markersize=6 if k != "override" else 9, label=lab)
               for k, lab in (("promoted", "promoted by the fold rule"), ("borderline", "promoted; tie-order sensitive"),
                              ("override", "owner override (not promoted)"), ("rejected", "not promoted (best variant)"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = ROOT / "paper/figures/decision_map"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"{len(rows)} families -> {out}.pdf")
    for r in rows:
        print(f"{r['panel']} {r['status']:10s} {r['family']:38s} {r['plotted_cell']:28s} VOC {r['voc_delta']:+.4f} COCO {r['coco_delta']:+.4f}")


if __name__ == "__main__":
    main()
