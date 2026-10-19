#!/usr/bin/env python3
"""Finalize SG10 (protocol Section 77): fold selection on VOC-Sim, diagnostics, COCO-Cap transfer.

Selection is protocol v1.2 (``scripts/select_propagation_cv.run_screen``), one
screen per track against its ``REF``, over the 16 registered candidates, with
the registered tie order. ``seed``, ``P1`` and the local-graph ``L_*`` cells are
reported only. COCO-Cap never selects.

Diagnostics (from ``diagnostics/*.jsonl``):
  connectivity of seed -> REF additions: share of each 20-bin histogram, mean
    Bn, and AUC of Bn for new TP against new FP (and against distant /
    separate-component new FP);
  per cell: new FP per new TP against the seed (AN1 ratio), and lost TP /
    removed FP against REF;
  REF components: max Bn of components that touch the object vs those that do not.

    python scripts/narmyn/attention/finalize_sg10.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.narmyn.attention.an1_propagation_depth import rates  # noqa: E402
from scripts.select_propagation_cv import best_threshold, image_ious, pooled_miou, run_screen  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402

CONFIG = ROOT / "configs" / "sg10_seed_connected_bottleneck.json"
OUT = ROOT / "outputs" / "segmentation" / "controlled_attention_cv" / "sg10_seed_connected_bottleneck"
OPS, GRAPHS, QS = ("B1", "B2a25", "B2a50", "B3"), ("G1", "G2"), (90, 95)
OP_RANK = {"B2a50": 4, "B2a25": 3, "B3": 2, "B1": 1}
CENTERS = (np.arange(20) + 0.5) / 20


def load(dataset: str) -> cv.Results | None:
    shards = sorted((OUT / dataset / "shards").glob("*.npz"))
    if not shards:
        return None
    r = cv.load_results(shards)
    if not all(r.complete) or len(set(r.classes)) != 20 or len(set(r.image_ids)) != len(r.image_ids):
        raise SystemExit(f"{dataset}: shards are incomplete or overlap")
    return r


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """P(Bn of a positive > Bn of a negative), ties counted half, from two histograms on the same bins."""
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    below = np.concatenate(([0], np.cumsum(neg)[:-1]))
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def diagnostics(dataset: str) -> dict:
    hist = defaultdict(lambda: np.zeros(20, dtype=np.int64))
    seed_add = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    ref_change = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    comps = defaultdict(list)
    for path in sorted((OUT / dataset / "diagnostics").glob("*.jsonl")):
        for line in path.open():
            rec = json.loads(line)
            for t, d in rec["tracks"].items():
                for key, groups in d["hist"].items():
                    for g, h in groups.items():
                        hist[(t, key, g)] += np.asarray(h)
                for cell, v in d["change_vs_seed"].items():
                    seed_add[(t, cell)] += (v["new_tp"], v["new_fp"])
                for cell, v in d["change_vs_ref"].items():
                    ref_change[(t, cell)] += (v["new_tp"], v["new_fp"], v["lost_tp"], v["removed_fp"])
                for c in d["components"]:
                    for key in (k for k in c if k not in ("size", "tp_fraction")):
                        comps[(t, key, "object" if c["tp_fraction"] > 0 else "separate")].append(c[key][0])
    out = {"connectivity_of_additions": {}, "addition_ratio_vs_seed": {}, "change_vs_ref": {}, "components": {}}
    for (t, key, g), h in sorted(hist.items()):
        out["connectivity_of_additions"].setdefault(t, {}).setdefault(key, {})[g] = {
            "pixels": int(h.sum()), "mean_bn": float((h * CENTERS).sum() / max(h.sum(), 1)), "hist": h.tolist()}
    for t, keys in out["connectivity_of_additions"].items():
        for key, groups in keys.items():
            tp = np.asarray(groups["new_tp"]["hist"])
            groups["auc_new_tp_vs"] = {g: auc(tp, np.asarray(groups[g]["hist"])) for g in ("new_fp", "new_fp_distant", "new_fp_separate")}
    for (t, cell), (tp, fp) in sorted(seed_add.items()):
        out["addition_ratio_vs_seed"].setdefault(t, {})[cell] = {"new_tp": int(tp), "new_fp": int(fp), "fp_per_tp": float(fp / max(tp, 1))}
    for (t, cell), v in sorted(ref_change.items()):
        out["change_vs_ref"].setdefault(t, {})[cell] = dict(zip(("new_tp", "new_fp", "lost_tp", "removed_fp"), map(int, v)))
    for (t, key, kind), vals in sorted(comps.items()):
        out["components"].setdefault(t, {}).setdefault(key, {})[kind] = {"components": len(vals), "mean_max_bn": float(np.mean(vals))}
    return out


def main() -> None:
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    voc, coco = load("voc"), load("coco")
    if voc is None:
        raise SystemExit("No VOC-Sim shards")
    full = np.ones(len(voc.image_ids), dtype=bool)
    t45 = voc.thresholds.index(0.45)
    vc = np.asarray(voc.classes)
    screens = {}
    for s in spec["screens"]:
        t = s["track"]
        cands = [f"{t}/{op}_{g}_q{q}" for g in GRAPHS for q in QS for op in OPS]
        incl = {f"{t}/{op}_{g}_q{q}": OP_RANK[op] * 100 + (2 if g == "G1" else 1) * 10 + (2 if q == 90 else 1)
                for g in GRAPHS for q in QS for op in OPS}
        screens[s["name"]] = run_screen(voc, s["reference"], cands, incl)
    thr = {n: best_threshold(voc, voc.index(n), full)[0] for n in voc.configurations}
    cells = {}
    for n in voc.configurations:
        i = voc.index(n)
        cells[n] = {"voc_miou@0.45": pooled_miou(voc.counts[i, :, t45], vc), "voc_threshold": voc.thresholds[thr[n]],
                    "voc_miou@threshold": pooled_miou(voc.counts[i, :, thr[n]], vc), "voc_an1@0.45": rates(voc.counts[i, :, t45], vc)}
    transfer = {}
    if coco is not None:
        cc = np.asarray(coco.classes)
        for n in coco.configurations:
            i = coco.index(n)
            cells[n].update({"coco_miou@voc_threshold": pooled_miou(coco.counts[i, :, thr[n]], cc),
                             "coco_an1@0.45": rates(coco.counts[i, :, t45], cc)})
        for name, item in screens.items():
            a = coco.counts[coco.index(item["frozen"]), :, thr[item["frozen"]]]
            b = coco.counts[coco.index(item["reference"]), :, thr[item["reference"]]]
            transfer[name] = {"frozen": item["frozen"], "reference": item["reference"], "coco_frozen": pooled_miou(a, cc),
                              "coco_reference": pooled_miou(b, cc), "paired_image_iou": cv.paired_bootstrap(image_ious(a), image_ious(b))}
    diag = {ds: diagnostics(ds) for ds in ("voc", "coco") if (OUT / ds / "diagnostics").exists()}
    envs = sorted({json.dumps({k: e.get(k) for k in ("torch", "gpu", "node", "conda_env")}, sort_keys=True)
                   for r in (voc, coco) if r for e in r.environments})
    summary = {"protocol": "Section 77", "voc_images": len(voc.image_ids), "coco_images": len(coco.image_ids) if coco else 0,
               "bottleneck_normalization": voc.settings["bottleneck_normalization"], "environments": envs,
               "screens": screens, "coco_transfer": transfer, "cells": cells, "diagnostics": diag}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = ["# SG10 seed-connected bottleneck propagation (protocol Section 77)", "",
             f"VOC-Sim {summary['voc_images']} images (selection, 5 folds); COCO-Cap {summary['coco_images']} (transfer only). "
             f"Bottleneck normalization: {summary['bottleneck_normalization']}.", "",
             "| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Why |", "|---|---|---:|---:|---:|---|---|"]
    for name, it in screens.items():
        lines.append(f"| {name} | `{it['vote_winner']}` | {it['vote_winner_votes']}/5 | {it['oof_miou_reference']:.4f} → "
                     f"{it['oof_miou_selection']:.4f} | {it['oof_gain']:+.4f} | `{it['frozen']}` | {it['freeze_reason']} |")
    if transfer:
        lines += ["", "| Screen | Frozen | COCO frozen | COCO reference | Paired Δ image IoU [95% CI] |", "|---|---|---:|---:|---:|"]
        for name, t in transfer.items():
            p = t["paired_image_iou"]
            lines.append(f"| {name} | `{t['frozen']}` | {t['coco_frozen']:.4f} | {t['coco_reference']:.4f} | "
                         f"{p['mean']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] |")
    lines += ["", "| Cell | VOC mIoU@thr | thr | VOC R / P @0.45 | COCO mIoU@VOC thr | COCO R / P @0.45 |", "|---|---:|---:|---:|---:|---:|"]
    for n, c in cells.items():
        v = c["voc_an1@0.45"]
        row = f"| `{n}` | {c['voc_miou@threshold']:.4f} | {c['voc_threshold']:.3f} | {v['recall']:.3f} / {v['precision']:.3f} |"
        if "coco_an1@0.45" in c:
            k = c["coco_an1@0.45"]
            row += f" {c['coco_miou@voc_threshold']:.4f} | {k['recall']:.3f} / {k['precision']:.3f} |"
        else:
            row += " – | – |"
        lines.append(row)
    for ds, d in diag.items():
        lines += ["", f"Connectivity of seed → REF additions ({ds}): mean Bn; AUC of Bn for new TP against new FP / distant / separate", "",
                  "| Track | Graph | New TP | New FP | Distant FP | Separate FP | AUC vs FP | vs distant | vs separate |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for t, keys in d["connectivity_of_additions"].items():
            for key, g in keys.items():
                a = g["auc_new_tp_vs"]
                fmt = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
                lines.append(f"| {t} | {key} | {g['new_tp']['mean_bn']:.3f} | {g['new_fp']['mean_bn']:.3f} | {g['new_fp_distant']['mean_bn']:.3f} | "
                             f"{g['new_fp_separate']['mean_bn']:.3f} | {fmt(a['new_fp'])} | {fmt(a['new_fp_distant'])} | {fmt(a['new_fp_separate'])} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(screens) + 1]))


if __name__ == "__main__":
    main()
