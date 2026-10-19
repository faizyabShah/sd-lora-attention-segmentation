#!/usr/bin/env python3
"""Finalize Claude's standalone SG2 or SG4 package (protocol Section 81).

Selection is protocol v1.2 (``scripts/select_propagation_cv.run_screen``) for
every registered screen in the family config, with its tie order; COCO-Cap
never selects. Diagnostics are summed over images from ``diagnostics/*.jsonl[.gz]``.

    python scripts/narmyn/attention/finalize_claude_sg.py --family sg2
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.narmyn.attention.an1_propagation_depth import rates  # noqa: E402
from scripts.narmyn.attention.run_claude_sg import FAMILIES  # noqa: E402
from scripts.select_propagation_cv import best_threshold, image_ious, pooled_miou, run_screen  # noqa: E402
from src.segmentation import controlled_cv as cv  # noqa: E402


def load(out: Path, dataset: str):
    shards = sorted((out / dataset / "shards").glob("*.npz"))
    if not shards:
        return None
    r = cv.load_results(shards)
    if not all(r.complete) or len(set(r.classes)) != 20 or len(set(r.image_ids)) != len(r.image_ids):
        raise SystemExit(f"{dataset}: shards are incomplete or overlap")
    return r


def diagnostics(out: Path, dataset: str) -> dict:
    seed, ref = defaultdict(lambda: np.zeros(2, np.int64)), defaultdict(lambda: np.zeros(4, np.int64))
    oracle, regions, edges = defaultdict(list), defaultdict(list), defaultdict(lambda: np.zeros(2))
    n = 0
    for path in sorted((out / dataset / "diagnostics").glob("*.jsonl*")):
        for line in (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()):
            d = json.loads(line); n += 1
            for k, v in d["change_vs_seed"].items():
                seed[k] += v
            for k, v in d["change_vs_ref"].items():
                ref[k] += v
            for k, v in d.get("oracle_iou", {}).items():
                oracle[k].append(v)
            for k, v in d.get("regions", {}).items():
                regions[k].append(v)
            for rel, pairs in d.get("edge_reliability", {}).items():
                for pair, (wr, w) in pairs.items():
                    edges[(rel, pair)] += (wr, w)
    res = {"images": n,
           "addition_vs_seed": {k: {"new_tp": int(v[0]), "new_fp": int(v[1]), "fp_per_tp": float(v[1] / max(v[0], 1))} for k, v in seed.items()},
           "change_vs_reference": {k: dict(zip(("new_tp", "new_fp", "lost_tp", "removed_fp"), map(int, v))) for k, v in ref.items()}}
    if oracle:
        res["oracle_iou_mean"] = {k: float(np.mean(v)) for k, v in oracle.items()}
        res["regions_mean"] = {k: float(np.mean(v)) for k, v in regions.items()}
    if edges:
        res["edge_reliability"] = {f"{rel} {pair}": float(wr / max(w, 1e-30)) for (rel, pair), (wr, w) in sorted(edges.items())}
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    args = parser.parse_args()
    fam = importlib.import_module(FAMILIES[args.family])
    out = fam.OUT
    spec = json.loads(fam.CONFIG.read_text(encoding="utf-8"))
    voc, coco = load(out, "voc"), load(out, "coco")
    if voc is None:
        raise SystemExit("No VOC-Sim shards")
    full = np.ones(len(voc.image_ids), bool)
    t45 = voc.thresholds.index(0.45)
    vc = np.asarray(voc.classes)
    screens = {s["name"]: run_screen(voc, s["reference"], s["candidates"], s["inclusiveness"]) for s in spec["screens"]}
    thr = {n: best_threshold(voc, voc.index(n), full)[0] for n in voc.configurations}
    cells = {}
    for n in voc.configurations:
        i = voc.index(n)
        cells[n] = {"voc_threshold": voc.thresholds[thr[n]], "voc_miou@threshold": pooled_miou(voc.counts[i, :, thr[n]], vc),
                    "voc_an1@0.45": rates(voc.counts[i, :, t45], vc)}
    transfer = {}
    if coco is not None:
        cc = np.asarray(coco.classes)
        for n in coco.configurations:
            i = coco.index(n)
            cells[n].update({"coco_miou@voc_threshold": pooled_miou(coco.counts[i, :, thr[n]], cc), "coco_an1@0.45": rates(coco.counts[i, :, t45], cc)})
        for name, it in screens.items():
            a = coco.counts[coco.index(it["frozen"]), :, thr[it["frozen"]]]
            b = coco.counts[coco.index(it["reference"]), :, thr[it["reference"]]]
            transfer[name] = {"frozen": it["frozen"], "reference": it["reference"], "coco_frozen": pooled_miou(a, cc),
                              "coco_reference": pooled_miou(b, cc), "paired_image_iou": cv.paired_bootstrap(image_ious(a), image_ious(b))}
    envs = sorted({json.dumps({k: e.get(k) for k in ("torch", "gpu", "node", "conda_env")}, sort_keys=True)
                   for r in (voc, coco) if r for e in r.environments})
    diag = {ds: diagnostics(out, ds) for ds in ("voc", "coco") if (out / ds / "diagnostics").exists()}
    summary = {"protocol": "Section 81", "family": args.family, "voc_images": len(voc.image_ids),
               "coco_images": len(coco.image_ids) if coco else 0, "environments": envs, "screens": screens,
               "coco_transfer": transfer, "cells": cells, "diagnostics": diag}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {spec['family']} (protocol Section 81)", "",
             f"VOC-Sim {summary['voc_images']} images (selection, 5 folds); COCO-Cap {summary['coco_images']} (transfer only).", "",
             "| Screen | Vote winner | Votes | OOF reference → selection | Gain | Frozen | Why |", "|---|---|---:|---:|---:|---|---|"]
    for name, it in screens.items():
        lines.append(f"| {name} | `{it['vote_winner']}` | {it['vote_winner_votes']}/5 | {it['oof_miou_reference']:.4f} → "
                     f"{it['oof_miou_selection']:.4f} | {it['oof_gain']:+.4f} | `{it['frozen']}` | {it['freeze_reason']} |")
    if transfer:
        lines += ["", "| Screen | Frozen | COCO frozen | COCO reference | Paired Δ image IoU [95% CI] |", "|---|---|---:|---:|---:|"]
        for name, t in transfer.items():
            p = t["paired_image_iou"]
            lines.append(f"| {name} | `{t['frozen']}` | {t['coco_frozen']:.4f} | {t['coco_reference']:.4f} | {p['mean']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}] |")
    lines += ["", "| Cell | VOC mIoU@thr | thr | VOC R / P @0.45 | COCO mIoU@VOC thr | COCO R / P @0.45 | VOC new FP per new TP vs seed |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for n, c in cells.items():
        v = c["voc_an1@0.45"]
        k = c.get("coco_an1@0.45")
        ratio = diag.get("voc", {}).get("addition_vs_seed", {}).get(n, {}).get("fp_per_tp", float("nan"))
        lines.append(f"| `{n}` | {c['voc_miou@threshold']:.4f} | {c['voc_threshold']:.3f} | {v['recall']:.3f} / {v['precision']:.3f} | "
                     + (f"{c['coco_miou@voc_threshold']:.4f} | {k['recall']:.3f} / {k['precision']:.3f} |" if k else "– | – |") + f" {ratio:.2f} |")
    for ds, d in diag.items():
        if "oracle_iou_mean" in d:
            lines += ["", f"Oracle-region IoU ({ds}; region labelled object when at least half its pixels are object):", "",
                      "| Partition | Oracle mean image IoU | Non-empty regions |", "|---|---:|---:|"]
            lines += [f"| {k} | {v:.4f} | {d['regions_mean'][k]:.1f} |" for k, v in d["oracle_iou_mean"].items()]
        if "edge_reliability" in d:
            lines += ["", f"A_bar-weighted mean edge reliability ({ds}):", "", "| Reliability, edge type | Mean |", "|---|---:|"]
            lines += [f"| {k} | {v:.4f} |" for k, v in d["edge_reliability"].items()]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len(screens) + 1]))


if __name__ == "__main__":
    main()
