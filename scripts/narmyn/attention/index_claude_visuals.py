#!/usr/bin/env python3
"""Validate and index Claude's SG2 / SG4 qualitative pages (protocol Section 81).

Same layout and per-sample contract as Codex's ``scripts/finalize_sg_visual_audit.py``
(one deterministic VOC-Sim image per class, every cell), written to a separate
directory so that the shared audit index is not modified.

    python scripts/narmyn/attention/index_claude_visuals.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VISUALS = ROOT / "outputs" / "segmentation" / "sg_visual_audit_claude"
EXPECTED = {"sg2": 56, "sg4": 40}


def main() -> None:
    grouped, records = defaultdict(list), []
    for path in sorted(VISUALS.glob("*/*/index.json")):
        rows = json.loads(path.read_text())
        family = rows[0]["family"]
        if len(rows) != EXPECTED[family]:
            raise SystemExit(f"{path}: expected {EXPECTED[family]} cells, found {len(rows)}")
        grouped[family].append((path.parent, rows)); records += rows
    for family in EXPECTED:
        if len(grouped[family]) != 20:
            raise SystemExit(f"{family}: {len(grouped[family])}/20 classes rendered")
    (VISUALS / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# SG2 and SG4 qualitative audit (Claude; protocol Section 81)", "",
             "One deterministic VOC-Sim image per class (the first of the class) for every SG2 and SG4 cell, in the layout of the shared SG audit "
             "(`src/segmentation/sg_visual_audit.py`): released image, CA seed overlay, SA-derived score overlay, predicted mask with the "
             "ground-truth outline (cyan). Thresholds are each cell's all-VOC threshold. These pages are for inspection only; nothing is selected from them, "
             "and 20 images do not represent the package scores.", "", "| Package | Cells per class | Classes |", "|---|---:|---:|"]
    lines += [f"| `{f}` | {EXPECTED[f]} | {len(grouped[f])} |" for f in EXPECTED]
    for family in EXPECTED:
        lines += ["", f"## {family}", "", "| Class | Image | Pages |", "|---|---|---|"]
        for folder, rows in sorted(grouped[family], key=lambda x: x[1][0]["class"]):
            pages = sorted({r["page"] for r in rows})
            lines.append(f"| {rows[0]['class']} | `{rows[0]['image_id']}` | " + " ".join(f"[{i + 1}]({folder.relative_to(VISUALS) / p})" for i, p in enumerate(pages)) + " |")
    (VISUALS / "README.md").write_text("\n".join(lines) + "\n")
    print(f"indexed {len(records)} rows")


if __name__ == "__main__":
    main()
