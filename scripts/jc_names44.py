#!/usr/bin/env python3
"""Run 44: map every method's class-name list onto run 44's canonical class names (scoring is by name).

Canonical names: VOC column = jc_real44.VOC20; Context = jc_real44.CTX59 (VOC classes written as VOC20);
COCO-Object = run 42's CLS80 (the 20 VOC classes under their VOC names via jc_coco40.C2V, the other 60 under COCO names).
`check` asserts a bijection between each method's foreground lines and the canonical classes of the column.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ALIAS = {"airplane": "aeroplane", "plane": "aeroplane", "aeroplane": "aeroplane", "motorcycle": "motorbike", "motorbike": "motorbike",
         "diningtable": "dining table", "dining table": "dining table", "pottedplant": "potted plant", "potted plant": "potted plant",
         "plant": "potted plant", "tvmonitor": "tv", "tv monitor": "tv", "monitor": "tv", "tv": "tv", "couch": "sofa", "sofa": "sofa",
         "bicycle": "bicycle", "person": "person", "people": "person", "ship": "boat", "television monitor": "tv", "television": "tv"}

CTX_ALIAS = {"laptop": "computer", "bedding": "bedclothes"}           # Context only ("laptop" is its own COCO class)


def canon_sets():
    from jc_real44 import VOC20, CTX59
    import jc_ov42
    return {"voc": list(VOC20), "context": list(CTX59), "coco": list(jc_ov42.CLS80)}


def to_canon(line, s, canon):
    """Canonical class for one vocabulary line (synonyms comma-separated, NERVE-style descriptors after the first word)."""
    first = line.split(",")[0].strip().lower()
    cands = [first, " ".join(first.split()[:2]), first.split()[0] if first else ""]
    for c in cands:
        if s == "context" and c in CTX_ALIAS:                                  # iSeg's Context prompt words
            return CTX_ALIAS[c]
        if s in ("voc", "context") and c == "table":
            return "dining table"
        if c in canon:
            return c
        if c in ALIAS and ALIAS[c] in canon:
            return ALIAS[c]
    return None


def method_index(method, s, lines, cls):
    """0-based index into `lines` (foreground lines only) whose canonical name is `cls`."""
    canon = canon_sets()[s]
    hits = [i for i, l in enumerate(lines) if to_canon(l, s, canon) == cls]
    assert len(hits) == 1, (method, s, cls, hits)
    return hits[0]


def check(name_files):
    canon = canon_sets(); bad = 0
    for (method, s), (lines, offset) in name_files.items():
        fg = lines[offset:]; m = [to_canon(l, s, canon[s]) for l in fg]
        miss = sorted(set(canon[s]) - set(m)); dup = sorted({x for x in m if x and m.count(x) > 1}); unk = [l for l, x in zip(fg, m) if x is None]
        ok = not miss and not dup and not unk
        bad += not ok
        print(f"{method:10s} {s:8s} lines {len(fg):3d} -> {'OK' if ok else 'FAIL'}" + ("" if ok else f" missing {miss} dup {dup} unknown {unk[:5]}"))
    return bad


if __name__ == "__main__":
    R = ROOT / "other repositories"
    read = lambda p: [l.strip() for l in open(p) if l.strip()]
    F = {}
    for repo in ("sclip", "clearclip", "naclip", "proxyclip"):
        for s, f in (("voc", "cls_voc21.txt"), ("context", "cls_context60.txt"), ("coco", "cls_coco_object.txt")):
            F[(repo, s)] = (read(R / repo / "configs" / f), 1)                 # line 0 = background
    for s, f in (("voc", "data/voc/cls_voc21.txt"), ("context", "data/voc/cls_context60.txt"), ("coco", "data/coco14/cls_coco81.txt")):
        F[("nerve", s)] = (read(R / "nerve" / f), 1)
    sys.exit(check(F))
