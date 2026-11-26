#!/usr/bin/env python3
"""Side-N execution of side F's pre-registered runs/25 (runs/25_token_ladder_nr/PREREG.md), unchanged.

Thin wrapper around the committed `scripts/jc_ladder_nr.py`; nothing in it is edited. Three redirections only:
- BRACE_CACHE_DIR -> ~/TRDP/external/n_side/token_ladder_nr/cache (maps and scores; F's cache is not readable here);
- the report goes to ~/TRDP/external/n_side/token_ladder_nr/report/ instead of runs/25_token_ladder_nr/;
- COCO-Cap rows come from the rebuilt `ovam_coco_cap.csv` (committed scripts/build_coco_cap_manifest.py), as in
  comparisons/official/brace_repro.py, because the CSV is not in the released archive.
F's fidelity check against the runs/20 capture needs F's cache, so it reads NaN here; `fidelity.py` compares the
class-word BRACE map and image with side N's runs/20 reproduction (comparisons/official/) instead.

  python comparisons/token_ladder_nr/run_nr.py capture --dataset voc --shard 0 --num-shards 2
  python comparisons/token_ladder_nr/run_nr.py score --workers 40
  python comparisons/token_ladder_nr/run_nr.py report
"""
import os
import sys
from pathlib import Path

EXT = Path.home() / "TRDP/external/n_side"
os.environ["BRACE_CACHE_DIR"] = str(EXT / "token_ladder_nr/cache")
ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "comparisons/official"):
    sys.path.insert(0, str(p))

import jc_ladder_nr as L  # noqa: E402


def _official_patch_rows():
    """comparisons/official/brace_repro.patch_rows, loaded by path (scripts/ has an unrelated brace_repro.py)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("n_official_brace_repro", ROOT / "comparisons/official/brace_repro.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.patch_rows


_official_patch_rows()(EXT / "brace_repro/ovam_coco_cap.csv")
L.OUT = EXT / "token_ladder_nr/report"

if __name__ == "__main__":
    sys.argv[0] = str(ROOT / "scripts/jc_ladder_nr.py")
    import runpy
    g = vars(L)
    src = (ROOT / "scripts/jc_ladder_nr.py").read_text()
    main = src[src.index('if __name__ == "__main__":'):].split("\n", 1)[1]
    exec(compile("\n".join(l[4:] for l in main.splitlines()), "jc_ladder_nr.__main__", "exec"), g)
