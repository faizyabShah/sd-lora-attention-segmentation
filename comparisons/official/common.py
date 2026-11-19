"""Shared constants for the official N-side comparison (comparisons/official/)."""
from pathlib import Path

import numpy as np

# Extended threshold grid (27 values): runs/20's grid (0.30-0.70, step 0.025, 17 values) plus 0.05-0.25 and
# 0.75-0.95 at step 0.05, so BRACE can be reproduced on runs/20's grid exactly and every method (some of which
# peak outside 0.30-0.70) is scored on the same wider grid. Step 0.05 outside keeps DenseCRF cost down.
THR_EXT = np.round(np.r_[0.05 * np.arange(1, 6), 0.30 + 0.025 * np.arange(17), 0.75 + 0.05 * np.arange(5)], 3)
THR_R20 = np.flatnonzero((THR_EXT >= 0.30 - 1e-9) & (THR_EXT <= 0.70 + 1e-9))   # indices of runs/20's grid
EXT = Path.home() / "TRDP/external/n_side"
OFFICIAL_OUT = EXT / "official"


def crf_site() -> Path:
    """SimpleCRF 0.2.1.1 (the repo's pinned OVAM DenseCRF wrapper), built for this Python outside the repo."""
    return EXT / "deps/simplecrf_py311"
