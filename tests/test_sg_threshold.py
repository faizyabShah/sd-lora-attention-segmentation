"""SG-threshold rule definitions on synthetic maps."""
import numpy as np

from src.segmentation.sg_threshold import (GRID, gmm_threshold, otsu, quantile_for_area, rule_slots,
                                           snap, stable_threshold)


def _bimodal(seed=0, low=.2, high=.8, fg=.3):
    rng = np.random.default_rng(seed)
    n = 4096
    y = np.where(rng.random(n) < fg, rng.normal(high, .05, n), rng.normal(low, .05, n))
    return np.clip(y, 0, 1)


def test_snap_and_edges():
    assert snap(.3) == (0, False) and snap(.7) == (16, False)
    assert snap(.51) == (8, False) and snap(.2) == (0, True) and snap(.75) == (16, True)


def test_otsu_and_gmm_split_two_clear_modes():
    y = _bimodal()
    assert .35 < otsu(y) < .65
    assert .35 < gmm_threshold(y) < .65
    # Unequal weights move the GMM crossing towards the smaller mode.
    assert gmm_threshold(_bimodal(fg=.1)) > gmm_threshold(_bimodal(fg=.5))


def test_stable_threshold_sits_in_the_histogram_gap():
    y = _bimodal(low=.15, high=.85)
    assert .4 <= stable_threshold(y) <= .6


def test_quantile_matches_area():
    y = np.linspace(0, 1, 1001)
    assert abs((y >= quantile_for_area(y, .25)).mean() - .25) < 2e-3
    assert (y >= quantile_for_area(y, 1.)).all() and not (y >= quantile_for_area(y, 0.)).any()


def test_rule_slots():
    y = _bimodal()
    t0, e0 = rule_slots("T0", y)
    assert (t0 == np.arange(17)).all() and not e0.any()
    for rule in ("T1", "T2", "T4"):
        s, _ = rule_slots(rule, y)
        assert len(set(s)) == 1
    t3, _ = rule_slots("T3", y)
    assert (np.diff(t3) >= 0).all()
    # T5 on R0 itself reproduces the global grid wherever the map has no ties.
    smooth = np.linspace(0, 1, 4096)
    t5, _ = rule_slots("T5", smooth, smooth)
    assert (t5 == np.arange(17)).all()
    t6, _ = rule_slots("T6", y)
    assert (np.diff(t6) >= 0).all() and GRID[t6[8]] == GRID[snap(otsu(y))[0]]
