"""Numerical checks for exact Pre-SG19 features and tie-aware AUC."""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.pre_sg19_late_support import FEATURES, features_at_pixels
from scripts.finalize_pre_sg19_late_support import auc_ap


def main():
    obs = np.zeros((80, 3), np.float32)
    obs[:, 0] = 2.0
    obs[:40, 1] = 0.0; obs[40:, 1] = 4.0
    obs[:, 2] = np.arange(80)/80
    ca = np.array([0.0, 1.0, 0.1], np.float32)
    middle = np.array([1.0, 2.0, 0.2], np.float32)
    x = features_at_pixels(obs, ca, middle, 1e-8)
    assert x.shape == (3, len(FEATURES))
    np.testing.assert_allclose(x[0, FEATURES.index('late_score')], 2.0)
    np.testing.assert_allclose(x[0, FEATURES.index('retention_ratio')], 2.0)
    np.testing.assert_allclose(x[1, FEATURES.index('late_median')], 2.0)
    np.testing.assert_allclose(x[1, FEATURES.index('late_mad')], 2.0)
    np.testing.assert_allclose(x[1, FEATURES.index('late_up_consensus')], 0.5)
    np.testing.assert_allclose(x[1, FEATURES.index('late_vs_middle_consensus')], 0.5)
    np.testing.assert_allclose(x[1, FEATURES.index('late_trimmed_mean')], 2.0)
    assert auc_ap(np.array([1., 1., 0.]), np.array([1, 0, 0])) == (0.75, 0.5)
    assert auc_ap(np.array([3., 2., 1.]), np.array([1, 0, 0])) == (1.0, 1.0)
    print('Pre-SG19 numerical checks passed')


if __name__ == '__main__': main()
