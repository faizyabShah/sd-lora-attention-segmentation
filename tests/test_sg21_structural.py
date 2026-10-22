"""Algebraic checks for the locked SG21 within-call weighting rules."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.segmentation.sg21_structural import descriptors, weights_from_descriptors


def test_descriptors_match_row_average_definition():
    a = torch.tensor([[[.75, .25], [.10, .90]]] * 8)
    distance = torch.tensor([[0., 1.], [1., 0.]])
    local = torch.eye(2, dtype=torch.bool)
    actual = descriptors(a, distance, local)
    expected_distance = ((.25 + .10) / 2)
    expected_local = ((.75 + .90) / 2)
    torch.testing.assert_close(actual[:, 1], torch.full((8,), expected_distance))
    torch.testing.assert_close(actual[:, 2], torch.full((8,), expected_local))


def test_locked_weights_and_deterministic_rank_ties():
    local = torch.tensor([.2, .3, .3, .4, .5, .6, .7, .8])
    entropy = torch.arange(8, dtype=torch.float32) / 8
    distance = torch.arange(1, 9, dtype=torch.float32)
    values = torch.stack((entropy, distance, local), -1)
    w = weights_from_descriptors(values)
    torch.testing.assert_close(w.sum(-1), torch.ones(4))
    torch.testing.assert_close(w[0], local / local.sum())
    torch.testing.assert_close(w[1], torch.arange(1., 9.) / 36)
    assert w[2, 0] > w[2, 7]
    assert w[3, 0] > w[3, 7]


if __name__ == '__main__':
    test_descriptors_match_row_average_definition()
    test_locked_weights_and_deterministic_rank_ties()
    print('SG21 structural algebra passed')
