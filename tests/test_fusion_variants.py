import torch

from src.segmentation.fusion_variants import (
    gate,
    geometric_mean,
    max_normalize,
    minmax,
    otsu_threshold,
    residual_mix,
)


def test_residual_mix_endpoints():
    y = torch.tensor([[0.2, 0.8]])
    c = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(residual_mix(y, c, 1.0), y)
    torch.testing.assert_close(residual_mix(y, c, 0.0), c)
    torch.testing.assert_close(residual_mix(y, c, 0.25), 0.25 * y + 0.75 * c)


def test_gate_and_geometric_mean():
    y = torch.tensor([[0.25, 1.0, 0.0]])
    c = torch.tensor([[1.0, 0.36, 0.5]])
    torch.testing.assert_close(gate(y, c), torch.tensor([[0.25, 0.36, 0.0]]))
    torch.testing.assert_close(geometric_mean(y, c), torch.tensor([[0.5, 0.6, 0.0]]))


def test_normalizations():
    maps = torch.tensor([[[2.0, 4.0], [3.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]])
    torch.testing.assert_close(minmax(maps)[0], torch.tensor([[0.0, 1.0], [0.5, 0.0]]))
    torch.testing.assert_close(minmax(maps)[1], torch.zeros(2, 2))
    torch.testing.assert_close(max_normalize(maps)[0], torch.tensor([[0.5, 1.0], [0.75, 0.5]]))


def test_otsu_separates_two_modes():
    values = torch.cat((torch.full((300,), 0.2), torch.full((100,), 0.8)))
    threshold = otsu_threshold(values)
    assert 0.2 < threshold <= 0.8
    assert ((values >= threshold) == (values > 0.5)).all()
