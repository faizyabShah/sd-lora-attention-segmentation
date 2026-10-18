import torch

from src.segmentation.reciprocal_propagation import reciprocal_affinity, row_normalize


def test_registered_reciprocal_operators_are_row_stochastic():
    affinity = torch.tensor([[0.7, 0.3], [0.1, 0.9]])
    for operator in ("R1_geometric", "R2_minimum", "R3_soft_gate"):
        result = reciprocal_affinity(affinity, operator)
        assert torch.all(result >= 0)
        assert torch.allclose(result.sum(dim=1), torch.ones(2))


def test_symmetric_affinity_is_unchanged_by_geometric_and_minimum():
    affinity = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    assert torch.allclose(reciprocal_affinity(affinity, "R1_geometric"), affinity)
    assert torch.allclose(reciprocal_affinity(affinity, "R2_minimum"), affinity)


def test_row_normalization_handles_tiny_rows():
    value = row_normalize(torch.zeros(2, 2), epsilon=1e-6)
    assert torch.equal(value, torch.zeros_like(value))
