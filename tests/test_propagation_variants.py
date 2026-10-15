import torch

from src.segmentation.propagation_variants import (
    power_trajectory,
    restart_trajectory,
    sparse_one_pass,
)


def test_power_trajectory_matches_manual_matrix_powers():
    affinity = torch.tensor([[0.75, 0.25], [0.10, 0.90]])
    seed = torch.tensor([1.0, 0.0])
    observed = power_trajectory(affinity, seed, 3)
    one = affinity @ seed
    two = affinity @ one
    three = affinity @ two
    torch.testing.assert_close(observed, torch.stack((one, two, three)))


def test_restart_endpoints_and_batching():
    affinity = torch.tensor(
        [[[0.75, 0.25], [0.10, 0.90]], [[0.20, 0.80], [0.60, 0.40]]]
    )
    seed = torch.tensor([1.0, 0.0])
    observed = restart_trajectory(affinity, seed, (0.0, 1.0), 2)
    torch.testing.assert_close(observed[:, :, 0], power_trajectory(affinity, seed, 2))
    torch.testing.assert_close(observed[:, :, 1], seed.expand(2, 2, 2))


def test_sparse_one_pass_renormalizes_selected_entries():
    affinity = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.45, 0.2, 0.35]])
    seed = torch.tensor([1.0, 0.5, 0.0])
    observed = sparse_one_pass(affinity, seed, (1, 2))
    expected_k1 = torch.tensor([1.0, 0.5, 1.0])
    expected_k2 = torch.tensor([(0.5 + 0.15) / 0.8, 0.3 / 0.9, 0.45 / 0.8])
    torch.testing.assert_close(observed[0], expected_k1)
    torch.testing.assert_close(observed[1], expected_k2)
