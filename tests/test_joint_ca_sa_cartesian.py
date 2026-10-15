import torch

from scripts.run_joint_ca_sa_cartesian import (
    CA_CONFIGURATIONS,
    JOINT_CONFIGURATIONS,
    REFERENCE,
    SA_CHOICES,
    _propagate_seed_batch,
)


def test_registered_cartesian_grid_is_complete_and_unique():
    assert len(CA_CONFIGURATIONS) == 6
    assert len(SA_CHOICES) == 13
    assert len(JOINT_CONFIGURATIONS) == 78
    assert len(set(JOINT_CONFIGURATIONS)) == 78
    assert REFERENCE in JOINT_CONFIGURATIONS


def test_batched_seed_propagation_matches_independent_matvecs():
    affinity = torch.tensor(
        [
            [[0.75, 0.25], [0.10, 0.90]],
            [[0.20, 0.80], [0.60, 0.40]],
        ]
    )
    seeds = torch.tensor([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]])
    observed = _propagate_seed_batch(affinity, seeds)
    expected = torch.stack(
        [torch.stack([head @ seed for head in affinity]) for seed in seeds]
    )
    torch.testing.assert_close(observed, expected)
