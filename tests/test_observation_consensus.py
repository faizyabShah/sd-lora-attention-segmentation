import torch

from src.segmentation.observation_consensus import configuration_names, consensus_maps


def test_sg8_maps_and_limits():
    seed = torch.tensor([[0.2, 0.5, 0.8], [0.1, 0.4, 0.9]])
    observations = torch.stack((seed - 0.1, seed, *(seed + 0.1 for _ in range(7)), seed + 5.0))
    maps, diagnostics = consensus_maps(observations, seed, (0.05,), (0.60, 0.90))
    names = configuration_names(("W", "T"), (0.05,), (0.60, 0.90))
    assert maps.shape == (18, 3)
    assert len(names) == maps.shape[0]
    assert torch.allclose(maps[names.index("W/REF")], observations[:, 0].mean(0))
    assert torch.allclose(diagnostics["sigma"], observations.std(0, correction=0))
    # Eight of ten observations increase the seed: rho=.60 accepts, rho=.90 rejects.
    assert torch.allclose(maps[names.index("W/C/rho0p60")], observations[:, 0].mean(0))
    assert torch.allclose(maps[names.index("W/C/rho0p90")], seed[0])
    assert torch.all(maps[names.index("W/A/median")] < maps[names.index("W/REF")])


def test_expansion_only_never_reduces_seed():
    seed = torch.tensor([[0.3, 0.7]])
    observations = torch.tensor([[[0.1, 0.8]], [[0.2, 0.9]], [[0.4, 1.0]], [[0.5, 1.1]],
                                 [[0.4, 0.9]], [[0.5, 1.0]], [[0.4, 0.8]], [[0.5, 0.9]],
                                 [[0.4, 1.0]], [[0.5, 1.1]]])
    maps, _ = consensus_maps(observations, seed, (0.1,), (0.6,))
    names = configuration_names(("W",), (0.1,), (0.6,))
    assert torch.all(maps[names.index("W/D/rho0p60")] >= seed[0])
