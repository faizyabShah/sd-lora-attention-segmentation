"""Observation-consensus operators for SG8 self-attention propagation.

The input contains already-propagated maps ``A_i c``.  Keeping these vectors
is inexpensive and avoids retaining any full spatial affinity matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def configuration_names(
    tracks: Sequence[str], taus: Sequence[float], rhos: Sequence[float]
) -> tuple[str, ...]:
    """Return the fixed SG8 output order, one independent family per track."""
    names: list[str] = []
    for track in tracks:
        names.extend((f"{track}/REF", f"{track}/A/trim10", f"{track}/A/trim20", f"{track}/A/median"))
        names.extend(f"{track}/B/tau{_tag(tau)}" for tau in taus)
        names.extend(f"{track}/C/rho{_tag(rho)}" for rho in rhos)
        names.extend(f"{track}/D/rho{_tag(rho)}" for rho in rhos)
    return tuple(names)


@torch.inference_mode()
def consensus_maps(
    propagated: torch.Tensor,
    seeds: torch.Tensor,
    taus: Sequence[float],
    rhos: Sequence[float],
    trim_fractions: Sequence[float] = (0.10, 0.20),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build SG8 maps from ``propagated[observation, track, pixel]``.

    The result is ordered as :func:`configuration_names`.  Standard deviation
    uses the population definition in the preregistration.
    """
    if propagated.ndim != 3 or seeds.ndim != 2:
        raise ValueError("expected propagated [O,S,P] and seeds [S,P]")
    if propagated.shape[1:] != seeds.shape:
        raise ValueError("propagated track/pixel dimensions must match seeds")
    if propagated.shape[0] < 2:
        raise ValueError("at least two observations are required")
    if tuple(trim_fractions) != (0.10, 0.20):
        raise ValueError("SG8-A is registered with 10% and 20% trimming")
    if any(tau <= 0 for tau in taus):
        raise ValueError("tau must be positive")
    if any(not 0.5 <= rho <= 1 for rho in rhos):
        raise ValueError("rho must lie in [0.5, 1]")

    observations = propagated.shape[0]
    mean = propagated.mean(dim=0)
    sigma = propagated.std(dim=0, correction=0)
    upward = (propagated > seeds.unsqueeze(0)).float().mean(dim=0)
    downward = (propagated < seeds.unsqueeze(0)).float().mean(dim=0)
    ordered = propagated.sort(dim=0).values
    robust: list[torch.Tensor] = []
    for fraction in trim_fractions:
        cut = int(observations * fraction)
        if cut == 0 or cut * 2 >= observations:
            raise ValueError("invalid trim fraction for observation count")
        robust.append(ordered[cut : observations - cut].mean(dim=0))
    # ``torch.median`` returns the lower middle element for an even number of
    # observations; SG8-A requires the conventional average of both middles.
    middle = observations // 2
    median = (ordered[middle - 1] + ordered[middle]) / 2 if observations % 2 == 0 else ordered[middle]

    outputs: list[torch.Tensor] = []
    for track in range(seeds.shape[0]):
        seed, mu, sd = seeds[track], mean[track], sigma[track]
        up, down = upward[track], downward[track]
        outputs.extend((mu, robust[0][track], robust[1][track], median[track]))
        for tau in taus:
            uncertainty = sd / (sd + float(tau))
            outputs.append(uncertainty * seed + (1.0 - uncertainty) * mu)
        for rho in rhos:
            agreed = (up >= float(rho)) | (down >= float(rho))
            outputs.append(torch.where(agreed, mu, seed))
        for rho in rhos:
            outputs.append(torch.where(up >= float(rho), torch.maximum(seed, mu), seed))
    diagnostics = {"mean": mean, "sigma": sigma, "r_up": upward, "r_down": downward}
    return torch.stack(outputs), diagnostics

