"""Map construction for controlled self-attention time and resolution screens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .controlled_ca import configuration_names, resolution_configurations, time_windows


@torch.inference_mode()
def compose_time_resolution_maps(
    observation_sums: torch.Tensor,
    observation_counts: torch.Tensor,
    resolution_values: Sequence[int],
    windows: Mapping[str, Sequence[int]] | None = None,
    resolution_sets: Mapping[str, Sequence[int]] | None = None,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Compose SA-T, SA-R, and joint maps from call-resolution observations.

    ``observation_sums[t, r]`` is the sum of the propagated maps produced by
    every selected module/head observation at call ``t`` and resolution ``r``.
    ``observation_counts`` stores the corresponding number of equally weighted
    observations. This preserves the protocol's equal observation weighting
    when different resolutions contain different numbers of UNet modules.
    """
    windows = dict(windows or time_windows())
    resolution_sets = dict(resolution_sets or resolution_configurations())
    resolution_values = tuple(int(value) for value in resolution_values)
    if observation_sums.ndim != 4:
        raise ValueError("observation_sums must have shape [calls, resolutions, H, W]")
    if tuple(observation_counts.shape) != tuple(observation_sums.shape[:2]):
        raise ValueError("observation_counts must match the first two sum dimensions")
    if len(resolution_values) != observation_sums.shape[1]:
        raise ValueError("resolution_values does not match the resolution dimension")

    resolution_index = {value: index for index, value in enumerate(resolution_values)}
    unknown = {
        value
        for selected in resolution_sets.values()
        for value in selected
        if value not in resolution_index
    }
    if unknown:
        raise ValueError(f"No observations were provided for resolutions {sorted(unknown)}")

    def combine(calls: Sequence[int], selected: Sequence[int]) -> torch.Tensor:
        call_indices = torch.as_tensor(tuple(calls), device=observation_sums.device)
        resolution_indices = torch.as_tensor(
            tuple(resolution_index[int(value)] for value in selected),
            device=observation_sums.device,
        )
        values = observation_sums.index_select(0, call_indices).index_select(
            1, resolution_indices
        )
        counts = observation_counts.index_select(0, call_indices).index_select(
            1, resolution_indices
        )
        count = counts.sum()
        if int(count) == 0:
            raise ValueError(f"No SA observations for calls={tuple(calls)}, resolutions={tuple(selected)}")
        return values.sum(dim=(0, 1)) / count.to(values.dtype)

    all_window = "all_00_30"
    if all_window not in windows:
        raise KeyError(f"Required inclusive time reference {all_window!r} is missing")
    all_resolutions = tuple(resolution_values)
    names: list[str] = []
    maps: list[torch.Tensor] = []
    for window_name, calls in windows.items():
        names.append(f"time/{window_name}")
        maps.append(combine(calls, all_resolutions))
    for resolution_name, selected in resolution_sets.items():
        names.append(f"resolution/{resolution_name}")
        maps.append(combine(windows[all_window], selected))
    for window_name, calls in windows.items():
        for resolution_name, selected in resolution_sets.items():
            names.append(f"time_resolution/{window_name}__{resolution_name}")
            maps.append(combine(calls, selected))

    expected = configuration_names(windows, resolution_sets)
    if tuple(names) != expected:
        raise RuntimeError("Controlled SA configuration order changed unexpectedly")
    return expected, torch.stack(maps)

