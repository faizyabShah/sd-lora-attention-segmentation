"""Map construction for the controlled CA time and resolution screens."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import torch

from .attention_aggregation import load_step_resolution_sums, resize_map
from .captured_dataset import AttentionSample


def time_windows() -> dict[str, tuple[int, ...]]:
    """Return the protocol Section 4 CA-T configurations."""
    windows: dict[str, tuple[int, ...]] = {"all_00_30": tuple(range(31))}
    windows.update({f"step_{step:02d}": (step,) for step in range(31)})
    for start, end in ((0, 4), (5, 9), (10, 14), (15, 20), (21, 25), (26, 30)):
        windows[f"window_{start:02d}_{end:02d}"] = tuple(range(start, end + 1))
    for start in (5, 10, 15, 21, 26):
        windows[f"trailing_{start:02d}_30"] = tuple(range(start, 31))
    for end in (4, 9, 14, 20, 25):
        windows[f"cumulative_00_{end:02d}"] = tuple(range(end + 1))
    windows.update(
        {
            f"leave_step_{step:02d}": tuple(i for i in range(31) if i != step)
            for step in range(31)
        }
    )
    return windows


def resolution_configurations() -> dict[str, tuple[int, ...]]:
    """Return all nonempty subsets of the four SD 1.5 CA resolutions."""
    return {
        "r" + "_r".join(map(str, values)): values
        for width in range(1, 5)
        for values in itertools.combinations((8, 16, 32, 64), width)
    }


def configuration_names(
    windows: Mapping[str, Sequence[int]] | None = None,
    resolutions: Mapping[str, Sequence[int]] | None = None,
) -> tuple[str, ...]:
    """Return the fixed order expected by the compact result writer."""
    windows = windows or time_windows()
    resolutions = resolutions or resolution_configurations()
    names = [f"time/{name}" for name in windows]
    names.extend(f"resolution/{name}" for name in resolutions)
    names.extend(
        f"time_resolution/{window_name}__{resolution_name}"
        for window_name in windows
        for resolution_name in resolutions
    )
    return tuple(names)


@torch.inference_mode()
def build_time_resolution_maps(
    sample: AttentionSample,
    token_positions: Sequence[int],
    windows: Mapping[str, Sequence[int]] | None = None,
    resolution_sets: Mapping[str, Sequence[int]] | None = None,
    working_size: int = 64,
    branch: int = 1,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Build every CA-T, CA-R and joint CA-TxR map for one sample.

    Every module-call observation has equal weight. Resizing the accumulated
    sum is equivalent to resizing each observation before summing because
    bilinear interpolation is linear.
    """
    windows = dict(windows or time_windows())
    resolution_sets = dict(resolution_sets or resolution_configurations())
    needed_resolutions = tuple(
        sorted({resolution for values in resolution_sets.values() for resolution in values})
    )
    memberships = {
        step: tuple(name for name, indices in windows.items() if step in indices)
        for step in range(len(sample.timesteps))
    }
    sums = {
        name: {
            resolution: torch.zeros((resolution, resolution), dtype=torch.float32)
            for resolution in needed_resolutions
        }
        for name in windows
    }
    counts = {
        name: {resolution: 0 for resolution in needed_resolutions}
        for name in windows
    }

    for step_index in range(len(sample.timesteps)):
        step_maps = load_step_resolution_sums(
            sample, step_index, token_positions, needed_resolutions, branch=branch
        )
        for window_name in memberships[step_index]:
            for resolution, (value, count) in step_maps.items():
                sums[window_name][resolution] += value
                counts[window_name][resolution] += count

    resized: dict[str, dict[int, torch.Tensor]] = {}
    for window_name in windows:
        resized[window_name] = {}
        for resolution in needed_resolutions:
            if counts[window_name][resolution] > 0:
                resized[window_name][resolution] = resize_map(
                    sums[window_name][resolution], working_size
                )

    def combine(window_name: str, selected: Sequence[int]) -> torch.Tensor:
        available = [r for r in selected if counts[window_name].get(r, 0) > 0]
        observation_count = sum(counts[window_name][r] for r in available)
        if not available or observation_count == 0:
            raise ValueError(
                f"No attention observations for {window_name} and resolutions {tuple(selected)}"
            )
        return sum((resized[window_name][r] for r in available), start=torch.zeros(
            (working_size, working_size), dtype=torch.float32
        )) / observation_count

    all_window = "all_00_30"
    if all_window not in windows:
        raise KeyError(f"Required inclusive time reference {all_window!r} is missing")
    maps: list[torch.Tensor] = []
    names: list[str] = []
    for window_name in windows:
        names.append(f"time/{window_name}")
        maps.append(combine(window_name, needed_resolutions))
    for resolution_name, selected in resolution_sets.items():
        names.append(f"resolution/{resolution_name}")
        maps.append(combine(all_window, selected))
    for window_name in windows:
        for resolution_name, selected in resolution_sets.items():
            names.append(f"time_resolution/{window_name}__{resolution_name}")
            maps.append(combine(window_name, selected))

    expected = configuration_names(windows, resolution_sets)
    if tuple(names) != expected:
        raise RuntimeError("Controlled CA configuration order changed unexpectedly")
    return expected, torch.stack(maps)
