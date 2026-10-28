"""Numerical equivalence checks for independently accumulated SA controls."""
from __future__ import annotations

import torch

CONTROL_ATOL = 1e-5
CONTROL_RTOL = 1e-4


def assert_control_close(actual: torch.Tensor, reference: torch.Tensor, name: str):
    """Tolerate float32 reduction-order noise while rejecting material drift."""
    try:
        torch.testing.assert_close(actual, reference, atol=CONTROL_ATOL,
                                   rtol=CONTROL_RTOL, msg=name)
    except AssertionError as error:
        max_abs = float((actual.float()-reference.float()).abs().max())
        raise AssertionError(f"{name}: max abs difference {max_abs:.8g}; "
                             f"atol={CONTROL_ATOL}, rtol={CONTROL_RTOL}") from error
