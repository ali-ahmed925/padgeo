"""Fail-loud checks for tensors crossing module boundaries."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def check_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: Sequence[int | None],
    dtype: torch.dtype,
    *,
    finite: bool = False,
    value_range: tuple[float, float] | None = None,
) -> None:
    """Raise a `ValueError` describing the first violated expectation.

    Args:
        name: Human-readable name used in the error message.
        tensor: The tensor to check.
        shape: Expected shape; `None` entries accept any size along that dimension.
        dtype: Expected dtype.
        finite: If True, every element must be finite (no NaN/inf).
        value_range: If given, every element must lie in the closed interval [low, high].
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name}: expected a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dim() != len(shape) or any(
        want is not None and got != want for got, want in zip(tensor.shape, shape, strict=True)
    ):
        raise ValueError(f"{name}: expected shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {tensor.dtype}")
    if finite and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name}: contains non-finite values")
    if value_range is not None and tensor.numel() > 0:
        low, high = value_range
        t_min, t_max = float(tensor.min()), float(tensor.max())
        if t_min < low or t_max > high:
            raise ValueError(f"{name}: values in [{t_min}, {t_max}] fall outside [{low}, {high}]")
