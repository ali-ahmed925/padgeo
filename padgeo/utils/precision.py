"""Precision settings shared by the geometry backends and the feature extractor."""

from __future__ import annotations

import contextlib

import torch

from padgeo.config import Precision

_DTYPES: dict[Precision, torch.dtype] = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def torch_dtype(precision: Precision) -> torch.dtype:
    """The torch dtype named by a config precision ("fp32", "fp16", "bf16")."""
    return _DTYPES[precision]


def autocast(precision: Precision, device: torch.device) -> contextlib.AbstractContextManager[None]:
    """Autocast context for a forward pass: disabled for "fp32", else autocast to the half dtype."""
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=_DTYPES[precision])
