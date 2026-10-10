"""Global seeding and deterministic-kernel settings.

The pipeline's only intended randomness is reference sampling, which uses its own explicitly seeded
generators (`padgeo.data.sampling`). Seeding the global RNGs here is a safety net so that any library
code that draws from them is reproducible too.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

# cuBLAS needs a fixed workspace configuration for deterministic GEMMs; it is read when the first cuBLAS
# handle is created, so it must be set before any CUDA work happens.
_CUBLAS_WORKSPACE = ":4096:8"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA) RNGs and request deterministic kernels.

    Args:
        seed: Non-negative integer seed.

    Raises:
        ValueError: If `seed` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE:
        logger.warning("CUDA was initialised before seeding; cuBLAS determinism setting may not apply.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG that third-party code may use
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: some inference kernels have no deterministic variant; they are reported, not fatal.
    torch.use_deterministic_algorithms(True, warn_only=True)
