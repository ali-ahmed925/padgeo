"""Wall-clock timing and peak GPU memory, reported for every query of every run."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch


def _cuda_active() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def reset_peak_vram() -> None:
    """Start a new peak-memory measurement window on the current CUDA device (no-op without CUDA)."""
    if _cuda_active():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float | None:
    """Peak memory allocated by tensors on the current CUDA device since the last reset, in MiB.

    Returns None if CUDA was never used in this process (CPU-only runs).
    """
    if not _cuda_active():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


@contextmanager
def stopwatch() -> Iterator[list[float]]:
    """Measure wall-clock seconds of a block, synchronising CUDA so queued kernels are counted.

    Usage: `with stopwatch() as elapsed: ...` then `elapsed[0]` holds the seconds.
    """
    elapsed = [0.0]
    if _cuda_active():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        if _cuda_active():
            torch.cuda.synchronize()
        elapsed[0] = time.perf_counter() - start
