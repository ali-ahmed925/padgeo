"""Logging setup shared by all entry-point scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the root logger to write to stderr and, optionally, to a file.

    Calling it again replaces the previous handlers (each seed of a run writes its own log file).

    Args:
        log_file: If given, log records are also appended to this file (parent directories are created).
        level: Logging level for both handlers.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    formatter = logging.Formatter(_FORMAT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
