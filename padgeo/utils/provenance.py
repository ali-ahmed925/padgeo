"""Provenance recorded with every result file: code version, library versions and hardware."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

# Distributions whose exact versions affect results.
_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "scikit-learn",
    "scikit-image",
    "Pillow",
    "pydantic",
    "PyYAML",
    "matplotlib",
    "huggingface-hub",
    "safetensors",
    "vggt",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True, timeout=30
    )
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    """Return the repository commit and whether the working tree has uncommitted changes.

    Raises:
        RuntimeError: If the code is not inside a git checkout (results must be traceable to a commit).
    """
    try:
        commit = _git("rev-parse", "HEAD")
        dirty_files = _git("status", "--porcelain", "--untracked-files=no")
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(f"Cannot read git state of {_REPO_ROOT}; run from a git checkout.") from error
    return {"commit": commit, "dirty": bool(dirty_files), "dirty_files": dirty_files.splitlines()}


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that affect results (None if a package is not installed).

    For VCS installs (VGGT is installed from a pinned git commit), the commit is appended.
    """
    versions: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            continue
        version = dist.version
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            vcs = json.loads(direct_url).get("vcs_info")
            if vcs:
                version = f"{version}+{vcs.get('vcs', 'vcs')}.{vcs.get('commit_id')}"
        versions[name] = version
    return versions


def hardware() -> dict[str, Any]:
    """CPU, GPU and CUDA details of the machine producing a result."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["gpus"].append(
                {
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / 2**20),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def collect_provenance() -> dict[str, Any]:
    """Everything needed to trace a result back to code, environment and hardware."""
    return {"git": git_state(), "packages": package_versions(), "hardware": hardware()}
