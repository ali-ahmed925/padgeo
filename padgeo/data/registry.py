"""Dataset construction from config. Adding a dataset means adding a loader and a branch here."""

from __future__ import annotations

from padgeo.config import DataConfig
from padgeo.data.base import AnomalyDataset
from padgeo.data.mad_sim import MADSim


def build_dataset(config: DataConfig) -> AnomalyDataset:
    """Instantiate the loader named by `config.dataset`."""
    if config.dataset == "mad_sim":
        return MADSim(config.root)
    raise ValueError(f"unknown dataset {config.dataset!r}")
