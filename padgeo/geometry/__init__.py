"""Geometry backends (VGGT and a model-free fake) and the join to the patch grid."""

from __future__ import annotations

from padgeo.config import FakeGeometryConfig, GeometryConfig, VGGTGeometryConfig
from padgeo.geometry.base import GeometryBackend
from padgeo.geometry.fake import FakeGeometryBackend
from padgeo.geometry.vggt_backend import VGGTBackend


def build_geometry_backend(config: GeometryConfig) -> GeometryBackend:
    """Instantiate the backend selected by `config.backend` (no weights are loaded here)."""
    if isinstance(config, FakeGeometryConfig):
        return FakeGeometryBackend(config)
    if isinstance(config, VGGTGeometryConfig):
        return VGGTBackend(config)
    raise TypeError(f"unknown geometry config {config!r}")
