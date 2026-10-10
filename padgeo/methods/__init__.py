"""Scoring methods behind one interface (`padgeo.methods.base.Method`)."""

from __future__ import annotations

from padgeo.config import AggregationConfig, AnchoredNNConfig, GlobalNNConfig, MethodConfig
from padgeo.methods.anchored_nn import AnchoredNN
from padgeo.methods.base import Method
from padgeo.methods.global_nn import GlobalNN


def build_method(config: MethodConfig, aggregation: AggregationConfig) -> Method:
    """Instantiate the method selected by `config.name` (a one-line config change switches method)."""
    if isinstance(config, GlobalNNConfig):
        return GlobalNN(config, aggregation)
    if isinstance(config, AnchoredNNConfig):
        return AnchoredNN(config, aggregation)
    raise TypeError(f"unknown method config {config!r}")
