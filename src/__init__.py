"""EMVisual/EMBase public package API (lazy exports — avoid heavy import side effects)."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "DatasetLoader",
    "GroundTruthDataset",
    "reshape_abundances",
    "DimensionalityReducer",
    "PCAReducer",
    "AEReducer",
    "turbo_extract_nfindr",
    "turbo_fcls_map",
    "estimate_n_endmembers",
    "reconstruct_and_compute_metrics",
    "Pipeline",
]

if TYPE_CHECKING:
    from .loaders import DatasetLoader
    from .dataset_loaders_gt import GroundTruthDataset, reshape_abundances
    from .dim_reduction import DimensionalityReducer, PCAReducer, AEReducer
    from .unmixing import (
        turbo_extract_nfindr,
        turbo_fcls_map,
        estimate_n_endmembers,
        reconstruct_and_compute_metrics,
    )
    from .pipeline import Pipeline


def __getattr__(name: str):
    if name == "DatasetLoader":
        from .loaders import DatasetLoader

        return DatasetLoader
    if name == "GroundTruthDataset":
        from .dataset_loaders_gt import GroundTruthDataset

        return GroundTruthDataset
    if name == "reshape_abundances":
        from .dataset_loaders_gt import reshape_abundances

        return reshape_abundances
    if name == "DimensionalityReducer":
        from .dim_reduction import DimensionalityReducer

        return DimensionalityReducer
    if name == "PCAReducer":
        from .dim_reduction import PCAReducer

        return PCAReducer
    if name == "AEReducer":
        from .dim_reduction import AEReducer

        return AEReducer
    if name == "turbo_extract_nfindr":
        from .unmixing import turbo_extract_nfindr

        return turbo_extract_nfindr
    if name == "turbo_fcls_map":
        from .unmixing import turbo_fcls_map

        return turbo_fcls_map
    if name == "estimate_n_endmembers":
        from .unmixing import estimate_n_endmembers

        return estimate_n_endmembers
    if name == "reconstruct_and_compute_metrics":
        from .unmixing import reconstruct_and_compute_metrics

        return reconstruct_and_compute_metrics
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
