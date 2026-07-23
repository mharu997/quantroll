"""Soft dependency handling: accept numpy / pandas / polars inputs uniformly.

The core computes on float64 numpy arrays; this module unwraps labeled inputs
and rebuilds matching labeled outputs, so DataFrames flow through the library
without pandas/polars being hard dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - import guard
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:  # pragma: no cover - import guard
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

__all__ = ["unwrap", "wrap_2d", "wrap_1d"]


def unwrap(X: Any) -> tuple[np.ndarray, Any, list[str] | None, str]:
    """Return ``(values, index, columns, kind)`` for numpy/pandas/polars input.

    ``kind`` is one of ``"numpy"``, ``"pandas"``, ``"polars"``. ``index`` is
    None except for pandas. Values are float64 and C-contiguous.
    """
    if pd is not None and isinstance(X, pd.DataFrame):
        values = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
        return values, X.index, [str(c) for c in X.columns], "pandas"
    if pd is not None and isinstance(X, pd.Series):
        values = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
        return values, X.index, None, "pandas"
    if pl is not None and isinstance(X, pl.DataFrame):
        values = np.ascontiguousarray(X.to_numpy().astype(np.float64))
        return values, None, list(X.columns), "polars"
    values = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    return values, None, None, "numpy"


def wrap_2d(values: np.ndarray, index: Any, columns: list[str], kind: str) -> Any:
    """Rebuild a 2-D labeled container matching the input ``kind``."""
    if kind == "pandas" and pd is not None:
        return pd.DataFrame(values, index=index, columns=columns)
    if kind == "polars" and pl is not None:
        return pl.DataFrame({c: values[:, i] for i, c in enumerate(columns)})
    return values


def wrap_1d(values: np.ndarray, index: Any, name: str, kind: str) -> Any:
    if kind == "pandas" and pd is not None:
        return pd.Series(values, index=index, name=name)
    return values
