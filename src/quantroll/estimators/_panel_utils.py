"""Shared input coercion for entity-panel estimators (K-Means, embeddings)."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - import guard
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

__all__ = ["split_periods", "coerce_panel", "coerce_cross_section", "pd"]


def split_periods(df):
    """Split a (time, entity) MultiIndex DataFrame into per-period
    cross-sections, preserving time and per-period entity order."""
    times = list(dict.fromkeys(df.index.get_level_values(0)))
    features = [str(c) for c in df.columns]
    crosses = []
    for t in times:
        cross = df.loc[t]
        if isinstance(cross, pd.Series):
            cross = cross.to_frame().T
        values = np.ascontiguousarray(cross.to_numpy(dtype=np.float64))
        if np.isnan(values).any():
            raise ValueError(
                f"period {t!r} contains NaN features; drop or fill incomplete rows"
            )
        crosses.append((list(map(str, cross.index)), values))
    return crosses, times, features


def coerce_panel(X: Any, allow_2d: bool = False):
    """Return ``(values (T,N,d), times, entities, features)`` from a 3-D
    array, a (time, entity) MultiIndex DataFrame, or (optionally) a single
    2-D cross-section."""
    if pd is not None and isinstance(X, pd.DataFrame):
        if isinstance(X.index, pd.MultiIndex):
            times = list(dict.fromkeys(X.index.get_level_values(0)))
            entities = X.loc[times[0]].index
            T, N, d = len(times), len(entities), X.shape[1]
            values = np.empty((T, N, d))
            for i, t in enumerate(times):
                cross = X.loc[t]
                if not cross.index.equals(entities):
                    cross = cross.reindex(entities)
                    if cross.isna().to_numpy().any():
                        raise ValueError(
                            f"unbalanced panel: period {t!r} is missing entities"
                        )
                values[i] = cross.to_numpy(dtype=np.float64)
            return values, times, list(map(str, entities)), [str(c) for c in X.columns]
        if allow_2d:
            values = X.to_numpy(dtype=np.float64)[None, :, :]
            return values, None, list(map(str, X.index)), [str(c) for c in X.columns]
        raise ValueError(
            "pandas input must use a (time, entity) MultiIndex; "
            "for a single period pass it to update()/predict()"
        )
    values = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if values.ndim == 2 and allow_2d:
        values = values[None, :, :]
    if values.ndim != 3:
        raise ValueError(f"expected a (T, N, d) panel; got shape {values.shape}")
    return values, None, None, None


def coerce_cross_section(X: Any):
    if pd is not None and isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float64), X.index
    values = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D cross-section (N, d); got {values.shape}")
    return values, None
