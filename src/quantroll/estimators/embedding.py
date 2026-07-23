"""Scikit-learn-style estimator for temporally stable rolling embeddings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.rolling_embedding import (
    RollingEmbeddingResult,
    _EmbeddingStepper,
    make_backend,
    rolling_embedding,
)
from ._panel_utils import pd
from ._panel_utils import split_periods as _split_periods

__all__ = ["RollingEmbedding"]


class RollingEmbedding:
    """Stable low-dimensional map of an entity universe over time.

    Wraps any per-period embedding backend (built-in PCA fallback, UMAP via
    the optional ``umap-learn`` package, or your own callable) with the
    stabilization the library applies everywhere: warm-started refits,
    closed-form Procrustes removal of global rotation / reflection /
    translation, and per-entity displacement diagnostics — so month-to-month
    maps are comparable and an entity's trajectory through the map is
    meaningful.

    Parameters
    ----------
    n_components : int
        Embedding dimension (2 for maps).
    backend : {"pca", "umap"} or callable ``f(X, init) -> (N, k)``
        Per-period reducer. ``init`` carries warm-start coordinates.
    scale : bool
        Also remove global scale changes during alignment.
    seed : int | None
        Seed for stochastic backends.
    backend_kwargs : dict | None
        Extra backend arguments (e.g. ``dict(n_neighbors=25, min_dist=0.3)``).

    Attributes
    ----------
    embeddings_ : (T, U, k) aligned coordinates (NaN rows where absent).
    disparity_ : (T,) structural map change after removing global motion.
    displacement_ : (T, U) per-entity movement between consecutive periods.
    entity_names_, times_, feature_names_in_ : labels when fitted from pandas.

    Examples
    --------
    >>> from quantroll import RollingEmbedding, simulate
    >>> X, membership, _ = simulate.drifting_blobs(40, seed=0)
    >>> emb = RollingEmbedding(backend="pca").fit(X)
    >>> emb.embeddings_.shape
    (40, 90, 2)
    >>> emb.update(X[-1])                     # stream one more period
    RollingEmbedding(backend='pca')
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        backend: str | Callable = "pca",
        scale: bool = False,
        seed: int | None = 0,
        backend_kwargs: dict | None = None,
    ) -> None:
        self.n_components = n_components
        self.backend = backend
        self.scale = scale
        self.seed = seed
        self.backend_kwargs = backend_kwargs

    # ------------------------------------------------------------------ fit

    def fit(self, X: Any, y: Any = None) -> RollingEmbedding:
        """Fit over a (T, N, d) numpy panel or a (time, entity) MultiIndex
        DataFrame (unbalanced panels welcome: absent entities get NaN rows)."""
        if pd is not None and isinstance(X, pd.DataFrame):
            if not isinstance(X.index, pd.MultiIndex):
                raise ValueError("pandas input must use a (time, entity) MultiIndex")
            crosses, times, features = _split_periods(X)
            self._fit_crosses(crosses, times, features)
            return self

        values = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        result = rolling_embedding(
            values,
            self.n_components,
            backend=self.backend,
            scale=self.scale,
            seed=self.seed,
            backend_kwargs=self.backend_kwargs,
        )
        self.result_: RollingEmbeddingResult = result
        run = make_backend(self.backend, self.n_components, self.seed, self.backend_kwargs)
        self._stepper = _EmbeddingStepper(run, self.n_components, self.scale)
        self._stepper.prev = result.embeddings[-1].copy()
        self._prev_entities = None
        self.times_ = None
        self.entity_names_ = None
        self.feature_names_in_ = None
        self._variable = False
        return self

    def _fit_crosses(self, crosses, times, features) -> None:
        run = make_backend(self.backend, self.n_components, self.seed, self.backend_kwargs)
        stepper = _EmbeddingStepper(run, self.n_components, self.scale)

        union: list[str] = []
        col: dict[str, int] = {}
        for entities, _ in crosses:
            for e in entities:
                if e not in col:
                    col[e] = len(union)
                    union.append(e)

        T, U, k = len(crosses), len(union), self.n_components
        embeddings = np.full((T, U, k), np.nan)
        disparity = np.full(T, np.nan)
        displacement = np.full((T, U), np.nan)

        prev_entities: list[str] = []
        for t, (entities, values) in enumerate(crosses):
            prev_pos = {e: i for i, e in enumerate(prev_entities)}
            cur_shared = np.array(
                [i for i, e in enumerate(entities) if e in prev_pos], dtype=np.int64
            )
            prev_shared = np.array(
                [prev_pos[e] for e in entities if e in prev_pos], dtype=np.int64
            )
            aligned, disp_t, disp_vec = stepper.step(
                np.ascontiguousarray(values), cur_shared, prev_shared
            )
            cols = np.array([col[e] for e in entities])
            embeddings[t, cols] = aligned
            disparity[t] = disp_t
            if cur_shared.size:
                displacement[t, cols[cur_shared]] = disp_vec
            prev_entities = list(entities)

        self.result_ = RollingEmbeddingResult(
            embeddings=embeddings,
            disparity=disparity,
            displacement=displacement,
            n_components=k,
        )
        self._stepper = stepper
        self._prev_entities = prev_entities
        self.times_ = times
        self.entity_names_ = union
        self.feature_names_in_ = features
        self._variable = True

    # -------------------------------------------------------------- update

    def update(self, X_new: Any) -> RollingEmbedding:
        """Advance with new period(s), continuing the alignment chain."""
        self._check_fitted()
        if self._variable:
            return self._update_variable(X_new)
        values = np.asarray(X_new, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, :, :]
        if values.ndim != 3 or values.shape[1] != self.result_.embeddings.shape[1]:
            raise ValueError(
                f"expected (m, {self.result_.embeddings.shape[1]}, d) new periods; "
                f"got {values.shape}"
            )
        r = self.result_
        N = values.shape[1]
        everyone = np.arange(N)
        emb, disp, dvec = [], [], []
        for x in values:
            aligned, disp_t, disp_v = self._stepper.step(
                np.ascontiguousarray(x), everyone, everyone
            )
            emb.append(aligned)
            disp.append(disp_t)
            dvec.append(disp_v)
        self.result_ = RollingEmbeddingResult(
            embeddings=np.concatenate([r.embeddings, np.stack(emb)]),
            disparity=np.concatenate([r.disparity, disp]),
            displacement=np.concatenate([r.displacement, np.stack(dvec)]),
            n_components=r.n_components,
        )
        return self

    def _update_variable(self, X_new: Any) -> RollingEmbedding:
        if pd is None or not isinstance(X_new, pd.DataFrame):
            raise ValueError(
                "this model was fitted with entity names; update() needs a "
                "DataFrame indexed by entity (or a (time, entity) MultiIndex)"
            )
        if isinstance(X_new.index, pd.MultiIndex):
            crosses, times, features = _split_periods(X_new)
        else:
            crosses = [(list(map(str, X_new.index)), X_new.to_numpy(dtype=np.float64))]
            times, features = None, [str(c) for c in X_new.columns]
        if features != self.feature_names_in_:
            raise ValueError("feature names do not match fitted columns")

        r = self.result_
        embeddings, disparity, displacement = (
            r.embeddings, r.disparity, r.displacement,
        )
        union = list(self.entity_names_)
        col = {e: i for i, e in enumerate(union)}
        prev_entities = list(self._prev_entities)

        for entities, values in crosses:
            fresh = [e for e in entities if e not in col]
            if fresh:
                for e in fresh:
                    col[e] = len(union)
                    union.append(e)
                pad = np.full((embeddings.shape[0], len(fresh), r.n_components), np.nan)
                embeddings = np.concatenate([embeddings, pad], axis=1)
                pad_d = np.full((displacement.shape[0], len(fresh)), np.nan)
                displacement = np.concatenate([displacement, pad_d], axis=1)

            prev_pos = {e: i for i, e in enumerate(prev_entities)}
            cur_shared = np.array(
                [i for i, e in enumerate(entities) if e in prev_pos], dtype=np.int64
            )
            prev_shared = np.array(
                [prev_pos[e] for e in entities if e in prev_pos], dtype=np.int64
            )
            aligned, disp_t, disp_v = self._stepper.step(
                np.ascontiguousarray(values), cur_shared, prev_shared
            )
            row = np.full((1, len(union), r.n_components), np.nan)
            cols = np.array([col[e] for e in entities])
            row[0, cols] = aligned
            drow = np.full((1, len(union)), np.nan)
            if cur_shared.size:
                drow[0, cols[cur_shared]] = disp_v
            embeddings = np.concatenate([embeddings, row])
            disparity = np.concatenate([disparity, [disp_t]])
            displacement = np.concatenate([displacement, drow])
            prev_entities = list(entities)

        self.result_ = RollingEmbeddingResult(
            embeddings=embeddings,
            disparity=disparity,
            displacement=displacement,
            n_components=r.n_components,
        )
        self.entity_names_ = union
        self._prev_entities = prev_entities
        if times is not None and self.times_ is not None:
            self.times_ = list(self.times_) + list(times)
        else:
            self.times_ = None
        return self

    # ------------------------------------------------------------ views

    def embedding_frame(self, t: int = -1):
        """One period's map as a pandas DataFrame (entities × coordinates)."""
        self._check_fitted()
        import pandas as pd

        coords = self.result_.embeddings[t]
        names = self.entity_names_ or [f"entity_{i}" for i in range(coords.shape[0])]
        cols = [f"e{j + 1}" for j in range(self.n_components)]
        df = pd.DataFrame(coords, index=names, columns=cols)
        return df.dropna(how="all")

    def trajectory(self, entity: str | int):
        """One entity's path through the map over time (pandas DataFrame)."""
        self._check_fitted()
        import pandas as pd

        if isinstance(entity, str):
            if self.entity_names_ is None:
                raise ValueError("model was fitted without entity names")
            idx = self.entity_names_.index(entity)
        else:
            idx = int(entity)
        cols = [f"e{j + 1}" for j in range(self.n_components)]
        return pd.DataFrame(
            self.result_.embeddings[:, idx, :], index=self.times_, columns=cols
        )

    # ----------------------------------------------------------- accessors

    @property
    def embeddings_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.embeddings

    @property
    def disparity_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.disparity

    @property
    def displacement_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.displacement

    def __repr__(self) -> str:
        name = self.backend if isinstance(self.backend, str) else "custom"
        return f"RollingEmbedding(backend={name!r})"

    def _check_fitted(self) -> None:
        if not hasattr(self, "result_"):
            raise RuntimeError("RollingEmbedding is not fitted; call fit(X) first")
