"""Scikit-learn-style estimator for temporally stable rolling K-Means."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.rolling_kmeans import (
    RollingKMeansResult,
    _KMeansStepper,
    _rolling_kmeans_impl,
)
from ._panel_utils import coerce_cross_section as _coerce_cross_section
from ._panel_utils import coerce_panel as _coerce_panel
from ._panel_utils import pd
from ._panel_utils import split_periods as _split_periods

__all__ = ["RollingKMeans"]


class RollingKMeans:
    """Stable clustering of an entity universe over time.

    Chains K-Means through time with warm starts and Hungarian-matched
    cluster identities (plus optional multi-centroid memory for nonlinear
    boundaries), so cluster labels are comparable across periods: unchanged
    data gives unchanged clusters, and "cluster 2" keeps meaning the same
    group of funds tomorrow that it does today.

    Parameters
    ----------
    n_clusters : int
        Number of clusters K.
    centroid_memory : int
        Recent centroids remembered per cluster for assignment (1 = classical
        linear boundaries; larger = piecewise-linear boundaries following
        each cluster's trajectory).
    max_iter, warm_iter, tol, init, seed, n_init
        See :func:`quantroll.rolling_kmeans`.

    Attributes
    ----------
    labels_ : (T, N) int stable cluster of each entity at each period.
    centroids_ : (T, K, d) matched centroids.
    inertia_, n_relabels_, switched_, n_iter_ : per-step diagnostics.
    entity_names_, feature_names_in_, times_ : labels when fitted from pandas.
    result_ : RollingKMeansResult from the last fit/update.

    Examples
    --------
    >>> from quantroll import RollingKMeans, simulate
    >>> X, membership, _ = simulate.drifting_blobs(100, seed=0)
    >>> km = RollingKMeans(n_clusters=3).fit(X)
    >>> km.labels_.shape
    (100, 90)
    >>> km.update(X[-1])                       # stream one more period
    RollingKMeans(n_clusters=3)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        *,
        centroid_memory: int = 1,
        max_iter: int = 300,
        warm_iter: int = 100,
        tol: float = 1e-8,
        init: str = "maxmin",
        seed: int | None = None,
        n_init: int = 10,
    ) -> None:
        self.n_clusters = n_clusters
        self.centroid_memory = centroid_memory
        self.max_iter = max_iter
        self.warm_iter = warm_iter
        self.tol = tol
        self.init = init
        self.seed = seed
        self.n_init = n_init

    # ------------------------------------------------------------------ fit

    def fit(self, X: Any, y: Any = None) -> RollingKMeans:
        """Fit over a panel of cross-sections.

        Accepts a (T, N, d) numpy array (fixed entity universe), or a pandas
        DataFrame with a (time, entity) MultiIndex and feature columns — the
        panel may be *unbalanced*: entities can appear and disappear over
        time, and absent entities carry label ``-1`` in the outputs.
        """
        if self.centroid_memory < 1:
            raise ValueError(f"centroid_memory must be >= 1; got {self.centroid_memory}")
        if pd is not None and isinstance(X, pd.DataFrame):
            if not isinstance(X.index, pd.MultiIndex):
                raise ValueError(
                    "pandas input must use a (time, entity) MultiIndex; "
                    "for a single period pass it to update()/predict()"
                )
            crosses, times, features = _split_periods(X)
            self._fit_crosses(crosses, times, features)
            return self

        values, _, _, _ = _coerce_panel(X)
        if not 1 <= self.n_clusters <= values.shape[1]:
            raise ValueError(
                f"n_clusters must be in [1, {values.shape[1]}]; got {self.n_clusters}"
            )
        result, stepper = _rolling_kmeans_impl(
            values,
            self.n_clusters,
            centroid_memory=self.centroid_memory,
            max_iter=self.max_iter,
            warm_iter=self.warm_iter,
            tol=self.tol,
            init=self.init,
            seed=self.seed,
            n_init=self.n_init,
        )
        self.result_: RollingKMeansResult = result
        self._stepper = stepper
        self.times_ = None
        self.entity_names_ = None
        self.feature_names_in_ = None
        self._variable = False
        return self

    def _fit_crosses(self, crosses, times, features) -> None:
        union: list[str] = []
        col: dict[str, int] = {}
        for entities, _ in crosses:
            for e in entities:
                if e not in col:
                    col[e] = len(union)
                    union.append(e)
        for i, (entities, values) in enumerate(crosses):
            if values.shape[0] < self.n_clusters:
                raise ValueError(
                    f"period {times[i]!r} has {values.shape[0]} entities, "
                    f"fewer than n_clusters={self.n_clusters}"
                )

        stepper = _KMeansStepper(
            self.n_clusters,
            centroid_memory=self.centroid_memory,
            max_iter=self.max_iter,
            warm_iter=self.warm_iter,
            tol=self.tol,
            init=self.init,
            seed=self.seed,
            n_init=self.n_init,
        )
        T, U, K = len(crosses), len(union), self.n_clusters
        d = crosses[0][1].shape[1]
        labels = np.full((T, U), -1, dtype=np.int64)
        centroids = np.empty((T, K, d))
        inertia = np.empty(T)
        n_relabels = np.zeros(T, dtype=np.int64)
        switched = np.zeros((T, U), dtype=bool)
        n_iter = np.zeros(T, dtype=np.int64)

        for t, (entities, values) in enumerate(crosses):
            out = stepper.step(np.ascontiguousarray(values))
            cols = np.array([col[e] for e in entities])
            labels[t, cols] = out["labels"]
            centroids[t] = out["centroids"]
            inertia[t] = out["inertia"]
            n_relabels[t] = out["n_relabels"]
            n_iter[t] = out["n_iter"]
            if t > 0:
                shared = (labels[t] >= 0) & (labels[t - 1] >= 0)
                switched[t, shared] = labels[t, shared] != labels[t - 1, shared]

        self.result_ = RollingKMeansResult(
            labels=labels,
            centroids=centroids,
            inertia=inertia,
            n_relabels=n_relabels,
            switched=switched,
            n_iter=n_iter,
            n_clusters=K,
            centroid_memory=self.centroid_memory,
        )
        self._stepper = stepper
        self.times_ = times
        self.entity_names_ = union
        self.feature_names_in_ = features
        self._variable = True

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X).labels_

    # ------------------------------------------------------------- predict

    def predict(self, X: Any = None) -> Any:
        """Cluster labels. No argument: the fitted (T, N) history. With a
        cross-section ``X`` (N', d): static assignment under the current
        remembered centroids (the model does not advance)."""
        self._check_fitted()
        if X is None:
            return self.labels_
        values, index = _coerce_cross_section(X)
        from ..core._kmeans_kernels import assign_multi

        labels, _ = assign_multi(
            np.ascontiguousarray(values), self._stepper.bank, self._stepper.bank_n
        )
        if index is not None and pd is not None:
            return pd.Series(labels, index=index, name="cluster")
        return labels

    def transform(self, X: Any) -> np.ndarray:
        """(N', K) distance of each row to each cluster's nearest remembered
        centroid, under the current state."""
        self._check_fitted()
        values, _ = _coerce_cross_section(X)
        st = self._stepper
        out = np.empty((values.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            bank = st.bank[k, : st.bank_n[k]]
            diff = values[:, None, :] - bank[None, :, :]
            out[:, k] = np.sqrt((diff**2).sum(axis=2).min(axis=1))
        return out

    # -------------------------------------------------------------- update

    def update(self, X_new: Any) -> RollingKMeans:
        """Advance with new period(s) — identical arithmetic to batch fit.

        Fixed-universe models (fitted from numpy) take (N, d) or (m, N, d)
        arrays. Variable-universe models (fitted from pandas) take a
        DataFrame: either one cross-section indexed by entity, or a
        (time, entity) MultiIndex chunk; new entities extend the universe.
        """
        self._check_fitted()
        if getattr(self, "_variable", False):
            return self._update_variable(X_new)

        values, _, _, _ = _coerce_panel(X_new, allow_2d=True)
        N = self.result_.labels.shape[1]
        if values.shape[1] != N:
            raise ValueError(f"expected {N} entities, got {values.shape[1]}")

        r = self.result_
        prev_labels = r.labels[-1]
        rows = {k: [] for k in ("labels", "centroids", "inertia", "n_relabels",
                                "switched", "n_iter")}
        for x in values:
            out = self._stepper.step(np.ascontiguousarray(x))
            rows["labels"].append(out["labels"])
            rows["centroids"].append(out["centroids"])
            rows["inertia"].append(out["inertia"])
            rows["n_relabels"].append(out["n_relabels"])
            rows["switched"].append(out["labels"] != prev_labels)
            rows["n_iter"].append(out["n_iter"])
            prev_labels = out["labels"]
        self.result_ = RollingKMeansResult(
            labels=np.concatenate([r.labels, np.stack(rows["labels"])]),
            centroids=np.concatenate([r.centroids, np.stack(rows["centroids"])]),
            inertia=np.concatenate([r.inertia, rows["inertia"]]),
            n_relabels=np.concatenate(
                [r.n_relabels, np.asarray(rows["n_relabels"], dtype=np.int64)]
            ),
            switched=np.concatenate([r.switched, np.stack(rows["switched"])]),
            n_iter=np.concatenate([r.n_iter, np.asarray(rows["n_iter"], dtype=np.int64)]),
            n_clusters=r.n_clusters,
            centroid_memory=r.centroid_memory,
        )
        return self

    def _update_variable(self, X_new: Any) -> RollingKMeans:
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
        labels = r.labels
        switched = r.switched
        union = list(self.entity_names_)
        col = {e: i for i, e in enumerate(union)}

        for i, (entities, values) in enumerate(crosses):
            if values.shape[0] < self.n_clusters:
                raise ValueError(
                    f"update period has {values.shape[0]} entities, fewer than "
                    f"n_clusters={self.n_clusters}"
                )
            fresh = [e for e in entities if e not in col]
            if fresh:
                for e in fresh:
                    col[e] = len(union)
                    union.append(e)
                pad_l = np.full((labels.shape[0], len(fresh)), -1, dtype=np.int64)
                labels = np.concatenate([labels, pad_l], axis=1)
                pad_s = np.zeros((switched.shape[0], len(fresh)), dtype=bool)
                switched = np.concatenate([switched, pad_s], axis=1)

            out = self._stepper.step(np.ascontiguousarray(values))
            row = np.full(len(union), -1, dtype=np.int64)
            row[[col[e] for e in entities]] = out["labels"]
            sw = np.zeros(len(union), dtype=bool)
            prev_row = labels[-1]
            shared = (row >= 0) & (prev_row >= 0)
            sw[shared] = row[shared] != prev_row[shared]

            labels = np.concatenate([labels, row[None]], axis=0)
            switched = np.concatenate([switched, sw[None]], axis=0)
            r = RollingKMeansResult(
                labels=labels,
                centroids=np.concatenate([r.centroids, out["centroids"][None]]),
                inertia=np.concatenate([r.inertia, [out["inertia"]]]),
                n_relabels=np.concatenate([r.n_relabels, [out["n_relabels"]]]),
                switched=switched,
                n_iter=np.concatenate([r.n_iter, [out["n_iter"]]]),
                n_clusters=r.n_clusters,
                centroid_memory=r.centroid_memory,
            )
        self.result_ = r
        self.entity_names_ = union
        if times is not None and self.times_ is not None:
            self.times_ = list(self.times_) + list(times)
        else:
            self.times_ = None
        return self

    # ------------------------------------------------------------ summary

    def labels_frame(self):
        """Labels as a pandas DataFrame (time × entity)."""
        self._check_fitted()
        import pandas as pd

        return pd.DataFrame(
            self.labels_, index=self.times_, columns=self.entity_names_
        )

    def cluster_summary(self):
        """Per-cluster table: average entity share, per-period switch rate,
        and the final centroid's feature values."""
        self._check_fitted()
        import pandas as pd

        labels = self.labels_
        T, N = labels.shape
        feats = self.feature_names_in_ or [
            f"f{j}" for j in range(self.centroids_.shape[2])
        ]
        n_present = max(int((labels >= 0).sum()), 1)
        rows = []
        for k in range(self.n_clusters):
            member = labels == k
            share = member.sum() / n_present
            entered = (member[1:] & ~member[:-1]).sum() if T > 1 else 0
            row = {
                "share": share,
                "n_final": int(member[-1].sum()),
                "moves_in": int(entered),
            }
            for j, f in enumerate(feats):
                row[f"centroid_{f}"] = self.centroids_[-1, k, j]
            rows.append(row)
        return pd.DataFrame(rows, index=[f"cluster_{k}" for k in range(self.n_clusters)])

    # ----------------------------------------------------------- accessors

    @property
    def labels_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.labels

    @property
    def centroids_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.centroids

    @property
    def inertia_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.inertia

    @property
    def n_relabels_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.n_relabels

    @property
    def switched_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.switched

    @property
    def n_iter_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.n_iter

    def __repr__(self) -> str:
        return f"RollingKMeans(n_clusters={self.n_clusters})"

    def _check_fitted(self) -> None:
        if not hasattr(self, "result_"):
            raise RuntimeError("RollingKMeans is not fitted; call fit(X) first")


