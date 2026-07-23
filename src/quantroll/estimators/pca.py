"""Scikit-learn-style estimator for temporally stable rolling PCA."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._compat import unwrap, wrap_2d
from ..core.align import canonical_signs, match_step
from ..core.rolling_pca import RollingPCAResult, rolling_pca

__all__ = ["RollingPCA"]


class RollingPCA:
    """Temporally stable PCA over rolling windows.

    Classical PCA recomputed per window suffers eigenvector sign flips and
    component reordering between adjacent windows. This estimator matches each
    window's eigenbasis to the previous one (absolute cosine similarity),
    un-flips signs and preserves component identities — following the rolling
    PCA methodology of Hirsa, Klinkert, Malhotra & Holmes (2023).

    Parameters
    ----------
    n_components : int
        Number of leading components to track.
    window : int
        Rolling window length in time steps (e.g. 252 for one trading year of
        daily data).
    matching : {"hungarian", "greedy"}
        Cross-window component matching. ``"hungarian"`` is an optimal
        one-to-one assignment; ``"greedy"`` follows the paper's per-component
        argmax.
    corr : bool
        Decompose the rolling correlation matrix instead of the covariance.
    ddof : int
        Delta degrees of freedom of the covariance estimate.
    refresh_every : int
        Steps between full recomputations of the sliding sums.
    solver : {"auto", "eigh", "subspace"}
        Per-step eigensolver for batch ``fit`` (see
        :func:`quantroll.rolling_pca`). Streaming ``update`` always uses the
        exact solver.
    solver_tol : float
        Residual tolerance for accepting subspace-iteration solutions.
    halflife : float | None
        If set, exponentially weighted moments with this half-life replace
        the flat window (``window`` then only sets the warm-up length).

    Attributes
    ----------
    components_ : (T, p, d) aligned components per time step.
    eigenvalues_ : (T, p)
    explained_variance_ratio_ : (T, p)
    similarity_ : (T, p) |cos| stability diagnostic per component.
    n_flips_, n_reorders_ : (T,) correction counts per step.
    feature_names_in_ : list[str] | None
    result_ : RollingPCAResult from the last full fit.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from quantroll import RollingPCA
    >>> rng = np.random.default_rng(0)
    >>> rets = pd.DataFrame(rng.standard_normal((400, 6)) / 100,
    ...                     columns=list("ABCDEF"))
    >>> pca = RollingPCA(n_components=2, window=120).fit(rets)
    >>> Z = pca.transform()          # DataFrame (400, 2), NaN warm-up
    >>> pca.update(rets.iloc[-1:])   # stream one more observation
    RollingPCA(n_components=2, window=120)
    """

    def __init__(
        self,
        n_components: int = 3,
        window: int = 252,
        *,
        matching: str = "hungarian",
        corr: bool = False,
        ddof: int = 1,
        refresh_every: int = 1024,
        solver: str = "auto",
        solver_tol: float = 1e-8,
        halflife: float | None = None,
    ) -> None:
        self.n_components = n_components
        self.window = window
        self.matching = matching
        self.corr = corr
        self.ddof = ddof
        self.refresh_every = refresh_every
        self.solver = solver
        self.solver_tol = solver_tol
        self.halflife = halflife

    # ------------------------------------------------------------------ fit

    def fit(self, X: Any, y: Any = None) -> RollingPCA:
        """Run the rolling decomposition over the full history ``X``.

        ``X`` may be a numpy array (T, d) or (F, T, d), a pandas DataFrame,
        or a polars DataFrame (rows = time).
        """
        values, index, columns, kind = unwrap(X)
        res = rolling_pca(
            values,
            window=self.window,
            n_components=self.n_components,
            matching=self.matching,
            corr=self.corr,
            ddof=self.ddof,
            refresh_every=self.refresh_every,
            solver=self.solver,
            solver_tol=self.solver_tol,
            halflife=self.halflife,
        )
        self.result_: RollingPCAResult = res
        self._kind = kind
        self._index = index
        self.feature_names_in_ = columns
        self._panel = values.ndim == 3
        self._projections = res.projections

        if values.ndim == 2 and not np.isnan(values).any():
            self._init_stream_state(values)
        else:
            self._stream = None
        return self

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X).transform()

    # ------------------------------------------------------------ transform

    def transform(self, X: Any = None) -> Any:
        """Return in-sample projections, or project new rows on the latest basis.

        With no argument, returns the stored (T, p) projections from ``fit``
        (labeled if the input was labeled). With ``X``, rows are centered with
        the latest rolling mean and projected onto the latest aligned basis —
        a static out-of-sample projection that does not advance the model.
        """
        self._check_fitted()
        names = [f"pc{k + 1}" for k in range(self.n_components)]
        if X is None:
            if self._panel:
                return self._projections
            return wrap_2d(self._projections, self._index, names, self._kind)
        values, index, _, kind = unwrap(X)
        if values.ndim != 2:
            raise ValueError("transform(X) expects 2-D input (T, d)")
        V, mean = self._latest_basis()
        Xc = np.where(np.isfinite(mean), values - mean, 0.0)
        proj = Xc @ np.nan_to_num(V, nan=0.0)
        return wrap_2d(proj, index, names, kind)

    # ------------------------------------------------------------- update

    def update(self, X_new: Any) -> RollingPCA:
        """Advance the model with new observation(s) — O(d²) per row.

        Streaming continuation of ``fit``: sliding sums are updated, one
        eigendecomposition is computed per row, and the basis is aligned to
        the previous step. Results are appended to the fitted attributes so
        ``transform()`` covers the full extended history.
        """
        self._check_fitted()
        if self._stream is None:
            raise NotImplementedError(
                "update() supports dense 2-D models only (no NaN universe gaps, no panels)"
            )
        values, index, columns, _ = unwrap(X_new)
        if values.ndim == 1:
            values = values[None, :]
        d = self._stream["s1"].shape[0]
        if values.shape[1] != d:
            raise ValueError(f"expected {d} features, got {values.shape[1]}")
        if columns is not None and self.feature_names_in_ is not None:
            if columns != self.feature_names_in_:
                raise ValueError("feature names do not match fitted columns")

        for row in values:
            self._update_one(row)
        if index is not None and self._index is not None:
            self._index = self._index.append(index)
        elif self._index is not None:
            self._index = None
        return self

    # ----------------------------------------------------------- accessors

    @property
    def components_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.components

    @property
    def eigenvalues_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.eigenvalues

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.explained_variance_ratio

    @property
    def similarity_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.similarity

    @property
    def n_flips_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.n_flips

    @property
    def n_reorders_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.n_reorders

    def __repr__(self) -> str:
        return f"RollingPCA(n_components={self.n_components}, window={self.window})"

    # ------------------------------------------------------------ internals

    def _check_fitted(self) -> None:
        if not hasattr(self, "result_"):
            raise RuntimeError("RollingPCA is not fitted; call fit(X) first")

    def _latest_basis(self) -> tuple[np.ndarray, np.ndarray]:
        vals = self.result_.eigenvalues[:, 0]
        valid = np.flatnonzero(~np.isnan(vals))
        if valid.size == 0:
            raise RuntimeError("no valid window was produced during fit")
        t = int(valid[-1])
        V = self.result_.components[t].T
        mean = self.result_.means[t]
        return V, mean

    def _init_stream_state(self, values: np.ndarray) -> None:
        if self.halflife is not None:
            from ..core._kernels import ewma_sums

            lam = 0.5 ** (1.0 / self.halflife)
            s1, s2, wsum = ewma_sums(values, lam)
            self._stream = {"mode": "ewma", "lam": lam, "s1": s1, "s2": s2, "wsum": wsum}
            return
        W = self.window
        buffer = values[-W:].copy()
        self._stream = {
            "buffer": buffer,
            "pos": 0,  # ring position of the oldest row
            "s1": buffer.sum(axis=0),
            "s2": buffer.T @ buffer,
            "since_refresh": 0,
        }

    def _update_one(self, x: np.ndarray) -> None:
        st = self._stream
        x = np.asarray(x, dtype=np.float64)
        if st.get("mode") == "ewma":
            lam = st["lam"]
            st["s1"] = lam * st["s1"] + x
            st["s2"] = lam * st["s2"] + np.outer(x, x)
            st["wsum"] = lam * st["wsum"] + 1.0
            mean = st["s1"] / st["wsum"]
            cov = st["s2"] / st["wsum"] - np.outer(mean, mean)
        else:
            W = self.window
            x_old = st["buffer"][st["pos"]].copy()
            st["buffer"][st["pos"]] = x
            st["pos"] = (st["pos"] + 1) % W
            st["s1"] += x - x_old
            st["s2"] += np.outer(x, x) - np.outer(x_old, x_old)
            st["since_refresh"] += 1
            if st["since_refresh"] >= self.refresh_every:
                st["s1"] = st["buffer"].sum(axis=0)
                st["s2"] = st["buffer"].T @ st["buffer"]
                st["since_refresh"] = 0

            w = float(W)
            mean = st["s1"] / w
            cov = (st["s2"] - np.outer(st["s1"], st["s1"]) / w) / (w - self.ddof)
        if self.corr:
            sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            denom = np.outer(sd, sd)
            with np.errstate(invalid="ignore", divide="ignore"):
                cov = np.where(denom > 0.0, cov / denom, 0.0)
            np.fill_diagonal(cov, np.where(sd > 0.0, 1.0, 0.0))
        cov = 0.5 * (cov + cov.T)
        total_var = float(np.trace(cov))

        vals, vecs = np.linalg.eigh(cov)
        vals = vals[::-1][: self.n_components]
        V = vecs[:, ::-1][:, : self.n_components]

        r = self.result_
        prev_valid = np.flatnonzero(~np.isnan(r.eigenvalues[:, 0]))
        if prev_valid.size:
            prev = r.components[int(prev_valid[-1])].T
            perm, signs, sim = match_step(prev, V, method=self.matching)
            V = V[:, perm] * signs
            vals = vals[perm]
            flips = int((signs < 0).sum())
            reorders = int((perm != np.arange(self.n_components)).sum())
        else:
            V = canonical_signs(V)
            sim = np.full(self.n_components, np.nan)
            flips = reorders = 0

        proj = (x - mean) @ V
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = vals / total_var if total_var else np.full_like(vals, np.nan)

        self.result_ = RollingPCAResult(
            components=np.concatenate([r.components, V.T[None]], axis=0),
            eigenvalues=np.concatenate([r.eigenvalues, vals[None]], axis=0),
            explained_variance_ratio=np.concatenate(
                [r.explained_variance_ratio, ratio[None]], axis=0
            ),
            similarity=np.concatenate([r.similarity, sim[None]], axis=0),
            projections=np.concatenate([r.projections, proj[None]], axis=0),
            means=np.concatenate([r.means, mean[None]], axis=0),
            total_variance=np.concatenate([r.total_variance, [total_var]], axis=0),
            n_flips=np.concatenate([r.n_flips, [flips]], axis=0),
            n_reorders=np.concatenate([r.n_reorders, [reorders]], axis=0),
            window=r.window,
            n_components=r.n_components,
        )
        self._projections = self.result_.projections
