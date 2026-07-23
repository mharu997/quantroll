"""Functional core: temporally stable rolling-window PCA on arrays.

This is the array-in / array-out layer. For labeled data (pandas) and a
scikit-learn-style stateful interface, use
:class:`quantroll.estimators.RollingPCA`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._kernels import (
    eig_from_sums,
    rebuild_sums,
    rolling_eig_panel,
    rolling_eig_panel_ewma,
    slide_sums,
)
from .align import align_sequence, align_sequence_masked

__all__ = ["rolling_pca", "RollingPCAResult"]


@dataclass(frozen=True)
class RollingPCAResult:
    """Output of :func:`rolling_pca`. All time-indexed arrays are full length
    ``T`` with NaN during the warm-up (``t < window - 1``)."""

    components: np.ndarray
    """(T, p, d) aligned components; ``components[t, k]`` is component k at t."""
    eigenvalues: np.ndarray
    """(T, p) eigenvalues in stable component order."""
    explained_variance_ratio: np.ndarray
    """(T, p) eigenvalue share of total variance at each step."""
    similarity: np.ndarray
    """(T, p) |cos| of each component vs. its predecessor (stability diagnostic)."""
    projections: np.ndarray
    """(T, p) — or (F, T, p) for panel input — window-mean-centered projections."""
    means: np.ndarray
    """(T, d) — or (F, T, d) for panel input — rolling window means."""
    total_variance: np.ndarray
    """(T,) trace of the decomposed matrix."""
    n_flips: np.ndarray
    """(T,) sign flips corrected at each step."""
    n_reorders: np.ndarray
    """(T,) component order changes at each step."""
    window: int = field(default=0)
    n_components: int = field(default=0)


def _as_panel(X: np.ndarray) -> tuple[np.ndarray, bool]:
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim == 2:
        return X[None, :, :], True
    if X.ndim == 3:
        return X, False
    raise ValueError(f"X must be 2-D (T, d) or 3-D (F, T, d); got shape {X.shape}")


def rolling_pca(
    X: np.ndarray,
    window: int,
    n_components: int = 3,
    *,
    matching: str = "hungarian",
    corr: bool = False,
    ddof: int = 1,
    refresh_every: int = 1024,
    align: bool = True,
    solver: str = "auto",
    solver_tol: float = 1e-8,
    halflife: float | None = None,
) -> RollingPCAResult:
    """Temporally stable PCA over rolling windows of a time series.

    Parameters
    ----------
    X : array, shape (T, d) or (F, T, d)
        Observations over time. For 3-D panel input the covariance is computed
        per item f and averaged across items at each step. For 2-D input, NaN
        marks a feature (asset) absent at that time: each step then uses only
        the columns fully observed within its window, and alignment runs on
        the coordinates shared between consecutive universes (variable
        feature spaces; the exact eigensolver is always used on this path).
    window : int
        Rolling window length W. The first valid output is at ``t = W - 1``.
    n_components : int
        Number of leading components p to track.
    matching : {"hungarian", "greedy"}
        Component matching mode across windows (see :mod:`quantroll.core.align`).
    corr : bool
        Decompose the correlation matrix instead of the covariance matrix.
    ddof : int
        Delta degrees of freedom for the covariance estimate.
    refresh_every : int
        Steps between full recomputations of the sliding sums (numerical hygiene).
    align : bool
        If False, skip temporal alignment and return raw eigendecompositions
        (useful to see the instability being corrected).
    solver : {"auto", "eigh", "subspace"}
        Per-step eigensolver. ``"eigh"`` is the exact full decomposition.
        ``"subspace"`` runs warm-started subspace iteration on the previous
        window's basis — O(d²p) instead of O(d³) — and falls back to exact
        ``eigh`` at any step failing a residual check, so accuracy is
        controlled by ``solver_tol``. ``"auto"`` picks subspace when
        ``n_components <= d / 4``.
    solver_tol : float
        Relative residual tolerance ``||C·V − V·λ||_F <= tol·||λ||₂`` for
        accepting a subspace-iteration solution.
    halflife : float | None
        If set, use *exponentially weighted* moments with this half-life (in
        steps) instead of a flat window — the RiskMetrics-style estimator
        that adapts smoothly and never hard-drops observations. ``window``
        then only sets the warm-up (first valid output at ``window − 1``);
        ``ddof`` and ``refresh_every`` are ignored (normalized-weight
        population covariance, recursive so nothing to refresh). Not
        supported together with NaN universe gaps.

    Examples
    --------
    >>> import numpy as np
    >>> from quantroll import rolling_pca
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 8))
    >>> res = rolling_pca(X, window=60, n_components=3)
    >>> res.projections.shape
    (500, 3)
    """
    P, was_2d = _as_panel(X)
    F, T, d = P.shape
    if not 1 <= n_components <= d:
        raise ValueError(f"n_components must be in [1, {d}]; got {n_components}")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    if window > T:
        raise ValueError(f"window ({window}) exceeds series length ({T})")
    if ddof >= window:
        raise ValueError(f"ddof ({ddof}) must be < window ({window})")
    if solver not in ("auto", "eigh", "subspace"):
        raise ValueError(f"solver must be 'auto', 'eigh' or 'subspace'; got {solver!r}")

    if np.isnan(P).any():
        if not was_2d:
            raise ValueError("NaN (variable universes) is supported for 2-D input only")
        if halflife is not None:
            raise NotImplementedError(
                "halflife (EWMA weighting) is not supported with NaN universe gaps"
            )
        return _rolling_pca_masked(
            P[0],
            window,
            n_components,
            matching=matching,
            corr=corr,
            ddof=ddof,
            refresh_every=int(refresh_every),
            align=align,
        )
    use_subspace = solver == "subspace" or (solver == "auto" and n_components * 4 <= d)

    if halflife is not None:
        if not halflife > 0:
            raise ValueError(f"halflife must be > 0; got {halflife}")
        lam = 0.5 ** (1.0 / halflife)
        eigvals, eigvecs, means, total_var = rolling_eig_panel_ewma(
            P, lam, window, n_components, corr,
            use_subspace, 2, float(solver_tol),
        )
    else:
        eigvals, eigvecs, means, total_var = rolling_eig_panel(
            P, window, n_components, ddof, corr, int(refresh_every),
            use_subspace, 2, float(solver_tol),
        )

    if align:
        aligned = align_sequence(eigvals, eigvecs, method=matching)
        eigvals, eigvecs = aligned.eigvals, aligned.eigvecs
        similarity, n_flips, n_reorders = (
            aligned.similarity,
            aligned.n_flips,
            aligned.n_reorders,
        )
    else:
        similarity = np.full_like(eigvals, np.nan)
        n_flips = np.zeros(T, dtype=np.int64)
        n_reorders = np.zeros(T, dtype=np.int64)

    projections = np.einsum("ftd,tdk->ftk", P - means, eigvecs)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = eigvals / total_var[:, None]

    if was_2d:
        projections = projections[0]
        means = means[0]

    return RollingPCAResult(
        components=np.ascontiguousarray(eigvecs.swapaxes(1, 2)),
        eigenvalues=eigvals,
        explained_variance_ratio=ratio,
        similarity=similarity,
        projections=projections,
        means=means,
        total_variance=total_var,
        n_flips=n_flips,
        n_reorders=n_reorders,
        window=window,
        n_components=n_components,
    )


def _rolling_pca_masked(
    X: np.ndarray,
    window: int,
    n_components: int,
    *,
    matching: str,
    corr: bool,
    ddof: int,
    refresh_every: int,
    align: bool,
) -> RollingPCAResult:
    """Rolling PCA over a changing feature universe (NaN = absent).

    A feature is *active* at step t when it is fully observed over the
    window ending at t. Sliding sums run while the active set is unchanged
    and are rebuilt from the window slice when it changes (or on the
    periodic refresh). Components carry NaN on inactive coordinates.
    """
    T, d = X.shape
    p = n_components
    finite = np.isfinite(X)
    csum = np.zeros((T + 1, d))
    np.cumsum(finite, axis=0, out=csum[1:])

    eigvals = np.full((T, p), np.nan)
    eigvecs = np.full((T, d, p), np.nan)
    means = np.full((T, d), np.nan)
    total_var = np.full(T, np.nan)

    prev_active: np.ndarray | None = None
    s1 = s2 = None
    since_refresh = 0
    for t in range(window - 1, T):
        cnt = csum[t + 1] - csum[t + 1 - window]
        A = np.flatnonzero(cnt == window)
        if A.size < p:
            prev_active = None
            continue
        if (
            prev_active is None
            or A.size != prev_active.size
            or not np.array_equal(A, prev_active)
            or since_refresh >= refresh_every
        ):
            w_slice = np.ascontiguousarray(X[t - window + 1 : t + 1][:, A])
            s1, s2 = rebuild_sums(w_slice)
            since_refresh = 0
        else:
            slide_sums(
                s1,
                s2,
                np.ascontiguousarray(X[t, A]),
                np.ascontiguousarray(X[t - window, A]),
            )
            since_refresh += 1
        prev_active = A

        vals, vecs, mean, tv = eig_from_sums(s1, s2, window, ddof, corr, p)
        eigvals[t] = vals
        eigvecs[t, A, :] = vecs
        means[t, A] = mean
        total_var[t] = tv

    if align:
        aligned = align_sequence_masked(eigvals, eigvecs, method=matching)
        eigvals, eigvecs = aligned.eigvals, aligned.eigvecs
        similarity, n_flips, n_reorders = (
            aligned.similarity,
            aligned.n_flips,
            aligned.n_reorders,
        )
    else:
        similarity = np.full_like(eigvals, np.nan)
        n_flips = np.zeros(T, dtype=np.int64)
        n_reorders = np.zeros(T, dtype=np.int64)

    Xc = np.where(np.isfinite(means), X - means, 0.0)
    V0 = np.nan_to_num(eigvecs, nan=0.0)
    projections = np.einsum("td,tdk->tk", Xc, V0)
    projections[np.isnan(eigvals[:, 0])] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = eigvals / total_var[:, None]

    return RollingPCAResult(
        components=np.ascontiguousarray(eigvecs.swapaxes(1, 2)),
        eigenvalues=eigvals,
        explained_variance_ratio=ratio,
        similarity=similarity,
        projections=projections,
        means=means,
        total_variance=total_var,
        n_flips=n_flips,
        n_reorders=n_reorders,
        window=window,
        n_components=n_components,
    )
