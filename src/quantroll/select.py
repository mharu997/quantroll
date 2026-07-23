"""Model-selection helpers: how many regimes, how many clusters.

Both helpers fit candidate models on a *representative slice* of data you
choose (e.g. the most recent few years, or the latest cross-section) and
return a transparent pandas table of criteria per candidate — the numbers
behind the recommendation, not just a number.
"""

from __future__ import annotations

import numpy as np

from ._compat import unwrap
from .core._gmm_kernels import hmm_em_fit
from .core._kmeans_kernels import lloyd, maxmin_init
from .core.rolling_regimes import _dispersion_init

__all__ = ["select_n_regimes", "select_n_clusters"]


def select_n_regimes(
    X,
    candidates=(1, 2, 3, 4, 5),
    *,
    reg_covar: float = 1e-6,
    p_switch: float = 0.02,
    max_iter: int = 300,
    tol: float = 1e-8,
):
    """Score candidate regime counts with AIC/BIC on Gaussian HMM fits.

    Each candidate K is fit by maximum-likelihood Baum-Welch (deterministic
    dispersion-quantile initialization) on the given feature matrix — pass a
    representative stretch, e.g. the window you intend to run
    :class:`~quantroll.RollingRegimes` with.

    Returns
    -------
    pandas.DataFrame indexed by K with columns ``log_likelihood`` (total),
    ``n_params``, ``aic``, ``bic``. The usual pick is ``df["bic"].idxmin()``.

    Examples
    --------
    >>> from quantroll import select_n_regimes, simulate
    >>> R, _ = simulate.regime_returns(2000, seed=0)
    >>> table = select_n_regimes(R, candidates=(1, 2, 3))
    >>> int(table["bic"].idxmin())
    2
    """
    import pandas as pd

    values, _, _, _ = unwrap(X)
    if values.ndim != 2:
        raise ValueError(f"X must be 2-D (T, d); got shape {values.shape}")
    values = np.ascontiguousarray(values)
    n, d = values.shape

    rows = []
    for K in candidates:
        if not 1 <= K <= 32 or n < 2 * K:
            raise ValueError(f"invalid candidate n_regimes={K} for {n} observations")
        means, covs, _ = _dispersion_init(values, K, reg_covar)
        if K > 1:
            A = np.full((K, K), p_switch / (K - 1))
            np.fill_diagonal(A, 1.0 - p_switch)
        else:
            A = np.ones((1, 1))
        pi = np.full(K, 1.0 / K)
        mean_ll, _ = hmm_em_fit(
            values, means, covs, A, pi, max_iter, tol, reg_covar,
            0.0, 2.0, A.copy(), means.copy(), covs.copy(), 0.0,
        )
        ll = mean_ll * n
        n_params = (K - 1) + K * (K - 1) + K * d + K * d * (d + 1) // 2
        rows.append(
            {
                "log_likelihood": ll,
                "n_params": n_params,
                "aic": -2.0 * ll + 2.0 * n_params,
                "bic": -2.0 * ll + n_params * np.log(n),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(list(candidates), name="n_regimes"))


def select_n_clusters(
    X,
    candidates=(2, 3, 4, 5, 6, 7, 8),
    *,
    max_iter: int = 300,
    tol: float = 1e-8,
):
    """Score candidate cluster counts with silhouette and inertia.

    Runs the deterministic K-Means (farthest-point init + Lloyd) on a single
    cross-section — pass a representative one, e.g. the latest period of the
    panel you intend to run :class:`~quantroll.RollingKMeans` on.

    Returns
    -------
    pandas.DataFrame indexed by K with columns ``silhouette`` (higher is
    better) and ``inertia``. The usual pick is ``df["silhouette"].idxmax()``.

    Examples
    --------
    >>> from quantroll import select_n_clusters, simulate
    >>> X, _, _ = simulate.drifting_blobs(1, n_entities=120, seed=0)
    >>> table = select_n_clusters(X[0], candidates=(2, 3, 4, 5))
    >>> int(table["silhouette"].idxmax())
    3
    """
    import pandas as pd

    values, _, _, _ = unwrap(X)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2-D cross-section (N, d); got {values.shape}")
    values = np.ascontiguousarray(values)
    N = values.shape[0]

    diff = values[:, None, :] - values[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))

    rows = []
    for K in candidates:
        if not 2 <= K <= N - 1:
            raise ValueError(f"invalid candidate n_clusters={K} for {N} points")
        centroids = maxmin_init(values, K)
        labels, inertia, _ = lloyd(values, centroids, max_iter, tol)
        rows.append({"silhouette": _silhouette(dist, labels, K), "inertia": inertia})
    return pd.DataFrame(rows, index=pd.Index(list(candidates), name="n_clusters"))


def _silhouette(dist: np.ndarray, labels: np.ndarray, K: int) -> float:
    N = dist.shape[0]
    scores = np.zeros(N)
    sizes = np.bincount(labels, minlength=K)
    for i in range(N):
        own = labels[i]
        if sizes[own] <= 1:
            continue
        a = dist[i, labels == own].sum() / (sizes[own] - 1)
        b = np.inf
        for k in range(K):
            if k == own or sizes[k] == 0:
                continue
            b = min(b, dist[i, labels == k].mean())
        m = max(a, b)
        scores[i] = (b - a) / m if m > 0 else 0.0
    return float(scores.mean())
