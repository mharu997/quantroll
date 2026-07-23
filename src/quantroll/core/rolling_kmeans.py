"""Temporally stable K-Means over rolling cross-sections of entities.

Clustering funds or assets per period with off-the-shelf K-Means is unstable
in two ways that have nothing to do with the data: random initialization
lands on different local optima each period, and cluster *indices* permute
arbitrarily between refits. Following the rolling K-Means methodology of
Hirsa, Holmes, Klinkert & Malhotra (2024), independently implemented, this
module chains the clustering through time:

- **warm starts** — each period's Lloyd iterations start from the previous
  period's centroids, so unchanged data yields unchanged clusters and slowly
  drifting data yields slowly drifting centroids;
- **matched identities** — new centroids are matched to the previous ones by
  Hungarian assignment on centroid distances, so "cluster 2" keeps meaning
  the same group through time;
- **centroid memory** — optionally, each cluster remembers its last
  ``centroid_memory`` centroids and claims any point nearest to one of them,
  producing piecewise-linear (nonlinear) decision boundaries that follow the
  cluster's trajectory;
- **deterministic everything** — farthest-point initialization and
  deterministic empty-cluster re-seeding; identical inputs give identical
  outputs, run to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._kmeans_kernels import assign_multi, lloyd, maxmin_init

__all__ = ["rolling_kmeans", "RollingKMeansResult", "match_centroids"]


@dataclass(frozen=True)
class RollingKMeansResult:
    """Output of :func:`rolling_kmeans` over a (T, N, d) panel."""

    labels: np.ndarray
    """(T, N) int cluster of each entity at each step (stable identities)."""
    centroids: np.ndarray
    """(T, K, d) matched centroids over time."""
    inertia: np.ndarray
    """(T,) sum of squared distances to the winning (remembered) centroid."""
    n_relabels: np.ndarray
    """(T,) clusters whose identity assignment changed at each step."""
    switched: np.ndarray
    """(T, N) bool — True where an entity's cluster changed from t-1."""
    n_iter: np.ndarray
    """(T,) Lloyd iterations used (warm starts keep this tiny)."""
    n_clusters: int = 0
    centroid_memory: int = 1


def match_centroids(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Match current centroids to previous identities.

    Returns ``perm`` with ``perm[k]`` = index of the current centroid
    assigned to previous identity ``k``, minimizing total squared distance
    via the Hungarian algorithm.
    """
    diff = prev[:, None, :] - cur[None, :, :]
    D = np.einsum("abd,abd->ab", diff, diff)
    _, perm = linear_sum_assignment(D)
    return perm


class _KMeansStepper:
    """Single-step engine shared verbatim by batch fit and streaming update,
    so both paths produce identical results by construction."""

    def __init__(
        self,
        n_clusters: int,
        *,
        centroid_memory: int,
        max_iter: int,
        warm_iter: int,
        tol: float,
        init: str,
        seed: int | None,
        n_init: int,
    ) -> None:
        self.K = n_clusters
        self.M = centroid_memory
        self.max_iter = max_iter
        self.warm_iter = warm_iter
        self.tol = tol
        self.init = init
        self.seed = seed
        self.n_init = n_init
        self.step_no = -1
        self.centroids: np.ndarray | None = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        if self.init == "maxmin":
            centroids = maxmin_init(X, self.K)
            _, _, _ = lloyd(X, centroids, self.max_iter, self.tol)
            return centroids
        if self.init == "k-means++":
            rng = np.random.default_rng(self.seed)
            best_c = None
            best_inertia = np.inf
            for _ in range(max(self.n_init, 1)):
                centroids = _kmeanspp(X, self.K, rng)
                _, inertia, _ = lloyd(X, centroids, self.max_iter, self.tol)
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_c = centroids
            return best_c
        raise ValueError(f"init must be 'maxmin' or 'k-means++'; got {self.init!r}")

    def step(self, X: np.ndarray) -> dict:
        self.step_no += 1
        n_relabels = 0
        if self.step_no == 0:
            centroids = self._init_centroids(X)
            order = np.lexsort(centroids.T[::-1])  # canonical: sort by features
            self.centroids = np.ascontiguousarray(centroids[order])
            d = X.shape[1]
            self.bank = np.zeros((self.K, self.M, d))
            self.bank_n = np.zeros(self.K, dtype=np.int64)
            self.bank_pos = np.zeros(self.K, dtype=np.int64)
            n_iter = 0
        else:
            prev = self.centroids.copy()
            centroids = prev.copy()
            _, _, n_iter = lloyd(X, centroids, self.warm_iter, self.tol)
            perm = match_centroids(prev, centroids)
            if not np.array_equal(perm, np.arange(self.K)):
                centroids = np.ascontiguousarray(centroids[perm])
                n_relabels = int((perm != np.arange(self.K)).sum())
            self.centroids = centroids

        for k in range(self.K):
            self.bank[k, self.bank_pos[k]] = self.centroids[k]
            self.bank_pos[k] = (self.bank_pos[k] + 1) % self.M
            self.bank_n[k] = min(self.bank_n[k] + 1, self.M)

        labels, dist2 = assign_multi(X, self.bank, self.bank_n)

        return {
            "labels": labels,
            "centroids": self.centroids.copy(),
            "inertia": float(dist2.sum()),
            "n_relabels": n_relabels,
            "n_iter": int(n_iter),
        }


def _kmeanspp(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    N, d = X.shape
    centroids = np.empty((K, d))
    centroids[0] = X[rng.integers(N)]
    dist2 = ((X - centroids[0]) ** 2).sum(axis=1)
    for k in range(1, K):
        total = dist2.sum()
        if total <= 0:
            centroids[k] = X[rng.integers(N)]
            continue
        idx = rng.choice(N, p=dist2 / total)
        centroids[k] = X[idx]
        dist2 = np.minimum(dist2, ((X - centroids[k]) ** 2).sum(axis=1))
    return centroids


def rolling_kmeans(
    X: np.ndarray,
    n_clusters: int,
    *,
    centroid_memory: int = 1,
    max_iter: int = 300,
    warm_iter: int = 100,
    tol: float = 1e-8,
    init: str = "maxmin",
    seed: int | None = None,
    n_init: int = 10,
) -> RollingKMeansResult:
    """Temporally stable K-Means over a panel of entity feature cross-sections.

    Parameters
    ----------
    X : array, shape (T, N, d)
        For each of T periods, an (N entities × d features) cross-section.
        The entity universe is fixed across periods.
    n_clusters : int
        Number of clusters K.
    centroid_memory : int
        Recent centroids remembered per cluster for assignment. 1 gives the
        classical linear boundaries; larger values let a cluster claim points
        near its recent past positions (nonlinear boundaries).
    max_iter, warm_iter : int
        Lloyd iteration caps for the initial fit and warm refits.
    tol : float
        Convergence threshold on centroid movement.
    init : {"maxmin", "k-means++"}
        Initial centroid strategy at t=0. ``"maxmin"`` is deterministic
        (identical data → identical results); ``"k-means++"`` draws
        ``n_init`` seeded restarts and keeps the lowest-inertia one.
    seed : int | None
        Random seed for ``"k-means++"`` (ignored by ``"maxmin"``).
    n_init : int
        Restarts for ``"k-means++"``.

    Examples
    --------
    >>> from quantroll import rolling_kmeans, simulate
    >>> X, membership, _ = simulate.drifting_blobs(80, seed=0)
    >>> res = rolling_kmeans(X, n_clusters=3)
    >>> res.labels.shape
    (80, 90)
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (T, N, d); got shape {X.shape}")
    T, N, d = X.shape
    if not 1 <= n_clusters <= N:
        raise ValueError(f"n_clusters must be in [1, {N}]; got {n_clusters}")
    if centroid_memory < 1:
        raise ValueError(f"centroid_memory must be >= 1; got {centroid_memory}")

    result, _ = _rolling_kmeans_impl(
        X,
        n_clusters,
        centroid_memory=centroid_memory,
        max_iter=max_iter,
        warm_iter=warm_iter,
        tol=tol,
        init=init,
        seed=seed,
        n_init=n_init,
    )
    return result


def _rolling_kmeans_impl(
    X: np.ndarray,
    n_clusters: int,
    *,
    centroid_memory: int,
    max_iter: int,
    warm_iter: int,
    tol: float,
    init: str,
    seed: int | None,
    n_init: int,
) -> tuple[RollingKMeansResult, _KMeansStepper]:
    T, N, d = X.shape
    K = n_clusters
    labels = np.empty((T, N), dtype=np.int64)
    centroids = np.empty((T, K, d))
    inertia = np.empty(T)
    n_relabels = np.zeros(T, dtype=np.int64)
    switched = np.zeros((T, N), dtype=bool)
    n_iter = np.zeros(T, dtype=np.int64)

    stepper = _KMeansStepper(
        K,
        centroid_memory=centroid_memory,
        max_iter=max_iter,
        warm_iter=warm_iter,
        tol=tol,
        init=init,
        seed=seed,
        n_init=n_init,
    )
    for t in range(T):
        out = stepper.step(X[t])
        labels[t] = out["labels"]
        centroids[t] = out["centroids"]
        inertia[t] = out["inertia"]
        n_relabels[t] = out["n_relabels"]
        n_iter[t] = out["n_iter"]
        if t > 0:
            switched[t] = labels[t] != labels[t - 1]

    result = RollingKMeansResult(
        labels=labels,
        centroids=centroids,
        inertia=inertia,
        n_relabels=n_relabels,
        switched=switched,
        n_iter=n_iter,
        n_clusters=K,
        centroid_memory=centroid_memory,
    )
    return result, stepper
