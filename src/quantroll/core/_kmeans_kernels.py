"""Numba kernels for K-Means on rolling cross-sections.

Built for repeated warm-started refits: Lloyd's algorithm restarted from the
previous step's centroids converges in a couple of iterations when the data
drifts slowly, and — crucially — returns the *same* solution when the data
has not changed at all. Empty clusters are re-seeded deterministically at the
points farthest from their assigned centroids.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["lloyd", "assign_multi", "maxmin_init"]


@njit(cache=True)
def _assign(X, centroids, labels, dist2):
    """Nearest-centroid assignment; fills labels/dist2, returns inertia."""
    N, d = X.shape
    K = centroids.shape[0]
    inertia = 0.0
    for i in range(N):
        best = 0
        bd = np.inf
        for k in range(K):
            acc = 0.0
            for j in range(d):
                diff = X[i, j] - centroids[k, j]
                acc += diff * diff
            if acc < bd:
                bd = acc
                best = k
        labels[i] = best
        dist2[i] = bd
        inertia += bd
    return inertia


@njit(cache=True)
def lloyd(X, centroids, max_iter, tol):
    """Lloyd's K-Means from the given centroids (updated in place).

    Returns ``(labels, inertia, n_iter)``. Deterministic: ties break to the
    lowest cluster index, and an emptied cluster is re-seeded at the point
    currently farthest from its assigned centroid (successively for several
    empties). Convergence when the largest centroid movement falls below
    ``tol`` (Euclidean).
    """
    N, d = X.shape
    K = centroids.shape[0]
    labels = np.empty(N, dtype=np.int64)
    dist2 = np.empty(N)
    counts = np.empty(K)
    sums = np.empty((K, d))
    tol2 = tol * tol

    it = 0
    for it in range(1, max_iter + 1):
        _assign(X, centroids, labels, dist2)

        for k in range(K):
            counts[k] = 0.0
            for j in range(d):
                sums[k, j] = 0.0
        for i in range(N):
            k = labels[i]
            counts[k] += 1.0
            for j in range(d):
                sums[k, j] += X[i, j]

        n_empty = 0
        for k in range(K):
            if counts[k] == 0.0:
                n_empty += 1
        if n_empty > 0:
            order = np.argsort(-dist2)
            used = 0
            for k in range(K):
                if counts[k] == 0.0:
                    idx = order[used]
                    used += 1
                    counts[k] = 1.0
                    for j in range(d):
                        sums[k, j] = X[idx, j]

        shift2 = 0.0
        for k in range(K):
            acc = 0.0
            for j in range(d):
                nc = sums[k, j] / counts[k]
                diff = nc - centroids[k, j]
                acc += diff * diff
                centroids[k, j] = nc
            if acc > shift2:
                shift2 = acc
        if shift2 <= tol2:
            break

    inertia = _assign(X, centroids, labels, dist2)
    return labels, inertia, it


@njit(cache=True)
def assign_multi(X, bank, bank_n):
    """Assign each point to the cluster with the nearest *remembered* centroid.

    ``bank`` is (K, M, d) — up to M recent centroids per cluster — and
    ``bank_n[k]`` how many are valid. A cluster claims a point if any of its
    remembered centroids is closest, which yields piecewise-linear (overall
    nonlinear) decision boundaries following each cluster's trajectory. With
    one centroid per cluster this reduces to the classical assignment.

    Returns ``(labels, dist2)`` with ``dist2`` the squared distance to the
    winning remembered centroid.
    """
    N, d = X.shape
    K = bank.shape[0]
    labels = np.empty(N, dtype=np.int64)
    dist2 = np.empty(N)
    for i in range(N):
        best_k = 0
        best = np.inf
        for k in range(K):
            for m in range(bank_n[k]):
                acc = 0.0
                for j in range(d):
                    diff = X[i, j] - bank[k, m, j]
                    acc += diff * diff
                if acc < best:
                    best = acc
                    best_k = k
        labels[i] = best_k
        dist2[i] = best
    return labels, dist2


@njit(cache=True)
def maxmin_init(X, K):
    """Deterministic farthest-point initial centroids.

    Starts from the most central observation (nearest the grand mean), then
    repeatedly adds the point farthest from all chosen centroids. No
    randomness — identical data always yields identical starting points.
    """
    N, d = X.shape
    centroids = np.empty((K, d))
    mean = np.zeros(d)
    for i in range(N):
        for j in range(d):
            mean[j] += X[i, j]
    for j in range(d):
        mean[j] /= N

    best = 0
    bd = np.inf
    for i in range(N):
        acc = 0.0
        for j in range(d):
            diff = X[i, j] - mean[j]
            acc += diff * diff
        if acc < bd:
            bd = acc
            best = i
    for j in range(d):
        centroids[0, j] = X[best, j]

    dist2 = np.empty(N)
    for i in range(N):
        acc = 0.0
        for j in range(d):
            diff = X[i, j] - centroids[0, j]
            acc += diff * diff
        dist2[i] = acc

    for k in range(1, K):
        far = 0
        fd = -1.0
        for i in range(N):
            if dist2[i] > fd:
                fd = dist2[i]
                far = i
        for j in range(d):
            centroids[k, j] = X[far, j]
        for i in range(N):
            acc = 0.0
            for j in range(d):
                diff = X[i, j] - centroids[k, j]
                acc += diff * diff
            if acc < dist2[i]:
                dist2[i] = acc
    return centroids
