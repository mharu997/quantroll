"""Numba kernels for streaming rolling-window moments and eigendecomposition.

The hot loop maintains sliding sums (S1 = Σx, S2 = Σxxᵀ) so each step costs
O(F·d²) for the covariance update plus one d×d eigendecomposition, instead of
recomputing O(F·W·d²) per window. Floating-point drift from the add/remove
updates is bounded by a periodic full recomputation (``refresh_every``).

All kernels operate on panel-shaped input ``(F, T, d)``: F items (e.g. funds),
T time steps, d features. Plain 2-D input is handled upstream as F = 1.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "rolling_eig_panel",
    "rolling_eig_panel_ewma",
    "ewma_sums",
    "rebuild_sums",
    "slide_sums",
    "eig_from_sums",
]


@njit(cache=True)
def _refresh_sums(X, f, t0, t1, s1, s2):
    d = X.shape[2]
    for i in range(d):
        s1[f, i] = 0.0
        for j in range(d):
            s2[f, i, j] = 0.0
    for k in range(t0, t1):
        for i in range(d):
            xi = X[f, k, i]
            s1[f, i] += xi
            for j in range(d):
                s2[f, i, j] += xi * X[f, k, j]


@njit(cache=True)
def rolling_eig_panel_ewma(X, lam, min_periods, n_components, use_corr,
                           use_subspace, subspace_iters, subspace_tol):
    """Exponentially weighted rolling eigendecomposition (RiskMetrics-style).

    Recursive moments — ``S1 ← λS1 + x``, ``S2 ← λS2 + xxᵀ`` — give the
    weighted covariance ``S2/Σw − μμᵀ`` (normalized weights, population
    convention) with O(F·d²) work per step and no window buffer, so the
    estimator adapts smoothly with effective memory ``1/(1−λ)`` steps.
    Outputs start at ``t = min_periods − 1``.
    """
    F, T, d = X.shape
    p = n_components
    eigvals = np.full((T, p), np.nan)
    eigvecs = np.full((T, d, p), np.nan)
    means = np.full((F, T, d), np.nan)
    total_var = np.full(T, np.nan)
    if T < min_periods:
        return eigvals, eigvecs, means, total_var

    s1 = np.zeros((F, d))
    s2 = np.zeros((F, d, d))
    wsum = 0.0
    cov = np.empty((d, d))
    sd = np.empty(d)
    Vprev = np.zeros((d, p))
    have_prev = False
    subspace = use_subspace and p < d

    for t in range(T):
        wsum = lam * wsum + 1.0
        for f in range(F):
            for i in range(d):
                xi = X[f, t, i]
                s1[f, i] = lam * s1[f, i] + xi
                for j in range(d):
                    s2[f, i, j] = lam * s2[f, i, j] + xi * X[f, t, j]
        if t < min_periods - 1:
            continue

        for i in range(d):
            for j in range(d):
                cov[i, j] = 0.0
        for f in range(F):
            for i in range(d):
                means[f, t, i] = s1[f, i] / wsum
            for i in range(d):
                mi = s1[f, i] / wsum
                for j in range(d):
                    cov[i, j] += s2[f, i, j] / wsum - mi * s1[f, j] / wsum
        if F > 1:
            inv_f = 1.0 / F
            for i in range(d):
                for j in range(d):
                    cov[i, j] *= inv_f

        if use_corr:
            for i in range(d):
                sd[i] = np.sqrt(cov[i, i]) if cov[i, i] > 0.0 else 0.0
            for i in range(d):
                for j in range(d):
                    denom = sd[i] * sd[j]
                    cov[i, j] = cov[i, j] / denom if denom > 0.0 else 0.0
                cov[i, i] = 1.0 if sd[i] > 0.0 else 0.0

        tv = 0.0
        for i in range(d):
            tv += cov[i, i]
        total_var[t] = tv

        for i in range(d):
            for j in range(i + 1, d):
                m = 0.5 * (cov[i, j] + cov[j, i])
                cov[i, j] = m
                cov[j, i] = m

        solved = False
        if subspace and have_prev:
            solved = _subspace_step(
                cov, Vprev, subspace_iters, subspace_tol, eigvals[t], eigvecs[t]
            )
        if not solved:
            vals, vecs = np.linalg.eigh(cov)
            for c in range(p):
                eigvals[t, c] = vals[d - 1 - c]
                for i in range(d):
                    eigvecs[t, i, c] = vecs[i, d - 1 - c]
        for i in range(d):
            for c in range(p):
                Vprev[i, c] = eigvecs[t, i, c]
        have_prev = True

    return eigvals, eigvecs, means, total_var


@njit(cache=True)
def ewma_sums(X, lam):
    """Final EWMA sums over a (T, d) series: ``(S1, S2, Σw)`` for streaming."""
    T, d = X.shape
    s1 = np.zeros(d)
    s2 = np.zeros((d, d))
    wsum = 0.0
    for t in range(T):
        wsum = lam * wsum + 1.0
        for i in range(d):
            xi = X[t, i]
            s1[i] = lam * s1[i] + xi
            for j in range(d):
                s2[i, j] = lam * s2[i, j] + xi * X[t, j]
    return s1, s2, wsum


@njit(cache=True)
def rebuild_sums(W_slice):
    """Fresh sliding sums (S1, S2) from a (W, m) window slice."""
    W, m = W_slice.shape
    s1 = np.zeros(m)
    s2 = np.zeros((m, m))
    for k in range(W):
        for i in range(m):
            xi = W_slice[k, i]
            s1[i] += xi
            for j in range(m):
                s2[i, j] += xi * W_slice[k, j]
    return s1, s2


@njit(cache=True)
def slide_sums(s1, s2, x_new, x_old):
    """Advance sliding sums by one row (add ``x_new``, drop ``x_old``)."""
    m = s1.shape[0]
    for i in range(m):
        s1[i] += x_new[i] - x_old[i]
    for i in range(m):
        xn = x_new[i]
        xo = x_old[i]
        for j in range(m):
            s2[i, j] += xn * x_new[j] - xo * x_old[j]


@njit(cache=True)
def eig_from_sums(s1, s2, window, ddof, use_corr, p):
    """Covariance (or correlation) from sliding sums, then top-p eigenpairs.

    Returns ``(vals (p,), vecs (m, p), mean (m,), total_var)`` for the
    active m-dimensional universe the sums were built over.
    """
    m = s1.shape[0]
    w = float(window)
    mean = s1 / w
    cov = np.empty((m, m))
    for i in range(m):
        si = s1[i]
        for j in range(m):
            cov[i, j] = (s2[i, j] - si * s1[j] / w) / (w - ddof)
    if use_corr:
        sd = np.empty(m)
        for i in range(m):
            sd[i] = np.sqrt(cov[i, i]) if cov[i, i] > 0.0 else 0.0
        for i in range(m):
            for j in range(m):
                denom = sd[i] * sd[j]
                cov[i, j] = cov[i, j] / denom if denom > 0.0 else 0.0
            cov[i, i] = 1.0 if sd[i] > 0.0 else 0.0
    tv = 0.0
    for i in range(m):
        tv += cov[i, i]
    for i in range(m):
        for j in range(i + 1, m):
            v = 0.5 * (cov[i, j] + cov[j, i])
            cov[i, j] = v
            cov[j, i] = v
    ev, evec = np.linalg.eigh(cov)
    vals = np.empty(p)
    vecs = np.empty((m, p))
    for c in range(p):
        vals[c] = ev[m - 1 - c]
        for i in range(m):
            vecs[i, c] = evec[i, m - 1 - c]
    return vals, vecs, mean, tv


@njit(cache=True)
def _subspace_step(cov, Vprev, iters, tol, vals_out, vecs_out):
    """Warm-started subspace iteration + Rayleigh-Ritz for the top-p eigenpairs.

    Returns True when the residual ``||C·V − V·λ||_F <= tol·||λ||_2`` accepts
    the solution; the caller falls back to a full eigendecomposition otherwise.
    """
    d, p = Vprev.shape
    Y = np.dot(cov, Vprev)
    for _ in range(iters - 1):
        Q, _ = np.linalg.qr(Y)
        Y = np.dot(cov, np.ascontiguousarray(Q))
    Q, _ = np.linalg.qr(Y)
    Q = np.ascontiguousarray(Q)

    T1 = np.dot(cov, Q)
    H = np.empty((p, p))
    for a in range(p):
        for b in range(p):
            acc = 0.0
            for i in range(d):
                acc += Q[i, a] * T1[i, b]
            H[a, b] = acc
    for a in range(p):
        for b in range(a + 1, p):
            m = 0.5 * (H[a, b] + H[b, a])
            H[a, b] = m
            H[b, a] = m

    hv, hw = np.linalg.eigh(H)
    W = np.empty((p, p))
    for a in range(p):
        for b in range(p):
            W[a, b] = hw[a, p - 1 - b]
    V = np.dot(Q, W)

    lam_norm2 = 0.0
    for c in range(p):
        lam = hv[p - 1 - c]
        vals_out[c] = lam
        lam_norm2 += lam * lam

    R = np.dot(cov, V)
    res2 = 0.0
    for c in range(p):
        lam = vals_out[c]
        for i in range(d):
            r = R[i, c] - V[i, c] * lam
            res2 += r * r
    if res2 > tol * tol * max(lam_norm2, 1e-300):
        return False
    for i in range(d):
        for c in range(p):
            vecs_out[i, c] = V[i, c]
    return True


@njit(cache=True)
def rolling_eig_panel(X, window, n_components, ddof, use_corr, refresh_every,
                      use_subspace, subspace_iters, subspace_tol):
    """Rolling eigendecomposition of the cross-item average covariance matrix.

    Parameters
    ----------
    X : float64 array, shape (F, T, d)
    window : int
        Rolling window length W (in time steps).
    n_components : int
        Number of leading eigenpairs to keep (p).
    ddof : int
        Delta degrees of freedom for the covariance (1 = sample covariance).
    use_corr : bool
        Decompose the correlation matrix instead of the covariance matrix.
    refresh_every : int
        Full sum recomputation cadence to cancel floating-point drift.
    use_subspace : bool
        Solve each step with warm-started subspace iteration (O(d²p)) instead
        of a full eigendecomposition (O(d³)), falling back to the exact solver
        whenever the residual check fails.
    subspace_iters : int
        Matrix-subspace multiplications per step (2 is usually plenty).
    subspace_tol : float
        Relative residual tolerance of the acceptance check.

    Returns
    -------
    eigvals : (T, p)   descending, NaN during the warm-up (t < W-1)
    eigvecs : (T, d, p) columns are unit eigenvectors (raw, unaligned)
    means   : (F, T, d) rolling window means per item
    total_var : (T,)   trace of the decomposed matrix (for variance ratios)
    """
    F, T, d = X.shape
    p = n_components
    eigvals = np.full((T, p), np.nan)
    eigvecs = np.full((T, d, p), np.nan)
    means = np.full((F, T, d), np.nan)
    total_var = np.full(T, np.nan)
    if T < window:
        return eigvals, eigvecs, means, total_var

    s1 = np.zeros((F, d))
    s2 = np.zeros((F, d, d))
    cov = np.empty((d, d))
    sd = np.empty(d)
    w = float(window)
    Vprev = np.zeros((d, p))
    have_prev = False
    subspace = use_subspace and p < d

    for f in range(F):
        _refresh_sums(X, f, 0, window, s1, s2)

    since_refresh = 0
    for t in range(window - 1, T):
        if t >= window:
            for f in range(F):
                for i in range(d):
                    s1[f, i] += X[f, t, i] - X[f, t - window, i]
                for i in range(d):
                    xn = X[f, t, i]
                    xo = X[f, t - window, i]
                    for j in range(d):
                        s2[f, i, j] += xn * X[f, t, j] - xo * X[f, t - window, j]
            since_refresh += 1
            if since_refresh >= refresh_every:
                for f in range(F):
                    _refresh_sums(X, f, t - window + 1, t + 1, s1, s2)
                since_refresh = 0

        for i in range(d):
            for j in range(d):
                cov[i, j] = 0.0
        for f in range(F):
            for i in range(d):
                means[f, t, i] = s1[f, i] / w
            for i in range(d):
                si = s1[f, i]
                for j in range(d):
                    cov[i, j] += (s2[f, i, j] - si * s1[f, j] / w) / (w - ddof)
        if F > 1:
            inv_f = 1.0 / F
            for i in range(d):
                for j in range(d):
                    cov[i, j] *= inv_f

        if use_corr:
            for i in range(d):
                sd[i] = np.sqrt(cov[i, i]) if cov[i, i] > 0.0 else 0.0
            for i in range(d):
                for j in range(d):
                    denom = sd[i] * sd[j]
                    cov[i, j] = cov[i, j] / denom if denom > 0.0 else 0.0
                cov[i, i] = 1.0 if sd[i] > 0.0 else 0.0

        tv = 0.0
        for i in range(d):
            tv += cov[i, i]
        total_var[t] = tv

        for i in range(d):
            for j in range(i + 1, d):
                m = 0.5 * (cov[i, j] + cov[j, i])
                cov[i, j] = m
                cov[j, i] = m

        solved = False
        if subspace and have_prev:
            solved = _subspace_step(
                cov, Vprev, subspace_iters, subspace_tol, eigvals[t], eigvecs[t]
            )
        if not solved:
            vals, vecs = np.linalg.eigh(cov)
            for c in range(p):
                eigvals[t, c] = vals[d - 1 - c]
                for i in range(d):
                    eigvecs[t, i, c] = vecs[i, d - 1 - c]
        for i in range(d):
            for c in range(p):
                Vprev[i, c] = eigvecs[t, i, c]
        have_prev = True

    return eigvals, eigvecs, means, total_var
