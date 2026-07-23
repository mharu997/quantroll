"""Numba kernels: Gaussian mixture and Gaussian HMM EM on rolling windows.

Small, allocation-light EM built for repeated warm-started refits: on rolling
windows the parameters barely move between steps, so EM restarted from the
previous window's solution converges in a couple of iterations. All routines
use full covariances (d is small in regime work) with a diagonal floor
(``reg_covar``) and a tiny uniform responsibility prior to prevent component
starvation.

The HMM routines (``hmm_em_fit``, ``hmm_filter_last``) implement scaled
forward-backward (Rabiner) so persistence is *learned* as a transition matrix
rather than imposed by ad-hoc smoothing.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["em_fit", "log_gauss_matrix", "posterior", "hmm_em_fit", "hmm_filter_last"]

_RESP_PRIOR = 1e-10


@njit(cache=True)
def _chol_logdet_prec(covs, reg_covar):
    """Cholesky-based log-determinants and precision matrices per component."""
    K, d, _ = covs.shape
    logdets = np.empty(K)
    precs = np.empty((K, d, d))
    for k in range(K):
        C = covs[k].copy()
        for i in range(d):
            C[i, i] += reg_covar
        L = np.linalg.cholesky(C)
        acc = 0.0
        for i in range(d):
            acc += np.log(L[i, i])
        logdets[k] = 2.0 * acc
        precs[k] = np.linalg.inv(C)
    return logdets, precs


@njit(cache=True)
def log_gauss_matrix(X, means, covs, reg_covar):
    """(n, K) log N(x_n | μ_k, Σ_k)."""
    n, d = X.shape
    K = means.shape[0]
    logdets, precs = _chol_logdet_prec(covs, reg_covar)
    const = d * np.log(2.0 * np.pi)
    out = np.empty((n, K))
    diff = np.empty(d)
    for i in range(n):
        for k in range(K):
            for j in range(d):
                diff[j] = X[i, j] - means[k, j]
            maha = 0.0
            for a in range(d):
                acc = 0.0
                for b in range(d):
                    acc += precs[k, a, b] * diff[b]
                maha += diff[a] * acc
            out[i, k] = -0.5 * (const + logdets[k] + maha)
    return out


@njit(cache=True)
def _e_step(X, means, covs, weights, reg_covar, resp):
    """Fill responsibilities; return mean log-likelihood per observation."""
    n, _ = X.shape
    K = means.shape[0]
    logp = log_gauss_matrix(X, means, covs, reg_covar)
    ll = 0.0
    for i in range(n):
        mx = -np.inf
        for k in range(K):
            logp[i, k] += np.log(weights[k])
            if logp[i, k] > mx:
                mx = logp[i, k]
        s = 0.0
        for k in range(K):
            s += np.exp(logp[i, k] - mx)
        lse = mx + np.log(s)
        ll += lse
        for k in range(K):
            r = np.exp(logp[i, k] - lse)
            resp[i, k] = r * (1.0 - _RESP_PRIOR) + _RESP_PRIOR / K
    return ll / n


@njit(cache=True)
def _m_step(X, resp, means, covs, weights, reg_covar):
    n, d = X.shape
    K = means.shape[0]
    for k in range(K):
        nk = 0.0
        for i in range(n):
            nk += resp[i, k]
        weights[k] = nk / n
        for j in range(d):
            acc = 0.0
            for i in range(n):
                acc += resp[i, k] * X[i, j]
            means[k, j] = acc / nk
        for a in range(d):
            for b in range(a, d):
                acc = 0.0
                for i in range(n):
                    acc += (
                        resp[i, k] * (X[i, a] - means[k, a]) * (X[i, b] - means[k, b])
                    )
                v = acc / nk
                covs[k, a, b] = v
                covs[k, b, a] = v
        for a in range(d):
            covs[k, a, a] += reg_covar


@njit(cache=True)
def em_fit(X, means, covs, weights, max_iter, tol, reg_covar):
    """Run EM from the given parameters (modified in place).

    Returns ``(mean_loglik, n_iter)``. Convergence: change in mean
    log-likelihood below ``tol``.
    """
    n = X.shape[0]
    K = means.shape[0]
    resp = np.empty((n, K))
    prev_ll = -np.inf
    ll = prev_ll
    it = 0
    for it in range(1, max_iter + 1):
        ll = _e_step(X, means, covs, weights, reg_covar, resp)
        _m_step(X, resp, means, covs, weights, reg_covar)
        if np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return ll, it


@njit(cache=True)
def _shifted_b(logb):
    """Per-row max-shifted emission likelihoods and the shifts (for scaling)."""
    n, K = logb.shape
    b = np.empty((n, K))
    shifts = np.empty(n)
    for t in range(n):
        m = -np.inf
        for k in range(K):
            if logb[t, k] > m:
                m = logb[t, k]
        shifts[t] = m
        for k in range(K):
            b[t, k] = np.exp(logb[t, k] - m)
    return b, shifts


@njit(cache=True)
def _forward(b, A, pi, alpha):
    """Scaled forward pass. Fills alpha (normalized rows); returns log c sums."""
    n, K = b.shape
    logc = 0.0
    c = 0.0
    for k in range(K):
        alpha[0, k] = pi[k] * b[0, k]
        c += alpha[0, k]
    for k in range(K):
        alpha[0, k] /= c
    logc += np.log(c)
    for t in range(1, n):
        c = 0.0
        for j in range(K):
            acc = 0.0
            for i in range(K):
                acc += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = acc * b[t, j]
            c += alpha[t, j]
        for j in range(K):
            alpha[t, j] /= c
        logc += np.log(c)
    return logc


@njit(cache=True)
def hmm_em_fit(X, means, covs, A, pi, max_iter, tol, reg_covar, trans_pseudo,
               min_state_mass, A_prior, prior_means, prior_covs, prior_strength):
    """MAP Baum-Welch for a Gaussian HMM, warm-started from the given parameters.

    ``means``, ``covs``, ``A`` (transition matrix) and ``pi`` (initial state
    distribution) are updated in place. Returns ``(mean_loglik, n_iter)``.

    Three MAP-style guards make rolling-window refits sane:

    - ``trans_pseudo`` pseudo-counts, distributed like the *fixed reference*
      matrix ``A_prior`` (not the warm start — a collapsed warm start would
      perpetuate itself), are added to the transition statistics: a state
      starved in this window has its transition row healed back toward the
      reference, while active states' real counts dominate the prior;
    - each state's emission parameters carry a quasi-Normal-Inverse-Wishart
      prior centered on (``prior_means``, ``prior_covs``) with weight
      ``prior_strength`` pseudo-observations. A regime absent from the
      window is *held* at its prior instead of being repurposed to model
      sub-structure of whatever regime is present — the failure mode of
      maximum-likelihood mixtures on single-regime stretches. Set
      ``prior_strength = 0`` for plain maximum likelihood;
    - states whose responsibility mass in the window falls below
      ``min_state_mass`` (in observations) keep their previous Gaussian
      parameters rather than being refit on numerical dust.
    """
    n, d = X.shape
    K = means.shape[0]
    alpha = np.empty((n, K))
    beta = np.empty((n, K))
    gamma = np.empty((n, K))
    xi_sum = np.empty((K, K))
    prev_ll = -np.inf
    ll = prev_ll
    it = 0
    for it in range(1, max_iter + 1):
        logb = log_gauss_matrix(X, means, covs, reg_covar)
        b, shifts = _shifted_b(logb)

        # forward (scaled) — track scale factors for the backward pass
        cs = np.empty(n)
        c = 0.0
        for k in range(K):
            alpha[0, k] = pi[k] * b[0, k]
            c += alpha[0, k]
        cs[0] = c
        for k in range(K):
            alpha[0, k] /= c
        for t in range(1, n):
            c = 0.0
            for j in range(K):
                acc = 0.0
                for i in range(K):
                    acc += alpha[t - 1, i] * A[i, j]
                alpha[t, j] = acc * b[t, j]
                c += alpha[t, j]
            cs[t] = c
            for j in range(K):
                alpha[t, j] /= c

        ll_new = 0.0
        for t in range(n):
            ll_new += np.log(cs[t]) + shifts[t]
        ll = ll_new / n

        # backward (scaled with the same factors)
        for k in range(K):
            beta[n - 1, k] = 1.0
        for t in range(n - 2, -1, -1):
            for i in range(K):
                acc = 0.0
                for j in range(K):
                    acc += A[i, j] * b[t + 1, j] * beta[t + 1, j]
                beta[t, i] = acc / cs[t + 1]

        for t in range(n):
            for k in range(K):
                g = alpha[t, k] * beta[t, k]
                gamma[t, k] = g * (1.0 - _RESP_PRIOR) + _RESP_PRIOR / K

        for i in range(K):
            for j in range(K):
                xi_sum[i, j] = 0.0
        for t in range(1, n):
            for i in range(K):
                for j in range(K):
                    xi_sum[i, j] += (
                        alpha[t - 1, i] * A[i, j] * b[t, j] * beta[t, j] / cs[t]
                    )

        # M-step: transitions, initial distribution, Gaussian parameters
        for i in range(K):
            row = 0.0
            for j in range(K):
                xi_sum[i, j] += _RESP_PRIOR + trans_pseudo * A_prior[i, j]
                row += xi_sum[i, j]
            for j in range(K):
                A[i, j] = xi_sum[i, j] / row
        # floor keeps every transition reachable (no absorbing lock-in)
        for i in range(K):
            row = 0.0
            for j in range(K):
                if A[i, j] < 1e-4:
                    A[i, j] = 1e-4
                row += A[i, j]
            for j in range(K):
                A[i, j] /= row
        pisum = 0.0
        for k in range(K):
            pi[k] = gamma[0, k]
            pisum += pi[k]
        for k in range(K):
            pi[k] /= pisum
        update_mask = np.empty(K, dtype=np.bool_)
        for k in range(K):
            mass = 0.0
            for t in range(n):
                mass += gamma[t, k]
            update_mask[k] = mass >= min_state_mass
        _m_step_gauss(
            X, gamma, means, covs, reg_covar, update_mask,
            prior_means, prior_covs, prior_strength,
        )

        if np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    return ll, it


@njit(cache=True)
def _m_step_gauss(X, resp, means, covs, reg_covar, update_mask,
                  prior_means, prior_covs, prior_strength):
    n, d = X.shape
    K = means.shape[0]
    kappa = prior_strength
    for k in range(K):
        if not update_mask[k]:
            continue
        nk = 0.0
        for i in range(n):
            nk += resp[i, k]
        denom = nk + kappa
        for j in range(d):
            acc = 0.0
            for i in range(n):
                acc += resp[i, k] * X[i, j]
            if kappa > 0.0:
                acc += kappa * prior_means[k, j]
            means[k, j] = acc / denom
        for a in range(d):
            for b_ in range(a, d):
                acc = 0.0
                for i in range(n):
                    acc += (
                        resp[i, k] * (X[i, a] - means[k, a]) * (X[i, b_] - means[k, b_])
                    )
                if kappa > 0.0:
                    da = prior_means[k, a] - means[k, a]
                    db = prior_means[k, b_] - means[k, b_]
                    acc += kappa * (prior_covs[k, a, b_] + da * db)
                v = acc / denom
                covs[k, a, b_] = v
                covs[k, b_, a] = v
        for a in range(d):
            covs[k, a, a] += reg_covar


@njit(cache=True)
def hmm_filter_last(X, means, covs, A, pi, reg_covar):
    """Filtered state distribution at the last observation of ``X``.

    One scaled forward pass; returns the (K,) normalized forward variable —
    P(state at T | all observations up to T).
    """
    n = X.shape[0]
    K = means.shape[0]
    logb = log_gauss_matrix(X, means, covs, reg_covar)
    b, _ = _shifted_b(logb)
    alpha = np.empty((n, K))
    _forward(b, A, pi, alpha)
    out = np.empty(K)
    for k in range(K):
        out[k] = alpha[n - 1, k]
    return out


@njit(cache=True)
def posterior(X, means, covs, weights, reg_covar):
    """(n, K) posterior membership probabilities under fixed parameters."""
    n = X.shape[0]
    K = means.shape[0]
    logp = log_gauss_matrix(X, means, covs, reg_covar)
    out = np.empty((n, K))
    for i in range(n):
        mx = -np.inf
        for k in range(K):
            logp[i, k] += np.log(weights[k])
            if logp[i, k] > mx:
                mx = logp[i, k]
        s = 0.0
        for k in range(K):
            s += np.exp(logp[i, k] - mx)
        for k in range(K):
            out[i, k] = np.exp(logp[i, k] - mx) / s
    return out
