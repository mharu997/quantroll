"""Temporally stable regime detection over rolling windows.

A Gaussian mixture is maintained over a rolling window of feature vectors.
Between steps the mixture is refit by EM **warm-started from the previous
window's parameters**, which is both fast (a couple of iterations) and
temporally coherent. Three mechanisms keep regime identities stable, in the
spirit of the rolling regime-detection methodology of Hirsa, Xu & Malhotra
(2024), independently implemented:

- **anchored identities** — each refit's components are matched (Hungarian
  assignment on Bhattacharyya distance) against slowly-updated *anchor*
  parameters — an exponential ensemble of each regime's own history — rather
  than just the previous step. Anchors freeze whenever the current components
  are too similar to distinguish, so a regime's identity survives long spells
  in which it is absent from the window and snaps back when it returns.
  Classification runs against the anchors (the identity bearers), while the
  window fit's role is to adapt them;
- **learned persistence** — each window is fit as a Gaussian *hidden Markov
  model* (warm-started Baum-Welch), so regime stickiness is estimated as a
  transition matrix rather than imposed; probabilities are the HMM forward
  filter at the newest observation, aggregating evidence across the window —
  individual observations of overlapping regimes are far too noisy to
  classify alone;
- **hysteresis** — the regime call switches only when the filtered
  probability of the challenger exceeds ``switch_threshold``, eliminating
  label flicker around regime boundaries.

Initial components are seeded deterministically from dispersion quantiles of
the first window (component 0 = calmest, K-1 = most turbulent), then refined
by full EM, so runs are reproducible without a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._gmm_kernels import hmm_em_fit, hmm_filter_last, posterior

__all__ = ["rolling_regimes", "RollingRegimesResult", "match_components"]


@dataclass(frozen=True)
class RollingRegimesResult:
    """Output of :func:`rolling_regimes`. Time-indexed arrays are full length
    ``T``; warm-up rows (``t < window - 1``) hold NaN / label ``-1``."""

    labels: np.ndarray
    """(T,) int regime call after filtering and hysteresis; -1 in warm-up."""
    probs: np.ndarray
    """(T, K) forward-filtered regime probabilities (anchor identities)."""
    raw_probs: np.ndarray
    """(T, K) unsmoothed posterior of each observation under the current model."""
    means: np.ndarray
    """(T, K, d) regime means over time (stable component identities)."""
    covariances: np.ndarray
    """(T, K, d, d) regime covariances over time."""
    weights: np.ndarray
    """(T, K) stationary regime frequencies implied by the learned transitions."""
    transition: np.ndarray
    """(T, K, K) learned HMM transition matrix (rows sum to 1)."""
    log_likelihood: np.ndarray
    """(T,) mean log-likelihood per observation at the most recent refit."""
    n_relabels: np.ndarray
    """(T,) components whose identity assignment changed at each refit."""
    switched: np.ndarray
    """(T,) bool — True where the regime call changed."""
    window: int = 0
    n_regimes: int = 0


def match_components(
    prev_means: np.ndarray,
    prev_covs: np.ndarray,
    cur_means: np.ndarray,
    cur_covs: np.ndarray,
) -> np.ndarray:
    """Match current mixture components to previous identities.

    Returns ``perm`` with ``perm[k]`` = index of the current component
    assigned to previous identity ``k``, minimizing total Bhattacharyya
    distance via the Hungarian algorithm.
    """
    K = prev_means.shape[0]
    D = np.empty((K, K))
    for a in range(K):
        for b in range(K):
            D[a, b] = _bhattacharyya(
                prev_means[a], prev_covs[a], cur_means[b], cur_covs[b]
            )
    _, perm = linear_sum_assignment(D)
    return perm


def _bhattacharyya(m1, c1, m2, c2) -> float:
    cbar = 0.5 * (c1 + c2)
    diff = m1 - m2
    sign, logdet_bar = np.linalg.slogdet(cbar)
    if sign <= 0:
        return np.inf
    _, logdet1 = np.linalg.slogdet(c1)
    _, logdet2 = np.linalg.slogdet(c2)
    maha = float(diff @ np.linalg.solve(cbar, diff))
    return 0.125 * maha + 0.5 * (logdet_bar - 0.5 * (logdet1 + logdet2))


def _stationary(A: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix (left eigenvector)."""
    w, v = np.linalg.eig(A.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    s = np.abs(np.real(v[:, idx]))
    total = s.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(A.shape[0], 1.0 / A.shape[0])
    return s / total


def _dispersion_init(win: np.ndarray, K: int, reg_covar: float):
    """Deterministic init: split window rows into K dispersion quantile groups.

    Rows are ranked by squared deviation from the window mean, so component 0
    starts on the calmest observations and component K-1 on the most
    turbulent — a natural, reproducible seeding for financial regimes.
    """
    W, d = win.shape
    disp = ((win - win.mean(axis=0)) ** 2).sum(axis=1)
    order = np.argsort(disp)
    means = np.empty((K, d))
    covs = np.empty((K, d, d))
    weights = np.full(K, 1.0 / K)
    bounds = np.linspace(0, W, K + 1).astype(np.int64)
    for k in range(K):
        rows = win[order[bounds[k] : bounds[k + 1]]]
        means[k] = rows.mean(axis=0)
        c = np.cov(rows, rowvar=False, ddof=0).reshape(d, d)
        covs[k] = c + reg_covar * np.eye(d)
    return means, covs, weights


class _RegimeStepper:
    """Single-step engine shared verbatim by batch fit and streaming update,
    so both paths produce identical results by construction."""

    def __init__(
        self,
        n_regimes: int,
        *,
        refit_every: int,
        p_switch: float,
        switch_threshold: float,
        reg_covar: float,
        warm_iter: int,
        init_iter: int,
        tol: float,
        anchor_rate: float,
        anchor_min_separation: float,
        anchor_strength: float,
        trans_pseudo: float,
        min_state_mass: float,
    ) -> None:
        self.K = n_regimes
        self.refit_every = refit_every
        self.p_switch = p_switch
        self.switch_threshold = switch_threshold
        self.reg_covar = reg_covar
        self.warm_iter = warm_iter
        self.init_iter = init_iter
        self.tol = tol
        self.anchor_rate = anchor_rate
        self.anchor_min_separation = anchor_min_separation
        self.anchor_strength = anchor_strength
        self.trans_pseudo = trans_pseudo
        self.min_state_mass = min_state_mass
        self.step_no = -1
        self.label = -1
        self.probs_s: np.ndarray | None = None
        self.ll = np.nan

    def _apply_perm(self, perm: np.ndarray) -> None:
        self.means = np.ascontiguousarray(self.means[perm])
        self.covariances = np.ascontiguousarray(self.covariances[perm])
        self.A = np.ascontiguousarray(self.A[perm][:, perm])
        self.pi = np.ascontiguousarray(self.pi[perm])

    def init_fit(self, win: np.ndarray) -> None:
        self.means, self.covariances, _ = _dispersion_init(
            win, self.K, self.reg_covar
        )
        if self.K > 1:
            A = np.full((self.K, self.K), self.p_switch / (self.K - 1))
            np.fill_diagonal(A, 1.0 - self.p_switch)
        else:
            A = np.ones((1, 1))
        self.A = A.copy()
        self.A_sticky = A  # fixed restorative reference for transition priors
        self.pi = np.full(self.K, 1.0 / self.K)
        self.ll, _ = hmm_em_fit(
            win, self.means, self.covariances, self.A, self.pi,
            self.init_iter, self.tol, self.reg_covar,
            self.trans_pseudo, self.min_state_mass, self.A_sticky,
            self.means.copy(), self.covariances.copy(), 0.0,
        )
        order = np.argsort([np.trace(c) for c in self.covariances])
        self._apply_perm(np.asarray(order))
        self.anchor_means = self.means.copy()
        self.anchor_covs = self.covariances.copy()

    def _separation(self) -> float:
        """Smallest pairwise Bhattacharyya distance among current components."""
        sep = np.inf
        for a in range(self.K):
            for b in range(a + 1, self.K):
                sep = min(
                    sep,
                    _bhattacharyya(
                        self.means[a], self.covariances[a],
                        self.means[b], self.covariances[b],
                    ),
                )
        return sep

    def step(self, win: np.ndarray, x_t: np.ndarray) -> dict:
        self.step_no += 1
        n_relabels = 0
        if self.step_no == 0:
            self.init_fit(win)
        elif self.step_no % self.refit_every == 0:
            self.ll, _ = hmm_em_fit(
                win, self.means, self.covariances, self.A, self.pi,
                self.warm_iter, self.tol, self.reg_covar,
                self.trans_pseudo, self.min_state_mass, self.A_sticky,
                self.anchor_means, self.anchor_covs, self.anchor_strength,
            )
            perm = match_components(
                self.anchor_means, self.anchor_covs, self.means, self.covariances
            )
            if not np.array_equal(perm, np.arange(self.K)):
                self._apply_perm(perm)
                n_relabels = int((perm != np.arange(self.K)).sum())
            if self.K == 1 or self._separation() >= self.anchor_min_separation:
                eta = self.anchor_rate
                self.anchor_means = (1.0 - eta) * self.anchor_means + eta * self.means
                self.anchor_covs = (1.0 - eta) * self.anchor_covs + eta * self.covariances

        self.probs_s = hmm_filter_last(
            win, self.means, self.covariances, self.A, self.pi, self.reg_covar
        )
        raw = posterior(
            x_t[None, :], self.means, self.covariances,
            np.full(self.K, 1.0 / self.K), self.reg_covar,
        )[0]

        switched = False
        challenger = int(np.argmax(self.probs_s))
        if self.label < 0:
            self.label = challenger
        elif challenger != self.label and self.probs_s[challenger] >= self.switch_threshold:
            self.label = challenger
            switched = True

        return {
            "raw": raw,
            "probs": self.probs_s.copy(),
            "label": self.label,
            "switched": switched,
            "n_relabels": n_relabels,
            "transition": self.A.copy(),
            "ll": self.ll,
            "means": self.means.copy(),
            "covariances": self.covariances.copy(),
            "weights": _stationary(self.A),
        }


def rolling_regimes(
    X: np.ndarray,
    window: int,
    n_regimes: int = 2,
    *,
    refit_every: int = 1,
    p_switch: float = 0.02,
    switch_threshold: float = 0.6,
    reg_covar: float = 1e-6,
    warm_iter: int = 10,
    init_iter: int = 200,
    tol: float = 1e-6,
    anchor_rate: float | None = None,
    anchor_min_separation: float = 0.1,
    anchor_strength: float = 20.0,
    trans_pseudo: float = 2.0,
    min_state_mass: float = 2.0,
) -> RollingRegimesResult:
    """Detect regimes with a temporally stable rolling Gaussian HMM.

    Parameters
    ----------
    X : array, shape (T, d)
        Feature vectors over time (e.g. returns, realized vol, spreads).
    window : int
        Rolling window length W; first regime call at ``t = W - 1``. The
        window must span several regime cycles — a regime that never appears
        in the window cannot be learned from it. For daily equity-style data
        think years, not months (e.g. 750-1250 steps).
    n_regimes : int
        Number of hidden states K.
    refit_every : int
        Refit cadence in steps (1 = every step; larger is faster, filtered
        probabilities still update every step under the last-fit model).
    p_switch : float in (0, 1)
        *Initial* per-step regime switch probability seeding the transition
        matrix; persistence is subsequently learned from the data by
        Baum-Welch.
    switch_threshold : float
        Filtered probability a challenger regime needs to take over the call.
    reg_covar : float
        Diagonal floor added to covariances.
    warm_iter, init_iter : int
        EM iteration caps for warm refits and the initial fit.
    tol : float
        EM convergence tolerance on mean log-likelihood.
    anchor_rate : float | None
        EWMA rate at which identity anchors absorb newly fitted parameters
        (None → ``2 / (window + 1)``). Anchors are the long-memory reference
        that component matching aligns to.
    anchor_min_separation : float
        Minimum pairwise Bhattacharyya distance among fitted components for
        anchors to update; below it the fit is considered degenerate (a
        single-regime window) and identities are held frozen.
    anchor_strength : float
        Weight of the anchors as an emission prior in each refit, in
        pseudo-observations (MAP-EM). This is what stops a regime absent
        from the window being repurposed to model sub-structure of the
        present one; a regime actually in the window outvotes it easily.
        0 disables the prior (plain maximum likelihood).
    trans_pseudo : float
        Pseudo-count strength regularizing each refit's transition matrix
        toward its warm start — windows with few or no switches keep a
        sensible sticky matrix.
    min_state_mass : float
        Minimum responsibility mass (in observations) a state needs in the
        window for its Gaussian parameters to update; below it the regime is
        absent and keeps its remembered shape.

    Examples
    --------
    >>> from quantroll import rolling_regimes, simulate
    >>> R, states = simulate.regime_returns(2500, seed=0)
    >>> res = rolling_regimes(R, window=750, n_regimes=2)
    >>> res.labels[-1] in (0, 1)
    True
    """
    result, _ = _rolling_regimes_impl(
        X,
        window,
        n_regimes,
        refit_every=refit_every,
        p_switch=p_switch,
        switch_threshold=switch_threshold,
        reg_covar=reg_covar,
        warm_iter=warm_iter,
        init_iter=init_iter,
        tol=tol,
        anchor_rate=anchor_rate,
        anchor_min_separation=anchor_min_separation,
        anchor_strength=anchor_strength,
        trans_pseudo=trans_pseudo,
        min_state_mass=min_state_mass,
    )
    return result


def _rolling_regimes_impl(
    X: np.ndarray,
    window: int,
    n_regimes: int,
    *,
    refit_every: int,
    p_switch: float,
    switch_threshold: float,
    reg_covar: float,
    warm_iter: int,
    init_iter: int,
    tol: float,
    anchor_rate: float | None,
    anchor_min_separation: float,
    anchor_strength: float = 20.0,
    trans_pseudo: float = 2.0,
    min_state_mass: float = 2.0,
) -> tuple[RollingRegimesResult, _RegimeStepper]:
    """Run the rolling loop and also return the live stepper (for streaming)."""
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (T, d); got shape {X.shape}")
    T, d = X.shape
    if not 1 <= n_regimes <= 32:
        raise ValueError(f"n_regimes must be in [1, 32]; got {n_regimes}")
    if window < 2 * n_regimes:
        raise ValueError(f"window must be >= 2*n_regimes; got {window}")
    if window > T:
        raise ValueError(f"window ({window}) exceeds series length ({T})")
    if not 0.0 < p_switch < 1.0:
        raise ValueError(f"p_switch must be in (0, 1); got {p_switch}")

    K = n_regimes
    labels = np.full(T, -1, dtype=np.int64)
    probs = np.full((T, K), np.nan)
    raw_probs = np.full((T, K), np.nan)
    means = np.full((T, K, d), np.nan)
    covariances = np.full((T, K, d, d), np.nan)
    weights = np.full((T, K), np.nan)
    transition = np.full((T, K, K), np.nan)
    log_likelihood = np.full(T, np.nan)
    n_relabels = np.zeros(T, dtype=np.int64)
    switched = np.zeros(T, dtype=bool)

    stepper = _RegimeStepper(
        K,
        refit_every=refit_every,
        p_switch=p_switch,
        switch_threshold=switch_threshold,
        reg_covar=reg_covar,
        warm_iter=warm_iter,
        init_iter=init_iter,
        tol=tol,
        anchor_rate=(2.0 / (window + 1) if anchor_rate is None else anchor_rate),
        anchor_min_separation=anchor_min_separation,
        anchor_strength=anchor_strength,
        trans_pseudo=trans_pseudo,
        min_state_mass=min_state_mass,
    )
    for t in range(window - 1, T):
        out = stepper.step(X[t - window + 1 : t + 1], X[t])
        labels[t] = out["label"]
        probs[t] = out["probs"]
        raw_probs[t] = out["raw"]
        means[t] = out["means"]
        covariances[t] = out["covariances"]
        weights[t] = out["weights"]
        transition[t] = out["transition"]
        log_likelihood[t] = out["ll"]
        n_relabels[t] = out["n_relabels"]
        switched[t] = out["switched"]

    result = RollingRegimesResult(
        labels=labels,
        probs=probs,
        raw_probs=raw_probs,
        means=means,
        covariances=covariances,
        weights=weights,
        transition=transition,
        log_likelihood=log_likelihood,
        n_relabels=n_relabels,
        switched=switched,
        window=window,
        n_regimes=n_regimes,
    )
    return result, stepper
