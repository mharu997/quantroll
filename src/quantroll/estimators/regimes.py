"""Scikit-learn-style estimator for temporally stable rolling regime detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._compat import unwrap, wrap_1d, wrap_2d
from ..core._gmm_kernels import posterior
from ..core.rolling_regimes import RollingRegimesResult, _rolling_regimes_impl

__all__ = ["RollingRegimes"]


class RollingRegimes:
    """Regime detection with stable labels over rolling windows.

    A rolling Gaussian mixture refit by warm-started EM. Regime *identities*
    live in slowly-updated anchor parameters (Hungarian-matched at each
    refit, frozen while the window fit is degenerate); regime *probabilities*
    come from an HMM-style forward filter with a sticky transition prior; the
    discrete call adds a hysteresis threshold. Together: no label switching,
    no flicker, and identities that survive long single-regime spells.

    Parameters
    ----------
    n_regimes : int
        Number of regimes K. Component 0 is seeded on the calmest
        observations of the first window, K-1 on the most turbulent.
    window : int
        Rolling window length in time steps. Must span several regime cycles
        — a regime absent from the window cannot be learned from it (default
        1008 ≈ four trading years of daily data).
    refit_every : int
        Refit cadence (1 = every step). Probabilities still update every step.
    p_switch : float in (0, 1)
        Prior per-step regime switch probability of the forward filter
        (expected spell length ≈ 1/p_switch).
    switch_threshold : float
        Filtered probability required for a challenger regime to take over.
    reg_covar : float
        Diagonal floor added to component covariances.
    warm_iter, init_iter : int
        EM iteration caps for warm refits / the initial fit.
    tol : float
        EM convergence tolerance.

    Attributes
    ----------
    labels_ : (T,) int regime calls (-1 during warm-up).
    probs_ : (T, K) forward-filtered regime probabilities.
    means_, covariances_, weights_ : regime parameters over time.
    transition_ : (T, K, K) learned HMM transition matrices over time.
    n_relabels_, switched_ : stability diagnostics.
    result_ : RollingRegimesResult from the last fit/update.

    Examples
    --------
    >>> from quantroll import RollingRegimes, simulate
    >>> R, states = simulate.regime_returns(3000, seed=0)
    >>> rr = RollingRegimes(n_regimes=2, window=1000).fit(R)
    >>> rr.predict()[-5:]                      # stable regime calls
    array([...])
    >>> rr.update(new_rows)                    # doctest: +SKIP
    >>> rr.regime_summary()                    # doctest: +SKIP
    """

    def __init__(
        self,
        n_regimes: int = 2,
        window: int = 1008,
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
    ) -> None:
        self.n_regimes = n_regimes
        self.window = window
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

    # ------------------------------------------------------------------ fit

    def fit(self, X: Any, y: Any = None) -> RollingRegimes:
        """Run rolling regime detection over the full history ``X``."""
        values, index, columns, kind = unwrap(X)
        if values.ndim != 2:
            raise ValueError("RollingRegimes expects 2-D input (T, d)")
        result, stepper = _rolling_regimes_impl(
            values,
            self.window,
            self.n_regimes,
            refit_every=self.refit_every,
            p_switch=self.p_switch,
            switch_threshold=self.switch_threshold,
            reg_covar=self.reg_covar,
            warm_iter=self.warm_iter,
            init_iter=self.init_iter,
            tol=self.tol,
            anchor_rate=self.anchor_rate,
            anchor_min_separation=self.anchor_min_separation,
            anchor_strength=self.anchor_strength,
        )
        self.result_: RollingRegimesResult = result
        self._stepper = stepper
        self._kind = kind
        self._index = index
        self.feature_names_in_ = columns
        self._X = values
        buffer = values[-self.window :].copy()
        self._buffer = buffer
        self._pos = 0
        return self

    # ------------------------------------------------------------- predict

    def predict(self, X: Any = None) -> Any:
        """Regime calls. No argument: the fitted history (with hysteresis).

        With ``X``: static maximum-posterior labels of new rows under the
        latest parameters — no temporal smoothing or hysteresis is applied
        and the model does not advance (use :meth:`update` for that).
        """
        self._check_fitted()
        if X is None:
            return wrap_1d(self.result_.labels, self._index, "regime", self._kind)
        probs = self.predict_proba(X)
        values = probs.to_numpy() if hasattr(probs, "to_numpy") else probs
        _, index, _, kind = unwrap(X)
        return wrap_1d(values.argmax(axis=1), index, "regime", kind)

    def predict_proba(self, X: Any = None) -> Any:
        """Regime probabilities. No argument: filtered fitted history; with
        ``X``: static posteriors of new rows under the latest parameters."""
        self._check_fitted()
        names = [f"regime_{k}" for k in range(self.n_regimes)]
        if X is None:
            return wrap_2d(self.result_.probs, self._index, names, self._kind)
        values, index, _, kind = unwrap(X)
        if values.ndim == 1:
            values = values[None, :]
        st = self._stepper
        raw = posterior(
            np.ascontiguousarray(values), st.means, st.covariances,
            np.full(self.n_regimes, 1.0 / self.n_regimes), self.reg_covar,
        )
        return wrap_2d(raw, index, names, kind)

    def fit_predict(self, X: Any, y: Any = None) -> Any:
        return self.fit(X).predict()

    # -------------------------------------------------------------- update

    def update(self, X_new: Any) -> RollingRegimes:
        """Advance the model with new observation(s), streaming.

        Identical arithmetic to batch ``fit`` — the same stepper continues —
        so ``fit(head); update(tail)`` equals ``fit(full)`` exactly.
        """
        self._check_fitted()
        values, index, columns, _ = unwrap(X_new)
        if values.ndim == 1:
            values = values[None, :]
        d = self._buffer.shape[1]
        if values.shape[1] != d:
            raise ValueError(f"expected {d} features, got {values.shape[1]}")
        if columns is not None and self.feature_names_in_ is not None:
            if columns != self.feature_names_in_:
                raise ValueError("feature names do not match fitted columns")

        r = self.result_
        appended = {
            "labels": [], "probs": [], "raw_probs": [], "means": [],
            "covariances": [], "weights": [], "transition": [],
            "log_likelihood": [], "n_relabels": [], "switched": [],
        }
        for row in np.ascontiguousarray(values, dtype=np.float64):
            self._buffer[self._pos] = row
            self._pos = (self._pos + 1) % self.window
            win = np.ascontiguousarray(
                np.concatenate([self._buffer[self._pos :], self._buffer[: self._pos]])
            )
            out = self._stepper.step(win, win[-1])
            appended["labels"].append(out["label"])
            appended["probs"].append(out["probs"])
            appended["raw_probs"].append(out["raw"])
            appended["means"].append(out["means"])
            appended["covariances"].append(out["covariances"])
            appended["weights"].append(out["weights"])
            appended["transition"].append(out["transition"])
            appended["log_likelihood"].append(out["ll"])
            appended["n_relabels"].append(out["n_relabels"])
            appended["switched"].append(out["switched"])

        self.result_ = RollingRegimesResult(
            labels=np.concatenate([r.labels, np.asarray(appended["labels"], dtype=np.int64)]),
            probs=np.concatenate([r.probs, np.asarray(appended["probs"])]),
            raw_probs=np.concatenate([r.raw_probs, np.asarray(appended["raw_probs"])]),
            means=np.concatenate([r.means, np.asarray(appended["means"])]),
            covariances=np.concatenate([r.covariances, np.asarray(appended["covariances"])]),
            weights=np.concatenate([r.weights, np.asarray(appended["weights"])]),
            transition=np.concatenate([r.transition, np.asarray(appended["transition"])]),
            log_likelihood=np.concatenate(
                [r.log_likelihood, np.asarray(appended["log_likelihood"])]
            ),
            n_relabels=np.concatenate(
                [r.n_relabels, np.asarray(appended["n_relabels"], dtype=np.int64)]
            ),
            switched=np.concatenate([r.switched, np.asarray(appended["switched"], dtype=bool)]),
            window=r.window,
            n_regimes=r.n_regimes,
        )
        self._X = np.concatenate([self._X, values])
        if index is not None and self._index is not None:
            self._index = self._index.append(index)
        elif self._index is not None:
            self._index = None
        return self

    # ------------------------------------------------------------ summary

    def regime_summary(self):
        """Per-regime table: share of time, average spell length, and
        per-feature mean/std of the observations assigned to each regime."""
        self._check_fitted()
        import pandas as pd

        labels = self.result_.labels
        valid = labels >= 0
        cols = self.feature_names_in_ or [
            f"f{j}" for j in range(self._X.shape[1])
        ]
        rows = []
        for k in range(self.n_regimes):
            mask = labels == k
            share = mask.sum() / max(valid.sum(), 1)
            spells = _spell_lengths(labels, k)
            row = {
                "share": share,
                "avg_spell": float(np.mean(spells)) if spells else 0.0,
                "n_spells": len(spells),
            }
            sel = self._X[mask]
            for j, c in enumerate(cols):
                row[f"mean_{c}"] = sel[:, j].mean() if sel.size else np.nan
                row[f"std_{c}"] = sel[:, j].std(ddof=1) if sel.shape[0] > 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows, index=[f"regime_{k}" for k in range(self.n_regimes)])

    # ----------------------------------------------------------- accessors

    @property
    def labels_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.labels

    @property
    def probs_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.probs

    @property
    def means_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.means

    @property
    def covariances_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.covariances

    @property
    def weights_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.weights

    @property
    def transition_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.transition

    @property
    def n_relabels_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.n_relabels

    @property
    def switched_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.switched

    def __repr__(self) -> str:
        return f"RollingRegimes(n_regimes={self.n_regimes}, window={self.window})"

    def _check_fitted(self) -> None:
        if not hasattr(self, "result_"):
            raise RuntimeError("RollingRegimes is not fitted; call fit(X) first")


def _spell_lengths(labels: np.ndarray, k: int) -> list[int]:
    spells = []
    run = 0
    for v in labels:
        if v == k:
            run += 1
        elif run:
            spells.append(run)
            run = 0
    if run:
        spells.append(run)
    return spells
