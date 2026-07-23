"""Target-relative scorecards: compare assets to a target across many measures.

Portfolios are usually built on one objective but judged on many. A scorecard
makes the multi-measure comparison explicit and explainable: every measure is
computed for each asset and for the target, normalized into an
orientation-adjusted relative score (positive = better than target), and
aggregated into a weighted composite plus a dispersion statistic showing how
unevenly the asset beats or misses the target across measures.

This is an independent, transparent construction inspired by the unifying
performance-measure framework concept of Hirsa, Ding & Malhotra (2023); it
does not reproduce their proprietary formulas.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._compat import unwrap
from . import metrics as m

__all__ = ["Measure", "Scorecard", "ScorecardResult", "DEFAULT_MEASURES"]

_EPS = 1e-12


@dataclass(frozen=True)
class Measure:
    """A named measure with orientation and (optional) benchmark dependence.

    ``higher_is_better`` sets the score orientation. ``relative`` measures are
    called as ``fn(asset_returns, target_returns, **kwargs)``.
    """

    fn: Callable[..., np.ndarray]
    higher_is_better: bool = True
    relative: bool = False
    kwargs: tuple = ()

    def compute(self, returns, target) -> np.ndarray:
        kw = dict(self.kwargs)
        if self.relative:
            return self.fn(returns, target, **kw)
        return self.fn(returns, **kw)


DEFAULT_MEASURES: dict[str, Measure] = {
    "cagr": Measure(m.cagr),
    "ann_vol": Measure(m.ann_vol, higher_is_better=False),
    "sharpe": Measure(m.sharpe),
    "sortino": Measure(m.sortino),
    "max_drawdown": Measure(m.max_drawdown),  # less negative is better
    "calmar": Measure(m.calmar),
    "cvar_95": Measure(m.cvar_hist, higher_is_better=False),
    "skewness": Measure(m.skewness),
    "hit_rate": Measure(m.hit_rate),
    "tracking_error": Measure(m.tracking_error, higher_is_better=False, relative=True),
    "information_ratio": Measure(m.information_ratio, relative=True),
    "up_capture": Measure(m.up_capture, relative=True),
    "down_capture": Measure(m.down_capture, higher_is_better=False, relative=True),
}


@dataclass(frozen=True)
class ScorecardResult:
    """Everything needed to explain a comparison, not just rank it."""

    measures: list[str]
    assets: list[str]
    values: np.ndarray
    """(M, K) raw measure values per asset."""
    target_values: np.ndarray
    """(M,) raw measure values of the target (NaN for relative measures)."""
    scores: np.ndarray
    """(M, K) orientation-adjusted relative scores (positive = beats target)."""
    weights: np.ndarray
    """(M,) weights used in the composite."""
    composite: np.ndarray
    """(K,) weighted mean score per asset."""
    dispersion: np.ndarray
    """(K,) weighted std of scores per asset — how unevenly the target is beaten."""

    def to_frame(self):
        """Return a pandas DataFrame (measures × assets + target column)."""
        import pandas as pd

        df = pd.DataFrame(self.values, index=self.measures, columns=self.assets)
        df.insert(0, "target", self.target_values)
        return df

    def scores_frame(self):
        """Scores as a pandas DataFrame with composite and dispersion rows."""
        import pandas as pd

        df = pd.DataFrame(self.scores, index=self.measures, columns=self.assets)
        df.loc["composite"] = self.composite
        df.loc["dispersion"] = self.dispersion
        return df

    def rank(self) -> list[str]:
        """Assets ordered best-first by composite score."""
        order = np.argsort(-self.composite)
        return [self.assets[i] for i in order]


class Scorecard:
    """Multi-measure comparison of assets against a target return series.

    Parameters
    ----------
    measures : dict[str, Measure] | None
        Measures to evaluate (defaults to :data:`DEFAULT_MEASURES`).
    weights : dict[str, float] | None
        Composite weights per measure name; unspecified measures get weight 1.
        Weights are normalized to sum to 1.
    periods_per_year : int
        Annualization frequency passed to measures that accept it.

    Examples
    --------
    >>> import numpy as np
    >>> from quantroll import Scorecard
    >>> rng = np.random.default_rng(0)
    >>> target = rng.normal(4e-4, 0.01, 1000)
    >>> funds = target[:, None] + rng.normal(1e-4, 0.004, (1000, 3))
    >>> result = Scorecard().compare(funds, target, names=["A", "B", "C"])
    >>> result.rank()  # doctest: +SKIP
    ['B', 'A', 'C']
    """

    _PPY_AWARE = {
        "cagr", "ann_vol", "sharpe", "sortino", "calmar",
        "tracking_error", "information_ratio",
    }

    def __init__(
        self,
        measures: dict[str, Measure] | None = None,
        weights: dict[str, float] | None = None,
        periods_per_year: int = 252,
    ) -> None:
        self.measures = dict(measures) if measures is not None else dict(DEFAULT_MEASURES)
        raw_w = np.array(
            [(weights or {}).get(name, 1.0) for name in self.measures], dtype=np.float64
        )
        if (raw_w < 0).any() or raw_w.sum() <= 0:
            raise ValueError("weights must be non-negative and not all zero")
        self.weights = raw_w / raw_w.sum()
        self.periods_per_year = periods_per_year

    def compare(self, returns: Any, target: Any, names: list[str] | None = None) -> ScorecardResult:
        """Score each asset column of ``returns`` against ``target`` returns."""
        R, _, columns, _ = unwrap(returns)
        tgt, _, _, _ = unwrap(target)
        if R.ndim == 1:
            R = R[:, None]
        if tgt.ndim != 1:
            raise ValueError("target must be a single return series")
        if R.shape[0] != tgt.shape[0]:
            raise ValueError(
                f"length mismatch: assets have {R.shape[0]} rows, target {tgt.shape[0]}"
            )
        K = R.shape[1]
        assets = names or columns or [f"asset_{k + 1}" for k in range(K)]
        if len(assets) != K:
            raise ValueError(f"expected {K} names, got {len(assets)}")

        names_m = list(self.measures)
        M = len(names_m)
        values = np.full((M, K), np.nan)
        target_values = np.full(M, np.nan)
        scores = np.full((M, K), np.nan)

        for i, name in enumerate(names_m):
            meas = self.measures[name]
            kw = dict(meas.kwargs)
            if name in self._PPY_AWARE and "periods_per_year" not in kw:
                kw["periods_per_year"] = self.periods_per_year
            meas = Measure(meas.fn, meas.higher_is_better, meas.relative, tuple(kw.items()))
            v = np.atleast_1d(np.asarray(meas.compute(R, tgt), dtype=np.float64))
            values[i] = v
            if meas.relative:
                t_val = _relative_neutral(name)
            else:
                t_val = float(np.asarray(meas.compute(tgt[:, None], tgt)).ravel()[0])
            target_values[i] = t_val
            scores[i] = _score(v, t_val, meas.higher_is_better)

        w = self.weights[:, None]
        with np.errstate(invalid="ignore"):
            wsum = np.nansum(np.where(np.isnan(scores), 0.0, w), axis=0)
            composite = np.nansum(np.where(np.isnan(scores), 0.0, scores * w), axis=0)
            composite = composite / np.where(wsum > 0, wsum, np.nan)
            dev = scores - composite[None, :]
            dispersion = np.sqrt(
                np.nansum(np.where(np.isnan(scores), 0.0, w * dev**2), axis=0)
                / np.where(wsum > 0, wsum, np.nan)
            )

        return ScorecardResult(
            measures=names_m,
            assets=list(assets),
            values=values,
            target_values=target_values,
            scores=scores,
            weights=self.weights.copy(),
            composite=composite,
            dispersion=dispersion,
        )


def _relative_neutral(name: str) -> float:
    """Neutral target value for measures defined *against* the target."""
    if name in ("up_capture", "down_capture"):
        return 1.0
    return 0.0


def _score(v: np.ndarray, target: float, higher_is_better: bool) -> np.ndarray:
    """Orientation-adjusted relative difference vs. the target value."""
    denom = abs(target)
    if not np.isfinite(denom) or denom < _EPS:
        finite = v[np.isfinite(v)]
        denom = float(np.mean(np.abs(finite))) if finite.size else 1.0
        denom = denom if denom > _EPS else 1.0
    raw = (v - target) / denom
    return raw if higher_is_better else -raw
