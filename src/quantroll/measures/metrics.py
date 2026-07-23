"""Vectorized, NaN-aware performance measures for return series.

All functions accept a 1-D ``(T,)`` or 2-D ``(T, K)`` array of simple periodic
returns (time along axis 0) and reduce over time, returning a scalar or
``(K,)`` vector. Benchmark-relative measures broadcast a 1-D benchmark against
2-D assets. Annualization uses ``periods_per_year`` (252 for daily data).

Conventions: ``max_drawdown`` is negative (a loss); ``var_hist`` / ``cvar_hist``
are positive loss magnitudes at the given confidence level.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "total_return",
    "cagr",
    "ann_return",
    "ann_vol",
    "downside_vol",
    "sharpe",
    "sortino",
    "calmar",
    "drawdown_series",
    "max_drawdown",
    "var_hist",
    "cvar_hist",
    "skewness",
    "kurtosis",
    "hit_rate",
    "gain_to_pain",
    "best_period",
    "worst_period",
    "active_return",
    "tracking_error",
    "information_ratio",
    "beta",
    "alpha",
    "up_capture",
    "down_capture",
    "correlation",
    "rolling_mean",
    "rolling_vol",
    "rolling_sharpe",
]

_EPS = 1e-12


def _r(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _count(r: np.ndarray) -> np.ndarray:
    return np.sum(~np.isnan(r), axis=0)


# ------------------------------------------------------------------ absolute

def total_return(returns) -> np.ndarray:
    """Compound total return over the sample."""
    r = _r(returns)
    return np.exp(np.nansum(np.log1p(r), axis=0)) - 1.0


def cagr(returns, periods_per_year: int = 252) -> np.ndarray:
    """Compound annual growth rate."""
    r = _r(returns)
    n = _count(r)
    growth = 1.0 + total_return(r)
    years = np.maximum(n, 1) / periods_per_year
    out = np.where(growth > 0, np.power(np.maximum(growth, _EPS), 1.0 / years) - 1.0, np.nan)
    return np.where(n > 0, out, np.nan)


def ann_return(returns, periods_per_year: int = 252) -> np.ndarray:
    """Arithmetic annualized mean return."""
    return np.nanmean(_r(returns), axis=0) * periods_per_year


def ann_vol(returns, periods_per_year: int = 252, ddof: int = 1) -> np.ndarray:
    """Annualized volatility of periodic returns."""
    return np.nanstd(_r(returns), axis=0, ddof=ddof) * np.sqrt(periods_per_year)


def downside_vol(returns, mar: float = 0.0, periods_per_year: int = 252) -> np.ndarray:
    """Annualized downside deviation below the minimum acceptable return."""
    r = _r(returns)
    dn = np.where(np.isnan(r), np.nan, np.minimum(r - mar, 0.0))
    return np.sqrt(np.nanmean(dn**2, axis=0)) * np.sqrt(periods_per_year)


def sharpe(returns, rf: float = 0.0, periods_per_year: int = 252) -> np.ndarray:
    """Annualized Sharpe ratio; ``rf`` is an annual risk-free rate."""
    r = _r(returns) - rf / periods_per_year
    vol = np.nanstd(r, axis=0, ddof=1)
    return np.nanmean(r, axis=0) * periods_per_year / np.where(vol > 0, vol, np.nan) / np.sqrt(
        periods_per_year
    )


def sortino(returns, mar: float = 0.0, periods_per_year: int = 252) -> np.ndarray:
    """Annualized Sortino ratio versus the minimum acceptable return (annual)."""
    r = _r(returns)
    mar_p = mar / periods_per_year
    dv = downside_vol(r, mar=mar_p, periods_per_year=periods_per_year)
    excess = (np.nanmean(r, axis=0) - mar_p) * periods_per_year
    return excess / np.where(dv > 0, dv, np.nan)


def drawdown_series(returns) -> np.ndarray:
    """Path of drawdowns from running peak wealth (same shape as input)."""
    r = _r(returns)
    log_wealth = np.nancumsum(np.log1p(r), axis=0)
    peak = np.maximum.accumulate(log_wealth, axis=0)
    return np.expm1(log_wealth - peak)


def max_drawdown(returns) -> np.ndarray:
    """Deepest peak-to-trough loss (negative number)."""
    return drawdown_series(returns).min(axis=0)


def calmar(returns, periods_per_year: int = 252) -> np.ndarray:
    """CAGR divided by absolute max drawdown."""
    mdd = np.abs(max_drawdown(returns))
    return cagr(returns, periods_per_year) / np.where(mdd > 0, mdd, np.nan)


def var_hist(returns, alpha: float = 0.05) -> np.ndarray:
    """Historical value-at-risk at level ``alpha`` (positive loss magnitude).

    Uses the empirical inverse-CDF quantile (historical-simulation convention),
    not interpolation, so with 5% of observations at -10% the 95% VaR is 10%.
    """
    return -np.nanquantile(_r(returns), alpha, axis=0, method="inverted_cdf")


def cvar_hist(returns, alpha: float = 0.05) -> np.ndarray:
    """Historical expected shortfall beyond the ``alpha`` quantile (positive)."""
    r = _r(returns)
    q = np.nanquantile(r, alpha, axis=0, method="inverted_cdf")
    tail = np.where(r <= q, r, np.nan)
    return -np.nanmean(tail, axis=0)


def skewness(returns) -> np.ndarray:
    r = _r(returns)
    m = np.nanmean(r, axis=0)
    s = np.nanstd(r, axis=0, ddof=0)
    return np.nanmean((r - m) ** 3, axis=0) / np.where(s > 0, s, np.nan) ** 3


def kurtosis(returns) -> np.ndarray:
    """Excess kurtosis (normal = 0)."""
    r = _r(returns)
    m = np.nanmean(r, axis=0)
    s = np.nanstd(r, axis=0, ddof=0)
    return np.nanmean((r - m) ** 4, axis=0) / np.where(s > 0, s, np.nan) ** 4 - 3.0


def hit_rate(returns) -> np.ndarray:
    """Share of periods with positive return."""
    r = _r(returns)
    return np.nansum(r > 0, axis=0) / _count(r)


def gain_to_pain(returns) -> np.ndarray:
    """Sum of gains divided by absolute sum of losses."""
    r = _r(returns)
    gains = np.nansum(np.where(r > 0, r, 0.0), axis=0)
    pains = np.abs(np.nansum(np.where(r < 0, r, 0.0), axis=0))
    return gains / np.where(pains > 0, pains, np.nan)


def best_period(returns) -> np.ndarray:
    return np.nanmax(_r(returns), axis=0)


def worst_period(returns) -> np.ndarray:
    return np.nanmin(_r(returns), axis=0)


# ------------------------------------------------------------------ relative

def _bench(returns, benchmark) -> tuple[np.ndarray, np.ndarray]:
    r, b = _r(returns), _r(benchmark)
    if r.ndim == 2 and b.ndim == 1:
        b = b[:, None]
    return r, b


def active_return(returns, benchmark, periods_per_year: int = 252) -> np.ndarray:
    """Annualized arithmetic mean of the active (excess) return."""
    r, b = _bench(returns, benchmark)
    out = np.nanmean(r - b, axis=0) * periods_per_year
    return out if out.ndim == 0 or out.shape[-1] > 1 else out.squeeze(-1)


def tracking_error(returns, benchmark, periods_per_year: int = 252) -> np.ndarray:
    r, b = _bench(returns, benchmark)
    return ann_vol(r - b, periods_per_year)


def information_ratio(returns, benchmark, periods_per_year: int = 252) -> np.ndarray:
    r, b = _bench(returns, benchmark)
    return sharpe(r - b, rf=0.0, periods_per_year=periods_per_year)


def beta(returns, benchmark) -> np.ndarray:
    r, b = _bench(returns, benchmark)
    rm = r - np.nanmean(r, axis=0)
    bm = b - np.nanmean(b, axis=0)
    cov = np.nanmean(rm * bm, axis=0)
    var = np.nanmean(bm * bm, axis=0)
    return cov / np.where(var > 0, var, np.nan)


def alpha(returns, benchmark, rf: float = 0.0, periods_per_year: int = 252) -> np.ndarray:
    """Annualized CAPM alpha versus the benchmark."""
    r, b = _bench(returns, benchmark)
    rf_p = rf / periods_per_year
    bet = beta(r, b if b.ndim == 1 else b[:, 0])
    return (
        (np.nanmean(r, axis=0) - rf_p) - bet * (np.nanmean(b, axis=0) - rf_p)
    ) * periods_per_year


def up_capture(returns, benchmark) -> np.ndarray:
    """Compound asset return over benchmark-up periods relative to benchmark's."""
    return _capture(returns, benchmark, up=True)


def down_capture(returns, benchmark) -> np.ndarray:
    """Compound asset return over benchmark-down periods relative to benchmark's."""
    return _capture(returns, benchmark, up=False)


def _capture(returns, benchmark, up: bool) -> np.ndarray:
    r, b = _bench(returns, benchmark)
    bench_1d = b if b.ndim == 1 else b[:, 0]
    mask = bench_1d > 0 if up else bench_1d < 0
    if not mask.any():
        shape = () if r.ndim == 1 else (r.shape[1],)
        return np.full(shape, np.nan)
    r_sel = r[mask]
    b_sel = bench_1d[mask]
    n = max(int(mask.sum()), 1)
    asset_g = np.exp(np.nansum(np.log1p(r_sel), axis=0) / n) - 1.0
    bench_g = np.exp(np.nansum(np.log1p(b_sel)) / n) - 1.0
    return asset_g / bench_g if abs(bench_g) > _EPS else np.full_like(asset_g, np.nan)


def correlation(returns, benchmark) -> np.ndarray:
    r, b = _bench(returns, benchmark)
    rm = r - np.nanmean(r, axis=0)
    bm = b - np.nanmean(b, axis=0)
    cov = np.nanmean(rm * bm, axis=0)
    denom = np.sqrt(np.nanmean(rm**2, axis=0) * np.nanmean(bm**2, axis=0))
    return cov / np.where(denom > 0, denom, np.nan)


# ------------------------------------------------------------------- rolling

def _sliding(r: np.ndarray, window: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(r, window, axis=0)


def _pad_warmup(out: np.ndarray, window: int) -> np.ndarray:
    pad_shape = (window - 1,) + out.shape[1:]
    return np.concatenate([np.full(pad_shape, np.nan), out], axis=0)


def rolling_mean(returns, window: int) -> np.ndarray:
    """Rolling mean with NaN warm-up padding to full length."""
    r = _r(returns)
    return _pad_warmup(np.nanmean(_sliding(r, window), axis=-1), window)


def rolling_vol(returns, window: int, periods_per_year: int = 252) -> np.ndarray:
    """Rolling annualized volatility."""
    r = _r(returns)
    out = np.nanstd(_sliding(r, window), axis=-1, ddof=1) * np.sqrt(periods_per_year)
    return _pad_warmup(out, window)


def rolling_sharpe(returns, window: int, rf: float = 0.0, periods_per_year: int = 252) -> np.ndarray:
    """Rolling annualized Sharpe ratio."""
    r = _r(returns) - rf / periods_per_year
    win = _sliding(r, window)
    mu = np.nanmean(win, axis=-1) * periods_per_year
    sd = np.nanstd(win, axis=-1, ddof=1) * np.sqrt(periods_per_year)
    return _pad_warmup(mu / np.where(sd > 0, sd, np.nan), window)
