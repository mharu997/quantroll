import numpy as np
import pytest

from quantroll import Scorecard
from quantroll.measures import metrics as m


def test_constant_positive_returns():
    r = np.full(252, 0.01)
    np.testing.assert_allclose(m.total_return(r), 1.01**252 - 1, rtol=1e-12)
    np.testing.assert_allclose(m.cagr(r), 1.01**252 - 1, rtol=1e-10)
    assert m.ann_vol(r) == 0.0
    assert np.isnan(m.sharpe(r))  # zero vol -> undefined, not inf
    assert m.max_drawdown(r) == 0.0
    assert m.hit_rate(r) == 1.0


def test_known_drawdown_path():
    r = np.array([0.10, -0.50, 0.10])
    np.testing.assert_allclose(m.max_drawdown(r), -0.5, rtol=1e-12)
    dd = m.drawdown_series(r)
    np.testing.assert_allclose(dd, [0.0, -0.5, -0.45], rtol=1e-12)


def test_var_cvar():
    r = np.concatenate([np.full(95, 0.01), np.full(5, -0.10)])
    assert m.var_hist(r, alpha=0.05) == pytest.approx(0.10, abs=1e-9)
    assert m.cvar_hist(r, alpha=0.05) == pytest.approx(0.10, abs=1e-9)


def test_relative_measures_leveraged_benchmark():
    rng = np.random.default_rng(0)
    b = rng.normal(3e-4, 0.01, 5000)
    r = 2.0 * b
    assert m.beta(r, b) == pytest.approx(2.0, abs=1e-9)
    assert m.correlation(r, b) == pytest.approx(1.0, abs=1e-9)
    assert m.alpha(r, b) == pytest.approx(0.0, abs=1e-9)
    assert m.tracking_error(r, b) == pytest.approx(m.ann_vol(b), rel=1e-9)


def test_capture_ratios_identical_series():
    rng = np.random.default_rng(1)
    b = rng.normal(0, 0.01, 1000)
    assert m.up_capture(b, b) == pytest.approx(1.0, rel=1e-9)
    assert m.down_capture(b, b) == pytest.approx(1.0, rel=1e-9)


def test_2d_reduction_shapes():
    rng = np.random.default_rng(2)
    R = rng.normal(0, 0.01, (500, 4))
    b = rng.normal(0, 0.01, 500)
    assert m.sharpe(R).shape == (4,)
    assert m.max_drawdown(R).shape == (4,)
    assert m.beta(R, b).shape == (4,)
    assert m.up_capture(R, b).shape == (4,)


def test_nan_awareness():
    r = np.array([0.01, np.nan, 0.02, -0.01])
    assert np.isfinite(m.total_return(r))
    assert np.isfinite(m.ann_vol(r))
    assert m.hit_rate(r) == pytest.approx(2 / 3)


def test_rolling_measures():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 300)
    rv = m.rolling_vol(r, 60)
    assert rv.shape == (300,)
    assert np.isnan(rv[:59]).all()
    np.testing.assert_allclose(rv[100], np.std(r[41:101], ddof=1) * np.sqrt(252), rtol=1e-10)


# ---------------------------------------------------------------- scorecard


def _fund_universe(seed=0):
    rng = np.random.default_rng(seed)
    target = rng.normal(4e-4, 0.010, 2500)
    same = target.copy()
    better = target + 3e-4
    worse = target - 3e-4 + rng.normal(0, 0.004, 2500)
    R = np.column_stack([same, better, worse])
    return R, target


def test_scorecard_ranks_obvious_universe():
    R, target = _fund_universe()
    res = Scorecard().compare(R, target, names=["same", "better", "worse"])
    assert res.rank() == ["better", "same", "worse"]
    assert res.composite[1] > res.composite[0] > res.composite[2]
    # identical asset scores ~0 against its own target
    assert abs(res.composite[0]) < 0.05


def test_scorecard_explainability_surface():
    R, target = _fund_universe(seed=1)
    res = Scorecard().compare(R, target, names=["same", "better", "worse"])
    df = res.to_frame()
    assert list(df.columns) == ["target", "same", "better", "worse"]
    sf = res.scores_frame()
    assert "composite" in sf.index and "dispersion" in sf.index
    assert (res.dispersion >= 0).all()


def test_scorecard_weights_and_validation():
    R, target = _fund_universe(seed=2)
    only_risk = Scorecard(weights={"ann_vol": 1.0, "cagr": 0.0, "sharpe": 0.0,
                                   "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                                   "cvar_95": 0.0, "skewness": 0.0, "hit_rate": 0.0,
                                   "tracking_error": 0.0, "information_ratio": 0.0,
                                   "up_capture": 0.0, "down_capture": 0.0})
    res = only_risk.compare(R[:, 2], target, names=["worse"])
    # noisy 'worse' fund has higher vol -> negative risk-only composite
    assert res.composite[0] < 0
    with pytest.raises(ValueError, match="weights"):
        Scorecard(weights={"cagr": -1.0})
    with pytest.raises(ValueError, match="length mismatch"):
        Scorecard().compare(R[:100], target)


def test_scorecard_pandas_input():
    import pandas as pd

    R, target = _fund_universe(seed=3)
    df = pd.DataFrame(R, columns=["same", "better", "worse"])
    res = Scorecard().compare(df, pd.Series(target))
    assert res.assets == ["same", "better", "worse"]
