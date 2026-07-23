import numpy as np
import pandas as pd
import pytest

from quantroll import RollingRegimes, rolling_regimes, simulate
from quantroll.core.rolling_regimes import match_components


def _sim(T=2500, seed=0):
    return simulate.regime_returns(
        T, n_assets=4, p_stay=0.995, bull=(8e-4, 0.008), bear=(-6e-4, 0.02), seed=seed
    )


def _accuracy(labels, states, K):
    """Best label->state mapping accuracy over valid rows."""
    valid = labels >= 0
    lab, st = labels[valid], states[valid]
    best = 0.0
    from itertools import permutations

    for perm in permutations(range(K)):
        mapped = np.array([perm[v] for v in lab])
        best = max(best, float(np.mean(mapped == st)))
    return best


def test_recovers_simulated_regimes():
    # window spans ~3-4 regime spells — the documented requirement
    R, states = _sim()
    res = rolling_regimes(R, window=750, n_regimes=2)
    assert _accuracy(res.labels, states, 2) > 0.9
    assert res.n_relabels.sum() == 0  # identities never scrambled


def test_switching_tracks_true_transitions():
    R, states = _sim(seed=2)
    res = rolling_regimes(R, window=750, n_regimes=2)
    v = res.labels >= 0
    true_trans = (np.diff(states[v]) != 0).sum()
    assert 1 <= res.switched.sum() <= 3 * true_trans + 5


def test_threshold_hysteresis_monotone():
    R, _ = _sim(T=2000, seed=3)
    loose = rolling_regimes(R, window=600, n_regimes=2, switch_threshold=0.0)
    tight = rolling_regimes(R, window=600, n_regimes=2, switch_threshold=0.9)
    assert tight.switched.sum() <= loose.switched.sum()


def test_deterministic_reproducibility():
    R, _ = _sim(T=1500, seed=1)
    a = rolling_regimes(R, window=600, n_regimes=2)
    b = rolling_regimes(R, window=600, n_regimes=2)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.probs, b.probs)


def test_learned_transitions_row_stochastic_and_sticky():
    R, _ = _sim(seed=0)
    res = rolling_regimes(R, window=750, n_regimes=2)
    valid = res.labels >= 0
    sums = res.transition[valid].sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    assert (res.transition[valid] > 0).all()
    # persistent simulated regimes -> sticky learned matrix at the end
    assert res.transition[-1].diagonal().min() > 0.5
    # stationary frequencies are a probability vector
    np.testing.assert_allclose(res.weights[valid].sum(axis=1), 1.0, atol=1e-9)


def test_match_components_realigns_shuffled_mixture():
    rng = np.random.default_rng(4)
    K, d = 3, 2
    means = rng.standard_normal((K, d)) * 3
    covs = np.stack([np.eye(d) * s for s in (0.5, 1.0, 2.0)])
    shuffle = np.array([2, 0, 1])
    cur_means, cur_covs = means[shuffle], covs[shuffle]
    perm = match_components(means, covs, cur_means, cur_covs)
    # applying perm restores original identities exactly
    np.testing.assert_allclose(cur_means[perm], means)
    np.testing.assert_allclose(cur_covs[perm], covs)


def test_warmup_and_shapes():
    R, _ = _sim(T=400, seed=5)
    res = rolling_regimes(R, window=100, n_regimes=3)
    assert (res.labels[:99] == -1).all()
    assert np.isnan(res.probs[:99]).all()
    assert (res.labels[99:] >= 0).all()
    assert res.means.shape == (400, 3, 4)
    assert res.covariances.shape == (400, 3, 4, 4)
    valid_probs = res.probs[99:]
    np.testing.assert_allclose(valid_probs.sum(axis=1), 1.0, atol=1e-6)


def test_streaming_update_equals_batch():
    R, _ = _sim(T=900, seed=6)
    df = pd.DataFrame(R, columns=list("wxyz"), index=pd.RangeIndex(900))
    full = RollingRegimes(n_regimes=2, window=300).fit(df)
    part = RollingRegimes(n_regimes=2, window=300).fit(df.iloc[:700])
    part.update(df.iloc[700:])

    np.testing.assert_array_equal(part.labels_, full.labels_)
    np.testing.assert_allclose(part.probs_, full.probs_, rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(part.means_, full.means_, rtol=1e-9, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        part.transition_, full.transition_, rtol=1e-9, atol=1e-12, equal_nan=True
    )


def test_estimator_pandas_and_predict():
    R, _ = _sim(T=1200, seed=7)
    idx = pd.bdate_range("2021-01-04", periods=1200)
    df = pd.DataFrame(R, index=idx, columns=[f"a{i}" for i in range(4)])
    rr = RollingRegimes(n_regimes=2, window=400).fit(df)

    labels = rr.predict()
    assert isinstance(labels, pd.Series)
    assert labels.index.equals(idx)

    probs = rr.predict_proba()
    assert list(probs.columns) == ["regime_0", "regime_1"]

    new = df.iloc[-3:]
    p_new = rr.predict_proba(new)
    assert p_new.shape == (3, 2)
    np.testing.assert_allclose(p_new.sum(axis=1), 1.0, atol=1e-9)
    l_new = rr.predict(new)
    assert set(np.unique(l_new)) <= {0, 1}

    summary = rr.regime_summary()
    assert summary.shape[0] == 2
    assert "share" in summary.columns and "avg_spell" in summary.columns
    np.testing.assert_allclose(summary["share"].sum(), 1.0, atol=1e-9)


def test_calm_regime_has_lower_vol_identity():
    R, states = _sim(T=2500, seed=8)
    rr = RollingRegimes(n_regimes=2, window=750).fit(R)
    summary = rr.regime_summary()
    vols = summary[[c for c in summary.columns if c.startswith("std_")]].mean(axis=1)
    # dispersion-quantile init seeds regime_0 on the calmest observations
    assert vols.loc["regime_0"] < vols.loc["regime_1"]


def test_refit_cadence_still_labels_every_step():
    R, states = _sim(seed=9)
    res = rolling_regimes(R, window=750, n_regimes=2, refit_every=5)
    assert (res.labels[749:] >= 0).all()
    assert _accuracy(res.labels, states, 2) > 0.85


def test_input_validation():
    R, _ = _sim(T=300, seed=10)
    with pytest.raises(ValueError, match="2-D"):
        rolling_regimes(R[:, 0], window=50)
    with pytest.raises(ValueError, match="window"):
        rolling_regimes(R, window=400)
    with pytest.raises(ValueError, match="p_switch"):
        rolling_regimes(R, window=100, p_switch=1.0)
    rr = RollingRegimes(n_regimes=2, window=100).fit(R)
    with pytest.raises(ValueError, match="expected 4 features"):
        rr.update(np.zeros((1, 3)))
