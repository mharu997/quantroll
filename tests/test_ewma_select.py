"""EWMA-weighted rolling PCA and model-selection helpers."""

import numpy as np
import pandas as pd
import pytest

from quantroll import (
    RollingPCA,
    rolling_pca,
    select_n_clusters,
    select_n_regimes,
    simulate,
)


def _structured(T=400, d=8, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((3, d))
    factors = rng.standard_normal((T, 3)) * np.array([5.0, 2.5, 1.2])
    return factors @ loadings + rng.standard_normal((T, d)) * 0.1


# ------------------------------------------------------------------- EWMA


def test_ewma_matches_direct_weighted_covariance():
    X = _structured()
    hl = 40.0
    lam = 0.5 ** (1.0 / hl)
    res = rolling_pca(X, window=60, n_components=3, halflife=hl, align=False)
    for t in [59, 200, 399]:
        w = lam ** np.arange(t, -1, -1.0)
        mu = (w[:, None] * X[: t + 1]).sum(axis=0) / w.sum()
        Xc = X[: t + 1] - mu
        cov = (w[:, None] * Xc).T @ Xc / w.sum()
        vals = np.linalg.eigvalsh(cov)[::-1][:3]
        np.testing.assert_allclose(res.eigenvalues[t], vals, rtol=1e-8)
        np.testing.assert_allclose(res.means[t], mu, rtol=1e-9)
        np.testing.assert_allclose(res.total_variance[t], np.trace(cov), rtol=1e-9)
    assert np.isnan(res.eigenvalues[:59]).all()  # window = warm-up length


def test_ewma_expanding_equal_weight_limit():
    X = _structured(seed=1)
    res = rolling_pca(X, window=50, n_components=3, halflife=1e12, align=False)
    cov = np.cov(X, rowvar=False, ddof=0)
    vals = np.linalg.eigvalsh(cov)[::-1][:3]
    np.testing.assert_allclose(res.eigenvalues[-1], vals, rtol=1e-6)


def test_ewma_no_sign_flips_on_rotating_cloud():
    X, _ = simulate.rotating_cloud(n_steps=900, theta_per_step=0.02,
                                   sigmas=(5.0, 0.3), seed=2)
    res = rolling_pca(X, window=60, n_components=1, halflife=30.0)
    v = res.components[59:, 0, :]
    dots = np.einsum("td,td->t", v[1:], v[:-1])
    assert (dots > 0).all()


def test_ewma_streaming_equals_batch():
    X = _structured(seed=3)
    full = RollingPCA(n_components=3, window=60, halflife=40.0).fit(X)
    part = RollingPCA(n_components=3, window=60, halflife=40.0).fit(X[:300])
    part.update(X[300:])
    np.testing.assert_allclose(
        part.transform(), full.transform(), rtol=1e-6, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        part.eigenvalues_, full.eigenvalues_, rtol=1e-6, atol=1e-12, equal_nan=True
    )


def test_ewma_panel_average():
    rng = np.random.default_rng(4)
    F, T, d, hl = 3, 150, 5, 25.0
    lam = 0.5 ** (1.0 / hl)
    X = rng.standard_normal((F, T, d)).cumsum(axis=1)
    res = rolling_pca(X, window=40, n_components=2, halflife=hl, align=False)
    t = 120
    w = lam ** np.arange(t, -1, -1.0)
    covs = []
    for f in range(F):
        mu = (w[:, None] * X[f, : t + 1]).sum(axis=0) / w.sum()
        Xc = X[f, : t + 1] - mu
        covs.append((w[:, None] * Xc).T @ Xc / w.sum())
    vals = np.linalg.eigvalsh(np.mean(covs, axis=0))[::-1][:2]
    np.testing.assert_allclose(res.eigenvalues[t], vals, rtol=1e-8)


def test_ewma_corr_mode_and_validation():
    X = _structured(seed=5)
    res = rolling_pca(X, window=50, n_components=2, halflife=30.0, corr=True)
    valid = ~np.isnan(res.total_variance)
    np.testing.assert_allclose(res.total_variance[valid], 8.0, rtol=1e-9)
    with pytest.raises(ValueError, match="halflife"):
        rolling_pca(X, window=50, n_components=2, halflife=0.0)
    Xn = X.copy()
    Xn[:100, 0] = np.nan
    with pytest.raises(NotImplementedError, match="halflife"):
        rolling_pca(Xn, window=50, n_components=2, halflife=30.0)


# --------------------------------------------------------------- selection


def test_select_n_regimes_recovers_true_count():
    R, _ = simulate.regime_returns(2000, n_assets=4, p_stay=0.995,
                                   bull=(8e-4, 0.008), bear=(-6e-4, 0.02), seed=0)
    table = select_n_regimes(R, candidates=(1, 2, 3, 4))
    assert list(table.columns) == ["log_likelihood", "n_params", "aic", "bic"]
    assert int(table["bic"].idxmin()) == 2
    # parameter count formula: K=2, d=4 -> 1 + 2 + 8 + 20 = 31
    assert table.loc[2, "n_params"] == 31
    assert table.loc[2, "log_likelihood"] > table.loc[1, "log_likelihood"]


def test_select_n_clusters_recovers_true_count():
    X, _, _ = simulate.drifting_blobs(1, n_entities=150, n_clusters=3,
                                      sep=6.0, spread=0.5, seed=1)
    table = select_n_clusters(X[0], candidates=(2, 3, 4, 5, 6))
    assert int(table["silhouette"].idxmax()) == 3
    inertia = table["inertia"].to_numpy()
    assert (np.diff(inertia) < 1e-9).all()  # more clusters never fit worse


def test_select_accepts_pandas_and_validates():
    R, _ = simulate.regime_returns(600, n_assets=3, seed=2)
    df = pd.DataFrame(R, columns=list("abc"))
    table = select_n_regimes(df, candidates=(1, 2))
    assert table.index.name == "n_regimes"
    with pytest.raises(ValueError, match="2-D"):
        select_n_regimes(R[:, 0])
    X, _, _ = simulate.drifting_blobs(1, n_entities=50, seed=3)
    with pytest.raises(ValueError, match="invalid candidate"):
        select_n_clusters(X[0], candidates=(1,))
    with pytest.raises(ValueError, match="2-D"):
        select_n_clusters(X)
