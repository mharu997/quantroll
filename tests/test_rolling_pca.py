import numpy as np
import pytest

from quantroll import rolling_pca
from quantroll.simulate import rotating_cloud


def _structured_data(T=300, d=8, seed=0):
    """Factor data with a well-separated spectrum (stable eigen-identities)."""
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((3, d))
    factors = rng.standard_normal((T, 3)) * np.array([5.0, 2.5, 1.2])
    return factors @ loadings + rng.standard_normal((T, d)) * 0.1


def test_matches_direct_per_window_recompute():
    X = _structured_data()
    W, p = 60, 3
    res = rolling_pca(X, window=W, n_components=p, align=False)
    for t in [W - 1, W, 150, 299]:
        win = X[t - W + 1 : t + 1]
        cov = np.cov(win, rowvar=False, ddof=1)
        vals = np.linalg.eigvalsh(cov)[::-1][:p]
        np.testing.assert_allclose(res.eigenvalues[t], vals, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(res.means[t], win.mean(axis=0), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(res.total_variance[t], np.trace(cov), rtol=1e-9)


def test_refresh_cadence_does_not_change_results():
    X = _structured_data(seed=1)
    a = rolling_pca(X, window=40, n_components=2, refresh_every=1)
    b = rolling_pca(X, window=40, n_components=2, refresh_every=10**9)
    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-8, equal_nan=True)
    np.testing.assert_allclose(a.projections, b.projections, rtol=1e-6, atol=1e-8, equal_nan=True)


def test_warmup_is_nan_and_shapes():
    X = _structured_data(T=100)
    res = rolling_pca(X, window=30, n_components=2)
    assert np.isnan(res.eigenvalues[:29]).all()
    assert np.isnan(res.projections[:29]).all()
    assert not np.isnan(res.eigenvalues[29:]).any()
    assert res.components.shape == (100, 2, 8)
    assert res.projections.shape == (100, 2)


def test_aligned_pc1_has_no_sign_flips_but_raw_does():
    X, _ = rotating_cloud(n_steps=1200, theta_per_step=0.02, sigmas=(5.0, 0.3), seed=7)
    W = 60
    raw = rolling_pca(X, window=W, n_components=2, align=False)
    stable = rolling_pca(X, window=W, n_components=2)

    v_raw = raw.components[W - 1 :, 0, :]
    v_st = stable.components[W - 1 :, 0, :]
    dots_raw = np.einsum("td,td->t", v_raw[1:], v_raw[:-1])
    dots_st = np.einsum("td,td->t", v_st[1:], v_st[:-1])

    assert (dots_raw < 0).any(), "rotating data should provoke raw sign flips"
    assert (dots_st > 0).all(), "aligned PC1 must never flip sign"
    assert stable.n_flips[W:].sum() > 0  # flips were detected and corrected


def test_aligned_pc1_tracks_true_rotating_direction():
    X, true_dir = rotating_cloud(n_steps=1200, theta_per_step=0.02, sigmas=(5.0, 0.3), seed=8)
    W = 60
    res = rolling_pca(X, window=W, n_components=1)
    ts = np.arange(W - 1, 1200)
    mid = ts - (W - 1) // 2
    v = res.components[ts, 0, :]
    track = np.abs(np.einsum("td,td->t", v, true_dir[mid]))
    assert np.median(track) > 0.98
    assert np.mean(track > 0.9) > 0.95


def test_panel_average_covariance():
    rng = np.random.default_rng(2)
    F, T, d, W, p = 3, 120, 5, 40, 2
    X = rng.standard_normal((F, T, d)).cumsum(axis=1)
    res = rolling_pca(X, window=W, n_components=p, align=False)
    t = 100
    covs = [np.cov(X[f, t - W + 1 : t + 1], rowvar=False, ddof=1) for f in range(F)]
    avg = np.mean(covs, axis=0)
    vals = np.linalg.eigvalsh(avg)[::-1][:p]
    np.testing.assert_allclose(res.eigenvalues[t], vals, rtol=1e-9)
    assert res.projections.shape == (F, T, p)
    assert res.means.shape == (F, T, d)


def test_corr_mode_unit_diagonal_trace():
    X = _structured_data(seed=3)
    res = rolling_pca(X, window=50, n_components=3, corr=True)
    valid = ~np.isnan(res.total_variance)
    np.testing.assert_allclose(res.total_variance[valid], 8.0, rtol=1e-9)
    t = 200
    win = X[t - 49 : t + 1]
    corr = np.corrcoef(win, rowvar=False)
    vals = np.linalg.eigvalsh(corr)[::-1][:3]
    np.testing.assert_allclose(res.eigenvalues[t], vals, rtol=1e-8)


def test_projection_definition():
    X = _structured_data(seed=4)
    res = rolling_pca(X, window=60, n_components=2)
    t = 150
    V = res.components[t].T  # (d, p)
    expected = (X[t] - res.means[t]) @ V
    np.testing.assert_allclose(res.projections[t], expected, rtol=1e-10)


def test_input_validation():
    X = np.zeros((50, 4))
    with pytest.raises(ValueError, match="n_components"):
        rolling_pca(X, window=20, n_components=5)
    with pytest.raises(ValueError, match="window"):
        rolling_pca(X, window=60, n_components=2)
    with pytest.raises(ValueError, match="2-D"):
        rolling_pca(np.zeros(50), window=10, n_components=1)
