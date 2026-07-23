import numpy as np

from quantroll import rolling_pca


def _wide_data(T=800, d=60, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((5, d))
    factors = rng.standard_normal((T, 5)) * np.array([6.0, 4.0, 2.5, 1.5, 1.0])
    return factors @ loadings + rng.standard_normal((T, d)) * 0.3


def test_subspace_matches_exact_solver():
    X = _wide_data()
    exact = rolling_pca(X, window=120, n_components=5, solver="eigh")
    fast = rolling_pca(X, window=120, n_components=5, solver="subspace")
    np.testing.assert_allclose(
        fast.eigenvalues, exact.eigenvalues, rtol=1e-6, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        fast.projections, exact.projections, rtol=1e-5, atol=1e-7, equal_nan=True
    )
    np.testing.assert_allclose(
        np.abs(fast.components), np.abs(exact.components), rtol=1e-5, atol=1e-6, equal_nan=True
    )


def test_subspace_matches_exact_in_corr_mode():
    X = _wide_data(seed=1)
    exact = rolling_pca(X, window=100, n_components=4, solver="eigh", corr=True)
    fast = rolling_pca(X, window=100, n_components=4, solver="subspace", corr=True)
    np.testing.assert_allclose(
        fast.eigenvalues, exact.eigenvalues, rtol=1e-6, atol=1e-9, equal_nan=True
    )


def test_auto_solver_selection_runs():
    X = _wide_data(seed=2)
    res = rolling_pca(X, window=100, n_components=3, solver="auto")
    valid = ~np.isnan(res.eigenvalues[:, 0])
    assert valid.sum() == 800 - 99
