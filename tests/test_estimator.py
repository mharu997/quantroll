import numpy as np
import pandas as pd
import pytest

from quantroll import RollingPCA


def _structured_df(T=260, d=6, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((3, d))
    factors = rng.standard_normal((T, 3)) * np.array([5.0, 2.5, 1.2])
    X = factors @ loadings + rng.standard_normal((T, d)) * 0.1
    idx = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(X, index=idx, columns=[f"a{i}" for i in range(d)])


def test_pandas_roundtrip_labels():
    df = _structured_df()
    pca = RollingPCA(n_components=2, window=60).fit(df)
    Z = pca.transform()
    assert isinstance(Z, pd.DataFrame)
    assert Z.index.equals(df.index)
    assert list(Z.columns) == ["pc1", "pc2"]
    assert pca.feature_names_in_ == list(df.columns)
    # numpy path produces identical numbers
    pca_np = RollingPCA(n_components=2, window=60).fit(df.to_numpy())
    np.testing.assert_allclose(Z.to_numpy(), pca_np.transform(), equal_nan=True)


def test_streaming_update_equals_batch_fit():
    df = _structured_df(T=300, seed=1)
    full = RollingPCA(n_components=3, window=60).fit(df)
    part = RollingPCA(n_components=3, window=60).fit(df.iloc[:200])
    part.update(df.iloc[200:])

    Zf = full.transform()
    Zp = part.transform()
    assert Zp.index.equals(Zf.index)
    np.testing.assert_allclose(
        Zp.to_numpy(), Zf.to_numpy(), rtol=1e-6, atol=1e-8, equal_nan=True
    )
    np.testing.assert_allclose(
        part.eigenvalues_, full.eigenvalues_, rtol=1e-6, atol=1e-10, equal_nan=True
    )
    np.testing.assert_allclose(
        part.components_[60:], full.components_[60:], rtol=1e-5, atol=1e-6
    )


def test_streaming_update_row_by_row():
    df = _structured_df(T=140, seed=2)
    full = RollingPCA(n_components=2, window=50).fit(df)
    part = RollingPCA(n_components=2, window=50).fit(df.iloc[:100])
    for i in range(100, 140):
        part.update(df.iloc[i : i + 1])
    np.testing.assert_allclose(
        part.transform().to_numpy(),
        full.transform().to_numpy(),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )


def test_transform_new_data_uses_latest_basis():
    df = _structured_df()
    pca = RollingPCA(n_components=2, window=60).fit(df)
    new = df.iloc[-5:]
    Z = pca.transform(new)
    V, mean = pca._latest_basis()
    expected = (new.to_numpy() - mean) @ V
    np.testing.assert_allclose(Z.to_numpy(), expected)
    assert Z.index.equals(new.index)


def test_corr_mode_streaming_consistency():
    df = _structured_df(T=200, seed=3)
    full = RollingPCA(n_components=2, window=40, corr=True).fit(df)
    part = RollingPCA(n_components=2, window=40, corr=True).fit(df.iloc[:150])
    part.update(df.iloc[150:])
    np.testing.assert_allclose(
        part.transform().to_numpy(),
        full.transform().to_numpy(),
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )


def test_errors():
    df = _structured_df()
    pca = RollingPCA(n_components=2, window=60)
    with pytest.raises(RuntimeError, match="not fitted"):
        pca.transform()
    pca.fit(df)
    with pytest.raises(ValueError, match="expected 6 features"):
        pca.update(np.zeros((1, 4)))
    bad = df.iloc[-1:].rename(columns={"a0": "zz"})
    with pytest.raises(ValueError, match="feature names"):
        pca.update(bad)
