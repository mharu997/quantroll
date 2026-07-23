"""Variable universes: NaN feature spaces (PCA) and unbalanced panels (K-Means)."""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from quantroll import RollingKMeans, RollingPCA, rolling_pca, simulate
from quantroll.core.rolling_pca import _rolling_pca_masked

W = 60


def _masked_data(T=600, d=6, seed=0):
    """Factor data where asset 5 launches at t=300 and asset 0 dies at t=450."""
    rng = np.random.default_rng(seed)
    loadings = rng.standard_normal((3, d))
    factors = rng.standard_normal((T, 3)) * np.array([5.0, 2.5, 1.2])
    X = factors @ loadings + rng.standard_normal((T, d)) * 0.1
    X[:300, 5] = np.nan
    X[450:, 0] = np.nan
    return X


# ------------------------------------------------------------- PCA masked


def test_masked_matches_direct_submatrix_eigh():
    X = _masked_data()
    raw = rolling_pca(X, window=W, n_components=3, align=False)
    aligned = rolling_pca(X, window=W, n_components=3)
    for t, active in [(200, [0, 1, 2, 3, 4]), (400, [0, 1, 2, 3, 4, 5]),
                      (500, [1, 2, 3, 4, 5])]:
        win = X[t - W + 1 : t + 1][:, active]
        assert np.isfinite(win).all()
        cov = np.cov(win, rowvar=False, ddof=1)
        vals = np.linalg.eigvalsh(cov)[::-1][:3]
        np.testing.assert_allclose(raw.eigenvalues[t], vals, rtol=1e-8)
        np.testing.assert_allclose(raw.total_variance[t], np.trace(cov), rtol=1e-9)
        # alignment may permute identities (crossings), never change the set
        np.testing.assert_allclose(
            np.sort(aligned.eigenvalues[t]), np.sort(vals), rtol=1e-8
        )
        inactive = sorted(set(range(6)) - set(active))
        assert np.isnan(raw.components[t][:, inactive]).all()
        assert np.isfinite(raw.components[t][:, active]).all()
        np.testing.assert_allclose(raw.means[t, active], win.mean(axis=0), rtol=1e-9)
        assert np.isnan(raw.means[t, inactive]).all()


def test_masked_active_set_transitions():
    X = _masked_data()
    res = rolling_pca(X, window=W, n_components=3)
    # asset 5 becomes active once its full window is observed
    assert np.isnan(res.components[300 + W - 2, 0, 5])
    assert np.isfinite(res.components[300 + W - 1, 0, 5])
    # asset 0 becomes inactive as soon as a NaN enters the window
    assert np.isfinite(res.components[449, 0, 0])
    assert np.isnan(res.components[450, 0, 0])


def test_masked_alignment_no_flips_across_universe_changes():
    X = _masked_data(seed=1)
    res = rolling_pca(X, window=W, n_components=2)
    V = res.components[:, 0, :]  # PC1 over time, (T, d) with NaNs
    valid = np.flatnonzero(~np.isnan(res.eigenvalues[:, 0]))
    for a, b in zip(valid[:-1], valid[1:]):
        shared = np.isfinite(V[a]) & np.isfinite(V[b])
        va, vb = V[a][shared], V[b][shared]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > 1e-9 and nb > 1e-9:
            assert va @ vb / (na * nb) > 0, f"sign flip between t={a} and t={b}"
    assert np.nanmedian(res.similarity[valid[1:]]) > 0.99


def test_masked_projection_definition():
    X = _masked_data(seed=2)
    res = rolling_pca(X, window=W, n_components=2)
    t = 350
    active = np.isfinite(res.means[t])
    V = res.components[t][:, active].T
    expected = (X[t, active] - res.means[t, active]) @ V
    np.testing.assert_allclose(res.projections[t], expected, rtol=1e-10)


def test_masked_path_equals_dense_on_dense_data():
    rng = np.random.default_rng(3)
    loadings = rng.standard_normal((3, 8))
    X = rng.standard_normal((300, 3)) * np.array([5, 2.5, 1.2]) @ loadings
    X = X + rng.standard_normal((300, 8)) * 0.1
    dense = rolling_pca(X, window=50, n_components=3, solver="eigh")
    masked = _rolling_pca_masked(
        X, 50, 3, matching="hungarian", corr=False, ddof=1,
        refresh_every=1024, align=True,
    )
    np.testing.assert_allclose(
        masked.eigenvalues, dense.eigenvalues, rtol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        masked.projections, dense.projections, rtol=1e-7, atol=1e-9, equal_nan=True
    )


def test_masked_too_few_active_columns_gives_nan():
    X = _masked_data(seed=4)
    X[:200, 3:] = np.nan  # early on only 3 columns are observed
    res = rolling_pca(X, window=W, n_components=4)
    assert np.isnan(res.eigenvalues[200]).all()  # 3 active < 4 components
    assert np.isfinite(res.eigenvalues[400]).all()


def test_masked_estimator_roundtrip_and_update_guard():
    X = _masked_data(seed=5)
    idx = pd.bdate_range("2020-01-01", periods=600)
    cols = [f"a{i}" for i in range(6)]
    df = pd.DataFrame(X, index=idx, columns=cols)
    pca = RollingPCA(n_components=2, window=W).fit(df)
    Z = pca.transform()
    assert isinstance(Z, pd.DataFrame) and Z.index.equals(idx)
    with pytest.raises(NotImplementedError, match="dense"):
        pca.update(df.iloc[-1:])
    Znew = pca.transform(df.iloc[-3:])
    assert np.isfinite(Znew.to_numpy()).all()
    with pytest.raises(ValueError, match="2-D"):
        rolling_pca(np.full((2, 100, 4), np.nan), window=20, n_components=2)


# --------------------------------------------------------- K-Means variable


def _unbalanced_df(T=60, seed=6):
    """3 drifting clusters; 9 extra entities launch at t=20, 6 die at t=40."""
    X, membership, _ = simulate.drifting_blobs(
        T, n_entities=60, n_clusters=3, drift=0.02, seed=seed
    )
    Xx, mem_x, _ = simulate.drifting_blobs(
        T, n_entities=9, n_clusters=3, drift=0.02, seed=seed
    )
    times = pd.bdate_range("2024-01-01", periods=T)
    base = [f"fund_{i:02d}" for i in range(60)]
    extra = [f"new_{i}" for i in range(9)]
    frames = []
    for t in range(T):
        ents, rows = list(base), [X[t]]
        if t >= 20:
            ents = ents + extra
            rows.append(Xx[t])
        vals = np.concatenate(rows)
        if t >= 40:  # first 6 base entities die
            ents, vals = ents[6:], vals[6:]
        frames.append(
            pd.DataFrame(
                vals,
                index=pd.MultiIndex.from_product([[times[t]], ents]),
                columns=["mom", "vol"],
            )
        )
    truth = dict(zip(base + extra, list(membership) + list(mem_x)))
    return pd.concat(frames), truth, times


def _mapped_accuracy(labels, truth_arr, K):
    best = 0.0
    for perm in permutations(range(K)):
        mapped = np.array([perm[v] if v >= 0 else -1 for v in labels])
        ok = labels >= 0
        best = max(best, float(np.mean(mapped[ok] == truth_arr[ok])))
    return best


def test_unbalanced_panel_fit():
    df, truth, times = _unbalanced_df()
    km = RollingKMeans(n_clusters=3).fit(df)
    lf = km.labels_frame()
    assert lf.shape == (60, 69)
    # absence pattern: extras are -1 before t=20, dead base entities -1 from t=40
    assert (lf.loc[times[0], [f"new_{i}" for i in range(9)]] == -1).all()
    assert (lf.loc[times[45], [f"fund_{i:02d}" for i in range(6)]] == -1).all()
    assert (lf.loc[times[30]] >= 0).all()
    # correct grouping at every period under one global mapping
    truth_arr = np.array([truth[e] for e in lf.columns])
    for t in [0, 25, 59]:
        assert _mapped_accuracy(lf.iloc[t].to_numpy(), truth_arr, 3) == 1.0
    assert km.n_relabels_.sum() == 0
    assert not km.switched_.any()
    summary = km.cluster_summary()
    np.testing.assert_allclose(summary["share"].sum(), 1.0, atol=1e-9)


def test_variable_streaming_equals_batch():
    df, _, times = _unbalanced_df(seed=7)
    full = RollingKMeans(n_clusters=3).fit(df)
    head = df[df.index.get_level_values(0) < times[35]]
    tail = df[df.index.get_level_values(0) >= times[35]]
    part = RollingKMeans(n_clusters=3).fit(head)
    part.update(tail)
    np.testing.assert_array_equal(part.labels_, full.labels_)
    np.testing.assert_allclose(part.centroids_, full.centroids_)
    np.testing.assert_array_equal(part.switched_, full.switched_)
    assert part.entity_names_ == full.entity_names_


def test_update_with_new_entities_extends_universe():
    df, _, times = _unbalanced_df(seed=8)
    early = df[df.index.get_level_values(0) < times[20]]  # extras not yet born
    km = RollingKMeans(n_clusters=3).fit(early)
    assert len(km.entity_names_) == 60
    chunk = df[
        (df.index.get_level_values(0) >= times[20])
        & (df.index.get_level_values(0) < times[25])
    ]
    km.update(chunk)
    assert len(km.entity_names_) == 69
    assert (km.labels_[:20, 60:] == -1).all()  # history padded for newcomers
    assert (km.labels_[-1, 60:] >= 0).all()
    # single cross-section update (plain entity index)
    last = df.loc[times[25]]
    km.update(last)
    assert km.labels_.shape[0] == 26


def test_variable_validation():
    df, _, _ = _unbalanced_df(seed=9)
    km = RollingKMeans(n_clusters=3).fit(df)
    with pytest.raises(ValueError, match="DataFrame indexed by entity"):
        km.update(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="feature names"):
        km.update(pd.DataFrame(np.zeros((5, 2)), columns=["x", "y"]))
    tiny = pd.DataFrame(
        np.zeros((2, 2)),
        index=pd.MultiIndex.from_product([["2024-01-01"], ["a", "b"]]),
        columns=["mom", "vol"],
    )
    with pytest.raises(ValueError, match="fewer than n_clusters"):
        RollingKMeans(n_clusters=3).fit(tiny)
    bad = pd.DataFrame(
        [[1.0, np.nan]] * 5,
        index=pd.MultiIndex.from_product([["2024-01-01"], list("abcde")]),
        columns=["mom", "vol"],
    )
    with pytest.raises(ValueError, match="NaN features"):
        RollingKMeans(n_clusters=2).fit(bad)
