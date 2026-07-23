import numpy as np
import pandas as pd
import pytest

from quantroll import RollingEmbedding, rolling_embedding, simulate
from quantroll.core.rolling_embedding import pca_embedding, procrustes_align


def test_procrustes_recovers_rigid_motion():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((40, 2))
    theta = 1.1
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    F = np.array([[1.0, 0.0], [0.0, -1.0]])  # reflection
    cur = ref @ (R @ F) + np.array([3.0, -7.0])
    aligned, disparity, _ = procrustes_align(ref, cur)
    np.testing.assert_allclose(aligned, ref, atol=1e-10)
    assert disparity < 1e-10


def test_procrustes_scale_option():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((30, 2))
    cur = ref * 3.5 + 1.0
    aligned_no, disp_no, _ = procrustes_align(ref, cur, scale=False)
    aligned_sc, disp_sc, _ = procrustes_align(ref, cur, scale=True)
    assert disp_sc < 1e-10
    assert disp_no > disp_sc  # scale change is signal unless requested


def test_static_data_gives_static_map():
    rng = np.random.default_rng(2)
    cross = rng.standard_normal((80, 5))
    X = np.repeat(cross[None], 20, axis=0)
    res = rolling_embedding(X, backend="pca")
    for t in range(1, 20):
        np.testing.assert_allclose(res.embeddings[t], res.embeddings[0], atol=1e-9)
    assert np.nanmax(res.displacement[1:]) < 1e-9
    assert np.nanmax(res.disparity[1:]) < 1e-9


def test_alignment_beats_naive_on_rotating_structure():
    # rotating blobs make raw per-period PCA reflect/rotate arbitrarily;
    # Procrustes-aligned maps move only as much as the data does
    X, membership, _ = simulate.drifting_blobs(80, drift=0.06, seed=3)
    res = rolling_embedding(X, backend="pca")
    aligned_move = np.nanmedian(res.displacement[1:])

    naive_prev = None
    naive_moves = []
    for t in range(80):
        emb = pca_embedding(X[t], None, 2)
        if naive_prev is not None:
            naive_moves.append(
                np.median(np.sqrt(((emb - naive_prev) ** 2).sum(axis=1)))
            )
        naive_prev = emb
    assert aligned_move < 0.25 * np.median(naive_moves)
    # cluster structure preserved in the map: tight within, far between
    last = res.embeddings[-1]
    within = np.mean([last[membership == k].std(axis=0).mean() for k in range(3)])
    centers = np.stack([last[membership == k].mean(axis=0) for k in range(3)])
    between = np.linalg.norm(centers[0] - centers[1])
    assert between > 3 * within


def test_streaming_update_equals_batch_fixed():
    X, _, _ = simulate.drifting_blobs(50, seed=4)
    full = RollingEmbedding(backend="pca").fit(X)
    part = RollingEmbedding(backend="pca").fit(X[:30])
    part.update(X[30:])
    np.testing.assert_allclose(
        part.embeddings_, full.embeddings_, rtol=1e-9, atol=1e-9
    )
    np.testing.assert_allclose(
        part.disparity_, full.disparity_, rtol=1e-9, atol=1e-12, equal_nan=True
    )


def _unbalanced_df(T=30, seed=5):
    X, membership, _ = simulate.drifting_blobs(T, n_entities=45, seed=seed)
    times = pd.bdate_range("2024-01-01", periods=T)
    names = [f"fund_{i:02d}" for i in range(45)]
    frames = []
    for t in range(T):
        ents, vals = names, X[t]
        if t < 10:  # last 5 funds not yet launched
            ents, vals = names[:40], X[t, :40]
        frames.append(
            pd.DataFrame(
                vals,
                index=pd.MultiIndex.from_product([[times[t]], ents]),
                columns=["f1", "f2"],
            )
        )
    return pd.concat(frames), membership, times


def test_unbalanced_panel_and_trajectory():
    df, membership, times = _unbalanced_df()
    emb = RollingEmbedding(backend="pca").fit(df)
    assert emb.embeddings_.shape == (30, 45, 2)
    assert np.isnan(emb.embeddings_[:10, 40:]).all()  # pre-launch
    assert np.isfinite(emb.embeddings_[10:]).all()

    traj = emb.trajectory("fund_00")
    assert traj.shape == (30, 2)
    assert list(traj.index) == list(times)
    frame = emb.embedding_frame(-1)
    assert frame.shape == (45, 2)
    # newcomers land near their cluster-mates on arrival
    m40 = membership[40]
    mates = [i for i in range(40) if membership[i] == m40]
    d_mates = np.linalg.norm(
        emb.embeddings_[10, mates] - emb.embeddings_[10, 40], axis=1
    ).min()
    others = [i for i in range(40) if membership[i] != m40]
    d_others = np.linalg.norm(
        emb.embeddings_[10, others] - emb.embeddings_[10, 40], axis=1
    ).min()
    assert d_mates < d_others


def test_unbalanced_streaming_and_new_entities():
    df, _, times = _unbalanced_df(seed=6)
    full = RollingEmbedding(backend="pca").fit(df)
    head = df[df.index.get_level_values(0) < times[15]]
    tail = df[df.index.get_level_values(0) >= times[15]]
    part = RollingEmbedding(backend="pca").fit(head)
    part.update(tail)
    np.testing.assert_allclose(
        part.embeddings_, full.embeddings_, rtol=1e-9, atol=1e-9, equal_nan=True
    )
    assert part.entity_names_ == full.entity_names_


def test_custom_callable_backend():
    calls = []

    def toy(X, init):
        calls.append(init is not None)
        return X[:, :2] * 2.0

    X, _, _ = simulate.drifting_blobs(5, seed=7)
    res = rolling_embedding(X, backend=toy)
    assert res.embeddings.shape == (5, 90, 2)
    assert calls == [False, True, True, True, True]  # warm starts passed


def test_validation():
    X, _, _ = simulate.drifting_blobs(5, seed=8)
    with pytest.raises(ValueError, match="3-D"):
        rolling_embedding(X[0])
    with pytest.raises(ValueError, match="backend"):
        rolling_embedding(X, backend="tsne")
    with pytest.raises(ValueError, match="n_components"):
        rolling_embedding(X, n_components=5)
    bad = lambda X, init: X[:, :1]  # noqa: E731 — wrong output shape
    with pytest.raises(ValueError, match="backend returned shape"):
        rolling_embedding(X, backend=bad)


def test_umap_backend_stable_maps():
    pytest.importorskip("umap")
    X, membership, _ = simulate.drifting_blobs(6, n_entities=60, seed=9)
    emb = RollingEmbedding(
        backend="umap", seed=0, backend_kwargs={"n_neighbors": 10, "n_epochs": 60}
    ).fit(X)
    assert np.isfinite(emb.embeddings_).all()
    # cluster structure survives in the map
    last = emb.embeddings_[-1]
    within = np.mean([last[membership == k].std(axis=0).mean() for k in range(3)])
    centers = np.stack([last[membership == k].mean(axis=0) for k in range(3)])
    between = min(
        np.linalg.norm(centers[a] - centers[b]) for a in range(3) for b in range(a)
    )
    assert between > 2 * within
    # warm start + Procrustes keep consecutive maps close
    assert np.nanmedian(emb.displacement_[1:]) < 0.5 * between
