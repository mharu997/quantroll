from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from quantroll import RollingKMeans, rolling_kmeans, simulate
from quantroll.core._kmeans_kernels import assign_multi, lloyd, maxmin_init
from quantroll.core.rolling_kmeans import match_centroids


def _mapped_accuracy(labels, truth, K):
    best = 0.0
    for perm in permutations(range(K)):
        mapped = np.array([perm[v] for v in labels])
        best = max(best, float(np.mean(mapped == truth)))
    return best


# ------------------------------------------------------------------ kernels


def test_lloyd_recovers_separated_blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    truth = np.repeat(np.arange(3), 50)
    X = centers[truth] + rng.normal(0, 0.4, (150, 2))
    centroids = maxmin_init(X, 3)
    labels, inertia, n_iter = lloyd(X, centroids, 300, 1e-8)
    assert _mapped_accuracy(labels, truth, 3) == 1.0
    # centroids near true centers (up to order)
    d = np.abs(centroids[:, None, :] - centers[None, :, :]).sum(axis=2).min(axis=1)
    assert (d < 0.5).all()


def test_assign_multi_reduces_to_plain_with_memory_one():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (40, 3))
    centroids = rng.normal(0, 1, (4, 3))
    bank = centroids[:, None, :].copy()
    labels_m, dist2_m = assign_multi(X, bank, np.ones(4, dtype=np.int64))
    labels_p = np.empty(40, dtype=np.int64)
    dist2_p = np.empty(40)
    from quantroll.core._kmeans_kernels import _assign

    _assign(X, centroids, labels_p, dist2_p)
    np.testing.assert_array_equal(labels_m, labels_p)
    np.testing.assert_allclose(dist2_m, dist2_p)


def test_assign_multi_nonlinear_memory_claims_point():
    # cluster 0 remembers an old centroid at the origin; cluster 1 is closer
    # to the point than cluster 0's current position
    bank = np.zeros((2, 2, 2))
    bank[0, 0] = [0.0, 0.0]  # old position of cluster 0
    bank[0, 1] = [4.0, 0.0]  # current position of cluster 0
    bank[1, 0] = [2.0, 0.0]  # cluster 1 (memory of one)
    point = np.array([[0.3, 0.0]])
    with_memory, _ = assign_multi(point, bank, np.array([2, 1], dtype=np.int64))
    bank_cur = np.zeros((2, 1, 2))
    bank_cur[0, 0] = [4.0, 0.0]
    bank_cur[1, 0] = [2.0, 0.0]
    current_only, _ = assign_multi(point, bank_cur, np.array([1, 1], dtype=np.int64))
    assert with_memory[0] == 0  # memory of the origin claims the point
    assert current_only[0] == 1  # without memory it goes to cluster 1


def test_match_centroids_recovers_shuffle():
    rng = np.random.default_rng(2)
    prev = rng.normal(0, 5, (4, 3))
    shuffle = np.array([3, 0, 2, 1])
    cur = prev[shuffle] + rng.normal(0, 0.01, (4, 3))
    perm = match_centroids(prev, cur)
    np.testing.assert_allclose(cur[perm], prev, atol=0.05)


# --------------------------------------------------------------- rolling


def test_static_data_is_perfectly_stable():
    rng = np.random.default_rng(3)
    centers = np.array([[0.0, 0.0], [7.0, 0.0], [0.0, 7.0]])
    truth = np.repeat(np.arange(3), 40)
    cross = centers[truth] + rng.normal(0, 0.4, (120, 2))
    X = np.repeat(cross[None, :, :], 50, axis=0)  # identical every period

    res = rolling_kmeans(X, n_clusters=3)
    assert res.n_relabels.sum() == 0
    assert not res.switched[1:].any()
    for t in range(1, 50):
        np.testing.assert_array_equal(res.labels[t], res.labels[0])
        np.testing.assert_allclose(res.centroids[t], res.centroids[0], atol=1e-12)
    assert res.n_iter[1:].max() <= 2  # warm starts converge immediately


def test_naive_reinit_flips_but_rolling_does_not():
    X, membership, _ = simulate.drifting_blobs(60, seed=4)
    res = rolling_kmeans(X, n_clusters=3)
    rolling_changes = res.switched[1:].sum()

    rng = np.random.default_rng(5)
    naive_changes = 0
    prev = None
    for t in range(60):
        idx = rng.choice(X.shape[1], 3, replace=False)
        centroids = X[t][idx].copy()
        labels, _, _ = lloyd(X[t], centroids, 300, 1e-8)
        if prev is not None:
            naive_changes += (labels != prev).sum()
        prev = labels
    assert rolling_changes == 0  # entities never change cluster
    assert naive_changes > 0  # naive per-period clustering flips labels


def test_tracks_drifting_blobs_with_stable_identities():
    X, membership, centers = simulate.drifting_blobs(150, drift=0.03, seed=6)
    res = rolling_kmeans(X, n_clusters=3)
    assert res.n_relabels.sum() == 0
    assert not res.switched[1:].any()
    # per-period accuracy vs true membership is perfect under one global map
    acc = _mapped_accuracy(res.labels[-1], membership, 3)
    assert acc == 1.0
    # matched centroids track the (rotating) true centers
    for t in [0, 75, 149]:
        d = np.linalg.norm(res.centroids[t][:, None] - centers[t][None, :], axis=2)
        assert d.min(axis=1).max() < 0.5


def test_streaming_update_equals_batch():
    X, _, _ = simulate.drifting_blobs(100, seed=7)
    full = RollingKMeans(n_clusters=3).fit(X)
    part = RollingKMeans(n_clusters=3).fit(X[:60])
    part.update(X[60:])
    np.testing.assert_array_equal(part.labels_, full.labels_)
    np.testing.assert_allclose(part.centroids_, full.centroids_)
    np.testing.assert_allclose(part.inertia_, full.inertia_)


def test_deterministic_and_seeded_kmeanspp():
    X, _, _ = simulate.drifting_blobs(30, seed=8)
    a = rolling_kmeans(X, n_clusters=3)
    b = rolling_kmeans(X, n_clusters=3)
    np.testing.assert_array_equal(a.labels, b.labels)
    c = rolling_kmeans(X, n_clusters=3, init="k-means++", seed=0)
    d = rolling_kmeans(X, n_clusters=3, init="k-means++", seed=0)
    np.testing.assert_array_equal(c.labels, d.labels)


def test_centroid_memory_recaptures_vacated_ground():
    # Cluster A's entities march from the origin to (8, 0); cluster B stays
    # at (4, 6). A point near the origin afterwards: A's memory claims it,
    # while memoryless assignment gives it to whichever current centroid is
    # nearer.
    T, per = 40, 30
    rng = np.random.default_rng(9)
    a_centers = np.linspace([0.0, 0.0], [8.0, 0.0], T)
    X = np.empty((T, 2 * per, 2))
    for t in range(T):
        X[t, :per] = a_centers[t] + rng.normal(0, 0.3, (per, 2))
        X[t, per:] = np.array([4.0, 6.0]) + rng.normal(0, 0.3, (per, 2))

    probe = np.array([[0.5, 0.0]])
    km_mem = RollingKMeans(n_clusters=2, centroid_memory=T).fit(X)
    km_plain = RollingKMeans(n_clusters=2, centroid_memory=1).fit(X)

    lab_mem = km_mem.predict(probe)[0]
    lab_plain = km_plain.predict(probe)[0]
    a_id = km_mem.labels_[0, 0]  # identity of cluster A (stable through time)
    assert lab_mem == a_id  # A's remembered trail claims the origin
    # plain assignment: current A is at (8,0), farther than B at (4,6)
    assert lab_plain != km_plain.labels_[0, 0]


def test_more_clusters_than_groups_no_crash():
    rng = np.random.default_rng(10)
    cross = np.concatenate(
        [rng.normal(0, 0.3, (30, 2)), rng.normal(5, 0.3, (30, 2))]
    )
    X = np.repeat(cross[None], 10, axis=0)
    res = rolling_kmeans(X, n_clusters=4)
    assert set(np.unique(res.labels)) <= {0, 1, 2, 3}
    assert np.isfinite(res.inertia).all()


def test_pandas_multiindex_roundtrip():
    X, membership, _ = simulate.drifting_blobs(20, n_entities=30, seed=11)
    times = pd.bdate_range("2024-01-01", periods=20)
    entities = [f"fund_{i:02d}" for i in range(30)]
    idx = pd.MultiIndex.from_product([times, entities], names=["date", "fund"])
    df = pd.DataFrame(X.reshape(-1, 2), index=idx, columns=["mom", "vol"])

    km = RollingKMeans(n_clusters=3).fit(df)
    lf = km.labels_frame()
    assert list(lf.columns) == entities
    assert list(lf.index) == list(times)
    km_np = RollingKMeans(n_clusters=3).fit(X)
    np.testing.assert_array_equal(lf.to_numpy(), km_np.labels_)

    summary = km.cluster_summary()
    assert "centroid_mom" in summary.columns
    np.testing.assert_allclose(summary["share"].sum(), 1.0, atol=1e-9)

    cross = df.loc[times[-1]]
    s = km.predict(cross)
    assert isinstance(s, pd.Series)
    assert s.index.equals(cross.index)
    dists = km.transform(cross)
    assert dists.shape == (30, 3)


def test_validation_errors():
    X, _, _ = simulate.drifting_blobs(10, n_entities=20, seed=12)
    with pytest.raises(ValueError, match="3-D"):
        rolling_kmeans(X[0], n_clusters=3)
    with pytest.raises(ValueError, match="n_clusters"):
        rolling_kmeans(X, n_clusters=21)
    with pytest.raises(ValueError, match="centroid_memory"):
        rolling_kmeans(X, n_clusters=3, centroid_memory=0)
    with pytest.raises(ValueError, match="init"):
        rolling_kmeans(X, n_clusters=3, init="random")
    km = RollingKMeans(n_clusters=3).fit(X)
    with pytest.raises(ValueError, match="expected 20 entities"):
        km.update(np.zeros((5, 2)))
    with pytest.raises(RuntimeError, match="not fitted"):
        RollingKMeans().predict(np.zeros((3, 2)))
