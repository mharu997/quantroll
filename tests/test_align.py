import numpy as np
import pytest

from quantroll.core.align import align_sequence, canonical_signs, match_step


def _random_orthonormal(d, p, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q[:, :p]


def test_canonical_signs_max_entry_positive():
    V = _random_orthonormal(6, 3, seed=1)
    C = canonical_signs(V)
    idx = np.argmax(np.abs(C), axis=0)
    assert (C[idx, np.arange(3)] > 0).all()
    # only signs may differ
    np.testing.assert_allclose(np.abs(C), np.abs(V))


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_match_recovers_permutation_and_signs(method):
    prev = _random_orthonormal(8, 4, seed=2)
    true_perm = np.array([2, 0, 3, 1])
    true_signs = np.array([1.0, -1.0, -1.0, 1.0])
    # cur columns are shuffled/flipped copies of prev
    cur = prev[:, true_perm] * true_signs
    perm, signs, sim = match_step(prev, cur, method=method)
    aligned = cur[:, perm] * signs
    np.testing.assert_allclose(aligned, prev, atol=1e-12)
    np.testing.assert_allclose(sim, 1.0, atol=1e-12)


def test_match_with_noise_prefers_correct_slot():
    rng = np.random.default_rng(3)
    prev = _random_orthonormal(10, 3, seed=4)
    noise = rng.standard_normal((10, 3)) * 0.05
    cur = prev * np.array([-1.0, 1.0, -1.0]) + noise
    cur /= np.linalg.norm(cur, axis=0)
    perm, signs, _ = match_step(prev, cur)
    np.testing.assert_array_equal(perm, [0, 1, 2])
    np.testing.assert_allclose(signs, [-1.0, 1.0, -1.0])


def test_align_sequence_chains_and_skips_nan_warmup():
    d, p, T = 5, 2, 6
    base = _random_orthonormal(d, p, seed=5)
    eigvecs = np.full((T, d, p), np.nan)
    eigvals = np.full((T, p), np.nan)
    flip = np.array([1.0, -1.0])
    for t in range(2, T):
        s = flip if t % 2 else np.array([1.0, 1.0])
        eigvecs[t] = base * s
        eigvals[t] = [3.0, 1.0]
    res = align_sequence(eigvals, eigvecs)
    assert np.isnan(res.similarity[:3]).all()  # warmup rows + first valid step
    ref = res.eigvecs[2]
    for t in range(3, T):
        np.testing.assert_allclose(res.eigvecs[t], ref, atol=1e-12)
        np.testing.assert_allclose(res.similarity[t], 1.0, atol=1e-12)


def test_unknown_method_raises():
    V = _random_orthonormal(4, 2)
    with pytest.raises(ValueError, match="unknown matching"):
        match_step(V, V, method="nope")
