"""Benchmark: quantroll rolling PCA vs naive per-window recomputation."""

import time

import numpy as np

from quantroll import RollingPCA, rolling_pca


def naive_rolling_pca(X, window, p):
    T, d = X.shape
    vals = np.full((T, p), np.nan)
    vecs = np.full((T, d, p), np.nan)
    for t in range(window - 1, T):
        cov = np.cov(X[t - window + 1 : t + 1], rowvar=False, ddof=1)
        ev, V = np.linalg.eigh(cov)
        vals[t] = ev[::-1][:p]
        vecs[t] = V[:, ::-1][:, :p]
    return vals, vecs


def timeit(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    for T, d, W, p in [(5000, 30, 252, 5), (5000, 100, 252, 5), (2000, 250, 252, 10)]:
        loadings = rng.standard_normal((p, d))
        factors = rng.standard_normal((T, p)) * np.linspace(5, 1, p)
        X = factors @ loadings + rng.standard_normal((T, d)) * 0.5

        rolling_pca(X[: W + 10], window=W, n_components=p)  # JIT warm-up

        t_naive = timeit(lambda: naive_rolling_pca(X, W, p), repeat=1)
        t_hung = timeit(lambda: rolling_pca(X, W, n_components=p, solver="eigh"))
        t_sub = timeit(lambda: rolling_pca(X, W, n_components=p, solver="subspace"))

        model = RollingPCA(n_components=p, window=W).fit(X[:-500])
        new = X[-500:]
        t0 = time.perf_counter()
        model.update(new)
        t_stream = time.perf_counter() - t0

        print(f"T={T} d={d} W={W} p={p}")
        print(f"  naive per-window recompute : {t_naive:8.3f}s  (no alignment)")
        print(f"  quantroll solver='eigh'    : {t_hung:8.3f}s  ({t_naive / t_hung:5.1f}x)")
        print(f"  quantroll solver='subspace': {t_sub:8.3f}s  ({t_naive / t_sub:5.1f}x)")
        print(f"  streaming update           : {t_stream / len(new) * 1e3:8.3f} ms/row")
        print()


if __name__ == "__main__":
    main()
