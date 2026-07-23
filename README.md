# quantroll

Fast, temporally **stable** rolling-window tools for quantitative work on financial time series.

Classical unsupervised models break on rolling windows of nonstationary data: eigenvectors flip sign and swap order between adjacent windows, cluster centroids jump, regime labels switch. None of that reflects the data — it's solver arbitrariness — and it silently corrupts every model consuming the outputs. `quantroll` provides rolling versions of these tools with **temporal alignment built in**, plus a vectorized performance-measure suite and explainable target scorecards.

- **`RollingPCA`** — rolling-window PCA whose eigenbasis is matched across time by absolute cosine similarity: signs un-flipped, component identities preserved. Streaming `update()` in O(d²) per new observation via sliding covariance sums. Supports covariance or correlation mode, 2-D `(T, d)` input or 3-D `(F, T, d)` panels (per-item covariances averaged per window), NumPy / pandas / polars in and out. **Variable feature spaces**: NaN marks an asset absent at that time — each step decomposes only the columns fully observed in its window, and alignment runs on the coordinates shared between consecutive universes (renormalized), so component identities survive assets entering and leaving.
- **`RollingRegimes`** — regime detection with a rolling Gaussian **hidden Markov model** (warm-started MAP Baum–Welch) whose regime identities are held by slowly-updated *anchor* parameters: Hungarian-matched at each refit, used as an emission prior so a regime absent from the window keeps its remembered shape instead of being repurposed, and frozen while the window is degenerate. Persistence is *learned* as a transition matrix; probabilities are the HMM forward filter; the discrete call adds hysteresis. Streaming `update()`, learned transition matrices, stationary frequencies, and a `regime_summary()` table.
- **`RollingKMeans`** — stable clustering of an entity universe (funds, assets) over time: warm-started Lloyd iterations chained through periods, Hungarian-matched cluster identities, deterministic farthest-point initialization and empty-cluster re-seeding — identical data gives identical clusters, and "cluster 2" keeps meaning the same group tomorrow. Optional **centroid memory** lets each cluster claim points near its recent past positions (piecewise-linear, nonlinear boundaries). Takes fixed-universe `(T, N, d)` panels or `(time, entity)` MultiIndex DataFrames — **unbalanced panels welcome**: entities may launch and die; absent entities carry label `-1`, switch tracking runs on shared entities, and `update()` grows the universe as newcomers appear.
- **`measures`** — NaN-aware, vectorized performance measures: CAGR, vol, Sharpe, Sortino, Calmar, drawdowns, VaR/CVaR, skew/kurtosis, hit rate, gain-to-pain, tracking error, information ratio, beta/alpha, up/down capture, rolling variants.
- **`RollingEmbedding`** — temporally stable low-dimensional *maps* of an entity universe (the nonlinear complement to `RollingPCA`). Wraps any per-period embedding backend — UMAP via the optional `umap-learn` extra, a built-in PCA fallback, or your own callable — with warm-started refits, closed-form **Procrustes** removal of the global rotation/reflection/translation the objectives are invariant under, newcomer placement via feature-space neighbors, and per-entity displacement + disparity diagnostics. Unchanged data gives an unchanged map; a fund's trajectory through the map becomes meaningful.
- **`select_n_regimes` / `select_n_clusters`** — transparent model-selection tables: AIC/BIC over maximum-likelihood Gaussian-HMM fits per candidate regime count; silhouette + inertia per candidate cluster count.
- **`Scorecard`** — compare assets against a target across many measures at once, with orientation-adjusted relative scores, a weighted composite, per-measure breakdown (the *explanation*), and a dispersion statistic showing how unevenly a target is beaten.
- **`simulate`** — synthetic generators (rotating anisotropic cloud, Markov regime-switching returns) for testing and demos.

## Install

```bash
pip install -e ".[dev]"   # from the repo root
```

**Start with the [use-case notebooks](notebooks/README.md)** — five executed walkthroughs on real ticker data (stable market factors, regime detection, stock peer groups, a stable universe map, and explainable scorecards), each showing the naive failure, the stable fix, and how to read every output.

Requires Python ≥ 3.10. Core dependencies: NumPy, SciPy, Numba. pandas/polars/matplotlib are optional.

## Quick start

```python
import numpy as np, pandas as pd
from quantroll import RollingPCA, Scorecard

# --- stable rolling PCA on daily returns ---------------------------------
rets = pd.DataFrame(...)                      # (T, d) returns, DatetimeIndex
pca = RollingPCA(n_components=3, window=252).fit(rets)

Z = pca.transform()                # (T, 3) DataFrame of stable projections
pca.similarity_                    # (T, 3) |cos| vs previous step — stability diagnostic
pca.n_flips_.sum()                 # how many solver sign-flips were corrected
pca.components_[t]                 # (3, d) aligned basis at time t

pca.update(new_rows)               # stream new observations, O(d²) each

# RiskMetrics-style exponential weighting instead of a flat window:
pca_ew = RollingPCA(n_components=3, window=60, halflife=40).fit(rets)

# --- explainable comparison vs a benchmark -------------------------------
result = Scorecard().compare(fund_returns, benchmark_returns)
result.rank()                      # assets ordered by weighted composite
result.scores_frame()              # per-measure breakdown: *why* the ranking
```

```python
# --- stable regime detection ---------------------------------------------
from quantroll import RollingRegimes

rr = RollingRegimes(n_regimes=2, window=1008).fit(features)  # window: years, not months
rr.predict()               # (T,) stable regime calls, no label switching
rr.predict_proba()         # (T, K) forward-filtered probabilities
rr.transition_[-1]         # learned transition matrix (persistence from data)
rr.regime_summary()        # share, spell lengths, per-feature mean/vol by regime
rr.update(new_rows)        # stream new observations

# how many regimes / clusters? transparent criteria tables:
from quantroll import select_n_regimes, select_n_clusters
select_n_regimes(features, candidates=(1, 2, 3, 4))["bic"].idxmin()
select_n_clusters(latest_cross_section, candidates=range(2, 9))["silhouette"].idxmax()

# --- stable universe maps (UMAP with umap-learn installed) ---------------
from quantroll import RollingEmbedding

emb = RollingEmbedding(backend="umap").fit(panel)   # or backend="pca" / callable
emb.embedding_frame()       # today's map: entities × (e1, e2)
emb.trajectory("fund_17")   # one fund's path through the map over time
emb.disparity_              # structural map change net of global motion
emb.update(next_period)
```

```python
# --- stable fund clustering ----------------------------------------------
from quantroll import RollingKMeans

km = RollingKMeans(n_clusters=5, centroid_memory=10).fit(panel)  # (T, N, d) or MultiIndex df
km.labels_frame()          # DataFrame (time × entity) of stable cluster ids
km.cluster_summary()       # share, membership moves, final centroids
km.predict(cross_section)  # assign new entities under current clusters
km.update(next_period)     # stream the next cross-section
```

The functional layer (`quantroll.rolling_pca`, `quantroll.rolling_regimes`, `quantroll.rolling_kmeans`, `quantroll.measures.*`) works directly on NumPy arrays for maximum speed and composability.

**Window sizing for regimes**: the window must span several regime cycles — a regime that never appears in the window cannot be learned from it. For daily equity-style data think years (750–1250 steps), not months. The anchor prior protects long single-regime stretches *within* such a window.

## Why alignment matters

Eigendecompositions are unique only up to sign (and order, when eigenvalues cross). Recomputing PCA per window therefore produces series that jump between `+v` and `−v` arbitrarily. `quantroll` matches each window's eigenvectors to the previous window's (Hungarian assignment on absolute cosine similarity — or the greedy per-component variant), flips negatives, and restores component identity. The similarity trace it returns is itself a useful signal: dips flag genuine covariance-structure change (regime shifts) rather than solver noise.

## Performance notes

- **Sliding covariance sums**: each step costs an O(d²) update instead of O(W·d²) per-window recomputation; a periodic full refresh bounds floating-point drift. EWMA mode (`halflife=`) is recursive — no buffer, nothing to refresh, naturally streaming.
- **Warm-started subspace iteration** (`solver="subspace"`, chosen automatically when `n_components ≤ d/4`): the previous window's basis seeds a Rayleigh–Ritz subspace solve — O(d²p) per step instead of the full O(d³) `eigh` — with a residual check that falls back to the exact solver at any step exceeding `solver_tol`.
- The rolling kernel is Numba-JIT-compiled (first call pays a one-time compile, cached afterwards); alignment is O(p²·d) per step, negligible.

Measured against naive per-window `np.cov` + `eigh` recomputation (M-series Mac, float64):

| T × d, W, p | naive | `solver="eigh"` | `solver="subspace"` | speedup |
|---|---|---|---|---|
| 5000 × 30, W=252, p=5 | 0.32 s | 0.23 s | 0.16 s | 2.0× |
| 5000 × 100, W=252, p=5 | 2.61 s | 2.45 s | **0.19 s** | **13.5×** |
| 2000 × 250, W=252, p=10 | 4.91 s | 4.79 s | **0.30 s** | **16.3×** |

Streaming `update()`: ~0.5 ms/row at d=30, ~1 ms/row at d=100 (exact solver). Reproduce with `python scripts/bench.py`.

`RollingRegimes` (K=2, d=4, W=750): 0.27 ms/step batch — warm-started Baum–Welch converges in 1–2 iterations per refit — 0.28 ms/row streaming, ~3× faster again with `refit_every=5`. Recovers simulated Markov regimes at 98–99% accuracy with zero identity relabels across seeds.

`RollingKMeans` (N=500 entities, d=8, K=5): 0.026 ms/period batch, 0.05 ms/period streaming — warm starts converge in ~2 Lloyd iterations. On drifting well-separated blobs: exact membership recovery, zero cluster-identity relabels, zero spurious entity switches (a naive per-period K-Means flips labels constantly on the same data).

## Method sources & independence

The stability methodology follows published research: Hirsa, Klinkert, Malhotra & Holmes, *"Robust Rolling PCA: Managing Time Series and Multiple Dimensions"* (SSRN 4400158, 2023); the regime module is in the spirit of Hirsa, Xu & Malhotra, *"Robust Rolling Regime Detection"* (2024) and its HMM extensions, built here as an anchored MAP rolling HMM of our own design; the clustering module follows the rolling K-Means concept of Hirsa, Holmes, Klinkert & Malhotra (2024) — warm-started, identity-matched, with a rolling window of centroids for nonlinear boundaries; the embedding module addresses the rolling-UMAP stabilization concept with a warm-start + Procrustes design of our own. The scorecard is an independent, transparent construction inspired by the multi-measure unification concept of Hirsa, Ding & Malhotra (SSRN 4335455, 2023). **This library is an independent open implementation of publicly published algorithm ideas. It is not affiliated with, endorsed by, or derived from proprietary code of Ask2.ai Inc.; no trademarked product names are used in the API.**

## Roadmap

- Regime extensions: diagonal-covariance mode, duration statistics
- Clustering extensions: spectral / density-based rolling variants
- Incremental eigenupdates; polars-native fast paths
- Factor extraction via autoencoders (optional torch extra)
