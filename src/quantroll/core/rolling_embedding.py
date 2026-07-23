"""Temporally stable low-dimensional embeddings over rolling cross-sections.

Nonlinear embedding algorithms (UMAP, t-SNE) are excellent cartographers of
an entity universe but terrible at consistency: their objectives are
invariant to rotation, reflection and translation of the whole map, their
optimizers are stochastic, and small data changes can teleport clusters.
Re-embedding a fund universe each month therefore produces maps that churn
violently even when nothing economic changed.

This module stabilizes *any* per-period embedding backend the same way the
rest of the library stabilizes its models:

- **warm starts** — each period's optimizer is initialized at the previous
  period's (aligned) coordinates, so unchanged data yields unchanged maps;
- **Procrustes alignment** — the leftover global rotation / reflection /
  translation (optionally scale) is removed in closed form by orthogonal
  Procrustes onto the previous embedding over shared entities;
- **diagnostics** — per-entity displacement and the Procrustes disparity
  separate genuine structural change from solver noise.

Backends: ``"pca"`` (built-in linear fallback, always available), ``"umap"``
(requires the optional ``umap-learn`` package), or any callable
``f(X, init) -> (N, k)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["rolling_embedding", "RollingEmbeddingResult", "procrustes_align"]


@dataclass(frozen=True)
class RollingEmbeddingResult:
    """Output of :func:`rolling_embedding` over a fixed-universe panel."""

    embeddings: np.ndarray
    """(T, N, k) aligned coordinates of each entity at each period."""
    disparity: np.ndarray
    """(T,) RMS Procrustes residual vs. the previous period (NaN at t=0) —
    the amount of *structural* map change after removing global motion."""
    displacement: np.ndarray
    """(T, N) per-entity movement from the previous aligned period."""
    n_components: int = 2


def procrustes_align(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    scale: bool = False,
) -> tuple[np.ndarray, float, tuple[np.ndarray, np.ndarray, float]]:
    """Rigidly align ``cur`` onto ``ref`` (orthogonal Procrustes).

    Removes translation and the best rotation/reflection (and optionally a
    global scale) — the symmetry group embedding objectives are invariant
    under. Rows are paired observations (shared entities).

    Returns ``(aligned, disparity, transform)`` where ``disparity`` is the
    RMS residual after alignment and ``transform = (R, shift, s)`` maps any
    further points from the current frame: ``y @ R * s + shift``.
    """
    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    mu_r = ref.mean(axis=0)
    mu_c = cur.mean(axis=0)
    Rc = ref - mu_r
    Cc = cur - mu_c
    U, sv, Vt = np.linalg.svd(Cc.T @ Rc)
    R = U @ Vt
    if scale:
        denom = (Cc**2).sum()
        s = sv.sum() / denom if denom > 0 else 1.0
    else:
        s = 1.0
    shift = mu_r - (mu_c @ R) * s
    aligned = cur @ R * s + shift
    disparity = float(np.sqrt(((aligned - ref) ** 2).sum(axis=1).mean()))
    return aligned, disparity, (R, shift, s)


def pca_embedding(X: np.ndarray, init: np.ndarray | None, n_components: int) -> np.ndarray:
    """Built-in linear backend: project the cross-section onto its top
    principal axes. Deterministic; ``init`` is unused (alignment alone makes
    the sequence stable)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    _, vecs = np.linalg.eigh(cov)
    V = vecs[:, ::-1][:, :n_components]
    return Xc @ V


def make_backend(
    backend: str | Callable,
    n_components: int,
    seed: int | None,
    backend_kwargs: dict | None,
) -> Callable[[np.ndarray, np.ndarray | None], np.ndarray]:
    """Resolve a backend spec into ``f(X, init) -> (N, k)``."""
    kwargs = dict(backend_kwargs or {})
    if callable(backend):
        return lambda X, init: np.asarray(backend(X, init), dtype=np.float64)
    if backend == "pca":
        return lambda X, init: pca_embedding(X, init, n_components)
    if backend == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "backend='umap' needs the optional dependency umap-learn "
                "(pip install quantroll[umap])"
            ) from exc

        def run_umap(X: np.ndarray, init: np.ndarray | None) -> np.ndarray:
            kw = dict(kwargs)
            kw.setdefault("n_neighbors", min(15, max(2, X.shape[0] - 1)))
            reducer = umap.UMAP(
                n_components=n_components,
                init=init if init is not None else "spectral",
                random_state=seed,
                **kw,
            )
            return np.asarray(reducer.fit_transform(X), dtype=np.float64)

        return run_umap
    raise ValueError(f"backend must be 'pca', 'umap' or a callable; got {backend!r}")


class _EmbeddingStepper:
    """Single-step engine shared by batch fit and streaming update.

    Operates on positionless arrays; the caller supplies, per step, which
    rows carry over from the previous step (for warm starts and alignment)
    via index arrays into the previous step's coordinates.
    """

    def __init__(self, run: Callable, n_components: int, scale: bool) -> None:
        self.run = run
        self.k = n_components
        self.scale = scale
        self.prev: np.ndarray | None = None  # (N_prev, k) aligned coords

    def step(
        self,
        X: np.ndarray,
        cur_shared: np.ndarray,
        prev_shared: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Embed one cross-section.

        ``cur_shared`` / ``prev_shared`` index the rows of ``X`` and of the
        previous coordinates that refer to the same entities. Returns
        ``(aligned (N, k), disparity, displacement_shared)``.
        """
        N = X.shape[0]
        init = None
        if self.prev is not None and cur_shared.size:
            init = np.empty((N, self.k))
            carried = np.zeros(N, dtype=bool)
            init[cur_shared] = self.prev[prev_shared]
            carried[cur_shared] = True
            if not carried.all():
                init[~carried] = _place_newcomers(
                    X, cur_shared, init[cur_shared], np.flatnonzero(~carried)
                )
        raw = self.run(X, init)
        if raw.shape != (N, self.k):
            raise ValueError(
                f"backend returned shape {raw.shape}, expected {(N, self.k)}"
            )

        if self.prev is None or cur_shared.size < max(self.k + 1, 3):
            aligned = raw
            disparity = np.nan
            disp = np.full(cur_shared.size, np.nan)
        else:
            _, disparity, (R, shift, s) = procrustes_align(
                self.prev[prev_shared], raw[cur_shared], scale=self.scale
            )
            aligned = raw @ R * s + shift
            disp = np.sqrt(
                ((aligned[cur_shared] - self.prev[prev_shared]) ** 2).sum(axis=1)
            )
        self.prev = aligned
        return aligned, disparity, disp


def _place_newcomers(
    X: np.ndarray,
    cur_shared: np.ndarray,
    shared_coords: np.ndarray,
    newcomers: np.ndarray,
) -> np.ndarray:
    """Seed entities without history at their nearest carried-over neighbor
    (in feature space), with a tiny deterministic offset to avoid exact
    overlaps."""
    anchors = X[cur_shared]
    out = np.empty((newcomers.size, shared_coords.shape[1]))
    for i, idx in enumerate(newcomers):
        d2 = ((anchors - X[idx]) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        out[i] = shared_coords[j] + 1e-3 * (i + 1)
    return out


def rolling_embedding(
    X: np.ndarray,
    n_components: int = 2,
    *,
    backend: str | Callable = "pca",
    scale: bool = False,
    seed: int | None = 0,
    backend_kwargs: dict | None = None,
) -> RollingEmbeddingResult:
    """Stable embedding of a fixed entity universe over time.

    Parameters
    ----------
    X : array, shape (T, N, d)
        For each of T periods, an (N entities × d features) cross-section.
    n_components : int
        Embedding dimension k (2 for maps).
    backend : {"pca", "umap"} or callable ``f(X, init) -> (N, k)``
        Per-period embedding algorithm. ``init`` is the warm-start coordinate
        array (or None on the first period); backends may ignore it —
        Procrustes alignment still applies.
    scale : bool
        Also remove a global scale change during alignment (off by default:
        genuine spread changes are usually signal).
    seed : int | None
        Random seed passed to stochastic backends (UMAP).
    backend_kwargs : dict | None
        Extra keyword arguments for the backend (e.g. ``n_neighbors``,
        ``min_dist`` for UMAP).

    Examples
    --------
    >>> from quantroll import rolling_embedding, simulate
    >>> X, membership, _ = simulate.drifting_blobs(30, seed=0)
    >>> res = rolling_embedding(X, backend="pca")
    >>> res.embeddings.shape
    (30, 90, 2)
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 3:
        raise ValueError(f"X must be 3-D (T, N, d); got shape {X.shape}")
    if np.isnan(X).any():
        raise ValueError("NaN features are not supported; use the estimator "
                         "with an unbalanced (time, entity) MultiIndex instead")
    T, N, d = X.shape
    if not 1 <= n_components <= d:
        raise ValueError(f"n_components must be in [1, {d}]; got {n_components}")

    run = make_backend(backend, n_components, seed, backend_kwargs)
    stepper = _EmbeddingStepper(run, n_components, scale)
    embeddings = np.empty((T, N, n_components))
    disparity = np.full(T, np.nan)
    displacement = np.full((T, N), np.nan)
    everyone = np.arange(N)
    for t in range(T):
        aligned, disp_t, disp_vec = stepper.step(X[t], everyone, everyone)
        embeddings[t] = aligned
        disparity[t] = disp_t
        if t > 0:
            displacement[t] = disp_vec
    return RollingEmbeddingResult(
        embeddings=embeddings,
        disparity=disparity,
        displacement=displacement,
        n_components=n_components,
    )
