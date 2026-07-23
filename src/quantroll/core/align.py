"""Temporal alignment of eigenbases across rolling windows.

Successive eigendecompositions of nearly identical covariance matrices can
return eigenvectors with arbitrary signs and, when eigenvalues cross, swapped
order. Both effects are artifacts of the solver, not the data, and they wreck
downstream models that consume the projected series.

The fix, following Hirsa, Klinkert, Malhotra & Holmes (2023), is to match each
window's eigenvectors to the previous window's by absolute cosine similarity,
flip signs where the dot product is negative, and reorder components to keep
their identities stable through time.

Two matching modes are provided:

- ``"hungarian"`` (default): optimal one-to-one assignment maximizing total
  absolute cosine similarity (``scipy.optimize.linear_sum_assignment``).
  Strictly better than greedy when several components rotate at once.
- ``"greedy"``: per-component argmax in order, as written in the paper's
  Algorithm 1, with a next-best fallback when two components claim the same
  predecessor.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "match_step",
    "align_sequence",
    "align_sequence_masked",
    "canonical_signs",
    "AlignResult",
]


def canonical_signs(V: np.ndarray) -> np.ndarray:
    """Deterministic sign convention: largest-|entry| of each column is positive.

    Applied to the first window's basis so results are reproducible run-to-run
    (LAPACK sign choices are otherwise arbitrary).
    """
    V = np.asarray(V)
    idx = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[idx, np.arange(V.shape[1])])
    signs[signs == 0] = 1.0
    return V * signs


def match_step(
    prev: np.ndarray,
    cur: np.ndarray,
    method: str = "hungarian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match current eigenvectors to the previous basis.

    Parameters
    ----------
    prev, cur : (d, p) arrays with unit-norm columns.
    method : "hungarian" or "greedy".

    Returns
    -------
    perm : (p,) int — ``perm[k]`` is the column of ``cur`` assigned to slot k.
    signs : (p,) float ±1 — sign to apply to the matched column.
    sim : (p,) float — absolute cosine similarity of each match.
    """
    M = prev.T @ cur
    return _match_from_M(M, method)


def _match_from_M(M: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assignment, signs and |cos| from a (p, p) cosine matrix."""
    A = np.abs(M)
    p = A.shape[0]

    if method == "hungarian":
        _, perm = linear_sum_assignment(-A)
    elif method == "greedy":
        perm = np.full(p, -1, dtype=np.intp)
        taken = np.zeros(p, dtype=bool)
        for i in range(p):
            order = np.argsort(-A[:, i])
            for k in order:
                if not taken[k]:
                    perm[k] = i
                    taken[k] = True
                    break
    else:
        raise ValueError(f"unknown matching method: {method!r}")

    slots = np.arange(p)
    signs = np.sign(M[slots, perm])
    signs[signs == 0] = 1.0
    sim = A[slots, perm]
    return perm, signs, sim


class AlignResult(NamedTuple):
    eigvals: np.ndarray
    """(T, p) eigenvalues reordered to stable component identities."""
    eigvecs: np.ndarray
    """(T, d, p) aligned eigenvectors (consistent sign and order)."""
    similarity: np.ndarray
    """(T, p) |cos| between each component and its predecessor; NaN in warm-up."""
    n_flips: np.ndarray
    """(T,) number of sign flips applied at each step."""
    n_reorders: np.ndarray
    """(T,) number of components whose order changed at each step."""


def align_sequence(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    method: str = "hungarian",
) -> AlignResult:
    """Chain-align a full sequence of eigendecompositions.

    Each step is matched against the *aligned* basis of the previous step, so
    component identities propagate through time. Rows whose eigenvalues are
    NaN (the rolling warm-up) are skipped.
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    T, p = eigvals.shape

    out_vals = eigvals.copy()
    out_vecs = eigvecs.copy()
    similarity = np.full((T, p), np.nan)
    n_flips = np.zeros(T, dtype=np.int64)
    n_reorders = np.zeros(T, dtype=np.int64)

    valid = ~np.isnan(eigvals[:, 0])
    if not valid.any():
        return AlignResult(out_vals, out_vecs, similarity, n_flips, n_reorders)
    start = int(np.argmax(valid))

    out_vecs[start] = canonical_signs(out_vecs[start])
    prev = out_vecs[start]
    slots = np.arange(p)

    for t in range(start + 1, T):
        if not valid[t]:
            continue
        perm, signs, sim = match_step(prev, eigvecs[t], method=method)
        out_vecs[t] = eigvecs[t][:, perm] * signs
        out_vals[t] = eigvals[t][perm]
        similarity[t] = sim
        n_flips[t] = int((signs < 0).sum())
        n_reorders[t] = int((perm != slots).sum())
        prev = out_vecs[t]

    return AlignResult(out_vals, out_vecs, similarity, n_flips, n_reorders)


def align_sequence_masked(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    method: str = "hungarian",
) -> AlignResult:
    """Chain-align eigenbases that live on *changing* feature universes.

    ``eigvecs[t]`` has NaN rows on features inactive at step t. Matching uses
    only the coordinates shared by consecutive active universes, with each
    restricted vector renormalized to unit length so coverage differences do
    not bias the cosine similarities. If two steps share no coordinates the
    chain restarts with a canonical sign convention (similarity stays NaN at
    the restart).
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    T, p = eigvals.shape

    out_vals = eigvals.copy()
    out_vecs = eigvecs.copy()
    similarity = np.full((T, p), np.nan)
    n_flips = np.zeros(T, dtype=np.int64)
    n_reorders = np.zeros(T, dtype=np.int64)

    valid = ~np.isnan(eigvals[:, 0])
    if not valid.any():
        return AlignResult(out_vals, out_vecs, similarity, n_flips, n_reorders)
    start = int(np.argmax(valid))
    slots = np.arange(p)

    mask = ~np.isnan(out_vecs[start, :, 0])
    out_vecs[start, mask] = canonical_signs(out_vecs[start, mask])
    prev = out_vecs[start]
    prev_mask = mask

    for t in range(start + 1, T):
        if not valid[t]:
            continue
        cur = out_vecs[t]
        cur_mask = ~np.isnan(cur[:, 0])
        shared = prev_mask & cur_mask
        if not shared.any():
            cur[cur_mask] = canonical_signs(cur[cur_mask])
            prev, prev_mask = cur, cur_mask
            continue

        P = prev[shared]
        C = cur[shared]
        pn = np.linalg.norm(P, axis=0)
        cn = np.linalg.norm(C, axis=0)
        Pn = np.where(pn > 1e-12, P / np.where(pn > 0, pn, 1.0), 0.0)
        Cn = np.where(cn > 1e-12, C / np.where(cn > 0, cn, 1.0), 0.0)
        perm, signs, sim = _match_from_M(Pn.T @ Cn, method)

        out_vecs[t] = cur[:, perm] * signs
        out_vals[t] = eigvals[t][perm]
        similarity[t] = sim
        n_flips[t] = int((signs < 0).sum())
        n_reorders[t] = int((perm != slots).sum())
        prev = out_vecs[t]
        prev_mask = cur_mask

    return AlignResult(out_vals, out_vecs, similarity, n_flips, n_reorders)
