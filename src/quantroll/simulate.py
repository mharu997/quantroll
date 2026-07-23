"""Synthetic data generators for testing and demos.

``rotating_cloud`` reproduces the covariance-instability stress test used in
the rolling-PCA literature: an anisotropic Gaussian cloud whose principal
direction rotates a little every step, so any per-window PCA must track a
moving basis without sign flips.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rotating_cloud", "regime_returns", "drifting_blobs"]


def rotating_cloud(
    n_steps: int = 1000,
    theta_per_step: float = 0.01,
    sigmas: tuple[float, float] = (3.0, 0.5),
    n_features: int = 2,
    noise: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Anisotropic 2-D Gaussian embedded in ``n_features`` dims, rotating over time.

    Returns
    -------
    X : (n_steps, n_features)
        One observation per step; the latent principal direction at step t is
        rotated by ``t * theta_per_step`` radians.
    true_direction : (n_steps, n_features)
        Unit vector of the latent leading principal axis at each step.
    """
    if n_features < 2:
        raise ValueError("n_features must be >= 2")
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    angles = t * theta_per_step
    cos, sin = np.cos(angles), np.sin(angles)

    z1 = rng.normal(0.0, sigmas[0], n_steps)
    z2 = rng.normal(0.0, sigmas[1], n_steps)
    plane = np.column_stack([cos * z1 - sin * z2, sin * z1 + cos * z2])

    X = rng.normal(0.0, noise, (n_steps, n_features))
    X[:, :2] += plane
    true_direction = np.zeros((n_steps, n_features))
    true_direction[:, 0] = cos
    true_direction[:, 1] = sin
    return X, true_direction


def regime_returns(
    n_steps: int = 2000,
    n_assets: int = 5,
    p_stay: float = 0.99,
    bull: tuple[float, float] = (8e-4, 0.008),
    bear: tuple[float, float] = (-6e-4, 0.02),
    factor_load: float = 0.7,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-state Markov regime-switching returns with a common factor.

    Returns
    -------
    returns : (n_steps, n_assets) simple periodic returns.
    states : (n_steps,) int — 0 = bull, 1 = bear.
    """
    rng = np.random.default_rng(seed)
    states = np.empty(n_steps, dtype=np.int64)
    s = 0
    for t in range(n_steps):
        states[t] = s
        if rng.random() > p_stay:
            s = 1 - s
    mu = np.where(states == 0, bull[0], bear[0])
    sig = np.where(states == 0, bull[1], bear[1])

    common = rng.standard_normal(n_steps)
    idio = rng.standard_normal((n_steps, n_assets))
    shocks = factor_load * common[:, None] + np.sqrt(1 - factor_load**2) * idio
    returns = mu[:, None] + sig[:, None] * shocks
    return returns, states


def drifting_blobs(
    n_steps: int = 200,
    n_entities: int = 90,
    n_clusters: int = 3,
    n_features: int = 2,
    sep: float = 6.0,
    spread: float = 0.5,
    drift: float = 0.02,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Entity panel whose cluster centers rotate slowly around the origin.

    Entities keep a fixed cluster membership; each period their features are
    drawn around their cluster's current center. Centers sit on a circle (in
    the first two feature dimensions) with nearest-neighbor separation
    ``sep`` and rotate by ``drift`` radians per step — a moving target that a
    temporally stable clustering should track without identity changes.

    Returns
    -------
    X : (n_steps, n_entities, n_features)
    membership : (n_entities,) int — true cluster of each entity.
    centers : (n_steps, n_clusters, n_features) — true centers over time.
    """
    if n_features < 2:
        raise ValueError("n_features must be >= 2")
    rng = np.random.default_rng(seed)
    membership = np.arange(n_entities) % n_clusters
    radius = sep / (2.0 * np.sin(np.pi / n_clusters)) if n_clusters > 1 else 0.0
    base = 2.0 * np.pi * np.arange(n_clusters) / n_clusters

    centers = np.zeros((n_steps, n_clusters, n_features))
    t = np.arange(n_steps)[:, None]
    centers[:, :, 0] = radius * np.cos(base[None, :] + drift * t)
    centers[:, :, 1] = radius * np.sin(base[None, :] + drift * t)

    X = centers[:, membership, :] + spread * rng.standard_normal(
        (n_steps, n_entities, n_features)
    )
    return X, membership, centers
