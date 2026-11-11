"""Simple exponential Hawkes.

Past events raise the chance of new ones for a while.
λ(t) = μ + sum of α * exp(-β * age) over past events.
Branching ratio α/β under 1 means the process stays stable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def neg_loglik(theta: np.ndarray, events: np.ndarray, T: float) -> float:
    mu, alpha, beta = theta
    if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
        return 1e50

    events = np.asarray(events, dtype=float)
    n = len(events)
    if n == 0:
        return float(mu * T)

    # Recursive excitation term S_i = sum_{j<i} exp(-β (t_i - t_j))
    ll = 0.0
    S = 0.0
    for i, t in enumerate(events):
        if i == 0:
            S = 0.0
        else:
            dt = t - events[i - 1]
            S = np.exp(-beta * dt) * (1.0 + S)
        lam = mu + alpha * S
        if lam <= 0:
            return 1e50
        ll += np.log(lam)

    # Compensator ∫_0^T λ(t) dt
    integral = mu * T
    for t in events:
        integral += (alpha / beta) * (1.0 - np.exp(-beta * (T - t)))

    return float(-(ll - integral))


def fit_hawkes(
    events: np.ndarray,
    T: float | None = None,
) -> dict[str, Any]:
    """MLE for an exponential Hawkes process on event times in [0, T]."""
    events = np.asarray(events, dtype=float)
    if len(events) < 3:
        raise ValueError("Need at least 3 events to fit Hawkes.")

    T = float(events[-1] if T is None else T)
    x0 = np.array([0.05, 0.4, 1.0])
    bounds = [(1e-8, None), (1e-8, None), (1e-8, None)]

    res = minimize(
        neg_loglik,
        x0,
        args=(events, T),
        method="L-BFGS-B",
        bounds=bounds,
    )
    mu, alpha, beta = map(float, res.x)
    return {
        "mu": mu,
        "alpha": alpha,
        "beta": beta,
        "branching_ratio": alpha / beta,
        "nll": float(res.fun),
        "success": bool(res.success),
        "n_events": int(len(events)),
        "T": T,
    }


def poisson_nll(events: np.ndarray, T: float) -> float:
    """Homogeneous Poisson null: λ = n / T."""
    n = len(events)
    if T <= 0:
        return np.inf
    lam = n / T
    # -log L = -n log λ + λ T
    return float(-n * np.log(lam) + lam * T)
