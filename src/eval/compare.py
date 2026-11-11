"""Small helpers to summarize a run."""

from __future__ import annotations

from typing import Any


def summarise_fit(
    garch_aic: float | None,
    hawkes: dict[str, Any],
    poisson_nll: float,
) -> dict[str, Any]:
    """Bundle headline numbers for results/fit_summary.json."""
    return {
        "garch_aic": garch_aic,
        "hawkes": {
            "mu": hawkes.get("mu"),
            "alpha": hawkes.get("alpha"),
            "beta": hawkes.get("beta"),
            "branching_ratio": hawkes.get("branching_ratio"),
            "nll": hawkes.get("nll"),
            "n_events": hawkes.get("n_events"),
            "success": hawkes.get("success"),
        },
        "poisson_nll": poisson_nll,
        "hawkes_beats_poisson_on_nll": (
            hawkes.get("nll") is not None and hawkes["nll"] < poisson_nll
        ),
        "note": "Early run on synthetic data not real MASI results",
    }
