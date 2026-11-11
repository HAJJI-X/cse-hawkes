"""Load prices and pick out the worst days."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a CSV with columns date, close; ensure log_ret exists."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if "log_ret" not in df.columns:
        df["log_ret"] = np.log(df["close"]).diff()
    return df.dropna(subset=["log_ret"]).reset_index(drop=True)


def exceedance_times(
    df: pd.DataFrame,
    q: float = 0.95,
    side: str = "neg",
) -> tuple[np.ndarray, float]:
    """
    Event times in trading-day index units for a Hawkes process.

    For side='neg' and q=0.95, threshold is the empirical 5% left-tail quantile
    (worst days).
    """
    x = df["log_ret"].to_numpy(dtype=float)
    if side == "neg":
        thr = float(np.quantile(x, 1.0 - q))
        mask = x <= thr
    else:
        thr = float(np.quantile(x, q))
        mask = x >= thr
    t = np.flatnonzero(mask).astype(float)
    return t, thr
