"""Turn FinSent comment times into daily counts (later step)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def daily_comment_intensity(
    events_csv: str | Path,
    tz: str = "Africa/Casablanca",
) -> pd.DataFrame:
    """Aggregate comment counts and code-switch share by Casablanca calendar day."""
    df = pd.read_csv(events_csv, parse_dates=["published_at"])
    ts = df["published_at"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    df = df.copy()
    df["day"] = ts.dt.tz_convert(tz).dt.floor("D")

    g = df.groupby("day").agg(
        n=("comment_id", "size"),
        n_cs=("is_cs", "sum") if "is_cs" in df.columns else ("comment_id", "size"),
    )
    if "is_cs" in df.columns:
        g["cs_share"] = g["n_cs"] / g["n"].clip(lower=1)
    return g
