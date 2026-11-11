#!/usr/bin/env python3
"""Make a fake daily price series so we can test the code without real MASI data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic" / "masi_synthetic.csv"


def main(seed: int = 42, n: int = 2500) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n, freq="C")

    # GARCH(1,1)-ish path
    omega, alpha, beta = 1e-6, 0.08, 0.90
    eps = rng.standard_t(df=7, size=n) / np.sqrt(7 / 5)
    r = np.zeros(n)
    h = np.full(n, omega / max(1e-12, (1 - alpha - beta)))
    for t in range(1, n):
        h[t] = omega + alpha * (r[t - 1] ** 2) + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * eps[t]

    # Extra self-exciting downside jumps (what Hawkes should pick up)
    mu, alpha_h, beta_h = 0.01, 0.55, 1.8
    intensity = mu
    for t in range(n):
        if rng.random() < intensity:
            r[t] -= abs(rng.normal(0.025, 0.008))
            intensity = mu + alpha_h
        else:
            intensity = mu + (intensity - mu) * np.exp(-beta_h)

    px = 1000.0 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"date": dates, "close": px, "log_ret": r})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
