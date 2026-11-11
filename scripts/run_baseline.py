#!/usr/bin/env python3
"""Fit GARCH and Hawkes, then write a small JSON summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.returns import exceedance_times, load_prices  # noqa: E402
from eval.compare import summarise_fit  # noqa: E402
from models.garch_fit import fit_garch  # noqa: E402
from models.hawkes_exp import fit_hawkes, poisson_nll  # noqa: E402

SYN = ROOT / "data" / "synthetic" / "masi_synthetic.csv"
RAW = ROOT / "data" / "raw" / "masi_daily.csv"
OUT = ROOT / "results" / "fit_summary.json"


def main() -> None:
    path = RAW if RAW.exists() else SYN
    if not path.exists():
        raise SystemExit(
            f"No data at {SYN}. Run: python scripts/make_synthetic.py"
        )

    print(f"using {path}")
    df = load_prices(path)
    res = fit_garch(df["log_ret"])
    events, thr = exceedance_times(df, q=0.95, side="neg")
    T = float(len(df) - 1)
    hawkes = fit_hawkes(events, T=T)
    p_nll = poisson_nll(events, T)

    summary = summarise_fit(float(res.aic), hawkes, p_nll)
    summary["data_path"] = str(path.relative_to(ROOT))
    summary["n_obs"] = int(len(df))
    summary["exceedance_threshold"] = thr
    summary["n_exceedances"] = int(len(events))
    summary["garch_persistence"] = float(
        res.params.get("alpha[1]", 0.0) + res.params.get("beta[1]", 0.0)
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
