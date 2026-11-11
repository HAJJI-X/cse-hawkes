"""GARCH baseline (usual volatility model)."""

from __future__ import annotations

import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


def fit_garch(
    returns: pd.Series,
    dist: str = "t",
    p: int = 1,
    q: int = 1,
) -> ARCHModelResult:
    """
    Fit a constant-mean GARCH(p,q).

    arch expects percent returns, so we scale by 100.
    Default Student-t errors match common MASI / emerging-market practice.
    """
    am = arch_model(
        returns * 100.0,
        mean="Constant",
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
    )
    return am.fit(disp="off")


def conditional_vol(res: ARCHModelResult) -> pd.Series:
    """Conditional volatility back on the log-return scale."""
    return res.conditional_volatility / 100.0
