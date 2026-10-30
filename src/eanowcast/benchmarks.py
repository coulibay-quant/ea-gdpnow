"""Benchmarks: the component model has to beat them to be worth showing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import Config

DIRECT_REGRESSORS = ["retail_volume", "ip_total", "construction", "esi"]


def _sample(cfg: Config, last: pd.Period) -> list[pd.Period]:
    return [q for q in pd.period_range(cfg.start_quarter, last, freq="Q") if q not in cfg.exclude_quarters]


def ar_gdp(gdp: pd.Series, cfg: Config, target: pd.Period, p: int = 2) -> float:
    """AR(p) on GDP growth, iterated up to the target quarter."""
    y = gdp.dropna()
    last = y.index.max()
    lags = pd.concat({f"lag{j}": y.shift(j) for j in range(1, p + 1)}, axis=1)
    data = lags.join(y.rename("y")).reindex(_sample(cfg, last)).dropna()
    res = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y"))).fit()
    hist = list(y.to_numpy())
    for _ in range(target.ordinal - last.ordinal):
        x = np.r_[1.0, [hist[-j] for j in range(1, p + 1)]]
        hist.append(float(x @ res.params.to_numpy()))
    return hist[-1]


def historical_mean(gdp: pd.Series, cfg: Config) -> float:
    y = gdp.dropna()
    return float(y.reindex(_sample(cfg, y.index.max())).mean())


def direct_bridge(gdp: pd.Series, X: pd.DataFrame, cfg: Config, target: pd.Period,
                  regressors: list[str] = DIRECT_REGRESSORS) -> float:
    """Bridge equation for aggregate GDP on the same indicators: the real test of the decomposition."""
    y = gdp.dropna()
    data = X[regressors].join(y.rename("y"), how="outer").reindex(_sample(cfg, y.index.max())).dropna()
    res = sm.OLS(data["y"], sm.add_constant(data.drop(columns="y"))).fit()
    x = np.r_[1.0, X.loc[target, regressors].to_numpy(dtype=float)]
    return float(x @ res.params.to_numpy())
