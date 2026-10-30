"""Extending the monthly indicators to the end of the target quarter (ragged edge).

Two methods:
- "ar" : one AR(p) per series (p chosen by AIC), estimated on the winsorized series;
- "dfm": a common dynamic factor model (statsmodels DynamicFactorMQ, Bańbura-Modugno
  approach); missing months are the model's smoothed values. This is the GDPNow idea:
  the factor is only used to complete the source data.

Only values missing AFTER the last observation of each series are filled.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import Config
from .transforms import winsorize


def _stationary(s: pd.Series, transform: str) -> pd.Series:
    return 100 * np.log(s).diff() if transform == "pct" else s


def _rebuild(level_last: float, path: np.ndarray, transform: str) -> np.ndarray:
    if transform == "pct":
        return level_last * np.exp(np.cumsum(path) / 100)
    return path


def _ar_forecast(x: pd.Series, steps: int, max_lag: int = 6) -> np.ndarray:
    """Iterated forecast of an AR(p) with constant, p by AIC, estimated by OLS on winsorized x."""
    xw = winsorize(x.dropna()).to_numpy()
    best = None
    n_eff = len(xw) - max_lag
    for p in range(1, max_lag + 1):
        y = xw[max_lag:]
        X = np.column_stack([np.ones(n_eff)] + [xw[max_lag - j:len(xw) - j] for j in range(1, p + 1)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        aic = n_eff * np.log(resid @ resid / n_eff) + 2 * (p + 1)
        if best is None or aic < best[0]:
            best = (aic, p, beta)
    _, p, beta = best
    hist = list(x.dropna().to_numpy()[-p:])  # start from the actual last values
    out = []
    for _ in range(steps):
        nxt = beta[0] + sum(beta[j] * hist[-j] for j in range(1, p + 1))
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)


def fill_monthly(monthly: pd.DataFrame, cfg: Config, until: pd.Period, method: str = "ar",
                 start: str = "2000-01") -> pd.DataFrame:
    idx = pd.period_range(monthly.index.min(), max(monthly.index.max(), until), freq="M")
    m = monthly.reindex(idx)
    if method == "dfm":
        return _fill_dfm(m, cfg, start)
    if method != "ar":
        raise ValueError(f"Unknown method: {method}")
    out = m.copy()
    for col in m.columns:
        s = m[col]
        last = s.last_valid_index()
        if last is None or last >= idx[-1]:
            continue
        steps = idx[-1].ordinal - last.ordinal
        tr = cfg.indicators[col].transform
        x = _stationary(s.loc[start:last], tr)
        path = _ar_forecast(x, steps)
        out.loc[last + 1:, col] = _rebuild(s.loc[last], path, tr)
    return out


def _fill_dfm(m: pd.DataFrame, cfg: Config, start: str) -> pd.DataFrame:
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    panel = m.loc[start:]
    X = pd.DataFrame({c: _stationary(panel[c], cfg.indicators[c].transform) for c in panel.columns})
    X = X.iloc[1:]  # the first difference is empty
    X = X.apply(winsorize)
    mu, sd = X.mean(), X.std()
    Z = (X - mu) / sd
    Z.index = Z.index.to_timestamp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = DynamicFactorMQ(Z, factors=1, factor_orders=2, idiosyncratic_ar1=True, standardize=False)
        res = mod.fit(disp=False, maxiter=300)
    smoothed = res.get_prediction(information_set="smoothed").predicted_mean
    smoothed.index = X.index
    fitted = smoothed * sd + mu

    out = m.copy()
    for col in m.columns:
        s = m[col]
        last = s.last_valid_index()
        if last is None or last >= m.index[-1]:
            continue
        path = fitted.loc[last + 1:, col].to_numpy()
        out.loc[last + 1:, col] = _rebuild(s.loc[last], path, cfg.indicators[col].transform)
    return out
