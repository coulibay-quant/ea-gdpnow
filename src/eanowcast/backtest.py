"""Pseudo-real-time evaluation and tracking of a quarter (GDPNow-style chart).

Pseudo real time: at each nowcast date, `Dataset.as_of` hides what had not been published
yet (lags from the config), then everything is re-estimated. Values come from the latest
vintage, as in Higgins (2014); true vintages accumulate in the database over time.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import benchmarks as bm
from .config import Config
from .data import Dataset
from .nowcast import run_nowcast
from .transforms import historical_contributions

# Nowcast dates, in days relative to the end of the target quarter
HORIZONS = {"J-60": -60, "J-30": -30, "J0": 0, "J+25": 25}


def nowcast_date(target: pd.Period, offset_days: int) -> pd.Timestamp:
    return target.to_timestamp(how="end").normalize() + pd.Timedelta(days=offset_days)


def backtest(ds: Dataset, cfg: Config, start: str, end: str, horizons: dict[str, int] = HORIZONS,
             fill_method: str = "ar", verbose: bool = True) -> pd.DataFrame:
    actual_gdp = ds.growth["gdp"]
    actual_contrib = historical_contributions(ds.growth, ds.levels, cfg)
    blocks = list(cfg.blocks)
    rows = []
    targets = pd.period_range(start, end, freq="Q")
    for i, target in enumerate(targets):
        if pd.isna(actual_gdp.get(target, np.nan)):
            continue
        for hname, offset in horizons.items():
            date = nowcast_date(target, offset)
            av = ds.as_of(date, cfg)
            try:
                res = run_nowcast(av, cfg, target, fill_method=fill_method)
            except ValueError as exc:  # not enough data at this date
                if verbose:
                    print(f"  {target} {hname}: skipped ({exc})")
                continue
            base = {"target": str(target), "horizon": hname, "date": date.date(), "actual": actual_gdp[target]}
            rec = {**base, "model": "components", "nowcast": res.gdp}
            for b in blocks:
                rec[f"c_{b}"] = res.table.loc[b, "contribution"]
                rec[f"a_{b}"] = actual_contrib[b].get(target, np.nan) if b in actual_contrib else np.nan
            rows.append(rec)
            rows.append({**base, "model": "AR(2) GDP", "nowcast": bm.ar_gdp(av.growth["gdp"], cfg, target)})
            rows.append({**base, "model": "Mean", "nowcast": bm.historical_mean(av.growth["gdp"], cfg)})
            rows.append({**base, "model": "Direct GDP bridge",
                         "nowcast": bm.direct_bridge(av.growth["gdp"], res.X, cfg, target)})
        if verbose:
            print(f"  {target} ({i + 1}/{len(targets)})")
    out = pd.DataFrame(rows)
    out["error"] = out["nowcast"] - out["actual"]
    return out


def _dm_test(e_model: np.ndarray, e_bench: np.ndarray, lag: int = 1) -> tuple[float, float]:
    """Diebold-Mariano, squared loss, Newey-West variance. Stat > 0: the model beats the benchmark."""
    d = e_bench ** 2 - e_model ** 2
    T = len(d)
    if T < 8:
        return np.nan, np.nan
    dc = d - d.mean()
    var = dc @ dc / T
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / math.sqrt(var / T)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p


def metrics(bt: pd.DataFrame) -> pd.DataFrame:
    """RMSFE, MAE, RMSFE relative to the AR(2) and DM test of the component model."""
    out = []
    for h, g in bt.groupby("horizon", sort=False):
        piv = g.pivot_table(index="target", columns="model", values="error")
        ar = piv["AR(2) GDP"]
        for model in piv.columns:
            e = piv[model].dropna()
            row = {"horizon": h, "model": model, "n": len(e),
                   "RMSFE": float(np.sqrt((e ** 2).mean())), "MAE": float(e.abs().mean()),
                   "RMSFE / AR(2)": float(np.sqrt((e ** 2).mean()) / np.sqrt((ar.loc[e.index] ** 2).mean()))}
            if model != "components":
                both = piv[["components", model]].dropna()
                row["DM vs components"], row["p-value"] = _dm_test(both["components"].to_numpy(),
                                                                    both[model].to_numpy())
            out.append(row)
    return pd.DataFrame(out)


def error_by_component(bt: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """RMSE of the contribution error by block and horizon (cf. Higgins, table 6)."""
    comp = bt[bt["model"] == "components"]
    out = {}
    for b in cfg.blocks:
        err = comp[f"c_{b}"] - comp[f"a_{b}"]
        out[b] = err.groupby(comp["horizon"], sort=False).apply(lambda e: float(np.sqrt((e ** 2).mean())))
    return pd.DataFrame(out)


def track(ds: Dataset, cfg: Config, target: str, dates: list[pd.Timestamp],
          fill_method: str = "ar") -> pd.DataFrame:
    """Nowcast of the same quarter at several dates: the GDPNow evolution chart."""
    target = pd.Period(target, freq="Q")
    rows = []
    for d in dates:
        res = run_nowcast(ds.as_of(d, cfg), cfg, target, fill_method=fill_method)
        row = {"date": pd.Timestamp(d)}
        row.update(res.table["contribution"].drop("gdp").to_dict())
        row["gdp"] = res.gdp
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")
