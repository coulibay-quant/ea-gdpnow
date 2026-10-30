"""Synthetic data with the same structure as the Eurostat data.

Used by the tests and the offline demo (`eanowcast demo`). Real data come from
`eanowcast fetch`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import Dataset


def make_dataset(cfg: Config, seed: int = 7, last_month: str = "2026-09",
                 last_quarter: str = "2026Q2") -> Dataset:
    rng = np.random.default_rng(seed)
    months = pd.period_range("1999-01", last_month, freq="M")
    n = len(months)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.normal(0, 0.6)
    smooth = pd.Series(f).ewm(alpha=0.3).mean().to_numpy()

    def index(drift, beta, noise):
        dlog = drift + beta * f + rng.normal(0, noise, n)
        return 100 * np.exp(np.cumsum(dlog) / 100)

    monthly = pd.DataFrame(
        {
            "retail_volume": index(0.08, 0.3, 0.4),
            "ip_total": index(0.08, 0.8, 0.8),
            "ip_capital_goods": index(0.10, 1.0, 1.2),
            "construction": index(0.05, 0.5, 1.2),
            "esi": 100 + 5 * smooth + rng.normal(0, 1.0, n),
            "consumer_conf": -12 + 3 * smooth + rng.normal(0, 1.0, n),
            "retail_conf": -6 + 3 * smooth + rng.normal(0, 1.5, n),
            "construction_conf": -10 + 4 * smooth + rng.normal(0, 2.0, n),
            "export_orders": -15 + 6 * smooth + rng.normal(0, 2.0, n),
            "stocks_finished": 8 - 2 * smooth + rng.normal(0, 1.5, n),
        },
        index=months,
    )
    monthly = monthly[[c for c in cfg.indicators if c in monthly]]

    # "True" quarterly indicators used to generate the components
    qidx = months.asfreq("Q")
    qavg = monthly.groupby(qidx).mean()
    qpct = 100 * (qavg / qavg.shift(1) - 1)
    quarters = pd.period_range("1999Q1", last_quarter, freq="Q")
    qpct = qpct.reindex(quarters).fillna(0.0)
    m = len(quarters)

    g = pd.DataFrame(index=quarters, dtype=float)
    g["consumption"] = 0.25 + 0.6 * qpct["retail_volume"] + rng.normal(0, 0.2, m)
    gov = np.zeros(m)
    for t in range(1, m):
        gov[t] = 0.25 + 0.3 * gov[t - 1] + rng.normal(0, 0.25)
    g["government"] = gov
    g["investment"] = 0.2 + 0.5 * qpct["construction"] + 0.3 * qpct["ip_capital_goods"] + rng.normal(0, 0.6, m)
    g["exports"] = 0.5 + 0.9 * qpct["ip_total"] + rng.normal(0, 0.7, m)
    g["imports"] = 0.2 + 0.5 * g["consumption"] + 0.4 * g["investment"] + 0.6 * g["exports"] + rng.normal(0, 0.6, m)

    # Nominal levels: realistic initial shares, 0.5% inflation per quarter
    shares0 = {"consumption": 0.52, "government": 0.21, "investment": 0.21, "exports": 0.48, "imports": 0.45}
    levels = pd.DataFrame(index=quarters, columns=list(shares0) + ["gdp"], dtype=float)
    lvl = {k: v * 2_500_000 for k, v in shares0.items()}
    inv_share = 0.03
    for i, q in enumerate(quarters):
        if i > 0:
            for k in lvl:
                lvl[k] *= (1 + g.loc[q, k] / 100) * 1.005
        inv = (inv_share + rng.normal(0, 0.004)) * 2_500_000 * 1.0065 ** i
        levels.loc[q, list(lvl)] = list(lvl.values())
        levels.loc[q, "gdp"] = lvl["consumption"] + lvl["government"] + lvl["investment"] + lvl["exports"] - lvl["imports"] + inv

    # GDP growth = sum of contributions (previous-year shares) + inventories
    gdp = []
    for q in quarters:
        prev = levels.loc[(levels.index.year == q.year - 1)]
        base = prev if len(prev) == 4 else levels.loc[:q].tail(4)
        w = base[list(shares0)].sum() / base["gdp"].sum()
        sign = pd.Series({"consumption": 1, "government": 1, "investment": 1, "exports": 1, "imports": -1})
        gdp.append(float((sign * w * g.loc[q, list(shares0)]).sum()) + rng.normal(0, 0.15))
    g.insert(0, "gdp", gdp)
    return Dataset(monthly=monthly, growth=g, levels=levels)
