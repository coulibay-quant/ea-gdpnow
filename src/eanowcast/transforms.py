"""Monthly-to-quarterly conversion and aggregation weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def to_quarterly(monthly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Quarterly average of each indicator (complete quarter required), then transformation.

    - "pct"  : % change of the quarterly average on the previous quarter
    - "level": quarterly average of the survey balance
    """
    out = {}
    q_index = monthly.index.asfreq("Q")
    for name in monthly.columns:
        grouped = monthly[name].groupby(q_index)
        avg = grouped.mean().where(grouped.count() == 3)
        if cfg.indicators[name].transform == "pct":
            out[name] = 100 * (avg / avg.shift(1) - 1)
        else:
            out[name] = avg
    frame = pd.DataFrame(out)
    frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return frame


def nominal_weights(levels: pd.DataFrame, quarter: pd.Period, blocks: list[str]) -> pd.Series:
    """Nominal shares of the previous year (previous-year-prices logic).

    If the previous year is not complete in the available data, the last four
    available quarters before `quarter` are used instead.
    """
    prev_year = [pd.Period(f"{quarter.year - 1}Q{i}", freq="Q") for i in range(1, 5)]
    avail = levels.loc[:quarter - 1].dropna(subset=["gdp"] + blocks)
    window = avail.loc[avail.index.isin(prev_year)]
    if len(window) < 4:
        window = avail.tail(4)
    if window.empty:
        raise ValueError(f"No nominal levels available before {quarter}")
    totals = window.sum()
    return totals[blocks] / totals["gdp"]


def historical_contributions(growth: pd.DataFrame, levels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Historical contributions (percentage points) of each component and implied residual.

    residual = GDP growth - sum of component contributions; it captures the change in
    inventories and the gap due to the non-additivity of chain-linked volumes.
    """
    names = [b.name for b in cfg.component_blocks]
    signs = pd.Series({b.name: b.sign for b in cfg.component_blocks})
    rows = {}
    complete = growth.dropna(subset=["gdp"] + names).index
    for q in complete:
        try:
            w = nominal_weights(levels, q, names)
        except ValueError:
            continue
        contrib = signs * w * growth.loc[q, names]
        contrib["residual"] = growth.loc[q, "gdp"] - contrib.sum()
        rows[q] = contrib
    frame = pd.DataFrame(rows).T
    if not frame.empty:
        frame.index = pd.PeriodIndex(frame.index, freq="Q")
    return frame


def winsorize(s: pd.Series, k: float = 4.0) -> pd.Series:
    """Clip at k median absolute deviations (limits the Covid effect on the monthly ARs)."""
    med = s.median()
    mad = 1.4826 * np.nanmedian(np.abs(s - med))
    if not np.isfinite(mad) or mad == 0:
        return s
    return s.clip(med - k * mad, med + k * mad)
