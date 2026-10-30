"""Data download, storage and shaping.

A `Dataset` holds:
- `monthly`: raw monthly indicators (indices or survey balances), monthly PeriodIndex;
- `growth` : quarterly volume growth (%, q/q) of GDP and of each component;
- `levels` : quarterly levels at current prices (EUR million), for the aggregation weights.

`Dataset.as_of(date)` rebuilds what had been published at a given date from the
publication lags (pseudo real time).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import Config
from .eurostat import EurostatError, fetch_jsonstat, fetch_series, jsonstat_to_frame, parse_period
from .store import Store


def _sid_growth(item: str) -> str:
    return f"na:growth:{item}"


def _sid_level(item: str) -> str:
    return f"na:level:{item}"


def _sid_ind(name: str) -> str:
    return f"ind:{name}"


@dataclass
class Dataset:
    monthly: pd.DataFrame
    growth: pd.DataFrame
    levels: pd.DataFrame

    def as_of(self, date: pd.Timestamp | str, cfg: Config) -> "Dataset":
        """Data available at `date`, given the publication lags in the config."""
        date = pd.Timestamp(date)
        monthly = self.monthly.copy()
        for name, ind in cfg.indicators.items():
            if name not in monthly:
                continue
            published = monthly.index.to_timestamp(how="end") + pd.Timedelta(days=ind.release_lag_days)
            monthly.loc[published > date, name] = float("nan")
        q_end = self.growth.index.to_timestamp(how="end")
        growth = self.growth.copy()
        comp_cols = [c for c in growth.columns if c != "gdp"]
        growth.loc[q_end + pd.Timedelta(days=cfg.components_release_lag_days) > date, comp_cols] = float("nan")
        growth.loc[q_end + pd.Timedelta(days=cfg.gdp_release_lag_days) > date, "gdp"] = float("nan")
        lq_end = self.levels.index.to_timestamp(how="end")
        levels = self.levels.copy()
        levels.loc[lq_end + pd.Timedelta(days=cfg.components_release_lag_days) > date, :] = float("nan")
        return Dataset(monthly=monthly, growth=growth, levels=levels)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(cfg: Config, store: Store, since_q: str = "1999-Q1", since_m: str = "1999-01",
             verbose: bool = True) -> dict[str, int]:
    """Download every series in the config and write it to the store (timestamped vintage)."""
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    http = requests.Session()
    written: dict[str, int] = {}
    na = cfg.na
    items = [na["gdp_item"]] + [b.na_item for b in cfg.component_blocks]

    def na_frame(unit: str) -> dict[str, pd.Series]:
        payload = fetch_jsonstat(
            na["dataset"],
            {"geo": cfg.geo, "unit": unit, "s_adj": na["s_adj"], "na_item": items},
            since=since_q, session=http,
        )
        frame = jsonstat_to_frame(payload)
        out = {}
        for item in items:
            sub = frame[frame["na_item"] == item]
            out[item] = pd.Series(sub["value"].to_numpy(),
                                  index=[parse_period(t) for t in sub["time"]], dtype=float).sort_index()
        return out

    # Volume growth: recomputed from chain-linked volume levels when available
    growth, source = None, na["growth_unit"]
    for unit in na.get("volume_level_units", []):
        try:
            lv = na_frame(unit)
        except EurostatError:
            continue
        if all(len(s) > 40 for s in lv.values()):
            growth = {k: 100 * s.pct_change().dropna() for k, s in lv.items()}
            source = unit
            break
    if growth is None:
        growth = na_frame(na["growth_unit"])
    if verbose:
        print(f"  volume growth computed from: {source}")
    if source != na["growth_unit"]:
        # The GDP flash (t+30) may be released as a growth rate before the chain-linked level:
        # append the quarters that only exist in the published growth rate (rounded to 0.1).
        # The level-based value replaces it later, as a revision in the store.
        try:
            published = na_frame(na["growth_unit"])
        except EurostatError:
            published = {}
        for item, s in published.items():
            extra = s[s.index > growth[item].index.max()].dropna()
            if len(extra):
                growth[item] = pd.concat([growth[item], extra])
                if verbose:
                    print(f"  {item}: {', '.join(map(str, extra.index))} from {na['growth_unit']} "
                          "(chain-linked level not yet published)")
    for item, s in growth.items():
        written[_sid_growth(item)] = store.write_series(_sid_growth(item), s, retrieved_at)
    for item, s in na_frame(na["level_unit"]).items():
        written[_sid_level(item)] = store.write_series(_sid_level(item), s, retrieved_at)
    for name, ind in cfg.indicators.items():
        time.sleep(0.5)  # be gentle with the API (rate limit)
        s = fetch_series(ind.dataset, {"geo": cfg.geo, **ind.filters}, since=since_m, session=http)
        written[_sid_ind(name)] = store.write_series(_sid_ind(name), s, retrieved_at)
    if verbose:
        for k, v in written.items():
            print(f"  {k:<32} {v:>5} new or revised values")
    return written


def load_from_store(cfg: Config, store: Store, as_of: str | None = None) -> Dataset:
    """Build a Dataset from the store (latest vintage, or the vintage known at `as_of`)."""
    na = cfg.na
    growth = {"gdp": store.read_series(_sid_growth(na["gdp_item"]), as_of)}
    levels = {"gdp": store.read_series(_sid_level(na["gdp_item"]), as_of)}
    for b in cfg.component_blocks:
        growth[b.name] = store.read_series(_sid_growth(b.na_item), as_of)
        levels[b.name] = store.read_series(_sid_level(b.na_item), as_of)
    monthly = {name: store.read_series(_sid_ind(name), as_of) for name in cfg.indicators}
    if all(s.empty for s in growth.values()):
        raise RuntimeError("Empty database: run `eanowcast fetch` first.")
    return Dataset(
        monthly=_to_frame(monthly, "M"),
        growth=_to_frame(growth, "Q"),
        levels=_to_frame(levels, "Q"),
    )


def _to_frame(series: dict[str, pd.Series], freq: str) -> pd.DataFrame:
    frame = pd.DataFrame({k: v for k, v in series.items()})
    if frame.empty:
        return frame
    start, end = frame.index.min(), frame.index.max()
    return frame.reindex(pd.period_range(start, end, freq=freq))
