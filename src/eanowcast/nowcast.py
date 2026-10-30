"""Core of the nowcast: one bridge equation per component, then re-aggregation into contributions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import Block, Config
from .data import Dataset
from .fill import fill_monthly
from .transforms import historical_contributions, nominal_weights, to_quarterly


@dataclass
class NowcastResult:
    target: pd.Period
    table: pd.DataFrame                      # components of the target quarter
    by_quarter: dict[pd.Period, pd.DataFrame]  # every forecast quarter (backcasts included)
    models: dict[str, object] = field(default_factory=dict)
    last_component_quarter: pd.Period | None = None
    gdp_flash_used: list[pd.Period] = field(default_factory=list)
    X: pd.DataFrame | None = None            # quarterly indicators after filling

    @property
    def gdp(self) -> float:
        return float(self.table.loc["gdp", "growth"])


def _regressor_value(name: str, q: pd.Period, X: pd.DataFrame, hist: pd.DataFrame,
                     pred: pd.DataFrame) -> float:
    if name.startswith("block:"):
        b = name.split(":", 1)[1]
        if q in pred.index and pd.notna(pred.loc[q, b]):
            return pred.loc[q, b]
        return hist.loc[q, b] if q in hist.index else np.nan
    return X.loc[q, name] if q in X.index else np.nan


def _lag_value(block: str, q: pd.Period, hist: pd.DataFrame, pred: pd.DataFrame) -> float:
    prev = q - 1
    if prev in pred.index and pd.notna(pred.loc[prev, block]):
        return pred.loc[prev, block]
    return hist.loc[prev, block] if prev in hist.index else np.nan


def _design_row(block: Block, q: pd.Period, X, hist, pred) -> dict:
    row = {}
    if block.model == "ar" or block.ar_lag:
        row["lag1"] = _lag_value(block.name, q, hist, pred)
    if block.model == "bridge":
        for r in block.regressors:
            row[r] = _regressor_value(r, q, X, hist, pred)
    return row


def fit_block(block: Block, hist: pd.DataFrame, X: pd.DataFrame, sample: list[pd.Period]):
    empty = pd.DataFrame(columns=hist.columns, dtype=float)
    rows = {q: _design_row(block, q, X, hist, empty) for q in sample}
    design = pd.DataFrame(rows).T
    data = design.join(hist[block.name].rename("y")).dropna()
    if len(data) < 20:
        raise ValueError(f"Block {block.name}: sample too short ({len(data)} quarters)")
    exog = sm.add_constant(data.drop(columns="y"), has_constant="add")
    return sm.OLS(data["y"], exog).fit()


def run_nowcast(ds: Dataset, cfg: Config, target: pd.Period | str, fill_method: str = "ar") -> NowcastResult:
    target = pd.Period(target, freq="Q")
    names = [b.name for b in cfg.component_blocks]
    growth = ds.growth

    complete = growth[names].dropna()
    q_c = complete.index.max()
    quarters = pd.period_range(q_c + 1, target, freq="Q")
    if len(quarters) == 0:
        raise ValueError(f"{target} is already published by component (last complete quarter: {q_c}).")

    monthly = fill_monthly(ds.monthly, cfg, until=target.asfreq("M", how="end"), method=fill_method)
    X = to_quarterly(monthly, cfg)

    contrib_hist = historical_contributions(growth, ds.levels, cfg)
    hist = growth[names].copy()
    hist["residual"] = contrib_hist["residual"].reindex(hist.index)

    sample = [q for q in pd.period_range(cfg.start_quarter, q_c, freq="Q") if q not in cfg.exclude_quarters]
    pred = pd.DataFrame(index=quarters, columns=names + ["residual"], dtype=float)
    models = {}
    weights = {q: nominal_weights(ds.levels, q, names) for q in quarters}
    signs = pd.Series({b.name: b.sign for b in cfg.component_blocks})
    flash_used = []

    for block in cfg.blocks.values():
        res = fit_block(block, hist, X, sample)
        models[block.name] = res
        for q in quarters:
            row = _design_row(block, q, X, hist, pred)
            if any(pd.isna(v) for v in row.values()):
                missing = [k for k, v in row.items() if pd.isna(v)]
                raise ValueError(f"Block {block.name}, {q}: missing regressors {missing}")
            x = np.r_[1.0, [row[k] for k in res.params.index[1:]]]
            pred.loc[q, block.name] = float(x @ res.params.to_numpy())
            # If the GDP flash is already out for q, the residual is pinned to it
            if block.is_residual and pd.notna(growth.get("gdp", pd.Series(dtype=float)).get(q, np.nan)):
                contrib = (signs * weights[q] * pred.loc[q, names]).sum()
                pred.loc[q, "residual"] = growth.loc[q, "gdp"] - contrib
                if q not in flash_used:
                    flash_used.append(q)

    by_quarter = {}
    labels = {b.name: b.label for b in cfg.blocks.values()}
    for q in quarters:
        w = weights[q]
        t = pd.DataFrame(index=names + ["residual", "gdp"], columns=["label", "growth", "weight", "contribution"])
        for n in names:
            t.loc[n] = [labels[n], pred.loc[q, n], w[n], signs[n] * w[n] * pred.loc[q, n]]
        t.loc["residual"] = [labels["residual"], np.nan, np.nan, pred.loc[q, "residual"]]
        t.loc["gdp"] = ["GDP", t.loc[names + ["residual"], "contribution"].sum(), np.nan, np.nan]
        t[["growth", "weight", "contribution"]] = t[["growth", "weight", "contribution"]].astype(float)
        by_quarter[q] = t

    return NowcastResult(target=target, table=by_quarter[target], by_quarter=by_quarter,
                         models=models, last_component_quarter=q_c, gdp_flash_used=flash_used, X=X)
