"""Loads series.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).with_name("series.toml")


@dataclass
class Indicator:
    name: str
    label: str
    dataset: str
    filters: dict
    transform: str  # "pct" | "level"
    release_lag_days: int


@dataclass
class Block:
    name: str
    label: str
    sign: int
    model: str  # "bridge" | "ar"
    na_item: str | None = None
    regressors: list[str] = field(default_factory=list)
    ar_lag: bool = False

    @property
    def is_residual(self) -> bool:
        return self.na_item is None


@dataclass
class Config:
    geo: str
    start_quarter: pd.Period
    gdp_release_lag_days: int
    components_release_lag_days: int
    exclude_quarters: list[pd.Period]
    na: dict
    indicators: dict[str, Indicator]
    blocks: dict[str, Block]  # estimation order = order in the file

    @property
    def component_blocks(self) -> list[Block]:
        return [b for b in self.blocks.values() if not b.is_residual]


def load_config(path: str | Path | None = None) -> Config:
    raw = tomllib.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    g = raw["general"]
    indicators = {
        name: Indicator(name=name, **spec) for name, spec in raw["indicators"].items()
    }
    blocks = {name: Block(name=name, **spec) for name, spec in raw["blocks"].items()}
    for b in blocks.values():
        for r in b.regressors:
            if r.startswith("block:"):
                dep = r.split(":", 1)[1]
                order = list(blocks)
                if dep not in blocks or order.index(dep) >= order.index(b.name):
                    raise ValueError(f"Block {b.name}: '{r}' must refer to a block defined before it")
            elif r not in indicators:
                raise ValueError(f"Block {b.name}: unknown indicator '{r}'")
    if sum(b.is_residual for b in blocks.values()) != 1:
        raise ValueError("Exactly one residual block (without na_item) is required")
    return Config(
        geo=g["geo"],
        start_quarter=pd.Period(g["start_quarter"], freq="Q"),
        gdp_release_lag_days=g["gdp_release_lag_days"],
        components_release_lag_days=g["components_release_lag_days"],
        exclude_quarters=[pd.Period(q, freq="Q") for q in g.get("exclude_quarters", [])],
        na=raw["national_accounts"],
        indicators=indicators,
        blocks=blocks,
    )
