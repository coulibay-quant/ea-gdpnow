"""Outputs: nowcast evolution chart (GDPNow style) and tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure  # object API: no global backend forced (notebook friendly)

from .config import Config

# Validated categorical palette (fixed order, never cycled) and text inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def tracking_chart(tr: pd.DataFrame, cfg: Config, target: str, path: str | Path,
                   actual: float | None = None, title: str | None = None) -> Path:
    """Stacked bars of contributions by nowcast date + line for the GDP nowcast."""
    blocks = [b for b in cfg.blocks if b in tr.columns]
    labels = {b: cfg.blocks[b].label for b in blocks}
    fig = Figure(figsize=(10, 5.2), dpi=150, facecolor=SURFACE)
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    x = mdates.date2num(tr.index.to_pydatetime())
    width = 0.7 * (x[1] - x[0]) if len(x) > 1 else 3
    pos_base = pd.Series(0.0, index=tr.index)
    neg_base = pd.Series(0.0, index=tr.index)
    for i, b in enumerate(blocks):
        v = tr[b].fillna(0.0)
        bottom = pos_base.where(v >= 0, neg_base)
        ax.bar(x, v, width=width, bottom=bottom, color=SERIES[i], edgecolor=SURFACE, linewidth=1.0,
               label=labels[b], zorder=2)
        pos_base = pos_base + v.clip(lower=0)
        neg_base = neg_base + v.clip(upper=0)
    ax.plot(x, tr["gdp"], color=INK, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="GDP (nowcast)", zorder=3)
    last = tr["gdp"].iloc[-1]
    ax.annotate(f"{last:+.2f}%", (x[-1], last), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, fontweight="bold", color=INK)
    if actual is not None:
        ax.axhline(actual, color=INK_2, linewidth=1.2, linestyle="--", zorder=1,
                   label=f"Published GDP: {actual:+.2f}%")
    ax.axhline(0, color=INK_2, linewidth=0.8, zorder=1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("q/q growth and contributions (percentage points)", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(title or f"Euro area GDP nowcast, {target}: evolution and contributions",
                 loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    return path


def format_table(table: pd.DataFrame) -> str:
    t = table.copy()
    t["growth"] = t["growth"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}%")
    t["weight"] = t["weight"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    t["contribution"] = t["contribution"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f} pt")
    t.columns = ["Block", "Growth", "Weight", "Contribution"]
    return t.to_string(index=False)
