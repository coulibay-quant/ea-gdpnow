"""SQLite store for observations, with vintages.

Every download is timestamped (`retrieved_at`). A row is only inserted when the value
is new or has been revised, so the table keeps the full revision history and any
vintage can be rebuilt with `read_series(as_of=...)`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    series_id    TEXT NOT NULL,
    period       TEXT NOT NULL,
    value        REAL NOT NULL,
    retrieved_at TEXT NOT NULL,
    PRIMARY KEY (series_id, period, retrieved_at)
);
CREATE INDEX IF NOT EXISTS idx_obs_series ON observations(series_id, period);

CREATE TABLE IF NOT EXISTS nowcasts (
    run_at        TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    target        TEXT NOT NULL,
    block         TEXT NOT NULL,
    growth        REAL,
    weight        REAL,
    contribution  REAL,
    PRIMARY KEY (run_at, target, block)
);
"""

# Latest vintage: the most recent value of each (series, period) known at a given time
LATEST_SQL = """
SELECT o.period, o.value
FROM observations o
JOIN (
    SELECT series_id, period, MAX(retrieved_at) AS last
    FROM observations
    WHERE series_id = ? AND retrieved_at <= ?
    GROUP BY series_id, period
) m ON o.series_id = m.series_id AND o.period = m.period AND o.retrieved_at = m.last
"""


def _period_to_str(p: pd.Period) -> str:
    return str(p)


def _str_to_period(s: str) -> pd.Period:
    return pd.Period(s, freq="Q" if "Q" in s else "M")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.executescript(SCHEMA)

    def close(self) -> None:
        self.con.close()

    # -- observations -------------------------------------------------------
    def write_series(self, series_id: str, s: pd.Series, retrieved_at: str) -> int:
        """Insert new or revised values. Returns the number of rows written."""
        current = self.read_series(series_id)
        rows = []
        for period, value in s.dropna().items():
            key = period
            if key in current.index and abs(current[key] - value) < 1e-9:
                continue
            rows.append((series_id, _period_to_str(period), float(value), retrieved_at))
        self.con.executemany("INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?)", rows)
        self.con.commit()
        return len(rows)

    def read_series(self, series_id: str, as_of: str | None = None) -> pd.Series:
        """Series as it was known at time `as_of` (ISO string). None = latest vintage."""
        as_of = as_of or "9999-12-31T23:59:59"
        cur = self.con.execute(LATEST_SQL, (series_id, as_of))
        data = cur.fetchall()
        if not data:
            return pd.Series(dtype=float)
        s = pd.Series({_str_to_period(p): v for p, v in data})
        return s.sort_index()

    def series_ids(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT series_id FROM observations")]

    def vintages(self, series_id: str) -> list[str]:
        cur = self.con.execute(
            "SELECT DISTINCT retrieved_at FROM observations WHERE series_id = ? ORDER BY 1",
            (series_id,),
        )
        return [r[0] for r in cur]

    # -- nowcasts -----------------------------------------------------------
    def write_nowcast(self, run_at: str, as_of: str, target: str, table: pd.DataFrame) -> None:
        rows = [
            (run_at, as_of, target, block, r.get("growth"), r.get("weight"), r.get("contribution"))
            for block, r in table.iterrows()
        ]
        self.con.executemany("INSERT OR REPLACE INTO nowcasts VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        self.con.commit()

    def read_nowcasts(self, target: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM nowcasts WHERE target = ? ORDER BY as_of, run_at", self.con, params=(target,)
        )
