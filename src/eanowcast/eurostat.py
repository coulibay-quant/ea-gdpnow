"""Minimal client for the Eurostat dissemination API (JSON-stat 2.0 format).

Endpoint: https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}
No key is required. A repeated parameter (na_item=P6&na_item=P7) selects several
categories of the same dimension.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

import pandas as pd
import requests

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


class EurostatError(RuntimeError):
    pass


def fetch_jsonstat(
    dataset: str,
    filters: Mapping[str, Any],
    since: str | None = None,
    session: requests.Session | None = None,
    retries: int = 4,
    timeout: int = 60,
) -> dict:
    """Download a JSON-stat extract. Values in `filters` may be lists."""
    params: dict[str, Any] = {k: v for k, v in filters.items()}
    params.setdefault("lang", "EN")
    if since:
        params["sinceTimePeriod"] = since
    http = session or requests.Session()
    url = f"{BASE_URL}/{dataset}"
    for attempt in range(retries):
        resp = http.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt * 5)
            continue
        if resp.status_code != 200:
            raise EurostatError(f"{dataset}: HTTP {resp.status_code} – {resp.text[:300]}")
        payload = resp.json()
        if "error" in payload:
            raise EurostatError(f"{dataset}: {payload['error']}")
        return payload
    raise EurostatError(f"{dataset}: failed after {retries} attempts (HTTP {resp.status_code})")


def _category_codes(dim: dict) -> list[str]:
    """Codes of a dimension, sorted by position (index given as dict or list)."""
    index = dim["category"]["index"]
    if isinstance(index, list):
        return list(index)
    return [code for code, _ in sorted(index.items(), key=lambda kv: kv[1])]


def jsonstat_to_frame(payload: dict) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 dataset into a long DataFrame (one column per dimension + value).

    Positions without data are left out. Storage order is row-major: the last
    dimension in `id` varies fastest.
    """
    dim_ids: list[str] = payload["id"]
    sizes: list[int] = payload["size"]
    codes = [_category_codes(payload["dimension"][d]) for d in dim_ids]
    values = payload.get("value", {})
    if isinstance(values, list):
        items = [(i, v) for i, v in enumerate(values) if v is not None]
    else:
        items = [(int(k), v) for k, v in values.items() if v is not None]

    # Stride of each dimension in the flat index
    strides = [1] * len(sizes)
    for j in range(len(sizes) - 2, -1, -1):
        strides[j] = strides[j + 1] * sizes[j + 1]

    rows = []
    for flat, val in items:
        row = {}
        for j, d in enumerate(dim_ids):
            row[d] = codes[j][(flat // strides[j]) % sizes[j]]
        row["value"] = float(val)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=dim_ids + ["value"])
    return frame


def parse_period(code: str) -> pd.Period:
    """'2026-Q2' -> Period('2026Q2', 'Q'); '2026-08' -> Period('2026-08', 'M')."""
    if "-Q" in code:
        return pd.Period(code.replace("-", ""), freq="Q")
    return pd.Period(code, freq="M")


def fetch_series(dataset: str, filters: Mapping[str, Any], since: str | None = None,
                 session: requests.Session | None = None) -> pd.Series:
    """Download ONE series (every filter must select a single category)."""
    frame = jsonstat_to_frame(fetch_jsonstat(dataset, filters, since=since, session=session))
    if frame.empty:
        raise EurostatError(f"{dataset} {dict(filters)}: no observations")
    varying = [c for c in frame.columns if c not in ("time", "value") and frame[c].nunique() > 1]
    if varying:
        raise EurostatError(f"{dataset}: ambiguous filters, several categories in {varying}")
    s = pd.Series(frame["value"].to_numpy(), index=[parse_period(t) for t in frame["time"]])
    return s.sort_index()
