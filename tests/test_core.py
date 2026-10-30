import numpy as np
import pandas as pd
import pytest

from eanowcast import data as data_mod
from eanowcast.backtest import backtest, error_by_component, metrics
from eanowcast.config import load_config
from eanowcast.data import download, load_from_store
from eanowcast.eurostat import jsonstat_to_frame, parse_period
from eanowcast.nowcast import run_nowcast
from eanowcast.store import Store
from eanowcast.synthetic import make_dataset


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def ds(cfg):
    return make_dataset(cfg)


def make_jsonstat(dims: dict[str, list[str]], values: dict[tuple, float]) -> dict:
    """Build a JSON-stat 2.0 payload in Eurostat format (dict index, sparse values)."""
    ids = list(dims)
    sizes = [len(v) for v in dims.values()]
    strides = [int(np.prod(sizes[j + 1:])) for j in range(len(sizes))]
    flat = {}
    for key, v in values.items():
        pos = sum(dims[d].index(k) * strides[j] for j, (d, k) in enumerate(zip(ids, key)))
        flat[str(pos)] = v
    return {
        "version": "2.0", "class": "dataset", "id": ids, "size": sizes, "value": flat,
        "dimension": {d: {"category": {"index": {c: i for i, c in enumerate(codes)}}} for d, codes in dims.items()},
    }


def test_jsonstat_parsing_row_major_and_missing():
    dims = {"freq": ["Q"], "na_item": ["B1GQ", "P6"], "geo": ["EA21"], "time": ["2026-Q1", "2026-Q2"]}
    payload = make_jsonstat(dims, {("Q", "B1GQ", "EA21", "2026-Q1"): 0.1,
                                   ("Q", "P6", "EA21", "2026-Q2"): 1.5})
    frame = jsonstat_to_frame(payload)
    assert len(frame) == 2
    row = frame[frame["na_item"] == "P6"].iloc[0]
    assert row["time"] == "2026-Q2" and row["value"] == 1.5
    assert parse_period("2026-Q2") == pd.Period("2026Q2", freq="Q")
    assert parse_period("2026-08") == pd.Period("2026-08", freq="M")


def test_store_keeps_vintages(tmp_path):
    st = Store(tmp_path / "db.sqlite")
    s1 = pd.Series({pd.Period("2026Q1", "Q"): 0.1, pd.Period("2026Q2", "Q"): 0.6})
    assert st.write_series("x", s1, "2026-09-07T10:00:00") == 2
    s2 = pd.Series({pd.Period("2026Q1", "Q"): 0.1, pd.Period("2026Q2", "Q"): 0.5})
    assert st.write_series("x", s2, "2026-10-15T10:00:00") == 1  # only the revision is written
    assert st.read_series("x", as_of="2026-09-30")[pd.Period("2026Q2", "Q")] == 0.6
    assert st.read_series("x")[pd.Period("2026Q2", "Q")] == 0.5
    assert st.vintages("x") == ["2026-09-07T10:00:00", "2026-10-15T10:00:00"]


def test_as_of_masks_unpublished(cfg, ds):
    av = ds.as_of("2026-10-15", cfg)
    assert pd.isna(av.monthly.loc["2026-09", "retail_volume"])     # 36-day lag
    assert pd.notna(av.monthly.loc["2026-08", "retail_volume"])
    assert pd.isna(av.monthly.loc["2026-08", "construction"])      # 52-day lag
    assert pd.notna(av.monthly.loc["2026-09", "esi"])              # surveys: end of month
    assert pd.notna(av.growth.loc["2026Q2", "consumption"])
    av2 = ds.as_of("2026-08-15", cfg)
    assert pd.notna(av2.growth.loc["2026Q2", "gdp"]) and pd.isna(av2.growth.loc["2026Q2", "consumption"])


@pytest.mark.parametrize("method", ["ar", "dfm"])
def test_nowcast_aggregates_consistently(cfg, ds, method):
    res = run_nowcast(ds.as_of("2026-10-15", cfg), cfg, "2026Q4", fill_method=method)
    assert list(res.by_quarter) == [pd.Period("2026Q3", "Q"), pd.Period("2026Q4", "Q")]
    t = res.table
    assert np.isclose(t.loc["gdp", "growth"], t["contribution"].drop("gdp").sum())
    assert t["contribution"].drop("gdp").notna().all()
    assert -5 < res.gdp < 5


def test_flash_pins_residual(cfg, ds):
    # On 15 Aug the Q2 flash is out but not its components
    av = ds.as_of("2026-08-15", cfg)
    res = run_nowcast(av, cfg, "2026Q3")
    q2 = pd.Period("2026Q2", "Q")
    assert q2 in res.gdp_flash_used
    assert np.isclose(res.by_quarter[q2].loc["gdp", "growth"], ds.growth.loc[q2, "gdp"])


def test_backtest_runs(cfg, ds):
    bt = backtest(ds, cfg, "2025Q3", "2026Q2", verbose=False)
    assert set(bt["model"]) == {"components", "AR(2) GDP", "Mean", "Direct GDP bridge"}
    m = metrics(bt)
    assert (m["RMSFE"] > 0).all()
    assert not error_by_component(bt, cfg).empty


def test_download_with_mocked_api(cfg, tmp_path, monkeypatch, ds):
    """Download writes growth (from chain-linked volumes) and levels to the database."""
    names = {b.na_item: b.name for b in cfg.component_blocks}
    names[cfg.na["gdp_item"]] = "gdp"
    quarters = [f"{q.year}-Q{q.quarter}" for q in ds.levels.index]

    def fake_fetch_jsonstat(dataset, filters, since=None, session=None):
        unit = filters["unit"]
        if unit == "CLV20_MEUR":
            raise data_mod.EurostatError("unit not available")
        dims = {"unit": [unit], "na_item": list(filters["na_item"]), "time": quarters}
        vals = {}
        for item in filters["na_item"]:
            col = names[item]
            if unit == "CLV15_MEUR":
                series = 1000 * (1 + ds.growth[col] / 100).cumprod()
            else:
                series = ds.levels[col]
            for q, v in zip(quarters, series.to_numpy()):
                vals[(unit, item, q)] = float(v)
        return make_jsonstat(dims, vals)

    def fake_fetch_series(dataset, filters, since=None, session=None):
        name = next(n for n, ind in cfg.indicators.items()
                    if ind.dataset == dataset and all(filters.get(k) == v for k, v in ind.filters.items()))
        return ds.monthly[name].dropna()

    monkeypatch.setattr(data_mod, "fetch_jsonstat", fake_fetch_jsonstat)
    monkeypatch.setattr(data_mod, "fetch_series", fake_fetch_series)
    st = Store(tmp_path / "db.sqlite")
    download(cfg, st, verbose=False)
    loaded = load_from_store(cfg, st)
    g = loaded.growth.loc["2010Q1":"2026Q2", "exports"]
    assert np.allclose(g, ds.growth.loc["2010Q1":"2026Q2", "exports"], atol=1e-8)
    assert np.allclose(loaded.monthly["esi"].dropna(), ds.monthly["esi"].dropna())


def test_download_flash_fallback(cfg, tmp_path, monkeypatch, ds):
    """A quarter published only as a growth rate (GDP flash) is appended to the level-based growth."""
    names = {b.na_item: b.name for b in cfg.component_blocks}
    names[cfg.na["gdp_item"]] = "gdp"
    quarters = [f"{q.year}-Q{q.quarter}" for q in ds.levels.index]

    def fake_fetch_jsonstat(dataset, filters, since=None, session=None):
        unit = filters["unit"]
        if unit == "CLV20_MEUR":
            raise data_mod.EurostatError("unit not available")
        items = list(filters["na_item"])
        if unit == "CLV_PCH_PRE":  # growth rate: one more quarter for GDP only (the flash)
            time = quarters + ["2026-Q3"]
        else:
            time = quarters
        dims = {"unit": [unit], "na_item": items, "time": time}
        vals = {}
        for item in items:
            col = names[item]
            if unit == "CLV15_MEUR":
                series = list((1000 * (1 + ds.growth[col].fillna(0) / 100).cumprod()).to_numpy())
            elif unit == "CLV_PCH_PRE":
                series = list(ds.growth[col].round(1).to_numpy()) + ([0.3] if col == "gdp" else [np.nan])
            else:
                series = list(ds.levels[col].to_numpy())
            for q, v in zip(time, series):
                if not np.isnan(v):
                    vals[(unit, item, q)] = float(v)
        return make_jsonstat(dims, vals)

    def fake_fetch_series(dataset, filters, since=None, session=None):
        name = next(n for n, ind in cfg.indicators.items()
                    if ind.dataset == dataset and all(filters.get(k) == v for k, v in ind.filters.items()))
        return ds.monthly[name].dropna()

    monkeypatch.setattr(data_mod, "fetch_jsonstat", fake_fetch_jsonstat)
    monkeypatch.setattr(data_mod, "fetch_series", fake_fetch_series)
    st = Store(tmp_path / "db.sqlite")
    download(cfg, st, verbose=False)
    loaded = load_from_store(cfg, st)
    flash = pd.Period("2026Q3", freq="Q")
    assert loaded.growth.loc[flash, "gdp"] == 0.3            # flash taken from the growth rate
    assert pd.isna(loaded.growth.loc[flash, "exports"])      # components not published yet
    g = loaded.growth.loc["2010Q1":"2026Q2", "gdp"]
    assert np.allclose(g, ds.growth.loc["2010Q1":"2026Q2", "gdp"], atol=1e-8)  # history stays level-based
