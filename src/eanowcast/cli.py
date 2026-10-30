"""Command line: eanowcast <command>  (or python -m eanowcast <command>).

  fetch     download the Eurostat series into the database (new timestamped vintage)
  nowcast   nowcast of the target quarter with the data available today
  track     evolution of the nowcast of one quarter, in pseudo real time, + chart
  backtest  pseudo-real-time evaluation against the benchmarks
  history   chart of the nowcasts actually saved over time (true vintages)
  demo      the whole pipeline on synthetic data (offline)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import benchmarks as bm
from .backtest import backtest, error_by_component, metrics, track
from .config import load_config
from .data import download, load_from_store
from .nowcast import run_nowcast
from .report import format_table, tracking_chart
from .store import Store

DEFAULT_DB = "data/eanowcast.sqlite"


def _default_target(as_of: pd.Timestamp) -> pd.Period:
    return pd.Period(as_of, freq="Q")


def _print_nowcast(res, ds, cfg) -> None:
    for q, table in res.by_quarter.items():
        flag = " (pinned to the GDP flash)" if q in res.gdp_flash_used else ""
        print(f"\n=== {q}{flag} ===")
        print(format_table(table))
    t = res.target
    print(f"\nBenchmarks {t}: AR(2) {bm.ar_gdp(ds.growth['gdp'], cfg, t):+.2f}% | "
          f"direct GDP bridge {bm.direct_bridge(ds.growth['gdp'], res.X, cfg, t):+.2f}% | "
          f"historical mean {bm.historical_mean(ds.growth['gdp'], cfg):+.2f}%")


def cmd_fetch(args, cfg):
    store = Store(args.db)
    print(f"Downloading Eurostat data ({cfg.geo}) -> {args.db}")
    download(cfg, store)


def cmd_nowcast(args, cfg):
    store = Store(args.db)
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.today().normalize()
    if args.as_of and args.vintage:  # true vintage saved in the database
        ds = load_from_store(cfg, store, as_of=f"{as_of.date()}T23:59:59")
    else:
        ds = load_from_store(cfg, store)
        if args.as_of:  # retrospective nowcast in pseudo real time (publication lags)
            ds = ds.as_of(as_of, cfg)
    target = pd.Period(args.target, freq="Q") if args.target else _default_target(as_of)
    res = run_nowcast(ds, cfg, target, fill_method=args.method)
    print(f"Nowcast as of {as_of.date()} | last quarter complete by component: {res.last_component_quarter}")
    _print_nowcast(res, ds, cfg)
    if args.save:
        run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        for q, table in res.by_quarter.items():
            store.write_nowcast(run_at, str(as_of.date()), str(q), table)
        print(f"\nSaved to {args.db} (nowcasts table)")


def _dates(start, end, freq):
    return list(pd.date_range(start, end, freq=freq))


def cmd_track(args, cfg, ds=None, outdir=None):
    ds = ds if ds is not None else load_from_store(cfg, Store(args.db))
    target = pd.Period(args.target, freq="Q")
    start = args.start or (target.start_time - pd.Timedelta(days=30)).date()
    end = args.end or (target.end_time + pd.Timedelta(days=25)).date()
    tr = track(ds, cfg, str(target), _dates(start, end, args.freq), fill_method=args.method)
    actual = ds.growth["gdp"].get(target)
    outdir = Path(outdir or args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tr.to_csv(outdir / f"track_{target}.csv")
    png = tracking_chart(tr, cfg, str(target), outdir / f"track_{target}.png",
                         actual=None if pd.isna(actual) else float(actual))
    print(tr.round(2).to_string())
    print(f"\nChart: {png}")


def cmd_backtest(args, cfg, ds=None, outdir=None):
    ds = ds if ds is not None else load_from_store(cfg, Store(args.db))
    end = args.end or str(ds.growth["gdp"].dropna().index.max())
    bt = backtest(ds, cfg, args.start, end, fill_method=args.method)
    m = metrics(bt)
    ec = error_by_component(bt, cfg)
    outdir = Path(outdir or args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    bt.to_csv(outdir / "backtest.csv", index=False)
    m.to_csv(outdir / "metrics.csv", index=False)
    ec.to_csv(outdir / "error_by_component.csv")
    print("\n" + m.round(3).to_string(index=False))
    print("\nRMSE of the contribution error by block (percentage points):")
    print(ec.round(3).to_string())


def cmd_history(args, cfg):
    store = Store(args.db)
    df = store.read_nowcasts(args.target)
    if df.empty:
        print("No saved nowcast for this quarter (run `nowcast --save` regularly).")
        return
    last_runs = df.sort_values("run_at").groupby("as_of")["run_at"].max()
    df = df[df["run_at"].isin(last_runs)]
    wide = df.pivot_table(index="as_of", columns="block", values="contribution")
    wide["gdp"] = df[df["block"] == "gdp"].set_index("as_of")["growth"]
    wide.index = pd.to_datetime(wide.index)
    png = tracking_chart(wide, cfg, args.target, Path(args.out) / f"history_{args.target}.png",
                         title=f"Nowcasts saved in real time, {args.target}")
    print(f"Chart: {png}")


def cmd_demo(args, cfg):
    from .synthetic import make_dataset

    ds = make_dataset(cfg)
    out = Path(args.out) / "demo"
    print("Synthetic data: nowcast as of 2026-10-15")
    av = ds.as_of("2026-10-15", cfg)
    res = run_nowcast(av, cfg, "2026Q4", fill_method=args.method)
    _print_nowcast(res, av, cfg)
    print("\nPseudo-real-time tracking of 2026Q2:")
    ns = argparse.Namespace(target="2026Q2", start=None, end=None, freq="W-FRI", method=args.method)
    cmd_track(ns, cfg, ds=ds, outdir=out)
    print("\nBacktest 2015Q1-2026Q2:")
    ns = argparse.Namespace(start="2015Q1", end="2026Q2", method=args.method)
    cmd_backtest(ns, cfg, ds=ds, outdir=out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="eanowcast", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to an alternative series.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, db=True, out=True, method=True):
        if db:
            sp.add_argument("--db", default=DEFAULT_DB)
        if out:
            sp.add_argument("--out", default="outputs")
        if method:
            sp.add_argument("--method", choices=["ar", "dfm"], default="ar",
                            help="filling of missing months: one AR per series or a dynamic factor model")

    common(sub.add_parser("fetch"), out=False, method=False)
    sp = sub.add_parser("nowcast")
    common(sp, out=False)
    sp.add_argument("--target", help="target quarter, e.g. 2026Q4 (default: current quarter)")
    sp.add_argument("--as-of", help="ISO date: retrospective nowcast in pseudo real time")
    sp.add_argument("--vintage", action="store_true",
                    help="with --as-of: use the vintage actually downloaded at that date")
    sp.add_argument("--save", action="store_true", help="save the nowcast to the database")
    sp = sub.add_parser("track")
    common(sp)
    sp.add_argument("--target", required=True)
    sp.add_argument("--start")
    sp.add_argument("--end")
    sp.add_argument("--freq", default="W-FRI", help="frequency of the nowcast dates (pandas alias)")
    sp = sub.add_parser("backtest")
    common(sp)
    sp.add_argument("--start", default="2010Q1")
    sp.add_argument("--end")
    sp = sub.add_parser("history")
    common(sp, method=False)
    sp.add_argument("--target", required=True)
    common(sub.add_parser("demo"), db=False)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {"fetch": cmd_fetch, "nowcast": cmd_nowcast, "track": cmd_track, "backtest": cmd_backtest,
     "history": cmd_history, "demo": cmd_demo}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
