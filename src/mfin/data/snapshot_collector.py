"""Save one daily snapshot of the full SPX option chain.

Run once per trading day after the US close (06:00 KST in winter):

    python -m mfin.data.snapshot_collector

The file goes to data_cache/snapshots/<ticker>_<YYYYMMDD>.parquet, dated by the
quote time (a Saturday run saves Friday's chain). See README for a Task Scheduler
command that runs this automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mfin.data.option_chain import fetch_chain
from mfin.paths import SNAPSHOT_DIR


def collect(ticker: str = "^SPX", max_days: float = 400, fallback: str | None = None) -> Path:
    chain = fetch_chain(ticker, min_days=0, max_days=max_days, fallback=fallback, refresh=True)
    name = str(chain["underlying"].iloc[0]).replace("^", "")
    day = chain["quote_time"].iloc[0].strftime("%Y%m%d")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}_{day}.parquet"
    chain.to_parquet(path)
    n_exp = chain["expiry"].nunique()
    print(f"saved {len(chain)} contracts, {n_exp} expiries, spot {chain['spot'].iloc[0]:.2f} -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="^SPX")
    ap.add_argument("--max-days", type=float, default=400)
    ap.add_argument("--fallback", default=None, help="ticker to use if the main one fails (e.g. SPY)")
    args = ap.parse_args()
    collect(args.ticker, args.max_days, args.fallback)


if __name__ == "__main__":
    main()
