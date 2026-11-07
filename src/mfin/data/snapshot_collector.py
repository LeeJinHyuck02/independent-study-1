"""SPX 옵션체인 전체의 하루치 스냅샷을 저장한다.

거래일마다 한 번, 반드시 미국 장중에 실행한다. 예: 15:30 ET (겨울 한국 시간 05:30,
서머타임 기간 04:30). 장이 끝나면 Yahoo가 매수·매도 호가를 지우는 것으로 보이며,
종목마다 체결 시각이 다른 마지막 체결가만 남는다.

    python -m mfin.data.snapshot_collector

yfinance는 과거 옵션체인을 제공하지 않으므로 놓친 날은 나중에 받을 수 없다.
파일은 data_cache/snapshots/<종목>_<YYYYMMDD>.parquet에 시세 날짜로 저장된다.
자동 실행용 작업 스케줄러 명령은 README를 참고한다.
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
    two_sided = float(((chain["bid"] > 0) & (chain["ask"] > 0)).mean())
    print(f"saved {len(chain)} contracts, {n_exp} expiries, spot {chain['spot'].iloc[0]:.2f}, "
          f"{two_sided:.0%} with live bid/ask -> {path}")
    if two_sided < 0.5:
        print("WARNING: most contracts have no bid/ask (market closed?). Collect during US market hours.")
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
