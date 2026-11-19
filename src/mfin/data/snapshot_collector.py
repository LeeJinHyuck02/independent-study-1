"""SPX 옵션체인 전체의 하루치 스냅샷을 저장한다.

거래일마다 반드시 미국 장중에 실행한다. 장이 끝나면 Yahoo가 매수·매도 호가를 지우는 것으로
보이며, 종목마다 체결 시각이 다른 마지막 체결가만 남는다. 기본 실행 경로는 GitHub Actions
예약 실행(.github/workflows/snapshot.yml)이고, 받은 파일은 data-snapshots 브랜치에 쌓인다.

    python -m mfin.data.snapshot_collector
    python -m mfin.data.snapshot_collector --out-dir store/snapshots --require-today

yfinance는 과거 옵션체인을 제공하지 않으므로 놓친 날은 나중에 받을 수 없다.
파일은 <out-dir>/<종목>_<YYYYMMDD>.parquet에 시세 날짜(뉴욕)로 저장된다.

저장 전 판정
-------------
- 호가: ATM 근방(|K/S - 1| < 10%, 만기 7-180일) 계약 중 매수·매도 호가가 모두 있는 비율이
  기준(기본 0.8) 이상이어야 한다. 전체 체인 비율은 0DTE·deep OTM 때문에 장중에도 낮을 수
  있어 로그로만 남긴다.
- 신선도: 지금 - 시세 시각 < 45분. 휴장일(시세 시각이 전 거래일), 반일장 마감 후, 장 마감 후
  실행을 걸러낸다.
- 같은 날짜 파일이 있으면 시세 시각이 더 늦은 유효 스냅샷이 덮어쓴다.
- 종료코드: 다운로드 실패 1, 판정 미달로 건너뜀 0. ``--require-today``이면 오늘 장이 열렸는데
  (시세 날짜 = 오늘) 오늘 파일이 없을 때 1 (예약 실행의 마지막 회차에서 실패 알림용).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mfin.data.cache import DataUnavailableError
from mfin.data.option_chain import NY, fetch_chain
from mfin.paths import SNAPSHOT_DIR

DAYS_PER_YEAR = 365.0  # option_chain의 T 규약 (달력일 / 365)


@dataclass
class Assessment:
    quote_time: pd.Timestamp
    now: pd.Timestamp
    stale_minutes: float
    two_sided_atm: float
    two_sided_all: float
    n_atm: int
    valid: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class CollectResult:
    status: str  # "saved" | "kept" (같은 날 더 늦은 파일이 이미 있음) | "skipped"
    name: str  # 파일 이름의 종목 부분 (예: "SPX")
    path: Path
    assessment: Assessment


def _now() -> pd.Timestamp:
    """현재 뉴욕 시각 (테스트에서 바꿔 끼운다)."""
    return pd.Timestamp.now(tz=NY)


def _two_sided(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return float(((df["bid"].fillna(0) > 0) & (df["ask"].fillna(0) > 0)).mean())


def assess(
    chain: pd.DataFrame,
    now: pd.Timestamp | None = None,
    min_two_sided: float = 0.8,
    max_stale_min: float = 45.0,
    atm_band: float = 0.10,
    atm_days: tuple[float, float] = (7.0, 180.0),
) -> Assessment:
    """체인이 장중의 살아 있는 호가인지 판정한다 (저장 여부 결정용, 부수효과 없음)."""
    now = _now() if now is None else pd.Timestamp(now).tz_convert(NY)
    quote_time = pd.Timestamp(chain["quote_time"].iloc[0]).tz_convert(NY)
    stale = (now - quote_time).total_seconds() / 60.0

    spot = float(chain["spot"].iloc[0])
    days = chain["T"] * DAYS_PER_YEAR
    near = (np.abs(chain["K"] / spot - 1.0) < atm_band) & days.between(*atm_days)
    atm = chain[near]
    frac_atm, frac_all = _two_sided(atm), _two_sided(chain)

    reasons = []
    if len(atm) == 0:
        reasons.append("no near-the-money contracts")
    elif not frac_atm >= min_two_sided:
        reasons.append(f"near-the-money two-sided quotes {frac_atm:.0%} < {min_two_sided:.0%}")
    if stale > max_stale_min:
        reasons.append(f"quotes are {stale:.0f} min old (> {max_stale_min:g}); market closed?")
    return Assessment(quote_time, now, stale, frac_atm, frac_all, int(len(atm)), not reasons, reasons)


def snapshot_path(out_dir: Path, name: str, day: pd.Timestamp) -> Path:
    return Path(out_dir) / f"{name}_{day.strftime('%Y%m%d')}.parquet"


def _existing_quote_time(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        qt = pd.read_parquet(path, columns=["quote_time"])["quote_time"].iloc[0]
        return pd.Timestamp(qt).tz_convert(NY)
    except Exception:  # noqa: BLE001 - 손상된 파일이면 새 스냅샷으로 교체한다
        return None


def collect(
    ticker: str = "^SPX",
    max_days: float = 400,
    fallback: str | None = None,
    out_dir: Path | str = SNAPSHOT_DIR,
    min_two_sided: float = 0.8,
    max_stale_min: float = 45.0,
    force: bool = False,
    now: pd.Timestamp | None = None,
) -> CollectResult:
    """체인을 받아 판정하고, 유효하면(또는 ``force``) 저장한다."""
    chain = fetch_chain(ticker, min_days=0, max_days=max_days, fallback=fallback, refresh=True)
    a = assess(chain, now=now, min_two_sided=min_two_sided, max_stale_min=max_stale_min)
    name = str(chain["underlying"].iloc[0]).replace("^", "")
    path = snapshot_path(Path(out_dir), name, a.quote_time)

    print(f"{len(chain)} contracts, {chain['expiry'].nunique()} expiries, spot {chain['spot'].iloc[0]:.2f}, "
          f"quote time {a.quote_time:%Y-%m-%d %H:%M} ET ({a.stale_minutes:.0f} min old), "
          f"two-sided quotes: near-the-money {a.two_sided_atm:.0%} (n={a.n_atm}), all {a.two_sided_all:.0%}",
          flush=True)

    if not (a.valid or force):
        return CollectResult("skipped", name, path, a)
    prev = _existing_quote_time(path)
    if prev is not None and prev > a.quote_time and not force:
        return CollectResult("kept", name, path, a)
    path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(path)
    return CollectResult("saved", name, path, a)


def _warn(msg: str) -> None:
    # GitHub Actions에서는 실행 요약에 경고로 표시된다
    prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") == "true" else "WARNING: "
    print(prefix + msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="^SPX")
    ap.add_argument("--max-days", type=float, default=400)
    ap.add_argument("--fallback", default=None, help="ticker to use if the main one fails (e.g. SPY)")
    ap.add_argument("--out-dir", type=Path, default=SNAPSHOT_DIR)
    ap.add_argument("--min-two-sided", type=float, default=0.8,
                    help="minimum share of near-the-money contracts with live bid and ask")
    ap.add_argument("--max-stale-min", type=float, default=45.0, help="maximum age of the quotes in minutes")
    ap.add_argument("--force", action="store_true", help="save even if the quotes look stale")
    ap.add_argument("--require-today", action="store_true",
                    help="exit 1 if the market traded today but no snapshot for today exists")
    args = ap.parse_args(argv)

    try:
        res = collect(args.ticker, args.max_days, args.fallback, args.out_dir,
                      args.min_two_sided, args.max_stale_min, args.force)
    except DataUnavailableError as e:
        print(f"ERROR: {e}", flush=True)
        return 1

    a = res.assessment
    if res.status == "saved":
        print(f"saved -> {res.path}", flush=True)
        if not a.valid:
            _warn("saved with --force although: " + "; ".join(a.reasons))
    elif res.status == "kept":
        print(f"kept the later snapshot already in {res.path}", flush=True)
    else:
        _warn("skipped: " + "; ".join(a.reasons))

    if args.require_today:
        today = a.now.normalize()
        traded_today = a.quote_time.normalize() == today
        if traded_today and not snapshot_path(args.out_dir, res.name, today).exists():
            print("ERROR: the market traded today but no valid snapshot for today was saved", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
