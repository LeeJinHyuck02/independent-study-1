"""옵션체인 스냅샷 수집기의 저장 판정 (네트워크 없이 가짜 체인으로)."""

import numpy as np
import pandas as pd
import pytest

import mfin.data.snapshot_collector as sc
from mfin.data.option_chain import NY

S = 6000.0
OPEN_DAY = pd.Timestamp("2026-11-20 14:00", tz=NY)  # 금요일 장중


def fake_chain(quote_time, atm_quoted=1.0, far_quoted=0.0):
    """ATM 근방(|K/S-1| < 10%, 7-180일) 계약 중 ``atm_quoted`` 비율에만 호가가 있는 체인."""
    rows = []
    for days in (0.5, 30.0, 90.0, 300.0):
        for K in np.linspace(0.5 * S, 1.5 * S, 41):
            rows.append({"K": K, "T": days / 365.0, "expiry": quote_time + pd.Timedelta(days=days)})
    c = pd.DataFrame(rows)
    near = (np.abs(c["K"] / S - 1) < 0.10) & (c["T"] * 365).between(7, 180)
    rank = near.cumsum().where(near, 0)
    quoted = np.where(near, rank <= round(atm_quoted * near.sum()), np.arange(len(c)) % 10 < far_quoted * 10)
    c["bid"] = np.where(quoted, 1.0, 0.0)
    c["ask"] = np.where(quoted, 1.2, 0.0)
    c["underlying"], c["spot"], c["quote_time"] = "^SPX", S, quote_time
    return c


@pytest.fixture
def market(monkeypatch):
    """fetch_chain과 현재 시각을 바꿔 끼운다. state['chain'], state['now']를 테스트가 정한다."""
    state = {"now": OPEN_DAY, "chain": fake_chain(OPEN_DAY - pd.Timedelta(minutes=2))}
    monkeypatch.setattr(sc, "fetch_chain", lambda *a, **k: state["chain"])
    monkeypatch.setattr(sc, "_now", lambda: state["now"])
    return state


def test_assess_live_quotes_valid_even_if_far_strikes_unquoted():
    a = sc.assess(fake_chain(OPEN_DAY, atm_quoted=0.9, far_quoted=0.0), now=OPEN_DAY + pd.Timedelta(minutes=5))
    assert a.valid and a.two_sided_atm == pytest.approx(0.9, abs=0.05) and a.two_sided_all < 0.5


def test_assess_rejects_thin_quotes_and_stale_quotes():
    thin = sc.assess(fake_chain(OPEN_DAY, atm_quoted=0.5), now=OPEN_DAY)
    assert not thin.valid and "two-sided" in thin.reasons[0]
    stale = sc.assess(fake_chain(OPEN_DAY), now=OPEN_DAY + pd.Timedelta(hours=3))
    assert not stale.valid and "min old" in stale.reasons[0]


def test_skip_does_not_write_unless_forced(market, tmp_path):
    market["chain"] = fake_chain(OPEN_DAY, atm_quoted=0.2)
    res = sc.collect(out_dir=tmp_path)
    assert res.status == "skipped" and not res.path.exists()
    res = sc.collect(out_dir=tmp_path, force=True)
    assert res.status == "saved" and res.path.exists() and res.path.parent == tmp_path
    assert res.path.name == "SPX_20261120.parquet"


def test_later_valid_snapshot_replaces_earlier(market, tmp_path):
    early, late = OPEN_DAY - pd.Timedelta(hours=2), OPEN_DAY
    market["chain"], market["now"] = fake_chain(late), late + pd.Timedelta(minutes=1)
    assert sc.collect(out_dir=tmp_path).status == "saved"
    # 시세 시각이 더 이른 유효 스냅샷은 기존 파일을 덮지 않는다
    market["chain"], market["now"] = fake_chain(early), early + pd.Timedelta(minutes=1)
    res = sc.collect(out_dir=tmp_path)
    assert res.status == "kept"
    assert pd.read_parquet(res.path)["quote_time"].iloc[0] == late
    # 더 늦은 스냅샷은 교체한다
    later = late + pd.Timedelta(hours=1)
    market["chain"], market["now"] = fake_chain(later), later + pd.Timedelta(minutes=1)
    assert sc.collect(out_dir=tmp_path).status == "saved"
    assert pd.read_parquet(res.path)["quote_time"].iloc[0] == later


def test_require_today_fails_only_when_market_traded_today(market, tmp_path):
    args = ["--out-dir", str(tmp_path), "--require-today"]
    # 오늘 장이 열렸는데(시세 날짜 = 오늘) 호가가 오래돼 저장하지 못함 -> 실패
    market["chain"] = fake_chain(OPEN_DAY.replace(hour=13))
    market["now"] = OPEN_DAY.replace(hour=15)
    assert sc.main(args) == 1
    # 같은 날 앞선 회차가 저장해 두었으면 성공
    market["chain"], market["now"] = fake_chain(OPEN_DAY), OPEN_DAY + pd.Timedelta(minutes=3)
    assert sc.main(["--out-dir", str(tmp_path)]) == 0
    market["chain"] = fake_chain(OPEN_DAY.replace(hour=13))
    market["now"] = OPEN_DAY.replace(hour=15)
    assert sc.main(args) == 0


def test_require_today_passes_on_holiday(market, tmp_path):
    # 추수감사절 11/26: 시세 시각이 전 거래일 마감이므로 실패로 치지 않는다
    market["chain"] = fake_chain(pd.Timestamp("2026-11-25 16:15", tz=NY))
    market["now"] = pd.Timestamp("2026-11-26 14:13", tz=NY)
    assert sc.main(["--out-dir", str(tmp_path), "--require-today"]) == 0
    assert not any(tmp_path.iterdir())


def test_download_failure_exits_1(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise sc.DataUnavailableError("blocked")

    monkeypatch.setattr(sc, "fetch_chain", boom)
    assert sc.main(["--out-dir", str(tmp_path)]) == 1
