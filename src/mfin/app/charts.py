"""앱 그림 (Altair, Vega-Lite). Streamlit에 의존하지 않는 순수 함수.

색은 대상(entity)에 고정한다: 섹터 XLK..XLU와 전략 5개는 계열색 1..5, 현금과 SPY는 중립색.
라이트 값은 :data:`mfin.experiments._plot.SERIES`와 같고, 다크 값은 같은 색상을 어두운 배경
(Streamlit 기본 #0e1117)에 맞춰 고른 단계다. 두 세트 모두 validate_palette.js 검사를 통과했다
(라이트 대비 WARN은 tooltip과 표로 보완).
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd

from mfin.app.data import RETURN_PARTS, STRATEGY_KO, sector_label, strategy_label
from mfin.portfolio.engine import CASH

# NAV 일별 시계열(약 1,250일 x 전략 5개)이 Altair 기본 한도 5,000행을 넘는다
alt.data_transformers.disable_max_rows()


@dataclass(frozen=True)
class Palette:
    series: tuple[str, ...]
    cash: str  # 현금: 배경 쪽으로 물러나는 중립색
    bench: str  # SPY: 진한 중립색 (6번째 색상을 새로 만들지 않는다)
    surface: str  # 차트 배경 (면 사이 간격선 색)
    ink: str  # 데이터 위에 쓰는 글자
    muted: str  # 기준선·보조선
    div: tuple[str, str, str]  # 발산: 음(빨강) - 중립 - 양(파랑)


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"),
    cash="#b4b2ac",
    bench="#6b6a66",
    surface="#ffffff",
    ink="#1f1f1d",
    muted="#8a8984",
    div=("#e34948", "#f0efec", "#2a78d6"),
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"),
    cash="#5f5e5a",
    bench="#a8a6a0",
    surface="#0e1117",
    ink="#f0efec",
    muted="#8a8984",
    div=("#e66767", "#383835", "#3987e5"),
)


def palette(dark: bool) -> Palette:
    return DARK if dark else LIGHT


def ticker_scale(tickers: list[str], dark: bool, universe: list[str] | None = None) -> alt.Scale:
    """섹터 티커 -> 색. SPY와 현금은 중립색, 나머지는 ``universe`` 안의 위치로 계열색을 정한다.

    universe(기본: tickers)를 고정해 두면 일부 자산을 빼고 그려도 남은 자산의 색이 바뀌지 않는다.
    """
    p = palette(dark)
    risky = [t for t in (universe or tickers) if t not in (CASH, "SPY")]
    domain, rng = [], []
    for t in tickers:
        domain.append(sector_label(t))
        if t == CASH:
            rng.append(p.cash)
        elif t == "SPY":
            rng.append(p.bench)
        else:
            rng.append(p.series[risky.index(t) % len(p.series)])
    return alt.Scale(domain=domain, range=rng)


def strategy_scale(dark: bool, present: list[str] | None = None) -> alt.Scale:
    """전략 -> 색. 색은 STRATEGY_KO 안의 위치로 고정하고, 범례에는 그린 전략만 남긴다."""
    p = palette(dark)
    names = list(STRATEGY_KO)
    keep = [s for s in names if present is None or s in present]
    return alt.Scale(domain=[strategy_label(s) for s in keep], range=[p.series[names.index(s) % 5] for s in keep])


def _bar_height(n: int) -> int:
    return max(120, 34 * n + 30)


# ------------------------------------------------------------------------------ 오늘의 추천


def weights_bar(wt: pd.DataFrame, dark: bool = False) -> alt.Chart:
    """목표 비중 가로 막대 (현금 포함). 막대 끝에 값을 쓴다."""
    p = palette(dark)
    order = wt["label"].tolist()
    base = alt.Chart(wt).encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X("target:Q", title="목표 비중", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("label:N", title="자산"), alt.Tooltip("target:Q", title="목표 비중", format=".2%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("label:N", scale=ticker_scale(wt["ticker"].tolist(), dark), legend=None)
    )
    text = base.mark_text(align="left", dx=6, color=p.ink).encode(text=alt.Text("target:Q", format=".1%"))
    return (bars + text).properties(height=_bar_height(len(wt)))


def risk_contrib_bar(wt: pd.DataFrame, dark: bool = False) -> alt.Chart:
    """자산별 위험기여도 (합 = 1, 현금 제외)."""
    p = palette(dark)
    df = wt[wt["ticker"] != CASH]
    base = alt.Chart(df).encode(
        y=alt.Y("label:N", sort=df["label"].tolist(), title=None),
        x=alt.X("risk_contrib:Q", title="위험기여도", axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("label:N", title="자산"), alt.Tooltip("risk_contrib:Q", title="위험기여도", format=".1%"),
                 alt.Tooltip("target:Q", title="목표 비중", format=".1%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("label:N", scale=ticker_scale(df["ticker"].tolist(), dark), legend=None)
    )
    text = base.mark_text(align="left", dx=6, color=p.ink).encode(text=alt.Text("risk_contrib:Q", format=".0%"))
    return (bars + text).properties(height=_bar_height(len(df)))


# -------------------------------------------------------------------------------- 추천 근거


def return_stack(parts: pd.DataFrame, decomp: pd.DataFrame, dark: bool = False) -> alt.Chart:
    """기대수익 3성분 누적 막대 + 보유기간 평균 기대수익 눈금."""
    p = palette(dark)
    names = list(RETURN_PARTS.values())
    scale = alt.Scale(domain=names, range=[p.cash, p.series[0], p.series[1]])
    order = decomp["label"].tolist()
    bars = alt.Chart(parts).mark_bar(width=36, stroke=p.surface, strokeWidth=2).encode(
        x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("sum(value):Q", title="기대수익률 (연)", axis=alt.Axis(format=".0%")),
        color=alt.Color("part:N", scale=scale, sort=names, title="성분", legend=alt.Legend(orient="top")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("label:N", title="자산"), alt.Tooltip("part:N", title="성분"),
                 alt.Tooltip("value:Q", title="값", format=".2%")],
    )
    tick = alt.Chart(decomp).mark_tick(thickness=3, size=46, color=p.ink).encode(
        x=alt.X("label:N", sort=order),
        y="horizon_mean:Q",
        tooltip=[alt.Tooltip("label:N", title="자산"), alt.Tooltip("mu0:Q", title="현재 μ0", format=".2%"),
                 alt.Tooltip("horizon_mean:Q", title="보유기간 평균", format=".2%")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=p.muted).encode(y="y:Q")
    return (bars + zero + tick).properties(height=320)


def vol_term_chart(term: pd.DataFrame, ewma_vol: float, horizon: float, dark: bool = False,
                   fwd: pd.DataFrame | None = None, hz_vol: float | None = None) -> alt.Chart:
    """기간 평균 내재변동성 sigma_bar(0, T) 곡선 + SPY EWMA 실현변동성 + 보유기간 표시.

    fwd를 주면 구간별 선도변동성 계단선을 옅게 겹친다. hz_vol은 보유기간 평균 내재변동성(강조 점).
    """
    p = palette(dark)
    x_max = float(term["days"].max())
    x = alt.X("days:Q", title="만기까지 기간 (일)", scale=alt.Scale(domain=[0, x_max]))
    y = alt.Y("vol:Q", title="연율 변동성", axis=alt.Axis(format=".0%", tickMinStep=0.01), scale=alt.Scale(zero=False))
    layers = []
    if fwd is not None:
        layers.append(alt.Chart(fwd).mark_line(interpolate="step-after", strokeWidth=1, color=p.muted, opacity=0.7).encode(
            x=x, y=y, tooltip=[alt.Tooltip("days:Q", title="구간 시작 (일)", format=".0f"),
                               alt.Tooltip("vol:Q", title="선도변동성", format=".1%")]))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["days"], empty=False, clear="pointerout")
    layers.append(alt.Chart(term).mark_line(strokeWidth=2, color=p.series[0]).encode(x=x, y=y))
    layers.append(alt.Chart(term).mark_point(filled=True, size=64, color=p.series[0], stroke=p.surface, strokeWidth=2).encode(
        x="days:Q", y="vol:Q",
        opacity=alt.condition(nearest, alt.value(1.0), alt.value(0.0)),
        tooltip=[alt.Tooltip("days:Q", title="만기 (일)", format=".0f"),
                 alt.Tooltip("vol:Q", title="기간 평균 내재변동성", format=".1%")],
    ).add_params(nearest))
    if pd.notna(ewma_vol):
        ref = pd.DataFrame({"vol": [ewma_vol], "days": [x_max],
                            "label": [f"SPY 실현변동성(EWMA) {ewma_vol:.1%}"]})
        layers.append(alt.Chart(ref).mark_rule(strokeDash=[6, 4], strokeWidth=2, color=p.bench).encode(
            y="vol:Q", tooltip=[alt.Tooltip("vol:Q", title="EWMA 실현변동성", format=".1%")]))
        # 왼쪽 아래에 둔다: 오른쪽 끝은 곡선이 실현변동성 근처를 지나며 겹치기 쉽다
        layers.append(alt.Chart(ref.assign(days=0.0)).mark_text(align="left", dx=6, dy=12, color=p.bench).encode(
            x="days:Q", y="vol:Q", text="label:N"))
    hz = pd.DataFrame({"days": [horizon * 365.0], "label": ["보유기간"]})
    layers.append(alt.Chart(hz).mark_rule(color=p.muted, strokeDash=[2, 3]).encode(x="days:Q"))
    layers.append(alt.Chart(hz).mark_text(align="left", dx=4, baseline="top", color=p.muted).encode(
        x="days:Q", y=alt.value(4), text="label:N"))
    if hz_vol is not None and np.isfinite(hz_vol):
        # 엔진이 이번 추천에 쓴 값: sqrt((w(h) - w(0)) / h)
        pt = pd.DataFrame({"days": [horizon * 365.0], "vol": [hz_vol], "label": [f"보유기간 평균 {hz_vol:.1%}"]})
        layers.append(alt.Chart(pt).mark_point(filled=True, size=110, color=p.series[0], stroke=p.surface,
                                               strokeWidth=2).encode(
            x="days:Q", y="vol:Q", tooltip=[alt.Tooltip("vol:Q", title="보유기간 평균 내재변동성", format=".1%")]))
        layers.append(alt.Chart(pt).mark_text(align="left", dx=9, dy=12, color=p.ink, fontWeight="bold").encode(
            x="days:Q", y="vol:Q", text="label:N"))
    return alt.layer(*layers).properties(height=300)


def corr_heatmap(cf: pd.DataFrame, tickers: list[str], dark: bool = False) -> alt.Chart:
    """Ledoit–Wolf 상관행렬. 발산 색(-1 빨강, 0 중립 회색, +1 파랑), 칸마다 값."""
    p = palette(dark)
    base = alt.Chart(cf).encode(
        x=alt.X("col:N", sort=tickers, title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("row:N", sort=tickers, title=None),
    )
    rect = base.mark_rect(stroke=p.surface, strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color("rho:Q", title="상관계수",
                        # lab 보간: 기본(hcl)은 중립색에서 파랑으로 가며 분홍을 거친다
                        scale=alt.Scale(domain=[-1, 0, 1], range=list(p.div), interpolate="lab"),
                        legend=alt.Legend(format=".1f")),
        tooltip=[alt.Tooltip("row:N", title="행"), alt.Tooltip("col:N", title="열"),
                 alt.Tooltip("rho:Q", title="ρ", format=".3f")],
    )
    # 라이트: 진한 칸(|rho| > 0.55)은 흰 글자, 옅은 칸은 검은 글자. 다크: 중립색이 어두우므로 항상 밝은 글자
    strong = "abs(datum.rho) > 0.55" if not dark else "true"
    text = base.mark_text(fontSize=13).encode(
        text=alt.Text("rho:Q", format=".2f"),
        color=alt.condition(strong, alt.value("#ffffff" if not dark else p.ink), alt.value(p.ink)),
    )
    side = 64 * len(tickers) + 20
    return (rect + text).properties(width=side, height=side)


def constraint_bars(cl: pd.DataFrame, dark: bool = False) -> alt.Chart:
    """무제약 Merton 해 vs 롱온리 제약 해 (자산별 묶음 막대, 0 기준선)."""
    p = palette(dark)
    kinds = ["무제약 Merton 해", "롱온리 제약 해(추천)"]
    order = cl["label"].drop_duplicates().tolist()
    bars = alt.Chart(cl).mark_bar(cornerRadiusEnd=4, height=14).encode(
        y=alt.Y("label:N", sort=order, title=None),
        yOffset=alt.YOffset("kind:N", sort=kinds),
        x=alt.X("weight:Q", title="비중", axis=alt.Axis(format=".0%")),
        color=alt.Color("kind:N", scale=alt.Scale(domain=kinds, range=[p.cash, p.series[0]]), sort=kinds,
                        title=None, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title="자산"), alt.Tooltip("kind:N", title="해"),
                 alt.Tooltip("weight:Q", title="비중", format=".1%")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=p.muted).encode(x="x:Q")
    return (bars + zero).properties(height=_bar_height(len(order)) + 30)


# ------------------------------------------------------------------------------ 백테스트

# Vega 기본 시간 라벨은 영어 월 이름("April")이라 숫자 형식으로 고정한다
DATE_AXIS = alt.Axis(format="%Y-%m", labelOverlap=True)


def _time_lines(long: pd.DataFrame, value: str, y_title: str, fmt: str, dark: bool, *,
                log: bool = False, step: bool = False, points: bool = False, height: int = 340) -> alt.Chart:
    """전략별 시계열 선 + 가장 가까운 날짜의 십자선과 tooltip."""
    p = palette(dark)
    keys = [s for s in STRATEGY_KO if s in set(long["strategy"])]
    wide = long.pivot_table(index="date", columns="strategy", values=value).reset_index()
    color = alt.Color("label:N", scale=strategy_scale(dark, keys), title=None, legend=alt.Legend(orient="top"))
    # nice=False: 로그축이 10의 거듭제곱(0.1~10)까지 늘어나 선이 눌리는 것을 막는다
    yscale = alt.Scale(type="log", nice=False, padding=8) if log else alt.Scale(zero=False)
    # % 축은 1%p보다 촘촘한 눈금을 막는다 (".0%" 서식에서 같은 라벨이 반복되지 않게)
    axis = alt.Axis(format=fmt, tickMinStep=0.01) if fmt.endswith("%") else alt.Axis(format=fmt)
    base = alt.Chart(long).encode(
        x=alt.X("date:T", title=None, axis=DATE_AXIS),
        y=alt.Y(f"{value}:Q", title=y_title, axis=axis, scale=yscale),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2, interpolate="step-after" if step else "linear")
    # 가장 가까운 날짜를 고르는 선택. 세로선 하나가 그날의 전략별 값을 모두 tooltip으로 보여 준다
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False, clear="pointerout")
    tips = [alt.Tooltip("date:T", title="날짜", format="%Y-%m-%d")] + [
        alt.Tooltip(f"{k}:Q", title=strategy_label(k), format=fmt) for k in keys
    ]
    rules = alt.Chart(wide).mark_rule(color=p.muted, strokeWidth=1.5).encode(
        x="date:T", opacity=alt.condition(nearest, alt.value(0.9), alt.value(0.0)), tooltip=tips,
    ).add_params(nearest)
    dots = base.mark_point(filled=True, size=64, stroke=p.surface, strokeWidth=2).encode(
        opacity=alt.condition(nearest, alt.value(1.0), alt.value(0.0))
    )
    layers = [lines, rules, dots]
    if points:
        layers.insert(1, base.mark_point(filled=True, size=36, opacity=0.9))
    return alt.layer(*layers).properties(height=height)


def nav_chart(nl: pd.DataFrame, log: bool = True, dark: bool = False) -> alt.Chart:
    return _time_lines(nl, "growth", "1달러의 성장 (비용 차감 후)", ".2f", dark, log=log)


def drawdown_chart(dl: pd.DataFrame, dark: bool = False) -> alt.Chart:
    return _time_lines(dl, "drawdown", "고점 대비 낙폭", ".0%", dark, height=260)


def diag_chart(dg: pd.DataFrame, y_title: str, fmt: str = ".0%", dark: bool = False) -> alt.Chart:
    """리밸런싱 시점별 진단값 (계단선 + 월별 점)."""
    return _time_lines(dg, "value", y_title, fmt, dark, step=True, points=True, height=260)


def weights_area(wl: pd.DataFrame, dark: bool = False, universe: list[str] | None = None) -> alt.Chart:
    """리밸런싱 시점별 비중 누적 면적 (계단형). 현금은 맨 위의 중립색.

    universe: 원래 비중 표의 열 전체. 전 기간 0이라 빠진 자산이 있어도 색이 밀리지 않게 한다.
    """
    p = palette(dark)
    tickers = wl.sort_values("order")["ticker"].drop_duplicates().tolist()
    labels = [sector_label(t) for t in tickers]
    return alt.Chart(wl).mark_area(interpolate="step-after", stroke=p.surface, strokeWidth=1.5).encode(
        x=alt.X("date:T", title=None, axis=DATE_AXIS),
        y=alt.Y("weight:Q", stack="zero", title="비중", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("label:N", scale=ticker_scale(tickers, dark, universe), sort=labels, title=None,
                        legend=alt.Legend(orient="top")),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("date:T", title="리밸런싱일", format="%Y-%m-%d"), alt.Tooltip("label:N", title="자산"),
                 alt.Tooltip("weight:Q", title="비중", format=".1%")],
    ).properties(height=320)
