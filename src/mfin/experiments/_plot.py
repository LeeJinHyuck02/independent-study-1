"""보고서 그림에 공통으로 쓰는 matplotlib 스타일.

색: 검증된 기준 팔레트의 처음 다섯 색(파랑, 주황, 청록, 노랑, 자홍)을 이 순서로 쓴다.
처음 세 색은 모든 쌍이, 다섯 색은 인접 쌍이 색각 이상 검사를 통과한다(선·막대용).
청록·노랑·자홍은 밝은 배경에서 명암비가 3:1에 못 미치므로, 그림마다 CSV 표를 함께
저장하고 계열도 마커 모양으로 구분한다. 산점도처럼 모든 쌍이 맞닿는 그림은 세 계열까지만 쓴다.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
CASH = "#b4b2ac"  # 현금 등 '나머지' 영역용 중립색 (계열색과 겹치지 않는다)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "font.size": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.labelcolor": TEXT,
        }
    )


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
