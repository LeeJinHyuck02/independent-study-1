"""Rolling-window 백테스트 (모듈 D): 전략, 시뮬레이션 루프, 성과 지표.

하위 모듈은 처음 쓸 때 불러온다(PEP 562). 패키지를 import할 때 rolling을 미리 불러오면
`python -m mfin.backtest.rolling` 실행 시 runpy가 경고를 내기 때문이다.
"""

from importlib import import_module

_EXPORTS = {
    "BacktestConfig": "rolling",
    "BacktestData": "rolling",
    "ProportionalCost": "rolling",
    "RebalanceContext": "strategies",
    "StrategyResult": "rolling",
    "build_strategy": "strategies",
    "drawdown": "metrics",
    "load_data": "rolling",
    "performance_table": "metrics",
    "rebalance_dates": "rolling",
    "run_backtest": "rolling",
    "run_strategy": "rolling",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
