"""추천 엔진 (모듈 C): 상태 추정, 비중 계산, 위험 리포트, 체결 계획.

하위 모듈은 처음 쓸 때 불러온다(PEP 562). `python -m mfin.portfolio.engine` 실행 시 runpy
경고를 피하기 위해서다.
"""

from importlib import import_module

_EXPORTS = {
    "CASH": "engine",
    "EstimatorConfig": "estimators",
    "ExecutionConfig": "execution",
    "ExecutionSchedule": "execution",
    "MarketState": "estimators",
    "Recommendation": "engine",
    "RiskReport": "risk",
    "chain_knots": "estimators",
    "estimate_state": "estimators",
    "ledoit_wolf_corr": "estimators",
    "live_state": "engine",
    "order_from_weights": "execution",
    "plan_execution": "execution",
    "recommend": "engine",
    "risk_report": "risk",
    "vix_knots": "estimators",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
