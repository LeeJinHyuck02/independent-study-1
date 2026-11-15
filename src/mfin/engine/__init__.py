"""기준해 계산기 (해석해 / 수치해).

모듈 A: 블랙-숄즈와 내재변동성. 모듈 B1: Merton HJB (해석해, 제약 QP).
모듈 B2: 다자산 Almgren–Chriss (행렬 Riccati, n = 1, 2 격자 FDM 교차검증).
"""

from mfin.engine.bs_analytic import (
    black_call,
    black_price,
    black_put,
    black_vega_s,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
)
from mfin.engine.fdm_hjb import FDMConfig, FDMResult, richardson, solve_ac_fdm
from mfin.engine.implied_vol import fit_constant_vol, implied_vol, implied_vols, smile
from mfin.engine.merton import (
    MertonCoeffs,
    QPResult,
    kkt_residuals,
    merton_policy,
    merton_policy_period,
    merton_unconstrained,
    merton_value,
    period_inputs,
    solve_mv_qp,
)
from mfin.engine.riccati_ac import (
    ACProblem,
    RiccatiSolution,
    optimal_cost,
    rate_closed_form,
    riccati_closed_form,
    riccati_inverse_closed_form,
    solve_riccati,
    trajectory_closed_form,
)
from mfin.engine.vol_term import (
    DEFAULT_SENSORS,
    forward_variance_knots,
    forward_vol_curve,
    term_structure,
    vix_forward_vol_curve,
)

__all__ = [
    "ACProblem",
    "DEFAULT_SENSORS",
    "FDMConfig",
    "FDMResult",
    "MertonCoeffs",
    "QPResult",
    "RiccatiSolution",
    "black_call",
    "black_price",
    "black_put",
    "black_vega_s",
    "bs_delta",
    "bs_gamma",
    "bs_price",
    "bs_vega",
    "fit_constant_vol",
    "forward_variance_knots",
    "forward_vol_curve",
    "implied_vol",
    "implied_vols",
    "kkt_residuals",
    "merton_policy",
    "merton_policy_period",
    "merton_unconstrained",
    "merton_value",
    "optimal_cost",
    "period_inputs",
    "rate_closed_form",
    "richardson",
    "riccati_closed_form",
    "riccati_inverse_closed_form",
    "smile",
    "solve_ac_fdm",
    "solve_mv_qp",
    "solve_riccati",
    "term_structure",
    "trajectory_closed_form",
    "vix_forward_vol_curve",
]
