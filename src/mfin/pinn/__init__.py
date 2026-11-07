"""Physics-informed neural networks (research path, compared against mfin.engine)."""

from mfin.pinn.bs_inverse import (
    InverseBSConfig,
    InverseBSPINN,
    InverseBSResult,
    price_error_vs_black,
    synthetic_quotes,
)
from mfin.pinn.core import DTYPE, MLP, TrainConfig, grad, set_seed, train

__all__ = [
    "DTYPE",
    "InverseBSConfig",
    "InverseBSPINN",
    "InverseBSResult",
    "MLP",
    "TrainConfig",
    "grad",
    "price_error_vs_black",
    "set_seed",
    "synthetic_quotes",
    "train",
]
