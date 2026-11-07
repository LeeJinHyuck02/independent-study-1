"""mfin: Black-Scholes / HJB based portfolio recommendation system.

Subpackages
-----------
data     : market data loaders (prices, risk-free rate, VIX term structure, option chains) with parquet cache
engine   : exact/numerical reference solvers (Black-Scholes closed form, Brent implied vol, vol term structure)
pinn     : physics-informed neural network core and the inverse Black-Scholes PINN
experiments : runnable scripts that produce tables/figures under results/
"""

__version__ = "0.1.0"
