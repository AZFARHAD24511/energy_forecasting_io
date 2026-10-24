from __future__ import annotations

import numpy as np
import pandas as pd


class LeontiefEconomicModel:
    """
    IO quantity model plus a separate simplified fixed-coefficient cost scenario.

    Convention: A[i, j] is input from sector i per unit output of sector j.
    Therefore the quantity model is x = (I - A)^(-1) y.
    The cost scenario is illustrative; it is not a causal or calibrated price forecast.
    """

    def __init__(self, A: np.ndarray, sector_names: list[str]):
        self.A = np.asarray(A, dtype=float)
        self.sector_names = list(sector_names)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square matrix.")
        if self.A.shape[0] != len(self.sector_names):
            raise ValueError("Number of sector names must match A dimensions.")
        if not np.isfinite(self.A).all() or (self.A < 0).any():
            raise ValueError("A must contain finite, non-negative coefficients.")
        self.n = self.A.shape[0]
        self.I = np.eye(self.n)
        self.leontief_inverse = np.linalg.inv(self.I - self.A)

    def output_effect(self, final_demand_shock: np.ndarray) -> pd.DataFrame:
        """Calculate gross-output changes for a final-demand shock, Δx=(I-A)^(-1)Δy."""
        shock = np.asarray(final_demand_shock, dtype=float)
        if shock.shape != (self.n,):
            raise ValueError(f"final_demand_shock must have shape ({self.n},).")
        dx = self.leontief_inverse @ shock
        return pd.DataFrame({"sector": self.sector_names, "final_demand_shock": shock, "gross_output_change": dx})

    def electricity_cost_scenario(
        self, electricity_sector: str, electricity_price_change_pct: float
    ) -> pd.DataFrame:
        """
        Approximate price-cost propagation with fixed input coefficients.

        First-round cost contribution for each producing sector is its electricity
        input coefficient times the electricity-price change. This contribution is
        propagated through (I-A.T)^(-1). The output is a stylized percentage-point
        scenario, not a fully calibrated IO price model. The input matrix must use
        the documented row-input/column-output convention.
        """
        if electricity_sector not in self.sector_names:
            raise ValueError(f"Unknown electricity sector: {electricity_sector}")
        if not np.isfinite(electricity_price_change_pct):
            raise ValueError("Price change must be finite.")
        elec_idx = self.sector_names.index(electricity_sector)
        # Electricity input coefficients across consuming sectors.
        direct_cost = self.A[elec_idx, :] * float(electricity_price_change_pct)
        propagated = np.linalg.solve(self.I - self.A.T, direct_cost)
        return pd.DataFrame({
            "sector": self.sector_names,
            "direct_cost_change_pct_points": direct_cost * 100,
            "propagated_cost_change_pct_points": propagated * 100,
        })
