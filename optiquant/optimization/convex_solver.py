"""
Analytical Convex Quadratic Optimizer for Exact Continuous Markowitz Pareto Frontiers.
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.optimize as sco


class AnalyticalConvexOptimizer:
    """
    Computes exact Markowitz optimal portfolios and continuous Pareto frontiers
    via Sequential Least Squares Programming (SLSQP).
    """

    def __init__(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.045
    ):
        self.mean_returns = np.asarray(mean_returns, dtype=float)
        self.cov_matrix = np.asarray(cov_matrix, dtype=float)
        self.rf = float(risk_free_rate)
        self.num_assets = len(self.mean_returns)

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """Calculates portfolio expected annual return, volatility, and Sharpe ratio."""
        w = np.asarray(weights, dtype=float)
        ret = float(np.sum(self.mean_returns * w))
        var = float(np.dot(w.T, np.dot(self.cov_matrix, w)))
        vol = float(np.sqrt(max(var, 1e-8)))
        sharpe = float((ret - self.rf) / vol)
        return ret, vol, sharpe

    def max_sharpe(self, max_weight: float = 0.50) -> Dict[str, Any]:
        """
        Maximizes the Sharpe ratio subject to 0 <= w_i <= max_weight and sum(w_i) = 1.
        """
        def neg_sharpe(w):
            return -self.portfolio_performance(w)[2]

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.0, max_weight) for _ in range(self.num_assets))
        init_guess = np.ones(self.num_assets) / self.num_assets

        res = sco.minimize(
            neg_sharpe,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000}
        )
        weights = res.x if res.success else init_guess
        weights = np.clip(weights, 0.0, 1.0)
        weights = weights / np.sum(weights)

        r, v, s = self.portfolio_performance(weights)
        return {"weights": weights, "return": r, "volatility": v, "sharpe": s}

    def minimum_variance(self) -> Dict[str, Any]:
        """
        Finds the global minimum variance portfolio.
        """
        def port_vol(w):
            return self.portfolio_performance(w)[1]

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        init_guess = np.ones(self.num_assets) / self.num_assets

        res = sco.minimize(
            port_vol,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000}
        )
        weights = res.x if res.success else init_guess
        weights = np.clip(weights, 0.0, 1.0)
        weights = weights / np.sum(weights)

        r, v, s = self.portfolio_performance(weights)
        return {"weights": weights, "return": r, "volatility": v, "sharpe": s}

    def compute_efficient_frontier(self, points: int = 50) -> Dict[str, np.ndarray]:
        """
        Calculates the exact Pareto frontier by minimizing variance across a target return grid.
        """
        min_v = self.minimum_variance()
        max_s = self.max_sharpe(max_weight=1.0)

        r_min = min_v["return"]
        r_max = max(max_s["return"] * 1.25, r_min + 0.05)
        target_returns = np.linspace(r_min, r_max, points)

        frontier_vols = []
        frontier_weights = []
        valid_returns = []

        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        guess = min_v["weights"]

        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, t=target: np.sum(self.mean_returns * w) - t}
            ]
            res = sco.minimize(
                lambda w: np.dot(w.T, np.dot(self.cov_matrix, w)),
                guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-8, "maxiter": 500}
            )
            if res.success:
                frontier_vols.append(float(np.sqrt(max(res.fun, 1e-8))))
                frontier_weights.append(res.x)
                valid_returns.append(target)
                guess = res.x  # Warm start for next target

        return {
            "returns": np.array(valid_returns),
            "volatilities": np.array(frontier_vols),
            "weights": np.array(frontier_weights)
        }
