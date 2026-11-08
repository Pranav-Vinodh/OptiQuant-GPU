"""
Unit tests for convex quadratic optimizer and Hierarchical Risk Parity.
"""

import unittest
import numpy as np
import pandas as pd
from optiquant.optimization.convex_solver import AnalyticalConvexOptimizer
from optiquant.models.hrp import HierarchicalRiskParity


class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.num_assets = 4
        self.mu = np.array([0.12, 0.18, 0.15, 0.09])
        # Well-conditioned covariance
        self.cov = np.array([
            [0.04, 0.01, 0.005, 0.002],
            [0.01, 0.09, 0.02, 0.01],
            [0.005, 0.02, 0.06, 0.015],
            [0.002, 0.01, 0.015, 0.03]
        ])
        std = np.sqrt(np.diag(self.cov))
        self.corr = self.cov / np.outer(std, std)
        self.asset_names = ["Asset1", "Asset2", "Asset3", "Asset4"]
        self.df_cov = pd.DataFrame(self.cov, index=self.asset_names, columns=self.asset_names)
        self.df_corr = pd.DataFrame(self.corr, index=self.asset_names, columns=self.asset_names)

    def test_convex_max_sharpe(self):
        solver = AnalyticalConvexOptimizer(self.mu, self.cov, risk_free_rate=0.04)
        res = solver.max_sharpe(max_weight=1.0)
        weights = res["weights"]

        # Weights sum to 1.0 and non-negative
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=4)
        self.assertTrue((weights >= -1e-6).all())
        self.assertGreater(res["sharpe"], 0.0)

    def test_convex_minimum_variance(self):
        solver = AnalyticalConvexOptimizer(self.mu, self.cov, risk_free_rate=0.04)
        min_v = solver.minimum_variance()
        weights = min_v["weights"]

        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=4)
        self.assertTrue((weights >= -1e-6).all())

        # An arbitrary portfolio should have >= volatility than minimum variance
        arbitrary_w = np.array([0.25, 0.25, 0.25, 0.25])
        _, arb_v, _ = solver.portfolio_performance(arbitrary_w)
        self.assertLessEqual(min_v["volatility"], arb_v + 1e-5)

    def test_hrp_solver(self):
        hrp = HierarchicalRiskParity(self.df_cov, self.df_corr)
        weights = hrp.solve()

        self.assertEqual(len(weights), self.num_assets)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=4)
        self.assertTrue((weights.values >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
