"""
Unit tests for GPU/CPU PyTorch Tensor Engine.
"""

import unittest
import numpy as np
import torch
from optiquant.systems.gpu_engine import GPUPortfolioEngine


class TestGPUEngine(unittest.TestCase):
    def test_tensor_simulation_correctness(self):
        mu = np.array([0.10, 0.15, 0.20], dtype=np.float32)
        cov = np.diag([0.04, 0.09, 0.16]).astype(np.float32)

        # Force CPU device to test mathematical correctness on any environment
        engine = GPUPortfolioEngine(device="cpu")
        res = engine.simulate_allocations(mu, cov, num_simulations=5000, risk_free_rate=0.03)

        self.assertEqual(res["num_simulations"], 5000)
        self.assertIn("max_sharpe", res)
        self.assertIn("min_volatility", res)

        max_s = res["max_sharpe"]
        min_v = res["min_volatility"]

        # Sharpe ratio of optimal should be strictly positive
        self.assertGreater(max_s["sharpe"], 0.0)
        # Minimum volatility should be less than or equal to max sharpe's volatility
        self.assertLessEqual(min_v["volatility"], max_s["volatility"] + 1e-4)

        # Weights sum to 1
        self.assertAlmostEqual(float(np.sum(max_s["weights"])), 1.0, places=3)
        self.assertAlmostEqual(float(np.sum(min_v["weights"])), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
