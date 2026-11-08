"""
Unit tests for 3-state Gaussian HMM regime detector.
"""

import unittest
import numpy as np
import pandas as pd
from optiquant.models.regime_detector import MarketRegimeDetector


class TestRegimeDetector(unittest.TestCase):
    def test_regime_detector_convergence_and_structure(self):
        rng = np.random.default_rng(100)
        # Create synthetic regime transitions: 100 days low-vol bull, 100 days high-vol bear
        bull_ret = rng.normal(0.001, 0.008, 120)
        bear_ret = rng.normal(-0.002, 0.030, 120)
        returns = pd.Series(np.concatenate([bull_ret, bear_ret]))

        detector = MarketRegimeDetector(n_regimes=3, random_state=42)
        states, summary = detector.fit_predict(returns)

        self.assertEqual(len(states), len(returns))
        self.assertEqual(summary.transition_matrix.shape, (3, 3))
        # Probabilities sum to 1.0 along rows
        row_sums = np.sum(summary.transition_matrix, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-5)

        # Suggested risk multiplier should be positive
        self.assertGreater(summary.suggested_risk_multiplier, 0.0)
        self.assertIn(summary.current_regime, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
