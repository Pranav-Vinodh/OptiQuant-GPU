"""
Unit tests for data fetcher and preprocessor.
"""

import unittest
import numpy as np
import pandas as pd
from optiquant.data.fetcher import MarketDataFetcher
from optiquant.data.preprocessor import MarketDataPreprocessor


class TestDataModules(unittest.TestCase):
    def test_nse_ticker_normalization(self):
        tickers = ["tcs", "RELIANCE.NS", "infy", "^NSEI"]
        normalized = MarketDataFetcher.normalize_nse_tickers(tickers)
        self.assertEqual(normalized, ["TCS.NS", "RELIANCE.NS", "INFY.NS", "^NSEI"])

    def test_synthetic_prices_generation(self):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        prices = MarketDataFetcher.generate_synthetic_prices(tickers, "2023-01-01", "2024-01-01", seed=42)
        self.assertEqual(list(prices.columns), tickers)
        self.assertGreater(len(prices), 200)
        self.assertTrue((prices > 0).all().all())

    def test_preprocessor_log_returns_and_conditioning(self):
        tickers = ["A", "B", "C"]
        prices = MarketDataFetcher.generate_synthetic_prices(tickers, "2023-01-01", "2024-01-01", seed=10)
        preprocessor = MarketDataPreprocessor()
        data = preprocessor.process(prices)

        self.assertEqual(len(data.mean_returns), 3)
        self.assertEqual(data.cov_matrix.shape, (3, 3))
        self.assertGreater(data.condition_number, 0)
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(data.cov_matrix.values)
        self.assertTrue((eigenvalues >= 0).all())


if __name__ == "__main__":
    unittest.main()
