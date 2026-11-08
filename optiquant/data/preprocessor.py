"""
Quantitative Preprocessor: Return Modeling, EWMA Volatility, and Covariance Conditioning.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass
class ProcessedMarketData:
    prices: pd.DataFrame
    log_returns: pd.DataFrame
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    corr_matrix: pd.DataFrame
    condition_number: float
    market_return_series: pd.Series
    market_ewma_vol_series: pd.Series


class MarketDataPreprocessor:
    """
    Transforms raw price series into rigorously conditioned quantitative matrices.
    """

    def __init__(self, trading_days_per_year: int = 252, ewma_span: int = 20):
        self.annual_factor = trading_days_per_year
        self.ewma_span = ewma_span

    def process(self, prices: pd.DataFrame) -> ProcessedMarketData:
        """
        Computes continuous returns, condition numbers, and regularized covariance.
        """
        if prices.empty or len(prices.columns) < 2:
            raise ValueError("Price matrix must contain at least 2 assets and valid price history.")

        # Continuous compounding returns
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Annualized expected return vector
        mean_returns = log_returns.mean() * self.annual_factor

        # Annualized sample covariance matrix
        cov_matrix = log_returns.cov() * self.annual_factor

        # Condition number calculation: kappa = lambda_max / lambda_min
        eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
        min_eig = max(eigenvalues[0], 1e-12)
        condition_number = float(eigenvalues[-1] / min_eig)

        # Regularize covariance if ill-conditioned (ridge shrinkage)
        if min_eig < 1e-6 or condition_number > 1e6:
            cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 1e-4

        # Correlation matrix
        corr_matrix = log_returns.corr()

        # Market-wide benchmark proxy (equal-weighted cross-sectional return)
        market_return_series = log_returns.mean(axis=1)

        # Exponentially Weighted Moving Average (EWMA) volatility
        market_ewma_vol_series = market_return_series.ewm(span=self.ewma_span).std() * np.sqrt(self.annual_factor)
        market_ewma_vol_series = market_ewma_vol_series.bfill().fillna(0.15)

        return ProcessedMarketData(
            prices=prices,
            log_returns=log_returns,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            corr_matrix=corr_matrix,
            condition_number=condition_number,
            market_return_series=market_return_series,
            market_ewma_vol_series=market_ewma_vol_series
        )
