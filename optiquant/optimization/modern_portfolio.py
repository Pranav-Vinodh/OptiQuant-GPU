"""
Modern Portfolio Theory Risk Analytics and Performance Evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd


@dataclass
class PortfolioMetrics:
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    cumulative_returns: pd.Series
    drawdown_series: pd.Series


class PortfolioRiskAnalytics:
    """
    Computes institutional risk and performance metrics for allocated portfolios.
    """

    def __init__(self, risk_free_rate: float = 0.045, trading_days: int = 252):
        self.rf = risk_free_rate
        self.annual_factor = trading_days

    def evaluate(self, weights: np.ndarray, log_returns: pd.DataFrame) -> PortfolioMetrics:
        """
        Evaluates historical portfolio trajectory given asset weights and log-returns.
        """
        w = np.asarray(weights, dtype=float)
        # Daily portfolio returns
        daily_returns = log_returns.dot(w)

        # Annualized return & volatility
        ann_return = float(daily_returns.mean() * self.annual_factor)
        ann_vol = float(daily_returns.std() * np.sqrt(self.annual_factor))

        # Sharpe ratio
        sharpe = (ann_return - self.rf) / max(ann_vol, 1e-6)

        # Downside deviation & Sortino ratio
        rf_daily = self.rf / self.annual_factor
        downside = daily_returns[daily_returns < rf_daily] - rf_daily
        downside_std = float(np.sqrt(np.mean(downside**2))) * np.sqrt(self.annual_factor) if len(downside) > 0 else 1e-6
        sortino = (ann_return - self.rf) / max(downside_std, 1e-6)

        # Cumulative wealth & Drawdown series
        cum_returns = np.exp(daily_returns.cumsum())
        peak = cum_returns.cummax()
        drawdown_series = (cum_returns - peak) / peak
        max_dd = float(drawdown_series.min())

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

        # Historical Value at Risk (VaR) & Conditional VaR (CVaR)
        var_95 = float(np.percentile(daily_returns, 5) * -np.sqrt(self.annual_factor))
        var_99 = float(np.percentile(daily_returns, 1) * -np.sqrt(self.annual_factor))

        tail_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)]
        cvar_95 = float(-tail_95.mean() * np.sqrt(self.annual_factor)) if len(tail_95) > 0 else var_95

        tail_99 = daily_returns[daily_returns <= np.percentile(daily_returns, 1)]
        cvar_99 = float(-tail_99.mean() * np.sqrt(self.annual_factor)) if len(tail_99) > 0 else var_99

        return PortfolioMetrics(
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            cumulative_returns=cum_returns,
            drawdown_series=drawdown_series
        )
