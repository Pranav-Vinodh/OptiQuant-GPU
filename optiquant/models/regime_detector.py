"""
Unsupervised Macro Market Regime Detection using 3-State Gaussian Hidden Markov Models.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm


@dataclass
class MarketRegimeState:
    current_regime: int
    current_regime_name: str
    regime_names: Dict[int, str]
    transition_matrix: np.ndarray
    regime_probabilities: np.ndarray
    suggested_risk_multiplier: float
    state_sequence: np.ndarray
    posterior_history: np.ndarray


class MarketRegimeDetector:
    """
    Fits a 3-State Gaussian Hidden Markov Model (HMM) on return-volatility phase space
    to infer latent macroeconomic regimes without supervised labels.
    """

    def __init__(self, n_regimes: int = 3, covariance_type: str = "diag", random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=1000,
            min_covar=1e-4,
            random_state=random_state
        )
        self.regime_map: Dict[int, str] = {}
        self.risk_multipliers: Dict[int, float] = {}

    def fit_predict(
        self,
        returns_series: pd.Series,
        vol_series: pd.Series = None
    ) -> Tuple[np.ndarray, MarketRegimeState]:
        """
        Trains HMM on (daily return, rolling volatility) features and extracts regime states.
        """
        clean_returns = returns_series.fillna(0.0).values
        if vol_series is None:
            rolling_vol = returns_series.rolling(window=20).std().fillna(returns_series.std()).values
        else:
            rolling_vol = vol_series.fillna(vol_series.mean()).values

        features = np.column_stack([clean_returns, rolling_vol])

        # Fit Gaussian HMM using Baum-Welch (EM)
        self.model.fit(features)
        hidden_states = self.model.predict(features)
        posteriors = self.model.predict_proba(features)

        # Sort regimes by Sharpe-like metric: mean return / volatility
        means = [self.model.means_[i][0] for i in range(self.n_regimes)]
        if self.model.covars_.ndim == 2:
            vols = [np.sqrt(np.maximum(self.model.covars_[i][1], 1e-8)) for i in range(self.n_regimes)]
        else:
            vols = [np.sqrt(np.maximum(self.model.covars_[i][1, 1], 1e-8)) for i in range(self.n_regimes)]
        ratios = [m / (v + 1e-6) for m, v in zip(means, vols)]

        sorted_indices = np.argsort(ratios)
        # 0: Bear (High Volatility, lowest Sharpe ratio)
        # 1: Neutral (Sideways, intermediate Sharpe ratio)
        # 2: Bull (Low Volatility, highest Sharpe ratio)
        self.regime_map = {
            int(sorted_indices[0]): "Bear (High Volatility)",
            int(sorted_indices[1]): "Neutral (Sideways)",
            int(sorted_indices[2]): "Bull (Low Volatility)"
        }

        self.risk_multipliers = {
            int(sorted_indices[0]): 0.50,  # De-risk and shift weight to low-vol assets
            int(sorted_indices[1]): 1.00,  # Normal risk budget
            int(sorted_indices[2]): 1.25   # Increase Sharpe factor
        }

        current_state = int(hidden_states[-1])
        current_name = self.regime_map.get(current_state, "Neutral (Sideways)")
        multiplier = self.risk_multipliers.get(current_state, 1.0)

        summary = MarketRegimeState(
            current_regime=current_state,
            current_regime_name=current_name,
            regime_names=self.regime_map,
            transition_matrix=self.model.transmat_,
            regime_probabilities=posteriors[-1],
            suggested_risk_multiplier=multiplier,
            state_sequence=hidden_states,
            posterior_history=posteriors
        )

        return hidden_states, summary
