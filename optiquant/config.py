"""
Global Configuration and Presets for OptiQuant-GPU Engine
"""

from dataclasses import dataclass
from typing import Dict, List

# Benchmark risk-free rates (annualized)
US_RISK_FREE_RATE = 0.045     # 10-Year US Treasury benchmark (~4.5%)
NSE_RISK_FREE_RATE = 0.065    # India 10-Year G-Sec / Repo benchmark (~6.5%)

# Market Asset Presets
US_MARKET_PRESETS: Dict[str, List[str]] = {
    "Mega-Cap Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "Diversified Blue Chip": ["AAPL", "JPM", "JNJ", "PG", "XOM", "UNH", "BRK-B"],
    "Quant Momentum": ["NVDA", "AMD", "AVGO", "SMCI", "META", "MSFT"],
    "Low Volatility & Dividend": ["PG", "KO", "JNJ", "PEP", "MCD", "WMT", "COST"],
}

NSE_MARKET_PRESETS: Dict[str, List[str]] = {
    "NIFTY 50 Bluechip": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "ITC.NS", "BHARTIARTL.NS"],
    "NIFTY IT Leaders": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"],
    "Banking & Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS"],
    "Auto & Manufacturing": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
}

DEFAULT_SIMULATION_COUNT = 500_000
BENCHMARK_SIMULATION_POINTS = [10_000, 50_000, 100_000, 500_000, 1_000_000]

# UI Chart Theme
PALETTE = {
    "primary": "#1E88E5",
    "accent": "#00E676",
    "danger": "#FF1744",
    "warning": "#FFD600",
    "background": "#0E1117",
    "surface": "#1E222D",
    "text": "#E0E6ED",
    "muted": "#8A99AD"
}
