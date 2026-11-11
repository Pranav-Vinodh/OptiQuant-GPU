# OptiQuant-GPU

**A High-Throughput, Regime-Adaptive Quantitative Portfolio Optimization Engine**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Accelerated-green.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OptiQuant-GPU is a quantitative portfolio optimization and risk analytics framework built in Python and PyTorch. It replaces slow, naive Monte Carlo scripts and unstable classical Markowitz models with unsupervised regime detection, graph-based asset allocation, exact convex optimization, and hardware-accelerated tensor simulation.

---

## Why This Project Exists

Classical portfolio optimization faces three major practical bottlenecks:

1. **Unstable Covariance Inversion:** Standard Markowitz optimization requires inverting the asset covariance matrix. When stocks are strongly correlated, this matrix becomes ill-conditioned, turning small estimation noise into extreme, fragile portfolio weights.
2. **Static Market Assumptions:** Markets switch between calm bull runs, sideways chop, and high-volatility crashes. Optimizing with a single static risk profile fails to adapt when market volatility spikes.
3. **Slow Simulation Loops:** Evaluating hundreds of thousands of portfolio allocations in pure Python or via `multiprocessing.Pool` spends most of its time fighting the GIL, serializing objects (`pickle`), and copying data across CPU processes.

### How OptiQuant-GPU Solves This

- **3-State Gaussian HMM Regime Detection:** Uses an unsupervised Hidden Markov Model on daily returns and rolling volatility to classify the current market state (*Bull*, *Neutral*, or *Bear*) and suggest dynamic risk scaling.
- **Hierarchical Risk Parity (HRP):** Clusters correlated assets into a tree structure and allocates risk top-down using recursive bisection—achieving stable, diversified portfolios without inverting the covariance matrix.
- **Vectorized PyTorch Tensor Engine:** Evaluates 1,000,000+ random portfolio allocations in a single batched matrix operation on GPU VRAM (with seamless multi-threaded CPU tensor fallback), cutting runtime from ~18.3 seconds down to ~204 milliseconds on CPU (89.4x speedup).
- **Analytical Convex Solver (SLSQP):** Mathematically solves for the exact Maximum Sharpe portfolio, Minimum Variance portfolio, and continuous Pareto frontier.
- **Multi-Market Support:** Fetches live market data for both **US Equities (NYSE/NASDAQ)** and **Indian Equities (NSE)** with localized risk-free benchmark rates (4.5% US Treasury / 6.5% India G-Sec) and synthetic GBM fallback when offline.

---

## System Architecture

```
  [Market Data Feed: US & Indian NSE via yfinance]
                         │
                         ▼
        [Preprocessor: Log Returns & EWMA Vol]
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
  [3-State Gaussian HMM]     [Hierarchical Risk Parity]
  (Bull / Neutral / Bear)    (Cluster-Based Weights)
           │                           │
           └─────────────┬─────────────┘
                         ▼
         [Optimization & Simulation Engines]
           ┌─────────────┴─────────────┐
           ▼                           ▼
  [Exact SLSQP Solver]      [PyTorch Tensor Kernel]
  (Pareto Frontier)         (1M+ Allocations in Parallel)
           │                           │
           └─────────────┬─────────────┘
                         ▼
        [Risk Metrics: Sharpe, Sortino, VaR, CVaR]
                         │
                         ▼
         [Interactive Streamlit Dashboard]
```

---

## Key Features

- **5-Tab Quantitative Terminal (`Streamlit` + `Plotly`):**
  1. **US Equity Portfolios:** Presets for Mega-Cap Tech, Blue Chips, Quant Momentum, and Dividend stocks.
  2. **Indian NSE Portfolios:** Presets for NIFTY 50, NIFTY IT, Banking, and Auto sectors with automatic `.NS` ticker normalization.
  3. **Macro Regime Detection:** Visualizes HMM transition probability matrices, market state timelines, and posterior regime confidence.
  4. **Systems Latency Benchmark:** Runs live head-to-head comparisons of Sequential Python, Multiprocessing Pool, Vectorized PyTorch, and Analytical Solvers.
  5. **Strategy Matrix:** Side-by-side comparison of Max Sharpe, Minimum Variance, HRP, and Equal-Weight strategies.
- **Tail Risk & Backtesting:** Evaluates annualized return/volatility, Sortino ratio, Calmar ratio, maximum drawdown curves, and Historical VaR / CVaR (95% & 99%).

---

## Performance Benchmarks

Measured on 1,000,000 simulated portfolios across 15 assets:

| Execution Engine | Latency | Speedup vs Sequential | Throughput |
| :--- | :---: | :---: | :---: |
| **Python Sequential (Single-Core)** | 18,275 ms | 1.0x | ~54,700 portfolios/sec |
| **Python Multiprocessing Pool** | 3,593 ms | 5.1x | ~278,300 portfolios/sec |
| **PyTorch Vectorized (CPU)** | 212 ms | 86.3x | ~4.72M portfolios/sec |
| **OptiQuant Tensor Engine (CPU)** | 204 ms | 89.4x | ~4.89M portfolios/sec |
| **OptiQuant Tensor Engine (CUDA GPU)** | Hardware-dependent | GPU-accelerated | Zero-copy VRAM execution |

*CPU timings above were measured on an 8-core host across 1,000,000 portfolios and 15 assets. When an NVIDIA CUDA runtime is active, the tensor engine runs directly in GPU VRAM.*

---

## Project Structure

```
Parallel-Portfolio-Optimization-Dashboard/
├── app.py                        # Streamlit entrypoint
├── pyproject.toml                # Package configuration
├── requirements.txt              # Dependencies
├── optiquant/
│   ├── config.py                 # Market presets and benchmark risk-free rates
│   ├── data/
│   │   ├── fetcher.py            # Yahoo Finance ingestion + synthetic fallback
│   │   └── preprocessor.py       # Log returns, EWMA volatility, covariance checks
│   ├── models/
│   │   ├── regime_detector.py    # 3-State Gaussian HMM classifier
│   │   └── hrp.py                # Hierarchical Risk Parity allocator
│   ├── optimization/
│   │   ├── convex_solver.py      # SLSQP Pareto frontier, Max Sharpe, Min Var
│   │   └── modern_portfolio.py   # Sharpe, Sortino, Drawdown, VaR, CVaR
│   ├── systems/
│   │   ├── gpu_engine.py         # Vectorized PyTorch CUDA/CPU portfolio simulator
│   │   └── cpu_baseline.py       # Sequential and multiprocessing baselines
│   └── ui/
│       ├── components.py         # Plotly interactive charts
│       └── dashboard.py          # Multi-tab Streamlit dashboard
├── benchmarks/
│   ├── benchmark_execution.py    # Latency and throughput benchmark script
│   └── memory_profile.py         # Tensor memory footprint profiler
└── tests/
    ├── test_data.py
    ├── test_regime.py
    ├── test_solvers.py
    └── test_gpu_engine.py
```

---

## Quickstart

### 1. Setup with Conda (Recommended)

```bash
git clone https://github.com/Pranav-Vinodh/Parallel-Portfolio-Optimization-Dashboard.git
cd Parallel-Portfolio-Optimization-Dashboard

conda create -n optiquant python=3.11 -y
conda activate optiquant
pip install -e .
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

### 3. Run Benchmarks & Tests

```bash
# Run unit tests
python -m unittest discover tests/

# Run systems latency benchmark
python -m benchmarks.benchmark_execution

# Run tensor memory profile
python -m benchmarks.memory_profile
```

---

## License

MIT License
