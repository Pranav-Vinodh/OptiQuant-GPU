"""
OptiQuant-GPU: High-Throughput Regime-Adaptive Quantitative Portfolio Engine
Main Streamlit Quantitative Terminal Application.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from optiquant.config import (
    US_MARKET_PRESETS,
    NSE_MARKET_PRESETS,
    US_RISK_FREE_RATE,
    NSE_RISK_FREE_RATE,
    DEFAULT_SIMULATION_COUNT,
)
from optiquant.data.fetcher import MarketDataFetcher
from optiquant.data.preprocessor import MarketDataPreprocessor
from optiquant.models.hrp import HierarchicalRiskParity
from optiquant.models.regime_detector import MarketRegimeDetector
from optiquant.optimization.convex_solver import AnalyticalConvexOptimizer
from optiquant.optimization.modern_portfolio import PortfolioRiskAnalytics
from optiquant.systems.gpu_engine import GPUPortfolioEngine
from optiquant.systems.cpu_baseline import CPUBaselineEngine
from optiquant.ui.components import UIPlotter


def render_sidebar():
    st.sidebar.title("⚡ OptiQuant-GPU Engine")
    st.sidebar.markdown(
        """
        **Institutional Portfolio Optimization**
        - Unsupervised Gaussian HMM
        - Hierarchical Risk Parity (HRP)
        - Vectorized CUDA Simplex Kernels
        - Exact Convex Optimization (OSQP)
        """
    )
    st.sidebar.markdown("---")
    sim_count = st.sidebar.select_slider(
        "Simulations (Batched Tensor Size)",
        options=[10_000, 50_000, 100_000, 500_000, 1_000_000],
        value=100_000
    )
    history_years = st.sidebar.slider("Historical Lookback (Years)", 1, 5, 2)
    return sim_count, history_years


def run_portfolio_analysis(tickers, start_date, end_date, risk_free_rate, sim_count, is_nse=False):
    fetcher = MarketDataFetcher()
    preprocessor = MarketDataPreprocessor()

    with st.spinner("Ingesting market feeds and conditioning covariance matrices..."):
        prices = fetcher.fetch_prices(tickers, start_date, end_date, is_nse=is_nse)
        data = preprocessor.process(prices)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Assets Analyzed", len(data.cov_matrix.columns))
    col_m2.metric("Condition Number κ(Σ)", f"{data.condition_number:.1f}")
    col_m3.metric("Annualized Volatility (EWMA)", f"{data.market_ewma_vol_series.iloc[-1]:.2%}")
    col_m4.metric("Benchmark Risk-Free Rate", f"{risk_free_rate:.2%}")

    # Solvers
    convex_solver = AnalyticalConvexOptimizer(data.mean_returns.values, data.cov_matrix.values, risk_free_rate)
    hrp_solver = HierarchicalRiskParity(data.cov_matrix, data.corr_matrix)
    gpu_engine = GPUPortfolioEngine()
    analytics = PortfolioRiskAnalytics(risk_free_rate)

    with st.spinner("Executing exact Pareto optimization & GPU tensor simulation..."):
        # Analytical Convex
        max_sharpe_opt = convex_solver.max_sharpe()
        min_vol_opt = convex_solver.minimum_variance()
        frontier = convex_solver.compute_efficient_frontier(points=40)

        # HRP Allocation
        hrp_weights = hrp_solver.solve()
        hrp_r, hrp_v, hrp_s = convex_solver.portfolio_performance(hrp_weights.values)
        hrp_metrics = {"return": hrp_r, "volatility": hrp_v, "sharpe": hrp_s, "weights": hrp_weights.values}

        # GPU Monte Carlo Tensor Engine
        gpu_result = gpu_engine.simulate_allocations(
            data.mean_returns.values,
            data.cov_matrix.values,
            num_simulations=sim_count,
            risk_free_rate=risk_free_rate
        )

    st.success(
        f"Simulated **{sim_count:,}** allocations on `{gpu_result['device']}` in **{gpu_result['elapsed_ms']:.2f} ms** "
        f"({sim_count / max(gpu_result['elapsed_ms'] / 1000.0, 1e-6):,.0f} allocations/sec)."
    )

    # 1. Continuous Pareto Frontier Chart
    st.subheader("1. Continuous Pareto Frontier vs. Simplex Monte Carlo Space")
    frontier_fig = UIPlotter.plot_efficient_frontier(
        scatter_sample=gpu_result["scatter_sample"],
        frontier=frontier,
        max_sharpe=max_sharpe_opt,
        min_vol=min_vol_opt,
        hrp=hrp_metrics
    )
    st.plotly_chart(frontier_fig, use_container_width=True)

    # 2. Optimal Allocation Breakdowns
    st.subheader("2. Institutional Allocation Weights")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.markdown("**Maximum Sharpe Ratio**")
        sharpe_dict = dict(zip(data.cov_matrix.columns, max_sharpe_opt["weights"]))
        st.plotly_chart(UIPlotter.plot_weights_donut(sharpe_dict, "Max Sharpe"), use_container_width=True)
        st.caption(f"Return: {max_sharpe_opt['return']:.2%} | Vol: {max_sharpe_opt['volatility']:.2%} | Sharpe: {max_sharpe_opt['sharpe']:.2f}")

    with col_w2:
        st.markdown("**Minimum Variance**")
        min_vol_dict = dict(zip(data.cov_matrix.columns, min_vol_opt["weights"]))
        st.plotly_chart(UIPlotter.plot_weights_donut(min_vol_dict, "Min Variance"), use_container_width=True)
        st.caption(f"Return: {min_vol_opt['return']:.2%} | Vol: {min_vol_opt['volatility']:.2%} | Sharpe: {min_vol_opt['sharpe']:.2f}")

    with col_w3:
        st.markdown("**Hierarchical Risk Parity (HRP)**")
        hrp_dict = dict(zip(data.cov_matrix.columns, hrp_weights.values))
        st.plotly_chart(UIPlotter.plot_weights_donut(hrp_dict, "HRP Cluster Allocation"), use_container_width=True)
        st.caption(f"Return: {hrp_r:.2%} | Vol: {hrp_v:.2%} | Sharpe: {hrp_s:.2f}")

    # 3. Historical Backtest & Underwater Drawdown
    st.subheader("3. Historical Cumulative Equity Trajectory & Tail Risk")
    selected_weights = max_sharpe_opt["weights"]
    perf = analytics.evaluate(selected_weights, data.log_returns)

    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    col_r1.metric("Annualized Return", f"{perf.annualized_return:.2%}")
    col_r2.metric("Annualized Vol", f"{perf.annualized_volatility:.2%}")
    col_r3.metric("Sortino Ratio", f"{perf.sortino_ratio:.2f}")
    col_r4.metric("Max Drawdown", f"{perf.max_drawdown:.2%}")
    col_r5.metric("VaR 95% (CVaR)", f"{perf.var_95:.2%} ({perf.cvar_95:.2%})")

    dates = data.log_returns.index
    bm_returns = data.log_returns.mean(axis=1)
    bm_cum = np.exp(bm_returns.cumsum())

    wealth_fig = UIPlotter.plot_wealth_and_drawdown(
        dates=dates,
        cum_returns=perf.cumulative_returns,
        drawdowns=perf.drawdown_series,
        benchmark_returns=bm_cum
    )
    st.plotly_chart(wealth_fig, use_container_width=True)


def main():
    st.set_page_config(
        page_title="OptiQuant-GPU | Quantitative Portfolio Engine",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    sim_count, lookback_years = render_sidebar()
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 US Equity Portfolios",
        "🇮🇳 Indian NSE Portfolios",
        "🧠 Macro Regime Detection (HMM)",
        "⚡ Systems Latency Benchmark",
        "⚖️ Strategy Matrix"
    ])

    # TAB 1: US Portfolios
    with tab1:
        st.header("🇺🇸 US Institutional Equity Portfolio Engine")
        preset_name = st.selectbox("Select Strategy Preset", list(US_MARKET_PRESETS.keys()), key="us_preset")
        default_tickers = US_MARKET_PRESETS[preset_name]
        tickers_input = st.text_input("Tickers (comma-separated)", ", ".join(default_tickers), key="us_tickers")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        if st.button("Run US Optimization", type="primary", key="btn_us"):
            run_portfolio_analysis(tickers, start_date, end_date, US_RISK_FREE_RATE, sim_count, is_nse=False)

    # TAB 2: Indian NSE Portfolios
    with tab2:
        st.header("🇮🇳 Indian NSE Equity Quantitative Engine")
        preset_nse = st.selectbox("Select NSE Strategy Preset", list(NSE_MARKET_PRESETS.keys()), key="nse_preset")
        default_nse_tickers = NSE_MARKET_PRESETS[preset_nse]
        nse_input = st.text_input("Tickers (comma-separated)", ", ".join(default_nse_tickers), key="nse_tickers")
        tickers_nse = [t.strip().upper() for t in nse_input.split(",") if t.strip()]

        if st.button("Run NSE Optimization", type="primary", key="btn_nse"):
            run_portfolio_analysis(tickers_nse, start_date, end_date, NSE_RISK_FREE_RATE, sim_count, is_nse=True)

    # TAB 3: Unsupervised Regime Detection
    with tab3:
        st.header("🧠 Unsupervised Market Regime Detection (3-State Gaussian HMM)")
        st.markdown(
            """
            Market dynamics exhibit non-stationary phase transitions across macroeconomic cycles.
            OptiQuant-GPU fits a **3-State Gaussian Hidden Markov Model (HMM)** via the Baum-Welch (EM) algorithm
            on cross-sectional returns and realized EWMA volatility to dynamically classify regimes.
            """
        )
        regime_tickers = st.text_input("Regime Feed Tickers", "SPY, QQQ, GLD, TLT, IWM", key="regime_input")
        tickers_r = [t.strip().upper() for t in regime_tickers.split(",") if t.strip()]

        if st.button("Train HMM & Detect Regimes", type="primary", key="btn_regime"):
            with st.spinner("Training Gaussian HMM on return-volatility phase space..."):
                fetcher = MarketDataFetcher()
                preprocessor = MarketDataPreprocessor()
                prices = fetcher.fetch_prices(tickers_r, start_date, end_date, is_nse=False)
                data = preprocessor.process(prices)

                detector = MarketRegimeDetector(n_regimes=3)
                states, summary = detector.fit_predict(data.market_return_series, data.market_ewma_vol_series)

            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Current Inferred Regime", summary.current_regime_name)
            col_s2.metric("Suggested Risk Multiplier", f"{summary.suggested_risk_multiplier:.2f}x")
            col_s3.metric("Regime Confidence", f"{summary.regime_probabilities[summary.current_regime]:.1%}")

            st.subheader("Transition Probability Matrix A")
            df_trans = pd.DataFrame(
                summary.transition_matrix,
                index=[summary.regime_names[i] for i in range(3)],
                columns=[summary.regime_names[i] for i in range(3)]
            )
            st.dataframe(df_trans.style.format("{:.2%}"), use_container_width=True)

            # Normalized baseline price
            norm_price = (1.0 + data.market_return_series).cumprod()
            regime_fig = UIPlotter.plot_regime_timeline(
                dates=data.log_returns.index,
                prices=norm_price,
                regimes=states,
                posteriors=summary.posterior_history,
                regime_names=summary.regime_names
            )
            st.plotly_chart(regime_fig, use_container_width=True)

    # TAB 4: Systems Performance Benchmark
    with tab4:
        st.header("⚡ High-Throughput Systems Latency & Throughput Benchmark")
        st.markdown(
            """
            Direct comparative benchmark evaluating execution latency and throughput across:
            1. **Sequential Single-Core Python** (Python GIL, scalar iterations)
            2. **Multiprocessing Multi-Core Pool** (Python IPC pickling overhead)
            3. **Vectorized PyTorch Tensor Engine** (Zero-copy memory, SIMD vectorization)
            4. **OptiQuant Analytical Solver** (SLSQP/OSQP continuous Pareto frontier)
            """
        )
        bench_count = st.select_slider(
            "Benchmark Allocation Volume",
            options=[10_000, 50_000, 100_000, 500_000, 1_000_000],
            value=100_000,
            key="bench_slider"
        )

        if st.button("Execute Live Systems Benchmark", type="primary", key="btn_bench"):
            with st.spinner("Benchmarking execution engines..."):
                # Synthetic benchmark setup
                n_assets = 10
                rng = np.random.default_rng(42)
                mu = rng.uniform(0.08, 0.20, n_assets)
                a = rng.normal(0, 1, (n_assets, n_assets))
                cov = np.dot(a, a.T) * 0.04

                cpu_engine = CPUBaselineEngine()
                gpu_engine = GPUPortfolioEngine()
                convex = AnalyticalConvexOptimizer(mu, cov)

                # 1. Sequential (scale down if > 200,000 to prevent timeout)
                seq_count = min(bench_count, 50_000)
                seq_res = cpu_engine.run_sequential(mu, cov, num_simulations=seq_count)
                seq_latency = (seq_res["elapsed_ms"] / seq_count) * bench_count
                seq_tp = seq_res["throughput"]

                # 2. Multiprocessing
                mp_count = min(bench_count, 100_000)
                mp_res = cpu_engine.run_multiprocessing(mu, cov, num_simulations=mp_count)
                mp_latency = (mp_res["elapsed_ms"] / mp_count) * bench_count
                mp_tp = mp_res["throughput"]

                # 3. Vectorized PyTorch Tensor Engine
                tensor_res = gpu_engine.simulate_allocations(mu, cov, num_simulations=bench_count)
                tensor_latency = tensor_res["elapsed_ms"]
                tensor_tp = bench_count / max(tensor_latency / 1000.0, 1e-6)

                # 4. Analytical Convex Solver
                t0 = datetime.now()
                convex.max_sharpe()
                convex.minimum_variance()
                convex.compute_efficient_frontier(points=30)
                convex_latency = (datetime.now() - t0).total_seconds() * 1000.0

            results = [
                {"Engine": "Sequential CPU", "Latency (ms)": seq_latency, "Speedup": "1.0x", "Throughput": seq_tp},
                {"Engine": "Multiprocessing CPU (8 Cores)", "Latency (ms)": mp_latency, "Speedup": f"{seq_latency / max(mp_latency, 1e-6):.1f}x", "Throughput": mp_tp},
                {"Engine": f"Vectorized Tensor ({tensor_res['device']})", "Latency (ms)": tensor_latency, "Speedup": f"{seq_latency / max(tensor_latency, 1e-6):.1f}x", "Throughput": tensor_tp},
                {"Engine": "Analytical Convex (SLSQP)", "Latency (ms)": convex_latency, "Speedup": f"{seq_latency / max(convex_latency, 1e-6):.1f}x", "Throughput": bench_count / max(convex_latency / 1000.0, 1e-6)}
            ]

            df_bench = pd.DataFrame(results)
            st.dataframe(df_bench.style.format({"Latency (ms)": "{:.2f}", "Throughput": "{:,.0f}"}), use_container_width=True)
            bench_fig = UIPlotter.plot_benchmark_latency(df_bench)
            st.plotly_chart(bench_fig, use_container_width=True)

    # TAB 5: Strategy Matrix
    with tab5:
        st.header("⚖️ Quantitative Strategy Allocation Matrix")
        st.markdown(
            """
            Direct matrix comparing classical Markowitz mean-variance optimization against
            Hierarchical Risk Parity (HRP) tree clustering.
            """
        )
        st.markdown(
            """
            | Strategy | Primary Objective | Mathematical Core | Covariance Sensitivity |
            | :--- | :--- | :--- | :--- |
            | **Max Sharpe Ratio** | Maximize risk-adjusted excess return | $\\max_{\\mathbf{w}} \\frac{\\mathbf{w}^T \\boldsymbol{\\mu} - R_f}{\\sqrt{\\mathbf{w}^T \\boldsymbol{\\Sigma} \\mathbf{w}}}$ | High ($\kappa(\boldsymbol{\Sigma}) > 10^3$ instability) |
            | **Minimum Variance** | Minimize absolute portfolio volatility | $\\min_{\\mathbf{w}} \\mathbf{w}^T \\boldsymbol{\\Sigma} \\mathbf{w}$ | Moderate |
            | **Hierarchical Risk Parity (HRP)** | Inverse-cluster variance parity | Graph clustering + Dendrogram quasi-diagonalization | **Zero matrix inversion** (Stable across all regimes) |
            | **Equal-Weight Baseline** | Naive $1/N$ diversification | $w_i = 1/N$ | None |
            """
        )


if __name__ == "__main__":
    main()
