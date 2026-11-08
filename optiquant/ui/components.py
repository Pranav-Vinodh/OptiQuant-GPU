"""
Interactive Plotly Visualizations for OptiQuant-GPU Terminal.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class UIPlotter:
    """
    Renders institutional-grade Plotly charts with dark financial styling.
    """

    DARK_LAYOUT = dict(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#C9D1D9", family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D")
    )

    @classmethod
    def plot_efficient_frontier(
        cls,
        scatter_sample: Dict[str, np.ndarray],
        frontier: Dict[str, np.ndarray],
        max_sharpe: Dict[str, float],
        min_vol: Dict[str, float],
        hrp: Dict[str, float] = None
    ) -> go.Figure:
        """Plots the continuous exact frontier alongside the GPU Monte Carlo cloud and optimal portfolios."""
        fig = go.Figure()

        # 1. Monte Carlo scatter cloud
        fig.add_trace(go.Scatter(
            x=scatter_sample["volatilities"],
            y=scatter_sample["returns"],
            mode="markers",
            marker=dict(
                size=3,
                color=scatter_sample["sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe Ratio", thickness=15),
                opacity=0.45
            ),
            name="GPU Monte Carlo Portfolios",
            hoverinfo="skip"
        ))

        # 2. Continuous Analytical Pareto Curve
        if len(frontier.get("returns", [])) > 0:
            fig.add_trace(go.Scatter(
                x=frontier["volatilities"],
                y=frontier["returns"],
                mode="lines",
                line=dict(color="#00E676", width=3, dash="solid"),
                name="Exact Analytical Pareto Frontier (SLSQP)"
            ))

        # 3. Maximum Sharpe Portfolio
        fig.add_trace(go.Scatter(
            x=[max_sharpe["volatility"]],
            y=[max_sharpe["return"]],
            mode="markers+text",
            marker=dict(color="#FFD600", size=14, symbol="star", line=dict(color="#FFFFFF", width=1.5)),
            name=f"Max Sharpe ({max_sharpe['sharpe']:.2f})",
            text=["Max Sharpe"],
            textposition="top center",
            hovertext=f"Return: {max_sharpe['return']:.2%}<br>Vol: {max_sharpe['volatility']:.2%}<br>Sharpe: {max_sharpe['sharpe']:.2f}"
        ))

        # 4. Minimum Volatility Portfolio
        fig.add_trace(go.Scatter(
            x=[min_vol["volatility"]],
            y=[min_vol["return"]],
            mode="markers+text",
            marker=dict(color="#29B6F6", size=13, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
            name=f"Min Volatility ({min_vol['volatility']:.2%})",
            text=["Min Vol"],
            textposition="bottom center",
            hovertext=f"Return: {min_vol['return']:.2%}<br>Vol: {min_vol['volatility']:.2%}<br>Sharpe: {min_vol['sharpe']:.2f}"
        ))

        # 5. Hierarchical Risk Parity (HRP)
        if hrp:
            fig.add_trace(go.Scatter(
                x=[hrp["volatility"]],
                y=[hrp["return"]],
                mode="markers+text",
                marker=dict(color="#FF4081", size=13, symbol="cross", line=dict(color="#FFFFFF", width=1.5)),
                name=f"HRP Cluster ({hrp['sharpe']:.2f})",
                text=["HRP"],
                textposition="middle right",
                hovertext=f"Return: {hrp['return']:.2%}<br>Vol: {hrp['volatility']:.2%}<br>Sharpe: {hrp['sharpe']:.2f}"
            ))

        layout = dict(**cls.DARK_LAYOUT)
        layout.update(
            title="<b>Continuous Pareto Frontier vs. GPU Tensor Simplex Space</b>",
            xaxis_title="Annualized Volatility (Risk)",
            yaxis_title="Expected Annual Return",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(22, 27, 34, 0.8)")
        )
        fig.update_layout(layout)
        return fig

    @classmethod
    def plot_weights_donut(cls, weights: Dict[str, float], title: str = "Asset Allocation") -> go.Figure:
        """Plots asset allocation weights as an institutional donut chart."""
        labels = list(weights.keys())
        values = [max(0.0, float(v)) for v in weights.values()]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            textinfo="label+percent",
            insidetextorientation="radial",
            marker=dict(colors=["#2962FF", "#00B0FF", "#00E676", "#FFD600", "#FF6D00", "#FF1744", "#AA00FF", "#7C4DFF"])
        )])

        layout = dict(**cls.DARK_LAYOUT)
        layout.update(title=f"<b>{title}</b>", showlegend=True)
        fig.update_layout(layout)
        return fig

    @classmethod
    def plot_regime_timeline(
        cls,
        dates: pd.DatetimeIndex,
        prices: pd.Series,
        regimes: np.ndarray,
        posteriors: np.ndarray,
        regime_names: Dict[int, str]
    ) -> go.Figure:
        """Dual-panel chart showing market prices colored by regime and posterior probabilities."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("<b>Market Price Dynamics across Unsupervised Macro Regimes</b>", "<b>Regime Posterior Probabilities (Gaussian HMM)</b>"),
            row_heights=[0.6, 0.4]
        )

        regime_colors = {0: "#FF1744", 1: "#FFD600", 2: "#00E676"}  # Bear, Neutral, Bull

        # Price line colored by dominant state
        df_p = pd.DataFrame({"Date": dates, "Price": prices, "Regime": regimes})
        for state, name in regime_names.items():
            mask = df_p["Regime"] == state
            fig.add_trace(go.Scatter(
                x=df_p["Date"][mask],
                y=df_p["Price"][mask],
                mode="markers",
                marker=dict(size=4, color=regime_colors.get(state, "#999999")),
                name=f"State: {name}"
            ), row=1, col=1)

        # Baseline price curve
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode="lines",
            line=dict(color="rgba(200, 200, 200, 0.3)", width=1),
            showlegend=False,
            hoverinfo="skip"
        ), row=1, col=1)

        # Posterior probabilities
        for state, name in regime_names.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=posteriors[:, state],
                mode="lines",
                name=f"P({name})",
                line=dict(color=regime_colors.get(state, "#999999"), width=1.5)
            ), row=2, col=1)

        layout = dict(**cls.DARK_LAYOUT)
        layout.update(
            height=650,
            xaxis2_title="Date",
            yaxis1_title="Normalized Index Price",
            yaxis2_title="State Probability",
            legend=dict(orientation="h", y=1.06, x=0.0)
        )
        fig.update_layout(layout)
        return fig

    @classmethod
    def plot_wealth_and_drawdown(
        cls,
        dates: pd.DatetimeIndex,
        cum_returns: pd.Series,
        drawdowns: pd.Series,
        benchmark_returns: pd.Series = None
    ) -> go.Figure:
        """Plots cumulative portfolio equity growth and underwater drawdown curve."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("<b>Cumulative Portfolio Wealth ($1.00 Base)</b>", "<b>Drawdown Underwater Curve</b>"),
            row_heights=[0.65, 0.35]
        )

        fig.add_trace(go.Scatter(
            x=dates,
            y=cum_returns.values,
            mode="lines",
            line=dict(color="#00E676", width=2.2),
            name="Optimized Strategy"
        ), row=1, col=1)

        if benchmark_returns is not None:
            fig.add_trace(go.Scatter(
                x=dates,
                y=benchmark_returns.values,
                mode="lines",
                line=dict(color="#90CAF9", width=1.5, dash="dash"),
                name="Equal-Weight Benchmark"
            ), row=1, col=1)

        # Drawdown filled area
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdowns.values,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#FF1744", width=1.5),
            name="Portfolio Drawdown"
        ), row=2, col=1)

        layout = dict(**cls.DARK_LAYOUT)
        layout.update(
            height=600,
            xaxis2_title="Date",
            yaxis1_title="Growth Factor",
            yaxis2_title="Drawdown (%)",
            legend=dict(orientation="h", y=1.06, x=0.0)
        )
        fig.update_layout(layout)
        return fig

    @classmethod
    def plot_benchmark_latency(cls, benchmark_df: pd.DataFrame) -> go.Figure:
        """Plots system latency and speedup across engines on logarithmic scale."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("<b>Execution Latency (ms, Log Scale)</b>", "<b>Throughput (Portfolios / Sec)</b>")
        )

        fig.add_trace(go.Bar(
            x=benchmark_df["Engine"],
            y=benchmark_df["Latency (ms)"],
            marker=dict(color=["#9E9E9E", "#FFB300", "#03A9F4", "#00E676", "#E040FB"]),
            name="Latency"
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=benchmark_df["Engine"],
            y=benchmark_df["Throughput"],
            marker=dict(color=["#9E9E9E", "#FFB300", "#03A9F4", "#00E676", "#E040FB"]),
            name="Throughput"
        ), row=1, col=2)

        layout = dict(**cls.DARK_LAYOUT)
        layout.update(
            title="<b>OptiQuant-GPU Latency & Throughput Benchmark Analysis</b>",
            yaxis1=dict(type="log", title="Latency (ms)", gridcolor="#30363D"),
            yaxis2=dict(type="log", title="Allocations / Sec", gridcolor="#30363D"),
            showlegend=False,
            height=450
        )
        fig.update_layout(layout)
        return fig
