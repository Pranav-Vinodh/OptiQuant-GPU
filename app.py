import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from multiprocessing import Pool

st.set_page_config(page_title="Parallel Portfolio Optimizer", layout="wide")

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Black background */
    .stApp {
        background-color: #000000;
    }
    
    /* Center main title */
    h1 {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding: 1rem 0;
    }
    
    /* Center headers */
    h2, h3, h4 {
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Professional parameter box */
    .param-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 2px solid #333333;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 90%;
        box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1), 
                    0 8px 40px rgba(255, 255, 255, 0.05),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Center content in parameter box */
    .param-container > div {
        text-align: center;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem auto;
        max-width: 90%;
        text-align: center;
        box-shadow: 0 2px 10px rgba(255, 255, 255, 0.05);
    }
    
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Section headers in boxes */
    .section-header {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #ffffff;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #444444;
    }
</style>
""", unsafe_allow_html=True)

st.title("💹 Parallel Portfolio Optimization Dashboard")

st.markdown("""
<div class="info-box">
    <p style="font-size: 1rem; line-height: 1.6; margin: 0;">
        This app demonstrates <strong>Parallel Portfolio Optimization</strong> —  
        evaluating thousands of random portfolio allocations in parallel to find  
        the most efficient risk–return tradeoff using real historical stock data.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# Helper functions
# =====================================================================

def fetch_data(tickers, start, end):
    """Download adjusted close prices and compute daily returns robustly."""
    data = yf.download(tickers, start=start, end=end, progress=False)

    # Handle single vs multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        else:
            # Fallback in case 'Adj Close' is missing
            data = data.xs("Close", axis=1, level=0)
    else:
        # Single ticker — data is a simple DataFrame
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]
        else:
            raise ValueError("No valid price column found in downloaded data.")

    # Drop any tickers that failed to download
    data = data.dropna(axis=1, how="all")
    if data.empty:
        raise ValueError("No valid stock data fetched. Check tickers or date range.")

    returns = np.log(data / data.shift(1)).dropna()
    mu = returns.mean() * 252
    Sigma = returns.cov() * 252
    return data, returns, mu, Sigma


def fetch_nse_data(tickers, start, end):
    """Download NSE stock data using yfinance with .NS suffix."""
    # Add .NS suffix to NSE tickers for yfinance
    nse_tickers = [f"{ticker}.NS" for ticker in tickers]
    
    # Use the same fetch_data logic as US stocks
    data = yf.download(nse_tickers, start=start, end=end, progress=False)
    
    # Handle single vs multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        else:
            # Fallback in case 'Adj Close' is missing
            data = data.xs("Close", axis=1, level=0)
    else:
        # Single ticker — data is a simple DataFrame
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]
        else:
            raise ValueError("No valid price column found in downloaded data.")
    
    # Drop any tickers that failed to download
    data = data.dropna(axis=1, how="all")
    
    if data.empty:
        raise ValueError("No valid NSE stock data fetched. Check tickers or date range.")
    
    # Remove .NS suffix from column names for cleaner display
    data.columns = [col.replace('.NS', '') for col in data.columns]
    
    # Compute returns and statistics
    returns = np.log(data / data.shift(1)).dropna()
    mu = returns.mean() * 252  # Annualized return
    Sigma = returns.cov() * 252  # Annualized covariance
    
    return data, returns, mu, Sigma


def sample_portfolios(seed_and_count):
    """Simulate random portfolio samples (used for parallel processing)."""
    seed, count, mu, Sigma, rf = seed_and_count
    np.random.seed(seed)
    results = []
    for _ in range(count):
        w = np.random.rand(len(mu))
        w /= w.sum()
        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)
        sharpe = (ret - rf) / vol if vol > 0 else -np.inf
        results.append((sharpe, ret, vol, w))
    return results


def run_parallel_optimization(mu, Sigma, rf, num_portfolios, num_workers):
    """Runs the optimization using multiprocessing."""
    samples_per_worker = num_portfolios // num_workers
    args = [(i, samples_per_worker, mu, Sigma, rf) for i in range(num_workers)]
    t0 = time.time()
    with Pool(num_workers) as pool:
        res = pool.map(sample_portfolios, args)
    total_time = time.time() - t0
    all_results = [item for sublist in res for item in sublist]
    df = pd.DataFrame(all_results, columns=["Sharpe", "Return", "Vol", "Weights"])
    return df, total_time


def calculate_portfolio_metrics(returns, rf=0.02):
    """Calculate comprehensive portfolio metrics."""
    cum_returns = (1 + returns).cumprod()
    
    # Basic metrics
    total_return = (cum_returns.iloc[-1] - 1) * 100
    annual_return = returns.mean() * 252 * 100
    annual_vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() * 252 - rf) / (returns.std() * np.sqrt(252))
    
    # Downside metrics
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(252)
    sortino = (returns.mean() * 252 - rf) / downside_std if downside_std > 0 else 0
    
    # Drawdown
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'cum_returns': cum_returns,
        'drawdowns': drawdowns
    }


def calculate_risk_metrics(returns, confidence_levels=[0.05, 0.01]):
    """Calculate Value at Risk (VaR) and Conditional VaR (CVaR)."""
    risk_metrics = {}
    for conf in confidence_levels:
        var = np.percentile(returns, conf * 100) * 100
        cvar = returns[returns <= np.percentile(returns, conf * 100)].mean() * 100
        risk_metrics[f'VaR_{int(conf*100)}'] = var
        risk_metrics[f'CVaR_{int(conf*100)}'] = cvar
    return risk_metrics


def monte_carlo_simulation(returns, weights, num_simulations=1000, days=252, initial_value=10000):
    """Run Monte Carlo simulation for future portfolio values."""
    # Calculate portfolio statistics
    portfolio_returns = (returns @ weights).dropna()
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    # Simulate future returns
    simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, days))
    simulated_values = (1 + simulated_returns).cumprod(axis=1) * initial_value
    
    return simulated_values


def calculate_portfolio_from_weights(weights, mu, Sigma, rf):
    """Calculate portfolio metrics from given weights."""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    ret = weights @ mu
    vol = np.sqrt(weights @ Sigma @ weights)
    sharpe = (ret - rf) / vol if vol > 0 else -np.inf
    
    return {
        'return': ret,
        'vol': vol,
        'sharpe': sharpe,
        'weights': weights
    }


# =====================================================================
# TABS
# =====================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Optimization (US)", "🇮🇳 Portfolio Optimization (NSE)", "⚙️ Parallel Speedup Benchmark", "🔀 Portfolio Comparison"])

# ---------------------------------------------------------------------
# TAB 1: Portfolio Optimization
# ---------------------------------------------------------------------
with tab1:
    st.header("📊 Portfolio Optimization")
    
    # Parameter selection in styled container
    st.markdown("""
    <div class="param-container">
        <div class="section-header">⚙️ Optimization Parameters</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickers_input = st.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, GOOG, AMZN, META", key="tickers_tab1")
        num_portfolios = st.number_input("Number of portfolios to simulate:", 1000, 1000000, 10000, step=1000, key="num_portfolios_tab1")
        num_workers = st.slider("Number of parallel workers:", 1, 8, 4, key="workers_tab1")
    
    with col2:
        metric = st.selectbox("Optimization Metric:", ["Max Sharpe", "Max Return", "Min Risk"], key="metric_tab1")
        rf = st.number_input("Risk-free rate (annual):", 0.0, 0.1, 0.02, step=0.005, key="rf_tab1")
    
    with col3:
        start_date = st.date_input("Start date:", pd.to_datetime("2018-01-01"), key="start_tab1")
        end_date = st.date_input("End date:", pd.to_datetime("2024-01-01"), key="end_tab1")
    
    st.markdown('</div>', unsafe_allow_html=True)

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if st.button("🚀 Run Optimization", key="run_tab1"):
        with st.spinner("Fetching data and running optimization..."):
            data, returns, mu, Sigma = fetch_data(tickers, start_date, end_date)
            df, total_time = run_parallel_optimization(mu, Sigma, rf, num_portfolios, num_workers)

        # --- Select best portfolio ---
        if metric == "Max Sharpe":
            best = df.loc[df["Sharpe"].idxmax()]
        elif metric == "Max Return":
            best = df.loc[df["Return"].idxmax()]
        else:
            best = df.loc[df["Vol"].idxmin()]

        # Calculate portfolio metrics
        best_weights = np.array(best["Weights"])
        portfolio_returns = (returns @ best_weights).dropna()
        metrics = calculate_portfolio_metrics(portfolio_returns, rf)
        
        # Store results in session state for persistence
        st.session_state['tab1_data'] = data
        st.session_state['tab1_returns'] = returns
        st.session_state['tab1_mu'] = mu
        st.session_state['tab1_Sigma'] = Sigma
        st.session_state['tab1_df'] = df
        st.session_state['tab1_total_time'] = total_time
        st.session_state['tab1_best'] = best
        st.session_state['tab1_best_weights'] = best_weights
        st.session_state['tab1_portfolio_returns'] = portfolio_returns
        st.session_state['tab1_metrics'] = metrics
        st.session_state['tab1_tickers'] = tickers
        st.session_state['tab1_rf'] = rf
        st.session_state['tab1_metric'] = metric

        st.success(f"✅ Optimization completed in **{total_time:.2f}s** using **{num_workers} workers**")

    # Check if we have stored results to display
    if 'tab1_data' in st.session_state:
        # Retrieve from session state
        data = st.session_state['tab1_data']
        returns = st.session_state['tab1_returns']
        mu = st.session_state['tab1_mu']
        Sigma = st.session_state['tab1_Sigma']
        df = st.session_state['tab1_df']
        total_time = st.session_state['tab1_total_time']
        best = st.session_state['tab1_best']
        best_weights = st.session_state['tab1_best_weights']
        portfolio_returns = st.session_state['tab1_portfolio_returns']
        metrics = st.session_state['tab1_metrics']
        tickers = st.session_state['tab1_tickers']
        rf = st.session_state['tab1_rf']
        metric = st.session_state['tab1_metric']

        # --- Key Metrics Row ---
        st.markdown("---")
        st.subheader("🏆 Optimized Portfolio Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            profit_color = "normal" if metrics['total_return'] >= 0 else "inverse"
            st.metric(
                "Total Return", 
                f"{metrics['total_return']:.2f}%",
                delta=f"{metrics['annual_return']:.2f}% annual",
                delta_color=profit_color
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio", 
                f"{best['Sharpe']:.2f}",
                delta=f"Sortino: {metrics['sortino']:.2f}"
            )
        
        with col3:
            st.metric(
                "Volatility (Risk)", 
                f"{best['Vol']*100:.2f}%",
                delta=f"Max DD: {metrics['max_drawdown']:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Win Rate", 
                f"{metrics['win_rate']:.1f}%",
                delta=f"{int((portfolio_returns > 0).sum())} winning days"
            )

        # --- Interactive Efficient Frontier ---
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Efficient Frontier")
            fig = go.Figure()
            
            # All portfolios
            fig.add_trace(go.Scatter(
                x=df["Vol"]*100,
                y=df["Return"]*100,
                mode='markers',
                marker=dict(
                    size=4,
                    color=df["Sharpe"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                    line=dict(width=0)
                ),
                text=[f"Return: {r*100:.2f}%<br>Risk: {v*100:.2f}%<br>Sharpe: {s:.2f}" 
                      for r, v, s in zip(df["Return"], df["Vol"], df["Sharpe"])],
                hovertemplate='%{text}<extra></extra>',
                name='Portfolios'
            ))
            
            # Best portfolio
            fig.add_trace(go.Scatter(
                x=[best["Vol"]*100],
                y=[best["Return"]*100],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                name='Best Portfolio',
                hovertemplate=f'<b>Best Portfolio</b><br>Return: {best["Return"]*100:.2f}%<br>Risk: {best["Vol"]*100:.2f}%<br>Sharpe: {best["Sharpe"]:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis_title="Volatility (Risk) %",
                yaxis_title="Expected Annual Return %",
                height=500,
                hovermode='closest',
                showlegend=True,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Portfolio Weights")
            weights_df = pd.DataFrame({"Ticker": tickers, "Weight": best["Weights"]})
            weights_df = weights_df.sort_values("Weight", ascending=False)
            
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=weights_df["Ticker"],
                values=weights_df["Weight"],
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set3),
                textinfo='label+percent',
                textposition='auto'
            )])
            fig_pie.update_layout(
                height=400,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Weight table
            weights_df["Weight %"] = (weights_df["Weight"] * 100).round(2)
            st.dataframe(
                weights_df[["Ticker", "Weight %"]],
                hide_index=True,
                use_container_width=True
            )

        # --- Backtest Performance ---
        st.markdown("---")
        st.subheader("💰 Historical Backtest Performance")
        
        # Create subplot with cumulative returns and drawdown
        fig_backtest = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Portfolio Value Growth ($10,000 Initial)', 'Drawdown %'),
            row_heights=[0.7, 0.3]
        )
        
        # Cumulative returns
        portfolio_value = metrics['cum_returns'] * 10000
        fig_backtest.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,176,80,0.2)',
                hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig_backtest.add_trace(
            go.Scatter(
                x=metrics['drawdowns'].index,
                y=metrics['drawdowns'],
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_backtest.update_xaxes(title_text="Date", row=2, col=1)
        fig_backtest.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig_backtest.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        fig_backtest.update_layout(
            height=600,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_backtest, use_container_width=True)
        
        # Performance summary
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_value = 10000
            final_value = portfolio_value.iloc[-1]
            profit = final_value - initial_value
            st.markdown(f"""
            **💵 Investment Growth**
            - Initial: ${initial_value:,.0f}
            - Final: ${final_value:,.2f}
            - Profit: ${profit:,.2f}
            """)
        
        with col2:
            best_day = portfolio_returns.max() * 100
            worst_day = portfolio_returns.min() * 100
            st.markdown(f"""
            **📊 Daily Performance**
            - Best Day: +{best_day:.2f}%
            - Worst Day: {worst_day:.2f}%
            - Avg Daily: {portfolio_returns.mean()*100:.3f}%
            """)
        
        with col3:
            years = len(portfolio_returns) / 252
            cagr = ((metrics['cum_returns'].iloc[-1]) ** (1/years) - 1) * 100 if years > 0 else 0
            st.markdown(f"""
            **📈 Long-term Stats**
            - CAGR: {cagr:.2f}%
            - Period: {years:.1f} years
            - Risk-Adj Return: {metrics['sharpe']:.2f}
            """)

        st.info("💡 **Tip:** Try varying the number of workers or portfolios to observe changes in runtime and results.")

        # =====================================================================
        # NEW INTERACTIVE FEATURES
        # =====================================================================
        
        st.markdown("---")
        st.subheader("🎮 Interactive Features")
        
        feature_tabs = st.tabs(["🔄 Portfolio Rebalancing", "📊 Correlation Analysis", "🎲 Monte Carlo Simulation", "⚠️ Risk Analysis", "📥 Export Results"])
        
        # Feature 1: Interactive Portfolio Rebalancing
        with feature_tabs[0]:
            st.markdown("### 🔄 Interactive Portfolio Rebalancing")
            st.markdown("Manually adjust portfolio weights using sliders and see real-time impact on metrics.")
            
            # Check if reset button was clicked (before creating sliders)
            reset_clicked = False
            if 'reset_weights_tab1' in st.session_state and st.session_state.get('reset_weights_tab1', False):
                # Delete all slider keys to reset them
                for ticker in tickers:
                    slider_key = f"slider_{ticker}_tab1"
                    if slider_key in st.session_state:
                        del st.session_state[slider_key]
                st.session_state.manual_weights_tab1 = best["Weights"].copy().tolist()
                st.session_state['reset_weights_tab1'] = False
                reset_clicked = True
            
            # Initialize manual weights in session state if not exists or if tickers changed
            if 'manual_weights_tab1' not in st.session_state or len(st.session_state.manual_weights_tab1) != len(tickers):
                st.session_state.manual_weights_tab1 = best["Weights"].copy().tolist()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create sliders for each ticker
                adjusted_weights = []
                total_weight = 0
                
                for i, ticker in enumerate(tickers):
                    slider_key = f"slider_{ticker}_tab1"
                    # Use stored weight or optimized weight as default
                    # If slider key exists, use its value; otherwise use default
                    if slider_key in st.session_state:
                        default_weight = st.session_state[slider_key]
                    else:
                        default_weight = st.session_state.manual_weights_tab1[i] if i < len(st.session_state.manual_weights_tab1) else float(best["Weights"][i])
                    
                    weight = st.slider(
                        f"{ticker} Weight",
                        0.0, 1.0, 
                        default_weight,
                        step=0.01,
                        key=slider_key
                    )
                    adjusted_weights.append(weight)
                    total_weight += weight
                
                # Normalize weights
                if total_weight > 0:
                    adjusted_weights = np.array(adjusted_weights) / total_weight
                else:
                    adjusted_weights = np.array([1.0/len(tickers)] * len(tickers))
                
                # Calculate metrics for adjusted portfolio
                adjusted_portfolio = calculate_portfolio_from_weights(adjusted_weights, mu, Sigma, rf)
                adjusted_returns = (returns @ adjusted_portfolio['weights']).dropna()
                adjusted_metrics = calculate_portfolio_metrics(adjusted_returns, rf)
                
                # Display comparison
                st.markdown("#### 📈 Portfolio Comparison")
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.metric("Expected Return", f"{adjusted_portfolio['return']*100:.2f}%", 
                             delta=f"{adjusted_portfolio['return']*100 - best['Return']*100:+.2f}% vs Optimized")
                
                with comp_col2:
                    st.metric("Volatility", f"{adjusted_portfolio['vol']*100:.2f}%",
                             delta=f"{adjusted_portfolio['vol']*100 - best['Vol']*100:+.2f}% vs Optimized")
                
                with comp_col3:
                    st.metric("Sharpe Ratio", f"{adjusted_portfolio['sharpe']:.2f}",
                             delta=f"{adjusted_portfolio['sharpe'] - best['Sharpe']:+.2f} vs Optimized")
                
                # Visual comparison on efficient frontier
                fig_rebal = go.Figure()
                
                # All portfolios
                fig_rebal.add_trace(go.Scatter(
                    x=df["Vol"]*100,
                    y=df["Return"]*100,
                    mode='markers',
                    marker=dict(size=3, color='lightgray', opacity=0.3),
                    name='All Portfolios'
                ))
                
                # Optimized portfolio
                fig_rebal.add_trace(go.Scatter(
                    x=[best["Vol"]*100],
                    y=[best["Return"]*100],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Optimized Portfolio'
                ))
                
                # Adjusted portfolio
                fig_rebal.add_trace(go.Scatter(
                    x=[adjusted_portfolio['vol']*100],
                    y=[adjusted_portfolio['return']*100],
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='circle'),
                    name='Your Adjusted Portfolio'
                ))
                
                fig_rebal.update_layout(
                    title="Portfolio Comparison on Efficient Frontier",
                    xaxis_title="Volatility (Risk) %",
                    yaxis_title="Expected Annual Return %",
                    height=400,
                    template='plotly_white',
                    showlegend=True
                )
                st.plotly_chart(fig_rebal, use_container_width=True)
            
            with col2:
                st.markdown("#### Current Weights")
                weights_comparison = pd.DataFrame({
                    'Ticker': tickers,
                    'Optimized %': (best["Weights"] * 100).round(2),
                    'Adjusted %': (adjusted_weights * 100).round(2),
                    'Change': ((adjusted_weights - best["Weights"]) * 100).round(2)
                })
                st.dataframe(weights_comparison, hide_index=True, use_container_width=True)
                
                if st.button("Reset to Optimized", key="reset_btn_tab1"):
                    # Set flag to reset sliders on next run
                    st.session_state['reset_weights_tab1'] = True
                    st.rerun()
        
        # Feature 2: Correlation Analysis
        with feature_tabs[1]:
            st.markdown("### 📊 Stock Correlation Analysis")
            st.markdown("Understand how stocks move together. High correlation (red) means stocks tend to move in the same direction.")
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Interactive heatmap
                fig_corr = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_matrix.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size":10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_corr.update_layout(
                    title="Stock Correlation Heatmap",
                    height=500,
                    xaxis_title="",
                    yaxis_title="",
                    template='plotly_white'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.markdown("#### Correlation Insights")
                
                # Find highest and lowest correlations
                corr_upper = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                
                max_corr = corr_upper.max().max()
                min_corr = corr_upper.min().min()
                
                max_pair = [(i, j) for i in correlation_matrix.index 
                           for j in correlation_matrix.columns 
                           if correlation_matrix.loc[i, j] == max_corr][0]
                min_pair = [(i, j) for i in correlation_matrix.index 
                           for j in correlation_matrix.columns 
                           if correlation_matrix.loc[i, j] == min_corr][0]
                
                st.metric("Highest Correlation", f"{max_corr:.3f}", 
                         delta=f"{max_pair[0]} ↔ {max_pair[1]}")
                st.metric("Lowest Correlation", f"{min_corr:.3f}",
                         delta=f"{min_pair[0]} ↔ {min_pair[1]}")
                
                st.markdown("""
                **💡 Interpretation:**
                - **High correlation (>0.7)**: Stocks move together, less diversification benefit
                - **Low correlation (<0.3)**: Stocks move independently, better diversification
                - **Negative correlation**: Stocks move opposite, excellent for risk reduction
                """)
        
        # Feature 3: Monte Carlo Simulation
        with feature_tabs[2]:
            st.markdown("### 🎲 Monte Carlo Future Projection")
            st.markdown("Simulate thousands of possible future scenarios based on historical patterns.")
            
            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                num_simulations = st.slider("Number of Simulations", 500, 5000, 1000, step=500, key="mc_sims_tab1")
            with mc_col2:
                projection_days = st.slider("Projection Period (days)", 30, 504, 252, step=30, key="mc_days_tab1")
            
            if st.button("Run Monte Carlo Simulation", key="run_mc_tab1"):
                with st.spinner("Running Monte Carlo simulation..."):
                    simulated_values = monte_carlo_simulation(returns, best_weights, num_simulations, projection_days)
                    
                    # Calculate percentiles
                    percentiles = [5, 25, 50, 75, 95]
                    percentile_values = np.percentile(simulated_values, percentiles, axis=0)
                    
                    # Create visualization
                    fig_mc = go.Figure()
                    
                    # Add percentile bands
                    colors = ['rgba(255,0,0,0.1)', 'rgba(255,165,0,0.2)', 'rgba(0,255,0,0.3)']
                    labels = ['5th-95th Percentile', '25th-75th Percentile', 'Median']
                    
                    for i, (p_low, p_high) in enumerate([(0, 4), (1, 3), (2, 2)]):
                        if p_low == p_high:
                            fig_mc.add_trace(go.Scatter(
                                x=list(range(projection_days)),
                                y=percentile_values[p_low],
                                mode='lines',
                                name=labels[i],
                                line=dict(color=['red', 'orange', 'green'][i], width=3),
                                fill=None
                            ))
                        else:
                            fig_mc.add_trace(go.Scatter(
                                x=list(range(projection_days)),
                                y=percentile_values[p_low],
                                mode='lines',
                                name=labels[i],
                                line=dict(width=0),
                                showlegend=False
                            ))
                            fig_mc.add_trace(go.Scatter(
                                x=list(range(projection_days)),
                                y=percentile_values[p_high],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=colors[i],
                                name=labels[i]
                            ))
                    
                    # Add some sample paths
                    for i in range(min(10, num_simulations)):
                        fig_mc.add_trace(go.Scatter(
                            x=list(range(projection_days)),
                            y=simulated_values[i],
                            mode='lines',
                            line=dict(color='lightblue', width=1),
                            opacity=0.3,
                            showlegend=(i == 0),
                            name='Sample Paths' if i == 0 else ''
                        ))
                    
                    fig_mc.update_layout(
                        title=f"Monte Carlo Portfolio Projection ({num_simulations:,} simulations)",
                        xaxis_title="Days into Future",
                        yaxis_title="Portfolio Value ($)",
                        height=500,
                        template='plotly_white',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)
                    
                    # Statistics
                    final_values = simulated_values[:, -1]
                    st.markdown("#### 📊 Projection Statistics")
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("Median Value", f"${np.median(final_values):,.2f}")
                    with stat_col2:
                        st.metric("5th Percentile", f"${np.percentile(final_values, 5):,.2f}")
                    with stat_col3:
                        st.metric("95th Percentile", f"${np.percentile(final_values, 95):,.2f}")
                    with stat_col4:
                        prob_positive = (final_values > 10000).sum() / len(final_values) * 100
                        st.metric("Probability of Gain", f"{prob_positive:.1f}%")
        
        # Feature 4: Risk Analysis
        with feature_tabs[3]:
            st.markdown("### ⚠️ Advanced Risk Analysis")
            st.markdown("Deep dive into portfolio risk metrics including Value at Risk (VaR) and Conditional VaR.")
            
            # Calculate risk metrics
            portfolio_returns_pct = portfolio_returns * 100  # Convert to percentage
            risk_metrics = calculate_risk_metrics(portfolio_returns, [0.05, 0.01])
            
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.markdown("#### Value at Risk (VaR)")
                st.markdown("""
                **VaR** measures the maximum expected loss over a given time period at a certain confidence level.
                - **VaR 5%**: Worst case loss expected 5% of the time
                - **VaR 1%**: Worst case loss expected 1% of the time (more extreme)
                """)
                
                var_col1, var_col2 = st.columns(2)
                with var_col1:
                    st.metric("VaR (5%)", f"{risk_metrics['VaR_5']:.2f}%",
                             delta="Daily loss threshold")
                with var_col2:
                    st.metric("VaR (1%)", f"{risk_metrics['VaR_1']:.2f}%",
                             delta="Extreme loss threshold")
            
            with risk_col2:
                st.markdown("#### Conditional VaR (CVaR)")
                st.markdown("""
                **CVaR** (Expected Shortfall) is the average loss when VaR is exceeded.
                It provides a more conservative risk measure than VaR.
                """)
                
                cvar_col1, cvar_col2 = st.columns(2)
                with cvar_col1:
                    st.metric("CVaR (5%)", f"{risk_metrics['CVaR_5']:.2f}%",
                             delta="Average loss beyond VaR")
                with cvar_col2:
                    st.metric("CVaR (1%)", f"{risk_metrics['CVaR_1']:.2f}%",
                             delta="Extreme average loss")
            
            # Risk distribution visualization
            st.markdown("---")
            st.markdown("#### 📉 Return Distribution & Risk Zones")
            
            fig_risk = go.Figure()
            
            # Histogram of returns
            fig_risk.add_trace(go.Histogram(
                x=portfolio_returns_pct,
                nbinsx=50,
                name='Daily Returns',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Add VaR lines
            for var_name, var_value in [('VaR 5%', risk_metrics['VaR_5']), ('VaR 1%', risk_metrics['VaR_1'])]:
                fig_risk.add_vline(
                    x=var_value,
                    line_dash="dash",
                    line_color="red" if '5%' in var_name else "darkred",
                    annotation_text=var_name,
                    annotation_position="top"
                )
            
            fig_risk.update_layout(
                title="Portfolio Return Distribution with VaR Thresholds",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Risk summary table
            risk_summary = pd.DataFrame({
                'Metric': ['Volatility', 'Max Drawdown', 'VaR (5%)', 'CVaR (5%)', 'VaR (1%)', 'CVaR (1%)'],
                'Value': [
                    f"{metrics['annual_vol']:.2f}%",
                    f"{metrics['max_drawdown']:.2f}%",
                    f"{risk_metrics['VaR_5']:.2f}%",
                    f"{risk_metrics['CVaR_5']:.2f}%",
                    f"{risk_metrics['VaR_1']:.2f}%",
                    f"{risk_metrics['CVaR_1']:.2f}%"
                ]
            })
            st.dataframe(risk_summary, hide_index=True, use_container_width=True)
        
        # Feature 5: Export Results
        with feature_tabs[4]:
            st.markdown("### 📥 Export Portfolio Results")
            st.markdown("Download your optimization results and analysis data.")
            
            # Prepare data for export
            export_data = {
                'Portfolio Weights': pd.DataFrame({
                    'Ticker': tickers,
                    'Weight': best["Weights"],
                    'Weight %': (best["Weights"] * 100).round(2)
                }),
                'Portfolio Metrics': pd.DataFrame([{
                    'Total Return (%)': metrics['total_return'],
                    'Annual Return (%)': metrics['annual_return'],
                    'Volatility (%)': metrics['annual_vol'],
                    'Sharpe Ratio': best['Sharpe'],
                    'Sortino Ratio': metrics['sortino'],
                    'Max Drawdown (%)': metrics['max_drawdown'],
                    'Win Rate (%)': metrics['win_rate']
                }]),
                'All Simulated Portfolios': df[['Sharpe', 'Return', 'Vol']].copy()
            }
            
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            
            with exp_col1:
                # Export weights
                csv_weights = export_data['Portfolio Weights'].to_csv(index=False)
                st.download_button(
                    label="📊 Download Weights (CSV)",
                    data=csv_weights,
                    file_name=f"portfolio_weights_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_weights_tab1"
                )
            
            with exp_col2:
                # Export metrics
                csv_metrics = export_data['Portfolio Metrics'].to_csv(index=False)
                st.download_button(
                    label="📈 Download Metrics (CSV)",
                    data=csv_metrics,
                    file_name=f"portfolio_metrics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_metrics_tab1"
                )
            
            with exp_col3:
                # Export all portfolios
                csv_all = export_data['All Simulated Portfolios'].to_csv(index=False)
                st.download_button(
                    label="🎯 Download All Portfolios (CSV)",
                    data=csv_all,
                    file_name=f"all_portfolios_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_all_tab1"
                )
            
            st.markdown("---")
            st.markdown("#### 📋 Export Summary")
            st.dataframe(export_data['Portfolio Metrics'], hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: Indian NSE Portfolio Optimization
# ---------------------------------------------------------------------
with tab2:
    st.header("🇮🇳 NSE Portfolio Optimization")
    
    st.markdown("""
    <div class="info-box">
        <p style="font-size: 1rem; line-height: 1.6; margin: 0;">
            This tab uses <strong>NSE (National Stock Exchange) India</strong> data to optimize portfolios.<br>
            Enter NSE ticker symbols <strong>without the .NS suffix</strong> (e.g., RELIANCE, TCS, INFY).<br>
            The app will automatically add <code>.NS</code> for Yahoo Finance API.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameter selection in styled container
    st.markdown("""
    <div class="param-container">
        <div class="section-header">⚙️ Optimization Parameters</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickers_input_nse = st.text_input("Enter NSE stock tickers (comma-separated):", "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK", key="tickers_nse")
        num_portfolios_nse = st.number_input("Number of portfolios to simulate:", 1000, 1000000, 10000, step=1000, key="num_portfolios_nse")
        num_workers_nse = st.slider("Number of parallel workers:", 1, 8, 4, key="workers_nse")
    
    with col2:
        metric_nse = st.selectbox("Optimization Metric:", ["Max Sharpe", "Max Return", "Min Risk"], key="metric_nse")
        rf_nse = st.number_input("Risk-free rate (annual):", 0.0, 0.1, 0.06, step=0.005, key="rf_nse")
    
    with col3:
        start_date_nse = st.date_input("Start date:", pd.to_datetime("2020-01-01"), key="start_nse")
        end_date_nse = st.date_input("End date:", pd.to_datetime("2024-01-01"), key="end_nse")
    
    st.markdown('</div>', unsafe_allow_html=True)

    tickers_nse = [t.strip().upper() for t in tickers_input_nse.split(",") if t.strip()]

    if st.button("🚀 Run NSE Optimization", key="run_nse"):
        with st.spinner("Fetching NSE data and running optimization..."):
            try:
                data_nse, returns_nse, mu_nse, Sigma_nse = fetch_nse_data(tickers_nse, start_date_nse, end_date_nse)
                df_nse, total_time_nse = run_parallel_optimization(mu_nse, Sigma_nse, rf_nse, num_portfolios_nse, num_workers_nse)
                
                # Update tickers_nse to only include successfully fetched ones
                tickers_nse = list(data_nse.columns)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()

        # --- Select best portfolio ---
        if metric_nse == "Max Sharpe":
            best_nse = df_nse.loc[df_nse["Sharpe"].idxmax()]
        elif metric_nse == "Max Return":
            best_nse = df_nse.loc[df_nse["Return"].idxmax()]
        else:
            best_nse = df_nse.loc[df_nse["Vol"].idxmin()]

        # Calculate portfolio metrics
        best_weights_nse = np.array(best_nse["Weights"])
        portfolio_returns_nse = (returns_nse @ best_weights_nse).dropna()
        metrics_nse = calculate_portfolio_metrics(portfolio_returns_nse, rf_nse)
        
        # Store results in session state for persistence
        st.session_state['tab2_data'] = data_nse
        st.session_state['tab2_returns'] = returns_nse
        st.session_state['tab2_mu'] = mu_nse
        st.session_state['tab2_Sigma'] = Sigma_nse
        st.session_state['tab2_df'] = df_nse
        st.session_state['tab2_total_time'] = total_time_nse
        st.session_state['tab2_best'] = best_nse
        st.session_state['tab2_best_weights'] = best_weights_nse
        st.session_state['tab2_portfolio_returns'] = portfolio_returns_nse
        st.session_state['tab2_metrics'] = metrics_nse
        st.session_state['tab2_tickers'] = tickers_nse
        st.session_state['tab2_rf'] = rf_nse
        st.session_state['tab2_metric'] = metric_nse

        st.success(f"✅ Optimization completed in **{total_time_nse:.2f}s** using **{num_workers_nse} workers**")

    # Check if we have stored results to display
    if 'tab2_data' in st.session_state:
        # Retrieve from session state
        data_nse = st.session_state['tab2_data']
        returns_nse = st.session_state['tab2_returns']
        mu_nse = st.session_state['tab2_mu']
        Sigma_nse = st.session_state['tab2_Sigma']
        df_nse = st.session_state['tab2_df']
        total_time_nse = st.session_state['tab2_total_time']
        best_nse = st.session_state['tab2_best']
        best_weights_nse = st.session_state['tab2_best_weights']
        portfolio_returns_nse = st.session_state['tab2_portfolio_returns']
        metrics_nse = st.session_state['tab2_metrics']
        tickers_nse = st.session_state['tab2_tickers']
        rf_nse = st.session_state['tab2_rf']
        metric_nse = st.session_state['tab2_metric']

        # --- Key Metrics Row ---
        st.markdown("---")
        st.subheader("🏆 Optimized Portfolio Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            profit_color = "normal" if metrics_nse['total_return'] >= 0 else "inverse"
            st.metric(
                "Total Return", 
                f"{metrics_nse['total_return']:.2f}%",
                delta=f"{metrics_nse['annual_return']:.2f}% annual",
                delta_color=profit_color
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio", 
                f"{best_nse['Sharpe']:.2f}",
                delta=f"Sortino: {metrics_nse['sortino']:.2f}"
            )
        
        with col3:
            st.metric(
                "Volatility (Risk)", 
                f"{best_nse['Vol']*100:.2f}%",
                delta=f"Max DD: {metrics_nse['max_drawdown']:.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Win Rate", 
                f"{metrics_nse['win_rate']:.1f}%",
                delta=f"{int((portfolio_returns_nse > 0).sum())} winning days"
            )

        # --- Interactive Efficient Frontier ---
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Efficient Frontier")
            fig_nse = go.Figure()
            
            # All portfolios
            fig_nse.add_trace(go.Scatter(
                x=df_nse["Vol"]*100,
                y=df_nse["Return"]*100,
                mode='markers',
                marker=dict(
                    size=4,
                    color=df_nse["Sharpe"],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                    line=dict(width=0)
                ),
                text=[f"Return: {r*100:.2f}%<br>Risk: {v*100:.2f}%<br>Sharpe: {s:.2f}" 
                      for r, v, s in zip(df_nse["Return"], df_nse["Vol"], df_nse["Sharpe"])],
                hovertemplate='%{text}<extra></extra>',
                name='Portfolios'
            ))
            
            # Best portfolio
            fig_nse.add_trace(go.Scatter(
                x=[best_nse["Vol"]*100],
                y=[best_nse["Return"]*100],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
                name='Best Portfolio',
                hovertemplate=f'<b>Best Portfolio</b><br>Return: {best_nse["Return"]*100:.2f}%<br>Risk: {best_nse["Vol"]*100:.2f}%<br>Sharpe: {best_nse["Sharpe"]:.2f}<extra></extra>'
            ))
            
            fig_nse.update_layout(
                xaxis_title="Volatility (Risk) %",
                yaxis_title="Expected Annual Return %",
                height=500,
                hovermode='closest',
                showlegend=True,
                template='plotly_white'
            )
            st.plotly_chart(fig_nse, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Portfolio Weights")
            weights_df_nse = pd.DataFrame({"Ticker": tickers_nse, "Weight": best_nse["Weights"]})
            weights_df_nse = weights_df_nse.sort_values("Weight", ascending=False)
            
            # Pie chart
            fig_pie_nse = go.Figure(data=[go.Pie(
                labels=weights_df_nse["Ticker"],
                values=weights_df_nse["Weight"],
                hole=0.4,
                marker=dict(colors=px.colors.qualitative.Set2),
                textinfo='label+percent',
                textposition='auto'
            )])
            fig_pie_nse.update_layout(
                height=400,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie_nse, use_container_width=True)
            
            # Weight table
            weights_df_nse["Weight %"] = (weights_df_nse["Weight"] * 100).round(2)
            st.dataframe(
                weights_df_nse[["Ticker", "Weight %"]],
                hide_index=True,
                use_container_width=True
            )

        # --- Backtest Performance ---
        st.markdown("---")
        st.subheader("💰 Historical Backtest Performance")
        
        # Create subplot with cumulative returns and drawdown
        fig_backtest_nse = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Portfolio Value Growth (₹1,00,000 Initial)', 'Drawdown %'),
            row_heights=[0.7, 0.3]
        )
        
        # Cumulative returns
        portfolio_value_nse = metrics_nse['cum_returns'] * 100000
        fig_backtest_nse.add_trace(
            go.Scatter(
                x=portfolio_value_nse.index,
                y=portfolio_value_nse,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#FF6B35', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,107,53,0.2)',
                hovertemplate='Date: %{x}<br>Value: ₹%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig_backtest_nse.add_trace(
            go.Scatter(
                x=metrics_nse['drawdowns'].index,
                y=metrics_nse['drawdowns'],
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_backtest_nse.update_xaxes(title_text="Date", row=2, col=1)
        fig_backtest_nse.update_yaxes(title_text="Value (₹)", row=1, col=1)
        fig_backtest_nse.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        fig_backtest_nse.update_layout(
            height=600,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_backtest_nse, use_container_width=True)
        
        # Performance summary
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_value = 100000
            final_value = portfolio_value_nse.iloc[-1]
            profit = final_value - initial_value
            st.markdown(f"""
            **💵 Investment Growth**
            - Initial: ₹{initial_value:,.0f}
            - Final: ₹{final_value:,.2f}
            - Profit: ₹{profit:,.2f}
            """)
        
        with col2:
            best_day = portfolio_returns_nse.max() * 100
            worst_day = portfolio_returns_nse.min() * 100
            st.markdown(f"""
            **📊 Daily Performance**
            - Best Day: +{best_day:.2f}%
            - Worst Day: {worst_day:.2f}%
            - Avg Daily: {portfolio_returns_nse.mean()*100:.3f}%
            """)
        
        with col3:
            years = len(portfolio_returns_nse) / 252
            cagr = ((metrics_nse['cum_returns'].iloc[-1]) ** (1/years) - 1) * 100 if years > 0 else 0
            st.markdown(f"""
            **📈 Long-term Stats**
            - CAGR: {cagr:.2f}%
            - Period: {years:.1f} years
            - Risk-Adj Return: {metrics_nse['sharpe']:.2f}
            """)

        st.info("💡 **Tip:** Popular NSE tickers include RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, BHARTIARTL, ITC, LT, etc.")

        # =====================================================================
        # NEW INTERACTIVE FEATURES FOR NSE
        # =====================================================================
        
        st.markdown("---")
        st.subheader("🎮 Interactive Features")
        
        feature_tabs_nse = st.tabs(["🔄 Portfolio Rebalancing", "📊 Correlation Analysis", "🎲 Monte Carlo Simulation", "⚠️ Risk Analysis", "📥 Export Results"])
        
        # Feature 1: Interactive Portfolio Rebalancing (NSE)
        with feature_tabs_nse[0]:
            st.markdown("### 🔄 Interactive Portfolio Rebalancing")
            st.markdown("Manually adjust portfolio weights using sliders and see real-time impact on metrics.")
            
            # Check if reset button was clicked (before creating sliders)
            if 'reset_weights_nse' in st.session_state and st.session_state.get('reset_weights_nse', False):
                # Delete all slider keys to reset them
                for ticker in tickers_nse:
                    slider_key = f"slider_{ticker}_nse"
                    if slider_key in st.session_state:
                        del st.session_state[slider_key]
                st.session_state.manual_weights_nse = best_nse["Weights"].copy().tolist()
                st.session_state['reset_weights_nse'] = False
            
            # Initialize manual weights in session state if not exists or if tickers changed
            if 'manual_weights_nse' not in st.session_state or len(st.session_state.manual_weights_nse) != len(tickers_nse):
                st.session_state.manual_weights_nse = best_nse["Weights"].copy().tolist()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                adjusted_weights_nse = []
                total_weight_nse = 0
                
                for i, ticker in enumerate(tickers_nse):
                    slider_key = f"slider_{ticker}_nse"
                    # Use stored weight or optimized weight as default
                    # If slider key exists, use its value; otherwise use default
                    if slider_key in st.session_state:
                        default_weight = st.session_state[slider_key]
                    else:
                        default_weight = st.session_state.manual_weights_nse[i] if i < len(st.session_state.manual_weights_nse) else float(best_nse["Weights"][i])
                    
                    weight = st.slider(
                        f"{ticker} Weight",
                        0.0, 1.0, 
                        default_weight,
                        step=0.01,
                        key=slider_key
                    )
                    adjusted_weights_nse.append(weight)
                    total_weight_nse += weight
                
                if total_weight_nse > 0:
                    adjusted_weights_nse = np.array(adjusted_weights_nse) / total_weight_nse
                else:
                    adjusted_weights_nse = np.array([1.0/len(tickers_nse)] * len(tickers_nse))
                
                adjusted_portfolio_nse = calculate_portfolio_from_weights(adjusted_weights_nse, mu_nse, Sigma_nse, rf_nse)
                adjusted_returns_nse = (returns_nse @ adjusted_portfolio_nse['weights']).dropna()
                
                st.markdown("#### 📈 Portfolio Comparison")
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.metric("Expected Return", f"{adjusted_portfolio_nse['return']*100:.2f}%", 
                             delta=f"{adjusted_portfolio_nse['return']*100 - best_nse['Return']*100:+.2f}% vs Optimized")
                
                with comp_col2:
                    st.metric("Volatility", f"{adjusted_portfolio_nse['vol']*100:.2f}%",
                             delta=f"{adjusted_portfolio_nse['vol']*100 - best_nse['Vol']*100:+.2f}% vs Optimized")
                
                with comp_col3:
                    st.metric("Sharpe Ratio", f"{adjusted_portfolio_nse['sharpe']:.2f}",
                             delta=f"{adjusted_portfolio_nse['sharpe'] - best_nse['Sharpe']:+.2f} vs Optimized")
                
                fig_rebal_nse = go.Figure()
                fig_rebal_nse.add_trace(go.Scatter(
                    x=df_nse["Vol"]*100,
                    y=df_nse["Return"]*100,
                    mode='markers',
                    marker=dict(size=3, color='lightgray', opacity=0.3),
                    name='All Portfolios'
                ))
                fig_rebal_nse.add_trace(go.Scatter(
                    x=[best_nse["Vol"]*100],
                    y=[best_nse["Return"]*100],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Optimized Portfolio'
                ))
                fig_rebal_nse.add_trace(go.Scatter(
                    x=[adjusted_portfolio_nse['vol']*100],
                    y=[adjusted_portfolio_nse['return']*100],
                    mode='markers',
                    marker=dict(size=15, color='blue', symbol='circle'),
                    name='Your Adjusted Portfolio'
                ))
                fig_rebal_nse.update_layout(
                    title="Portfolio Comparison on Efficient Frontier",
                    xaxis_title="Volatility (Risk) %",
                    yaxis_title="Expected Annual Return %",
                    height=400,
                    template='plotly_white',
                    showlegend=True
                )
                st.plotly_chart(fig_rebal_nse, use_container_width=True)
            
            with col2:
                st.markdown("#### Current Weights")
                weights_comparison_nse = pd.DataFrame({
                    'Ticker': tickers_nse,
                    'Optimized %': (best_nse["Weights"] * 100).round(2),
                    'Adjusted %': (adjusted_weights_nse * 100).round(2),
                    'Change': ((adjusted_weights_nse - best_nse["Weights"]) * 100).round(2)
                })
                st.dataframe(weights_comparison_nse, hide_index=True, use_container_width=True)
                
                if st.button("Reset to Optimized", key="reset_btn_nse"):
                    # Set flag to reset sliders on next run
                    st.session_state['reset_weights_nse'] = True
                    st.rerun()
        
        # Feature 2: Correlation Analysis (NSE)
        with feature_tabs_nse[1]:
            st.markdown("### 📊 Stock Correlation Analysis")
            correlation_matrix_nse = returns_nse.corr()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_corr_nse = go.Figure(data=go.Heatmap(
                    z=correlation_matrix_nse.values,
                    x=correlation_matrix_nse.columns,
                    y=correlation_matrix_nse.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_matrix_nse.values.round(2),
                    texttemplate='%{text}',
                    textfont={"size":10},
                    colorbar=dict(title="Correlation")
                ))
                fig_corr_nse.update_layout(
                    title="Stock Correlation Heatmap",
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig_corr_nse, use_container_width=True)
            
            with col2:
                st.markdown("#### Correlation Insights")
                corr_upper_nse = correlation_matrix_nse.where(
                    np.triu(np.ones(correlation_matrix_nse.shape), k=1).astype(bool)
                )
                max_corr_nse = corr_upper_nse.max().max()
                min_corr_nse = corr_upper_nse.min().min()
                max_pair_nse = [(i, j) for i in correlation_matrix_nse.index 
                               for j in correlation_matrix_nse.columns 
                               if correlation_matrix_nse.loc[i, j] == max_corr_nse][0]
                min_pair_nse = [(i, j) for i in correlation_matrix_nse.index 
                               for j in correlation_matrix_nse.columns 
                               if correlation_matrix_nse.loc[i, j] == min_corr_nse][0]
                
                st.metric("Highest Correlation", f"{max_corr_nse:.3f}", 
                         delta=f"{max_pair_nse[0]} ↔ {max_pair_nse[1]}")
                st.metric("Lowest Correlation", f"{min_corr_nse:.3f}",
                         delta=f"{min_pair_nse[0]} ↔ {min_pair_nse[1]}")
        
        # Feature 3: Monte Carlo Simulation (NSE)
        with feature_tabs_nse[2]:
            st.markdown("### 🎲 Monte Carlo Future Projection")
            mc_col1, mc_col2 = st.columns(2)
            with mc_col1:
                num_simulations_nse = st.slider("Number of Simulations", 500, 5000, 1000, step=500, key="mc_sims_nse")
            with mc_col2:
                projection_days_nse = st.slider("Projection Period (days)", 30, 504, 252, step=30, key="mc_days_nse")
            
            if st.button("Run Monte Carlo Simulation", key="run_mc_nse"):
                with st.spinner("Running Monte Carlo simulation..."):
                    # Use ₹1,00,000 as initial value for NSE (matching the backtest)
                    simulated_values_nse = monte_carlo_simulation(returns_nse, best_weights_nse, num_simulations_nse, projection_days_nse, initial_value=100000)
                    percentiles_nse = [5, 25, 50, 75, 95]
                    percentile_values_nse = np.percentile(simulated_values_nse, percentiles_nse, axis=0)
                    
                    fig_mc_nse = go.Figure()
                    colors_nse = ['rgba(255,0,0,0.1)', 'rgba(255,165,0,0.2)', 'rgba(0,255,0,0.3)']
                    labels_nse = ['5th-95th Percentile', '25th-75th Percentile', 'Median']
                    
                    for i, (p_low, p_high) in enumerate([(0, 4), (1, 3), (2, 2)]):
                        if p_low == p_high:
                            fig_mc_nse.add_trace(go.Scatter(
                                x=list(range(projection_days_nse)),
                                y=percentile_values_nse[p_low],
                                mode='lines',
                                name=labels_nse[i],
                                line=dict(color=['red', 'orange', 'green'][i], width=3)
                            ))
                        else:
                            fig_mc_nse.add_trace(go.Scatter(
                                x=list(range(projection_days_nse)),
                                y=percentile_values_nse[p_low],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            fig_mc_nse.add_trace(go.Scatter(
                                x=list(range(projection_days_nse)),
                                y=percentile_values_nse[p_high],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=colors_nse[i],
                                name=labels_nse[i]
                            ))
                    
                    for i in range(min(10, num_simulations_nse)):
                        fig_mc_nse.add_trace(go.Scatter(
                            x=list(range(projection_days_nse)),
                            y=simulated_values_nse[i],
                            mode='lines',
                            line=dict(color='lightblue', width=1),
                            opacity=0.3,
                            showlegend=(i == 0),
                            name='Sample Paths' if i == 0 else ''
                        ))
                    
                    fig_mc_nse.update_layout(
                        title=f"Monte Carlo Portfolio Projection ({num_simulations_nse:,} simulations)",
                        xaxis_title="Days into Future",
                        yaxis_title="Portfolio Value (₹)",
                        height=500,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_mc_nse, use_container_width=True)
                    
                    final_values_nse = simulated_values_nse[:, -1]
                    st.markdown("#### 📊 Projection Statistics")
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.metric("Median Value", f"₹{np.median(final_values_nse):,.2f}")
                    with stat_col2:
                        st.metric("5th Percentile", f"₹{np.percentile(final_values_nse, 5):,.2f}")
                    with stat_col3:
                        st.metric("95th Percentile", f"₹{np.percentile(final_values_nse, 95):,.2f}")
                    with stat_col4:
                        prob_positive_nse = (final_values_nse > 100000).sum() / len(final_values_nse) * 100
                        st.metric("Probability of Gain", f"{prob_positive_nse:.1f}%")
        
        # Feature 4: Risk Analysis (NSE)
        with feature_tabs_nse[3]:
            st.markdown("### ⚠️ Advanced Risk Analysis")
            portfolio_returns_pct_nse = portfolio_returns_nse * 100
            risk_metrics_nse = calculate_risk_metrics(portfolio_returns_nse, [0.05, 0.01])
            
            risk_col1, risk_col2 = st.columns(2)
            
            with risk_col1:
                st.markdown("#### Value at Risk (VaR)")
                var_col1, var_col2 = st.columns(2)
                with var_col1:
                    st.metric("VaR (5%)", f"{risk_metrics_nse['VaR_5']:.2f}%")
                with var_col2:
                    st.metric("VaR (1%)", f"{risk_metrics_nse['VaR_1']:.2f}%")
            
            with risk_col2:
                st.markdown("#### Conditional VaR (CVaR)")
                cvar_col1, cvar_col2 = st.columns(2)
                with cvar_col1:
                    st.metric("CVaR (5%)", f"{risk_metrics_nse['CVaR_5']:.2f}%")
                with cvar_col2:
                    st.metric("CVaR (1%)", f"{risk_metrics_nse['CVaR_1']:.2f}%")
            
            fig_risk_nse = go.Figure()
            fig_risk_nse.add_trace(go.Histogram(
                x=portfolio_returns_pct_nse,
                nbinsx=50,
                marker_color='lightblue',
                opacity=0.7
            ))
            
            for var_name, var_value in [('VaR 5%', risk_metrics_nse['VaR_5']), ('VaR 1%', risk_metrics_nse['VaR_1'])]:
                fig_risk_nse.add_vline(
                    x=var_value,
                    line_dash="dash",
                    line_color="red" if '5%' in var_name else "darkred",
                    annotation_text=var_name
                )
            
            fig_risk_nse.update_layout(
                title="Portfolio Return Distribution with VaR Thresholds",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_risk_nse, use_container_width=True)
        
        # Feature 5: Export Results (NSE)
        with feature_tabs_nse[4]:
            st.markdown("### 📥 Export Portfolio Results")
            export_data_nse = {
                'Portfolio Weights': pd.DataFrame({
                    'Ticker': tickers_nse,
                    'Weight': best_nse["Weights"],
                    'Weight %': (best_nse["Weights"] * 100).round(2)
                }),
                'Portfolio Metrics': pd.DataFrame([{
                    'Total Return (%)': metrics_nse['total_return'],
                    'Annual Return (%)': metrics_nse['annual_return'],
                    'Volatility (%)': metrics_nse['annual_vol'],
                    'Sharpe Ratio': best_nse['Sharpe'],
                    'Sortino Ratio': metrics_nse['sortino'],
                    'Max Drawdown (%)': metrics_nse['max_drawdown'],
                    'Win Rate (%)': metrics_nse['win_rate']
                }])
            }
            
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                csv_weights_nse = export_data_nse['Portfolio Weights'].to_csv(index=False)
                st.download_button(
                    label="📊 Download Weights (CSV)",
                    data=csv_weights_nse,
                    file_name=f"nse_portfolio_weights_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_weights_nse"
                )
            with exp_col2:
                csv_metrics_nse = export_data_nse['Portfolio Metrics'].to_csv(index=False)
                st.download_button(
                    label="📈 Download Metrics (CSV)",
                    data=csv_metrics_nse,
                    file_name=f"nse_portfolio_metrics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_metrics_nse"
                )

# ---------------------------------------------------------------------
# TAB 3: Parallel Speedup Benchmark
# ---------------------------------------------------------------------
with tab3:
    st.header("⚙️ Parallel Speedup Benchmark")

    st.markdown("""
    <div class="info-box">
        <p style="font-size: 1rem; line-height: 1.6; margin: 0;">
            This tab measures <strong>how parallelization affects runtime performance</strong>.<br>
            The same optimization task is run with different numbers of workers (1, 2, 4, 8),<br>
            and the runtime is plotted to show speedup and efficiency.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameter selection in styled container
    st.markdown("""
    <div class="param-container">
        <div class="section-header">⚙️ Benchmark Parameters</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickers_input = st.text_input("Tickers (comma-separated):", "AAPL, MSFT, GOOG, AMZN", key="tickers_tab2")
        num_portfolios = st.number_input("Number of portfolios per test:", 5000, 200000, 20000, step=5000, key="num_portfolios_tab2")
    
    with col2:
        rf = st.number_input("Risk-free rate (annual):", 0.0, 0.1, 0.02, step=0.005, key="rf_tab2")
    
    with col3:
        start_date = st.date_input("Start Date:", pd.to_datetime("2020-01-01"), key="start_tab2")
        end_date = st.date_input("End Date:", pd.to_datetime("2024-01-01"), key="end_tab2")
    
    st.markdown('</div>', unsafe_allow_html=True)

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if st.button("🧪 Run Speedup Test", key="run_tab2"):
        with st.spinner("Fetching data and running benchmarks..."):
            data, returns, mu, Sigma = fetch_data(tickers, start_date, end_date)

            worker_counts = [1, 2, 4, 8]
            runtimes = []

            for w in worker_counts:
                st.write(f"Running with {w} workers...")
                _, ttime = run_parallel_optimization(mu, Sigma, rf, num_portfolios, w)
                runtimes.append(ttime)

        # Compute speedup and efficiency
        base_time = runtimes[0]
        speedup = [base_time / t for t in runtimes]
        efficiency = [s / w for s, w in zip(speedup, worker_counts)]
        ideal_speedup = worker_counts  # Linear speedup

        results_df = pd.DataFrame({
            "Workers": worker_counts,
            "Runtime (s)": np.round(runtimes, 2),
            "Speedup": np.round(speedup, 2),
            "Efficiency": np.round(efficiency, 2)
        })

        st.markdown("---")
        st.subheader("📊 Benchmark Results")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sequential Time", f"{base_time:.2f}s", delta="Baseline")
        with col2:
            st.metric("8-Worker Time", f"{runtimes[-1]:.2f}s", delta=f"-{base_time-runtimes[-1]:.2f}s", delta_color="normal")
        with col3:
            st.metric("Max Speedup", f"{max(speedup):.2f}x", delta=f"{(max(speedup)/8)*100:.1f}% efficient")
        with col4:
            parallel_efficiency = (speedup[-1] / worker_counts[-1]) * 100
            st.metric("Parallel Efficiency", f"{parallel_efficiency:.1f}%", delta="at 8 workers")
        
        st.markdown("---")
        
        # Results table
        st.dataframe(
            results_df.style.background_gradient(subset=['Speedup'], cmap='Greens')
                           .background_gradient(subset=['Runtime (s)'], cmap='Reds_r')
                           .format({'Runtime (s)': '{:.2f}', 'Speedup': '{:.2f}', 'Efficiency': '{:.2f}'}),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Interactive Plotly charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Speedup chart
            fig_speedup = go.Figure()
            
            # Ideal linear speedup
            fig_speedup.add_trace(go.Scatter(
                x=worker_counts,
                y=ideal_speedup,
                mode='lines',
                name='Ideal Linear Speedup',
                line=dict(color='lightgray', width=2, dash='dash'),
                hovertemplate='Workers: %{x}<br>Ideal Speedup: %{y:.2f}x<extra></extra>'
            ))
            
            # Actual speedup
            fig_speedup.add_trace(go.Scatter(
                x=worker_counts,
                y=speedup,
                mode='lines+markers',
                name='Actual Speedup',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=10, color='#2E86AB', line=dict(width=2, color='white')),
                hovertemplate='Workers: %{x}<br>Speedup: %{y:.2f}x<extra></extra>'
            ))
            
            fig_speedup.update_layout(
                title="Speedup vs. Number of Workers",
                xaxis_title="Number of Workers",
                yaxis_title="Speedup Factor",
                height=400,
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_speedup, use_container_width=True)
        
        with col2:
            # Efficiency chart
            fig_efficiency = go.Figure()
            
            fig_efficiency.add_trace(go.Scatter(
                x=worker_counts,
                y=efficiency,
                mode='lines+markers',
                name='Efficiency',
                line=dict(color='#F18F01', width=3),
                marker=dict(size=10, color='#F18F01', line=dict(width=2, color='white')),
                fill='tozeroy',
                fillcolor='rgba(241,143,1,0.2)',
                hovertemplate='Workers: %{x}<br>Efficiency: %{y:.2f}<extra></extra>'
            ))
            
            # Add 100% efficiency line
            fig_efficiency.add_hline(
                y=1.0, 
                line_dash="dash", 
                line_color="green",
                annotation_text="Perfect Efficiency",
                annotation_position="right"
            )
            
            fig_efficiency.update_layout(
                title="Parallel Efficiency vs. Number of Workers",
                xaxis_title="Number of Workers",
                yaxis_title="Efficiency (Speedup/Workers)",
                height=400,
                hovermode='x unified',
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        # Runtime comparison
        st.markdown("---")
        st.subheader("⏱️ Runtime Comparison")
        
        fig_runtime = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        fig_runtime.add_trace(go.Bar(
            x=[f"{w} Worker{'s' if w > 1 else ''}" for w in worker_counts],
            y=runtimes,
            text=[f"{t:.2f}s" for t in runtimes],
            textposition='auto',
            marker=dict(
                color=runtimes,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Time (s)")
            ),
            hovertemplate='%{x}<br>Runtime: %{y:.2f}s<extra></extra>'
        ))
        
        fig_runtime.update_layout(
            title=f"Runtime Comparison ({num_portfolios:,} portfolios)",
            xaxis_title="Configuration",
            yaxis_title="Runtime (seconds)",
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_runtime, use_container_width=True)

        # Analysis
        st.markdown("---")
        st.subheader("📈 Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **⚡ Speedup Analysis**
            - Maximum speedup achieved: **{max(speedup):.2f}x** (with {worker_counts[speedup.index(max(speedup))]} workers)
            - Time saved with 8 workers: **{base_time - runtimes[-1]:.2f}s** ({((base_time - runtimes[-1])/base_time)*100:.1f}% reduction)
            - Speedup efficiency: **{(speedup[-1]/worker_counts[-1])*100:.1f}%** of ideal
            
            **📊 Amdahl's Law Implications**
            - The parallel portion of the code shows good scalability
            - Some overhead exists due to process creation and data serialization
            - Diminishing returns observed as worker count increases
            """)
        
        with col2:
            st.markdown(f"""
            **🎯 Optimization Insights**
            - Best performance/efficiency tradeoff: **{worker_counts[1]}-{worker_counts[2]} workers**
            - Sequential overhead: **{((1 - efficiency[-1]) * 100):.1f}%**
            - Parallel efficiency at 8 workers: **{efficiency[-1]*100:.1f}%**
            
            **💡 Recommendations**
            - For quick iterations: Use 2-4 workers
            - For maximum throughput: Use all available cores
            - Consider I/O vs CPU bound operations when scaling
            """)

        st.success("✅ Benchmark completed! Use these insights in your report to demonstrate parallel computing concepts.")

# ---------------------------------------------------------------------
# TAB 4: Portfolio Comparison
# ---------------------------------------------------------------------
with tab4:
    st.header("🔀 Portfolio Comparison Tool")
    
    st.markdown("""
    <div class="info-box">
        <p style="font-size: 1rem; line-height: 1.6; margin: 0;">
            Compare different optimization strategies side-by-side to understand trade-offs.<br>
            Run multiple optimizations with different metrics and see how they perform.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Parameter selection in styled container
    st.markdown("""
    <div class="param-container">
        <div class="section-header">⚙️ Comparison Parameters</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tickers_comp = st.text_input("Enter stock tickers (comma-separated):", "AAPL, MSFT, GOOG, AMZN, META", key="tickers_comp")
        num_portfolios_comp = st.number_input("Number of portfolios to simulate:", 1000, 100000, 10000, step=1000, key="num_portfolios_comp")
        num_workers_comp = st.slider("Number of parallel workers:", 1, 8, 4, key="workers_comp")
    
    with col2:
        rf_comp = st.number_input("Risk-free rate (annual):", 0.0, 0.1, 0.02, step=0.005, key="rf_comp")
    
    with col3:
        start_date_comp = st.date_input("Start date:", pd.to_datetime("2018-01-01"), key="start_comp")
        end_date_comp = st.date_input("End date:", pd.to_datetime("2024-01-01"), key="end_comp")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    tickers_comp_list = [t.strip().upper() for t in tickers_comp.split(",") if t.strip()]
    
    if st.button("🚀 Compare All Strategies", key="run_comp"):
        with st.spinner("Running optimizations for all strategies..."):
            data_comp, returns_comp, mu_comp, Sigma_comp = fetch_data(tickers_comp_list, start_date_comp, end_date_comp)
            df_comp, total_time_comp = run_parallel_optimization(mu_comp, Sigma_comp, rf_comp, num_portfolios_comp, num_workers_comp)
            
            # Find best portfolios for each metric
            strategies = {
                'Max Sharpe': df_comp.loc[df_comp["Sharpe"].idxmax()],
                'Max Return': df_comp.loc[df_comp["Return"].idxmax()],
                'Min Risk': df_comp.loc[df_comp["Vol"].idxmin()]
            }
            
            # Calculate metrics for each strategy
            comparison_data = []
            for strategy_name, best_port in strategies.items():
                weights_comp = np.array(best_port["Weights"])
                portfolio_returns_comp = (returns_comp @ weights_comp).dropna()
                metrics_comp = calculate_portfolio_metrics(portfolio_returns_comp, rf_comp)
                
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Return (%)': best_port["Return"] * 100,
                    'Risk (%)': best_port["Vol"] * 100,
                    'Sharpe': best_port["Sharpe"],
                    'Sortino': metrics_comp['sortino'],
                    'Max Drawdown (%)': metrics_comp['max_drawdown'],
                    'Win Rate (%)': metrics_comp['win_rate'],
                    'Weights': best_port["Weights"]
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Store data in session state for persistence
            st.session_state['comp_df'] = df_comp
            st.session_state['comp_returns'] = returns_comp
            st.session_state['comp_mu'] = mu_comp
            st.session_state['comp_Sigma'] = Sigma_comp
            st.session_state['comp_rf'] = rf_comp
            st.session_state['comp_strategies'] = strategies
            st.session_state['comp_comparison_df'] = comparison_df
            st.session_state['comp_tickers'] = tickers_comp_list
            
            st.success(f"✅ Comparison completed in **{total_time_comp:.2f}s**")

    # Check if we have stored results to display (for persistence after reruns)
    if 'comp_df' in st.session_state:
        # Retrieve from session state
        df_comp = st.session_state['comp_df']
        returns_comp = st.session_state['comp_returns']
        mu_comp = st.session_state['comp_mu']
        Sigma_comp = st.session_state['comp_Sigma']
        rf_comp = st.session_state['comp_rf']
        strategies = st.session_state['comp_strategies']
        comparison_df = st.session_state['comp_comparison_df']
        tickers_comp_list = st.session_state['comp_tickers']
        
        # Display comparison table
        st.markdown("---")
        st.subheader("📊 Strategy Comparison")
        
        # Metrics comparison
        metrics_comparison = comparison_df[['Strategy', 'Return (%)', 'Risk (%)', 'Sharpe', 'Sortino', 'Max Drawdown (%)', 'Win Rate (%)']].copy()
        st.dataframe(
            metrics_comparison.style.background_gradient(subset=['Sharpe', 'Return (%)'], cmap='Greens')
                               .background_gradient(subset=['Risk (%)', 'Max Drawdown (%)'], cmap='Reds_r'),
            use_container_width=True,
            hide_index=True
        )
        
        # Visual comparison
        st.markdown("---")
        st.subheader("📈 Visual Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create portfolio selection dropdown
            # Sort portfolios by Sharpe ratio for easy selection
            if 'comp_df_sorted' in st.session_state:
                df_comp_sorted = st.session_state['comp_df_sorted']
                portfolio_options = st.session_state['comp_portfolio_options']
            else:
                df_comp_sorted = df_comp.sort_values('Sharpe', ascending=False).reset_index(drop=True)
                
                # Create selection options
                portfolio_options = []
                for idx in range(min(100, len(df_comp_sorted))):  # Top 100 by Sharpe
                    port = df_comp_sorted.iloc[idx]
                    portfolio_options.append({
                        'index': df_comp_sorted.index[idx],
                        'label': f"Portfolio #{idx+1}: Return={port['Return']*100:.2f}%, Risk={port['Vol']*100:.2f}%, Sharpe={port['Sharpe']:.2f}"
                    })
                
                # Store options in session state
                st.session_state['comp_portfolio_options'] = portfolio_options
                st.session_state['comp_df_sorted'] = df_comp_sorted
            
            # Portfolio selector
            st.markdown("**🎯 Select a Portfolio to Analyze:**")
            selected_option_idx = st.selectbox(
                "Choose a portfolio from the efficient frontier:",
                range(len(portfolio_options)),
                format_func=lambda x: portfolio_options[x]['label'],
                key="portfolio_selector_comp"
            )
            
            # Get selected portfolio
            selected_port_idx = portfolio_options[selected_option_idx]['index']
            selected_portfolio = df_comp_sorted.iloc[selected_port_idx]
            
            # Efficient frontier with all strategies and selected portfolio
            fig_comp = go.Figure()
            
            # All portfolios
            fig_comp.add_trace(go.Scatter(
                x=df_comp["Vol"]*100,
                y=df_comp["Return"]*100,
                mode='markers',
                marker=dict(size=3, color='lightgray', opacity=0.2),
                name='All Portfolios',
                hovertemplate='Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>'
            ))
            
            # Selected portfolio (highlighted)
            fig_comp.add_trace(go.Scatter(
                x=[selected_portfolio["Vol"]*100],
                y=[selected_portfolio["Return"]*100],
                mode='markers',
                marker=dict(size=25, color='purple', symbol='star', 
                          line=dict(width=3, color='white')),
                name='Selected Portfolio',
                hovertemplate='<b>Selected Portfolio</b><br>Return: %{y:.2f}%<br>Risk: %{x:.2f}%<br>Sharpe: ' + f'{selected_portfolio["Sharpe"]:.2f}<extra></extra>'
            ))
            
            # Strategy portfolios
            colors_strat = ['red', 'blue', 'green']
            symbols_strat = ['star', 'diamond', 'circle']
            for idx, (strategy_name, best_port) in enumerate(strategies.items()):
                fig_comp.add_trace(go.Scatter(
                    x=[best_port["Vol"]*100],
                    y=[best_port["Return"]*100],
                    mode='markers',
                    marker=dict(size=20, color=colors_strat[idx], symbol=symbols_strat[idx], 
                              line=dict(width=2, color='white')),
                    name=strategy_name,
                    hovertemplate=f'<b>{strategy_name}</b><br>Return: %{{y:.2f}}%<br>Risk: %{{x:.2f}}%<extra></extra>'
                ))
            
            fig_comp.update_layout(
                title="Strategy Comparison on Efficient Frontier (Select from dropdown above)",
                xaxis_title="Volatility (Risk) %",
                yaxis_title="Expected Annual Return %",
                height=500,
                template='plotly_white',
                showlegend=True,
                hovermode='closest'
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Display selected portfolio metrics and comparison
            st.markdown("---")
            st.subheader("📊 Selected Portfolio Analysis")
            
            # Calculate metrics for selected portfolio
            selected_weights = np.array(selected_portfolio["Weights"])
            selected_portfolio_returns = (returns_comp @ selected_weights).dropna()
            selected_metrics = calculate_portfolio_metrics(selected_portfolio_returns, rf_comp)
            
            # Compare with optimal strategies
            st.markdown("#### Selected Portfolio vs Optimal Strategies")
            
            # Create comparison table
            selected_comparison = {
                'Metric': ['Expected Return (%)', 'Volatility (Risk) (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Win Rate (%)'],
                'Selected Portfolio': [
                    f"{selected_portfolio['Return']*100:.2f}",
                    f"{selected_portfolio['Vol']*100:.2f}",
                    f"{selected_portfolio['Sharpe']:.2f}",
                    f"{selected_metrics['sortino']:.2f}",
                    f"{selected_metrics['max_drawdown']:.2f}",
                    f"{selected_metrics['win_rate']:.2f}"
                ],
                'Max Sharpe': [
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Return (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Risk (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Sharpe'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Sortino'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Max Drawdown (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Win Rate (%)'].values[0]:.2f}"
                ],
                'Max Return': [
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Return', 'Return (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Return', 'Risk (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Return', 'Sharpe'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Return', 'Sortino'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Return', 'Max Drawdown (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Max Return', 'Win Rate (%)'].values[0]:.2f}"
                ],
                'Min Risk': [
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Min Risk', 'Return (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Min Risk', 'Risk (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Min Risk', 'Sharpe'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Min Risk', 'Sortino'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Min Risk', 'Max Drawdown (%)'].values[0]:.2f}",
                    f"{comparison_df.loc[comparison_df['Strategy'] == 'Min Risk', 'Win Rate (%)'].values[0]:.2f}"
                ]
            }
            
            selected_comparison_df = pd.DataFrame(selected_comparison)
            st.dataframe(selected_comparison_df, use_container_width=True, hide_index=True)
            
            # Key insights
            st.markdown("#### 💡 Key Insights")
            
            # Calculate differences
            max_sharpe_return = comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Return (%)'].values[0]
            max_sharpe_risk = comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Risk (%)'].values[0]
            max_sharpe_sharpe = comparison_df.loc[comparison_df['Strategy'] == 'Max Sharpe', 'Sharpe'].values[0]
            
            return_diff = selected_portfolio['Return']*100 - max_sharpe_return
            risk_diff = selected_portfolio['Vol']*100 - max_sharpe_risk
            sharpe_diff = selected_portfolio['Sharpe'] - max_sharpe_sharpe
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.markdown(f"""
                **📈 Performance Comparison:**
                - Return vs Max Sharpe: {return_diff:+.2f}%
                - Risk vs Max Sharpe: {risk_diff:+.2f}%
                - Sharpe vs Max Sharpe: {sharpe_diff:+.2f}
                """)
            
            with insight_col2:
                # Determine if selected is better/worse
                if sharpe_diff > 0:
                    verdict = "✅ Better risk-adjusted return than Max Sharpe"
                elif sharpe_diff < -0.1:
                    verdict = "⚠️ Lower risk-adjusted return than Max Sharpe"
                else:
                    verdict = "≈ Similar to Max Sharpe"
                
                st.markdown(f"""
                **🎯 Assessment:**
                {verdict}
                
                **Portfolio Weights:**
                {', '.join([f"{t}: {(w*100):.1f}%" for t, w in zip(tickers_comp_list, selected_weights)])}
                """)
            
            with col2:
                # Bar chart comparison
                fig_bar = go.Figure()
                
                strategies_list = comparison_df['Strategy'].tolist()
                metrics_to_compare = ['Return (%)', 'Risk (%)', 'Sharpe']
                
                for metric in metrics_to_compare:
                    fig_bar.add_trace(go.Bar(
                        name=metric,
                        x=strategies_list,
                        y=comparison_df[metric],
                        text=[f"{v:.2f}" for v in comparison_df[metric]],
                        textposition='auto'
                    ))
                
                fig_bar.update_layout(
                    title="Metrics Comparison",
                    xaxis_title="Strategy",
                    yaxis_title="Value",
                    barmode='group',
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Weight comparison
            st.markdown("---")
            st.subheader("🎯 Portfolio Weights Comparison")
            
            weights_comparison_all = pd.DataFrame({
                'Ticker': tickers_comp_list
            })
            
            for strategy_name in strategies.keys():
                strategy_idx = comparison_df[comparison_df['Strategy'] == strategy_name].index[0]
                weights_comparison_all[strategy_name] = [
                    f"{(w * 100):.2f}%" for w in comparison_df.loc[strategy_idx, 'Weights']
                ]
            
            st.dataframe(weights_comparison_all, use_container_width=True, hide_index=True)
            
            # Performance over time comparison
            st.markdown("---")
            st.subheader("💰 Historical Performance Comparison")
            
            fig_perf = go.Figure()
            
            colors_perf = ['red', 'blue', 'green']
            for idx, (strategy_name, best_port) in enumerate(strategies.items()):
                weights_perf = np.array(best_port["Weights"])
                portfolio_returns_perf = (returns_comp @ weights_perf).dropna()
                cum_returns_perf = (1 + portfolio_returns_perf).cumprod() * 10000
                
                fig_perf.add_trace(go.Scatter(
                    x=cum_returns_perf.index,
                    y=cum_returns_perf,
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=colors_perf[idx], width=2)
                ))
            
            fig_perf.update_layout(
                title="Portfolio Value Over Time ($10,000 Initial)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                template='plotly_white',
                hovermode='x unified'
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Summary insights
            st.markdown("---")
            st.subheader("💡 Key Insights")
            
            best_sharpe_strat = comparison_df.loc[comparison_df['Sharpe'].idxmax(), 'Strategy']
            best_return_strat = comparison_df.loc[comparison_df['Return (%)'].idxmax(), 'Strategy']
            best_risk_strat = comparison_df.loc[comparison_df['Risk (%)'].idxmin(), 'Strategy']
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.markdown(f"""
                **🏆 Best Strategies:**
                - **Best Risk-Adjusted Return**: {best_sharpe_strat} (Sharpe: {comparison_df.loc[comparison_df['Sharpe'].idxmax(), 'Sharpe']:.2f})
                - **Highest Return**: {best_return_strat} ({comparison_df.loc[comparison_df['Return (%)'].idxmax(), 'Return (%)']:.2f}%)
                - **Lowest Risk**: {best_risk_strat} ({comparison_df.loc[comparison_df['Risk (%)'].idxmin(), 'Risk (%)']:.2f}%)
                """)
            
            with insight_col2:
                st.markdown(f"""
                **📊 Trade-offs:**
                - **Max Return** typically has highest risk
                - **Min Risk** sacrifices return for stability
                - **Max Sharpe** balances return and risk optimally
                - Choose based on your risk tolerance and investment goals
                """)
