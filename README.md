# 💹 Parallel Portfolio Optimization Dashboard

A comprehensive Streamlit application demonstrating **parallel computing** for portfolio optimization using real stock market data from both US and Indian NSE markets via Yahoo Finance API. The app features advanced interactive tools for portfolio analysis, risk assessment, and performance comparison.

## 🚀 Key Features

### Core Functionality
- **US Market Portfolio Optimization**: Optimize portfolios using US stocks (NASDAQ, NYSE, etc.)
- **Indian NSE Market Portfolio Optimization**: Optimize portfolios using NSE India stocks
- **Parallel Processing**: Leverage multiprocessing to speed up portfolio simulations across multiple CPU cores
- **Portfolio Comparison Tool**: Compare different optimization strategies side-by-side
- **Speedup Benchmarking**: Measure and visualize parallel performance improvements

---

## 📊 Tab 1: Portfolio Optimization (US Markets)

### Main Features
- **Portfolio Optimization**: Optimize portfolios using three strategies:
  - **Max Sharpe**: Maximum risk-adjusted return
  - **Max Return**: Maximum expected return
  - **Min Risk**: Minimum volatility/risk
- **Efficient Frontier Visualization**: Interactive Plotly chart showing all simulated portfolios
- **Portfolio Weights**: Donut chart and table showing optimal asset allocation
- **Historical Backtest**: Dual-panel chart showing portfolio value growth and drawdown analysis
- **Performance Metrics**: Comprehensive metrics including:
  - Total return, annual return, CAGR
  - Sharpe & Sortino ratios
  - Maximum drawdown
  - Win rate and daily performance stats

### 🎮 Interactive Features (5 Sub-tabs)

#### 1. 🔄 Portfolio Rebalancing
- **Interactive Sliders**: Manually adjust portfolio weights for each stock
- **Real-time Metrics**: See immediate impact on return, volatility, and Sharpe ratio
- **Visual Comparison**: Compare your adjusted portfolio with the optimized one on the efficient frontier
- **Reset Button**: Quickly return to optimized weights

#### 2. 📊 Correlation Analysis
- **Correlation Heatmap**: Interactive heatmap showing how stocks move together
- **Correlation Insights**: Identifies highest and lowest correlation pairs
- **Diversification Guidance**: Understand portfolio diversification benefits

#### 3. 🎲 Monte Carlo Simulation
- **Future Projections**: Simulate thousands of possible future scenarios (500-5000 simulations)
- **Percentile Bands**: Visualize 5th-95th, 25th-75th percentiles and median projections
- **Probability Analysis**: Calculate probability of gains over different time horizons
- **Customizable Periods**: Project 30-504 days into the future

#### 4. ⚠️ Risk Analysis
- **Value at Risk (VaR)**: Calculate VaR at 5% and 1% confidence levels
- **Conditional VaR (CVaR)**: Expected shortfall analysis
- **Return Distribution**: Histogram showing portfolio return distribution with VaR thresholds
- **Risk Summary Table**: Comprehensive risk metrics dashboard

#### 5. 📥 Export Results
- **Download Weights**: Export portfolio weights as CSV
- **Download Metrics**: Export performance metrics as CSV
- **Download All Portfolios**: Export all simulated portfolios for further analysis
- **Timestamped Files**: Automatic file naming with dates

---

## 🇮🇳 Tab 2: Portfolio Optimization (NSE India)

### Main Features
- **NSE Stock Support**: Optimize portfolios using Indian National Stock Exchange stocks
- **Automatic .NS Suffix**: Enter tickers without suffix (e.g., RELIANCE, TCS, INFY)
- **Same Optimization Strategies**: Max Sharpe, Max Return, Min Risk
- **Rupee Currency Display**: All values displayed in ₹ (Indian Rupees)
- **Initial Investment**: ₹1,00,000 default for backtesting

### 🎮 Interactive Features (5 Sub-tabs)
Same interactive features as Tab 1, adapted for NSE stocks:
- Portfolio Rebalancing with sliders
- Correlation Analysis for Indian stocks
- Monte Carlo Simulation with ₹1,00,000 initial value
- Risk Analysis (VaR, CVaR)
- Export Results functionality

---

## ⚙️ Tab 3: Parallel Speedup Benchmark

### Features
- **Performance Measurement**: Compare runtime with 1, 2, 4, and 8 workers
- **Speedup Visualization**: Interactive charts showing:
  - Speedup vs. Number of Workers (with ideal linear speedup reference)
  - Parallel Efficiency vs. Number of Workers
  - Runtime Comparison bar chart
- **Performance Metrics**:
  - Sequential time (baseline)
  - Maximum speedup achieved
  - Parallel efficiency percentage
  - Time saved with parallelization
- **Amdahl's Law Analysis**: Insights into parallel scalability
- **Performance Recommendations**: Optimal worker count suggestions

---

## 🔀 Tab 4: Portfolio Comparison Tool

### Features
- **Multi-Strategy Comparison**: Compare Max Sharpe, Max Return, and Min Risk strategies simultaneously
- **Interactive Portfolio Selection**: 
  - Dropdown to select from top 100 portfolios (sorted by Sharpe ratio)
  - Visual highlighting of selected portfolio on efficient frontier
  - Detailed metrics comparison table
- **Side-by-Side Metrics**: Compare selected portfolio with all optimal strategies
- **Visual Comparisons**:
  - Efficient frontier with all strategies and selected portfolio
  - Bar chart comparing key metrics
  - Historical performance overlay
- **Portfolio Weights Comparison**: See how different strategies allocate assets
- **Key Insights**: Automated assessment of portfolio performance vs. optimal strategies

---

## 📈 Recommended Stock Tickers

### 🇺🇸 US Stock Tickers (Groups of 5 by Sector)

#### Technology
- **AAPL, MSFT, GOOG, AMZN, META** (Apple, Microsoft, Google, Amazon, Meta)
- **NVDA, AMD, INTC, QCOM, AVGO** (NVIDIA, AMD, Intel, Qualcomm, Broadcom)

#### Finance & Banking
- **JPM, BAC, WFC, GS, MS** (JPMorgan, Bank of America, Wells Fargo, Goldman Sachs, Morgan Stanley)
- **V, MA, AXP, COF, DFS** (Visa, Mastercard, American Express, Capital One, Discover)

#### Healthcare & Pharmaceuticals
- **JNJ, PFE, UNH, ABT, TMO** (Johnson & Johnson, Pfizer, UnitedHealth, Abbott, Thermo Fisher)
- **LLY, MRK, ABBV, BMY, GILD** (Eli Lilly, Merck, AbbVie, Bristol-Myers, Gilead)

#### Consumer & Retail
- **WMT, HD, MCD, NKE, SBUX** (Walmart, Home Depot, McDonald's, Nike, Starbucks)
- **TGT, COST, LOW, TJX, DG** (Target, Costco, Lowe's, TJX, Dollar General)

#### Energy & Utilities
- **XOM, CVX, SLB, EOG, COP** (ExxonMobil, Chevron, Schlumberger, EOG Resources, ConocoPhillips)
- **NEE, DUK, SO, AEP, SRE** (NextEra Energy, Duke Energy, Southern Company, AEP, Sempra)

#### Industrial & Manufacturing
- **BA, CAT, GE, HON, RTX** (Boeing, Caterpillar, General Electric, Honeywell, Raytheon)
- **UNP, CSX, NSC, UPS, FDX** (Union Pacific, CSX, Norfolk Southern, UPS, FedEx)

#### Communication & Media
- **DIS, NFLX, CMCSA, T, VZ** (Disney, Netflix, Comcast, AT&T, Verizon)
- **GOOGL, META, SNAP, TWTR, ROKU** (Google, Meta, Snap, Twitter, Roku)

---

### 🇮🇳 NSE Stock Tickers (Groups of 5 by Sector)

#### Banking & Finance
- **HDFCBANK, ICICIBANK, SBIN, KOTAKBANK, AXISBANK** (HDFC Bank, ICICI Bank, SBI, Kotak Bank, Axis Bank)
- **INDUSINDBK, FEDERALBNK, BANDHANBNK, IDFCFIRSTB, YESBANK** (IndusInd Bank, Federal Bank, Bandhan Bank, IDFC First Bank, Yes Bank)

#### Information Technology
- **TCS, INFY, WIPRO, HCLTECH, TECHM** (TCS, Infosys, Wipro, HCL Tech, Tech Mahindra)
- **LTIM, MINDTREE, MPHASIS, ZENSAR, PERSISTENT** (LTI Mindtree, Mindtree, Mphasis, Zensar, Persistent)

#### Consumer & Retail
- **RELIANCE, ITC, HINDUNILVR, ASIANPAINT, MARUTI** (Reliance, ITC, HUL, Asian Paints, Maruti)
- **TITAN, TATACONSUM, GODREJCP, DABUR, BRITANNIA** (Titan, Tata Consumer, Godrej CP, Dabur, Britannia)

#### Pharmaceuticals & Healthcare
- **SUNPHARMA, DRREDDY, CIPLA, LUPIN, AUROPHARMA** (Sun Pharma, Dr. Reddy's, Cipla, Lupin, Aurobindo)
- **DIVISLAB, BIOCON, TORONTPHARM, GLENMARK, CADILAHC** (Divis Labs, Biocon, Torrent Pharma, Glenmark, Cadila)

#### Energy & Power
- **ONGC, NTPC, POWERGRID, GAIL, IOC** (ONGC, NTPC, Power Grid, GAIL, Indian Oil)
- **BPCL, HPCL, ADANIPOWER, TATAPOWER, RELIANCE** (BPCL, HPCL, Adani Power, Tata Power, Reliance)

#### Infrastructure & Engineering
- **LT, ADANIPORTS, JSWSTEEL, TATASTEEL, SAIL** (L&T, Adani Ports, JSW Steel, Tata Steel, SAIL)
- **ULTRACEMCO, AMBUJACEM, SHREECEM, ACC, RAMCOCEM** (UltraTech, Ambuja, Shree Cement, ACC, Ramco)

#### Telecom & Communication
- **BHARTIARTL, RELIANCE, IDEA, TATACOMM, MTNL** (Bharti Airtel, Reliance Jio, Vodafone Idea, Tata Comm, MTNL)

#### Automobiles
- **MARUTI, TATAMOTORS, M&M, BAJAJ-AUTO, EICHERMOT** (Maruti, Tata Motors, Mahindra, Bajaj Auto, Eicher)
- **HEROMOTOCO, TVSMOTOR, ASHOKLEY, FORCEMOT, ESCORTS** (Hero MotoCorp, TVS Motor, Ashok Leyland, Force Motors, Escorts)

---

## 📦 Installation

1. **Create a Python virtual environment** (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

2. **Install the required dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:

```bash
streamlit run app.py
```

### Note on Virtual Environment

Using a virtual environment is recommended to:
- Isolate project dependencies
- Avoid conflicts with other Python projects
- Ensure consistent package versions

**To deactivate the virtual environment** when done:
```bash
deactivate
```

## 📊 Usage Guide

### Getting Started

1. **Select a Tab**: Choose from 4 main tabs based on your needs
2. **Enter Stock Tickers**: Use comma-separated tickers (see recommended groups above)
3. **Configure Parameters**:
   - Number of portfolios to simulate (1,000 - 1,000,000)
   - Number of parallel workers (1-8)
   - Optimization metric (Max Sharpe, Max Return, Min Risk)
   - Risk-free rate
   - Date range for historical data
4. **Run Optimization**: Click the optimization button
5. **Explore Results**: Use interactive features to analyze and compare portfolios

### Tips for Best Results

- **US Stocks**: Use standard ticker symbols (e.g., AAPL, MSFT)
- **NSE Stocks**: Enter without .NS suffix (e.g., RELIANCE, TCS)
- **Portfolio Size**: More portfolios (10,000+) give better optimization results
- **Workers**: Use 4-8 workers for faster computation on multi-core systems
- **Date Range**: Use at least 2-3 years of data for reliable statistics

---

## 🔧 Technical Details

### Parallel Computing Architecture

The application uses Python's `multiprocessing.Pool` to distribute portfolio simulations across multiple CPU cores, demonstrating:

- **Process-based Parallelism**: Each worker runs in a separate process
- **Workload Distribution**: Portfolios are evenly distributed across workers
- **Performance Measurement**: Real-time benchmarking of parallel speedup
- **Scalability Analysis**: Amdahl's Law implications and efficiency metrics

### Portfolio Optimization Methods

- **Monte Carlo Simulation**: Randomly generates thousands of portfolio allocations
- **Mean-Variance Optimization**: Evaluates risk-return tradeoffs using historical data
- **Sharpe Ratio Maximization**: Measures risk-adjusted returns
- **Efficient Frontier Analysis**: Visualizes optimal portfolios in risk-return space

### Risk Metrics

- **Value at Risk (VaR)**: Maximum expected loss at confidence levels
- **Conditional VaR (CVaR)**: Expected shortfall beyond VaR threshold
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sortino Ratio**: Risk-adjusted return focusing on downside volatility

### Data Sources

- **US Markets**: Yahoo Finance API via `yfinance` library
- **Indian NSE**: Yahoo Finance API via `yfinance` with `.NS` suffix
- **Data Frequency**: Daily adjusted close prices
- **Return Calculation**: Logarithmic returns for accurate compounding

---

## 📝 Notes for Academic Projects

This project is ideal for demonstrating:

1. **Parallel Computing Concepts**: 
   - Process pools and workload distribution
   - Speedup analysis and efficiency metrics
   - Amdahl's Law in practice

2. **Financial Modeling**:
   - Portfolio optimization theory
   - Risk-return analysis
   - Monte Carlo simulation methods

3. **Real-World Applications**:
   - Financial technology (FinTech)
   - Data-driven decision making
   - Interactive data visualization

4. **Performance Benchmarking**:
   - Sequential vs. parallel execution
   - Scalability analysis
   - Resource utilization

### Performance Considerations

- **Data Fetching**: Sequential (I/O bound) - cannot be parallelized
- **Portfolio Simulation**: Parallel (CPU bound) - benefits from multiprocessing
- **Optimal Worker Count**: Depends on available CPU cores (typically 4-8)
- **Memory Usage**: Increases with number of portfolios and workers

---

## 🎯 Key Metrics Explained

### Return Metrics
- **Total Return**: Cumulative return over the entire period
- **Annual Return**: Average annualized return
- **CAGR**: Compound Annual Growth Rate

### Risk Metrics
- **Volatility**: Standard deviation of returns (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline percentage
- **VaR (5%)**: Loss threshold expected 5% of the time
- **CVaR**: Average loss when VaR is exceeded

### Risk-Adjusted Metrics
- **Sharpe Ratio**: (Return - Risk-free rate) / Volatility
- **Sortino Ratio**: (Return - Risk-free rate) / Downside volatility
- **Win Rate**: Percentage of positive return days

---

## 🤝 Dependencies

- `streamlit`: Web application framework
- `yfinance`: Stock market data (US and Indian NSE)
- `numpy`: Numerical computations and array operations
- `pandas`: Data manipulation and analysis
- `matplotlib`: Basic plotting (legacy support)
- `plotly`: Interactive visualizations and charts

---

## 📄 License

This project is created for educational purposes as part of a Parallel Computing course.

---

## 🆘 Troubleshooting

### Common Issues

1. **Data Fetching Errors**: 
   - Check internet connection
   - Verify ticker symbols are correct
   - Try a different date range

2. **NSE Tickers Not Working**:
   - Ensure you're entering without .NS suffix
   - Check if ticker is listed on NSE
   - Some tickers may have limited historical data

3. **Slow Performance**:
   - Reduce number of portfolios
   - Use fewer workers if system is overloaded
   - Check date range (longer ranges = more data)

4. **Memory Issues**:
   - Reduce number of portfolios
   - Use shorter date ranges
   - Close other applications

---

## 📚 Additional Resources

- **Portfolio Theory**: Modern Portfolio Theory (Markowitz)
- **Parallel Computing**: Python multiprocessing documentation
- **Financial Data**: Yahoo Finance API documentation
- **Streamlit**: Streamlit documentation for UI components

---

**Happy Portfolio Optimizing! 🚀📈**
