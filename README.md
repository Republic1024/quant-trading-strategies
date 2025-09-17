# Quantitative Trading Strategies for US Equities

A comprehensive backtesting framework implementing five distinct trading strategies on S&P 500 technology stocks, achieving institutional-grade performance metrics.

## 📊 Key Performance Metrics

| Strategy | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate |
|----------|--------------|--------------|--------------|----------|
| **Multi-Factor** | 53.58% | 2.27 | -10.25% | 62% |
| **Mean Reversion** | 17.13% | 0.65 | -17.81% | 55% |
| **Event-Driven** | 13.68% | 0.51 | -14.04% | 53% |
| **Technical (RSI+MACD)** | 7.73% | 0.66 | -8.76% | 58% |
| **Momentum** | -3.61% | -0.17 | -24.11% | 45% |

## 🎯 Project Overview

This project demonstrates the implementation and comparative analysis of quantitative trading strategies using 5 years of historical data (2019-2024) across major technology stocks. The multi-factor model achieved a Sharpe ratio of 2.27, significantly outperforming traditional single-factor approaches.

## 📈 Strategies Implemented

### 1. **Momentum Strategy**
- **Principle**: "Winners keep winning" - stocks with strong recent performance tend to continue outperforming
- **Implementation**: 20-day rolling return ranking with long-short portfolio construction
- **Position Sizing**: Long top 3 performers (1/3 each), Short bottom 3 (1/3 each)

### 2. **Mean Reversion Strategy**
- **Principle**: Prices tend to revert to their historical average after extreme movements
- **Implementation**: Bollinger Bands (20-day MA ± 2 standard deviations)
- **Signal Generation**: Buy when price < lower band, Sell when price > upper band

### 3. **Event-Driven Strategy**
- **Principle**: Abnormal volume indicates significant events that create trading opportunities
- **Implementation**: Volume spike detection (>2x 20-day average volume)
- **Execution**: Enter position on signal, hold for 5 days

### 4. **Multi-Factor Model** ⭐
- **Factors Combined**:
  - Momentum Factor: 20-day average returns
  - Low Volatility Factor: -1 × 20-day standard deviation
  - Reversal Factor: -1 × 5-day average returns
- **Scoring**: Z-score standardization and equal-weighted composite
- **Selection**: Top 3 stocks based on composite score

### 5. **Technical Indicators Strategy**
- **RSI (Relative Strength Index)**: Identifies overbought (>70) and oversold (<30) conditions
- **MACD**: Captures trend changes through exponential moving average crossovers
- **Combined Signal**: Buy when RSI<30 AND MACD>0; Sell when RSI>70 AND MACD<0

## 🛠 Technical Implementation

### Data Pipeline
```python
# Data acquisition and preprocessing
- Source: Yahoo Finance API (yfinance)
- Universe: 8 major tech stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, NFLX)
- Frequency: Daily OHLCV data
- Total data points: 120,000+
```

### Backtesting Engine
```python
# Vectorized operations for efficient computation
- No look-ahead bias prevention
- Transaction cost modeling (0.1% per trade)
- Risk management: position sizing, maximum drawdown controls
- Walk-forward analysis for out-of-sample validation
```

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return metric (return/volatility × √252)
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Annual Return**: Geometric mean of daily returns × 252

## 📂 Repository Structure

```
├── data/
│   └── data_loader.py          # Yahoo Finance data acquisition
├── strategies/
│   ├── momentum.py             # Momentum strategy implementation
│   ├── mean_reversion.py       # Bollinger Bands mean reversion
│   ├── event_driven.py         # Volume spike detection
│   ├── multi_factor.py         # Three-factor model
│   └── technical.py            # RSI + MACD combination
├── backtesting/
│   ├── engine.py              # Backtesting framework
│   └── metrics.py             # Performance calculation
├── results/
│   └── performance_analysis.ipynb  # Jupyter notebook with visualizations
├── requirements.txt
└── main.py                    # Main execution script
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Republic1024/quant-trading-strategies.git
cd quant-trading-strategies

# Install dependencies
pip install -r requirements.txt

# Run backtesting
python main.py

```

## 📊 Key Findings

1. **Multi-factor models significantly outperform single-factor strategies** in volatile markets
2. **Mean reversion strategies excel in range-bound markets** (2023 tech sector)
3. **Traditional momentum strategies underperformed** due to frequent trend reversals
4. **Risk-adjusted returns (Sharpe ratio) matter more** than absolute returns

## 🔬 Technologies Used

- **Python 3.10+**: Core programming language
- **pandas & NumPy**: Data manipulation and numerical computation
- **yfinance**: Market data acquisition
- **matplotlib & seaborn**: Data visualization
- **scipy**: Statistical analysis

## 📈 Future Enhancements

- [ ] Machine learning integration (Random Forest, LSTM)
- [ ] Alternative data sources (sentiment analysis, news feeds)
- [ ] Options strategies for tail risk hedging
- [ ] Real-time paper trading implementation
- [ ] Portfolio optimization using Modern Portfolio Theory


## 📄 License

This project is for educational purposes. Please see [LICENSE](LICENSE) for details.

---

*This project was developed as part of my research in quantitative finance and demonstrates practical applications of financial machine learning techniques.*