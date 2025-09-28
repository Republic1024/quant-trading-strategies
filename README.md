# Quantitative Trading Strategies for US Equities

A comprehensive backtesting framework implementing five distinct trading strategies on major technology stocks, achieving institutional-grade performance metrics during the 2023-2025 cycle.

## 📊 Key Performance Metrics

| Strategy | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate |
|----------|--------------|--------------|--------------|----------|
| **Multi-Factor** | **44.91%** | **1.78** | **-22.71%** | 62% |
| **Event-Driven** | **39.01%** | **1.32** | **-17.50%** | 53% |
| **Momentum** | 5.89% | 0.23 | -33.23% | 45% |
| **Mean Reversion** | 0.62% | 0.02 | -30.84% | 55% |
| **Technical (RSI+MACD)** | -5.18% | -0.51 | -23.52% | 58% |

## 🎯 Project Overview

This project demonstrates the implementation and comparative analysis of quantitative trading strategies using **2.5 years of historical data (2023-2025)** across major technology stocks. The multi-factor model achieved a Sharpe ratio of **1.78**, significantly outperforming traditional single-factor approaches in this period.

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
  - **Value Factor**: -1 × Price Position (Price relative to 20-day High-Low range)
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
- **Total data points: 26,680**