# Fuzzy Analytics

A comprehensive fuzzy logic analytics framework for trading, marketing, and shareholder value optimization. This package leverages fuzzy inference systems to handle uncertainty and imprecision in financial markets, marketing campaigns, and strategic business decisions.

## Overview

**Fuzzy Analytics** provides three integrated systems that apply fuzzy logic principles to real-world business challenges:

- **Trading System** – Generate intelligent trading signals from market data using fuzzy logic rules
- **Marketing System** – Evaluate marketing campaign performance and segment customers using fuzzy reasoning
- **Shareholder Value Optimizer** – Synthesize trading and marketing metrics into actionable strategic guidance

### Why Fuzzy Logic?

Traditional binary logic struggles with the inherent uncertainty in financial markets and business metrics. Fuzzy logic excels at:

- **Handling Imprecision**: Market conditions like "high volatility" or "moderate growth" are naturally fuzzy concepts
- **Rule-Based Reasoning**: Express domain expertise as intuitive IF-THEN rules
- **Graceful Degradation**: Provides meaningful outputs even with incomplete or noisy data
- **Human-Interpretable**: Decisions can be traced back to explicit rules, unlike black-box ML models

## Features

### Trading Module
- Technical indicator computation (RSI, MACD, Bollinger Bands, Moving Averages)
- Fuzzy trading signal generation (Strong Buy/Buy/Hold/Sell/Strong Sell)
- Multi-indicator fusion for robust decision making
- Handles uncertainty in market trends and momentum

### Marketing Module
- Campaign performance evaluation using fuzzy metrics
- Customer segmentation based on behavioral patterns
- Marketing ROI optimization with fuzzy rules
- Recommendation system for campaign strategies

### Shareholder Value Module
- Integrates trading and marketing performance
- Generates composite shareholder value scores
- Strategic recommendations based on business performance
- Holistic view of business health

## Installation

### Prerequisites
- Python 3.8+
- NumPy
- scikit-fuzzy
- pandas (for data handling)

### Install Dependencies

```bash
pip install numpy scikit-fuzzy pandas matplotlib
```

### Clone Repository

```bash
git clone https://github.com/samburwood23/Fuzzy-Market_predicition.git
cd Fuzzy-Market_predicition
```

## Quick Start

### Trading System

```python
from fuzzy_analytics import TradingFuzzySystem, TradingIndicators
import pandas as pd

# Load your market data
market_data = pd.read_csv('market_data.csv')

# Initialize trading system
trading_system = TradingFuzzySystem()

# Compute technical indicators
indicators = TradingIndicators()
rsi = indicators.calculate_rsi(market_data['close'])
macd, signal = indicators.calculate_macd(market_data['close'])

# Generate trading signal
signal = trading_system.evaluate(
    rsi=rsi[-1],
    macd_histogram=macd[-1] - signal[-1],
    price_trend='bullish'
)

print(f"Trading Signal: {signal}")
# Output: "Strong Buy", "Buy", "Hold", "Sell", or "Strong Sell"
```

### Marketing System

```python
from fuzzy_analytics import MarketingFuzzySystem, CustomerSegmentationSystem

# Initialize marketing system
marketing_system = MarketingFuzzySystem()

# Evaluate campaign performance
campaign_score = marketing_system.evaluate_campaign(
    reach=0.75,           # 75% reach
    engagement=0.60,      # 60% engagement
    conversion=0.08       # 8% conversion rate
)

print(f"Campaign Performance: {campaign_score}")

# Customer segmentation
segmentation = CustomerSegmentationSystem()
segment = segmentation.classify_customer(
    purchase_frequency=0.70,
    average_order_value=0.85,
    engagement_score=0.65
)

print(f"Customer Segment: {segment}")
# Output: e.g., "High-Value", "At-Risk", "Potential Growth"
```

### Shareholder Value Optimization

```python
from fuzzy_analytics import ShareholderValueOptimizer

# Initialize optimizer
optimizer = ShareholderValueOptimizer()

# Synthesize trading and marketing performance
result = optimizer.optimize(
    trading_performance=0.75,    # 75% positive trading signals
    marketing_roi=2.5,            # 2.5x return on marketing spend
    customer_retention=0.82,      # 82% retention rate
    market_share_growth=0.15      # 15% market share growth
)

print(f"Shareholder Value Score: {result['score']}")
print(f"Strategic Recommendation: {result['recommendation']}")
# Output: Actionable strategic guidance based on fuzzy inference
```

## Architecture

```
fuzzy_analytics/
│
├── __init__.py              # Package initialization
├── trading.py               # Trading fuzzy system
│   ├── TradingFuzzySystem
│   └── TradingIndicators
├── marketing.py             # Marketing fuzzy system
│   ├── MarketingFuzzySystem
│   └── CustomerSegmentationSystem
└── shareholder_value.py     # Shareholder value optimizer
    └── ShareholderValueOptimizer
```

## Fuzzy Logic Approach

### Membership Functions

The system uses multiple membership function types:

- **Triangular**: For well-defined ranges (e.g., RSI overbought/oversold)
- **Trapezoidal**: For plateaus in linguistic variables
- **Gaussian**: For smooth transitions between states

### Inference Methods

- **Mamdani Inference**: Used for trading signals where output is a fuzzy set
- **Sugeno Inference**: Used for shareholder value scores requiring crisp numerical output
- **Fuzzy C-Means Clustering**: Applied in customer segmentation

### Rule Base Examples

**Trading Rules:**
```
IF RSI is Oversold AND MACD is Bullish THEN Signal is Strong_Buy
IF RSI is Neutral AND MACD is Neutral THEN Signal is Hold
IF RSI is Overbought AND MACD is Bearish THEN Signal is Strong_Sell
```

**Marketing Rules:**
```
IF Reach is High AND Engagement is High THEN Campaign_Quality is Excellent
IF Conversion is Low AND Engagement is High THEN Recommendation is Optimize_Funnel
```

## Technical Indicators

The `TradingIndicators` class computes:

- **RSI (Relative Strength Index)**: Momentum indicator (0-100)
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum
- **Bollinger Bands**: Volatility indicator
- **SMA/EMA**: Simple and Exponential Moving Averages
- **ATR (Average True Range)**: Volatility measurement
- **Volume Analysis**: Trading volume patterns

## Use Cases

### 1. Algorithmic Trading
Integrate fuzzy signals into automated trading strategies where market uncertainty requires nuanced decision-making beyond simple threshold-based rules.

### 2. Marketing Campaign Optimization
Evaluate multiple campaigns simultaneously and allocate budget based on fuzzy performance metrics that account for reach, engagement, and conversion holistically.

### 3. Customer Lifetime Value Prediction
Use fuzzy customer segmentation to identify high-value customers and predict churn risk with interpretable rules.

### 4. Strategic Business Planning
Combine trading performance and marketing effectiveness to provide executive-level insights on shareholder value creation.

## Configuration

The fuzzy systems can be customized by modifying membership functions and rule bases:

```python
# Example: Customize RSI thresholds
trading_system = TradingFuzzySystem()
trading_system.configure_rsi(
    oversold_threshold=25,  # More aggressive oversold level
    overbought_threshold=75  # More aggressive overbought level
)
```

## Performance Considerations

- **Computational Efficiency**: Fuzzy inference is lightweight compared to deep learning
- **Real-Time Capability**: Suitable for high-frequency trading applications
- **Scalability**: Can process multiple assets/campaigns in parallel
- **Interpretability**: Every decision can be traced to specific rules

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific module tests
python -m pytest tests/test_trading.py
python -m pytest tests/test_marketing.py
```

## Roadmap

- [ ] Add sentiment analysis integration for trading signals
- [ ] Implement genetic algorithm for rule optimization
- [ ] Support for multi-asset portfolio optimization
- [ ] Real-time market data integration via APIs
- [ ] Web dashboard for visualization
- [ ] Backtesting framework with historical data
- [ ] Export to ONNX for production deployment

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for:

- New fuzzy rule implementations
- Additional technical indicators
- Marketing metrics integration
- Documentation improvements
- Bug fixes

## Research Background

This project builds on established research in fuzzy logic applications:

- **Fuzzy Technical Analysis**: Zadeh's fuzzy set theory applied to trading
- **Adaptive Neuro-Fuzzy Inference Systems (ANFIS)**: Hybrid learning approaches
- **Fuzzy Customer Analytics**: Marketing segmentation with imprecise data
- **Multi-Criteria Decision Making**: Shareholder value as a fuzzy optimization problem

## References

- Zadeh, L.A. (1965). "Fuzzy Sets". Information and Control.
- Jang, J.-S.R. (1993). "ANFIS: Adaptive-Network-Based Fuzzy Inference System". IEEE Transactions on Systems, Man, and Cybernetics.
- Ross, T.J. (2010). "Fuzzy Logic with Engineering Applications". Wiley.

## License

MIT License - see LICENSE file for details

## Author

Sam Burwood  
Specialist Engineering (DevOps) | HashiCorp Vault SME  
[GitHub](https://github.com/samburwood23)

## Acknowledgments

Built with:
- [scikit-fuzzy](https://github.com/scikit-fuzzy/scikit-fuzzy) - Fuzzy logic toolkit for Python
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation

---

**Disclaimer**: This software is for educational and research purposes. Trading and investment decisions involve substantial risk. Always perform due diligence and consider consulting financial professionals before making investment decisions.
