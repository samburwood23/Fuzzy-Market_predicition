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
- Technical indicators: RSI (Wilder), MACD, annualised volatility, Bollinger Bands, SMA/EMA and ATR
- Five-level fuzzy signal (`strong_sell`, `sell`, `hold`, `buy`, `strong_buy`) with a confidence score
- Combines momentum (MACD) and mean reversion (RSI); high volatility pulls signals towards `hold`
- Configurable RSI oversold/overbought thresholds

### Marketing Module
- Campaign quality evaluation from engagement rate, conversion rate and ROI
- RFM-style customer segmentation (`champion`, `loyal`, `potential_loyalist`, `at_risk`)
- A recommended next step for every rating

### Shareholder Value Module
- Combines trading performance, marketing ROI, customer retention and market share growth
- Composite score from 0 to 100 with a `poor` / `fair` / `good` / `excellent` rating
- Targeted actions for each metric that is clearly weak

## Installation

Requires Python 3.9+ and NumPy 2.

```bash
git clone https://github.com/samburwood23/Fuzzy-Market_predicition.git
cd Fuzzy-Market_predicition
pip install -e '.[dev]'
```

## Quick Start

### Trading System

```python
from fuzzy_analytics import TradingFuzzySystem

trading_system = TradingFuzzySystem()

# Either pass indicator values directly...
result = trading_system.evaluate(
    rsi=25,               # 0–100
    macd_histogram=0.4,   # MACD histogram as a % of price
    volatility=0.2,       # annualised, 0.2 = 20%
)
print(result["action"], round(result["confidence"], 2))   # strong_buy 1.0

# ...or let the system compute them from closing prices (oldest first)
import pandas as pd
closes = pd.read_csv("market_data.csv")["close"]
result = trading_system.evaluate_prices(closes)
print(result["action"], result["indicators"])
```

### Marketing System

```python
from fuzzy_analytics import MarketingFuzzySystem, CustomerSegmentationSystem

marketing_system = MarketingFuzzySystem()
campaign = marketing_system.evaluate_campaign(
    engagement_rate=0.60,   # 60% of the audience engaged
    conversion_rate=0.08,   # 8% of engaged users converted
    roi=2.5,                # 2.5x return on spend
)
print(campaign["quality"], "-", campaign["recommendation"])

segmentation = CustomerSegmentationSystem()
customer = segmentation.segment_customer(
    recency_days=12,
    purchase_frequency=22,
    annual_spend=8200,
)
print(customer["segment"])   # champion
```

### Shareholder Value Optimization

```python
from fuzzy_analytics import ShareholderValueOptimizer

optimizer = ShareholderValueOptimizer()
result = optimizer.optimize(
    trading_performance=0.75,    # 75% positive trading signals
    marketing_roi=2.5,           # 2.5x return on marketing spend
    customer_retention=0.82,     # 82% retention rate
    market_share_growth=0.15,    # +15% market share
)

print(f"Score: {result['score']:.0f} ({result['rating']})")
print(result["recommendation"])
for action in result["actions"]:
    print("-", action)
```

### Technical Indicators

```python
from fuzzy_analytics import TradingIndicators

rsi = TradingIndicators.calculate_rsi(closes)
macd, signal, histogram = TradingIndicators.calculate_macd(closes)
middle, upper, lower = TradingIndicators.calculate_bollinger_bands(closes)
atr = TradingIndicators.calculate_atr(highs, lows, closes)
```

## Architecture

```
src/fuzzy_analytics/
├── __init__.py               # Public API
├── core/                     # Generic Mamdani fuzzy engine
│   ├── membership_functions.py   # triangular, trapezoidal, gaussian, sigmoid
│   ├── fuzzy_variable.py         # FuzzySet, FuzzyVariable
│   ├── fuzzy_rule.py             # FuzzyRule (AND of antecedents, weighted)
│   └── fuzzy_system.py           # FuzzyInferenceSystem (centroid defuzzification)
├── indicators/               # RSI, MACD, volatility, Bollinger, SMA/EMA, ATR
├── trading/                  # TradingFuzzySystem
├── marketing/                # MarketingFuzzySystem, CustomerSegmentationSystem
└── shareholder_value/        # ShareholderValueOptimizer
tests/                        # pytest suite
```

## Fuzzy Logic Approach

All systems use **Mamdani inference** built on the small engine in `fuzzy_analytics.core`:

1. **Fuzzification**: each input is clipped to its variable's domain and mapped to membership degrees. Edge sets are trapezoidal "shoulders", so extreme values (an ROI of 50, say) still count fully as "high".
2. **Rule evaluation**: antecedents are combined with the minimum (AND) and scaled by the rule's weight.
3. **Aggregation**: each rule clips its output set at its firing strength; clipped sets are combined with the maximum.
4. **Defuzzification**: the centroid of the aggregated set gives the crisp output, which is then mapped to the label it belongs to most strongly.

Membership functions accept NumPy arrays, so the output universe is evaluated in a single vectorised pass.

### Building Your Own System

```python
from fuzzy_analytics.core import FuzzyInferenceSystem, FuzzyRule, FuzzySet, FuzzyVariable
from fuzzy_analytics.core import membership_functions as mf

temp = FuzzyVariable("temperature", (0, 40))
temp.add_set("cold", FuzzySet(mf.trapezoidal(0, 0, 10, 20)))
temp.add_set("hot", FuzzySet(mf.trapezoidal(20, 30, 40, 40)))

fan = FuzzyVariable("fan_speed", (0, 100))
fan.add_set("slow", FuzzySet(mf.triangular(0, 20, 50)))
fan.add_set("fast", FuzzySet(mf.triangular(50, 80, 100)))

system = FuzzyInferenceSystem(
    [temp], fan,
    [FuzzyRule([(temp, "cold")], (fan, "slow")),
     FuzzyRule([(temp, "hot")], (fan, "fast"))],
)
print(system.evaluate({"temperature": 27}))

# See which rules fired and how strongly
for rule, strength in system.rule_activations({"temperature": 27}):
    print(f"{strength:.2f}  {rule}")
```

### Rule Base Examples

**Trading rules:**
```
IF rsi is oversold AND macd_histogram is bullish THEN signal is strong_buy
IF rsi is neutral AND macd_histogram is neutral THEN signal is hold
IF rsi is overbought AND macd_histogram is bearish THEN signal is strong_sell
IF volatility is high THEN signal is hold (weight 0.7)
```

**Marketing rules:**
```
IF engagement_rate is high AND conversion_rate is high AND roi is high THEN quality is excellent
IF engagement_rate is high AND conversion_rate is low THEN quality is average (weight 0.8)
```

## Use Cases

### 1. Algorithmic Trading
Integrate fuzzy signals into automated trading strategies where market uncertainty requires nuanced decision-making beyond simple threshold-based rules.

### 2. Marketing Campaign Optimization
Evaluate multiple campaigns simultaneously and allocate budget based on fuzzy performance metrics that account for reach, engagement, and conversion holistically.

### 3. Customer Lifetime Value Prediction
Use fuzzy customer segmentation to identify high-value customers and predict churn risk with interpretable rules.

### 4. Strategic Business Planning
Combine trading performance and marketing effectiveness to provide executive-level insights on shareholder value creation.

## Performance

- A trading evaluation takes roughly a quarter of a millisecond on a laptop, which suits batch screening and bar-by-bar strategies (not high-frequency trading).
- Systems are stateless after construction, so you can evaluate many assets or campaigns in parallel.
- Every decision can be traced to specific rules via `rule_activations`.

## Testing

```bash
pytest
```

This runs the unit tests in `tests/` and the examples in the docstrings. GitHub Actions runs the same suite on every push.

## Roadmap

- [ ] Add sentiment analysis integration for trading signals
- [ ] Implement genetic algorithm for rule optimization
- [ ] Support for multi-asset portfolio optimization
- [ ] Real-time market data integration via APIs
- [ ] Web dashboard for visualization
- [ ] Backtesting framework with historical data
- [ ] OR operators and Sugeno-style inference in the core engine
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

Built with [NumPy](https://numpy.org/). The fuzzy engine is implemented from scratch, with no fuzzy logic library dependency.

---

**Disclaimer**: This software is for educational and research purposes. Trading and investment decisions involve substantial risk. Always perform due diligence and consider consulting financial professionals before making investment decisions.
