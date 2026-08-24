# Sentiment Trading Research Pipeline

## Objective
Build an end-to-end research pipeline that tests whether financial-news sentiment carries useful information about a stock's next-day price direction — covering news ingestion, FinBERT-based NLP scoring, feature construction, model evaluation, and a simple trading simulation, applied to a configurable ticker (default: AAPL).

## Table of Contents
- [Abstract & Overview](#abstract--overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Core Subsystems](#core-subsystems)
  - [Data Collection](#1-data-collection)
  - [Sentiment Scoring](#2-sentiment-scoring)
  - [Feature Engineering](#3-feature-engineering)
  - [Modelling](#4-modelling)
  - [Backtesting](#5-backtesting)
  - [Visualization](#6-visualization)
- [Feature and Target Definition](#feature-and-target-definition)
- [How to Read the Results](#how-to-read-the-results)
- [Research Cautions](#research-cautions)
- [Getting Started](#getting-started)
- [Disclaimer](#disclaimer)

## Abstract & Overview
Financial news is often argued to move markets before price fully reflects new information. This project builds the full quantitative pipeline needed to test that claim rigorously: raw news collection, NLP-based sentiment scoring, feature construction, a time-ordered classifier, and a realistic backtest — rather than a single accuracy number in isolation.

This repository is intentionally a **research prototype**. Its value lies in making the entire workflow transparent and reproducible: data collection → NLP → feature construction → model evaluation → trading simulation. A positive backtest result is not evidence of a tradable edge until it survives stronger controls, realistic execution assumptions, and out-of-sample validation.

## System Architecture
The system follows a layered pipeline design:

```
Data Layer          Alpha Vantage NEWS_SENTIMENT API → news_data.csv
                     yfinance → adjusted daily prices
                            |
NLP Layer            ProsusAI/finbert → article-level sentiment scores
                            |
Feature Layer        Daily aggregation → lagged sentiment, moving average,
                      momentum, news volume
                            |
Modelling Layer       Time-ordered Logistic Regression → next-day direction
                            |
Evaluation Layer      Backtest vs. buy-and-hold → equity curve, Sharpe,
                       drawdown, win rate
                            |
Output Layer           Diagnostic plots (sentiment, correlation,
                        classification, equity curve)
```

## Repository Structure
```
├── main.py                    # Orchestrates the complete workflow
├── config.py                  # Loads environment variables; ticker/date/output settings
├── news_fetcher.py            # Downloads and saves Alpha Vantage news
├── sentiment_analyzer.py      # Loads FinBERT and scores articles
├── price_data.py              # Downloads adjusted prices and computes returns
├── feature_engineering.py     # Aggregates, merges, and constructs model inputs
├── model.py                   # Fits and evaluates logistic regression
├── backtest.py                # Computes strategy returns, costs, equity, metrics
├── visualize.py                # Produces exploratory and evaluation charts
├── data/                       # Local CSV output and research data
├── requirements.txt
└── .env                        # API keys (not committed)
```

## Core Subsystems

### 1. Data Collection
News is fetched from Alpha Vantage's `NEWS_SENTIMENT` endpoint one month at a time (rate-limit aware) via the `AlphaVantageNewsFetcher` class. Adjusted daily prices are downloaded separately with `yfinance` and converted to returns.

### 2. Sentiment Scoring
Article summaries are scored with **`ProsusAI/finbert`**, a BERT model fine-tuned on financial text, producing positive/negative/neutral probabilities per article — rather than relying on a general-purpose sentiment model.

### 3. Feature Engineering
Article-level scores are aggregated by day into positive, negative, neutral, and article-count measures, then transformed into lagged and rolling features (see below).

### 4. Modelling
A **time-ordered logistic regression** is trained to predict next-day price direction, with a chronological 80/20 train/test split (no shuffling — sentiment problems are sequential by nature).

### 5. Backtesting
The test period is backtested against buy-and-hold using a simple long/flat rule, with a configurable per-trade cost.

### 6. Visualization
Diagnostic plots cover sentiment over time, sentiment-return correlation, classification performance, feature coefficients, and the resulting equity curve.

## Feature and Target Definition
The model uses:

| Feature | Description |
| --- | --- |
| `sentiment_lag1` | Previous day's net sentiment |
| `sentiment_ma3` | 3-day moving average of net sentiment |
| `sentiment_momentum` | Current net sentiment minus the value 3 days earlier |
| `news_count` | Daily article count after the merge |

**Net sentiment** = `avg_positive - avg_negative`.
**Label** = `1` if the next trading day's adjusted-close return is positive, else `0`.
**Split**: chronological — first 80% of rows for training, remaining 20% for testing.

## How to Read the Results
Accuracy alone is a weak trading metric. Compare the strategy against the majority-class baseline and buy-and-hold, then inspect:

- Net return after costs
- Annualized volatility and Sharpe ratio
- Maximum drawdown
- Win rate and trade frequency
- Stability across different periods, tickers, and cost assumptions

The current backtester uses a simple long/flat rule: predicted direction `1` earns the next-day return, predicted direction `0` earns zero. It subtracts `0.0005` per position change. This does **not** model spreads, slippage, borrow fees, market impact, latency, partial fills, or portfolio sizing.

## Research Cautions
- **Data handling.** News timestamps must be aligned to market close to avoid lookahead bias, and forward-filled sentiment/article-count values are a modeling choice worth testing against alternatives (zero-fill, decay).
- **Validation is limited.** This uses a single chronological holdout — treat results as preliminary until confirmed with walk-forward validation across multiple regimes, with research decisions kept separate from the final test period.
- **Signal and metrics are simplified.** FinBERT sentiment isn't causal (it can reflect already-priced information or source bias), and reported metrics (252-day annualization, raw win rate) are basic, not risk-adjusted.

## Getting Started

### 1. Clone and install
```bash
git clone <your-repo-url>
cd stocks-sentimentanalysis-finbert
python -m venv venv
source venv/bin/activate       # macOS/Linux
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Configure environment
Create `.env` in the repository root:
```dotenv
AV_API_KEY=your_alpha_vantage_key
```
`config.py` loads this value with `python-dotenv` and raises an error if it's missing. Never commit `.env` or paste a live key into source code. If a key has ever been exposed, revoke or rotate it with the provider.

### 3. Run
```bash
python main.py
```
By default, `main.py` runs the pipeline for ticker **AAPL** over news dates **2025-01-01 to 2026-08-19** and price dates **2025-01-01 to 2026-08-20**. These defaults live in `config.py`, not `main.py` — to use a different ticker or date range, edit the corresponding values there; `main.py` simply reads them.

The first run may take time — Alpha Vantage is rate-limited and FinBERT downloads model weights from Hugging Face on first use. The script writes news to `data/news_data.csv`, downloads price data, opens several Matplotlib windows, prints classification metrics, and prints a side-by-side backtest summary.

For a smaller, faster experiment, use a shorter date range. The Alpha Vantage free tier may require longer delays.

## Disclaimer
This project is for education and research, not investment advice. Historical or simulated performance does not guarantee future results. Validate data, assumptions, and execution behavior independently before using any strategy with capital.