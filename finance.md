# Finance Concepts 

This document explains the financial ideas behind the sentiment-trading research pipeline. It focuses on the concepts implemented in the code rather than attempting to provide a complete finance textbook.

## 1. Financial News as Information

A stock price reflects market participants' expectations about a company's future cash flows, risk, and growth. News can change those expectations by providing information about:

- earnings and guidance
- products, technology, and competition
- regulation and litigation
- mergers, acquisitions, and capital decisions
- macroeconomic or geopolitical conditions

The project tests whether the tone of financial news contains information about the stock's **next trading-day return**. This is an information-signal hypothesis, not an assumption that every positive article should produce a higher price.

### Why the relationship may be weak

News may already be incorporated into the price before the data is collected. Articles can also be repetitive, ambiguous, delayed, or unrelated to the stock's short-term movement. A sentiment score therefore represents a noisy research feature, not a causal explanation of returns.

This connects to two complementary ideas:

- **Efficient-market reasoning:** public information is quickly reflected in prices, making persistent excess returns difficult to obtain.
- **Behavioral-finance reasoning:** investors may underreact, overreact, herd, or trade on emotion, creating short-lived patterns after news arrives.

The project empirically tests this tension instead of assuming either theory is always correct.

## 2. Sentiment as a Quantitative Signal

FinBERT assigns each article probabilities for positive, negative, and neutral financial sentiment. For each calendar day, the project averages article scores so that a day with many articles is not automatically treated as more positive or negative merely because it contains more text.

The daily net sentiment feature is:

$$
S_t = \overline{P}_{t}^{+} - \overline{P}_{t}^{-}
$$

where:

- $\overline{P}_{t}^{+}$ is the average positive probability on day $t$;
- $\overline{P}_{t}^{-}$ is the average negative probability on day $t$;
- $S_t > 0$ indicates relatively more positive than negative tone;
- $S_t < 0$ indicates relatively more negative than positive tone.

The daily article count is retained as `news_count`. It can act as a rough measure of news intensity, although it does not distinguish important events from duplicated or low-value coverage.

## 3. Prices, Adjustments, and Returns

The project downloads daily prices with `yfinance` using adjusted prices. Adjustments help account for events such as stock splits and dividends, making historical price comparisons more meaningful.

The one-period simple return is:

$$
R_t = \frac{P_t}{P_{t-1}} - 1
$$

where $P_t$ is the adjusted closing price at time $t$. The code then defines the prediction target as the following trading day's return:

$$
R_{t+1} = \frac{P_{t+1}}{P_t} - 1
$$

The classification label is:

$$
y_t =
\begin{cases}
1, & \text{if } R_{t+1} > 0 \\
0, & \text{otherwise}
\end{cases}
$$

Thus, the model is predicting **direction**, not the size of the return. A correct direction prediction can still correspond to a very small gain, while an incorrect prediction can coincide with a large loss.

## 4. Lagged Features and Information Timing

A trading feature must only use information available before the return it is intended to predict. The project uses:

| Feature | Financial interpretation |
| --- | --- |
| `sentiment_lag1` | Prior day's net sentiment, reducing direct overlap with the target day |
| `sentiment_ma3` | Three-day moving average that smooths one-day noise |
| `sentiment_momentum` | Change in sentiment relative to three days earlier |
| `news_count` | Approximate daily news intensity |

The lagged sentiment is:

$$
S_{t-1}
$$

The three-day moving average is:

$$
MA_{3,t} = \frac{S_t + S_{t-1} + S_{t-2}}{3}
$$

The sentiment momentum feature is:

$$
M_t = S_t - S_{t-3}
$$

These features are simple proxies for persistence, trend, and information intensity. They should not be interpreted as established risk factors.

### Lookahead bias

Lookahead bias occurs when a model uses information that would not have been known at the time of the simulated trade. News timestamps must be aligned with market hours and market close. For example, a story released after the close should generally affect the next tradable session, not the return that ended before publication.

The current project is a research prototype and does not fully model intraday publication times, exchange calendars, or execution delays. These are important improvements before treating results as realistic.

## 5. Time-Series Model Validation

Financial observations are ordered in time and are often dependent across adjacent periods. The project therefore uses a chronological split:

- first 80% of observations: training set;
- final 20%: test set;
- no random shuffling.

This preserves the basic direction of information flow. Random train-test splitting could place future observations in the training set and produce overly optimistic results.

A single holdout is still limited. Stronger research would use walk-forward or expanding-window validation across multiple market regimes, while keeping a final untouched test period for the last evaluation.

## 6. Portfolio and Strategy Returns

The backtest converts the predicted direction into a simple **long/flat** position:

- prediction `1`: hold the asset for the next period;
- prediction `0`: hold cash and earn zero in the simplified model.

The gross strategy return is:

$$
R^{strategy}_t = position_t \times R_{t+1}
$$

where $position_t \in \{0,1\}$. The buy-and-hold benchmark uses the asset return on every test-period observation:

$$
R^{BH}_t = R_{t+1}
$$

Equity is compounded through time:

$$
E_t = E_{t-1}(1 + R_t), \qquad E_0 = 1
$$

Compounding is important because a sequence of returns is not equivalent to simply adding percentage changes.

## 7. Transaction Costs

Trading is not free. The backtester charges a cost whenever the predicted position changes. With cost $c$ and position change $\Delta position_t$:

$$
R^{net}_t = R^{gross}_t - c \left|position_t - position_{t-1}\right|
$$

The configured cost is `0.0005`, or 5 basis points, per position change. One basis point is one hundredth of one percentage point:

$$
1\text{ bp} = 0.01\% = 0.0001
$$

The current cost model is intentionally simple. It excludes bid-ask spread, slippage, market impact, commissions, taxes, latency, borrow costs, and portfolio financing. A strategy that only works before realistic costs is not a convincing trading strategy.

## 8. Performance Measures

The backtester reports several complementary metrics.

### Total return

For a return series $R_1, \ldots, R_T$:

$$
Total\ Return = \prod_{t=1}^{T}(1+R_t) - 1
$$

This is the cumulative growth over the evaluated period.

### Annualized return

The implementation estimates annualized return as:

$$
Annualized\ Return \approx \bar{R} \times 252
$$

where $252$ is the approximate number of US trading days in a year. This is a simple arithmetic annualization and is not the same as a compounded annual growth rate.

### Annualized volatility

The implementation estimates volatility as:

$$
Annualized\ Volatility = \sigma_R \sqrt{252}
$$

where $\sigma_R$ is the standard deviation of periodic returns. Volatility measures dispersion, not whether returns are favorable.

### Sharpe ratio

With a simplified zero risk-free rate, the reported Sharpe ratio is:

$$
Sharpe = \frac{Annualized\ Return}{Annualized\ Volatility}
$$

A higher value indicates more average return per unit of measured volatility. The interpretation is limited because the implementation does not subtract a time-varying risk-free rate and uses a simple annualization convention.

### Maximum drawdown

Drawdown measures decline from a previous equity peak. If $E_t$ is equity and $H_t = \max_{u \leq t} E_u$ is the running high-water mark:

$$
Drawdown_t = \frac{E_t - H_t}{H_t}
$$

Maximum drawdown is the most negative drawdown during the period. It gives a direct view of historical peak-to-trough pain that average volatility may hide.

### Win rate

Win rate is the fraction of periods with positive returns:

$$
Win\ Rate = \frac{\#\{t : R_t > 0\}}{T}
$$

Win rate alone is not sufficient. A strategy can win often and still lose money if its losses are larger than its gains. It can also have a low win rate while remaining profitable if winners are sufficiently large.

## 9. How to Interpret Results

The strongest comparison is not model accuracy alone. Review:

1. Strategy net return versus gross return.
2. Strategy performance versus buy-and-hold.
3. Sharpe ratio and volatility.
4. Maximum drawdown.
5. Number of trades and sensitivity to transaction costs.
6. Stability across tickers, date ranges, and market regimes.
7. Performance on a genuinely out-of-sample period.

A positive result may reflect chance, data revisions, survivorship bias, timing leakage, or overfitting. A useful signal should remain credible after controls, alternative assumptions, and realistic execution costs.

## 10. Scope and Limitations

This project does not implement a complete asset-pricing model, portfolio optimization framework, derivatives model, or risk-management system. It also does not estimate statistical significance, confidence intervals, factor exposures, turnover-adjusted capacity, or risk-adjusted performance against a formal multi-factor benchmark.

The appropriate conclusion from this project is therefore modest: it is a transparent environment for investigating whether news sentiment is associated with next-day direction under stated assumptions. It is not evidence that the strategy will generate profits with real capital.
