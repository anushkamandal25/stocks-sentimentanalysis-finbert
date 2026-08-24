import numpy as np
import pandas as pd


class Backtester:
    """
    A class for running a simple vectorized backtest of the sentiment strategy
    against buy-and-hold, including transaction costs.

    Attributes:
    - cost_per_trade (float): Round-trip transaction cost as a fraction (e.g. 0.0005 = 5 bps).
    """

    def __init__(self, cost_per_trade=0.0005):
        """
        Initializes the Backtester object.

        Args:
        - cost_per_trade (float): Transaction cost per position change.
        """
        self.cost_per_trade = cost_per_trade

    def build_backtest_frame(self, merged_df, X_test, preds, clf, y_test, split):
        """
        Builds the test-period DataFrame with predictions attached.

        Args:
        - merged_df (pd.DataFrame): Feature-engineered DataFrame.
        - X_test (pd.DataFrame): Test features.
        - preds (np.ndarray): Predicted directions.
        - clf: Fitted classifier (for predict_proba).
        - y_test (pd.Series): True directions.
        - split (int): Index where the test set starts.

        Returns:
        - pd.DataFrame: bt_df with Date, Close, Next_Return, pred_direction, pred_proba, actual_direction.
        """
        bt_df = merged_df[['Date', 'Close', 'Next_Return']].iloc[split:].reset_index(drop=True)
        bt_df['pred_direction'] = preds
        bt_df['pred_proba'] = clf.predict_proba(X_test)[:, 1]
        bt_df['actual_direction'] = y_test.reset_index(drop=True)
        return bt_df

    def add_strategy_returns(self, bt_df):
        """
        Computes strategy and buy-and-hold returns/equity curves, net of costs.

        Args:
        - bt_df (pd.DataFrame): Output of build_backtest_frame().

        Returns:
        - pd.DataFrame: bt_df with strategy_return, buyhold_return, strategy_equity,
          buyhold_equity, strategy_return_net, strategy_equity_net columns added.
        """
        bt_df['strategy_return'] = np.where(bt_df['pred_direction'] == 1, bt_df['Next_Return'], 0.0)
        bt_df['buyhold_return'] = bt_df['Next_Return']

        bt_df['strategy_equity'] = (1 + bt_df['strategy_return']).cumprod()
        bt_df['buyhold_equity'] = (1 + bt_df['buyhold_return']).cumprod()

        position = bt_df['pred_direction'].values
        trades = np.abs(np.diff(position, prepend=position[0]))
        bt_df['strategy_return_net'] = bt_df['strategy_return'] - trades * self.cost_per_trade
        bt_df['strategy_equity_net'] = (1 + bt_df['strategy_return_net']).cumprod()
        return bt_df

    def compute_metrics(self, returns, freq=252):
        """
        Computes annualized return, volatility, Sharpe, max drawdown, and win rate.

        Args:
        - returns (pd.Series): A return series.
        - freq (int): Trading periods per year.

        Returns:
        - pd.Series: Summary metrics.
        """
        returns = returns.dropna()
        total_return = (1 + returns).prod() - 1
        ann_return = returns.mean() * freq
        ann_vol = returns.std() * np.sqrt(freq)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()

        win_rate = (returns > 0).mean()

        return pd.Series({
            'Total Return': total_return,
            'Annualized Return': ann_return,
            'Annualized Vol': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Win Rate': win_rate,
        })

    def summarize_results(self, bt_df):
        """
        Builds a side-by-side comparison table of strategy (net/gross) vs buy-and-hold.

        Args:
        - bt_df (pd.DataFrame): Output of add_strategy_returns().

        Returns:
        - pd.DataFrame: Metrics table.
        """
        return pd.DataFrame({
            'Strategy (net)': self.compute_metrics(bt_df['strategy_return_net']),
            'Strategy (gross)': self.compute_metrics(bt_df['strategy_return']),
            'Buy & Hold': self.compute_metrics(bt_df['buyhold_return']),
        }).round(4)


if __name__ == '__main__':
    # Example Usage:
    # bt = Backtester(cost_per_trade=0.0005)
    # bt_df = bt.build_backtest_frame(merged, X_test, preds, clf, y_test, split)
    # bt_df = bt.add_strategy_returns(bt_df)
    # print(bt.summarize_results(bt_df))
    pass