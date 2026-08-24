import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


class Visualizer:
    """
    A class holding all plotting functions used for exploring sentiment
    and evaluating the model/backtest.
    """

    def plot_daily_sentiment(self, daily_sentiment_df):
        """
        Plots daily average positive/neutral/negative sentiment as grouped bars.

        Args:
        - daily_sentiment_df (pd.DataFrame): Output of FeatureEngineer.build_daily_sentiment().
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        width = 0.2
        ax.bar(daily_sentiment_df.index - pd.DateOffset(days=width), daily_sentiment_df['avg_positive'], width=width, label='Positive', color='green')
        ax.bar(daily_sentiment_df.index, daily_sentiment_df['avg_neutral'], width=width, label='Neutral', color='orange')
        ax.bar(daily_sentiment_df.index + pd.DateOffset(days=width), daily_sentiment_df['avg_negative'], width=width, label='Negative', color='red')
        ax.set_title('Daily Average Sentiment')
        ax.legend()
        ax.grid(True, axis='y', linestyle='--')
        plt.show()

    def plot_monthly_sentiment(self, daily_sentiment_df):
        """
        Plots monthly average sentiment as grouped bars.

        Args:
        - daily_sentiment_df (pd.DataFrame): Output of FeatureEngineer.build_daily_sentiment().
        """
        monthly = daily_sentiment_df[['avg_positive', 'avg_neutral', 'avg_negative']].resample('ME').mean()
        fig, ax = plt.subplots(figsize=(14, 6))
        width = 0.25
        x = np.arange(len(monthly))
        ax.bar(x - width, monthly['avg_positive'], width=width, label='Positive', color='seagreen')
        ax.bar(x, monthly['avg_neutral'], width=width, label='Neutral', color='goldenrod')
        ax.bar(x + width, monthly['avg_negative'], width=width, label='Negative', color='firebrick')
        ax.set_xticks(x)
        ax.set_xticklabels(monthly.index.strftime('%b %Y'), rotation=45, ha='right')
        ax.set_title('Monthly Average Sentiment')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_monthly_volume(self, news_df):
        """
        Plots monthly news article volume.

        Args:
        - news_df (pd.DataFrame): Raw news DataFrame with a Date column.
        """
        monthly_volume = news_df.groupby(news_df['Date'].dt.to_period('M')).size()
        fig, ax = plt.subplots(figsize=(14, 4))
        monthly_volume.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Monthly News Volume')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, merged_df):
        """
        Plots a correlation heatmap of sentiment and return features.

        Args:
        - merged_df (pd.DataFrame): Feature-engineered DataFrame.
        """
        corr = merged_df[['sentiment_net', 'sentiment_lag1', 'sentiment_ma3', 'sentiment_momentum',
                           'news_count', 'Return', 'Next_Return']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_sentiment_vs_return(self, merged_df):
        """
        Plots a regression scatter of lagged sentiment vs next-day return.

        Args:
        - merged_df (pd.DataFrame): Feature-engineered DataFrame.
        """
        plt.figure(figsize=(8, 6))
        sns.regplot(x='sentiment_lag1', y='Next_Return', data=merged_df, scatter_kws={'alpha': 0.4})
        plt.title('Lagged Sentiment vs Next-Day Return')
        plt.show()

    def plot_price_vs_sentiment(self, merged_df):
        """
        Plots price and 3-day sentiment moving average on a dual-axis chart.

        Args:
        - merged_df (pd.DataFrame): Feature-engineered DataFrame.
        """
        fig, ax1 = plt.subplots(figsize=(14, 6))
        line1, = ax1.plot(merged_df['Date'], merged_df['Close'], color='steelblue', label='Close')
        ax2 = ax1.twinx()
        line2, = ax2.plot(merged_df['Date'], merged_df['sentiment_ma3'], color='darkorange', alpha=0.7, label='Sentiment (3d MA)')
        lines = [line1, line2]
        ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
        plt.title('Price vs News Sentiment')
        fig.tight_layout()
        plt.show()

    def plot_return_by_bucket(self, merged_df):
        """
        Plots next-day return distribution grouped by sentiment bucket.

        Args:
        - merged_df (pd.DataFrame): Feature-engineered DataFrame with sentiment_bucket column.
        """
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='sentiment_bucket', y='Next_Return', data=merged_df, hue='sentiment_bucket', palette='coolwarm', legend=False)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Next-Day Return by Sentiment Bucket')
        plt.show()

    def plot_confusion_matrix(self, y_test, preds):
        """
        Plots a confusion matrix for the classifier's test predictions.

        Args:
        - y_test (pd.Series): True labels.
        - preds (np.ndarray): Predicted labels.
        """
        ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Blues')
        plt.title('Confusion Matrix — Logistic Regression')
        plt.show()

    def plot_feature_importance(self, clf, features):
        """
        Plots logistic regression coefficients as a horizontal bar chart.

        Args:
        - clf (LogisticRegression): Fitted classifier.
        - features (list): Feature names in the same order used for training.
        """
        importance = pd.Series(clf.coef_[0], index=features).sort_values()
        importance.plot(kind='barh', figsize=(6, 4), color='teal')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title('Logistic Regression Coefficients')
        plt.tight_layout()
        plt.show()

    def plot_equity_curve(self, bt_df):
        """
        Plots strategy (net of costs) vs buy-and-hold equity curves.

        Args:
        - bt_df (pd.DataFrame): Output of Backtester.add_strategy_returns().
        """
        plt.figure(figsize=(12, 6))
        plt.plot(bt_df['Date'], bt_df['strategy_equity_net'], label='Sentiment Strategy (net)')
        plt.plot(bt_df['Date'], bt_df['buyhold_equity'], label='Buy & Hold')
        plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Strategy vs Buy & Hold')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example Usage:
    # viz = Visualizer()
    # viz.plot_equity_curve(bt_df)
    pass