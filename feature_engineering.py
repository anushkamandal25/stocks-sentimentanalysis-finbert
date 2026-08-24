import pandas as pd


class FeatureEngineer:
    """
    A class for merging price and sentiment data and building model features.
    """

    def build_daily_sentiment(self, news_df):
        """
        Aggregates article-level sentiment scores into one row per day.

        Args:
        - news_df (pd.DataFrame): News DataFrame with Sent_positive/negative/neutral columns.

        Returns:
        - pd.DataFrame: Daily average sentiment indexed by Date, with news_count column.
        """
        grouped = news_df.groupby('Date')[['Sent_positive', 'Sent_negative', 'Sent_neutral']].agg(['mean', 'count'])
        grouped.columns = ['avg_positive', 'count_pos', 'avg_negative', 'count_neg', 'avg_neutral', 'count_neu']
        grouped['news_count'] = grouped['count_pos']
        grouped = grouped[['avg_positive', 'avg_negative', 'avg_neutral', 'news_count']]
        grouped.index = pd.to_datetime(grouped.index)
        return grouped

    def merge_price_and_sentiment(self, price_df, daily_sentiment_df):
        """
        Merges price data with daily sentiment, forward-filling on non-news days.

        Args:
        - price_df (pd.DataFrame): Price DataFrame from PriceDataFetcher.
        - daily_sentiment_df (pd.DataFrame): Output of build_daily_sentiment().

        Returns:
        - pd.DataFrame: Merged DataFrame with sentiment_net column added.
        """
        sentiment_reset = daily_sentiment_df.reset_index()
        merged = pd.merge(
            price_df,
            sentiment_reset[['Date', 'avg_positive', 'avg_negative', 'avg_neutral', 'news_count']],
            on='Date',
            how='left'
        )
        merged[['avg_positive', 'avg_negative', 'avg_neutral', 'news_count']] = \
            merged[['avg_positive', 'avg_negative', 'avg_neutral', 'news_count']].ffill().fillna(0)
        merged['sentiment_net'] = merged['avg_positive'] - merged['avg_negative']
        return merged

    def add_features(self, merged_df):
        """
        Adds lagged/rolling sentiment features and the prediction target.

        Args:
        - merged_df (pd.DataFrame): Output of merge_price_and_sentiment().

        Returns:
        - pd.DataFrame: DataFrame with sentiment_lag1, sentiment_ma3, sentiment_momentum,
          direction, and sentiment_bucket columns added; NaN rows dropped.
        """
        merged_df['sentiment_lag1'] = merged_df['sentiment_net'].shift(1)
        merged_df['sentiment_ma3'] = merged_df['sentiment_net'].rolling(3).mean()
        merged_df['sentiment_momentum'] = merged_df['sentiment_net'] - merged_df['sentiment_net'].shift(3)
        merged_df['direction'] = (merged_df['Next_Return'] > 0).astype(int)

        merged_df = merged_df.dropna(
            subset=['sentiment_lag1', 'sentiment_ma3', 'sentiment_momentum', 'Next_Return']
        ).reset_index(drop=True)

        merged_df['sentiment_bucket'] = pd.cut(
            merged_df['sentiment_lag1'], bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        return merged_df


if __name__ == '__main__':
    # Example Usage:
    # fe = FeatureEngineer()
    # daily_sentiment = fe.build_daily_sentiment(news_df)
    # merged = fe.merge_price_and_sentiment(price_df, daily_sentiment)
    # merged = fe.add_features(merged)
    pass