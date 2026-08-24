import config
from news_fetcher import AlphaVantageNewsFetcher
from sentiment_analyzer import FinBertSentimentAnalyzer
from price_data import PriceDataFetcher
from feature_engineering import FeatureEngineer
from model import DirectionModel, FEATURES
from backtest import Backtester
from visualize import Visualizer


def main():
    # 1. News
    news_fetcher = AlphaVantageNewsFetcher(config.AV_API_KEY)
    news_df = news_fetcher.fetch_news(config.TICKER, config.NEWS_START_DATE, config.NEWS_END_DATE)
    news_fetcher.save_to_csv(news_df, config.NEWS_CSV_PATH)

    # 2. Sentiment (FinBERT)
    sentiment_analyzer = FinBertSentimentAnalyzer()
    news_df = sentiment_analyzer.add_sentiment_columns(news_df)

    # 3. Feature engineering
    fe = FeatureEngineer()
    daily_sentiment = fe.build_daily_sentiment(news_df)

    price_fetcher = PriceDataFetcher(config.TICKER)
    price_df = price_fetcher.fetch_price_data(config.PRICE_START_DATE, config.PRICE_END_DATE)

    merged = fe.merge_price_and_sentiment(price_df, daily_sentiment)
    merged = fe.add_features(merged)

    # 4. Plots
    viz = Visualizer()
    viz.plot_daily_sentiment(daily_sentiment)
    viz.plot_monthly_sentiment(daily_sentiment)
    viz.plot_monthly_volume(news_df)
    viz.plot_correlation_matrix(merged)
    viz.plot_sentiment_vs_return(merged)
    viz.plot_price_vs_sentiment(merged)
    viz.plot_return_by_bucket(merged)

    # 5. Model
    dm = DirectionModel()
    X_train, X_test, y_train, y_test, split = dm.train_test_split_by_time(merged)
    dm.train(X_train, y_train)
    preds = dm.evaluate(X_train, y_train, X_test, y_test)

    viz.plot_confusion_matrix(y_test, preds)
    viz.plot_feature_importance(dm.clf, FEATURES)

    # 6. Backtest
    bt = Backtester(cost_per_trade=0.0005)
    bt_df = bt.build_backtest_frame(merged, X_test, preds, dm.clf, y_test, split)
    bt_df = bt.add_strategy_returns(bt_df)
    viz.plot_equity_curve(bt_df)

    results = bt.summarize_results(bt_df)
    print(results)


if __name__ == "__main__":
    main()