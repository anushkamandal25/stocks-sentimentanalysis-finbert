import time
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta



class AlphaVantageNewsFetcher:
    """
    A class for fetching stock news + sentiment data from Alpha Vantage.

    Attributes:
    - api_key (str): Alpha Vantage API key.
    """

    def __init__(self, api_key):
        """
        Initializes the AlphaVantageNewsFetcher object.

        Args:
        - api_key (str): Alpha Vantage API key.
        """
        self.api_key = api_key

    def fetch_news(self, ticker, start_date, end_date):
        """
        Fetches news articles for a ticker between two dates, one month at a time.

        Args:
        - ticker (str): Stock ticker symbol.
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
        - pd.DataFrame: DataFrame of news articles with Date, URL, Source, Author, Title, Description, Content.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_results = []
        current = start

        while current < end:
            window_end = min(current + relativedelta(months=1), end)
            time_from = current.strftime('%Y%m%dT0000')
            time_to = window_end.strftime('%Y%m%dT0000')

            url = (
                f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                f"&tickers={ticker}&time_from={time_from}&time_to={time_to}"
                f"&limit=1000&apikey={self.api_key}"
            )
            response = requests.get(url)

            if response.status_code == 200:
                feed = response.json().get('feed', [])
                print(f"{current.date()} to {window_end.date()}: {len(feed)} articles")
                for article in feed:
                    all_results.append({
                        'Date': article['time_published'][:8],
                        'URL': article['url'],
                        'Source': article['source'],
                        'Author': ", ".join(article['authors']) if article.get('authors') else None,
                        'Title': article['title'],
                        'Description': article['summary'],
                        'Content': article['summary'],
                    })
            else:
                print(f"Error {response.status_code} for window {current.date()}–{window_end.date()}: {response.text}")

            current = window_end
            time.sleep(12)  # free tier = 5 calls/min

        df = pd.DataFrame(all_results).drop_duplicates(subset='URL')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        return df

    def save_to_csv(self, df, path):
        """
        Saves a news DataFrame to CSV.

        Args:
        - df (pd.DataFrame): News DataFrame.
        - path (str): Output file path.
        """
        df.to_csv(path, index=False)


if __name__ == '__main__':
    # Example Usage:
    import config
    fetcher = AlphaVantageNewsFetcher(api_key=config.ALPHA_VANTAGE_API_KEY)
    news_df = fetcher.fetch_news("AAPL", "2025-01-01", "2026-08-19")
    fetcher.save_to_csv(news_df, "data/news_data.csv")
    print(news_df.head())