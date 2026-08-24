import yfinance as yf


class PriceDataFetcher:
    """
    A class for fetching and preparing historical stock price data.

    Attributes:
    - ticker (str): Stock ticker symbol.
    """

    def __init__(self, ticker):
        """
        Initializes the PriceDataFetcher object.

        Args:
        - ticker (str): Stock ticker symbol.
        """
        self.ticker = ticker

    def fetch_price_data(self, start_date, end_date):
        """
        Downloads daily price data and computes return columns.

        Args:
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.

        Returns:
        - pd.DataFrame: Price data with Return and Next_Return columns added.
        """
        price = yf.download(self.ticker, start=start_date, end=end_date, auto_adjust=True)
        price = price.reset_index()
        price.columns = [c if isinstance(c, str) else c[0] for c in price.columns]
        price['Return'] = price['Close'].pct_change()
        price['Next_Return'] = price['Return'].shift(-1)
        return price


if __name__ == '__main__':
    # Example Usage:
    fetcher = PriceDataFetcher("AAPL")
    price_df = fetcher.fetch_price_data("2025-01-01", "2026-08-20")
    print(price_df.head())