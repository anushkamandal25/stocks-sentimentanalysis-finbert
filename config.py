import os
from dotenv import load_dotenv

load_dotenv()  # reads .env and loads it into environment variables

AV_API_KEY = os.getenv("AV_API_KEY", "")
TICKER = "AAPL"
NEWS_START_DATE = "2025-01-01"
NEWS_END_DATE = "2026-08-19"
PRICE_START_DATE = "2025-01-01"
PRICE_END_DATE = "2026-08-20"
NEWS_CSV_PATH = "data/news_data.csv"

if not AV_API_KEY:
    raise EnvironmentError("AV_API_KEY is missing. Add it to your .env file.")