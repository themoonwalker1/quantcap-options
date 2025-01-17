import json
from datetime import datetime, timedelta

import requests
from dateutil.relativedelta import relativedelta


def download_sentiment_data(symbols: list[str], interval: str = "1d") -> None:
    """
    Downloads sentiment data from StockGeist API and stores it in JSON files.
    """
    start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-01-10", "%Y-%m-%d")

    def fetch_data(url, params, headers):
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def fetch_sentiment_data(symbols, start_date, end_date, interval, data_type):
        base_url = "https://api.stockgeist.ai"
        asset_class = "stock"
        location = "us"
        url = f"{base_url}/{asset_class}/{location}/hist/{data_type}-metrics"
        all_data = {}

        current_date = start_date
        while current_date <= end_date:
            month_start = current_date.replace(day=1)
            next_month = month_start + relativedelta(months=1)
            month_end = next_month - timedelta(days=1)
            if month_end > end_date:
                month_end = end_date

            params = {
                "symbols": ','.join(symbols),
                "start": month_start.strftime("%Y-%m-%d"),
                "end": month_end.strftime("%Y-%m-%d"),
                "timeframe": interval
            }

            headers = {
                "token": "b2s0InjGYj5JJN5SvLf2gWjiAorevDVd"
            }  

            data = fetch_data(url, params, headers)
            data = data.get("data", [])
            for symbol in symbols:
                if symbol not in all_data:
                    all_data[symbol] = []
                all_data[symbol].extend(data.get(symbol, []))
            current_date = next_month
        # Store the fetched data in corresponding files
        file_name = f"sentiment_data_{data_type}_stocks_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        with open(file_name, "w") as f:
            json.dump(all_data, f)

    fetch_sentiment_data(symbols, start_date, end_date, interval, "message")
    fetch_sentiment_data(symbols, start_date, end_date, interval, "article")



if __name__ == "__main__":
    print("Downloading sentiment data...")
    download_sentiment_data(["GOOGL", "AAPL"], "1d")
    print("Sentiment data downloaded successfully.")
