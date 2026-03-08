import datetime
import matplotlib.pyplot as plt


class StockDataCollector:
    def __init__(self, symbol):
        self.symbol = symbol

    def grabDataFromRange(self, start_date, end_date):
        data = self.symbol.history(start=start_date, end=end_date)
        return data
    def grabDataFromPeriod(self, period):
        data = self.symbol.history(period=period)
        return data.to_string()

    def grabMetadata(self):
        for key, value in self.symbol.info.items():
            print(f"{key}: {value}")

    def plotStockData(self, start=None, end=None, period=None):
        if period:
            data = self.grabDataFromPeriod(period)
            title = f"{self.symbol} Stock Price ({period})"
        elif start and end:
            data = self.grabDataFromRange(start, end)
            title = f"{self.symbol.ticker} Stock Price ({start} to {end})"

        fig, ax = plt.subplots(figsize=(10, 6))
        data["Close"].plot(title=title, ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        return fig


if __name__ == "__main__":
    import yfinance as yf

    stockData = StockDataCollector(yf.Ticker("VOO"))

    start_date = datetime.datetime(2026, 2, 10)
    end_date = datetime.datetime(2026, 2, 17)
    print(stockData.grabDataFromRange(start_date, end_date))
