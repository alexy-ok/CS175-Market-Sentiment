"""
Script to collect stock market data from Yahoo Finance API.
"""
import datetime
import matplotlib.pyplot as plt

class StockDataCollector:
    def __init__(self, symbol):
        self.symbol = symbol # e.g. yf.Ticker("VOO")
    
    def grabDataFromRange(self, start_date, end_date):
        """Grab OHLCV data within a given timeframe."""
        data = self.symbol.history(start=start_date, end=end_date)
        return data
    
    def grabDataFromPeriod(self, period):
         """Grab OHLCV data within a given period."""
         data = self.symbol.history(period=period)
         return data.to_string()

    def grabMetadata(self):
        """Grab the stock's metadata.""" 
        for key, value in self.symbol.info.items():
            print(f"{key}: {value}")
    
    def plotStockData(self, start=None, end=None, period=None):
        """Visualize the stock's data in a tabular format.""" 
        if period:
            data = self.grabDataFromPeriod(period)
            title = f"{self.symbol} Stock Price ({period})"
        elif start and end:
            data = self.grabDataFromRange(start, end)
            title = f"{self.symbol.ticker} Stock Price ({start} to {end})"

        data['Close'].plot(title=title)
        plt.xlabel("Date")
        plt.ylabel("Close Price (USD)")
        plt.show()

if __name__ == '__main__':
    import yfinance as yf

    stockData = StockDataCollector(yf.Ticker("VOO"))
    
    start_date = datetime.datetime(2026, 2, 10)
    end_date = datetime.datetime(2026, 2, 17)
    print(stockData.grabDataFromRange(start_date, end_date))

