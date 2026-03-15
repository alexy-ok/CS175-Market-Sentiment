import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# load sentiment data
with open('data/processed/averaged_labels.json', 'r') as f:
    sentiment_data = json.load(f)

# month mapping for parsing dates from article IDs
month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def extract_date(article_id):
    """
    Extract date from article ID.
    Article IDs are in format: business/YYYY/mon/DD/title
    """
    parts = article_id.split('/')
    if len(parts) < 4:
        return None
    
    try:
        year = int(parts[1])
        month_str = parts[2].lower()
        if month_str not in month_map:
            return None
        month = month_map[month_str]
        day = int(parts[3])
        
        date = datetime(year, month, day)
        return date.date()
    except (ValueError, IndexError):
        return None

# group sentiment by date
sentiment_by_date = {}
for article_id, sentiment in sentiment_data.items():
    date = extract_date(article_id)
    if date:
        if date not in sentiment_by_date:
            sentiment_by_date[date] = []
        sentiment_by_date[date].append(sentiment)

# average sentiment per date
avg_sentiment = {date: sum(sentiments) / len(sentiments) for date, sentiments in sentiment_by_date.items()}

# create data frame for sentiment data
sentiment_df = pd.DataFrame(
    list(avg_sentiment.items()),
    columns=['Date', 'Sentiment']
)

sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
sentiment_df.set_index('Date', inplace=True)

# load stock data
stock_df = pd.read_csv('results/stock_results.csv', sep=',', index_col=0, parse_dates=True)

# compute daily returns
stock_df['Return'] = stock_df['Close'].pct_change()

# normalize index to date only (remove time and timezone)
stock_df.index = pd.to_datetime(stock_df.index, utc=True)
stock_df['Date'] = stock_df.index.date
stock_df.set_index('Date', inplace=True)

combined_df = sentiment_df.join(stock_df[['Return']], how='inner')
combined_df = combined_df.sort_index()

# smooth signals, 5-day roll
combined_df['Return_Smoothed'] = combined_df['Return'].rolling(5).mean()
combined_df['Sentiment_Smoothed'] = combined_df['Sentiment'].rolling(5).mean()

if not combined_df.empty:
    print(combined_df.head())

    # plot dual time series
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(
        combined_df.index,
        combined_df['Return_Smoothed'],
        color='blue',
        label='Stock Return',
        alpha=0.7
    )

    ax1.set_ylabel("Stock Return", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # sentiment axis
    ax2 = ax1.twinx()
    ax2.plot(
        combined_df.index,
        combined_df['Sentiment_Smoothed'],
        color='red',
        label='Sentiment Score',
        alpha=0.7
    )
    ax2.set_ylabel("Average Sentiment", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title("Market Sentiment vs Stock Returns Over Time")
    plt.grid(True, alpha=0.3)

    plt.savefig(
        'results/sentiment_stock_timeseries.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    
    # Save combined data
    combined_df.to_csv('results/sentiment_stock_combined.csv', index=False)
    print("Combined data saved to results/sentiment_stock_combined.csv")
    print("Plot saved to results/sentiment_stock_correlation.png")
else:
    print("No overlapping dates found between sentiment data and stock data.")