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

# load stock data
stock_df = pd.read_csv('results/stock_results.csv', sep=',', index_col=0, parse_dates=True)

# compute daily returns
stock_df['Return'] = stock_df['Close'].pct_change()

# normalize index to date only (remove time and timezone)
stock_df.index = pd.to_datetime(stock_df.index, utc=True)
stock_df['Date'] = stock_df.index.date
stock_df.set_index('Date', inplace=True)

print(f"Stock data dates: {stock_df.index[:5]}")
print(f"Sentiment dates: {list(avg_sentiment.keys())[:5]}")

# create combined DataFrame with sentiment and next day return
data = []
for date, sent in avg_sentiment.items():
    next_date = date + pd.Timedelta(days=1)
    if next_date in stock_df.index:
        ret = stock_df.loc[next_date, 'Return']
        data.append({'Date': date, 'Sentiment': sent, 'Next_Return': ret})

combined_df = pd.DataFrame(data)

if not combined_df.empty:
    # Compute correlation
    correlation = combined_df['Sentiment'].corr(combined_df['Next_Return'])
    print(f"Correlation between average daily sentiment and next day's stock return: {correlation:.4f}")
    
    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_df['Sentiment'], combined_df['Next_Return'], alpha=0.6)
    plt.xlabel('Average Daily Sentiment (0-4)')
    plt.ylabel('Next Day Stock Return')
    plt.title(f'Sentiment vs Next Day Stock Return\nCorrelation: {correlation:.4f}')
    plt.grid(True, alpha=0.3)
    # plt.savefig('../results/sentiment_stock_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save combined data
    combined_df.to_csv('../results/sentiment_stock_combined.csv', index=False)
    print("Combined data saved to ../results/sentiment_stock_combined.csv")
    # print("Plot saved to ../results/sentiment_stock_correlation.png")
else:
    print("No matching dates found between sentiment data and stock data.")