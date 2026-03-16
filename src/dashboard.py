import streamlit as st
from datetime import datetime, timedelta
import stock_data_collection as sdc
import yfinance as yf
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.title("Market Sentiment Dashboard")

month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def extract_date(article_id):
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

try:
    label_file = Path(__file__).parent.parent / "data" / "processed" / "averaged_labels.json"
    with open(label_file, 'r') as f:
        sentiment_data = json.load(f)
    
    all_dates = []
    for article_id in sentiment_data.keys():
        date = extract_date(article_id)
        if date:
            all_dates.append(date)
    
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        st.write("This is a dashboard to visualize the market sentiment.")
        st.info(f"📅 Available data range: **{min_date}** to **{max_date}**")
        st.write("Select a date range")
        
        default_start = max(min_date, max_date - timedelta(days=30))
        
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input(
                "Date from", 
                value=default_start,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            date_to = st.date_input(
                "Date to", 
                value=max_date,
                min_value=date_from,
                max_value=max_date
            )
    else:
        st.error("No valid dates found in sentiment data.")
        st.stop()
        
except FileNotFoundError:
    st.error("Sentiment data file not found. Please ensure 'data/processed/averaged_labels.json' exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading sentiment data: {str(e)}")
    st.stop()

stock_data = sdc.StockDataCollector(yf.Ticker("VOO"))

st.write("## Stock Data")
fig = stock_data.plotStockData(start=date_from, end=date_to)
st.pyplot(fig)

st.write("## Model Sentiment Analysis")

try:
    sentiment_by_date = {}
    for article_id, sentiment in sentiment_data.items():
        date = extract_date(article_id)
        if date and date_from <= date <= date_to:
            if date not in sentiment_by_date:
                sentiment_by_date[date] = []
            sentiment_by_date[date].append(sentiment)
    
    if sentiment_by_date:
        avg_sentiment = {date: sum(sentiments) / len(sentiments) 
                        for date, sentiments in sentiment_by_date.items()}
        
        sentiment_df = pd.DataFrame(
            list(avg_sentiment.items()),
            columns=['Date', 'Sentiment']
        ).sort_values('Date')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Sentiment", f"{sentiment_df['Sentiment'].mean():.2f}")
        with col2:
            st.metric("Min Sentiment", f"{sentiment_df['Sentiment'].min():.2f}")
        with col3:
            st.metric("Max Sentiment", f"{sentiment_df['Sentiment'].max():.2f}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sentiment_df['Date'], sentiment_df['Sentiment'], 
                color='#2E86AB', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Neutral (2)')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sentiment Score', fontsize=12)
        ax.set_title('Model Sentiment Analysis Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        with st.expander("View Sentiment Data"):
            st.dataframe(sentiment_df.set_index('Date'))
            st.write(f"Total articles analyzed: {sum(len(v) for v in sentiment_by_date.values())}")
            st.write(f"Days with data: {len(sentiment_by_date)}")
    else:
        st.warning(f"No sentiment data available for the selected date range ({date_from} to {date_to})")
        
except Exception as e:
    st.error(f"Error processing sentiment data: {str(e)}")
