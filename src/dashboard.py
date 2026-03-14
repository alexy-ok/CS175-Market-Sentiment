import streamlit as st
from datetime import datetime, timedelta
import stock_data_collection as sdc
import yfinance as yf

st.title("Market Sentiment Dashboard")
st.write("This is a dashboard to visualize the market sentiment.")
st.write("Select a date range")
col1, col2 = st.columns(2)
with col1:
    date_from = st.date_input(
        "Date from", value=datetime.now() - timedelta(days=30), max_value=datetime.now()
    )
with col2:
    date_to = st.date_input(
        "Date to", value=datetime.now(), min_value=date_from, max_value=datetime.now()
    )

stock_data = sdc.StockDataCollector(yf.Ticker("VOO"))


st.write("Stock data")
fig = stock_data.plotStockData(start=date_from, end=date_to)
st.pyplot(fig)
