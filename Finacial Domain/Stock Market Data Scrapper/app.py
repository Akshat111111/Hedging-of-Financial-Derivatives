import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
import json

# Function to get historical stock data
@st.cache_data(ttl=600)
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        st.error("No data found. Please check the ticker symbol and date range.")
        return None
    stock_data.reset_index(inplace=True)
    return stock_data

# Function to get SEC ticker symbols and company names from provided JSON
@st.cache_data(ttl=3600)
def get_sec_ticker_info():
    try:
        with open("ticker.json", "r") as file:
            data = json.load(file)
            df = pd.DataFrame(data).transpose()
            df.columns = ["CIK", "Ticker", "Title"]
            return df
    except Exception as e:
        st.error(f"Error reading ticker information: {e}")
        return None

# Streamlit interface

# Sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Stock Price Data Scraper", "Ticker Symbol Information"])

if menu == "Stock Price Data Scraper":
    st.header("Stock Price Data Scraper")
    ticker = st.text_input("Enter the ticker symbol of the company:", value="AAPL")
    start_date = st.date_input("Select start date:", value=datetime(2020, 1, 1))
    end_date = st.date_input("Select end date:", value=datetime.today())

    if st.button("Get Stock Data"):
        df = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df is not None:
            st.write(f"Historical stock data for {ticker} from {start_date} to {end_date}")
            st.dataframe(df.style.set_table_attributes('style="text-align: center; width: auto;"'))
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name=f"{ticker}_historical_data.csv", mime='text/csv')

elif menu == "Ticker Symbol Information":
    st.header("Ticker Symbol Information")
    ticker_info_df = get_sec_ticker_info()
    if ticker_info_df is not None:
        search_query = st.text_input("Search for a company name or ticker symbol:")
        if search_query:
            filtered_df = ticker_info_df[ticker_info_df.apply(lambda row: search_query.lower() in row['Ticker'].lower() or search_query.lower() in row['Title'].lower(), axis=1)]
            st.dataframe(filtered_df[['Title', 'Ticker']].style.set_table_attributes('style="text-align: center; width: auto;"'))
        else:
            st.dataframe(ticker_info_df[['Title', 'Ticker']].style.set_table_attributes('style="text-align: center; width: auto;"'))
            csv = ticker_info_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name=f"stock_ticker_information.csv",mime='text/csv')
