# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:32:23 2024

@author: Plutonium
"""

#%%
import os
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%%

sp500 = yf.Ticker("^GSPC")
sp500_data = sp500.history(period="1mo")
print(sp500_data.head())



#%%

import requests
from bs4 import BeautifulSoup

# List of Yahoo Finance pages to scrape
urls = [
    "https://finance.yahoo.com/markets/stocks/most-active/",
    "https://finance.yahoo.com/markets/stocks/trending/",
    "https://finance.yahoo.com/markets/stocks/gainers/",
    "https://finance.yahoo.com/markets/stocks/losers/",
    "https://finance.yahoo.com/markets/stocks/52-week-gainers/",
    "https://finance.yahoo.com/markets/stocks/52-week-losers/",
]

def get_tickers_from_page(url):
    """
    Scrape tickers from a Yahoo Finance page.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Locate the table containing tickers
    table = soup.find("table")
    if not table:
        return []

    # Extract tickers from the table
    tickers = []
    for row in table.find_all("tr")[1:]:  # Skip header row
        cells = row.find_all("td")
        if cells:
            tickers.append(cells[0].text.strip())  # The ticker is in the first column

    return tickers

# Collect tickers from all pages
all_tickers = {}
for url in urls:
    page_name = url.split("/")[-2]
    tickers = get_tickers_from_page(url)
    all_tickers[page_name] = tickers

# Print the results
for category, tickers in all_tickers.items():
    print(f"{category.capitalize()} Tickers ({len(tickers)}): {tickers}\n")
#%%


# Define the tickers for the DOW 30
dow30_tickers = [
    "AAPL", "MSFT", "IBM", "VZ", "JNJ", "WMT", "PG", "KO", "CSCO", "TRV",
    "INTC", "MMM", "NKE", "DIS", "V", "JPM", "GS", "AXP", "BA", "CAT",
    "CVX", "XOM", "MRK", "AMGN", "HD", "HON", "UNH", "WBA"
]


# Add the target stock (example: "TSLA")
target_ticker = ["TSLA"]
# all_tickers = dow30_tickers + target_ticker
all_tickers = dow30_tickers

# Fetch historical data
data = yf.download(all_tickers, start="2010-01-01", end="2024-11-29", group_by="ticker", auto_adjust=False)
print(data.head())

#%%
# Step 1: Check for missing days in the raw data
raw_missing_days = data.index.to_series().diff()[data.index.to_series().diff() > pd.Timedelta(days=1)]

print(f"Missing days in raw data: {len(raw_missing_days)}")
if not raw_missing_days.empty:
    print("Details of missing days in raw data:")
    print(raw_missing_days)


        

#%%
from technical_indicators import *

processed_data = {}
ticker_count = 0
for ticker in all_tickers:
    print(ticker)
    ticker_count += 1
    try:
        # Extract relevant columns
        ticker_data = data[ticker][["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
        
        # Calculate additional features
        ticker_data["Daily Change %"] = ticker_data["Close"].pct_change() * 100
        ticker_data["Volatility"] = ticker_data["High"] - ticker_data["Low"]  # Daily range
        ticker_data["Gap"] = ticker_data["Close"] - ticker_data["Open"]       # Gap between close and open
        
        # Add technical indicators
        # ticker_data["MACD"], ticker_data["MACD Signal"] = calculate_macd(ticker_data)
        # ticker_data["Bollinger Upper"], ticker_data["Bollinger Middle"], ticker_data["Bollinger Lower"] = calculate_bollinger_bands(ticker_data)
        # ticker_data["RSI"] = calculate_rsi(ticker_data)
        # ticker_data["CCI"] = calculate_cci(ticker_data)
        # ticker_data["ADX"] = calculate_adx(ticker_data)
        # ticker_data["SMA_30"] = calculate_sma(ticker_data, period=30)
        # ticker_data["SMA_60"] = calculate_sma(ticker_data, period=60)
        
        
        # Find rows with NaN values
        nan_rows = ticker_data[ticker_data.isna().any(axis=1)]
    
        # Print rows with NaN values
        print("Rows with NaN values:")
        print(nan_rows)
    
        # Optionally, print the tickers causing NaN values
        nan_columns = ticker_data.columns[ticker_data.isna().any()]
        print("\nTickers with NaN values:")
        print(nan_columns)
    
    
        # Drop NaN rows caused by pct_change or other calculations
        ticker_data.dropna(inplace=True)
    
        # Store the processed data
        processed_data[ticker] = ticker_data

    except KeyError:
        print(f"No data found for {ticker}")
        



feature_rich_data = pd.concat(
    [df for df in processed_data.values()],  # Include all columns for each ticker
    axis=1,
    keys=processed_data.keys()
)

print(feature_rich_data.head())

print("Unique Tickers : ", ticker_count)
print("Unique Column Types Per Ticker : ", feature_rich_data.shape[1]/ticker_count)
#%%

# Fetch VIX data
vix_data = yf.download("^VIX", start="2010-01-01", end="2023-01-01")
vix_data["Daily Change %"] = vix_data["Close"].pct_change() * 100  # Daily percentage change
vix_data.dropna(inplace=True)

# Add VIX to the existing pipeline
# Align VIX with your stock data by date
feature_rich_data["VIX Close"] = vix_data["Close"].reindex(feature_rich_data.index, method="pad")
feature_rich_data["VIX Daily Change %"] = vix_data["Daily Change %"].reindex(feature_rich_data.index, method="pad")

print(feature_rich_data.head())


#%%
# Fetch stock split and dividend history for a specific ticker
# These might be good features to add to the data set later.
ticker = yf.Ticker("AAPL")  # Replace with the desired ticker
splits = ticker.splits
dividends = ticker.dividends

print("Stock Split History:")
print(splits)

print("Dividend History:")
print(dividends)








#%%
# Ensure the index is in datetime format
feature_rich_data.index = pd.to_datetime(feature_rich_data.index)

# Add temporal features
feature_rich_data["Day of Week"] = feature_rich_data.index.weekday  # 0 = Monday, 6 = Sunday
feature_rich_data["Day of Month"] = feature_rich_data.index.day
feature_rich_data["Day of Year"] = feature_rich_data.index.dayofyear

# Perform sine and cosine encoding for cyclical features
feature_rich_data["Day of Week Sin"] = np.sin(2 * np.pi * feature_rich_data["Day of Week"] / 7)
feature_rich_data["Day of Week Cos"] = np.cos(2 * np.pi * feature_rich_data["Day of Week"] / 7)

feature_rich_data["Day of Month Sin"] = np.sin(2 * np.pi * feature_rich_data["Day of Month"] / 31)
feature_rich_data["Day of Month Cos"] = np.cos(2 * np.pi * feature_rich_data["Day of Month"] / 31)

feature_rich_data["Day of Year Sin"] = np.sin(2 * np.pi * feature_rich_data["Day of Year"] / 365)
feature_rich_data["Day of Year Cos"] = np.cos(2 * np.pi * feature_rich_data["Day of Year"] / 365)

# Drop the original day columns if you don't want them
# feature_rich_data.drop(["Day of Week", "Day of Month", "Day of Year"], axis=1, inplace=True)

# Print the updated dataset
print(feature_rich_data.head())


fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
# Format x-axis with date ticks
locator = mdates.AutoDateLocator()
formatter = mdates.DateFormatter("%Y-%m-%d")

# Day of the Week: Sin and Cos Encoding
axes[0].plot(feature_rich_data.index, feature_rich_data["Day of Week Sin"], label="Day of Week Sin", alpha=0.8)
axes[0].plot(feature_rich_data.index, feature_rich_data["Day of Week Cos"], label="Day of Week Cos", alpha=0.8)
axes[0].set_title("Day of the Week: Sin and Cos Encoding")
axes[0].legend()
axes[0].grid()

# Day of the Month: Sin and Cos Encoding
axes[1].plot(feature_rich_data.index, feature_rich_data["Day of Month Sin"], label="Day of Month Sin", alpha=0.8)
axes[1].plot(feature_rich_data.index, feature_rich_data["Day of Month Cos"], label="Day of Month Cos", alpha=0.8)
axes[1].set_title("Day of the Month: Sin and Cos Encoding")
axes[1].legend()
axes[1].grid()

# Day of the Year: Sin and Cos Encoding
axes[2].plot(feature_rich_data.index, feature_rich_data["Day of Year Sin"], label="Day of Year Sin", alpha=0.8)
axes[2].plot(feature_rich_data.index, feature_rich_data["Day of Year Cos"], label="Day of Year Cos", alpha=0.8)
axes[2].set_title("Day of the Year: Sin and Cos Encoding")
axes[2].legend()
axes[2].grid()

# Apply date formatting to the x-axis of all subplots
for ax in axes:
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Display the plots
plt.tight_layout()
plt.show()

#%%
# Ensure the index is sorted and in datetime format
feature_rich_data.index = pd.to_datetime(feature_rich_data.index)
feature_rich_data = feature_rich_data.sort_index()

# Calculate the difference between consecutive days
date_deltas = feature_rich_data.index.to_series().diff()

# Identify missing days (gaps greater than 1 day)
missing_days = date_deltas[date_deltas > pd.Timedelta(days=1)]

# Print results
if missing_days.empty:
    print("No missing days detected.")
else:
    print(f"Missing days detected: {len(missing_days)}")
    print("Details of missing days:")
    print(missing_days)
    

# Get the first missing day
first_missing_date = missing_days.index[0]
delta = missing_days.iloc[0]

# Define the range of days to inspect (1 day before and after the missing period)
start_date = first_missing_date - pd.Timedelta(days=delta.days + 1)
end_date = first_missing_date + pd.Timedelta(days=1)

# Slice the data around the missing day
data_around_missing = feature_rich_data.loc[start_date:end_date]

# Display the data
print(f"Data around the first missing day ({first_missing_date}):")
print(data_around_missing)


#%% Clean Up Data Frame
feature_rich_data.columns = ['_'.join(col).strip() for col in feature_rich_data.columns.values]
# Replace the Date index with an incremental index
feature_rich_data = feature_rich_data.reset_index(drop=True)
print(feature_rich_data.head())
    


#%%
# Save the preprocessed dataset to a new CSV file
output_csv = "preprocessed_stock_data.csv"
feature_rich_data.to_csv(output_csv, index=False)

print(f"Dataset saved to {output_csv}")

#%%
# Save the preprocessed dataset to a new CSV file
output_csv = "missing_days.csv"
missing_days.to_csv(output_csv)

print(f"Dataset saved to {output_csv}")




#%%

# # Define the stock ticker symbol
# ticker = "AAPL"  # Apple Inc.

# # Download historical data
# data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

# # Display the first few rows
# print(data.head())

# #%%
# tickers = ["AAPL", "MSFT", "GOOGL"]
# data = yf.download(tickers, start="2020-01-01", end="2023-01-01")
# print(data.head())
# #%%
# stock = yf.Ticker("AAPL")
# print(stock.dividends)  # Dividends history
# print(stock.splits)     # Splits history
# #%%
# stock = yf.Ticker("AAPL")
# print(stock.info)  # Fundamental information as a dictionary
# #%%
# # Define the stock ticker symbol
# ticker = "AAPL"  # Apple Inc.

# # Get the maximum date range available
# stock = yf.Ticker(ticker)
# data = stock.history(period="max")

# # Display the first and last dates
# print("Earliest data point:", data.index.min())
# print("Latest data point:", data.index.max())
# #%%
# # Fetch data for the maximum available period
# data = yf.download("AAPL", period="max")

# # Display the first few rows
# print(data.head())
