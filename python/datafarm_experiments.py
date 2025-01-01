# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:32:23 2024

@author: Plutonium
"""

#%%
import os
import pandas as pd
import kagglehub

#%%
# def list_csv_files_recursively(data_dir):
#     files_dict = {}
#     for root, dirs, files in os.walk(data_dir):
#         csv_files = [file for file in files if file.endswith('.csv')]
#         if csv_files:
#             files_dict[root] = csv_files
#     return files_dict

# def list_txt_files_recursively(data_dir):
#     files_dict = {}
#     for root, dirs, files in os.walk(data_dir):
#         txt_files = [file for file in files if file.endswith('.txt')]
#         if txt_files:
#             files_dict[root] = txt_files
#     return files_dict

# #%%

# # Dataset from 2010 to 2017.. 
# path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")
# print("Path to dataset files:", path)
# txt_files = list_txt_files_recursively(path)
# if txt_files:
#     for p in txt_files.keys():
#         print(p)
# else:
#     print("No TXT files found in the dataset directory.")
          
# # Dataset from 1999 to 2020
# path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
# print("Path to dataset files:", path)

# # Datset from 2010 to 2024
# path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
# print("Path to dataset files:", path)

# Stock Market Sentiment Dataset
# path = kagglehub.dataset_download("yash612/stockmarket-sentiment-dataset")

# Bitcoin Minute by Minute from Jan 2012
# path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")

# Bitcoin Tweets 
# path = kagglehub.dataset_download("kaushiksuresh147/bitcoin-tweets")


#%%



    
#%% Yahoo Finance API (via yfinance Python library):
import yfinance as yf
# Define the stock ticker symbol
ticker = "AAPL"  # Apple Inc.

# Download historical data
data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

# Display the first few rows
print(data.head())

#%%
tickers = ["AAPL", "MSFT", "GOOGL"]
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")
print(data.head())
#%%
stock = yf.Ticker("AAPL")
print(stock.dividends)  # Dividends history
print(stock.splits)     # Splits history
#%%
stock = yf.Ticker("AAPL")
print(stock.info)  # Fundamental information as a dictionary
#%%
# Define the stock ticker symbol
ticker = "AAPL"  # Apple Inc.

# Get the maximum date range available
stock = yf.Ticker(ticker)
data = stock.history(period="max")

# Display the first and last dates
print("Earliest data point:", data.index.min())
print("Latest data point:", data.index.max())
#%%
# Fetch data for the maximum available period
data = yf.download("AAPL", period="max")

# Display the first few rows
print(data.head())


#%% Alpha Vantage:
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')
print(data.head())

#%% Quandl
import quandl
quandl.ApiConfig.api_key = "YOUR_API_KEY"
data = quandl.get("WIKI/AAPL", start_date="2020-01-01", end_date="2023-01-01")
print(data.head())

#%% Polygon.io
# Needs a subscription. Lets avoid this until we need more data

