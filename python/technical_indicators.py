# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 00:42:42 2024

@author: Plutonium
"""
import pandas as pd
import numpy as np

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD and Signal Line.
    """
    fast_ema = data["Close"].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data["Close"].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    """
    sma = data["Close"].rolling(window=period).mean()
    std = data["Close"].rolling(window=period).std()
    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)
    return upper_band, sma, lower_band

def calculate_rsi(data, period=14):
    """
    Calculate RSI (Relative Strength Index).
    """
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_cci(data, period=20):
    """
    Calculate CCI (Commodity Channel Index).
    """
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: (np.abs(x - x.mean())).mean(), raw=True)
    cci = (typical_price - sma) / (0.015 * mad)
    return cci

def calculate_adx(data, period=14):
    """
    Calculate ADX (Average Directional Index).
    """
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / abs(plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_sma(data, period):
    """
    Calculate SMA (Simple Moving Average).
    """
    return data["Close"].rolling(window=period).mean()
