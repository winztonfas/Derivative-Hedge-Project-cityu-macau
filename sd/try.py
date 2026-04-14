import yfinance as yf
import pandas as pd
import duckdb

tickers = yf.Tickers(['JPM', 'SPY', '^IXIC'])
df = tickers.download(interval='1d', start="2020-01-01", end="2026-04-01", group_by='ticker')

df.columns = [f"{ticker.replace('^', '')}_{field}" for ticker, field in df.columns] # for fixing the issue of '^' in column names
df = df.reset_index()
df.to_csv("jpm.csv", index=False)
