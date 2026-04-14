import yfinance as yf
import pandas as pd
import duckdb

odd_tickers = yf.Tickers(['MNST', 'MSFT', 'AXP', 'SPY', '^IXIC', 'NVDA', 'NFLX', 'TSM']) 
hedge_futures = yf.Tickers(["MNQM26.CME","MESM26.CME"])

df = odd_tickers.download(interval='1d', start="2020-01-01", group_by='ticker')
df2 = hedge_futures.download(interval='1d', start="2026-03-01",end="2026-03-28", group_by='ticker')


df.columns = [f"{ticker.replace('^', '')}_{field}" for ticker, field in df.columns] # for fixing the issue of '^' in column names
df = df.reset_index()
df2.columns = [f"{ticker.replace('.', '')}_{field}" for ticker, field in df2.columns] # for fixing the issue of '^' in column names
df2 = df2.reset_index()

# df.to_parquet("yahoo1.parquet", engine="pyarrow")

conn = duckdb.connect('test.ddb')
conn.sql('CREATE OR REPLACE TABLE yahoo AS SELECT * FROM df')
conn.sql('CREATE OR REPLACE TABLE hedge_futures AS SELECT * FROM df2')

query = """
WITH MNST AS (
    SELECT date, MNST_Open, MNST_High, MNST_Low, MNST_Close, MNST_Volume
    FROM yahoo
    WHERE date < '2026-03-28'
),

MSFT AS (
    SELECT date, MSFT_Open, MSFT_High, MSFT_Low, MSFT_Close, MSFT_Volume
    FROM yahoo
    WHERE date < '2026-03-28'
),

AXP AS (
    SELECT date, AXP_Open, AXP_High, AXP_Low, AXP_Close, AXP_Volume
    FROM yahoo
    WHERE date < '2026-03-28'
),

SPY AS (
    SELECT date, SPY_Open, SPY_High, SPY_Low, SPY_Close, SPY_Volume
    FROM yahoo
    WHERE date < '2026-03-28'
),

IXIC AS (
    SELECT date, IXIC_Open, IXIC_High, IXIC_Low, IXIC_Close, IXIC_Volume
    FROM yahoo
    WHERE date < '2026-03-28'
)

SELECT * FROM MNST, MSFT, AXP, SPY, IXIC
WHERE MNST.date = MSFT.date AND MSFT.date = AXP.date AND AXP.date = SPY.date AND SPY.date = IXIC.date
"""

print(conn.sql(query).fetchdf())
print(conn.sql("SELECT * FROM hedge_futures").fetchdf())

conn.commit()
conn.close()
