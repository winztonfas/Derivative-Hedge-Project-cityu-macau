import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

tickers = ['MNST', 'MSFT', 'AXP', 'NFLX', 'NVDA', 'TSM']

# print(df)

def returns(df):
    df['MNST_Return'] = np.log(df['MNST_Close'] / df['MNST_Close'].shift(1))
    df['MSFT_Return'] = np.log(df['MSFT_Close'] / df['MSFT_Close'].shift(1))
    df['AXP_Return'] = np.log(df['AXP_Close'] / df['AXP_Close'].shift(1))

    # new tickers for examine
    df['NVDA_Return'] = np.log(df['NVDA_Close'] / df['NVDA_Close'].shift(1))
    df['NFLX_Return'] = np.log(df['NFLX_Close'] / df['NFLX_Close'].shift(1))
    df['TSM_Return'] = np.log(df['TSM_Close'] / df['TSM_Close'].shift(1))
    
    # index
    df['SPY_Return'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
    df['IXIC_Return'] = np.log(df['IXIC_Close'] / df['IXIC_Close'].shift(1))
    return df

def volatility(df):
    df['MNST_Volatility'] = df['MNST_Return'].rolling(window=30).std()
    df['MSFT_Volatility'] = df['MSFT_Return'].rolling(window=30).std()
    df['AXP_Volatility'] = df['AXP_Return'].rolling(window=30).std()
    df['NFLX_Volatility'] = df['NFLX_Return'].rolling(window=30).std()
    df['NVDA_Volatility'] = df['NVDA_Return'].rolling(window=30).std()
    df['TSM_Volatility'] = df['TSM_Return'].rolling(window=30).std()
    
    df['SPY_Volatility'] = df['SPY_Return'].rolling(window=30).std()
    df['IXIC_Volatility'] = df['IXIC_Return'].rolling(window=30).std()
    return df

def correlation_pair(df):
    return df[['MNST_Return', 'MSFT_Return', 'AXP_Return', 'NFLX_Return', 'NVDA_Return', 'TSM_Return', 'SPY_Return', 'IXIC_Return']].corr()

def avg_corr(df):
    df['Mean_return'] = df[['MNST_Return', 'MSFT_Return', 'AXP_Return', 'NFLX_Return', 'NVDA_Return', 'TSM_Return']].mean(axis=1)
    return df[['Mean_return', 'SPY_Return', 'IXIC_Return']].corr()

def run_regression(y, X):
    data = pd.concat([X, y], axis=1).dropna()

    X_clean = data[X.columns]
    y_clean = data[y.name]

    model = LinearRegression()
    model.fit(X_clean, y_clean)

    return {
        "beta_SPY": float(model.coef_[0]),
        "beta_IXIC": float(model.coef_[1]),
        "alpha": float(model.intercept_)
    }

def basket_regression(df):
    X = df[["SPY_Return", "IXIC_Return"]]
    y = df["Mean_return"]
    return run_regression(y, X)

def basket_SPY_regression(df):
    X = df[["SPY_Return"]]
    y = df["Mean_return"]
    data = pd.concat([X, y], axis=1).dropna()

    X_clean = data[X.columns]
    y_clean = data[y.name]

    model = LinearRegression()
    model.fit(X_clean, y_clean)

    return {
        "beta_SPY": float(model.coef_[0]),
        "alpha": float(model.intercept_)
    }

def basket_IXIC_regression(df):
    X = df[["IXIC_Return"]]
    y = df["Mean_return"]
    data = pd.concat([X, y], axis=1).dropna()

    X_clean = data[X.columns]
    y_clean = data[y.name]

    model = LinearRegression()
    model.fit(X_clean, y_clean)

    return {
        "beta_IXIC": float(model.coef_[0]),
        "alpha": float(model.intercept_)
    }

def ticker_regression(df, ticker):
    X = df[["SPY_Return", "IXIC_Return"]]
    y = df[f"{ticker}_Return"]
    return run_regression(y, X)

def SPY_regression(df, ticker):
    X = df[["SPY_Return"]]
    y = df[f"{ticker}_Return"]
    data = pd.concat([X, y], axis=1).dropna()

    X_clean = data[X.columns]
    y_clean = data[y.name]

    model = LinearRegression()
    model.fit(X_clean, y_clean)

    return {
        "beta_SPY": float(model.coef_[0]),
        "alpha": float(model.intercept_)
    }

def IXIC_regression(df, ticker):    
    X = df[["IXIC_Return"]]
    y = df[f"{ticker}_Return"]
    data = pd.concat([X, y], axis=1).dropna()

    X_clean = data[X.columns]
    y_clean = data[y.name]

    model = LinearRegression()
    model.fit(X_clean, y_clean)

    return {
        "beta_IXIC": float(model.coef_[0]),
        "alpha": float(model.intercept_)
    }

# Historical amalysis for 6 benchmarks tickers and 2 market indices
def main():
    conn = duckdb.connect("test.ddb")
    try:
        df = conn.sql(
            "SELECT Date, MNST_Close, MSFT_Close, AXP_Close, NFLX_Close, NVDA_Close, TSM_Close, SPY_Close, IXIC_Close "
            "FROM yahoo "
            "WHERE Date < '2026-3-2'"
        ).fetchdf()

        results_SPY = []
        results_IXIC = []
        results = []

        df = returns(df)
        df = volatility(df)

        pair_corr = correlation_pair(df)
        mean_corr = avg_corr(df)
        basket_results = basket_regression(df)
        basket_SPY_regression_results = basket_SPY_regression(df)
        basket_IXIC_regression_results = basket_IXIC_regression(df)

        # Add basket regression results to summary
        results.append({
            "Target": "Basket",
            "beta_SPY": basket_results["beta_SPY"],
            "beta_IXIC": basket_results["beta_IXIC"],
            "alpha": basket_results["alpha"]
        })

        results_SPY.append({
            "Target": "Basket",
            "beta_SPY": basket_SPY_regression_results["beta_SPY"],
            "alpha": basket_SPY_regression_results["alpha"]
        })


        results_IXIC.append({
            "Target": "Basket",
            "beta_IXIC": basket_IXIC_regression_results["beta_IXIC"],
            "alpha": basket_IXIC_regression_results["alpha"]
        })

        
        # single index regression for each ticker
        for ticker in tickers:
            spy_results = SPY_regression(df, ticker)
            ixic_results = IXIC_regression(df, ticker)

            results_SPY.append({
                "Target": f"{ticker} vs SPY",
                "beta_SPY": spy_results["beta_SPY"],
                "alpha": spy_results["alpha"]
            })

            results_IXIC.append({
                "Target": f"{ticker} vs IXIC",
                "beta_IXIC": ixic_results["beta_IXIC"],
                "alpha": ixic_results["alpha"]
            })

        # two index regression for each ticker
        for ticker in tickers:
            ticker_results = ticker_regression(df, ticker)
            results.append({
                "Target": ticker,
                "beta_SPY": ticker_results["beta_SPY"],
                "beta_IXIC": ticker_results["beta_IXIC"],
                "alpha": ticker_results["alpha"]
            })

        # df 
        summary_dfSPY = pd.DataFrame(results_SPY)
        summary_dfIXIC = pd.DataFrame(results_IXIC)
        summary_dfALL = pd.DataFrame(results)

        print("Pair Correlation")
        print(pair_corr)
        print()
        
        print("Basket vs Benchmarks Correlation")
        print(mean_corr)
        print()

        print('Mean Volatility')
        for ticker in tickers:
            print(f"{ticker}_MeanVolatility: {df[f'{ticker}_Volatility'].mean()}")
            print(f"{ticker} vs SPY Volatility Correlation: {df[[f'{ticker}_Volatility', 'SPY_Volatility']].corr().iloc[0,1]}")
            print(f"{ticker} vs IXIC Volatility Correlation: {df[[f'{ticker}_Volatility', 'IXIC_Volatility']].corr().iloc[0,1]}")
            print()

        print("Single Index Regression Summary_SPY")
        print(summary_dfSPY)
        print()
        print("Single Index Regression Summary_IXIC")
        print(summary_dfIXIC)
        print()
        
        print("Two Index Regression Summary")
        print(summary_dfALL)
        print()

    except Exception as e:
        print(e)

    finally:
        conn.commit()
        conn.close()


if __name__ == "__main__":
    main()
  
