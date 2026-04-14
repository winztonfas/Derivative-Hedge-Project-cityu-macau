import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

tickers = ['MNST', 'MSFT', 'AXP', 'NFLX']

# print(df)

def returns(df):
    df['MNST_Return'] = np.log(df['MNST_Close'] / df['MNST_Close'].shift(1))
    df['MSFT_Return'] = np.log(df['MSFT_Close'] / df['MSFT_Close'].shift(1))
    df['AXP_Return'] = np.log(df['AXP_Close'] / df['AXP_Close'].shift(1))
    df['NFLX_Return'] = np.log(df['NFLX_Close'] / df['NFLX_Close'].shift(1))
    
    # index
    df['SPY_Return'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
    df['IXIC_Return'] = np.log(df['IXIC_Close'] / df['IXIC_Close'].shift(1))
    return df

def returns_hedge(df):
    df['MESM26CME_Return'] = np.log(df['MESM26CME_Close'] / df['MESM26CME_Close'].shift(1))
    df['MNQM26CME_Return'] = np.log(df['MNQM26CME_Close'] / df['MNQM26CME_Close'].shift(1))
    return df

# def volatility(df):
#     df['MNST_Volatility'] = df['MNST_Return'].rolling(window=30).std()
#     df['MSFT_Volatility'] = df['MSFT_Return'].rolling(window=30).std()
#     df['AXP_Volatility'] = df['AXP_Return'].rolling(window=30).std()
#     df['NFLX_Volatility'] = df['NFLX_Return'].rolling(window=30).std()

#     df['SPY_Volatility'] = df['SPY_Return'].rolling(window=30).std()
#     df['IXIC_Volatility'] = df['IXIC_Return'].rolling(window=30).std()
#     return df

def correlation_pair(df):
    return df[['MNST_Return', 'MSFT_Return', 'AXP_Return', 'NFLX_Return', 'SPY_Return', 'IXIC_Return']].corr()

def avg_corr(df):
    df['Mean_return'] = df[['MNST_Return', 'MSFT_Return', 'AXP_Return', 'NFLX_Return']].mean(axis=1)
    return df[['Mean_return', 'SPY_Return', 'IXIC_Return']].corr()

def heged_correlation_pair(df):
    return df[['Hedge_Adjusted_Return', 'SPY_Return', 'IXIC_Return']].corr()

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

def hedged_basket_regression(df):
    X = df[["SPY_Return", "IXIC_Return"]]
    y = df["Hedge_Adjusted_Return"]
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

def calculate_hedge_adjusted_PL(df, beta_spy, beta_ixic, Value):
    df['Basket_PL'] = df['Mean_return'] * Value

    df['SPY_Hedge_Dollar'] = -Value * beta_spy
    df['IXIC_Hedge_Dollar'] = -Value * beta_ixic

    df['MES_Notional'] = df['MESM26CME_Close'] * 5
    df['MNQ_Notional'] = df['MNQM26CME_Close'] * 2
    df['ES_Notional'] = df['MESM26CME_Close'] * 50
    df['NQ_Notional'] = df['MNQM26CME_Close'] * 20

    df['MES_Contracts'] = df['SPY_Hedge_Dollar'] /df['MES_Notional']
    df['MNQ_Contracts'] = df['IXIC_Hedge_Dollar'] /df['MNQ_Notional']
    df['ES_Contracts'] = df['SPY_Hedge_Dollar'] /df['ES_Notional']
    df['NQ_Contracts'] = df['IXIC_Hedge_Dollar'] /df['NQ_Notional']

    df['MES_Price_Change'] = df['MESM26CME_Close'].diff().fillna(0)
    df['MNQ_Price_Change'] = df['MNQM26CME_Close'].diff().fillna(0)

    MES_Contracts = int(np.floor(abs(df['MES_Contracts'].iloc[0]))) * np.sign(df['MES_Contracts'].iloc[0])
    MNQ_Contracts = int(np.floor(abs(df['MNQ_Contracts'].iloc[0]))) * np.sign(df['MNQ_Contracts'].iloc[0])

    df['MES_Contracts_Actual'] = MES_Contracts
    df['MNQ_Contracts_Actual'] = MNQ_Contracts

    df['Hedge_PL_MES'] = MES_Contracts * 5 * df['MES_Price_Change']
    df['Hedge_PL_MNQ'] = MNQ_Contracts * 2 * df['MNQ_Price_Change']

    df['Hedge_PL'] = df['Hedge_PL_MES']+ df['Hedge_PL_MNQ']
    df['Hedge_Adjusted_PL'] = df['Basket_PL'] + df['Hedge_PL_MES'] + df['Hedge_PL_MNQ']

    df["Hedge_Adjusted_Return"] = df["Hedge_Adjusted_PL"] / Value
    return df

# Historical amalysis for 6 benchmarks tickers and 2 market indices
def main():
    conn = duckdb.connect("test.ddb")
    try:
        df = conn.sql(
            "SELECT Date, MNST_Close, MSFT_Close, AXP_Close, NFLX_Close, SPY_Close, IXIC_Close "
            "FROM yahoo "
            "WHERE Date > '2026-3-1' AND Date < '2026-3-28'"
        ).fetchdf()

        df2 = conn.sql(
            "SELECT Date, MESM26CME_Close, MNQM26CME_Close " 
            "FROM hedge_futures").fetchdf()
        
        results_SPY = []
        results_IXIC = []
        results = []
        hedged_summary = []
        betas = []

        # returns calculation for hedge futures
        df2 = returns_hedge(df2)

        # for stocks 
        df = returns(df)
        # df = volatility(df)

        # merge df
        df = pd.merge(df, df2, on='Date', how='inner')

        # correlation analysis
        pair_corr = correlation_pair(df)
        mean_corr = avg_corr(df)

        # regression analysis for basket and each ticker
        basket_results = basket_regression(df)
        basket_SPY_regression_results = basket_SPY_regression(df)
        basket_IXIC_regression_results = basket_IXIC_regression(df)

        # Basket without NFLX for comparison
        df['Mean_return_Original'] = df[['MNST_Return', 'MSFT_Return', 'AXP_Return']].mean(axis=1)
        basket_original_results = run_regression(df['Mean_return_Original'], df[["SPY_Return", "IXIC_Return"]])

        # Add basket regression results to summary
        ## two index regression results
        results.append({
            "Target": "Basket",
            "beta_SPY": basket_results["beta_SPY"],
            "beta_IXIC": basket_results["beta_IXIC"],
            "alpha": basket_results["alpha"]
        })

        ## single index regression results for basket
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

        # Add stocks regression results to summary
        # two index regression for each ticker
        for ticker in tickers:
            ticker_results = ticker_regression(df, ticker)
            results.append({
                "Target": ticker,
                "beta_SPY": ticker_results["beta_SPY"],
                "beta_IXIC": ticker_results["beta_IXIC"],
                "alpha": ticker_results["alpha"]
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

        # construction for hedging set up 
        betas.append({
            "beta_SPY": basket_results["beta_SPY"],
            "beta_IXIC": basket_results["beta_IXIC"]
                })
        
        # hedge results
        beta_spy = betas[0]["beta_SPY"]
        beta_ixic = betas[0]["beta_IXIC"]
        df = calculate_hedge_adjusted_PL(df, beta_spy, beta_ixic, Value=1000000)
        hedged_results = hedged_basket_regression(df)
        hedge_corr = heged_correlation_pair(df)

        hedged_summary.append({
            "Target": "Basket",
            "hedged_beta_SPY": hedged_results["beta_SPY"],
            "hedged_beta_IXIC": hedged_results["beta_IXIC"],
            "alpha": hedged_results["alpha"]
        })
        
        # ouput list to df 
        summary_dfSPY = pd.DataFrame(results_SPY)
        summary_dfIXIC = pd.DataFrame(results_IXIC)
        summary_dfALL = pd.DataFrame(results) 

        print("Pair Correlation")
        print(pair_corr)
        print()
        
        print("Basket vs Benchmarks Correlation")
        print(mean_corr)
        print()

        print("*Hedge Adjusted Correlation")
        print(hedge_corr)
        print()

        # Volatility analysis window not enough for 30 day rolling, so skipping for now
        # print('Mean Volatility')
        # for ticker in tickers:
        #     print(f"{ticker}_MeanVolatility: {df[f'{ticker}_Volatility'].mean()}")
        #     print(f"{ticker} vs SPY Volatility Correlation: {df[[f'{ticker}_Volatility', 'SPY_Volatility']].corr().iloc[0,1]}")
        #     print(f"{ticker} vs IXIC Volatility Correlation: {df[[f'{ticker}_Volatility', 'IXIC_Volatility']].corr().iloc[0,1]}")
        #     print()

        print("Single Index Regression Summary_SPY")
        print(summary_dfSPY)
        print()
        print("Single Index Regression Summary_IXIC")
        print(summary_dfIXIC)
        print()
        
        print("Two Index Regression Summary")
        print(summary_dfALL)
        print()

        print("===== Backtesting Basket Mean Results without hedge [1st March to 28th March 2026] =====")
        print(f"Basket Mean Return: {df['Mean_return'].mean()}")
        print(f'Basket P&L Mean Return: {df["Basket_PL"].mean()}')
        print(f'benchmark SP500 Mean Return: {df["SPY_Return"].mean()} & Nasdaq Mean Return: {df["IXIC_Return"].mean()}')    
        print(f"Basket Mean alpha: {basket_results['alpha']}")
        print(f"Basket Mean beta: SP500 beta:{summary_dfALL['beta_SPY'].iloc[0]}, Nasdaq beta:{summary_dfALL['beta_IXIC'].iloc[0]}")
        print()

        print("===== Comparison with Original 3 Tickers (without NFLX) without hedge =====")
        print(f"Original Basket Mean Return: {df['Mean_return_Original'].mean()}")
        print(f"Original Basket Mean alpha: {basket_original_results['alpha']}")
        print(f"Original Basket Mean beta: SP500 beta:{basket_original_results['beta_SPY']}, Nasdaq beta:{basket_original_results['beta_IXIC']}")
        print()

        print("===== P&L Summary =====")
        print(f"Basket P&L Mean Return: {df['Basket_PL'].mean()}")
        print(f"Hedge PL Mean Return: {df['Hedge_PL'].mean()}")
        print(f"Hedge Adjusted P&L Mean Return: {df['Hedge_Adjusted_PL'].mean()}")
        print(f"Hedge Position for MES:{df['MES_Contracts'].iloc[0]} & MNQ:{df['MNQ_Contracts'].iloc[0]}")
        print()

        print("===== Backtesting Basket Mean Results with hedge [1st March to 28th March 2026] =====")
        print(f"Basket Hedged Mean Return: {df['Hedge_Adjusted_Return'].mean()} ")
        print(f'Basket Hedged P&L Mean Return: {df["Hedge_Adjusted_PL"].mean()}')
        print(f"Hedge Position for MES:{df['MES_Contracts_Actual'].iloc[0]} & MNQ:{df['MNQ_Contracts_Actual'].iloc[0]}")

        # Since the hedgge contract of ES and NQ are the 10 times of MSE and MNQ, less flexible for small size. 
        # print(f"Assume hedge Position for ES: {df['ES_Contracts'].iloc[0]} & NQ: {df['NQ_Contracts'].iloc[0]}") 
        print(f"Hedge Adjusted Beta for SPY: {hedged_summary[0]['hedged_beta_SPY']}, IXIC: {hedged_summary[0]['hedged_beta_IXIC']}")        

    except Exception as e:
        print(e)

    finally:
        conn.commit()
        conn.close()


if __name__ == "__main__":
    main()
  
