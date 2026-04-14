from datetime import timedelta
import duckdb
import pandas as pd
import yfinance as yf


DB_PATH = "somedb.ddb"
TABLE_NAME = "yahoo"
TICKERS = ["MNST", "MSFT", "AXP", "SPY"] # Add more tickers as needed
START_DATE = "2020-01-01"
INTERVAL = "1d"
OVERLAP_DAYS = 5


def table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    query = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
    """
    return conn.execute(query, [table_name]).fetchone() is not None


def fetch_yahoo_data(start_date: str) -> pd.DataFrame:
    df = yf.download(
        " ".join(TICKERS),
        start=start_date,
        interval=INTERVAL,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return df

    df.columns = [f"{ticker}_{field}" for ticker, field in df.columns]
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


def create_initial_table(conn: duckdb.DuckDBPyConnection, fresh_df: pd.DataFrame) -> None:
    conn.register("fresh_yahoo", fresh_df)
    conn.sql(
        f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT
            CAST(Date AS DATE) AS Date,
            * EXCLUDE (Date)
        FROM fresh_yahoo
        ORDER BY Date
        """
    )
    conn.unregister("fresh_yahoo")


def merge_incremental_data(conn: duckdb.DuckDBPyConnection, fresh_df: pd.DataFrame) -> None:
    conn.register("fresh_yahoo", fresh_df)
    conn.sql(
        f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT
            CAST(Date AS DATE) AS Date,
            * EXCLUDE (Date, _source_priority, rn)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(Date AS DATE)
                       ORDER BY _source_priority DESC
                   ) AS rn
            FROM (
                SELECT *, 0 AS _source_priority FROM {TABLE_NAME}
                UNION ALL BY NAME
                SELECT *, 1 AS _source_priority FROM fresh_yahoo
            )
        )
        WHERE rn = 1
        ORDER BY Date
        """
    )
    conn.unregister("fresh_yahoo")


def main() -> None:
    conn = duckdb.connect(DB_PATH)
    try:
        has_table = table_exists(conn, TABLE_NAME)
        if has_table:
            max_date = conn.sql(f"SELECT MAX(Date) AS max_date FROM {TABLE_NAME}").fetchone()[0]
            max_date = pd.Timestamp(max_date).date()
            fetch_start = (max_date - timedelta(days=OVERLAP_DAYS)).isoformat()
        else:
            fetch_start = START_DATE

        fresh_df = fetch_yahoo_data(fetch_start)
        if fresh_df.empty:
            print(f"No data returned from Yahoo Finance for start={fetch_start}.")
            return

        if has_table:
            merge_incremental_data(conn, fresh_df)
        else:
            create_initial_table(conn, fresh_df)

        summary = conn.sql(
            f"""
            SELECT
                MIN(Date) AS min_date,
                MAX(Date) AS max_date,
                COUNT(*) AS row_count
            FROM {TABLE_NAME}
            """
        ).fetchdf()
        print(summary)
    finally:
        conn.close()


if __name__ == "__main__":
    main()


