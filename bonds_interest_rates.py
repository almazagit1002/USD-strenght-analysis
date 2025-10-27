import pandas as pd
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web

# ------------------------------------
# Ticker Symbols from Stooq
# ------------------------------------
TICKERS = {
    "iShares 20+ Year Treasury Bond ETF": "tlt.us",
    "iShares 7-10 Year Treasury Bond ETF": "ief.us",
    "iShares 1-3 Year Treasury Bond ETF": "shy.us",
    "Vanguard Total Bond Market ETF": "bnd.us",
    "iShares Core U.S. Aggregate Bond ETF": "agg.us",
    "iShares TIPS Bond ETF": "tip.us",
    "iShares iBoxx $ Investment Grade Corporate Bond ETF": "lqd.us",
    "iShares iBoxx $ High Yield Corporate Bond ETF": "hyg.us",
    "SPDR Bloomberg High Yield Bond ETF": "jnk.us",
    "SPDR Bloomberg 1-3 Month T-Bill ETF": "bil.us",
    "Global X Interest Rate Hedge ETF": "rate.us"
}

# ------------------------------------
# Fetch ETF Data from Stooq
# ------------------------------------
def fetch_ticker_data(ticker_symbol: str, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    url = f"https://stooq.com/q/d/l/?s={ticker_symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=["Date"]).set_index("Date").sort_index()
        time.sleep(1.0 + random.uniform(0, 1.0))  # rate limiting
        print(f"Successfully fetched data for {ticker_symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def fetch_all_etf_data(years: int):
    data = {}
    for name, symbol in TICKERS.items():
        df = fetch_ticker_data(symbol, years)
        if not df.empty:
            data[name] = df['Close']
        else:
            print(f"No data for {name}")
    return pd.DataFrame(data)

# ------------------------------------
# Fetch Interest Rate Data from FRED
# ------------------------------------
def fetch_fred_yields(start_date, end_date):
    ten_year = web.DataReader('DGS10', 'fred', start_date, end_date)
    three_month = web.DataReader('DTB3', 'fred', start_date, end_date)
    return ten_year, three_month

# ------------------------------------
# Plotting Correlation Matrix
# ------------------------------------
def plot_correlations(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Between Bond and Interest Rate ETFs")
    plt.tight_layout()
    plt.show()

# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 5, 20)

    # Fetch ETF price data
    etfs_df = fetch_all_etf_data(years=3)
    etfs_df.dropna(inplace=True)
    print(f"ETF data shape: {etfs_df.shape}")

    # Fetch FRED yield data
    ten_year_yield, three_month_yield = fetch_fred_yields(start_date, end_date)
    print(f"10-Year Yield shape: {ten_year_yield.shape}")
    print(f"3-Month Yield shape: {three_month_yield.shape}")

    # Normalize ETF prices (start at 100)
    df_100 = etfs_df / etfs_df.iloc[0] * 100

    # Plot Normalized ETF Prices
    plt.figure(figsize=(14, 7))
    for col in df_100.columns:
        plt.plot(df_100.index, df_100[col], label=col)
    plt.title("Normalized Bond ETF Prices (Start = 100)")
    plt.xlabel("Date")
    plt.ylabel("Price Index")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Interest Rate Yields
    plt.figure(figsize=(14, 5))
    plt.plot(ten_year_yield.index, ten_year_yield, label="10-Year Treasury Yield")
    plt.plot(three_month_yield.index, three_month_yield, label="3-Month Treasury Yield")
    plt.title("US Treasury Yields from FRED")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Correlation Matrix
    plot_correlations(etfs_df)
