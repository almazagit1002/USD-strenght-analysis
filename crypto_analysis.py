import pandas as pd
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import numpy as np

TICKERS = {
    "BTC": "btc.v",  
    "ETH": "eth.v",
    "XRP": "xrp.v",
    "BNB": "bnb.v",
    "SOL": "sol.v"
}

def fetch_ticker_data(ticker_symbol: str, years: int ) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    url = f"https://stooq.com/q/d/l/?s={ticker_symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    df = pd.read_csv(url, parse_dates=["Date"]).set_index("Date").sort_index()
    time.sleep(1.0 + random.uniform(0, 1.0))  # avoid rate limits
    return df

def fetch_all_data(years: int):
    data = {}
    for name, symbol in TICKERS.items():
        df = fetch_ticker_data(symbol, years)
        if not df.empty:
            data[name] = df['Close']
    return pd.DataFrame(data)

if __name__ == "__main__":
    crypto_df = fetch_all_data(years=1)

    # Drop rows with any missing values
    crypto_df = crypto_df.dropna()

    # Normalizations
    df_100 = crypto_df / crypto_df.iloc[0] * 100
    df_zscore = (crypto_df - crypto_df.mean()) / crypto_df.std()
    df_pct_change = crypto_df.pct_change().dropna()
    df_minmax = (crypto_df - crypto_df.min()) / (crypto_df.max() - crypto_df.min())

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    # 1. Start at 100
    for col in df_100.columns:
        axs[0].plot(df_100.index, df_100[col], label=col)
    axs[0].set_title("Normalized (Start = 100)")
    axs[0].legend()
    axs[0].grid(True)

    # 2. Z-score
    for col in df_zscore.columns:
        axs[1].plot(df_zscore.index, df_zscore[col], label=col)
    axs[1].set_title("Z-score Normalization")
    axs[1].legend()
    axs[1].grid(True)

    # 3. Percentage Change
    for col in df_pct_change.columns:
        axs[2].plot(df_pct_change.index, df_pct_change[col], label=col)
    axs[2].set_title("Percentage Change")
    axs[2].legend()
    axs[2].grid(True)

    # 4. Min-Max Normalization
    for col in df_minmax.columns:
        axs[3].plot(df_minmax.index, df_minmax[col], label=col)
    axs[3].set_title("Min-Max Normalization")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
