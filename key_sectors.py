import pandas as pd
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------
# Sector ETFs via Stooq
# ------------------------------------
SECTOR_TICKERS = {
    "Technology": "xlk.us",
    "Energy": "xle.us",
    "Consumer Staples": "xlp.us",
    "Consumer Discretionary": "xly.us",
    "Healthcare": "xlv.us",
    "Industrials": "xli.us",
    "Materials": "xlb.us"
}

# ------------------------------------
# Fetch Data from Stooq
# ------------------------------------
def fetch_ticker_data(ticker_symbol: str, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    url = f"https://stooq.com/q/d/l/?s={ticker_symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    try:
        df = pd.read_csv(url, parse_dates=["Date"]).set_index("Date").sort_index()
        time.sleep(1.0 + random.uniform(0.5, 1.0))  # delay to prevent server overload
        print(f"Fetched {ticker_symbol}")
        return df
    except Exception as e:
        print(f"Failed to fetch {ticker_symbol}: {e}")
        return pd.DataFrame()

def fetch_all_sector_data(years: int) -> pd.DataFrame:
    data = {}
    for name, symbol in SECTOR_TICKERS.items():
        df = fetch_ticker_data(symbol, years)
        if not df.empty:
            data[name] = df['Close']
    return pd.DataFrame(data)

# ------------------------------------
# Plot Correlation Matrix
# ------------------------------------
def plot_correlations(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", 
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Sector Correlation Matrix")
    plt.tight_layout()
    plt.savefig("sector_correlation_matrix.png")
    plt.show()

# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    years = 3
    df = fetch_all_sector_data(years)
    df.dropna(inplace=True)

    print(f"Data shape: {df.shape}")

    # Normalizations
    df_100 = df / df.iloc[0] * 100
    df_z = (df - df.mean()) / df.std()
    df_pct = df.pct_change().dropna()
    df_minmax = (df - df.min()) / (df.max() - df.min())

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    for col in df.columns:
        axs[0].plot(df_100.index, df_100[col], label=col)
    axs[0].set_title("Normalized (Start = 100)")
    axs[0].legend()
    axs[0].grid(True)

    for col in df.columns:
        axs[1].plot(df_z.index, df_z[col], label=col)
    axs[1].set_title("Z-Score Normalization")
    axs[1].legend()
    axs[1].grid(True)

    for col in df_pct.columns:
        axs[2].plot(df_pct.index, df_pct[col], label=col, alpha=0.7)
    axs[2].set_title("Daily % Change")
    axs[2].legend()
    axs[2].grid(True)

    for col in df_minmax.columns:
        axs[3].plot(df_minmax.index, df_minmax[col], label=col)
    axs[3].set_title("Min-Max Normalization")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("sector_comparison.png")
    plt.show()

    # Correlation Matrix
    plot_correlations(df)

    # Rolling correlation with Technology (example)
    window = 90
    rolling_corr = pd.DataFrame()
    for col in df.columns:
        if col != "Technology":
            rolling_corr[col] = df[col].rolling(window).corr(df["Technology"])

    plt.figure(figsize=(12, 6))
    for col in rolling_corr.columns:
        plt.plot(rolling_corr.index, rolling_corr[col], label=col)
    plt.title(f"{window}-Day Rolling Correlation with Technology Sector")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sector_rolling_correlation.png")
    plt.show()
