import pandas as pd
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# Major global stock market indices with their Stooq symbols
TICKERS = {
    "S&P500": "^spx",
    "NASDAQ": "^ndq",
    "Dow Jones": "^dji",
    "FTSE250": "^ftm",
    "DAX": "^dax",
    "Nikkei225": "^nkx",
    "Shanghai": "^shc",
    "Russell2000": "iwm.us",
    "MSCI Emerging Markets": "eem.us",
    "MSCI World": "urth.us",
    "All Country World": "acwi.us",
    "Euro Stoxx 50":  "fez.us"}

def fetch_ticker_data(ticker_symbol: str, years: int) -> pd.DataFrame:
    """Fetch historical data for a given ticker symbol"""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    url = f"https://stooq.com/q/d/l/?s={ticker_symbol}&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d"
    
    try:
        df = pd.read_csv(url, parse_dates=["Date"]).set_index("Date").sort_index()
        time.sleep(1.0 + random.uniform(0, 1.0))  # avoid rate limits
        print(f"Successfully fetched data for {ticker_symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        return pd.DataFrame()

def fetch_all_data(years: int):
    """Fetch data for all tickers specified in the TICKERS dictionary"""
    data = {}
    for name, symbol in TICKERS.items():
        df = fetch_ticker_data(symbol, years)
        if not df.empty:
            data[name] = df['Close']
        else:
            print(f"No data available for {name}")
    
    return pd.DataFrame(data)

def plot_correlations(df):
    """Plot correlation matrix between all markets"""
    corr = df.corr()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", 
                     ha="center", va="center", 
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Between Global Markets")
    
    return plt.gcf()

if __name__ == "__main__":
    # Fetch data for the last 3 years
    markets_df = fetch_all_data(years=3)
    
    # Drop rows with any missing values
    markets_df = markets_df.dropna()
    
    print(f"Data shape after cleaning: {markets_df.shape}")
    
    # Normalizations
    df_100 = markets_df / markets_df.iloc[0] * 100
    df_zscore = (markets_df - markets_df.mean()) / markets_df.std()
    df_pct_change = markets_df.pct_change().dropna()
    df_minmax = (markets_df - markets_df.min()) / (markets_df.max() - markets_df.min())
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
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
        axs[2].plot(df_pct_change.index, df_pct_change[col], label=col, alpha=0.7)
    axs[2].set_title("Daily Percentage Change")
    axs[2].legend()
    axs[2].grid(True)
    
    # 4. Min-Max Normalization
    for col in df_minmax.columns:
        axs[3].plot(df_minmax.index, df_minmax[col], label=col)
    axs[3].set_title("Min-Max Normalization")
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig("global_markets_comparison.png")
    
    # Additional analysis - plot correlation matrix
    corr_fig = plot_correlations(markets_df)
    plt.tight_layout()
    plt.savefig("global_markets_correlation.png")
    
    # Calculate rolling correlations to S&P 500
    window = 90  # 90-day rolling window
    rolling_corr = pd.DataFrame()
    
    for col in markets_df.columns:
        if col != "S&P500":
            rolling_corr[col] = markets_df[col].rolling(window).corr(markets_df["S&P500"])
    
    # Plot rolling correlations
    plt.figure(figsize=(12, 6))
    for col in rolling_corr.columns:
        plt.plot(rolling_corr.index, rolling_corr[col], label=col)
    
    plt.title(f"{window}-day Rolling Correlation with S&P500")
    plt.legend()
    plt.grid(True)
    plt.savefig("rolling_correlations.png")
    
    plt.show()