import pandas as pd
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import numpy as np

TICKERS = {
    "US Financials ETF": "xlf.us",
    "European Banks ETF": "eufn.us",
    "Global Financials ETF": "ixg.us",
    "Asia-Pacific ETF": "aia.us",
    "China Financials ETF": "fxi.us",
    "Japan Banks ETF": "ewj.us",
    "Latin America Financials ETF": "ilf.us",
    "Canada Banks ETF": "ewc.us",
    "Australia Banks ETF": "ewa.us"
}

# Define reference index for correlation analysis (using US Financials as reference)
REFERENCE_INDEX = "US Financials ETF"

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
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Between Global Financial Markets")
    
    return plt.gcf()

if __name__ == "__main__":
    # Fetch data for the last 3 years
    financial_df = fetch_all_data(years=3)
    
    # Drop rows with any missing values
    financial_df = financial_df.dropna()
    
    print(f"Data shape after cleaning: {financial_df.shape}")
    
    # Normalizations
    df_100 = financial_df / financial_df.iloc[0] * 100
    df_zscore = (financial_df - financial_df.mean()) / financial_df.std()
    df_pct_change = financial_df.pct_change().dropna()
    df_minmax = (financial_df - financial_df.min()) / (financial_df.max() - financial_df.min())
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # 1. Start at 100
    for col in df_100.columns:
        axs[0].plot(df_100.index, df_100[col], label=col)
    axs[0].set_title("Normalized (Start = 100)")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[0].grid(True)
    
    # 2. Z-score
    for col in df_zscore.columns:
        axs[1].plot(df_zscore.index, df_zscore[col], label=col)
    axs[1].set_title("Z-score Normalization")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[1].grid(True)
    
    # 3. Percentage Change
    for col in df_pct_change.columns:
        axs[2].plot(df_pct_change.index, df_pct_change[col], label=col, alpha=0.7)
    axs[2].set_title("Daily Percentage Change")
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].grid(True)
    
    # 4. Min-Max Normalization
    for col in df_minmax.columns:
        axs[3].plot(df_minmax.index, df_minmax[col], label=col)
    axs[3].set_title("Min-Max Normalization")
    axs[3].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.savefig("global_markets_comparison.png", bbox_inches='tight')
    
    # Additional analysis - plot correlation matrix
    corr_fig = plot_correlations(financial_df)
    plt.tight_layout()
    plt.savefig("global_markets_correlation.png", bbox_inches='tight')
    
    # Calculate rolling correlations to US Financials ETF instead of S&P 500
    window = 90  # 90-day rolling window
    rolling_corr = pd.DataFrame()
    
    for col in financial_df.columns:
        if col != REFERENCE_INDEX:
            rolling_corr[col] = financial_df[col].rolling(window).corr(financial_df[REFERENCE_INDEX])
    
    # Plot rolling correlations
    plt.figure(figsize=(12, 6))
    for col in rolling_corr.columns:
        plt.plot(rolling_corr.index, rolling_corr[col], label=col)
    
    plt.title(f"{window}-day Rolling Correlation with {REFERENCE_INDEX}")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig("rolling_correlations.png", bbox_inches='tight')
    
    plt.show()