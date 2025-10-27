import pandas as pd
from datetime import datetime, timedelta
import time
import random
import logging
from typing import Optional
import pandas_datareader.data as web

class BaseFetcher:
    """Base class for all data fetchers"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def fetch_ticker_data(self, ticker_symbol: str, years: int) -> Optional[pd.DataFrame]:
        raise NotImplementedError("Subclasses must implement fetch_ticker_data")

class StooqFetcher(BaseFetcher):
    """Fetcher for Stooq data source"""
    
    def __init__(self, base_url: str, logger: logging.Logger):
        super().__init__(logger)
        self.base_url = base_url
    
    def fetch_ticker_data(self, ticker_symbol: str, years: int) -> Optional[pd.DataFrame]:
        """Your existing Stooq fetching logic"""
        try:
            end = datetime.today()
            start = end - timedelta(days=years * 365)
            url = (f"{self.base_url}?s={ticker_symbol}"
                  f"&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d")
            
            self.logger.info(f"Fetching data for {ticker_symbol} from Stooq")
            
            df = pd.read_csv(url, parse_dates=["Date"]).set_index("Date").sort_index()
            
            if df.empty:
                self.logger.warning(f"No data returned for {ticker_symbol}")
                return None
                
            # Add delay to respect rate limits
            delay = 1.0 + random.uniform(0, 1.0)
            time.sleep(delay)
            
            self.logger.info(f"Successfully fetched {len(df)} records for {ticker_symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker_symbol}: {str(e)}")
            return None

class FredFetcher(BaseFetcher):
    """Fetcher for FRED (Federal Reserve Economic Data) source"""
    
    def __init__(self, logger: logging.Logger):
        super().__init__(logger)
    
    def fetch_ticker_data(self, ticker_symbol: str, years: int) -> Optional[pd.DataFrame]:
        """Fetch data from FRED using pandas_datareader"""
        try:
            end = datetime.today()
            start = end - timedelta(days=years * 365)
            
            self.logger.info(f"Fetching {ticker_symbol} from FRED from {start.date()} to {end.date()}")
            
            # Use pandas_datareader to get FRED data
            df = web.DataReader(ticker_symbol, 'fred', start, end)
            
            if df.empty:
                self.logger.warning(f"No data returned for {ticker_symbol} from FRED")
                return None
            
            # Rename the column to match Stooq format
            df.columns = ['Close']  # FRED data becomes 'Close' for consistency
            
            # Add delay to respect rate limits
            time.sleep(0.5)  # FRED is generally more lenient
            
            self.logger.info(f"Successfully fetched {len(df)} records for {ticker_symbol} from FRED")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching FRED data for {ticker_symbol}: {str(e)}")
            return None