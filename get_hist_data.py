import pandas as pd
import yfinance as yf
import boto3
from datetime import datetime, timedelta
import calendar
import logging
import os
import io
from dotenv import load_dotenv


load_dotenv()



class CurrencyDataCollector:
    def __init__(self):
        """
        Initialize the CurrencyDataCollector with AWS credentials.
        
        Parameters:
        -----------
        aws_access_key : str
            AWS access key for S3 bucket access
        aws_secret_key : str
            AWS secret key for S3 bucket access
        region_name : str
            AWS region name
        """
        self.logger = self._setup_logger()
        
        # Dictionary to hold the currency pairs
        self.currency_pairs = {
            #Currency Pairs
            "DXY": "DX-Y.NYB",
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X", 
            "USDJPY": "USDJPY=X",
            "USDCAD": "USDCAD=X",
            "USDSEK": "USDSEK=X",
            "USDCHF": "USDCHF=X",
            
            # Cryptocurrencies
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            
            # US Equity Indices
            "S&P500": "^GSPC",
            "SPY": "SPY",  # S&P 500 ETF
            "NASDAQ": "^IXIC",
            "NASDAQ100": "^NDX",
            "QQQ": "QQQ",  # NASDAQ 100 ETF
            "DowJones": "^DJI",
            "DIA": "DIA",  # Dow Jones ETF
            
            # International Indices
            "FTSE100": "^FTSE",  # UK
            "DAX": "^GDAXI",     # Germany
            "Nikkei225": "^N225", # Japan
            "ShanghaiComp": "^SSEC", # China
            "HangSeng": "^HSI",   # Hong Kong
            "VEA": "VEA",        # Developed Markets ETF
            "VWO": "VWO",        # Emerging Markets ETF
            
            # Commodities
            "Gold": "GC=F",
            "Silver": "SI=F",
            "CrudeOil": "CL=F",
            "BrentOil": "BZ=F",
            "NaturalGas": "NG=F",
            
            # Interest Rates and Bonds
            "US10Y": "^TNX",      # 10-Year Treasury Yield
            "US2Y": "^UST2Y",     # 2-Year Treasury Yield
            "US30Y": "^TYX",      # 30-Year Treasury Yield
            "TLT": "TLT",         # 20+ Year Treasury Bond ETF
            "IEF": "IEF",         # 7-10 Year Treasury Bond ETF
            "SHY": "SHY",         # 1-3 Year Treasury Bond ETF
            
            # Volatility and Market Sentiment
            "VIX": "^VIX",        # CBOE Volatility Index
            "VVIX": "^VVIX",      # VIX Volatility Index
            "MOVE": "^MOVE",      # Bond Market Volatility
            "SKEW": "^SKEW",      # Black Swan Index
            "PUT_CALL": "^PCR",   # Put/Call Ratio
            
            # Major Sector ETFs
            "XLF": "XLF",         # Financial Sector
            "XLE": "XLE",         # Energy Sector
            "XLK": "XLK",         # Technology Sector
            "XLV": "XLV",         # Healthcare Sector
            "XLI": "XLI",         # Industrial Sector
            "XLP": "XLP",         # Consumer Staples
            "XLY": "XLY",         # Consumer Discretionary
            "XLU": "XLU",         # Utilities Sector
            "XLB": "XLB",         # Materials Sector
            "XLRE": "XLRE",       # Real Estate Sector
            "XLC": "XLC",         # Communication Services
            
            # Thematic ETFs
            "ARKK": "ARKK",       # Innovation ETF
            "SMH": "SMH",         # Semiconductor ETF
            "IBB": "IBB",         # Biotech ETF
            "ITA": "ITA",         # Aerospace & Defense ETF
            "JETS": "JETS",       # Airline ETF
            "KWEB": "KWEB"        # China Internet ETF
        }
        
        # Initialize AWS S3 client
        
        self.s3_client = boto3.client('s3')
               
    def _setup_logger(self):
        """Set up and return a logger."""
        logger = logging.getLogger('CurrencyDataCollector')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        return logger
    
    def fetch_and_save_currency_data(self, bucket_name, years=2, remove_incomplete_months=True,create_bucket_if_not_exists=True):
        """
        Fetch currency data for the specified number of years, remove incomplete months if requested,
        and save to an S3 bucket.
        
        Parameters:
        -----------
        bucket_name : str
            Name of the S3 bucket
        years : int
            Number of years of data to fetch (default 5)
        remove_incomplete_months : bool
            Whether to remove incomplete first and last months
        
        Returns:
        --------
        dict
            Dictionary containing the S3 URLs of saved files
        """
        self.logger.info(f"Starting data collection for {years} years for {len(self.currency_pairs)} currency pairs")
        
        # Calculate start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        # If we need to remove incomplete months, adjust dates
        if remove_incomplete_months:
            # Adjust start date to the first day of the next month if current month is incomplete
            if start_date.day != 1:
                # Move to the first day of the next month
                if start_date.month == 12:
                    start_date = datetime(start_date.year + 1, 1, 1)
                else:
                    start_date = datetime(start_date.year, start_date.month + 1, 1)
            
            # Adjust end date to the last day of the previous month if current month is incomplete
            if end_date.day < calendar.monthrange(end_date.year, end_date.month)[1]:
                # Move to the last day of the previous month
                if end_date.month == 1:
                    last_day = calendar.monthrange(end_date.year - 1, 12)[1]
                    end_date = datetime(end_date.year - 1, 12, last_day)
                else:
                    last_day = calendar.monthrange(end_date.year, end_date.month - 1)[1]
                    end_date = datetime(end_date.year, end_date.month - 1, last_day)
        
        # Format dates for yfinance
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        self.logger.info(f"Adjusted date range: {start_date_str} to {end_date_str}")

        if create_bucket_if_not_exists:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                self.logger.info(f"Bucket {bucket_name} exists")
            except Exception as e:
                if "404" in str(e) or "NoSuchBucket" in str(e):
                    self.logger.info(f"Bucket {bucket_name} does not exist. Creating it...")
                    try:
                        
                        self.s3_client.create_bucket(Bucket=bucket_name)
                       
                        self.logger.info(f"Bucket {bucket_name} created successfully")
                    except Exception as create_error:
                        self.logger.error(f"Error creating bucket: {str(create_error)}")
                        raise
                else:
                    self.logger.error(f"Error checking bucket: {str(e)}")
                    raise
        
        # Dictionary to store results
        results = {}
        
        # Fetch data for each currency pair
        all_data = {}
        for name, ticker in self.currency_pairs.items():
            try:
                self.logger.info(f"Fetching data for {name} ({ticker})")
                data = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
                
                # Check if data is not empty
                if data.empty:
                    self.logger.warning(f"No data found for {name} ({ticker})")
                    continue
                
                # Store the data
                all_data[name] = data
                self.logger.info(f"Successfully fetched {len(data)} records for {name}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {name}: {str(e)}")
        
        # Save individual currency pair data
        for name, data in all_data.items():
            try:
                
                # Save as Parquet
                parquet_key = f"currency_data/{name}/{start_date_str}_to_{end_date_str}.parquet"
                self._save_to_s3(data, bucket_name, parquet_key, format='parquet')
                results[f"{name}_parquet"] = f"s3://{bucket_name}/{parquet_key}"
            except Exception as e:
                self.logger.error(f"Error saving {name} data: {str(e)}")
        
        # Save combined dataset
        try:
            # Create a combined dataset with adjusted close prices
            combined_df = pd.DataFrame()
            for name, data in all_data.items():
                if 'Adj Close' in data.columns:
                    combined_df[name] = data['Adj Close']
                elif 'Close' in data.columns:
                    combined_df[name] = data['Close']
            
            if not combined_df.empty:
          
                # Save combined data as Parquet
                combined_parquet_key = f"currency_data/combined/{start_date_str}_to_{end_date_str}.parquet"
                self._save_to_s3(combined_df, bucket_name, combined_parquet_key, format='parquet')
                results["combined_parquet"] = f"s3://{bucket_name}/{combined_parquet_key}"
        except Exception as e:
            self.logger.error(f"Error saving combined data: {str(e)}")
        
        # Save metadata
        try:
            metadata = {
                "currency_pairs": list(self.currency_pairs.keys()),
                "start_date": start_date_str,
                "end_date": end_date_str,
                "total_days": (end_date - start_date).days,
                "collection_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "removed_incomplete_months": remove_incomplete_months
            }
            
            # Convert metadata to DataFrame for easy saving
            metadata_df = pd.DataFrame([metadata])
            metadata_key = f"currency_data/metadata/{start_date_str}_to_{end_date_str}.csv"
            self._save_to_s3(metadata_df, bucket_name, metadata_key, format='csv')
            results["metadata"] = f"s3://{bucket_name}/{metadata_key}"
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
        
        return results
    
    def _save_to_s3(self, df, bucket_name, key, format='csv'):
        """
        Save DataFrame to S3 bucket.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to save
        bucket_name : str
            S3 bucket name
        key : str
            S3 object key (path)
        format : str
            File format ('csv' or 'parquet')
        """
        buffer = io.BytesIO()
        
        if format.lower() == 'csv':
            df.to_csv(buffer)
        elif format.lower() == 'parquet':
            df.to_parquet(buffer)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        buffer.seek(0)
        
        try:
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=key,
                Body=buffer.getvalue()
            )
            self.logger.info(f"Successfully saved to s3://{bucket_name}/{key}")
        except Exception as e:
            self.logger.error(f"Error saving to S3: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":

    BUCKET_NAME = os.getenv("BUCKET_NAME")

    
    # Initialize the collector
    collector = CurrencyDataCollector()
    
    # Fetch and save data
    # Retrieve the S3 bucket name

    bucket_name = BUCKET_NAME  
    results = collector.fetch_and_save_currency_data(
        bucket_name=bucket_name,
        years=2,
        remove_incomplete_months=True,
        create_bucket_if_not_exists=True
    )
    print(results)
    # Print results
    for name, url in results.items():
        print(f"{name}: {url}")