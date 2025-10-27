import pandas as pd
import logging
from typing import Dict, Optional

from src.fetcher_factory import FetcherFactory
from src.configurations import DataSourceConfig


class DataManager:
    """Enhanced data manager with intelligent missing data handling and DXY calculations"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{config.fetch_name}")
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        
   
    
    def fetch_all_data(self, years: int) -> bool:
        """Enhanced version that handles multiple sources"""
        if self.config.is_multi_source():
            return self._fetch_multi_source_data(years)
        else:
            return self._fetch_single_source_data(years)
    
    def _fetch_single_source_data(self, years: int) -> bool:
        """Your existing fetch logic for single source"""
        self.logger.info(f"Starting to fetch data for {len(self.config.tickers)} pairs")
        
        # Create fetcher (defaults to Stooq for backward compatibility)
        fetcher = FetcherFactory.create_fetcher('stooq', self.config.base_url, self.logger)
        
        success_count = 0
        for name, symbol in self.config.tickers.items():
            df = fetcher.fetch_ticker_data(symbol, years)
            if df is not None:
                self.raw_data[name] = df
                success_count += 1
            else:
                self.logger.warning(f"Failed to fetch data for {name} ({symbol})")
        
        self.logger.info(f"Successfully fetched data for {success_count}/{len(self.config.tickers)}")
        return success_count > 0
    
    def _fetch_multi_source_data(self, years: int) -> bool:
        """logic for multiple sources"""
        self.logger.info(f"Starting multi-source data fetch for {len(self.config.sources)} sources")
        
        total_success = 0
        total_tickers = 0
        
        for source in self.config.sources:
            self.logger.info(f"Fetching from {source.type} source")
            
            # Create appropriate fetcher
            fetcher = FetcherFactory.create_fetcher(source.type, source.base_url, self.logger)
            
            # Fetch data for this source
            for name, symbol in source.tickers.items():
                total_tickers += 1
                df = fetcher.fetch_ticker_data(symbol, years)
                if df is not None:
                    self.raw_data[name] = df
                    total_success += 1
                else:
                    self.logger.warning(f"Failed to fetch data for {name} ({symbol}) from {source.type}")
        
        self.logger.info(f"Multi-source fetch complete: {total_success}/{total_tickers} successful")
        return total_success > 0
    
    
    
    def combine_data_smart(self, missing_threshold: float = 0.7, 
                          fill_method: str = 'forward') -> Optional[pd.DataFrame]:
        """
        Intelligently combine data with better missing data handling
        
        Args:
            missing_threshold: Maximum allowed missing data ratio per column (0.7 = 70%)
            fill_method: Method to handle missing data ('forward', 'backward', 'interpolate', 'drop', 'hybrid')
        
        Returns:
            Combined DataFrame with intelligent missing data handling
        """
        if not self.raw_data:
            self.logger.error("No raw data available to combine")
            return None
        
        # Get all dataframes
        dataframes = []
        symbols = []
        
        for symbol, df in self.raw_data.items():
            if df is not None and not df.empty:
                # Use 'close' price as the main metric (adjust based on your data structure)
                price_col = self._get_price_column(df)
                if price_col:
                    series = df[price_col].copy()
                    series.name = symbol
                    dataframes.append(series)
                    symbols.append(symbol)
        
        if not dataframes:
            self.logger.error("No valid data found in any dataset")
            return None
        
        # Combine all series
        combined_df = pd.concat(dataframes, axis=1, sort=True)
        original_shape = combined_df.shape
        self.logger.info(f"Initial combined shape: {original_shape}")
        
        # Analyze missing data
        missing_stats = self._analyze_missing_data(combined_df)
        self.logger.info(f"Missing data analysis:\n{missing_stats}")
        
        # Remove columns with too much missing data
        columns_to_keep = []
        for col in combined_df.columns:
            missing_ratio = combined_df[col].isna().mean()
            if missing_ratio <= missing_threshold:
                columns_to_keep.append(col)
            else:
                self.logger.warning(f"Removing {col}: {missing_ratio:.2%} missing data")
        
        if not columns_to_keep:
            self.logger.error("All columns exceed missing data threshold")
            return None
        
        combined_df = combined_df[columns_to_keep]
        
        # Handle missing data based on method
        combined_df = self._handle_missing_data(combined_df, fill_method)
        
        final_shape = combined_df.shape
        self.logger.info(f"Final combined shape: {final_shape}")
        self.logger.info(f"Data retention: {final_shape[0]/original_shape[0]:.2%} rows, "
                        f"{final_shape[1]/original_shape[1]:.2%} columns")
        
        self.combined_data = combined_df
        return combined_df
    
    def _get_price_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if col.lower() in ['close', 'price', 'value']:
                return col
        # Fallback to first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else None
        
    def _analyze_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing data patterns"""
        missing_stats = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': [df[col].isna().sum() for col in df.columns],
            'Missing_Percentage': [df[col].isna().mean() * 100 for col in df.columns],
            'Data_Points': [df[col].notna().sum() for col in df.columns]
        })
        return missing_stats.sort_values('Missing_Percentage', ascending=False)
    
    def _handle_missing_data(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Handle missing data based on specified method"""
        original_rows = len(df)
        
        if method == 'forward':
            df_filled = df.fillna(method='ffill')
            self.logger.info("Applied forward fill")
        
        elif method == 'backward':
            df_filled = df.fillna(method='bfill')
            self.logger.info("Applied backward fill")
        
        elif method == 'interpolate':
            df_filled = df.interpolate(method='linear')
            self.logger.info("Applied linear interpolation")
        
        elif method == 'drop':
            df_filled = df.dropna()
            self.logger.info(f"Dropped rows with missing data: {original_rows - len(df_filled)} rows removed")
        
        else:
            # Hybrid approach: interpolate then forward fill
            df_filled = df.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
            self.logger.info("Applied hybrid approach: interpolation + forward/backward fill")
        
        # Final cleanup - remove any remaining NaN rows if they exist
        if method != 'drop':
            before_final_cleanup = len(df_filled)
            df_filled = df_filled.dropna()
            removed_final = before_final_cleanup - len(df_filled)
            if removed_final > 0:
                self.logger.info(f"Final cleanup: removed {removed_final} rows with remaining NaN values")
        
        return df_filled
    
    
   