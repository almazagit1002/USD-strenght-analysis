import yfinance as yf
import pandas as pd

import numpy as np
import seaborn as sns
from datetime import datetime
import sys

from data_saver import CurrencyDataSaver
from visualizations import Visualizer
from utils.utils import load_config
from utils.error_handler import ErrorHandler


CONFIG_PATH = 'config.yaml'
class CurrencyAnalyzer:
    """
    A class to fetch, analyze and visualize currency pair data.
    
    This class handles the complete workflow of currency analysis:
    - Fetching data from Yahoo Finance
    - Data validation and preprocessing
    - Calculating percentage changes and correlations
    - Creating visualizations
    - Generating reports
    """
    


    def __init__(self):
        """
        Initialize the CurrencyAnalyzer with configuration.
        
        Parameters:
        -----------
        config_path : str
            Path to the main configuration file
        plot_config_path : str
            Path to the plotting configuration file
        """
        self.config = load_config(CONFIG_PATH)
        self.error_handler = ErrorHandler(
            logger_name="CurrencyAnalyzer", 
            debug_mode=False
        )
        
        # Dictionary to hold the currency pairs
        self.currency_pairs = {
            "DXY": "DX-Y.NYB",
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X", 
            "USDJPY": "USDJPY=X",
            "USDCAD": "USDCAD=X",
            "USDSEK": "USDSEK=X",
            "USDCHF": "USDCHF=X"
        }
        
        # Colors for consistent plotting
        self.colors = {
            'DXY': 'blue', 
            'EURUSD': 'green', 
            'GBPUSD': 'red', 
            'USDJPY': 'purple',
            'USDCAD': 'orange',
            'USDSEK': 'brown',
            'USDCHF': 'teal'
        }
        
        # Initialize data storage
        self.data = None
        self.returns = None
        self.correlation_matrix = None
        self.percent_changes = {}
        self.correlation_pairs = []
    
    def fetch_data(self):
        """
        Fetch currency pair data from Yahoo Finance.
        
        Returns:
        --------
        bool
            True if data was successfully fetched, False otherwise
        """
        self.error_handler.logger.info(f"Current date and time: {datetime.now()}")
        self.error_handler.logger.info("Fetching currency pair data from Yahoo Finance")
        
        # Create a DataFrame to store the data
        all_data = pd.DataFrame()
        
        # Fetch data for each currency pair
        for name, ticker in self.currency_pairs.items():
            try:
                self.error_handler.logger.info(f"Fetching {name} ({ticker})...")
                # Try direct download method which sometimes works better
                hist = yf.download(ticker, period=self.config['period'], interval=self.config['interval'])
                
                if hist.empty:
                    self.error_handler.logger.warning(f"No data received for {name}")
                    # Try alternative method
                    ticker_data = yf.Ticker(ticker)
                    hist = ticker_data.history(period=self.config['period'])
                    if hist.empty:
                        self.error_handler.logger.warning(f"Alternative method also failed for {name}")
                        continue
                
                self.error_handler.logger.info(f"Received {len(hist)} rows")
                self.error_handler.logger.debug(f"Date range: {hist.index.min()} to {hist.index.max()}")
                
                # Add to our combined DataFrame
                all_data[name] = hist['Close']
                
            except Exception as e:
                self.error_handler.logger.error(f"Error fetching {name}: {str(e)}")
        
        # Check if we have any data to plot
        if all_data.empty:
            self.error_handler.logger.error("No data was fetched for any currency pair. Cannot proceed.")
            return False
        
        # Store the data
        self.data = all_data.fillna(method='ffill')
        return True
    
    def calculate_percentage_changes(self):
        """
        Calculate percentage changes for each currency pair from the start date.
        
        Returns:
        --------
        dict
            Dictionary of currency pair names and their percentage changes
        """
        if self.data is None:
            self.error_handler.logger.error("No data available to calculate percentage changes.")
            return {}
        
        self.percent_changes = {}
        
        for column in self.data.columns:
            # Calculate percentage change using safe_calculation
            def calc_percent_change(df=self.data, col=column):
                # Check for valid data
                if df[col].isna().all() or len(df[col]) < 2:
                    self.error_handler.logger.warning(f"Insufficient data for {col} to calculate percentage change")
                    return 0.0
                    
                start_value = df[col].iloc[0]
                end_value = df[col].iloc[-1]
                
                # Log values to help with debugging
                self.error_handler.logger.debug(f"{col} start value: {start_value}, end value: {end_value}")
                
                # Validate the start value to avoid extreme percentage calculations
                if abs(start_value) < 0.0001:  # Avoid division by very small numbers
                    self.error_handler.logger.warning(f"Very small start value for {col}: {start_value}. Using substitute value.")
                    return 0.0
                    
                # Use handle_division for safe division
                percent_change = self.error_handler.handle_division(
                    numerator=(end_value - start_value), 
                    denominator=start_value, 
                    replace_value=0.0
                ) * 100
                
                # Cap extreme values that might indicate data problems
                if abs(percent_change) > 50:  # Threshold for unreasonable percentage change
                    self.error_handler.logger.warning(
                        f"Extreme percentage change detected for {col}: {percent_change}%. "
                        f"Values: start={start_value}, end={end_value}. Capping at ±50%."
                    )
                    return 50.0 if percent_change > 0 else -50.0
                    
                return percent_change
                
            percent_change = self.error_handler.safe_calculation(calc_percent_change, default_value=0.0)
            self.percent_changes[column] = percent_change
        
        return self.percent_changes
    
    def calculate_correlations(self):
        """
        Calculate correlation matrix for the currency pairs based on daily returns.
        
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix of currency pairs
        """
        if self.data is None:
            self.error_handler.logger.error("No data available to calculate correlations.")
            return None
        
        # Calculate returns for correlation analysis
        self.returns = self.data.pct_change().dropna()
        
        # Check if we have enough data
        if len(self.returns) < 5:
            self.error_handler.logger.warning("Not enough data points for reliable correlation analysis")
            return None
        
        # Calculate correlation matrix
        def calc_correlation_matrix(df=self.returns):
            return df.corr(method='pearson')
        
        self.correlation_matrix = self.error_handler.safe_calculation(
            calc_correlation_matrix, 
            default_value=pd.DataFrame(np.eye(len(self.returns.columns)), 
                                      index=self.returns.columns, 
                                      columns=self.returns.columns)
        )
        
        # Analyze correlation pairs
        self._analyze_correlation_pairs()
        
        return self.correlation_matrix
    
    def _analyze_correlation_pairs(self):
        """
        Analyze and rank correlation pairs from strongest to weakest.
        
        This is a helper method for calculate_correlations().
        """
        if self.correlation_matrix is None:
            return
        
        # Reset correlation pairs
        self.correlation_pairs = []
        
        # Find all unique pairs
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                col1 = self.correlation_matrix.columns[i]
                col2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                self.correlation_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation (descending)
        self.correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    
    def run_analysis(self):
        """
        Run the complete currency analysis workflow.
        
        This method orchestrates the entire analysis process from data fetching to visualization
        and report generation.
        
        Returns:
        --------
        bool
            True if analysis completed successfully, False otherwise
        """
        self.error_handler.logger.info("Starting currency analysis workflow...")
        
        # Step 1: Fetch the data
        if not self.fetch_data():
            self.error_handler.logger.error("Data fetching failed. Aborting analysis.")
            return False
        
        self.calculate_percentage_changes()
        
        #visualizations
        visualizer = Visualizer()
        visualizer.plot_percentage_changes(self.data, self.percent_changes)
        
        # Calculate correlations
        self.calculate_correlations()
        
       
        visualizer.plot_correlation_heatmap(self.correlation_matrix)
        
        # Save summary statistics
        data_saver = CurrencyDataSaver()
        
        data_saver.save_summary_stats(self.data,self.percent_changes, self.correlation_pairs)
        
        # Save data to CSV
        data_saver.save_data_to_csv(self.data,self.returns,self.correlation_matrix)
        
        self.error_handler.logger.info("Currency analysis workflow completed successfully.")
        


# Example usage
if __name__ == "__main__":
    analyzer = CurrencyAnalyzer()
    analyzer.run_analysis()