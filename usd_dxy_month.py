import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import  timedelta
from dateutil.relativedelta import relativedelta

from usd_dxy import CurrencyAnalyzer
from visualizations import Visualizer
from data_saver import CurrencyDataSaver

from utils.error_handler import ErrorHandler




class MonthlyCurrencyAnalyzer:
    """
    A class that extends CurrencyAnalyzer to perform rolling monthly analysis over a 1-year period.
    
    This class analyzes currency pairs using sliding windows:
    - First window: Month 1
    - Second window: Month 1 + Month 2
    - Third window: Month 1 + Month 2 + Month 3
    - And so on until completing the year
    
    It also provides overall analysis for the full year period.
    """
    
    def __init__(self):
        """
        Initialize the MonthlyCurrencyAnalyzer with a base CurrencyAnalyzer instance.
        
        Parameters:
        -----------
        base_analyzer : CurrencyAnalyzer
            An instance of the CurrencyAnalyzer class
        config_path : str
            Path to the configuration file
        """
        # Create a new CurrencyAnalyzer if not provided
        
        self.base_analyzer = CurrencyAnalyzer()
        
     
        # Initialize with a 1-year period
        self.base_analyzer.config['period'] = '1y'
        
        # Storage for monthly analysis results
        self.monthly_data = {}
        self.monthly_percent_changes = {}
        self.monthly_correlation_matrices = {}
        self.monthly_correlation_pairs = {}
        
        # Complete year data
        self.year_data = None
        self.year_correlation_matrix = None
        self.year_correlation_pairs = None
        self.year_percent_changes = None

        self.error_handler = ErrorHandler(
            logger_name="MonthlyCurrencyAnalyzer", 
            debug_mode=False
        )
        
    
    
    def generate_monthly_windows(self):
        """
        Generate data windows for monthly cumulative analysis.
        
        This creates 12 windows:
        - Window 1: Month 1
        - Window 2: Month 1 + Month 2
        - Window 3: Month 1 + Month 2 + Month 3
        ...and so on
        
        Returns:
        --------
        dict
            Dictionary with window data for each period
        """
        if self.year_data is None:
            self.error_handler.logger.error("Year data not available. Call fetch_year_data first.")
            return False
        
        # Get the start date from the data
        start_date = self.year_data.index[0]
        
        # Generate 12 windows
        for i in range(1, 13):
            # Calculate end date for this window
            end_date = start_date + relativedelta(months=i) - timedelta(days=1)
            
            # Ensure we don't go beyond our data range
            if end_date > self.year_data.index[-1]:
                end_date = self.year_data.index[-1]
                
            # Get window name based on month count
            window_name = f"Month_{i}"
            
            # Get data slice for this window
            window_data = self.year_data.loc[start_date:end_date].copy()
            
            if not window_data.empty:
                self.monthly_data[window_name] = window_data
                self.error_handler.logger.info(
                    f"Created window {window_name}: {start_date.date()} to {end_date.date()} ({len(window_data)} days)"
                )
            else:
                self.error_handler.logger.warning(
                    f"Empty data for window {window_name}: {start_date.date()} to {end_date.date()}"
                )
        print(self.monthly_data)
        return self.monthly_data
    
    def analyze_monthly_windows(self):
        """
        Analyze each monthly window for percentage changes and correlations.
        
        Returns:
        --------
        dict
            Dictionary with analysis results for each window
        """
        if not self.monthly_data:
            self.error_handler.logger.error("Monthly window data not available. Call generate_monthly_windows first.")
            return False
        
        # Create temporary storage for the base analyzer data
        original_data = self.base_analyzer.data
        
        # Analyze each window
        for window_name, window_data in self.monthly_data.items():
            self.error_handler.logger.info(f"Analyzing {window_name}...")
            
            # Set the data in the base analyzer
            self.base_analyzer.data = window_data
            
            # Calculate percentage changes
            percent_changes = self.base_analyzer.calculate_percentage_changes()
            self.monthly_percent_changes[window_name] = percent_changes
            
            # Calculate correlations
            correlation_matrix = self.base_analyzer.calculate_correlations()
            if correlation_matrix is not None:
                self.monthly_correlation_matrices[window_name] = correlation_matrix
                self.monthly_correlation_pairs[window_name] = self.base_analyzer.correlation_pairs.copy()
        print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
        print(self.monthly_percent_changes)
        print("¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤")
        # Restore original data
        self.base_analyzer.data = original_data
        
        return {
            'percent_changes': self.monthly_percent_changes,
            'correlation_matrices': self.monthly_correlation_matrices,
            'correlation_pairs': self.monthly_correlation_pairs
        }
    
    def analyze_full_year_data(self):
        """
        Analyze the full year data for percentage changes and correlations.
        
        Returns:
        --------
        bool
            True if analysis completed successfully, False otherwise
        """
        if self.year_data is None:
            self.error_handler.logger.error("Year data not available. Call fetch_year_data first.")
            return False
        
        # Create temporary storage for the base analyzer data
        original_data = self.base_analyzer.data
        
        # Set the data in the base analyzer to the full year data
        self.base_analyzer.data = self.year_data
        
        # Calculate percentage changes for the full year
        self.year_percent_changes = self.base_analyzer.calculate_percentage_changes()
        
        # Calculate correlations for the full year
        self.year_correlation_matrix = self.base_analyzer.calculate_correlations()
        if self.year_correlation_matrix is not None:
            self.year_correlation_pairs = self.base_analyzer.correlation_pairs.copy()
        
        # Restore original data
        self.base_analyzer.data = original_data
        
        return True
    
    def run_analysis(self):
        """
        Run the complete monthly currency analysis workflow.
        
        Returns:
        --------
        bool
            True if analysis completed successfully, False otherwise
        """
        self.error_handler.logger.info("Starting monthly currency analysis workflow...")
        
        # Step 1: Fetch the year data
        success = self.base_analyzer.fetch_data()
        if success:
            self.year_data = self.base_analyzer.data.copy()
            self.error_handler.logger.info(f"Fetched year data with shape: {self.year_data.shape}")
        else:
            self.error_handler.logger.error("Year data fetching failed. Aborting analysis.")
            return False
        
        # Step 2: Analyze full year data first (before processing monthly windows)
        self.analyze_full_year_data()
            
        # Step 3: Generate monthly windows
        self.generate_monthly_windows()
        
        # # Step 4: Analyze monthly windows
        # self.analyze_monthly_windows()
        
        # # Step 5: Plot the results
        # visualizer = Visualizer()
        # visualizer.plot_percentage_changes(self.year_data, self.monthly_percent_changes['Month_12'],'./output/year_currency_trends.png')
        
        # visualizer.plot_monthly_summary(self.monthly_percent_changes,'./output/monthly_summary.png')

        # # Now use the pre-calculated correlation pairs for the full year data
        # # No need to recalculate correlations for the full year data again
        # if self.year_correlation_pairs:
        #     # Get top n most correlated pairs
        #     currency_pairs = [(pair[0], pair[1]) for pair in self.year_correlation_pairs[:3]]
        #     visualizer.plot_correlation_evolution(self.monthly_correlation_matrices, currency_pairs,'./output/correlation_evolution.png')
        
        # # Step 6: Save monthly statistics
        # data_saver = CurrencyDataSaver()
        # print("#########################")
        # print(self.monthly_percent_changes)
        # print("#########################")
        # data_saver.save_monthly_statistics(self.monthly_percent_changes, self.monthly_correlation_matrices,
        #                          self.monthly_correlation_pairs)

        
        # self.error_handler.logger.info("Monthly currency analysis workflow completed successfully.")
        # return True


# Example usage
if __name__ == "__main__":
    
   
    
    # Create monthly analyzer using the base analyzer
    monthly_analyzer = MonthlyCurrencyAnalyzer()
    
    # Run the analysis
    monthly_analyzer.run_analysis()