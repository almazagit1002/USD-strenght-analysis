import pandas as pd
from datetime import datetime
from utils.error_handler import ErrorHandler
from utils.utils import load_config


class CurrencyDataSaver:
    """
    A class responsible for saving currency data and statistics to files.
    """

    def __init__(self):
        """
        Initialize the CurrencyDataSaver class without data arguments.
        """
        self.config = load_config('config.yaml')
        self.error_handler = ErrorHandler(
            logger_name="CurrencyDataSaver", 
            debug_mode=False
        )
    
    def save_data_to_csv(self, data=None, returns=None, correlation_matrix=None, base_filename='currency_data'):
        """
        Save data to CSV files.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data for currency pairs
        returns : pandas.DataFrame
            Returns data for currency pairs
        correlation_matrix : pandas.DataFrame
            Correlation matrix between currency pairs
        base_filename : str
            Base filename for the CSV files
            
        Returns:
        --------
        bool
            True if all data was saved successfully, False otherwise
        """
        success = True
        
        try:
            # Save price data
            if data is not None:
                data.to_csv(f'{base_filename}_prices.csv')
                self.error_handler.logger.info(f"Price data saved to '{base_filename}_prices.csv'")
            else:
                success = False
                self.error_handler.logger.warning("No price data available to save.")
                
            # Save returns data
            if returns is not None:
                returns.to_csv(f'{base_filename}_returns.csv')
                self.error_handler.logger.info(f"Returns data saved to '{base_filename}_returns.csv'")
            else:
                self.error_handler.logger.warning("No returns data available to save.")
                
            # Save correlation matrix
            if correlation_matrix is not None:
                correlation_matrix.to_csv(f'{base_filename}_correlation_matrix.csv')
                self.error_handler.logger.info(f"Correlation matrix saved to '{base_filename}_correlation_matrix.csv'")
            else:
                self.error_handler.logger.warning("No correlation matrix available to save.")
                
        except Exception as e:
            self.error_handler.logger.error(f"Error saving data to CSV: {str(e)}")
            success = False
            
        return success
    
    def save_summary_stats(self, data=None, percent_changes=None, correlation_pairs=None, filename='currency_summary_stats.txt'):
        """
        Save summary statistics to a text file.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data for currency pairs
        percent_changes : dict
            Dictionary containing percentage changes for each currency pair
        correlation_pairs : list
            List of tuples containing correlated pairs and their correlation values
        filename : str
            Filename to save the stats to
            
        Returns:
        --------
        bool
            True if stats were saved successfully, False otherwise
        """
        if data is None:
            self.error_handler.logger.error("No data available to save statistics.")
            return False
        
        try:
            with open(filename, 'w') as f:
                f.write("Currency Pairs - Summary Statistics\n")
                f.write("=================================\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Check if the data has a datetime index before using strftime
                if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                    start_date = data.index.min()
                    end_date = data.index.max()
                    
                    # Ensure the index values are datetime objects
                    if hasattr(start_date, 'strftime') and hasattr(end_date, 'strftime'):
                        f.write(f"Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n")
                    else:
                        f.write(f"Time Period: {start_date} to {end_date}\n\n")
                else:
                    f.write("Time Period: Not available (index is not datetime)\n\n")
                
                # Write percentage changes
                period = self.config['period'][0]
                f.write(f"Percentage Changes Over {period} Months Period:\n")
                f.write("---------------------------------\n")
                
                if percent_changes:
                    for column, change in percent_changes.items():
                        stats_line = f"{column}: {change:.2f}% change over period"
                        self.error_handler.logger.info(stats_line)
                        f.write(stats_line + "\n")
                else:
                    f.write("No percentage change data available.\n")
                
                # Add correlation section if available
                if correlation_pairs:
                    f.write("\n\nCorrelation Analysis:\n")
                    f.write("--------------------\n")
                    f.write("Top correlations between currency pairs:\n")
                    
                    top_n = min(5, len(correlation_pairs))
                    for i in range(top_n):
                        col1, col2, corr = correlation_pairs[i]
                        relationship = "positive" if corr > 0 else "negative"
                        
                        # Categorize correlation strength
                        if abs(corr) > 0.7:
                            strength = "strong"
                        elif abs(corr) > 0.4:
                            strength = "moderate"
                        else:
                            strength = "weak"
                        
                        f.write(f"- {col1} and {col2}: {strength} {relationship} correlation ({corr:.3f})\n")
            
            self.error_handler.logger.info(f"Summary statistics saved to '{filename}'")
            return True
            
        except Exception as e:
            self.error_handler.logger.error(f"Error saving summary statistics: {str(e)}")
            return False
        
    def save_monthly_statistics(self, monthly_percent_changes, monthly_correlation_matrices,
                             monthly_correlation_pairs, output_dir="./output"):
        """
        Save monthly statistics to CSV files and a summary text file with yearly analysis.
        
        Parameters:
        -----------
        monthly_percent_changes : dict
            Dictionary of monthly percentage changes
        monthly_correlation_matrices : dict
            Dictionary of monthly correlation matrices
        monthly_correlation_pairs : dict
            Dictionary of monthly correlation pairs
        output_dir : str
            Directory to save the output files
        """
        import os
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save percentage changes
        percent_changes_df = pd.DataFrame()
        for window, changes in monthly_percent_changes.items():
            for pair, value in changes.items():
                percent_changes_df.loc[window, pair] = value
        
        percent_changes_df.to_csv(f"{output_dir}/monthly_percent_changes.csv")
        
        # Save correlation matrices
        all_correlations = {}
        for window, corr_matrix in monthly_correlation_matrices.items():
            for pair in monthly_correlation_pairs[window]:
                pair_name = f"{pair[0]}-{pair[1]}"
                if pair_name not in all_correlations:
                    all_correlations[pair_name] = {}
                all_correlations[pair_name][window] = pair[2]
        
        correlations_df = pd.DataFrame(all_correlations)
        correlations_df.to_csv(f"{output_dir}/monthly_correlations.csv")
        
        try:
            with open(f"{output_dir}/monthly_statistics_summary.txt", 'w') as f:
                # Write header
                f.write("Currency Pairs - Summary Statistics\n")
                f.write("=================================\n\n")
                
                # Write analysis date and period
                current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"Analysis Date: {current_date}\n")
                
                # Get first and last window to determine the analysis period
                all_windows = sorted(monthly_percent_changes.keys())
                first_window = all_windows[0]
                last_window = all_windows[-1]
                
                # Extract dates from window strings (assuming format like "2024-01" or similar)
                start_date = first_window
                end_date = last_window
                f.write(f"Time Period: {start_date} to {end_date}\n\n")
                
                # Calculate yearly percentage changes
                f.write(f"Percentage Changes Over {len(all_windows)} Months Period:\n")
                f.write("---------------------------------\n")
                
                # Get all currency pairs
                all_pairs = set()
                for changes in monthly_percent_changes.values():
                    all_pairs.update(changes.keys())
                
                # Calculate yearly changes for each pair
                yearly_changes = {}
                for pair in all_pairs:
                    # Convert monthly percentage changes to multipliers
                    monthly_multipliers = []
                    for window in all_windows:
                        if pair in monthly_percent_changes[window]:
                            monthly_multipliers.append(1 + monthly_percent_changes[window][pair] / 100)
                        else:
                            # If data is missing for a month, assume no change
                            monthly_multipliers.append(1.0)
                    
                    # Calculate cumulative change over the entire period
                    cumulative_multiplier = 1.0
                    for multiplier in monthly_multipliers:
                        cumulative_multiplier *= multiplier
                    
                    # Convert back to percentage
                    yearly_changes[pair] = (cumulative_multiplier - 1) * 100
                    f.write(f"{pair}: {yearly_changes[pair]:.2f}% change over period\n")
                
                f.write("\n\n")
                
                # Write correlation analysis for the full period
                f.write("Correlation Analysis:\n")
                f.write("--------------------\n")
                
                # Combine all correlation pairs across all windows
                all_pair_correlations = {}
                for window, pairs in monthly_correlation_pairs.items():
                    for pair in pairs:
                        pair_key = (pair[0], pair[1])
                        if pair_key not in all_pair_correlations:
                            all_pair_correlations[pair_key] = []
                        all_pair_correlations[pair_key].append(pair[2])
                
                # Calculate average correlation for each pair
                avg_correlations = []
                for pair_key, values in all_pair_correlations.items():
                    avg_corr = sum(values) / len(values)
                    avg_correlations.append((pair_key[0], pair_key[1], avg_corr))
                
                # Sort by absolute correlation value
                sorted_correlations = sorted(avg_correlations, key=lambda x: abs(x[2]), reverse=True)
                
                # Display top correlations
                f.write("Top correlations between currency pairs:\n")
                top_n = min(5, len(sorted_correlations))
                for i in range(top_n):
                    pair1, pair2, corr_value = sorted_correlations[i]
                    
                    relationship = "positive" if corr_value > 0 else "negative"
                    
                    if abs(corr_value) > 0.7:
                        strength = "strong"
                    elif abs(corr_value) > 0.4:
                        strength = "moderate"
                    else:
                        strength = "weak"
                    
                    f.write(f"- {pair1} and {pair2}: {strength} {relationship} correlation ({corr_value:.3f})\n")
                    
                # Also include the monthly analysis if needed
                if len(all_windows) > 1:  # Only if there are multiple months
                    f.write("\n\nMonthly Breakdown:\n")
                    f.write("----------------\n")
                    
                    # For each month, show brief statistics
                    for window in all_windows:
                        f.write(f"\n{window}:\n")
                        
                        # Show percentage changes
                        f.write("Percentage changes:\n")
                        for pair, value in sorted(monthly_percent_changes[window].items()):
                            f.write(f"  {pair}: {value:.2f}%\n")
                        
                        # Show top correlation for this month
                        f.write("Top correlation:\n")
                        if monthly_correlation_pairs[window]:
                            sorted_pairs = sorted(monthly_correlation_pairs[window], 
                                                key=lambda x: abs(x[2]), reverse=True)
                            pair1, pair2, corr_value = sorted_pairs[0]
                            relationship = "positive" if corr_value > 0 else "negative"
                            f.write(f"  {pair1} and {pair2}: {relationship} correlation ({corr_value:.3f})\n")
        
        except Exception as e:
            self.error_handler.logger.error(f"Error creating monthly statistics summary text file: {str(e)}")
        
        self.error_handler.logger.info(f"Saved monthly statistics to {output_dir}")