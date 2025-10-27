import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from utils.utils import load_config
from utils.error_handler import ErrorHandler


class Visualizer:
    """
    A utility class for creating and saving currency data visualizations.
    
    This class handles various plotting functions for currency data analysis,
    including percentage change plots and correlation heatmaps.
    """
    
    def __init__(self, debug_mode=False):
        """
        Initialize the CurrencyPlotter with configuration and error handling.
        
        Parameters:
        -----------
        plot_config_path : str
            Path to the plotting configuration file
        error_handler : ErrorHandler, optional
            An existing error handler instance to use
        debug_mode : bool, optional
            Whether to enable debug mode if creating a new error handler
        """
        self.plot_config = load_config('plot_config.yaml')
        
        # Use the provided error handler or create a new one
        
        self.error_handler = ErrorHandler(
                logger_name="Visualizer", 
                debug_mode=debug_mode
            )
        
        # Default colors for currency pairs
        self.colors = {
            'DXY': 'blue', 
            'EURUSD': 'green', 
            'GBPUSD': 'red', 
            'USDJPY': 'purple',
            'USDCAD': 'orange',
            'USDSEK': 'brown',
            'USDCHF': 'teal'
        }
    
    def plot_percentage_changes(self, data, percent_changes, filename='currency_pairs_chart.png'):
        """
        Create and save a plot of percentage changes for all currency pairs.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data with currency pairs as columns
        percent_changes : dict
            Dictionary mapping column names to their percentage changes
        colors : dict, optional
            Dictionary mapping column names to their plot colors
        filename : str, optional
            Filename to save the plot to
            
        Returns:
        --------
        bool
            True if plot was created and saved successfully, False otherwise
        """
        if data is None or data.empty:
            self.error_handler.logger.error("No data available to plot.")
            return False
        
        # Use provided colors or fall back to default colors
        plot_colors =  self.colors
        try:
            # Create figure with one subplot for percentage change
            fig, ax = plt.subplots(figsize=self.plot_config['figsize'])
            
            # Dictionary to store coordinates for annotations
            line_ends = {}
            
            # Plot percentage changes for each currency pair
            for column in data.columns:
                # Calculate normalized percentage change
                def normalize_data(df=data, col=column):
                    if df[col].iloc[0] == 0:
                        self.error_handler.logger.warning(f"First value for {col} is zero, cannot normalize")
                        return pd.Series([0] * len(df), index=df.index)
                    return df[col] / df[col].iloc[0] * 100 - 100
                
                normalized = self.error_handler.safe_calculation(
                    normalize_data, 
                    default_value=pd.Series([0] * len(data), index=data.index)
                )
                
                # Plot the line with color from dict or use default matplotlib color
                line, = ax.plot(
                    data.index, 
                    normalized, 
                    label=column, 
                    color=plot_colors.get(column), 
                    linewidth=2
                )
                
                # Store the coordinates of the final point for annotation
                line_ends[column] = (data.index[-1], normalized.iloc[-1])
            
            # Add annotations for percentage changes
            for column, change in percent_changes.items():
                if column not in data.columns:
                    continue  # Skip if column is not in the data
                    
                try:
                    x, y = line_ends[column]
                    ax.annotate(
                        f"{column}: {change:.2f}%", 
                        xy=(x, y),
                        xytext=(10, 0),  # Small offset from the end of line
                        textcoords="offset points",
                        fontsize=12,
                        color=plot_colors.get(column),
                        weight='bold'
                    )
                except Exception as e:
                    self.error_handler.logger.error(f"Error adding annotation for {column}: {str(e)}")
            
            # Set plot formatting
            ax.set_title('Currency Pairs - Percentage Change Since Start', 
                        fontsize=self.plot_config['title'], 
                        weight='bold')
            ax.set_ylabel('Percentage Change (%)', fontsize=self.plot_config['ax_labels'])
            ax.set_xlabel('Date', fontsize=self.plot_config['ax_labels'])
            ax.tick_params(axis='both', which='major', labelsize=self.plot_config['tick_label'])
            ax.grid(True)
            ax.legend(loc='best', fontsize=self.plot_config['legend'])
            
            # Format x-axis dates
            fig.autofmt_xdate()
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            self.error_handler.logger.info(f"Plot saved as '{filename}'")
            plt.close(fig)
            
            return True
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating percentage change plot: {str(e)}")
            return False
    
    def plot_correlation_heatmap(self, correlation_matrix, filename='currency_correlation_heatmap.png'):
        """
        Create and save a heatmap of currency pair correlations.
        
        Parameters:
        -----------
        correlation_matrix : pandas.DataFrame
            Correlation matrix of currency pairs
        filename : str, optional
            Filename to save the heatmap to
            
        Returns:
        --------
        bool
            True if heatmap was created and saved successfully, False otherwise
        """
        if correlation_matrix is None or correlation_matrix.empty:
            self.error_handler.logger.error("No correlation data available to plot.")
            return False
        
        try:
            # Create a new figure for the correlation heatmap
            plt.figure(figsize=self.plot_config['heatmap_fig_size'])
            
            # Use seaborn to create a heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask for the upper triangle
            
            # Custom colormap: dark blue for negative correlations, white for neutral, dark red for positive
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            # Plot the heatmap
            heatmap = sns.heatmap(
                correlation_matrix, 
                mask=mask,
                annot=True,  # Show the correlation values
                fmt=".2f",   # Format with 2 decimal places
                linewidths=0.5,
                cmap=cmap,
                vmin=-1, vmax=1,  # Correlation ranges from -1 to 1
                square=True,      # Make cells square
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
            )
            
            plt.title('Currency Pair Correlation Matrix', fontsize=self.plot_config['title'], weight='bold')
            
            # Increase font size for the annotations
            for text in heatmap.texts:
                text.set_fontsize(self.plot_config['heatmap_font_size'])
            
            plt.tight_layout()
            
            # Save the correlation heatmap
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            self.error_handler.logger.info(f"Correlation heatmap saved as '{filename}'")
            plt.close()
            
            return True
            
        except Exception as e:
            self.error_handler.logger.error(f"Error creating correlation heatmap: {str(e)}")
            return False

    def plot_monthly_summary(self, monthly_percent_changes, save_path=None):
            """
            Plot summary of percentage changes for each cumulative month window.
            
            Parameters:
            -----------
            save_path : str
                Path to save the plot
            """
            if not monthly_percent_changes:
                self.error_handler.logger.error("Monthly analysis not available. Call analyze_monthly_windows first.")
                return
            
            # Prepare data for plotting
            windows = list(monthly_percent_changes.keys())
            currency_pairs = list(monthly_percent_changes[windows[0]].keys())
            
            # Create a DataFrame to hold all percentage changes
            df_changes = pd.DataFrame(index=windows, columns=currency_pairs)
            
            for window in windows:
                for pair in currency_pairs:
                    df_changes.loc[window, pair] = monthly_percent_changes[window][pair]
            
            # Plot the data
            plt.figure(figsize=(14, 10))
            
            # Create bar chart
            x = np.arange(len(windows))
            width = 0.1
            
            # Plot each currency pair
            for i, pair in enumerate(currency_pairs):
                offset = width * (i - len(currency_pairs)/2 + 0.5)
                color = self.colors[pair]
                plt.bar(x + offset, df_changes[pair], width, label=pair, color=color)
            
            # Add annotations
            plt.title('Cumulative Percentage Changes by Month Window', fontsize=16)
            plt.ylabel('Percentage Change (%)', fontsize=12)
            plt.xlabel('Cumulative Month Window', fontsize=12)
            plt.xticks(x, windows, rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
            plt.legend(loc='best')
            
            # Add zero line
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                self.error_handler.logger.info(f"Saved monthly summary plot to {save_path}")
            
    def plot_correlation_evolution(self,monthly_correlation_matrices, currency_pairs, save_path=None):
            """
            Plot the evolution of correlation between selected currency pairs over the months.
            
            Parameters:
            -----------
            currency_pairs : list of tuple
                List of currency pair tuples to plot, e.g. [('DXY', 'EURUSD'), ('GBPUSD', 'USDJPY')]
                If None, top 3 most correlated pairs from the full year are used
            save_path : str
                Path to save the plot
            """
            if not monthly_correlation_matrices:
                self.error_handler.logger.error("Monthly correlation data not available. Call analyze_monthly_windows first.")
             
            
            # Get correlation values for each window
            windows = list(monthly_correlation_matrices.keys())
            correlation_values = {}
            
            for pair in currency_pairs:
                pair_name = f"{pair[0]}-{pair[1]}"
                correlation_values[pair_name] = []
                
                for window in windows:
                    corr_matrix = monthly_correlation_matrices[window]
                    corr_value = corr_matrix.loc[pair[0], pair[1]]
                    correlation_values[pair_name].append(corr_value)
            
            # Plot the data
            plt.figure(figsize=(14, 8))
            
            # Plot each pair's correlation evolution
            for pair_name, values in correlation_values.items():
                plt.plot(windows, values, marker='o', label=pair_name)
            
            # Add annotations
            plt.title('Evolution of Currency Pair Correlations', fontsize=16)
            plt.ylabel('Correlation Coefficient', fontsize=12)
            plt.xlabel('Cumulative Month Window', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='best')
            
            # Add reference lines
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.3)
            plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                self.error_handler.logger.info(f"Saved correlation evolution plot to {save_path}")
            
  