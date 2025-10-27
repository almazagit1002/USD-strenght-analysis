import pandas as pd
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
import yaml

from src.configurations import CurrencyConfig



def load_currency_config(filepath: str) -> CurrencyConfig:
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return CurrencyConfig(**data)

# Load config from YAML
CURRENCY_CONFIG = load_currency_config('config/currency_config.yaml')
# ------------------------------------
# Setup Logging
# ------------------------------------
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('currency_analysis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ------------------------------------
# Data Fetching and DXY Calculation Class
# ------------------------------------
class CurrencyDataManager:
    """Handles data fetching and DXY calculations for currency analysis"""
    
    def __init__(self, config: CurrencyConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.raw_data: Dict[str, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        
    def fetch_ticker_data(self, ticker_symbol: str, years: int) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a given ticker symbol from Stooq
        
        Args:
            ticker_symbol: Symbol to fetch (e.g., 'eurusd')
            years: Number of years of historical data
            
        Returns:
            DataFrame with Date index and OHLCV data, or None if failed
        """
        try:
            end = datetime.today()
            start = end - timedelta(days=years * 365)
            url = (f"{self.config.base_url}?s={ticker_symbol}"
                  f"&d1={start:%Y%m%d}&d2={end:%Y%m%d}&i=d")
            
            self.logger.info(f"Fetching data for {ticker_symbol} from {start.date()} to {end.date()}")
            
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
    
    def fetch_all_currency_data(self, years: int) -> bool:
        """
        Fetch data for all configured currency pairs
        
        Args:
            years: Number of years of historical data
            
        Returns:
            True if at least some data was fetched successfully
        """
        self.logger.info(f"Starting to fetch data for {len(self.config.tickers)} currency pairs")
        
        success_count = 0
        for name, symbol in self.config.tickers.items():
            df = self.fetch_ticker_data(symbol, years)
            if df is not None:
                self.raw_data[name] = df
                success_count += 1
            else:
                self.logger.warning(f"Failed to fetch data for {name} ({symbol})")
        
        self.logger.info(f"Successfully fetched data for {success_count}/{len(self.config.tickers)} currencies")
        return success_count > 0
    
    def combine_currency_data(self) -> Optional[pd.DataFrame]:
        """
        Combine all fetched currency data into a single DataFrame
        
        Returns:
            Combined DataFrame with Close prices for all currencies
        """
        if not self.raw_data:
            self.logger.error("No raw data available to combine")
            return None
        
        try:
            # Extract close prices and combine
            close_prices = {}
            for name, df in self.raw_data.items():
                if 'Close' in df.columns:
                    close_prices[name] = df['Close']
                else:
                    self.logger.warning(f"No 'Close' column found for {name}")
            
            if not close_prices:
                self.logger.error("No close price data available")
                return None
            
            combined_df = pd.DataFrame(close_prices)
            
            # Remove rows with any missing values
            initial_rows = len(combined_df)
            combined_df = combined_df.dropna()
            final_rows = len(combined_df)
            
            if initial_rows != final_rows:
                self.logger.info(f"Removed {initial_rows - final_rows} rows with missing data")
            
            self.combined_data = combined_df
            self.logger.info(f"Combined data shape: {combined_df.shape}")
            self.logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Error combining currency data: {str(e)}")
            return None
    
    def calculate_dxy(self, currency_df: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
        """
        Calculate Dollar Index (DXY) using the standard formula
        
        Args:
            currency_df: DataFrame with currency data (uses self.combined_data if None)
            
        Returns:
            Series with calculated DXY values
        """
        if currency_df is None:
            currency_df = self.combined_data
            
        if currency_df is None:
            self.logger.error("No currency data available for DXY calculation")
            return None
        
        try:
            # Check if all required currencies are available
            required_currencies = list(self.config.dxy_weights.keys())
            available_currencies = [curr for curr in required_currencies if curr in currency_df.columns]
            
            if len(available_currencies) != len(required_currencies):
                missing = set(required_currencies) - set(available_currencies)
                self.logger.warning(f"Missing currencies for DXY calculation: {missing}")
                if len(available_currencies) < len(required_currencies) * 0.8:  # Less than 80% available
                    return None
            
            # Calculate DXY using the formula: DXY = base_constant * ∏(rate^weight)
            self.logger.info("Calculating DXY using standard formula")
            dxy_series = pd.Series(
                self.config.dxy_base_constant, 
                index=currency_df.index, 
                name="DXY_Calculated"
            )
            
            for currency, weight in self.config.dxy_weights.items():
                if currency in currency_df.columns:
                    dxy_series *= currency_df[currency] ** weight
                    self.logger.debug(f"Applied weight {weight} for {currency}")
            
            self.logger.info("DXY calculation completed successfully")
            return dxy_series
            
        except Exception as e:
            self.logger.error(f"Error calculating DXY: {str(e)}")
            return None
    
    def get_currency_data(self, years: int = 3, include_dxy: bool = True) -> Optional[pd.DataFrame]:
        """
        Main method to fetch and prepare all currency data
        
        Args:
            years: Number of years of historical data
            include_dxy: Whether to calculate and include DXY
            
        Returns:
            Complete DataFrame with all currency data
        """
        self.logger.info("Starting complete currency data preparation")
        
        # Fetch all data
        if not self.fetch_all_currency_data(years):
            self.logger.error("Failed to fetch any currency data")
            return None
        
        # Combine data
        combined_df = self.combine_currency_data()
        if combined_df is None:
            return None
        
        # Add DXY if requested
        if include_dxy:
            dxy_series = self.calculate_dxy(combined_df)
            if dxy_series is not None:
                combined_df['DXY_Calculated'] = dxy_series
                self.logger.info("Added calculated DXY to dataset")
        
        self.logger.info("Currency data preparation completed successfully")
        return combined_df

# ------------------------------------
# Analysis and Plotting Class
# ------------------------------------
class CurrencyAnalyzer:
    """Handles analysis, visualization, and statistics for currency data"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized analyzer with data shape: {data.shape}")
        
        # Pre-calculate normalized datasets
        self._calculate_normalizations()
    
    def _calculate_normalizations(self) -> None:
        """Pre-calculate different data normalizations"""
        try:
            self.normalized_100 = self.data / self.data.iloc[0] * 100
            self.zscore_normalized = (self.data - self.data.mean()) / self.data.std()
            self.pct_change = self.data.pct_change().dropna()
            self.minmax_normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min())
            
            self.logger.info("Calculated all data normalizations")
        except Exception as e:
            self.logger.error(f"Error calculating normalizations: {str(e)}")
            raise
    
    def calculate_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive performance metrics for all currencies
        
        Returns:
            Dictionary with performance metrics for each currency
        """
        try:
            metrics = {}
            
            for currency in self.data.columns:
                # Total return
                total_return = (self.data[currency].iloc[-1] / self.data[currency].iloc[0] - 1) * 100
                
                # Volatility (annualized)
                daily_returns = self.data[currency].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100
                
                # Sharpe ratio (assuming 0% risk-free rate)
                sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
                
                # Maximum drawdown
                cumulative = (1 + daily_returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative / rolling_max - 1) * 100
                max_drawdown = drawdown.min()
                
                metrics[currency] = {
                    'total_return_pct': total_return,
                    'volatility_pct': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown_pct': max_drawdown
                }
            
            self.logger.info("Calculated performance metrics for all currencies")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def plot_correlation_matrix(self, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create correlation matrix heatmap
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib Figure object
        """
        try:
            corr = self.data.corr()
            
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add correlation values as text
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    text_color = "white" if abs(corr.iloc[i, j]) > 0.5 else "black"
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}", 
                           ha="center", va="center", color=text_color, fontsize=10)
            
            # Formatting
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            ax.set_title("Currency Correlation Matrix", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            self.logger.info("Generated correlation matrix plot")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating correlation matrix: {str(e)}")
            raise
    
    def plot_comprehensive_analysis(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Create comprehensive 2x2 analysis plot
        
        Args:
            figsize: Figure size tuple
        
        Returns:
            Matplotlib Figure object
        """
        try:
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            axs = axs.flatten()
            
            # Plot 1: Normalized to 100
            for col in self.normalized_100.columns:
                axs[0].plot(self.normalized_100.index, self.normalized_100[col], 
                           label=col, linewidth=1.5)
            axs[0].set_title("Normalized Currency Prices (Start = 100)", fontweight='bold')
            axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[0].grid(True, alpha=0.3)
            axs[0].set_ylabel("Normalized Value")
            
            # Plot 2: Z-score normalization
            for col in self.zscore_normalized.columns:
                axs[1].plot(self.zscore_normalized.index, self.zscore_normalized[col], 
                           label=col, linewidth=1.5)
            axs[1].set_title("Z-score Normalization", fontweight='bold')
            axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[1].grid(True, alpha=0.3)
            axs[1].set_ylabel("Z-score")
            
            # Plot 3: Daily percentage change
            for col in self.pct_change.columns:
                axs[2].plot(self.pct_change.index, self.pct_change[col] * 100, 
                           label=col, alpha=0.7, linewidth=1)
            axs[2].set_title("Daily Percentage Change", fontweight='bold')
            axs[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[2].grid(True, alpha=0.3)
            axs[2].set_ylabel("Daily Change (%)")
            
            # Plot 4: Min-max normalization
            for col in self.minmax_normalized.columns:
                axs[3].plot(self.minmax_normalized.index, self.minmax_normalized[col], 
                           label=col, linewidth=1.5)
            axs[3].set_title("Min-Max Normalization (0-1)", fontweight='bold')
            axs[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axs[3].grid(True, alpha=0.3)
            axs[3].set_ylabel("Normalized (0-1)")
            
            plt.tight_layout()
            self.logger.info("Generated comprehensive analysis plot")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive analysis plot: {str(e)}")
            raise
    
    def plot_rolling_correlations(self, reference_currency: str = "USD Index", 
                                window: int = 90, figsize: Tuple[int, int] = (12, 6)) -> Optional[plt.Figure]:
        """
        Plot rolling correlations with reference currency
        
        Args:
            reference_currency: Currency to use as reference
            window: Rolling window size in days
            figsize: Figure size tuple
            
        Returns:
            Matplotlib Figure object or None if reference currency not found
        """
        if reference_currency not in self.data.columns:
            self.logger.warning(f"Reference currency {reference_currency} not found in data")
            return None
        
        try:
            rolling_corr = pd.DataFrame()
            
            for col in self.data.columns:
                if col != reference_currency:
                    rolling_corr[col] = self.data[col].rolling(window).corr(self.data[reference_currency])
            
            fig, ax = plt.subplots(figsize=figsize)
            for col in rolling_corr.columns:
                ax.plot(rolling_corr.index, rolling_corr[col], label=col, linewidth=1.5)
            
            ax.set_title(f"{window}-day Rolling Correlation with {reference_currency}", 
                        fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Correlation Coefficient")
            ax.set_ylim(-1.1, 1.1)
            
            plt.tight_layout()
            self.logger.info(f"Generated rolling correlation plot with {reference_currency}")
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating rolling correlation plot: {str(e)}")
            raise
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive text report
        
        Returns:
            Formatted string report
        """
        try:
            metrics = self.calculate_performance_metrics()
            
            report_lines = [
                "=" * 60,
                "CURRENCY ANALYSIS REPORT",
                "=" * 60,
                f"Analysis Period: {self.data.index.min().date()} to {self.data.index.max().date()}",
                f"Number of Trading Days: {len(self.data)}",
                f"Currencies Analyzed: {len(self.data.columns)}",
                "",
                "PERFORMANCE SUMMARY",
                "-" * 30
            ]
            
            for currency, metric in metrics.items():
                report_lines.extend([
                    f"{currency}:",
                    f"  Total Return: {metric['total_return_pct']:+7.2f}%",
                    f"  Volatility:   {metric['volatility_pct']:7.2f}%",
                    f"  Sharpe Ratio: {metric['sharpe_ratio']:7.2f}",
                    f"  Max Drawdown: {metric['max_drawdown_pct']:7.2f}%",
                    ""
                ])
            
            # Correlation summary
            corr_matrix = self.data.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            report_lines.extend([
                "CORRELATION ANALYSIS",
                "-" * 30,
                f"Average Correlation: {avg_correlation:.3f}",
                "",
                "Highest Correlations:"
            ])
            
            # Find highest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
            
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for pair in corr_pairs[:5]:
                report_lines.append(f"  {pair[0]} vs {pair[1]}: {pair[2]:+.3f}")
            
            report_text = "\n".join(report_lines)
            self.logger.info("Generated comprehensive analysis report")
            return report_text
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"

# ------------------------------------
# Main Analysis Runner
# ------------------------------------
def run_currency_analysis(years: int = 3, save_plots: bool = True, 
                         show_plots: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Run complete currency analysis
    
    Args:
        years: Number of years of historical data
        save_plots: Whether to save plots to files
        show_plots: Whether to display plots
        
    Returns:
        Tuple of (currency_data, analysis_report)
    """
    logger.info("Starting complete currency analysis")
    logger.info("=" * 60)
    
    try:
        # Initialize data manager and fetch data
        data_manager = CurrencyDataManager(CURRENCY_CONFIG)
        currency_data = data_manager.get_currency_data(years=years, include_dxy=True)
        
        if currency_data is None:
            logger.error("Failed to fetch currency data")
            return None, None
        
        # Initialize analyzer
        analyzer = CurrencyAnalyzer(currency_data)
        
        # Generate plots
        logger.info("Generating analysis visualizations...")
        
        # 1. Comprehensive analysis plot
        analysis_fig = analyzer.plot_comprehensive_analysis()
        if save_plots:
            analysis_fig.savefig("currency_analysis.png", dpi=300, bbox_inches='tight')
            logger.info("Saved: currency_analysis.png")
        
        # 2. Correlation matrix
        corr_fig = analyzer.plot_correlation_matrix()
        if save_plots:
            corr_fig.savefig("currency_correlation_matrix.png", dpi=300, bbox_inches='tight')
            logger.info("Saved: currency_correlation_matrix.png")
        
        # 3. Rolling correlations
        rolling_fig = analyzer.plot_rolling_correlations()
        if rolling_fig and save_plots:
            rolling_fig.savefig("currency_rolling_correlations.png", dpi=300, bbox_inches='tight')
            logger.info("Saved: currency_rolling_correlations.png")
        
        # Generate report
        report = analyzer.generate_report()
        if save_plots:
            with open("currency_analysis_report.txt", "w") as f:
                f.write(report)
            logger.info("Saved: currency_analysis_report.txt")
        
        # Display plots if requested
        if show_plots:
            plt.show()
        
        logger.info("Currency analysis completed successfully!")
        print("\n" + report)
        
        return currency_data, report
        
    except Exception as e:
        logger.error(f"Error in currency analysis: {str(e)}")
        return None, None

# ------------------------------------
# Main Execution
# ------------------------------------
if __name__ == "__main__":
    # Run the complete analysis
    data, report = run_currency_analysis(years=3, save_plots=True, show_plots=True)
    
    if data is not None:
        print(f"\nAnalysis completed! Generated files:")
        print("- currency_analysis.png")
        print("- currency_correlation_matrix.png") 
        print("- currency_rolling_correlations.png")
        print("- currency_analysis_report.txt")
        print("- currency_analysis.log")
    else:
        print("Analysis failed. Check the log file for details.")