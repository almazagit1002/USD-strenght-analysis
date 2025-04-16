import logging
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple

# Configure a root logger with standard settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class ErrorHandler:
    """
    Centralized error handling class for trading strategy components.
    Provides standard methods for data validation and logging.
    """
    
    def __init__(self, logger_name: str, debug_mode: bool = False):
        """
        Initialize the error handler with component-specific logger.
        
        Parameters:
        -----------
        logger_name : str
            Name of the logger, typically the component name
        debug_mode : bool, optional
            Whether to enable debug level logging (default: False)
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False  # Prevent affecting other loggers
        
        # Set log level based on debug mode
        self.set_debug_mode(debug_mode)
        
        # Make sure handler isn't duplicated
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def set_debug_mode(self, debug_mode: bool):
        """Set the logging level based on debug mode."""
        if debug_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
    
    def check_dataframe_validity(self, df: pd.DataFrame, required_columns: List[str], 
                                 min_rows: int = 0, check_full: bool = False) -> Tuple[bool, str]:
        """
        Validate a DataFrame for trading operations.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to validate
        required_columns : list
            List of column names that must be present
        min_rows : int, optional
            Minimum number of rows required (default: 0)
        check_full : bool, optional
            Whether to check all rows for NaN values (default: False)
            If False, only checks the last row
            
        Returns:
        --------
        (valid, message) : tuple
            valid (bool): Whether the DataFrame is valid
            message (str): Error message if invalid, empty string if valid
        """
        # Check if DataFrame exists
        if df is None:
            self.logger.error("DataFrame is None")
            return False, "DataFrame is None"
        
        # Check for empty DataFrame
        if len(df) == 0:
            self.logger.error("Empty DataFrame")
            return False, "Empty DataFrame"
        
        # Check minimum rows
        if len(df) < min_rows:
            self.logger.error(f"Not enough data points ({len(df)}). Need at least {min_rows}.")
            return False, f"Not enough data points ({len(df)}). Need at least {min_rows}."
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for NaN values
        if check_full:
            # Check entire DataFrame for NaNs in required columns
            for col in required_columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    nan_percent = (nan_count / len(df)) * 100
                    self.logger.warning(f"Column {col} has {nan_count} NaN values ({nan_percent:.2f}%)")
                    # Still log but don't fail for NaNs in the middle of the DataFrame
        else:
            # Only check the last row for NaNs
            last_index = df.index[-1]
            for col in required_columns:
                if pd.isna(df.at[last_index, col]):
                    self.logger.error(f"NaN value in column {col} at last row")
                    return False, f"NaN value in column {col} at last row"
        
        return True, ""
    
    def safe_calculation(self, func, default_value=None, **kwargs):
        """
        Safely execute a calculation function with error handling.
        
        Parameters:
        -----------
        func : callable
            Function to execute
        default_value : any, optional
            Value to return if function raises an exception
        **kwargs : dict
            Arguments to pass to the function
            
        Returns:
        --------
        result : any
            Result of the function or default_value if an exception occurs
        """
        try:
            return func(**kwargs)
        except Exception as e:
            self.logger.error(f"Calculation error: {str(e)}")
            return default_value
    
    def log_dataframe_stats(self, df: pd.DataFrame, columns: List[str]):
        """
        Log statistics about key columns in a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to analyze
        columns : list
            List of column names to log statistics for
        """
        self.logger.info(f"DataFrame shape: {df.shape}")
        
        for col in columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                nan_percent = (nan_count / len(df)) * 100
                inf_count = np.isinf(df[col].replace([np.inf, -np.inf], np.nan)).sum()
                zero_count = (df[col] == 0).sum()
                
                self.logger.info(
                    f"Column {col}: "
                    f"NaN={nan_count}/{len(df)} ({nan_percent:.2f}%), "
                    f"Inf={inf_count}/{len(df)} ({(inf_count/len(df))*100:.2f}%), "
                    f"Zero={zero_count}/{len(df)} ({(zero_count/len(df))*100:.2f}%)"
                )
                
                if not df[col].empty and not df[col].isna().all():
                    self.logger.debug(
                        f"Column {col} stats: "
                        f"Min={df[col].min():.4f}, "
                        f"Max={df[col].max():.4f}, "
                        f"Mean={df[col].mean():.4f}, "
                        f"Median={df[col].median():.4f}"
                    )
    
    def handle_division(self, numerator, denominator, replace_value=0.0, min_denominator=1e-10):
        """
        Safely handle division operations to avoid divide by zero errors.
        
        Parameters:
        -----------
        numerator : numeric or array-like
            Numerator in the division
        denominator : numeric or array-like
            Denominator in the division
        replace_value : numeric, optional
            Value to use when division is invalid (default: 0.0)
        min_denominator : float, optional
            Minimum value for denominator to be considered valid (default: 1e-10)
            
        Returns:
        --------
        result : numeric or array-like
            Result of the division operation with safety handling
        """
        # Convert inputs to numpy arrays for consistent handling
        num_array = np.asarray(numerator)
        denom_array = np.asarray(denominator)
        
        # Create a mask for safe division (denominator not too close to zero)
        safe_mask = np.abs(denom_array) >= min_denominator
        
        # Initialize result array with the replace_value
        result = np.full_like(num_array, replace_value, dtype=float)
        
        # Only perform division where it's safe
        if np.any(safe_mask):
            result[safe_mask] = num_array[safe_mask] / denom_array[safe_mask]
        
        return result