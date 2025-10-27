import pandas as pd
import logging
from typing import Optional, Dict, List
import yaml
from pathlib import Path

from src.configurations import DataSourceConfig, SourceConfig
from data_manager import DataManager


class ConfigurationManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str = 'config/data_manager_config.yaml'):
        self.config_path = Path(config_path)
        self._base_config = None
        self._loaded_configs = {}
    
    def _load_base_config(self) -> Dict:
        """Load the base configuration file"""
        if self._base_config is None:
            with open(self.config_path, 'r') as file:
                self._base_config = yaml.safe_load(file)
        return self._base_config
    
    def load_config(self, data_type: str) -> DataSourceConfig:
        """Load configuration for a specific data type"""
        if data_type in self._loaded_configs:
            return self._loaded_configs[data_type]
        
        config = self._load_base_config()
        data = config[data_type].copy()
        
        # Handle multi-source configuration
        if "sources" in data:
            # Multi-source configuration
            sources = []
            for source_data in data["sources"]:
                source_config = SourceConfig(
                    type=source_data["type"],
                    base_url=source_data["base_url"],
                    tickers=source_data["tickers"]
                )
                sources.append(source_config)
            
            config_obj = DataSourceConfig(
                fetch_name=data["fetch_name"],
                sources=sources
            )
        else:
            # Single-source configuration (backward compatibility)
            base_url = data.get("base_url", config.get("base_url"))
            config_obj = DataSourceConfig(
                tickers=data["tickers"],
                base_url=base_url,
                fetch_name=data["fetch_name"],
                dxy_base_constant=data.get("dxy_base_constant"),
                dxy_weights=data.get("dxy_weights")
            )
        
        self._loaded_configs[data_type] = config_obj
        return config_obj


class DataAnalysisFramework:
    """Enhanced framework for data analysis operations with smart data handling"""
    
    AVAILABLE_DATA_TYPES = [
        "currencies", "key_sectors", "financial", 
        "stocks", "commodities", "bonds_interest", "cryptos"
    ]
    
    def __init__(self, config_path: str = 'config/data_manager_config.yaml'):
        self.config_manager = ConfigurationManager(config_path)
        self.logger = self._setup_logging()
        self.data_managers = {}
    
    def _setup_logging(self, level: int = logging.INFO) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def initialize_data_manager(self, data_type: str) -> DataManager:
        """Initialize an enhanced data manager for a specific data type"""
        if data_type not in self.data_managers:
            config = self.config_manager.load_config(data_type)
            self.data_managers[data_type] = DataManager(config)
            self.logger.info(f"Initialized data manager for {data_type}")
        return self.data_managers[data_type]
    
    def get_data_by_type_smart(self, data_type: str, years: int = 3,
                              missing_threshold: float = 0.8,
                              fill_method: str = 'hybrid') -> Optional[pd.DataFrame]:
        """
        Generic method to fetch data with enhanced missing data handling
        
        Args:
            data_type: Type of data to fetch
            years: Number of years of historical data
            missing_threshold: Maximum allowed missing data ratio per column
            fill_method: Method to handle missing data
            
        Returns:
            DataFrame with requested data using smart data handling
        """
        if data_type not in self.AVAILABLE_DATA_TYPES:
            self.logger.error(f"Unsupported data type: {data_type}")
            return None
        
        data_manager = self.initialize_data_manager(data_type)
        
        # Fetch data
        if not data_manager.fetch_all_data(years):
            self.logger.error(f"Failed to fetch {data_type} data")
            return None
        
        # Use smart combine method
        combined_df = data_manager.combine_data_smart(
            missing_threshold=missing_threshold,
            fill_method=fill_method
        )
        
        if combined_df is not None:
            self.logger.info(f"{data_type.title()} data preparation completed successfully")
        
        return combined_df
    
    def get_multi_type_data_smart(self, data_types: List[str], years: int = 3,
                                 missing_threshold: float = 0.8,
                                 fill_method: str = 'hybrid') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple data types with smart handling
        
        Args:
            data_types: List of data types to fetch
            years: Number of years of historical data
            missing_threshold: Maximum allowed missing data ratio per column
            fill_method: Method to handle missing data
            
        Returns:
            Dictionary mapping data types to their DataFrames
        """
        results = {}
        
        for data_type in data_types:
            data = self.get_data_by_type_smart(
                data_type, 
                years=years,
                missing_threshold=missing_threshold,
                fill_method=fill_method
            )
            if data is not None:
                results[data_type] = data
        
        self.logger.info(f"Successfully loaded data for {len(results)} data types")
        return results
    
    def calculate_dxy(self, currency_df: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
        """
        Calculate Dollar Index (DXY) using the standard formula
        
        Args:
            currency_df: DataFrame with currency data (if None, will fetch currencies data)
            
        Returns:
            Series with calculated DXY values
        """
        # If no currency data provided, fetch it
        if currency_df is None:
            self.logger.info("No currency data provided, fetching currencies data for DXY calculation")
            currency_df = self.get_data_by_type_smart("currencies")
            
        if currency_df is None:
            self.logger.error("No currency data available for DXY calculation")
            return None
        
        # Get the currency data manager for configuration
        if "currencies" not in self.data_managers:
            currency_manager = self.initialize_data_manager("currencies")
        else:
            currency_manager = self.data_managers["currencies"]
            
        config = currency_manager.config
        
        # Check if DXY configuration exists
        if not hasattr(config, 'dxy_weights') or config.dxy_weights is None:
            self.logger.info("No DXY weights provided in configuration. Skipping DXY calculation.")
            return None
            
        if not hasattr(config, 'dxy_base_constant') or config.dxy_base_constant is None:
            self.logger.info("No DXY base constant provided in configuration. Skipping DXY calculation.")
            return None
        
        try:
            # Check if all required currencies are available
            required_currencies = list(config.dxy_weights.keys())
            available_currencies = [curr for curr in required_currencies if curr in currency_df.columns]
            
            if len(available_currencies) != len(required_currencies):
                missing = set(required_currencies) - set(available_currencies)
                self.logger.warning(f"Missing currencies for DXY calculation: {missing}")
                if len(available_currencies) < len(required_currencies) * 0.8:  # Less than 80% available
                    self.logger.error("Too many currencies missing for reliable DXY calculation")
                    return None
            
            # Calculate DXY using the formula: DXY = base_constant * ∏(rate^weight)
            self.logger.info("Calculating DXY using standard formula")
            dxy_series = pd.Series(
                config.dxy_base_constant, 
                index=currency_df.index, 
                name="DXY_Calculated"
            )
            
            for currency, weight in config.dxy_weights.items():
                if currency in currency_df.columns:
                    dxy_series *= currency_df[currency] ** weight
                    self.logger.debug(f"Applied weight {weight} for {currency}")
            
            self.logger.info("DXY calculation completed successfully")
            return dxy_series
            
        except Exception as e:
            self.logger.error(f"Error calculating DXY: {str(e)}")
            return None