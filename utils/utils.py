import os
import yaml
import pandas as pd
import logging
import numpy as np
import json


def save_config_yaml(df, config_path):
    """
    Updates specific sections of a YAML configuration file with optimized parameters
    while preserving all other sections and values.
    
    Parameters:
    -----------
    df : pd.Series or pd.DataFrame
        DataFrame or Series containing the optimization parameters
    config_path : str
        Path to the YAML file to update
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print(f"Saving.....................{config_path}")
    try:
        # If a DataFrame is passed, get the first row
        if isinstance(df, pd.DataFrame):
            if len(df) > 0:
                params = df.iloc[0]
            else:
                print("Error: Empty DataFrame provided")
                return False
        else:
            params = df  # Assume it's a Series
        
        # Load existing configuration if the file exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            config = {}
            print(f"No existing configuration found at {config_path}. Creating new file.")
        
        # Only update the specific sections we want to change
        # Initialize sections if they don't exist
        if 'signals' not in config:
            config['signals'] = {}
        if 'indicators' not in config:
            config['indicators'] = {}
        if 'backtest' not in config:
            config['backtest'] = {}
        
        # Update only the parameters in the specific sections
        # Signals section
        if 'CCI_up_threshold' in params:
            config['signals']['CCI_up_threshold'] = int(params['CCI_up_threshold'])
        if 'CCI_low_threshold' in params:
            config['signals']['CCI_low_threshold'] = int(params['CCI_low_threshold'])
        if 'Bollinger_Keltner_alignment' in params:
            config['signals']['Bollinger_Keltner_alignment'] = float(params['Bollinger_Keltner_alignment'])
        if 'window_size' in params:
            config['signals']['window_size'] = int(params['window_size'])
        
        # Indicators section
        if 'cci_period' in params:
            config['indicators']['cci_period'] = int(params['cci_period'])
        if 'bollinger_period' in params:
            config['indicators']['bollinger_period'] = int(params['bollinger_period'])
        if 'std_multiplier' in params:
            config['indicators']['std_multiplier'] = float(params['std_multiplier'])
        
        # Backtest section
        if 'max_positions' in params:
            config['backtest']['max_positions'] = int(params['max_positions'])
        if 'tp_level' in params:
            config['backtest']['tp_level'] = float(params['tp_level'])
        if 'sl_level' in params:
            config['backtest']['sl_level'] = float(params['sl_level'])
        
        # Write to YAML file with preserved formatting
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            
        print(f"Configuration updated and saved to {config_path}")
        return True
        
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False
    
def load_config(config_path):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        if not os.path.exists(config_path):
            logging.warning(f"Config file {config_path} not found. Using default configuration.")

                
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info(f"Configuration loaded from {config_path}")
            return config
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}. Using default configuration.")
    
def convert_ndarrays(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # handles np.float64, np.int64, etc.
        return obj.item()
    else:
        return obj