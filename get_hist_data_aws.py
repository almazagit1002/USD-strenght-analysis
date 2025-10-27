import boto3
import pandas as pd
import io
import os
import logging
from botocore.exceptions import ClientError
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime

import numpy as np

load_dotenv()

def fetch_combined_currency_data(bucket_name, date_range=None):
    """
    Fetch the combined currency data parquet file from S3.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    date_range : str, optional
        Specific date range to fetch (e.g., "2020-04-01_to_2025-03-31")
        If None, will attempt to fetch the most recent data
    aws_access_key : str, optional
        AWS access key for S3 bucket access
    aws_secret_key : str, optional
        AWS secret key for S3 bucket access
    region_name : str, optional
        AWS region name (default: 'us-east-1')
    
    Returns:
    --------
    pandas.DataFrame
        Combined currency data for all pairs
    """
    # Set up logger
    logger = logging.getLogger('fetch_combined_currency_data')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    
    
    s3_client = boto3.client('s3')
    
    # Path to look for the combined data
    base_path = "currency_data/combined/"
    
    try:
        if date_range:
            # If a specific date range is provided, use it
            object_key = f"{base_path}{date_range}.parquet"
            logger.info(f"Fetching specific combined data file: {object_key}")
        else:
            # Otherwise, list all objects in the combined directory and get the most recent one
            logger.info(f"Looking for most recent combined data file in {base_path}")
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=base_path
            )
            
            if 'Contents' not in response or not response['Contents']:
                logger.error(f"No combined currency data files found in {bucket_name}/{base_path}")
                return None
            
            # Find the most recent .parquet file
            parquet_files = [obj for obj in response['Contents'] 
                             if obj['Key'].endswith('.parquet')]
            
            if not parquet_files:
                logger.error(f"No parquet files found in {bucket_name}/{base_path}")
                return None
            
            # Sort by last modified date (most recent first)
            parquet_files.sort(key=lambda x: x['LastModified'], reverse=True)
            
            # Get the most recent file
            object_key = parquet_files[0]['Key']
            logger.info(f"Found most recent combined data file: {object_key}")
        
        # Get the object from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        
        # Read the parquet data
        parquet_data = response['Body'].read()
        buffer = io.BytesIO(parquet_data)
        df = pd.read_parquet(buffer)
        
        logger.info(f"Successfully loaded combined currency data with shape {df.shape}")
        
        # Basic data validation
        if df.empty:
            logger.warning("The loaded DataFrame is empty")
        else:
            logger.info(f"Date range in data: {df.index.min()} to {df.index.max()}")
            logger.info(f"Currency pairs in data: {', '.join(df.columns)}")
            
            # Check for missing values
            missing_counts = df.isna().sum()
            if missing_counts.sum() > 0:
                logger.warning(f"Missing values detected in data: {missing_counts.to_dict()}")
            
        return df
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'NoSuchKey':
            logger.error(f"The specified file does not exist: {object_key}")
        elif error_code == 'NoSuchBucket':
            logger.error(f"The specified bucket does not exist: {bucket_name}")
        else:
            logger.error(f"Error accessing S3: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None


def calculate_monthly_percentage_changes(currency_df):
    """
    Calculate the percentage change for each currency pair on a monthly basis.
    
    Parameters:
    -----------
    currency_df : pandas.DataFrame
        DataFrame containing currency data with dates as index
        and currency pairs as columns
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with monthly percentage changes for each currency pair
    """
    logger = logging.getLogger('calculate_monthly_percentage_changes')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Ensure the DataFrame has a datetime index
    if not isinstance(currency_df.index, pd.DatetimeIndex):
        try:
            currency_df.index = pd.to_datetime(currency_df.index)
            logger.info("Converted index to datetime")
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {str(e)}")
            return None
    
    logger.info(f"Processing data from {currency_df.index.min()} to {currency_df.index.max()}")
    
    # Method 1: Resample to month-end and calculate percentage change
    try:
        # Get month-end values
        monthly_data = currency_df.resample('M').last()
        
        # Calculate percentage change month over month
        monthly_pct_change = monthly_data.pct_change() * 100
        
        logger.info(f"Calculated monthly percentage changes using resample method")
        logger.info(f"Result shape: {monthly_pct_change.shape}")
        
        # Add month and year columns for easier analysis
        monthly_pct_change['Year'] = monthly_pct_change.index.year
        monthly_pct_change['Month'] = monthly_pct_change.index.month
        
        return monthly_pct_change
        
    except Exception as e:
        logger.error(f"Error calculating percentage changes using resample method: {str(e)}")
        
        # Method 2 (Alternative): Calculate manually by month
        try:
            logger.info("Trying alternative method for calculating monthly percentage changes")
            
            # Add month and year columns
            temp_df = currency_df.copy()
            temp_df['year'] = temp_df.index.year
            temp_df['month'] = temp_df.index.month
            
            # Get first and last day of each month
            monthly_grouped = temp_df.groupby(['year', 'month'])
            monthly_first = monthly_grouped.first()
            monthly_last = monthly_grouped.last()
            
            # Calculate percentage change between last day of each month
            currency_cols = [col for col in currency_df.columns]
            monthly_pct_change = pd.DataFrame(index=monthly_last.index)
            
            for col in currency_cols:
                # Current month's last day value / previous month's last day value - 1
                values = monthly_last[col].values
                pct_changes = np.zeros_like(values)
                
                # Start from second element (skip first month which has no previous month)
                for i in range(1, len(values)):
                    if values[i-1] != 0:  # Avoid division by zero
                        pct_changes[i] = ((values[i] / values[i-1]) - 1) * 100
                    else:
                        pct_changes[i] = np.nan
                
                monthly_pct_change[col] = pct_changes
            
            # Create a proper datetime index
            years = [idx[0] for idx in monthly_pct_change.index]
            months = [idx[1] for idx in monthly_pct_change.index]
            dates = pd.to_datetime([f"{year}-{month:02d}-01" for year, month in zip(years, months)])
            monthly_pct_change.index = dates
            
            # Add month and year columns for easier analysis
            monthly_pct_change['Year'] = monthly_pct_change.index.year
            monthly_pct_change['Month'] = monthly_pct_change.index.month
            
            logger.info(f"Calculated monthly percentage changes using alternative method")
            logger.info(f"Result shape: {monthly_pct_change.shape}")
            
            return monthly_pct_change
            
        except Exception as e2:
            logger.error(f"Error in alternative method for calculating percentage changes: {str(e2)}")
            return None
def calculate_yearly_changes(monthly_df):
    """
    Calculate yearly percentage changes and year-to-date changes from a DataFrame 
    of monthly percentage changes.
    
    Parameters:
    -----------
    monthly_df : pandas.DataFrame
        DataFrame with monthly percentage changes. Must include 'Year' and 'Month' columns,
        and the index should be a DatetimeIndex.
        
    Returns:
    --------
    tuple (yearly_changes_df, ytd_changes_df)
        yearly_changes_df: DataFrame with compounded yearly percentage changes
        ytd_changes_df: DataFrame with year-to-date percentage changes
    """
    # Ensure DataFrame has the expected structure
    if 'Year' not in monthly_df.columns or 'Month' not in monthly_df.columns:
        raise ValueError("Input DataFrame must contain 'Year' and 'Month' columns")
    
    # Get currency pair columns (exclude Year and Month)
    currency_cols = [col for col in monthly_df.columns if col not in ['Year', 'Month']]
    
    if not currency_cols:
        raise ValueError("No currency pair columns found in the DataFrame")
    
    # Make a copy of the input DataFrame to avoid modifying it
    df = monthly_df.copy()
    
    # Convert percentage values to multipliers (e.g., 5% -> 1.05)
    for col in currency_cols:
        df[col] = 1 + (df[col] / 100)
    
    # 1. Calculate yearly compound changes
    yearly_changes = {}
    
    # Group by year and calculate the compound product
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year]
        
        # Calculate compounded yearly change by multiplying all monthly multipliers
        year_changes = {}
        for col in currency_cols:
            # Compound all monthly changes
            compound_change = year_data[col].prod()
            # Convert back to percentage
            year_changes[col] = (compound_change - 1) * 100
        
        yearly_changes[year] = year_changes
    
    # Convert to DataFrame
    yearly_changes_df = pd.DataFrame.from_dict(yearly_changes, orient='index')
    yearly_changes_df.index.name = 'Year'
    yearly_changes_df = yearly_changes_df.sort_index()
    
    # 2. Calculate year-to-date changes
    ytd_changes = {}
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Get all available years in the data
    available_years = sorted(df['Year'].unique())
    
    for year in available_years:
        # For each year, calculate the YTD change
        # For past years, consider all months
        # For current year, consider only months up to current month
        if year < current_year:
            # Use all months for past years
            end_month = 12
        else:
            # Use months up to current month for current year
            end_month = current_month
        
        year_data = df[(df['Year'] == year) & (df['Month'] <= end_month)]
        
        # Calculate YTD change
        ytd_changes_dict = {}
        for col in currency_cols:
            # Compound the monthly changes
            ytd_compound_change = year_data[col].prod()
            # Convert back to percentage
            ytd_changes_dict[col] = (ytd_compound_change - 1) * 100
        
        ytd_changes[year] = ytd_changes_dict
    
    # Convert to DataFrame
    ytd_changes_df = pd.DataFrame.from_dict(ytd_changes, orient='index')
    ytd_changes_df.index.name = 'Year'
    ytd_changes_df = ytd_changes_df.sort_index()
    
    # Calculate previous year's complete performance for comparison
    # This allows comparing YTD performance with the same period in previous years
    previous_ytd_changes = {}
    
    for i, year in enumerate(available_years):
        if i > 0:  # Skip the first year as it has no previous year
            prev_year = available_years[i-1]
            
            # For current year, get the YTD months
            if year == current_year:
                months_to_consider = current_month
            else:
                months_to_consider = 12
            
            # Get data from previous year for the same months
            prev_year_partial = df[(df['Year'] == prev_year) & (df['Month'] <= months_to_consider)]
            
            prev_ytd_dict = {}
            for col in currency_cols:
                # Compound changes for comparable period in previous year
                prev_compound_change = prev_year_partial[col].prod()
                prev_ytd_dict[col] = (prev_compound_change - 1) * 100
            
            previous_ytd_changes[year] = prev_ytd_dict
    
    # Convert to DataFrame
    prev_ytd_df = pd.DataFrame.from_dict(previous_ytd_changes, orient='index')
    prev_ytd_df.index.name = 'Year'
    prev_ytd_df = prev_ytd_df.sort_index()
    
    # Rename columns to indicate they're previous year values
    prev_ytd_df.columns = [f"{col}_prev_ytd" for col in prev_ytd_df.columns]
    
    # Combine with current YTD for easy comparison
    combined_ytd_df = pd.concat([ytd_changes_df, prev_ytd_df], axis=1)
    
    return yearly_changes_df, combined_ytd_df


def calculate_monthly_correlations(currency_df):
    """
    Calculate the correlation matrix for each currency pair on a monthly basis.
    If the current month is incomplete, it will still include it up to the latest date.
    
    Parameters:
    -----------
    currency_df : pandas.DataFrame
        DataFrame containing currency data with dates as index
        and currency pairs as columns (DXY, EURUSD, GBPUSD, USDJPY, USDCAD, USDSEK, USDCHF)
    
    Returns:
    --------
    dict
        Dictionary where keys are month-year (YYYY-MM format) and 
        values are correlation matrices for that month
    pandas.DataFrame
        DataFrame with all correlation data combined with multi-index (Month, Pair1, Pair2)
    """
   
    
    logger = logging.getLogger('calculate_monthly_correlations')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Ensure the DataFrame has a datetime index
    if not isinstance(currency_df.index, pd.DatetimeIndex):
        try:
            currency_df.index = pd.to_datetime(currency_df.index)
            logger.info("Converted index to datetime")
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {str(e)}")
            return None, None
    
    logger.info(f"Processing data from {currency_df.index.min()} to {currency_df.index.max()}")
    
    # Add year-month column for grouping
    currency_df = currency_df.copy()
    currency_df['year_month'] = currency_df.index.strftime('%Y-%m')
    
    # Get all unique months
    unique_months = currency_df['year_month'].unique()
    logger.info(f"Found {len(unique_months)} unique months in the data")
    
    # Calculate correlations for each month
    monthly_correlations = {}
    
    for month in unique_months:
        month_data = currency_df[currency_df['year_month'] == month].drop(columns=['year_month'])
        corr_matrix = month_data.corr()
        monthly_correlations[month] = corr_matrix
        logger.info(f"Calculated correlation matrix for {month} with shape {corr_matrix.shape}")
    
    # Create a combined DataFrame with multi-index
    correlation_rows = []
    
    for month, corr_matrix in monthly_correlations.items():
        for pair1 in corr_matrix.columns:
            for pair2 in corr_matrix.columns:
                correlation_rows.append({
                    'Month': month,
                    'Pair1': pair1,
                    'Pair2': pair2,
                    'Correlation': corr_matrix.loc[pair1, pair2]
                })
    
    combined_df = pd.DataFrame(correlation_rows)
    combined_df = combined_df.set_index(['Month', 'Pair1', 'Pair2'])
    
    # Check if current month is in the data and is incomplete
    now = datetime.now()
    current_month = now.strftime('%Y-%m')
    
    is_current_month_incomplete = False
    if current_month in unique_months:
        month_data = currency_df[currency_df['year_month'] == current_month]
        latest_date = month_data.index.max()
        is_current_month_incomplete = latest_date.day < now.day
        logger.info(f"Current month ({current_month}) is in the data")
        logger.info(f"Latest date in current month: {latest_date}")
        logger.info(f"Current month is incomplete: {is_current_month_incomplete}")
    
    return monthly_correlations, combined_df

def analyze_monthly_changes(currency_df):
    """
    Process the currency DataFrame and provide analysis of monthly changes.
    
    Parameters:
    -----------
    currency_df : pandas.DataFrame
        DataFrame containing currency data with dates as index
        and currency pairs as columns
    
    Returns:
    --------
    tuple
        (monthly_pct_changes, summary_stats, extreme_months)
    """
    # Calculate monthly percentage changes
    monthly_pct_changes = calculate_monthly_percentage_changes(currency_df)

    if monthly_pct_changes is None:
        return None, None, None
    
    # Get currency pair columns (exclude Year and Month columns)
    currency_cols = [col for col in monthly_pct_changes.columns 
                     if col not in ['Year', 'Month']]
    
    # Calculate summary statistics
    summary_stats = {}
    for col in currency_cols:
        stats = {
            'mean': monthly_pct_changes[col].mean(),
            'median': monthly_pct_changes[col].median(),
            'std': monthly_pct_changes[col].std(),
            'min': monthly_pct_changes[col].min(),
            'max': monthly_pct_changes[col].max(),
            'positive_months': (monthly_pct_changes[col] > 0).sum(),
            'negative_months': (monthly_pct_changes[col] < 0).sum(),
            'volatile_months': (abs(monthly_pct_changes[col]) > 5).sum()  # Months with >5% change
        }
        summary_stats[col] = stats
    
    # Find extreme months for each currency pair
    extreme_months = {}
    for col in currency_cols:
        # Get months with largest positive and negative changes
        extreme_pos = monthly_pct_changes[[col, 'Year', 'Month']].nlargest(3, col)
        extreme_neg = monthly_pct_changes[[col, 'Year', 'Month']].nsmallest(3, col)
        
        extreme_months[col] = {
            'largest_gains': extreme_pos,
            'largest_losses': extreme_neg
        }
    
    return monthly_pct_changes, summary_stats, extreme_months

def plot_annotated_changes_subplots(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    groups = [
        ['DXY'],
        ['EURUSD', 'GBPUSD'],
        ['USDJPY', 'USDCAD'],
        ['USDSEK', 'USDCHF']
    ]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    # Adjust the spacing
    plt.subplots_adjust(
        # wspace=1.9,  # Width space between plots (horizontal spacing)
        hspace=1.9  # Height space between plots (vertical spacing)
    )

    years = df['Year'].unique()
    color_idx = 0  # <- Global color index

    for ax, group in zip(axs, groups):
        for col in group:
            col_color = colors[color_idx % len(colors)]
            color_idx += 1

            ax.plot(df.index, df[col], label=col, color=col_color)

            # Annotate yearly changes
            for j in range(1, len(years)):
                prev_year = years[j - 1]
                current_year = years[j]

                prev_val = df[df['Year'] == prev_year][col].iloc[-1]
                curr_val = df[df['Year'] == current_year][col].iloc[-1]
                change = curr_val - prev_val

                date_to_plot = df[df['Year'] == current_year].index[-1]
                ax.scatter(date_to_plot, curr_val, color=col_color, zorder=5)
                ax.annotate(f"{change:.2f}", (date_to_plot, curr_val),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=11)

            # Annotate last point
            last_date = df.index[-1]
            last_val = df[col].iloc[-1]
            ax.scatter(last_date, last_val, color=col_color, zorder=5)
            ax.annotate(f"{last_val:.2f}", (last_date, last_val),
                        textcoords="offset points", xytext=(0, -12),
                        ha='center', fontsize=8, color=col_color)

        ax.set_title(", ".join(group))
        ax.grid(True)
        ax.legend()

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig('test.png')



# Example usage
if __name__ == "__main__":
    
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    # Replace with your AWS credentials or configure AWS CLI
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    # Option 1: Fetch the most recent combined data
    df = fetch_combined_currency_data(bucket_name=BUCKET_NAME)
    
   
    if df is not None:
        # Display basic information about the data
        print(f"Loaded data with shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Currency pairs: {df.columns.tolist()}")
        # Calculate and analyze monthly changes
        monthly_changes, stats, extreme_months = analyze_monthly_changes(df)

        monthly_changes.dropna(inplace=True)
        current_year = pd.Timestamp.today().year
        monthly_changes_current_year = monthly_changes[monthly_changes['Year'] == current_year]
        monthly_changes_current_year.to_csv("monthly_changes_cy.csv")
        monthly_changes.to_csv('monthly_changes_all.csv')
        yearly_changes, ytd_changes = calculate_yearly_changes(monthly_changes)
        yearly_changes.dropna(inplace=True)
        ytd_changes.dropna(inplace=True)
        yearly_changes.to_csv("year_changes.csv")
        ytd_changes.to_csv("year_to_date.csv")


        # print("Yearly Percentage Changes:")
        # print(yearly_changes)
        # print("\nYear-to-Date Changes (with previous year comparison):")
        # print(ytd_changes)
        # if monthly_changes is not None:
        #     print("\nMonthly Percentage Changes (First 5 months):")
        #     print(monthly_changes.head())
        #     plot_annotated_changes_subplots(monthly_changes)
            # print("\nSummary Statistics:")
            # for pair, pair_stats in stats.items():
            #     print(f"\n{pair}:")
            #     for stat_name, value in pair_stats.items():
            #         print(f"  {stat_name}: {value:.2f}" if isinstance(value, float) else f"  {stat_name}: {value}")
            
            # print("\nExtreme Months:")
            # for pair, extremes in extreme_months.items():
            #     print(f"\n{pair} - Largest Gains:")
            #     print(extremes['largest_gains'])
            #     print(f"\n{pair} - Largest Losses:")
            #     print(extremes['largest_losses'])