from data_quality_analysis import DataAnalysisFramework


def main():
    """Main execution function focusing on multi-type smart data fetching with DXY calculation"""
    
    # Initialize the framework
    framework = DataAnalysisFramework()
    
    # Define the data types to fetch
    data_types = [
        "currencies","key_sectors","financial","stocks","commodities","bonds_interest", "cryptos"
    ]
    
 
    # Execute multi-type smart data fetching
    multi_data = framework.get_multi_type_data_smart(
        data_types, 
        years=5,
        missing_threshold=0.8,
        fill_method='hybrid'
    )
    
    # Additional analysis and reporting
    if multi_data:
        print("\n" + "="*60)
        print("🔍 DETAILED ANALYSIS")
        print("="*60)
        
        for data_type, df in multi_data.items():
            print(f"\n📊 {data_type.upper()} DATA DETAILS:")
            print(f"   Shape: {df.shape}")
            print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
           
            # Check for missing data
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                symbols_with_missing = (missing_data > 0).sum()
                print(f"   Missing data: {symbols_with_missing} symbols have missing values")
            else:
                print("   Missing data: None ✅")
            
            # Show data types
            numeric_cols = df.select_dtypes(include=['number']).shape[1]
            print(f"   Data types: {numeric_cols} numeric columns")
            
            # Quick statistics
            if not df.empty:
                print(f"   Value range: {df.min().min():.2f} to {df.max().max():.2f}")

            # 💵 DXY CALCULATION (only for currencies)
            if data_type == "currencies":
                print("\n💵 DXY CALCULATION")
                print("="*30)
                
                dxy_series = framework.calculate_dxy(df)
                
                if dxy_series is not None:
                    print(f"   DXY data points: {len(dxy_series)}")
                    print(f"   DXY range: {dxy_series.min():.2f} to {dxy_series.max():.2f}")
                    print(f"   Current DXY value: {dxy_series.iloc[-1]:.2f}")
                    print("   ✅ DXY calculation successful!")
                    
                    # Optionally, add DXY to the currencies dataframe
                    df["DXY"] = dxy_series
                    print("   📈 DXY added to currencies dataset")
                else:
                    print("   ❌ DXY calculation failed - check configuration")
            df.to_csv(f"data/{data_type}.csv")
            print(f"\n Data saved in data/{data_type}.csv")
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Log file: data_analysis.log")
        
    else:
        print("\n❌ No data was loaded. Check the logs for detailed error information.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()