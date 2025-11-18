import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
 
def analyze_historical_trends(df):
    """Analyze historical Volume trends to understand patterns"""
    print("=== HISTORICAL TREND ANALYSIS ===")
   
    # Overall trends
    monthly_trends = df.groupby('Date').agg({
        'Volume': ['mean', 'std', 'count']
    }).round(2)
    print("Monthly Volume Trends:")
    print(monthly_trends)
   
    # By Product
    product_trends = df.groupby(['Product', 'Date']).agg({
        'Volume': 'mean'
    }).groupby('Product').agg({
        'Volume': ['mean', 'std', 'min', 'max']
    }).round(2)
    print("\nProduct-wise Volume Analysis:")
    print(product_trends)
   
    # By Customer
    customer_trends = df.groupby('Customer').agg({
        'Volume': ['mean', 'std', 'count']
    }).round(2)
    print("\nCustomer-wise Volume Analysis:")
    print(customer_trends)
   
    # Last 3 months vs previous months
    last_date = df['Date'].max()
    last_3_months = df[df['Date'] >= (last_date - pd.DateOffset(months=2))]
    previous_months = df[df['Date'] < (last_date - pd.DateOffset(months=2))]
   
    print(f"\nLast 3 Months vs Previous Period:")
    print(f"Last 3 months Volume: {last_3_months['Volume'].mean():.2f}")
    print(f"Previous period Volume: {previous_months['Volume'].mean():.2f}")
    print(f"Change: {((last_3_months['Volume'].mean() - previous_months['Volume'].mean()) / previous_months['Volume'].mean() * 100):.1f}%")
   
    return monthly_trends, product_trends, customer_trends
 
def prepare_forecast_data():
    # Define file path - UPDATED with your specific path and file name
    desktop_path = r'C:\Users\Prakhar.Parashar\Desktop'
    file_name = 'volume.xlsx'  # UPDATED file name
    file_path = os.path.join(desktop_path, file_name)
   
    print(f"Looking for file at: {file_path}")
   
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please make sure the file 'volume.xlsx' is at the specified location")
       
        # List files in desktop to help debug
        try:
            desktop_files = os.listdir(desktop_path)
            excel_files = [f for f in desktop_files if f.endswith('.xlsx')]
            print(f"Excel files found on desktop: {excel_files}")
        except:
            print("Could not list desktop files")
       
        return None, None, None, None, None, None, None, None, None
   
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print("File loaded successfully!")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
       
        # Data preprocessing
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
       
        # Analyze historical trends first
        monthly_trends, product_trends, customer_trends = analyze_historical_trends(df)
       
        # Extract comprehensive time-based features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Days_in_month'] = df['Date'].dt.days_in_month
       
        # Create proper time index starting from 0
        min_date = df['Date'].min()
        df['Time_Index'] = (df['Date'] - min_date).dt.days
       
        # Add cyclical features for seasonality
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter']/4)
        df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter']/4)
       
        # Calculate trend component
        df['Trend'] = df.groupby(['Product', 'Profit Center', 'Customer', 'Company Code'])['Time_Index'].rank(method='dense')
       
        # Encode categorical variables
        categorical_cols = ['Product', 'Profit Center', 'Customer', 'Company Code']
        label_encoders = {}
       
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
       
        # Prepare features for training
        feature_cols = ['Time_Index', 'Product_encoded', 'Profit Center_encoded',
                       'Customer_encoded', 'Company Code_encoded', 'Year', 'Month',
                       'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos',
                       'Days_in_month', 'Trend']
       
        X = df[feature_cols]
        y = df['Volume']  # Changed from 'ASP' to 'Volume'
       
        return df, X, y, label_encoders, feature_cols, min_date, monthly_trends, product_trends, customer_trends
   
    except Exception as e:
        print(f"Error loading or processing file: {e}")
        return None, None, None, None, None, None, None, None, None
 
def train_random_forest_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
   
    rf_model.fit(X_train, y_train)
   
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   
    print(f"\nModel Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
   
    return rf_model
 
def generate_intelligent_forecast(rf_model, original_df, label_encoders, feature_cols, min_date, product_trends):
    # Get the last date in the data
    last_date = original_df['Date'].max()
    print(f"\nLast date in data: {last_date}")
   
    # Get unique combinations of categorical variables
    categorical_combinations = original_df[['Product', 'Profit Center', 'Customer', 'Company Code']].drop_duplicates()
    print(f"Number of unique combinations: {len(categorical_combinations)}")
   
    # Generate next 3 months
    forecast_dates = []
    for i in range(1, 4):
        if last_date.month + i <= 12:
            year = last_date.year
            month = last_date.month + i
        else:
            year = last_date.year + 1
            month = last_date.month + i - 12
        forecast_date = pd.Timestamp(year=year, month=month, day=1)
        forecast_dates.append(forecast_date)
        print(f"Forecast month {i}: {forecast_date.strftime('%Y%m')}")
   
    # Calculate historical averages for reference
    historical_avg = original_df['Volume'].mean()
    recent_avg = original_df[original_df['Date'] >= (last_date - pd.DateOffset(months=3))]['Volume'].mean()
   
    print(f"\nHistorical Reference Points:")
    print(f"Overall historical Volume average: {historical_avg:.2f}")
    print(f"Last 3 months Volume average: {recent_avg:.2f}")
   
    # Prepare forecast data
    forecast_records = []
   
    for i, date in enumerate(forecast_dates):
        for _, combo in categorical_combinations.iterrows():
            # Get historical pattern for this combination
            combo_history = original_df[
                (original_df['Product'] == combo['Product']) &
                (original_df['Profit Center'] == combo['Profit Center']) &
                (original_df['Customer'] == combo['Customer']) &
                (original_df['Company Code'] == combo['Company Code'])
            ]
           
            # Create proper time features
            days_from_start = (date - min_date).days
           
            # Seasonal features
            month_sin = np.sin(2 * np.pi * date.month/12)
            month_cos = np.cos(2 * np.pi * date.month/12)
            quarter = (date.month - 1) // 3 + 1
            quarter_sin = np.sin(2 * np.pi * quarter/4)
            quarter_cos = np.cos(2 * np.pi * quarter/4)
            days_in_month = date.days_in_month
           
            # Trend continues from last point
            if len(combo_history) > 0:
                last_trend = combo_history['Trend'].max() + i + 1
            else:
                last_trend = original_df['Trend'].max() + i + 1
           
            # Encode categorical variables
            product_encoded = label_encoders['Product'].transform([str(combo['Product'])])[0]
            profit_center_encoded = label_encoders['Profit Center'].transform([combo['Profit Center']])[0]
            customer_encoded = label_encoders['Customer'].transform([str(combo['Customer'])])[0]
            company_code_encoded = label_encoders['Company Code'].transform([str(combo['Company Code'])])[0]
           
            # Create feature array
            features = np.array([[days_from_start, product_encoded, profit_center_encoded,
                                customer_encoded, company_code_encoded, date.year, date.month,
                                month_sin, month_cos, quarter_sin, quarter_cos,
                                days_in_month, last_trend]])
           
            # Predict Volume
            volume_pred = rf_model.predict(features)[0]
           
            # Apply business logic based on product type and historical patterns
            # Get product-specific historical ranges
            product_history = original_df[original_df['Product'] == combo['Product']]
            if len(product_history) > 0:
                product_min = product_history['Volume'].min()
                product_max = product_history['Volume'].max()
                product_avg = product_history['Volume'].mean()
               
                # Apply reasonable bounds based on historical data
                volume_pred = max(product_min * 0.8, min(product_max * 1.2, volume_pred))
           
            # Round to appropriate units based on product type and typical Volume values
            # For Volume, we typically round to whole numbers
            volume_pred = round(volume_pred)
           
            # Add small random variation (1-2% of value)
            random_variation = np.random.normal(0, volume_pred * 0.01)
            volume_pred += random_variation
           
            # Ensure positive Volume
            volume_pred = max(0, volume_pred)
           
            # Final rounding
            volume_pred = round(volume_pred)
           
            # Create forecast record
            forecast_record = {
                'Version': 'Forecast',
                'Date': date.strftime('%Y%m'),
                'Company': 'Vertex Chemical Systems',
                'Company Code': combo['Company Code'],
                'Currency': 'USD',
                'Sales Org': 7004,  # Default sales org
                'Country': 'US',
                'Region': 'Northeast',  # Default region
                'City': 'New York',  # Default city
                'Product': combo['Product'],
                'UOM': 'L' if combo['Product'] == 'IS101' else ('g' if combo['Product'] == 'OR102' else 'kg'),
                'Product Group': 'Solvents' if combo['Product'] == 'IS101' else ('Reagents' if combo['Product'] == 'OR102' else 'Polymers'),
                'Customer': combo['Customer'],
                'Customer Country': 'USA',
                'Customer City': 'Florham Park',
                'Profit Center': combo['Profit Center'],
                'Account': 400000,
                'Audit Trail': 'ML_VOLUME_FORECAST',  # Updated audit trail
                'Volume': volume_pred  # Changed from ASP to Volume
            }
           
            forecast_records.append(forecast_record)
   
    forecast_df = pd.DataFrame(forecast_records)
   
    # Analyze forecast vs historical
    forecast_avg = forecast_df['Volume'].mean()
    print(f"\n=== FORECAST ANALYSIS ===")
    print(f"Forecast period Volume average: {forecast_avg:.2f}")
    print(f"Historical Volume average: {historical_avg:.2f}")
    print(f"Difference: {forecast_avg - historical_avg:.2f} ({((forecast_avg - historical_avg) / historical_avg * 100):.1f}%)")
   
    # Monthly breakdown
    print(f"\nMonthly Forecast Averages:")
    monthly_forecast = forecast_df.groupby('Date')['Volume'].mean()
    for date, avg in monthly_forecast.items():
        print(f"  {date}: {avg:.2f}")
   
    return forecast_df, historical_avg, forecast_avg
 
def create_comprehensive_analysis_graph(original_df, forecast_df, historical_avg, forecast_avg):
    # Prepare data
    original_df['Date_numeric'] = (original_df['Date'] - original_df['Date'].min()).dt.days
   
    # Aggregate data
    monthly_original = original_df.groupby('Date').agg({
        'Volume': 'mean'
    }).reset_index()
   
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], format='%Y%m')
    monthly_forecast = forecast_df.groupby('Date').agg({
        'Volume': 'mean'
    }).reset_index()
   
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
   
    # Plot 1: Timeline with historical and forecast
    ax1.plot(monthly_original['Date'], monthly_original['Volume'],
             marker='o', linewidth=2, markersize=6, label='Historical Monthly Average', color='blue')
    ax1.plot(monthly_forecast['Date'], monthly_forecast['Volume'],
             marker='s', linewidth=2, markersize=8, label='Forecast Monthly Average', color='red')
   
    # Add reference lines
    ax1.axhline(y=historical_avg, color='green', linestyle='--', alpha=0.7, label=f'Historical Avg: {historical_avg:.0f}')
    ax1.axhline(y=forecast_avg, color='orange', linestyle='--', alpha=0.7, label=f'Forecast Avg: {forecast_avg:.0f}')
   
    ax1.set_title('Volume Forecast vs Historical Trends\n(Are the forecast levels reasonable?)',
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Volume')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
   
    # Plot 2: Product-wise comparison
    product_historical = original_df.groupby('Product')['Volume'].mean()
    product_forecast = forecast_df.groupby('Product')['Volume'].mean()
   
    products = product_historical.index.tolist()
    x_pos = np.arange(len(products))
   
    ax2.bar(x_pos - 0.2, product_historical.values, 0.4, label='Historical Average', alpha=0.7)
    ax2.bar(x_pos + 0.2, product_forecast.values, 0.4, label='Forecast Average', alpha=0.7)
   
    ax2.set_title('Product-wise Volume: Historical vs Forecast', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Product')
    ax2.set_ylabel('Volume')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(products)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
   
    # Add value labels
    for i, v in enumerate(product_historical.values):
        ax2.text(i - 0.2, v + max(product_historical.values)*0.01, f'{v:.0f}', ha='center', va='bottom')
    for i, v in enumerate(product_forecast.values):
        ax2.text(i + 0.2, v + max(product_forecast.values)*0.01, f'{v:.0f}', ha='center', va='bottom')
   
    # Add analysis text
    overall_change = ((forecast_avg - historical_avg) / historical_avg * 100)
    analysis_text = f"Overall Forecast Change: {overall_change:+.1f}%"
   
    if abs(overall_change) < 5:
        analysis_text += "\n→ Forecast is relatively stable (good)"
    elif overall_change > 5:
        analysis_text += "\n→ Forecast shows volume increase (positive)"
    else:
        analysis_text += "\n→ Forecast shows volume decline (watch out)"
   
    ax1.annotate(analysis_text, xy=(0.02, 0.02), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                fontsize=10)
   
    plt.tight_layout()
    return fig
 
def main():
    print("Starting Volume Forecast Analysis...")
    result = prepare_forecast_data()
    if result[0] is None:
        print("Failed to load data. Exiting.")
        return None
   
    df, X, y, label_encoders, feature_cols, min_date, monthly_trends, product_trends, customer_trends = result
   
    print(f"\nTraining model on {X.shape[0]} samples...")
    rf_model = train_random_forest_model(X, y)
   
    print("\nGenerating intelligent forecast...")
    forecast_df, historical_avg, forecast_avg = generate_intelligent_forecast(
        rf_model, df, label_encoders, feature_cols, min_date, product_trends
    )
   
    # Save forecast data to desktop - UPDATED path
    desktop_path = r'C:\Users\Prakhar.Parashar\Desktop'
    forecast_filename = os.path.join(desktop_path, 'next_3_months_volume_forecast.xlsx')
    forecast_df.to_excel(forecast_filename, index=False)
   
    # Create comprehensive analysis graph
    print("\nCreating comprehensive analysis graph...")
    fig = create_comprehensive_analysis_graph(df, forecast_df, historical_avg, forecast_avg)
   
    # Save graph to desktop - UPDATED path
    graph_filename = os.path.join(desktop_path, 'volume_forecast_analysis.png')
    fig.savefig(graph_filename, dpi=300, bbox_inches='tight')
    plt.show()
   
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Forecast saved to: {forecast_filename}")
    print(f"Analysis graph saved to: {graph_filename}")
    print(f"Historical Volume average: {historical_avg:.0f}")
    print(f"Forecast Volume average: {forecast_avg:.0f}")
    print(f"Change: {((forecast_avg - historical_avg) / historical_avg * 100):+.1f}%")
   
    # Show product-wise forecast summary
    print(f"\nProduct-wise Forecast Summary:")
    product_summary = forecast_df.groupby('Product')['Volume'].agg(['mean', 'std', 'count']).round(2)
    print(product_summary)
   
    return forecast_df
 
if __name__ == "__main__":
    forecast_results = main()