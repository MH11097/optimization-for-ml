#!/usr/bin/env python3
"""
02. Data Preprocessing with Advanced Feature Engineering
Optimized version based on EDA results
Input: data/00_raw/used_cars_data.csv
Output: data/02_processed/ (cleaned and engineered features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import re
import json
import joblib
import warnings
from src.utils.data_loader import load_data_chunked
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

# Columns to load (based on EDA results)
SELECTED_COLUMNS = [
    # Target
    'price',
    
    # Core features
    'year', 'make_name', 'model_name', 'body_type', 
    'fuel_type', 'engine_type', 'transmission', 'wheel_system',
    
    # Numeric features
    'mileage', 'horsepower', 'engine_displacement', 
    'city_fuel_economy', 'highway_fuel_economy',
    'maximum_seating', 'daysonmarket',
    
    # Dimensions (for creating composite features)
    'length', 'width', 'height', 'wheelbase',
    'back_legroom', 'front_legroom', 'fuel_tank_volume',
    
    # Condition indicators
    'is_new', 'has_accidents', 'frame_damaged', 
    'fleet', 'owner_count',
    
    # Additional features
    'seller_rating', 'exterior_color', 'interior_color',
    'torque', 'power', 'listing_color'
]

# Columns to drop (high missing or low importance)
COLUMNS_TO_DROP = [
    'bed', 'bed_height', 'bed_length', 'cabin',
    'combine_fuel_economy', 'is_certified', 'is_cpo', 
    'is_oemcpo', 'vehicle_damage_category',
    'vin', 'listing_id', 'trimId', 'sp_id',
    'description', 'main_picture_url', 'dealer_zip',
    'latitude', 'longitude', 'listed_date', 'sp_name',
    'franchise_make', 'transmission_display', 
    'wheel_system_display', 'salvage', 'theft_title',
    'isCab', 'trim_name', 'major_options'
]

# ==================== HELPER FUNCTIONS ====================

def setup_environment():
    """Setup output directory"""
    output_dir = Path("data/02_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_data_for_preprocessing(file_path=None, columns=None, sample_size=None, chunk_size=10000):
    """Load data with chunked processing for memory efficiency"""
    print("📂 Loading data for preprocessing with chunking...")
    
    # Default file path
    if file_path is None:
        file_path = Path("data/00_raw/used_cars_data.csv")
    
    # Use selected columns if not specified
    if columns is None:
        columns = SELECTED_COLUMNS
    
    # Use the shared chunking function from utils
    df = load_data_chunked(
        file_path=file_path,
        chunk_size=chunk_size,
        max_rows=sample_size,
        columns=columns
    )
    
    return df

# optimize_dtypes function removed - now using optimize_dataframe_dtypes from utils

# ==================== CLEANING FUNCTIONS ====================

def initial_cleaning(df):
    """Perform initial data cleaning"""
    print("\n" + "="*60)
    print("🧹 INITIAL CLEANING")
    print("="*60)
    
    initial_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_shape[0] - df.shape[0]
    
    # Remove invalid prices
    if 'price' in df.columns:
        initial_count = len(df)
        df = df[df['price'] > 0]
        df = df[df['price'] < df['price'].quantile(0.995)]  # Remove extreme outliers
        price_filtered = initial_count - len(df)
        print(f"  Removed {price_filtered:,} rows with invalid prices")
    
    # Remove invalid years
    if 'year' in df.columns:
        current_year = pd.Timestamp.now().year
        df = df[(df['year'] >= 1900) & (df['year'] <= current_year + 1)]
    
    # Remove invalid mileage
    if 'mileage' in df.columns:
        df = df[(df['mileage'] >= 0) & (df['mileage'] < 500000)]
    
    print(f"  Removed {duplicates_removed:,} duplicate rows")
    print(f"  Final shape: {df.shape}")
    
    return df

def clean_text_columns(df):
    """Clean text/string columns"""
    print("\n" + "="*60)
    print("📝 CLEANING TEXT COLUMNS")
    print("="*60)
    
    text_columns = df.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        # Convert to lowercase
        df[col] = df[col].str.lower()
        
        # Remove extra whitespace
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Replace empty strings with NaN
        df[col] = df[col].replace('', np.nan)
        
    print(f"  Cleaned {len(text_columns)} text columns")
    
    return df

def extract_numeric_from_string(df):
    """Extract numeric values from string columns"""
    print("\n" + "="*60)
    print("🔢 EXTRACTING NUMERIC VALUES")
    print("="*60)
    
    # Columns that might contain numeric values with units
    columns_with_units = {
        'back_legroom': 'inches',
        'front_legroom': 'inches',
        'height': 'inches',
        'length': 'inches',
        'width': 'inches',
        'wheelbase': 'inches',
        'fuel_tank_volume': 'gallons',
        'maximum_seating': 'seats'
    }
    
    for col, unit in columns_with_units.items():
        if col in df.columns and df[col].dtype == 'object':
            # Extract first number found
            df[col] = df[col].str.extract(r'(\d+\.?\d*)', expand=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"  Extracted numeric from {col} ({unit})")
    
    # Handle power and torque (complex format)
    if 'power' in df.columns and df['power'].dtype == 'object':
        # Extract HP and RPM from format like "200 hp @ 5000 rpm"
        df['power_hp'] = df['power'].str.extract(r'(\d+)\s*hp', expand=False)
        df['power_rpm'] = df['power'].str.extract(r'@\s*(\d+)', expand=False)
        df['power_hp'] = pd.to_numeric(df['power_hp'], errors='coerce')
        df['power_rpm'] = pd.to_numeric(df['power_rpm'], errors='coerce')
        print("  Extracted power_hp and power_rpm from power column")
    
    if 'torque' in df.columns and df['torque'].dtype == 'object':
        # Extract torque value and RPM
        df['torque_value'] = df['torque'].str.extract(r'(\d+)', expand=False)
        df['torque_rpm'] = df['torque'].str.extract(r'@\s*(\d+)', expand=False)
        df['torque_value'] = pd.to_numeric(df['torque_value'], errors='coerce')
        df['torque_rpm'] = pd.to_numeric(df['torque_rpm'], errors='coerce')
        print("  Extracted torque_value and torque_rpm from torque column")
    
    return df

# ==================== MISSING VALUE HANDLING ====================

def handle_missing_values(df, strategy='smart'):
    """Handle missing values with multiple strategies"""
    print("\n" + "="*60)
    print("🔧 HANDLING MISSING VALUES")
    print("="*60)
    
    initial_missing = df.isnull().sum().sum()
    print(f"  Initial missing values: {initial_missing:,}")
    
    # Drop columns with >50% missing
    threshold = 0.5
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing")
    
    # Smart imputation based on column type and distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Numeric imputation
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct < 0.05:  # Low missing - use median
                df[col].fillna(df[col].median(), inplace=True)
            elif missing_pct < 0.2:  # Moderate missing - use group-based imputation
                # Try to impute based on similar vehicles
                if 'make_name' in df.columns and 'model_name' in df.columns:
                    df[col] = df.groupby(['make_name', 'model_name'])[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                # Fill remaining with overall median
                df[col].fillna(df[col].median(), inplace=True)
            else:  # High missing - create indicator and impute
                df[f'{col}_was_missing'] = df[col].isnull().astype(int)
                df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical imputation
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct < 0.1:  # Low missing - use mode
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_value, inplace=True)
            else:  # Higher missing - use 'unknown' category
                df[col].fillna('unknown', inplace=True)
    
    final_missing = df.isnull().sum().sum()
    print(f"  Final missing values: {final_missing:,}")
    print(f"  Reduction: {initial_missing - final_missing:,} values handled")
    
    return df

# ==================== FEATURE ENGINEERING ====================

def create_basic_features(df):
    """Create basic engineered features"""
    print("\n" + "="*60)
    print("⚡ BASIC FEATURE ENGINEERING")
    print("="*60)
    
    features_created = []
    
    # Helper function to safely convert to numeric
    def safe_numeric(series, name):
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            print(f"  Warning: Could not convert {name} to numeric: {e}")
            return series
    
    # Age-related features
    if 'year' in df.columns:
        df['year'] = safe_numeric(df['year'], 'year')
        current_year = pd.Timestamp.now().year
        df['age'] = current_year - df['year']
        df['age_squared'] = df['age'] ** 2
        df['is_classic'] = (df['age'] > 25).astype(int)
        features_created.extend(['age', 'age_squared', 'is_classic'])
    
    # Mileage-related features
    if 'mileage' in df.columns and 'age' in df.columns:
        df['mileage'] = safe_numeric(df['mileage'], 'mileage')
        df['mileage_per_year'] = df['mileage'] / (df['age'] + 1)
        df['high_mileage'] = (df['mileage_per_year'] > 15000).astype(int)
        features_created.extend(['mileage_per_year', 'high_mileage'])
    
    # Fuel economy features
    if 'city_fuel_economy' in df.columns and 'highway_fuel_economy' in df.columns:
        df['city_fuel_economy'] = safe_numeric(df['city_fuel_economy'], 'city_fuel_economy')
        df['highway_fuel_economy'] = safe_numeric(df['highway_fuel_economy'], 'highway_fuel_economy')
        df['combined_fuel_economy'] = (df['city_fuel_economy'] + df['highway_fuel_economy']) / 2
        df['fuel_economy_diff'] = df['highway_fuel_economy'] - df['city_fuel_economy']
        features_created.extend(['combined_fuel_economy', 'fuel_economy_diff'])
    
    # Size features - with proper numeric conversion
    size_cols = ['length', 'width', 'height']
    if all(col in df.columns for col in size_cols):
        # Convert size columns to numeric
        for col in size_cols:
            df[col] = safe_numeric(df[col], col)
        
        # Check if all size columns are now numeric
        all_numeric = all(pd.api.types.is_numeric_dtype(df[col]) for col in size_cols)
        
        if all_numeric:
            df['vehicle_volume'] = df['length'] * df['width'] * df['height']
            df['vehicle_footprint'] = df['length'] * df['width']
            features_created.extend(['vehicle_volume', 'vehicle_footprint'])
        else:
            print("  Warning: Size columns contain non-numeric values, skipping volume calculations")
    
    # Legroom features
    if 'back_legroom' in df.columns and 'front_legroom' in df.columns:
        df['back_legroom'] = safe_numeric(df['back_legroom'], 'back_legroom')
        df['front_legroom'] = safe_numeric(df['front_legroom'], 'front_legroom')
        df['total_legroom'] = df['back_legroom'] + df['front_legroom']
        df['legroom_ratio'] = df['back_legroom'] / (df['front_legroom'] + 1)
        features_created.extend(['total_legroom', 'legroom_ratio'])
    
    # Power-to-weight proxy (using size as weight proxy)
    if 'horsepower' in df.columns and 'vehicle_volume' in features_created:
        df['horsepower'] = safe_numeric(df['horsepower'], 'horsepower')
        df['power_to_size_ratio'] = df['horsepower'] / (df['vehicle_volume'] + 1)
        features_created.append('power_to_size_ratio')
    
    # Market time features
    if 'daysonmarket' in df.columns:
        df['daysonmarket'] = safe_numeric(df['daysonmarket'], 'daysonmarket')
        df['weeks_on_market'] = df['daysonmarket'] / 7
        df['quick_sale'] = (df['daysonmarket'] < 30).astype(int)
        features_created.extend(['weeks_on_market', 'quick_sale'])
    
    print(f"  Created {len(features_created)} new features:")
    for feat in features_created:
        if feat in df.columns:
            print(f"    - {feat}")
    
    return df

def create_interaction_features(df):
    """Create interaction features between important variables"""
    print("\n" + "="*60)
    print("🔄 INTERACTION FEATURES")
    print("="*60)
    
    interactions_created = []
    
    # Brand-Model combinations
    if 'make_name' in df.columns and 'model_name' in df.columns:
        # Convert to string to avoid categorical + string issues
        make_str = df['make_name'].astype(str)
        model_str = df['model_name'].astype(str)
        df['make_model'] = make_str + '_' + model_str
        interactions_created.append('make_model')
    
    # Luxury indicators
    luxury_brands = ['mercedes-benz', 'bmw', 'audi', 'lexus', 'porsche', 
                    'jaguar', 'land rover', 'tesla', 'cadillac', 'lincoln']
    if 'make_name' in df.columns:
        # Convert to string and lowercase for comparison
        make_lower = df['make_name'].astype(str).str.lower()
        df['is_luxury'] = make_lower.isin(luxury_brands).astype(int)
        interactions_created.append('is_luxury')
    
    # Performance categories
    if 'horsepower' in df.columns:
        # Ensure horsepower is numeric
        hp_numeric = pd.to_numeric(df['horsepower'], errors='coerce')
        df['performance_category'] = pd.cut(
            hp_numeric,
            bins=[0, 150, 250, 350, float('inf')],
            labels=['economy', 'standard', 'performance', 'high_performance']
        )
        interactions_created.append('performance_category')
    
    # Fuel type groups
    if 'fuel_type' in df.columns:
        # Convert to string for safe string operations
        fuel_str = df['fuel_type'].astype(str).str.lower()
        df['is_electric'] = (fuel_str == 'electric').astype(int)
        df['is_hybrid'] = fuel_str.str.contains('hybrid', na=False).astype(int)
        interactions_created.extend(['is_electric', 'is_hybrid'])
    
    # Condition score
    condition_cols = ['has_accidents', 'frame_damaged', 'fleet']
    available_condition = [col for col in condition_cols if col in df.columns]
    if available_condition:
        # Convert to numeric, handling boolean/categorical types
        condition_sum = 0
        for col in available_condition:
            try:
                # Convert to numeric (1 for True/positive, 0 for False/negative)
                col_numeric = pd.to_numeric(df[col], errors='coerce').fillna(0)
                condition_sum += col_numeric
            except Exception as e:
                print(f"  Warning: Could not process condition column {col}: {e}")
        
        df['condition_score'] = 3 - condition_sum
        interactions_created.append('condition_score')
    
    # Age-mileage interaction
    if 'age' in df.columns and 'mileage' in df.columns:
        age_numeric = pd.to_numeric(df['age'], errors='coerce')
        mileage_numeric = pd.to_numeric(df['mileage'], errors='coerce')
        df['age_mileage_ratio'] = mileage_numeric / (age_numeric + 1)
        interactions_created.append('age_mileage_ratio')
    
    # Price tier (if price exists and we're not in target encoding stage)
    if 'price' in df.columns:
        price_numeric = pd.to_numeric(df['price'], errors='coerce')
        df['price_tier'] = pd.cut(
            price_numeric,
            bins=[0, 15000, 30000, 50000, float('inf')],
            labels=['budget', 'mid_range', 'premium', 'luxury']
        )
        interactions_created.append('price_tier')
    
    print(f"  Created {len(interactions_created)} interaction features:")
    for feat in interactions_created:
        if feat in df.columns:
            print(f"    - {feat}")
    
    return df

def create_target_encoding_features(df, target='price', categorical_cols=None):
    """Create target encoding for high-cardinality categorical features"""
    print("\n" + "="*60)
    print("🎯 TARGET ENCODING")
    print("="*60)
    
    if target not in df.columns:
        print("  ⚠️ Target variable not found, skipping target encoding")
        return df
    
    if categorical_cols is None:
        categorical_cols = ['make_name', 'model_name', 'make_model', 'body_type']
    
    # Filter to available columns
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    encoded_features = []
    
    for col in categorical_cols:
        if df[col].nunique() > 10:  # Only for high-cardinality
            # Calculate mean target by category
            encoding_map = df.groupby(col)[target].agg(['mean', 'count'])
            
            # Smooth with global mean for rare categories
            global_mean = df[target].mean()
            min_samples = 30
            encoding_map['smoothed_mean'] = (
                (encoding_map['mean'] * encoding_map['count'] + global_mean * min_samples) /
                (encoding_map['count'] + min_samples)
            )
            
            # Apply encoding
            df[f'{col}_target_encoded'] = df[col].map(encoding_map['smoothed_mean'])
            df[f'{col}_target_encoded'].fillna(global_mean, inplace=True)
            
            encoded_features.append(f'{col}_target_encoded')
    
    if encoded_features:
        print(f"  Created {len(encoded_features)} target-encoded features")
    
    return df

# ==================== ENCODING & SCALING ====================

def encode_categorical_features(df, max_categories=50):
    """Encode categorical features"""
    print("\n" + "="*60)
    print("🔤 CATEGORICAL ENCODING")
    print("="*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target if present
    if 'price' in categorical_cols:
        categorical_cols.remove('price')
    
    encoded_cols = []
    dropped_cols = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        
        if unique_count <= 2:
            # Binary encoding
            df[f'{col}_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))
            encoded_cols.append(col)
            
        elif unique_count <= 10:
            # One-hot encoding for low cardinality
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            encoded_cols.append(col)
            
        elif unique_count <= max_categories:
            # Label encoding for moderate cardinality
            df[f'{col}_label_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))
            encoded_cols.append(col)
            
        else:
            # Drop or handle high cardinality differently
            if f'{col}_target_encoded' not in df.columns:
                dropped_cols.append(col)
    
    # Drop original categorical columns that were encoded
    df = df.drop(columns=encoded_cols)
    
    # Drop high cardinality columns without target encoding
    if dropped_cols:
        df = df.drop(columns=dropped_cols)
        print(f"  ⚠️ Dropped {len(dropped_cols)} high-cardinality columns")
    
    print(f"  Encoded {len(encoded_cols)} categorical features")
    
    return df

def scale_features(df, method='robust', exclude_cols=['price']):
    """Scale numeric features"""
    print("\n" + "="*60)
    print("📏 FEATURE SCALING")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude specified columns
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    # Apply scaling
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    print(f"  Scaled {len(scale_cols)} numeric features using {method} scaling")
    
    return df, scaler

# ==================== FINAL PREPARATION ====================


def prepare_final_dataset(df, target='price', test_size=0.2, random_state=42):
    """Prepare final dataset for modeling"""
    print("\n" + "="*60)
    print("🎯 PREPARING FINAL DATASET")
    print("="*60)
    
    print(f"  Input dataframe shape: {df.shape}")
    
    # Ensure target exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    
    # Log transform target if skewed
    target_col = target
    if df[target].skew() > 1:
        df['log_price'] = np.log1p(df[target])
        print("  Applied log transformation to target (high skewness)")
        target_col = 'log_price'
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [target, 'log_price']]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"  Before filtering - Features: {X.shape[1]} columns")
    print(f"  Target column: {target_col}")
    print(f"  Target values count: {len(y.dropna())}")
    
    # Improved dtype handling - don't filter too aggressively
    columns_to_keep = []
    
    for col in X.columns:
        try:
            # Check if column has any non-null values
            if X[col].notna().sum() == 0:
                print(f"    Dropping empty column: {col}")
                continue
                
            # Try to convert to numeric for non-numeric dtypes
            if X[col].dtype == 'object':
                # Try numeric conversion
                numeric_converted = pd.to_numeric(X[col], errors='coerce')
                if numeric_converted.notna().sum() > 0:
                    X[col] = numeric_converted
                    columns_to_keep.append(col)
                else:
                    print(f"    Dropping non-convertible object column: {col}")
            
            elif X[col].dtype == 'category':
                # Convert category to numeric codes
                X[col] = X[col].cat.codes
                # Replace -1 (NaN category code) with NaN
                X[col] = X[col].replace(-1, np.nan)
                columns_to_keep.append(col)
                
            elif X[col].dtype == 'bool' or X[col].dtype == 'boolean':
                # Convert boolean to int
                X[col] = X[col].astype(int)
                columns_to_keep.append(col)
                
            elif pd.api.types.is_numeric_dtype(X[col]):
                # Keep numeric columns
                columns_to_keep.append(col)
                
            else:
                print(f"    Dropping unsupported dtype column: {col} ({X[col].dtype})")
                
        except Exception as e:
            print(f"    Error processing column {col}: {e}")
    
    # Filter to keep only processed columns
    X = X[columns_to_keep]
    
    # Fill any remaining NaN values with median for numeric columns
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(X[col]):
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
            else:
                # Fill with 0 for any remaining non-numeric
                X[col] = X[col].fillna(0)
    
    print(f"  After filtering - Features: {X.shape[1]} columns")
    print(f"  Columns kept: {len(columns_to_keep)}")
    
    # Validate target
    y_clean = y.dropna()
    if len(y_clean) == 0:
        raise ValueError("Target column has no valid values!")
    
    # Remove rows where target is NaN
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"  Final dataset shape: {X.shape}")
    print(f"  Valid target values: {len(y)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Train set: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"  Test set: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
    print(f"  y_train range: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"  y_test range: {y_test.min():.2f} - {y_test.max():.2f}")
    
    return X_train, X_test, y_train, y_test, columns_to_keep

def save_processed_data(X_train, X_test, y_train, y_test, feature_names, 
                        scaler, output_dir):
    """Save processed data and metadata with validation"""
    print("\n" + "="*60)
    print("💾 SAVING PROCESSED DATA")
    print("="*60)
    
    # Validation checks before saving
    print("  Performing validation checks...")
    
    # Check shapes
    if X_train.shape[0] != len(y_train):
        raise ValueError(f"X_train and y_train shape mismatch: {X_train.shape[0]} vs {len(y_train)}")
    if X_test.shape[0] != len(y_test):
        raise ValueError(f"X_test and y_test shape mismatch: {X_test.shape[0]} vs {len(y_test)}")
    
    # Check for empty data
    if X_train.empty or X_test.empty:
        raise ValueError("Training or test features are empty!")
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Training or test targets are empty!")
    
    # Check for actual values in target
    if y_train.isna().all() or y_test.isna().all():
        raise ValueError("Target variables contain only NaN values!")
    
    print(f"  ✅ X_train: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"  ✅ X_test: {X_test.shape[0]:,} rows × {X_test.shape[1]} features")
    print(f"  ✅ y_train: {len(y_train):,} values (range: {y_train.min():.2f} - {y_train.max():.2f})")
    print(f"  ✅ y_test: {len(y_test):,} values (range: {y_test.min():.2f} - {y_test.max():.2f})")
    
    # Save datasets with proper formatting
    print("  Saving datasets...")
    
    # Save feature matrices
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    
    # Save targets as proper DataFrames with values
    y_train_df = pd.DataFrame({'target': y_train})
    y_test_df = pd.DataFrame({'target': y_test})
    
    y_train_df.to_csv(output_dir / "y_train.csv", index=False)
    y_test_df.to_csv(output_dir / "y_test.csv", index=False)
    
    # Validate saved files
    print("  Validating saved files...")
    
    # Check if files exist and have content
    for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        filepath = output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Failed to save {filename}")
        
        # Check file size
        file_size = filepath.stat().st_size
        if file_size < 100:  # Less than 100 bytes is suspicious
            print(f"  ⚠️  Warning: {filename} is very small ({file_size} bytes)")
    
    # Save feature information with updated details
    feature_info = {
        'n_features': len(feature_names),
        'feature_names': list(feature_names),
        'train_shape': list(X_train.shape),
        'test_shape': list(X_test.shape),
        'target_name': 'target',
        'target_stats': {
            'train_min': float(y_train.min()),
            'train_max': float(y_train.max()),
            'train_mean': float(y_train.mean()),
            'test_min': float(y_test.min()),
            'test_max': float(y_test.max()),
            'test_mean': float(y_test.mean())
        },
        'validation': {
            'features_match_names': len(feature_names) == X_train.shape[1],
            'no_empty_features': not X_train.isnull().all().any(),
            'no_empty_targets': not y_train.isnull().all()
        }
    }
    
    with open(output_dir / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    
    # Save scaler if provided
    if scaler:
        joblib.dump(scaler, output_dir / "scaler.pkl")
    
    # Print final summary
    print("✅ Saved files:")
    print(f"  - X_train.csv: {X_train.shape}")
    print(f"  - X_test.csv: {X_test.shape}")
    print(f"  - y_train.csv: {len(y_train)} target values")
    print(f"  - y_test.csv: {len(y_test)} target values")
    print(f"  - feature_info.json: metadata for {len(feature_names)} features")
    if scaler:
        print(f"  - scaler.pkl: fitted scaler object")
    
    # Additional validation info
    print("\n📊 Data Quality Check:")
    print(f"  - Feature-name consistency: {'✅' if feature_info['validation']['features_match_names'] else '❌'}")
    print(f"  - No empty features: {'✅' if feature_info['validation']['no_empty_features'] else '❌'}")
    print(f"  - Valid targets: {'✅' if feature_info['validation']['no_empty_targets'] else '❌'}")
    
    return feature_info

def generate_preprocessing_summary(df_initial, df_final, output_dir):
    """Generate preprocessing summary"""
    print("\n📋 GENERATING PREPROCESSING SUMMARY")
    print("=" * 50)
    
    summary = {
        'initial_data': {
            'shape': list(df_initial.shape),
            'memory_mb': round(df_initial.memory_usage(deep=True).sum() / 1024**2, 2),
            'missing_values': int(df_initial.isnull().sum().sum())
        },
        'final_data': {
            'shape': list(df_final.shape),
            'memory_mb': round(df_final.memory_usage(deep=True).sum() / 1024**2, 2),
            'missing_values': int(df_final.isnull().sum().sum())
        },
        'preprocessing_steps': [
            'Memory optimization',
            'Missing values handling',
            'Duplicate removal',
            'Feature engineering',
            'Categorical encoding',
            'Train/test split'
        ]
    }
    
    # Calculate improvements
    memory_reduction = ((summary['initial_data']['memory_mb'] - summary['final_data']['memory_mb']) 
                       / summary['initial_data']['memory_mb'] * 100)
    
    summary['improvements'] = {
        'memory_reduction_pct': round(memory_reduction, 1),
        'missing_values_eliminated': summary['initial_data']['missing_values'] - summary['final_data']['missing_values']
    }
    
    # Save summary
    with open(output_dir / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Summary generated and saved")
    print(f"  - Memory reduction: {summary['improvements']['memory_reduction_pct']:.1f}%")
    print(f"  - Missing values eliminated: {summary['improvements']['missing_values_eliminated']:,}")
    
    return summary

def main():
    """Main preprocessing pipeline"""
    print("🔧 Starting Data Preprocessing")
    print("=" * 60)
    
    # Setup environment
    output_dir = setup_environment()
    # Configuration
    DATA_PATH = Path("data/00_raw/used_cars_data.csv")  # Update this path
    SAMPLE_SIZE = None  # Set to number (e.g., 100000) for faster testing
    
    # Load data with selected columns
    print("\n📂 Data Loading Configuration:")
    print(f"  - Using {len(SELECTED_COLUMNS)} essential columns")
    print(f"  - Sample size: {'All rows' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE:,} rows'}")
    
    # Load initial data
    print("\n📂 Loading data...")
    df_initial = load_data_for_preprocessing(
        DATA_PATH,
        columns=SELECTED_COLUMNS,
        sample_size=SAMPLE_SIZE
    )
    df = df_initial.copy()
    
    print(f"📊 Initial data shape: {df.shape}")
    print(f"💾 Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Preprocessing pipeline
    print("\n🧹 Starting preprocessing pipeline...")
    
    # Step 1: Initial cleaning
    df = initial_cleaning(df)
    
    # Step 2: Clean text columns
    df = clean_text_columns(df)
    
    # Step 3: Handle missing values
    df = handle_missing_values(df)
    
    # Step 4: Create basic features
    df = create_basic_features(df)
    
    # Step 5: Create interaction features
    df = create_interaction_features(df)
    
    # Step 6: Create target encoding features
    df = create_target_encoding_features(df)
    
    # Step 7: Encode categorical features
    df = encode_categorical_features(df)
    
    # Step 8: Scale features
    df, scaler = scale_features(df)
    
    # Step 9: Prepare final dataset
    X_train, X_test, y_train, y_test, feature_names = prepare_final_dataset(df)
    
    # Step 10: Save processed data
    save_processed_data(X_train, X_test, y_train, y_test, feature_names, scaler, output_dir)
    
    # Step 11: Generate summary
    summary = generate_preprocessing_summary(df_initial, df, output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETED!")
    print("=" * 60)
    print(f"📊 Initial shape: {summary['initial_data']['shape']}")
    print(f"📊 Final shape: {summary['final_data']['shape']}")
    print(f"💾 Memory reduced: {summary['improvements']['memory_reduction_pct']:.1f}%")
    print(f"🕳️ Missing values eliminated: {summary['improvements']['missing_values_eliminated']:,}")
    print(f"📁 Results saved to: {output_dir}")
    print("▶️  Next step: Run algorithms for optimization")
    
    return summary

if __name__ == "__main__":
    main()
