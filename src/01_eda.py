#!/usr/bin/env python3
"""
01. Exploratory Data Analysis (EDA) 
Optimized version with selective column loading
Input: data/00_raw/used_cars_data.csv
Output: data/01_eda/ (plots, reports, statistics)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from src.utils.data_loader import load_data_chunked
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# Define columns to use from the beginning to avoid loading all data
ESSENTIAL_COLUMNS = [
    # Target variable
    'price',
    
    # Vehicle identification
    'vin', 'year', 'make_name', 'model_name', 'trim_name',
    
    # Core features
    'body_type', 'fuel_type', 'engine_type', 'transmission', 
    'wheel_system', 'exterior_color', 'interior_color',
    
    # Numeric features
    'mileage', 'horsepower', 'engine_displacement', 'fuel_tank_volume',
    'city_fuel_economy', 'highway_fuel_economy', 
    'maximum_seating', 'daysonmarket',
    
    # Dimensions
    'length', 'width', 'height', 'wheelbase',
    'back_legroom', 'front_legroom',
    
    # Dealer info
    'seller_rating', 'dealer_zip', 'city', 'franchise_dealer',
    
    # Condition indicators
    'is_new', 'has_accidents', 'frame_damaged', 'salvage',
    'fleet', 'theft_title', 'owner_count',
    
    # Additional features
    'torque', 'power', 'major_options', 'listing_color'
]

# Columns known to have >50% missing values (from initial analysis)
HIGH_MISSING_COLUMNS = [
    'bed', 'bed_height', 'bed_length', 'cabin', 
    'combine_fuel_economy', 'is_certified', 'is_cpo', 
    'is_oemcpo', 'vehicle_damage_category'
]

# ==================== HELPER FUNCTIONS ====================

def setup_environment():
    """Setup plotting style and output directory"""
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = Path("data/01_eda")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir

def load_data_optimized(file_path, use_columns=None, sample_size=None, chunk_size=10000):
    """
    Load data with chunked processing for memory efficiency
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    use_columns : list
        List of columns to load (None = all columns)
    sample_size : int
        Number of rows to sample (None = all rows)
    chunk_size : int
        Size of chunks for processing
    """
    print("📂 Loading data with EDA optimization...")
    
    # Use the shared chunking function from utils
    df = load_data_chunked(
        file_path=file_path,
        chunk_size=chunk_size,
        max_rows=sample_size,
        columns=use_columns
    )
    
    return df

# optimize_dtypes_eda function removed - now using optimize_dataframe_dtypes from utils

# ==================== ANALYSIS FUNCTIONS ====================

def basic_info_analysis(df, output_dir):
    """Perform basic data information analysis"""
    print("\n" + "="*60)
    print("📊 BASIC DATA INFORMATION")
    print("="*60)
    
    info = {
        'dataset_shape': df.shape,
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'total_missing_values': int(df.isnull().sum().sum()),
        'missing_percentage': round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_percentage': round(df.duplicated().sum() / df.shape[0] * 100, 2)
    }
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    info['data_types'] = {str(k): int(v) for k, v in dtype_counts.items()}
    
    # Column categorization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    info['column_types'] = {
        'numeric': len(numeric_cols),
        'categorical': len(categorical_cols),
        'boolean': len(boolean_cols),
        'numeric_cols': numeric_cols[:10],  # Save first 10 for reference
        'categorical_cols': categorical_cols[:10]
    }
    
    # Print summary
    print(f"Shape: {info['dataset_shape'][0]:,} rows × {info['dataset_shape'][1]} columns")
    print(f"Memory: {info['memory_usage_mb']:.2f} MB")
    print(f"Missing: {info['total_missing_values']:,} ({info['missing_percentage']:.1f}%)")
    print(f"Duplicates: {info['duplicate_rows']:,} ({info['duplicate_percentage']:.1f}%)")
    print(f"\nColumn Types:")
    print(f"  - Numeric: {info['column_types']['numeric']}")
    print(f"  - Categorical: {info['column_types']['categorical']}")
    print(f"  - Boolean: {info['column_types']['boolean']}")
    
    # Save to file
    with open(output_dir / "basic_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    
    return info

def missing_values_analysis(df, output_dir):
    """Analyze missing values in detail"""
    print("\n" + "="*60)
    print("🕳️ MISSING VALUES ANALYSIS")
    print("="*60)
    
    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'column': missing_count.index,
        'missing_count': missing_count.values,
        'missing_pct': missing_pct.values,
        'dtype': df.dtypes.values
    })
    
    # Sort by missing percentage
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    # Categorize by missing percentage
    critical_missing = missing_df[missing_df['missing_pct'] > 50]
    high_missing = missing_df[(missing_df['missing_pct'] > 20) & (missing_df['missing_pct'] <= 50)]
    moderate_missing = missing_df[(missing_df['missing_pct'] > 5) & (missing_df['missing_pct'] <= 20)]
    low_missing = missing_df[missing_df['missing_pct'] <= 5]
    
    print(f"Columns with missing values: {len(missing_df)}/{len(df.columns)}")
    print(f"\nMissing Value Categories:")
    print(f"  - Critical (>50%): {len(critical_missing)} columns")
    print(f"  - High (20-50%): {len(high_missing)} columns")
    print(f"  - Moderate (5-20%): {len(moderate_missing)} columns")
    print(f"  - Low (≤5%): {len(low_missing)} columns")
    
    if len(critical_missing) > 0:
        print(f"\n⚠️ Critical missing columns (consider dropping):")
        for _, row in critical_missing.head(10).iterrows():
            print(f"  - {row['column']}: {row['missing_pct']:.1f}%")
    
    # Visualize missing values
    if len(missing_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of top missing columns
        top_missing = missing_df.head(20)
        axes[0].barh(range(len(top_missing)), top_missing['missing_pct'])
        axes[0].set_yticks(range(len(top_missing)))
        axes[0].set_yticklabels(top_missing['column'])
        axes[0].set_xlabel('Missing Percentage (%)')
        axes[0].set_title('Top 20 Columns with Missing Values')
        axes[0].axvline(x=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        axes[0].legend()
        
        # Histogram of missing percentages
        axes[1].hist(missing_df['missing_pct'], bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Missing Percentage (%)')
        axes[1].set_ylabel('Number of Columns')
        axes[1].set_title('Distribution of Missing Values Across Columns')
        axes[1].axvline(x=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "missing_values_analysis.png", dpi=100, bbox_inches='tight')
        plt.show()
    
    # Save detailed report
    missing_df.to_csv(output_dir / "missing_values_report.csv", index=False)
    
    return missing_df

def target_variable_analysis(df, output_dir, target='price'):
    """Analyze the target variable (price)"""
    print("\n" + "="*60)
    print("🎯 TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    if target not in df.columns:
        print(f"❌ Target column '{target}' not found!")
        return None
    
    # Remove invalid prices
    price_data = df[target].dropna()
    price_data = price_data[price_data > 0]  # Remove zero/negative prices
    
    # Calculate statistics
    stats = {
        'count': int(len(price_data)),
        'mean': float(price_data.mean()),
        'median': float(price_data.median()),
        'std': float(price_data.std()),
        'min': float(price_data.min()),
        'max': float(price_data.max()),
        'q1': float(price_data.quantile(0.25)),
        'q3': float(price_data.quantile(0.75)),
        'iqr': float(price_data.quantile(0.75) - price_data.quantile(0.25)),
        'skewness': float(price_data.skew()),
        'kurtosis': float(price_data.kurtosis())
    }
    
    # Identify outliers using IQR method
    lower_bound = stats['q1'] - 1.5 * stats['iqr']
    upper_bound = stats['q3'] + 1.5 * stats['iqr']
    outliers_count = len(price_data[(price_data < lower_bound) | (price_data > upper_bound)])
    stats['outliers_count'] = outliers_count
    stats['outliers_pct'] = round(outliers_count / len(price_data) * 100, 2)
    
    print(f"Valid prices: {stats['count']:,}")
    print(f"Mean: ${stats['mean']:,.0f}")
    print(f"Median: ${stats['median']:,.0f}")
    print(f"Std Dev: ${stats['std']:,.0f}")
    print(f"Range: ${stats['min']:,.0f} - ${stats['max']:,.0f}")
    print(f"IQR: ${stats['iqr']:,.0f}")
    print(f"Outliers: {stats['outliers_count']:,} ({stats['outliers_pct']:.1f}%)")
    print(f"Skewness: {stats['skewness']:.2f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Histogram
    axes[0, 0].hist(price_data, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].axvline(stats['mean'], color='r', linestyle='--', label=f'Mean: ${stats["mean"]:,.0f}')
    axes[0, 0].axvline(stats['median'], color='g', linestyle='--', label=f'Median: ${stats["median"]:,.0f}')
    axes[0, 0].legend()
    
    # 2. Log-scale histogram
    axes[0, 1].hist(price_data, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Price ($)')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].set_title('Price Distribution (Log Scale)')
    axes[0, 1].set_yscale('log')
    
    # 3. Box plot
    axes[0, 2].boxplot(price_data, vert=True)
    axes[0, 2].set_ylabel('Price ($)')
    axes[0, 2].set_title('Price Box Plot')
    axes[0, 2].set_xticklabels(['Price'])
    
    # 4. Q-Q plot
    from scipy import stats as scipy_stats
    scipy_stats.probplot(price_data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    
    # 5. Log-transformed distribution
    log_price = np.log10(price_data)
    axes[1, 1].hist(log_price, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Log10(Price)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Log-Transformed Price Distribution')
    
    # 6. Price ranges distribution
    price_ranges = pd.cut(price_data, bins=[0, 10000, 25000, 50000, 75000, 100000, float('inf')],
                          labels=['<$10K', '$10-25K', '$25-50K', '$50-75K', '$75-100K', '>$100K'])
    price_ranges.value_counts().sort_index().plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_xlabel('Price Range')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Distribution by Price Range')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "target_analysis.png", dpi=100, bbox_inches='tight')
    plt.show()
    
    # Save statistics
    with open(output_dir / "target_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return stats

def numeric_features_analysis(df, output_dir):
    """Analyze numeric features"""
    print("\n" + "="*60)
    print("📈 NUMERIC FEATURES ANALYSIS")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'price' in numeric_cols:
        numeric_cols.remove('price')  # Remove target variable
    
    print(f"Found {len(numeric_cols)} numeric features")
    
    # Basic statistics for numeric columns
    numeric_stats = df[numeric_cols].describe().T
    numeric_stats['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
    numeric_stats['zeros_pct'] = ((df[numeric_cols] == 0).sum() / len(df) * 100).round(2)
    numeric_stats['unique_count'] = df[numeric_cols].nunique()
    
    print("\nTop numeric features by correlation with price:")
    if 'price' in df.columns:
        price_corr = df[numeric_cols + ['price']].corr()['price'].drop('price').abs().sort_values(ascending=False)
        for feat, corr in price_corr.head(10).items():
            print(f"  - {feat}: {corr:.3f}")
        
        # Save correlation matrix
        price_corr.to_csv(output_dir / "price_correlations.csv")
    
    # Visualize top numeric features
    top_numeric = numeric_cols[:min(12, len(numeric_cols))]
    if len(top_numeric) > 0:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(top_numeric):
            data = df[col].dropna()
            axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col}\n(missing: {df[col].isnull().sum() / len(df) * 100:.1f}%)')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        # Hide unused subplots
        for idx in range(len(top_numeric), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Distribution of Top Numeric Features', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "numeric_features_distributions.png", dpi=100, bbox_inches='tight')
        plt.show()
    
    # Save statistics
    numeric_stats.to_csv(output_dir / "numeric_features_stats.csv")
    
    return numeric_stats

def categorical_features_analysis(df, output_dir):
    """Analyze categorical features"""
    print("\n" + "="*60)
    print("📊 CATEGORICAL FEATURES ANALYSIS")
    print("="*60)
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Found {len(categorical_cols)} categorical features")
    
    cat_analysis = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        missing_pct = df[col].isnull().sum() / len(df) * 100
        
        cat_analysis.append({
            'column': col,
            'unique_values': unique_count,
            'missing_pct': round(missing_pct, 2),
            'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
            'top_value_pct': round(df[col].value_counts().iloc[0] / len(df) * 100, 2) if len(df[col].value_counts()) > 0 else 0
        })
    
    cat_df = pd.DataFrame(cat_analysis).sort_values('unique_values', ascending=False)
    
    # Categorize by cardinality
    high_cardinality = cat_df[cat_df['unique_values'] > 50]
    medium_cardinality = cat_df[(cat_df['unique_values'] > 10) & (cat_df['unique_values'] <= 50)]
    low_cardinality = cat_df[cat_df['unique_values'] <= 10]
    
    print(f"\nCardinality Distribution:")
    print(f"  - High (>50): {len(high_cardinality)} columns")
    print(f"  - Medium (11-50): {len(medium_cardinality)} columns")
    print(f"  - Low (≤10): {len(low_cardinality)} columns")
    
    if len(high_cardinality) > 0:
        print(f"\n⚠️ High cardinality columns (consider encoding/dropping):")
        for _, row in high_cardinality.head(5).iterrows():
            print(f"  - {row['column']}: {row['unique_values']} unique values")
    
    # Visualize top categorical features
    important_cats = ['body_type', 'fuel_type', 'transmission', 'make_name', 'engine_type', 'wheel_system']
    available_cats = [col for col in important_cats if col in categorical_cols]
    
    if len(available_cats) > 0 and 'price' in df.columns:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(available_cats[:6]):
            # Get top categories
            top_cats = df[col].value_counts().head(10)
            
            # Calculate mean price for each category
            price_by_cat = df.groupby(col)['price'].mean().loc[top_cats.index]
            
            axes[idx].bar(range(len(price_by_cat)), price_by_cat.values)
            axes[idx].set_xticks(range(len(price_by_cat)))
            axes[idx].set_xticklabels(price_by_cat.index, rotation=45, ha='right')
            axes[idx].set_title(f'Mean Price by {col}')
            axes[idx].set_ylabel('Mean Price ($)')
        
        plt.suptitle('Price Analysis by Categorical Features', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "categorical_price_analysis.png", dpi=100, bbox_inches='tight')
        plt.show()
    
    # Save analysis
    cat_df.to_csv(output_dir / "categorical_features_analysis.csv", index=False)
    
    return cat_df

def correlation_analysis(df, output_dir):
    """Perform correlation analysis"""
    print("\n" + "="*60)
    print("🔗 CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("❌ Not enough numeric columns for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated features (excluding self-correlation)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': round(corr_matrix.iloc[i, j], 3)
                })
    
    if high_corr_pairs:
        print(f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
        for pair in high_corr_pairs[:5]:
            print(f"  - {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']}")
    
    # Visualize correlation matrix
    if 'price' in corr_matrix.columns:
        # Focus on features most correlated with price
        price_corr = corr_matrix['price'].abs().sort_values(ascending=False)
        top_features = price_corr.head(min(15, len(price_corr))).index.tolist()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix.loc[top_features, top_features], 
                   annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix - Top Features with Price')
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=100, bbox_inches='tight')
        plt.show()
    
    # Save correlation matrix and high correlation pairs
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")
    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs).to_csv(output_dir / "high_correlation_pairs.csv", index=False)
    
    return corr_matrix

def data_quality_report(df, output_dir):
    """Generate comprehensive data quality report"""
    print("\n" + "="*60)
    print("📋 DATA QUALITY REPORT")
    print("="*60)
    
    report = {
        'overview': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_rows_pct': round(df.duplicated().sum() / len(df) * 100, 2)
        },
        'missing_data': {
            'total_missing_values': int(df.isnull().sum().sum()),
            'columns_with_missing': int((df.isnull().sum() > 0).sum()),
            'complete_rows': int(df.dropna().shape[0]),
            'complete_rows_pct': round(df.dropna().shape[0] / len(df) * 100, 2)
        },
        'column_types': {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object']).columns),
            'datetime': len(df.select_dtypes(include=['datetime']).columns),
            'boolean': len(df.select_dtypes(include=['bool']).columns)
        }
    }
    
    # Add target variable info if exists
    if 'price' in df.columns:
        price_data = df['price'].dropna()
        price_data = price_data[price_data > 0]
        report['target_variable'] = {
            'name': 'price',
            'valid_count': len(price_data),
            'missing_count': df['price'].isnull().sum(),
            'zero_negative_count': len(df[(df['price'] <= 0) & (df['price'].notna())]),
            'mean': round(price_data.mean(), 2),
            'median': round(price_data.median(), 2),
            'std': round(price_data.std(), 2)
        }
    
    # Identify potential issues
    issues = []
    
    # Check for high missing values
    high_missing = df.columns[df.isnull().sum() / len(df) > 0.5].tolist()
    if high_missing:
        issues.append(f"High missing values (>50%): {', '.join(high_missing[:5])}")
    
    # Check for high cardinality
    cat_cols = df.select_dtypes(include=['object']).columns
    high_card = [col for col in cat_cols if df[col].nunique() > 100]
    if high_card:
        issues.append(f"High cardinality categorical: {', '.join(high_card[:5])}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant/single-value columns: {', '.join(constant_cols)}")
    
    report['potential_issues'] = issues
    
    # Print summary
    print("Data Quality Summary:")
    print(f"  ✓ Completeness: {report['missing_data']['complete_rows_pct']:.1f}% rows are complete")
    print(f"  ✓ Uniqueness: {100 - report['overview']['duplicate_rows_pct']:.1f}% rows are unique")
    print(f"  ✓ Columns: {report['overview']['total_columns']} total")
    
    if issues:
        print(f"\n⚠️ Potential Issues Found ({len(issues)}):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No major data quality issues detected")
    
    # Save report
    with open(output_dir / "data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    return report

def generate_eda_summary(all_results, output_dir):
    """Generate final EDA summary and recommendations"""
    print("\n" + "="*60)
    print("📝 EDA SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # Based on missing values
    if 'missing_df' in all_results and len(all_results['missing_df']) > 0:
        critical_missing = all_results['missing_df'][all_results['missing_df']['missing_pct'] > 50]
        if len(critical_missing) > 0:
            recommendations.append({
                'type': 'DROP_COLUMNS',
                'priority': 'HIGH',
                'description': f"Drop {len(critical_missing)} columns with >50% missing values",
                'columns': critical_missing['column'].tolist()
            })
    
    # Based on duplicates
    if 'basic_info' in all_results:
        if all_results['basic_info']['duplicate_rows'] > 0:
            recommendations.append({
                'type': 'REMOVE_DUPLICATES',
                'priority': 'HIGH',
                'description': f"Remove {all_results['basic_info']['duplicate_rows']:,} duplicate rows"
            })
    
    # Based on target variable
    if 'target_stats' in all_results:
        if all_results['target_stats']['outliers_pct'] > 5:
            recommendations.append({
                'type': 'HANDLE_OUTLIERS',
                'priority': 'MEDIUM',
                'description': f"Handle price outliers ({all_results['target_stats']['outliers_pct']:.1f}% of data)",
                'suggestion': "Consider using IQR method or percentile capping"
            })
        
        if all_results['target_stats']['skewness'] > 1:
            recommendations.append({
                'type': 'TRANSFORM_TARGET',
                'priority': 'MEDIUM',
                'description': "Apply log transformation to target variable (high skewness)",
                'skewness': all_results['target_stats']['skewness']
            })
    
    # Based on categorical features
    if 'categorical_df' in all_results:
        high_card = all_results['categorical_df'][all_results['categorical_df']['unique_values'] > 50]
        if len(high_card) > 0:
            recommendations.append({
                'type': 'ENCODE_CATEGORICAL',
                'priority': 'HIGH',
                'description': f"Handle {len(high_card)} high-cardinality categorical features",
                'columns': high_card['column'].tolist(),
                'suggestion': "Use target encoding or feature hashing"
            })
    
    # Print recommendations
    print("Key Recommendations:")
    priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    sorted_recs = sorted(recommendations, key=lambda x: priority_order.get(x['priority'], 4))
    
    for i, rec in enumerate(sorted_recs, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['type']}")
        print(f"   {rec['description']}")
        if 'suggestion' in rec:
            print(f"   💡 {rec['suggestion']}")
    
    # Create summary report
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_overview': all_results.get('basic_info', {}),
        'recommendations': recommendations,
        'next_steps': [
            "Run preprocessing pipeline (02_preprocessing.py)",
            "Handle missing values based on analysis",
            "Encode categorical variables appropriately",
            "Create train/test splits",
            "Apply feature engineering"
        ]
    }
    
    # Save summary
    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("✅ EDA COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 All results saved to: {output_dir}")
    print("🚀 Next step: Run preprocessing pipeline")
    
    return summary

# ==================== MAIN PIPELINE ====================

def main():
    """Main EDA pipeline with optimized data loading"""
    print("="*60)
    print("🔍 STARTING EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Setup
    output_dir = setup_environment()
    
    # Configuration
    DATA_PATH = Path("data/00_raw/used_cars_data.csv")  # Update this path
    SAMPLE_SIZE = None  # Set to number (e.g., 100000) for faster testing
    
    # Load data with selected columns
    print("\n📂 Data Loading Configuration:")
    print(f"  - Using {len(ESSENTIAL_COLUMNS)} essential columns")
    print(f"  - Sample size: {'All rows' if SAMPLE_SIZE is None else f'{SAMPLE_SIZE:,} rows'}")
    
    df = load_data_optimized(
        DATA_PATH,
        use_columns=ESSENTIAL_COLUMNS,
        sample_size=SAMPLE_SIZE
    )
    
    # Store all results
    all_results = {}
    
    # Run analyses
    all_results['basic_info'] = basic_info_analysis(df, output_dir)
    all_results['missing_df'] = missing_values_analysis(df, output_dir)
    all_results['target_stats'] = target_variable_analysis(df, output_dir)
    all_results['numeric_stats'] = numeric_features_analysis(df, output_dir)
    all_results['categorical_df'] = categorical_features_analysis(df, output_dir)
    all_results['correlation_matrix'] = correlation_analysis(df, output_dir)
    all_results['quality_report'] = data_quality_report(df, output_dir)
    
    # Generate summary and recommendations
    summary = generate_eda_summary(all_results, output_dir)
    
    return df, all_results, summary

if __name__ == "__main__":
    df, results, summary = main()