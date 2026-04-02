"""
Analyze the existing financial.csv to understand data patterns
"""
import pandas as pd
import numpy as np

print("=" * 80)
print("ANALYZING FINANCIAL.CSV")
print("=" * 80)

# Load data
print("\n1. Loading data...")
try:
    df = pd.read_csv('data/financial.csv', nrows=10000)  # Load first 10k rows for analysis
    print(f"   Loaded {len(df):,} rows for analysis")
except Exception as e:
    print(f"   Error loading file: {e}")
    exit(1)

# Basic info
print(f"\n2. Dataset shape: {df.shape}")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")

# Column names
print(f"\n3. Columns ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

# Data types
print("\n4. Data types:")
print(df.dtypes)

# First few rows
print("\n5. First 5 rows:")
print(df.head())

# Check for fraud column
print("\n6. Fraud distribution:")
if 'is_fraud' in df.columns:
    fraud_counts = df['is_fraud'].value_counts()
    print(f"   Legitimate: {fraud_counts.get(0, 0):,} ({fraud_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"   Fraud: {fraud_counts.get(1, 0):,} ({fraud_counts.get(1, 0)/len(df)*100:.2f}%)")
else:
    print("   No 'is_fraud' column found")
    print(f"   Available columns: {list(df.columns)}")

# Missing values
print("\n7. Missing values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   No missing values")

# Numeric columns statistics
print("\n8. Numeric columns summary:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe())
else:
    print("   No numeric columns found")

# Categorical columns
print("\n9. Categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"   {col}: {unique_count} unique values")
    if unique_count <= 10:
        print(f"      Values: {df[col].value_counts().to_dict()}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
