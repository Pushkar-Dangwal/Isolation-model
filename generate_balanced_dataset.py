"""
Generate a balanced 300k dataset for fraud detection
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate 300k samples (150k fraud, 150k legitimate)
n_samples = 300000
n_fraud = 150000
n_legit = 150000

print(f"Generating {n_samples:,} samples ({n_fraud:,} fraud, {n_legit:,} legitimate)...")

# Generate legitimate transactions
legit_data = {
    'transaction_amount': np.random.lognormal(4, 1.5, n_legit).clip(10, 5000),
    'account_age_days': np.random.gamma(5, 100, n_legit).clip(30, 3650),
    'transaction_hour': np.random.choice(range(6, 23), n_legit),
    'merchant_category': np.random.choice(['retail', 'grocery', 'gas', 'restaurant', 'online'], n_legit),
    'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_legit, p=[0.6, 0.3, 0.1]),
    'location_match': np.random.choice([True, False], n_legit, p=[0.95, 0.05]),
    'previous_transactions': np.random.poisson(50, n_legit).clip(0, 500),
    'avg_transaction_amount': np.random.lognormal(3.5, 1, n_legit).clip(20, 2000),
    'days_since_last_transaction': np.random.exponential(3, n_legit).clip(0, 30),
    'is_fraud': [0] * n_legit
}

# Generate fraudulent transactions
fraud_data = {
    'transaction_amount': np.random.lognormal(6, 1.8, n_fraud).clip(500, 10000),
    'account_age_days': np.random.gamma(2, 30, n_fraud).clip(1, 365),
    'transaction_hour': np.random.choice(range(0, 6), n_fraud),
    'merchant_category': np.random.choice(['online', 'electronics', 'jewelry', 'travel', 'crypto'], n_fraud),
    'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_fraud, p=[0.4, 0.5, 0.1]),
    'location_match': np.random.choice([True, False], n_fraud, p=[0.2, 0.8]),
    'previous_transactions': np.random.poisson(10, n_fraud).clip(0, 100),
    'avg_transaction_amount': np.random.lognormal(3, 0.8, n_fraud).clip(10, 1000),
    'days_since_last_transaction': np.random.exponential(10, n_fraud).clip(1, 90),
    'is_fraud': [1] * n_fraud
}

# Combine and shuffle
df_legit = pd.DataFrame(legit_data)
df_fraud = pd.DataFrame(fraud_data)
df = pd.concat([df_legit, df_fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add transaction IDs and timestamps
df.insert(0, 'transaction_id', [f'TXN{i:08d}' for i in range(len(df))])
start_date = datetime(2025, 1, 1)
df['transaction_date'] = [start_date + timedelta(minutes=i*2) for i in range(len(df))]

# Save to CSV
output_file = 'data/financial_300k_balanced.csv'
df.to_csv(output_file, index=False)

print(f"\nDataset saved to: {output_file}")
print(f"Total samples: {len(df):,}")
print(f"Fraud samples: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"Legitimate samples: {(~df['is_fraud'].astype(bool)).sum():,} ({(1-df['is_fraud'].mean())*100:.1f}%)")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
