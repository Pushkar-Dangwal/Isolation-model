"""
Generate realistic balanced dataset using patterns from financial.csv
Uses imblearn SMOTE for intelligent oversampling
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATING REALISTIC BALANCED DATASET FROM FINANCIAL.CSV")
print("=" * 80)

# Load original data
print("\n1. Loading original financial.csv...")
df_original = pd.read_csv('data/financial.csv', nrows=50000)  # Load 50k rows
print(f"   Loaded {len(df_original):,} rows")
print(f"   Fraud: {df_original['is_fraud'].sum():,} ({df_original['is_fraud'].mean()*100:.4f}%)")
print(f"   Legitimate: {(~df_original['is_fraud']).sum():,} ({(~df_original['is_fraud']).mean()*100:.2f}%)")

# Prepare features for SMOTE
print("\n2. Preparing features for SMOTE...")

# Select relevant numeric and categorical features
feature_cols = [
    'amount', 'time_since_last_transaction', 'spending_deviation_score',
    'velocity_score', 'geo_anomaly_score'
]

categorical_cols = [
    'transaction_type', 'merchant_category', 'location', 
    'device_used', 'payment_channel'
]

# Handle missing values
df_clean = df_original.copy()
df_clean['time_since_last_transaction'].fillna(df_clean['time_since_last_transaction'].median(), inplace=True)
df_clean['fraud_type'].fillna('none', inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# Prepare feature matrix
encoded_cols = [f'{col}_encoded' for col in categorical_cols]
all_features = feature_cols + encoded_cols

X = df_clean[all_features].values
y = df_clean['is_fraud'].values

print(f"   Features: {len(all_features)}")
print(f"   Samples: {len(X):,}")

# Apply SMOTE to generate balanced data
print("\n3. Applying SMOTE to generate balanced samples...")
print("   Target: 300,000 samples (150k fraud, 150k legitimate)")

# Calculate sampling strategy
n_legitimate = (~df_original['is_fraud']).sum()
n_fraud_original = df_original['is_fraud'].sum()

print(f"   Original fraud samples: {n_fraud_original}")
print(f"   Original legitimate samples: {n_legitimate}")

# Use SMOTE to oversample minority class (fraud)
smote = SMOTE(
    sampling_strategy={
        True: 150000,   # Generate 150k fraud samples
        False: 150000   # Keep/sample 150k legitimate
    },
    random_state=42,
    k_neighbors=min(5, n_fraud_original - 1) if n_fraud_original > 1 else 1
)

try:
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"   ✓ SMOTE completed successfully")
    print(f"   Generated {len(X_resampled):,} samples")
except Exception as e:
    print(f"   SMOTE failed: {e}")
    print("   Falling back to random oversampling...")
    
    # Fallback: Manual balanced sampling
    fraud_indices = np.where(y)[0]
    legit_indices = np.where(~y)[0]
    
    # Sample with replacement
    fraud_sample_indices = np.random.choice(fraud_indices, size=150000, replace=True)
    legit_sample_indices = np.random.choice(legit_indices, size=150000, replace=True)
    
    all_indices = np.concatenate([fraud_sample_indices, legit_sample_indices])
    np.random.shuffle(all_indices)
    
    X_resampled = X[all_indices]
    y_resampled = y[all_indices]

print(f"   Final fraud samples: {y_resampled.sum():,} ({y_resampled.mean()*100:.1f}%)")
print(f"   Final legitimate samples: {(~y_resampled).sum():,} ({(~y_resampled).mean()*100:.1f}%)")

# Create DataFrame with resampled data
print("\n4. Creating balanced dataset...")
df_balanced = pd.DataFrame(X_resampled, columns=all_features)
df_balanced['is_fraud'] = y_resampled

# Decode categorical variables
for col in categorical_cols:
    encoded_col = f'{col}_encoded'
    # Clip values to valid range
    df_balanced[encoded_col] = df_balanced[encoded_col].clip(
        0, len(label_encoders[col].classes_) - 1
    ).astype(int)
    df_balanced[col] = label_encoders[col].inverse_transform(df_balanced[encoded_col])
    df_balanced.drop(encoded_col, axis=1, inplace=True)

# Add realistic transaction IDs and timestamps
print("\n5. Adding transaction metadata...")
df_balanced['transaction_id'] = [f'TXN{i:08d}' for i in range(len(df_balanced))]
start_date = datetime(2024, 1, 1)
df_balanced['timestamp'] = [
    (start_date + timedelta(minutes=i*2)).isoformat() 
    for i in range(len(df_balanced))
]

# Add other fields based on patterns
df_balanced['sender_account'] = [f'ACC{np.random.randint(100000, 999999)}' for _ in range(len(df_balanced))]
df_balanced['receiver_account'] = [f'ACC{np.random.randint(100000, 999999)}' for _ in range(len(df_balanced))]
df_balanced['ip_address'] = [f'{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}.{np.random.randint(1,255)}' for _ in range(len(df_balanced))]
df_balanced['device_hash'] = [f'D{np.random.randint(1000000, 9999999)}' for _ in range(len(df_balanced))]
df_balanced['fraud_type'] = df_balanced['is_fraud'].apply(lambda x: 'synthetic_fraud' if x else 'none')

# Reorder columns to match original
column_order = [
    'transaction_id', 'timestamp', 'sender_account', 'receiver_account',
    'amount', 'transaction_type', 'merchant_category', 'location', 'device_used',
    'is_fraud', 'fraud_type', 'time_since_last_transaction', 
    'spending_deviation_score', 'velocity_score', 'geo_anomaly_score',
    'payment_channel', 'ip_address', 'device_hash'
]

df_balanced = df_balanced[column_order]

# Save to CSV
print("\n6. Saving balanced dataset...")
output_file = 'data/financial_balanced_300k.csv'
df_balanced.to_csv(output_file, index=False)

print(f"   ✓ Saved to: {output_file}")
print(f"   Total samples: {len(df_balanced):,}")
print(f"   Fraud: {df_balanced['is_fraud'].sum():,} ({df_balanced['is_fraud'].mean()*100:.1f}%)")
print(f"   Legitimate: {(~df_balanced['is_fraud']).sum():,} ({(~df_balanced['is_fraud']).mean()*100:.1f}%)")

# Show statistics
print("\n7. Dataset statistics:")
print("\nNumeric features:")
print(df_balanced[feature_cols].describe())

print("\nCategorical features:")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df_balanced[col].value_counts().head())

print("\n" + "=" * 80)
print("DATASET GENERATION COMPLETE")
print("=" * 80)
print(f"\nGenerated file: {output_file}")
print(f"Total samples: {len(df_balanced):,}")
print(f"Balanced: 50% fraud, 50% legitimate")
print("\nReady for training with original model architecture!")
