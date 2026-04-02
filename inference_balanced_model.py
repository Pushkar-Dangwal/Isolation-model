"""
Inference script for the balanced fraud detection model
"""
import pandas as pd
import numpy as np
import joblib

print("=" * 80)
print("BALANCED FRAUD DETECTION MODEL - INFERENCE")
print("=" * 80)

# Load model
print("\n1. Loading model...")
model = joblib.load('models/balanced_rf_fraud_detector_20260402_135854.joblib')
print("   Model loaded successfully!")

# Create sample transactions for inference
print("\n2. Creating sample transactions...")
sample_data = {
    'transaction_amount': [150.0, 5000.0, 75.0, 8000.0, 200.0],
    'account_age_days': [365, 30, 730, 15, 500],
    'transaction_hour': [14, 2, 10, 3, 16],
    'previous_transactions': [50, 5, 100, 2, 75],
    'avg_transaction_amount': [120.0, 100.0, 150.0, 80.0, 180.0],
    'days_since_last_transaction': [2, 15, 1, 20, 3],
    'location_match': [1, 0, 1, 0, 1],
    'merchant_category': ['grocery', 'online', 'retail', 'crypto', 'restaurant'],
    'device_type': ['mobile', 'desktop', 'mobile', 'desktop', 'tablet']
}

df_samples = pd.DataFrame(sample_data)
print(f"   Created {len(df_samples)} sample transactions")

# Prepare features (same encoding as training)
print("\n3. Preparing features...")
df_encoded = pd.get_dummies(df_samples, columns=['merchant_category', 'device_type'], drop_first=True)

# Ensure all training features are present
training_features = [
    'transaction_amount', 'account_age_days', 'transaction_hour',
    'previous_transactions', 'avg_transaction_amount', 'days_since_last_transaction',
    'location_match', 'merchant_category_gas', 'merchant_category_grocery',
    'merchant_category_online', 'merchant_category_restaurant', 'merchant_category_retail',
    'merchant_category_crypto', 'merchant_category_electronics', 'merchant_category_jewelry',
    'merchant_category_travel', 'device_type_mobile', 'device_type_tablet'
]

# Add missing columns with 0
for col in training_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Reorder columns to match training
X_inference = df_encoded[training_features]

# Make predictions
print("\n4. Making predictions...")
predictions = model.predict(X_inference)
probabilities = model.predict_proba(X_inference)

# Display results
print("\n" + "=" * 80)
print("PREDICTION RESULTS")
print("=" * 80)

for i in range(len(df_samples)):
    print(f"\nTransaction {i+1}:")
    print(f"  Amount: ${df_samples.iloc[i]['transaction_amount']:.2f}")
    print(f"  Hour: {df_samples.iloc[i]['transaction_hour']}:00")
    print(f"  Account Age: {df_samples.iloc[i]['account_age_days']} days")
    print(f"  Previous Transactions: {df_samples.iloc[i]['previous_transactions']}")
    print(f"  Location Match: {'Yes' if df_samples.iloc[i]['location_match'] else 'No'}")
    print(f"  Merchant: {df_samples.iloc[i]['merchant_category']}")
    print(f"  Device: {df_samples.iloc[i]['device_type']}")
    print(f"  ---")
    print(f"  Prediction: {'🚨 FRAUD' if predictions[i] == 1 else '✓ LEGITIMATE'}")
    print(f"  Fraud Probability: {probabilities[i][1]:.2%}")
    print(f"  Legitimate Probability: {probabilities[i][0]:.2%}")

print("\n" + "=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)

# Summary
fraud_count = sum(predictions)
print(f"\nSummary: {fraud_count} out of {len(predictions)} transactions flagged as fraud")
