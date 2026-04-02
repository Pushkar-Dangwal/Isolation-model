#!/usr/bin/env python3
"""
Check what predictions are actually being made on test data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from fraud_detector import FraudDetector
from config import setup_logging, DATA_DIR

setup_logging('INFO')

# Load model
detector = FraudDetector()
detector.load_model("models/fraud_detector_20260320_133015", verify_integrity=False)

# Load data
df = pd.read_csv(DATA_DIR / 'financial.csv', nrows=5000000)
print(f"Loaded {len(df)} transactions, fraud rate: {df['is_fraud'].mean():.1%}")

# Split
y = df['is_fraud'].values
train_indices, test_indices = train_test_split(
    np.arange(len(df)), test_size=0.2, stratify=y, random_state=42
)

# Predict on full dataset
print("\nGenerating predictions...")
all_predictions = detector.predict(df, return_probabilities=True)

# Extract test predictions
test_preds = all_predictions.iloc[test_indices]
y_test = y[test_indices]

# Analyze
print(f"\nTest set: {len(y_test)} samples")
print(f"Actual frauds: {y_test.sum()} ({y_test.mean():.1%})")
print(f"\nPrediction probabilities:")
print(f"  Unique values: {len(np.unique(test_preds['fraud_probability']))}")
print(f"  Min: {test_preds['fraud_probability'].min():.6f}")
print(f"  Max: {test_preds['fraud_probability'].max():.6f}")
print(f"  Mean: {test_preds['fraud_probability'].mean():.6f}")
print(f"  Std: {test_preds['fraud_probability'].std():.6f}")

# Check if all predictions are the same
if test_preds['fraud_probability'].std() < 0.001:
    print("\n⚠️ WARNING: All predictions are nearly identical!")
    print("This explains the 0.500 ROC-AUC (random performance)")
else:
    print("\n✓ Predictions have variance")

# Show distribution
print(f"\nPrediction distribution:")
print(test_preds['fraud_probability'].describe())

# Check actual fraud cases
fraud_mask = y_test == 1
print(f"\nFraud cases (n={fraud_mask.sum()}):")
print(f"  Probability mean: {test_preds.loc[fraud_mask, 'fraud_probability'].mean():.6f}")
print(f"  Probability std: {test_preds.loc[fraud_mask, 'fraud_probability'].std():.6f}")

legit_mask = y_test == 0
print(f"\nLegitimate cases (n={legit_mask.sum()}):")
print(f"  Probability mean: {test_preds.loc[legit_mask, 'fraud_probability'].mean():.6f}")
print(f"  Probability std: {test_preds.loc[legit_mask, 'fraud_probability'].std():.6f}")
