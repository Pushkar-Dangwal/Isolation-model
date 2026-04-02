#!/usr/bin/env python3
"""
Test script to verify the evaluation fix generates all features correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fraud_detector import FraudDetector
from config import setup_logging, DATA_DIR

# Set up logging
setup_logging('INFO')
logger = logging.getLogger(__name__)

def test_evaluation_fix():
    """Test that evaluation now generates all features correctly."""
    
    logger.info("="*80)
    logger.info("TESTING EVALUATION FIX")
    logger.info("="*80)
    
    # Load the trained model
    model_path = "models/fraud_detector_20260320_130507"
    logger.info(f"Loading model from {model_path}")
    
    detector = FraudDetector()
    detector.load_model(model_path, verify_integrity=False)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Expected features: {len(detector.feature_names)}")
    logger.info(f"Feature names: {detector.feature_names[:10]}...")  # Show first 10
    
    # Load test data
    data_path = DATA_DIR / 'financial.csv'
    logger.info(f"\nLoading test data from {data_path}")
    
    # Load a sample for testing
    df = pd.read_csv(data_path, nrows=100000)
    logger.info(f"Loaded {len(df)} transactions")
    logger.info(f"Fraud rate: {df['is_fraud'].mean():.1%}")
    
    # Test prediction on full dataset (should work)
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Predictions on FULL dataset (with historical context)")
    logger.info("="*80)
    
    predictions_full = detector.predict(df, return_probabilities=True)
    logger.info(f"✓ Generated {len(predictions_full)} predictions")
    logger.info(f"  - Mean fraud probability: {predictions_full['fraud_probability'].mean():.3f}")
    logger.info(f"  - Predicted frauds: {predictions_full['fraud_prediction'].sum()}")
    
    # Test prediction on subset (should now work with the fix)
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Predictions on SUBSET (without historical context)")
    logger.info("="*80)
    
    # Take last 20% as test set
    test_size = int(len(df) * 0.2)
    df_test = df.iloc[-test_size:].reset_index(drop=True)
    
    logger.info(f"Test subset size: {len(df_test)}")
    
    try:
        predictions_subset = detector.predict(df_test, return_probabilities=True)
        logger.info(f"✓ Generated {len(predictions_subset)} predictions")
        logger.info(f"  - Mean fraud probability: {predictions_subset['fraud_probability'].mean():.3f}")
        logger.info(f"  - Predicted frauds: {predictions_subset['fraud_prediction'].sum()}")
        
        # Check if predictions are reasonable (not all 0.5)
        if predictions_subset['fraud_probability'].std() < 0.01:
            logger.warning("⚠ WARNING: Predictions have very low variance - features may be missing!")
        else:
            logger.info(f"  - Prediction variance: {predictions_subset['fraud_probability'].std():.3f} (GOOD)")
            
    except Exception as e:
        logger.error(f"✗ Prediction on subset failed: {e}")
    
    # Test the new evaluation approach
    logger.info("\n" + "="*80)
    logger.info("TEST 3: New evaluation approach (full dataset for features)")
    logger.info("="*80)
    
    from sklearn.model_selection import train_test_split
    
    # Split indices
    train_indices, test_indices = train_test_split(
        np.arange(len(df)), 
        test_size=0.2, 
        stratify=df['is_fraud'].values,
        random_state=42
    )
    
    logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test")
    
    # Generate predictions on FULL dataset
    all_predictions = detector.predict(df, return_probabilities=True)
    
    # Extract test subset predictions
    test_predictions = all_predictions.iloc[test_indices].reset_index(drop=True)
    y_test = df['is_fraud'].values[test_indices]
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    y_proba = test_predictions['fraud_probability'].values
    y_pred = test_predictions['fraud_prediction'].values
    
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"✓ Evaluation metrics on test subset:")
    logger.info(f"  - ROC-AUC: {roc_auc:.3f}")
    logger.info(f"  - Precision: {precision:.3f}")
    logger.info(f"  - Recall: {recall:.3f}")
    logger.info(f"  - F1-Score: {f1:.3f}")
    
    # Compare with expected training performance
    expected_auc = 0.948
    if roc_auc > 0.7:
        logger.info(f"\n✓ SUCCESS: ROC-AUC {roc_auc:.3f} is reasonable (expected ~{expected_auc:.3f})")
    else:
        logger.warning(f"\n⚠ WARNING: ROC-AUC {roc_auc:.3f} is too low (expected ~{expected_auc:.3f})")
    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETED")
    logger.info("="*80)

if __name__ == '__main__':
    test_evaluation_fix()
