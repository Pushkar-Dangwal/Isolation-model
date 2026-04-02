#!/usr/bin/env python3
"""
Diagnose why evaluation metrics are still poor despite fixed feature engineering.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from fraud_detector import FraudDetector
from config import setup_logging, DATA_DIR

setup_logging('INFO')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("DIAGNOSING EVALUATION ISSUE")
    logger.info("="*80)
    
    # Load the newly trained model
    model_path = "models/fraud_detector_fixed_20260320_132428"
    logger.info(f"Loading model from {model_path}")
    
    detector = FraudDetector()
    detector.load_model(model_path, verify_integrity=False)
    
    # Load the same data used for training
    data_path = DATA_DIR / 'financial.csv'
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, nrows=500000)
    
    logger.info(f"Loaded {len(df)} transactions, fraud rate: {df['is_fraud'].mean():.1%}")
    
    # Split the same way as training
    y = df['is_fraud'].values
    train_indices, test_indices = train_test_split(
        np.arange(len(df)), 
        test_size=0.2, 
        stratify=y,
        random_state=42
    )
    
    logger.info(f"Split: {len(train_indices)} train, {len(test_indices)} test")
    
    # Generate predictions on FULL dataset
    logger.info("\nGenerating predictions on full dataset...")
    all_predictions = detector.predict(df, return_probabilities=True, return_risk_levels=True)
    
    # Extract test subset
    test_predictions = all_predictions.iloc[test_indices].reset_index(drop=True)
    y_test = y[test_indices]
    
    # Analyze predictions
    logger.info("\n" + "="*80)
    logger.info("PREDICTION ANALYSIS")
    logger.info("="*80)
    
    y_proba = test_predictions['fraud_probability'].values
    y_pred = test_predictions['fraud_prediction'].values
    
    logger.info(f"Test set size: {len(y_test)}")
    logger.info(f"Actual frauds: {y_test.sum()} ({y_test.mean():.1%})")
    logger.info(f"Predicted frauds: {y_pred.sum()} ({y_pred.mean():.1%})")
    
    logger.info(f"\nFraud probability statistics:")
    logger.info(f"  Min: {y_proba.min():.6f}")
    logger.info(f"  Max: {y_proba.max():.6f}")
    logger.info(f"  Mean: {y_proba.mean():.6f}")
    logger.info(f"  Median: {np.median(y_proba):.6f}")
    logger.info(f"  Std: {y_proba.std():.6f}")
    
    # Check distribution
    logger.info(f"\nProbability distribution:")
    logger.info(f"  < 0.01: {(y_proba < 0.01).sum()} ({(y_proba < 0.01).mean():.1%})")
    logger.info(f"  0.01-0.1: {((y_proba >= 0.01) & (y_proba < 0.1)).sum()}")
    logger.info(f"  0.1-0.5: {((y_proba >= 0.1) & (y_proba < 0.5)).sum()}")
    logger.info(f"  >= 0.5: {(y_proba >= 0.5).sum()}")
    
    # Calculate metrics with different thresholds
    logger.info(f"\n" + "="*80)
    logger.info("METRICS AT DIFFERENT THRESHOLDS")
    logger.info("="*80)
    
    for threshold in [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        if y_pred_thresh.sum() == 0:
            logger.info(f"\nThreshold {threshold:.2f}: No predictions (all negative)")
            continue
            
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        logger.info(f"\nThreshold {threshold:.2f}:")
        logger.info(f"  Predicted frauds: {y_pred_thresh.sum()} ({y_pred_thresh.mean():.1%})")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1:.3f}")
    
    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        logger.info(f"\n" + "="*80)
        logger.info(f"ROC-AUC: {roc_auc:.3f}")
        logger.info("="*80)
        
        if roc_auc < 0.6:
            logger.warning("ROC-AUC is very low! Model is not discriminating between fraud and legitimate.")
        elif roc_auc > 0.9:
            logger.info("ROC-AUC is excellent! Model is discriminating well.")
    except Exception as e:
        logger.error(f"Failed to calculate ROC-AUC: {e}")
    
    # Check for actual fraud cases
    logger.info(f"\n" + "="*80)
    logger.info("FRAUD CASE ANALYSIS")
    logger.info("="*80)
    
    fraud_indices = np.where(y_test == 1)[0]
    if len(fraud_indices) > 0:
        fraud_probas = y_proba[fraud_indices]
        logger.info(f"Fraud cases: {len(fraud_indices)}")
        logger.info(f"Fraud probability stats for actual frauds:")
        logger.info(f"  Min: {fraud_probas.min():.6f}")
        logger.info(f"  Max: {fraud_probas.max():.6f}")
        logger.info(f"  Mean: {fraud_probas.mean():.6f}")
        logger.info(f"  Median: {np.median(fraud_probas):.6f}")
        
        # Check how many frauds have high probability
        logger.info(f"\nFraud detection at different thresholds:")
        for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
            detected = (fraud_probas >= threshold).sum()
            logger.info(f"  >= {threshold:.2f}: {detected}/{len(fraud_indices)} ({detected/len(fraud_indices):.1%})")
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS COMPLETED")
    logger.info("="*80)

if __name__ == '__main__':
    main()
