#!/usr/bin/env python3
"""
Training script without error handling to debug the issue
"""

import sys
sys.path.append('src')

import pandas as pd
from fraud_detector import FraudDetector
from config import setup_logging

def train_without_error_handling():
    setup_logging('INFO')
    
    # Load sample data
    print("Loading data...")
    df = pd.read_csv('data/financial.csv', nrows=50000)
    print(f"Loaded {len(df)} transactions")
    
    # Initialize detector WITHOUT error handling
    detector = FraudDetector(enable_error_handling=False)
    
    print("Training detector...")
    try:
        detector.fit(df, target_column='is_fraud', transaction_id_column='transaction_id')
        print("Training completed successfully!")
        
        # Check model state
        print(f"detector.is_fitted: {detector.is_fitted}")
        
        model_info = detector.get_model_info()
        print(f"Model status: {model_info.get('status')}")
        print(f"Feature count: {model_info.get('feature_count')}")
        print(f"Performance metrics: {model_info.get('performance_metrics')}")
        
        # Try to save the model
        print("Attempting to save model...")
        detector.save_model('models/test_no_error_handling')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_without_error_handling()