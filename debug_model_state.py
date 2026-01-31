#!/usr/bin/env python3
"""
Debug script to check the state of the trained model
"""

import sys
sys.path.append('src')

import pandas as pd
from fraud_detector import FraudDetector
from config import setup_logging

def debug_model_state():
    setup_logging('INFO')
    
    # Load some sample data
    df = pd.read_csv('data/financial.csv', nrows=1000)
    
    # Initialize detector
    detector = FraudDetector(enable_error_handling=False)  # Disable error handling
    
    print("Training detector...")
    try:
        detector.fit(df, target_column='is_fraud', transaction_id_column='transaction_id')
        print("Training completed successfully!")
        
        # Check model state
        print(f"detector.is_fitted: {detector.is_fitted}")
        
        model_info = detector.get_model_info()
        print(f"Model info: {model_info}")
        
        # Check individual components
        print(f"Preprocessor fitted: {detector.preprocessor.is_fitted}")
        print(f"Feature engineer fitted: {detector.feature_engineer.is_fitted}")
        print(f"Anomaly detector fitted: {detector.anomaly_detector.is_fitted}")
        print(f"Classifier fitted: {detector.classifier.is_fitted}")
        print(f"Risk scorer fitted: {detector.risk_scorer.is_fitted}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_state()