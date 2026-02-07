#!/usr/bin/env python3
"""
Fraud Prediction Demo Script

This script demonstrates how to use the trained fraud detection model
to predict fraud on new transactions.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fraud_detector import FraudDetector
from model_persistence import ModelPersistenceManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_transactions(n_transactions: int = 10) -> pd.DataFrame:
    """
    Create sample transactions for fraud prediction demonstration.
    
    Args:
        n_transactions: Number of sample transactions to create
        
    Returns:
        DataFrame with sample transaction data
    """
    np.random.seed(42)  
    transactions = []
    base_time = datetime.now() - timedelta(days=1)
    
    for i in range(n_transactions):
        # Mix of normal and potentially fraudulent transactions
        is_suspicious = i % 4 == 0  # Every 4th transaction is suspicious
        
        if is_suspicious:
            # Suspicious transaction patterns
            amount = np.random.uniform(5000, 50000)  # High amounts
            hour = np.random.choice([2, 3, 23, 24])  # Unusual hours
            device = np.random.choice(['mobile_suspicious', 'web_tor'])
            location = np.random.choice(['foreign_country', 'high_risk_area'])
            merchant_cat = np.random.choice(['gambling', 'crypto', 'cash_advance'])
        else:
            # Normal transaction patterns
            amount = np.random.uniform(10, 500)  # Normal amounts
            hour = np.random.choice(range(9, 18))  # Business hours
            device = np.random.choice(['mobile', 'web', 'pos'])
            location = np.random.choice(['home_city', 'work_area', 'shopping_mall'])
            merchant_cat = np.random.choice(['grocery', 'gas_station', 'restaurant', 'retail'])
        
        transaction = {
            'transaction_id': f'demo_tx_{i+1:03d}',
            'timestamp': base_time + timedelta(hours=i, minutes=np.random.randint(0, 60)),
            'sender_account': f'account_{np.random.randint(1000, 9999)}',
            'receiver_account': f'merchant_{np.random.randint(100, 999)}',
            'amount': round(amount, 2),
            'transaction_type': np.random.choice(['purchase', 'transfer', 'withdrawal']),
            'merchant_category': merchant_cat,
            'location': location,
            'device_used': device
        }
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    
    # Convert timestamp to string format (as it would come from a real system)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def load_trained_model(model_name: str = "production_fraud_detector_v1_optimized") -> FraudDetector:
    """
    Load the trained fraud detection model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded FraudDetector instance
    """
    logger.info(f"Loading trained model: {model_name}")
    
    # Find the model file directly
    models_dir = Path("models/models")
    model_dirs = list(models_dir.glob(f"{model_name}/*"))
    
    if not model_dirs:
        raise ValueError(f"No model directories found for {model_name}")
    
    # Get the latest version directory
    latest_dir = max(model_dirs, key=lambda x: x.name)
    model_file = latest_dir / f"{model_name}.joblib"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    logger.info(f"Loading model from: {model_file}")
    
    # Load the model data
    import joblib
    model_data = joblib.load(model_file)
    
    # Create a new FraudDetector and restore its state
    detector = FraudDetector()
    
    # Restore all components and state
    detector.preprocessor = model_data['preprocessor']
    detector.feature_engineer = model_data['feature_engineer'] 
    detector.anomaly_detector = model_data['anomaly_detector']
    detector.classifier = model_data['classifier']
    detector.risk_scorer = model_data['risk_scorer']
    detector.is_fitted = model_data['is_fitted']
    detector.feature_names = model_data['feature_names']
    detector.column_mapping = model_data['column_mapping']
    detector.training_metadata = model_data['training_metadata']
    detector.performance_metrics = model_data['performance_metrics']
    detector.random_state = model_data['random_state']
    detector.enable_error_handling = model_data['enable_error_handling']
    detector.ensure_reproducibility = model_data.get('ensure_reproducibility', False)
    
    logger.info("Model loaded and restored successfully")
    return detector


def predict_fraud(detector: FraudDetector, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict fraud for a batch of transactions.
    
    Args:
        detector: Trained FraudDetector instance
        transactions_df: DataFrame with transaction data
        
    Returns:
        DataFrame with fraud predictions and explanations
    """
    logger.info(f"Predicting fraud for {len(transactions_df)} transactions")
    
    # Make predictions with full details
    predictions = detector.predict(
        transactions_df,
        transaction_id_column='transaction_id',
        return_probabilities=True,
        return_risk_levels=True,
        return_explanations=True
    )
    
    logger.info("Fraud predictions completed")
    return predictions


def display_results(transactions_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """
    Display the fraud prediction results in a readable format.
    
    Args:
        transactions_df: Original transaction data
        predictions_df: Fraud predictions
    """
    print("\n" + "="*100)
    print("FRAUD DETECTION RESULTS")
    print("="*100)
    
    # Merge transactions with predictions for display
    results = transactions_df.merge(predictions_df, on='transaction_id', how='left')
    
    # Sort by fraud probability (highest first)
    results = results.sort_values('fraud_probability', ascending=False)
    
    print(f"\nAnalyzed {len(results)} transactions:")
    print(f"High Risk: {len(results[results['risk_level'] == 'high'])}")
    print(f"Medium Risk: {len(results[results['risk_level'] == 'medium'])}")
    print(f"Low Risk: {len(results[results['risk_level'] == 'low'])}")
    
    print(f"\nFraud Predictions: {results['fraud_prediction'].sum()} flagged as fraud")
    print(f"Average Fraud Probability: {results['fraud_probability'].mean():.3f}")
    
    print("\n" + "-"*100)
    print("DETAILED TRANSACTION ANALYSIS")
    print("-"*100)
    
    for idx, row in results.iterrows():
        risk_color = {
            'high': '🔴',
            'medium': '🟡', 
            'low': '🟢'
        }.get(row['risk_level'], '⚪')
        
        fraud_flag = "🚨 FRAUD DETECTED" if row['fraud_prediction'] == 1 else "✅ LEGITIMATE"
        
        print(f"\n{risk_color} Transaction ID: {row['transaction_id']}")
        print(f"   Amount: ${row['amount']:,.2f} | Time: {row['timestamp']}")
        print(f"   Merchant: {row['merchant_category']} | Location: {row['location']} | Device: {row['device_used']}")
        print(f"   {fraud_flag}")
        print(f"   Fraud Probability: {row['fraud_probability']:.3f} | Risk Level: {row['risk_level'].upper()}")
        print(f"   Anomaly Score: {row['anomaly_score']:.3f}")
        
        if 'explanation' in row and pd.notna(row['explanation']):
            print(f"   Explanation: {row['explanation']}")


def main():
    """Main demonstration function."""
    print("🔍 FRAUD DETECTION SYSTEM DEMONSTRATION")
    print("="*50)
    
    try:
        # Step 1: Create sample transactions
        print("\n📊 Step 1: Creating sample transactions...")
        sample_transactions = create_sample_transactions(n_transactions=10)
        print(f"Created {len(sample_transactions)} sample transactions")
        
        # Display sample transactions
        print("\nSample transactions created:")
        for idx, row in sample_transactions.iterrows():
            print(f"  {row['transaction_id']}: ${row['amount']:,.2f} at {row['merchant_category']} ({row['location']})")
        
        # Step 2: Load trained model
        print("\n🤖 Step 2: Loading trained fraud detection model...")
        detector = load_trained_model()
        
        # Display model info
        model_info = detector.get_model_info()
        print(f"Model Status: {model_info['status']}")
        print(f"Features: {model_info['feature_count']}")
        if 'training_metadata' in model_info:
            training_meta = model_info['training_metadata']
            print(f"Training Samples: {training_meta.get('training_samples', 'N/A'):,}")
            print(f"Model Performance: PR-AUC = {training_meta.get('pr_auc', 'N/A')}")
        
        # Step 3: Predict fraud
        print("\n🎯 Step 3: Predicting fraud for sample transactions...")
        predictions = predict_fraud(detector, sample_transactions)
        
        # Step 4: Display results
        print("\n📋 Step 4: Displaying fraud detection results...")
        display_results(sample_transactions, predictions)
        
        # Summary statistics
        fraud_count = predictions['fraud_prediction'].sum()
        high_risk_count = len(predictions[predictions['risk_level'] == 'high'])
        avg_fraud_prob = predictions['fraud_probability'].mean()
        
        print(f"\n📈 SUMMARY STATISTICS")
        print(f"Total Transactions Analyzed: {len(predictions)}")
        print(f"Transactions Flagged as Fraud: {fraud_count}")
        print(f"High Risk Transactions: {high_risk_count}")
        print(f"Average Fraud Probability: {avg_fraud_prob:.3f}")
        print(f"Fraud Detection Rate: {fraud_count/len(predictions)*100:.1f}%")
        
        print(f"\n✅ Fraud detection demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)