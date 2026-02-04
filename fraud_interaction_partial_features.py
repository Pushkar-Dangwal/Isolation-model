#!/usr/bin/env python3
"""
Fraud Detection Interaction Demo - Partial Features

This script demonstrates fraud detection with partial feature sets,
showing how the system handles missing features and degrades gracefully.
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
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_partial_feature_transactions(n_transactions: int = 6) -> pd.DataFrame:
    """
    Create transactions with only partial feature sets.
    
    This simulates real-world scenarios where some data might be missing
    or unavailable from certain data sources.
    """
    np.random.seed(456)  # Different seed
    
    transactions = []
    base_time = datetime.now() - timedelta(hours=8)
    
    for i in range(n_transactions):
        # Create transactions with varying levels of missing features
        missing_level = i % 3  # 0=few missing, 1=some missing, 2=many missing
        
        # Base transaction data (always present)
        transaction = {
            'transaction_id': f'partial_tx_{i+1:03d}',
            'timestamp': base_time + timedelta(hours=i*1.5, minutes=np.random.randint(0, 60)),
            'sender_account': f'account_{np.random.randint(1000, 9999)}',
            'receiver_account': f'merchant_{np.random.randint(100, 999)}',
            'amount': round(np.random.uniform(50, 5000), 2),
            'transaction_type': np.random.choice(['purchase', 'transfer', 'withdrawal']),
            'merchant_category': np.random.choice(['grocery', 'gas_station', 'restaurant', 'retail', 'gambling', 'crypto']),
            'location': np.random.choice(['home_city', 'work_area', 'foreign_country', 'shopping_mall']),
            'device_used': np.random.choice(['mobile', 'web', 'pos', 'mobile_suspicious']),
        }
        
        # Add features based on missing level
        if missing_level == 0:  # Few missing features (80% complete)
            transaction.update({
                'time_since_last_transaction': np.random.randint(60, 7200),
                'spending_deviation_score': np.random.uniform(0, 2),
                'velocity_score': np.random.uniform(0, 1.5),
                'payment_channel': np.random.choice(['online', 'pos', 'mobile_app']),
                'tx_count_last_1h': np.random.randint(0, 3),
                'tx_count_last_24h': np.random.randint(1, 15),
                'total_amount_last_24h': transaction['amount'] * np.random.uniform(1, 5),
                'new_location_flag': np.random.choice([0, 1]),
                'new_device_flag': np.random.choice([0, 1]),
                'receiver_tx_count': np.random.randint(1, 50),
                'sender_receiver_frequency': np.random.randint(1, 10),
            })
            
        elif missing_level == 1:  # Some missing features (50% complete)
            transaction.update({
                'spending_deviation_score': np.random.uniform(0, 2),
                'tx_count_last_24h': np.random.randint(1, 15),
                'new_location_flag': np.random.choice([0, 1]),
                'receiver_tx_count': np.random.randint(1, 50),
            })
            
        # missing_level == 2: Many missing features (only basic data ~30% complete)
        # Only the base transaction data is included
        
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def create_minimal_feature_transactions(n_transactions: int = 4) -> pd.DataFrame:
    """
    Create transactions with only the absolute minimum required features.
    
    This tests the system's ability to work with very limited data.
    """
    np.random.seed(789)
    
    transactions = []
    base_time = datetime.now() - timedelta(hours=4)
    
    for i in range(n_transactions):
        # Only the most basic required fields
        transaction = {
            'transaction_id': f'minimal_tx_{i+1:03d}',
            'timestamp': base_time + timedelta(hours=i, minutes=np.random.randint(0, 60)),
            'sender_account': f'account_{np.random.randint(1000, 9999)}',
            'receiver_account': f'merchant_{np.random.randint(100, 999)}',
            'amount': round(np.random.uniform(10, 10000), 2),
            'transaction_type': 'purchase',  # Fixed value
            'merchant_category': 'unknown',  # Unknown category
            'location': 'unknown',  # Unknown location
            'device_used': 'unknown',  # Unknown device
        }
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def load_model():
    """Load the trained fraud detection model."""
    logger.info("Loading trained fraud detection model...")
    
    models_dir = Path("models/models")
    model_dirs = list(models_dir.glob("production_fraud_detector_v1_optimized/*"))
    
    if not model_dirs:
        raise ValueError("No model found")
    
    latest_dir = max(model_dirs, key=lambda x: x.name)
    model_file = latest_dir / "production_fraud_detector_v1_optimized.joblib"
    
    model_data = joblib.load(model_file)
    detector = FraudDetector()
    
    for key, value in model_data.items():
        if hasattr(detector, key):
            setattr(detector, key, value)
    
    logger.info("Model loaded successfully")
    return detector


def analyze_feature_impact(transactions_df: pd.DataFrame, predictions_df: pd.DataFrame, dataset_name: str):
    """Analyze how feature completeness affects predictions."""
    print(f"\n" + "="*100)
    print(f"FRAUD DETECTION ANALYSIS - {dataset_name.upper()}")
    print("="*100)
    
    results = transactions_df.merge(predictions_df, on='transaction_id', how='left')
    results = results.sort_values('fraud_probability', ascending=False)
    
    # Calculate feature completeness
    expected_features = [
        'fraud_type', 'time_since_last_transaction', 'spending_deviation_score', 
        'velocity_score', 'geo_anomaly_score', 'payment_channel', 'ip_address', 
        'device_hash', 'tx_count_last_1h', 'tx_count_last_24h', 'total_amount_last_24h',
        'avg_amount_last_24h', 'max_amount_last_24h', 'velocity_1h', 'velocity_24h',
        'receiver_tx_count', 'receiver_fraud_count', 'receiver_fraud_rate',
        'receiver_total_amount', 'receiver_avg_amount', 'new_location_flag',
        'new_device_flag', 'location_frequency', 'device_frequency',
        'sender_location_count', 'sender_device_count', 'sender_receiver_frequency',
        'sender_receiver_total_amount', 'sender_receiver_avg_amount'
    ]
    
    # Count available features for each transaction
    feature_completeness = []
    for idx, row in transactions_df.iterrows():
        available_features = sum(1 for feature in expected_features if feature in row and pd.notna(row[feature]))
        completeness_pct = (available_features / len(expected_features)) * 100
        feature_completeness.append(completeness_pct)
    
    results['feature_completeness'] = feature_completeness
    
    # Summary statistics
    avg_completeness = np.mean(feature_completeness)
    high_risk = len(results[results['risk_level'] == 'high'])
    medium_risk = len(results[results['risk_level'] == 'medium'])
    low_risk = len(results[results['risk_level'] == 'low'])
    fraud_flagged = results['fraud_prediction'].sum()
    avg_prob = results['fraud_probability'].mean()
    
    print(f"\n📊 FEATURE COMPLETENESS ANALYSIS:")
    print(f"Average Feature Completeness: {avg_completeness:.1f}%")
    print(f"Total Transactions: {len(results)}")
    print(f"Risk Distribution - High: {high_risk} | Medium: {medium_risk} | Low: {low_risk}")
    print(f"Flagged as Fraud: {fraud_flagged}")
    print(f"Average Fraud Probability: {avg_prob:.3f}")
    
    print(f"\n🔍 TRANSACTION DETAILS:")
    print("-" * 100)
    
    for idx, row in results.iterrows():
        risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(row['risk_level'], '⚪')
        fraud_status = "🚨 FRAUD" if row['fraud_prediction'] == 1 else "✅ LEGIT"
        
        print(f"\n{risk_emoji} {row['transaction_id']} | {fraud_status}")
        print(f"   💰 ${row['amount']:,.2f} | 🕐 {row['timestamp']} | 📊 Features: {row['feature_completeness']:.1f}%")
        print(f"   🏪 {row['merchant_category']} | 📍 {row['location']} | 📱 {row['device_used']}")
        print(f"   🎯 Fraud Prob: {row['fraud_probability']:.3f} | 🚨 Risk: {row['risk_level'].upper()} | 🔍 Anomaly: {row['anomaly_score']:.3f}")
        
        # Show which features are available
        available_features = [f for f in expected_features if f in transactions_df.columns and pd.notna(row.get(f))]
        if available_features:
            print(f"   ✅ Available Features: {', '.join(available_features[:5])}{'...' if len(available_features) > 5 else ''}")
        
        if 'explanation' in row and pd.notna(row['explanation']):
            print(f"   💬 {row['explanation']}")


def main():
    """Main demonstration function."""
    print("🔍 FRAUD DETECTION - FEATURE COMPLETENESS COMPARISON")
    print("=" * 70)
    
    try:
        # Load model once
        print("\n🤖 Loading trained model...")
        detector = load_model()
        
        # Test 1: Partial features (mixed completeness)
        print("\n📊 Test 1: Creating transactions with partial features...")
        partial_transactions = create_partial_feature_transactions(6)
        
        print("🎯 Generating predictions for partial feature set...")
        partial_predictions = detector.predict(
            partial_transactions,
            transaction_id_column='transaction_id',
            return_probabilities=True,
            return_risk_levels=True,
            return_explanations=True
        )
        
        analyze_feature_impact(partial_transactions, partial_predictions, "Partial Features")
        
        # Test 2: Minimal features
        print("\n📊 Test 2: Creating transactions with minimal features...")
        minimal_transactions = create_minimal_feature_transactions(4)
        
        print("🎯 Generating predictions for minimal feature set...")
        minimal_predictions = detector.predict(
            minimal_transactions,
            transaction_id_column='transaction_id',
            return_probabilities=True,
            return_risk_levels=True,
            return_explanations=True
        )
        
        analyze_feature_impact(minimal_transactions, minimal_predictions, "Minimal Features")
        
        # Summary comparison
        print(f"\n📈 FEATURE IMPACT SUMMARY:")
        print("=" * 70)
        print("This demonstration shows how the fraud detection system:")
        print("• Handles missing features gracefully with fallback values")
        print("• Maintains functionality even with minimal data")
        print("• Provides consistent risk assessment across feature completeness levels")
        print("• Uses error handling to prevent system failures")
        
        print(f"\n✅ Feature completeness demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)