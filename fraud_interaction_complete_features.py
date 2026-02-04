#!/usr/bin/env python3
"""
Fraud Detection Interaction Demo - Complete Features

This script demonstrates fraud detection with complete feature sets,
showing how the system performs when all expected features are present.
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


def create_complete_feature_transactions(n_transactions: int = 8) -> pd.DataFrame:
    """
    Create transactions with complete feature sets that match training data.
    
    This includes all the features the model was trained on to show
    optimal performance.
    """
    np.random.seed(123)  # Different seed for variety
    
    transactions = []
    base_time = datetime.now() - timedelta(hours=12)
    
    # Define realistic feature ranges based on training data
    fraud_scenarios = [
        # High-risk fraud patterns
        {
            'is_fraud_likely': True,
            'amount_range': (5000, 50000),
            'hours': [2, 3, 23, 24],
            'locations': ['foreign_country', 'high_risk_area', 'unknown_location'],
            'devices': ['mobile_suspicious', 'web_tor', 'unknown_device'],
            'merchants': ['gambling', 'crypto', 'cash_advance', 'adult_entertainment'],
            'fraud_type': ['account_takeover', 'synthetic_identity', 'card_testing'],
            'velocity_multiplier': 5.0,
            'geo_anomaly': 0.8
        },
        # Medium-risk patterns
        {
            'is_fraud_likely': False,
            'amount_range': (500, 2000),
            'hours': [20, 21, 22, 7, 8],
            'locations': ['new_city', 'airport', 'hotel'],
            'devices': ['mobile_new', 'web_public'],
            'merchants': ['electronics', 'jewelry', 'luxury_goods'],
            'fraud_type': ['legitimate'],
            'velocity_multiplier': 2.0,
            'geo_anomaly': 0.3
        },
        # Low-risk normal patterns
        {
            'is_fraud_likely': False,
            'amount_range': (10, 300),
            'hours': list(range(9, 18)),
            'locations': ['home_city', 'work_area', 'shopping_mall'],
            'devices': ['mobile', 'web', 'pos'],
            'merchants': ['grocery', 'gas_station', 'restaurant', 'retail'],
            'fraud_type': ['legitimate'],
            'velocity_multiplier': 1.0,
            'geo_anomaly': 0.1
        }
    ]
    
    for i in range(n_transactions):
        # Choose scenario based on index to ensure variety
        scenario_idx = i % len(fraud_scenarios)
        scenario = fraud_scenarios[scenario_idx]
        
        # Generate transaction with complete features
        amount = np.random.uniform(*scenario['amount_range'])
        hour = np.random.choice(scenario['hours'])
        location = np.random.choice(scenario['locations'])
        device = np.random.choice(scenario['devices'])
        merchant = np.random.choice(scenario['merchants'])
        fraud_type = np.random.choice(scenario['fraud_type'])
        
        # Generate comprehensive features
        transaction = {
            # Basic transaction info
            'transaction_id': f'complete_tx_{i+1:03d}',
            'timestamp': base_time + timedelta(hours=i*2, minutes=np.random.randint(0, 60)),
            'sender_account': f'account_{np.random.randint(1000, 9999)}',
            'receiver_account': f'merchant_{np.random.randint(100, 999)}',
            'amount': round(amount, 2),
            'transaction_type': np.random.choice(['purchase', 'transfer', 'withdrawal']),
            'merchant_category': merchant,
            'location': location,
            'device_used': device,
            
            # Advanced features that the model expects
            'fraud_type': fraud_type,
            'time_since_last_transaction': np.random.randint(10, 3600),  # seconds
            'spending_deviation_score': np.random.uniform(0, scenario['velocity_multiplier']),
            'velocity_score': np.random.uniform(0, scenario['velocity_multiplier']),
            'geo_anomaly_score': scenario['geo_anomaly'] + np.random.uniform(-0.1, 0.1),
            'payment_channel': np.random.choice(['online', 'pos', 'atm', 'mobile_app']),
            'ip_address': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
            'device_hash': f"device_{np.random.randint(10000, 99999)}",
            
            # Behavioral features
            'tx_count_last_1h': np.random.randint(0, int(5 * scenario['velocity_multiplier'])),
            'tx_count_last_24h': np.random.randint(1, int(20 * scenario['velocity_multiplier'])),
            'total_amount_last_24h': amount * np.random.uniform(1, scenario['velocity_multiplier']),
            'avg_amount_last_24h': amount * np.random.uniform(0.5, 1.5),
            'max_amount_last_24h': amount * np.random.uniform(1, 2),
            'velocity_1h': np.random.uniform(0, scenario['velocity_multiplier']),
            'velocity_24h': np.random.uniform(0, scenario['velocity_multiplier']),
            
            # Receiver risk features
            'receiver_tx_count': np.random.randint(1, 100),
            'receiver_fraud_count': np.random.randint(0, 5) if scenario['is_fraud_likely'] else 0,
            'receiver_fraud_rate': np.random.uniform(0.1, 0.3) if scenario['is_fraud_likely'] else np.random.uniform(0, 0.05),
            'receiver_total_amount': amount * np.random.uniform(10, 100),
            'receiver_avg_amount': amount * np.random.uniform(0.8, 1.2),
            
            # Location and device features
            'new_location_flag': 1 if 'new' in location or 'foreign' in location else 0,
            'new_device_flag': 1 if 'new' in device or 'suspicious' in device else 0,
            'location_frequency': np.random.uniform(0.1, 1.0),
            'device_frequency': np.random.uniform(0.1, 1.0),
            'sender_location_count': np.random.randint(1, 10),
            'sender_device_count': np.random.randint(1, 5),
            
            # Interaction features
            'sender_receiver_frequency': np.random.randint(1, 20),
            'sender_receiver_total_amount': amount * np.random.uniform(1, 10),
            'sender_receiver_avg_amount': amount * np.random.uniform(0.8, 1.2),
        }
        
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    
    # Convert timestamp to string format
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df


def load_model():
    """Load the trained fraud detection model."""
    logger.info("Loading trained fraud detection model...")
    
    # Find the model file
    models_dir = Path("models/models")
    model_dirs = list(models_dir.glob("production_fraud_detector_v1_optimized/*"))
    
    if not model_dirs:
        raise ValueError("No model found")
    
    latest_dir = max(model_dirs, key=lambda x: x.name)
    model_file = latest_dir / "production_fraud_detector_v1_optimized.joblib"
    
    # Load and restore model
    model_data = joblib.load(model_file)
    detector = FraudDetector()
    
    # Restore state
    for key, value in model_data.items():
        if hasattr(detector, key):
            setattr(detector, key, value)
    
    logger.info("Model loaded successfully")
    return detector


def analyze_predictions(transactions_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """Analyze and display detailed prediction results."""
    print("\n" + "="*120)
    print("FRAUD DETECTION ANALYSIS - COMPLETE FEATURES")
    print("="*120)
    
    # Merge data
    results = transactions_df.merge(predictions_df, on='transaction_id', how='left')
    results = results.sort_values('fraud_probability', ascending=False)
    
    # Summary statistics
    high_risk = len(results[results['risk_level'] == 'high'])
    medium_risk = len(results[results['risk_level'] == 'medium'])
    low_risk = len(results[results['risk_level'] == 'low'])
    fraud_flagged = results['fraud_prediction'].sum()
    avg_prob = results['fraud_probability'].mean()
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Total Transactions: {len(results)}")
    print(f"High Risk: {high_risk} | Medium Risk: {medium_risk} | Low Risk: {low_risk}")
    print(f"Flagged as Fraud: {fraud_flagged}")
    print(f"Average Fraud Probability: {avg_prob:.3f}")
    
    print(f"\n🔍 DETAILED ANALYSIS:")
    print("-" * 120)
    
    for idx, row in results.iterrows():
        risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(row['risk_level'], '⚪')
        fraud_status = "🚨 FRAUD" if row['fraud_prediction'] == 1 else "✅ LEGIT"
        
        print(f"\n{risk_emoji} {row['transaction_id']} | {fraud_status}")
        print(f"   💰 Amount: ${row['amount']:,.2f} | 🕐 Time: {row['timestamp']}")
        print(f"   🏪 {row['merchant_category']} | 📍 {row['location']} | 📱 {row['device_used']}")
        print(f"   🎯 Fraud Prob: {row['fraud_probability']:.3f} | 🚨 Risk: {row['risk_level'].upper()}")
        print(f"   🔍 Anomaly: {row['anomaly_score']:.3f}")
        
        # Show key features that influenced the decision
        print(f"   📈 Key Features:")
        print(f"      • Velocity (1h): {row.get('velocity_1h', 'N/A')}")
        print(f"      • Geo Anomaly: {row.get('geo_anomaly_score', 'N/A')}")
        print(f"      • New Location: {'Yes' if row.get('new_location_flag', 0) else 'No'}")
        print(f"      • Receiver Risk: {row.get('receiver_fraud_rate', 'N/A')}")
        
        if 'explanation' in row and pd.notna(row['explanation']):
            print(f"   💬 {row['explanation']}")


def main():
    """Main demonstration function."""
    print("🔍 FRAUD DETECTION - COMPLETE FEATURES DEMO")
    print("=" * 60)
    
    try:
        # Create transactions with complete features
        print("\n📊 Creating transactions with complete feature sets...")
        transactions = create_complete_feature_transactions(8)
        
        print(f"Created {len(transactions)} transactions with complete features:")
        for idx, row in transactions.iterrows():
            risk_indicator = "🔴" if any(x in row['location'] for x in ['foreign', 'high_risk']) else "🟢"
            print(f"  {risk_indicator} {row['transaction_id']}: ${row['amount']:,.2f} at {row['merchant_category']} ({row['location']})")
        
        # Load model
        print("\n🤖 Loading trained model...")
        detector = load_model()
        
        # Make predictions
        print("\n🎯 Generating fraud predictions...")
        predictions = detector.predict(
            transactions,
            transaction_id_column='transaction_id',
            return_probabilities=True,
            return_risk_levels=True,
            return_explanations=True
        )
        
        # Analyze results
        analyze_predictions(transactions, predictions)
        
        print(f"\n✅ Complete features demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)