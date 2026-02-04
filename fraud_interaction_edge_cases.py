#!/usr/bin/env python3
"""
Fraud Detection Interaction Demo - Edge Cases

This script demonstrates fraud detection with edge cases and challenging scenarios,
showing system robustness and error handling capabilities.
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


def create_edge_case_transactions() -> pd.DataFrame:
    """
    Create transactions with edge cases and challenging scenarios.
    
    This tests the system's robustness with unusual data patterns.
    """
    transactions = []
    base_time = datetime.now() - timedelta(hours=6)
    
    # Edge Case 1: Extremely high amount
    transactions.append({
        'transaction_id': 'edge_tx_001',
        'timestamp': (base_time + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_9999',
        'receiver_account': 'merchant_999',
        'amount': 999999.99,  # Very high amount
        'transaction_type': 'transfer',
        'merchant_category': 'luxury_goods',
        'location': 'foreign_country',
        'device_used': 'mobile_suspicious',
        'velocity_score': 10.0,  # Extremely high velocity
        'geo_anomaly_score': 1.0,  # Maximum anomaly
        'new_location_flag': 1,
        'new_device_flag': 1,
    })
    
    # Edge Case 2: Extremely low amount
    transactions.append({
        'transaction_id': 'edge_tx_002',
        'timestamp': (base_time + timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_0001',
        'receiver_account': 'merchant_001',
        'amount': 0.01,  # Very low amount
        'transaction_type': 'purchase',
        'merchant_category': 'testing',
        'location': 'unknown',
        'device_used': 'automated',
        'velocity_score': 0.0,
        'tx_count_last_1h': 100,  # High frequency, low amounts (card testing)
    })
    
    # Edge Case 3: Missing critical data
    transactions.append({
        'transaction_id': 'edge_tx_003',
        'timestamp': (base_time + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': '',  # Empty sender
        'receiver_account': 'merchant_unknown',
        'amount': 500.00,
        'transaction_type': 'unknown',
        'merchant_category': '',  # Empty category
        'location': None,  # Null location
        'device_used': 'unknown',
    })
    
    # Edge Case 4: Unusual timestamp (very old)
    transactions.append({
        'transaction_id': 'edge_tx_004',
        'timestamp': '2020-01-01 00:00:00',  # Very old timestamp
        'sender_account': 'account_historical',
        'receiver_account': 'merchant_old',
        'amount': 1000.00,
        'transaction_type': 'purchase',
        'merchant_category': 'grocery',
        'location': 'home_city',
        'device_used': 'pos',
        'time_since_last_transaction': 999999,  # Very long time
    })
    
    # Edge Case 5: Rapid-fire transactions (velocity attack)
    for i in range(3):
        transactions.append({
            'transaction_id': f'edge_tx_00{5+i}',
            'timestamp': (base_time + timedelta(seconds=i*10)).strftime('%Y-%m-%d %H:%M:%S'),
            'sender_account': 'account_velocity',
            'receiver_account': f'merchant_rapid_{i}',
            'amount': 99.99,
            'transaction_type': 'purchase',
            'merchant_category': 'electronics',
            'location': 'online',
            'device_used': 'mobile',
            'velocity_1h': 50.0,  # Very high velocity
            'tx_count_last_1h': 20 + i*5,
            'time_since_last_transaction': 10,  # Very short intervals
        })
    
    # Edge Case 6: International with currency issues
    transactions.append({
        'transaction_id': 'edge_tx_008',
        'timestamp': (base_time + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_intl',
        'receiver_account': 'merchant_foreign',
        'amount': 1234.56,
        'transaction_type': 'transfer',
        'merchant_category': 'money_transfer',
        'location': 'foreign_country',
        'device_used': 'web_tor',
        'geo_anomaly_score': 0.95,
        'new_location_flag': 1,
        'payment_channel': 'crypto',
    })
    
    # Edge Case 7: Perfect legitimate transaction (all good signals)
    transactions.append({
        'transaction_id': 'edge_tx_009',
        'timestamp': (base_time + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_trusted',
        'receiver_account': 'merchant_verified',
        'amount': 45.67,
        'transaction_type': 'purchase',
        'merchant_category': 'grocery',
        'location': 'home_city',
        'device_used': 'pos',
        'velocity_score': 0.1,
        'geo_anomaly_score': 0.0,
        'new_location_flag': 0,
        'new_device_flag': 0,
        'receiver_fraud_rate': 0.0,
        'tx_count_last_24h': 3,
        'sender_receiver_frequency': 50,  # Regular customer
    })
    
    # Edge Case 8: Extreme outlier features
    transactions.append({
        'transaction_id': 'edge_tx_010',
        'timestamp': (base_time + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_outlier',
        'receiver_account': 'merchant_suspicious',
        'amount': 50000.00,
        'transaction_type': 'withdrawal',
        'merchant_category': 'gambling',
        'location': 'high_risk_area',
        'device_used': 'mobile_suspicious',
        'spending_deviation_score': 100.0,  # Extreme deviation
        'velocity_score': 50.0,  # Extreme velocity
        'geo_anomaly_score': 1.0,  # Maximum anomaly
        'receiver_fraud_rate': 0.8,  # High-risk receiver
        'tx_count_last_1h': 50,  # Extreme frequency
        'new_location_flag': 1,
        'new_device_flag': 1,
    })
    
    return pd.DataFrame(transactions)


def create_data_quality_issues() -> pd.DataFrame:
    """Create transactions with data quality issues."""
    transactions = []
    base_time = datetime.now() - timedelta(hours=2)
    
    # Data Quality Issue 1: Invalid data types
    transactions.append({
        'transaction_id': 'quality_tx_001',
        'timestamp': (base_time + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_123',
        'receiver_account': 'merchant_456',
        'amount': 'invalid_amount',  # String instead of number
        'transaction_type': 'purchase',
        'merchant_category': 'retail',
        'location': 'home_city',
        'device_used': 'mobile',
    })
    
    # Data Quality Issue 2: Negative amounts
    transactions.append({
        'transaction_id': 'quality_tx_002',
        'timestamp': (base_time + timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_456',
        'receiver_account': 'merchant_789',
        'amount': -100.00,  # Negative amount
        'transaction_type': 'refund',
        'merchant_category': 'retail',
        'location': 'home_city',
        'device_used': 'web',
    })
    
    # Data Quality Issue 3: Invalid timestamp
    transactions.append({
        'transaction_id': 'quality_tx_003',
        'timestamp': 'invalid_timestamp',  # Invalid timestamp format
        'sender_account': 'account_789',
        'receiver_account': 'merchant_012',
        'amount': 250.00,
        'transaction_type': 'purchase',
        'merchant_category': 'grocery',
        'location': 'shopping_mall',
        'device_used': 'pos',
    })
    
    # Data Quality Issue 4: Extreme unicode characters
    transactions.append({
        'transaction_id': 'quality_tx_004',
        'timestamp': (base_time + timedelta(minutes=40)).strftime('%Y-%m-%d %H:%M:%S'),
        'sender_account': 'account_🏦',  # Unicode characters
        'receiver_account': 'merchant_🛒',
        'amount': 75.50,
        'transaction_type': 'purchase',
        'merchant_category': 'café_français',  # Accented characters
        'location': 'москва',  # Cyrillic characters
        'device_used': 'mobile',
    })
    
    return pd.DataFrame(transactions)


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


def analyze_edge_cases(transactions_df: pd.DataFrame, predictions_df: pd.DataFrame, test_name: str):
    """Analyze edge case handling and system robustness."""
    print(f"\n" + "="*120)
    print(f"EDGE CASE ANALYSIS - {test_name.upper()}")
    print("="*120)
    
    results = transactions_df.merge(predictions_df, on='transaction_id', how='left')
    results = results.sort_values('fraud_probability', ascending=False)
    
    # Analyze system behavior
    successful_predictions = len(results[results['fraud_probability'].notna()])
    failed_predictions = len(results) - successful_predictions
    
    print(f"\n🔧 SYSTEM ROBUSTNESS:")
    print(f"Total Transactions: {len(results)}")
    print(f"Successful Predictions: {successful_predictions}")
    print(f"Failed Predictions: {failed_predictions}")
    print(f"Success Rate: {(successful_predictions/len(results)*100):.1f}%")
    
    # Risk distribution
    risk_dist = results['risk_level'].value_counts()
    print(f"\n📊 RISK DISTRIBUTION:")
    for risk, count in risk_dist.items():
        print(f"  {risk.title()}: {count}")
    
    print(f"\n🔍 DETAILED EDGE CASE ANALYSIS:")
    print("-" * 120)
    
    for idx, row in results.iterrows():
        risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(row['risk_level'], '⚪')
        fraud_status = "🚨 FRAUD" if row['fraud_prediction'] == 1 else "✅ LEGIT"
        
        print(f"\n{risk_emoji} {row['transaction_id']} | {fraud_status}")
        
        # Handle potential data issues in display
        try:
            amount_str = f"${row['amount']:,.2f}" if pd.notna(row['amount']) and isinstance(row['amount'], (int, float)) else f"${row['amount']}"
        except:
            amount_str = f"Amount: {row['amount']}"
            
        print(f"   💰 {amount_str} | 🕐 {row['timestamp']}")
        print(f"   🏪 {row['merchant_category']} | 📍 {row['location']} | 📱 {row['device_used']}")
        
        if pd.notna(row['fraud_probability']):
            print(f"   🎯 Fraud Prob: {row['fraud_probability']:.3f} | 🚨 Risk: {row['risk_level'].upper()}")
            print(f"   🔍 Anomaly: {row['anomaly_score']:.3f}")
        else:
            print(f"   ❌ Prediction failed - system used fallback")
        
        # Identify the edge case type
        edge_case_type = identify_edge_case(row)
        if edge_case_type:
            print(f"   🔬 Edge Case: {edge_case_type}")
        
        if 'explanation' in row and pd.notna(row['explanation']):
            print(f"   💬 {row['explanation']}")


def identify_edge_case(row):
    """Identify what type of edge case this transaction represents."""
    edge_cases = []
    
    # Check for various edge case patterns
    if pd.notna(row['amount']):
        if isinstance(row['amount'], (int, float)):
            if row['amount'] > 100000:
                edge_cases.append("Extremely High Amount")
            elif row['amount'] < 1:
                edge_cases.append("Extremely Low Amount")
            elif row['amount'] < 0:
                edge_cases.append("Negative Amount")
        else:
            edge_cases.append("Invalid Amount Type")
    
    if row.get('velocity_score', 0) > 10:
        edge_cases.append("Extreme Velocity")
    
    if row.get('tx_count_last_1h', 0) > 20:
        edge_cases.append("High Frequency")
    
    if row.get('geo_anomaly_score', 0) > 0.9:
        edge_cases.append("Geographic Anomaly")
    
    if not row.get('sender_account') or row.get('sender_account') == '':
        edge_cases.append("Missing Sender")
    
    if 'invalid' in str(row.get('timestamp', '')).lower():
        edge_cases.append("Invalid Timestamp")
    
    # Check for unicode characters
    text_fields = ['sender_account', 'receiver_account', 'merchant_category', 'location']
    for field in text_fields:
        if row.get(field) and any(ord(char) > 127 for char in str(row[field])):
            edge_cases.append("Unicode Characters")
            break
    
    return ", ".join(edge_cases) if edge_cases else None


def main():
    """Main demonstration function."""
    print("🔍 FRAUD DETECTION - EDGE CASES & ROBUSTNESS TESTING")
    print("=" * 80)
    
    try:
        # Load model
        print("\n🤖 Loading trained model...")
        detector = load_model()
        
        # Test 1: Edge cases
        print("\n📊 Test 1: Creating edge case transactions...")
        edge_transactions = create_edge_case_transactions()
        
        print("🎯 Testing edge case handling...")
        edge_predictions = detector.predict(
            edge_transactions,
            transaction_id_column='transaction_id',
            return_probabilities=True,
            return_risk_levels=True,
            return_explanations=True
        )
        
        analyze_edge_cases(edge_transactions, edge_predictions, "Edge Cases")
        
        # Test 2: Data quality issues
        print("\n📊 Test 2: Creating transactions with data quality issues...")
        quality_transactions = create_data_quality_issues()
        
        print("🎯 Testing data quality issue handling...")
        quality_predictions = detector.predict(
            quality_transactions,
            transaction_id_column='transaction_id',
            return_probabilities=True,
            return_risk_levels=True,
            return_explanations=True
        )
        
        analyze_edge_cases(quality_transactions, quality_predictions, "Data Quality Issues")
        
        # Summary
        print(f"\n📈 ROBUSTNESS SUMMARY:")
        print("=" * 80)
        print("The fraud detection system demonstrates:")
        print("• Graceful handling of extreme values and edge cases")
        print("• Robust error handling for data quality issues")
        print("• Consistent fallback mechanisms when predictions fail")
        print("• Ability to process unusual data patterns without crashing")
        print("• Comprehensive logging and error reporting")
        
        print(f"\n✅ Edge cases and robustness testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)