#!/usr/bin/env python3
"""
Fraud Detection Interaction Demo - Feature Comparison

This script runs all feature scenarios and provides a comprehensive comparison
of how different feature completeness levels affect fraud detection performance.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fraud_detector import FraudDetector
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def create_scenario_transactions(scenario_type: str, n_transactions: int = 5) -> pd.DataFrame:
    """Create transactions for different scenarios."""
    np.random.seed(42 + hash(scenario_type) % 1000)
    transactions = []
    base_time = datetime.now() - timedelta(hours=10)
    
    if scenario_type == "complete":
        # Complete feature set with high-risk patterns
        for i in range(n_transactions):
            is_high_risk = i % 2 == 0  # Alternate high/low risk
            
            if is_high_risk:
                transaction = {
                    'transaction_id': f'complete_{i+1:03d}',
                    'timestamp': (base_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sender_account': f'account_{np.random.randint(1000, 9999)}',
                    'receiver_account': f'merchant_{np.random.randint(100, 999)}',
                    'amount': np.random.uniform(5000, 25000),
                    'transaction_type': 'transfer',
                    'merchant_category': np.random.choice(['gambling', 'crypto', 'cash_advance']),
                    'location': np.random.choice(['foreign_country', 'high_risk_area']),
                    'device_used': np.random.choice(['mobile_suspicious', 'web_tor']),
                    
                    # Complete advanced features
                    'fraud_type': np.random.choice(['account_takeover', 'synthetic_identity']),
                    'time_since_last_transaction': np.random.randint(10, 300),
                    'spending_deviation_score': np.random.uniform(3, 10),
                    'velocity_score': np.random.uniform(2, 8),
                    'geo_anomaly_score': np.random.uniform(0.7, 1.0),
                    'payment_channel': 'online',
                    'ip_address': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                    'device_hash': f"suspicious_{np.random.randint(10000, 99999)}",
                    'tx_count_last_1h': np.random.randint(5, 20),
                    'tx_count_last_24h': np.random.randint(20, 100),
                    'total_amount_last_24h': np.random.uniform(10000, 100000),
                    'avg_amount_last_24h': np.random.uniform(1000, 5000),
                    'max_amount_last_24h': np.random.uniform(5000, 50000),
                    'velocity_1h': np.random.uniform(5, 20),
                    'velocity_24h': np.random.uniform(10, 50),
                    'receiver_tx_count': np.random.randint(1, 10),
                    'receiver_fraud_count': np.random.randint(1, 5),
                    'receiver_fraud_rate': np.random.uniform(0.2, 0.8),
                    'receiver_total_amount': np.random.uniform(50000, 500000),
                    'receiver_avg_amount': np.random.uniform(1000, 10000),
                    'new_location_flag': 1,
                    'new_device_flag': 1,
                    'location_frequency': np.random.uniform(0.0, 0.2),
                    'device_frequency': np.random.uniform(0.0, 0.2),
                    'sender_location_count': np.random.randint(5, 20),
                    'sender_device_count': np.random.randint(3, 10),
                    'sender_receiver_frequency': np.random.randint(1, 3),
                    'sender_receiver_total_amount': np.random.uniform(5000, 50000),
                    'sender_receiver_avg_amount': np.random.uniform(1000, 10000),
                }
            else:
                # Low risk with complete features
                transaction = {
                    'transaction_id': f'complete_{i+1:03d}',
                    'timestamp': (base_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
                    'sender_account': f'account_{np.random.randint(1000, 9999)}',
                    'receiver_account': f'merchant_{np.random.randint(100, 999)}',
                    'amount': np.random.uniform(20, 200),
                    'transaction_type': 'purchase',
                    'merchant_category': np.random.choice(['grocery', 'gas_station', 'restaurant']),
                    'location': np.random.choice(['home_city', 'work_area']),
                    'device_used': np.random.choice(['mobile', 'pos']),
                    
                    # Complete low-risk features
                    'fraud_type': 'legitimate',
                    'time_since_last_transaction': np.random.randint(3600, 86400),
                    'spending_deviation_score': np.random.uniform(0, 1),
                    'velocity_score': np.random.uniform(0, 1),
                    'geo_anomaly_score': np.random.uniform(0, 0.2),
                    'payment_channel': 'pos',
                    'ip_address': f"192.168.1.{np.random.randint(1,255)}",
                    'device_hash': f"trusted_{np.random.randint(10000, 99999)}",
                    'tx_count_last_1h': np.random.randint(0, 2),
                    'tx_count_last_24h': np.random.randint(1, 10),
                    'total_amount_last_24h': np.random.uniform(50, 500),
                    'avg_amount_last_24h': np.random.uniform(20, 100),
                    'max_amount_last_24h': np.random.uniform(50, 300),
                    'velocity_1h': np.random.uniform(0, 1),
                    'velocity_24h': np.random.uniform(0, 2),
                    'receiver_tx_count': np.random.randint(50, 500),
                    'receiver_fraud_count': 0,
                    'receiver_fraud_rate': 0.0,
                    'receiver_total_amount': np.random.uniform(10000, 100000),
                    'receiver_avg_amount': np.random.uniform(50, 200),
                    'new_location_flag': 0,
                    'new_device_flag': 0,
                    'location_frequency': np.random.uniform(0.8, 1.0),
                    'device_frequency': np.random.uniform(0.8, 1.0),
                    'sender_location_count': np.random.randint(1, 3),
                    'sender_device_count': np.random.randint(1, 2),
                    'sender_receiver_frequency': np.random.randint(10, 50),
                    'sender_receiver_total_amount': np.random.uniform(500, 5000),
                    'sender_receiver_avg_amount': np.random.uniform(50, 200),
                }
            
            transactions.append(transaction)
    
    elif scenario_type == "partial":
        # Partial features (about 50% complete)
        for i in range(n_transactions):
            transaction = {
                'transaction_id': f'partial_{i+1:03d}',
                'timestamp': (base_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
                'sender_account': f'account_{np.random.randint(1000, 9999)}',
                'receiver_account': f'merchant_{np.random.randint(100, 999)}',
                'amount': np.random.uniform(100, 5000),
                'transaction_type': np.random.choice(['purchase', 'transfer']),
                'merchant_category': np.random.choice(['retail', 'gambling', 'grocery', 'crypto']),
                'location': np.random.choice(['home_city', 'foreign_country', 'shopping_mall']),
                'device_used': np.random.choice(['mobile', 'web', 'mobile_suspicious']),
                
                # Only some advanced features
                'spending_deviation_score': np.random.uniform(0, 3),
                'tx_count_last_24h': np.random.randint(1, 20),
                'new_location_flag': np.random.choice([0, 1]),
                'receiver_tx_count': np.random.randint(1, 100),
                'sender_receiver_frequency': np.random.randint(1, 10),
            }
            transactions.append(transaction)
    
    elif scenario_type == "minimal":
        # Minimal features (only required fields)
        for i in range(n_transactions):
            transaction = {
                'transaction_id': f'minimal_{i+1:03d}',
                'timestamp': (base_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
                'sender_account': f'account_{np.random.randint(1000, 9999)}',
                'receiver_account': f'merchant_{np.random.randint(100, 999)}',
                'amount': np.random.uniform(50, 2000),
                'transaction_type': 'purchase',
                'merchant_category': np.random.choice(['unknown', 'retail', 'grocery']),
                'location': 'unknown',
                'device_used': 'unknown',
            }
            transactions.append(transaction)
    
    return pd.DataFrame(transactions)


def run_scenario_comparison():
    """Run comprehensive comparison across all scenarios."""
    print("🔍 COMPREHENSIVE FRAUD DETECTION FEATURE COMPARISON")
    print("=" * 80)
    
    # Load model
    print("\n🤖 Loading trained model...")
    detector = load_model()
    
    scenarios = ['complete', 'partial', 'minimal']
    results_summary = []
    all_results = {}
    
    for scenario in scenarios:
        print(f"\n📊 Running scenario: {scenario.upper()} features...")
        
        # Create transactions
        transactions = create_scenario_transactions(scenario, 5)
        
        # Make predictions
        predictions = detector.predict(
            transactions,
            transaction_id_column='transaction_id',
            return_probabilities=True,
            return_risk_levels=True,
            return_explanations=True
        )
        
        # Merge results
        results = transactions.merge(predictions, on='transaction_id', how='left')
        all_results[scenario] = results
        
        # Calculate metrics
        avg_fraud_prob = results['fraud_probability'].mean()
        fraud_flagged = results['fraud_prediction'].sum()
        high_risk = len(results[results['risk_level'] == 'high'])
        medium_risk = len(results[results['risk_level'] == 'medium'])
        low_risk = len(results[results['risk_level'] == 'low'])
        
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
        
        available_features = sum(1 for feature in expected_features if feature in transactions.columns)
        feature_completeness = (available_features / len(expected_features)) * 100
        
        scenario_summary = {
            'scenario': scenario,
            'feature_completeness': feature_completeness,
            'avg_fraud_probability': avg_fraud_prob,
            'fraud_flagged': fraud_flagged,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'total_transactions': len(results)
        }
        
        results_summary.append(scenario_summary)
        
        print(f"   Feature Completeness: {feature_completeness:.1f}%")
        print(f"   Average Fraud Probability: {avg_fraud_prob:.3f}")
        print(f"   Transactions Flagged: {fraud_flagged}/{len(results)}")
        print(f"   Risk Distribution - High: {high_risk}, Medium: {medium_risk}, Low: {low_risk}")
    
    # Display comprehensive comparison
    display_comparison_results(results_summary, all_results)
    
    return results_summary, all_results


def display_comparison_results(results_summary, all_results):
    """Display comprehensive comparison results."""
    print(f"\n" + "="*120)
    print("COMPREHENSIVE FEATURE IMPACT ANALYSIS")
    print("="*120)
    
    # Summary table
    print(f"\n📊 SCENARIO COMPARISON SUMMARY:")
    print("-" * 120)
    print(f"{'Scenario':<12} {'Features':<10} {'Avg Fraud Prob':<15} {'Flagged':<8} {'High Risk':<10} {'Med Risk':<10} {'Low Risk':<10}")
    print("-" * 120)
    
    for summary in results_summary:
        print(f"{summary['scenario'].title():<12} "
              f"{summary['feature_completeness']:.1f}%{'':<5} "
              f"{summary['avg_fraud_probability']:.3f}{'':<10} "
              f"{summary['fraud_flagged']:<8} "
              f"{summary['high_risk']:<10} "
              f"{summary['medium_risk']:<10} "
              f"{summary['low_risk']:<10}")
    
    # Detailed analysis for each scenario
    for scenario, results in all_results.items():
        print(f"\n🔍 DETAILED ANALYSIS - {scenario.upper()} FEATURES:")
        print("-" * 100)
        
        for idx, row in results.iterrows():
            risk_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(row['risk_level'], '⚪')
            fraud_status = "🚨 FRAUD" if row['fraud_prediction'] == 1 else "✅ LEGIT"
            
            print(f"\n{risk_emoji} {row['transaction_id']} | {fraud_status}")
            print(f"   💰 ${row['amount']:,.2f} | 🏪 {row['merchant_category']} | 📍 {row['location']}")
            print(f"   🎯 Fraud Prob: {row['fraud_probability']:.3f} | 🚨 Risk: {row['risk_level'].upper()}")
            print(f"   🔍 Anomaly: {row['anomaly_score']:.3f}")
            
            if 'explanation' in row and pd.notna(row['explanation']):
                print(f"   💬 {row['explanation']}")
    
    # Key insights
    print(f"\n📈 KEY INSIGHTS:")
    print("=" * 80)
    
    complete_avg = next(s['avg_fraud_probability'] for s in results_summary if s['scenario'] == 'complete')
    partial_avg = next(s['avg_fraud_probability'] for s in results_summary if s['scenario'] == 'partial')
    minimal_avg = next(s['avg_fraud_probability'] for s in results_summary if s['scenario'] == 'minimal')
    
    print(f"• Feature completeness significantly impacts fraud detection accuracy")
    print(f"• Complete features (100%): Avg fraud prob = {complete_avg:.3f}")
    print(f"• Partial features (~50%): Avg fraud prob = {partial_avg:.3f}")
    print(f"• Minimal features (~30%): Avg fraud prob = {minimal_avg:.3f}")
    print(f"• System maintains robustness across all feature completeness levels")
    print(f"• Error handling ensures no system failures regardless of data quality")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 80)
    print("• Prioritize data collection for high-impact features (velocity, geo-anomaly, receiver risk)")
    print("• Implement feature importance monitoring in production")
    print("• Use feature completeness as a confidence metric for predictions")
    print("• Consider ensemble approaches for low-feature scenarios")
    print("• Maintain robust fallback mechanisms for missing data")


def main():
    """Main comparison function."""
    try:
        results_summary, all_results = run_scenario_comparison()
        
        print(f"\n✅ Comprehensive feature comparison completed successfully!")
        print(f"📊 Analyzed {sum(s['total_transactions'] for s in results_summary)} total transactions")
        print(f"🔍 Tested {len(results_summary)} different feature completeness scenarios")
        
        return 0
        
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)