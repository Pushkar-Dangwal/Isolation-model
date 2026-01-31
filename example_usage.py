#!/usr/bin/env python3
"""
Example usage of the fraud detection training and inference scripts.

This script demonstrates how to:
1. Generate sample data for testing
2. Train a fraud detection model
3. Run inference on new transactions
4. Interpret the results
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000, fraud_rate=0.04, save_path="sample_transactions.csv"):
    """Generate realistic sample transaction data for testing."""
    print(f"Generating {n_samples} sample transactions with {fraud_rate:.1%} fraud rate...")
    
    np.random.seed(42)
    
    # Generate transaction IDs
    transaction_ids = [f'tx_{i:06d}' for i in range(n_samples)]
    
    # Generate timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=np.random.randint(0, 30*24*3600)
    ) for _ in range(n_samples)]
    
    # Generate account IDs
    n_senders = max(100, n_samples // 20)  # Realistic sender distribution
    n_receivers = max(200, n_samples // 10)  # More receivers than senders
    
    sender_accounts = [f'sender_{i % n_senders:04d}' for i in range(n_samples)]
    receiver_accounts = [f'receiver_{i % n_receivers:04d}' for i in range(n_samples)]
    
    # Generate transaction amounts (log-normal distribution)
    amounts = np.random.lognormal(mean=4.0, sigma=1.5, size=n_samples)
    amounts = np.round(amounts, 2)
    
    # Generate categorical features
    transaction_types = np.random.choice(
        ['transfer', 'payment', 'withdrawal', 'deposit'], 
        n_samples, 
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    merchant_categories = np.random.choice(
        ['grocery', 'gas', 'restaurant', 'online', 'retail', 'entertainment'],
        n_samples,
        p=[0.2, 0.15, 0.2, 0.25, 0.15, 0.05]
    )
    
    locations = np.random.choice(
        ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego'],
        n_samples,
        p=[0.2, 0.15, 0.12, 0.1, 0.08, 0.08, 0.07, 0.2]  # Last is "other"
    )
    
    devices = np.random.choice(
        ['mobile', 'web', 'atm', 'pos'],
        n_samples,
        p=[0.45, 0.25, 0.15, 0.15]
    )
    
    # Generate fraud labels
    is_fraud = np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
    
    # Make fraud cases more realistic (higher amounts, specific patterns)
    fraud_mask = is_fraud == 1
    if np.any(fraud_mask):
        # Fraudulent transactions tend to be higher amounts
        amounts[fraud_mask] *= np.random.uniform(2, 5, np.sum(fraud_mask))
        
        # Fraudulent transactions more likely at certain times
        night_hours = np.random.choice([22, 23, 0, 1, 2, 3], np.sum(fraud_mask))
        for i, (idx, hour) in enumerate(zip(np.where(fraud_mask)[0], night_hours)):
            timestamps[idx] = timestamps[idx].replace(hour=hour)
    
    # Create DataFrame
    data = {
        'transaction_id': transaction_ids,
        'timestamp': timestamps,
        'sender_account': sender_accounts,
        'receiver_account': receiver_accounts,
        'amount': amounts,
        'transaction_type': transaction_types,
        'merchant_category': merchant_categories,
        'location': locations,
        'device_used': devices,
        'is_fraud': is_fraud
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Sample data saved to: {save_path}")
    print(f"Data summary:")
    print(f"  - Total transactions: {len(df)}")
    print(f"  - Fraud transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.1%})")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():.2f}")
    
    return df


def demonstrate_training():
    """Demonstrate model training."""
    print("\n" + "="*60)
    print("TRAINING DEMONSTRATION")
    print("="*60)
    
    # Generate training data
    training_data = generate_sample_data(
        n_samples=5000, 
        fraud_rate=0.04, 
        save_path="training_data.csv"
    )
    
    print("\nTo train the model, run:")
    print("python train_model.py \\")
    print("  --data-path training_data.csv \\")
    print("  --model-name demo_fraud_model \\")
    print("  --enable-reproducibility \\")
    print("  --optimize-thresholds \\")
    print("  --save-evaluation \\")
    print("  --verbose")
    
    print("\nThis will:")
    print("  1. Load and validate the training data")
    print("  2. Train the complete fraud detection pipeline")
    print("  3. Evaluate model performance")
    print("  4. Save the trained model with metadata")
    print("  5. Generate comprehensive evaluation reports")


def demonstrate_inference():
    """Demonstrate model inference."""
    print("\n" + "="*60)
    print("INFERENCE DEMONSTRATION")
    print("="*60)
    
    # Generate inference data (without fraud labels)
    inference_data = generate_sample_data(
        n_samples=100, 
        fraud_rate=0.04, 
        save_path="inference_data.csv"
    )
    
    # Remove fraud labels for realistic inference scenario
    inference_data_clean = inference_data.drop(columns=['is_fraud'])
    inference_data_clean.to_csv("inference_data_clean.csv", index=False)
    
    print("\nGenerated inference data (without fraud labels)")
    
    print("\nTo run batch inference, use:")
    print("python inference.py \\")
    print("  --model-path models/demo_fraud_model \\")
    print("  --input-data inference_data_clean.csv \\")
    print("  --output-file predictions.csv \\")
    print("  --output-format both \\")
    print("  --enable-explanations \\")
    print("  --verbose")
    
    print("\nTo run real-time inference, use:")
    print("python inference.py \\")
    print("  --model-path models/demo_fraud_model \\")
    print("  --input-data inference_data_clean.csv \\")
    print("  --real-time \\")
    print("  --benchmark \\")
    print("  --verbose")
    
    print("\nTo process a single transaction:")
    single_transaction = {
        "transaction_id": "tx_single_001",
        "timestamp": "2024-01-15 14:30:00",
        "sender_account": "sender_0001",
        "receiver_account": "receiver_0001", 
        "amount": 1500.00,
        "transaction_type": "transfer",
        "merchant_category": "online",
        "location": "NYC",
        "device_used": "mobile"
    }
    
    print(f"python inference.py \\")
    print(f"  --model-path models/demo_fraud_model \\")
    print(f"  --single-transaction '{json.dumps(single_transaction)}' \\")
    print(f"  --enable-explanations \\")
    print(f"  --verbose")


def demonstrate_results_interpretation():
    """Demonstrate how to interpret results."""
    print("\n" + "="*60)
    print("RESULTS INTERPRETATION")
    print("="*60)
    
    # Create sample prediction results
    sample_results = pd.DataFrame({
        'transaction_id': ['tx_001', 'tx_002', 'tx_003', 'tx_004', 'tx_005'],
        'fraud_probability': [0.05, 0.35, 0.75, 0.92, 0.15],
        'anomaly_score': [0.1, 0.4, 0.8, 0.9, 0.2],
        'risk_level': ['low', 'medium', 'high', 'high', 'low'],
        'fraud_prediction': [0, 0, 1, 1, 0],
        'explanation': [
            'Low fraud risk (probability: 0.05) - transaction appears legitimate',
            'Medium fraud risk (probability: 0.35) - requires additional verification',
            'High fraud risk (probability: 0.75) based on learned fraud patterns',
            'High fraud risk (probability: 0.92) due to unusual transaction patterns (anomaly score: 0.90)',
            'Low fraud risk (probability: 0.15) - transaction appears legitimate'
        ]
    })
    
    print("Sample prediction results:")
    print(sample_results.to_string(index=False))
    
    print("\nInterpretation guide:")
    print("• fraud_probability: 0.0-1.0 score indicating likelihood of fraud")
    print("• anomaly_score: 0.0-1.0 score from unsupervised anomaly detection")
    print("• risk_level: low/medium/high based on optimized thresholds")
    print("• fraud_prediction: Binary classification (0=legitimate, 1=fraud)")
    print("• explanation: Human-readable explanation of the decision")
    
    print("\nRecommended actions by risk level:")
    print("• LOW RISK: Automatic approval, minimal friction")
    print("• MEDIUM RISK: Additional verification (SMS, email confirmation)")
    print("• HIGH RISK: Manual review, potential transaction blocking")
    
    print("\nBusiness metrics to monitor:")
    print("• Fraud Detection Rate: % of actual fraud cases caught")
    print("• False Positive Rate: % of legitimate transactions flagged")
    print("• Customer Friction Rate: % of customers experiencing delays")
    print("• Precision at High Risk: Accuracy of high-risk predictions")


def main():
    """Main demonstration function."""
    print("FRAUD DETECTION SYSTEM - USAGE DEMONSTRATION")
    print("=" * 80)
    print("This script demonstrates how to use the training and inference scripts.")
    print("It will generate sample data and show example commands.")
    
    try:
        # Demonstrate training
        demonstrate_training()
        
        # Demonstrate inference
        demonstrate_inference()
        
        # Demonstrate results interpretation
        demonstrate_results_interpretation()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("1. Run the training command to train your model")
        print("2. Use the inference script to score new transactions")
        print("3. Monitor performance and retrain as needed")
        print("4. Integrate with your production systems")
        
        print("\nFor more options and advanced usage:")
        print("  python train_model.py --help")
        print("  python inference.py --help")
        
    except Exception as e:
        print(f"Demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)