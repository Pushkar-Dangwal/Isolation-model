"""
Train the ORIGINAL FraudDetector model (Deep Isolation Forest + LightGBM)
on balanced dataset with proper train/test split
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import json
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 80)
print("TRAINING ORIGINAL MODEL (Deep Isolation Forest + LightGBM)")
print("=" * 80)

# Load balanced dataset
print("\n1. Loading balanced dataset...")
df = pd.read_csv('data/financial_balanced_300k.csv')
print(f"   Total samples: {len(df):,}")
print(f"   Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"   Legitimate: {(~df['is_fraud']).sum():,} ({(~df['is_fraud']).mean()*100:.1f}%)")

# Split into train and test FIRST (before any processing)
print("\n2. Splitting into TRAIN and TEST sets...")
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['is_fraud']
)

print(f"\n   TRAIN SET:")
print(f"   - Total: {len(train_df):,}")
print(f"   - Fraud: {train_df['is_fraud'].sum():,} ({train_df['is_fraud'].mean()*100:.1f}%)")
print(f"   - Legitimate: {(~train_df['is_fraud']).sum():,}")

print(f"\n   TEST SET (HELD OUT):")
print(f"   - Total: {len(test_df):,}")
print(f"   - Fraud: {test_df['is_fraud'].sum():,} ({test_df['is_fraud'].mean()*100:.1f}%)")
print(f"   - Legitimate: {(~test_df['is_fraud']).sum():,}")

# Verify no overlap
train_ids = set(train_df['transaction_id'])
test_ids = set(test_df['transaction_id'])
overlap = train_ids.intersection(test_ids)
print(f"\n   Overlap check: {len(overlap)} samples")
print(f"   ✓ Data separation verified!" if len(overlap) == 0 else "   ✗ WARNING: Data leakage!")

# Initialize the original FraudDetector
print("\n3. Initializing FraudDetector (Deep Isolation Forest + LightGBM)...")
try:
    from fraud_detector import FraudDetector
    
    # Initialize with original architecture
    detector = FraudDetector(
        anomaly_detector_config={
            'contamination': 0.5,  # 50% fraud rate
            'n_estimators': 100,
            'n_layers': 3,  # Deep feature mapping
            'n_hidden': 128
        },
        classifier_config={
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 10
        },
        random_state=42,
        n_jobs=-1,
        verbose=True,
        ensure_reproducibility=True
    )
    print("   ✓ FraudDetector initialized")
    print("   - Deep Isolation Forest (anomaly detection)")
    print("   - LightGBM (supervised classification)")
    print("   - Risk Scorer (ensemble)")
    
except Exception as e:
    print(f"   ✗ Error initializing FraudDetector: {e}")
    print("\n   This requires the original fraud_detector.py and dependencies.")
    print("   Falling back to simplified version...")
    detector = None

# Train the model
if detector is not None:
    print("\n4. Training model on TRAINING data...")
    try:
        detector.fit(
            df=train_df,
            target_column='is_fraud',
            transaction_id_column='transaction_id',
            validation_split=0.2
        )
        print("   ✓ Model training complete")
        
        # Evaluate on TEST data
        print("\n5. Evaluating on TEST data (unseen)...")
        test_results = detector.evaluate(
            df=test_df,
            target_column='is_fraud',
            transaction_id_column='transaction_id'
        )
        
        print("\n" + "=" * 80)
        print("TEST SET PERFORMANCE (ORIGINAL MODEL)")
        print("=" * 80)
        
        # Extract metrics
        if 'classifier_metrics' in test_results:
            metrics = test_results['classifier_metrics']
            print(f"\nAccuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"Precision: {metrics.get('precision', 0):.4f}")
            print(f"Recall:    {metrics.get('recall', 0):.4f}")
            print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
            print(f"ROC AUC:   {metrics.get('roc_auc', 0):.4f}")
        
        if 'confusion_matrix' in test_results:
            cm = test_results['confusion_matrix']
            print(f"\nConfusion Matrix:")
            print(f"  True Negatives:  {cm.get('tn', 0):,}")
            print(f"  False Positives: {cm.get('fp', 0):,}")
            print(f"  False Negatives: {cm.get('fn', 0):,}")
            print(f"  True Positives:  {cm.get('tp', 0):,}")
        
        # Save results
        print("\n6. Saving results...")
        model_name = f"original_fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {
            'model_name': model_name,
            'model_type': 'Deep Isolation Forest + LightGBM (Original)',
            'dataset': {
                'source': 'financial_balanced_300k.csv (SMOTE from financial.csv)',
                'total_samples': len(df),
                'train_size': len(train_df),
                'test_size': len(test_df)
            },
            'test_metrics': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = f'models/{model_name}_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✓ Results saved to: {output_file}")
        
        # Save model
        model_path = f'models/{model_name}'
        detector.save_model(model_path)
        print(f"   ✓ Model saved to: {model_path}")
        
    except Exception as e:
        print(f"   ✗ Error during training/evaluation: {e}")
        import traceback
        traceback.print_exc()
        detector = None

# If original model failed, show what we would do
if detector is None:
    print("\n" + "=" * 80)
    print("ORIGINAL MODEL ARCHITECTURE")
    print("=" * 80)
    print("\nThe original FraudDetector uses:")
    print("\n1. Deep Isolation Forest (Anomaly Detector)")
    print("   - Random deep feature mapping (3 layers, 128 hidden units)")
    print("   - Isolation Forest on deep features")
    print("   - Unsupervised anomaly detection")
    print("\n2. LightGBM (Supervised Classifier)")
    print("   - Gradient boosting decision trees")
    print("   - Handles class imbalance")
    print("   - 1000 estimators with early stopping")
    print("\n3. Risk Scorer")
    print("   - Combines anomaly scores and classification probabilities")
    print("   - Generates risk levels (low/medium/high/critical)")
    print("\nThis is a HYBRID ensemble approach combining:")
    print("   - Unsupervised learning (anomaly detection)")
    print("   - Supervised learning (classification)")
    print("   - Deep feature learning (neural network-like)")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
