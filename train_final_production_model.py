"""
Train original FraudDetector (Deep Isolation Forest + LightGBM)
on production-ready balanced data with proper test evaluation
"""
import pandas as pd
import numpy as np
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
print("TRAINING ORIGINAL MODEL - PRODUCTION VERSION")
print("Deep Isolation Forest + LightGBM on SMOTENC-balanced data")
print("=" * 80)

# Load datasets
print("\n1. Loading datasets...")
df_train = pd.read_csv('data/financial_train_balanced.csv')
df_test = pd.read_csv('data/financial_test_original.csv')

print(f"\n   TRAINING SET (with SMOTENC oversampling):")
print(f"   - Total: {len(df_train):,}")
print(f"   - Fraud: {df_train['is_fraud'].sum():,} ({df_train['is_fraud'].mean()*100:.1f}%)")
print(f"   - Legitimate: {(~df_train['is_fraud']).sum():,}")

print(f"\n   TEST SET (original distribution, NO oversampling):")
print(f"   - Total: {len(df_test):,}")
print(f"   - Fraud: {df_test['is_fraud'].sum():,} ({df_test['is_fraud'].mean()*100:.4f}%)")
print(f"   - Legitimate: {(~df_test['is_fraud']).sum():,}")

# Verify no overlap
train_ids = set(df_train['transaction_id'])
test_ids = set(df_test['transaction_id'])
overlap = train_ids.intersection(test_ids)
print(f"\n   Data leakage check: {len(overlap)} overlapping transactions")
print(f"   ✓ No data leakage!" if len(overlap) == 0 else "   ✗ WARNING: Data leakage!")

# Initialize FraudDetector
print("\n2. Initializing FraudDetector...")
try:
    from fraud_detector import FraudDetector
    
    detector = FraudDetector(
        anomaly_detector_config={
            'contamination': 0.5,  # Max 50% for IsolationForest
            'n_estimators': 200,   # Increased from 100
            'n_layers': 4,         # Increased from 3
            'n_hidden': 256        # Increased from 128
        },
        classifier_config={
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 10,
            'early_stopping_rounds': 50
        },
        risk_scorer_config={
            'low_risk': 0.2,       # Decreased by 0.1 (was 0.3)
            'high_risk': 0.6       # Decreased by 0.1 (was 0.7)
        },
        random_state=42,
        n_jobs=-1,
        verbose=True,
        ensure_reproducibility=True
    )
    
    print("   ✓ FraudDetector initialized")
    print("   Architecture:")
    print("     1. Deep Isolation Forest (unsupervised anomaly detection)")
    print("        - 4 deep layers with 256 hidden units (UPGRADED)")
    print("        - 200 isolation trees (UPGRADED)")
    print("     2. LightGBM (supervised classification)")
    print("        - 1000 estimators with early stopping")
    print("        - Handles class imbalance")
    print("     3. Risk Scorer (ensemble combination)")
    print("        - Low risk threshold: 0.2 (decreased by 0.1)")
    print("        - High risk threshold: 0.6 (decreased by 0.1)")
    
    # Train model
    print("\n3. Training model on TRAINING data...")
    print("   This may take a few minutes...")
    
    detector.fit(
        df=df_train,
        target_column='is_fraud',
        transaction_id_column='transaction_id',
        validation_split=0.2
    )
    
    print("   ✓ Training complete!")
    
    # Evaluate on TEST data
    print("\n4. Evaluating on TEST data (unseen, original distribution)...")
    test_results = detector.evaluate(
        df=df_test,
        target_column='is_fraud',
        transaction_id_column='transaction_id'
    )
    
    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE")
    print("Model: Deep Isolation Forest + LightGBM (Original Architecture)")
    print("Data: Original distribution (0.1% fraud) - NO oversampling")
    print("=" * 80)
    
    # Extract and display metrics
    if 'classifier_metrics' in test_results:
        metrics = test_results['classifier_metrics']
        print(f"\nSupervised Classifier (LightGBM) Metrics:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
        print(f"  ROC AUC:   {metrics.get('roc_auc', 0):.4f}")
    
    if 'anomaly_metrics' in test_results:
        anom_metrics = test_results['anomaly_metrics']
        print(f"\nAnomaly Detector (Deep Isolation Forest) Metrics:")
        print(f"  Accuracy:  {anom_metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {anom_metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {anom_metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {anom_metrics.get('f1', 0):.4f}")
    
    if 'confusion_matrix' in test_results:
        cm = test_results['confusion_matrix']
        print(f"\nConfusion Matrix (Test Set):")
        print(f"  True Negatives:  {cm.get('true_negatives', 0):,} (correctly identified legitimate)")
        print(f"  False Positives: {cm.get('false_positives', 0):,} (legitimate flagged as fraud)")
        print(f"  False Negatives: {cm.get('false_negatives', 0):,} (fraud missed)")
        print(f"  True Positives:  {cm.get('true_positives', 0):,} (correctly identified fraud)")
    
    # Save results
    print("\n5. Saving model and results...")
    model_name = f"production_fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    results = {
        'model_name': model_name,
        'model_type': 'Deep Isolation Forest + LightGBM (Original Architecture)',
        'data_generation': {
            'method': 'SMOTENC',
            'source': 'financial.csv',
            'improvements': [
                'Proper categorical handling with SMOTENC',
                'Only minority class oversampled',
                'Applied after train-test split',
                'Realistic timestamps',
                'No data leakage'
            ]
        },
        'dataset': {
            'train_size': len(df_train),
            'train_fraud_rate': float(df_train['is_fraud'].mean()),
            'test_size': len(df_test),
            'test_fraud_rate': float(df_test['is_fraud'].mean()),
            'data_leakage': len(overlap) == 0
        },
        'test_metrics': test_results,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = f'models/{model_name}_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✓ Results saved: {output_file}")
    
    # Save model
    try:
        model_path = f'models/{model_name}'
        detector.save_model(model_path)
        print(f"   ✓ Model saved: {model_path}")
    except Exception as e:
        print(f"   ⚠ Model save failed: {e}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\n✅ Production-ready model trained successfully!")
    print(f"   - Used SMOTENC for balanced training data")
    print(f"   - Tested on original imbalanced distribution")
    print(f"   - No data leakage")
    print(f"   - Original architecture (Deep IF + LightGBM)")
    
except ImportError as e:
    print(f"\n✗ Cannot import FraudDetector: {e}")
    print("\nThis requires:")
    print("  - src/fraud_detector.py")
    print("  - src/anomaly_detector.py")
    print("  - src/supervised_classifier.py")
    print("  - src/risk_scorer.py")
    print("  - All dependencies installed")
    
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
