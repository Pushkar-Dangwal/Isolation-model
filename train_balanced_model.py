"""
Train a balanced fraud detection model with comprehensive metrics
"""
import pandas as pd
import numpy as np
from src.dual_evaluation.balanced_pipeline import BalancedPipeline
import json
from datetime import datetime

print("=" * 80)
print("BALANCED FRAUD DETECTION MODEL TRAINING")
print("=" * 80)

# Load the balanced dataset
print("\n1. Loading balanced dataset...")
df = pd.read_csv('data/financial_300k_balanced.csv')
print(f"   Total samples: {len(df):,}")
print(f"   Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"   Legitimate: {(~df['is_fraud'].astype(bool)).sum():,} ({(1-df['is_fraud'].mean())*100:.1f}%)")

# Prepare features and target
print("\n2. Preparing features...")
feature_cols = [
    'transaction_amount', 'account_age_days', 'transaction_hour',
    'previous_transactions', 'avg_transaction_amount', 'days_since_last_transaction'
]

# Convert boolean to int
df['location_match'] = df['location_match'].astype(int)
feature_cols.append('location_match')

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['merchant_category', 'device_type'], drop_first=True)

# Get all feature columns (numeric + encoded)
all_feature_cols = [col for col in df_encoded.columns if col not in ['transaction_id', 'is_fraud', 'transaction_date']]
print(f"   Total features: {len(all_feature_cols)}")

X = df_encoded[all_feature_cols]
y = df_encoded['is_fraud']

# Initialize and train the balanced pipeline
print("\n3. Training balanced model...")
print("   Using IsolationForest with balanced sampling...")
pipeline = BalancedPipeline(random_state=42, n_jobs=-1)

# Prepare training dataframe
train_df = df_encoded.copy()

trained_model = pipeline.train_model(
    train_df=train_df,
    fraud_col='is_fraud',
    validation_split=0.2
)

print(f"\n   Model trained successfully")

# Get test data for evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# Evaluate with comprehensive metrics
print("\n4. Evaluating model...")
evaluation = pipeline.evaluate(
    test_df=df_encoded.iloc[X_test.index],
    fraud_col='is_fraud'
)

# Display metrics
print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)

print(f"\nAccuracy: {evaluation['accuracy']:.4f}")
print(f"Precision: {evaluation['precision']:.4f}")
print(f"Recall: {evaluation['recall']:.4f}")
print(f"F1 Score: {evaluation['f1']:.4f}")
print(f"ROC AUC: {evaluation['roc_auc']:.4f}")
print(f"PR AUC: {evaluation['pr_auc']:.4f}")

print("\nConfusion Matrix:")
cm = evaluation['confusion_matrix']
print(f"  True Negatives:  {cm['tn']:,}")
print(f"  False Positives: {cm['fp']:,}")
print(f"  False Negatives: {cm['fn']:,}")
print(f"  True Positives:  {cm['tp']:,}")

print("\nClass-wise Metrics:")
for cls, metrics in evaluation['class_metrics'].items():
    print(f"  Class {cls}:")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1-Score: {metrics['f1-score']:.4f}")
    print(f"    Support: {metrics['support']:,}")

# Optimize threshold
print("\n5. Optimizing decision threshold...")
test_df = df_encoded.iloc[X_test.index].copy()
threshold_results = pipeline.optimize_threshold(
    test_df=test_df,
    fraud_col='is_fraud',
    metric='f1'
)

print(f"\n   Optimal threshold: {threshold_results['optimal_threshold']:.4f}")
print(f"   F1 Score at optimal threshold: {threshold_results['best_score']:.4f}")

# Save results
print("\n6. Saving results...")
model_name = f"balanced_fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
results = {
    'model_name': model_name,
    'dataset': {
        'total_samples': len(df),
        'fraud_samples': int(df['is_fraud'].sum()),
        'legitimate_samples': int((~df['is_fraud'].astype(bool)).sum()),
        'fraud_rate': float(df['is_fraud'].mean())
    },
    'features': all_feature_cols,
    'training': {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'validation_split': 0.2
    },
    'evaluation': {
        'accuracy': float(evaluation['accuracy']),
        'precision': float(evaluation['precision']),
        'recall': float(evaluation['recall']),
        'f1': float(evaluation['f1']),
        'roc_auc': float(evaluation['roc_auc']),
        'pr_auc': float(evaluation['pr_auc']),
        'confusion_matrix': {k: int(v) for k, v in cm.items()},
        'class_metrics': {k: {mk: float(mv) if mk != 'support' else int(mv) 
                              for mk, mv in v.items()} 
                         for k, v in evaluation['class_metrics'].items()}
    },
    'threshold_optimization': {
        'optimal_threshold': float(threshold_results['optimal_threshold']),
        'best_f1_score': float(threshold_results['best_score'])
    },
    'timestamp': datetime.now().isoformat()
}

output_file = f'models/{model_name}_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved to: {output_file}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
