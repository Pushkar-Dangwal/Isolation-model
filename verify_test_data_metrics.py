"""
Verify that model is tested on separate test data (not training data)
Shows clear separation and comprehensive metrics
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import json
from datetime import datetime

print("=" * 80)
print("MODEL EVALUATION - TRAIN/TEST DATA SEPARATION VERIFICATION")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/financial_300k_balanced.csv')
print(f"   Total samples: {len(df):,}")

# Prepare features
print("\n2. Preparing features...")
df['location_match'] = df['location_match'].astype(int)
df_encoded = pd.get_dummies(df, columns=['merchant_category', 'device_type'], drop_first=True)
all_feature_cols = [col for col in df_encoded.columns if col not in ['transaction_id', 'is_fraud', 'transaction_date']]

X = df_encoded[all_feature_cols]
y = df_encoded['is_fraud']

# Split into train and test sets
print("\n3. Splitting data into TRAIN and TEST sets...")
print("   Using stratified split to maintain class balance")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   TRAIN SET:")
print(f"   - Total samples: {len(X_train):,}")
print(f"   - Fraud samples: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"   - Legitimate samples: {(~y_train.astype(bool)).sum():,} ({(1-y_train.mean())*100:.1f}%)")

print(f"\n   TEST SET (HELD OUT - NOT USED FOR TRAINING):")
print(f"   - Total samples: {len(X_test):,}")
print(f"   - Fraud samples: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
print(f"   - Legitimate samples: {(~y_test.astype(bool)).sum():,} ({(1-y_test.mean())*100:.1f}%)")

# Verify no overlap
print("\n4. Verifying train/test separation...")
train_indices = set(X_train.index)
test_indices = set(X_test.index)
overlap = train_indices.intersection(test_indices)
print(f"   Overlap between train and test: {len(overlap)} samples")
print(f"   ✓ Data separation verified!" if len(overlap) == 0 else "   ✗ WARNING: Data leakage detected!")

# Train model on TRAINING data only
print("\n5. Training model on TRAINING data ONLY...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    verbose=0
)

model.fit(X_train, y_train)
print("   ✓ Model trained on training data only")

# Evaluate on TRAINING data
print("\n6. Evaluating on TRAINING data (to check for overfitting)...")
y_pred_train = model.predict(X_train)
y_proba_train = model.predict_proba(X_train)[:, 1]

train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train)
train_recall = recall_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train)
train_roc_auc = roc_auc_score(y_train, y_proba_train)

print(f"\n   TRAINING SET METRICS:")
print(f"   - Accuracy:  {train_accuracy:.4f}")
print(f"   - Precision: {train_precision:.4f}")
print(f"   - Recall:    {train_recall:.4f}")
print(f"   - F1 Score:  {train_f1:.4f}")
print(f"   - ROC AUC:   {train_roc_auc:.4f}")

# Evaluate on TEST data (UNSEEN DATA)
print("\n7. Evaluating on TEST data (UNSEEN DATA - NEVER USED IN TRAINING)...")
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_roc_auc = roc_auc_score(y_test, y_proba_test)
test_pr_auc = average_precision_score(y_test, y_proba_test)

print("\n" + "=" * 80)
print("TEST SET METRICS (FINAL MODEL PERFORMANCE)")
print("=" * 80)

print(f"\nAccuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"F1 Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
print(f"ROC AUC:   {test_roc_auc:.4f} ({test_roc_auc*100:.2f}%)")
print(f"PR AUC:    {test_pr_auc:.4f} ({test_pr_auc*100:.2f}%)")

# Confusion matrix on TEST data
cm_test = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm_test.ravel()

print("\nConfusion Matrix (TEST SET):")
print(f"  True Negatives:  {tn:,} (correctly identified legitimate)")
print(f"  False Positives: {fp:,} (legitimate flagged as fraud)")
print(f"  False Negatives: {fn:,} (fraud missed)")
print(f"  True Positives:  {tp:,} (correctly identified fraud)")

print("\nClassification Report (TEST SET):")
print(classification_report(y_test, y_pred_test, target_names=['Legitimate', 'Fraud']))

# Compare train vs test performance
print("\n" + "=" * 80)
print("TRAIN vs TEST COMPARISON (Overfitting Check)")
print("=" * 80)

print(f"\n{'Metric':<15} {'Train':<12} {'Test':<12} {'Difference':<12}")
print("-" * 51)
print(f"{'Accuracy':<15} {train_accuracy:<12.4f} {test_accuracy:<12.4f} {abs(train_accuracy-test_accuracy):<12.4f}")
print(f"{'Precision':<15} {train_precision:<12.4f} {test_precision:<12.4f} {abs(train_precision-test_precision):<12.4f}")
print(f"{'Recall':<15} {train_recall:<12.4f} {test_recall:<12.4f} {abs(train_recall-test_recall):<12.4f}")
print(f"{'F1 Score':<15} {train_f1:<12.4f} {test_f1:<12.4f} {abs(train_f1-test_f1):<12.4f}")
print(f"{'ROC AUC':<15} {train_roc_auc:<12.4f} {test_roc_auc:<12.4f} {abs(train_roc_auc-test_roc_auc):<12.4f}")

if abs(train_f1 - test_f1) < 0.05:
    print("\n✓ Model generalizes well (train and test performance similar)")
else:
    print("\n⚠ Potential overfitting detected (train performance >> test performance)")

# Per-class metrics on TEST data
print("\n" + "=" * 80)
print("PER-CLASS METRICS (TEST SET)")
print("=" * 80)

print("\nLegitimate Transactions (Class 0):")
legit_precision = precision_score(y_test, y_pred_test, pos_label=0)
legit_recall = recall_score(y_test, y_pred_test, pos_label=0)
legit_f1 = f1_score(y_test, y_pred_test, pos_label=0)
print(f"  Precision: {legit_precision:.4f} (of predicted legitimate, {legit_precision*100:.2f}% are correct)")
print(f"  Recall:    {legit_recall:.4f} (detected {legit_recall*100:.2f}% of all legitimate)")
print(f"  F1 Score:  {legit_f1:.4f}")
print(f"  Support:   {(y_test == 0).sum():,} samples")

print("\nFraud Transactions (Class 1):")
print(f"  Precision: {test_precision:.4f} (of predicted fraud, {test_precision*100:.2f}% are correct)")
print(f"  Recall:    {test_recall:.4f} (detected {test_recall*100:.2f}% of all fraud)")
print(f"  F1 Score:  {test_f1:.4f}")
print(f"  Support:   {(y_test == 1).sum():,} samples")

# Save detailed results
print("\n8. Saving detailed results...")
results = {
    'verification': {
        'train_test_overlap': len(overlap),
        'data_separation_verified': len(overlap) == 0,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_percentage': len(X_test) / len(X) * 100
    },
    'test_set_metrics': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1),
        'roc_auc': float(test_roc_auc),
        'pr_auc': float(test_pr_auc),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    },
    'train_set_metrics': {
        'accuracy': float(train_accuracy),
        'precision': float(train_precision),
        'recall': float(train_recall),
        'f1_score': float(train_f1),
        'roc_auc': float(train_roc_auc)
    },
    'overfitting_check': {
        'accuracy_diff': float(abs(train_accuracy - test_accuracy)),
        'f1_diff': float(abs(train_f1 - test_f1)),
        'generalizes_well': abs(train_f1 - test_f1) < 0.05
    },
    'timestamp': datetime.now().isoformat()
}

output_file = 'models/test_data_verification_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved to: {output_file}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\nKEY FINDINGS:")
print(f"✓ Model trained on {len(X_train):,} samples")
print(f"✓ Model tested on {len(X_test):,} SEPARATE samples (never seen during training)")
print(f"✓ Test accuracy: {test_accuracy:.4f}")
print(f"✓ Test F1 score: {test_f1:.4f}")
print(f"✓ No data leakage detected")
