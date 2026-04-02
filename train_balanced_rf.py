"""
Train a balanced fraud detection model using Random Forest
Optimized for balanced datasets with comprehensive metrics
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve
)
import json
from datetime import datetime
import joblib

print("=" * 80)
print("BALANCED FRAUD DETECTION MODEL TRAINING - RANDOM FOREST")
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

# Split data
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train):,}")
print(f"   Test samples: {len(X_test):,}")

# Train Random Forest
print("\n4. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    verbose=1
)

model.fit(X_train, y_train)
print("   Model training complete!")

# Make predictions
print("\n5. Making predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

# Evaluate
print("\n6. Evaluating model...")
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_proba_test)
pr_auc = average_precision_score(y_test, y_proba_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

# Display metrics
print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC: {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(f"  True Negatives:  {tn:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Positives:  {tp:,}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['Legitimate', 'Fraud']))

# Training set performance
print("\nTraining Set Performance:")
train_accuracy = accuracy_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train)
print(f"  Accuracy: {train_accuracy:.4f}")
print(f"  F1 Score: {train_f1:.4f}")

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model
print("\n7. Saving model and results...")
model_name = f"balanced_rf_fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
model_path = f'models/{model_name}.joblib'
joblib.dump(model, model_path)
print(f"   Model saved to: {model_path}")

# Save results
results = {
    'model_name': model_name,
    'model_type': 'RandomForestClassifier',
    'dataset': {
        'total_samples': len(df),
        'fraud_samples': int(df['is_fraud'].sum()),
        'legitimate_samples': int((~df['is_fraud'].astype(bool)).sum()),
        'fraud_rate': float(df['is_fraud'].mean())
    },
    'features': all_feature_cols,
    'feature_importance': feature_importance.to_dict('records'),
    'training': {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_estimators': 100,
        'max_depth': 20,
        'class_weight': 'balanced',
        'random_state': 42
    },
    'evaluation': {
        'test_set': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        },
        'train_set': {
            'accuracy': float(train_accuracy),
            'f1': float(train_f1)
        }
    },
    'timestamp': datetime.now().isoformat()
}

results_path = f'models/{model_name}_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"   Results saved to: {results_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nModel: {model_name}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test ROC AUC: {roc_auc:.4f}")
