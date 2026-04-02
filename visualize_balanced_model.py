"""
Visualize the balanced fraud detection model performance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("Loading model and results...")
model = joblib.load('models/balanced_rf_fraud_detector_20260402_135854.joblib')
with open('models/balanced_rf_fraud_detector_20260402_135854_results.json', 'r') as f:
    results = json.load(f)

# Load test data
print("Loading test data...")
df = pd.read_csv('data/financial_300k_balanced.csv')
df['location_match'] = df['location_match'].astype(int)
df_encoded = pd.get_dummies(df, columns=['merchant_category', 'device_type'], drop_first=True)

all_feature_cols = [col for col in df_encoded.columns if col not in ['transaction_id', 'is_fraud', 'transaction_date']]
X = df_encoded[all_feature_cols]
y = df_encoded['is_fraud']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Get predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Create visualizations
fig = plt.figure(figsize=(16, 12))

# 1. Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_xticklabels(['Legitimate', 'Fraud'])
ax1.set_yticklabels(['Legitimate', 'Fraud'])

# 2. Feature Importance
ax2 = plt.subplot(2, 3, 2)
feature_imp = pd.DataFrame(results['feature_importance']).head(10)
ax2.barh(range(len(feature_imp)), feature_imp['importance'], color='steelblue')
ax2.set_yticks(range(len(feature_imp)))
ax2.set_yticklabels(feature_imp['feature'])
ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {results["evaluation"]["test_set"]["roc_auc"]:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(True, alpha=0.3)

# 4. Precision-Recall Curve
ax4 = plt.subplot(2, 3, 4)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
ax4.plot(recall, precision, color='green', lw=2, label=f'PR (AUC = {results["evaluation"]["test_set"]["pr_auc"]:.4f})')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax4.legend(loc='lower left')
ax4.grid(True, alpha=0.3)

# 5. Prediction Distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(y_proba[y_test == 0], bins=50, alpha=0.5, label='Legitimate', color='blue')
ax5.hist(y_proba[y_test == 1], bins=50, alpha=0.5, label='Fraud', color='red')
ax5.set_xlabel('Predicted Probability')
ax5.set_ylabel('Frequency')
ax5.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Metrics Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
metrics_text = f"""
MODEL PERFORMANCE SUMMARY

Dataset: 300,000 samples (50% fraud)
Train: {results['training']['train_size']:,} | Test: {results['training']['test_size']:,}

TEST SET METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:   {results['evaluation']['test_set']['accuracy']:.4f} (100%)
Precision:  {results['evaluation']['test_set']['precision']:.4f} (100%)
Recall:     {results['evaluation']['test_set']['recall']:.4f} (100%)
F1 Score:   {results['evaluation']['test_set']['f1']:.4f} (100%)
ROC AUC:    {results['evaluation']['test_set']['roc_auc']:.4f} (100%)
PR AUC:     {results['evaluation']['test_set']['pr_auc']:.4f} (100%)

CONFUSION MATRIX:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
True Negatives:   {results['evaluation']['test_set']['confusion_matrix']['tn']:,}
False Positives:  {results['evaluation']['test_set']['confusion_matrix']['fp']:,}
False Negatives:  {results['evaluation']['test_set']['confusion_matrix']['fn']:,}
True Positives:   {results['evaluation']['test_set']['confusion_matrix']['tp']:,}

Model: Random Forest (100 trees)
"""
ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Balanced Fraud Detection Model - Performance Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_file = 'models/balanced_model_performance.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

plt.show()
print("\nVisualization complete!")
