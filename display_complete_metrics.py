"""
Display complete metrics from the trained model
"""
import json
from datetime import datetime

print("=" * 80)
print("COMPLETE MODEL METRICS - PRODUCTION FRAUD DETECTOR")
print("=" * 80)

# Load results
results_file = 'models/production_fraud_detector_20260402_142537_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Model info
print(f"\n📋 MODEL INFORMATION")
print(f"   Model: {results['model_name']}")
print(f"   Type: {results['model_type']}")
print(f"   Timestamp: {results['timestamp']}")

# Dataset info
print(f"\n📊 DATASET INFORMATION")
dataset = results['dataset']
print(f"   Training samples: {dataset['train_size']:,}")
print(f"   Training fraud rate: {dataset['train_fraud_rate']*100:.2f}%")
print(f"   Test samples: {dataset['test_size']:,}")
print(f"   Test fraud rate: {dataset['test_fraud_rate']*100:.2f}%")
print(f"   Data leakage: {'No ✓' if not dataset['data_leakage'] else 'Yes ✗'}")

# Overall metrics
print(f"\n🎯 OVERALL PERFORMANCE METRICS (Test Set)")
print("=" * 80)
overall = results['test_metrics']['overall_metrics']
print(f"   Accuracy:   {overall['accuracy']*100:6.2f}%  ({int(overall['accuracy']*dataset['test_size']):,} out of {dataset['test_size']:,} correct)")
print(f"   Precision:  {overall['precision']*100:6.2f}%  (of predicted fraud, how many are correct)")
print(f"   Recall:     {overall['recall']*100:6.2f}%  (of actual fraud, how many detected)")
print(f"   F1 Score:   {overall['f1_score']*100:6.2f}%  (harmonic mean of precision & recall)")
print(f"   PR-AUC:     {overall['pr_auc']*100:6.2f}%  (precision-recall curve area)")
print(f"   ROC-AUC:    {overall['roc_auc']*100:6.2f}%  (ROC curve area)")

# Confusion matrix
print(f"\n📈 CONFUSION MATRIX (Test Set)")
print("=" * 80)
cm = results['test_metrics']['confusion_matrix']
tn = cm['true_negatives']
fp = cm['false_positives']
fn = cm['false_negatives']
tp = cm['true_positives']

print(f"\n                    Predicted")
print(f"                Legitimate    Fraud")
print(f"   Actual Legitimate   {tn:>6,}    {fp:>6,}")
print(f"          Fraud        {fn:>6,}    {tp:>6,}")
print(f"\n   True Negatives:  {tn:>6,} (correctly identified legitimate)")
print(f"   False Positives: {fp:>6,} (legitimate wrongly flagged as fraud)")
print(f"   False Negatives: {fn:>6,} (fraud missed - CRITICAL!)")
print(f"   True Positives:  {tp:>6,} (correctly identified fraud)")

# Calculate rates
total_legit = tn + fp
total_fraud = fn + tp
print(f"\n   Legitimate Detection Rate: {tn/total_legit*100:.2f}% ({tn}/{total_legit})")
print(f"   Fraud Detection Rate:      {tp/total_fraud*100:.2f}% ({tp}/{total_fraud})")

# Risk analysis
print(f"\n🎲 RISK LEVEL ANALYSIS")
print("=" * 80)
risk_dist = results['test_metrics']['risk_analysis']['risk_distribution']
fraud_by_risk = results['test_metrics']['risk_analysis']['fraud_by_risk_level']

print(f"\n   Risk Distribution:")
print(f"   - Low Risk:    {risk_dist['low']:>6,} transactions")
print(f"   - Medium Risk: {risk_dist['medium']:>6,} transactions")
print(f"   - High Risk:   {risk_dist['high']:>6,} transactions")

print(f"\n   Fraud by Risk Level:")
for level in ['low', 'medium', 'high']:
    if level in fraud_by_risk:
        data = fraud_by_risk[level]
        print(f"   {level.upper():>6} Risk: {data['fraud_count']:>3} fraud out of {data['count']:>6,} ({data['fraud_rate']*100:6.2f}%)")

# Business metrics
print(f"\n💼 BUSINESS METRICS")
print("=" * 80)
business = results['test_metrics']['business_metrics']
print(f"   Fraud Detection Rate:     {business['fraud_detection_rate']*100:6.2f}%")
print(f"   False Positive Rate:      {business['false_positive_rate']*100:6.2f}%")
print(f"   Customer Friction Rate:   {business['customer_friction_rate']*100:6.2f}%")
print(f"   Precision at High Risk:   {business['precision_at_high_risk']*100:6.2f}%")

# Component performance
print(f"\n🔧 COMPONENT PERFORMANCE")
print("=" * 80)
components = results['test_metrics']['component_performance']

print(f"\n   LightGBM Classifier:")
clf = components['classifier']['training_history']
print(f"   - Training samples: {clf['n_train_samples']:,}")
print(f"   - Features: {clf['n_features']}")
print(f"   - Best iteration: {clf['best_iteration']} (out of 1000)")
print(f"   - Training AUC: {clf['train_auc']*100:.4f}%")
print(f"   - Validation AUC: {clf['val_auc']*100:.4f}%")
print(f"   - Training F1: {clf['train_f1']*100:.4f}%")
print(f"   - Validation F1: {clf['val_f1']*100:.4f}%")

print(f"\n   Deep Isolation Forest:")
anom = components['anomaly_detector']
print(f"   - Status: {anom['component_status']}")
print(f"   - Features: {anom['feature_count']}")

print(f"\n   Risk Scorer:")
risk = components['risk_scorer']
print(f"   - Status: {risk['component_status']}")
print(f"   - Low risk threshold: {risk['threshold_config']['low_risk']}")
print(f"   - High risk threshold: {risk['threshold_config']['high_risk']}")

# Key insights
print(f"\n💡 KEY INSIGHTS")
print("=" * 80)
print(f"   ✅ STRENGTHS:")
print(f"      - 100% precision (no false alarms)")
print(f"      - 99.98% accuracy")
print(f"      - 99.92% ROC-AUC (excellent discrimination)")
print(f"      - Zero false positives (no customer friction)")
print(f"\n   ⚠️  AREAS FOR IMPROVEMENT:")
print(f"      - 80% recall (missed 2 out of 10 frauds)")
print(f"      - 1 fraud hidden in 'low risk' category")
print(f"      - May need threshold tuning for higher recall")

print(f"\n📝 RECOMMENDATIONS")
print("=" * 80)
print(f"   1. Lower decision threshold to catch more fraud (trade precision for recall)")
print(f"   2. Investigate the 2 missed fraud cases (false negatives)")
print(f"   3. Consider ensemble with other models")
print(f"   4. Add more features for better fraud detection")
print(f"   5. Monitor performance on new data (concept drift)")

print(f"\n" + "=" * 80)
print("METRICS DISPLAY COMPLETE")
print("=" * 80)
