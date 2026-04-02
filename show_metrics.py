"""
Quick script to display key fraud detection metrics from evaluation results.
"""

import json
import sys

def display_metrics(evaluation_file):
    """Load and display key metrics from evaluation JSON."""
    
    with open(evaluation_file, 'r') as f:
        data = json.load(f)
    
    # Extract metrics from the evaluation file
    metrics = data['pipeline_evaluation']['overall_metrics']
    
    print("\n" + "="*60)
    print("FRAUD DETECTION MODEL METRICS")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Value':<15} {'Why Needed'}")
    print("-"*60)
    print(f"{'Accuracy':<20} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)  Basic metric")
    print(f"{'Precision':<20} {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)  How correct fraud predictions are")
    print(f"{'Recall':<20} {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)  How many frauds you catch")
    print(f"{'F1-Score':<20} {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)  Balance of precision & recall")
    
    print("\n" + "="*60)
    print("ADDITIONAL METRICS")
    print("="*60)
    print(f"{'PR-AUC':<20} {metrics['pr_auc']:.4f}        Overall fraud detection quality")
    print(f"{'ROC-AUC':<20} {metrics['roc_auc']:.4f}        Classification performance")
    
    # Confusion Matrix
    cm = data['pipeline_evaluation']['confusion_matrix']
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"True Positives:  {cm['true_positives']:,} (Frauds correctly detected)")
    print(f"False Positives: {cm['false_positives']:,} (Legitimate flagged as fraud)")
    print(f"True Negatives:  {cm['true_negatives']:,} (Legitimate correctly identified)")
    print(f"False Negatives: {cm['false_negatives']:,} (Frauds missed)")
    
    # Business Impact
    business = data['pipeline_evaluation']['business_metrics']
    print("\n" + "="*60)
    print("BUSINESS IMPACT")
    print("="*60)
    print(f"Fraud Detection Rate:    {business['fraud_detection_rate']*100:.1f}%")
    print(f"False Positive Rate:     {business['false_positive_rate']*100:.1f}%")
    print(f"Customer Friction Rate:  {business['customer_friction_rate']*100:.1f}%")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    evaluation_file = "models/production_fraud_detector_v1_optimized_evaluation.json"
    
    if len(sys.argv) > 1:
        evaluation_file = sys.argv[1]
    
    try:
        display_metrics(evaluation_file)
    except FileNotFoundError:
        print(f"Error: Could not find {evaluation_file}")
        print("Usage: python show_metrics.py [evaluation_file.json]")
    except Exception as e:
        print(f"Error: {e}")
