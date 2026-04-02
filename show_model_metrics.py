#!/usr/bin/env python3
"""
Display Key Model Metrics

Shows the essential performance metrics for the fraud detection model:
- Accuracy: Basic metric
- Precision: How correct fraud predictions are
- Recall: How many frauds you catch
- F1-score: Balance of precision & recall
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))


def load_model_metrics():
    """Load metrics from the trained model's evaluation file."""
    
    # Find the most recent evaluation file
    models_dir = Path("models")
    eval_files = list(models_dir.glob("*_evaluation.json"))
    
    if not eval_files:
        print("❌ No evaluation files found. Please train the model first.")
        return None
    
    # Get the most recent evaluation file
    latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📊 Loading metrics from: {latest_eval.name}\n")
    
    with open(latest_eval, 'r') as f:
        eval_data = json.load(f)
    
    return eval_data


def display_metrics(eval_data):
    """Display the key metrics in a clean format."""
    
    # Extract metrics from the evaluation data
    if 'pipeline_evaluation' in eval_data and 'overall_metrics' in eval_data['pipeline_evaluation']:
        metrics = eval_data['pipeline_evaluation']['overall_metrics']
    else:
        print("❌ Could not find metrics in evaluation file")
        return
    
    # Get the metrics
    accuracy = metrics.get('accuracy', 0.0)
    precision = metrics.get('precision', 0.0)
    recall = metrics.get('recall', 0.0)
    f1_score = metrics.get('f1_score', 0.0)
    
    # Display in a clean table format
    print("=" * 70)
    print("🎯 FRAUD DETECTION MODEL - KEY PERFORMANCE METRICS")
    print("=" * 70)
    print()
    
    print(f"{'Metric':<20} {'Value':<15} {'Why Needed':<35}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {accuracy:<15.3f} {'Basic metric':<35}")
    print(f"{'Precision':<20} {precision:<15.3f} {'How correct fraud predictions are':<35}")
    print(f"{'Recall':<20} {recall:<15.3f} {'How many frauds you catch':<35}")
    print(f"{'F1-Score':<20} {f1_score:<15.3f} {'Balance of precision & recall':<35}")
    print("-" * 70)
    print()
    
    # Additional context
    print("📈 INTERPRETATION:")
    print()
    
    # Precision interpretation
    if precision > 0:
        print(f"   • Precision ({precision:.1%}): Out of all transactions flagged as fraud,")
        print(f"     {precision:.1%} are actually fraudulent")
    else:
        print(f"   • Precision ({precision:.1%}): Model is very conservative")
    
    # Recall interpretation
    if recall > 0:
        print(f"   • Recall ({recall:.1%}): The model catches {recall:.1%} of all fraudulent")
        print(f"     transactions in the dataset")
    else:
        print(f"   • Recall ({recall:.1%}): Model is not detecting frauds")
    
    # F1-Score interpretation
    print(f"   • F1-Score ({f1_score:.3f}): Overall balance between precision and recall")
    
    if f1_score > 0.7:
        print(f"     ✅ Good balance - model performs well")
    elif f1_score > 0.4:
        print(f"     ⚠️  Moderate performance - room for improvement")
    else:
        print(f"     ❌ Low performance - model needs tuning")
    
    print()
    
    # Business metrics if available
    if 'business_metrics' in eval_data['pipeline_evaluation']:
        business = eval_data['pipeline_evaluation']['business_metrics']
        
        print("💼 BUSINESS IMPACT:")
        print()
        print(f"   • Fraud Detection Rate: {business.get('fraud_detection_rate', 0):.1%}")
        print(f"   • False Positive Rate: {business.get('false_positive_rate', 0):.3f}")
        print(f"   • Customer Friction Rate: {business.get('customer_friction_rate', 0):.1%}")
        print()
    
    # Model info if available
    if 'evaluation_metadata' in eval_data:
        metadata = eval_data['evaluation_metadata']
        print("ℹ️  MODEL INFO:")
        print()
        print(f"   • Test Samples: {metadata.get('test_samples', 'N/A'):,}")
        print(f"   • Test Fraud Rate: {metadata.get('test_fraud_rate', 0):.1%}")
        print(f"   • Model: {metadata.get('model_name', 'N/A')}")
        print()
    
    print("=" * 70)


def main():
    """Main function."""
    print()
    
    # Load metrics
    eval_data = load_model_metrics()
    
    if eval_data is None:
        return 1
    
    # Display metrics
    display_metrics(eval_data)
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
