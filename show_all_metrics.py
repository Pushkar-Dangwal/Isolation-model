#!/usr/bin/env python3
"""
Display comprehensive metrics from the trained fraud detection model.
Shows all available metrics including accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and more.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_evaluation_results(model_name):
    """Load evaluation results from JSON file."""
    eval_path = Path('models') / f'{model_name}_evaluation.json'
    
    if not eval_path.exists():
        print(f"Error: Evaluation file not found: {eval_path}")
        return None
    
    with open(eval_path, 'r') as f:
        return json.load(f)

def load_metadata(model_name):
    """Load model metadata from JSON file."""
    metadata_path = Path('models') / f'{model_name}_metadata.json'
    
    if not metadata_path.exists():
        print(f"Warning: Metadata file not found: {metadata_path}")
        return None
    
    with open(metadata_path, 'r') as f:
        return json.load(f)

def print_section(title, width=80):
    """Print a section header."""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_subsection(title):
    """Print a subsection header."""
    print(f"\n{title}")
    print("-" * len(title))

def display_all_metrics(model_name):
    """Display all available metrics for the model."""
    
    # Load evaluation results
    eval_results = load_evaluation_results(model_name)
    if not eval_results:
        return
    
    # Load metadata
    metadata = load_metadata(model_name)
    
    # Print header
    print_section(f"COMPREHENSIVE METRICS REPORT")
    print(f"Model: {model_name}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. EVALUATION METADATA
    if 'evaluation_metadata' in eval_results:
        print_section("EVALUATION INFORMATION")
        meta = eval_results['evaluation_metadata']
        print(f"Evaluation Method: {meta.get('evaluation_method', 'N/A')}")
        print(f"Test Samples: {meta.get('test_samples', 'N/A'):,}")
        print(f"Test Fraud Rate: {meta.get('test_fraud_rate', 0):.2%}")
        print(f"Evaluation Timestamp: {meta.get('evaluation_timestamp', 'N/A')}")
        if 'train_period' in meta:
            print(f"Training Period: {meta['train_period']}")
        if 'test_period' in meta:
            print(f"Test Period: {meta['test_period']}")
    
    # 2. OVERALL METRICS
    if 'pipeline_evaluation' in eval_results and 'overall_metrics' in eval_results['pipeline_evaluation']:
        print_section("OVERALL PERFORMANCE METRICS")
        metrics = eval_results['pipeline_evaluation']['overall_metrics']
        
        print_subsection("Classification Metrics")
        print(f"  Accuracy:           {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
        print(f"  Precision:          {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)")
        print(f"  Recall:             {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)")
        print(f"  F1-Score:           {metrics.get('f1_score', 0):.4f} ({metrics.get('f1_score', 0)*100:.2f}%)")
        
        print_subsection("Area Under Curve Metrics")
        print(f"  ROC-AUC:            {metrics.get('roc_auc', 0):.4f} ({metrics.get('roc_auc', 0)*100:.2f}%)")
        print(f"  PR-AUC:             {metrics.get('pr_auc', 0):.4f} ({metrics.get('pr_auc', 0)*100:.2f}%)")
    
    # 3. CONFUSION MATRIX
    if 'pipeline_evaluation' in eval_results and 'confusion_matrix' in eval_results['pipeline_evaluation']:
        print_section("CONFUSION MATRIX")
        cm = eval_results['pipeline_evaluation']['confusion_matrix']
        
        tn = cm.get('true_negatives', 0)
        fp = cm.get('false_positives', 0)
        fn = cm.get('false_negatives', 0)
        tp = cm.get('true_positives', 0)
        
        total = tn + fp + fn + tp
        
        print(f"\n                    Predicted")
        print(f"                Legitimate    Fraud")
        print(f"Actual  Legit   {tn:>10,}  {fp:>10,}")
        print(f"        Fraud   {fn:>10,}  {tp:>10,}")
        
        print_subsection("Confusion Matrix Breakdown")
        print(f"  True Negatives (TN):   {tn:>10,}  ({tn/total*100:.2f}% of total)")
        print(f"  False Positives (FP):  {fp:>10,}  ({fp/total*100:.2f}% of total)")
        print(f"  False Negatives (FN):  {fn:>10,}  ({fn/total*100:.2f}% of total)")
        print(f"  True Positives (TP):   {tp:>10,}  ({tp/total*100:.2f}% of total)")
        print(f"  Total Predictions:     {total:>10,}")
    
    # 4. BUSINESS METRICS
    if 'pipeline_evaluation' in eval_results and 'business_metrics' in eval_results['pipeline_evaluation']:
        print_section("BUSINESS IMPACT METRICS")
        biz = eval_results['pipeline_evaluation']['business_metrics']
        
        print(f"  Fraud Detection Rate:      {biz.get('fraud_detection_rate', 0):.2%}")
        print(f"    (Percentage of frauds caught)")
        
        print(f"\n  False Positive Rate:       {biz.get('false_positive_rate', 0):.4f} ({biz.get('false_positive_rate', 0)*100:.2f}%)")
        print(f"    (Legitimate transactions flagged as fraud)")
        
        print(f"\n  Customer Friction Rate:    {biz.get('customer_friction_rate', 0):.2%}")
        print(f"    (Total transactions flagged for review)")
    
    # 5. RISK DISTRIBUTION
    if 'pipeline_evaluation' in eval_results and 'risk_analysis' in eval_results['pipeline_evaluation']:
        risk_analysis = eval_results['pipeline_evaluation']['risk_analysis']
        if 'risk_distribution' in risk_analysis:
            print_section("RISK LEVEL DISTRIBUTION")
            risk_dist = risk_analysis['risk_distribution']
            
            total_transactions = sum(risk_dist.values())
            
            for risk_level in ['low', 'medium', 'high', 'critical']:
                count = risk_dist.get(risk_level, 0)
                percentage = (count / total_transactions * 100) if total_transactions > 0 else 0
                print(f"  {risk_level.capitalize():>10}: {count:>10,} ({percentage:>6.2f}%)")
            
            print(f"  {'Total':>10}: {total_transactions:>10,}")
    
    # 6. TRAINING METRICS (from metadata)
    if metadata and 'performance_metrics' in metadata:
        print_section("TRAINING PERFORMANCE METRICS")
        train_metrics = metadata['performance_metrics']
        
        print_subsection("Training Validation Metrics")
        print(f"  ROC-AUC:            {train_metrics.get('roc_auc', 0):.4f} ({train_metrics.get('roc_auc', 0)*100:.2f}%)")
        print(f"  PR-AUC:             {train_metrics.get('pr_auc', 0):.4f} ({train_metrics.get('pr_auc', 0)*100:.2f}%)")
        
        if 'best_threshold' in train_metrics:
            print(f"\n  Optimal Threshold:  {train_metrics['best_threshold']:.4f}")
    
    # 7. PERFORMANCE COMPARISON
    if metadata and 'performance_metrics' in metadata and 'pipeline_evaluation' in eval_results:
        print_section("TRAINING vs EVALUATION COMPARISON")
        
        train_roc = metadata['performance_metrics'].get('roc_auc', 0)
        eval_roc = eval_results['pipeline_evaluation']['overall_metrics'].get('roc_auc', 0)
        roc_gap = train_roc - eval_roc
        
        train_pr = metadata['performance_metrics'].get('pr_auc', 0)
        eval_pr = eval_results['pipeline_evaluation']['overall_metrics'].get('pr_auc', 0)
        pr_gap = train_pr - eval_pr
        
        print(f"  Metric          Training    Evaluation    Gap")
        print(f"  ROC-AUC         {train_roc:.4f}      {eval_roc:.4f}        {roc_gap:+.4f}")
        print(f"  PR-AUC          {train_pr:.4f}      {eval_pr:.4f}        {pr_gap:+.4f}")
        
        print(f"\n  Performance Gap Analysis:")
        if abs(roc_gap) < 0.05:
            print(f"    ✓ Excellent generalization (ROC-AUC gap < 5%)")
        elif abs(roc_gap) < 0.10:
            print(f"    ✓ Good generalization (ROC-AUC gap < 10%)")
        else:
            print(f"    ⚠ Significant gap (ROC-AUC gap > 10%) - possible overfitting")
    
    # 8. MODEL INFORMATION
    if metadata:
        print_section("MODEL INFORMATION")
        
        if 'training_metadata' in metadata:
            train_meta = metadata['training_metadata']
            print(f"  Training Samples:   {train_meta.get('training_samples', 'N/A'):,}")
            print(f"  Validation Samples: {train_meta.get('validation_samples', 'N/A'):,}")
            print(f"  Fraud Rate:         {train_meta.get('fraud_rate_total', 0):.2%}")
        
        if 'feature_count' in metadata:
            print(f"  Feature Count:      {metadata['feature_count']}")
        
        if 'model_version' in metadata:
            print(f"  Model Version:      {metadata['model_version']}")
        
        if 'training_timestamp' in metadata:
            print(f"  Training Date:      {metadata['training_timestamp']}")
    
    # 9. SUMMARY
    print_section("SUMMARY")
    
    if 'pipeline_evaluation' in eval_results:
        metrics = eval_results['pipeline_evaluation']['overall_metrics']
        biz = eval_results['pipeline_evaluation'].get('business_metrics', {})
        
        print("\nKey Takeaways:")
        print(f"  • Model achieves {metrics.get('roc_auc', 0)*100:.1f}% ROC-AUC on test data")
        print(f"  • Catches {biz.get('fraud_detection_rate', 0)*100:.1f}% of fraudulent transactions")
        print(f"  • Only {biz.get('false_positive_rate', 0)*100:.2f}% false positive rate")
        print(f"  • Overall accuracy: {metrics.get('accuracy', 0)*100:.2f}%")
        
        # Performance assessment
        roc_auc = metrics.get('roc_auc', 0)
        if roc_auc >= 0.90:
            assessment = "Excellent"
        elif roc_auc >= 0.80:
            assessment = "Good"
        elif roc_auc >= 0.70:
            assessment = "Fair"
        else:
            assessment = "Needs Improvement"
        
        print(f"\n  Overall Assessment: {assessment}")
    
    print("\n" + "=" * 80 + "\n")

def main():
    """Main function."""
    # Find the latest model
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("Error: models directory not found")
        return 1
    
    # Look for evaluation files
    eval_files = list(models_dir.glob('*_evaluation.json'))
    
    if not eval_files:
        print("Error: No evaluation files found in models directory")
        return 1
    
    # Get the latest evaluation file
    latest_eval = max(eval_files, key=lambda p: p.stat().st_mtime)
    model_name = latest_eval.stem.replace('_evaluation', '')
    
    print(f"Loading metrics for model: {model_name}\n")
    
    # Display all metrics
    display_all_metrics(model_name)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
