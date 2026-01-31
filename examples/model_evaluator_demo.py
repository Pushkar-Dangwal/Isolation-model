"""
Demonstration of the ModelEvaluator class for fraud detection evaluation.
This example shows how to use the comprehensive evaluation capabilities.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_evaluator import ModelEvaluator


def create_sample_fraud_data(n_samples=5000, fraud_rate=0.04):
    """Create realistic sample fraud detection data."""
    np.random.seed(42)
    
    # Generate true labels
    y_true = np.random.choice([0, 1], n_samples, p=[1-fraud_rate, fraud_rate])
    
    # Generate probabilities that are correlated with true labels
    # Fraud cases get higher probabilities, legitimate cases get lower probabilities
    fraud_mask = y_true == 1
    legitimate_mask = y_true == 0
    
    # Use beta distributions to create realistic probability distributions
    fraud_probs = np.random.beta(3, 2, np.sum(fraud_mask))  # Skewed towards higher values
    legitimate_probs = np.random.beta(1, 4, np.sum(legitimate_mask))  # Skewed towards lower values
    
    y_proba = np.zeros(n_samples)
    y_proba[fraud_mask] = fraud_probs
    y_proba[legitimate_mask] = legitimate_probs
    
    return y_true, y_proba


def demonstrate_basic_evaluation():
    """Demonstrate basic evaluation metrics."""
    print("=== Basic Evaluation Metrics Demo ===")
    
    # Create sample data
    y_true, y_proba = create_sample_fraud_data()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(save_plots=False)
    
    # Calculate comprehensive metrics
    metrics = evaluator.calculate_comprehensive_metrics(y_true, y_proba, threshold=0.5)
    
    print(f"Dataset: {len(y_true)} transactions, {np.sum(y_true)} fraud cases ({np.mean(y_true):.1%} fraud rate)")
    print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
    print(f"Customer Friction Rate: {metrics['customer_friction_rate']:.3f}")
    print()


def demonstrate_threshold_analysis():
    """Demonstrate threshold performance analysis."""
    print("=== Threshold Analysis Demo ===")
    
    # Create sample data
    y_true, y_proba = create_sample_fraud_data()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(save_plots=False)
    
    # Analyze threshold performance
    thresholds = np.arange(0.1, 0.9, 0.1)
    analysis = evaluator.analyze_threshold_performance(y_true, y_proba, thresholds=thresholds)
    
    print("Threshold Analysis Results:")
    print("Threshold | Precision | Recall | F1-Score | Customer Friction")
    print("-" * 65)
    
    for result in analysis['threshold_analysis']:
        print(f"{result['threshold']:.1f}       | "
              f"{result['precision']:.3f}     | "
              f"{result['recall']:.3f}  | "
              f"{result['f1_score']:.3f}    | "
              f"{result['customer_friction_rate']:.3f}")
    
    # Show optimal thresholds
    print("\nOptimal Thresholds:")
    for strategy, details in analysis['optimal_thresholds'].items():
        if isinstance(details, dict) and 'threshold' in details:
            print(f"{strategy}: {details['threshold']:.3f}")
    print()


def demonstrate_performance_report():
    """Demonstrate comprehensive performance report generation."""
    print("=== Performance Report Demo ===")
    
    # Create sample data
    y_true, y_proba = create_sample_fraud_data()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(save_plots=False)
    
    # Generate comprehensive performance report
    report = evaluator.generate_performance_report(
        y_true, y_proba,
        current_threshold=0.5,
        model_name="Demo Fraud Detection Model",
        include_threshold_analysis=True,
        include_business_metrics=True
    )
    
    # Display executive summary
    exec_summary = report['executive_summary']
    print(f"Model: {report['model_name']}")
    print(f"Overall Assessment: {exec_summary['overall_assessment']}")
    print(f"Key Metrics:")
    for metric, value in exec_summary['key_metrics'].items():
        print(f"  {metric.upper()}: {value:.3f}")
    
    print(f"\nBusiness Impact: {exec_summary['business_impact_summary']}")
    
    print("\nKey Findings:")
    for finding in exec_summary['key_findings']:
        print(f"  - {finding}")
    
    print("\nPrimary Recommendations:")
    for i, rec in enumerate(exec_summary['primary_recommendations'][:3], 1):
        if isinstance(rec, dict):
            print(f"  {i}. [{rec.get('priority', 'medium').upper()}] {rec.get('action', rec)}")
        else:
            print(f"  {i}. {rec}")
    print()


def demonstrate_model_comparison():
    """Demonstrate model comparison functionality."""
    print("=== Model Comparison Demo ===")
    
    # Create sample data
    y_true, y_proba = create_sample_fraud_data()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(save_plots=False)
    
    # Simulate two different models with different performance
    # Model 1: Higher precision, lower recall
    y_proba_model1 = y_proba * 0.8 + 0.1  # Shift probabilities
    
    # Model 2: Lower precision, higher recall  
    y_proba_model2 = y_proba * 1.2  # Scale probabilities
    y_proba_model2 = np.clip(y_proba_model2, 0, 1)  # Ensure valid range
    
    # Evaluate both models
    model1_metrics = evaluator.calculate_comprehensive_metrics(y_true, y_proba_model1, threshold=0.5)
    model2_metrics = evaluator.calculate_comprehensive_metrics(y_true, y_proba_model2, threshold=0.5)
    
    # Create comparison
    model_results = {
        'Conservative Model': {'performance_metrics': model1_metrics},
        'Aggressive Model': {'performance_metrics': model2_metrics}
    }
    
    comparison = evaluator.compare_models(model_results)
    
    print("Model Comparison Results:")
    print("Metric      | Conservative | Aggressive | Winner")
    print("-" * 50)
    
    for metric in ['pr_auc', 'f1_score', 'precision', 'recall']:
        if metric in comparison['metric_comparison']:
            conservative_val = comparison['metric_comparison'][metric].get('Conservative Model', 0)
            aggressive_val = comparison['metric_comparison'][metric].get('Aggressive Model', 0)
            winner = 'Conservative' if conservative_val > aggressive_val else 'Aggressive'
            
            print(f"{metric.upper():<11} | {conservative_val:.3f}        | {aggressive_val:.3f}      | {winner}")
    
    print(f"\nOverall Winner: {comparison['recommendations']['best_overall']}")
    print(f"Rationale: {comparison['recommendations']['rationale']}")
    print()


def main():
    """Run all demonstration examples."""
    print("ModelEvaluator Demonstration")
    print("=" * 50)
    print()
    
    try:
        demonstrate_basic_evaluation()
        demonstrate_threshold_analysis()
        demonstrate_performance_report()
        demonstrate_model_comparison()
        
        print("Demo completed successfully!")
        print("\nThe ModelEvaluator provides comprehensive evaluation capabilities including:")
        print("- Basic classification metrics (precision, recall, F1, AUC)")
        print("- Confusion matrix analysis and visualization")
        print("- Threshold optimization across different operating points")
        print("- Business impact analysis with cost-benefit calculations")
        print("- Performance reports with actionable recommendations")
        print("- Model comparison and benchmarking")
        
    except Exception as e:
        print(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()