"""
Calculate metrics by loading and running the actual fraud detection model.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fraud_detector import FraudDetector
from model_evaluator import ModelEvaluator
from config import setup_logging, DATA_DIR

# Set up logging
setup_logging('INFO')
logger = logging.getLogger(__name__)


def load_test_data(data_path: str, sample_size: int = 10000):
    """Load a sample of test data."""
    logger.info(f"Loading test data from {data_path}")
    
    dtype_mapping = {
        'transaction_id': 'string',
        'sender_account': 'string',
        'receiver_account': 'string',
        'transaction_type': 'category',
        'merchant_category': 'category',
        'location': 'category',
        'device_used': 'category',
        'amount': 'float32',
        'is_fraud': 'int8'
    }
    
    # Load sample
    df = pd.read_csv(data_path, dtype=dtype_mapping, nrows=sample_size)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    logger.info(f"Loaded {len(df)} transactions")
    logger.info(f"Fraud rate: {df['is_fraud'].mean():.1%}")
    
    return df


def calculate_metrics_from_model(model_path: str, test_data: pd.DataFrame):
    """Load model and calculate metrics on test data."""
    
    logger.info(f"Loading model from {model_path}")
    
    # Load the trained model
    detector = FraudDetector()
    detector.load_model(model_path)
    
    logger.info("Model loaded successfully")
    
    # Get predictions
    logger.info("Generating predictions...")
    predictions = detector.predict(
        test_data,
        return_probabilities=True,
        return_risk_levels=True
    )
    
    # Extract true labels and predictions
    y_true = test_data['is_fraud'].values
    y_proba = predictions['fraud_probability'].values
    # Convert probabilities to binary predictions using threshold 0.5
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Calculate comprehensive metrics
    logger.info("Calculating metrics...")
    metrics = evaluator.calculate_comprehensive_metrics(
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.5
    )
    
    return metrics, y_true, y_proba


def display_metrics(metrics: dict):
    """Display metrics in a formatted way."""
    
    print("\n" + "="*60)
    print("FRAUD DETECTION MODEL METRICS (FROM MODEL RUN)")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Value':<15} {'Why Needed'}")
    print("-"*60)
    print(f"{'Accuracy':<20} {metrics['dataset_stats']['fraud_rate']:.4f} ({metrics['dataset_stats']['fraud_rate']*100:.2f}%)  Basic metric")
    print(f"{'Precision':<20} {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)  How correct fraud predictions are")
    print(f"{'Recall':<20} {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)  How many frauds you catch")
    print(f"{'F1-Score':<20} {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)  Balance of precision & recall")
    
    print("\n" + "="*60)
    print("ADDITIONAL METRICS")
    print("="*60)
    print(f"{'PR-AUC':<20} {metrics['pr_auc']:.4f}        Overall fraud detection quality")
    print(f"{'ROC-AUC':<20} {metrics['roc_auc']:.4f}        Classification performance")
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"True Positives:  {cm['true_positives']:,} (Frauds correctly detected)")
    print(f"False Positives: {cm['false_positives']:,} (Legitimate flagged as fraud)")
    print(f"True Negatives:  {cm['true_negatives']:,} (Legitimate correctly identified)")
    print(f"False Negatives: {cm['false_negatives']:,} (Frauds missed)")
    
    # Business Impact
    print("\n" + "="*60)
    print("BUSINESS IMPACT")
    print("="*60)
    print(f"Fraud Detection Rate:    {metrics['fraud_detection_rate']*100:.1f}%")
    print(f"False Positive Rate:     {metrics['false_positive_rate']*100:.1f}%")
    print(f"Customer Friction Rate:  {metrics['customer_friction_rate']*100:.1f}%")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main execution."""
    
    # Paths
    model_path = "models/production_fraud_detector_v1_optimized"
    data_path = DATA_DIR / "financial.csv"
    
    try:
        # Load test data
        print("Loading test data...")
        test_data = load_test_data(str(data_path), sample_size=100000)
        
        # Calculate metrics from model
        print("Running model and calculating metrics...")
        metrics, y_true, y_proba = calculate_metrics_from_model(model_path, test_data)
        
        # Display results
        display_metrics(metrics)
        
        # Additional analysis
        print("INTERPRETATION:")
        print("-" * 60)
        
        if metrics['precision'] < 0.1:
            print("⚠ Very low precision - model flags too many legitimate transactions")
        if metrics['recall'] > 0.95:
            print("✓ Excellent recall - catching almost all frauds")
        if metrics['f1_score'] < 0.3:
            print("⚠ Low F1-score - poor balance between precision and recall")
        if metrics['pr_auc'] < 0.6:
            print("⚠ Low PR-AUC - model needs improvement")
        
        print("\nRECOMMENDATION:")
        if metrics['precision'] < 0.1 and metrics['recall'] > 0.9:
            print("Consider increasing the decision threshold to reduce false positives")
            print("This will improve precision but may slightly reduce recall")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure the model exists at: models/production_fraud_detector_v1_optimized")
        print("And the data file exists at: data/financial.csv")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError occurred: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
