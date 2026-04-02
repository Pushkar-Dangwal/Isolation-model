"""
Example usage of DualEvaluationPipeline.

This script demonstrates how to use the DualEvaluationPipeline to compare
fraud detection performance between imbalanced and balanced approaches.
"""

import logging
from pathlib import Path
from src.dual_evaluation import DualEvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run dual evaluation pipeline example."""
    
    # Configuration
    data_path = 'data/financial.csv'
    pretrained_model_path = 'models/fraud_detector_20260320_134826'
    output_dir = 'results/dual_evaluation'
    random_state = 42
    
    logger.info("Starting Dual Evaluation Pipeline Example")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Model path: {pretrained_model_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize pipeline
    pipeline = DualEvaluationPipeline(
        data_path=data_path,
        pretrained_model_path=pretrained_model_path,
        output_dir=output_dir,
        random_state=random_state
    )
    
    # Run complete evaluation
    logger.info("\nRunning dual evaluation...")
    results = pipeline.run_evaluation()
    
    # Display summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    imb_results = results['imbalanced_results']
    bal_results = results['balanced_results']
    
    logger.info("\nImbalanced Pipeline Results:")
    logger.info(f"  Test Samples: {imb_results['test_samples']:,}")
    logger.info(f"  Fraud Rate: {imb_results['test_fraud_rate']:.4f}")
    logger.info(f"  Accuracy: {imb_results['accuracy']:.4f}")
    logger.info(f"  Precision: {imb_results['precision']:.4f}")
    logger.info(f"  Recall: {imb_results['recall']:.4f}")
    logger.info(f"  F1-Score: {imb_results['f1_score']:.4f}")
    logger.info(f"  ROC-AUC: {imb_results['roc_auc']:.4f}")
    
    logger.info("\nBalanced Pipeline Results:")
    logger.info(f"  Test Samples: {bal_results['test_samples']:,}")
    logger.info(f"  Fraud Rate: {bal_results['test_fraud_rate']:.4f}")
    logger.info(f"  Accuracy: {bal_results['accuracy']:.4f}")
    logger.info(f"  Precision: {bal_results['precision']:.4f}")
    logger.info(f"  Recall: {bal_results['recall']:.4f}")
    logger.info(f"  F1-Score: {bal_results['f1_score']:.4f}")
    logger.info(f"  ROC-AUC: {bal_results['roc_auc']:.4f}")
    
    logger.info("\nKey Differences:")
    metric_diffs = results['metric_differences']
    logger.info(f"  Precision Change: {metric_diffs.get('precision', 0):+.4f}")
    logger.info(f"  Recall Change: {metric_diffs.get('recall', 0):+.4f}")
    logger.info(f"  F1-Score Change: {metric_diffs.get('f1_score', 0):+.4f}")
    
    logger.info("\nRecommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        logger.info(f"  {i}. {rec}")
    
    logger.info(f"\nOutputs saved to: {output_dir}")
    logger.info(f"  - comparison_table.csv")
    logger.info(f"  - comparison_report.json")
    logger.info(f"  - interpretation.txt")
    logger.info(f"  - recommendations.txt")
    logger.info(f"  - Visualizations: {len(results['visualization_paths'])} files")
    
    logger.info("\n" + "=" * 80)
    logger.info("DUAL EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
