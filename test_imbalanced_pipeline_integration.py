"""
Integration test for ImbalancedPipeline with actual model structure.

This script tests the ImbalancedPipeline component with a real model file
to verify it can load and evaluate pretrained models correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.dual_evaluation import ImbalancedPipeline, DataLoader

def test_imbalanced_pipeline_integration():
    """Test ImbalancedPipeline with actual model structure."""
    
    # Find a pretrained model
    model_dir = Path('models/models/fraud_detector_20260320_134826')
    
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        print("Skipping integration test")
        return
    
    print(f"Testing with model: {model_dir}")
    
    # Initialize pipeline
    pipeline = ImbalancedPipeline(model_path=str(model_dir), random_state=42)
    
    # Load pretrained model
    print("Loading pretrained model...")
    try:
        model = pipeline.load_pretrained_model()
        print(f"✓ Model loaded successfully")
        print(f"  - is_fitted: {model.is_fitted}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Load test data
    print("\nLoading test data...")
    data_loader = DataLoader(random_state=42)
    
    try:
        full_df = data_loader.load_full_dataset('data/financial.csv')
        print(f"✓ Loaded {len(full_df):,} transactions")
        
        # Use time-based split to get test data
        train_df, test_df = data_loader.time_based_split(full_df, test_size=0.2)
        print(f"✓ Split data: train={len(train_df):,}, test={len(test_df):,}")
        
        # Use a smaller subset for quick testing
        test_subset = test_df.head(1000)
        print(f"  Using subset of {len(test_subset):,} samples for testing")
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return
    
    # Evaluate model
    print("\nEvaluating model...")
    try:
        result = pipeline.evaluate(test_subset)
        print(f"✓ Evaluation complete")
        print(f"\nResults:")
        print(f"  - Test samples: {result.test_samples:,}")
        print(f"  - Test fraud rate: {result.test_fraud_rate:.4f}")
        print(f"  - Accuracy: {result.accuracy:.4f}")
        print(f"  - Precision: {result.precision:.4f}")
        print(f"  - Recall: {result.recall:.4f}")
        print(f"  - F1-score: {result.f1_score:.4f}")
        print(f"  - ROC-AUC: {result.roc_auc:.4f}")
        print(f"  - FPR: {result.false_positive_rate:.4f}")
        print(f"  - Evaluation time: {result.evaluation_time:.2f}s")
        
    except Exception as e:
        print(f"✗ Failed to evaluate: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test threshold optimization
    print("\nOptimizing threshold...")
    try:
        predictions = model.predict(test_subset, return_probabilities=True)
        y_proba = predictions['fraud_probability'].values
        
        threshold, optimized_result = pipeline.optimize_threshold(
            test_subset,
            y_proba,
            target_recall_min=0.70,
            target_recall_max=0.85,
            max_fpr=0.05
        )
        
        print(f"✓ Threshold optimization complete")
        print(f"\nOptimized Results:")
        print(f"  - Optimal threshold: {threshold:.4f}")
        print(f"  - Recall: {optimized_result.recall:.4f}")
        print(f"  - Precision: {optimized_result.precision:.4f}")
        print(f"  - FPR: {optimized_result.false_positive_rate:.4f}")
        
    except Exception as e:
        print(f"✗ Failed to optimize threshold: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ All integration tests passed!")


if __name__ == '__main__':
    test_imbalanced_pipeline_integration()
