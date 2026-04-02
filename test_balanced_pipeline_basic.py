"""
Basic integration test for BalancedPipeline.

This script verifies that the BalancedPipeline can be instantiated and
its basic methods work correctly without requiring actual model training.
"""

import pandas as pd
import numpy as np
from src.dual_evaluation import BalancedPipeline

def test_balanced_pipeline_instantiation():
    """Test that BalancedPipeline can be instantiated."""
    print("Testing BalancedPipeline instantiation...")
    
    pipeline = BalancedPipeline(random_state=42, n_jobs=1)
    
    assert pipeline.random_state == 42
    assert pipeline.n_jobs == 1
    assert pipeline.model is None
    
    print("✓ BalancedPipeline instantiation successful")


def test_balanced_pipeline_metrics_calculation():
    """Test that metrics calculation works correctly."""
    print("\nTesting metrics calculation...")
    
    pipeline = BalancedPipeline(random_state=42)
    
    # Create simple test case
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.5, 0.7, 0.6, 0.2, 0.1])
    
    metrics = pipeline._calculate_all_metrics(y_true, y_pred, y_proba)
    
    # Verify metrics
    assert metrics['true_positives'] == 2
    assert metrics['true_negatives'] == 4
    assert metrics['false_positives'] == 2
    assert metrics['false_negatives'] == 2
    assert metrics['precision'] == 0.5
    assert metrics['recall'] == 0.5
    assert metrics['f1_score'] == 0.5
    
    print("✓ Metrics calculation successful")
    print(f"  - Precision: {metrics['precision']:.3f}")
    print(f"  - Recall: {metrics['recall']:.3f}")
    print(f"  - F1-score: {metrics['f1_score']:.3f}")


def test_balanced_pipeline_error_handling():
    """Test that error handling works correctly."""
    print("\nTesting error handling...")
    
    pipeline = BalancedPipeline(random_state=42)
    
    # Test evaluate without trained model
    try:
        test_df = pd.DataFrame({
            'transaction_id': ['T001', 'T002'],
            'is_fraud': [1, 0]
        })
        pipeline.evaluate(test_df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Model not trained" in str(e)
        print("✓ Error handling for untrained model works correctly")
    
    # Test train_model with empty data
    try:
        empty_df = pd.DataFrame()
        pipeline.train_model(empty_df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()
        print("✓ Error handling for empty data works correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("BalancedPipeline Basic Integration Tests")
    print("=" * 60)
    
    test_balanced_pipeline_instantiation()
    test_balanced_pipeline_metrics_calculation()
    test_balanced_pipeline_error_handling()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
