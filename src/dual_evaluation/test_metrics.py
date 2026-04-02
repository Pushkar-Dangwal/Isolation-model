"""
Unit tests for metrics calculation utilities.

This module tests the calculate_all_metrics function to ensure it correctly
computes all required evaluation metrics and handles edge cases properly.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
"""

import numpy as np
import pytest

from src.dual_evaluation.metrics import calculate_all_metrics


class TestCalculateAllMetrics:
    """Test suite for calculate_all_metrics function."""
    
    def test_basic_metrics_calculation(self):
        """Test basic metric calculation with known values."""
        # Create simple test case with known values
        # TP=2, TN=4, FP=2, FN=2
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.5, 0.7, 0.6, 0.2, 0.1])
        
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        # Verify confusion matrix (Requirement 8.3)
        assert metrics['true_positives'] == 2
        assert metrics['true_negatives'] == 4
        assert metrics['false_positives'] == 2
        assert metrics['false_negatives'] == 2
        
        # Verify confusion matrix sums to total samples (Requirement 8.5)
        cm_sum = (metrics['true_positives'] + metrics['true_negatives'] + 
                  metrics['false_positives'] + metrics['false_negatives'])
        assert cm_sum == len(y_true)
        
        # Verify precision formula (Requirement 8.6)
        # TP / (TP + FP) = 2 / (2 + 2) = 0.5
        assert metrics['precision'] == 0.5
        
        # Verify recall formula (Requirement 8.7)
        # TP / (TP + FN) = 2 / (2 + 2) = 0.5
        assert metrics['recall'] == 0.5
        
        # Verify F1-score formula (Requirement 8.8)
        # 2 * (precision * recall) / (precision + recall) = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        assert metrics['f1_score'] == 0.5
        
        # Verify accuracy (Requirement 8.1)
        # (TP + TN) / total = (2 + 4) / 10 = 0.6
        assert metrics['accuracy'] == 0.6
        
        # Verify false positive rate (Requirement 8.4)
        # FP / (FP + TN) = 2 / (2 + 4) = 0.333...
        assert abs(metrics['false_positive_rate'] - (2/6)) < 0.001
        
        # Verify ROC-AUC and PR-AUC are present (Requirement 8.2)
        assert 'roc_auc' in metrics
        assert 'pr_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1
    
    def test_edge_case_no_positive_predictions(self):
        """Test metric calculation when no positive predictions (Requirement 8.6, 8.8)."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.2])
        
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        # When no positive predictions, precision should be 0
        assert metrics['precision'] == 0.0
        # Recall should also be 0 (no true positives)
        assert metrics['recall'] == 0.0
        # F1-score should be 0
        assert metrics['f1_score'] == 0.0
        
        # Confusion matrix should still be valid
        assert metrics['true_positives'] == 0
        assert metrics['false_positives'] == 0
        assert metrics['false_negatives'] == 2
        assert metrics['true_negatives'] == 2
    
    def test_edge_case_all_positive_predictions(self):
        """Test metric calculation when all predictions are positive (Requirement 8.7)."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6])
        
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        # TP=2, FP=2, FN=0, TN=0
        assert metrics['true_positives'] == 2
        assert metrics['false_positives'] == 2
        assert metrics['false_negatives'] == 0
        assert metrics['true_negatives'] == 0
        
        # Precision = TP / (TP + FP) = 2 / 4 = 0.5
        assert metrics['precision'] == 0.5
        # Recall = TP / (TP + FN) = 2 / 2 = 1.0
        assert metrics['recall'] == 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
        assert abs(metrics['f1_score'] - (2/3)) < 0.001
    
    def test_edge_case_perfect_predictions(self):
        """Test metric calculation with perfect predictions."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        y_proba = np.array([0.9, 0.8, 0.1, 0.2, 0.95, 0.05])
        
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        # All metrics should be perfect
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
        assert metrics['false_positive_rate'] == 0.0
        
        # Confusion matrix
        assert metrics['true_positives'] == 3
        assert metrics['true_negatives'] == 3
        assert metrics['false_positives'] == 0
        assert metrics['false_negatives'] == 0
    
    def test_edge_case_no_true_negatives(self):
        """Test metric calculation when no true negatives (FPR edge case)."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.4, 0.3])
        
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        # FPR = FP / (FP + TN) should be 0 when denominator is 0
        assert metrics['false_positive_rate'] == 0.0
        
        # Other metrics should still be valid
        assert metrics['true_positives'] == 2
        assert metrics['false_negatives'] == 2
        assert metrics['recall'] == 0.5
    
    def test_confusion_matrix_invariant(self):
        """Test that confusion matrix always sums to total samples (Requirement 8.5)."""
        # Test with various random cases
        np.random.seed(42)
        
        for _ in range(10):
            n_samples = np.random.randint(50, 200)
            y_true = np.random.randint(0, 2, n_samples)
            y_pred = np.random.randint(0, 2, n_samples)
            y_proba = np.random.random(n_samples)
            
            metrics = calculate_all_metrics(y_true, y_pred, y_proba)
            
            cm_sum = (metrics['true_positives'] + metrics['true_negatives'] + 
                      metrics['false_positives'] + metrics['false_negatives'])
            
            assert cm_sum == n_samples, \
                f"Confusion matrix sum {cm_sum} != total samples {n_samples}"
    
    def test_metric_ranges(self):
        """Test that all metrics are within valid ranges (Requirement 8.1-8.4)."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
        y_proba = np.array([0.9, 0.4, 0.8, 0.2, 0.6, 0.1, 0.85, 0.55, 0.3, 0.15])
        
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        # All rate/score metrics should be in [0, 1]
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['pr_auc'] <= 1
        assert 0 <= metrics['false_positive_rate'] <= 1
        
        # Confusion matrix values should be non-negative integers
        assert metrics['true_positives'] >= 0
        assert metrics['true_negatives'] >= 0
        assert metrics['false_positives'] >= 0
        assert metrics['false_negatives'] >= 0
        assert isinstance(metrics['true_positives'], int)
        assert isinstance(metrics['true_negatives'], int)
        assert isinstance(metrics['false_positives'], int)
        assert isinstance(metrics['false_negatives'], int)
    
    def test_formula_correctness(self):
        """Test that formulas match requirements exactly (Requirements 8.6, 8.7, 8.8)."""
        # Create specific case to verify formulas
        # TP=3, TN=2, FP=1, FN=4
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.6, 0.15, 0.05])
        
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        tp = metrics['true_positives']
        tn = metrics['true_negatives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        
        # Verify confusion matrix values
        assert tp == 3
        assert tn == 2
        assert fp == 1
        assert fn == 4
        
        # Verify precision formula: TP / (TP + FP)
        expected_precision = tp / (tp + fp)
        assert abs(metrics['precision'] - expected_precision) < 0.0001
        
        # Verify recall formula: TP / (TP + FN)
        expected_recall = tp / (tp + fn)
        assert abs(metrics['recall'] - expected_recall) < 0.0001
        
        # Verify F1-score formula: 2 * (precision * recall) / (precision + recall)
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        assert abs(metrics['f1_score'] - expected_f1) < 0.0001
        
        # Verify FPR formula: FP / (FP + TN)
        expected_fpr = fp / (fp + tn)
        assert abs(metrics['false_positive_rate'] - expected_fpr) < 0.0001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
