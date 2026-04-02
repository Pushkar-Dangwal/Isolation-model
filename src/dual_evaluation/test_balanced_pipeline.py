"""
Unit tests for BalancedPipeline component.

This module tests the BalancedPipeline functionality including:
- Model training on balanced data
- Model validation
- Prediction generation
- Metrics calculation
- Threshold optimization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from .balanced_pipeline import BalancedPipeline, ModelError
from .data_models import EvaluationResult


class TestBalancedPipeline:
    """Test suite for BalancedPipeline class."""
    
    @pytest.fixture
    def sample_balanced_data(self):
        """Create sample balanced dataset."""
        np.random.seed(42)
        n_fraud = 500
        n_legitimate = 500
        n_samples = n_fraud + n_legitimate
        
        # Create fraud labels (50/50 split)
        is_fraud = np.array([1] * n_fraud + [0] * n_legitimate)
        np.random.shuffle(is_fraud)
        
        df = pd.DataFrame({
            'transaction_id': [f'T{i:06d}' for i in range(n_samples)],
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'sender_account': [f'A{i%100:04d}' for i in range(n_samples)],
            'receiver_account': [f'B{i%100:04d}' for i in range(n_samples)],
            'amount': np.random.uniform(10, 10000, n_samples),
            'transaction_type': np.random.choice(['online', 'in_store', 'atm'], n_samples),
            'merchant_category': np.random.choice(['retail', 'food', 'travel'], n_samples),
            'location': np.random.choice(['US', 'UK', 'CA'], n_samples),
            'device_used': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'is_fraud': is_fraud
        })
        
        return df
    
    def test_initialization(self):
        """Test BalancedPipeline initialization."""
        pipeline = BalancedPipeline(random_state=42, n_jobs=1)
        
        assert pipeline.random_state == 42
        assert pipeline.n_jobs == 1
        assert pipeline.model is None
    
    def test_initialization_defaults(self):
        """Test BalancedPipeline initialization with defaults."""
        pipeline = BalancedPipeline()
        
        assert pipeline.random_state == 42
        assert pipeline.n_jobs == -1
        assert pipeline.model is None
    
    def test_calculate_all_metrics_correctness(self):
        """Test metric calculation correctness."""
        pipeline = BalancedPipeline(random_state=42)
        
        # Create simple test case with known values
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.5, 0.7, 0.6, 0.2, 0.1])
        
        metrics = pipeline._calculate_all_metrics(y_true, y_pred, y_proba)
        
        # Verify confusion matrix
        assert metrics['true_positives'] == 2
        assert metrics['true_negatives'] == 4
        assert metrics['false_positives'] == 2
        assert metrics['false_negatives'] == 2
        
        # Verify precision = TP / (TP + FP) = 2 / 4 = 0.5
        assert metrics['precision'] == 0.5
        
        # Verify recall = TP / (TP + FN) = 2 / 4 = 0.5
        assert metrics['recall'] == 0.5
        
        # Verify F1 = 2 * (precision * recall) / (precision + recall) = 2 * 0.25 / 1.0 = 0.5
        assert metrics['f1_score'] == 0.5
        
        # Verify confusion matrix sums to total
        cm_sum = (metrics['true_positives'] + metrics['true_negatives'] + 
                 metrics['false_positives'] + metrics['false_negatives'])
        assert cm_sum == len(y_true)
    
    def test_calculate_all_metrics_edge_case_no_positives(self):
        """Test metric calculation when no positive predictions."""
        pipeline = BalancedPipeline(random_state=42)
        
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.2])
        
        metrics = pipeline._calculate_all_metrics(y_true, y_pred, y_proba)
        
        # When no positive predictions, precision should be 0
        assert metrics['precision'] == 0.0
        # Recall should also be 0 (no true positives)
        assert metrics['recall'] == 0.0
        # F1 should be 0
        assert metrics['f1_score'] == 0.0
    
    def test_calculate_all_metrics_edge_case_all_positives(self):
        """Test metric calculation when all predictions are positive."""
        pipeline = BalancedPipeline(random_state=42)
        
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6])
        
        metrics = pipeline._calculate_all_metrics(y_true, y_pred, y_proba)
        
        # TP=2, FP=2, FN=0, TN=0
        assert metrics['true_positives'] == 2
        assert metrics['false_positives'] == 2
        assert metrics['false_negatives'] == 0
        assert metrics['true_negatives'] == 0
        
        # Precision = 2/4 = 0.5
        assert metrics['precision'] == 0.5
        # Recall = 2/2 = 1.0
        assert metrics['recall'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
