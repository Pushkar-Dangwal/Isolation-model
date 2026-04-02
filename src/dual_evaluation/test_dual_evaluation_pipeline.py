"""
Unit tests for DualEvaluationPipeline orchestrator.

This module tests the main orchestrator that coordinates the complete dual-evaluation
workflow.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from .dual_evaluation_pipeline import DualEvaluationPipeline
from .data_models import EvaluationResult, ComparisonReport


class TestDualEvaluationPipelineInitialization:
    """Test DualEvaluationPipeline initialization."""
    
    def test_initialization_creates_components(self, tmp_path):
        """Test that initialization creates all required components."""
        pipeline = DualEvaluationPipeline(
            data_path='data/test.csv',
            pretrained_model_path='models/test_model',
            output_dir=str(tmp_path),
            random_state=42
        )
        
        assert pipeline.data_path == 'data/test.csv'
        assert pipeline.pretrained_model_path == 'models/test_model'
        assert pipeline.output_dir == str(tmp_path)
        assert pipeline.random_state == 42
        
        # Verify components are initialized
        assert pipeline.data_loader is not None
        assert pipeline.imbalanced_pipeline is not None
        assert pipeline.balanced_pipeline is not None
        assert pipeline.metrics_comparator is not None
        
        # Verify comparison report is initially None
        assert pipeline.comparison_report is None
    
    def test_initialization_creates_output_directory(self, tmp_path):
        """Test that initialization creates output directory if it doesn't exist."""
        output_dir = tmp_path / 'new_output_dir'
        assert not output_dir.exists()
        
        pipeline = DualEvaluationPipeline(
            data_path='data/test.csv',
            pretrained_model_path='models/test_model',
            output_dir=str(output_dir),
            random_state=42
        )
        
        assert output_dir.exists()
        assert output_dir.is_dir()


class TestGetComparisonReport:
    """Test get_comparison_report method."""
    
    def test_get_comparison_report_raises_error_when_not_run(self, tmp_path):
        """Test that get_comparison_report raises error if run_evaluation not called."""
        pipeline = DualEvaluationPipeline(
            data_path='data/test.csv',
            pretrained_model_path='models/test_model',
            output_dir=str(tmp_path),
            random_state=42
        )
        
        with pytest.raises(ValueError, match="Comparison report not available"):
            pipeline.get_comparison_report()


class TestHelperMethods:
    """Test helper methods."""
    
    def test_evaluation_result_to_dict(self, tmp_path):
        """Test conversion of EvaluationResult to dictionary."""
        pipeline = DualEvaluationPipeline(
            data_path='data/test.csv',
            pretrained_model_path='models/test_model',
            output_dir=str(tmp_path),
            random_state=42
        )
        
        result = EvaluationResult(
            model_name='test_model',
            dataset_type='imbalanced',
            train_samples=1000,
            test_samples=200,
            train_fraud_rate=0.036,
            test_fraud_rate=0.035,
            accuracy=0.95,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            roc_auc=0.90,
            pr_auc=0.85,
            true_positives=50,
            true_negatives=140,
            false_positives=10,
            false_negatives=0,
            fraud_detection_rate=0.75,
            false_positive_rate=0.05,
            customer_friction_rate=0.05,
            optimal_threshold=0.5,
            threshold_range=(0.0, 1.0),
            training_time=None,
            evaluation_time=10.5
        )
        
        result_dict = pipeline._evaluation_result_to_dict(result)
        
        assert result_dict['model_name'] == 'test_model'
        assert result_dict['dataset_type'] == 'imbalanced'
        assert result_dict['accuracy'] == 0.95
        assert result_dict['precision'] == 0.80
        assert result_dict['recall'] == 0.75
    
    def test_calculate_metric_differences(self, tmp_path):
        """Test calculation of metric differences from comparison table."""
        pipeline = DualEvaluationPipeline(
            data_path='data/test.csv',
            pretrained_model_path='models/test_model',
            output_dir=str(tmp_path),
            random_state=42
        )
        
        comparison_table = pd.DataFrame([
            {'metric_name': 'accuracy', 'imbalanced_value': 0.95, 'balanced_value': 0.93, 'difference': -0.02, 'percent_change': -2.1},
            {'metric_name': 'precision', 'imbalanced_value': 0.80, 'balanced_value': 0.85, 'difference': 0.05, 'percent_change': 6.25},
            {'metric_name': 'recall', 'imbalanced_value': 0.75, 'balanced_value': 0.90, 'difference': 0.15, 'percent_change': 20.0}
        ])
        
        differences = pipeline._calculate_metric_differences(comparison_table)
        
        assert differences['accuracy'] == -0.02
        assert differences['precision'] == 0.05
        assert differences['recall'] == 0.15
    
    def test_generate_recommendations(self, tmp_path):
        """Test generation of recommendations."""
        pipeline = DualEvaluationPipeline(
            data_path='data/test.csv',
            pretrained_model_path='models/test_model',
            output_dir=str(tmp_path),
            random_state=42
        )
        
        trade_offs = {
            'f1_improvement': True,
            'precision_improvement': True,
            'recall_improvement': True,
            'lower_customer_friction': True,
            'training_data_reduction': 0.85
        }
        
        comparison_table = pd.DataFrame([
            {'metric_name': 'f1_score', 'imbalanced_value': 0.77, 'balanced_value': 0.87, 'difference': 0.10, 'percent_change': 13.0}
        ])
        
        recommendations = pipeline._generate_recommendations(trade_offs, comparison_table)
        
        assert len(recommendations) > 0
        assert any('balanced approach' in rec.lower() for rec in recommendations)
        assert any('training data' in rec.lower() for rec in recommendations)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
