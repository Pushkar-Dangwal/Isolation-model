"""
Tests for the ModelEvaluator class.
Tests comprehensive evaluation metrics and performance analysis functionality.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch

# Set matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test cases for the ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data for testing
        np.random.seed(42)
        self.n_samples = 1000
        self.fraud_rate = 0.04
        
        # Generate realistic fraud detection scenario
        self.y_true = np.random.choice([0, 1], self.n_samples, p=[1-self.fraud_rate, self.fraud_rate])
        
        # Generate probabilities that are somewhat correlated with true labels
        fraud_mask = self.y_true == 1
        legitimate_mask = self.y_true == 0
        
        # Fraud cases get higher probabilities (but not perfect)
        fraud_probs = np.random.beta(3, 2, np.sum(fraud_mask))  # Skewed towards higher values
        legitimate_probs = np.random.beta(1, 4, np.sum(legitimate_mask))  # Skewed towards lower values
        
        self.y_proba = np.zeros(self.n_samples)
        self.y_proba[fraud_mask] = fraud_probs
        self.y_proba[legitimate_mask] = legitimate_probs
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(save_plots=False)
    
    def test_model_evaluator_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(figsize=(10, 6), style='darkgrid', save_plots=True, plot_dir='test_plots')
        
        assert evaluator.figsize == (10, 6)
        assert evaluator.style == 'darkgrid'
        assert evaluator.save_plots is True
        assert evaluator.plot_dir == 'test_plots'
    
    def test_calculate_pr_auc(self):
        """Test PR-AUC calculation."""
        pr_auc = self.evaluator.calculate_pr_auc(self.y_true, self.y_proba)
        
        # PR-AUC should be a float between 0 and 1
        assert isinstance(pr_auc, float)
        assert 0 <= pr_auc <= 1
        
        # Should be better than random (fraud rate)
        assert pr_auc > self.fraud_rate
    
    def test_calculate_pr_auc_with_invalid_input(self):
        """Test PR-AUC calculation with invalid input."""
        # Test with None inputs
        with pytest.raises(ValueError, match="cannot be None"):
            self.evaluator.calculate_pr_auc(None, self.y_proba)
        
        # Test with empty arrays
        with pytest.raises(ValueError, match="cannot be empty"):
            self.evaluator.calculate_pr_auc(np.array([]), np.array([]))
        
        # Test with mismatched lengths
        with pytest.raises(ValueError, match="must have same length"):
            self.evaluator.calculate_pr_auc(self.y_true[:100], self.y_proba)
        
        # Test with invalid probability range
        invalid_proba = np.array([0.5, 1.5, -0.1])
        y_true_small = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="must be in \\[0, 1\\] range"):
            self.evaluator.calculate_pr_auc(y_true_small, invalid_proba)
    
    def test_calculate_precision_recall_f1(self):
        """Test precision, recall, and F1-score calculation."""
        y_pred = (self.y_proba >= 0.5).astype(int)
        metrics = self.evaluator.calculate_precision_recall_f1(self.y_true, y_pred)
        
        # Check that all metrics are present and valid
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        for metric_name, value in metrics.items():
            assert isinstance(value, float)
            assert 0 <= value <= 1
    
    def test_calculate_precision_recall_f1_with_invalid_input(self):
        """Test precision/recall/F1 calculation with invalid input."""
        # Test with invalid binary labels
        invalid_y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 1])
        
        with pytest.raises(ValueError, match="must contain only 0 and 1 values"):
            self.evaluator.calculate_precision_recall_f1(invalid_y_true, y_pred)
    
    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation."""
        y_pred = (self.y_proba >= 0.5).astype(int)
        cm = self.evaluator.generate_confusion_matrix(self.y_true, y_pred)
        
        # Should be 2x2 matrix for binary classification
        assert cm.shape == (2, 2)
        
        # All values should be non-negative integers
        assert np.all(cm >= 0)
        assert cm.dtype in [np.int32, np.int64]
        
        # Sum should equal total samples
        assert np.sum(cm) == len(self.y_true)
    
    def test_generate_confusion_matrix_normalized(self):
        """Test normalized confusion matrix generation."""
        y_pred = (self.y_proba >= 0.5).astype(int)
        cm_norm = self.evaluator.generate_confusion_matrix(self.y_true, y_pred, normalize='true')
        
        # Should be 2x2 matrix
        assert cm_norm.shape == (2, 2)
        
        # Normalized values should be between 0 and 1
        assert np.all(cm_norm >= 0)
        assert np.all(cm_norm <= 1)
        
        # Each row should sum to 1 (approximately, due to floating point)
        row_sums = np.sum(cm_norm, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
    
    def test_visualize_confusion_matrix(self):
        """Test confusion matrix visualization."""
        y_pred = (self.y_proba >= 0.5).astype(int)
        
        with patch('matplotlib.pyplot.show'):  # Prevent actual plot display
            fig = self.evaluator.visualize_confusion_matrix(self.y_true, y_pred)
        
        # Should return a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
    
    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = self.evaluator.calculate_comprehensive_metrics(self.y_true, self.y_proba)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'precision', 'recall', 'f1_score', 'pr_auc', 'roc_auc',
            'confusion_matrix', 'false_positive_rate', 'true_positive_rate',
            'fraud_detection_rate', 'customer_friction_rate', 'dataset_stats'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check confusion matrix structure
        cm = metrics['confusion_matrix']
        assert 'true_negatives' in cm
        assert 'false_positives' in cm
        assert 'false_negatives' in cm
        assert 'true_positives' in cm
        
        # Check dataset stats
        stats = metrics['dataset_stats']
        assert stats['total_transactions'] == len(self.y_true)
        assert stats['total_fraud'] == np.sum(self.y_true)
        assert abs(stats['fraud_rate'] - np.mean(self.y_true)) < 1e-10
    
    def test_plot_precision_recall_curve(self):
        """Test precision-recall curve plotting."""
        with patch('matplotlib.pyplot.show'):  # Prevent actual plot display
            fig = self.evaluator.plot_precision_recall_curve(self.y_true, self.y_proba)
        
        # Should return a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
    
    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        with patch('matplotlib.pyplot.show'):  # Prevent actual plot display
            fig = self.evaluator.plot_roc_curve(self.y_true, self.y_proba)
        
        # Should return a matplotlib figure
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
    
    def test_create_evaluation_report(self):
        """Test comprehensive evaluation report creation."""
        with patch('matplotlib.pyplot.show'):  # Prevent actual plot display
            report = self.evaluator.create_evaluation_report(
                self.y_true, self.y_proba, 
                model_name="Test Model",
                include_plots=True
            )
        
        # Check report structure
        assert 'model_name' in report
        assert 'evaluation_summary' in report
        assert 'performance_metrics' in report
        assert 'plots' in report
        assert 'recommendations' in report
        assert 'business_impact' in report
        
        assert report['model_name'] == "Test Model"
        
        # Check that plots were generated
        if 'error' not in report['plots']:
            assert 'confusion_matrix' in report['plots']
            assert 'precision_recall_curve' in report['plots']
            assert 'roc_curve' in report['plots']
        
        # Clean up any plots
        for plot_name, fig in report['plots'].items():
            if isinstance(fig, plt.Figure):
                plt.close(fig)
    
    def test_analyze_threshold_performance(self):
        """Test threshold performance analysis."""
        # Use a smaller set of thresholds for faster testing
        thresholds = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        analysis = self.evaluator.analyze_threshold_performance(
            self.y_true, self.y_proba, thresholds=thresholds
        )
        
        # Check analysis structure
        assert 'threshold_analysis' in analysis
        assert 'optimal_thresholds' in analysis
        assert 'recommendations' in analysis
        assert 'summary_statistics' in analysis
        
        # Check that we have results for all thresholds
        threshold_results = analysis['threshold_analysis']
        assert len(threshold_results) == len(thresholds)
        
        # Check that each threshold result has expected metrics
        for result in threshold_results:
            assert 'threshold' in result
            assert 'precision' in result
            assert 'recall' in result
            assert 'f1_score' in result
    
    def test_generate_performance_report(self):
        """Test comprehensive performance report generation."""
        with patch('matplotlib.pyplot.show'):  # Prevent actual plot display
            report = self.evaluator.generate_performance_report(
                self.y_true, self.y_proba,
                model_name="Test Performance Model",
                include_threshold_analysis=True,
                include_business_metrics=True
            )
        
        # Check report structure
        expected_sections = [
            'model_name', 'report_metadata', 'current_performance',
            'threshold_analysis', 'business_impact', 'recommendations',
            'monitoring_guidance', 'executive_summary'
        ]
        
        for section in expected_sections:
            assert section in report
        
        assert report['model_name'] == "Test Performance Model"
        
        # Check that threshold analysis was included
        assert 'threshold_analysis' in report['threshold_analysis']
        
        # Check that business impact was included
        assert 'financial_impact' in report['business_impact']
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock results for two models
        model1_results = {
            'performance_metrics': {
                'pr_auc': 0.75,
                'roc_auc': 0.85,
                'f1_score': 0.60,
                'precision': 0.70,
                'recall': 0.55
            }
        }
        
        model2_results = {
            'performance_metrics': {
                'pr_auc': 0.80,
                'roc_auc': 0.82,
                'f1_score': 0.65,
                'precision': 0.65,
                'recall': 0.65
            }
        }
        
        model_results = {
            'Model 1': model1_results,
            'Model 2': model2_results
        }
        
        comparison = self.evaluator.compare_models(model_results)
        
        # Check comparison structure
        assert 'models_compared' in comparison
        assert 'comparison_metrics' in comparison
        assert 'metric_comparison' in comparison
        assert 'ranking' in comparison
        assert 'recommendations' in comparison
        
        # Check that both models are included
        assert 'Model 1' in comparison['models_compared']
        assert 'Model 2' in comparison['models_compared']
        
        # Check that rankings exist
        assert 'overall' in comparison['ranking']
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with all fraud cases
        y_true_all_fraud = np.ones(100)
        y_proba_all_fraud = np.random.uniform(0.5, 1.0, 100)
        
        metrics = self.evaluator.calculate_comprehensive_metrics(y_true_all_fraud, y_proba_all_fraud)
        assert metrics['dataset_stats']['fraud_rate'] == 1.0
        
        # Test with no fraud cases
        y_true_no_fraud = np.zeros(100)
        y_proba_no_fraud = np.random.uniform(0.0, 0.5, 100)
        
        metrics = self.evaluator.calculate_comprehensive_metrics(y_true_no_fraud, y_proba_no_fraud)
        assert metrics['dataset_stats']['fraud_rate'] == 0.0
        
        # Test with perfect predictions
        y_true_perfect = np.array([0, 0, 1, 1])
        y_proba_perfect = np.array([0.1, 0.2, 0.8, 0.9])
        
        metrics = self.evaluator.calculate_comprehensive_metrics(y_true_perfect, y_proba_perfect, threshold=0.5)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_sample_weights(self):
        """Test functionality with sample weights."""
        # Create sample weights (higher weight for fraud cases)
        sample_weights = np.ones(len(self.y_true))
        sample_weights[self.y_true == 1] = 2.0
        
        # Test PR-AUC with weights
        pr_auc_weighted = self.evaluator.calculate_pr_auc(self.y_true, self.y_proba, sample_weights)
        pr_auc_unweighted = self.evaluator.calculate_pr_auc(self.y_true, self.y_proba)
        
        # Both should be valid
        assert 0 <= pr_auc_weighted <= 1
        assert 0 <= pr_auc_unweighted <= 1
        
        # Test comprehensive metrics with weights
        metrics_weighted = self.evaluator.calculate_comprehensive_metrics(
            self.y_true, self.y_proba, sample_weight=sample_weights
        )
        
        # Should have all expected metrics
        assert 'precision' in metrics_weighted
        assert 'recall' in metrics_weighted
        assert 'pr_auc' in metrics_weighted


if __name__ == '__main__':
    pytest.main([__file__])