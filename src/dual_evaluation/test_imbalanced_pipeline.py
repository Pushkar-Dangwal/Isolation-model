"""
Unit tests for ImbalancedPipeline component.

This module tests the ImbalancedPipeline functionality including:
- Pretrained model loading
- Model validation
- Prediction generation
- Metrics calculation
- Threshold optimization
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from pathlib import Path
import joblib

from .imbalanced_pipeline import ImbalancedPipeline, ModelError
from .data_models import EvaluationResult


class MockFraudDetector:
    """Mock FraudDetector for testing."""
    
    def __init__(self, is_fitted=True):
        self.is_fitted = is_fitted
        self.preprocessor = None
        self.feature_engineer = None
        self.anomaly_detector = None
        self.classifier = None
        self.risk_scorer = None
        self.feature_names = ['feature1', 'feature2']
        self.column_mapping = {}
        self.training_metadata = {}
        self.performance_metrics = {}
        self.random_state = 42
    
    def predict(self, df, return_probabilities=True, return_risk_levels=False, return_explanations=False):
        """Mock predict method."""
        n = len(df)
        
        # Generate mock predictions
        # Use is_fraud if available for realistic predictions
        if 'is_fraud' in df.columns:
            # Generate probabilities correlated with ground truth
            y_true = df['is_fraud'].values
            # Add noise to make it realistic
            y_proba = y_true * 0.7 + np.random.uniform(0, 0.3, n)
            y_proba = np.clip(y_proba, 0, 1)
        else:
            y_proba = np.random.uniform(0, 1, n)
        
        y_pred = (y_proba >= 0.5).astype(int)
        
        return pd.DataFrame({
            'fraud_prediction': y_pred,
            'fraud_probability': y_proba
        })


class TestImbalancedPipeline:
    """Test suite for ImbalancedPipeline class."""
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_fraud = 100  # 10% fraud rate (imbalanced)
        
        # Create fraud labels
        is_fraud = np.array([1] * n_fraud + [0] * (n_samples - n_fraud))
        np.random.shuffle(is_fraud)
        
        df = pd.DataFrame({
            'transaction_id': [f'T{i:06d}' for i in range(n_samples)],
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'sender_account': [f'A{i%100:04d}' for i in range(n_samples)],
            'receiver_account': [f'B{i%100:04d}' for i in range(n_samples)],
            'amount': np.random.uniform(10, 10000, n_samples),
            'is_fraud': is_fraud
        })
        
        return df
    
    @pytest.fixture
    def mock_model_file(self):
        """Create temporary mock model file."""
        # Create mock model
        mock_model = MockFraudDetector(is_fitted=True)
        
        # Save as dictionary (common format)
        model_data = {
            'preprocessor': mock_model.preprocessor,
            'feature_engineer': mock_model.feature_engineer,
            'anomaly_detector': mock_model.anomaly_detector,
            'classifier': mock_model.classifier,
            'risk_scorer': mock_model.risk_scorer,
            'is_fitted': mock_model.is_fitted,
            'feature_names': mock_model.feature_names,
            'column_mapping': mock_model.column_mapping,
            'training_metadata': mock_model.training_metadata,
            'performance_metrics': mock_model.performance_metrics,
            'random_state': mock_model.random_state
        }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.joblib', delete=False) as f:
            joblib.dump(model_data, f.name)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def mock_model_dir(self, mock_model_file):
        """Create temporary directory with mock model file."""
        temp_dir = tempfile.mkdtemp()
        
        # Copy model file to directory
        dest_path = Path(temp_dir) / 'model.joblib'
        import shutil
        shutil.copy(mock_model_file, dest_path)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ImbalancedPipeline initialization."""
        pipeline = ImbalancedPipeline(model_path='test_model.joblib', random_state=42)
        
        assert pipeline.model_path == 'test_model.joblib'
        assert pipeline.random_state == 42
        assert pipeline.model is None

    def test_load_pretrained_model_success_file(self, mock_model_file):
        """Test successful model loading from file."""
        pipeline = ImbalancedPipeline(model_path=mock_model_file)
        model = pipeline.load_pretrained_model()
        
        assert model is not None
        assert pipeline.model is not None
        assert pipeline.model.is_fitted is True
    
    def test_load_pretrained_model_file_not_found(self):
        """Test model loading with non-existent file raises FileNotFoundError."""
        pipeline = ImbalancedPipeline(model_path='nonexistent_model.joblib')
        
        with pytest.raises(FileNotFoundError, match="not found"):
            pipeline.load_pretrained_model()
    
    def test_evaluate_success(self, mock_model_file, sample_test_data):
        """Test successful model evaluation."""
        pipeline = ImbalancedPipeline(model_path=mock_model_file)
        pipeline.load_pretrained_model()
        
        # Replace with mock that returns predictions
        pipeline.model = MockFraudDetector(is_fitted=True)
        
        result = pipeline.evaluate(sample_test_data)
        
        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "pretrained_imbalanced"
        assert result.dataset_type == "imbalanced"
        assert result.test_samples == len(sample_test_data)
        
        # Verify metrics are calculated
        assert 0 <= result.accuracy <= 1
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1
        assert 0 <= result.f1_score <= 1
        
        # Verify confusion matrix sums to total
        cm_sum = (result.true_positives + result.true_negatives + 
                 result.false_positives + result.false_negatives)
        assert cm_sum == len(sample_test_data)
    
    def test_calculate_all_metrics_correctness(self, mock_model_file):
        """Test metric calculation correctness."""
        pipeline = ImbalancedPipeline(model_path=mock_model_file)
        
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
