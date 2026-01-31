"""
Integration tests for the FraudDetector main class.
Tests the complete pipeline integration and error handling.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detector import FraudDetector
from error_handling import DataValidationError, ModelError, PipelineError


class TestFraudDetectorIntegration:
    """Integration tests for the complete FraudDetector pipeline."""
    
    def create_sample_data(self, n_samples=100):
        """Create sample transaction data for testing."""
        np.random.seed(42)
        
        data = {
            'transaction_id': [f'tx_{i}' for i in range(n_samples)],
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h'),
            'sender_account': [f'sender_{i % 20}' for i in range(n_samples)],
            'receiver_account': [f'receiver_{i % 30}' for i in range(n_samples)],
            'amount': np.random.lognormal(4, 1, n_samples),
            'transaction_type': np.random.choice(['transfer', 'payment', 'withdrawal'], n_samples),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'online'], n_samples),
            'location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
            'device_used': np.random.choice(['mobile', 'web', 'atm'], n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.96, 0.04])
        }
        
        return pd.DataFrame(data)
    
    def test_fraud_detector_initialization(self):
        """Test FraudDetector initialization with different configurations."""
        # Test with error handling enabled
        detector_with_eh = FraudDetector(enable_error_handling=True)
        assert detector_with_eh.enable_error_handling is True
        assert detector_with_eh.error_handler is not None
        
        # Test with error handling disabled
        detector_without_eh = FraudDetector(enable_error_handling=False)
        assert detector_without_eh.enable_error_handling is False
        assert detector_without_eh.error_handler is None
    
    def test_fraud_detector_system_health(self):
        """Test system health reporting."""
        detector = FraudDetector(enable_error_handling=True)
        
        health = detector.get_system_health()
        
        # Check basic health structure
        assert 'overall_status' in health
        assert 'pipeline_fitted' in health
        assert 'components_status' in health
        assert 'error_handling' in health
        
        # Should be not ready since not fitted
        assert health['overall_status'] == 'not_ready'
        assert health['pipeline_fitted'] is False
        
        # Error handling should be enabled
        assert health['error_handling']['enabled'] is True
        assert health['error_handling']['status'] == 'active'
    
    def test_fraud_detector_model_info(self):
        """Test model info reporting."""
        detector = FraudDetector(enable_error_handling=True)
        
        info = detector.get_model_info()
        
        # Should indicate not fitted
        assert info['status'] == 'not_fitted'
        
        # After mocking as fitted
        detector.is_fitted = True
        detector.feature_names = ['feature1', 'feature2']
        
        info = detector.get_model_info()
        assert info['status'] == 'fitted'
        assert info['feature_count'] == 2
        assert info['error_handling_enabled'] is True
    
    def test_predict_without_fitting(self):
        """Test prediction without fitting should raise error."""
        detector = FraudDetector(enable_error_handling=True)
        
        sample_data = self.create_sample_data(10)
        
        with pytest.raises(ValueError, match="must be fitted"):
            detector.predict(sample_data)
    
    def test_predict_with_empty_data(self):
        """Test prediction with empty data."""
        detector = FraudDetector(enable_error_handling=True)
        detector.is_fitted = True  # Mock as fitted
        
        empty_df = pd.DataFrame()
        
        # Test that the public predict method handles empty data gracefully
        result = detector.predict(empty_df)
        
        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        expected_columns = ['transaction_id', 'fraud_probability', 'anomaly_score', 
                          'risk_level', 'fraud_prediction', 'processed_at', 'model_version', 'has_error']
        for col in expected_columns:
            assert col in result.columns
    
    def test_predict_with_missing_transaction_id(self):
        """Test prediction when transaction ID column is missing."""
        detector = FraudDetector(enable_error_handling=True)
        detector.is_fitted = True
        detector.feature_names = ['amount', 'hour']
        
        # Mock components to avoid actual model calls
        detector.preprocessor = Mock()
        detector.feature_engineer = Mock()
        detector.anomaly_detector = Mock()
        detector.classifier = Mock()
        detector.risk_scorer = Mock()
        
        # Configure mocks
        detector.preprocessor.transform.return_value = pd.DataFrame({
            'amount': [100, 200],
            'hour': [10, 15]
        })
        detector.feature_engineer.transform.return_value = pd.DataFrame({
            'amount': [100, 200],
            'hour': [10, 15]
        })
        detector.anomaly_detector.predict_anomaly_scores.return_value = np.array([0.3, 0.7])
        detector.classifier.predict_proba.return_value = np.array([0.2, 0.8])
        detector.risk_scorer.assign_risk_levels.return_value = np.array(['low', 'high'])
        
        sample_data = pd.DataFrame({
            'amount': [100, 200],
            'sender_account': ['user1', 'user2']
        })
        
        result = detector.predict(sample_data)
        
        # Should handle missing transaction ID gracefully
        assert len(result) == 2
        assert 'transaction_id' in result.columns
    
    def test_error_statistics_reset(self):
        """Test error statistics reset functionality."""
        detector = FraudDetector(enable_error_handling=True)
        
        # Should not raise error
        detector.reset_error_statistics()
        
        # Test with error handling disabled
        detector_no_eh = FraudDetector(enable_error_handling=False)
        detector_no_eh.reset_error_statistics()  # Should log warning but not crash
    
    def test_single_transaction_prediction(self):
        """Test prediction for a single transaction."""
        detector = FraudDetector(enable_error_handling=True)
        detector.is_fitted = True
        detector.feature_names = ['amount']
        
        # Mock the predict method to return a simple result
        with patch.object(detector, 'predict') as mock_predict:
            mock_predict.return_value = pd.DataFrame({
                'transaction_id': ['tx1'],
                'fraud_probability': [0.3],
                'risk_level': ['low'],
                'fraud_prediction': [0],
                'explanation': ['Low risk transaction']
            })
            
            transaction = {
                'amount': 100.0,
                'sender_account': 'user1',
                'receiver_account': 'merchant1'
            }
            
            result = detector.predict_single(transaction, 'tx1')
            
            assert isinstance(result, dict)
            assert result['transaction_id'] == 'tx1'
            assert 'fraud_probability' in result
    
    def test_column_mapping_flexibility(self):
        """Test that column mapping allows for flexible input schemas."""
        detector = FraudDetector(enable_error_handling=True)
        
        # Test default column mapping
        assert detector.column_mapping['transaction_id'] == 'transaction_id'
        assert detector.column_mapping['amount'] == 'amount'
        
        # Column mapping should be configurable
        detector.column_mapping['amount'] = 'transaction_amount'
        assert detector.column_mapping['amount'] == 'transaction_amount'
    
    def test_graceful_degradation_with_component_failures(self):
        """Test graceful degradation when individual components fail."""
        detector = FraudDetector(enable_error_handling=True)
        detector.is_fitted = True
        detector.feature_names = ['amount', 'hour']
        
        # Mock components with some failing
        detector.preprocessor = Mock()
        detector.feature_engineer = Mock()
        detector.anomaly_detector = Mock()
        detector.classifier = Mock()
        detector.risk_scorer = Mock()
        
        # Configure preprocessor to work
        detector.preprocessor.transform.return_value = pd.DataFrame({
            'amount': [100],
            'hour': [10]
        })
        
        # Configure feature engineer to work
        detector.feature_engineer.transform.return_value = pd.DataFrame({
            'amount': [100],
            'hour': [10]
        })
        
        # Make anomaly detector fail
        detector.anomaly_detector.predict_anomaly_scores.side_effect = Exception("Anomaly detector failed")
        
        # Configure classifier and risk scorer to work
        detector.classifier.predict_proba.return_value = np.array([0.5])
        detector.risk_scorer.assign_risk_levels.return_value = np.array(['medium'])
        
        sample_data = pd.DataFrame({
            'transaction_id': ['tx1'],
            'amount': [100]
        })
        
        result = detector.predict(sample_data)
        
        # Should still return results with fallback anomaly scores
        assert len(result) == 1
        assert 'fraud_probability' in result.columns
        # Anomaly score should be fallback value (0.5) due to failure
        if 'anomaly_score' in result.columns:
            assert result['anomaly_score'].iloc[0] == 0.5


if __name__ == '__main__':
    pytest.main([__file__])