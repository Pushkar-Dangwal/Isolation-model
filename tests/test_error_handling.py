"""
Tests for error handling and edge case management in the fraud detection system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from error_handling import (
    ErrorHandler, DataValidationError, ModelError, PipelineError,
    safe_divide, safe_log, validate_probability_array, create_error_response
)
from fraud_detector import FraudDetector


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler(
            enable_fallback=True,
            fallback_fraud_probability=0.5,
            fallback_risk_level='medium'
        )
    
    def test_validate_input_data_valid(self):
        """Test input data validation with valid data."""
        df = pd.DataFrame({
            'transaction_id': ['tx1', 'tx2', 'tx3'],
            'amount': [100.0, 200.0, 300.0],
            'sender': ['user1', 'user2', 'user3']
        })
        
        result = self.error_handler.validate_input_data(
            df, required_columns=['transaction_id', 'amount']
        )
        
        assert len(result) == 3
        assert 'transaction_id' in result.columns
        assert 'amount' in result.columns
    
    def test_validate_input_data_missing_columns(self):
        """Test input data validation with missing required columns."""
        df = pd.DataFrame({
            'transaction_id': ['tx1', 'tx2'],
            'amount': [100.0, 200.0]
        })
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            self.error_handler.validate_input_data(
                df, required_columns=['transaction_id', 'amount', 'sender']
            )
    
    def test_validate_input_data_empty_dataframe(self):
        """Test input data validation with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="must have at least"):
            self.error_handler.validate_input_data(df, min_rows=1)
    
    def test_validate_input_data_none(self):
        """Test input data validation with None input."""
        with pytest.raises(DataValidationError, match="cannot be None"):
            self.error_handler.validate_input_data(None)
    
    def test_validate_model_output_valid(self):
        """Test model output validation with valid output."""
        output = np.array([0.1, 0.5, 0.9])
        
        result = self.error_handler.validate_model_output(
            output, expected_type=np.ndarray, value_range=(0, 1)
        )
        
        assert np.array_equal(result, output)
    
    def test_validate_model_output_with_nan(self):
        """Test model output validation with NaN values."""
        output = np.array([0.1, np.nan, 0.9])
        
        result = self.error_handler.validate_model_output(
            output, expected_type=np.ndarray, value_range=(0, 1)
        )
        
        # NaN should be replaced with fallback value
        assert not np.any(np.isnan(result))
        assert result[1] == self.error_handler.fallback_fraud_probability
    
    def test_validate_model_output_with_inf(self):
        """Test model output validation with infinite values."""
        output = np.array([0.1, np.inf, -np.inf])
        
        result = self.error_handler.validate_model_output(
            output, expected_type=np.ndarray, value_range=(0, 1)
        )
        
        # Infinite values should be replaced
        assert not np.any(np.isinf(result))
        assert result[1] == 1.0  # positive infinity -> 1.0
        assert result[2] == 0.0  # negative infinity -> 0.0
    
    def test_handle_missing_features(self):
        """Test handling of missing features."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        expected_features = ['feature1', 'feature2', 'feature3']
        
        result = self.error_handler.handle_missing_features(
            df, expected_features, fill_strategy='zero'
        )
        
        assert 'feature3' in result.columns
        assert all(result['feature3'] == 0.0)
        assert list(result.columns) == expected_features
    
    def test_handle_data_quality_issues(self):
        """Test handling of various data quality issues."""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3, np.inf],
            'col2': ['a', None, 'c', 'd'],
            'col3': [1, 2, 2, 4]  # Has duplicates when combined
        })
        
        result = self.error_handler.handle_data_quality_issues(df)
        
        # Check that missing values are filled
        assert not result['col1'].isna().any()
        assert not result['col2'].isna().any()
        
        # Check that infinite values are handled
        assert not np.isinf(result['col1']).any()
    
    def test_create_fallback_predictions(self):
        """Test creation of fallback predictions."""
        n_samples = 5
        transaction_ids = [f'tx_{i}' for i in range(n_samples)]
        
        result = self.error_handler.create_fallback_predictions(
            n_samples, transaction_ids
        )
        
        assert len(result) == n_samples
        assert 'fraud_probability' in result.columns
        assert 'risk_level' in result.columns
        assert 'is_fallback' in result.columns
        assert all(result['is_fallback'] == True)
        assert all(result['fraud_probability'] == self.error_handler.fallback_fraud_probability)
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality."""
        # Configure for quick circuit breaking
        handler = ErrorHandler(max_failures=2, failure_window_minutes=1)
        
        # Simulate failures
        handler._record_failure()
        assert not handler._is_circuit_open()
        
        handler._record_failure()
        assert handler._is_circuit_open()
        
        # Reset should clear the circuit breaker
        handler._reset_failures()
        assert not handler._is_circuit_open()
    
    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        initial_stats = self.error_handler.get_error_summary()
        assert initial_stats['error_statistics']['total_errors'] == 0
        
        # Simulate different types of errors
        self.error_handler._update_error_stats(DataValidationError("test"))
        self.error_handler._update_error_stats(ModelError("test"))
        
        updated_stats = self.error_handler.get_error_summary()
        assert updated_stats['error_statistics']['total_errors'] == 2
        assert updated_stats['error_statistics']['data_validation_errors'] == 1
        assert updated_stats['error_statistics']['model_errors'] == 1


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_safe_divide(self):
        """Test safe division function."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0  # Default value
        assert safe_divide(10, 0, default=1.0) == 1.0
        assert safe_divide("invalid", 2) == 0.0
    
    def test_safe_log(self):
        """Test safe logarithm function."""
        assert safe_log(np.e) == pytest.approx(1.0)
        assert safe_log(0) == 0.0  # Default value
        assert safe_log(-1) == 0.0  # Default value
        assert safe_log(-1, default=1.0) == 1.0
    
    def test_validate_probability_array(self):
        """Test probability array validation."""
        # Valid probabilities
        valid_probs = np.array([0.0, 0.5, 1.0])
        result = validate_probability_array(valid_probs)
        assert np.array_equal(result, valid_probs)
        
        # Probabilities with NaN
        nan_probs = np.array([0.1, np.nan, 0.9])
        result = validate_probability_array(nan_probs, fix_invalid=True)
        assert not np.any(np.isnan(result))
        assert result[1] == 0.5  # Default replacement
        
        # Probabilities outside range
        invalid_probs = np.array([-0.1, 0.5, 1.5])
        result = validate_probability_array(invalid_probs, fix_invalid=True)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        
        # Test with fix_invalid=False
        with pytest.raises(ValueError, match="NaN values"):
            validate_probability_array(nan_probs, fix_invalid=False)
    
    def test_create_error_response(self):
        """Test error response creation."""
        error = ValueError("Test error")
        transaction_ids = ['tx1', 'tx2', 'tx3']
        
        result = create_error_response(error, transaction_ids)
        
        assert len(result) == 3
        assert 'transaction_id' in result.columns
        assert 'fraud_probability' in result.columns
        assert 'has_error' in result.columns
        assert all(result['has_error'] == True)
        assert all(result['error_type'] == 'ValueError')


class TestFraudDetectorErrorHandling:
    """Test error handling integration in FraudDetector."""
    
    def test_fraud_detector_with_error_handling_enabled(self):
        """Test FraudDetector initialization with error handling enabled."""
        detector = FraudDetector(enable_error_handling=True)
        
        assert detector.enable_error_handling is True
        assert detector.error_handler is not None
        
        health = detector.get_system_health()
        assert health['error_handling']['enabled'] is True
    
    def test_fraud_detector_with_error_handling_disabled(self):
        """Test FraudDetector initialization with error handling disabled."""
        detector = FraudDetector(enable_error_handling=False)
        
        assert detector.enable_error_handling is False
        assert detector.error_handler is None
        
        health = detector.get_system_health()
        assert health['error_handling']['enabled'] is False
    
    def test_predict_with_invalid_data_and_error_handling(self):
        """Test prediction with invalid data when error handling is enabled."""
        detector = FraudDetector(enable_error_handling=True)
        
        # Mock the detector as fitted
        detector.is_fitted = True
        detector.feature_names = ['feature1', 'feature2']
        
        # Create invalid input data
        invalid_df = pd.DataFrame({
            'transaction_id': ['tx1'],
            'invalid_column': [None]
        })
        
        # Mock components to avoid actual model calls
        with patch.object(detector.error_handler, 'create_fallback_predictions') as mock_fallback:
            mock_fallback.return_value = pd.DataFrame({
                'transaction_id': ['tx1'],
                'fraud_probability': [0.5],
                'risk_level': ['medium'],
                'fraud_prediction': [1],
                'is_fallback': [True]
            })
            
            result = detector.predict(invalid_df)
            
            # Should return fallback predictions
            assert len(result) == 1
            assert 'is_fallback' in result.columns or 'fraud_probability' in result.columns
    
    def test_system_health_reporting(self):
        """Test system health reporting functionality."""
        detector = FraudDetector(enable_error_handling=True)
        
        health = detector.get_system_health()
        
        assert 'overall_status' in health
        assert 'pipeline_fitted' in health
        assert 'components_status' in health
        assert 'error_handling' in health
        
        # Should report not ready since not fitted
        assert health['overall_status'] == 'not_ready'
        assert health['pipeline_fitted'] is False


if __name__ == '__main__':
    pytest.main([__file__])