"""
Error handling and edge case management for the fraud detection system.
Implements comprehensive error handling, graceful degradation, and fallback mechanisms.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import warnings
from functools import wraps
import traceback

logger = logging.getLogger(__name__)


class FraudDetectionError(Exception):
    """Base exception class for fraud detection system errors."""
    pass


class DataValidationError(FraudDetectionError):
    """Raised when input data validation fails."""
    pass


class ModelError(FraudDetectionError):
    """Raised when model operations fail."""
    pass


class PipelineError(FraudDetectionError):
    """Raised when pipeline operations fail."""
    pass


class ErrorHandler:
    """
    Comprehensive error handling and edge case management for fraud detection.
    
    This class provides:
    - Graceful error handling with fallback mechanisms
    - Data validation and sanitization
    - Model failure recovery
    - Circuit breaker pattern for model serving
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, 
                 enable_fallback: bool = True,
                 fallback_fraud_probability: float = 0.5,
                 fallback_risk_level: str = 'medium',
                 max_failures: int = 5,
                 failure_window_minutes: int = 10):
        """
        Initialize the ErrorHandler with configuration.
        
        Args:
            enable_fallback: Whether to enable fallback mechanisms
            fallback_fraud_probability: Default fraud probability for fallback
            fallback_risk_level: Default risk level for fallback
            max_failures: Maximum failures before circuit breaker opens
            failure_window_minutes: Time window for failure counting
        """
        self.enable_fallback = enable_fallback
        self.fallback_fraud_probability = fallback_fraud_probability
        self.fallback_risk_level = fallback_risk_level
        self.max_failures = max_failures
        self.failure_window_minutes = failure_window_minutes
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'data_validation_errors': 0,
            'model_errors': 0,
            'pipeline_errors': 0,
            'fallback_activations': 0
        }
        
        logger.info("ErrorHandler initialized with fallback mechanisms")
    
    def handle_errors(self, operation_name: str = "operation"):
        """
        Decorator for comprehensive error handling with fallback mechanisms.
        
        Args:
            operation_name: Name of the operation for logging
            
        Returns:
            Decorated function with error handling
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # Check circuit breaker
                    if self._is_circuit_open():
                        logger.warning(f"Circuit breaker open for {operation_name}, using fallback")
                        return self._get_fallback_result(*args, **kwargs)
                    
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Reset failure count on success
                    self._reset_failures()
                    
                    return result
                    
                except Exception as e:
                    # Record failure
                    self._record_failure()
                    
                    # Log error with context
                    logger.error(f"Error in {operation_name}: {str(e)}")
                    logger.debug(f"Error traceback: {traceback.format_exc()}")
                    
                    # Update error statistics
                    self._update_error_stats(e)
                    
                    # Return fallback result if enabled
                    if self.enable_fallback:
                        logger.info(f"Using fallback mechanism for {operation_name}")
                        return self._get_fallback_result(*args, **kwargs)
                    else:
                        # Re-raise the exception
                        raise
            
            return wrapper
        return decorator
    
    def validate_input_data(self, df: pd.DataFrame, 
                           required_columns: List[str] = None,
                           min_rows: int = 1,
                           max_rows: int = 10000000) -> pd.DataFrame:
        """
        Validate and sanitize input data with comprehensive checks.
        
        Args:
            df: Input DataFrame to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            max_rows: Maximum number of rows allowed
            
        Returns:
            Validated and sanitized DataFrame
            
        Raises:
            DataValidationError: If validation fails
        """
        if df is None:
            raise DataValidationError("Input DataFrame cannot be None")
        
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError(f"Input must be a pandas DataFrame, got {type(df)}")
        
        # Check row count
        if len(df) < min_rows:
            raise DataValidationError(f"DataFrame must have at least {min_rows} rows, got {len(df)}")
        
        if len(df) > max_rows:
            logger.warning(f"DataFrame has {len(df)} rows, truncating to {max_rows}")
            df = df.head(max_rows)
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Sanitize data
        df_clean = self._sanitize_dataframe(df)
        
        logger.debug(f"Input data validated: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        
        return df_clean
    
    def validate_model_output(self, output: Any, 
                            expected_type: type = None,
                            expected_shape: tuple = None,
                            value_range: tuple = None) -> Any:
        """
        Validate model output with comprehensive checks.
        
        Args:
            output: Model output to validate
            expected_type: Expected output type
            expected_shape: Expected output shape (for arrays)
            value_range: Expected value range (min, max)
            
        Returns:
            Validated output
            
        Raises:
            ModelError: If validation fails
        """
        if output is None:
            raise ModelError("Model output cannot be None")
        
        # Type validation
        if expected_type and not isinstance(output, expected_type):
            raise ModelError(f"Expected output type {expected_type}, got {type(output)}")
        
        # Shape validation for arrays
        if expected_shape and hasattr(output, 'shape'):
            if output.shape != expected_shape:
                raise ModelError(f"Expected output shape {expected_shape}, got {output.shape}")
        
        # Check for invalid values first (before range validation)
        if hasattr(output, '__iter__'):
            if np.any(np.isnan(output)):
                logger.warning("Model output contains NaN values, replacing with fallback")
                output = np.nan_to_num(output, nan=self.fallback_fraud_probability)
            
            if np.any(np.isinf(output)):
                logger.warning("Model output contains infinite values, replacing with fallback")
                output = np.nan_to_num(output, posinf=1.0, neginf=0.0)
        
        # Value range validation (after fixing invalid values)
        if value_range and hasattr(output, '__iter__'):
            min_val, max_val = value_range
            if hasattr(output, 'min') and hasattr(output, 'max'):
                if output.min() < min_val or output.max() > max_val:
                    raise ModelError(f"Output values outside expected range {value_range}")
        
        return output
    
    def handle_missing_features(self, df: pd.DataFrame, 
                              expected_features: List[str],
                              fill_strategy: str = 'zero') -> pd.DataFrame:
        """
        Handle missing features in input data.
        
        Args:
            df: Input DataFrame
            expected_features: List of expected feature names
            fill_strategy: Strategy for filling missing features ('zero', 'mean', 'median')
            
        Returns:
            DataFrame with all expected features
        """
        missing_features = [feat for feat in expected_features if feat not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features in input data: {missing_features}")
            
            for feature in missing_features:
                if fill_strategy == 'zero':
                    df[feature] = 0.0
                elif fill_strategy == 'mean':
                    # Use a reasonable default mean value
                    df[feature] = 0.5
                elif fill_strategy == 'median':
                    # Use a reasonable default median value
                    df[feature] = 0.0
                else:
                    df[feature] = 0.0
                    
            logger.info(f"Filled {len(missing_features)} missing features with {fill_strategy} strategy")
        
        # Ensure feature order matches expected
        return df[expected_features]
    
    def handle_data_quality_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle various data quality issues in input data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with data quality issues resolved
        """
        df_clean = df.copy()
        issues_found = []
        
        # Handle missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.any():
            issues_found.append(f"missing_values: {missing_counts.sum()}")
            
            # Fill missing values based on column type
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if df_clean[col].dtype in ['int64', 'float64']:
                        # Fill numeric columns with median
                        fill_value = df_clean[col].median()
                        if pd.isna(fill_value):
                            fill_value = 0.0
                        df_clean[col] = df_clean[col].fillna(fill_value)
                    else:
                        # Fill categorical columns with mode or 'unknown'
                        mode_value = df_clean[col].mode()
                        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else 'unknown'
                        df_clean[col] = df_clean[col].fillna(fill_value)
        
        # Handle infinite values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if np.isinf(df_clean[col]).any():
                issues_found.append(f"infinite_values_in_{col}")
                # Replace infinite values with column max/min
                col_max = df_clean[col][~np.isinf(df_clean[col])].max()
                col_min = df_clean[col][~np.isinf(df_clean[col])].min()
                
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], [col_max, col_min])
        
        # Handle extreme outliers (values beyond 5 standard deviations)
        for col in numeric_columns:
            if len(df_clean[col].unique()) > 10:  # Skip categorical-like numeric columns
                mean_val = df_clean[col].mean()
                std_val = df_clean[col].std()
                
                if std_val > 0:
                    outlier_mask = np.abs(df_clean[col] - mean_val) > 5 * std_val
                    if outlier_mask.any():
                        issues_found.append(f"extreme_outliers_in_{col}: {outlier_mask.sum()}")
                        # Cap outliers at 5 standard deviations
                        upper_bound = mean_val + 5 * std_val
                        lower_bound = mean_val - 5 * std_val
                        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        # Handle duplicate rows
        duplicate_count = df_clean.duplicated().sum()
        if duplicate_count > 0:
            issues_found.append(f"duplicate_rows: {duplicate_count}")
            df_clean = df_clean.drop_duplicates()
        
        # Log issues found and resolved
        if issues_found:
            logger.info(f"Data quality issues resolved: {', '.join(issues_found)}")
        
        return df_clean
    
    def create_fallback_predictions(self, n_samples: int,
                                  transaction_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create fallback predictions when models fail.
        
        Args:
            n_samples: Number of samples to create predictions for
            transaction_ids: Optional list of transaction IDs
            
        Returns:
            DataFrame with fallback predictions
        """
        logger.info(f"Creating fallback predictions for {n_samples} samples")
        
        # Create transaction IDs if not provided
        if transaction_ids is None:
            transaction_ids = [f"fallback_tx_{i}" for i in range(n_samples)]
        
        # Create conservative fallback predictions
        fallback_results = pd.DataFrame({
            'transaction_id': transaction_ids[:n_samples],
            'fraud_probability': self.fallback_fraud_probability,
            'anomaly_score': 0.5,  # Neutral anomaly score
            'risk_level': self.fallback_risk_level,
            'fraud_prediction': 1 if self.fallback_fraud_probability >= 0.5 else 0,
            'explanation': f'Fallback prediction due to model failure (probability: {self.fallback_fraud_probability})',
            'processed_at': datetime.now(),
            'model_version': 'fallback_v1.0',
            'is_fallback': True
        })
        
        # Update fallback statistics
        self.error_stats['fallback_activations'] += n_samples
        
        return fallback_results
    
    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize DataFrame by handling common data issues."""
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Remove completely empty columns
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # Handle string columns with whitespace issues
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for col in string_columns:
            if df_clean[col].dtype == 'object':
                # Strip whitespace and handle empty strings
                df_clean[col] = df_clean[col].astype(str).str.strip()
                df_clean[col] = df_clean[col].replace('', np.nan)
        
        # Ensure numeric columns are properly typed
        for col in df_clean.columns:
            if col.lower() in ['amount', 'transaction_amount', 'value']:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    pass
        
        return df_clean
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_open:
            return False
        
        # Check if enough time has passed to try again
        if self.last_failure_time:
            minutes_since_failure = (datetime.now() - self.last_failure_time).total_seconds() / 60
            if minutes_since_failure > self.failure_window_minutes:
                logger.info("Circuit breaker reset after timeout")
                self._reset_failures()
                return False
        
        return True
    
    def _record_failure(self) -> None:
        """Record a failure for circuit breaker logic."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset_failures(self) -> None:
        """Reset failure count and circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False
    
    def _update_error_stats(self, error: Exception) -> None:
        """Update error statistics based on error type."""
        self.error_stats['total_errors'] += 1
        
        if isinstance(error, DataValidationError):
            self.error_stats['data_validation_errors'] += 1
        elif isinstance(error, ModelError):
            self.error_stats['model_errors'] += 1
        elif isinstance(error, PipelineError):
            self.error_stats['pipeline_errors'] += 1
    
    def _get_fallback_result(self, *args, **kwargs) -> Any:
        """Get appropriate fallback result based on function context."""
        # This is a simplified fallback - in practice, you'd analyze the function signature
        # and arguments to provide more appropriate fallback results
        
        # Try to determine the expected result type from arguments
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Assume this is a prediction function
                return self.create_fallback_predictions(len(arg))
        
        # Default fallback
        return {
            'error': 'Operation failed, fallback result provided',
            'fallback': True,
            'timestamp': datetime.now()
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error statistics and system health."""
        return {
            'error_statistics': self.error_stats.copy(),
            'circuit_breaker': {
                'is_open': self.circuit_open,
                'failure_count': self.failure_count,
                'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
            },
            'configuration': {
                'fallback_enabled': self.enable_fallback,
                'max_failures': self.max_failures,
                'failure_window_minutes': self.failure_window_minutes
            },
            'health_status': 'degraded' if self.circuit_open else 'healthy'
        }
    
    def reset_error_stats(self) -> None:
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'data_validation_errors': 0,
            'model_errors': 0,
            'pipeline_errors': 0,
            'fallback_activations': 0
        }
        self._reset_failures()
        logger.info("Error statistics reset")


# Utility functions for common error handling patterns

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def safe_log(value: float, default: float = 0.0) -> float:
    """Safely compute logarithm, handling negative and zero values."""
    try:
        if value <= 0:
            return default
        return np.log(value)
    except (TypeError, ValueError):
        return default


def safe_array_operation(arr: np.ndarray, operation: Callable, 
                        default_value: float = 0.0) -> np.ndarray:
    """Safely apply operation to array, handling NaN and infinite values."""
    try:
        result = operation(arr)
        
        # Handle NaN values
        if np.any(np.isnan(result)):
            result = np.nan_to_num(result, nan=default_value)
        
        # Handle infinite values
        if np.any(np.isinf(result)):
            result = np.nan_to_num(result, posinf=1.0, neginf=0.0)
        
        return result
    except Exception:
        # Return array of default values with same shape
        return np.full_like(arr, default_value, dtype=float)


def validate_probability_array(probabilities: np.ndarray, 
                             fix_invalid: bool = True) -> np.ndarray:
    """Validate and optionally fix probability array to ensure [0, 1] range."""
    if probabilities is None:
        raise ValueError("Probability array cannot be None")
    
    # Check for NaN values
    if np.any(np.isnan(probabilities)):
        if fix_invalid:
            logger.warning("Found NaN values in probabilities, replacing with 0.5")
            probabilities = np.nan_to_num(probabilities, nan=0.5)
        else:
            raise ValueError("Probability array contains NaN values")
    
    # Check for infinite values
    if np.any(np.isinf(probabilities)):
        if fix_invalid:
            logger.warning("Found infinite values in probabilities, clipping to [0, 1]")
            probabilities = np.nan_to_num(probabilities, posinf=1.0, neginf=0.0)
        else:
            raise ValueError("Probability array contains infinite values")
    
    # Check range
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        if fix_invalid:
            logger.warning("Found probabilities outside [0, 1] range, clipping")
            probabilities = np.clip(probabilities, 0.0, 1.0)
        else:
            raise ValueError("Probability array contains values outside [0, 1] range")
    
    return probabilities


def create_error_response(error: Exception, 
                         transaction_ids: Optional[List[str]] = None,
                         n_samples: int = None) -> pd.DataFrame:
    """Create standardized error response DataFrame."""
    if transaction_ids is None and n_samples is None:
        raise ValueError("Either transaction_ids or n_samples must be provided")
    
    if transaction_ids is None:
        transaction_ids = [f"error_tx_{i}" for i in range(n_samples)]
    
    return pd.DataFrame({
        'transaction_id': transaction_ids,
        'fraud_probability': 0.5,  # Conservative default
        'anomaly_score': 0.5,
        'risk_level': 'medium',  # Conservative default
        'fraud_prediction': 1,  # Conservative default (flag for review)
        'explanation': f'Error during processing: {str(error)}',
        'processed_at': datetime.now(),
        'model_version': 'error_handler_v1.0',
        'has_error': True,
        'error_type': type(error).__name__,
        'error_message': str(error)
    })