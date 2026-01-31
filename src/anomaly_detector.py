"""
Anomaly detection module for the fraud detection system.
Implements Deep Isolation Forest (DIF/ODIF) for unsupervised anomaly detection.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from typing import Dict, List, Optional, Tuple, Any
import joblib
import warnings
from datetime import datetime
from pathlib import Path

from config import ANOMALY_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)


class RandomDeepFeatureMapper(BaseEstimator, TransformerMixin):
    """
    Random Deep Feature Mapper for Deep Isolation Forest (DIF/ODIF).
    
    Creates deep nonlinear feature representations using random neural network layers
    without training. This mimics the ODIF approach of deep random feature mapping
    before applying Isolation Forest.
    """
    
    def __init__(self, n_layers: int = 3, n_hidden: int = 128, 
                 activation: str = 'tanh', random_state: int = 42):
        """
        Initialize the Random Deep Feature Mapper.
        
        Args:
            n_layers: Number of random neural network layers
            n_hidden: Number of hidden units per layer
            activation: Activation function ('tanh', 'relu', 'sigmoid')
            random_state: Random seed for reproducibility
        """
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state
        self.weights_ = []
        self.biases_ = []
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y=None) -> 'RandomDeepFeatureMapper':
        """
        Fit the random deep feature mapper by generating random weights.
        
        Args:
            X: Input feature matrix
            y: Ignored (unsupervised)
            
        Returns:
            Self for method chaining
        """
        rng = np.random.RandomState(self.random_state)
        self.weights_ = []
        self.biases_ = []
        
        n_features = X.shape[1]
        
        # Generate random weights and biases for each layer
        for i in range(self.n_layers):
            # Xavier/Glorot initialization for better gradient flow
            fan_in = n_features
            fan_out = self.n_hidden
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            W = rng.uniform(-limit, limit, size=(n_features, self.n_hidden))
            b = rng.uniform(-limit, limit, size=(self.n_hidden,))
            
            self.weights_.append(W)
            self.biases_.append(b)
            n_features = self.n_hidden
        
        self.is_fitted = True
        logger.debug(f"Fitted RandomDeepFeatureMapper with {self.n_layers} layers, "
                    f"{self.n_hidden} hidden units each")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features through random deep layers.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Deep feature representation
        """
        if not self.is_fitted:
            raise ValueError("RandomDeepFeatureMapper must be fitted before transform")
        
        X_out = X.astype(np.float32)
        
        # Forward pass through random layers
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            X_out = X_out @ W + b
            
            # Apply activation function
            if self.activation == 'tanh':
                X_out = np.tanh(X_out)
            elif self.activation == 'relu':
                X_out = np.maximum(0, X_out)
            elif self.activation == 'sigmoid':
                X_out = 1 / (1 + np.exp(-np.clip(X_out, -500, 500)))  # Clip to prevent overflow
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
        
        return X_out
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class AnomalyDetector:
    """
    Implements Deep Isolation Forest (DIF/ODIF) for unsupervised anomaly detection in fraud detection.
    
    This class combines random deep feature mapping with Isolation Forest to create
    a Deep Isolation Forest approach. The deep feature mapper creates nonlinear
    representations without training, then Isolation Forest identifies anomalies
    in the deep feature space.
    """
    
    def __init__(self, 
                 contamination: float = None,
                 n_estimators: int = None,
                 max_samples: str = None,
                 random_state: int = None,
                 n_jobs: int = -1,
                 # Deep feature mapping parameters
                 n_layers: int = 3,
                 n_hidden: int = 128,
                 activation: str = 'tanh'):
        """
        Initialize the Deep Isolation Forest AnomalyDetector.
        
        Args:
            contamination: Expected proportion of outliers (fraud rate)
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw to train each base estimator
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all processors)
            n_layers: Number of random deep layers for feature mapping
            n_hidden: Number of hidden units per layer
            activation: Activation function for deep layers
        """
        # Use config defaults if not specified
        self.contamination = contamination or ANOMALY_CONFIG['contamination']
        self.n_estimators = n_estimators or ANOMALY_CONFIG['n_estimators']
        self.max_samples = max_samples or ANOMALY_CONFIG['max_samples']
        self.random_state = random_state or ANOMALY_CONFIG['random_state']
        self.n_jobs = n_jobs
        
        # Deep feature mapping parameters
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.activation = activation
        
        # Initialize components
        self.deep_mapper = RandomDeepFeatureMapper(
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            activation=self.activation,
            random_state=self.random_state
        )
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Performance tracking
        self.training_stats = {}
        
        logger.info(f"Initialized Deep Isolation Forest AnomalyDetector with "
                   f"contamination={self.contamination}, n_estimators={self.n_estimators}, "
                   f"deep_layers={self.n_layers}, hidden_units={self.n_hidden}")
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'AnomalyDetector':
        """
        Train the Deep Isolation Forest on transaction features.
        
        This implements the DIF/ODIF approach:
        1. Scale input features
        2. Apply random deep feature mapping
        3. Train Isolation Forest on deep features
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names for interpretability
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If input data is invalid
        """
        if X is None or len(X) == 0:
            raise ValueError("Input data X cannot be None or empty")
        
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input X must be numpy array or pandas DataFrame")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        logger.info(f"Training Deep Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Validate input dimensions
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2-dimensional, got shape {X.shape}")
        
        if X.shape[0] < 10:
            logger.warning(f"Very small training set ({X.shape[0]} samples), results may be unreliable")
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Handle missing values - ensure X is numeric first
        try:
            # Convert to float64 to ensure compatibility with np.isnan
            X = X.astype(np.float64)
            if np.isnan(X).any():
                logger.warning("Found NaN values in input data, filling with median")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
        except (ValueError, TypeError) as e:
            logger.warning(f"Data type conversion issue: {e}. Attempting to handle mixed types.")
            # Handle mixed data types by converting each column individually
            X_converted = np.zeros_like(X, dtype=np.float64)
            for i in range(X.shape[1]):
                try:
                    X_converted[:, i] = pd.to_numeric(X[:, i], errors='coerce')
                except:
                    # If conversion fails, fill with zeros
                    X_converted[:, i] = 0.0
            X = X_converted
            
            # Now handle NaN values after conversion
            if np.isnan(X).any():
                logger.warning("Found NaN values after type conversion, filling with median")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
        
        # Handle infinite values
        if np.isinf(X).any():
            logger.warning("Found infinite values in input data, clipping to finite range")
            X = np.clip(X, -1e10, 1e10)
        
        # Step 1: Scale features for better deep feature mapping
        logger.debug("Scaling features for deep feature mapping")
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: Apply random deep feature mapping (core DIF/ODIF approach)
        logger.debug("Applying random deep feature mapping")
        X_deep = self.deep_mapper.fit_transform(X_scaled)
        
        logger.debug(f"Deep features shape: {X_deep.shape}")
        
        # Step 3: Initialize and train Isolation Forest on deep features
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Train the model on deep features
        logger.debug("Training Isolation Forest on deep features")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress sklearn warnings
            self.isolation_forest.fit(X_deep)
        
        # Calculate training statistics
        self._calculate_training_stats(X_deep)
        
        self.is_fitted = True
        logger.info("Deep Isolation Forest training completed successfully")
        
        return self
    
    def predict_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Generate anomaly scores for transactions using Deep Isolation Forest.
        
        Scores are normalized to [0, 1] range where higher scores indicate
        greater anomaly likelihood (higher fraud probability). This method
        ensures proper normalization and handles edge cases for robust scoring.
        
        The process follows the DIF/ODIF approach:
        1. Scale input features
        2. Apply deep feature mapping
        3. Generate anomaly scores from Isolation Forest
        4. Normalize to [0, 1] range
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of anomaly scores in [0, 1] range
            
        Raises:
            ValueError: If detector is not fitted or input is invalid
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector must be fitted before predicting")
        
        if X is None or len(X) == 0:
            raise ValueError("Input data X cannot be None or empty")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if X.ndim != 2:
            raise ValueError(f"Input X must be 2-dimensional, got shape {X.shape}")
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        logger.debug(f"Generating Deep Isolation Forest anomaly scores for {X.shape[0]} samples")
        
        # Handle missing values (same strategy as training) - ensure X is numeric first
        try:
            # Convert to float64 to ensure compatibility with np.isnan
            X = X.astype(np.float64)
            if np.isnan(X).any():
                logger.warning("Found NaN values in input data, filling with median")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
        except (ValueError, TypeError) as e:
            logger.warning(f"Data type conversion issue: {e}. Attempting to handle mixed types.")
            # Handle mixed data types by converting each column individually
            X_converted = np.zeros_like(X, dtype=np.float64)
            for i in range(X.shape[1]):
                try:
                    X_converted[:, i] = pd.to_numeric(X[:, i], errors='coerce')
                except:
                    # If conversion fails, fill with zeros
                    X_converted[:, i] = 0.0
            X = X_converted
            
            # Now handle NaN values after conversion
            if np.isnan(X).any():
                logger.warning("Found NaN values after type conversion, filling with median")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
        
        # Handle infinite values
        if np.isinf(X).any():
            logger.warning("Found infinite values in input data, clipping to finite range")
            X = np.clip(X, -1e10, 1e10)
        
        # Step 1: Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Step 2: Apply deep feature mapping
        X_deep = self.deep_mapper.transform(X_scaled)
        
        # Step 3: Get anomaly scores from Isolation Forest on deep features
        # decision_function returns negative scores for outliers
        raw_scores = self.isolation_forest.decision_function(X_deep)
        
        # Step 4: Normalize scores to [0, 1] range with proper validation
        # Higher scores should indicate higher anomaly likelihood
        normalized_scores = self._normalize_scores(raw_scores)
        
        # Validate that all scores are in [0, 1] range as required by Requirements 3.2
        assert np.all(normalized_scores >= 0.0), "Anomaly scores must be >= 0"
        assert np.all(normalized_scores <= 1.0), "Anomaly scores must be <= 1"
        
        logger.debug(f"Generated Deep Isolation Forest anomaly scores - "
                    f"min: {normalized_scores.min():.4f}, "
                    f"max: {normalized_scores.max():.4f}, "
                    f"mean: {normalized_scores.mean():.4f}")
        
        return normalized_scores
    
    def predict_anomalies(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """
        Predict binary anomaly labels for transactions using Deep Isolation Forest.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            threshold: Anomaly score threshold (default uses contamination rate)
            
        Returns:
            Binary array where 1 indicates anomaly, 0 indicates normal
        """
        scores = self.predict_anomaly_scores(X)
        
        if threshold is None:
            # Use contamination rate to determine threshold
            threshold = np.percentile(scores, (1 - self.contamination) * 100)
        
        return (scores >= threshold).astype(int)
    
    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Normalize raw isolation forest scores to [0, 1] range.
        
        Isolation Forest returns negative scores for outliers, so we need to
        transform them to a more intuitive scale where higher values indicate
        higher anomaly likelihood. This method ensures robust normalization
        that handles edge cases and guarantees [0, 1] output range.
        
        Args:
            raw_scores: Raw scores from isolation forest
            
        Returns:
            Normalized scores in [0, 1] range
        """
        # Isolation Forest scores are typically in range [-0.5, 0.5]
        # Outliers have more negative scores
        
        # Handle edge case where all scores are identical
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        
        if max_score == min_score:
            # All scores are the same, return uniform scores at contamination level
            logger.debug("All raw scores identical, returning uniform anomaly scores")
            return np.full_like(raw_scores, self.contamination)
        
        # Invert and normalize: lower raw scores -> higher normalized scores
        # This ensures outliers (negative scores) get higher anomaly scores
        normalized = (max_score - raw_scores) / (max_score - min_score)
        
        # Ensure scores are strictly in [0, 1] range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Additional validation to ensure proper range
        assert np.all(normalized >= 0.0), "Normalized scores contain values < 0"
        assert np.all(normalized <= 1.0), "Normalized scores contain values > 1"
        
        # Safe NaN and infinity checks with proper data type handling
        try:
            assert not np.any(np.isnan(normalized.astype(np.float64))), "Normalized scores contain NaN values"
            assert not np.any(np.isinf(normalized.astype(np.float64))), "Normalized scores contain infinite values"
        except (ValueError, TypeError):
            # If type conversion fails, do element-wise checking
            has_nan = False
            has_inf = False
            for val in normalized.flat:
                try:
                    if np.isnan(float(val)):
                        has_nan = True
                        break
                    if np.isinf(float(val)):
                        has_inf = True
                        break
                except (ValueError, TypeError):
                    continue
            assert not has_nan, "Normalized scores contain NaN values"
            assert not has_inf, "Normalized scores contain infinite values"
        
        return normalized
    
    def _calculate_training_stats(self, X_deep: np.ndarray) -> None:
        """
        Calculate and store training statistics for model evaluation.
        
        Args:
            X_deep: Deep feature representation of training data
        """
        # Get training anomaly scores
        training_scores = self.isolation_forest.decision_function(X_deep)
        normalized_training_scores = self._normalize_scores(training_scores)
        
        # Calculate statistics
        self.training_stats = {
            'n_samples': X_deep.shape[0],
            'n_original_features': len(self.feature_names),
            'n_deep_features': X_deep.shape[1],
            'deep_layers': self.n_layers,
            'hidden_units': self.n_hidden,
            'activation': self.activation,
            'contamination': self.contamination,
            'score_mean': float(normalized_training_scores.mean()),
            'score_std': float(normalized_training_scores.std()),
            'score_min': float(normalized_training_scores.min()),
            'score_max': float(normalized_training_scores.max()),
            'threshold_percentile': float(np.percentile(normalized_training_scores, 
                                                       (1 - self.contamination) * 100))
        }
        
        logger.debug(f"Deep Isolation Forest training statistics: {self.training_stats}")
    
    def get_feature_importance(self, X: np.ndarray = None, n_samples: int = 1000) -> Dict[str, float]:
        """
        Calculate feature importance for anomaly detection.
        
        This is an approximation since Isolation Forest doesn't provide
        direct feature importance. We use permutation-based importance.
        
        Args:
            X: Optional data to calculate importance on (uses subset if large)
            n_samples: Maximum number of samples to use for importance calculation
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("AnomalyDetector must be fitted before calculating feature importance")
        
        if X is None:
            logger.warning("No data provided for feature importance calculation")
            return {name: 0.0 for name in self.feature_names}
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Use subset for efficiency
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
        
        logger.debug(f"Calculating feature importance using {len(X_subset)} samples")
        
        # Get baseline scores
        baseline_scores = self.predict_anomaly_scores(X_subset)
        baseline_mean = baseline_scores.mean()
        
        importance_scores = {}
        
        # Calculate permutation importance for each feature
        for i, feature_name in enumerate(self.feature_names):
            # Create permuted version of the data
            X_permuted = X_subset.copy()
            np.random.shuffle(X_permuted[:, i])  # Shuffle this feature
            
            # Get scores with permuted feature
            permuted_scores = self.predict_anomaly_scores(X_permuted)
            permuted_mean = permuted_scores.mean()
            
            # Importance is the change in mean score
            importance = abs(baseline_mean - permuted_mean)
            importance_scores[feature_name] = float(importance)
        
        # Normalize importance scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
        
        return importance_scores
    
    def save_model(self, filepath: str, version: Optional[str] = None,
                   include_metadata: bool = True) -> str:
        """
        Save the trained Deep Isolation Forest anomaly detector to disk with comprehensive metadata.
        
        Args:
            filepath: Path to save the model
            version: Optional version string
            include_metadata: Whether to include comprehensive metadata
            
        Returns:
            Path to the saved model
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        from model_persistence import ModelPersistenceManager, ModelMetadata
        
        # Initialize persistence manager
        persistence_manager = ModelPersistenceManager()
        
        # Extract model name from filepath
        model_name = Path(filepath).stem
        
        model_data = {
            'isolation_forest': self.isolation_forest,
            'deep_mapper': self.deep_mapper,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'random_state': self.random_state,
            'n_layers': self.n_layers,
            'n_hidden': self.n_hidden,
            'activation': self.activation,
            'training_stats': self.training_stats,
            'is_fitted': self.is_fitted
        }
        
        # Create metadata if requested
        metadata = None
        if include_metadata:
            hyperparameters = {
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'max_samples': self.max_samples,
                'random_state': self.random_state,
                'n_layers': self.n_layers,
                'n_hidden': self.n_hidden,
                'activation': self.activation
            }
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_type='deep_isolation_forest',
                version=version or datetime.now().strftime("%Y%m%d_%H%M%S"),
                created_at=datetime.now().isoformat(),
                training_samples=self.training_stats.get('n_samples'),
                feature_count=len(self.feature_names) if self.feature_names else None,
                feature_names=self.feature_names,
                hyperparameters=hyperparameters,
                performance_metrics={
                    'score_mean': self.training_stats.get('score_mean'),
                    'score_std': self.training_stats.get('score_std'),
                    'threshold_percentile': self.training_stats.get('threshold_percentile')
                },
                random_state=self.random_state,
                tags=['anomaly_detection', 'isolation_forest', 'deep_learning'],
                description=f"Deep Isolation Forest with {self.n_layers} layers and {self.n_hidden} hidden units"
            )
        
        # Save using persistence manager
        saved_path = persistence_manager.save_model(
            model=model_data,
            model_name=model_name,
            model_type='deep_isolation_forest',
            metadata=metadata,
            version=version,
            format='joblib',
            compress=True
        )
        
        logger.info(f"Deep Isolation Forest AnomalyDetector saved to {saved_path}")
        return saved_path
    
    def load_model(self, filepath: str, version: Optional[str] = None,
                   verify_integrity: bool = True) -> 'AnomalyDetector':
        """
        Load a trained Deep Isolation Forest anomaly detector from disk with metadata validation.
        
        Args:
            filepath: Path to load the model from
            version: Specific version to load
            verify_integrity: Whether to verify model integrity
            
        Returns:
            Self for method chaining
        """
        from model_persistence import ModelPersistenceManager
        
        # Initialize persistence manager
        persistence_manager = ModelPersistenceManager()
        
        # Extract model name from filepath
        model_name = Path(filepath).stem
        
        # Load using persistence manager
        model_data, metadata = persistence_manager.load_model(
            model_name=model_name,
            version=version,
            verify_integrity=verify_integrity
        )
        
        # Restore model components
        self.isolation_forest = model_data['isolation_forest']
        self.deep_mapper = model_data['deep_mapper']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']
        self.n_estimators = model_data['n_estimators']
        self.max_samples = model_data['max_samples']
        self.random_state = model_data['random_state']
        self.n_layers = model_data['n_layers']
        self.n_hidden = model_data['n_hidden']
        self.activation = model_data['activation']
        self.training_stats = model_data['training_stats']
        self.is_fitted = model_data['is_fitted']
        
        # Log metadata information if available
        if metadata:
            logger.info(f"Loaded anomaly detector with metadata - Version: {metadata.version}, "
                       f"Features: {metadata.feature_count}")
        
        logger.info(f"Deep Isolation Forest AnomalyDetector loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the trained Deep Isolation Forest model.
        
        Returns:
            Dictionary containing model configuration and statistics
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        return {
            'status': 'fitted',
            'model_type': 'Deep Isolation Forest (DIF/ODIF)',
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'random_state': self.random_state,
            'deep_layers': self.n_layers,
            'hidden_units': self.n_hidden,
            'activation': self.activation,
            'n_original_features': len(self.feature_names),
            'n_deep_features': self.training_stats.get('n_deep_features', 'unknown'),
            'feature_names': self.feature_names,
            'training_stats': self.training_stats
        }