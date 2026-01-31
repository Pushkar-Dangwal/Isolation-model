"""
Feature integration module for the fraud detection system.
Combines anomaly scores with engineered features for supervised learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from anomaly_detector import AnomalyDetector
from feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class FeatureIntegrator:
    """
    Integrates anomaly scores with engineered features for supervised model training.
    
    This class ensures that anomaly_score is properly integrated as a feature
    and creates a complete feature combination pipeline for the supervised classifier.
    """
    
    def __init__(self):
        """Initialize the FeatureIntegrator."""
        self.anomaly_detector = None
        self.feature_engineer = None
        self.feature_columns = None
        self.anomaly_feature_name = 'anomaly_score'
        self.is_fitted = False
        
        logger.info("Initialized FeatureIntegrator")
    
    def fit(self, 
            df: pd.DataFrame,
            anomaly_detector: AnomalyDetector,
            feature_engineer: FeatureEngineer,
            feature_columns: Optional[List[str]] = None,
            exclude_columns: Optional[List[str]] = None) -> 'FeatureIntegrator':
        """
        Fit the feature integrator with anomaly detector and feature engineer.
        
        Args:
            df: Training DataFrame with engineered features
            anomaly_detector: Fitted anomaly detector
            feature_engineer: Fitted feature engineer
            feature_columns: Specific columns to use as features (if None, auto-detect)
            exclude_columns: Columns to exclude from features
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If components are not fitted or data is invalid
        """
        if not anomaly_detector.is_fitted:
            raise ValueError("AnomalyDetector must be fitted before integration")
        
        if not feature_engineer.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before integration")
        
        if df is None or len(df) == 0:
            raise ValueError("Training DataFrame cannot be None or empty")
        
        logger.info(f"Fitting FeatureIntegrator on {len(df)} samples")
        
        # Store components
        self.anomaly_detector = anomaly_detector
        self.feature_engineer = feature_engineer
        
        # Determine feature columns
        if feature_columns is None:
            # Auto-detect feature columns (exclude metadata and target columns)
            exclude_columns = exclude_columns or []
            default_excludes = [
                'transaction_id', 'timestamp', 'sender_account', 'receiver_account',
                'is_fraud', 'fraud_label', 'target', 'label'
            ]
            exclude_set = set(exclude_columns + default_excludes)
            
            # Get all numeric columns that are not in exclude set
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numeric_columns if col not in exclude_set]
        else:
            self.feature_columns = feature_columns.copy()
        
        # Ensure anomaly_score will be included
        if self.anomaly_feature_name not in self.feature_columns:
            self.feature_columns.append(self.anomaly_feature_name)
        
        logger.info(f"Selected {len(self.feature_columns)} feature columns for integration")
        logger.debug(f"Feature columns: {self.feature_columns}")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Transform DataFrame to feature matrix with integrated anomaly scores.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
            
        Raises:
            ValueError: If integrator is not fitted or required columns are missing
        """
        if not self.is_fitted:
            raise ValueError("FeatureIntegrator must be fitted before transforming")
        
        if df is None or len(df) == 0:
            raise ValueError("Input DataFrame cannot be None or empty")
        
        logger.debug(f"Transforming {len(df)} samples with feature integration")
        
        df_processed = df.copy()
        
        # Generate anomaly scores if not already present
        if self.anomaly_feature_name not in df_processed.columns:
            logger.debug("Generating Deep Isolation Forest anomaly scores for feature integration")
            
            # Extract features for Deep Isolation Forest anomaly detection
            anomaly_features = self._extract_anomaly_features(df_processed)
            
            # Generate anomaly scores using Deep Isolation Forest
            anomaly_scores = self.anomaly_detector.predict_anomaly_scores(anomaly_features)
            
            # Add anomaly scores to DataFrame
            df_processed[self.anomaly_feature_name] = anomaly_scores
        
        # Validate that all required feature columns are present
        missing_columns = [col for col in self.feature_columns if col not in df_processed.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")
        
        # Extract feature matrix
        feature_matrix = df_processed[self.feature_columns].values
        
        # Handle missing values
        if np.isnan(feature_matrix).any():
            logger.warning("Found NaN values in feature matrix, filling with median")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            feature_matrix = imputer.fit_transform(feature_matrix)
        
        # Handle infinite values
        if np.isinf(feature_matrix).any():
            logger.warning("Found infinite values in feature matrix, clipping to finite range")
            feature_matrix = np.clip(feature_matrix, -1e10, 1e10)
        
        # Validate anomaly score integration
        anomaly_col_idx = self.feature_columns.index(self.anomaly_feature_name)
        anomaly_scores_in_matrix = feature_matrix[:, anomaly_col_idx]
        
        # Ensure anomaly scores are in [0, 1] range as required by Requirements 3.3
        if not (np.all(anomaly_scores_in_matrix >= 0.0) and np.all(anomaly_scores_in_matrix <= 1.0)):
            logger.warning("Anomaly scores not in [0, 1] range, clipping values")
            feature_matrix[:, anomaly_col_idx] = np.clip(anomaly_scores_in_matrix, 0.0, 1.0)
        
        logger.debug(f"Generated feature matrix with shape {feature_matrix.shape}")
        logger.debug(f"Anomaly score statistics - min: {feature_matrix[:, anomaly_col_idx].min():.4f}, "
                    f"max: {feature_matrix[:, anomaly_col_idx].max():.4f}, "
                    f"mean: {feature_matrix[:, anomaly_col_idx].mean():.4f}")
        
        return feature_matrix, self.feature_columns.copy()
    
    def fit_transform(self, 
                     df: pd.DataFrame,
                     anomaly_detector: AnomalyDetector,
                     feature_engineer: FeatureEngineer,
                     feature_columns: Optional[List[str]] = None,
                     exclude_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the integrator and transform data in one step.
        
        Args:
            df: Training DataFrame with engineered features
            anomaly_detector: Fitted anomaly detector
            feature_engineer: Fitted feature engineer
            feature_columns: Specific columns to use as features
            exclude_columns: Columns to exclude from features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.fit(df, anomaly_detector, feature_engineer, feature_columns, exclude_columns)
        return self.transform(df)
    
    def _extract_anomaly_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features suitable for Deep Isolation Forest anomaly detection from the DataFrame.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Feature matrix for Deep Isolation Forest anomaly detection
        """
        # Use the same features that the Deep Isolation Forest was trained on
        if self.anomaly_detector.feature_names:
            # Try to match feature names from anomaly detector
            available_features = []
            for feature_name in self.anomaly_detector.feature_names:
                if feature_name in df.columns:
                    available_features.append(feature_name)
                else:
                    logger.warning(f"Deep Isolation Forest feature '{feature_name}' not found in DataFrame")
            
            if available_features:
                return df[available_features].values
        
        # Fallback: use all numeric columns except metadata
        exclude_columns = [
            'transaction_id', 'timestamp', 'sender_account', 'receiver_account',
            'is_fraud', 'fraud_label', 'target', 'label', self.anomaly_feature_name
        ]
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        if not feature_columns:
            raise ValueError("No suitable features found for Deep Isolation Forest anomaly detection")
        
        logger.debug(f"Using {len(feature_columns)} features for Deep Isolation Forest anomaly detection")
        return df[feature_columns].values
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the integrated features.
        
        Returns:
            Dictionary containing feature information
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        # Find anomaly score position
        anomaly_score_index = None
        if self.anomaly_feature_name in self.feature_columns:
            anomaly_score_index = self.feature_columns.index(self.anomaly_feature_name)
        
        return {
            'status': 'fitted',
            'n_features': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'anomaly_feature_name': self.anomaly_feature_name,
            'anomaly_score_index': anomaly_score_index,
            'has_anomaly_detector': self.anomaly_detector is not None,
            'has_feature_engineer': self.feature_engineer is not None
        }
    
    def validate_feature_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that the DataFrame contains all required features for integration.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        if not self.is_fitted:
            return {'status': 'not_fitted', 'valid': False}
        
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        extra_features = [col for col in df.columns if col not in self.feature_columns 
                         and col not in ['transaction_id', 'timestamp', 'sender_account', 
                                       'receiver_account', 'is_fraud', 'fraud_label']]
        
        has_anomaly_score = self.anomaly_feature_name in df.columns
        
        validation_result = {
            'status': 'fitted',
            'valid': len(missing_features) == 0,
            'missing_features': missing_features,
            'extra_features': extra_features,
            'has_anomaly_score': has_anomaly_score,
            'total_features_expected': len(self.feature_columns),
            'total_features_found': len([col for col in self.feature_columns if col in df.columns])
        }
        
        return validation_result
    
    def create_feature_pipeline(self) -> Dict[str, Any]:
        """
        Create a complete feature pipeline configuration.
        
        Returns:
            Dictionary containing pipeline configuration
        """
        if not self.is_fitted:
            raise ValueError("FeatureIntegrator must be fitted before creating pipeline")
        
        pipeline_config = {
            'feature_engineer': {
                'class': 'FeatureEngineer',
                'fitted': self.feature_engineer.is_fitted if self.feature_engineer else False
            },
            'anomaly_detector': {
                'class': 'AnomalyDetector (Deep Isolation Forest)',
                'fitted': self.anomaly_detector.is_fitted if self.anomaly_detector else False,
                'contamination': self.anomaly_detector.contamination if self.anomaly_detector else None,
                'model_type': 'Deep Isolation Forest (DIF/ODIF)',
                'deep_layers': getattr(self.anomaly_detector, 'n_layers', None),
                'hidden_units': getattr(self.anomaly_detector, 'n_hidden', None)
            },
            'feature_integration': {
                'class': 'FeatureIntegrator',
                'fitted': self.is_fitted,
                'n_features': len(self.feature_columns),
                'anomaly_feature_included': self.anomaly_feature_name in self.feature_columns
            },
            'output_features': self.feature_columns,
            'pipeline_steps': [
                'feature_engineering',
                'anomaly_detection',
                'feature_integration',
                'supervised_classification'
            ]
        }
        
        return pipeline_config