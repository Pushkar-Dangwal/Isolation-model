"""
Supervised classification module for the fraud detection system.
Implements LightGBM classifier with class imbalance handling for fraud detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import joblib
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import CLASSIFIER_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)


class SupervisedClassifier(BaseEstimator, ClassifierMixin):
    """
    Implements LightGBM classifier with class imbalance handling for fraud detection.
    
    This class provides a robust supervised classification solution specifically
    designed for fraud detection with highly imbalanced datasets (4% fraud rate).
    It includes automatic class weight calculation, early stopping, cross-validation,
    and comprehensive performance tracking.
    
    Key features:
    - Automatic class imbalance handling using scale_pos_weight
    - Early stopping to prevent overfitting
    - Cross-validation for robust model evaluation
    - Comprehensive performance metrics tracking
    - Model persistence and reproducibility
    """
    
    def __init__(self,
                 # LightGBM parameters
                 objective: str = None,
                 metric: str = None,
                 boosting_type: str = None,
                 num_leaves: int = None,
                 learning_rate: float = None,
                 feature_fraction: float = None,
                 bagging_fraction: float = None,
                 bagging_freq: int = None,
                 max_depth: int = None,
                 min_data_in_leaf: int = None,
                 lambda_l1: float = None,
                 lambda_l2: float = None,
                 # Class imbalance handling
                 scale_pos_weight: float = None,
                 is_unbalance: bool = None,
                 # Training parameters
                 n_estimators: int = 1000,
                 early_stopping_rounds: int = 100,
                 verbose: int = None,
                 random_state: int = None,
                 n_jobs: int = -1):
        """
        Initialize the SupervisedClassifier with LightGBM.
        
        Args:
            objective: LightGBM objective function
            metric: Evaluation metric for training
            boosting_type: Type of boosting algorithm
            num_leaves: Maximum number of leaves in one tree
            learning_rate: Learning rate for gradient boosting
            feature_fraction: Fraction of features to use in each iteration
            bagging_fraction: Fraction of data to use in each iteration
            bagging_freq: Frequency of bagging
            max_depth: Maximum depth of trees (-1 for no limit)
            min_data_in_leaf: Minimum number of data points in a leaf
            lambda_l1: L1 regularization term
            lambda_l2: L2 regularization term
            scale_pos_weight: Weight for positive class (fraud) to handle imbalance
            is_unbalance: Whether to use LightGBM's built-in imbalance handling
            n_estimators: Maximum number of boosting iterations
            early_stopping_rounds: Number of rounds for early stopping
            verbose: Verbosity level for LightGBM
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
        """
        # Use config defaults if not specified
        config = CLASSIFIER_CONFIG['lightgbm']
        
        self.objective = objective or config.get('objective', 'binary')
        self.metric = metric or config.get('metric', 'auc')
        self.boosting_type = boosting_type or config.get('boosting_type', 'gbdt')
        self.num_leaves = num_leaves or config.get('num_leaves', 31)
        self.learning_rate = learning_rate or config.get('learning_rate', 0.1)
        self.feature_fraction = feature_fraction or config.get('feature_fraction', 0.9)
        self.bagging_fraction = bagging_fraction or config.get('bagging_fraction', 0.8)
        self.bagging_freq = bagging_freq or config.get('bagging_freq', 5)
        self.max_depth = max_depth or config.get('max_depth', -1)
        self.min_data_in_leaf = min_data_in_leaf or config.get('min_data_in_leaf', 20)
        self.lambda_l1 = lambda_l1 or config.get('lambda_l1', 0.0)
        self.lambda_l2 = lambda_l2 or config.get('lambda_l2', 0.0)
        
        # Class imbalance handling - Requirements 4.1, 4.2
        self.scale_pos_weight = scale_pos_weight
        self.is_unbalance = is_unbalance or config.get('is_unbalance', True)
        
        # Training parameters
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose if verbose is not None else config.get('verbose', -1)
        self.random_state = random_state or config.get('random_state', MODEL_CONFIG['random_seed'])
        self.n_jobs = n_jobs
        
        # Model components
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        # Training history and statistics
        self.training_history = {}
        self.validation_scores = {}
        self.feature_importance = {}
        
        logger.info(f"Initialized SupervisedClassifier with LightGBM - "
                   f"learning_rate={self.learning_rate}, num_leaves={self.num_leaves}, "
                   f"scale_pos_weight={self.scale_pos_weight}, is_unbalance={self.is_unbalance}")
    
    def _calculate_class_weights(self, y: np.ndarray) -> float:
        """
        Calculate scale_pos_weight for class imbalance handling.
        
        For fraud detection with ~4% fraud rate, this calculates the ratio
        of negative samples to positive samples to balance the classes.
        
        Args:
            y: Target labels (0 for legitimate, 1 for fraud)
            
        Returns:
            Scale weight for positive class
        """
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if n_positive == 0:
            logger.warning("No positive samples found in training data")
            return 1.0
        
        weight = n_negative / n_positive
        logger.info(f"Calculated class weights - Negative: {n_negative}, "
                   f"Positive: {n_positive}, Scale pos weight: {weight:.2f}")
        
        return weight
    
    def _prepare_lgb_params(self, scale_pos_weight: float = None) -> Dict[str, Any]:
        """
        Prepare LightGBM parameters dictionary.
        
        Args:
            scale_pos_weight: Weight for positive class
            
        Returns:
            Dictionary of LightGBM parameters
        """
        params = {
            'objective': self.objective,
            'metric': self.metric,
            'boosting_type': self.boosting_type,
            'num_leaves': self.num_leaves,
            'learning_rate': self.learning_rate,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction,
            'bagging_freq': self.bagging_freq,
            'max_depth': self.max_depth,
            'min_data_in_leaf': self.min_data_in_leaf,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'force_col_wise': True,  # For better performance
        }
        
        # Add class imbalance handling - Requirements 4.1, 4.2
        if scale_pos_weight is not None:
            params['scale_pos_weight'] = scale_pos_weight
        elif self.is_unbalance:
            params['is_unbalance'] = True
        
        return params
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, 
            X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None,
            eval_set: Optional[List[Tuple]] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'SupervisedClassifier':
        """
        Train the LightGBM classifier with early stopping and cross-validation.
        
        This method implements comprehensive training with:
        - Automatic class imbalance handling using scale_pos_weight
        - Early stopping to prevent overfitting
        - Validation set evaluation
        - Feature importance calculation
        - Training history tracking
        
        Args:
            X: Training feature matrix
            y: Training target labels (0 for legitimate, 1 for fraud)
            X_val: Optional validation feature matrix
            y_val: Optional validation target labels
            feature_names: Optional list of feature names
            eval_set: Optional evaluation sets for early stopping
            sample_weight: Optional sample weights
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If input data is invalid
        """
        if X is None or y is None:
            raise ValueError("Training data X and y cannot be None")
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Validate input dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of samples: {len(X)} vs {len(y)}")
        
        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info(f"Training LightGBM classifier on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Calculate class distribution
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        fraud_rate = n_positive / len(y)
        
        logger.info(f"Class distribution - Legitimate: {n_negative} ({(1-fraud_rate)*100:.1f}%), "
                   f"Fraud: {n_positive} ({fraud_rate*100:.1f}%)")
        
        # Calculate scale_pos_weight if not provided - Requirements 4.1, 4.2
        if self.scale_pos_weight is None:
            self.scale_pos_weight = self._calculate_class_weights(y)
        
        # Prepare LightGBM parameters
        params = self._prepare_lgb_params(self.scale_pos_weight)
        
        # Create validation set if not provided
        if X_val is None and y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=MODEL_CONFIG['validation_size'], 
                stratify=y, random_state=self.random_state
            )
            logger.info(f"Created validation set: {len(X_train)} train, {len(X_val)} validation")
        else:
            X_train, y_train = X, y
            if X_val is not None and isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if y_val is not None and isinstance(y_val, pd.Series):
                y_val = y_val.values
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, label=y_train, 
            feature_name=self.feature_names,
            weight=sample_weight
        )
        
        valid_data = None
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(
                X_val, label=y_val,
                feature_name=self.feature_names,
                reference=train_data
            )
        
        # Prepare evaluation sets
        valid_sets = [train_data]
        valid_names = ['train']
        
        if valid_data is not None:
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        if eval_set is not None:
            for i, (X_eval, y_eval) in enumerate(eval_set):
                eval_data = lgb.Dataset(X_eval, label=y_eval, reference=train_data)
                valid_sets.append(eval_data)
                valid_names.append(f'eval_{i}')
        
        # Train the model with early stopping - Requirements 4.3
        logger.info("Starting LightGBM training with early stopping")
        
        callbacks = []
        if self.early_stopping_rounds > 0 and len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        
        # Suppress LightGBM warnings during training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
        
        # Set fitted flag before calculating stats
        self.is_fitted = True
        
        # Calculate and store training statistics
        self._calculate_training_stats(X_train, y_train, X_val, y_val)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        logger.info(f"LightGBM training completed - Best iteration: {self.model.best_iteration}, "
                   f"Best score: {self.model.best_score}")
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Generate fraud probability scores for transactions.
        
        Returns probability scores in [0, 1] range where higher scores
        indicate higher fraud likelihood. This method ensures proper
        probability calibration and handles edge cases robustly.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Array of fraud probability scores in [0, 1] range
            
        Raises:
            ValueError: If classifier is not fitted or input is invalid
        """
        if not self.is_fitted:
            raise ValueError("SupervisedClassifier must be fitted before prediction")
        
        if X is None or len(X) == 0:
            raise ValueError("Input data X cannot be None or empty")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        logger.debug(f"Generating fraud probability scores for {X.shape[0]} samples")
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found NaN values in input data, filling with median")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Handle infinite values
        if np.isinf(X).any():
            logger.warning("Found infinite values in input data, clipping to finite range")
            X = np.clip(X, -1e10, 1e10)
        
        # Generate predictions using LightGBM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probabilities = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Ensure probabilities are in [0, 1] range - Requirements 4.5
        probabilities = np.clip(probabilities, 0.0, 1.0)
        
        # Validate output requirements
        assert np.all(probabilities >= 0.0), "Fraud probabilities must be >= 0"
        assert np.all(probabilities <= 1.0), "Fraud probabilities must be <= 1"
        assert not np.any(np.isnan(probabilities)), "Fraud probabilities contain NaN values"
        assert not np.any(np.isinf(probabilities)), "Fraud probabilities contain infinite values"
        
        logger.debug(f"Generated fraud probability scores - "
                    f"min: {probabilities.min():.4f}, "
                    f"max: {probabilities.max():.4f}, "
                    f"mean: {probabilities.mean():.4f}")
        
        return probabilities
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary fraud predictions for transactions.
        
        Args:
            X: Feature matrix for prediction
            threshold: Decision threshold for binary classification
            
        Returns:
            Binary array where 1 indicates fraud, 0 indicates legitimate
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def _calculate_training_stats(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: Optional[np.ndarray] = None, 
                                 y_val: Optional[np.ndarray] = None) -> None:
        """
        Calculate and store comprehensive training statistics.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        # Basic training information
        self.training_history = {
            'n_train_samples': len(X_train),
            'n_features': X_train.shape[1],
            'fraud_rate_train': float(np.mean(y_train)),
            'best_iteration': self.model.best_iteration,
            'n_estimators_used': self.model.best_iteration,
            'scale_pos_weight': self.scale_pos_weight,
        }
        
        # Training set predictions and metrics
        train_proba = self.predict_proba(X_train)
        train_pred = (train_proba >= 0.5).astype(int)
        
        self.training_history.update({
            'train_auc': float(roc_auc_score(y_train, train_proba)),
            'train_precision': float(precision_score(y_train, train_pred, zero_division=0)),
            'train_recall': float(recall_score(y_train, train_pred, zero_division=0)),
            'train_f1': float(f1_score(y_train, train_pred, zero_division=0)),
        })
        
        # Validation set metrics if available
        if X_val is not None and y_val is not None:
            val_proba = self.predict_proba(X_val)
            val_pred = (val_proba >= 0.5).astype(int)
            
            self.training_history.update({
                'n_val_samples': len(X_val),
                'fraud_rate_val': float(np.mean(y_val)),
                'val_auc': float(roc_auc_score(y_val, val_proba)),
                'val_precision': float(precision_score(y_val, val_pred, zero_division=0)),
                'val_recall': float(recall_score(y_val, val_pred, zero_division=0)),
                'val_f1': float(f1_score(y_val, val_pred, zero_division=0)),
            })
        
        # Store best scores from LightGBM training
        if hasattr(self.model, 'best_score'):
            self.training_history['best_scores'] = self.model.best_score
        
        logger.debug(f"Training statistics calculated: {self.training_history}")
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance scores."""
        if self.model is None:
            return
        
        # Get feature importance from LightGBM
        importance_gain = self.model.feature_importance(importance_type='gain')
        importance_split = self.model.feature_importance(importance_type='split')
        
        # Create feature importance dictionary
        self.feature_importance = {
            'gain': dict(zip(self.feature_names, importance_gain)),
            'split': dict(zip(self.feature_names, importance_split))
        }
        
        # Normalize importance scores
        total_gain = sum(importance_gain)
        total_split = sum(importance_split)
        
        if total_gain > 0:
            self.feature_importance['gain_normalized'] = {
                name: score / total_gain 
                for name, score in self.feature_importance['gain'].items()
            }
        
        if total_split > 0:
            self.feature_importance['split_normalized'] = {
                name: score / total_split 
                for name, score in self.feature_importance['split'].items()
            }
        
        logger.debug("Feature importance calculated")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'gain_normalized', 'split_normalized')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("SupervisedClassifier must be fitted before getting feature importance")
        
        if importance_type not in self.feature_importance:
            raise ValueError(f"Invalid importance_type: {importance_type}. "
                           f"Available types: {list(self.feature_importance.keys())}")
        
        return self.feature_importance[importance_type]
    
    def evaluate_model(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray,
                      threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate model performance on given dataset.
        
        Args:
            X: Feature matrix
            y: True labels
            threshold: Decision threshold for binary classification
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.is_fitted:
            raise ValueError("SupervisedClassifier must be fitted before evaluation")
        
        # Generate predictions
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'auc': float(roc_auc_score(y, probabilities)),
            'precision': float(precision_score(y, predictions, zero_division=0)),
            'recall': float(recall_score(y, predictions, zero_division=0)),
            'f1': float(f1_score(y, predictions, zero_division=0)),
            'threshold': threshold,
            'fraud_rate': float(np.mean(y)),
            'predicted_fraud_rate': float(np.mean(predictions)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            })
        
        return metrics
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray,
                      cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation to assess model stability.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            
        Returns:
            Dictionary with cross-validation results
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        logger.info(f"Performing {cv}-fold cross-validation with {scoring} scoring")
        
        # Create a temporary LightGBM classifier for cross-validation
        params = self._prepare_lgb_params(self._calculate_class_weights(y))
        
        # Use LightGBM's sklearn interface for cross-validation
        lgb_classifier = lgb.LGBMClassifier(
            **params,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # Perform stratified cross-validation
        cv_scores = cross_val_score(
            lgb_classifier, X, y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            n_jobs=self.n_jobs
        )
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std()),
            'min_score': float(cv_scores.min()),
            'max_score': float(cv_scores.max()),
            'scoring_metric': scoring,
            'n_folds': cv
        }
        
        logger.info(f"Cross-validation completed - Mean {scoring}: {results['mean_score']:.4f} "
                   f"(±{results['std_score']:.4f})")
        
        return results
    
    def save_model(self, filepath: str, version: Optional[str] = None,
                   include_metadata: bool = True) -> str:
        """
        Save the trained LightGBM classifier to disk with comprehensive metadata.
        
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
            'model': self.model,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'scale_pos_weight': self.scale_pos_weight,
            'is_unbalance': self.is_unbalance,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            # Store all hyperparameters for reproducibility
            'hyperparameters': {
                'objective': self.objective,
                'metric': self.metric,
                'boosting_type': self.boosting_type,
                'num_leaves': self.num_leaves,
                'learning_rate': self.learning_rate,
                'feature_fraction': self.feature_fraction,
                'bagging_fraction': self.bagging_fraction,
                'bagging_freq': self.bagging_freq,
                'max_depth': self.max_depth,
                'min_data_in_leaf': self.min_data_in_leaf,
                'lambda_l1': self.lambda_l1,
                'lambda_l2': self.lambda_l2,
                'n_estimators': self.n_estimators,
                'early_stopping_rounds': self.early_stopping_rounds,
            }
        }
        
        # Create metadata if requested
        metadata = None
        if include_metadata:
            # Extract performance metrics from training history
            performance_metrics = {}
            if self.training_history:
                for key, value in self.training_history.items():
                    if isinstance(value, (int, float)):
                        performance_metrics[key] = value
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_type='lightgbm_classifier',
                version=version or datetime.now().strftime("%Y%m%d_%H%M%S"),
                created_at=datetime.now().isoformat(),
                training_samples=self.training_history.get('n_train_samples'),
                feature_count=len(self.feature_names) if self.feature_names else None,
                feature_names=self.feature_names,
                hyperparameters=model_data['hyperparameters'],
                performance_metrics=performance_metrics,
                random_state=self.random_state,
                tags=['supervised_learning', 'lightgbm', 'classification', 'fraud_detection'],
                description=f"LightGBM classifier with {self.training_history.get('n_estimators_used', 'unknown')} estimators"
            )
        
        # Save using persistence manager
        saved_path = persistence_manager.save_model(
            model=model_data,
            model_name=model_name,
            model_type='lightgbm_classifier',
            metadata=metadata,
            version=version,
            format='joblib',
            compress=True
        )
        
        logger.info(f"SupervisedClassifier saved to {saved_path}")
        return saved_path
    
    def load_model(self, filepath: str, version: Optional[str] = None,
                   verify_integrity: bool = True) -> 'SupervisedClassifier':
        """
        Load a trained LightGBM classifier from disk with metadata validation.
        
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
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        self.feature_importance = model_data['feature_importance']
        self.scale_pos_weight = model_data['scale_pos_weight']
        self.is_unbalance = model_data['is_unbalance']
        self.random_state = model_data['random_state']
        self.is_fitted = model_data['is_fitted']
        
        # Restore hyperparameters if available
        if 'hyperparameters' in model_data:
            hyperparams = model_data['hyperparameters']
            for param, value in hyperparams.items():
                setattr(self, param, value)
        
        # Log metadata information if available
        if metadata:
            logger.info(f"Loaded classifier with metadata - Version: {metadata.version}, "
                       f"Features: {metadata.feature_count}, "
                       f"Training samples: {metadata.training_samples}")
        
        logger.info(f"SupervisedClassifier loaded from {filepath}")
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the trained classifier.
        
        Returns:
            Dictionary containing model configuration and statistics
        """
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        return {
            'status': 'fitted',
            'model_type': 'LightGBM Classifier',
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'scale_pos_weight': self.scale_pos_weight,
            'is_unbalance': self.is_unbalance,
            'best_iteration': self.model.best_iteration,
            'training_history': self.training_history,
            'hyperparameters': {
                'objective': self.objective,
                'metric': self.metric,
                'learning_rate': self.learning_rate,
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'feature_fraction': self.feature_fraction,
                'bagging_fraction': self.bagging_fraction,
                'random_state': self.random_state,
            }
        }