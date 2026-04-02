"""
ImbalancedPipeline component for dual-evaluation pipeline.

This module provides functionality for evaluating pretrained fraud detection models
on imbalanced test data. It loads existing models without retraining, generates
predictions, calculates comprehensive metrics, and optimizes classification thresholds
for imbalanced data constraints.

Requirements: 4.1-4.5, 6.1-6.5, 8.1-8.4, 16.1, 16.4
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
import joblib
import pickle
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import time

from .data_models import EvaluationResult
from .metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Raised when model loading or validation fails."""
    pass


class ImbalancedPipeline:
    """
    Evaluates pretrained models on imbalanced test data.
    
    This class provides methods for:
    - Loading pretrained FraudDetector models from disk
    - Generating predictions with probabilities
    - Calculating comprehensive evaluation metrics
    - Optimizing thresholds for imbalanced data constraints
    
    Requirements: 4.1-4.5, 6.1-6.5
    """
    
    def __init__(self, model_path: str, random_state: int = 42):
        """
        Initialize ImbalancedPipeline.
        
        Args:
            model_path: Path to pretrained model directory or file
            random_state: Random seed for reproducibility
            
        Requirements: 4.1
        """
        self.model_path = model_path
        self.random_state = random_state
        self.model = None
        
        logger.info(f"ImbalancedPipeline initialized with model_path={model_path}")
    
    def load_pretrained_model(self):
        """
        Load pretrained FraudDetector from disk without retraining.
        
        This method loads an existing trained model using joblib, verifies it is
        fitted, and handles various error conditions.
        
        Returns:
            Loaded FraudDetector model
            
        Raises:
            FileNotFoundError: If model file doesn't exist at specified path
            ModelError: If model file is corrupted or model is not fitted
            
        Requirements: 4.1, 4.2, 4.3, 16.1, 16.4
        """
        logger.info(f"Loading pretrained model from {self.model_path}")
        
        # Convert to Path object for easier manipulation
        model_path = Path(self.model_path)
        
        # Determine the actual model file path
        if model_path.is_dir():
            # Look for .joblib file in directory structure
            # Check for versioned structure: model_name/version/model_name.joblib
            joblib_files = list(model_path.rglob("*.joblib"))
            
            if not joblib_files:
                raise FileNotFoundError(
                    f"No .joblib model files found in directory: {self.model_path}"
                )
            
            # Use the most recent model file (by modification time)
            model_file = max(joblib_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found model file: {model_file}")
        else:
            model_file = model_path
        
        # Verify file exists (Requirement 4.2)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Pretrained model file not found: {model_file}. "
                f"Expected path: {self.model_path}"
            )
        
        try:
            # Load model using joblib (Requirement 4.1)
            logger.info(f"Loading model from {model_file}")
            
            # Try to import FraudDetector if available
            try:
                from src.fraud_detector import FraudDetector
                fraud_detector_available = True
            except ImportError:
                fraud_detector_available = False
                logger.warning("FraudDetector import not available, will use generic model loading")
            
            model_data = joblib.load(model_file)
            
            # Handle different model storage formats
            if isinstance(model_data, dict):
                # Model stored as dictionary with components
                if fraud_detector_available:
                    # Try to create proper FraudDetector instance
                    try:
                        self.model = FraudDetector()
                        
                        # Restore all components
                        self.model.preprocessor = model_data['preprocessor']
                        self.model.feature_engineer = model_data['feature_engineer']
                        self.model.anomaly_detector = model_data['anomaly_detector']
                        self.model.classifier = model_data['classifier']
                        self.model.risk_scorer = model_data['risk_scorer']
                        
                        # Restore state
                        self.model.is_fitted = model_data['is_fitted']
                        self.model.feature_names = model_data['feature_names']
                        self.model.column_mapping = model_data['column_mapping']
                        self.model.training_metadata = model_data.get('training_metadata', {})
                        self.model.performance_metrics = model_data.get('performance_metrics', {})
                        self.model.random_state = model_data.get('random_state', self.random_state)
                    except Exception as e:
                        logger.warning(f"Failed to create FraudDetector instance: {e}, using generic model")
                        fraud_detector_available = False
                
                if not fraud_detector_available:
                    # Create a simple object to hold the model data
                    class LoadedModel:
                        pass
                    
                    self.model = LoadedModel()
                    
                    # Restore all components
                    self.model.preprocessor = model_data['preprocessor']
                    self.model.feature_engineer = model_data['feature_engineer']
                    self.model.anomaly_detector = model_data['anomaly_detector']
                    self.model.classifier = model_data['classifier']
                    self.model.risk_scorer = model_data['risk_scorer']
                    
                    # Restore state
                    self.model.is_fitted = model_data['is_fitted']
                    self.model.feature_names = model_data['feature_names']
                    self.model.column_mapping = model_data['column_mapping']
                    self.model.training_metadata = model_data.get('training_metadata', {})
                    self.model.performance_metrics = model_data.get('performance_metrics', {})
                    self.model.random_state = model_data.get('random_state', self.random_state)
                
            else:
                # Model stored directly as FraudDetector object
                self.model = model_data
            
            # Verify model is fitted (Requirement 4.3)
            if not hasattr(self.model, 'is_fitted') or not self.model.is_fitted:
                raise ModelError(
                    f"Loaded model is not fitted. Model must be trained before evaluation."
                )
            
            logger.info("Pretrained model loaded successfully")
            logger.info(f"Model is_fitted: {self.model.is_fitted}")
            
            return self.model
            
        except (EOFError, pickle.UnpicklingError, ValueError) as e:
            # Handle corrupted model file (Requirement 16.4)
            logger.error(f"Model file is corrupted: {str(e)}")
            raise ModelError(
                f"Failed to load model from {model_file}. "
                f"File may be corrupted or incompatible. Error: {str(e)}"
            )
        except ModuleNotFoundError as e:
            # Handle missing module dependencies
            logger.error(f"Missing module dependency: {str(e)}")
            raise ModelError(
                f"Failed to load model from {model_file}. "
                f"Missing required module: {str(e)}. "
                f"Ensure all fraud detection components are properly installed."
            )
        except Exception as e:
            logger.error(f"Unexpected error loading model: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        fraud_col: str = 'is_fraud'
    ) -> EvaluationResult:
        """
        Evaluate pretrained model on imbalanced test data.
        
        This method generates predictions with probabilities, calculates all required
        metrics, and returns a complete EvaluationResult.
        
        Args:
            test_df: Test DataFrame with ground truth labels
            fraud_col: Name of fraud indicator column
            
        Returns:
            EvaluationResult with complete evaluation metrics
            
        Raises:
            ValueError: If model not loaded or test data invalid
            
        Requirements: 4.4, 4.5, 8.1, 8.2, 8.3, 8.4
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_pretrained_model() first."
            )
        
        if fraud_col not in test_df.columns:
            raise ValueError(
                f"Fraud column '{fraud_col}' not found in test DataFrame"
            )
        
        logger.info(f"Evaluating model on {len(test_df):,} test samples")
        
        start_time = time.time()
        
        # Generate predictions with probabilities (Requirement 4.4)
        logger.info("Generating predictions...")
        predictions = self.model.predict(
            test_df,
            return_probabilities=True,
            return_risk_levels=False,
            return_explanations=False
        )
        
        # Extract ground truth and predictions
        y_true = test_df[fraud_col].values
        y_pred = predictions['fraud_prediction'].values
        y_proba = predictions['fraud_probability'].values
        
        # Calculate all required metrics (Requirement 4.5, 8.1-8.4)
        metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        
        evaluation_time = time.time() - start_time
        
        # Calculate dataset statistics
        test_fraud_rate = y_true.mean()
        
        # Create EvaluationResult (Requirement 8.1-8.4)
        result = EvaluationResult(
            model_name="pretrained_imbalanced",
            dataset_type="imbalanced",
            train_samples=0,  # Not applicable for pretrained model
            test_samples=len(test_df),
            train_fraud_rate=0.0,  # Not applicable
            test_fraud_rate=test_fraud_rate,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            roc_auc=metrics['roc_auc'],
            pr_auc=metrics['pr_auc'],
            true_positives=metrics['true_positives'],
            true_negatives=metrics['true_negatives'],
            false_positives=metrics['false_positives'],
            false_negatives=metrics['false_negatives'],
            fraud_detection_rate=metrics['recall'],  # Same as recall
            false_positive_rate=metrics['false_positive_rate'],
            customer_friction_rate=metrics['false_positive_rate'],  # FPR indicates friction
            optimal_threshold=0.5,  # Default, will be updated by optimize_threshold
            threshold_range=(0.0, 1.0),
            training_time=None,  # No training performed
            evaluation_time=evaluation_time
        )
        
        logger.info(f"Evaluation complete in {evaluation_time:.2f}s")
        logger.info(f"Metrics - Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"Recall: {metrics['recall']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return result
    
    def optimize_threshold(
        self,
        test_df: pd.DataFrame,
        y_proba: np.ndarray,
        fraud_col: str = 'is_fraud',
        target_recall_min: float = 0.70,
        target_recall_max: float = 0.85,
        max_fpr: float = 0.05
    ) -> Tuple[float, EvaluationResult]:
        """
        Optimize classification threshold for imbalanced data constraints.
        
        This method searches for a threshold that achieves recall between 70-85%
        while keeping false positive rate below 5%. If no threshold meets all
        constraints, returns the threshold with best recall under FPR constraint.
        
        Args:
            test_df: Test DataFrame with ground truth labels
            y_proba: Predicted fraud probabilities
            fraud_col: Name of fraud indicator column
            target_recall_min: Minimum target recall (default 0.70)
            target_recall_max: Maximum target recall (default 0.85)
            max_fpr: Maximum allowed false positive rate (default 0.05)
            
        Returns:
            Tuple of (optimal_threshold, updated_evaluation_result)
            
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
        """
        logger.info("Optimizing threshold for imbalanced data...")
        logger.info(f"Target recall: {target_recall_min:.2f}-{target_recall_max:.2f}, "
                   f"Max FPR: {max_fpr:.2f}")
        
        y_true = test_df[fraud_col].values
        
        # Search thresholds from 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)
        
        best_threshold = 0.5
        best_recall = 0.0
        best_metrics = None
        constraint_met = False
        
        # Search for threshold meeting constraints (Requirement 6.1, 6.2)
        for threshold in thresholds:
            y_pred_temp = (y_proba >= threshold).astype(int)
            
            # Calculate metrics for this threshold
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_temp).ravel()
            
            # Calculate recall and FPR
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # Check if FPR constraint is met
            if fpr <= max_fpr:
                # Check if recall is in target range
                if target_recall_min <= recall <= target_recall_max:
                    # Found threshold meeting all constraints
                    best_threshold = threshold
                    best_recall = recall
                    constraint_met = True
                    logger.info(f"Found threshold {threshold:.3f} meeting all constraints: "
                               f"recall={recall:.4f}, FPR={fpr:.4f}")
                    break
                elif recall > best_recall:
                    # Track best recall under FPR constraint (Requirement 6.3)
                    best_threshold = threshold
                    best_recall = recall
                    best_metrics = (recall, fpr)
        
        # Log warning if constraints not met (Requirement 6.5)
        if not constraint_met:
            logger.warning(
                f"Could not find threshold meeting all constraints. "
                f"Using threshold {best_threshold:.3f} with recall={best_recall:.4f}, "
                f"FPR={best_metrics[1]:.4f} (best recall under FPR constraint)"
            )
        
        # Recalculate all metrics with optimal threshold (Requirement 6.4)
        y_pred_optimal = (y_proba >= best_threshold).astype(int)
        metrics = calculate_all_metrics(y_true, y_pred_optimal, y_proba)
        
        # Create updated EvaluationResult
        test_fraud_rate = y_true.mean()
        
        result = EvaluationResult(
            model_name="pretrained_imbalanced",
            dataset_type="imbalanced",
            train_samples=0,
            test_samples=len(test_df),
            train_fraud_rate=0.0,
            test_fraud_rate=test_fraud_rate,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            roc_auc=metrics['roc_auc'],
            pr_auc=metrics['pr_auc'],
            true_positives=metrics['true_positives'],
            true_negatives=metrics['true_negatives'],
            false_positives=metrics['false_positives'],
            false_negatives=metrics['false_negatives'],
            fraud_detection_rate=metrics['recall'],
            false_positive_rate=metrics['false_positive_rate'],
            customer_friction_rate=metrics['false_positive_rate'],
            optimal_threshold=best_threshold,
            threshold_range=(target_recall_min, target_recall_max),
            training_time=None,
            evaluation_time=0.0  # Optimization time not tracked separately
        )
        
        logger.info(f"Optimal threshold: {best_threshold:.3f}")
        logger.info(f"Final metrics - Recall: {metrics['recall']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"FPR: {metrics['false_positive_rate']:.4f}")
        
        return best_threshold, result

    def _calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate all required evaluation metrics.
        
        This is a wrapper method that delegates to the centralized
        calculate_all_metrics function in the metrics module.
        
        Args:
            y_true: Ground truth labels
            y_pred: Binary predictions
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary with all metrics
            
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
        """
        return calculate_all_metrics(y_true, y_pred, y_proba)
