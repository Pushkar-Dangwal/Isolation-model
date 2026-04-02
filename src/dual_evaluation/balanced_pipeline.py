"""
BalancedPipeline component for dual-evaluation pipeline.

This module provides functionality for training fraud detection models on balanced
datasets and evaluating their performance. It creates new FraudDetector instances,
trains them on balanced data, generates predictions, calculates comprehensive metrics,
and optimizes classification thresholds for balanced data constraints.

Requirements: 5.1-5.5, 7.1-7.5, 8.1-8.4, 15.1-15.4, 20.4
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import time

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from .data_models import EvaluationResult
from .metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Raised when model training or validation fails."""
    pass


class BalancedPipeline:
    """
    Trains and evaluates models on balanced datasets.
    
    This class provides methods for:
    - Training new FraudDetector models on balanced data
    - Generating predictions with probabilities
    - Calculating comprehensive evaluation metrics
    - Optimizing thresholds for balanced data constraints
    
    Requirements: 5.1-5.5, 7.1-7.5
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize BalancedPipeline.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for model training
            
        Requirements: 5.1
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
        
        logger.info(f"BalancedPipeline initialized with random_state={random_state}, n_jobs={n_jobs}")
    
    def train_model(
        self,
        train_df: pd.DataFrame,
        fraud_col: str = 'is_fraud',
        validation_split: float = 0.2
    ):
        """
        Train new FraudDetector model on balanced dataset.
        
        This method creates a new FraudDetector instance with the same architecture
        as the pretrained model, trains it on balanced data, and tracks training time.
        
        Args:
            train_df: Training DataFrame with balanced data
            fraud_col: Name of fraud indicator column
            validation_split: Fraction of data to use for validation
            
        Returns:
            Trained FraudDetector model
            
        Raises:
            ValueError: If training data is invalid
            ModelError: If model training fails
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 15.1, 15.2, 15.3, 15.4, 20.4
        """
        if train_df is None or len(train_df) == 0:
            raise ValueError("Training data cannot be None or empty")
        
        if fraud_col not in train_df.columns:
            raise ValueError(f"Fraud column '{fraud_col}' not found in training DataFrame")
        
        logger.info(f"Training new model on {len(train_df):,} balanced samples")
        
        try:
            # Import FraudDetector (Requirement 5.1)
            from src.fraud_detector import FraudDetector
            
            # Create new FraudDetector instance with same architecture (Requirements 5.2, 15.1-15.4)
            # Use same random_state for reproducibility (Requirement 20.4)
            logger.info("Creating new FraudDetector instance with same architecture as pretrained model")
            self.model = FraudDetector(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=True,
                ensure_reproducibility=True,
                strict_determinism=False
            )
            
            # Track training time (Requirement 5.4)
            training_start = time.time()
            
            # Train model on balanced data (Requirement 5.1, 5.3)
            logger.info("Training model on balanced dataset...")
            self.model.fit(
                train_df,
                target_column=fraud_col,
                transaction_id_column='transaction_id',
                validation_split=validation_split,
                optimize_thresholds=True
            )
            
            training_time = time.time() - training_start
            
            # Verify model is fitted (Requirement 5.5)
            if not hasattr(self.model, 'is_fitted') or not self.model.is_fitted:
                raise ModelError(
                    "Model training completed but model is not fitted. "
                    "Training may have failed silently."
                )
            
            logger.info(f"Model training completed successfully in {training_time:.2f}s")
            logger.info(f"Model is_fitted: {self.model.is_fitted}")
            
            # Store training time in model metadata
            if not hasattr(self.model, 'training_metadata'):
                self.model.training_metadata = {}
            self.model.training_metadata['training_time'] = training_time
            
            return self.model
            
        except ImportError as e:
            logger.error(f"Failed to import FraudDetector: {str(e)}")
            raise ModelError(
                f"Failed to import FraudDetector. "
                f"Ensure fraud detection components are properly installed. Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise ModelError(f"Failed to train model on balanced data: {str(e)}")
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        fraud_col: str = 'is_fraud'
    ) -> EvaluationResult:
        """
        Evaluate trained model on balanced test data.
        
        This method generates predictions with probabilities, calculates all required
        metrics, and returns a complete EvaluationResult.
        
        Args:
            test_df: Test DataFrame with ground truth labels
            fraud_col: Name of fraud indicator column
            
        Returns:
            EvaluationResult with complete evaluation metrics
            
        Raises:
            ValueError: If model not trained or test data invalid
            
        Requirements: 4.4, 4.5, 8.1, 8.2, 8.3, 8.4
        """
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train_model() first."
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
        metrics = self._calculate_all_metrics(y_true, y_pred, y_proba)
        
        evaluation_time = time.time() - start_time
        
        # Calculate dataset statistics
        test_fraud_rate = y_true.mean()
        
        # Get training time from model metadata
        training_time = None
        if hasattr(self.model, 'training_metadata'):
            training_time = self.model.training_metadata.get('training_time')
        
        # Create EvaluationResult (Requirement 8.1-8.4)
        result = EvaluationResult(
            model_name="new_balanced",
            dataset_type="balanced",
            train_samples=0,  # Will be updated by caller if needed
            test_samples=len(test_df),
            train_fraud_rate=0.0,  # Will be updated by caller if needed
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
            training_time=training_time,
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
        target_recall_min: float = 0.85,
        target_recall_max: float = 0.95,
        target_precision_min: float = 0.70,
        target_precision_max: float = 0.90,
        target_f1_min: float = 0.75
    ) -> Tuple[float, EvaluationResult]:
        """
        Optimize classification threshold for balanced data constraints.
        
        This method searches for a threshold that achieves recall between 85-95%,
        precision between 70-90%, and F1-score above 75%. If no threshold meets all
        constraints, returns the threshold with best F1-score.
        
        Args:
            test_df: Test DataFrame with ground truth labels
            y_proba: Predicted fraud probabilities
            fraud_col: Name of fraud indicator column
            target_recall_min: Minimum target recall (default 0.85)
            target_recall_max: Maximum target recall (default 0.95)
            target_precision_min: Minimum target precision (default 0.70)
            target_precision_max: Maximum target precision (default 0.90)
            target_f1_min: Minimum target F1-score (default 0.75)
            
        Returns:
            Tuple of (optimal_threshold, updated_evaluation_result)
            
        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        logger.info("Optimizing threshold for balanced data...")
        logger.info(f"Target recall: {target_recall_min:.2f}-{target_recall_max:.2f}, "
                   f"Target precision: {target_precision_min:.2f}-{target_precision_max:.2f}, "
                   f"Target F1: >{target_f1_min:.2f}")
        
        y_true = test_df[fraud_col].values
        
        # Search thresholds from 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 99)
        
        best_threshold = 0.5
        best_f1 = 0.0
        best_metrics = None
        constraint_met = False
        
        # Search for threshold meeting constraints (Requirement 7.1, 7.2, 7.3)
        for threshold in thresholds:
            y_pred_temp = (y_proba >= threshold).astype(int)
            
            # Calculate metrics for this threshold
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_temp).ravel()
            
            # Calculate precision, recall, and F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Check if all constraints are met
            recall_ok = target_recall_min <= recall <= target_recall_max
            precision_ok = target_precision_min <= precision <= target_precision_max
            f1_ok = f1 > target_f1_min
            
            if recall_ok and precision_ok and f1_ok:
                # Found threshold meeting all constraints
                best_threshold = threshold
                best_f1 = f1
                constraint_met = True
                logger.info(f"Found threshold {threshold:.3f} meeting all constraints: "
                           f"recall={recall:.4f}, precision={precision:.4f}, F1={f1:.4f}")
                break
            elif f1 > best_f1:
                # Track best F1-score (Requirement 7.4)
                best_threshold = threshold
                best_f1 = f1
                best_metrics = (recall, precision, f1)
        
        # Log warning if constraints not met (Requirement 7.5)
        if not constraint_met:
            logger.warning(
                f"Could not find threshold meeting all constraints. "
                f"Using threshold {best_threshold:.3f} with best F1-score={best_f1:.4f}, "
                f"recall={best_metrics[0]:.4f}, precision={best_metrics[1]:.4f}"
            )
        
        # Recalculate all metrics with optimal threshold (Requirement 7.5)
        y_pred_optimal = (y_proba >= best_threshold).astype(int)
        metrics = self._calculate_all_metrics(y_true, y_pred_optimal, y_proba)
        
        # Create updated EvaluationResult
        test_fraud_rate = y_true.mean()
        
        # Get training time from model metadata
        training_time = None
        if hasattr(self.model, 'training_metadata'):
            training_time = self.model.training_metadata.get('training_time')
        
        result = EvaluationResult(
            model_name="new_balanced",
            dataset_type="balanced",
            train_samples=0,  # Will be updated by caller if needed
            test_samples=len(test_df),
            train_fraud_rate=0.0,  # Will be updated by caller if needed
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
            training_time=training_time,
            evaluation_time=0.0  # Optimization time not tracked separately
        )
        
        logger.info(f"Optimal threshold: {best_threshold:.3f}")
        logger.info(f"Final metrics - Recall: {metrics['recall']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
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

