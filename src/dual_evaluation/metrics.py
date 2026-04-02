"""
Metrics calculation utilities for dual evaluation pipeline.

This module provides functions for calculating comprehensive evaluation metrics
including accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrix,
and false positive rate.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
"""

import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate all required evaluation metrics.
    
    This function computes comprehensive classification metrics including:
    - Accuracy, precision, recall, F1-score (Requirement 8.1)
    - ROC-AUC and PR-AUC scores (Requirement 8.2)
    - Confusion matrix values: TP, TN, FP, FN (Requirement 8.3)
    - False positive rate (Requirement 8.4)
    
    The function handles edge cases such as division by zero and ensures
    all metrics are calculated according to standard formulas.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Binary predictions (0 or 1)
        y_proba: Prediction probabilities in range [0, 1]
        
    Returns:
        Dictionary containing all calculated metrics:
            - accuracy: Overall accuracy
            - precision: TP / (TP + FP) if denominator > 0, else 0
            - recall: TP / (TP + FN) if denominator > 0, else 0
            - f1_score: 2 * (precision * recall) / (precision + recall) if denominator > 0, else 0
            - roc_auc: Area under ROC curve
            - pr_auc: Area under precision-recall curve
            - true_positives: Count of true positives
            - true_negatives: Count of true negatives
            - false_positives: Count of false positives
            - false_negatives: Count of false negatives
            - false_positive_rate: FP / (FP + TN) if denominator > 0, else 0
            
    Raises:
        AssertionError: If confusion matrix values don't sum to total samples
        
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8
    
    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> y_proba = np.array([0.9, 0.2, 0.8, 0.4, 0.1])
        >>> metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        >>> metrics['accuracy']
        0.8
        >>> metrics['precision']
        1.0
        >>> metrics['recall']
        0.666...
    """
    # Calculate confusion matrix (Requirement 8.3)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Verify confusion matrix sums to total samples (Requirement 8.5)
    assert (tp + tn + fp + fn) == len(y_true), \
        "Confusion matrix values must sum to total samples"
    
    # Calculate precision (Requirement 8.6)
    # Formula: TP / (TP + FP) if denominator is positive, else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Calculate recall (Requirement 8.7)
    # Formula: TP / (TP + FN) if denominator is positive, else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F1-score (Requirement 8.8)
    # Formula: 2 * (precision * recall) / (precision + recall) if denominator is positive, else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate accuracy (Requirement 8.1)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate ROC-AUC and PR-AUC (Requirement 8.2)
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        # Handle case where only one class is present
        roc_auc = 0.0
        logger.warning("ROC-AUC could not be calculated (only one class present)")
    
    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        pr_auc = 0.0
        logger.warning("PR-AUC could not be calculated (only one class present)")
    
    # Calculate false positive rate (Requirement 8.4)
    # Formula: FP / (FP + TN) if denominator is positive, else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'false_positive_rate': fpr
    }
