"""
Dual-Evaluation Pipeline for Fraud Detection

This package provides a comprehensive system for comparing fraud detection performance
between models trained on imbalanced versus balanced datasets.
"""

from .data_models import EvaluationResult, ComparisonReport
from .data_loader import DataLoader, DataValidationError
from .imbalanced_pipeline import ImbalancedPipeline, ModelError
from .balanced_pipeline import BalancedPipeline
from .metrics_comparator import MetricsComparator
from .dual_evaluation_pipeline import DualEvaluationPipeline

__all__ = [
    'EvaluationResult', 
    'ComparisonReport', 
    'DataLoader', 
    'DataValidationError',
    'ImbalancedPipeline',
    'BalancedPipeline',
    'ModelError',
    'MetricsComparator',
    'DualEvaluationPipeline'
]
