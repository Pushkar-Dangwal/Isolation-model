"""
Data models for dual-evaluation pipeline.

This module defines the core data structures used throughout the dual-evaluation
pipeline for storing evaluation results and comparison reports.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd


@dataclass
class EvaluationResult:
    """
    Stores comprehensive evaluation results for a single pipeline.
    
    This dataclass captures all metrics, statistics, and metadata from evaluating
    a fraud detection model on either imbalanced or balanced data.
    
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    
    # Model identification
    model_name: str
    dataset_type: str  # 'imbalanced' or 'balanced'
    
    # Dataset statistics
    train_samples: int
    test_samples: int
    train_fraud_rate: float
    test_fraud_rate: float
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Curve metrics
    roc_auc: float
    pr_auc: float
    
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Business metrics
    fraud_detection_rate: float
    false_positive_rate: float
    customer_friction_rate: float
    
    # Threshold information
    optimal_threshold: float
    threshold_range: Tuple[float, float]
    
    # Timing
    training_time: Optional[float]
    evaluation_time: float


@dataclass
class ComparisonReport:
    """
    Stores comprehensive comparison between imbalanced and balanced pipelines.
    
    This dataclass aggregates results from both evaluation pipelines and includes
    side-by-side comparisons, trade-off analysis, interpretations, and visualizations.
    
    Requirements: 9.1, 13.1
    """
    
    # Timestamp
    timestamp: str
    
    # Results from both pipelines
    imbalanced_results: EvaluationResult
    balanced_results: EvaluationResult
    
    # Comparison table
    comparison_table: pd.DataFrame
    
    # Metric differences
    metric_differences: Dict[str, float]
    
    # Trade-off analysis
    trade_offs: Dict[str, Any]
    
    # Interpretation
    interpretation: str
    
    # Recommendations
    recommendations: List[str]
    
    # Visualization paths
    visualization_paths: List[str]
