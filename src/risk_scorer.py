"""
Risk scoring module for the fraud detection system.
Implements threshold optimization and risk level assignment for fraud detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from sklearn.base import BaseEstimator, TransformerMixin

import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import RISK_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)


class RiskScorer(BaseEstimator, TransformerMixin):
    """
    Implements risk scoring with threshold optimization for fraud detection.
    
    This class provides comprehensive risk assessment capabilities including:
    - Precision-recall threshold optimization
    - 3-level risk strategy (low/medium/high)
    - Business impact analysis
    - Optimal threshold recommendations
    
    The risk scorer is designed to minimize false positives while maintaining
    acceptable fraud recall, balancing customer experience with fraud prevention.
    """
    
    def __init__(self,
                 low_risk_threshold: float = None,
                 high_risk_threshold: float = None,
                 risk_levels: List[str] = None,
                 optimization_metric: str = 'f1',
                 min_precision: float = 0.1,
                 min_recall: float = 0.8):
        """
        Initialize the RiskScorer with configurable thresholds and optimization settings.
        
        Args:
            low_risk_threshold: Threshold below which transactions are low risk
            high_risk_threshold: Threshold above which transactions are high risk
            risk_levels: List of risk level names
            optimization_metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced')
            min_precision: Minimum acceptable precision for threshold optimization
            min_recall: Minimum acceptable recall for threshold optimization
        """
        # Use config defaults if not specified
        config = RISK_CONFIG
        
        self.low_risk_threshold = low_risk_threshold or config['thresholds']['low_risk']
        self.high_risk_threshold = high_risk_threshold or config['thresholds']['high_risk']
        self.risk_levels = risk_levels or config['risk_levels']
        
        # Optimization parameters
        self.optimization_metric = optimization_metric
        self.min_precision = min_precision
        self.min_recall = min_recall
        
        # Fitted parameters
        self.is_fitted = False
        self.optimal_thresholds = {}
        self.threshold_analysis = {}
        self.business_impact = {}
        
        # Validation
        if self.low_risk_threshold >= self.high_risk_threshold:
            raise ValueError("low_risk_threshold must be less than high_risk_threshold")
        
        if len(self.risk_levels) != 3:
            raise ValueError("risk_levels must contain exactly 3 levels")
        
        logger.info(f"Initialized RiskScorer - Low: {self.low_risk_threshold}, "
                   f"High: {self.high_risk_threshold}, Metric: {self.optimization_metric}")
    
    def tune_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray, 
                       sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize decision thresholds for precision-recall tradeoff.
        
        This method performs comprehensive threshold analysis to find optimal
        operating points that balance fraud detection performance with false
        positive minimization. It evaluates multiple metrics and provides
        recommendations for different business scenarios.
        
        Args:
            y_true: True binary labels (0 for legitimate, 1 for fraud)
            y_proba: Predicted fraud probabilities in [0, 1] range
            sample_weight: Optional sample weights for evaluation
            
        Returns:
            Dictionary containing optimal thresholds and analysis results
            
        Raises:
            ValueError: If input data is invalid
        """
        if y_true is None or y_proba is None:
            raise ValueError("y_true and y_proba cannot be None")
        
        if len(y_true) == 0 or len(y_proba) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(y_true) != len(y_proba):
            raise ValueError(f"y_true and y_proba must have same length: {len(y_true)} vs {len(y_proba)}")
        
        # Validate probability range
        if np.any(y_proba < 0) or np.any(y_proba > 1):
            raise ValueError("y_proba must be in [0, 1] range")
        
        # Validate binary labels
        unique_labels = np.unique(y_true)
        if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [0]) and not np.array_equal(unique_labels, [1]):
            raise ValueError("y_true must contain only 0 and 1 values")
        
        logger.info(f"Tuning thresholds on {len(y_true)} samples - "
                   f"Fraud rate: {np.mean(y_true):.3f}, "
                   f"Optimization metric: {self.optimization_metric}")
        
        # Calculate precision-recall curve
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_proba, sample_weight=sample_weight)
        
        # Calculate ROC curve for additional analysis
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba, sample_weight=sample_weight)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        
        # Find optimal thresholds based on different criteria
        optimal_thresholds = self._find_optimal_thresholds(
            precisions, recalls, f1_scores, pr_thresholds, fpr, tpr, roc_thresholds
        )
        
        # Perform comprehensive threshold analysis
        threshold_analysis = self._analyze_thresholds(
            y_true, y_proba, precisions, recalls, pr_thresholds, sample_weight
        )
        
        # Store results
        self.optimal_thresholds = optimal_thresholds
        self.threshold_analysis = threshold_analysis
        self.is_fitted = True
        
        # Calculate business impact analysis
        self.business_impact = self._calculate_business_impact(y_true, y_proba, optimal_thresholds)
        
        results = {
            'optimal_thresholds': optimal_thresholds,
            'threshold_analysis': threshold_analysis,
            'business_impact': self.business_impact,
            'pr_auc': auc(recalls, precisions),
            'roc_auc': auc(fpr, tpr),
            'baseline_fraud_rate': float(np.mean(y_true)),
            'n_samples': len(y_true),
            'n_fraud_samples': int(np.sum(y_true)),
            'n_legitimate_samples': int(np.sum(y_true == 0))
        }
        
        logger.info(f"Threshold tuning completed - "
                   f"Optimal F1 threshold: {optimal_thresholds['f1_optimal']:.3f}, "
                   f"PR-AUC: {results['pr_auc']:.3f}")
        
        return results
    
    def _find_optimal_thresholds(self, precisions: np.ndarray, recalls: np.ndarray, 
                                f1_scores: np.ndarray, pr_thresholds: np.ndarray,
                                fpr: np.ndarray, tpr: np.ndarray, 
                                roc_thresholds: np.ndarray) -> Dict[str, float]:
        """
        Find optimal thresholds based on different optimization criteria.
        
        Args:
            precisions: Precision values from precision-recall curve
            recalls: Recall values from precision-recall curve
            f1_scores: F1 scores for each threshold
            pr_thresholds: Thresholds from precision-recall curve
            fpr: False positive rates from ROC curve
            tpr: True positive rates from ROC curve
            roc_thresholds: Thresholds from ROC curve
            
        Returns:
            Dictionary of optimal thresholds for different criteria
        """
        optimal_thresholds = {}
        
        # F1-optimal threshold
        if len(f1_scores) > 0:
            f1_optimal_idx = np.argmax(f1_scores)
            optimal_thresholds['f1_optimal'] = float(pr_thresholds[f1_optimal_idx])
            optimal_thresholds['f1_score'] = float(f1_scores[f1_optimal_idx])
            optimal_thresholds['f1_precision'] = float(precisions[f1_optimal_idx])
            optimal_thresholds['f1_recall'] = float(recalls[f1_optimal_idx])
        
        # Precision-focused threshold (minimize false positives)
        precision_mask = precisions[:-1] >= self.min_precision
        if np.any(precision_mask):
            valid_recalls = recalls[:-1][precision_mask]
            valid_thresholds = pr_thresholds[precision_mask]
            if len(valid_recalls) > 0:
                precision_optimal_idx = np.argmax(valid_recalls)
                optimal_thresholds['precision_focused'] = float(valid_thresholds[precision_optimal_idx])
                optimal_thresholds['precision_focused_precision'] = float(precisions[:-1][precision_mask][precision_optimal_idx])
                optimal_thresholds['precision_focused_recall'] = float(valid_recalls[precision_optimal_idx])
        
        # Recall-focused threshold (minimize false negatives)
        recall_mask = recalls[:-1] >= self.min_recall
        if np.any(recall_mask):
            valid_precisions = precisions[:-1][recall_mask]
            valid_thresholds = pr_thresholds[recall_mask]
            if len(valid_precisions) > 0:
                recall_optimal_idx = np.argmax(valid_precisions)
                optimal_thresholds['recall_focused'] = float(valid_thresholds[recall_optimal_idx])
                optimal_thresholds['recall_focused_precision'] = float(valid_precisions[recall_optimal_idx])
                optimal_thresholds['recall_focused_recall'] = float(recalls[:-1][recall_mask][recall_optimal_idx])
        
        # Balanced threshold (Youden's J statistic from ROC curve)
        if len(tpr) > 0 and len(fpr) > 0:
            j_scores = tpr - fpr  # Youden's J statistic
            j_optimal_idx = np.argmax(j_scores)
            optimal_thresholds['balanced'] = float(roc_thresholds[j_optimal_idx])
            optimal_thresholds['balanced_tpr'] = float(tpr[j_optimal_idx])
            optimal_thresholds['balanced_fpr'] = float(fpr[j_optimal_idx])
            optimal_thresholds['balanced_j_score'] = float(j_scores[j_optimal_idx])
        
        # Business-focused thresholds based on cost considerations
        # High precision threshold (conservative, minimize customer friction)
        high_precision_mask = precisions[:-1] >= 0.5  # At least 50% precision
        if np.any(high_precision_mask):
            valid_recalls = recalls[:-1][high_precision_mask]
            valid_thresholds = pr_thresholds[high_precision_mask]
            if len(valid_recalls) > 0:
                business_conservative_idx = np.argmax(valid_recalls)
                optimal_thresholds['business_conservative'] = float(valid_thresholds[business_conservative_idx])
        
        # High recall threshold (aggressive, catch more fraud)
        high_recall_mask = recalls[:-1] >= 0.9  # At least 90% recall
        if np.any(high_recall_mask):
            valid_precisions = precisions[:-1][high_recall_mask]
            valid_thresholds = pr_thresholds[high_recall_mask]
            if len(valid_precisions) > 0:
                business_aggressive_idx = np.argmax(valid_precisions)
                optimal_thresholds['business_aggressive'] = float(valid_thresholds[business_aggressive_idx])
        
        return optimal_thresholds
    
    def _analyze_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray,
                           precisions: np.ndarray, recalls: np.ndarray, 
                           thresholds: np.ndarray,
                           sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of threshold performance.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            precisions: Precision values from PR curve
            recalls: Recall values from PR curve
            thresholds: Threshold values from PR curve
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary containing detailed threshold analysis
        """
        analysis = {
            'threshold_range': {
                'min': float(np.min(thresholds)),
                'max': float(np.max(thresholds)),
                'mean': float(np.mean(thresholds)),
                'std': float(np.std(thresholds))
            },
            'performance_at_thresholds': []
        }
        
        # Analyze performance at key threshold points
        key_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for threshold in key_thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
            recall = recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
            f1 = f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            else:
                tn = fp = fn = tp = 0
                specificity = fpr = 0.0
            
            threshold_performance = {
                'threshold': threshold,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'false_positive_rate': float(fpr),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'predicted_fraud_rate': float(np.mean(y_pred)),
                'fraud_detection_rate': float(tp / np.sum(y_true)) if np.sum(y_true) > 0 else 0.0
            }
            
            analysis['performance_at_thresholds'].append(threshold_performance)
        
        # Find threshold ranges for different performance criteria
        analysis['threshold_recommendations'] = {
            'high_precision_range': self._find_threshold_range(
                precisions[:-1], thresholds, min_value=0.8, target='precision'
            ),
            'high_recall_range': self._find_threshold_range(
                recalls[:-1], thresholds, min_value=0.8, target='recall'
            ),
            'balanced_range': self._find_balanced_threshold_range(
                precisions[:-1], recalls[:-1], thresholds
            )
        }
        
        return analysis
    
    def _find_threshold_range(self, metric_values: np.ndarray, thresholds: np.ndarray,
                             min_value: float, target: str) -> Dict[str, float]:
        """
        Find threshold range that achieves minimum performance on a metric.
        
        Args:
            metric_values: Performance metric values
            thresholds: Corresponding threshold values
            min_value: Minimum acceptable value for the metric
            target: Name of the target metric
            
        Returns:
            Dictionary with threshold range information
        """
        valid_mask = metric_values >= min_value
        
        if not np.any(valid_mask):
            return {
                'available': False,
                'reason': f'No thresholds achieve {target} >= {min_value}'
            }
        
        valid_thresholds = thresholds[valid_mask]
        valid_metrics = metric_values[valid_mask]
        
        return {
            'available': True,
            'min_threshold': float(np.min(valid_thresholds)),
            'max_threshold': float(np.max(valid_thresholds)),
            'best_threshold': float(valid_thresholds[np.argmax(valid_metrics)]),
            'best_metric_value': float(np.max(valid_metrics)),
            'n_valid_thresholds': int(np.sum(valid_mask))
        }
    
    def _find_balanced_threshold_range(self, precisions: np.ndarray, recalls: np.ndarray,
                                      thresholds: np.ndarray) -> Dict[str, float]:
        """
        Find threshold range that balances precision and recall.
        
        Args:
            precisions: Precision values
            recalls: Recall values
            thresholds: Threshold values
            
        Returns:
            Dictionary with balanced threshold range information
        """
        # Find thresholds where precision and recall are reasonably balanced
        precision_recall_diff = np.abs(precisions - recalls)
        balanced_mask = precision_recall_diff <= 0.1  # Within 10% of each other
        
        if not np.any(balanced_mask):
            return {
                'available': False,
                'reason': 'No thresholds achieve balanced precision and recall'
            }
        
        valid_thresholds = thresholds[balanced_mask]
        valid_f1_scores = 2 * (precisions[balanced_mask] * recalls[balanced_mask]) / (
            precisions[balanced_mask] + recalls[balanced_mask] + 1e-8
        )
        
        best_idx = np.argmax(valid_f1_scores)
        
        return {
            'available': True,
            'min_threshold': float(np.min(valid_thresholds)),
            'max_threshold': float(np.max(valid_thresholds)),
            'best_threshold': float(valid_thresholds[best_idx]),
            'best_f1_score': float(valid_f1_scores[best_idx]),
            'best_precision': float(precisions[balanced_mask][best_idx]),
            'best_recall': float(recalls[balanced_mask][best_idx]),
            'n_valid_thresholds': int(np.sum(balanced_mask))
        }
    
    def _calculate_business_impact(self, y_true: np.ndarray, y_proba: np.ndarray,
                                  optimal_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate business impact analysis for different threshold strategies.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            optimal_thresholds: Dictionary of optimal thresholds
            
        Returns:
            Dictionary containing business impact analysis
        """
        business_impact = {
            'baseline_metrics': {
                'total_transactions': len(y_true),
                'fraud_transactions': int(np.sum(y_true)),
                'legitimate_transactions': int(np.sum(y_true == 0)),
                'baseline_fraud_rate': float(np.mean(y_true))
            },
            'threshold_impacts': {}
        }
        
        # Analyze impact for each optimal threshold
        for threshold_name, threshold_value in optimal_thresholds.items():
            if isinstance(threshold_value, (int, float)):
                y_pred = (y_proba >= threshold_value).astype(int)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
                
                # Calculate business metrics
                impact = {
                    'threshold': float(threshold_value),
                    'confusion_matrix': {
                        'true_negatives': int(tn),
                        'false_positives': int(fp),
                        'false_negatives': int(fn),
                        'true_positives': int(tp)
                    },
                    'detection_metrics': {
                        'fraud_detected': int(tp),
                        'fraud_missed': int(fn),
                        'fraud_detection_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                        'false_alarm_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                    },
                    'customer_impact': {
                        'customers_flagged': int(tp + fp),
                        'legitimate_customers_flagged': int(fp),
                        'customer_friction_rate': float((tp + fp) / len(y_true)),
                        'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                    }
                }
                
                # Add cost-benefit analysis (placeholder values - should be configured based on business)
                fraud_cost_per_transaction = 100  # Average fraud loss
                false_positive_cost = 5  # Cost of investigating false positive
                
                impact['cost_analysis'] = {
                    'fraud_prevented_value': float(tp * fraud_cost_per_transaction),
                    'fraud_loss_value': float(fn * fraud_cost_per_transaction),
                    'false_positive_cost': float(fp * false_positive_cost),
                    'net_benefit': float(tp * fraud_cost_per_transaction - fp * false_positive_cost),
                    'cost_per_fraud_detected': float(fp * false_positive_cost / tp) if tp > 0 else float('inf')
                }
                
                business_impact['threshold_impacts'][threshold_name] = impact
        
        return business_impact
    
    def assign_risk_levels(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Assign risk levels (low/medium/high) to transactions based on fraud probabilities.
        
        This method implements the 3-level risk strategy using configured thresholds:
        - Low risk: probability < low_risk_threshold
        - Medium risk: low_risk_threshold <= probability < high_risk_threshold  
        - High risk: probability >= high_risk_threshold
        
        Args:
            probabilities: Array of fraud probability scores in [0, 1] range
            
        Returns:
            Array of risk level strings ('low', 'medium', 'high')
            
        Raises:
            ValueError: If probabilities are invalid or thresholds not configured
        """
        if probabilities is None:
            raise ValueError("probabilities cannot be None")
        
        if len(probabilities) == 0:
            return np.array([], dtype=str)
        
        # Validate probability range
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("probabilities must be in [0, 1] range")
        
        if np.any(np.isnan(probabilities)):
            raise ValueError("probabilities cannot contain NaN values")
        
        if np.any(np.isinf(probabilities)):
            raise ValueError("probabilities cannot contain infinite values")
        
        logger.debug(f"Assigning risk levels to {len(probabilities)} transactions - "
                    f"Low threshold: {self.low_risk_threshold}, "
                    f"High threshold: {self.high_risk_threshold}")
        
        # Initialize risk levels array
        risk_levels = np.full(len(probabilities), self.risk_levels[1], dtype=object)  # Default to medium
        
        # Assign risk levels based on thresholds - Requirements 6.2
        low_mask = probabilities < self.low_risk_threshold
        high_mask = probabilities >= self.high_risk_threshold
        
        risk_levels[low_mask] = self.risk_levels[0]    # 'low'
        risk_levels[high_mask] = self.risk_levels[2]   # 'high'
        # Medium risk is already assigned as default
        
        # Validate risk level assignment consistency
        assert np.all(np.isin(risk_levels, self.risk_levels)), "Invalid risk levels assigned"
        
        # Log distribution of risk levels
        unique, counts = np.unique(risk_levels, return_counts=True)
        risk_distribution = dict(zip(unique, counts))
        
        logger.debug(f"Risk level distribution: {risk_distribution}")
        
        return risk_levels
    
    def get_risk_summary(self, probabilities: np.ndarray, 
                        risk_levels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate summary statistics for risk assessment.
        
        Args:
            probabilities: Array of fraud probability scores
            risk_levels: Optional array of risk levels (will be computed if not provided)
            
        Returns:
            Dictionary containing risk assessment summary
        """
        if risk_levels is None:
            risk_levels = self.assign_risk_levels(probabilities)
        
        # Calculate basic statistics
        summary = {
            'total_transactions': len(probabilities),
            'probability_stats': {
                'min': float(np.min(probabilities)),
                'max': float(np.max(probabilities)),
                'mean': float(np.mean(probabilities)),
                'median': float(np.median(probabilities)),
                'std': float(np.std(probabilities)),
                'percentiles': {
                    '25th': float(np.percentile(probabilities, 25)),
                    '75th': float(np.percentile(probabilities, 75)),
                    '90th': float(np.percentile(probabilities, 90)),
                    '95th': float(np.percentile(probabilities, 95)),
                    '99th': float(np.percentile(probabilities, 99))
                }
            },
            'risk_distribution': {},
            'threshold_config': {
                'low_risk_threshold': self.low_risk_threshold,
                'high_risk_threshold': self.high_risk_threshold,
                'risk_levels': self.risk_levels
            }
        }
        
        # Calculate risk level distribution
        unique_levels, counts = np.unique(risk_levels, return_counts=True)
        for level, count in zip(unique_levels, counts):
            summary['risk_distribution'][level] = {
                'count': int(count),
                'percentage': float(count / len(risk_levels) * 100)
            }
        
        # Calculate probability statistics by risk level
        summary['risk_level_stats'] = {}
        for level in self.risk_levels:
            level_mask = risk_levels == level
            if np.any(level_mask):
                level_probs = probabilities[level_mask]
                summary['risk_level_stats'][level] = {
                    'count': int(np.sum(level_mask)),
                    'min_probability': float(np.min(level_probs)),
                    'max_probability': float(np.max(level_probs)),
                    'mean_probability': float(np.mean(level_probs)),
                    'median_probability': float(np.median(level_probs))
                }
        
        return summary
    
    def generate_recommendations(self, include_business_context: bool = True) -> Dict[str, Any]:
        """
        Generate recommendations for optimal operating thresholds with business impact analysis.
        
        This method provides comprehensive recommendations for threshold selection
        based on the optimization results, including business context and
        implementation guidance for different operational scenarios.
        
        Args:
            include_business_context: Whether to include detailed business impact analysis
            
        Returns:
            Dictionary containing threshold recommendations and implementation guidance
            
        Raises:
            ValueError: If threshold tuning has not been performed
        """
        if not self.is_fitted:
            raise ValueError("Must call tune_thresholds() before generating recommendations")
        
        if not self.optimal_thresholds:
            raise ValueError("No optimal thresholds available - threshold tuning may have failed")
        
        logger.info("Generating threshold recommendations with business impact analysis")
        
        recommendations = {
            'executive_summary': self._generate_executive_summary(),
            'threshold_recommendations': self._generate_threshold_recommendations(),
            'implementation_guidance': self._generate_implementation_guidance(),
            'risk_strategy_options': self._generate_risk_strategy_options(),
            'monitoring_recommendations': self._generate_monitoring_recommendations()
        }
        
        if include_business_context:
            recommendations['business_impact_analysis'] = self._generate_business_impact_recommendations()
            recommendations['cost_benefit_analysis'] = self._generate_cost_benefit_recommendations()
        
        recommendations['metadata'] = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'optimization_metric': self.optimization_metric,
            'min_precision': self.min_precision,
            'min_recall': self.min_recall,
            'current_thresholds': {
                'low_risk': self.low_risk_threshold,
                'high_risk': self.high_risk_threshold
            }
        }
        
        logger.info("Threshold recommendations generated successfully")
        return recommendations
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of threshold optimization results."""
        # Find the best performing threshold based on optimization metric
        best_threshold_key = f'{self.optimization_metric}_optimal'
        if best_threshold_key not in self.optimal_thresholds:
            best_threshold_key = 'f1_optimal'  # Fallback to F1
        
        best_threshold = self.optimal_thresholds.get(best_threshold_key, 0.5)
        
        # Get performance metrics for the best threshold
        best_performance = {}
        for key, value in self.optimal_thresholds.items():
            if key.startswith(self.optimization_metric) and isinstance(value, (int, float)):
                metric_name = key.replace(f'{self.optimization_metric}_', '')
                best_performance[metric_name] = value
        
        # Calculate expected impact
        baseline_fraud_rate = self.business_impact.get('baseline_metrics', {}).get('baseline_fraud_rate', 0.04)
        
        summary = {
            'recommended_primary_threshold': float(best_threshold),
            'optimization_focus': self.optimization_metric,
            'expected_performance': best_performance,
            'key_insights': [
                f"Optimized for {self.optimization_metric} performance with {best_threshold:.3f} threshold",
                f"Expected to handle {baseline_fraud_rate:.1%} baseline fraud rate effectively",
                "Balances fraud detection with customer experience considerations"
            ],
            'implementation_priority': 'high' if best_performance.get('f1_score', 0) > 0.5 else 'medium',
            'confidence_level': self._calculate_confidence_level()
        }
        
        # Add specific insights based on business impact
        if self.business_impact and 'threshold_impacts' in self.business_impact:
            best_impact = self.business_impact['threshold_impacts'].get(best_threshold_key, {})
            if best_impact:
                detection_rate = best_impact.get('detection_metrics', {}).get('fraud_detection_rate', 0)
                false_alarm_rate = best_impact.get('detection_metrics', {}).get('false_alarm_rate', 0)
                
                summary['key_insights'].extend([
                    f"Expected fraud detection rate: {detection_rate:.1%}",
                    f"Expected false alarm rate: {false_alarm_rate:.1%}"
                ])
        
        return summary
    
    def _generate_threshold_recommendations(self) -> Dict[str, Any]:
        """Generate specific threshold recommendations for different scenarios."""
        recommendations = {
            'primary_recommendation': {},
            'alternative_strategies': {},
            'scenario_based_recommendations': {}
        }
        
        # Primary recommendation based on optimization metric
        primary_key = f'{self.optimization_metric}_optimal'
        if primary_key in self.optimal_thresholds:
            recommendations['primary_recommendation'] = {
                'threshold': self.optimal_thresholds[primary_key],
                'strategy': self.optimization_metric,
                'rationale': f"Optimized for {self.optimization_metric} performance",
                'expected_metrics': {
                    key.replace(f'{self.optimization_metric}_', ''): value
                    for key, value in self.optimal_thresholds.items()
                    if key.startswith(self.optimization_metric) and isinstance(value, (int, float))
                }
            }
        
        # Alternative strategies
        strategy_mapping = {
            'precision_focused': {
                'name': 'Conservative Strategy',
                'description': 'Minimizes false positives, prioritizes customer experience',
                'use_case': 'High-value customers, premium services'
            },
            'recall_focused': {
                'name': 'Aggressive Strategy', 
                'description': 'Maximizes fraud detection, accepts higher false positive rate',
                'use_case': 'High-risk transactions, new customer onboarding'
            },
            'balanced': {
                'name': 'Balanced Strategy',
                'description': 'Balances fraud detection with customer experience',
                'use_case': 'General transaction processing'
            },
            'business_conservative': {
                'name': 'Business Conservative',
                'description': 'High precision focus for business operations',
                'use_case': 'Automated processing with minimal manual review'
            },
            'business_aggressive': {
                'name': 'Business Aggressive',
                'description': 'High recall focus for maximum fraud prevention',
                'use_case': 'High-risk periods, fraud pattern emergence'
            }
        }
        
        for strategy_key, strategy_info in strategy_mapping.items():
            if strategy_key in self.optimal_thresholds:
                threshold_value = self.optimal_thresholds[strategy_key]
                if isinstance(threshold_value, (int, float)):
                    recommendations['alternative_strategies'][strategy_key] = {
                        'threshold': float(threshold_value),
                        'name': strategy_info['name'],
                        'description': strategy_info['description'],
                        'recommended_use_case': strategy_info['use_case'],
                        'performance_metrics': {
                            key.replace(f'{strategy_key}_', ''): value
                            for key, value in self.optimal_thresholds.items()
                            if key.startswith(strategy_key) and isinstance(value, (int, float))
                        }
                    }
        
        # Scenario-based recommendations
        recommendations['scenario_based_recommendations'] = {
            'high_volume_periods': {
                'recommended_strategy': 'business_conservative',
                'rationale': 'Minimize manual review overhead during peak processing',
                'threshold_adjustment': 'Use higher threshold to reduce false positives'
            },
            'fraud_alert_periods': {
                'recommended_strategy': 'business_aggressive',
                'rationale': 'Increase detection sensitivity during known fraud campaigns',
                'threshold_adjustment': 'Use lower threshold to catch more potential fraud'
            },
            'new_customer_onboarding': {
                'recommended_strategy': 'recall_focused',
                'rationale': 'Higher scrutiny for unknown customer patterns',
                'threshold_adjustment': 'Apply stricter thresholds for new accounts'
            },
            'vip_customer_processing': {
                'recommended_strategy': 'precision_focused',
                'rationale': 'Minimize friction for high-value customers',
                'threshold_adjustment': 'Use higher thresholds to reduce false positives'
            }
        }
        
        return recommendations
    
    def _generate_implementation_guidance(self) -> Dict[str, Any]:
        """Generate practical implementation guidance for threshold deployment."""
        guidance = {
            'deployment_strategy': {
                'recommended_approach': 'gradual_rollout',
                'phases': [
                    {
                        'phase': 1,
                        'description': 'Shadow mode testing',
                        'duration': '1-2 weeks',
                        'actions': [
                            'Deploy new thresholds in shadow mode',
                            'Compare predictions with current system',
                            'Monitor performance metrics',
                            'Validate business impact estimates'
                        ]
                    },
                    {
                        'phase': 2,
                        'description': 'Limited rollout',
                        'duration': '2-4 weeks',
                        'actions': [
                            'Apply new thresholds to 10-20% of traffic',
                            'Monitor false positive and detection rates',
                            'Collect feedback from fraud analysts',
                            'Adjust thresholds based on real-world performance'
                        ]
                    },
                    {
                        'phase': 3,
                        'description': 'Full deployment',
                        'duration': 'Ongoing',
                        'actions': [
                            'Roll out to all traffic',
                            'Implement continuous monitoring',
                            'Set up automated alerting',
                            'Schedule regular threshold reviews'
                        ]
                    }
                ]
            },
            'technical_requirements': {
                'monitoring_setup': [
                    'Real-time fraud detection rate tracking',
                    'False positive rate monitoring',
                    'Customer friction metrics',
                    'Model performance degradation alerts'
                ],
                'infrastructure_considerations': [
                    'Ensure threshold updates can be deployed without downtime',
                    'Implement A/B testing capability for threshold comparison',
                    'Set up data pipeline for continuous model evaluation',
                    'Configure automated rollback mechanisms'
                ],
                'integration_points': [
                    'Update risk scoring service with new thresholds',
                    'Modify fraud analyst dashboard for new risk levels',
                    'Integrate with customer notification systems',
                    'Update reporting and analytics pipelines'
                ]
            },
            'validation_checklist': [
                'Verify threshold values are correctly configured',
                'Test risk level assignment logic',
                'Validate integration with downstream systems',
                'Confirm monitoring and alerting setup',
                'Review business impact calculations',
                'Test rollback procedures'
            ]
        }
        
        return guidance
    
    def _generate_risk_strategy_options(self) -> Dict[str, Any]:
        """Generate different risk strategy options with trade-off analysis."""
        strategies = {
            'current_strategy': {
                'low_threshold': self.low_risk_threshold,
                'high_threshold': self.high_risk_threshold,
                'description': 'Current 3-level risk strategy configuration'
            },
            'recommended_adjustments': {},
            'alternative_strategies': {}
        }
        
        # Analyze current threshold performance
        if self.threshold_analysis and 'performance_at_thresholds' in self.threshold_analysis:
            current_low_perf = None
            current_high_perf = None
            
            for perf in self.threshold_analysis['performance_at_thresholds']:
                if abs(perf['threshold'] - self.low_risk_threshold) < 0.05:
                    current_low_perf = perf
                if abs(perf['threshold'] - self.high_risk_threshold) < 0.05:
                    current_high_perf = perf
            
            # Generate recommendations based on optimal thresholds
            if 'f1_optimal' in self.optimal_thresholds:
                optimal_threshold = self.optimal_thresholds['f1_optimal']
                
                # Suggest new risk level boundaries
                new_low = max(0.1, optimal_threshold - 0.2)
                new_high = min(0.9, optimal_threshold + 0.2)
                
                strategies['recommended_adjustments'] = {
                    'new_low_threshold': float(new_low),
                    'new_high_threshold': float(new_high),
                    'rationale': f'Centered around F1-optimal threshold of {optimal_threshold:.3f}',
                    'expected_impact': {
                        'more_balanced_risk_distribution': True,
                        'improved_fraud_detection': True,
                        'optimized_resource_allocation': True
                    }
                }
        
        # Alternative strategy configurations
        strategies['alternative_strategies'] = {
            'conservative': {
                'low_threshold': 0.4,
                'high_threshold': 0.8,
                'description': 'Conservative approach with higher thresholds',
                'trade_offs': {
                    'pros': ['Lower false positive rate', 'Better customer experience'],
                    'cons': ['May miss some fraud', 'Higher risk tolerance required']
                }
            },
            'aggressive': {
                'low_threshold': 0.2,
                'high_threshold': 0.6,
                'description': 'Aggressive approach with lower thresholds',
                'trade_offs': {
                    'pros': ['Higher fraud detection rate', 'Better fraud prevention'],
                    'cons': ['Higher false positive rate', 'More manual reviews needed']
                }
            },
            'dynamic': {
                'description': 'Dynamic thresholds based on transaction context',
                'implementation': 'Adjust thresholds based on customer risk profile, transaction amount, time of day',
                'trade_offs': {
                    'pros': ['Personalized risk assessment', 'Optimized for different scenarios'],
                    'cons': ['More complex implementation', 'Requires additional data and logic']
                }
            }
        }
        
        return strategies
    
    def _generate_monitoring_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for ongoing monitoring and maintenance."""
        return {
            'key_metrics_to_monitor': {
                'performance_metrics': [
                    'Fraud detection rate (recall)',
                    'False positive rate',
                    'Precision at different thresholds',
                    'F1 score trends',
                    'PR-AUC and ROC-AUC'
                ],
                'business_metrics': [
                    'Customer friction rate',
                    'Manual review volume',
                    'Fraud losses prevented',
                    'Investigation costs',
                    'Customer satisfaction scores'
                ],
                'operational_metrics': [
                    'Model prediction latency',
                    'System availability',
                    'Data quality indicators',
                    'Model drift indicators'
                ]
            },
            'alerting_thresholds': {
                'critical_alerts': [
                    'Fraud detection rate drops below 70%',
                    'False positive rate exceeds 10%',
                    'Model prediction errors increase by 50%',
                    'System latency exceeds 500ms'
                ],
                'warning_alerts': [
                    'F1 score decreases by more than 5%',
                    'Customer friction rate increases by 20%',
                    'Manual review volume increases by 30%',
                    'Data quality score drops below 95%'
                ]
            },
            'review_schedule': {
                'daily': [
                    'Monitor key performance metrics',
                    'Review high-risk transaction flags',
                    'Check system health indicators'
                ],
                'weekly': [
                    'Analyze fraud detection trends',
                    'Review false positive cases',
                    'Assess customer impact metrics',
                    'Update fraud pattern analysis'
                ],
                'monthly': [
                    'Comprehensive model performance review',
                    'Threshold optimization analysis',
                    'Business impact assessment',
                    'Strategy adjustment recommendations'
                ],
                'quarterly': [
                    'Full model retraining evaluation',
                    'Threshold strategy review',
                    'Business requirements reassessment',
                    'Technology stack evaluation'
                ]
            },
            'maintenance_actions': {
                'threshold_updates': 'Review and update thresholds monthly based on performance data',
                'model_retraining': 'Retrain models quarterly or when performance degrades significantly',
                'data_quality_checks': 'Implement automated data quality monitoring and validation',
                'business_alignment': 'Regular alignment meetings with business stakeholders'
            }
        }
    
    def _generate_business_impact_recommendations(self) -> Dict[str, Any]:
        """Generate business impact analysis and recommendations."""
        if not self.business_impact or 'threshold_impacts' not in self.business_impact:
            return {'status': 'no_business_impact_data_available'}
        
        baseline = self.business_impact.get('baseline_metrics', {})
        impacts = self.business_impact['threshold_impacts']
        
        recommendations = {
            'baseline_analysis': {
                'current_fraud_exposure': {
                    'total_transactions': baseline.get('total_transactions', 0),
                    'fraud_transactions': baseline.get('fraud_transactions', 0),
                    'fraud_rate': baseline.get('baseline_fraud_rate', 0),
                    'estimated_annual_fraud_loss': baseline.get('fraud_transactions', 0) * 100 * 12  # Placeholder calculation
                }
            },
            'impact_comparison': {},
            'roi_analysis': {},
            'resource_planning': {}
        }
        
        # Compare different threshold strategies
        for strategy_name, impact_data in impacts.items():
            if isinstance(impact_data, dict) and 'detection_metrics' in impact_data:
                detection = impact_data['detection_metrics']
                customer = impact_data.get('customer_impact', {})
                cost = impact_data.get('cost_analysis', {})
                
                recommendations['impact_comparison'][strategy_name] = {
                    'fraud_prevention': {
                        'fraud_detected': detection.get('fraud_detected', 0),
                        'fraud_missed': detection.get('fraud_missed', 0),
                        'detection_improvement': detection.get('fraud_detection_rate', 0) - baseline.get('baseline_fraud_rate', 0)
                    },
                    'customer_experience': {
                        'customers_affected': customer.get('customers_flagged', 0),
                        'false_positives': customer.get('legitimate_customers_flagged', 0),
                        'friction_rate': customer.get('customer_friction_rate', 0)
                    },
                    'financial_impact': {
                        'fraud_prevented_value': cost.get('fraud_prevented_value', 0),
                        'investigation_costs': cost.get('false_positive_cost', 0),
                        'net_benefit': cost.get('net_benefit', 0)
                    }
                }
        
        # ROI analysis
        best_strategy = max(impacts.keys(), 
                           key=lambda k: impacts[k].get('cost_analysis', {}).get('net_benefit', 0)
                           if isinstance(impacts[k], dict) else 0)
        
        if best_strategy in impacts:
            best_impact = impacts[best_strategy]
            cost_analysis = best_impact.get('cost_analysis', {})
            
            recommendations['roi_analysis'] = {
                'recommended_strategy': best_strategy,
                'expected_annual_savings': cost_analysis.get('net_benefit', 0) * 12,  # Annualized
                'payback_period': 'immediate',  # Fraud prevention typically has immediate ROI
                'risk_adjusted_return': cost_analysis.get('net_benefit', 0) / max(cost_analysis.get('false_positive_cost', 1), 1)
            }
        
        # Resource planning recommendations
        total_flagged = sum(
            impact.get('customer_impact', {}).get('customers_flagged', 0)
            for impact in impacts.values()
            if isinstance(impact, dict)
        )
        
        recommendations['resource_planning'] = {
            'fraud_analyst_capacity': {
                'estimated_daily_reviews': total_flagged / len(impacts) if impacts else 0,
                'recommended_team_size': max(1, int((total_flagged / len(impacts)) / 50)) if impacts else 1,  # Assuming 50 reviews per analyst per day
                'peak_capacity_buffer': '20-30% above average for fraud campaign periods'
            },
            'technology_requirements': {
                'automated_processing_rate': '70-80% of low-risk transactions',
                'manual_review_queue_capacity': 'Handle 2x average volume for surge periods',
                'response_time_targets': 'High-risk: <1 hour, Medium-risk: <4 hours'
            }
        }
        
        return recommendations
    
    def _generate_cost_benefit_recommendations(self) -> Dict[str, Any]:
        """Generate detailed cost-benefit analysis recommendations."""
        return {
            'cost_model_assumptions': {
                'fraud_loss_per_transaction': 100,  # Should be configured based on business data
                'false_positive_investigation_cost': 5,
                'manual_review_cost': 10,
                'customer_churn_cost': 200,
                'note': 'These are placeholder values - should be calibrated with actual business data'
            },
            'optimization_opportunities': [
                'Implement dynamic thresholds based on transaction context',
                'Use ensemble methods to improve precision at high recall',
                'Develop customer-specific risk profiles',
                'Integrate real-time fraud intelligence feeds',
                'Implement automated case prioritization'
            ],
            'investment_priorities': [
                {
                    'priority': 1,
                    'investment': 'Enhanced feature engineering',
                    'expected_impact': 'Improve model precision by 10-15%',
                    'estimated_cost': 'Medium',
                    'timeline': '2-3 months'
                },
                {
                    'priority': 2,
                    'investment': 'Real-time model updates',
                    'expected_impact': 'Reduce model drift and maintain performance',
                    'estimated_cost': 'High',
                    'timeline': '4-6 months'
                },
                {
                    'priority': 3,
                    'investment': 'Advanced ensemble methods',
                    'expected_impact': 'Improve overall model performance by 5-10%',
                    'estimated_cost': 'Medium',
                    'timeline': '3-4 months'
                }
            ],
            'risk_mitigation': [
                'Implement gradual threshold rollout to minimize business disruption',
                'Maintain fallback to previous thresholds in case of performance issues',
                'Set up comprehensive monitoring to detect performance degradation early',
                'Establish clear escalation procedures for threshold-related issues'
            ]
        }
    
    def _calculate_confidence_level(self) -> str:
        """Calculate confidence level in recommendations based on data quality and performance."""
        # This is a simplified confidence calculation
        # In practice, this would consider factors like:
        # - Sample size adequacy
        # - Cross-validation stability
        # - Business metric alignment
        # - Historical performance consistency
        
        if not self.optimal_thresholds:
            return 'low'
        
        # Check if we have multiple optimization strategies
        strategy_count = len([k for k, v in self.optimal_thresholds.items() 
                             if isinstance(v, (int, float)) and k.endswith('_optimal')])
        
        # Check if we have business impact data
        has_business_data = bool(self.business_impact and 'threshold_impacts' in self.business_impact)
        
        # Simple confidence scoring
        confidence_score = 0
        confidence_score += min(strategy_count * 20, 60)  # Up to 60 points for multiple strategies
        confidence_score += 30 if has_business_data else 0  # 30 points for business impact data
        confidence_score += 10  # Base 10 points for having any optimization results
        
        if confidence_score >= 80:
            return 'high'
        elif confidence_score >= 60:
            return 'medium'
        else:
            return 'low'
    
    def save_recommendations(self, filepath: str, recommendations: Optional[Dict[str, Any]] = None) -> None:
        """
        Save threshold recommendations to a file.
        
        Args:
            filepath: Path to save the recommendations
            recommendations: Optional recommendations dict (will generate if not provided)
        """
        if recommendations is None:
            recommendations = self.generate_recommendations()
        
        import json
        with open(filepath, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        logger.info(f"Recommendations saved to {filepath}")
    
    def load_recommendations(self, filepath: str) -> Dict[str, Any]:
        """
        Load threshold recommendations from a file.
        
        Args:
            filepath: Path to load the recommendations from
            
        Returns:
            Dictionary containing loaded recommendations
        """
        import json
        with open(filepath, 'r') as f:
            recommendations = json.load(f)
        
        logger.info(f"Recommendations loaded from {filepath}")
        return recommendations