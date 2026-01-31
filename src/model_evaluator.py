"""
Model evaluation module for the fraud detection system.
Implements comprehensive evaluation metrics and performance analysis for fraud detection models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report,
    average_precision_score, roc_auc_score
)

import sys
import os
sys.path.append(os.path.dirname(__file__))
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation class for fraud detection systems.
    
    This class provides extensive evaluation capabilities including:
    - PR-AUC, precision, recall, F1-score calculations
    - Confusion matrix generation and visualization
    - Threshold analysis across different operating points
    - Performance reports with business impact metrics
    - Visualization of model performance
    
    The evaluator is designed specifically for fraud detection with imbalanced datasets,
    focusing on metrics that are most relevant for fraud prevention while minimizing
    customer friction.
    """
    
    def __init__(self,
                 figsize: Tuple[int, int] = (12, 8),
                 style: str = 'whitegrid',
                 save_plots: bool = False,
                 plot_dir: str = 'plots'):
        """
        Initialize the ModelEvaluator with visualization settings.
        
        Args:
            figsize: Default figure size for plots
            style: Seaborn style for plots
            save_plots: Whether to save plots to disk
            plot_dir: Directory to save plots
        """
        self.figsize = figsize
        self.style = style
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        # Set up plotting style
        sns.set_style(self.style)
        plt.rcParams['figure.figsize'] = self.figsize
        
        # Create plot directory if saving plots
        if self.save_plots:
            os.makedirs(self.plot_dir, exist_ok=True)
        
        logger.info(f"ModelEvaluator initialized - figsize: {figsize}, style: {style}, save_plots: {save_plots}")
    
    def calculate_pr_auc(self, y_true: np.ndarray, y_proba: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Calculate Precision-Recall Area Under Curve (PR-AUC).
        
        PR-AUC is particularly important for imbalanced datasets like fraud detection
        as it focuses on the performance on the minority class (fraud).
        
        Args:
            y_true: True binary labels (0 for legitimate, 1 for fraud)
            y_proba: Predicted fraud probabilities in [0, 1] range
            sample_weight: Optional sample weights
            
        Returns:
            PR-AUC score
            
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
        
        # Calculate PR-AUC - Requirements 5.1
        pr_auc = average_precision_score(y_true, y_proba, sample_weight=sample_weight)
        
        logger.debug(f"Calculated PR-AUC: {pr_auc:.4f}")
        return float(pr_auc)
    
    def calculate_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score metrics.
        
        These metrics are essential for fraud detection evaluation:
        - Precision: Fraction of predicted fraud that is actually fraud (minimize false positives)
        - Recall: Fraction of actual fraud that is detected (maximize fraud detection)
        - F1-score: Harmonic mean of precision and recall (balanced metric)
        
        Args:
            y_true: True binary labels (0 for legitimate, 1 for fraud)
            y_pred: Predicted binary labels (0 for legitimate, 1 for fraud)
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary containing precision, recall, and F1-score
            
        Raises:
            ValueError: If input data is invalid
        """
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred cannot be None")
        
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
        
        # Validate binary labels
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        
        if not all(label in [0, 1] for label in unique_true):
            raise ValueError("y_true must contain only 0 and 1 values")
        
        if not all(label in [0, 1] for label in unique_pred):
            raise ValueError("y_pred must contain only 0 and 1 values")
        
        # Calculate metrics - Requirements 5.2
        precision = precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        recall = recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        f1 = f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight)
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        logger.debug(f"Calculated metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return metrics
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 sample_weight: Optional[np.ndarray] = None,
                                 normalize: Optional[str] = None) -> np.ndarray:
        """
        Generate confusion matrix for binary classification.
        
        The confusion matrix provides detailed breakdown of prediction performance:
        - True Negatives (TN): Correctly identified legitimate transactions
        - False Positives (FP): Legitimate transactions incorrectly flagged as fraud
        - False Negatives (FN): Fraud transactions missed by the model
        - True Positives (TP): Correctly identified fraud transactions
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            sample_weight: Optional sample weights
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            
        Returns:
            Confusion matrix as 2x2 numpy array
            
        Raises:
            ValueError: If input data is invalid
        """
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred cannot be None")
        
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} vs {len(y_pred)}")
        
        # Generate confusion matrix - Requirements 5.3
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, normalize=normalize)
        
        logger.debug(f"Generated confusion matrix with shape: {cm.shape}")
        return cm
    
    def visualize_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  sample_weight: Optional[np.ndarray] = None,
                                  normalize: Optional[str] = None,
                                  title: str = "Confusion Matrix",
                                  class_names: List[str] = None) -> plt.Figure:
        """
        Create a visualization of the confusion matrix.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            sample_weight: Optional sample weights
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            title: Title for the plot
            class_names: Names for the classes (default: ['Legitimate', 'Fraud'])
            
        Returns:
            Matplotlib figure object
        """
        if class_names is None:
            class_names = ['Legitimate', 'Fraud']
        
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_true, y_pred, sample_weight, normalize)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Add performance metrics as text
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel() if not normalize else (cm * len(y_true)).astype(int).ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add text box with metrics
            metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots:
            filename = f"{title.lower().replace(' ', '_')}.png"
            filepath = os.path.join(self.plot_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix plot to {filepath}")
        
        return fig
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_proba: np.ndarray,
                                      threshold: float = 0.5,
                                      sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics for fraud detection.
        
        This method provides a complete evaluation including:
        - Basic classification metrics (precision, recall, F1)
        - Area under curve metrics (PR-AUC, ROC-AUC)
        - Confusion matrix analysis
        - Business-relevant metrics (false positive rate, detection rate)
        
        Args:
            y_true: True binary labels
            y_proba: Predicted fraud probabilities
            threshold: Decision threshold for binary classification
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary containing comprehensive metrics
        """
        if y_true is None or y_proba is None:
            raise ValueError("y_true and y_proba cannot be None")
        
        # Convert probabilities to binary predictions
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate basic metrics
        basic_metrics = self.calculate_precision_recall_f1(y_true, y_pred, sample_weight)
        
        # Calculate AUC metrics
        pr_auc = self.calculate_pr_auc(y_true, y_proba, sample_weight)
        roc_auc = roc_auc_score(y_true, y_proba, sample_weight=sample_weight)
        
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_true, y_pred, sample_weight)
        
        # Extract confusion matrix components
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        # Calculate additional business metrics
        total_transactions = len(y_true)
        total_fraud = np.sum(y_true)
        total_legitimate = total_transactions - total_fraud
        
        # Business metrics
        false_positive_rate = fp / total_legitimate if total_legitimate > 0 else 0.0
        true_positive_rate = tp / total_fraud if total_fraud > 0 else 0.0
        false_negative_rate = fn / total_fraud if total_fraud > 0 else 0.0
        specificity = tn / total_legitimate if total_legitimate > 0 else 0.0
        
        # Fraud detection metrics
        fraud_detection_rate = true_positive_rate
        fraud_miss_rate = false_negative_rate
        customer_friction_rate = (tp + fp) / total_transactions
        
        # Compile comprehensive results
        comprehensive_metrics = {
            # Basic classification metrics
            **basic_metrics,
            
            # AUC metrics
            'pr_auc': float(pr_auc),
            'roc_auc': float(roc_auc),
            
            # Confusion matrix
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            
            # Rate metrics
            'false_positive_rate': float(false_positive_rate),
            'true_positive_rate': float(true_positive_rate),
            'false_negative_rate': float(false_negative_rate),
            'specificity': float(specificity),
            
            # Business metrics
            'fraud_detection_rate': float(fraud_detection_rate),
            'fraud_miss_rate': float(fraud_miss_rate),
            'customer_friction_rate': float(customer_friction_rate),
            
            # Dataset statistics
            'dataset_stats': {
                'total_transactions': int(total_transactions),
                'total_fraud': int(total_fraud),
                'total_legitimate': int(total_legitimate),
                'fraud_rate': float(total_fraud / total_transactions) if total_transactions > 0 else 0.0,
                'threshold_used': float(threshold)
            },
            
            # Evaluation metadata
            'evaluation_metadata': {
                'evaluated_at': datetime.now().isoformat(),
                'sample_weighted': sample_weight is not None,
                'n_samples': int(total_transactions)
            }
        }
        
        logger.info(f"Comprehensive metrics calculated - PR-AUC: {pr_auc:.3f}, "
                   f"ROC-AUC: {roc_auc:.3f}, F1: {basic_metrics['f1_score']:.3f}")
        
        return comprehensive_metrics
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   sample_weight: Optional[np.ndarray] = None,
                                   title: str = "Precision-Recall Curve") -> plt.Figure:
        """
        Plot the Precision-Recall curve.
        
        The PR curve is particularly important for imbalanced datasets as it focuses
        on the performance on the minority class (fraud). It shows the trade-off
        between precision and recall for different threshold values.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted fraud probabilities
            sample_weight: Optional sample weights
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba, sample_weight=sample_weight)
        pr_auc = self.calculate_pr_auc(y_true, y_proba, sample_weight)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PR curve
        ax.plot(recalls, precisions, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        
        # Add baseline (random classifier)
        fraud_rate = np.mean(y_true)
        ax.axhline(y=fraud_rate, color='red', linestyle='--', alpha=0.7, 
                  label=f'Random Classifier (AUC = {fraud_rate:.3f})')
        
        # Formatting
        ax.set_xlabel('Recall (True Positive Rate)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add text with key metrics
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        best_precision = precisions[best_f1_idx]
        best_recall = recalls[best_f1_idx]
        
        # Mark best F1 point
        ax.plot(best_recall, best_precision, 'ro', markersize=8, 
               label=f'Best F1 = {best_f1:.3f} (threshold = {best_threshold:.3f})')
        
        # Update legend
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots:
            filename = f"{title.lower().replace(' ', '_').replace('-', '_')}.png"
            filepath = os.path.join(self.plot_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve plot to {filepath}")
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      sample_weight: Optional[np.ndarray] = None,
                      title: str = "ROC Curve") -> plt.Figure:
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        
        The ROC curve shows the trade-off between true positive rate (recall)
        and false positive rate across different threshold values.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted fraud probabilities
            sample_weight: Optional sample weights
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba, sample_weight=sample_weight)
        roc_auc = roc_auc_score(y_true, y_proba, sample_weight=sample_weight)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        
        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random Classifier (AUC = 0.5)')
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Find optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_threshold_idx]
        best_fpr = fpr[best_threshold_idx]
        best_tpr = tpr[best_threshold_idx]
        
        # Mark optimal point
        ax.plot(best_fpr, best_tpr, 'ro', markersize=8,
               label=f'Optimal Point (threshold = {best_threshold:.3f})')
        
        # Update legend
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots:
            filename = f"{title.lower().replace(' ', '_')}.png"
            filepath = os.path.join(self.plot_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve plot to {filepath}")
        
        return fig
    
    def create_evaluation_report(self, y_true: np.ndarray, y_proba: np.ndarray,
                               threshold: float = 0.5,
                               sample_weight: Optional[np.ndarray] = None,
                               model_name: str = "Fraud Detection Model",
                               include_plots: bool = True) -> Dict[str, Any]:
        """
        Create a comprehensive evaluation report with metrics and visualizations.
        
        This method generates a complete evaluation report including:
        - Comprehensive metrics calculation
        - Confusion matrix visualization
        - PR and ROC curve plots
        - Business impact analysis
        - Recommendations for threshold optimization
        
        Args:
            y_true: True binary labels
            y_proba: Predicted fraud probabilities
            threshold: Decision threshold for binary classification
            sample_weight: Optional sample weights
            model_name: Name of the model being evaluated
            include_plots: Whether to generate visualization plots
            
        Returns:
            Dictionary containing complete evaluation report
        """
        logger.info(f"Creating comprehensive evaluation report for {model_name}")
        
        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_proba, threshold, sample_weight)
        
        # Initialize report
        report = {
            'model_name': model_name,
            'evaluation_summary': {
                'total_samples': len(y_true),
                'fraud_samples': int(np.sum(y_true)),
                'fraud_rate': float(np.mean(y_true)),
                'threshold_used': threshold,
                'evaluation_date': datetime.now().isoformat()
            },
            'performance_metrics': metrics,
            'plots': {},
            'recommendations': {}
        }
        
        # Generate plots if requested
        if include_plots:
            try:
                # Confusion matrix
                cm_fig = self.visualize_confusion_matrix(
                    y_true, (y_proba >= threshold).astype(int), 
                    sample_weight=sample_weight,
                    title=f"{model_name} - Confusion Matrix"
                )
                report['plots']['confusion_matrix'] = cm_fig
                
                # Precision-Recall curve
                pr_fig = self.plot_precision_recall_curve(
                    y_true, y_proba, sample_weight=sample_weight,
                    title=f"{model_name} - Precision-Recall Curve"
                )
                report['plots']['precision_recall_curve'] = pr_fig
                
                # ROC curve
                roc_fig = self.plot_roc_curve(
                    y_true, y_proba, sample_weight=sample_weight,
                    title=f"{model_name} - ROC Curve"
                )
                report['plots']['roc_curve'] = roc_fig
                
                logger.info("Generated all evaluation plots successfully")
                
            except Exception as e:
                logger.warning(f"Failed to generate some plots: {str(e)}")
                report['plots']['error'] = str(e)
        
        # Generate recommendations based on metrics
        report['recommendations'] = self._generate_evaluation_recommendations(metrics)
        
        # Add business impact analysis
        report['business_impact'] = self._analyze_business_impact(metrics)
        
        logger.info(f"Evaluation report completed for {model_name} - "
                   f"PR-AUC: {metrics['pr_auc']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        return report
    
    def _generate_evaluation_recommendations(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on evaluation metrics.
        
        Args:
            metrics: Comprehensive metrics dictionary
            
        Returns:
            Dictionary containing recommendations
        """
        recommendations = {
            'performance_assessment': {},
            'threshold_recommendations': {},
            'model_improvements': [],
            'business_considerations': []
        }
        
        # Performance assessment
        pr_auc = metrics['pr_auc']
        f1_score = metrics['f1_score']
        precision = metrics['precision']
        recall = metrics['recall']
        
        if pr_auc >= 0.8:
            recommendations['performance_assessment']['overall'] = 'Excellent'
        elif pr_auc >= 0.6:
            recommendations['performance_assessment']['overall'] = 'Good'
        elif pr_auc >= 0.4:
            recommendations['performance_assessment']['overall'] = 'Fair'
        else:
            recommendations['performance_assessment']['overall'] = 'Poor'
        
        recommendations['performance_assessment']['details'] = {
            'pr_auc_assessment': f"PR-AUC of {pr_auc:.3f} indicates {recommendations['performance_assessment']['overall'].lower()} fraud detection capability",
            'f1_assessment': f"F1-score of {f1_score:.3f} shows {'balanced' if f1_score >= 0.5 else 'imbalanced'} precision-recall trade-off",
            'precision_assessment': f"Precision of {precision:.3f} means {precision*100:.1f}% of fraud predictions are correct",
            'recall_assessment': f"Recall of {recall:.3f} means {recall*100:.1f}% of actual fraud is detected"
        }
        
        # Threshold recommendations
        if precision < 0.3:
            recommendations['threshold_recommendations']['increase_threshold'] = {
                'reason': 'Low precision causing too many false positives',
                'impact': 'Will reduce customer friction but may miss some fraud'
            }
        elif recall < 0.7:
            recommendations['threshold_recommendations']['decrease_threshold'] = {
                'reason': 'Low recall missing too much fraud',
                'impact': 'Will catch more fraud but may increase false positives'
            }
        else:
            recommendations['threshold_recommendations']['current_threshold'] = {
                'assessment': 'Current threshold provides reasonable balance',
                'suggestion': 'Consider A/B testing with slight adjustments'
            }
        
        # Model improvement suggestions
        if pr_auc < 0.6:
            recommendations['model_improvements'].extend([
                'Consider feature engineering to capture more fraud patterns',
                'Evaluate ensemble methods or different algorithms',
                'Increase training data size, especially fraud examples',
                'Review data quality and feature selection'
            ])
        
        if precision < 0.5:
            recommendations['model_improvements'].extend([
                'Focus on reducing false positives through better feature selection',
                'Consider cost-sensitive learning approaches',
                'Implement more sophisticated threshold optimization'
            ])
        
        if recall < 0.8:
            recommendations['model_improvements'].extend([
                'Investigate missed fraud patterns in false negatives',
                'Consider oversampling techniques for fraud class',
                'Review feature importance for fraud detection signals'
            ])
        
        # Business considerations
        fpr = metrics['false_positive_rate']
        customer_friction = metrics['customer_friction_rate']
        
        recommendations['business_considerations'].extend([
            f"False positive rate of {fpr:.3f} affects {fpr*100:.1f}% of legitimate customers",
            f"Customer friction rate of {customer_friction:.3f} means {customer_friction*100:.1f}% of transactions require additional verification",
            "Consider implementing risk-based authentication for medium-risk transactions",
            "Monitor customer satisfaction metrics alongside fraud detection performance"
        ])
        
        return recommendations
    
    def _analyze_business_impact(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze business impact of model performance.
        
        Args:
            metrics: Comprehensive metrics dictionary
            
        Returns:
            Dictionary containing business impact analysis
        """
        # Extract key metrics
        tp = metrics['confusion_matrix']['true_positives']
        fp = metrics['confusion_matrix']['false_positives']
        fn = metrics['confusion_matrix']['false_negatives']
        tn = metrics['confusion_matrix']['true_negatives']
        
        total_transactions = metrics['dataset_stats']['total_transactions']
        fraud_rate = metrics['dataset_stats']['fraud_rate']
        
        # Business impact calculations (using placeholder values - should be configured)
        avg_fraud_loss = 100  # Average loss per fraud transaction
        false_positive_cost = 5  # Cost of investigating false positive
        
        business_impact = {
            'fraud_prevention': {
                'fraud_detected': tp,
                'fraud_missed': fn,
                'fraud_prevented_value': tp * avg_fraud_loss,
                'fraud_loss_value': fn * avg_fraud_loss,
                'fraud_prevention_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
            },
            'operational_costs': {
                'false_positive_investigations': fp,
                'false_positive_cost': fp * false_positive_cost,
                'cost_per_fraud_detected': (fp * false_positive_cost) / tp if tp > 0 else float('inf'),
                'investigation_workload': tp + fp
            },
            'customer_experience': {
                'customers_affected': tp + fp,
                'legitimate_customers_inconvenienced': fp,
                'customer_friction_percentage': ((tp + fp) / total_transactions) * 100,
                'false_positive_percentage': (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
            },
            'net_benefit': {
                'gross_fraud_prevented': tp * avg_fraud_loss,
                'operational_costs': fp * false_positive_cost,
                'net_benefit': (tp * avg_fraud_loss) - (fp * false_positive_cost),
                'roi_ratio': ((tp * avg_fraud_loss) - (fp * false_positive_cost)) / (fp * false_positive_cost) if fp > 0 else float('inf')
            }
        }
        
        return business_impact
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]],
                      metrics_to_compare: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: Dictionary mapping model names to their evaluation results
            metrics_to_compare: List of metrics to include in comparison
            
        Returns:
            Dictionary containing model comparison analysis
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['pr_auc', 'roc_auc', 'f1_score', 'precision', 'recall']
        
        comparison = {
            'models_compared': list(model_results.keys()),
            'comparison_metrics': metrics_to_compare,
            'metric_comparison': {},
            'ranking': {},
            'recommendations': {}
        }
        
        # Extract metrics for comparison
        for metric in metrics_to_compare:
            comparison['metric_comparison'][metric] = {}
            metric_values = []
            
            for model_name, results in model_results.items():
                if 'performance_metrics' in results and metric in results['performance_metrics']:
                    value = results['performance_metrics'][metric]
                    comparison['metric_comparison'][metric][model_name] = value
                    metric_values.append((model_name, value))
            
            # Rank models by this metric
            metric_values.sort(key=lambda x: x[1], reverse=True)
            comparison['ranking'][metric] = [model_name for model_name, _ in metric_values]
        
        # Overall ranking (weighted average of key metrics)
        weights = {'pr_auc': 0.3, 'f1_score': 0.3, 'precision': 0.2, 'recall': 0.2}
        overall_scores = {}
        
        for model_name in model_results.keys():
            score = 0
            total_weight = 0
            
            for metric, weight in weights.items():
                if metric in comparison['metric_comparison'] and model_name in comparison['metric_comparison'][metric]:
                    score += comparison['metric_comparison'][metric][model_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                overall_scores[model_name] = score / total_weight
        
        # Rank by overall score
        overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['ranking']['overall'] = [model_name for model_name, _ in overall_ranking]
        comparison['overall_scores'] = overall_scores
        
        # Generate recommendations
        if overall_ranking:
            best_model = overall_ranking[0][0]
            comparison['recommendations']['best_overall'] = best_model
            comparison['recommendations']['rationale'] = f"{best_model} achieved the highest weighted score across key metrics"
        
        return comparison
    def analyze_threshold_performance(self, y_true: np.ndarray, y_proba: np.ndarray,
                                     thresholds: Optional[np.ndarray] = None,
                                     sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze model performance across different threshold values.
        
        This method provides comprehensive threshold analysis to help optimize
        the decision threshold for different business scenarios. It evaluates
        performance metrics across a range of thresholds to identify optimal
        operating points.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted fraud probabilities
            thresholds: Optional array of thresholds to evaluate (default: 0.1 to 0.9 in 0.05 steps)
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary containing threshold analysis results
            
        Raises:
            ValueError: If input data is invalid
        """
        if y_true is None or y_proba is None:
            raise ValueError("y_true and y_proba cannot be None")
        
        if len(y_true) == 0 or len(y_proba) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(y_true) != len(y_proba):
            raise ValueError(f"y_true and y_proba must have same length: {len(y_true)} vs {len(y_proba)}")
        
        # Default threshold range if not provided
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)
        
        logger.info(f"Analyzing performance across {len(thresholds)} thresholds")
        
        # Initialize results storage
        threshold_results = []
        
        # Analyze each threshold
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate comprehensive metrics for this threshold
            metrics = self.calculate_comprehensive_metrics(y_true, y_proba, threshold, sample_weight)
            
            # Extract key metrics for threshold analysis
            threshold_result = {
                'threshold': float(threshold),
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'false_positive_rate': metrics['false_positive_rate'],
                'true_positive_rate': metrics['true_positive_rate'],
                'specificity': metrics['specificity'],
                'fraud_detection_rate': metrics['fraud_detection_rate'],
                'customer_friction_rate': metrics['customer_friction_rate'],
                'confusion_matrix': metrics['confusion_matrix']
            }
            
            threshold_results.append(threshold_result)
        
        # Convert to DataFrame for easier analysis
        df_results = pd.DataFrame(threshold_results)
        
        # Find optimal thresholds for different objectives - Requirements 5.4
        optimal_thresholds = self._find_optimal_operating_points(df_results)
        
        # Generate threshold recommendations - Requirements 5.5
        recommendations = self._generate_threshold_recommendations(df_results, optimal_thresholds)
        
        # Create summary statistics
        summary_stats = {
            'threshold_range': {
                'min': float(np.min(thresholds)),
                'max': float(np.max(thresholds)),
                'count': len(thresholds)
            },
            'performance_range': {
                'precision': {
                    'min': float(df_results['precision'].min()),
                    'max': float(df_results['precision'].max()),
                    'mean': float(df_results['precision'].mean())
                },
                'recall': {
                    'min': float(df_results['recall'].min()),
                    'max': float(df_results['recall'].max()),
                    'mean': float(df_results['recall'].mean())
                },
                'f1_score': {
                    'min': float(df_results['f1_score'].min()),
                    'max': float(df_results['f1_score'].max()),
                    'mean': float(df_results['f1_score'].mean())
                }
            }
        }
        
        analysis_results = {
            'threshold_analysis': threshold_results,
            'optimal_thresholds': optimal_thresholds,
            'recommendations': recommendations,
            'summary_statistics': summary_stats,
            'analysis_metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'n_thresholds_analyzed': len(thresholds),
                'n_samples': len(y_true),
                'fraud_rate': float(np.mean(y_true))
            }
        }
        
        logger.info(f"Threshold analysis completed - Analyzed {len(thresholds)} thresholds")
        return analysis_results
    
    def _find_optimal_operating_points(self, df_results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Find optimal operating points for different business objectives.
        
        Args:
            df_results: DataFrame containing threshold analysis results
            
        Returns:
            Dictionary containing optimal thresholds for different objectives
        """
        optimal_points = {}
        
        # F1-optimal threshold
        f1_optimal_idx = df_results['f1_score'].idxmax()
        optimal_points['f1_optimal'] = {
            'threshold': df_results.loc[f1_optimal_idx, 'threshold'],
            'f1_score': df_results.loc[f1_optimal_idx, 'f1_score'],
            'precision': df_results.loc[f1_optimal_idx, 'precision'],
            'recall': df_results.loc[f1_optimal_idx, 'recall']
        }
        
        # Precision-focused (minimize false positives)
        high_precision_mask = df_results['precision'] >= 0.5
        if high_precision_mask.any():
            precision_focused_df = df_results[high_precision_mask]
            precision_optimal_idx = precision_focused_df['recall'].idxmax()
            optimal_points['precision_focused'] = {
                'threshold': df_results.loc[precision_optimal_idx, 'threshold'],
                'precision': df_results.loc[precision_optimal_idx, 'precision'],
                'recall': df_results.loc[precision_optimal_idx, 'recall'],
                'f1_score': df_results.loc[precision_optimal_idx, 'f1_score']
            }
        
        # Recall-focused (minimize false negatives)
        high_recall_mask = df_results['recall'] >= 0.8
        if high_recall_mask.any():
            recall_focused_df = df_results[high_recall_mask]
            recall_optimal_idx = recall_focused_df['precision'].idxmax()
            optimal_points['recall_focused'] = {
                'threshold': df_results.loc[recall_optimal_idx, 'threshold'],
                'precision': df_results.loc[recall_optimal_idx, 'precision'],
                'recall': df_results.loc[recall_optimal_idx, 'recall'],
                'f1_score': df_results.loc[recall_optimal_idx, 'f1_score']
            }
        
        # Business-balanced (balance customer experience and fraud detection)
        # Find threshold that minimizes customer friction while maintaining reasonable fraud detection
        df_results['business_score'] = (
            df_results['recall'] * 0.6 +  # Weight fraud detection heavily
            (1 - df_results['customer_friction_rate']) * 0.4  # Weight customer experience
        )
        business_optimal_idx = df_results['business_score'].idxmax()
        optimal_points['business_balanced'] = {
            'threshold': df_results.loc[business_optimal_idx, 'threshold'],
            'business_score': df_results.loc[business_optimal_idx, 'business_score'],
            'precision': df_results.loc[business_optimal_idx, 'precision'],
            'recall': df_results.loc[business_optimal_idx, 'recall'],
            'customer_friction_rate': df_results.loc[business_optimal_idx, 'customer_friction_rate']
        }
        
        # Conservative (high precision, low customer friction)
        conservative_mask = (df_results['precision'] >= 0.7) & (df_results['customer_friction_rate'] <= 0.1)
        if conservative_mask.any():
            conservative_df = df_results[conservative_mask]
            conservative_optimal_idx = conservative_df['recall'].idxmax()
            optimal_points['conservative'] = {
                'threshold': df_results.loc[conservative_optimal_idx, 'threshold'],
                'precision': df_results.loc[conservative_optimal_idx, 'precision'],
                'recall': df_results.loc[conservative_optimal_idx, 'recall'],
                'customer_friction_rate': df_results.loc[conservative_optimal_idx, 'customer_friction_rate']
            }
        
        # Aggressive (high recall, catch more fraud)
        aggressive_mask = df_results['recall'] >= 0.9
        if aggressive_mask.any():
            aggressive_df = df_results[aggressive_mask]
            aggressive_optimal_idx = aggressive_df['precision'].idxmax()
            optimal_points['aggressive'] = {
                'threshold': df_results.loc[aggressive_optimal_idx, 'threshold'],
                'precision': df_results.loc[aggressive_optimal_idx, 'precision'],
                'recall': df_results.loc[aggressive_optimal_idx, 'recall'],
                'customer_friction_rate': df_results.loc[aggressive_optimal_idx, 'customer_friction_rate']
            }
        
        return optimal_points
    
    def _generate_threshold_recommendations(self, df_results: pd.DataFrame, 
                                          optimal_points: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate detailed recommendations for threshold selection.
        
        Args:
            df_results: DataFrame containing threshold analysis results
            optimal_points: Dictionary of optimal operating points
            
        Returns:
            Dictionary containing detailed recommendations
        """
        recommendations = {
            'strategy_recommendations': {},
            'business_scenarios': {},
            'implementation_guidance': {},
            'monitoring_recommendations': {}
        }
        
        # Strategy recommendations based on optimal points
        if 'f1_optimal' in optimal_points:
            f1_point = optimal_points['f1_optimal']
            recommendations['strategy_recommendations']['balanced_approach'] = {
                'threshold': f1_point['threshold'],
                'rationale': f"Optimal F1-score of {f1_point['f1_score']:.3f} provides best balance between precision and recall",
                'expected_performance': {
                    'precision': f1_point['precision'],
                    'recall': f1_point['recall'],
                    'f1_score': f1_point['f1_score']
                },
                'use_case': 'General fraud detection with balanced performance requirements'
            }
        
        if 'precision_focused' in optimal_points:
            precision_point = optimal_points['precision_focused']
            recommendations['strategy_recommendations']['low_friction_approach'] = {
                'threshold': precision_point['threshold'],
                'rationale': f"High precision of {precision_point['precision']:.3f} minimizes customer inconvenience",
                'expected_performance': {
                    'precision': precision_point['precision'],
                    'recall': precision_point['recall'],
                    'f1_score': precision_point['f1_score']
                },
                'use_case': 'Customer experience focused, willing to miss some fraud to reduce false positives'
            }
        
        if 'recall_focused' in optimal_points:
            recall_point = optimal_points['recall_focused']
            recommendations['strategy_recommendations']['fraud_focused_approach'] = {
                'threshold': recall_point['threshold'],
                'rationale': f"High recall of {recall_point['recall']:.3f} maximizes fraud detection",
                'expected_performance': {
                    'precision': recall_point['precision'],
                    'recall': recall_point['recall'],
                    'f1_score': recall_point['f1_score']
                },
                'use_case': 'Maximum fraud prevention, willing to accept higher false positive rate'
            }
        
        # Business scenario recommendations
        recommendations['business_scenarios'] = {
            'startup_phase': {
                'recommended_strategy': 'balanced_approach',
                'rationale': 'Balance fraud prevention with customer acquisition',
                'monitoring_focus': 'Customer feedback and fraud losses'
            },
            'mature_business': {
                'recommended_strategy': 'low_friction_approach',
                'rationale': 'Prioritize customer experience with established customer base',
                'monitoring_focus': 'Customer satisfaction and retention'
            },
            'high_risk_environment': {
                'recommended_strategy': 'fraud_focused_approach',
                'rationale': 'Maximum fraud prevention in high-risk scenarios',
                'monitoring_focus': 'Fraud detection rate and financial losses'
            },
            'regulatory_compliance': {
                'recommended_strategy': 'conservative',
                'rationale': 'High precision required for regulatory reporting',
                'monitoring_focus': 'Compliance metrics and audit requirements'
            }
        }
        
        # Implementation guidance
        recommendations['implementation_guidance'] = {
            'gradual_rollout': [
                'Start with conservative threshold to minimize customer impact',
                'Monitor customer feedback and fraud detection performance',
                'Gradually adjust threshold based on observed performance',
                'Implement A/B testing to compare threshold strategies'
            ],
            'monitoring_setup': [
                'Set up real-time monitoring of key metrics (precision, recall, customer complaints)',
                'Implement automated alerts for significant performance changes',
                'Create dashboards for business stakeholders',
                'Establish regular review cycles for threshold optimization'
            ],
            'fallback_strategies': [
                'Implement circuit breaker for model failures',
                'Define manual review processes for edge cases',
                'Create escalation procedures for high-risk transactions',
                'Maintain rule-based backup system'
            ]
        }
        
        # Monitoring recommendations
        recommendations['monitoring_recommendations'] = {
            'key_metrics': [
                'Precision and recall trends over time',
                'False positive rate and customer complaints',
                'Fraud detection rate and financial impact',
                'Model performance degradation indicators'
            ],
            'alert_thresholds': {
                'precision_drop': 'Alert if precision drops below 80% of baseline',
                'recall_drop': 'Alert if recall drops below 85% of baseline',
                'false_positive_spike': 'Alert if false positive rate increases by 50%',
                'model_drift': 'Alert if prediction distribution changes significantly'
            },
            'review_schedule': {
                'daily': 'Monitor key performance indicators',
                'weekly': 'Review threshold performance and customer feedback',
                'monthly': 'Comprehensive performance analysis and threshold optimization',
                'quarterly': 'Strategic review of fraud detection strategy'
            }
        }
        
        return recommendations
    
    def generate_performance_report(self, y_true: np.ndarray, y_proba: np.ndarray,
                                  current_threshold: float = 0.5,
                                  sample_weight: Optional[np.ndarray] = None,
                                  model_name: str = "Fraud Detection Model",
                                  include_threshold_analysis: bool = True,
                                  include_business_metrics: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with business impact metrics.
        
        This method creates a detailed performance analysis report that includes:
        - Current model performance at specified threshold
        - Threshold analysis across different operating points
        - Business impact analysis with cost-benefit calculations
        - Recommendations for optimization and monitoring
        
        Args:
            y_true: True binary labels
            y_proba: Predicted fraud probabilities
            current_threshold: Current operating threshold
            sample_weight: Optional sample weights
            model_name: Name of the model being analyzed
            include_threshold_analysis: Whether to include comprehensive threshold analysis
            include_business_metrics: Whether to include business impact calculations
            
        Returns:
            Dictionary containing comprehensive performance report
        """
        logger.info(f"Generating comprehensive performance report for {model_name}")
        
        # Initialize report structure
        report = {
            'model_name': model_name,
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'current_threshold': current_threshold,
                'total_samples': len(y_true),
                'fraud_samples': int(np.sum(y_true)),
                'fraud_rate': float(np.mean(y_true))
            },
            'current_performance': {},
            'threshold_analysis': {},
            'business_impact': {},
            'recommendations': {},
            'monitoring_guidance': {}
        }
        
        # Current performance analysis
        current_metrics = self.calculate_comprehensive_metrics(
            y_true, y_proba, current_threshold, sample_weight
        )
        report['current_performance'] = current_metrics
        
        # Threshold analysis if requested - Requirements 5.4
        if include_threshold_analysis:
            threshold_analysis = self.analyze_threshold_performance(
                y_true, y_proba, sample_weight=sample_weight
            )
            report['threshold_analysis'] = threshold_analysis
        
        # Business impact analysis if requested - Requirements 5.5
        if include_business_metrics:
            business_impact = self._analyze_detailed_business_impact(
                current_metrics, y_true, y_proba, current_threshold
            )
            report['business_impact'] = business_impact
        
        # Generate comprehensive recommendations
        report['recommendations'] = self._generate_comprehensive_recommendations(
            current_metrics, 
            report.get('threshold_analysis', {}),
            report.get('business_impact', {})
        )
        
        # Monitoring and maintenance guidance
        report['monitoring_guidance'] = self._generate_monitoring_guidance(current_metrics)
        
        # Executive summary
        report['executive_summary'] = self._generate_executive_summary(report)
        
        logger.info(f"Performance report completed for {model_name} - "
                   f"Current PR-AUC: {current_metrics['pr_auc']:.3f}, "
                   f"F1: {current_metrics['f1_score']:.3f}")
        
        return report
    
    def _analyze_detailed_business_impact(self, metrics: Dict[str, Any], 
                                        y_true: np.ndarray, y_proba: np.ndarray,
                                        threshold: float) -> Dict[str, Any]:
        """
        Analyze detailed business impact with cost-benefit calculations.
        
        Args:
            metrics: Current performance metrics
            y_true: True binary labels
            y_proba: Predicted probabilities
            threshold: Current threshold
            
        Returns:
            Dictionary containing detailed business impact analysis
        """
        # Extract confusion matrix components
        cm = metrics['confusion_matrix']
        tp, fp, fn, tn = cm['true_positives'], cm['false_positives'], cm['false_negatives'], cm['true_negatives']
        
        # Business parameters (should be configurable in production)
        avg_fraud_amount = 150  # Average fraud transaction amount
        investigation_cost = 10  # Cost to investigate each flagged transaction
        customer_service_cost = 25  # Cost when legitimate customer is inconvenienced
        fraud_prevention_value = 0.8  # Fraction of fraud amount that can be recovered
        
        # Financial impact calculations
        financial_impact = {
            'fraud_prevention': {
                'fraud_detected_count': tp,
                'fraud_missed_count': fn,
                'fraud_prevented_value': tp * avg_fraud_amount * fraud_prevention_value,
                'fraud_losses': fn * avg_fraud_amount,
                'fraud_prevention_rate': tp / (tp + fn) if (tp + fn) > 0 else 0
            },
            'operational_costs': {
                'investigations_required': tp + fp,
                'investigation_costs': (tp + fp) * investigation_cost,
                'customer_service_costs': fp * customer_service_cost,
                'total_operational_costs': (tp + fp) * investigation_cost + fp * customer_service_cost
            },
            'customer_impact': {
                'customers_inconvenienced': fp,
                'customer_satisfaction_impact': fp * 0.1,  # Estimated satisfaction score impact
                'potential_customer_churn': fp * 0.02,  # Estimated churn rate from false positives
                'customer_friction_percentage': ((tp + fp) / len(y_true)) * 100
            },
            'net_benefit_analysis': {
                'gross_benefit': tp * avg_fraud_amount * fraud_prevention_value,
                'total_costs': (tp + fp) * investigation_cost + fp * customer_service_cost,
                'fraud_losses': fn * avg_fraud_amount,
                'net_benefit': (tp * avg_fraud_amount * fraud_prevention_value) - 
                              ((tp + fp) * investigation_cost + fp * customer_service_cost) - 
                              (fn * avg_fraud_amount),
                'roi_percentage': (((tp * avg_fraud_amount * fraud_prevention_value) - 
                                  ((tp + fp) * investigation_cost + fp * customer_service_cost)) / 
                                 ((tp + fp) * investigation_cost + fp * customer_service_cost)) * 100 
                                if ((tp + fp) * investigation_cost + fp * customer_service_cost) > 0 else 0
            }
        }
        
        # Performance benchmarks
        industry_benchmarks = {
            'fraud_detection_rate': {'excellent': 0.95, 'good': 0.85, 'acceptable': 0.75},
            'false_positive_rate': {'excellent': 0.01, 'good': 0.03, 'acceptable': 0.05},
            'precision': {'excellent': 0.8, 'good': 0.6, 'acceptable': 0.4},
            'customer_friction_rate': {'excellent': 0.02, 'good': 0.05, 'acceptable': 0.1}
        }
        
        # Benchmark comparison
        current_fdr = metrics['fraud_detection_rate']
        current_fpr = metrics['false_positive_rate']
        current_precision = metrics['precision']
        current_cfr = metrics['customer_friction_rate']
        
        benchmark_analysis = {
            'fraud_detection_rate': self._categorize_performance(current_fdr, industry_benchmarks['fraud_detection_rate']),
            'false_positive_rate': self._categorize_performance(current_fpr, industry_benchmarks['false_positive_rate'], lower_is_better=True),
            'precision': self._categorize_performance(current_precision, industry_benchmarks['precision']),
            'customer_friction_rate': self._categorize_performance(current_cfr, industry_benchmarks['customer_friction_rate'], lower_is_better=True)
        }
        
        return {
            'financial_impact': financial_impact,
            'benchmark_analysis': benchmark_analysis,
            'industry_benchmarks': industry_benchmarks,
            'business_kpis': {
                'cost_per_fraud_detected': financial_impact['operational_costs']['total_operational_costs'] / tp if tp > 0 else float('inf'),
                'fraud_prevention_efficiency': financial_impact['fraud_prevention']['fraud_prevented_value'] / financial_impact['operational_costs']['total_operational_costs'] if financial_impact['operational_costs']['total_operational_costs'] > 0 else 0,
                'customer_impact_ratio': fp / (tp + fp) if (tp + fp) > 0 else 0,
                'net_benefit_per_transaction': financial_impact['net_benefit_analysis']['net_benefit'] / len(y_true)
            }
        }
    
    def _categorize_performance(self, value: float, benchmarks: Dict[str, float], 
                               lower_is_better: bool = False) -> Dict[str, Any]:
        """
        Categorize performance against industry benchmarks.
        
        Args:
            value: Current performance value
            benchmarks: Dictionary of benchmark thresholds
            lower_is_better: Whether lower values indicate better performance
            
        Returns:
            Dictionary containing performance category and analysis
        """
        if lower_is_better:
            if value <= benchmarks['excellent']:
                category = 'excellent'
            elif value <= benchmarks['good']:
                category = 'good'
            elif value <= benchmarks['acceptable']:
                category = 'acceptable'
            else:
                category = 'needs_improvement'
        else:
            if value >= benchmarks['excellent']:
                category = 'excellent'
            elif value >= benchmarks['good']:
                category = 'good'
            elif value >= benchmarks['acceptable']:
                category = 'acceptable'
            else:
                category = 'needs_improvement'
        
        return {
            'current_value': value,
            'category': category,
            'benchmarks': benchmarks,
            'gap_to_excellent': benchmarks['excellent'] - value if not lower_is_better else value - benchmarks['excellent']
        }
    
    def _generate_comprehensive_recommendations(self, current_metrics: Dict[str, Any],
                                              threshold_analysis: Dict[str, Any],
                                              business_impact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations based on all analyses.
        
        Args:
            current_metrics: Current performance metrics
            threshold_analysis: Threshold analysis results
            business_impact: Business impact analysis
            
        Returns:
            Dictionary containing comprehensive recommendations
        """
        recommendations = {
            'immediate_actions': [],
            'strategic_improvements': [],
            'threshold_optimization': {},
            'business_optimization': [],
            'monitoring_enhancements': []
        }
        
        # Analyze current performance and generate immediate actions
        pr_auc = current_metrics['pr_auc']
        f1_score = current_metrics['f1_score']
        precision = current_metrics['precision']
        recall = current_metrics['recall']
        
        if pr_auc < 0.5:
            recommendations['immediate_actions'].append({
                'priority': 'high',
                'action': 'Model performance is below random baseline - immediate model retraining required',
                'rationale': f'PR-AUC of {pr_auc:.3f} indicates poor fraud detection capability'
            })
        
        if precision < 0.3:
            recommendations['immediate_actions'].append({
                'priority': 'high',
                'action': 'Increase decision threshold to reduce false positives',
                'rationale': f'Precision of {precision:.3f} is causing excessive customer friction'
            })
        
        if recall < 0.6:
            recommendations['immediate_actions'].append({
                'priority': 'medium',
                'action': 'Review feature engineering and model architecture',
                'rationale': f'Recall of {recall:.3f} indicates significant fraud is being missed'
            })
        
        # Strategic improvements based on performance gaps
        if pr_auc < 0.8:
            recommendations['strategic_improvements'].extend([
                'Invest in advanced feature engineering to capture more fraud patterns',
                'Consider ensemble methods or deep learning approaches',
                'Expand training dataset with more diverse fraud examples',
                'Implement active learning to continuously improve model'
            ])
        
        # Threshold optimization recommendations
        if threshold_analysis and 'optimal_thresholds' in threshold_analysis:
            optimal_points = threshold_analysis['optimal_thresholds']
            
            if 'business_balanced' in optimal_points:
                business_point = optimal_points['business_balanced']
                recommendations['threshold_optimization']['primary_recommendation'] = {
                    'threshold': business_point['threshold'],
                    'rationale': 'Optimizes balance between fraud detection and customer experience',
                    'expected_improvement': f"Business score: {business_point.get('business_score', 'N/A')}"
                }
        
        # Business optimization recommendations
        if business_impact and 'business_kpis' in business_impact:
            kpis = business_impact['business_kpis']
            
            if kpis['cost_per_fraud_detected'] > 50:
                recommendations['business_optimization'].append({
                    'area': 'operational_efficiency',
                    'recommendation': 'Optimize investigation processes to reduce cost per fraud detected',
                    'current_cost': kpis['cost_per_fraud_detected']
                })
            
            if kpis['customer_impact_ratio'] > 0.5:
                recommendations['business_optimization'].append({
                    'area': 'customer_experience',
                    'recommendation': 'Implement risk-based authentication to reduce customer friction',
                    'current_ratio': kpis['customer_impact_ratio']
                })
        
        # Monitoring enhancements
        recommendations['monitoring_enhancements'] = [
            'Implement real-time model performance monitoring',
            'Set up automated alerts for performance degradation',
            'Create business impact dashboards for stakeholders',
            'Establish regular model retraining schedules',
            'Monitor customer feedback and satisfaction metrics'
        ]
        
        return recommendations
    
    def _generate_monitoring_guidance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate monitoring and maintenance guidance.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            Dictionary containing monitoring guidance
        """
        return {
            'key_metrics_to_monitor': [
                'PR-AUC and ROC-AUC trends',
                'Precision and recall stability',
                'False positive rate changes',
                'Customer complaint rates',
                'Fraud detection effectiveness'
            ],
            'alert_thresholds': {
                'pr_auc_drop': f"Alert if PR-AUC drops below {metrics['pr_auc'] * 0.9:.3f}",
                'precision_drop': f"Alert if precision drops below {metrics['precision'] * 0.8:.3f}",
                'recall_drop': f"Alert if recall drops below {metrics['recall'] * 0.85:.3f}",
                'false_positive_spike': f"Alert if FPR exceeds {metrics['false_positive_rate'] * 1.5:.3f}"
            },
            'monitoring_frequency': {
                'real_time': ['Transaction volume', 'System availability', 'Prediction latency'],
                'hourly': ['Fraud detection rate', 'False positive rate'],
                'daily': ['Model performance metrics', 'Customer feedback'],
                'weekly': ['Threshold optimization analysis', 'Business impact review'],
                'monthly': ['Comprehensive model evaluation', 'Strategic performance review']
            },
            'maintenance_schedule': {
                'daily': 'Monitor key performance indicators and system health',
                'weekly': 'Review model performance and investigate anomalies',
                'monthly': 'Comprehensive performance analysis and threshold optimization',
                'quarterly': 'Model retraining evaluation and strategic review'
            }
        }
    
    def _generate_executive_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate executive summary of the performance report.
        
        Args:
            report: Complete performance report
            
        Returns:
            Dictionary containing executive summary
        """
        current_perf = report['current_performance']
        
        # Overall performance assessment
        pr_auc = current_perf['pr_auc']
        if pr_auc >= 0.8:
            overall_assessment = 'Excellent'
        elif pr_auc >= 0.6:
            overall_assessment = 'Good'
        elif pr_auc >= 0.4:
            overall_assessment = 'Fair'
        else:
            overall_assessment = 'Poor'
        
        # Key findings
        key_findings = []
        
        if current_perf['precision'] < 0.5:
            key_findings.append(f"High false positive rate ({current_perf['false_positive_rate']:.1%}) impacting customer experience")
        
        if current_perf['recall'] < 0.7:
            key_findings.append(f"Missing {(1-current_perf['recall']):.1%} of fraud cases")
        
        if current_perf['pr_auc'] > 0.7:
            key_findings.append(f"Strong fraud detection capability with PR-AUC of {pr_auc:.3f}")
        
        # Business impact summary
        business_summary = "Performance analysis completed"
        if 'business_impact' in report and 'business_kpis' in report['business_impact']:
            kpis = report['business_impact']['business_kpis']
            net_benefit = kpis.get('net_benefit_per_transaction', 0)
            if net_benefit > 0:
                business_summary = f"Positive business impact with net benefit of ${net_benefit:.2f} per transaction"
            else:
                business_summary = f"Negative business impact with net loss of ${abs(net_benefit):.2f} per transaction"
        
        return {
            'overall_assessment': overall_assessment,
            'key_metrics': {
                'pr_auc': current_perf['pr_auc'],
                'f1_score': current_perf['f1_score'],
                'precision': current_perf['precision'],
                'recall': current_perf['recall']
            },
            'key_findings': key_findings,
            'business_impact_summary': business_summary,
            'primary_recommendations': report.get('recommendations', {}).get('immediate_actions', [])[:3],
            'next_steps': [
                'Review and implement immediate action items',
                'Monitor key performance indicators',
                'Schedule regular performance reviews'
            ]
        }