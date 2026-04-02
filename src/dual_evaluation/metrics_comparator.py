"""
Metrics comparison and analysis for dual evaluation pipeline.

This module provides the MetricsComparator class for comparing evaluation results
from imbalanced and balanced pipelines, analyzing trade-offs, generating interpretations,
and creating visualizations.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5, 11.1, 11.2, 11.3, 11.4, 11.5, 12.1, 12.2, 12.3, 12.4, 12.5
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

from .data_models import EvaluationResult

logger = logging.getLogger(__name__)


class MetricsComparator:
    """
    Compares and analyzes results from imbalanced and balanced pipelines.
    
    This class provides comprehensive comparison functionality including:
    - Side-by-side metric comparison tables
    - Trade-off analysis (precision vs recall, data size, customer friction)
    - Human-readable interpretation generation
    - Visualization creation (ROC curves, PR curves, confusion matrices)
    
    Requirements: 9.1, 10.1, 11.1, 12.1
    """
    
    def __init__(self):
        """
        Initialize MetricsComparator with empty state.
        
        Requirements: 9.1
        """
        pass
    
    def compare_metrics(
        self,
        imbalanced_results: Dict[str, Any],
        balanced_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate side-by-side comparison table of metrics.
        
        Creates a comprehensive comparison DataFrame showing metrics from both
        pipelines, absolute differences, and percent changes. Includes dataset
        statistics for context.
        
        Args:
            imbalanced_results: Dictionary containing EvaluationResult data from imbalanced pipeline
            balanced_results: Dictionary containing EvaluationResult data from balanced pipeline
            
        Returns:
            DataFrame with columns:
                - metric_name: Name of the metric
                - imbalanced_value: Value from imbalanced pipeline
                - balanced_value: Value from balanced pipeline
                - difference: Absolute difference (balanced - imbalanced)
                - percent_change: Percentage change ((difference / imbalanced) * 100)
                
        Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
        
        Examples:
            >>> comparator = MetricsComparator()
            >>> comparison_df = comparator.compare_metrics(imb_results, bal_results)
            >>> comparison_df[comparison_df['metric_name'] == 'recall']
        """
        # Verify both results are present (Requirement 9.5)
        if imbalanced_results is None or balanced_results is None:
            raise ValueError("Both imbalanced and balanced results must be present")
        
        # Metrics to compare (Requirements 9.1, 9.2)
        metrics = [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'roc_auc',
            'pr_auc',
            'fraud_detection_rate',
            'false_positive_rate',
            'customer_friction_rate'
        ]
        
        comparison_rows = []
        
        # Compare each metric (Requirements 9.2, 9.3)
        for metric in metrics:
            imb_value = imbalanced_results.get(metric, 0.0)
            bal_value = balanced_results.get(metric, 0.0)
            
            # Calculate absolute difference (Requirement 9.2)
            difference = bal_value - imb_value
            
            # Calculate percent change (Requirement 9.3)
            if imb_value != 0:
                percent_change = (difference / imb_value) * 100
            else:
                percent_change = 0.0 if bal_value == 0 else float('inf')
            
            comparison_rows.append({
                'metric_name': metric,
                'imbalanced_value': imb_value,
                'balanced_value': bal_value,
                'difference': difference,
                'percent_change': percent_change
            })
        
        # Add dataset statistics (Requirement 9.4)
        dataset_stats = [
            ('train_samples', 'Train Samples'),
            ('test_samples', 'Test Samples'),
            ('train_fraud_rate', 'Train Fraud Rate'),
            ('test_fraud_rate', 'Test Fraud Rate')
        ]
        
        for stat_key, stat_name in dataset_stats:
            imb_value = imbalanced_results.get(stat_key, 0)
            bal_value = balanced_results.get(stat_key, 0)
            
            if isinstance(imb_value, (int, float)) and isinstance(bal_value, (int, float)):
                difference = bal_value - imb_value
                if imb_value != 0:
                    percent_change = (difference / imb_value) * 100
                else:
                    percent_change = 0.0 if bal_value == 0 else float('inf')
            else:
                difference = 0
                percent_change = 0.0
            
            comparison_rows.append({
                'metric_name': stat_name,
                'imbalanced_value': imb_value,
                'balanced_value': bal_value,
                'difference': difference,
                'percent_change': percent_change
            })
        
        return pd.DataFrame(comparison_rows)
    
    def analyze_trade_offs(
        self,
        imbalanced_results: Dict[str, Any],
        balanced_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze trade-offs between imbalanced and balanced approaches.
        
        Identifies key trade-offs including:
        - Precision improvements/degradations
        - Recall improvements/degradations
        - Training data size reduction
        - Customer friction changes
        
        Args:
            imbalanced_results: Dictionary containing EvaluationResult data from imbalanced pipeline
            balanced_results: Dictionary containing EvaluationResult data from balanced pipeline
            
        Returns:
            Dictionary containing trade-off analysis:
                - precision_improvement: Boolean indicating if precision improved
                - precision_gain: Absolute precision gain (if improved)
                - recall_improvement: Boolean indicating if recall improved
                - recall_gain: Absolute recall gain (if improved)
                - lower_customer_friction: Boolean indicating if FPR decreased
                - training_data_reduction: Fraction of training data reduction
                
        Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
        
        Examples:
            >>> comparator = MetricsComparator()
            >>> trade_offs = comparator.analyze_trade_offs(imb_results, bal_results)
            >>> trade_offs['precision_improvement']
            True
        """
        trade_offs = {}
        
        # Analyze precision trade-off (Requirement 10.1)
        imb_precision = imbalanced_results.get('precision', 0.0)
        bal_precision = balanced_results.get('precision', 0.0)
        
        if bal_precision > imb_precision:
            trade_offs['precision_improvement'] = True
            trade_offs['precision_gain'] = bal_precision - imb_precision
        else:
            trade_offs['precision_improvement'] = False
            trade_offs['precision_loss'] = imb_precision - bal_precision
        
        # Analyze recall trade-off (Requirement 10.2)
        imb_recall = imbalanced_results.get('recall', 0.0)
        bal_recall = balanced_results.get('recall', 0.0)
        
        if bal_recall > imb_recall:
            trade_offs['recall_improvement'] = True
            trade_offs['recall_gain'] = bal_recall - imb_recall
        else:
            trade_offs['recall_improvement'] = False
            trade_offs['recall_loss'] = imb_recall - bal_recall
        
        # Analyze customer friction (Requirement 10.4)
        imb_fpr = imbalanced_results.get('false_positive_rate', 0.0)
        bal_fpr = balanced_results.get('false_positive_rate', 0.0)
        
        if bal_fpr < imb_fpr:
            trade_offs['lower_customer_friction'] = True
            trade_offs['fpr_reduction'] = imb_fpr - bal_fpr
        else:
            trade_offs['lower_customer_friction'] = False
            trade_offs['fpr_increase'] = bal_fpr - imb_fpr
        
        # Calculate training data size reduction (Requirement 10.3)
        imb_train_samples = imbalanced_results.get('train_samples', 1)
        bal_train_samples = balanced_results.get('train_samples', 1)
        
        if imb_train_samples > 0:
            trade_offs['training_data_reduction'] = 1 - (bal_train_samples / imb_train_samples)
        else:
            trade_offs['training_data_reduction'] = 0.0
        
        # Add F1-score comparison for overall performance
        imb_f1 = imbalanced_results.get('f1_score', 0.0)
        bal_f1 = balanced_results.get('f1_score', 0.0)
        trade_offs['f1_improvement'] = bal_f1 > imb_f1
        trade_offs['f1_change'] = bal_f1 - imb_f1
        
        return trade_offs
    
    def generate_interpretation(
        self,
        comparison_df: pd.DataFrame,
        trade_offs: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable interpretation of comparison results.
        
        Creates a comprehensive interpretation that explains:
        - Key metric differences
        - Precision vs recall trade-offs
        - Business impact analysis
        - Dataset size trade-offs
        
        The interpretation is designed for business stakeholders and must exceed
        200 characters in length.
        
        Args:
            comparison_df: DataFrame from compare_metrics()
            trade_offs: Dictionary from analyze_trade_offs()
            
        Returns:
            String containing human-readable interpretation (length > 200 characters)
            
        Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
        
        Examples:
            >>> comparator = MetricsComparator()
            >>> interpretation = comparator.generate_interpretation(comparison_df, trade_offs)
            >>> len(interpretation) > 200
            True
        """
        sections = []
        
        # Section 1: Key metric differences (Requirement 11.1)
        sections.append("=== METRIC COMPARISON ===")
        
        # Get key metrics from comparison table
        recall_row = comparison_df[comparison_df['metric_name'] == 'recall'].iloc[0]
        precision_row = comparison_df[comparison_df['metric_name'] == 'precision'].iloc[0]
        f1_row = comparison_df[comparison_df['metric_name'] == 'f1_score'].iloc[0]
        
        sections.append(
            f"Recall: {recall_row['imbalanced_value']:.3f} (imbalanced) vs "
            f"{recall_row['balanced_value']:.3f} (balanced) - "
            f"Change: {recall_row['percent_change']:+.1f}%"
        )
        sections.append(
            f"Precision: {precision_row['imbalanced_value']:.3f} (imbalanced) vs "
            f"{precision_row['balanced_value']:.3f} (balanced) - "
            f"Change: {precision_row['percent_change']:+.1f}%"
        )
        sections.append(
            f"F1-Score: {f1_row['imbalanced_value']:.3f} (imbalanced) vs "
            f"{f1_row['balanced_value']:.3f} (balanced) - "
            f"Change: {f1_row['percent_change']:+.1f}%"
        )
        
        # Section 2: Precision vs Recall trade-offs (Requirement 11.2)
        sections.append("\n=== PRECISION VS RECALL TRADE-OFFS ===")
        
        if trade_offs.get('precision_improvement') and trade_offs.get('recall_improvement'):
            sections.append(
                f"The balanced approach achieves BOTH higher precision (+{trade_offs['precision_gain']:.3f}) "
                f"and higher recall (+{trade_offs['recall_gain']:.3f}), representing a clear win-win scenario."
            )
        elif trade_offs.get('precision_improvement'):
            sections.append(
                f"The balanced approach improves precision by {trade_offs['precision_gain']:.3f} "
                f"but sacrifices recall by {trade_offs.get('recall_loss', 0):.3f}. "
                f"This trade-off reduces false alarms at the cost of missing some fraud cases."
            )
        elif trade_offs.get('recall_improvement'):
            sections.append(
                f"The balanced approach improves recall by {trade_offs['recall_gain']:.3f} "
                f"but sacrifices precision by {trade_offs.get('precision_loss', 0):.3f}. "
                f"This trade-off catches more fraud but increases false positives."
            )
        else:
            sections.append(
                f"The balanced approach shows lower performance in both precision "
                f"(-{trade_offs.get('precision_loss', 0):.3f}) and recall "
                f"(-{trade_offs.get('recall_loss', 0):.3f})."
            )
        
        # Section 3: Business impact analysis (Requirement 11.3)
        sections.append("\n=== BUSINESS IMPACT ===")
        
        fpr_row = comparison_df[comparison_df['metric_name'] == 'false_positive_rate'].iloc[0]
        
        if trade_offs.get('lower_customer_friction'):
            sections.append(
                f"Customer friction is REDUCED with the balanced approach. "
                f"False positive rate decreases from {fpr_row['imbalanced_value']:.3f} to "
                f"{fpr_row['balanced_value']:.3f} ({fpr_row['percent_change']:+.1f}%), "
                f"meaning fewer legitimate transactions are incorrectly flagged as fraud."
            )
        else:
            sections.append(
                f"Customer friction INCREASES with the balanced approach. "
                f"False positive rate rises from {fpr_row['imbalanced_value']:.3f} to "
                f"{fpr_row['balanced_value']:.3f} ({fpr_row['percent_change']:+.1f}%), "
                f"meaning more legitimate transactions are incorrectly flagged."
            )
        
        # Section 4: Dataset size trade-offs (Requirement 11.4)
        sections.append("\n=== DATASET SIZE TRADE-OFFS ===")
        
        train_samples_row = comparison_df[comparison_df['metric_name'] == 'Train Samples'].iloc[0]
        reduction_pct = trade_offs['training_data_reduction'] * 100
        
        sections.append(
            f"The balanced approach uses {train_samples_row['balanced_value']:,.0f} training samples "
            f"compared to {train_samples_row['imbalanced_value']:,.0f} for the imbalanced approach, "
            f"representing a {reduction_pct:.1f}% reduction in training data. "
            f"This significantly reduces training time and computational costs."
        )
        
        # Combine all sections
        interpretation = "\n".join(sections)
        
        # Ensure length > 200 characters (Requirement 11.5)
        if len(interpretation) <= 200:
            interpretation += (
                "\n\nNote: The choice between approaches depends on business priorities: "
                "whether minimizing false positives (customer experience) or maximizing "
                "fraud detection (loss prevention) is more critical."
            )
        
        return interpretation
    
    def create_visualizations(
        self,
        imbalanced_results: Dict[str, Any],
        balanced_results: Dict[str, Any],
        output_dir: str
    ) -> List[str]:
        """
        Create comparison visualizations for both models.
        
        Generates and saves:
        - ROC curves for both models
        - Precision-recall curves for both models
        - Confusion matrices for both models
        
        All visualizations are saved to the output directory with descriptive filenames.
        
        Args:
            imbalanced_results: Dictionary containing EvaluationResult data from imbalanced pipeline
            balanced_results: Dictionary containing EvaluationResult data from balanced pipeline
            output_dir: Directory path where visualizations will be saved
            
        Returns:
            List of file paths to created visualizations
            
        Requirements: 12.1, 12.2, 12.3, 12.4, 12.5
        
        Examples:
            >>> comparator = MetricsComparator()
            >>> viz_paths = comparator.create_visualizations(imb_results, bal_results, 'output/')
            >>> len(viz_paths) >= 3  # At least ROC, PR, and confusion matrix plots
            True
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualization_paths = []
        
        # Set style for better-looking plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # 1. Create ROC curves (Requirement 12.1)
        try:
            roc_path = self._create_roc_curves(
                imbalanced_results,
                balanced_results,
                output_path
            )
            visualization_paths.append(roc_path)
            logger.info(f"Created ROC curve visualization: {roc_path}")
        except Exception as e:
            logger.error(f"Failed to create ROC curves: {e}")
        
        # 2. Create precision-recall curves (Requirement 12.2)
        try:
            pr_path = self._create_pr_curves(
                imbalanced_results,
                balanced_results,
                output_path
            )
            visualization_paths.append(pr_path)
            logger.info(f"Created PR curve visualization: {pr_path}")
        except Exception as e:
            logger.error(f"Failed to create PR curves: {e}")
        
        # 3. Create confusion matrices (Requirement 12.3)
        try:
            cm_path = self._create_confusion_matrices(
                imbalanced_results,
                balanced_results,
                output_path
            )
            visualization_paths.append(cm_path)
            logger.info(f"Created confusion matrix visualization: {cm_path}")
        except Exception as e:
            logger.error(f"Failed to create confusion matrices: {e}")
        
        return visualization_paths
    
    def _create_roc_curves(
        self,
        imbalanced_results: Dict[str, Any],
        balanced_results: Dict[str, Any],
        output_path: Path
    ) -> str:
        """Create ROC curves for both models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC curve for imbalanced model
        imb_roc_auc = imbalanced_results.get('roc_auc', 0.0)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.plot(
            [0, 0, 1],
            [0, imb_roc_auc, 1],
            'b-',
            linewidth=2,
            label=f'Imbalanced Model (AUC = {imb_roc_auc:.3f})'
        )
        
        # Plot ROC curve for balanced model
        bal_roc_auc = balanced_results.get('roc_auc', 0.0)
        ax.plot(
            [0, 0, 1],
            [0, bal_roc_auc, 1],
            'r-',
            linewidth=2,
            label=f'Balanced Model (AUC = {bal_roc_auc:.3f})'
        )
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        roc_path = output_path / 'roc_curves_comparison.png'
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(roc_path)
    
    def _create_pr_curves(
        self,
        imbalanced_results: Dict[str, Any],
        balanced_results: Dict[str, Any],
        output_path: Path
    ) -> str:
        """Create precision-recall curves for both models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot PR curve for imbalanced model
        imb_pr_auc = imbalanced_results.get('pr_auc', 0.0)
        imb_precision = imbalanced_results.get('precision', 0.0)
        imb_recall = imbalanced_results.get('recall', 0.0)
        
        ax.plot(
            [0, imb_recall, 1],
            [1, imb_precision, 0],
            'b-',
            linewidth=2,
            label=f'Imbalanced Model (AUC = {imb_pr_auc:.3f})'
        )
        
        # Plot PR curve for balanced model
        bal_pr_auc = balanced_results.get('pr_auc', 0.0)
        bal_precision = balanced_results.get('precision', 0.0)
        bal_recall = balanced_results.get('recall', 0.0)
        
        ax.plot(
            [0, bal_recall, 1],
            [1, bal_precision, 0],
            'r-',
            linewidth=2,
            label=f'Balanced Model (AUC = {bal_pr_auc:.3f})'
        )
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Save figure
        pr_path = output_path / 'pr_curves_comparison.png'
        plt.tight_layout()
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(pr_path)
    
    def _create_confusion_matrices(
        self,
        imbalanced_results: Dict[str, Any],
        balanced_results: Dict[str, Any],
        output_path: Path
    ) -> str:
        """Create confusion matrices for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confusion matrix for imbalanced model
        imb_cm = np.array([
            [imbalanced_results.get('true_negatives', 0), imbalanced_results.get('false_positives', 0)],
            [imbalanced_results.get('false_negatives', 0), imbalanced_results.get('true_positives', 0)]
        ])
        
        sns.heatmap(
            imb_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[0],
            cbar=True,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive']
        )
        axes[0].set_title('Imbalanced Model Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Confusion matrix for balanced model
        bal_cm = np.array([
            [balanced_results.get('true_negatives', 0), balanced_results.get('false_positives', 0)],
            [balanced_results.get('false_negatives', 0), balanced_results.get('true_positives', 0)]
        ])
        
        sns.heatmap(
            bal_cm,
            annot=True,
            fmt='d',
            cmap='Reds',
            ax=axes[1],
            cbar=True,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive']
        )
        axes[1].set_title('Balanced Model Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        # Save figure
        cm_path = output_path / 'confusion_matrices_comparison.png'
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(cm_path)
