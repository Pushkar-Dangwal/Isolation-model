"""
DualEvaluationPipeline orchestrator for dual-evaluation pipeline.

This module provides the main orchestrator that coordinates the complete dual-evaluation
workflow, comparing fraud detection performance between imbalanced and balanced approaches.

Requirements: 13.1-13.7, 20.1
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import json

from .data_loader import DataLoader
from .imbalanced_pipeline import ImbalancedPipeline
from .balanced_pipeline import BalancedPipeline
from .metrics_comparator import MetricsComparator
from .data_models import EvaluationResult, ComparisonReport

logger = logging.getLogger(__name__)


class DualEvaluationPipeline:
    """
    Orchestrates complete dual-evaluation workflow.
    
    This class coordinates the execution of both imbalanced and balanced evaluation
    pipelines, aggregates results, generates comprehensive comparison reports, and
    saves all outputs and visualizations.
    
    Requirements: 13.1-13.7, 20.1
    """
    
    def __init__(
        self,
        data_path: str,
        pretrained_model_path: str,
        output_dir: str,
        random_state: int = 42
    ):
        """
        Initialize DualEvaluationPipeline.
        
        Args:
            data_path: Path to full dataset CSV file
            pretrained_model_path: Path to pretrained model directory or file
            output_dir: Directory where outputs will be saved
            random_state: Random seed for reproducibility
            
        Requirements: 20.1
        """
        self.data_path = data_path
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.random_state = random_state
        
        # Initialize component references
        self.data_loader = DataLoader(random_state=random_state)
        self.imbalanced_pipeline = ImbalancedPipeline(
            model_path=pretrained_model_path,
            random_state=random_state
        )
        self.balanced_pipeline = BalancedPipeline(
            random_state=random_state,
            n_jobs=-1
        )
        self.metrics_comparator = MetricsComparator()
        
        # Storage for comparison report
        self.comparison_report: Optional[ComparisonReport] = None
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DualEvaluationPipeline initialized")
        logger.info(f"  Data path: {data_path}")
        logger.info(f"  Model path: {pretrained_model_path}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Random state: {random_state}")

    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Execute complete dual evaluation workflow.
        
        This method orchestrates the entire evaluation process:
        1. Load full dataset
        2. Execute imbalanced pipeline evaluation
        3. Execute balanced pipeline evaluation
        4. Generate comparison report
        5. Save all outputs
        
        Returns:
            Dictionary containing comparison report data
            
        Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7
        """
        logger.info("=" * 80)
        logger.info("STARTING DUAL EVALUATION PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load full dataset
        logger.info("\n[STEP 1/5] Loading full dataset...")
        full_dataset = self.data_loader.load_full_dataset(self.data_path)
        logger.info(f"Loaded {len(full_dataset):,} transactions")
        logger.info(f"Fraud rate: {full_dataset['is_fraud'].mean():.4f}")
        
        # Step 2: Execute imbalanced pipeline evaluation
        logger.info("\n[STEP 2/5] Evaluating pretrained model on imbalanced data...")
        imbalanced_results = self._run_imbalanced_evaluation(full_dataset)
        logger.info("Imbalanced evaluation complete")
        logger.info(f"  Recall: {imbalanced_results.recall:.4f}")
        logger.info(f"  Precision: {imbalanced_results.precision:.4f}")
        logger.info(f"  F1-Score: {imbalanced_results.f1_score:.4f}")
        
        # Step 3: Execute balanced pipeline evaluation
        logger.info("\n[STEP 3/5] Training and evaluating on balanced data...")
        balanced_results = self._run_balanced_evaluation(full_dataset)
        logger.info("Balanced evaluation complete")
        logger.info(f"  Recall: {balanced_results.recall:.4f}")
        logger.info(f"  Precision: {balanced_results.precision:.4f}")
        logger.info(f"  F1-Score: {balanced_results.f1_score:.4f}")
        
        # Step 4: Generate comparison report
        logger.info("\n[STEP 4/5] Generating comparison report...")
        comparison_report = self._generate_comparison_report(
            imbalanced_results,
            balanced_results
        )
        logger.info("Comparison report generated")
        
        # Step 5: Save all outputs
        logger.info("\n[STEP 5/5] Saving outputs...")
        self._save_outputs(comparison_report)
        logger.info(f"All outputs saved to {self.output_dir}")
        
        logger.info("\n" + "=" * 80)
        logger.info("DUAL EVALUATION PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        # Store comparison report for later retrieval
        self.comparison_report = comparison_report
        
        # Return report as dictionary
        return self._comparison_report_to_dict(comparison_report)
    
    def _run_imbalanced_evaluation(
        self,
        full_dataset: pd.DataFrame
    ) -> EvaluationResult:
        """
        Execute imbalanced pipeline evaluation.
        
        Args:
            full_dataset: Full dataset with all transactions
            
        Returns:
            EvaluationResult from imbalanced pipeline
        """
        # Load pretrained model
        logger.info("Loading pretrained model...")
        self.imbalanced_pipeline.load_pretrained_model()
        
        # Time-based split (80/20)
        logger.info("Creating time-based train-test split...")
        train_df, test_df = self.data_loader.time_based_split(
            full_dataset,
            test_size=0.2
        )
        
        # Evaluate on test data
        logger.info("Evaluating pretrained model on test data...")
        result = self.imbalanced_pipeline.evaluate(test_df)
        
        # Update train statistics
        result.train_samples = len(train_df)
        result.train_fraud_rate = train_df['is_fraud'].mean()
        
        # Optimize threshold
        logger.info("Optimizing threshold for imbalanced data...")
        predictions = self.imbalanced_pipeline.model.predict(
            test_df,
            return_probabilities=True,
            return_risk_levels=False,
            return_explanations=False
        )
        y_proba = predictions['fraud_probability'].values
        
        optimal_threshold, optimized_result = self.imbalanced_pipeline.optimize_threshold(
            test_df,
            y_proba
        )
        
        # Update train statistics in optimized result
        optimized_result.train_samples = len(train_df)
        optimized_result.train_fraud_rate = train_df['is_fraud'].mean()
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        
        return optimized_result
    
    def _run_balanced_evaluation(
        self,
        full_dataset: pd.DataFrame
    ) -> EvaluationResult:
        """
        Execute balanced pipeline evaluation.
        
        Args:
            full_dataset: Full dataset with all transactions
            
        Returns:
            EvaluationResult from balanced pipeline
        """
        # Create balanced dataset
        logger.info("Creating balanced dataset...")
        balanced_dataset = self.data_loader.create_balanced_dataset(full_dataset)
        logger.info(f"Balanced dataset size: {len(balanced_dataset):,}")
        logger.info(f"Fraud rate: {balanced_dataset['is_fraud'].mean():.4f}")
        
        # Stratified split (80/20)
        logger.info("Creating stratified train-test split...")
        train_df, test_df = self.data_loader.stratified_split(
            balanced_dataset,
            test_size=0.2
        )
        
        # Train new model
        logger.info("Training new model on balanced data...")
        self.balanced_pipeline.train_model(train_df)
        
        # Evaluate on test data
        logger.info("Evaluating balanced model on test data...")
        result = self.balanced_pipeline.evaluate(test_df)
        
        # Update train statistics
        result.train_samples = len(train_df)
        result.train_fraud_rate = train_df['is_fraud'].mean()
        
        # Optimize threshold
        logger.info("Optimizing threshold for balanced data...")
        predictions = self.balanced_pipeline.model.predict(
            test_df,
            return_probabilities=True,
            return_risk_levels=False,
            return_explanations=False
        )
        y_proba = predictions['fraud_probability'].values
        
        optimal_threshold, optimized_result = self.balanced_pipeline.optimize_threshold(
            test_df,
            y_proba
        )
        
        # Update train statistics in optimized result
        optimized_result.train_samples = len(train_df)
        optimized_result.train_fraud_rate = train_df['is_fraud'].mean()
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        
        return optimized_result

    
    def _generate_comparison_report(
        self,
        imbalanced_results: EvaluationResult,
        balanced_results: EvaluationResult
    ) -> ComparisonReport:
        """
        Generate comprehensive comparison report.
        
        Args:
            imbalanced_results: Results from imbalanced pipeline
            balanced_results: Results from balanced pipeline
            
        Returns:
            ComparisonReport with complete analysis
        """
        # Convert EvaluationResult to dictionary for comparator
        imb_dict = self._evaluation_result_to_dict(imbalanced_results)
        bal_dict = self._evaluation_result_to_dict(balanced_results)
        
        # Generate comparison table
        comparison_table = self.metrics_comparator.compare_metrics(
            imb_dict,
            bal_dict
        )
        
        # Analyze trade-offs
        trade_offs = self.metrics_comparator.analyze_trade_offs(
            imb_dict,
            bal_dict
        )
        
        # Generate interpretation
        interpretation = self.metrics_comparator.generate_interpretation(
            comparison_table,
            trade_offs
        )
        
        # Create visualizations
        visualization_paths = self.metrics_comparator.create_visualizations(
            imb_dict,
            bal_dict,
            self.output_dir
        )
        
        # Calculate metric differences
        metric_differences = self._calculate_metric_differences(comparison_table)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trade_offs, comparison_table)
        
        # Create comparison report
        report = ComparisonReport(
            timestamp=datetime.now().isoformat(),
            imbalanced_results=imbalanced_results,
            balanced_results=balanced_results,
            comparison_table=comparison_table,
            metric_differences=metric_differences,
            trade_offs=trade_offs,
            interpretation=interpretation,
            recommendations=recommendations,
            visualization_paths=visualization_paths
        )
        
        return report

    
    def _save_outputs(self, report: ComparisonReport):
        """
        Save all outputs to output directory.
        
        Args:
            report: ComparisonReport to save
        """
        output_path = Path(self.output_dir)
        
        # Save comparison table as CSV
        table_path = output_path / 'comparison_table.csv'
        report.comparison_table.to_csv(table_path, index=False)
        logger.info(f"Saved comparison table to {table_path}")
        
        # Save full report as JSON
        report_path = output_path / 'comparison_report.json'
        report_dict = self._comparison_report_to_dict(report)
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        logger.info(f"Saved comparison report to {report_path}")
        
        # Save interpretation as text file
        interpretation_path = output_path / 'interpretation.txt'
        with open(interpretation_path, 'w') as f:
            f.write(report.interpretation)
        logger.info(f"Saved interpretation to {interpretation_path}")
        
        # Save recommendations as text file
        recommendations_path = output_path / 'recommendations.txt'
        with open(recommendations_path, 'w') as f:
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n\n")
        logger.info(f"Saved recommendations to {recommendations_path}")
        
        logger.info(f"Visualizations saved: {len(report.visualization_paths)} files")

    
    def get_comparison_report(self) -> Dict[str, Any]:
        """
        Get comprehensive comparison report.
        
        Returns:
            Dictionary containing comparison report data
            
        Raises:
            ValueError: If report doesn't exist (run_evaluation not called)
            
        Requirements: 13.1
        """
        if self.comparison_report is None:
            raise ValueError(
                "Comparison report not available. "
                "Call run_evaluation() first to generate the report."
            )
        
        return self._comparison_report_to_dict(self.comparison_report)
    
    def _evaluation_result_to_dict(self, result: EvaluationResult) -> Dict[str, Any]:
        """Convert EvaluationResult to dictionary."""
        return {
            'model_name': result.model_name,
            'dataset_type': result.dataset_type,
            'train_samples': result.train_samples,
            'test_samples': result.test_samples,
            'train_fraud_rate': result.train_fraud_rate,
            'test_fraud_rate': result.test_fraud_rate,
            'accuracy': result.accuracy,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'roc_auc': result.roc_auc,
            'pr_auc': result.pr_auc,
            'true_positives': result.true_positives,
            'true_negatives': result.true_negatives,
            'false_positives': result.false_positives,
            'false_negatives': result.false_negatives,
            'fraud_detection_rate': result.fraud_detection_rate,
            'false_positive_rate': result.false_positive_rate,
            'customer_friction_rate': result.customer_friction_rate,
            'optimal_threshold': result.optimal_threshold,
            'threshold_range': result.threshold_range,
            'training_time': result.training_time,
            'evaluation_time': result.evaluation_time
        }

    
    def _comparison_report_to_dict(self, report: ComparisonReport) -> Dict[str, Any]:
        """Convert ComparisonReport to dictionary."""
        return {
            'timestamp': report.timestamp,
            'imbalanced_results': self._evaluation_result_to_dict(report.imbalanced_results),
            'balanced_results': self._evaluation_result_to_dict(report.balanced_results),
            'comparison_table': report.comparison_table.to_dict(orient='records'),
            'metric_differences': report.metric_differences,
            'trade_offs': report.trade_offs,
            'interpretation': report.interpretation,
            'recommendations': report.recommendations,
            'visualization_paths': report.visualization_paths
        }
    
    def _calculate_metric_differences(
        self,
        comparison_table: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate metric differences from comparison table."""
        differences = {}
        
        for _, row in comparison_table.iterrows():
            metric_name = row['metric_name']
            difference = row['difference']
            differences[metric_name] = difference
        
        return differences

    
    def _generate_recommendations(
        self,
        trade_offs: Dict[str, Any],
        comparison_table: pd.DataFrame
    ) -> list:
        """
        Generate actionable recommendations based on trade-off analysis.
        
        Args:
            trade_offs: Trade-off analysis results
            comparison_table: Comparison table with metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Recommendation based on overall performance
        if trade_offs.get('f1_improvement'):
            recommendations.append(
                "The balanced approach shows better overall performance (higher F1-score). "
                "Consider using the balanced model if training time and data reduction are acceptable."
            )
        else:
            recommendations.append(
                "The imbalanced approach maintains better overall performance (higher F1-score). "
                "Consider keeping the pretrained model unless specific metrics need improvement."
            )
        
        # Recommendation based on precision vs recall
        if trade_offs.get('precision_improvement') and trade_offs.get('recall_improvement'):
            recommendations.append(
                "The balanced approach achieves both higher precision and recall - a clear win. "
                "This suggests the balanced dataset helps the model learn better decision boundaries."
            )
        elif trade_offs.get('recall_improvement'):
            recommendations.append(
                "The balanced approach catches more fraud cases (higher recall) but with more false alarms. "
                "Use this if fraud detection is the top priority and customer friction is acceptable."
            )
        elif trade_offs.get('precision_improvement'):
            recommendations.append(
                "The balanced approach reduces false alarms (higher precision) but misses more fraud. "
                "Use this if customer experience is the top priority and some fraud loss is acceptable."
            )
        
        # Recommendation based on customer friction
        if trade_offs.get('lower_customer_friction'):
            recommendations.append(
                "The balanced approach reduces customer friction (lower false positive rate). "
                "This can improve customer satisfaction and reduce operational costs from false alerts."
            )
        else:
            recommendations.append(
                "The imbalanced approach has lower customer friction (lower false positive rate). "
                "This is important for maintaining good customer experience."
            )
        
        # Recommendation based on training efficiency
        reduction_pct = trade_offs.get('training_data_reduction', 0) * 100
        if reduction_pct > 50:
            recommendations.append(
                f"The balanced approach uses {reduction_pct:.1f}% less training data, "
                f"significantly reducing training time and computational costs. "
                f"This makes model retraining more feasible for frequent updates."
            )
        
        return recommendations
