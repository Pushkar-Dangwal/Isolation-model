#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / 'src'))

from fraud_detector import FraudDetector
from model_evaluator import ModelEvaluator
from config import setup_logging, DATA_DIR, MODELS_DIR, MODEL_CONFIG
from error_handling import DataValidationError, PipelineError


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Train the fraud detection model on transaction data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=str(DATA_DIR / 'financial.csv'),
        help='Path to the training data CSV file'
    )
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=None,
        help='Number of samples to use for training (use all if not specified)'
    )
    parser.add_argument(
        '--validation-split', 
        type=float, 
        default=0.2,
        help='Fraction of data to use for validation'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name', 
        type=str, 
        default=f'fraud_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Name for the trained model'
    )
    parser.add_argument(
        '--random-seed', 
        type=int, 
        default=MODEL_CONFIG['random_seed'],
        help='Random seed for reproducible training'
    )
    parser.add_argument(
        '--enable-reproducibility', 
        action='store_true',
        help='Enable strict reproducibility controls'
    )
    parser.add_argument(
        '--strict-determinism', 
        action='store_true',
        help='Enable strict deterministic mode (slower but fully reproducible)'
    )
    
    # Training arguments
    parser.add_argument(
        '--optimize-thresholds', 
        action='store_true', 
        default=True,
        help='Perform threshold optimization during training'
    )
    parser.add_argument(
        '--enable-error-handling', 
        action='store_true', 
        default=True,
        help='Enable comprehensive error handling'
    )
    parser.add_argument(
        '--n-jobs', 
        type=int, 
        default=-1,
        help='Number of parallel jobs (-1 for all cores)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=str(MODELS_DIR),
        help='Directory to save the trained model'
    )
    parser.add_argument(
        '--save-evaluation', 
        action='store_true', 
        default=True,
        help='Save detailed evaluation results'
    )
    parser.add_argument(
        '--save-plots', 
        action='store_true',
        help='Save evaluation plots and visualizations'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def load_and_validate_data(data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load and validate training data.
    
    Args:
        data_path: Path to the CSV data file
        sample_size: Optional number of samples to load
        
    Returns:
        Validated DataFrame
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        DataValidationError: If data validation fails
    """
    logger = logging.getLogger(__name__)
    
    # Check if file exists
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    logger.info(f"Loading training data from {data_path}")
    start_time = time.time()
    
    try:
        # Load data with efficient dtypes
        dtype_mapping = {
            'transaction_id': 'string',
            'sender_account': 'string', 
            'receiver_account': 'string',
            'transaction_type': 'category',
            'merchant_category': 'category',
            'location': 'category',
            'device_used': 'category',
            'amount': 'float32',
            'is_fraud': 'int8'
        }
        
        # Load data in chunks if it's large
        if sample_size:
            df = pd.read_csv(data_path, dtype=dtype_mapping, nrows=sample_size)
            logger.info(f"Loaded sample of {len(df)} transactions")
        else:
            # For large files, load in chunks and combine
            chunk_size = 100000
            chunks = []
            total_rows = 0
            
            for chunk in pd.read_csv(data_path, dtype=dtype_mapping, chunksize=chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows % 500000 == 0:
                    logger.info(f"Loaded {total_rows} transactions...")
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded complete dataset: {len(df)} transactions")
        
        load_time = time.time() - start_time
        logger.info(f"Data loading completed in {load_time:.2f} seconds")
        
        # Basic data validation
        required_columns = [
            'transaction_id', 'timestamp', 'sender_account', 'receiver_account',
            'amount', 'transaction_type', 'merchant_category', 'location', 
            'device_used', 'is_fraud'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Convert timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Basic data quality checks
        logger.info("Performing data quality validation...")
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            logger.warning(f"Found {empty_rows} completely empty rows - will be removed")
            df = df.dropna(how='all')
        
        # Check fraud rate
        fraud_rate = df['is_fraud'].mean()
        logger.info(f"Dataset fraud rate: {fraud_rate:.1%}")
        
        if fraud_rate < 0.001 or fraud_rate > 0.5:
            logger.warning(f"Unusual fraud rate detected: {fraud_rate:.1%}")
        
        # Check for duplicate transaction IDs
        duplicate_ids = df['transaction_id'].duplicated().sum()
        if duplicate_ids > 0:
            logger.warning(f"Found {duplicate_ids} duplicate transaction IDs")
            df = df.drop_duplicates(subset=['transaction_id'], keep='first')
        
        # Memory usage optimization
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Dataset memory usage: {memory_usage:.1f} MB")
        
        logger.info("Data validation completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load training data: {str(e)}")
        raise DataValidationError(f"Data loading failed: {str(e)}") from e


def train_fraud_detector(df: pd.DataFrame, config: Dict[str, Any]) -> FraudDetector:
    """
    Train the fraud detection model.
    
    Args:
        df: Training DataFrame
        config: Training configuration
        
    Returns:
        Trained FraudDetector instance
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing fraud detection pipeline...")
    
    # Initialize detector with configuration
    detector = FraudDetector(
        random_state=config['random_seed'],
        n_jobs=config['n_jobs'],
        verbose=config['verbose'],
        enable_error_handling=config['enable_error_handling'],
        ensure_reproducibility=config['enable_reproducibility'],
        strict_determinism=config['strict_determinism']
    )
    
    logger.info(f"Training fraud detector on {len(df)} transactions...")
    logger.info(f"Configuration: reproducibility={config['enable_reproducibility']}, "
               f"error_handling={config['enable_error_handling']}, "
               f"n_jobs={config['n_jobs']}")
    
    # Start training
    training_start = time.time()
    
    try:
        detector.fit(
            df=df,
            target_column='is_fraud',
            transaction_id_column='transaction_id',
            validation_split=config['validation_split'],
            optimize_thresholds=config['optimize_thresholds']
        )
        
        training_time = time.time() - training_start
        logger.info(f"Training completed successfully in {training_time:.2f} seconds")
        
        # Log training summary
        model_info = detector.get_model_info()
        logger.info(f"Model summary:")
        logger.info(f"  - Features: {model_info.get('feature_count', 'N/A')}")
        logger.info(f"  - Training samples: {model_info.get('training_metadata', {}).get('training_samples', 'N/A')}")
        logger.info(f"  - Validation samples: {model_info.get('training_metadata', {}).get('validation_samples', 'N/A')}")
        
        fraud_rate = model_info.get('training_metadata', {}).get('fraud_rate_total', None)
        if fraud_rate is not None:
            logger.info(f"  - Fraud rate: {fraud_rate:.1%}")
        else:
            logger.info(f"  - Fraud rate: N/A")
        
        if 'performance_metrics' in model_info:
            metrics = model_info['performance_metrics']
            logger.info(f"  - PR-AUC: {metrics.get('pr_auc', 'N/A'):.3f}")
            logger.info(f"  - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.3f}")
        
        return detector
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise PipelineError(f"Training failed: {str(e)}") from e


def evaluate_model(detector: FraudDetector, df: pd.DataFrame, 
                  config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the trained model comprehensively using time-based split to prevent data leakage.
    
    Args:
        detector: Trained FraudDetector
        df: Evaluation DataFrame (full dataset)
        config: Evaluation configuration
        
    Returns:
        Comprehensive evaluation results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting comprehensive model evaluation with time-based split...")
    
    # CRITICAL FIX: Use time-based split to prevent data leakage
    # Sort by timestamp to simulate real-world scenario
    if 'timestamp' not in df.columns:
        logger.error("Timestamp column required for proper evaluation")
        raise ValueError("Timestamp column required for time-based split")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Time-based split: first 80% for train, last 20% for test
    # This simulates training on past data, evaluating on future data
    split_idx = int(len(df_sorted) * (1 - config['validation_split']))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    y_test = test_df['is_fraud'].values
    
    logger.info(f"Time-based evaluation split:")
    logger.info(f"  Training period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    logger.info(f"  Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    logger.info(f"  Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logger.info(f"  Test fraud rate: {np.mean(y_test):.1%}")
    
    # CRITICAL: Generate predictions ONLY on test data
    # Features will be computed using only test data (no data leakage)
    logger.info("Generating predictions on test data only (no data leakage)...")
    predictions = detector.predict(
        test_df, 
        transaction_id_column='transaction_id',
        return_probabilities=True, 
        return_risk_levels=True
    )
    
    y_proba = predictions['fraud_probability'].values
    y_pred = predictions['fraud_prediction'].values
    
    # Calculate comprehensive metrics on test subset only
    from sklearn.metrics import (
        precision_recall_curve, roc_curve, auc, precision_score,
        recall_score, f1_score, confusion_matrix
    )
    
    # Basic metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Curve metrics
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    # Risk level analysis
    risk_levels = predictions['risk_level'].values
    risk_distribution = pd.Series(risk_levels).value_counts().to_dict()
    
    # Create evaluation results matching the expected format
    evaluation_results = {
        'overall_metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'pr_auc': float(pr_auc),
            'roc_auc': float(roc_auc),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'risk_analysis': {
            'risk_distribution': risk_distribution
        },
        'business_metrics': {
            'fraud_detection_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'customer_friction_rate': float((tp + fp) / len(y_test))
        }
    }
    
    # Initialize model evaluator for additional analysis
    evaluator = ModelEvaluator(save_plots=config.get('save_plots', False))
    
    # Generate comprehensive performance report
    performance_report = evaluator.generate_performance_report(
        y_test, y_proba,
        current_threshold=0.5,
        model_name=config['model_name'],
        include_threshold_analysis=True,
        include_business_metrics=True
    )
    
    # Combine results
    comprehensive_results = {
        'pipeline_evaluation': evaluation_results,
        'performance_report': performance_report,
        'evaluation_metadata': {
            'test_samples': len(test_df),
            'test_fraud_rate': float(np.mean(y_test)),
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_name': config['model_name'],
            'evaluation_method': 'time_based_split_no_leakage',
            'train_period': f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()}",
            'test_period': f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()}"
        }
    }
    
    # Log key metrics
    overall_metrics = evaluation_results['overall_metrics']
    logger.info("Evaluation Results:")
    logger.info(f"  - Precision: {overall_metrics['precision']:.3f}")
    logger.info(f"  - Recall: {overall_metrics['recall']:.3f}")
    logger.info(f"  - F1-Score: {overall_metrics['f1_score']:.3f}")
    logger.info(f"  - PR-AUC: {overall_metrics['pr_auc']:.3f}")
    logger.info(f"  - ROC-AUC: {overall_metrics['roc_auc']:.3f}")
    logger.info(f"  - False Positive Rate: {evaluation_results['business_metrics']['false_positive_rate']:.3f}")
    
    # Log business impact
    business_metrics = evaluation_results['business_metrics']
    logger.info("Business Impact:")
    logger.info(f"  - Fraud Detection Rate: {business_metrics['fraud_detection_rate']:.1%}")
    logger.info(f"  - Customer Friction Rate: {business_metrics['customer_friction_rate']:.1%}")
    
    return comprehensive_results


def save_model_and_results(detector: FraudDetector, evaluation_results: Dict[str, Any], 
                          config: Dict[str, Any]) -> str:
    """
    Save the trained model and evaluation results.
    
    Args:
        detector: Trained FraudDetector
        evaluation_results: Evaluation results
        config: Configuration
        
    Returns:
        Path to saved model
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / config['model_name']
    logger.info(f"Saving trained model to {model_path}")
    
    saved_path = detector.save_model(
        filepath=str(model_path),
        version=f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        include_metadata=True,
        save_pipeline=True,
        training_data=None  # Don't save training data to reduce size
    )
    
    # Save evaluation results if requested
    if config.get('save_evaluation', True):
        eval_path = output_dir / f"{config['model_name']}_evaluation.json"
        logger.info(f"Saving evaluation results to {eval_path}")
        
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(evaluation_results)
        
        with open(eval_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    # Save training configuration
    config_path = output_dir / f"{config['model_name']}_config.json"
    logger.info(f"Saving training configuration to {config_path}")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Model and results saved successfully")
    logger.info(f"Model path: {saved_path}")
    
    return saved_path


def main():
    """Main training script execution."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Training started at: {datetime.now()}")
    logger.info(f"Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # Convert args to config dict
    config = vars(args)
    
    try:
        # Step 1: Load and validate data
        logger.info("\n" + "="*50)
        logger.info("STEP 1: DATA LOADING AND VALIDATION")
        logger.info("="*50)
        
        df = load_and_validate_data(args.data_path, args.sample_size)
        
        # Step 2: Train model
        logger.info("\n" + "="*50)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*50)
        
        detector = train_fraud_detector(df, config)
        
        # Step 3: Evaluate model
        logger.info("\n" + "="*50)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*50)
        
        # Check if detector is properly fitted before evaluation
        try:
            # Try to get model info to check if training was successful
            model_info = detector.get_model_info()
            if model_info.get('is_fitted', False):
                evaluation_results = evaluate_model(detector, df, config)
            else:
                logger.warning("Model training failed or used fallback mechanism. Skipping detailed evaluation.")
                evaluation_results = {
                    'pipeline_evaluation': {'overall_metrics': {'precision': 0, 'recall': 0, 'f1_score': 0, 'pr_auc': 0, 'roc_auc': 0}},
                    'business_metrics': {'fraud_detection_rate': 0, 'false_positive_rate': 0, 'customer_friction_rate': 0},
                    'evaluation_metadata': {
                        'test_samples': 0,
                        'test_fraud_rate': 0,
                        'evaluation_timestamp': datetime.now().isoformat(),
                        'model_name': config['model_name'],
                        'evaluation_skipped': True,
                        'reason': 'Training failed or insufficient data'
                    }
                }
        except Exception as e:
            logger.warning(f"Model evaluation failed: {str(e)}. Using fallback evaluation.")
            evaluation_results = {
                'pipeline_evaluation': {'overall_metrics': {'precision': 0, 'recall': 0, 'f1_score': 0, 'pr_auc': 0, 'roc_auc': 0}},
                'business_metrics': {'fraud_detection_rate': 0, 'false_positive_rate': 0, 'customer_friction_rate': 0},
                'evaluation_metadata': {
                    'test_samples': 0,
                    'test_fraud_rate': 0,
                    'evaluation_timestamp': datetime.now().isoformat(),
                    'model_name': config['model_name'],
                    'evaluation_skipped': True,
                    'reason': f'Evaluation error: {str(e)}'
                }
            }
        
        # Step 4: Save model and results
        logger.info("\n" + "="*50)
        logger.info("STEP 4: SAVING MODEL AND RESULTS")
        logger.info("="*50)
        
        # Check if model is fitted before saving
        try:
            model_info = detector.get_model_info()
            if model_info.get('is_fitted', False):
                saved_path = save_model_and_results(detector, evaluation_results, config)
            else:
                logger.warning("Model is not properly fitted. Skipping model saving, but saving evaluation results.")
                # Save only evaluation results
                output_dir = Path(config['output_dir'])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if config.get('save_evaluation', True):
                    eval_path = output_dir / f"{config['model_name']}_evaluation.json"
                    logger.info(f"Saving evaluation results to {eval_path}")
                    
                    import json
                    with open(eval_path, 'w') as f:
                        json.dump(evaluation_results, f, indent=2, default=str)
                
                saved_path = str(output_dir / config['model_name'])
                logger.info(f"Evaluation results saved to {saved_path}_evaluation.json")
        except Exception as e:
            logger.error(f"Failed to save model or results: {str(e)}")
            saved_path = "Not saved due to errors"
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Model saved to: {saved_path}")
        logger.info(f"Training completed at: {datetime.now()}")
        
        # Print key results
        overall_metrics = evaluation_results['pipeline_evaluation']['overall_metrics']
        logger.info("\nFinal Model Performance:")
        logger.info(f"  PR-AUC: {overall_metrics['pr_auc']:.3f}")
        logger.info(f"  F1-Score: {overall_metrics['f1_score']:.3f}")
        logger.info(f"  Precision: {overall_metrics['precision']:.3f}")
        logger.info(f"  Recall: {overall_metrics['recall']:.3f}")
        
        # Check if business_metrics exists
        if 'business_metrics' in evaluation_results:
            business_metrics = evaluation_results['business_metrics']
            logger.info(f"  Fraud Detection Rate: {business_metrics.get('fraud_detection_rate', 0):.1%}")
            logger.info(f"  False Positive Rate: {business_metrics.get('false_positive_rate', 0):.3f}")
        else:
            logger.info("  Business metrics: Not available (training used fallback)")
        
        logger.info("\nModel training completed!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)