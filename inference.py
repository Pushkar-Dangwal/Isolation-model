#!/usr/bin/env python3
"""
Inference script for the fraud detection system.

This script implements real-time scoring pipeline for new transactions with:
1. Model loading and validation
2. Real-time transaction processing
3. Batch processing capabilities
4. Output formatting for downstream systems
5. Performance monitoring and error handling

Requirements: 8.4 - Implement efficient feature computation for real-time scoring
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union

import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from fraud_detector import FraudDetector
from config import setup_logging, MODELS_DIR, LOGS_DIR
from error_handling import DataValidationError, PipelineError


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run fraud detection inference on transaction data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        '--model-path', 
        type=str, 
        required=True,
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--model-version', 
        type=str, 
        default=None,
        help='Specific model version to load (loads latest if not specified)'
    )
    
    # Input arguments
    parser.add_argument(
        '--input-data', 
        type=str, 
        default=None,
        help='Path to input CSV file for batch processing'
    )
    parser.add_argument(
        '--input-json', 
        type=str, 
        default=None,
        help='Path to input JSON file with transaction data'
    )
    parser.add_argument(
        '--single-transaction', 
        type=str, 
        default=None,
        help='JSON string with single transaction data'
    )
    
    # Processing arguments
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1000,
        help='Batch size for processing large datasets'
    )
    parser.add_argument(
        '--real-time', 
        action='store_true',
        help='Enable real-time processing mode with optimizations'
    )
    parser.add_argument(
        '--enable-explanations', 
        action='store_true',
        help='Include prediction explanations in output'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-file', 
        type=str, 
        default=None,
        help='Path to save prediction results (CSV format)'
    )
    parser.add_argument(
        '--output-json', 
        type=str, 
        default=None,
        help='Path to save prediction results (JSON format)'
    )
    parser.add_argument(
        '--output-format', 
        type=str, 
        choices=['csv', 'json', 'both'],
        default='csv',
        help='Output format for results'
    )
    
    # Performance arguments
    parser.add_argument(
        '--benchmark', 
        action='store_true',
        help='Run performance benchmarking'
    )
    parser.add_argument(
        '--profile-memory', 
        action='store_true',
        help='Profile memory usage during inference'
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


def load_model(model_path: str, model_version: Optional[str] = None) -> FraudDetector:
    """
    Load the trained fraud detection model.
    
    Args:
        model_path: Path to the model file
        model_version: Optional specific version to load
        
    Returns:
        Loaded FraudDetector instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        IOError: If model loading fails
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading fraud detection model from {model_path}")
    if model_version:
        logger.info(f"Requesting specific version: {model_version}")
    
    start_time = time.time()
    
    try:
        # Initialize detector and load model
        detector = FraudDetector()
        detector.load_model(
            filepath=model_path,
            version=model_version,
            verify_integrity=True,
            load_pipeline=True
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Log model information
        model_info = detector.get_model_info()
        logger.info(f"Model details:")
        logger.info(f"  - Status: {model_info['status']}")
        logger.info(f"  - Version: {model_info['model_version']}")
        logger.info(f"  - Features: {model_info['feature_count']}")
        logger.info(f"  - Error handling: {model_info.get('error_handling_enabled', 'N/A')}")
        
        # Check system health
        health = detector.get_system_health()
        logger.info(f"  - System health: {health['overall_status']}")
        
        if health['overall_status'] != 'healthy':
            logger.warning("Model system health is not optimal")
            for component, status in health['components_status'].items():
                if status != 'fitted':
                    logger.warning(f"  - {component}: {status}")
        
        return detector
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise IOError(f"Model loading failed: {str(e)}") from e


def load_input_data(input_data: Optional[str] = None, 
                   input_json: Optional[str] = None,
                   single_transaction: Optional[str] = None) -> pd.DataFrame:
    """
    Load input data from various sources.
    
    Args:
        input_data: Path to CSV file
        input_json: Path to JSON file
        single_transaction: JSON string with single transaction
        
    Returns:
        DataFrame with transaction data
        
    Raises:
        ValueError: If no input source provided or invalid data
        FileNotFoundError: If input file doesn't exist
    """
    logger = logging.getLogger(__name__)
    
    if single_transaction:
        logger.info("Processing single transaction from command line")
        try:
            transaction_data = json.loads(single_transaction)
            df = pd.DataFrame([transaction_data])
            logger.info(f"Loaded single transaction: {df.iloc[0].to_dict()}")
            return df
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in single transaction: {e}")
    
    elif input_json:
        logger.info(f"Loading transaction data from JSON file: {input_json}")
        if not Path(input_json).exists():
            raise FileNotFoundError(f"JSON input file not found: {input_json}")
        
        try:
            with open(input_json, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'transactions' in data:
                    df = pd.DataFrame(data['transactions'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            logger.info(f"Loaded {len(df)} transactions from JSON")
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load JSON data: {e}")
    
    elif input_data:
        logger.info(f"Loading transaction data from CSV file: {input_data}")
        if not Path(input_data).exists():
            raise FileNotFoundError(f"CSV input file not found: {input_data}")
        
        try:
            # Load with efficient dtypes for inference
            dtype_mapping = {
                'transaction_id': 'string',
                'sender_account': 'string', 
                'receiver_account': 'string',
                'transaction_type': 'category',
                'merchant_category': 'category',
                'location': 'category',
                'device_used': 'category',
                'amount': 'float32'
            }
            
            df = pd.read_csv(input_data, dtype=dtype_mapping)
            
            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            logger.info(f"Loaded {len(df)} transactions from CSV")
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV data: {e}")
    
    else:
        raise ValueError("No input data source provided. Use --input-data, --input-json, or --single-transaction")


def process_batch(detector: FraudDetector, df: pd.DataFrame, 
                 enable_explanations: bool = False) -> pd.DataFrame:
    """
    Process a batch of transactions.
    
    Args:
        detector: Loaded FraudDetector
        df: DataFrame with transactions
        enable_explanations: Whether to include explanations
        
    Returns:
        DataFrame with predictions
    """
    logger = logging.getLogger(__name__)
    
    logger.debug(f"Processing batch of {len(df)} transactions")
    
    try:
        # Generate predictions
        results = detector.predict(
            df,
            transaction_id_column='transaction_id',
            return_probabilities=True,
            return_risk_levels=True,
            return_explanations=enable_explanations
        )
        
        logger.debug(f"Generated predictions for {len(results)} transactions")
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise PipelineError(f"Batch processing failed: {str(e)}") from e


def process_real_time(detector: FraudDetector, df: pd.DataFrame, 
                     enable_explanations: bool = False) -> List[Dict[str, Any]]:
    """
    Process transactions in real-time mode (one by one).
    
    Args:
        detector: Loaded FraudDetector
        df: DataFrame with transactions
        enable_explanations: Whether to include explanations
        
    Returns:
        List of prediction dictionaries
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing {len(df)} transactions in real-time mode")
    
    results = []
    processing_times = []
    
    for idx, row in df.iterrows():
        start_time = time.time()
        
        try:
            # Convert row to transaction dictionary
            transaction = row.to_dict()
            transaction_id = transaction.get('transaction_id', f'tx_{idx}')
            
            # Process single transaction
            result = detector.predict_single(
                transaction=transaction,
                transaction_id=transaction_id
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Add processing metadata
            result['processing_time_ms'] = processing_time * 1000
            result['processed_at'] = datetime.now().isoformat()
            
            results.append(result)
            
            if idx % 100 == 0 and idx > 0:
                avg_time = np.mean(processing_times[-100:])
                logger.info(f"Processed {idx+1} transactions, avg time: {avg_time*1000:.2f}ms")
        
        except Exception as e:
            logger.error(f"Failed to process transaction {idx}: {str(e)}")
            # Create error result
            error_result = {
                'transaction_id': transaction.get('transaction_id', f'tx_{idx}'),
                'fraud_probability': 0.5,  # Default fallback
                'risk_level': 'medium',
                'fraud_prediction': 0,
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000,
                'processed_at': datetime.now().isoformat()
            }
            results.append(error_result)
    
    # Log performance statistics
    if processing_times:
        avg_time = np.mean(processing_times)
        median_time = np.median(processing_times)
        p95_time = np.percentile(processing_times, 95)
        
        logger.info(f"Real-time processing completed:")
        logger.info(f"  - Average time: {avg_time*1000:.2f}ms")
        logger.info(f"  - Median time: {median_time*1000:.2f}ms")
        logger.info(f"  - 95th percentile: {p95_time*1000:.2f}ms")
        logger.info(f"  - Throughput: {1/avg_time:.1f} transactions/second")
    
    return results


def run_benchmark(detector: FraudDetector, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run performance benchmarking.
    
    Args:
        detector: Loaded FraudDetector
        df: DataFrame with test transactions
        
    Returns:
        Benchmark results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Running performance benchmark...")
    
    # Test different batch sizes
    batch_sizes = [1, 10, 100, 1000]
    benchmark_results = {
        'batch_performance': {},
        'memory_usage': {},
        'system_info': {
            'total_transactions': len(df),
            'benchmark_timestamp': datetime.now().isoformat()
        }
    }
    
    for batch_size in batch_sizes:
        if batch_size > len(df):
            continue
            
        logger.info(f"Testing batch size: {batch_size}")
        
        # Sample data for this batch size
        sample_df = df.head(batch_size).copy()
        
        # Measure processing time
        start_time = time.time()
        
        try:
            results = detector.predict(sample_df, return_probabilities=True)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            throughput = len(sample_df) / processing_time
            avg_time_per_transaction = processing_time / len(sample_df)
            
            benchmark_results['batch_performance'][batch_size] = {
                'processing_time_seconds': processing_time,
                'throughput_per_second': throughput,
                'avg_time_per_transaction_ms': avg_time_per_transaction * 1000,
                'success': True
            }
            
            logger.info(f"  - Processing time: {processing_time:.3f}s")
            logger.info(f"  - Throughput: {throughput:.1f} tx/s")
            logger.info(f"  - Avg per transaction: {avg_time_per_transaction*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"Benchmark failed for batch size {batch_size}: {e}")
            benchmark_results['batch_performance'][batch_size] = {
                'error': str(e),
                'success': False
            }
    
    # Memory usage estimation
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        benchmark_results['memory_usage'] = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'memory_percent': process.memory_percent()
        }
        
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB RSS")
        
    except ImportError:
        logger.warning("psutil not available for memory profiling")
        benchmark_results['memory_usage'] = {'error': 'psutil not available'}
    
    return benchmark_results


def save_results(results: Union[pd.DataFrame, List[Dict]], 
                output_file: Optional[str] = None,
                output_json: Optional[str] = None,
                output_format: str = 'csv') -> None:
    """
    Save prediction results to file(s).
    
    Args:
        results: Prediction results
        output_file: CSV output path
        output_json: JSON output path
        output_format: Output format preference
    """
    logger = logging.getLogger(__name__)
    
    # Convert to DataFrame if needed
    if isinstance(results, list):
        df_results = pd.DataFrame(results)
    else:
        df_results = results
    
    # Save CSV
    if output_format in ['csv', 'both'] and output_file:
        logger.info(f"Saving results to CSV: {output_file}")
        df_results.to_csv(output_file, index=False)
    
    # Save JSON
    if output_format in ['json', 'both'] and output_json:
        logger.info(f"Saving results to JSON: {output_json}")
        
        # Convert DataFrame to JSON-serializable format
        json_data = df_results.to_dict('records')
        
        # Handle numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # Convert all numpy types
        for record in json_data:
            for key, value in record.items():
                record[key] = convert_numpy_types(value)
        
        with open(output_json, 'w') as f:
            json.dump({
                'predictions': json_data,
                'metadata': {
                    'total_predictions': len(json_data),
                    'generated_at': datetime.now().isoformat(),
                    'format_version': '1.0'
                }
            }, f, indent=2, default=str)
    
    logger.info(f"Results saved successfully ({len(df_results)} predictions)")


def main():
    """Main inference script execution."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("FRAUD DETECTION INFERENCE")
    logger.info("=" * 80)
    logger.info(f"Inference started at: {datetime.now()}")
    logger.info(f"Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Step 1: Load model
        logger.info("\n" + "="*50)
        logger.info("STEP 1: MODEL LOADING")
        logger.info("="*50)
        
        detector = load_model(args.model_path, args.model_version)
        
        # Step 2: Load input data
        logger.info("\n" + "="*50)
        logger.info("STEP 2: INPUT DATA LOADING")
        logger.info("="*50)
        
        df = load_input_data(args.input_data, args.input_json, args.single_transaction)
        
        # Validate input data against model expectations
        logger.info("Validating input data compatibility...")
        validation_results = detector.validate_inference_data(df)
        
        if not validation_results.get('fraud_detection_validation', True):
            logger.warning("Input data validation issues detected:")
            for issue in validation_results.get('data_quality_issues', []):
                logger.warning(f"  - {issue}")
            for rec in validation_results.get('recommendations', []):
                logger.info(f"  - Recommendation: {rec}")
        
        # Step 3: Run inference
        logger.info("\n" + "="*50)
        logger.info("STEP 3: INFERENCE PROCESSING")
        logger.info("="*50)
        
        start_time = time.time()
        
        if args.real_time:
            logger.info("Running in real-time processing mode")
            results = process_real_time(detector, df, args.enable_explanations)
            results_df = pd.DataFrame(results)
        else:
            logger.info("Running in batch processing mode")
            if len(df) > args.batch_size:
                logger.info(f"Processing in batches of {args.batch_size}")
                batch_results = []
                
                for i in range(0, len(df), args.batch_size):
                    batch_df = df.iloc[i:i+args.batch_size]
                    logger.info(f"Processing batch {i//args.batch_size + 1}: {len(batch_df)} transactions")
                    
                    batch_result = process_batch(detector, batch_df, args.enable_explanations)
                    batch_results.append(batch_result)
                
                results_df = pd.concat(batch_results, ignore_index=True)
            else:
                results_df = process_batch(detector, df, args.enable_explanations)
        
        total_time = time.time() - start_time
        
        logger.info(f"Inference completed in {total_time:.2f} seconds")
        logger.info(f"Processed {len(results_df)} transactions")
        logger.info(f"Average processing time: {total_time/len(results_df)*1000:.2f}ms per transaction")
        
        # Log prediction summary
        if 'fraud_probability' in results_df.columns:
            avg_fraud_prob = results_df['fraud_probability'].mean()
            logger.info(f"Average fraud probability: {avg_fraud_prob:.3f}")
        
        if 'risk_level' in results_df.columns:
            risk_distribution = results_df['risk_level'].value_counts()
            logger.info("Risk level distribution:")
            for level, count in risk_distribution.items():
                logger.info(f"  {level}: {count} ({count/len(results_df)*100:.1f}%)")
        
        # Step 4: Run benchmark if requested
        if args.benchmark:
            logger.info("\n" + "="*50)
            logger.info("STEP 4: PERFORMANCE BENCHMARKING")
            logger.info("="*50)
            
            benchmark_results = run_benchmark(detector, df.head(1000))  # Limit for benchmarking
            
            # Save benchmark results
            benchmark_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(benchmark_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            logger.info(f"Benchmark results saved to: {benchmark_file}")
        
        # Step 5: Save results
        logger.info("\n" + "="*50)
        logger.info("STEP 5: SAVING RESULTS")
        logger.info("="*50)
        
        # Generate output filenames if not provided
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not args.output_file and args.output_format in ['csv', 'both']:
            args.output_file = f"fraud_predictions_{timestamp}.csv"
        
        if not args.output_json and args.output_format in ['json', 'both']:
            args.output_json = f"fraud_predictions_{timestamp}.json"
        
        save_results(results_df, args.output_file, args.output_json, args.output_format)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Processed {len(results_df)} transactions")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Throughput: {len(results_df)/total_time:.1f} transactions/second")
        
        if args.output_file:
            logger.info(f"Results saved to: {args.output_file}")
        if args.output_json:
            logger.info(f"Results saved to: {args.output_json}")
        
        logger.info(f"Inference completed at: {datetime.now()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed with error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)