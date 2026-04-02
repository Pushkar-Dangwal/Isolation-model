#!/usr/bin/env python3
"""
Retrain the model with the fixed feature engineering pipeline.
This will ensure training and evaluation use the same feature generation logic.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import logging
from datetime import datetime
from config import setup_logging

# Set up logging
setup_logging('INFO')
logger = logging.getLogger(__name__)

def main():
    """Retrain model with fixed pipeline."""
    
    logger.info("="*80)
    logger.info("RETRAINING MODEL WITH FIXED FEATURE ENGINEERING")
    logger.info("="*80)
    
    # Import train_model main function
    from train_model import main as train_main
    
    # Override sys.argv to pass arguments
    sys.argv = [
        'train_model.py',
        '--data-path', 'data/financial.csv',
        '--sample-size', '500000',  # Use 500K samples for faster training
        '--model-name', f'fraud_detector_fixed_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        '--validation-split', '0.2',
        '--random-seed', '42',
        '--optimize-thresholds',
        '--enable-error-handling',
        '--log-level', 'INFO'
    ]
    
    logger.info("Starting training with fixed feature engineering pipeline...")
    logger.info(f"Arguments: {sys.argv[1:]}")
    
    # Run training
    exit_code = train_main()
    
    if exit_code == 0:
        logger.info("\n" + "="*80)
        logger.info("RETRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("The new model should now have consistent metrics between training and evaluation.")
    else:
        logger.error("\n" + "="*80)
        logger.error("RETRAINING FAILED")
        logger.error("="*80)
    
    return exit_code

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
