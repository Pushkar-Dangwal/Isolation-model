"""
Configuration module for the fraud detection system.
Handles logging setup and system-wide configuration.
"""

import logging
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(exist_ok=True)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging(log_level: str = LOG_LEVEL) -> None:
    """
    Set up logging configuration for the fraud detection system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(LOGS_DIR / "fraud_detection.log")  # File output
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)

# Model configuration
MODEL_CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
    "fraud_rate": 0.04,  # Expected fraud rate (4%)
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "time_windows": {
        "short_term": "1H",  # 1 hour
        "medium_term": "24H",  # 24 hours
    },
    "categorical_encoding": "label",  # or "onehot"
    "amount_transformation": "log",
}

# Anomaly detection configuration
ANOMALY_CONFIG = {
    "contamination": 0.04,  # Expected fraud rate
    "n_estimators": 100,
    "max_samples": "auto",
    "random_state": MODEL_CONFIG["random_seed"],
}

# Supervised classification configuration
CLASSIFIER_CONFIG = {
    "lightgbm": {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": MODEL_CONFIG["random_seed"],
        "is_unbalance": True,
    },
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "random_state": MODEL_CONFIG["random_seed"],
        "scale_pos_weight": 24,  # (96% / 4%) for class imbalance
    }
}

# Risk scoring configuration
RISK_CONFIG = {
    "thresholds": {
        "low_risk": 0.3,
        "high_risk": 0.7,
    },
    "risk_levels": ["low", "medium", "high"],
}