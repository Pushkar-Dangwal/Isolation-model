"""
Main fraud detection system integration module.
Implements the FraudDetector class that orchestrates all components for end-to-end transaction processing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
import joblib
from pathlib import Path

# Import all components
from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from anomaly_detector import AnomalyDetector
from supervised_classifier import SupervisedClassifier
from risk_scorer import RiskScorer
from error_handling import ErrorHandler, DataValidationError, ModelError, PipelineError, create_error_response
from config import MODEL_CONFIG, MODELS_DIR, setup_logging
from model_persistence import ModelPersistenceManager, ModelMetadata
from pipeline_persistence import PipelinePersistenceManager, PipelineMetadata
from reproducibility import reproducibility_manager, get_component_seed
from model_metadata import ModelMetadataGenerator, metadata_generator

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Main fraud detection system that integrates all components into a single pipeline.
    
    This class provides end-to-end transaction processing capabilities including:
    - Data preprocessing and feature engineering
    - Anomaly detection using Deep Isolation Forest
    - Supervised classification with LightGBM
    - Risk scoring and threshold optimization
    - Model persistence and reproducibility
    
    The system is designed to handle large transaction volumes efficiently while
    maintaining high fraud detection accuracy and minimizing false positives.
    """
    
    def __init__(self,
                 # Component configurations
                 preprocessor_config: Optional[Dict] = None,
                 feature_engineer_config: Optional[Dict] = None,
                 anomaly_detector_config: Optional[Dict] = None,
                 classifier_config: Optional[Dict] = None,
                 risk_scorer_config: Optional[Dict] = None,
                 # Error handling configuration
                 error_handler_config: Optional[Dict] = None,
                 enable_error_handling: bool = True,
                 # System configuration
                 random_state: int = None,
                 n_jobs: int = -1,
                 verbose: bool = True,
                 # Reproducibility configuration
                 ensure_reproducibility: bool = True,
                 strict_determinism: bool = False):
        """
        Initialize the FraudDetector with all component configurations.
        
        Args:
            preprocessor_config: Configuration for DataPreprocessor
            feature_engineer_config: Configuration for FeatureEngineer
            anomaly_detector_config: Configuration for AnomalyDetector
            classifier_config: Configuration for SupervisedClassifier
            risk_scorer_config: Configuration for RiskScorer
            error_handler_config: Configuration for ErrorHandler
            enable_error_handling: Whether to enable comprehensive error handling
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs for components that support it
            verbose: Whether to enable verbose logging
            ensure_reproducibility: Whether to enforce reproducible behavior
            strict_determinism: Whether to use strict deterministic mode
        """
        # Set up logging
        if verbose:
            setup_logging()
        
        self.random_state = random_state or MODEL_CONFIG['random_seed']
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.enable_error_handling = enable_error_handling
        self.ensure_reproducibility = ensure_reproducibility
        self.strict_determinism = strict_determinism
        
        # Set up reproducibility if requested - Requirements 9.2
        if self.ensure_reproducibility:
            logger.info("Setting up reproducibility controls")
            self.reproducibility_state = reproducibility_manager.set_global_seed(
                seed=self.random_state,
                strict_mode=self.strict_determinism
            )
        else:
            self.reproducibility_state = None
        
        # Initialize error handler
        error_config = error_handler_config or {}
        self.error_handler = ErrorHandler(**error_config) if enable_error_handling else None
        
        # Initialize components with deterministic seeds
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Initialize anomaly detector with component-specific seed
        anomaly_config = anomaly_detector_config or {}
        anomaly_seed = get_component_seed('anomaly_detector', self.random_state) if self.ensure_reproducibility else self.random_state
        self.anomaly_detector = AnomalyDetector(
            random_state=anomaly_seed,
            n_jobs=self.n_jobs,
            **anomaly_config
        )
        
        # Initialize supervised classifier with component-specific seed
        classifier_config = classifier_config or {}
        classifier_seed = get_component_seed('classifier', self.random_state) if self.ensure_reproducibility else self.random_state
        self.classifier = SupervisedClassifier(
            random_state=classifier_seed,
            n_jobs=self.n_jobs,
            **classifier_config
        )
        
        # Initialize risk scorer with config
        risk_config = risk_scorer_config or {}
        self.risk_scorer = RiskScorer(**risk_config)
        
        # Pipeline state
        self.is_fitted = False
        self.feature_names = None
        self.training_metadata = {}
        self.performance_metrics = {}
        
        # Column mappings for flexibility
        self.column_mapping = {
            'transaction_id': 'transaction_id',
            'timestamp': 'timestamp',
            'sender_account': 'sender_account',
            'receiver_account': 'receiver_account',
            'amount': 'amount',
            'transaction_type': 'transaction_type',
            'merchant_category': 'merchant_category',
            'location': 'location',
            'device_used': 'device_used',
            'is_fraud': 'is_fraud'
        }
        
        logger.info(f"FraudDetector initialized with all components (error handling: {enable_error_handling}, "
                   f"reproducibility: {ensure_reproducibility}, strict_determinism: {strict_determinism})")
    
    def fit(self, 
            df: pd.DataFrame,
            target_column: str = 'is_fraud',
            transaction_id_column: str = 'transaction_id',
            validation_split: float = 0.2,
            optimize_thresholds: bool = True) -> 'FraudDetector':
        """
        Train the complete fraud detection pipeline on historical transaction data.
        
        This method orchestrates the training of all components:
        1. Data preprocessing and feature engineering
        2. Anomaly detector training
        3. Supervised classifier training with anomaly features
        4. Risk threshold optimization
        
        Args:
            df: Training DataFrame with transaction data
            target_column: Name of the fraud indicator column
            transaction_id_column: Name of the transaction ID column
            validation_split: Fraction of data to use for validation
            optimize_thresholds: Whether to perform threshold optimization
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If input data is invalid or required columns are missing
            PipelineError: If training pipeline fails
        """
        # Apply error handling if enabled
        if self.enable_error_handling and self.error_handler:
            return self._fit_with_error_handling(
                df, target_column, transaction_id_column, validation_split, optimize_thresholds
            )
        else:
            return self._fit_internal(
                df, target_column, transaction_id_column, validation_split, optimize_thresholds
            )
    
    @ErrorHandler().handle_errors("training_pipeline")
    def _fit_with_error_handling(self, df, target_column, transaction_id_column, validation_split, optimize_thresholds):
        """Internal fit method with error handling."""
        return self._fit_internal(df, target_column, transaction_id_column, validation_split, optimize_thresholds)
    
    def _fit_internal(self, df, target_column, transaction_id_column, validation_split, optimize_thresholds):
        """Internal fit method without error handling wrapper."""
        if df is None or len(df) == 0:
            raise DataValidationError("Training data cannot be None or empty")
        
        # Ensure reproducibility for training - Requirements 9.2
        if self.ensure_reproducibility:
            logger.info("Ensuring reproducible training with deterministic seed management")
            # Create deterministic context for training
            with reproducibility_manager.create_deterministic_context(self.random_state):
                return self._fit_with_reproducibility(df, target_column, transaction_id_column, validation_split, optimize_thresholds)
        else:
            return self._fit_without_reproducibility(df, target_column, transaction_id_column, validation_split, optimize_thresholds)
    
    def _fit_with_reproducibility(self, df, target_column, transaction_id_column, validation_split, optimize_thresholds):
        """Training method with reproducibility controls."""
        # Validate input data with error handler if available
        if self.enable_error_handling and self.error_handler:
            required_columns = [target_column, transaction_id_column]
            df = self.error_handler.validate_input_data(df, required_columns)
        else:
            # Basic validation
            required_columns = [target_column, transaction_id_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Starting reproducible fraud detection pipeline training on {len(df)} transactions")
        
        # Store original transaction IDs for traceability - Requirements 7.4
        original_transaction_ids = df[transaction_id_column].copy()
        
        try:
            # Step 1: Data preprocessing with deterministic seed
            logger.info("Step 1: Data preprocessing (deterministic)")
            preprocessing_seed = get_component_seed('preprocessing', self.random_state)
            np.random.seed(preprocessing_seed)  # Ensure deterministic preprocessing
            
            df_processed = self.preprocessor.fit_transform(
                df.copy(),
                timestamp_col=self.column_mapping['timestamp'],
                categorical_columns=[
                    self.column_mapping['transaction_type'],
                    self.column_mapping['merchant_category'],
                    self.column_mapping['location'],
                    self.column_mapping['device_used']
                ],
                amount_columns=[self.column_mapping['amount']]
            )
            
            # Handle data quality issues if error handler is available
            if self.enable_error_handling and self.error_handler:
                df_processed = self.error_handler.handle_data_quality_issues(df_processed)
            
            # Step 2: Feature engineering with deterministic seed
            logger.info("Step 2: Feature engineering (deterministic)")
            feature_seed = get_component_seed('feature_engineering', self.random_state)
            np.random.seed(feature_seed)
            
            df_features = self.feature_engineer.fit_transform(
                df_processed,
                timestamp_col=self.column_mapping['timestamp'],
                sender_col=self.column_mapping['sender_account'],
                receiver_col=self.column_mapping['receiver_account'],
                amount_col=self.column_mapping['amount'],
                location_col=self.column_mapping['location'],
                device_col=self.column_mapping['device_used'],
                fraud_col=target_column
            )
            
            # Prepare feature matrix using helper method
            X, y, feature_columns = self._prepare_feature_matrix(df_features, target_column, transaction_id_column)
            
            # Store feature names for later use
            self.feature_names = feature_columns
            
            # Split data for validation with deterministic seed
            from sklearn.model_selection import train_test_split
            split_seed = get_component_seed('train_test_split', self.random_state)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                stratify=y, random_state=split_seed
            )
            
            # Step 3: Train anomaly detector with component-specific seed
            logger.info("Step 3: Training anomaly detector (deterministic)")
            self.anomaly_detector.fit(X_train, feature_names=self.feature_names)
            
            # Generate anomaly scores for training data
            anomaly_scores_train = self.anomaly_detector.predict_anomaly_scores(X_train)
            anomaly_scores_val = self.anomaly_detector.predict_anomaly_scores(X_val)
            
            # Validate anomaly scores if error handler is available
            if self.enable_error_handling and self.error_handler:
                anomaly_scores_train = self.error_handler.validate_model_output(
                    anomaly_scores_train, expected_type=np.ndarray, value_range=(0, 1)
                )
                anomaly_scores_val = self.error_handler.validate_model_output(
                    anomaly_scores_val, expected_type=np.ndarray, value_range=(0, 1)
                )
            
            # Step 4: Integrate anomaly scores as features for supervised learning
            logger.info("Step 4: Integrating anomaly scores with features (deterministic)")
            X_train_with_anomaly = np.column_stack([X_train, anomaly_scores_train])
            X_val_with_anomaly = np.column_stack([X_val, anomaly_scores_val])
            
            # Update feature names to include anomaly score
            extended_feature_names = self.feature_names + ['anomaly_score']
            
            # Step 5: Train supervised classifier with component-specific seed
            logger.info("Step 5: Training supervised classifier (deterministic)")
            self.classifier.fit(
                X_train_with_anomaly, y_train,
                X_val=X_val_with_anomaly, y_val=y_val,
                feature_names=extended_feature_names
            )
            
            # Step 6: Generate fraud probabilities for threshold optimization
            logger.info("Step 6: Generating fraud probabilities (deterministic)")
            fraud_probabilities_val = self.classifier.predict_proba(X_val_with_anomaly)
            
            # Validate fraud probabilities if error handler is available
            if self.enable_error_handling and self.error_handler:
                fraud_probabilities_val = self.error_handler.validate_model_output(
                    fraud_probabilities_val, expected_type=np.ndarray, value_range=(0, 1)
                )
            
            # Step 7: Optimize risk thresholds with deterministic seed
            if optimize_thresholds:
                logger.info("Step 7: Optimizing risk thresholds (deterministic)")
                threshold_seed = get_component_seed('threshold_optimization', self.random_state)
                np.random.seed(threshold_seed)
                
                threshold_results = self.risk_scorer.tune_thresholds(y_val, fraud_probabilities_val)
                logger.info(f"Threshold optimization completed - PR-AUC: {threshold_results['pr_auc']:.3f}")
            
            # Step 8: Calculate training metadata and performance metrics
            self._calculate_training_metadata(df, X_train, X_val, y_train, y_val)
            self._calculate_performance_metrics(X_val_with_anomaly, y_val)
            
            # Step 9: Generate comprehensive model metadata - Requirements 9.5
            logger.info("Step 9: Generating comprehensive model metadata")
            self._generate_comprehensive_metadata(
                df, X_train, X_val, y_train, y_val, 
                fraud_probabilities_val, training_start_time=datetime.now().isoformat()
            )
            
            # Store reproducibility information in metadata
            if self.reproducibility_state:
                self.training_metadata['reproducibility_state'] = self.reproducibility_state.to_dict()
                self.training_metadata['deterministic_training'] = True
                self.training_metadata['component_seeds'] = {
                    'preprocessing': get_component_seed('preprocessing', self.random_state),
                    'feature_engineering': get_component_seed('feature_engineering', self.random_state),
                    'anomaly_detector': get_component_seed('anomaly_detector', self.random_state),
                    'classifier': get_component_seed('classifier', self.random_state),
                    'train_test_split': get_component_seed('train_test_split', self.random_state),
                    'threshold_optimization': get_component_seed('threshold_optimization', self.random_state)
                }
            
            # Mark as fitted
            self.is_fitted = True
            
            logger.info("Reproducible fraud detection pipeline training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Reproducible training pipeline failed: {str(e)}")
            if isinstance(e, (DataValidationError, ModelError)):
                raise
            else:
                raise PipelineError(f"Reproducible training pipeline failed: {str(e)}") from e
    
    def _prepare_feature_matrix(self, df_features: pd.DataFrame, target_column: str, 
                               transaction_id_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix by selecting only numeric columns and excluding non-feature columns.
        
        Args:
            df_features: DataFrame with engineered features
            target_column: Name of target column to exclude
            transaction_id_column: Name of transaction ID column to exclude
            
        Returns:
            Tuple of (X, y, feature_names) where X is feature matrix, y is target, feature_names is list of feature names
        """
        # Prepare feature matrix (exclude target, ID, and non-numeric columns)
        exclude_columns = [target_column, transaction_id_column]
        
        # Also exclude common non-numeric columns that shouldn't be features
        non_feature_columns = [
            self.column_mapping['timestamp'],
            self.column_mapping['sender_account'], 
            self.column_mapping['receiver_account'],
            self.column_mapping['location'],
            self.column_mapping['device_used'],
            'merchant_category',  # categorical, should be encoded already
            'transaction_type'    # categorical, should be encoded already
        ]
        exclude_columns.extend(non_feature_columns)
        
        # Select only numeric columns for features
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Ensure we have some features
        if len(feature_columns) == 0:
            raise ValueError("No numeric feature columns found after filtering")
        
        X = df_features[feature_columns].values
        y = df_features[target_column].values
        
        logger.info(f"Prepared feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_columns
    
    def _fit_without_reproducibility(self, df, target_column, transaction_id_column, validation_split, optimize_thresholds):
        """Training method without reproducibility controls (original implementation)."""
        # Validate input data with error handler if available
        if self.enable_error_handling and self.error_handler:
            required_columns = [target_column, transaction_id_column]
            df = self.error_handler.validate_input_data(df, required_columns)
        else:
            # Basic validation
            required_columns = [target_column, transaction_id_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Starting fraud detection pipeline training on {len(df)} transactions")
        
        # Store original transaction IDs for traceability - Requirements 7.4
        original_transaction_ids = df[transaction_id_column].copy()
        
        try:
            # Step 1: Data preprocessing
            logger.info("Step 1: Data preprocessing")
            df_processed = self.preprocessor.fit_transform(
                df.copy(),
                timestamp_col=self.column_mapping['timestamp'],
                categorical_columns=[
                    self.column_mapping['transaction_type'],
                    self.column_mapping['merchant_category'],
                    self.column_mapping['location'],
                    self.column_mapping['device_used']
                ],
                amount_columns=[self.column_mapping['amount']]
            )
            
            # Handle data quality issues if error handler is available
            if self.enable_error_handling and self.error_handler:
                df_processed = self.error_handler.handle_data_quality_issues(df_processed)
            
            # Step 2: Feature engineering
            logger.info("Step 2: Feature engineering")
            df_features = self.feature_engineer.fit_transform(
                df_processed,
                timestamp_col=self.column_mapping['timestamp'],
                sender_col=self.column_mapping['sender_account'],
                receiver_col=self.column_mapping['receiver_account'],
                amount_col=self.column_mapping['amount'],
                location_col=self.column_mapping['location'],
                device_col=self.column_mapping['device_used'],
                fraud_col=target_column
            )
            
            # Prepare feature matrix (exclude target and ID columns)
            exclude_columns = [target_column, transaction_id_column]
            
            # Also exclude common non-numeric columns that shouldn't be features
            non_feature_columns = [
                self.column_mapping['timestamp'],
                self.column_mapping['sender_account'], 
                self.column_mapping['receiver_account'],
                self.column_mapping['location'],
                self.column_mapping['device_used'],
                'merchant_category',  # categorical, should be encoded already
                'transaction_type'    # categorical, should be encoded already
            ]
            exclude_columns.extend(non_feature_columns)
            
            # Select only numeric columns for features
            numeric_columns = df_features.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            # Ensure we have some features
            if len(feature_columns) == 0:
                raise ValueError("No numeric feature columns found after filtering")
            
            X = df_features[feature_columns].values
            y = df_features[target_column].values
            
            # Store feature names for later use
            self.feature_names = feature_columns
            
            logger.info(f"Prepared feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                stratify=y, random_state=self.random_state
            )
            
            # Step 3: Train anomaly detector
            logger.info("Step 3: Training anomaly detector")
            self.anomaly_detector.fit(X_train, feature_names=self.feature_names)
            
            # Generate anomaly scores for training data
            anomaly_scores_train = self.anomaly_detector.predict_anomaly_scores(X_train)
            anomaly_scores_val = self.anomaly_detector.predict_anomaly_scores(X_val)
            
            # Validate anomaly scores if error handler is available
            if self.enable_error_handling and self.error_handler:
                anomaly_scores_train = self.error_handler.validate_model_output(
                    anomaly_scores_train, expected_type=np.ndarray, value_range=(0, 1)
                )
                anomaly_scores_val = self.error_handler.validate_model_output(
                    anomaly_scores_val, expected_type=np.ndarray, value_range=(0, 1)
                )
            
            # Step 4: Integrate anomaly scores as features for supervised learning
            logger.info("Step 4: Integrating anomaly scores with features")
            X_train_with_anomaly = np.column_stack([X_train, anomaly_scores_train])
            X_val_with_anomaly = np.column_stack([X_val, anomaly_scores_val])
            
            # Update feature names to include anomaly score
            extended_feature_names = self.feature_names + ['anomaly_score']
            
            # Step 5: Train supervised classifier
            logger.info("Step 5: Training supervised classifier")
            self.classifier.fit(
                X_train_with_anomaly, y_train,
                X_val=X_val_with_anomaly, y_val=y_val,
                feature_names=extended_feature_names
            )
            
            # Step 6: Generate fraud probabilities for threshold optimization
            logger.info("Step 6: Generating fraud probabilities")
            fraud_probabilities_val = self.classifier.predict_proba(X_val_with_anomaly)
            
            # Validate fraud probabilities if error handler is available
            if self.enable_error_handling and self.error_handler:
                fraud_probabilities_val = self.error_handler.validate_model_output(
                    fraud_probabilities_val, expected_type=np.ndarray, value_range=(0, 1)
                )
            
            # Step 7: Optimize risk thresholds
            if optimize_thresholds:
                logger.info("Step 7: Optimizing risk thresholds")
                threshold_results = self.risk_scorer.tune_thresholds(y_val, fraud_probabilities_val)
                logger.info(f"Threshold optimization completed - PR-AUC: {threshold_results['pr_auc']:.3f}")
            
            # Step 8: Calculate training metadata and performance metrics
            self._calculate_training_metadata(df, X_train, X_val, y_train, y_val)
            self._calculate_performance_metrics(X_val_with_anomaly, y_val)

            # Step 9: Generate comprehensive model metadata - Requirements 9.5
            logger.info("Step 9: Generating comprehensive model metadata")
            self._generate_comprehensive_metadata(
                df, X_train, X_val, y_train, y_val, 
                fraud_probabilities_val, training_start_time=datetime.now().isoformat()
            )
            
            # Mark as fitted
            self.is_fitted = True
            
            logger.info("Fraud detection pipeline training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            if isinstance(e, (DataValidationError, ModelError)):
                raise
            else:
                raise PipelineError(f"Training pipeline failed: {str(e)}") from e
            # Step 1: Data preprocessing
            logger.info("Step 1: Data preprocessing")
            df_processed = self.preprocessor.fit_transform(
                df.copy(),
                timestamp_col=self.column_mapping['timestamp'],
                categorical_columns=[
                    self.column_mapping['transaction_type'],
                    self.column_mapping['merchant_category'],
                    self.column_mapping['location'],
                    self.column_mapping['device_used']
                ],
                amount_columns=[self.column_mapping['amount']]
            )
            
            # Handle data quality issues if error handler is available
            if self.enable_error_handling and self.error_handler:
                df_processed = self.error_handler.handle_data_quality_issues(df_processed)
            
            # Step 2: Feature engineering
            logger.info("Step 2: Feature engineering")
            df_features = self.feature_engineer.fit_transform(
                df_processed,
                timestamp_col=self.column_mapping['timestamp'],
                sender_col=self.column_mapping['sender_account'],
                receiver_col=self.column_mapping['receiver_account'],
                amount_col=self.column_mapping['amount'],
                location_col=self.column_mapping['location'],
                device_col=self.column_mapping['device_used'],
                fraud_col=target_column
            )
            
            # Prepare feature matrix (exclude target and ID columns)
            exclude_columns = [target_column, transaction_id_column]
            
            # Also exclude common non-numeric columns that shouldn't be features
            non_feature_columns = [
                self.column_mapping['timestamp'],
                self.column_mapping['sender_account'], 
                self.column_mapping['receiver_account'],
                self.column_mapping['location'],
                self.column_mapping['device_used'],
                'merchant_category',  # categorical, should be encoded already
                'transaction_type'    # categorical, should be encoded already
            ]
            exclude_columns.extend(non_feature_columns)
            
            # Select only numeric columns for features
            numeric_columns = df_features.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]
            
            # Ensure we have some features
            if len(feature_columns) == 0:
                raise ValueError("No numeric feature columns found after filtering")
            
            X = df_features[feature_columns].values
            y = df_features[target_column].values
            
            # Store feature names for later use
            self.feature_names = feature_columns
            
            logger.info(f"Prepared feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Split data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                stratify=y, random_state=self.random_state
            )
            
            # Step 3: Train anomaly detector
            logger.info("Step 3: Training anomaly detector")
            self.anomaly_detector.fit(X_train, feature_names=self.feature_names)
            
            # Generate anomaly scores for training data
            anomaly_scores_train = self.anomaly_detector.predict_anomaly_scores(X_train)
            anomaly_scores_val = self.anomaly_detector.predict_anomaly_scores(X_val)
            
            # Validate anomaly scores if error handler is available
            if self.enable_error_handling and self.error_handler:
                anomaly_scores_train = self.error_handler.validate_model_output(
                    anomaly_scores_train, expected_type=np.ndarray, value_range=(0, 1)
                )
                anomaly_scores_val = self.error_handler.validate_model_output(
                    anomaly_scores_val, expected_type=np.ndarray, value_range=(0, 1)
                )
            
            # Step 4: Integrate anomaly scores as features for supervised learning
            logger.info("Step 4: Integrating anomaly scores with features")
            X_train_with_anomaly = np.column_stack([X_train, anomaly_scores_train])
            X_val_with_anomaly = np.column_stack([X_val, anomaly_scores_val])
            
            # Update feature names to include anomaly score
            extended_feature_names = self.feature_names + ['anomaly_score']
            
            # Step 5: Train supervised classifier
            logger.info("Step 5: Training supervised classifier")
            self.classifier.fit(
                X_train_with_anomaly, y_train,
                X_val=X_val_with_anomaly, y_val=y_val,
                feature_names=extended_feature_names
            )
            
            # Step 6: Generate fraud probabilities for threshold optimization
            logger.info("Step 6: Generating fraud probabilities")
            fraud_probabilities_val = self.classifier.predict_proba(X_val_with_anomaly)
            
            # Validate fraud probabilities if error handler is available
            if self.enable_error_handling and self.error_handler:
                fraud_probabilities_val = self.error_handler.validate_model_output(
                    fraud_probabilities_val, expected_type=np.ndarray, value_range=(0, 1)
                )
            
            # Step 7: Optimize risk thresholds
            if optimize_thresholds:
                logger.info("Step 7: Optimizing risk thresholds")
                threshold_results = self.risk_scorer.tune_thresholds(y_val, fraud_probabilities_val)
                logger.info(f"Threshold optimization completed - PR-AUC: {threshold_results['pr_auc']:.3f}")
            
            # Step 8: Calculate training metadata and performance metrics
            self._calculate_training_metadata(df, X_train, X_val, y_train, y_val)
            self._calculate_performance_metrics(X_val_with_anomaly, y_val)
            
            # Mark as fitted
            self.is_fitted = True
            
            logger.info("Fraud detection pipeline training completed successfully")
            return self
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            if isinstance(e, (DataValidationError, ModelError)):
                raise
            else:
                raise PipelineError(f"Training pipeline failed: {str(e)}") from e
    
    def predict(self, 
                df: pd.DataFrame,
                transaction_id_column: str = 'transaction_id',
                return_probabilities: bool = True,
                return_risk_levels: bool = True,
                return_explanations: bool = False) -> pd.DataFrame:
        """
        Generate fraud predictions for new transactions using the complete pipeline.
        
        This method processes transactions through the entire pipeline:
        1. Data preprocessing and feature engineering
        2. Anomaly score generation
        3. Fraud probability prediction
        4. Risk level assignment
        
        Args:
            df: DataFrame with transaction data to score
            transaction_id_column: Name of the transaction ID column
            return_probabilities: Whether to include fraud probability scores
            return_risk_levels: Whether to include risk level assignments
            return_explanations: Whether to include prediction explanations
            
        Returns:
            DataFrame with predictions and risk assessments
            
        Raises:
            ValueError: If pipeline is not fitted or input data is invalid
            PipelineError: If prediction pipeline fails
        """
        # Apply error handling if enabled
        if self.enable_error_handling and self.error_handler:
            return self._predict_with_error_handling(
                df, transaction_id_column, return_probabilities, return_risk_levels, return_explanations
            )
        else:
            return self._predict_internal(
                df, transaction_id_column, return_probabilities, return_risk_levels, return_explanations
            )
    
    def _predict_with_error_handling(self, df, transaction_id_column, return_probabilities, return_risk_levels, return_explanations):
        """Internal predict method with comprehensive error handling."""
        if not self.is_fitted:
            raise ValueError("FraudDetector must be fitted before making predictions")
        
        if df is None or len(df) == 0:
            logger.warning("Empty input data provided, returning empty result")
            return pd.DataFrame(columns=[
                transaction_id_column, 'fraud_probability', 'anomaly_score', 
                'risk_level', 'fraud_prediction', 'processed_at', 'model_version', 'has_error'
            ])
        
        logger.info(f"Generating fraud predictions for {len(df)} transactions")
        
        # Store original transaction IDs for output - Requirements 7.4
        if transaction_id_column not in df.columns:
            logger.warning(f"Transaction ID column '{transaction_id_column}' not found, using index")
            transaction_ids = df.index.astype(str).tolist()
        else:
            transaction_ids = df[transaction_id_column].copy().tolist()
        
        try:
            # Validate and sanitize input data
            df_validated = self.error_handler.validate_input_data(df.copy(), min_rows=1)
            df_clean = self.error_handler.handle_data_quality_issues(df_validated)
            
            # Step 1: Data preprocessing with error handling
            logger.debug("Preprocessing transaction data")
            try:
                df_processed = self.preprocessor.transform(
                    df_clean,
                    timestamp_col=self.column_mapping['timestamp']
                )
            except Exception as e:
                logger.error(f"Preprocessing failed: {str(e)}")
                return create_error_response(e, transaction_ids)
            
            # Step 2: Feature engineering with error handling
            logger.debug("Engineering features")
            try:
                df_features = self.feature_engineer.transform(
                    df_processed,
                    timestamp_col=self.column_mapping['timestamp'],
                    sender_col=self.column_mapping['sender_account'],
                    receiver_col=self.column_mapping['receiver_account'],
                    amount_col=self.column_mapping['amount'],
                    location_col=self.column_mapping['location'],
                    device_col=self.column_mapping['device_used'],
                    fraud_col='is_fraud'  # May not exist in inference data
                )
            except Exception as e:
                logger.error(f"Feature engineering failed: {str(e)}")
                return create_error_response(e, transaction_ids)
            
            # Handle missing features
            df_features = self.error_handler.handle_missing_features(
                df_features, self.feature_names, fill_strategy='zero'
            )
            
            X = df_features[self.feature_names].values
            
            # Step 3: Generate anomaly scores with error handling
            logger.debug("Generating anomaly scores")
            try:
                anomaly_scores = self.anomaly_detector.predict_anomaly_scores(X)
                anomaly_scores = self.error_handler.validate_model_output(
                    anomaly_scores, expected_type=np.ndarray, value_range=(0, 1)
                )
            except Exception as e:
                logger.error(f"Anomaly detection failed: {str(e)}")
                # Use fallback anomaly scores
                anomaly_scores = np.full(len(X), 0.5)
            
            # Step 4: Combine features with anomaly scores
            X_with_anomaly = np.column_stack([X, anomaly_scores])
            
            # Step 5: Generate fraud probabilities with error handling
            logger.debug("Generating fraud probabilities")
            try:
                fraud_probabilities = self.classifier.predict_proba(X_with_anomaly)
                fraud_probabilities = self.error_handler.validate_model_output(
                    fraud_probabilities, expected_type=np.ndarray, value_range=(0, 1)
                )
            except Exception as e:
                logger.error(f"Classification failed: {str(e)}")
                # Use fallback probabilities
                fraud_probabilities = np.full(len(X), 0.5)
            
            # Step 6: Assign risk levels with error handling
            logger.debug("Assigning risk levels")
            try:
                risk_levels = self.risk_scorer.assign_risk_levels(fraud_probabilities)
            except Exception as e:
                logger.error(f"Risk scoring failed: {str(e)}")
                # Use fallback risk levels
                risk_levels = np.full(len(X), 'medium')
            
            # Step 7: Prepare output DataFrame - Requirements 7.5
            results = pd.DataFrame({
                transaction_id_column: transaction_ids[:len(fraud_probabilities)]
            })
            
            if return_probabilities:
                results['fraud_probability'] = fraud_probabilities
                results['anomaly_score'] = anomaly_scores
            
            if return_risk_levels:
                results['risk_level'] = risk_levels
            
            # Add binary predictions based on optimal threshold
            if hasattr(self.risk_scorer, 'optimal_thresholds') and 'f1_optimal' in self.risk_scorer.optimal_thresholds:
                optimal_threshold = self.risk_scorer.optimal_thresholds['f1_optimal']
                results['fraud_prediction'] = (fraud_probabilities >= optimal_threshold).astype(int)
            else:
                # Use default threshold
                results['fraud_prediction'] = (fraud_probabilities >= 0.5).astype(int)
            
            # Add explanations if requested
            if return_explanations:
                results['explanation'] = self._generate_explanations(
                    fraud_probabilities, risk_levels, anomaly_scores
                )
            
            # Add processing metadata
            results['processed_at'] = datetime.now()
            results['model_version'] = self._get_model_version()
            results['has_error'] = False
            
            logger.info(f"Generated predictions for {len(results)} transactions")
            
            # Validate output schema compliance - Requirements 7.5
            self._validate_output_schema(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {str(e)}")
            # Return comprehensive error response with fallback predictions
            return self.error_handler.create_fallback_predictions(
                len(transaction_ids), transaction_ids
            )
    
    def _predict_internal(self, df, transaction_id_column, return_probabilities, return_risk_levels, return_explanations):
        """Internal predict method without error handling wrapper."""
        if not self.is_fitted:
            raise ValueError("FraudDetector must be fitted before making predictions")
        
        if df is None or len(df) == 0:
            raise ValueError("Input data cannot be None or empty")
        
        logger.info(f"Generating fraud predictions for {len(df)} transactions")
        
        # Store original transaction IDs for output - Requirements 7.4
        if transaction_id_column not in df.columns:
            logger.warning(f"Transaction ID column '{transaction_id_column}' not found, using index")
            transaction_ids = df.index.astype(str)
        else:
            transaction_ids = df[transaction_id_column].copy()
        
        # Step 1: Data preprocessing
        logger.debug("Preprocessing transaction data")
        df_processed = self.preprocessor.transform(
            df.copy(),
            timestamp_col=self.column_mapping['timestamp']
        )
        
        # Step 2: Feature engineering
        logger.debug("Engineering features")
        df_features = self.feature_engineer.transform(
            df_processed,
            timestamp_col=self.column_mapping['timestamp'],
            sender_col=self.column_mapping['sender_account'],
            receiver_col=self.column_mapping['receiver_account'],
            amount_col=self.column_mapping['amount'],
            location_col=self.column_mapping['location'],
            device_col=self.column_mapping['device_used'],
            fraud_col='is_fraud'  # May not exist in inference data
        )
        
        # Prepare feature matrix
        available_features = [col for col in self.feature_names if col in df_features.columns]
        if len(available_features) != len(self.feature_names):
            missing_features = set(self.feature_names) - set(available_features)
            logger.warning(f"Missing features in inference data: {missing_features}")
            # Fill missing features with zeros
            for feature in missing_features:
                df_features[feature] = 0.0
        
        X = df_features[self.feature_names].values
        
        # Step 3: Generate anomaly scores
        logger.debug("Generating anomaly scores")
        anomaly_scores = self.anomaly_detector.predict_anomaly_scores(X)
        
        # Step 4: Combine features with anomaly scores
        X_with_anomaly = np.column_stack([X, anomaly_scores])
        
        # Step 5: Generate fraud probabilities
        logger.debug("Generating fraud probabilities")
        fraud_probabilities = self.classifier.predict_proba(X_with_anomaly)
        
        # Step 6: Assign risk levels
        logger.debug("Assigning risk levels")
        risk_levels = self.risk_scorer.assign_risk_levels(fraud_probabilities)
        
        # Step 7: Prepare output DataFrame - Requirements 7.5
        results = pd.DataFrame({
            transaction_id_column: transaction_ids
        })
        
        if return_probabilities:
            results['fraud_probability'] = fraud_probabilities
            results['anomaly_score'] = anomaly_scores
        
        if return_risk_levels:
            results['risk_level'] = risk_levels
        
        # Add binary predictions based on optimal threshold
        if hasattr(self.risk_scorer, 'optimal_thresholds') and 'f1_optimal' in self.risk_scorer.optimal_thresholds:
            optimal_threshold = self.risk_scorer.optimal_thresholds['f1_optimal']
            results['fraud_prediction'] = (fraud_probabilities >= optimal_threshold).astype(int)
        else:
            # Use default threshold
            results['fraud_prediction'] = (fraud_probabilities >= 0.5).astype(int)
        
        # Add explanations if requested
        if return_explanations:
            results['explanation'] = self._generate_explanations(
                fraud_probabilities, risk_levels, anomaly_scores
            )
        
        # Add processing metadata
        results['processed_at'] = datetime.now()
        results['model_version'] = self._get_model_version()
        
        logger.info(f"Generated predictions for {len(results)} transactions")
        
        # Validate output schema compliance - Requirements 7.5
        self._validate_output_schema(results)
        
        return results
    
    def predict_single(self, 
                      transaction: Dict[str, Any],
                      transaction_id: str = None) -> Dict[str, Any]:
        """
        Generate fraud prediction for a single transaction.
        
        Args:
            transaction: Dictionary containing transaction data
            transaction_id: Optional transaction ID
            
        Returns:
            Dictionary with prediction results
        """
        # Convert single transaction to DataFrame
        df = pd.DataFrame([transaction])
        if transaction_id:
            df['transaction_id'] = transaction_id
        
        # Get predictions
        results = self.predict(df, return_explanations=True)
        
        # Return as dictionary
        return results.iloc[0].to_dict()
    
    def evaluate(self, 
                df: pd.DataFrame,
                target_column: str = 'is_fraud',
                transaction_id_column: str = 'transaction_id') -> Dict[str, Any]:
        """
        Evaluate the fraud detection pipeline on test data.
        
        Args:
            df: Test DataFrame with ground truth labels
            target_column: Name of the fraud indicator column
            transaction_id_column: Name of the transaction ID column
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("FraudDetector must be fitted before evaluation")
        
        logger.info(f"Evaluating fraud detection pipeline on {len(df)} transactions")
        
        # Generate predictions
        predictions = self.predict(df, transaction_id_column, return_probabilities=True)
        
        # Get ground truth
        y_true = df[target_column].values
        y_proba = predictions['fraud_probability'].values
        y_pred = predictions['fraud_prediction'].values
        
        # Calculate comprehensive metrics
        from sklearn.metrics import (
            precision_recall_curve, roc_curve, auc, precision_score,
            recall_score, f1_score, confusion_matrix, classification_report
        )
        
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Curve metrics
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        roc_auc = auc(fpr, tpr)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        # Risk level analysis
        risk_levels = predictions['risk_level'].values
        risk_distribution = pd.Series(risk_levels).value_counts().to_dict()
        
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
                'risk_distribution': risk_distribution,
                'fraud_by_risk_level': self._analyze_fraud_by_risk_level(y_true, risk_levels)
            },
            'component_performance': {
                'anomaly_detector': self._evaluate_anomaly_detector(df, y_true),
                'classifier': self._evaluate_classifier(df, y_true),
                'risk_scorer': self._evaluate_risk_scorer(y_true, y_proba, risk_levels)
            },
            'business_metrics': {
                'fraud_detection_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'customer_friction_rate': float((tp + fp) / len(y_true)),
                'precision_at_high_risk': self._calculate_precision_at_risk_level(y_true, risk_levels, 'high')
            }
        }
        
        logger.info(f"Evaluation completed - F1: {f1:.3f}, PR-AUC: {pr_auc:.3f}, ROC-AUC: {roc_auc:.3f}")
        
        return evaluation_results
    
    def save_model(self, filepath: str, version: Optional[str] = None, 
                   include_metadata: bool = True, save_pipeline: bool = True,
                   training_data: Optional[pd.DataFrame] = None) -> str:
        """
        Save the complete trained fraud detection pipeline to disk with comprehensive metadata.
        
        This method saves all pipeline components using the centralized model persistence
        system with versioning, metadata tracking, and integrity verification. Optionally
        saves the preprocessing pipeline separately for inference consistency.
        
        Args:
            filepath: Base path for saving the model (without extension)
            version: Optional version string (auto-generated if not provided)
            include_metadata: Whether to include comprehensive metadata
            save_pipeline: Whether to save preprocessing pipeline separately
            training_data: Optional training data for pipeline schema validation
            
        Returns:
            Path to the saved model
            
        Raises:
            ValueError: If pipeline is not fitted
            IOError: If saving fails
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted FraudDetector")
        
        logger.info(f"Saving FraudDetector pipeline to {filepath}")
        
        # Initialize persistence managers
        model_persistence_manager = ModelPersistenceManager()
        pipeline_persistence_manager = PipelinePersistenceManager()
        
        # Extract model name from filepath
        model_name = Path(filepath).name
        
        try:
            # Save preprocessing pipeline separately if requested - Requirements 9.3
            pipeline_path = None
            if save_pipeline:
                logger.info("Saving preprocessing pipeline for inference consistency")
                
                pipeline_metadata = None
                if include_metadata:
                    pipeline_metadata = PipelineMetadata(
                        pipeline_name=f"{model_name}_preprocessing",
                        pipeline_type='fraud_detection_preprocessing',
                        version=version or datetime.now().strftime("%Y%m%d_%H%M%S"),
                        created_at=datetime.now().isoformat(),
                        feature_names=self.feature_names,
                        pipeline_config={
                            'column_mapping': self.column_mapping,
                            'random_state': self.random_state,
                            'ensure_reproducibility': self.ensure_reproducibility
                        },
                        tags=['preprocessing', 'inference', 'fraud_detection'],
                        description=f"Preprocessing pipeline for {model_name} fraud detection model"
                    )
                
                pipeline_path = pipeline_persistence_manager.save_pipeline(
                    preprocessor=self.preprocessor,
                    feature_engineer=self.feature_engineer,
                    pipeline_name=f"{model_name}_preprocessing",
                    training_data=training_data,
                    version=version,
                    metadata=pipeline_metadata,
                    validate_consistency=True
                )
            
            # Create comprehensive pipeline data
            pipeline_data = {
                'preprocessor': self.preprocessor,
                'feature_engineer': self.feature_engineer,
                'anomaly_detector': self.anomaly_detector,
                'classifier': self.classifier,
                'risk_scorer': self.risk_scorer,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names,
                'column_mapping': self.column_mapping,
                'training_metadata': self.training_metadata,
                'performance_metrics': self.performance_metrics,
                'random_state': self.random_state,
                'enable_error_handling': self.enable_error_handling,
                'ensure_reproducibility': self.ensure_reproducibility,
                'strict_determinism': getattr(self, 'strict_determinism', False),
                'reproducibility_state': self.reproducibility_state.to_dict() if self.reproducibility_state else None,
                'model_version': self._get_model_version(),
                'saved_at': datetime.now().isoformat(),
                'pipeline_path': pipeline_path
            }
            
            # Create comprehensive metadata if requested
            metadata = None
            if include_metadata:
                # Check if comprehensive metadata was generated during training
                if 'comprehensive_metadata' in self.training_metadata:
                    logger.info("Using comprehensive metadata generated during training")
                    
                    # Load the comprehensive metadata and update it for saving
                    from model_metadata import ComprehensiveModelMetadata
                    comprehensive_metadata = ComprehensiveModelMetadata.from_dict(
                        self.training_metadata['comprehensive_metadata']
                    )
                    
                    # Update metadata for saving context
                    comprehensive_metadata.model_name = model_name
                    comprehensive_metadata.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
                    comprehensive_metadata.created_at = datetime.now().isoformat()
                    comprehensive_metadata.deployment_target = "Production"
                    comprehensive_metadata.tags.extend(['saved_model', 'production_ready'])
                    
                    # Save comprehensive metadata to separate file
                    metadata_path = f"{filepath}_metadata.json"
                    comprehensive_metadata.to_json(metadata_path)
                    logger.info(f"Comprehensive metadata saved to {metadata_path}")
                    
                    # Create basic metadata for model persistence system
                    metadata = ModelMetadata(
                        model_name=model_name,
                        model_type='fraud_detection_pipeline',
                        version=comprehensive_metadata.version,
                        created_at=comprehensive_metadata.created_at,
                        training_samples=comprehensive_metadata.training_info.training_data_shape[0] if comprehensive_metadata.training_info.training_data_shape else None,
                        feature_count=len(comprehensive_metadata.data_info.feature_names),
                        feature_names=comprehensive_metadata.data_info.feature_names,
                        hyperparameters=comprehensive_metadata.training_info.hyperparameters,
                        performance_metrics=comprehensive_metadata.performance_info.metrics,
                        training_duration=comprehensive_metadata.training_info.training_duration_seconds,
                        random_state=comprehensive_metadata.training_info.random_seeds.get('master_seed'),
                        tags=comprehensive_metadata.tags,
                        description=comprehensive_metadata.description
                    )
                    
                else:
                    # Fallback to basic metadata generation
                    logger.info("Generating basic metadata for model saving")
                    
                    # Calculate training duration if available
                    training_duration = None
                    if 'training_date' in self.training_metadata:
                        try:
                            training_start = datetime.fromisoformat(self.training_metadata['training_date'])
                            training_duration = (datetime.now() - training_start).total_seconds()
                        except:
                            pass
                    
                    # Get component hyperparameters
                    hyperparameters = {
                        'random_state': self.random_state,
                        'enable_error_handling': self.enable_error_handling,
                        'ensure_reproducibility': self.ensure_reproducibility,
                        'strict_determinism': getattr(self, 'strict_determinism', False),
                        'anomaly_detector': {
                            'contamination': getattr(self.anomaly_detector, 'contamination', None),
                            'n_estimators': getattr(self.anomaly_detector, 'n_estimators', None),
                            'n_layers': getattr(self.anomaly_detector, 'n_layers', None),
                            'n_hidden': getattr(self.anomaly_detector, 'n_hidden', None)
                        },
                        'classifier': {
                            'learning_rate': getattr(self.classifier, 'learning_rate', None),
                            'num_leaves': getattr(self.classifier, 'num_leaves', None),
                            'max_depth': getattr(self.classifier, 'max_depth', None),
                            'scale_pos_weight': getattr(self.classifier, 'scale_pos_weight', None)
                        },
                        'risk_scorer': {
                            'low_risk_threshold': getattr(self.risk_scorer, 'low_risk_threshold', None),
                            'high_risk_threshold': getattr(self.risk_scorer, 'high_risk_threshold', None)
                        }
                    }
                    
                    metadata = ModelMetadata(
                        model_name=model_name,
                        model_type='fraud_detection_pipeline',
                        version=version or datetime.now().strftime("%Y%m%d_%H%M%S"),
                        created_at=datetime.now().isoformat(),
                        training_samples=self.training_metadata.get('training_samples'),
                        feature_count=len(self.feature_names) if self.feature_names else None,
                        feature_names=self.feature_names,
                        hyperparameters=hyperparameters,
                        performance_metrics=self.performance_metrics,
                        training_duration=training_duration,
                        random_state=self.random_state,
                        tags=['fraud_detection', 'pipeline', 'production'],
                        description=f"Complete fraud detection pipeline with {len(self.feature_names) if self.feature_names else 0} features and preprocessing consistency"
                    )
            
            # Save using persistence manager
            saved_path = model_persistence_manager.save_model(
                model=pipeline_data,
                model_name=model_name,
                model_type='fraud_detection_pipeline',
                metadata=metadata,
                version=version,
                format='joblib',
                compress=True
            )
            
            logger.info("FraudDetector pipeline saved successfully with comprehensive metadata and preprocessing consistency")
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to save FraudDetector pipeline: {e}")
            raise IOError(f"Failed to save pipeline: {e}")
    
    def load_model(self, filepath: str, version: Optional[str] = None,
                   verify_integrity: bool = True, load_pipeline: bool = True) -> 'FraudDetector':
        """
        Load a trained fraud detection pipeline from disk with metadata validation.
        
        This method loads the complete pipeline using the centralized model persistence
        system with integrity verification and metadata restoration. Optionally loads
        the preprocessing pipeline separately for consistency validation.
        
        Args:
            filepath: Base path to load the model from
            version: Specific version to load (loads latest if not provided)
            verify_integrity: Whether to verify model integrity
            load_pipeline: Whether to load preprocessing pipeline separately
            
        Returns:
            Self for method chaining
            
        Raises:
            FileNotFoundError: If model is not found
            IOError: If loading fails
        """
        logger.info(f"Loading FraudDetector pipeline from {filepath}")
        
        # Initialize persistence managers
        model_persistence_manager = ModelPersistenceManager()
        pipeline_persistence_manager = PipelinePersistenceManager()
        
        # Extract model name from filepath
        model_name = Path(filepath).name
        
        try:
            # Load using persistence manager
            pipeline_data, metadata = model_persistence_manager.load_model(
                model_name=model_name,
                version=version,
                verify_integrity=verify_integrity
            )
            
            # Load preprocessing pipeline separately if available and requested
            if load_pipeline and 'pipeline_path' in pipeline_data and pipeline_data['pipeline_path']:
                logger.info("Loading preprocessing pipeline for consistency validation")
                try:
                    preprocessor, feature_engineer, pipeline_metadata = pipeline_persistence_manager.load_pipeline(
                        pipeline_name=f"{model_name}_preprocessing",
                        version=version,
                        validate_consistency=True
                    )
                    
                    # Use the separately saved preprocessing components for better consistency
                    pipeline_data['preprocessor'] = preprocessor
                    pipeline_data['feature_engineer'] = feature_engineer
                    
                    logger.info("Preprocessing pipeline loaded successfully with consistency validation")
                    
                except Exception as e:
                    logger.warning(f"Failed to load separate preprocessing pipeline: {e}")
                    logger.info("Using preprocessing components from main model file")
            
            # Restore pipeline components
            self.preprocessor = pipeline_data['preprocessor']
            self.feature_engineer = pipeline_data['feature_engineer']
            self.anomaly_detector = pipeline_data['anomaly_detector']
            self.classifier = pipeline_data['classifier']
            self.risk_scorer = pipeline_data['risk_scorer']
            
            # Restore pipeline state
            self.is_fitted = pipeline_data['is_fitted']
            self.feature_names = pipeline_data['feature_names']
            self.column_mapping = pipeline_data['column_mapping']
            self.training_metadata = pipeline_data['training_metadata']
            self.performance_metrics = pipeline_data['performance_metrics']
            self.random_state = pipeline_data['random_state']
            self.enable_error_handling = pipeline_data.get('enable_error_handling', True)
            self.ensure_reproducibility = pipeline_data.get('ensure_reproducibility', False)
            self.strict_determinism = pipeline_data.get('strict_determinism', False)
            
            # Restore reproducibility state if available
            if pipeline_data.get('reproducibility_state'):
                from reproducibility import ReproducibilityState
                self.reproducibility_state = ReproducibilityState.from_dict(
                    pipeline_data['reproducibility_state']
                )
                
                # Restore reproducibility if it was enabled
                if self.ensure_reproducibility:
                    reproducibility_manager.restore_state(
                        self.reproducibility_state, 
                        strict_mode=self.strict_determinism
                    )
            else:
                self.reproducibility_state = None
            
            # Log metadata information if available
            if metadata:
                logger.info(f"Loaded model with metadata - Version: {metadata.version}, "
                           f"Features: {metadata.feature_count}, "
                           f"Training samples: {metadata.training_samples}, "
                           f"Reproducibility: {self.ensure_reproducibility}")
            
            logger.info("FraudDetector pipeline loaded successfully with preprocessing consistency")
            return self
            
        except Exception as e:
            logger.error(f"Failed to load FraudDetector pipeline: {e}")
            raise IOError(f"Failed to load pipeline: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the trained fraud detection pipeline.
        
        Returns:
            Dictionary containing model configuration and statistics
        """
        if not self.is_fitted:
            return {'status': 'not_fitted', 'is_fitted': False}
        
        model_info = {
            'status': 'fitted',
            'is_fitted': True,
            'model_version': self._get_model_version(),
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'training_metadata': self.training_metadata,
            'performance_metrics': self.performance_metrics,
            'component_info': {
                'preprocessor': {'fitted': self.preprocessor.is_fitted},
                'feature_engineer': {'fitted': self.feature_engineer.is_fitted},
                'anomaly_detector': self.anomaly_detector.get_model_info(),
                'classifier': self.classifier.get_model_info(),
                'risk_scorer': {
                    'fitted': self.risk_scorer.is_fitted,
                    'thresholds': {
                        'low_risk': self.risk_scorer.low_risk_threshold,
                        'high_risk': self.risk_scorer.high_risk_threshold
                    }
                }
            },
            'column_mapping': self.column_mapping,
            'random_state': self.random_state,
            'error_handling_enabled': self.enable_error_handling
        }
        
        # Add error handling statistics if available
        if self.enable_error_handling and self.error_handler:
            model_info['error_handling'] = self.error_handler.get_error_summary()
        
        return model_info
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information including error statistics.
        
        Returns:
            Dictionary containing system health metrics
        """
        health_info = {
            'overall_status': 'healthy',
            'pipeline_fitted': self.is_fitted,
            'components_status': {
                'preprocessor': 'fitted' if self.preprocessor.is_fitted else 'not_fitted',
                'feature_engineer': 'fitted' if self.feature_engineer.is_fitted else 'not_fitted',
                'anomaly_detector': 'fitted' if self.anomaly_detector.is_fitted else 'not_fitted',
                'classifier': 'fitted' if self.classifier.is_fitted else 'not_fitted',
                'risk_scorer': 'fitted' if self.risk_scorer.is_fitted else 'not_fitted'
            },
            'error_handling': {
                'enabled': self.enable_error_handling,
                'status': 'active' if self.enable_error_handling else 'disabled'
            }
        }
        
        # Add detailed error handling information if available
        if self.enable_error_handling and self.error_handler:
            error_summary = self.error_handler.get_error_summary()
            health_info['error_handling'].update(error_summary)
            
            # Determine overall health status
            if error_summary.get('health_status') == 'degraded':
                health_info['overall_status'] = 'degraded'
        
        # Check if all components are fitted
        if not all(status == 'fitted' for status in health_info['components_status'].values()):
            health_info['overall_status'] = 'not_ready'
        
        return health_info
    
    def reset_error_statistics(self) -> None:
        """Reset error handling statistics."""
        if self.enable_error_handling and self.error_handler:
            self.error_handler.reset_error_stats()
            logger.info("Error statistics reset")
        else:
            logger.warning("Error handling not enabled, no statistics to reset")
    
    def _calculate_training_metadata(self, df: pd.DataFrame, X_train: np.ndarray, 
                                   X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> None:
        """Calculate and store training metadata."""
        self.training_metadata = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'total_samples': len(df),
            'feature_count': len(self.feature_names),
            'fraud_rate_train': float(np.mean(y_train)),
            'fraud_rate_val': float(np.mean(y_val)),
            'fraud_rate_total': float(np.mean(df[self.column_mapping.get('is_fraud', 'is_fraud')])),
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state
        }
    
    def _calculate_performance_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Calculate and store performance metrics on validation data."""
        # Generate predictions on validation set
        fraud_probabilities = self.classifier.predict_proba(X_val)
        risk_levels = self.risk_scorer.assign_risk_levels(fraud_probabilities)
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_curve, roc_curve, auc
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, fraud_probabilities)
        fpr, tpr, _ = roc_curve(y_val, fraud_probabilities)
        
        self.performance_metrics = {
            'pr_auc': float(auc(recall_curve, precision_curve)),
            'roc_auc': float(auc(fpr, tpr)),
            'risk_distribution': pd.Series(risk_levels).value_counts().to_dict()
        }
    
    def _generate_explanations(self, probabilities: np.ndarray, 
                             risk_levels: np.ndarray, anomaly_scores: np.ndarray) -> List[str]:
        """Generate human-readable explanations for predictions."""
        explanations = []
        
        for prob, risk, anomaly in zip(probabilities, risk_levels, anomaly_scores):
            if risk == 'high':
                if anomaly > 0.7:
                    explanation = f"High fraud risk (probability: {prob:.2f}) due to unusual transaction patterns (anomaly score: {anomaly:.2f})"
                else:
                    explanation = f"High fraud risk (probability: {prob:.2f}) based on learned fraud patterns"
            elif risk == 'medium':
                explanation = f"Medium fraud risk (probability: {prob:.2f}) - requires additional verification"
            else:
                explanation = f"Low fraud risk (probability: {prob:.2f}) - transaction appears legitimate"
            
            explanations.append(explanation)
        
        return explanations
    
    def _validate_output_schema(self, results: pd.DataFrame) -> None:
        """Validate that output conforms to required schema - Requirements 7.5."""
        required_columns = ['fraud_probability', 'risk_level', 'fraud_prediction']
        
        for col in required_columns:
            if col not in results.columns:
                raise ValueError(f"Output schema validation failed: missing column '{col}'")
        
        # Validate data types and ranges
        if not results['fraud_probability'].between(0, 1).all():
            raise ValueError("fraud_probability values must be in [0, 1] range")
        
        valid_risk_levels = {'low', 'medium', 'high'}
        if not results['risk_level'].isin(valid_risk_levels).all():
            raise ValueError(f"risk_level values must be in {valid_risk_levels}")
        
        if not results['fraud_prediction'].isin([0, 1]).all():
            raise ValueError("fraud_prediction values must be 0 or 1")
    
    def _get_model_version(self) -> str:
        """Get model version string."""
        return f"fraud_detector_v1.0_{datetime.now().strftime('%Y%m%d')}"
    
    def _analyze_fraud_by_risk_level(self, y_true: np.ndarray, risk_levels: np.ndarray) -> Dict[str, Dict]:
        """Analyze fraud distribution by risk level."""
        analysis = {}
        
        for risk_level in ['low', 'medium', 'high']:
            mask = risk_levels == risk_level
            if np.any(mask):
                level_true = y_true[mask]
                analysis[risk_level] = {
                    'count': int(np.sum(mask)),
                    'fraud_count': int(np.sum(level_true)),
                    'fraud_rate': float(np.mean(level_true)),
                    'percentage_of_total': float(np.sum(mask) / len(y_true) * 100)
                }
        
        return analysis
    
    def _evaluate_anomaly_detector(self, df: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate anomaly detector performance."""
        # This is a simplified evaluation - in practice, you'd want more comprehensive metrics
        return {
            'component_status': 'fitted' if self.anomaly_detector.is_fitted else 'not_fitted',
            'feature_count': len(self.feature_names) if self.feature_names else 0
        }
    
    def _evaluate_classifier(self, df: pd.DataFrame, y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluate classifier performance."""
        return {
            'component_status': 'fitted' if self.classifier.is_fitted else 'not_fitted',
            'model_type': 'LightGBM',
            'training_history': getattr(self.classifier, 'training_history', {})
        }
    
    def _evaluate_risk_scorer(self, y_true: np.ndarray, y_proba: np.ndarray, 
                            risk_levels: np.ndarray) -> Dict[str, Any]:
        """Evaluate risk scorer performance."""
        return {
            'component_status': 'fitted' if self.risk_scorer.is_fitted else 'not_fitted',
            'threshold_config': {
                'low_risk': self.risk_scorer.low_risk_threshold,
                'high_risk': self.risk_scorer.high_risk_threshold
            },
            'risk_distribution': pd.Series(risk_levels).value_counts().to_dict()
        }
    
    def get_reproducibility_info(self) -> Dict[str, Any]:
        """
        Get comprehensive reproducibility information for the trained model.
        
        Returns:
            Dictionary containing reproducibility state and component seeds
        """
        if not self.ensure_reproducibility:
            return {
                'reproducibility_enabled': False,
                'message': 'Reproducibility was not enabled during initialization'
            }
        
        info = {
            'reproducibility_enabled': True,
            'strict_determinism': self.strict_determinism,
            'master_seed': self.random_state,
            'current_state': self.reproducibility_state.to_dict() if self.reproducibility_state else None
        }
        
        # Add component seeds if available in training metadata
        if 'component_seeds' in self.training_metadata:
            info['component_seeds'] = self.training_metadata['component_seeds']
        
        # Add reproducibility state from training if available
        if 'reproducibility_state' in self.training_metadata:
            info['training_reproducibility_state'] = self.training_metadata['reproducibility_state']
        
        return info
    
    def verify_reproducibility(self, other_detector: 'FraudDetector') -> Dict[str, bool]:
        """
        Verify reproducibility compatibility with another FraudDetector instance.
        
        Args:
            other_detector: Another FraudDetector instance to compare with
            
        Returns:
            Dictionary indicating reproducibility compatibility
        """
        if not self.ensure_reproducibility or not other_detector.ensure_reproducibility:
            return {
                'compatible': False,
                'reason': 'One or both detectors do not have reproducibility enabled'
            }
        
        verification = {
            'master_seed_match': self.random_state == other_detector.random_state,
            'strict_determinism_match': self.strict_determinism == other_detector.strict_determinism,
            'reproducibility_states_compatible': False
        }
        
        # Compare reproducibility states if both exist
        if self.reproducibility_state and other_detector.reproducibility_state:
            from reproducibility import reproducibility_manager
            state_verification = reproducibility_manager.verify_reproducibility(
                self.reproducibility_state, other_detector.reproducibility_state
            )
            verification['reproducibility_states_compatible'] = state_verification['all_match']
            verification['state_details'] = state_verification
        
        verification['compatible'] = all([
            verification['master_seed_match'],
            verification['strict_determinism_match'],
            verification['reproducibility_states_compatible']
        ])
        
        return verification
    
    def save_reproducibility_state(self, filepath: str) -> None:
        """
        Save the current reproducibility state to a file.
        
        Args:
            filepath: Path to save the reproducibility state
        """
        if not self.ensure_reproducibility or not self.reproducibility_state:
            raise ValueError("Reproducibility is not enabled or no state available")
        
        reproducibility_manager.save_state(filepath, self.reproducibility_state)
        logger.info(f"Reproducibility state saved to {filepath}")
    
    def load_reproducibility_state(self, filepath: str) -> None:
        """
        Load and apply a reproducibility state from a file.
        
        Args:
            filepath: Path to load the reproducibility state from
        """
        if not self.ensure_reproducibility:
            logger.warning("Reproducibility is not enabled, loading state anyway")
            self.ensure_reproducibility = True
        
        state = reproducibility_manager.load_state(filepath)
        reproducibility_manager.restore_state(state, strict_mode=self.strict_determinism)
        self.reproducibility_state = state
        self.random_state = state.master_seed
        
        logger.info(f"Reproducibility state loaded from {filepath}")
    
    def create_reproducible_prediction_context(self, seed: Optional[int] = None):
        """
        Create a context manager for reproducible predictions.
        
        This ensures that predictions are deterministic when using the same input data.
        
        Args:
            seed: Seed to use for predictions (uses master seed if not provided)
            
        Returns:
            Context manager for reproducible predictions
        """
        if not self.ensure_reproducibility:
            logger.warning("Reproducibility is not enabled, creating context anyway")
        
        prediction_seed = seed or self.random_state
        return reproducibility_manager.create_deterministic_context(prediction_seed)
    
    def validate_inference_data(self, df: pd.DataFrame, 
                               pipeline_name: Optional[str] = None,
                               version: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate inference data against the expected preprocessing pipeline schema.
        
        This method ensures that new data is compatible with the trained pipeline's
        preprocessing requirements, helping prevent inference errors.
        
        Args:
            df: DataFrame to validate
            pipeline_name: Optional pipeline name (auto-generated if not provided)
            version: Optional pipeline version
            
        Returns:
            Dictionary containing validation results
        """
        if not pipeline_name:
            # Try to infer pipeline name from model
            pipeline_name = f"{self.__class__.__name__.lower()}_preprocessing"
        
        try:
            from pipeline_persistence import pipeline_persistence
            
            validation_results = pipeline_persistence.validate_inference_data(
                pipeline_name=pipeline_name,
                version=version or "latest",
                inference_data=df
            )
            
            # Add additional validation specific to fraud detection
            fraud_specific_validation = self._validate_fraud_detection_data(df)
            validation_results.update(fraud_specific_validation)
            
            return validation_results
            
        except Exception as e:
            logger.warning(f"Pipeline validation failed: {e}")
            return {
                'validation_performed': False,
                'error': str(e),
                'basic_validation': self._validate_fraud_detection_data(df)
            }
    
    def _validate_fraud_detection_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform fraud detection specific data validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation = {
            'fraud_detection_validation': True,
            'required_columns_present': True,
            'missing_columns': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        # Check for required columns based on column mapping
        required_columns = [
            self.column_mapping['timestamp'],
            self.column_mapping['sender_account'],
            self.column_mapping['receiver_account'],
            self.column_mapping['amount'],
            self.column_mapping['transaction_type'],
            self.column_mapping['merchant_category'],
            self.column_mapping['location'],
            self.column_mapping['device_used']
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['required_columns_present'] = False
            validation['missing_columns'] = missing_columns
            validation['fraud_detection_validation'] = False
        
        # Check data quality issues
        for col in df.columns:
            if col in required_columns:
                # Check for excessive missing values
                missing_pct = df[col].isna().sum() / len(df)
                if missing_pct > 0.5:
                    validation['data_quality_issues'].append(
                        f"Column '{col}' has {missing_pct:.1%} missing values"
                    )
                
                # Check for data type issues
                if col == self.column_mapping['amount']:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        validation['data_quality_issues'].append(
                            f"Amount column '{col}' should be numeric"
                        )
                elif col == self.column_mapping['timestamp']:
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        validation['recommendations'].append(
                            f"Timestamp column '{col}' should be datetime type"
                        )
        
        # Overall validation status
        if validation['data_quality_issues']:
            validation['fraud_detection_validation'] = False
        
        return validation
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current pipeline configuration.
        
        Returns:
            Dictionary containing pipeline information
        """
        info = {
            'pipeline_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'column_mapping': self.column_mapping,
            'random_state': self.random_state,
            'enable_error_handling': self.enable_error_handling,
            'ensure_reproducibility': getattr(self, 'ensure_reproducibility', False),
            'strict_determinism': getattr(self, 'strict_determinism', False),
            'components': {
                'preprocessor_fitted': getattr(self.preprocessor, 'is_fitted', False),
                'feature_engineer_fitted': getattr(self.feature_engineer, 'is_fitted', False),
                'anomaly_detector_fitted': getattr(self.anomaly_detector, 'is_fitted', False),
                'classifier_fitted': getattr(self.classifier, 'is_fitted', False),
                'risk_scorer_fitted': getattr(self.risk_scorer, 'is_fitted', False)
            }
        }
        
        # Add training metadata if available
        if self.training_metadata:
            info['training_metadata'] = self.training_metadata
        
        # Add performance metrics if available
        if self.performance_metrics:
            info['performance_metrics'] = self.performance_metrics
        
        # Add reproducibility information if available
        if hasattr(self, 'reproducibility_state') and self.reproducibility_state:
            info['reproducibility_info'] = self.get_reproducibility_info()
        
        return info
    
    def get_comprehensive_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get the comprehensive model metadata if available.
        
        Returns:
            Dictionary containing comprehensive metadata or None if not available
        """
        if 'comprehensive_metadata' in self.training_metadata:
            return self.training_metadata['comprehensive_metadata']
        return None
    
    def export_comprehensive_metadata(self, filepath: str) -> bool:
        """
        Export comprehensive metadata to a JSON file.
        
        Args:
            filepath: Path to save the metadata JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            comprehensive_metadata = self.get_comprehensive_metadata()
            if comprehensive_metadata:
                from model_metadata import ComprehensiveModelMetadata
                metadata_obj = ComprehensiveModelMetadata.from_dict(comprehensive_metadata)
                metadata_obj.to_json(filepath)
                logger.info(f"Comprehensive metadata exported to {filepath}")
                return True
            else:
                logger.warning("No comprehensive metadata available to export")
                return False
        except Exception as e:
            logger.error(f"Failed to export comprehensive metadata: {e}")
            return False
    
    def _calculate_precision_at_risk_level(self, y_true: np.ndarray, 
                                         risk_levels: np.ndarray, risk_level: str) -> float:
        """Calculate precision for a specific risk level."""
        mask = risk_levels == risk_level
        if not np.any(mask):
            return 0.0
        
        level_true = y_true[mask]
        if len(level_true) == 0:
            return 0.0
        
        return float(np.mean(level_true))
    
    def _generate_comprehensive_metadata(self, 
                                       df: pd.DataFrame,
                                       X_train: np.ndarray,
                                       X_val: np.ndarray,
                                       y_train: np.ndarray,
                                       y_val: np.ndarray,
                                       fraud_probabilities_val: np.ndarray,
                                       training_start_time: str) -> None:
        """
        Generate comprehensive model metadata using the ModelMetadataGenerator.
        
        This method creates detailed metadata including system information, environment details,
        training parameters, performance metrics, and data characteristics for governance
        and reproducibility purposes.
        
        Args:
            df: Original training DataFrame
            X_train: Training feature matrix
            X_val: Validation feature matrix
            y_train: Training labels
            y_val: Validation labels
            fraud_probabilities_val: Validation fraud probabilities
            training_start_time: ISO format timestamp of training start
        """
        try:
            logger.info("Generating comprehensive model metadata")
            
            # Calculate training end time and duration
            training_end_time = datetime.now().isoformat()
            
            # Get hyperparameters from all components
            hyperparameters = {
                'fraud_detector': {
                    'random_state': self.random_state,
                    'enable_error_handling': self.enable_error_handling,
                    'ensure_reproducibility': getattr(self, 'ensure_reproducibility', False),
                    'strict_determinism': getattr(self, 'strict_determinism', False)
                },
                'anomaly_detector': {
                    'contamination': getattr(self.anomaly_detector, 'contamination', None),
                    'n_estimators': getattr(self.anomaly_detector, 'n_estimators', None),
                    'n_layers': getattr(self.anomaly_detector, 'n_layers', None),
                    'n_hidden': getattr(self.anomaly_detector, 'n_hidden', None),
                    'random_state': getattr(self.anomaly_detector, 'random_state', None)
                },
                'classifier': {
                    'learning_rate': getattr(self.classifier, 'learning_rate', None),
                    'num_leaves': getattr(self.classifier, 'num_leaves', None),
                    'max_depth': getattr(self.classifier, 'max_depth', None),
                    'scale_pos_weight': getattr(self.classifier, 'scale_pos_weight', None),
                    'random_state': getattr(self.classifier, 'random_state', None)
                },
                'risk_scorer': {
                    'low_risk_threshold': getattr(self.risk_scorer, 'low_risk_threshold', None),
                    'high_risk_threshold': getattr(self.risk_scorer, 'high_risk_threshold', None)
                }
            }
            
            # Get random seeds used
            random_seeds = {
                'master_seed': self.random_state,
                'anomaly_detector': getattr(self.anomaly_detector, 'random_state', self.random_state),
                'classifier': getattr(self.classifier, 'random_state', self.random_state)
            }
            
            # Add component seeds if reproducibility is enabled
            if hasattr(self, 'ensure_reproducibility') and self.ensure_reproducibility:
                random_seeds.update({
                    'preprocessing': get_component_seed('preprocessing', self.random_state),
                    'feature_engineering': get_component_seed('feature_engineering', self.random_state),
                    'train_test_split': get_component_seed('train_test_split', self.random_state),
                    'threshold_optimization': get_component_seed('threshold_optimization', self.random_state)
                })
            
            # Calculate data hash for integrity
            data_hash = metadata_generator.calculate_data_hash(df)
            
            # Create training information
            training_info = metadata_generator.create_training_info(
                training_start_time=training_start_time,
                training_end_time=training_end_time,
                hyperparameters=hyperparameters,
                random_seeds=random_seeds,
                training_data_shape=X_train.shape,
                training_data_hash=data_hash,
                validation_data_shape=X_val.shape
            )
            
            # Calculate performance metrics
            from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
            
            # Generate binary predictions for confusion matrix
            optimal_threshold = 0.4
            if hasattr(self.risk_scorer, 'optimal_thresholds') and 'f1_optimal' in self.risk_scorer.optimal_thresholds:
                optimal_threshold = self.risk_scorer.optimal_thresholds['f1_optimal']
            
            y_pred_val = (fraud_probabilities_val >= optimal_threshold).astype(int)
            
            # Calculate metrics
            precision_curve, recall_curve, _ = precision_recall_curve(y_val, fraud_probabilities_val)
            fpr, tpr, _ = roc_curve(y_val, fraud_probabilities_val)
            pr_auc = auc(recall_curve, precision_curve)
            roc_auc = auc(fpr, tpr)
            
            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred_val)
            cm_list = cm.tolist() if cm.size > 0 else [[0, 0], [0, 0]]
            
            # Feature importance if available
            feature_importance = None
            if hasattr(self.classifier, 'get_feature_importance'):
                try:
                    importance_dict = self.classifier.get_feature_importance()
                    if importance_dict:
                        feature_importance = importance_dict
                except:
                    pass
            
            # Create performance information
            performance_info = metadata_generator.create_performance_info(
                metrics={
                    'pr_auc': float(pr_auc),
                    'roc_auc': float(roc_auc),
                    'fraud_rate_train': float(np.mean(y_train)),
                    'fraud_rate_val': float(np.mean(y_val))
                },
                validation_metrics={
                    'pr_auc': float(pr_auc),
                    'roc_auc': float(roc_auc),
                    'fraud_rate': float(np.mean(y_val))
                },
                confusion_matrix=cm_list,
                feature_importance=feature_importance
            )
            
            # Determine feature types
            feature_types = {}
            categorical_features = []
            numerical_features = []
            
            if self.feature_names:
                for feature in self.feature_names:
                    if any(keyword in feature.lower() for keyword in ['category', 'type', 'flag', 'device', 'location']):
                        feature_types[feature] = 'categorical'
                        categorical_features.append(feature)
                    else:
                        feature_types[feature] = 'numerical'
                        numerical_features.append(feature)
            
            # Create data information
            target_column = self.column_mapping.get('is_fraud', 'is_fraud')
            data_info = metadata_generator.create_data_info(
                feature_names=self.feature_names or [],
                feature_types=feature_types,
                target_column=target_column,
                training_data=df,
                categorical_features=categorical_features,
                numerical_features=numerical_features
            )
            
            # Generate comprehensive metadata
            model_name = f"fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_version = f"v1.0_{datetime.now().strftime('%Y%m%d')}"
            
            comprehensive_metadata = metadata_generator.generate_comprehensive_metadata(
                model_name=model_name,
                model_type='fraud_detection_pipeline',
                version=model_version,
                description=f"Complete fraud detection pipeline with {len(self.feature_names) if self.feature_names else 0} features, anomaly detection, and supervised classification",
                training_info=training_info,
                performance_info=performance_info,
                data_info=data_info,
                model_purpose="Financial Fraud Detection",
                business_context="Real-time transaction monitoring and fraud prevention",
                deployment_target="Production",
                tags=['fraud_detection', 'anomaly_detection', 'supervised_learning', 'lightgbm', 'isolation_forest'],
                compliance_info={
                    'data_privacy': 'PII anonymized',
                    'model_governance': 'Comprehensive metadata tracking',
                    'reproducibility': 'Deterministic training with seed management' if getattr(self, 'ensure_reproducibility', False) else 'Non-deterministic training'
                },
                lineage_info={
                    'training_pipeline': 'FraudDetector.fit()',
                    'data_source': 'Historical transaction data',
                    'preprocessing_steps': ['timestamp_parsing', 'categorical_encoding', 'feature_engineering', 'anomaly_detection'],
                    'model_components': ['DataPreprocessor', 'FeatureEngineer', 'AnomalyDetector', 'SupervisedClassifier', 'RiskScorer']
                }
            )
            
            # Store comprehensive metadata in training metadata
            self.training_metadata['comprehensive_metadata'] = comprehensive_metadata.to_dict()
            self.training_metadata['metadata_generated_at'] = datetime.now().isoformat()
            
            logger.info(f"Comprehensive model metadata generated successfully for {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to generate comprehensive metadata: {e}")
            # Store basic metadata as fallback
            self.training_metadata['metadata_generation_error'] = str(e)
            self.training_metadata['basic_metadata'] = {
                'model_name': f"fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_type': 'fraud_detection_pipeline',
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'training_completed_at': datetime.now().isoformat()
            }