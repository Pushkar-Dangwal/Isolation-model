"""
Model metadata generation module for the fraud detection system.
Provides comprehensive metadata tracking for models and pipelines.
"""

import logging
import json
import os
import platform
import sys
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import hashlib
import subprocess

# Try to import optional dependencies
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information for reproducibility and debugging."""
    python_version: str
    platform: str
    processor: str
    architecture: tuple
    memory_total_gb: float
    cpu_count: int
    hostname: str
    username: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentInfo:
    """Environment information including dependencies and versions."""
    library_versions: Dict[str, str]
    environment_variables: Dict[str, str]
    python_path: List[str]
    working_directory: str
    git_info: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingInfo:
    """Training process information and parameters."""
    training_start_time: str
    training_end_time: Optional[str]
    training_duration_seconds: Optional[float]
    training_data_shape: Optional[tuple]
    training_data_hash: Optional[str]
    validation_data_shape: Optional[tuple]
    hyperparameters: Dict[str, Any]
    random_seeds: Dict[str, int]
    cross_validation_folds: Optional[int] = None
    early_stopping_rounds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceInfo:
    """Model performance metrics and evaluation results."""
    metrics: Dict[str, float]
    validation_metrics: Optional[Dict[str, float]]
    test_metrics: Optional[Dict[str, float]]
    confusion_matrix: Optional[List[List[int]]]
    feature_importance: Optional[Dict[str, float]]
    model_size_bytes: Optional[int]
    inference_time_ms: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataInfo:
    """Information about the data used for training and validation."""
    feature_names: List[str]
    feature_types: Dict[str, str]
    target_column: str
    categorical_features: List[str]
    numerical_features: List[str]
    missing_value_counts: Dict[str, int]
    data_quality_score: Optional[float]
    class_distribution: Optional[Dict[str, int]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComprehensiveModelMetadata:
    """
    Comprehensive model metadata containing all relevant information
    for model tracking, reproducibility, and governance.
    """
    model_id: str
    model_name: str
    model_type: str
    version: str
    created_at: str
    created_by: str
    description: str
    tags: List[str]
    
    # Core information
    system_info: SystemInfo
    environment_info: EnvironmentInfo
    training_info: TrainingInfo
    performance_info: PerformanceInfo
    data_info: DataInfo
    
    # Additional metadata
    model_purpose: str
    business_context: str
    deployment_target: str
    compliance_info: Optional[Dict[str, Any]] = None
    lineage_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Convert metadata to JSON string or save to file."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Model metadata saved to {filepath}")
        
        return json_str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComprehensiveModelMetadata':
        """Create metadata from dictionary."""
        # Convert nested dictionaries back to dataclasses
        system_info = SystemInfo(**data['system_info'])
        environment_info = EnvironmentInfo(**data['environment_info'])
        training_info = TrainingInfo(**data['training_info'])
        performance_info = PerformanceInfo(**data['performance_info'])
        data_info = DataInfo(**data['data_info'])
        
        # Create main metadata object
        metadata_dict = data.copy()
        metadata_dict['system_info'] = system_info
        metadata_dict['environment_info'] = environment_info
        metadata_dict['training_info'] = training_info
        metadata_dict['performance_info'] = performance_info
        metadata_dict['data_info'] = data_info
        
        return cls(**metadata_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ComprehensiveModelMetadata':
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ModelMetadataGenerator:
    """
    Comprehensive model metadata generator for the fraud detection system.
    
    Automatically captures system information, environment details, training
    parameters, performance metrics, and data characteristics to create
    comprehensive model metadata for governance and reproducibility.
    """
    
    def __init__(self):
        """Initialize the metadata generator."""
        self.logger = logging.getLogger(__name__)
        
    def capture_system_info(self) -> SystemInfo:
        """
        Capture comprehensive system information.
        
        Returns:
            SystemInfo object containing system details
        """
        try:
            # Get memory information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            system_info = SystemInfo(
                python_version=sys.version,
                platform=platform.platform(),
                processor=platform.processor(),
                architecture=platform.architecture(),
                memory_total_gb=round(memory_gb, 2),
                cpu_count=psutil.cpu_count(),
                hostname=platform.node(),
                username=os.getenv('USER', os.getenv('USERNAME', 'unknown')),
                timestamp=datetime.now().isoformat()
            )
            
            logger.debug("System information captured successfully")
            return system_info
            
        except Exception as e:
            logger.warning(f"Failed to capture complete system info: {e}")
            # Return minimal system info
            return SystemInfo(
                python_version=sys.version,
                platform=platform.platform(),
                processor="unknown",
                architecture=("unknown", "unknown"),
                memory_total_gb=0.0,
                cpu_count=1,
                hostname="unknown",
                username="unknown",
                timestamp=datetime.now().isoformat()
            )
    
    def capture_environment_info(self) -> EnvironmentInfo:
        """
        Capture comprehensive environment information.
        
        Returns:
            EnvironmentInfo object containing environment details
        """
        try:
            # Capture library versions
            library_versions = {
                'python': sys.version.split()[0],
                'numpy': np.__version__,
                'pandas': pd.__version__,
            }
            
            # Add optional library versions
            if SKLEARN_AVAILABLE:
                library_versions['scikit-learn'] = sklearn.__version__
            
            if LIGHTGBM_AVAILABLE:
                library_versions['lightgbm'] = lightgbm.__version__
            
            try:
                import joblib
                library_versions['joblib'] = joblib.__version__
            except ImportError:
                pass
            
            # Capture relevant environment variables
            relevant_env_vars = [
                'PYTHONPATH', 'PATH', 'PYTHONHASHSEED', 'OMP_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'CUDA_VISIBLE_DEVICES'
            ]
            
            environment_variables = {}
            for var in relevant_env_vars:
                value = os.getenv(var)
                if value:
                    environment_variables[var] = value
            
            # Capture Git information if available
            git_info = None
            if GIT_AVAILABLE:
                git_info = self._capture_git_info()
            
            environment_info = EnvironmentInfo(
                library_versions=library_versions,
                environment_variables=environment_variables,
                python_path=sys.path,
                working_directory=os.getcwd(),
                git_info=git_info
            )
            
            logger.debug("Environment information captured successfully")
            return environment_info
            
        except Exception as e:
            logger.warning(f"Failed to capture complete environment info: {e}")
            # Return minimal environment info
            return EnvironmentInfo(
                library_versions={'python': sys.version.split()[0]},
                environment_variables={},
                python_path=[],
                working_directory=os.getcwd()
            )
    
    def _capture_git_info(self) -> Optional[Dict[str, str]]:
        """Capture Git repository information."""
        try:
            repo = git.Repo(search_parent_directories=True)
            
            git_info = {
                'repository_url': next(repo.remote().urls),
                'branch': repo.active_branch.name,
                'commit_hash': repo.head.commit.hexsha,
                'commit_message': repo.head.commit.message.strip(),
                'commit_author': str(repo.head.commit.author),
                'commit_date': repo.head.commit.committed_datetime.isoformat(),
                'is_dirty': repo.is_dirty(),
                'untracked_files': repo.untracked_files
            }
            
            return git_info
            
        except Exception as e:
            logger.debug(f"Failed to capture Git info: {e}")
            return None
    
    def create_training_info(self,
                           training_start_time: str,
                           hyperparameters: Dict[str, Any],
                           random_seeds: Dict[str, int],
                           training_data_shape: Optional[tuple] = None,
                           training_data_hash: Optional[str] = None,
                           validation_data_shape: Optional[tuple] = None,
                           training_end_time: Optional[str] = None,
                           cross_validation_folds: Optional[int] = None,
                           early_stopping_rounds: Optional[int] = None) -> TrainingInfo:
        """
        Create training information metadata.
        
        Args:
            training_start_time: ISO format timestamp of training start
            hyperparameters: Dictionary of model hyperparameters
            random_seeds: Dictionary of random seeds used
            training_data_shape: Shape of training data
            training_data_hash: Hash of training data for integrity
            validation_data_shape: Shape of validation data
            training_end_time: ISO format timestamp of training end
            cross_validation_folds: Number of CV folds used
            early_stopping_rounds: Early stopping rounds used
            
        Returns:
            TrainingInfo object
        """
        training_duration = None
        if training_end_time:
            try:
                start = datetime.fromisoformat(training_start_time)
                end = datetime.fromisoformat(training_end_time)
                training_duration = (end - start).total_seconds()
            except Exception as e:
                logger.warning(f"Failed to calculate training duration: {e}")
        
        return TrainingInfo(
            training_start_time=training_start_time,
            training_end_time=training_end_time,
            training_duration_seconds=training_duration,
            training_data_shape=training_data_shape,
            training_data_hash=training_data_hash,
            validation_data_shape=validation_data_shape,
            hyperparameters=hyperparameters,
            random_seeds=random_seeds,
            cross_validation_folds=cross_validation_folds,
            early_stopping_rounds=early_stopping_rounds
        )
    
    def create_performance_info(self,
                              metrics: Dict[str, float],
                              validation_metrics: Optional[Dict[str, float]] = None,
                              test_metrics: Optional[Dict[str, float]] = None,
                              confusion_matrix: Optional[List[List[int]]] = None,
                              feature_importance: Optional[Dict[str, float]] = None,
                              model_size_bytes: Optional[int] = None,
                              inference_time_ms: Optional[float] = None) -> PerformanceInfo:
        """
        Create performance information metadata.
        
        Args:
            metrics: Training metrics
            validation_metrics: Validation metrics
            test_metrics: Test metrics
            confusion_matrix: Confusion matrix as nested list
            feature_importance: Feature importance scores
            model_size_bytes: Model size in bytes
            inference_time_ms: Average inference time in milliseconds
            
        Returns:
            PerformanceInfo object
        """
        return PerformanceInfo(
            metrics=metrics,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            confusion_matrix=confusion_matrix,
            feature_importance=feature_importance,
            model_size_bytes=model_size_bytes,
            inference_time_ms=inference_time_ms
        )
    
    def create_data_info(self,
                        feature_names: List[str],
                        feature_types: Dict[str, str],
                        target_column: str,
                        training_data: Optional[pd.DataFrame] = None,
                        categorical_features: Optional[List[str]] = None,
                        numerical_features: Optional[List[str]] = None) -> DataInfo:
        """
        Create data information metadata.
        
        Args:
            feature_names: List of feature names
            feature_types: Dictionary mapping features to their types
            target_column: Name of target column
            training_data: Optional training DataFrame for analysis
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            
        Returns:
            DataInfo object
        """
        missing_value_counts = {}
        data_quality_score = None
        class_distribution = None
        
        # Analyze training data if provided
        if training_data is not None:
            # Calculate missing value counts
            missing_value_counts = training_data.isnull().sum().to_dict()
            
            # Calculate data quality score (percentage of non-missing values)
            total_cells = training_data.shape[0] * training_data.shape[1]
            missing_cells = training_data.isnull().sum().sum()
            data_quality_score = (total_cells - missing_cells) / total_cells
            
            # Calculate class distribution if target column exists
            if target_column in training_data.columns:
                class_distribution = training_data[target_column].value_counts().to_dict()
        
        # Auto-detect categorical and numerical features if not provided
        if training_data is not None:
            if categorical_features is None:
                categorical_features = training_data.select_dtypes(
                    include=['object', 'category']
                ).columns.tolist()
            
            if numerical_features is None:
                numerical_features = training_data.select_dtypes(
                    include=['int64', 'float64']
                ).columns.tolist()
        
        return DataInfo(
            feature_names=feature_names,
            feature_types=feature_types,
            target_column=target_column,
            categorical_features=categorical_features or [],
            numerical_features=numerical_features or [],
            missing_value_counts=missing_value_counts,
            data_quality_score=data_quality_score,
            class_distribution=class_distribution
        )
    
    def generate_comprehensive_metadata(self,
                                      model_name: str,
                                      model_type: str,
                                      version: str,
                                      description: str,
                                      training_info: TrainingInfo,
                                      performance_info: PerformanceInfo,
                                      data_info: DataInfo,
                                      model_purpose: str = "Fraud Detection",
                                      business_context: str = "Financial Transaction Monitoring",
                                      deployment_target: str = "Production",
                                      tags: Optional[List[str]] = None,
                                      compliance_info: Optional[Dict[str, Any]] = None,
                                      lineage_info: Optional[Dict[str, Any]] = None) -> ComprehensiveModelMetadata:
        """
        Generate comprehensive model metadata.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'lightgbm_classifier')
            version: Model version
            description: Model description
            training_info: Training information
            performance_info: Performance information
            data_info: Data information
            model_purpose: Purpose of the model
            business_context: Business context
            deployment_target: Target deployment environment
            tags: List of tags for categorization
            compliance_info: Compliance and governance information
            lineage_info: Model lineage and provenance information
            
        Returns:
            ComprehensiveModelMetadata object
        """
        # Generate unique model ID
        model_id = self._generate_model_id(model_name, version)
        
        # Capture system and environment information
        system_info = self.capture_system_info()
        environment_info = self.capture_environment_info()
        
        # Set default tags if not provided
        if tags is None:
            tags = ['fraud_detection', 'machine_learning', model_type]
        
        # Get current user
        created_by = os.getenv('USER', os.getenv('USERNAME', 'unknown'))
        
        metadata = ComprehensiveModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            description=description,
            tags=tags,
            system_info=system_info,
            environment_info=environment_info,
            training_info=training_info,
            performance_info=performance_info,
            data_info=data_info,
            model_purpose=model_purpose,
            business_context=business_context,
            deployment_target=deployment_target,
            compliance_info=compliance_info,
            lineage_info=lineage_info
        )
        
        logger.info(f"Generated comprehensive metadata for model {model_name} v{version}")
        return metadata
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """Generate a unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        id_string = f"{model_name}_{version}_{timestamp}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]
    
    def calculate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Calculate a hash of the training data for integrity verification.
        
        Args:
            data: DataFrame to hash
            
        Returns:
            SHA256 hash of the data
        """
        try:
            # Create hash of the data
            data_hash = hashlib.sha256(
                pd.util.hash_pandas_object(data, index=True).values
            ).hexdigest()
            
            logger.debug("Data hash calculated successfully")
            return data_hash
            
        except Exception as e:
            logger.warning(f"Failed to calculate data hash: {e}")
            return "unknown"
    
    def benchmark_inference_time(self, model: Any, sample_data: np.ndarray, 
                                n_iterations: int = 100) -> float:
        """
        Benchmark model inference time.
        
        Args:
            model: Trained model with predict method
            sample_data: Sample data for inference
            n_iterations: Number of iterations for benchmarking
            
        Returns:
            Average inference time in milliseconds
        """
        try:
            import time
            
            # Warm up
            for _ in range(10):
                _ = model.predict(sample_data[:1])
            
            # Benchmark
            start_time = time.time()
            for _ in range(n_iterations):
                _ = model.predict(sample_data[:1])
            end_time = time.time()
            
            avg_time_ms = ((end_time - start_time) / n_iterations) * 1000
            
            logger.debug(f"Inference time benchmarked: {avg_time_ms:.2f}ms")
            return avg_time_ms
            
        except Exception as e:
            logger.warning(f"Failed to benchmark inference time: {e}")
            return 0.0


# Global instance for easy access
metadata_generator = ModelMetadataGenerator()