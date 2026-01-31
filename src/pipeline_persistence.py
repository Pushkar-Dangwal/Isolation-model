"""
Pipeline persistence module for the fraud detection system.
Provides comprehensive pipeline serialization with preprocessing consistency.
"""

import logging
import joblib
import json
import pickle
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import hashlib

from config import MODELS_DIR, MODEL_CONFIG
from model_persistence import ModelPersistenceManager, ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetadata:
    """
    Comprehensive metadata for pipeline artifacts.
    Tracks preprocessing steps, feature engineering, and pipeline configuration.
    """
    pipeline_name: str
    pipeline_type: str
    version: str
    created_at: str
    preprocessing_steps: Optional[List[str]] = None
    feature_engineering_steps: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    categorical_encoders: Optional[Dict[str, str]] = None
    numerical_transformers: Optional[Dict[str, str]] = None
    pipeline_config: Optional[Dict[str, Any]] = None
    data_schema: Optional[Dict[str, str]] = None
    preprocessing_hash: Optional[str] = None
    training_data_hash: Optional[str] = None
    consistency_checks: Optional[Dict[str, bool]] = None
    dependencies: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


class PipelinePersistenceManager:
    """
    Comprehensive pipeline persistence manager for the fraud detection system.
    
    Ensures preprocessing consistency between training and inference by
    serializing complete preprocessing pipelines with validation and
    integrity checks.
    """
    
    def __init__(self, base_dir: Union[str, Path] = None):
        """
        Initialize the pipeline persistence manager.
        
        Args:
            base_dir: Base directory for pipeline storage
        """
        self.base_dir = Path(base_dir) if base_dir else MODELS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.pipelines_dir = self.base_dir / "pipelines"
        self.preprocessors_dir = self.base_dir / "preprocessors"
        self.schemas_dir = self.base_dir / "schemas"
        self.metadata_dir = self.base_dir / "pipeline_metadata"
        
        for directory in [self.pipelines_dir, self.preprocessors_dir, self.schemas_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Registry for tracking pipelines
        self.registry_file = self.base_dir / "pipeline_registry.json"
        self._initialize_registry()
        
        logger.info(f"PipelinePersistenceManager initialized with base directory: {self.base_dir}")
    
    def _initialize_registry(self) -> None:
        """Initialize the pipeline registry if it doesn't exist."""
        if not self.registry_file.exists():
            registry = {
                "created_at": datetime.now().isoformat(),
                "pipelines": {},
                "version": "1.0"
            }
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info("Initialized pipeline registry")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the pipeline registry."""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pipeline registry: {e}")
            return {"pipelines": {}, "version": "1.0"}
    
    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the pipeline registry."""
        try:
            registry["last_updated"] = datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save pipeline registry: {e}")
    
    def _generate_preprocessing_hash(self, preprocessor: Any, feature_engineer: Any) -> str:
        """Generate a hash for preprocessing pipeline integrity."""
        try:
            # Create a representation of the preprocessing pipeline
            preprocessing_info = {
                'preprocessor_type': type(preprocessor).__name__,
                'preprocessor_fitted': getattr(preprocessor, 'is_fitted', False),
                'feature_engineer_type': type(feature_engineer).__name__,
                'feature_engineer_fitted': getattr(feature_engineer, 'is_fitted', False),
            }
            
            # Add encoder information if available
            if hasattr(preprocessor, 'label_encoders'):
                preprocessing_info['label_encoders'] = list(preprocessor.label_encoders.keys())
            
            if hasattr(preprocessor, 'onehot_encoders'):
                preprocessing_info['onehot_encoders'] = list(preprocessor.onehot_encoders.keys())
            
            # Create hash
            preprocessing_string = json.dumps(preprocessing_info, sort_keys=True)
            return hashlib.sha256(preprocessing_string.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate preprocessing hash: {e}")
            return "unknown"
    
    def _extract_data_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        """Extract data schema from a DataFrame."""
        schema = {}
        for column in df.columns:
            dtype = str(df[column].dtype)
            schema[column] = dtype
        return schema
    
    def _validate_data_schema(self, df: pd.DataFrame, expected_schema: Dict[str, str]) -> Dict[str, bool]:
        """Validate DataFrame against expected schema."""
        validation = {
            'columns_match': set(df.columns) == set(expected_schema.keys()),
            'dtypes_match': True,
            'missing_columns': [],
            'extra_columns': [],
            'dtype_mismatches': []
        }
        
        # Check for missing and extra columns
        expected_columns = set(expected_schema.keys())
        actual_columns = set(df.columns)
        
        validation['missing_columns'] = list(expected_columns - actual_columns)
        validation['extra_columns'] = list(actual_columns - expected_columns)
        
        # Check data types for common columns
        common_columns = expected_columns & actual_columns
        for column in common_columns:
            expected_dtype = expected_schema[column]
            actual_dtype = str(df[column].dtype)
            
            if expected_dtype != actual_dtype:
                validation['dtype_mismatches'].append({
                    'column': column,
                    'expected': expected_dtype,
                    'actual': actual_dtype
                })
                validation['dtypes_match'] = False
        
        validation['schema_valid'] = (
            validation['columns_match'] and 
            validation['dtypes_match'] and 
            len(validation['missing_columns']) == 0 and 
            len(validation['extra_columns']) == 0
        )
        
        return validation
    
    def save_pipeline(self,
                     preprocessor: Any,
                     feature_engineer: Any,
                     pipeline_name: str,
                     training_data: Optional[pd.DataFrame] = None,
                     version: Optional[str] = None,
                     metadata: Optional[PipelineMetadata] = None,
                     validate_consistency: bool = True) -> str:
        """
        Save a complete preprocessing pipeline with consistency validation.
        
        Args:
            preprocessor: Fitted data preprocessor
            feature_engineer: Fitted feature engineer
            pipeline_name: Name identifier for the pipeline
            training_data: Optional training data for schema extraction
            version: Optional version string
            metadata: Optional pipeline metadata
            validate_consistency: Whether to validate preprocessing consistency
            
        Returns:
            Path to the saved pipeline
            
        Raises:
            ValueError: If pipeline components are not fitted
            IOError: If saving fails
        """
        # Validate that components are fitted
        if not getattr(preprocessor, 'is_fitted', False):
            raise ValueError("Preprocessor must be fitted before saving")
        
        if not getattr(feature_engineer, 'is_fitted', False):
            raise ValueError("FeatureEngineer must be fitted before saving")
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create pipeline directory
        pipeline_dir = self.pipelines_dir / pipeline_name / version
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving pipeline '{pipeline_name}' version '{version}'")
        
        try:
            # Save preprocessor
            preprocessor_file = pipeline_dir / "preprocessor.joblib"
            joblib.dump(preprocessor, preprocessor_file, compress=3)
            
            # Save feature engineer
            feature_engineer_file = pipeline_dir / "feature_engineer.joblib"
            joblib.dump(feature_engineer, feature_engineer_file, compress=3)
            
            # Extract preprocessing steps information
            preprocessing_steps = []
            feature_engineering_steps = []
            categorical_encoders = {}
            numerical_transformers = {}
            
            # Extract preprocessor information
            if hasattr(preprocessor, 'label_encoders'):
                for col, encoder in preprocessor.label_encoders.items():
                    categorical_encoders[col] = 'label_encoder'
                    preprocessing_steps.append(f"label_encode_{col}")
            
            if hasattr(preprocessor, 'onehot_encoders'):
                for col, encoder in preprocessor.onehot_encoders.items():
                    categorical_encoders[col] = 'onehot_encoder'
                    preprocessing_steps.append(f"onehot_encode_{col}")
            
            if hasattr(preprocessor, 'imputers'):
                for col, imputer in preprocessor.imputers.items():
                    numerical_transformers[col] = f"imputer_{imputer.strategy}"
                    preprocessing_steps.append(f"impute_{col}")
            
            # Add common preprocessing steps
            preprocessing_steps.extend([
                "parse_timestamps",
                "transform_amounts",
                "handle_missing_values"
            ])
            
            # Add feature engineering steps
            feature_engineering_steps.extend([
                "create_time_features",
                "compute_sender_behavior",
                "compute_receiver_risk",
                "detect_anomalies",
                "compute_interaction_features"
            ])
            
            # Extract data schema if training data provided
            data_schema = None
            training_data_hash = None
            if training_data is not None:
                data_schema = self._extract_data_schema(training_data)
                training_data_hash = hashlib.sha256(
                    pd.util.hash_pandas_object(training_data).values
                ).hexdigest()
                
                # Save data schema
                schema_file = pipeline_dir / "data_schema.json"
                with open(schema_file, 'w') as f:
                    json.dump(data_schema, f, indent=2)
            
            # Generate preprocessing hash
            preprocessing_hash = self._generate_preprocessing_hash(preprocessor, feature_engineer)
            
            # Create or update metadata
            if metadata is None:
                metadata = PipelineMetadata(
                    pipeline_name=pipeline_name,
                    pipeline_type='fraud_detection_preprocessing',
                    version=version,
                    created_at=datetime.now().isoformat(),
                    preprocessing_steps=preprocessing_steps,
                    feature_engineering_steps=feature_engineering_steps,
                    categorical_encoders=categorical_encoders,
                    numerical_transformers=numerical_transformers,
                    data_schema=data_schema,
                    preprocessing_hash=preprocessing_hash,
                    training_data_hash=training_data_hash,
                    tags=['preprocessing', 'feature_engineering', 'fraud_detection'],
                    description=f"Complete preprocessing pipeline for fraud detection with {len(preprocessing_steps)} preprocessing steps"
                )
            else:
                # Update metadata with calculated values
                metadata.preprocessing_hash = preprocessing_hash
                metadata.training_data_hash = training_data_hash
                metadata.created_at = datetime.now().isoformat()
            
            # Save metadata
            metadata_file = pipeline_dir / "pipeline_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
            
            # Perform consistency validation if requested
            consistency_checks = {}
            if validate_consistency and training_data is not None:
                consistency_checks = self._validate_pipeline_consistency(
                    preprocessor, feature_engineer, training_data
                )
                
                # Save consistency check results
                consistency_file = pipeline_dir / "consistency_checks.json"
                with open(consistency_file, 'w') as f:
                    json.dump(consistency_checks, f, indent=2, default=str)
            
            # Update registry
            registry = self._load_registry()
            if pipeline_name not in registry["pipelines"]:
                registry["pipelines"][pipeline_name] = {}
            
            registry["pipelines"][pipeline_name][version] = {
                "pipeline_type": "fraud_detection_preprocessing",
                "created_at": metadata.created_at,
                "preprocessor_path": str(preprocessor_file.relative_to(self.base_dir)),
                "feature_engineer_path": str(feature_engineer_file.relative_to(self.base_dir)),
                "metadata_path": str(metadata_file.relative_to(self.base_dir)),
                "schema_path": str((pipeline_dir / "data_schema.json").relative_to(self.base_dir)) if data_schema else None,
                "preprocessing_hash": preprocessing_hash,
                "consistency_validated": validate_consistency,
                "consistency_checks": consistency_checks
            }
            
            self._save_registry(registry)
            
            logger.info(f"Pipeline saved successfully: {pipeline_name} v{version}")
            return str(pipeline_dir)
            
        except Exception as e:
            logger.error(f"Failed to save pipeline {pipeline_name}: {e}")
            # Clean up partial files
            if pipeline_dir.exists():
                shutil.rmtree(pipeline_dir, ignore_errors=True)
            raise IOError(f"Failed to save pipeline: {e}")
    
    def load_pipeline(self,
                     pipeline_name: str,
                     version: Optional[str] = None,
                     validate_consistency: bool = True) -> Tuple[Any, Any, PipelineMetadata]:
        """
        Load a complete preprocessing pipeline with consistency validation.
        
        Args:
            pipeline_name: Name of the pipeline to load
            version: Specific version to load (loads latest if not specified)
            validate_consistency: Whether to validate pipeline consistency
            
        Returns:
            Tuple of (preprocessor, feature_engineer, metadata)
            
        Raises:
            FileNotFoundError: If pipeline is not found
            IOError: If loading fails
        """
        registry = self._load_registry()
        
        if pipeline_name not in registry["pipelines"]:
            raise FileNotFoundError(f"Pipeline '{pipeline_name}' not found in registry")
        
        pipeline_versions = registry["pipelines"][pipeline_name]
        
        # Determine version to load
        if version is None:
            version = max(pipeline_versions.keys(), key=lambda v: pipeline_versions[v]["created_at"])
            logger.info(f"Loading latest version of {pipeline_name}: {version}")
        
        if version not in pipeline_versions:
            available_versions = list(pipeline_versions.keys())
            raise FileNotFoundError(f"Version '{version}' of pipeline '{pipeline_name}' not found. "
                                  f"Available versions: {available_versions}")
        
        pipeline_info = pipeline_versions[version]
        
        try:
            # Load preprocessor
            preprocessor_file = self.base_dir / pipeline_info["preprocessor_path"]
            preprocessor = joblib.load(preprocessor_file)
            
            # Load feature engineer
            feature_engineer_file = self.base_dir / pipeline_info["feature_engineer_path"]
            feature_engineer = joblib.load(feature_engineer_file)
            
            # Load metadata
            metadata_file = self.base_dir / pipeline_info["metadata_path"]
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                metadata = PipelineMetadata.from_dict(metadata_dict)
            
            # Validate consistency if requested
            if validate_consistency:
                # Check if components are properly fitted
                if not getattr(preprocessor, 'is_fitted', False):
                    logger.warning(f"Loaded preprocessor for {pipeline_name} v{version} is not fitted")
                
                if not getattr(feature_engineer, 'is_fitted', False):
                    logger.warning(f"Loaded feature engineer for {pipeline_name} v{version} is not fitted")
                
                # Verify preprocessing hash if available
                current_hash = self._generate_preprocessing_hash(preprocessor, feature_engineer)
                stored_hash = pipeline_info.get("preprocessing_hash")
                
                if stored_hash and current_hash != stored_hash and stored_hash != "unknown":
                    logger.warning(f"Preprocessing hash mismatch for {pipeline_name} v{version}")
            
            logger.info(f"Pipeline loaded successfully: {pipeline_name} v{version}")
            return preprocessor, feature_engineer, metadata
            
        except Exception as e:
            logger.error(f"Failed to load pipeline {pipeline_name} v{version}: {e}")
            raise IOError(f"Failed to load pipeline: {e}")
    
    def _validate_pipeline_consistency(self, preprocessor: Any, feature_engineer: Any, 
                                     test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate pipeline consistency by testing on sample data.
        
        Args:
            preprocessor: Fitted preprocessor
            feature_engineer: Fitted feature engineer
            test_data: Sample data for testing
            
        Returns:
            Dictionary containing consistency check results
        """
        consistency_checks = {
            'timestamp': datetime.now().isoformat(),
            'test_data_shape': test_data.shape,
            'checks_performed': [],
            'all_checks_passed': True,
            'issues_found': []
        }
        
        try:
            # Test preprocessing
            logger.debug("Testing preprocessing consistency")
            test_sample = test_data.head(100).copy()  # Use small sample for testing
            
            # Test preprocessor
            try:
                processed_data = preprocessor.transform(test_sample)
                consistency_checks['checks_performed'].append('preprocessor_transform')
                consistency_checks['preprocessor_output_shape'] = processed_data.shape
            except Exception as e:
                consistency_checks['all_checks_passed'] = False
                consistency_checks['issues_found'].append(f"Preprocessor transform failed: {str(e)}")
            
            # Test feature engineer
            try:
                if 'processed_data' in locals():
                    engineered_data = feature_engineer.transform(processed_data)
                    consistency_checks['checks_performed'].append('feature_engineer_transform')
                    consistency_checks['feature_engineer_output_shape'] = engineered_data.shape
                else:
                    # Try with original data if preprocessing failed
                    engineered_data = feature_engineer.transform(test_sample)
                    consistency_checks['checks_performed'].append('feature_engineer_transform_direct')
                    consistency_checks['feature_engineer_output_shape'] = engineered_data.shape
            except Exception as e:
                consistency_checks['all_checks_passed'] = False
                consistency_checks['issues_found'].append(f"Feature engineer transform failed: {str(e)}")
            
            # Test end-to-end pipeline
            try:
                if 'processed_data' in locals() and 'engineered_data' in locals():
                    consistency_checks['checks_performed'].append('end_to_end_pipeline')
                    consistency_checks['pipeline_successful'] = True
                else:
                    consistency_checks['pipeline_successful'] = False
            except Exception as e:
                consistency_checks['all_checks_passed'] = False
                consistency_checks['issues_found'].append(f"End-to-end pipeline failed: {str(e)}")
            
        except Exception as e:
            consistency_checks['all_checks_passed'] = False
            consistency_checks['issues_found'].append(f"Consistency validation failed: {str(e)}")
        
        return consistency_checks
    
    def validate_inference_data(self, pipeline_name: str, version: str, 
                               inference_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate inference data against the pipeline's expected schema.
        
        Args:
            pipeline_name: Name of the pipeline
            version: Version of the pipeline
            inference_data: Data to validate
            
        Returns:
            Dictionary containing validation results
        """
        registry = self._load_registry()
        
        if pipeline_name not in registry["pipelines"] or version not in registry["pipelines"][pipeline_name]:
            raise FileNotFoundError(f"Pipeline '{pipeline_name}' version '{version}' not found")
        
        pipeline_info = registry["pipelines"][pipeline_name][version]
        
        # Load expected schema if available
        schema_path = pipeline_info.get("schema_path")
        if not schema_path:
            return {
                'validation_performed': False,
                'reason': 'No schema available for validation'
            }
        
        schema_file = self.base_dir / schema_path
        if not schema_file.exists():
            return {
                'validation_performed': False,
                'reason': 'Schema file not found'
            }
        
        with open(schema_file, 'r') as f:
            expected_schema = json.load(f)
        
        # Validate schema
        validation_results = self._validate_data_schema(inference_data, expected_schema)
        validation_results['validation_performed'] = True
        validation_results['expected_schema'] = expected_schema
        validation_results['inference_data_shape'] = inference_data.shape
        
        return validation_results
    
    def list_pipelines(self) -> Dict[str, List[str]]:
        """
        List all available pipelines and their versions.
        
        Returns:
            Dictionary mapping pipeline names to lists of available versions
        """
        registry = self._load_registry()
        return {name: list(versions.keys()) for name, versions in registry["pipelines"].items()}
    
    def get_pipeline_info(self, pipeline_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            version: Specific version (latest if not specified)
            
        Returns:
            Dictionary containing pipeline information
        """
        registry = self._load_registry()
        
        if pipeline_name not in registry["pipelines"]:
            raise FileNotFoundError(f"Pipeline '{pipeline_name}' not found")
        
        pipeline_versions = registry["pipelines"][pipeline_name]
        
        if version is None:
            version = max(pipeline_versions.keys(), key=lambda v: pipeline_versions[v]["created_at"])
        
        if version not in pipeline_versions:
            raise FileNotFoundError(f"Version '{version}' not found")
        
        pipeline_info = pipeline_versions[version].copy()
        
        # Load metadata if available
        metadata_path = self.base_dir / pipeline_info["metadata_path"]
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                pipeline_info["metadata"] = metadata
        
        return pipeline_info
    
    def delete_pipeline(self, pipeline_name: str, version: Optional[str] = None) -> None:
        """
        Delete a pipeline and its associated files.
        
        Args:
            pipeline_name: Name of the pipeline to delete
            version: Specific version to delete (deletes all versions if not specified)
        """
        registry = self._load_registry()
        
        if pipeline_name not in registry["pipelines"]:
            raise FileNotFoundError(f"Pipeline '{pipeline_name}' not found")
        
        if version is None:
            # Delete all versions
            pipeline_dir = self.pipelines_dir / pipeline_name
            if pipeline_dir.exists():
                shutil.rmtree(pipeline_dir)
            del registry["pipelines"][pipeline_name]
            logger.info(f"Deleted all versions of pipeline '{pipeline_name}'")
        else:
            # Delete specific version
            if version not in registry["pipelines"][pipeline_name]:
                raise FileNotFoundError(f"Version '{version}' not found")
            
            version_dir = self.pipelines_dir / pipeline_name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            del registry["pipelines"][pipeline_name][version]
            
            # If no versions left, remove the pipeline entry
            if not registry["pipelines"][pipeline_name]:
                del registry["pipelines"][pipeline_name]
                pipeline_dir = self.pipelines_dir / pipeline_name
                if pipeline_dir.exists():
                    shutil.rmtree(pipeline_dir)
            
            logger.info(f"Deleted pipeline '{pipeline_name}' version '{version}'")
        
        self._save_registry(registry)


# Global instance for easy access
pipeline_persistence = PipelinePersistenceManager()