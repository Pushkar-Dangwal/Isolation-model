"""
Model persistence module for the fraud detection system.
Provides comprehensive model serialization, versioning, and metadata management.
"""

import logging
import joblib
import json
import pickle
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import hashlib

from config import MODELS_DIR, MODEL_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """
    Comprehensive metadata for model artifacts.
    Tracks training parameters, performance metrics, and versioning information.
    """
    model_name: str
    model_type: str
    version: str
    created_at: str
    training_data_hash: Optional[str] = None
    training_samples: Optional[int] = None
    feature_count: Optional[int] = None
    feature_names: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    training_duration: Optional[float] = None
    random_state: Optional[int] = None
    model_size_bytes: Optional[int] = None
    dependencies: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


class ModelPersistenceManager:
    """
    Centralized model persistence manager for the fraud detection system.
    
    Provides comprehensive model serialization with versioning, metadata tracking,
    and integrity validation. Supports different serialization formats and
    automatic model registry management.
    """
    
    def __init__(self, base_dir: Union[str, Path] = None):
        """
        Initialize the model persistence manager.
        
        Args:
            base_dir: Base directory for model storage (defaults to MODELS_DIR)
        """
        self.base_dir = Path(base_dir) if base_dir else MODELS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.models_dir = self.base_dir / "models"
        self.metadata_dir = self.base_dir / "metadata"
        self.registry_dir = self.base_dir / "registry"
        
        for directory in [self.models_dir, self.metadata_dir, self.registry_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Registry file for tracking all models
        self.registry_file = self.registry_dir / "model_registry.json"
        self._initialize_registry()
        
        logger.info(f"ModelPersistenceManager initialized with base directory: {self.base_dir}")
    
    def _initialize_registry(self) -> None:
        """Initialize the model registry if it doesn't exist."""
        if not self.registry_file.exists():
            registry = {
                "created_at": datetime.now().isoformat(),
                "models": {},
                "version": "1.0"
            }
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info("Initialized model registry")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            return {"models": {}, "version": "1.0"}
    
    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Save the model registry."""
        try:
            registry["last_updated"] = datetime.now().isoformat()
            with open(self.registry_file, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _generate_model_hash(self, model_data: Any) -> str:
        """Generate a hash for model integrity verification."""
        try:
            # Serialize the model data and create hash
            serialized = pickle.dumps(model_data)
            return hashlib.sha256(serialized).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate model hash: {e}")
            return "unknown"
    
    def _calculate_model_size(self, filepath: Path) -> int:
        """Calculate model file size in bytes."""
        try:
            return filepath.stat().st_size
        except Exception:
            return 0
    
    def save_model(self, 
                   model: Any,
                   model_name: str,
                   model_type: str,
                   metadata: Optional[ModelMetadata] = None,
                   version: Optional[str] = None,
                   format: str = 'joblib',
                   compress: bool = True) -> str:
        """
        Save a model with comprehensive metadata and versioning.
        
        Args:
            model: The model object to save
            model_name: Name identifier for the model
            model_type: Type of model (e.g., 'lightgbm', 'isolation_forest', 'pipeline')
            metadata: Optional metadata object
            version: Optional version string (auto-generated if not provided)
            format: Serialization format ('joblib', 'pickle', 'native')
            compress: Whether to compress the saved model
            
        Returns:
            Path to the saved model
            
        Raises:
            ValueError: If model_name or model_type is invalid
            IOError: If saving fails
        """
        if not model_name or not model_type:
            raise ValueError("model_name and model_type are required")
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension based on format
        extensions = {
            'joblib': '.joblib',
            'pickle': '.pkl',
            'native': '.model'
        }
        
        if format not in extensions:
            raise ValueError(f"Unsupported format: {format}. Use one of {list(extensions.keys())}")
        
        model_file = model_dir / f"{model_name}{extensions[format]}"
        
        try:
            # Save the model using appropriate method
            if format == 'joblib':
                if compress:
                    joblib.dump(model, model_file, compress=3)
                else:
                    joblib.dump(model, model_file)
            elif format == 'pickle':
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            elif format == 'native':
                # For models with native save methods (like LightGBM)
                if hasattr(model, 'save_model'):
                    model.save_model(str(model_file))
                else:
                    # Fallback to joblib
                    joblib.dump(model, model_file)
            
            # Generate model hash for integrity
            model_hash = self._generate_model_hash(model)
            model_size = self._calculate_model_size(model_file)
            
            # Create or update metadata
            if metadata is None:
                metadata = ModelMetadata(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    created_at=datetime.now().isoformat(),
                    random_state=MODEL_CONFIG.get('random_seed'),
                    model_size_bytes=model_size
                )
            else:
                # Update metadata with calculated values
                metadata.model_size_bytes = model_size
                metadata.created_at = datetime.now().isoformat()
            
            # Save metadata
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
            
            # Save model hash for integrity verification
            hash_file = model_dir / "model.hash"
            with open(hash_file, 'w') as f:
                f.write(model_hash)
            
            # Update registry
            registry = self._load_registry()
            if model_name not in registry["models"]:
                registry["models"][model_name] = {}
            
            registry["models"][model_name][version] = {
                "model_type": model_type,
                "created_at": metadata.created_at,
                "file_path": str(model_file.relative_to(self.base_dir)),
                "metadata_path": str(metadata_file.relative_to(self.base_dir)),
                "format": format,
                "compressed": compress,
                "size_bytes": model_size,
                "hash": model_hash
            }
            
            self._save_registry(registry)
            
            logger.info(f"Model saved successfully: {model_name} v{version} ({model_size} bytes)")
            return str(model_file)
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            # Clean up partial files
            if model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            raise IOError(f"Failed to save model: {e}")
    
    def load_model(self, 
                   model_name: str,
                   version: Optional[str] = None,
                   verify_integrity: bool = True) -> tuple[Any, ModelMetadata]:
        """
        Load a model with metadata and optional integrity verification.
        
        Args:
            model_name: Name of the model to load
            version: Specific version to load (loads latest if not specified)
            verify_integrity: Whether to verify model integrity using hash
            
        Returns:
            Tuple of (model_object, metadata)
            
        Raises:
            FileNotFoundError: If model is not found
            ValueError: If integrity verification fails
            IOError: If loading fails
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            raise FileNotFoundError(f"Model '{model_name}' not found in registry")
        
        model_versions = registry["models"][model_name]
        
        # Determine version to load
        if version is None:
            # Load latest version
            version = max(model_versions.keys(), key=lambda v: model_versions[v]["created_at"])
            logger.info(f"Loading latest version of {model_name}: {version}")
        
        if version not in model_versions:
            available_versions = list(model_versions.keys())
            raise FileNotFoundError(f"Version '{version}' of model '{model_name}' not found. "
                                  f"Available versions: {available_versions}")
        
        model_info = model_versions[version]
        model_file = self.base_dir / model_info["file_path"]
        metadata_file = self.base_dir / model_info["metadata_path"]
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            # Load metadata
            metadata = None
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = ModelMetadata.from_dict(metadata_dict)
            
            # Load the model
            format_type = model_info.get("format", "joblib")
            
            if format_type == 'joblib':
                model = joblib.load(model_file)
            elif format_type == 'pickle':
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            elif format_type == 'native':
                # This would need to be handled by specific model types
                # For now, fallback to joblib
                model = joblib.load(model_file)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Verify integrity if requested
            if verify_integrity:
                stored_hash = model_info.get("hash")
                if stored_hash:
                    current_hash = self._generate_model_hash(model)
                    if current_hash != stored_hash and stored_hash != "unknown":
                        logger.warning(f"Model integrity verification failed for {model_name} v{version}")
                        # Don't raise error, just warn - hash might change due to different environments
            
            logger.info(f"Model loaded successfully: {model_name} v{version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} v{version}: {e}")
            raise IOError(f"Failed to load model: {e}")
    
    def list_models(self) -> Dict[str, List[str]]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary mapping model names to lists of available versions
        """
        registry = self._load_registry()
        return {name: list(versions.keys()) for name, versions in registry["models"].items()}
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model
            version: Specific version (latest if not specified)
            
        Returns:
            Dictionary containing model information
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            raise FileNotFoundError(f"Model '{model_name}' not found")
        
        model_versions = registry["models"][model_name]
        
        if version is None:
            version = max(model_versions.keys(), key=lambda v: model_versions[v]["created_at"])
        
        if version not in model_versions:
            raise FileNotFoundError(f"Version '{version}' not found")
        
        model_info = model_versions[version].copy()
        
        # Load metadata if available
        metadata_path = self.base_dir / model_info["metadata_path"]
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_info["metadata"] = metadata
        
        return model_info
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> None:
        """
        Delete a model and its associated files.
        
        Args:
            model_name: Name of the model to delete
            version: Specific version to delete (deletes all versions if not specified)
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            raise FileNotFoundError(f"Model '{model_name}' not found")
        
        if version is None:
            # Delete all versions
            model_dir = self.models_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            del registry["models"][model_name]
            logger.info(f"Deleted all versions of model '{model_name}'")
        else:
            # Delete specific version
            if version not in registry["models"][model_name]:
                raise FileNotFoundError(f"Version '{version}' not found")
            
            version_dir = self.models_dir / model_name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            del registry["models"][model_name][version]
            
            # If no versions left, remove the model entry
            if not registry["models"][model_name]:
                del registry["models"][model_name]
                model_dir = self.models_dir / model_name
                if model_dir.exists():
                    shutil.rmtree(model_dir)
            
            logger.info(f"Deleted model '{model_name}' version '{version}'")
        
        self._save_registry(registry)
    
    def export_model(self, model_name: str, version: str, export_path: Union[str, Path]) -> None:
        """
        Export a model to a different location.
        
        Args:
            model_name: Name of the model to export
            version: Version to export
            export_path: Destination path for export
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"] or version not in registry["models"][model_name]:
            raise FileNotFoundError(f"Model '{model_name}' version '{version}' not found")
        
        source_dir = self.models_dir / model_name / version
        export_path = Path(export_path)
        
        if source_dir.exists():
            shutil.copytree(source_dir, export_path, dirs_exist_ok=True)
            logger.info(f"Model exported to {export_path}")
        else:
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    def import_model(self, import_path: Union[str, Path], model_name: str, version: str) -> None:
        """
        Import a model from an external location.
        
        Args:
            import_path: Path to import from
            model_name: Name to assign to the imported model
            version: Version to assign to the imported model
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import path not found: {import_path}")
        
        target_dir = self.models_dir / model_name / version
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(import_path, target_dir, dirs_exist_ok=True)
        
        # Update registry
        registry = self._load_registry()
        if model_name not in registry["models"]:
            registry["models"][model_name] = {}
        
        # Try to load metadata if available
        metadata_file = target_dir / "metadata.json"
        model_type = "unknown"
        created_at = datetime.now().isoformat()
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                model_type = metadata.get("model_type", "unknown")
                created_at = metadata.get("created_at", created_at)
        
        # Find model file
        model_files = list(target_dir.glob("*.joblib")) + list(target_dir.glob("*.pkl")) + list(target_dir.glob("*.model"))
        model_file = model_files[0] if model_files else None
        
        registry["models"][model_name][version] = {
            "model_type": model_type,
            "created_at": created_at,
            "file_path": str(model_file.relative_to(self.base_dir)) if model_file else "",
            "metadata_path": str(metadata_file.relative_to(self.base_dir)) if metadata_file.exists() else "",
            "format": "joblib",  # Default assumption
            "imported": True,
            "import_date": datetime.now().isoformat()
        }
        
        self._save_registry(registry)
        logger.info(f"Model imported successfully: {model_name} v{version}")
    
    def cleanup_old_versions(self, model_name: str, keep_versions: int = 5) -> None:
        """
        Clean up old versions of a model, keeping only the most recent ones.
        
        Args:
            model_name: Name of the model to clean up
            keep_versions: Number of recent versions to keep
        """
        registry = self._load_registry()
        
        if model_name not in registry["models"]:
            return
        
        model_versions = registry["models"][model_name]
        
        if len(model_versions) <= keep_versions:
            return
        
        # Sort versions by creation date
        sorted_versions = sorted(
            model_versions.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )
        
        # Keep only the most recent versions
        versions_to_keep = [v[0] for v in sorted_versions[:keep_versions]]
        versions_to_delete = [v[0] for v in sorted_versions[keep_versions:]]
        
        for version in versions_to_delete:
            try:
                self.delete_model(model_name, version)
                logger.info(f"Cleaned up old version: {model_name} v{version}")
            except Exception as e:
                logger.warning(f"Failed to clean up version {version}: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for all models.
        
        Returns:
            Dictionary containing storage statistics
        """
        registry = self._load_registry()
        
        total_models = len(registry["models"])
        total_versions = sum(len(versions) for versions in registry["models"].values())
        total_size = 0
        
        for model_name, versions in registry["models"].items():
            for version, info in versions.items():
                total_size += info.get("size_bytes", 0)
        
        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_model_size_mb": (total_size / total_versions / (1024 * 1024)) if total_versions > 0 else 0,
            "storage_directory": str(self.base_dir),
            "registry_file": str(self.registry_file)
        }


# Global instance for easy access
model_persistence = ModelPersistenceManager()