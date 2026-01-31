"""
Reproducibility module for the fraud detection system.
Provides comprehensive random seed management and deterministic behavior controls.
"""

import logging
import random
import numpy as np
import os
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityState:
    """
    Captures the complete reproducibility state of the system.
    Includes random seeds, environment variables, and system configuration.
    """
    master_seed: int
    python_seed: int
    numpy_seed: int
    torch_seed: Optional[int] = None
    tf_seed: Optional[int] = None
    environment_hash: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReproducibilityState':
        """Create state from dictionary."""
        return cls(**data)


class ReproducibilityManager:
    """
    Comprehensive reproducibility manager for the fraud detection system.
    
    Manages random seeds across all libraries and frameworks to ensure
    deterministic behavior for training, testing, and debugging.
    Provides state capture and restoration capabilities.
    """
    
    def __init__(self, master_seed: Optional[int] = None):
        """
        Initialize the reproducibility manager.
        
        Args:
            master_seed: Master seed for all random number generators
        """
        self.master_seed = master_seed or MODEL_CONFIG.get('random_seed', 42)
        self.current_state: Optional[ReproducibilityState] = None
        self.seed_history: List[ReproducibilityState] = []
        
        logger.info(f"ReproducibilityManager initialized with master seed: {self.master_seed}")
    
    def set_global_seed(self, seed: Optional[int] = None, 
                       strict_mode: bool = True,
                       capture_environment: bool = True) -> ReproducibilityState:
        """
        Set random seeds for all available libraries and frameworks.
        
        This method ensures deterministic behavior across:
        - Python's built-in random module
        - NumPy random number generation
        - PyTorch (if available)
        - TensorFlow (if available)
        - Environment variables affecting randomness
        
        Args:
            seed: Seed value to use (uses master_seed if not provided)
            strict_mode: Whether to enforce strict deterministic behavior
            capture_environment: Whether to capture environment state
            
        Returns:
            ReproducibilityState object containing all seed information
        """
        if seed is None:
            seed = self.master_seed
        
        logger.info(f"Setting global random seed to {seed} (strict_mode={strict_mode})")
        
        # Set Python built-in random seed
        random.seed(seed)
        python_seed = seed
        
        # Set NumPy random seed
        np.random.seed(seed)
        numpy_seed = seed
        
        # Set PyTorch seeds if available
        torch_seed = None
        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            torch_seed = seed
            
            if strict_mode:
                # Enable deterministic algorithms (may impact performance)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # Set additional deterministic flags for newer PyTorch versions
                try:
                    torch.use_deterministic_algorithms(True)
                except AttributeError:
                    # Older PyTorch versions don't have this function
                    pass
        
        # Set TensorFlow seeds if available
        tf_seed = None
        if TF_AVAILABLE:
            tf.random.set_seed(seed)
            tf_seed = seed
            
            if strict_mode:
                # Enable deterministic operations
                os.environ['TF_DETERMINISTIC_OPS'] = '1'
                os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
        # Set environment variables for other libraries
        if strict_mode:
            # Ensure deterministic behavior in other libraries
            os.environ['PYTHONHASHSEED'] = str(seed)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For CUDA determinism
        
        # Set specific library seeds
        self._set_library_seeds(seed, strict_mode)
        
        # Capture environment state if requested
        environment_hash = None
        system_info = None
        if capture_environment:
            environment_hash = self._capture_environment_hash()
            system_info = self._capture_system_info()
        
        # Create reproducibility state
        state = ReproducibilityState(
            master_seed=seed,
            python_seed=python_seed,
            numpy_seed=numpy_seed,
            torch_seed=torch_seed,
            tf_seed=tf_seed,
            environment_hash=environment_hash,
            system_info=system_info,
            timestamp=datetime.now().isoformat()
        )
        
        # Store current state and add to history
        self.current_state = state
        self.seed_history.append(state)
        
        logger.info("Global random seed set successfully")
        return state
    
    def _set_library_seeds(self, seed: int, strict_mode: bool) -> None:
        """Set seeds for specific ML libraries."""
        try:
            # LightGBM doesn't have a global seed function, but uses random_state parameter
            # XGBoost also uses random_state parameter
            # Scikit-learn uses random_state parameter in individual estimators
            
            # Set seeds for other libraries that might be used
            import sklearn.utils
            sklearn.utils.check_random_state(seed)
            
        except ImportError:
            pass
        
        # Set additional environment variables for reproducibility
        if strict_mode:
            # OpenMP settings for consistent threading
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['NUMEXPR_NUM_THREADS'] = '1'
            
            # Disable GPU non-deterministic operations
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    def _capture_environment_hash(self) -> str:
        """Capture a hash of relevant environment variables."""
        relevant_env_vars = [
            'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'CUDA_LAUNCH_BLOCKING',
            'TF_DETERMINISTIC_OPS', 'TF_CUDNN_DETERMINISTIC',
            'CUBLAS_WORKSPACE_CONFIG'
        ]
        
        env_data = {}
        for var in relevant_env_vars:
            env_data[var] = os.environ.get(var, 'not_set')
        
        # Create hash of environment state
        env_string = json.dumps(env_data, sort_keys=True)
        return hashlib.sha256(env_string.encode()).hexdigest()
    
    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture relevant system information for reproducibility."""
        import platform
        import sys
        
        system_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'numpy_version': np.__version__,
        }
        
        # Add library versions if available
        try:
            import sklearn
            system_info['sklearn_version'] = sklearn.__version__
        except ImportError:
            pass
        
        try:
            import lightgbm
            system_info['lightgbm_version'] = lightgbm.__version__
        except ImportError:
            pass
        
        if TORCH_AVAILABLE:
            system_info['torch_version'] = torch.__version__
            system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info['cuda_version'] = torch.version.cuda
        
        if TF_AVAILABLE:
            system_info['tensorflow_version'] = tf.__version__
        
        return system_info
    
    def get_current_state(self) -> Optional[ReproducibilityState]:
        """Get the current reproducibility state."""
        return self.current_state
    
    def restore_state(self, state: ReproducibilityState, 
                     strict_mode: bool = True) -> None:
        """
        Restore a previous reproducibility state.
        
        Args:
            state: ReproducibilityState to restore
            strict_mode: Whether to enforce strict deterministic behavior
        """
        logger.info(f"Restoring reproducibility state with master seed {state.master_seed}")
        
        # Restore the global seed using the master seed from the state
        self.set_global_seed(
            seed=state.master_seed,
            strict_mode=strict_mode,
            capture_environment=False
        )
        
        # Update current state
        self.current_state = state
    
    def save_state(self, filepath: str, state: Optional[ReproducibilityState] = None) -> None:
        """
        Save reproducibility state to a file.
        
        Args:
            filepath: Path to save the state
            state: State to save (uses current state if not provided)
        """
        if state is None:
            state = self.current_state
        
        if state is None:
            raise ValueError("No state to save. Call set_global_seed() first.")
        
        with open(filepath, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Reproducibility state saved to {filepath}")
    
    def load_state(self, filepath: str) -> ReproducibilityState:
        """
        Load reproducibility state from a file.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            Loaded ReproducibilityState
        """
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        state = ReproducibilityState.from_dict(state_dict)
        logger.info(f"Reproducibility state loaded from {filepath}")
        
        return state
    
    def verify_reproducibility(self, state1: ReproducibilityState, 
                             state2: ReproducibilityState,
                             check_environment: bool = True) -> Dict[str, bool]:
        """
        Verify that two reproducibility states are equivalent.
        
        Args:
            state1: First state to compare
            state2: Second state to compare
            check_environment: Whether to check environment hash
            
        Returns:
            Dictionary indicating which aspects match
        """
        verification = {
            'master_seed_match': state1.master_seed == state2.master_seed,
            'python_seed_match': state1.python_seed == state2.python_seed,
            'numpy_seed_match': state1.numpy_seed == state2.numpy_seed,
            'torch_seed_match': state1.torch_seed == state2.torch_seed,
            'tf_seed_match': state1.tf_seed == state2.tf_seed,
        }
        
        if check_environment:
            verification['environment_match'] = state1.environment_hash == state2.environment_hash
        
        verification['all_match'] = all(verification.values())
        
        return verification
    
    def create_deterministic_context(self, seed: Optional[int] = None):
        """
        Create a context manager for deterministic execution.
        
        Args:
            seed: Seed to use for the context (uses master_seed if not provided)
            
        Returns:
            Context manager that ensures deterministic behavior
        """
        return DeterministicContext(self, seed)
    
    def get_seed_for_component(self, component_name: str, 
                              base_seed: Optional[int] = None) -> int:
        """
        Generate a deterministic seed for a specific component.
        
        This ensures different components get different but reproducible seeds
        derived from the master seed.
        
        Args:
            component_name: Name of the component
            base_seed: Base seed to use (uses master_seed if not provided)
            
        Returns:
            Deterministic seed for the component
        """
        if base_seed is None:
            base_seed = self.master_seed
        
        # Create deterministic seed based on component name and base seed
        component_hash = hashlib.sha256(component_name.encode()).hexdigest()
        component_int = int(component_hash[:8], 16)  # Use first 8 hex chars
        
        # Combine with base seed
        component_seed = (base_seed + component_int) % (2**31 - 1)
        
        logger.debug(f"Generated seed {component_seed} for component '{component_name}'")
        return component_seed
    
    def get_history(self) -> List[ReproducibilityState]:
        """Get the history of all reproducibility states."""
        return self.seed_history.copy()
    
    def clear_history(self) -> None:
        """Clear the reproducibility state history."""
        self.seed_history.clear()
        logger.info("Reproducibility state history cleared")


class DeterministicContext:
    """
    Context manager for deterministic execution.
    
    Ensures that code executed within the context has deterministic behavior
    by setting and restoring random seeds.
    """
    
    def __init__(self, manager: ReproducibilityManager, seed: Optional[int] = None):
        """
        Initialize the deterministic context.
        
        Args:
            manager: ReproducibilityManager instance
            seed: Seed to use for the context
        """
        self.manager = manager
        self.seed = seed or manager.master_seed
        self.previous_state = None
    
    def __enter__(self):
        """Enter the deterministic context."""
        # Save current state
        self.previous_state = self.manager.get_current_state()
        
        # Set deterministic seed
        self.manager.set_global_seed(self.seed, strict_mode=True)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the deterministic context."""
        # Restore previous state if it existed
        if self.previous_state:
            self.manager.restore_state(self.previous_state)


# Global reproducibility manager instance
reproducibility_manager = ReproducibilityManager()


def set_global_seed(seed: Optional[int] = None, strict_mode: bool = True) -> ReproducibilityState:
    """
    Convenience function to set global random seed.
    
    Args:
        seed: Seed value to use
        strict_mode: Whether to enforce strict deterministic behavior
        
    Returns:
        ReproducibilityState object
    """
    return reproducibility_manager.set_global_seed(seed, strict_mode)


def get_deterministic_context(seed: Optional[int] = None):
    """
    Convenience function to create a deterministic context.
    
    Args:
        seed: Seed to use for the context
        
    Returns:
        Context manager for deterministic execution
    """
    return reproducibility_manager.create_deterministic_context(seed)


def get_component_seed(component_name: str, base_seed: Optional[int] = None) -> int:
    """
    Convenience function to get a deterministic seed for a component.
    
    Args:
        component_name: Name of the component
        base_seed: Base seed to use
        
    Returns:
        Deterministic seed for the component
    """
    return reproducibility_manager.get_seed_for_component(component_name, base_seed)