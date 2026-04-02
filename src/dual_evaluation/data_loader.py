"""
DataLoader component for dual-evaluation pipeline.

This module provides functionality for loading and preparing datasets for both
imbalanced and balanced evaluation pipelines. It handles CSV loading, balanced
dataset creation, time-based splitting, and stratified splitting.

Requirements: 1.1-1.6, 2.1-2.5, 3.1-3.4, 14.1, 14.2, 17.1, 18.1-18.6, 20.1-20.3
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class DataLoader:
    """
    Loads and prepares datasets for dual-evaluation pipeline.
    
    This class provides methods for:
    - Loading full datasets efficiently
    - Creating balanced datasets with equal fraud/legitimate samples
    - Time-based splitting to prevent temporal leakage
    - Stratified splitting to maintain class balance
    
    Requirements: 1.1-1.6, 2.1-2.5, 3.1-3.4
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            random_state: Random seed for reproducibility
            
        Requirements: 20.1
        """
        self.random_state = random_state
        logger.info(f"DataLoader initialized with random_state={random_state}")
    
    def load_full_dataset(self, path: str) -> pd.DataFrame:
        """
        Load complete dataset from CSV file with efficient data types.
        
        This method loads the full 5M transaction dataset using optimized data types
        for memory efficiency and validates the CSV structure.
        
        Args:
            path: Path to CSV file
            
        Returns:
            DataFrame with all transactions
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV structure is invalid or required columns missing
            
        Requirements: 1.1, 17.1, 18.1, 18.2
        """
        # Validate file path for security (prevent path traversal)
        if '..' in path or path.startswith('/'):
            raise ValueError(f"Invalid file path: {path}. Path traversal attempts not allowed.")
        
        try:
            logger.info(f"Loading dataset from {path}...")
            
            # Define efficient data types for memory optimization
            dtype_spec = {
                'transaction_id': 'object',
                'sender_account': 'object',
                'receiver_account': 'object',
                'amount': 'float32',
                'transaction_type': 'category',
                'merchant_category': 'category',
                'location': 'category',
                'device_used': 'category',
                'is_fraud': 'int8',  # Binary column
                'fraud_type': 'category',
                'time_since_last_transaction': 'float32',
                'spending_deviation_score': 'float32',
                'velocity_score': 'float32',
                'geo_anomaly_score': 'float32',
                'payment_channel': 'category',
                'ip_address': 'object',
                'device_hash': 'object'
            }
            
            # Load CSV with efficient data types
            # Try to parse timestamp if it exists, otherwise load without parsing
            try:
                df = pd.read_csv(path, dtype=dtype_spec, parse_dates=['timestamp'])
            except ValueError as e:
                if 'parse_dates' in str(e):
                    # Timestamp column doesn't exist, load without parsing
                    df = pd.read_csv(path, dtype=dtype_spec)
                else:
                    raise
            
            # Validate required columns
            required_columns = ['transaction_id', 'timestamp', 'is_fraud']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(
                    f"CSV missing required columns: {missing_columns}. "
                    f"Required columns: {required_columns}"
                )
            
            # Validate is_fraud column contains binary values
            if not df['is_fraud'].isin([0, 1]).all():
                raise ValueError("Column 'is_fraud' must contain only binary values (0 or 1)")
            
            logger.info(f"Loaded {len(df):,} transactions successfully")
            logger.info(f"Fraud rate: {df['is_fraud'].mean():.4f}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {path}")
            raise FileNotFoundError(f"Dataset file not found: {path}")
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {path}")
            raise ValueError(f"CSV file is empty: {path}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def create_balanced_dataset(
        self, 
        df: pd.DataFrame, 
        fraud_col: str = 'is_fraud'
    ) -> pd.DataFrame:
        """
        Create balanced dataset with equal fraud and legitimate samples.
        
        This method extracts ALL fraud samples and samples an equal number of
        legitimate transactions, then combines and shuffles them to create a
        50/50 balanced dataset.
        
        Args:
            df: Input DataFrame
            fraud_col: Name of fraud indicator column
            
        Returns:
            Balanced DataFrame with 50% fraud rate (±1%)
            
        Raises:
            DataValidationError: If insufficient fraud samples (< 1000)
            ValueError: If fraud_col not in DataFrame
            
        Requirements: 1.2, 1.3, 1.4, 1.5, 20.2, 20.3
        """
        # Validate fraud column exists
        if fraud_col not in df.columns:
            raise ValueError(f"Column '{fraud_col}' not found in DataFrame")
        
        # Separate fraud and legitimate samples
        fraud_samples = df[df[fraud_col] == 1].copy()
        legitimate_samples = df[df[fraud_col] == 0].copy()
        
        n_fraud = len(fraud_samples)
        n_legitimate = len(legitimate_samples)
        
        logger.info(f"Found {n_fraud:,} fraud samples and {n_legitimate:,} legitimate samples")
        
        # Validate sufficient fraud samples (Requirement 1.6)
        if n_fraud < 1000:
            raise DataValidationError(
                f"Insufficient fraud samples: {n_fraud}. Minimum required: 1000"
            )
        
        # Validate sufficient legitimate samples
        if n_legitimate < n_fraud:
            raise DataValidationError(
                f"Insufficient legitimate samples: {n_legitimate}. "
                f"Need at least {n_fraud} to match fraud samples"
            )
        
        # Sample equal number of legitimate transactions (Requirement 1.3)
        legitimate_sampled = legitimate_samples.sample(
            n=n_fraud, 
            random_state=self.random_state,
            replace=False
        )
        
        logger.info(f"Sampled {len(legitimate_sampled):,} legitimate transactions")
        
        # Combine fraud and legitimate samples (Requirement 1.2 - ALL fraud samples included)
        balanced_df = pd.concat([fraud_samples, legitimate_sampled], ignore_index=True)
        
        # Shuffle the combined dataset (Requirement 1.5)
        balanced_df = balanced_df.sample(
            frac=1.0, 
            random_state=self.random_state
        ).reset_index(drop=True)
        
        # Validate fraud rate is 50% ± 1% (Requirement 1.4)
        fraud_rate = balanced_df[fraud_col].mean()
        if abs(fraud_rate - 0.5) >= 0.01:
            logger.warning(
                f"Fraud rate {fraud_rate:.4f} is outside target range [0.49, 0.51]"
            )
        
        logger.info(f"Created balanced dataset with {len(balanced_df):,} samples")
        logger.info(f"Fraud rate: {fraud_rate:.4f}")
        
        return balanced_df
    
    def time_based_split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset chronologically to prevent temporal data leakage.
        
        This method sorts the dataset by timestamp and splits it such that all
        test data comes chronologically after all training data, preventing
        temporal leakage.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            test_size: Proportion of data for test set (0 < test_size < 1)
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            ValueError: If timestamp column missing or test_size invalid
            
        Requirements: 2.1, 2.2, 2.3, 2.4, 14.1
        """
        # Validate timestamp column exists (Requirement 2.5)
        if timestamp_col not in df.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_col}' not found in DataFrame. "
                f"Time-based split requires a valid timestamp column."
            )
        
        # Validate test_size parameter (Requirement 18.3)
        if not (0 < test_size < 1):
            raise ValueError(
                f"test_size must be between 0 and 1, got {test_size}"
            )
        
        # Sort by timestamp (Requirement 2.1)
        df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
        
        # Calculate split index (Requirement 2.3)
        split_index = int(len(df_sorted) * (1 - test_size))
        
        # Split chronologically
        train_df = df_sorted.iloc[:split_index].copy()
        test_df = df_sorted.iloc[split_index:].copy()
        
        # Verify no temporal overlap (Requirement 2.2, 2.4)
        train_max_time = train_df[timestamp_col].max()
        test_min_time = test_df[timestamp_col].min()
        
        if train_max_time >= test_min_time:
            logger.warning(
                f"Temporal overlap detected: train_max={train_max_time}, "
                f"test_min={test_min_time}"
            )
        
        # Verify test size proportion
        actual_test_size = len(test_df) / len(df_sorted)
        if abs(actual_test_size - test_size) > 0.01:
            logger.warning(
                f"Actual test size {actual_test_size:.4f} differs from "
                f"requested {test_size:.4f}"
            )
        
        logger.info(f"Time-based split: train={len(train_df):,}, test={len(test_df):,}")
        logger.info(f"Train period: {train_df[timestamp_col].min()} to {train_max_time}")
        logger.info(f"Test period: {test_min_time} to {test_df[timestamp_col].max()}")
        
        return train_df, test_df
    
    def stratified_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify_col: str = 'is_fraud'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset with stratification to maintain class balance.
        
        This method uses sklearn's stratified split to ensure the fraud rate
        is maintained in both train and test sets within 1% of the original.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set (0 < test_size < 1)
            stratify_col: Column to stratify on
            
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            ValueError: If stratify column missing or test_size invalid
            
        Requirements: 3.1, 3.2, 3.3, 3.4, 14.2
        """
        # Validate stratify column exists
        if stratify_col not in df.columns:
            raise ValueError(
                f"Stratify column '{stratify_col}' not found in DataFrame"
            )
        
        # Validate test_size parameter (Requirement 18.3)
        if not (0 < test_size < 1):
            raise ValueError(
                f"test_size must be between 0 and 1, got {test_size}"
            )
        
        # Calculate original fraud rate
        original_fraud_rate = df[stratify_col].mean()
        
        # Perform stratified split using sklearn (Requirement 3.1)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[stratify_col],
            random_state=self.random_state
        )
        
        # Calculate fraud rates in splits
        train_fraud_rate = train_df[stratify_col].mean()
        test_fraud_rate = test_df[stratify_col].mean()
        
        # Verify fraud rate maintained within 1% (Requirement 3.1)
        train_diff = abs(train_fraud_rate - original_fraud_rate)
        test_diff = abs(test_fraud_rate - original_fraud_rate)
        
        if train_diff > 0.01:
            logger.warning(
                f"Train fraud rate {train_fraud_rate:.4f} differs from original "
                f"{original_fraud_rate:.4f} by more than 1%"
            )
        
        if test_diff > 0.01:
            logger.warning(
                f"Test fraud rate {test_fraud_rate:.4f} differs from original "
                f"{original_fraud_rate:.4f} by more than 1%"
            )
        
        # Verify no overlapping indices (Requirement 3.2, 3.4)
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        overlap = train_indices.intersection(test_indices)
        
        if overlap:
            logger.error(f"Found {len(overlap)} overlapping indices between train and test")
            raise DataValidationError(
                f"Data leakage detected: {len(overlap)} samples appear in both train and test sets"
            )
        
        # Verify test size proportion (Requirement 3.3)
        actual_test_size = len(test_df) / len(df)
        if abs(actual_test_size - test_size) > 0.01:
            logger.warning(
                f"Actual test size {actual_test_size:.4f} differs from "
                f"requested {test_size:.4f}"
            )
        
        logger.info(f"Stratified split: train={len(train_df):,}, test={len(test_df):,}")
        logger.info(f"Original fraud rate: {original_fraud_rate:.4f}")
        logger.info(f"Train fraud rate: {train_fraud_rate:.4f}")
        logger.info(f"Test fraud rate: {test_fraud_rate:.4f}")
        
        return train_df, test_df
