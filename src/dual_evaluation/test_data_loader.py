"""
Unit tests for DataLoader component.

This module tests the DataLoader functionality including:
- CSV loading with validation
- Balanced dataset creation
- Time-based splitting
- Stratified splitting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from .data_loader import DataLoader, DataValidationError


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance for testing."""
        return DataLoader(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 10000
        n_fraud = 1200  # Sufficient fraud samples (> 1000)
        
        # Create timestamps
        base_time = datetime(2023, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(n_samples)]
        
        # Create fraud labels
        is_fraud = np.array([1] * n_fraud + [0] * (n_samples - n_fraud))
        np.random.shuffle(is_fraud)
        
        df = pd.DataFrame({
            'transaction_id': [f'T{i:06d}' for i in range(n_samples)],
            'timestamp': timestamps,
            'sender_account': [f'A{i%1000:04d}' for i in range(n_samples)],
            'receiver_account': [f'B{i%1000:04d}' for i in range(n_samples)],
            'amount': np.random.uniform(10, 10000, n_samples).astype('float32'),
            'transaction_type': np.random.choice(['transfer', 'payment', 'withdrawal'], n_samples),
            'merchant_category': np.random.choice(['retail', 'food', 'travel'], n_samples),
            'location': np.random.choice(['US', 'UK', 'CA'], n_samples),
            'device_used': np.random.choice(['mobile', 'web', 'atm'], n_samples),
            'is_fraud': is_fraud.astype('int8'),
            'fraud_type': ['none'] * n_samples,
            'time_since_last_transaction': np.random.uniform(0, 1000, n_samples).astype('float32'),
            'spending_deviation_score': np.random.uniform(0, 1, n_samples).astype('float32'),
            'velocity_score': np.random.uniform(0, 1, n_samples).astype('float32'),
            'geo_anomaly_score': np.random.uniform(0, 1, n_samples).astype('float32'),
            'payment_channel': np.random.choice(['online', 'pos', 'atm'], n_samples),
            'ip_address': [f'192.168.{i%256}.{i%256}' for i in range(n_samples)],
            'device_hash': [f'D{i%10000:04d}' for i in range(n_samples)]
        })
        
        return df
    
    def test_initialization(self, data_loader):
        """Test DataLoader initialization."""
        assert data_loader.random_state == 42
    
    def test_load_full_dataset_success(self, data_loader, sample_data):
        """Test successful dataset loading from CSV."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Load dataset
            df = data_loader.load_full_dataset(temp_path)
            
            # Verify loaded correctly
            assert len(df) == len(sample_data)
            assert 'transaction_id' in df.columns
            assert 'timestamp' in df.columns
            assert 'is_fraud' in df.columns
            assert df['is_fraud'].dtype == np.int8
            
        finally:
            os.unlink(temp_path)
    
    def test_load_full_dataset_missing_file(self, data_loader):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_full_dataset('nonexistent.csv')
    
    def test_load_full_dataset_path_traversal(self, data_loader):
        """Test path traversal attempt raises ValueError."""
        with pytest.raises(ValueError, match="Path traversal"):
            data_loader.load_full_dataset('../../../etc/passwd')
    
    def test_load_full_dataset_missing_columns(self, data_loader):
        """Test loading CSV with missing required columns raises ValueError."""
        # Create CSV without required columns
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="missing required columns"):
                data_loader.load_full_dataset(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_create_balanced_dataset_success(self, data_loader, sample_data):
        """Test successful balanced dataset creation."""
        balanced_df = data_loader.create_balanced_dataset(sample_data)
        
        # Verify fraud rate is 50% ± 1%
        fraud_rate = balanced_df['is_fraud'].mean()
        assert abs(fraud_rate - 0.5) < 0.01
        
        # Verify all fraud samples included
        original_fraud_count = sample_data['is_fraud'].sum()
        balanced_fraud_count = balanced_df['is_fraud'].sum()
        assert balanced_fraud_count == original_fraud_count
        
        # Verify equal fraud and legitimate samples
        fraud_count = (balanced_df['is_fraud'] == 1).sum()
        legit_count = (balanced_df['is_fraud'] == 0).sum()
        assert fraud_count == legit_count
        
        # Verify no duplicates
        assert not balanced_df.duplicated().any()
    
    def test_create_balanced_dataset_insufficient_fraud(self, data_loader):
        """Test balanced dataset creation with insufficient fraud samples."""
        # Create dataset with too few fraud samples
        df = pd.DataFrame({
            'transaction_id': [f'T{i:06d}' for i in range(1500)],
            'is_fraud': [1] * 500 + [0] * 1000  # Only 500 fraud samples
        })
        
        with pytest.raises(DataValidationError, match="Insufficient fraud samples"):
            data_loader.create_balanced_dataset(df)
    
    def test_create_balanced_dataset_missing_column(self, data_loader, sample_data):
        """Test balanced dataset creation with missing fraud column."""
        df = sample_data.drop(columns=['is_fraud'])
        
        with pytest.raises(ValueError, match="not found in DataFrame"):
            data_loader.create_balanced_dataset(df)
    
    def test_create_balanced_dataset_shuffling(self, data_loader, sample_data):
        """Test that balanced dataset is properly shuffled."""
        balanced_df = data_loader.create_balanced_dataset(sample_data)
        
        # Check that samples are not sorted by class
        # If properly shuffled, first half should not be all one class
        first_half = balanced_df.iloc[:len(balanced_df)//2]
        first_half_fraud_rate = first_half['is_fraud'].mean()
        
        # Should be close to 0.5, not 0 or 1
        assert 0.2 < first_half_fraud_rate < 0.8
    
    def test_time_based_split_success(self, data_loader, sample_data):
        """Test successful time-based split."""
        train_df, test_df = data_loader.time_based_split(sample_data, test_size=0.2)
        
        # Verify split sizes
        assert len(train_df) + len(test_df) == len(sample_data)
        test_proportion = len(test_df) / len(sample_data)
        assert abs(test_proportion - 0.2) < 0.01
        
        # Verify no temporal overlap
        train_max_time = train_df['timestamp'].max()
        test_min_time = test_df['timestamp'].min()
        assert train_max_time < test_min_time
    
    def test_time_based_split_missing_timestamp(self, data_loader, sample_data):
        """Test time-based split with missing timestamp column."""
        df = sample_data.drop(columns=['timestamp'])
        
        with pytest.raises(ValueError, match="Timestamp column.*not found"):
            data_loader.time_based_split(df)
    
    def test_time_based_split_invalid_test_size(self, data_loader, sample_data):
        """Test time-based split with invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            data_loader.time_based_split(sample_data, test_size=1.5)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            data_loader.time_based_split(sample_data, test_size=-0.1)
    
    def test_stratified_split_success(self, data_loader, sample_data):
        """Test successful stratified split."""
        train_df, test_df = data_loader.stratified_split(sample_data, test_size=0.2)
        
        # Verify split sizes
        assert len(train_df) + len(test_df) == len(sample_data)
        test_proportion = len(test_df) / len(sample_data)
        assert abs(test_proportion - 0.2) < 0.01
        
        # Verify fraud rates maintained
        original_fraud_rate = sample_data['is_fraud'].mean()
        train_fraud_rate = train_df['is_fraud'].mean()
        test_fraud_rate = test_df['is_fraud'].mean()
        
        assert abs(train_fraud_rate - original_fraud_rate) < 0.01
        assert abs(test_fraud_rate - original_fraud_rate) < 0.01
        
        # Verify no overlapping indices
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        assert len(train_indices.intersection(test_indices)) == 0
    
    def test_stratified_split_missing_column(self, data_loader, sample_data):
        """Test stratified split with missing stratify column."""
        df = sample_data.drop(columns=['is_fraud'])
        
        with pytest.raises(ValueError, match="not found in DataFrame"):
            data_loader.stratified_split(df)
    
    def test_stratified_split_invalid_test_size(self, data_loader, sample_data):
        """Test stratified split with invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            data_loader.stratified_split(sample_data, test_size=1.5)
    
    def test_reproducibility_balanced_dataset(self, sample_data):
        """Test that balanced dataset creation is reproducible with same random_state."""
        loader1 = DataLoader(random_state=42)
        loader2 = DataLoader(random_state=42)
        
        balanced1 = loader1.create_balanced_dataset(sample_data)
        balanced2 = loader2.create_balanced_dataset(sample_data)
        
        # Should produce identical results
        pd.testing.assert_frame_equal(balanced1, balanced2)
    
    def test_reproducibility_stratified_split(self, sample_data):
        """Test that stratified split is reproducible with same random_state."""
        loader1 = DataLoader(random_state=42)
        loader2 = DataLoader(random_state=42)
        
        train1, test1 = loader1.stratified_split(sample_data)
        train2, test2 = loader2.stratified_split(sample_data)
        
        # Should produce identical results
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
    
    def test_edge_case_small_dataset(self, data_loader):
        """Test handling of small dataset."""
        # Create minimal dataset
        df = pd.DataFrame({
            'transaction_id': [f'T{i:06d}' for i in range(2000)],
            'timestamp': pd.date_range('2023-01-01', periods=2000, freq='H'),
            'is_fraud': [1] * 1000 + [0] * 1000
        })
        
        # Should work for balanced dataset
        balanced_df = data_loader.create_balanced_dataset(df)
        assert len(balanced_df) == 2000
        
        # Should work for splits
        train, test = data_loader.time_based_split(df, test_size=0.2)
        assert len(train) + len(test) == 2000
    
    def test_edge_case_perfect_balance(self, data_loader):
        """Test balanced dataset creation when input is already balanced."""
        df = pd.DataFrame({
            'transaction_id': [f'T{i:06d}' for i in range(2000)],
            'is_fraud': [1] * 1000 + [0] * 1000
        })
        
        balanced_df = data_loader.create_balanced_dataset(df)
        
        # Should still have 50/50 balance
        fraud_rate = balanced_df['is_fraud'].mean()
        assert abs(fraud_rate - 0.5) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
