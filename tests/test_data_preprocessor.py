"""
Tests for the DataPreprocessor class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'transaction_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'timestamp': [
                '2023-08-22T09:22:43.516168',
                '2023-08-04T01:58:02.606711',
                '2023-05-12 11:39:33.742963',  # Different format
                '2023-10-10T06:04:43',         # No microseconds
                '2023-12-01T15:30:00.000000'
            ],
            'amount': [343.78, 419.65, 2773.86, 1666.22, 100.0],  # No zero for this test
            'transaction_type': ['withdrawal', 'withdrawal', 'deposit', 'deposit', 'transfer'],
            'location': ['Tokyo', 'Toronto', 'London', 'Sydney', 'Paris'],  # No missing values initially
            'is_fraud': [False, False, False, False, True]
        })
    
    def test_timestamp_parsing_basic(self):
        """Test basic timestamp parsing functionality."""
        df = self.sample_data.copy()
        result = self.preprocessor.parse_timestamps(df)
        
        # Check that timestamps were parsed
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        
        # Check that time components were extracted
        expected_components = ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 
                              'weekend_flag', 'is_business_hours', 'is_night_time']
        for component in expected_components:
            assert component in result.columns
        
        # Check specific values
        assert result.iloc[0]['hour'] == 9
        assert result.iloc[0]['weekend_flag'] == 0  # Tuesday
        assert result.iloc[0]['is_business_hours'] == 1
    
    def test_timestamp_parsing_missing_column(self):
        """Test error handling for missing timestamp column."""
        df = self.sample_data.drop(columns=['timestamp'])
        
        with pytest.raises(ValueError, match="Timestamp column 'timestamp' not found"):
            self.preprocessor.parse_timestamps(df)
    
    def test_categorical_encoding_label(self):
        """Test label encoding of categorical variables."""
        df = self.sample_data.copy()
        result = self.preprocessor.encode_categoricals(
            df, 
            categorical_columns=['transaction_type', 'location'],
            encoding_type='label'
        )
        
        # Check that categorical columns were encoded
        assert result['transaction_type'].dtype in ['int32', 'int64']
        assert result['location'].dtype in ['int32', 'int64']
        
        # Check that missing values were handled
        assert not result['location'].isna().any()
    
    def test_categorical_encoding_onehot(self):
        """Test one-hot encoding of categorical variables."""
        df = self.sample_data.copy()
        result = self.preprocessor.encode_categoricals(
            df, 
            categorical_columns=['transaction_type'],
            encoding_type='onehot'
        )
        
        # Check that one-hot columns were created
        onehot_cols = [col for col in result.columns if col.startswith('transaction_type_')]
        assert len(onehot_cols) > 0
        
        # Check that original column was removed
        assert 'transaction_type' not in result.columns
    
    def test_amount_transformation(self):
        """Test log transformation of amount columns."""
        # Create test data with zero amount for this specific test
        df = self.sample_data.copy()
        df.loc[4, 'amount'] = 0.0  # Add zero amount
        
        result = self.preprocessor.transform_amounts(df, amount_columns=['amount'])
        
        # Check that transformation was applied
        assert 'amount_original' in result.columns
        
        # Check that zero values were handled
        assert (result['amount'] > 0).all()
        
        # Check monotonicity: if original A < B, then log(A) < log(B)
        original_amounts = result['amount_original'].values
        log_amounts = result['amount'].values
        
        for i in range(len(original_amounts)):
            for j in range(i + 1, len(original_amounts)):
                if original_amounts[i] < original_amounts[j]:
                    assert log_amounts[i] < log_amounts[j], f"Monotonicity violated: {original_amounts[i]} vs {original_amounts[j]}"
    
    def test_missing_value_handling(self):
        """Test missing value imputation."""
        df = self.sample_data.copy()
        
        # Add some missing values
        df.loc[1, 'amount'] = np.nan
        df.loc[4, 'location'] = np.nan  # Use np.nan instead of None
        
        result = self.preprocessor.handle_missing_values(df)
        
        # Check that missing values were handled
        assert not result.isna().any().any()
    
    def test_full_pipeline(self):
        """Test the complete preprocessing pipeline."""
        df = self.sample_data.copy()
        result = self.preprocessor.fit_transform(df)
        
        # Check that all steps were applied
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        assert 'hour' in result.columns
        assert not result.isna().any().any()
        
        # Check that preprocessor is marked as fitted
        assert self.preprocessor.is_fitted
    
    def test_transform_unfitted_error(self):
        """Test error when trying to transform without fitting."""
        df = self.sample_data.copy()
        
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            self.preprocessor.transform(df)


if __name__ == '__main__':
    # Run a simple test
    test_instance = TestDataPreprocessor()
    test_instance.setup_method()
    
    print("Testing timestamp parsing...")
    test_instance.test_timestamp_parsing_basic()
    print("✓ Timestamp parsing test passed")
    
    print("Testing categorical encoding...")
    test_instance.test_categorical_encoding_label()
    print("✓ Categorical encoding test passed")
    
    print("Testing amount transformation...")
    test_instance.test_amount_transformation()
    print("✓ Amount transformation test passed")
    
    print("Testing full pipeline...")
    test_instance.test_full_pipeline()
    print("✓ Full pipeline test passed")
    
    print("\nAll tests passed! ✓")