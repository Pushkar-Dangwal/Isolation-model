"""
Tests for the FeatureEngineer class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_engineer = FeatureEngineer()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'transaction_id': ['T1', 'T2', 'T3', 'T4', 'T5'],
            'timestamp': [
                '2023-08-22T09:22:43.516168',
                '2023-08-22T10:58:02.606711',
                '2023-08-22T11:39:33.742963',
                '2023-08-22T15:04:43.000000',
                '2023-08-23T02:30:00.000000'
            ],
            'sender_account': ['S1', 'S1', 'S2', 'S1', 'S2'],
            'receiver_account': ['R1', 'R2', 'R1', 'R1', 'R3'],
            'amount': [100.0, 200.0, 150.0, 300.0, 50.0],
            'location': ['Tokyo', 'Tokyo', 'London', 'Paris', 'London'],
            'device_used': ['mobile', 'desktop', 'mobile', 'mobile', 'desktop'],
            'is_fraud': [False, False, False, False, True]
        })
        
        # Convert timestamp to datetime
        self.sample_data['timestamp'] = pd.to_datetime(self.sample_data['timestamp'])
    
    def test_create_time_features(self):
        """Test time feature creation."""
        df = self.sample_data.copy()
        result = self.feature_engineer.create_time_features(df)
        
        # Check that time features were created
        expected_features = ['hour', 'day_of_week', 'weekend_flag', 'is_business_hours']
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check specific values
        assert result.iloc[0]['hour'] == 9
        assert result.iloc[0]['weekend_flag'] == 0  # Tuesday
        assert result.iloc[0]['is_business_hours'] == 1
        assert result.iloc[4]['is_business_hours'] == 0  # 2:30 AM
    
    def test_compute_sender_behavior(self):
        """Test sender behavior feature computation."""
        df = self.sample_data.copy()
        result = self.feature_engineer.compute_sender_behavior(df)
        
        # Check that sender behavior features were created
        expected_features = ['tx_count_last_1h', 'tx_count_last_24h', 'total_amount_last_24h']
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check that features are numeric
        for feature in expected_features:
            assert pd.api.types.is_numeric_dtype(result[feature])
    
    def test_compute_receiver_risk(self):
        """Test receiver risk feature computation."""
        df = self.sample_data.copy()
        result = self.feature_engineer.compute_receiver_risk(df)
        
        # Check that receiver risk features were created
        expected_features = ['receiver_tx_count', 'receiver_fraud_rate']
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check that features are numeric
        for feature in expected_features:
            assert pd.api.types.is_numeric_dtype(result[feature])
    
    def test_detect_anomalies(self):
        """Test anomaly detection features."""
        df = self.sample_data.copy()
        result = self.feature_engineer.detect_anomalies(df)
        
        # Check that anomaly features were created
        expected_features = ['new_location_flag', 'new_device_flag']
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check that first transaction for each sender has new location/device flags
        # First transaction for S1 should have new flags
        s1_first = result[result['sender_account'] == 'S1'].iloc[0]
        assert s1_first['new_location_flag'] == 1
        assert s1_first['new_device_flag'] == 1
    
    def test_compute_interaction_features(self):
        """Test interaction feature computation."""
        df = self.sample_data.copy()
        result = self.feature_engineer.compute_interaction_features(df)
        
        # Check that interaction features were created
        expected_features = ['sender_receiver_frequency', 'sender_total_receivers']
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check that features are numeric
        for feature in expected_features:
            assert pd.api.types.is_numeric_dtype(result[feature])
    
    def test_fit_transform_pipeline(self):
        """Test the complete feature engineering pipeline."""
        df = self.sample_data.copy()
        result = self.feature_engineer.fit_transform(df)
        
        # Check that all feature categories were created
        time_features = ['hour', 'day_of_week', 'weekend_flag']
        behavior_features = ['tx_count_last_1h', 'tx_count_last_24h']
        risk_features = ['receiver_tx_count', 'receiver_fraud_rate']
        anomaly_features = ['new_location_flag', 'new_device_flag']
        interaction_features = ['sender_receiver_frequency', 'sender_total_receivers']
        
        all_expected_features = time_features + behavior_features + risk_features + anomaly_features + interaction_features
        
        for feature in all_expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check that feature engineer is marked as fitted
        assert self.feature_engineer.is_fitted
        
        # Check that no NaN values exist in key features
        for feature in all_expected_features:
            assert not result[feature].isna().any(), f"Feature {feature} contains NaN values"
    
    def test_missing_columns_error(self):
        """Test error handling for missing required columns."""
        df = self.sample_data.drop(columns=['sender_account'])
        
        with pytest.raises(ValueError):
            self.feature_engineer.compute_sender_behavior(df)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame(columns=self.sample_data.columns)
        result = self.feature_engineer.create_time_features(empty_df)
        
        # Should return empty dataframe with additional columns
        assert len(result) == 0
        assert 'hour' in result.columns


if __name__ == '__main__':
    # Run a simple test
    test_instance = TestFeatureEngineer()
    test_instance.setup_method()
    
    print("Testing time features...")
    test_instance.test_create_time_features()
    print("✓ Time features test passed")
    
    print("Testing sender behavior...")
    test_instance.test_compute_sender_behavior()
    print("✓ Sender behavior test passed")
    
    print("Testing receiver risk...")
    test_instance.test_compute_receiver_risk()
    print("✓ Receiver risk test passed")
    
    print("Testing anomaly detection...")
    test_instance.test_detect_anomalies()
    print("✓ Anomaly detection test passed")
    
    print("Testing interaction features...")
    test_instance.test_compute_interaction_features()
    print("✓ Interaction features test passed")
    
    print("Testing full pipeline...")
    test_instance.test_fit_transform_pipeline()
    print("✓ Full pipeline test passed")
    
    print("\nAll tests passed! ✓")