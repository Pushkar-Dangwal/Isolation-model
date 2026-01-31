#!/usr/bin/env python3
"""
Script to fix all instances of feature matrix preparation in fraud_detector.py
"""

def fix_fraud_detector():
    with open('src/fraud_detector.py', 'r') as f:
        content = f.read()
    
    # Define the old pattern to replace
    old_pattern = """            exclude_columns = [target_column, transaction_id_column]
            feature_columns = [col for col in df_features.columns if col not in exclude_columns]
            X = df_features[feature_columns].values
            y = df_features[target_column].values"""
    
    # Define the new pattern
    new_pattern = """            exclude_columns = [target_column, transaction_id_column]
            
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
            y = df_features[target_column].values"""
    
    # Replace all instances
    new_content = content.replace(old_pattern, new_pattern)
    
    # Write back to file
    with open('src/fraud_detector.py', 'w') as f:
        f.write(new_content)
    
    print("Fixed all instances of feature matrix preparation")

if __name__ == "__main__":
    fix_fraud_detector()