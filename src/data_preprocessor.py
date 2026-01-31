"""
Data preprocessing module for the fraud detection system.
Handles raw transaction data cleaning and transformation.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles preprocessing of raw transaction data for fraud detection.
    
    This class transforms raw transaction data into a clean, standardized format
    suitable for machine learning by handling timestamps, categorical encoding,
    amount transformations, and missing values.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor with encoders and imputers."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.is_fitted = False
        
    def parse_timestamps(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Convert timestamp strings to datetime objects and extract time components.
        
        Handles various timestamp formats and timezone issues by attempting multiple
        parsing strategies with fallback options.
        
        Args:
            df: DataFrame containing timestamp column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataFrame with parsed timestamps and extracted time components
            
        Raises:
            ValueError: If timestamp column is missing or all timestamps are invalid
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            
        df = df.copy()
        logger.info(f"Parsing timestamps from column '{timestamp_col}'")
        
        # Track parsing success
        original_count = len(df)
        parsed_count = 0
        
        # Try multiple timestamp formats in order of likelihood
        timestamp_formats = [
            '%Y-%m-%dT%H:%M:%S.%f',  # ISO format with microseconds
            '%Y-%m-%dT%H:%M:%S',     # ISO format without microseconds
            '%Y-%m-%d %H:%M:%S.%f',  # Space-separated with microseconds
            '%Y-%m-%d %H:%M:%S',     # Space-separated without microseconds
            '%m/%d/%Y %H:%M:%S',     # US format
            '%d/%m/%Y %H:%M:%S',     # European format
        ]
        
        # Initialize the parsed timestamp column
        df[timestamp_col + '_parsed'] = pd.NaT
        
        for fmt in timestamp_formats:
            # Find rows that haven't been parsed yet
            mask = df[timestamp_col + '_parsed'].isna()
            if not mask.any():
                break
                
            try:
                # Attempt to parse remaining timestamps with current format
                parsed_timestamps = pd.to_datetime(
                    df.loc[mask, timestamp_col], 
                    format=fmt, 
                    errors='coerce'
                )
                
                # Update successfully parsed timestamps
                valid_mask = mask & ~parsed_timestamps.isna()
                df.loc[valid_mask, timestamp_col + '_parsed'] = parsed_timestamps[~parsed_timestamps.isna()]
                
                newly_parsed = valid_mask.sum()
                if newly_parsed > 0:
                    logger.debug(f"Parsed {newly_parsed} timestamps with format '{fmt}'")
                    parsed_count += newly_parsed
                    
            except Exception as e:
                logger.debug(f"Failed to parse with format '{fmt}': {e}")
                continue
        
        # Final attempt with pandas' flexible parser for remaining timestamps
        remaining_mask = df[timestamp_col + '_parsed'].isna()
        if remaining_mask.any():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    flexible_parsed = pd.to_datetime(
                        df.loc[remaining_mask, timestamp_col], 
                        errors='coerce',
                        infer_datetime_format=True
                    )
                    
                    valid_flexible = ~flexible_parsed.isna()
                    if valid_flexible.any():
                        df.loc[remaining_mask, timestamp_col + '_parsed'] = flexible_parsed
                        newly_parsed = valid_flexible.sum()
                        parsed_count += newly_parsed
                        logger.debug(f"Parsed {newly_parsed} timestamps with flexible parser")
                        
            except Exception as e:
                logger.warning(f"Flexible timestamp parsing failed: {e}")
        
        # Check parsing success rate
        final_parsed = (~df[timestamp_col + '_parsed'].isna()).sum()
        success_rate = final_parsed / original_count
        
        if success_rate == 0:
            raise ValueError("Failed to parse any timestamps. Please check timestamp format.")
        elif success_rate < 0.95:
            logger.warning(f"Only parsed {success_rate:.1%} of timestamps successfully")
        else:
            logger.info(f"Successfully parsed {success_rate:.1%} of timestamps")
        
        # Replace original timestamp column with parsed version
        df[timestamp_col] = df[timestamp_col + '_parsed']
        df.drop(columns=[timestamp_col + '_parsed'], inplace=True)
        
        # Extract time components
        df = self._extract_time_components(df, timestamp_col)
        
        return df
    
    def _extract_time_components(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Extract time components from parsed timestamps.
        
        Args:
            df: DataFrame with parsed timestamp column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataFrame with additional time component columns
        """
        logger.debug("Extracting time components from timestamps")
        
        # Extract basic time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['year'] = df[timestamp_col].dt.year
        
        # Create derived features
        df['weekend_flag'] = (df['day_of_week'] >= 5).astype(int)  # Saturday=5, Sunday=6
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Handle any remaining NaT values in extracted components
        time_columns = ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 
                       'weekend_flag', 'is_business_hours', 'is_night_time']
        
        for col in time_columns:
            if df[col].isna().any():
                # Fill with mode for categorical-like features, median for others
                if col in ['weekend_flag', 'is_business_hours', 'is_night_time']:
                    fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                else:
                    fill_value = df[col].median()
                    
                df[col] = df[col].fillna(fill_value)
                logger.debug(f"Filled {df[col].isna().sum()} missing values in '{col}' with {fill_value}")
        
        return df
    def encode_categoricals(self, df: pd.DataFrame, 
                          categorical_columns: Optional[List[str]] = None,
                          encoding_type: str = 'label',
                          handle_unknown: str = 'ignore') -> pd.DataFrame:
        """
        Encode categorical variables using label encoding or one-hot encoding.
        
        Handles unknown categories in test data by either ignoring them or
        assigning them to a special 'unknown' category.
        
        Args:
            df: DataFrame containing categorical columns
            categorical_columns: List of column names to encode. If None, auto-detect
            encoding_type: 'label' for label encoding, 'onehot' for one-hot encoding
            handle_unknown: How to handle unknown categories ('ignore', 'error')
            
        Returns:
            DataFrame with encoded categorical variables
            
        Raises:
            ValueError: If encoding_type is not supported or columns are missing
        """
        if encoding_type not in ['label', 'onehot']:
            raise ValueError("encoding_type must be 'label' or 'onehot'")
            
        df = df.copy()
        
        # Auto-detect categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Remove timestamp column if it exists
            categorical_columns = [col for col in categorical_columns if 'timestamp' not in col.lower()]
            
        logger.info(f"Encoding {len(categorical_columns)} categorical columns using {encoding_type} encoding")
        
        for col in categorical_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame, skipping")
                continue
                
            # Handle missing values first
            if df[col].isna().any():
                df[col] = df[col].fillna('MISSING')
                logger.debug(f"Filled missing values in '{col}' with 'MISSING'")
            
            if encoding_type == 'label':
                df = self._apply_label_encoding(df, col, handle_unknown)
            else:  # onehot
                df = self._apply_onehot_encoding(df, col, handle_unknown)
                
        return df
    
    def _apply_label_encoding(self, df: pd.DataFrame, column: str, handle_unknown: str) -> pd.DataFrame:
        """Apply label encoding to a single column."""
        if column not in self.label_encoders:
            # First time fitting this column
            self.label_encoders[column] = LabelEncoder()
            
            # Fit the encoder
            unique_values = df[column].unique()
            self.label_encoders[column].fit(unique_values)
            logger.debug(f"Fitted label encoder for '{column}' with {len(unique_values)} unique values")
            
        # Transform the column
        try:
            df[column + '_encoded'] = self.label_encoders[column].transform(df[column])
        except ValueError as e:
            if handle_unknown == 'ignore':
                # Handle unknown categories by assigning them a special value
                known_classes = set(self.label_encoders[column].classes_)
                unknown_mask = ~df[column].isin(known_classes)
                
                if unknown_mask.any():
                    logger.warning(f"Found {unknown_mask.sum()} unknown categories in '{column}', assigning to -1")
                    
                    # Transform known values
                    df[column + '_encoded'] = -1  # Default for unknown
                    known_mask = df[column].isin(known_classes)
                    df.loc[known_mask, column + '_encoded'] = self.label_encoders[column].transform(
                        df.loc[known_mask, column]
                    )
                else:
                    df[column + '_encoded'] = self.label_encoders[column].transform(df[column])
            else:
                raise e
                
        # Replace original column
        df.drop(columns=[column], inplace=True)
        df.rename(columns={column + '_encoded': column}, inplace=True)
        
        return df
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, column: str, handle_unknown: str) -> pd.DataFrame:
        """Apply one-hot encoding to a single column."""
        if column not in self.onehot_encoders:
            # First time fitting this column
            self.onehot_encoders[column] = OneHotEncoder(
                handle_unknown=handle_unknown,
                sparse_output=False,
                dtype=np.int32
            )
            
            # Fit the encoder
            self.onehot_encoders[column].fit(df[[column]])
            logger.debug(f"Fitted one-hot encoder for '{column}' with {len(self.onehot_encoders[column].categories_[0])} categories")
        
        # Transform the column
        encoded_array = self.onehot_encoders[column].transform(df[[column]])
        
        # Create column names
        categories = self.onehot_encoders[column].categories_[0]
        encoded_columns = [f"{column}_{cat}" for cat in categories]
        
        # Create DataFrame with encoded columns
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=df.index)
        
        # Drop original column and add encoded columns
        df = df.drop(columns=[column])
        df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def transform_amounts(self, df: pd.DataFrame, 
                         amount_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply log transformation to transaction amounts to normalize distribution.
        
        Handles zero and negative amounts by adding a small constant before
        log transformation.
        
        Args:
            df: DataFrame containing amount columns
            amount_columns: List of amount column names. If None, auto-detect
            
        Returns:
            DataFrame with log-transformed amount columns
        """
        df = df.copy()
        
        # Auto-detect amount columns if not provided
        if amount_columns is None:
            # Look for columns with 'amount' in the name or numeric columns with positive values
            amount_columns = []
            for col in df.columns:
                if 'amount' in col.lower():
                    amount_columns.append(col)
                elif df[col].dtype in ['int64', 'float64'] and (df[col] > 0).any():
                    # Check if this looks like an amount column (positive values, reasonable range)
                    if df[col].max() > 1 and df[col].min() >= 0:
                        amount_columns.append(col)
        
        logger.info(f"Applying log transformation to {len(amount_columns)} amount columns")
        
        for col in amount_columns:
            if col not in df.columns:
                logger.warning(f"Amount column '{col}' not found in DataFrame, skipping")
                continue
                
            # Handle missing values
            if df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.debug(f"Filled {df[col].isna().sum()} missing values in '{col}' with median: {median_value}")
            
            # Handle zero and negative values
            min_positive = df[df[col] > 0][col].min() if (df[col] > 0).any() else 1.0
            epsilon = min_positive / 1000  # Small constant to add to zero/negative values
            
            # Count problematic values
            zero_count = (df[col] == 0).sum()
            negative_count = (df[col] < 0).sum()
            
            if zero_count > 0 or negative_count > 0:
                logger.warning(f"Found {zero_count} zero and {negative_count} negative values in '{col}', adding epsilon={epsilon}")
                df[col] = df[col] + epsilon
            
            # Apply log transformation
            original_col = col + '_original'
            df[original_col] = df[col] - epsilon  # Keep original for reference
            df[col] = np.log1p(df[col])  # log1p is more numerically stable
            
            logger.debug(f"Applied log transformation to '{col}': range [{df[col].min():.3f}, {df[col].max():.3f}]")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values using appropriate imputation strategies.
        
        Uses different strategies for different types of columns:
        - Numerical: median or mean
        - Categorical: mode or constant
        - Boolean: mode
        
        Args:
            df: DataFrame with potential missing values
            strategy: Dictionary mapping column names to imputation strategies
                     ('mean', 'median', 'mode', 'constant', 'drop')
                     
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Default strategies by data type
        if strategy is None:
            strategy = {}
        
        # Identify columns with missing values
        missing_cols = df.columns[df.isna().any()].tolist()
        
        if not missing_cols:
            logger.info("No missing values found")
            return df
            
        logger.info(f"Handling missing values in {len(missing_cols)} columns")
        
        for col in missing_cols:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df) * 100
            
            logger.debug(f"Column '{col}': {missing_count} missing values ({missing_pct:.1f}%)")
            
            # Determine strategy for this column
            col_strategy = strategy.get(col, self._get_default_strategy(df[col]))
            
            if col_strategy == 'drop':
                # Drop rows with missing values in this column
                df = df.dropna(subset=[col])
                logger.debug(f"Dropped {missing_count} rows with missing values in '{col}'")
                continue
            
            # Check if column has any non-missing values
            if df[col].notna().sum() == 0:
                # All values are missing - use constant strategy
                logger.warning(f"Column '{col}' has all missing values, using constant fill")
                if df[col].dtype in ['int64', 'float64']:
                    fill_value = 0
                elif df[col].dtype == 'bool':
                    fill_value = False
                else:
                    fill_value = 'UNKNOWN'
                df[col] = df[col].fillna(fill_value)
                continue
            
            # Apply imputation strategy
            if col_strategy == 'datetime_mode':
                # Handle datetime columns specially
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    # For datetime columns, use the mode (most frequent value)
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value.iloc[0]
                    else:
                        # If no mode, use the median datetime
                        fill_value = df[col].median()
                    df[col] = df[col].fillna(fill_value)
                    logger.debug(f"Filled {missing_count} missing datetime values in '{col}' with {fill_value}")
                continue
            
            if col not in self.imputers:
                if col_strategy in ['mean', 'median']:
                    self.imputers[col] = SimpleImputer(strategy=col_strategy)
                elif col_strategy == 'mode':
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                elif col_strategy == 'constant':
                    # Use appropriate constant based on data type
                    if df[col].dtype in ['int64', 'float64']:
                        fill_value = 0
                    elif df[col].dtype == 'bool':
                        fill_value = False
                    else:
                        fill_value = 'UNKNOWN'
                    self.imputers[col] = SimpleImputer(strategy='constant', fill_value=fill_value)
                
                # Fit the imputer
                self.imputers[col].fit(df[[col]])
            
            # Transform the column
            df[col] = self.imputers[col].transform(df[[col]]).ravel()
            
            logger.debug(f"Imputed missing values in '{col}' using {col_strategy} strategy")
        
        return df
    
    def _get_default_strategy(self, series: pd.Series) -> str:
        """Determine default imputation strategy based on data type and distribution."""
        if series.dtype in ['int64', 'float64']:
            # For numerical data, use median (more robust to outliers)
            return 'median'
        elif series.dtype == 'bool':
            return 'mode'
        elif pd.api.types.is_datetime64_any_dtype(series):
            # For datetime columns, use a special strategy
            return 'datetime_mode'
        else:
            # For categorical data, use mode
            return 'mode'
    
    def fit_transform(self, df: pd.DataFrame, 
                     timestamp_col: str = 'timestamp',
                     categorical_columns: Optional[List[str]] = None,
                     amount_columns: Optional[List[str]] = None,
                     encoding_type: str = 'label',
                     missing_strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline: fit and transform the data.
        
        Args:
            df: Raw transaction DataFrame
            timestamp_col: Name of timestamp column
            categorical_columns: List of categorical columns to encode
            amount_columns: List of amount columns to transform
            encoding_type: Type of categorical encoding ('label' or 'onehot')
            missing_strategy: Missing value handling strategy
            
        Returns:
            Fully preprocessed DataFrame
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Step 1: Parse timestamps
        df = self.parse_timestamps(df, timestamp_col)
        
        # Step 2: Handle missing values (before encoding to avoid issues)
        df = self.handle_missing_values(df, missing_strategy)
        
        # Step 3: Transform amounts
        df = self.transform_amounts(df, amount_columns)
        
        # Step 4: Encode categoricals (last to avoid issues with missing values)
        df = self.encode_categoricals(df, categorical_columns, encoding_type)
        
        self.is_fitted = True
        logger.info("Preprocessing pipeline completed successfully")
        
        return df
    
    def transform(self, df: pd.DataFrame, 
                  timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: New transaction DataFrame to transform
            timestamp_col: Name of timestamp column
            
        Returns:
            Preprocessed DataFrame
            
        Raises:
            ValueError: If preprocessor has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
            
        logger.info("Transforming new data using fitted preprocessors")
        
        # Apply same preprocessing steps
        df = self.parse_timestamps(df, timestamp_col)
        
        # Apply fitted imputers
        for col, imputer in self.imputers.items():
            if col in df.columns and df[col].isna().any():
                df[col] = imputer.transform(df[[col]]).ravel()
        
        # Apply fitted encoders
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df = self._apply_label_encoding(df, col, 'ignore')
                
        for col, encoder in self.onehot_encoders.items():
            if col in df.columns:
                df = self._apply_onehot_encoding(df, col, 'ignore')
        
        # Transform amounts (this doesn't require fitting)
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]
        if amount_cols:
            df = self.transform_amounts(df, amount_cols)
        
        return df