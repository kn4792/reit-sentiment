"""
Data Validation Module for REIT Sentiment Analysis

Provides data quality checks, missing value handling, and outlier detection.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class DataValidator:
    """
    Validate data quality and handle missing values.
    
    Implements:
    - Missing value detection and handling
    - Outlier detection (>5 standard deviations)
    - Data type validation
    - Required field checks
    - Duplicate detection
    
    Attributes:
        required_fields: List of column names that must be non-null
        outlier_threshold: Number of std devs for outlier detection
    
    Example:
        >>> validator = DataValidator(required_fields=['ticker', 'date'])
        >>> df = validator.validate(df)
    """
    
    def __init__(self, 
                 required_fields: Optional[List[str]] = None,
                 outlier_std_threshold: float = 5.0):
        """
        Initialize data validator.
        
        Args:
            required_fields: List of column names that must be non-null
            outlier_std_threshold: Number of std devs for outlier detection
        """
        self.required_fields = required_fields or ['ticker', 'date']
        self.outlier_threshold = outlier_std_threshold
    
    def check_required_fields(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Remove rows with missing required fields.
        
        Args:
            df: Input DataFrame
            verbose: Whether to print removal statistics
            
        Returns:
            DataFrame with complete required fields
            
        Example:
            >>> validator = DataValidator(required_fields=['ticker', 'date'])
            >>> df = validator.check_required_fields(df)
            ⚠️  Removed 5 rows with missing required fields
        """
        initial_len = len(df)
        
        for field in self.required_fields:
            if field in df.columns:
                df = df[df[field].notna()]
        
        removed = initial_len - len(df)
        if removed > 0 and verbose:
            print(f"⚠️  Removed {removed} rows with missing required fields")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies.
        
        Strategy:
        - Text fields: Fill with empty string
        - Numeric fields: Forward-fill (limit 5)
        - Dates: Drop rows
        
        Args:
            df: Input DataFrame
            verbose: Whether to print statistics
            
        Returns:
            DataFrame with missing values handled
            
        Example:
            >>> validator = DataValidator()
            >>> df = validator.handle_missing_values(df)
        """
        # For text fields, fill with empty string
        text_fields = ['pros', 'cons', 'title', 'employee_info', 
                      'pros_cleaned', 'cons_cleaned']
        for field in text_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                df[field] = df[field].fillna('')
                if missing_count > 0 and verbose:
                    print(f"  • Filled {missing_count} missing values in '{field}'")
        
        # For rating, forward-fill within company (limit 5)
        if 'rating' in df.columns and 'ticker' in df.columns:
            missing_before = df['rating'].isna().sum()
            df['rating'] = df.groupby('ticker')['rating'].fillna(method='ffill', limit=5)
            missing_after = df['rating'].isna().sum()
            filled = missing_before - missing_after
            if filled > 0 and verbose:
                print(f"  • Forward-filled {filled} missing ratings")
        
        return df
    
    def remove_outliers(self, 
                       df: pd.DataFrame, 
                       column: str = 'rating',
                       verbose: bool = True) -> pd.DataFrame:
        """
        Remove outliers using standard deviation threshold.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            verbose: Whether to print removal statistics
            
        Returns:
            DataFrame with outliers removed
            
        Example:
            >>> validator = DataValidator(outlier_std_threshold=5.0)
            >>> df = validator.remove_outliers(df, column='rating')
            ⚠️  Removed 3 outlier rows in rating
        """
        if column not in df.columns:
            return df
        
        # Convert to numeric
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Calculate mean and std
        mean = df[column].mean()
        std = df[column].std()
        
        # Remove outliers
        initial_len = len(df)
        df = df[np.abs(df[column] - mean) <= (self.outlier_threshold * std)]
        
        removed = initial_len - len(df)
        if removed > 0 and verbose:
            print(f"⚠️  Removed {removed} outlier rows in {column}")
        
        return df
    
    def validate_ratings(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Validate rating values are in reasonable range (0-5).
        
        Args:
            df: Input DataFrame
            verbose: Whether to print removal statistics
            
        Returns:
            DataFrame with valid ratings
            
        Example:
            >>> validator = DataValidator()
            >>> df = validator.validate_ratings(df)
            ⚠️  Removed 2 rows with invalid ratings
        """
        if 'rating' not in df.columns:
            return df
        
        # Convert to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Filter to 0-5 range
        initial_len = len(df)
        df = df[(df['rating'] >= 0) & (df['rating'] <= 5)]
        
        removed = initial_len - len(df)
        if removed > 0 and verbose:
            print(f"⚠️  Removed {removed} rows with invalid ratings")
        
        return df
    
    def remove_duplicates(self, 
                         df: pd.DataFrame,
                         subset: Optional[List[str]] = None,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates (None = all columns)
            verbose: Whether to print removal statistics
            
        Returns:
            DataFrame with duplicates removed
            
        Example:
            >>> validator = DataValidator()
            >>> df = validator.remove_duplicates(df, subset=['ticker', 'date', 'title'])
            ⚠️  Removed 15 duplicate rows
        """
        if subset is None:
            subset = ['ticker', 'date', 'title', 'pros']
        
        # Keep only existing columns
        subset = [col for col in subset if col in df.columns]
        
        if not subset:
            return df
        
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        
        removed = initial_len - len(df)
        if removed > 0 and verbose:
            print(f"⚠️  Removed {removed} duplicate rows")
        
        return df
    
    def validate_dates(self, 
                      df: pd.DataFrame,
                      date_column: str = 'date',
                      verbose: bool = True) -> pd.DataFrame:
        """
        Validate and convert dates to standard format.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            verbose: Whether to print statistics
            
        Returns:
            DataFrame with validated dates
            
        Example:
            >>> validator = DataValidator()
            >>> df = validator.validate_dates(df, date_column='date')
        """
        if date_column not in df.columns:
            return df
        
        # Convert to datetime
        initial_len = len(df)
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Remove rows with invalid dates
        df = df[df[date_column].notna()]
        
        removed = initial_len - len(df)
        if removed > 0 and verbose:
            print(f"⚠️  Removed {removed} rows with invalid dates")
        
        return df
    
    def check_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Check data types of all columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping data types to column names
            
        Example:
            >>> validator = DataValidator()
            >>> types = validator.check_data_types(df)
            >>> print(types)
            {'object': ['ticker', 'pros'], 'float64': ['rating'], ...}
        """
        type_dict = {}
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype not in type_dict:
                type_dict[dtype] = []
            type_dict[dtype].append(col)
        
        return type_dict
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
            
        Example:
            >>> validator = DataValidator()
            >>> report = validator.get_data_quality_report(df)
            >>> print(report['missing_values'])
            {'pros': 5, 'cons': 3, 'rating': 0}
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicate_rows': 0,
            'data_types': {},
            'numeric_stats': {}
        }
        
        # Missing values
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                report['missing_values'][col] = int(missing)
        
        # Duplicates
        report['duplicate_rows'] = int(df.duplicated().sum())
        
        # Data types
        report['data_types'] = self.check_data_types(df)
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        
        return report
    
    def validate(self, 
                df: pd.DataFrame,
                remove_outliers_cols: Optional[List[str]] = None,
                verbose: bool = True) -> pd.DataFrame:
        """
        Run complete validation pipeline.
        
        Args:
            df: Input DataFrame
            remove_outliers_cols: Columns to check for outliers
            verbose: Whether to print statistics
            
        Returns:
            Validated DataFrame
            
        Example:
            >>> validator = DataValidator()
            >>> df = validator.validate(df, remove_outliers_cols=['rating'])
        """
        if verbose:
            print(f"\n{'='*60}")
            print("DATA VALIDATION")
            print('='*60)
            print(f"Initial rows: {len(df):,}")
        
        # Check required fields
        df = self.check_required_fields(df, verbose=verbose)
        
        # Validate dates
        if 'date' in df.columns:
            df = self.validate_dates(df, verbose=verbose)
        
        # Validate ratings
        if 'rating' in df.columns:
            df = self.validate_ratings(df, verbose=verbose)
        
        # Handle missing values
        df = self.handle_missing_values(df, verbose=verbose)
        
        # Remove outliers
        if remove_outliers_cols:
            for col in remove_outliers_cols:
                if col in df.columns:
                    df = self.remove_outliers(df, column=col, verbose=verbose)
        
        # Remove duplicates
        df = self.remove_duplicates(df, verbose=verbose)
        
        if verbose:
            print(f"\nFinal rows: {len(df):,}")
            print('='*60)
        
        return df


if __name__ == '__main__':
    # Example usage
    print("Data Validator Example")
    print("=" * 60)
    
    # Create sample data with quality issues
    df = pd.DataFrame({
        'ticker': ['PLD', 'AMT', None, 'EQIX', 'PLD', 'PLD'],
        'date': ['2025-01-15', '2025-01-16', '2025-01-17', 'invalid', '2025-01-15', '2025-01-15'],
        'pros': ['Great benefits', None, 'Good culture', 'Excellent', 'Great benefits', 'Great benefits'],
        'rating': [4.5, 3.5, 10.0, 5.0, 4.5, 4.5]
    })
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Validate
    validator = DataValidator(required_fields=['ticker', 'date'])
    df_clean = validator.validate(df, remove_outliers_cols=['rating'])
    
    print("\nCleaned DataFrame:")
    print(df_clean)
    
    # Get quality report
    report = validator.get_data_quality_report(df_clean)
    print("\nData Quality Report:")
    print(f"  Total rows: {report['total_rows']}")
    print(f"  Missing values: {report['missing_values']}")
    print(f"  Duplicates: {report['duplicate_rows']}")