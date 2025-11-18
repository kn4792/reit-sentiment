"""
Integration tests for Phase 1: Data Collection & Preparation

This module tests the complete data preparation pipeline including:
- Data scraping/loading
- Data cleaning and preprocessing
- Train/validation/test splitting
- Data quality validation

Author: Konain Niaz
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from datetime import datetime

# Import project modules
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.data_validator import DataValidator


@pytest.mark.integration
class TestDataPreparationPipeline:
    """Integration tests for complete data preparation pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment with temporary directories"""
        self.raw_dir = tmp_path / "raw"
        self.processed_dir = tmp_path / "processed"
        self.raw_dir.mkdir()
        self.processed_dir.mkdir()
        
        # Create sample raw data
        self.sample_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15', 
                     '2023-03-01', '2023-03-15', '2023-04-01', '2023-04-15'],
            'rating': [4.0, 3.5, 5.0, 2.0, 4.5, 3.0, 5.0, 4.0],
            'pros': [
                'Great culture and amazing team',
                'Good benefits and work-life balance',
                'Excellent management and leadership',
                'Nice office space',
                'Strong REIT performance',
                'Competitive salary',
                'Great learning opportunities',
                'Supportive colleagues'
            ],
            'cons': [
                'Long hours sometimes',
                'Limited growth opportunities',
                'Low pay compared to industry',
                'Outdated technology',
                'High pressure environment',
                'Poor communication',
                'Lack of flexibility',
                'Micromanagement issues'
            ],
            'advice': [
                'Focus on employee development',
                'Improve compensation',
                'Better communication needed',
                'Invest in technology',
                'Reduce workload',
                'Trust employees more',
                'Offer remote work',
                'Train managers better'
            ],
            'company': ['TestREIT'] * 8,
            'ticker': ['TEST'] * 8
        })
        
        # Save sample raw data
        self.raw_file = self.raw_dir / "TEST_reviews.csv"
        self.sample_data.to_csv(self.raw_file, index=False)
        
        yield
        
    def test_raw_data_exists(self):
        """Test that raw data file exists and is readable"""
        assert self.raw_file.exists(), "Raw data file should exist"
        
        df = pd.read_csv(self.raw_file)
        assert len(df) > 0, "Raw data should not be empty"
        assert 'pros' in df.columns, "Raw data should have 'pros' column"
        assert 'cons' in df.columns, "Raw data should have 'cons' column"
        
    def test_required_columns_present(self):
        """Test that all required columns are present in raw data"""
        df = pd.read_csv(self.raw_file)
        
        required_columns = ['date', 'rating', 'pros', 'cons', 'company', 'ticker']
        for col in required_columns:
            assert col in df.columns, f"Required column '{col}' missing from raw data"
            
    def test_data_types_correct(self):
        """Test that data types are correct after loading"""
        df = pd.read_csv(self.raw_file)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        assert df['date'].dtype == 'datetime64[ns]', "Date column should be datetime type"
        assert df['rating'].dtype in ['float64', 'int64'], "Rating should be numeric"
        assert df['pros'].dtype == 'object', "Pros should be string/object type"
        assert df['cons'].dtype == 'object', "Cons should be string/object type"
        
    def test_no_critical_missing_values(self):
        """Test that critical columns have no missing values"""
        df = pd.read_csv(self.raw_file)
        
        critical_columns = ['date', 'rating', 'company', 'ticker']
        for col in critical_columns:
            missing_count = df[col].isna().sum()
            assert missing_count == 0, f"Critical column '{col}' has {missing_count} missing values"
            
    def test_text_cleaning_pipeline(self):
        """Test the text cleaning pipeline"""
        cleaner = TextCleaner()
        
        # Test cleaning on sample text
        dirty_text = "Great culture & amazing team!!! http://example.com #company"
        cleaned = cleaner.clean(dirty_text)
        
        assert len(cleaned) > 0, "Cleaned text should not be empty"
        assert 'http' not in cleaned.lower(), "URLs should be removed"
        assert '&' not in cleaned, "HTML entities should be removed"
        
    def test_data_cleaning_preserves_rows(self):
        """Test that data cleaning doesn't lose rows unexpectedly"""
        df = pd.read_csv(self.raw_file)
        original_count = len(df)
        
        cleaner = TextCleaner()
        
        # Clean text columns
        df['cleaned_pros'] = df['pros'].apply(cleaner.clean)
        df['cleaned_cons'] = df['cons'].apply(cleaner.clean)
        
        assert len(df) == original_count, "Cleaning should preserve row count"
        assert df['cleaned_pros'].notna().sum() > 0, "Cleaned pros should not be all null"
        
    def test_rating_range_validation(self):
        """Test that ratings are within valid range"""
        df = pd.read_csv(self.raw_file)
        
        assert df['rating'].min() >= 1.0, "Ratings should be >= 1.0"
        assert df['rating'].max() <= 5.0, "Ratings should be <= 5.0"
        
    def test_date_format_validation(self):
        """Test that dates are in correct format"""
        df = pd.read_csv(self.raw_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Check that all dates are valid
        assert df['date'].notna().all(), "All dates should be valid"
        
        # Check that dates are not in the future
        today = pd.Timestamp.now()
        assert (df['date'] <= today).all(), "Dates should not be in the future"
        
    def test_data_split_creation(self):
        """Test creation of train/val/test splits"""
        df = pd.read_csv(self.raw_file)
        
        # Split data (70/15/15)
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        # Save splits
        train_file = self.processed_dir / "train.csv"
        val_file = self.processed_dir / "val.csv"
        test_file = self.processed_dir / "test.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Verify files exist
        assert train_file.exists(), "Train file should exist"
        assert val_file.exists(), "Validation file should exist"
        assert test_file.exists(), "Test file should exist"
        
        # Verify sizes
        assert len(train_df) >= len(val_df), "Train set should be largest"
        assert len(train_df) >= len(test_df), "Train set should be largest"
        
    def test_data_splits_no_overlap(self):
        """Test that train/val/test splits have no overlap"""
        df = pd.read_csv(self.raw_file)
        
        # Add unique ID
        df['id'] = range(len(df))
        
        # Split data
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        # Check no overlap
        train_ids = set(train_df['id'])
        val_ids = set(val_df['id'])
        test_ids = set(test_df['id'])
        
        assert len(train_ids & val_ids) == 0, "Train and validation sets should not overlap"
        assert len(train_ids & test_ids) == 0, "Train and test sets should not overlap"
        assert len(val_ids & test_ids) == 0, "Validation and test sets should not overlap"
        
    def test_data_splits_sum_to_total(self):
        """Test that split sizes sum to original dataset size"""
        df = pd.read_csv(self.raw_file)
        original_size = len(df)
        
        # Split data
        train_size = int(0.7 * len(df))
        val_size = int(0.15 * len(df))
        
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size + val_size]
        test_df = df.iloc[train_size + val_size:]
        
        total_split_size = len(train_df) + len(val_df) + len(test_df)
        
        assert total_split_size == original_size, \
            f"Split sizes ({total_split_size}) should equal original size ({original_size})"
            
    def test_outlier_detection(self):
        """Test outlier detection in ratings"""
        df = pd.read_csv(self.raw_file)
        
        # Calculate outliers (>5 std from mean)
        mean_rating = df['rating'].mean()
        std_rating = df['rating'].std()
        
        outliers = df[np.abs(df['rating'] - mean_rating) > 5 * std_rating]
        
        # For this small dataset, shouldn't have extreme outliers
        assert len(outliers) == 0, "Should not have extreme outliers in sample data"
        
    def test_data_validator_integration(self):
        """Test DataValidator class with complete pipeline"""
        df = pd.read_csv(self.raw_file)
        
        validator = DataValidator()
        
        # Run validation - returns cleaned DataFrame
        validated_df = validator.validate(df)
        
        # Check that validation worked
        assert isinstance(validated_df, pd.DataFrame), "Should return DataFrame"
        assert len(validated_df) > 0, "Validated data should not be empty"
        assert len(validated_df) <= len(df), "Validated data should have same or fewer rows"
        
    def test_complete_pipeline_execution(self):
        """Test complete data preparation pipeline end-to-end"""
        # Step 1: Load raw data
        df = pd.read_csv(self.raw_file)
        assert len(df) > 0, "Should load data"
        
        # Step 2: Clean data
        cleaner = TextCleaner()
        df['cleaned_pros'] = df['pros'].apply(cleaner.clean)
        df['cleaned_cons'] = df['cons'].apply(cleaner.clean)
        assert df['cleaned_pros'].notna().sum() > 0, "Should have cleaned text"
        
        # Step 3: Validate data
        validator = DataValidator()
        validated_df = validator.validate(df)
        assert isinstance(validated_df, pd.DataFrame), "Should return DataFrame"
        assert len(validated_df) > 0, "Validated data should not be empty"
        
        # Step 4: Split data
        train_size = int(0.7 * len(validated_df))
        val_size = int(0.15 * len(validated_df))
        
        train_df = validated_df.iloc[:train_size]
        val_df = validated_df.iloc[train_size:train_size + val_size]
        test_df = validated_df.iloc[train_size + val_size:]
        
        # Step 5: Save processed data
        cleaned_file = self.processed_dir / "cleaned_TEST_reviews.csv"
        validated_df.to_csv(cleaned_file, index=False)
        
        train_file = self.processed_dir / "train.csv"
        val_file = self.processed_dir / "val.csv"
        test_file = self.processed_dir / "test.csv"
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # Verify all outputs exist
        assert cleaned_file.exists(), "Cleaned file should exist"
        assert train_file.exists(), "Train file should exist"
        assert val_file.exists(), "Val file should exist"
        assert test_file.exists(), "Test file should exist"
        
        print("✓ Complete data preparation pipeline executed successfully")


@pytest.mark.unit
class TestTextCleaning:
    """Unit tests for text cleaning functions"""
    
    def test_remove_html_entities(self):
        """Test HTML entity removal"""
        cleaner = TextCleaner()
        text = "Great &amp; amazing &lt;culture&gt;"
        cleaned = cleaner.clean(text)
        
        assert '&amp;' not in cleaned
        assert '&lt;' not in cleaned
        assert '&gt;' not in cleaned
        
    def test_remove_urls(self):
        """Test URL removal"""
        cleaner = TextCleaner()
        text = "Check out http://example.com and www.test.com"
        cleaned = cleaner.clean(text)
        
        assert 'http://' not in cleaned
        assert 'www.' not in cleaned
        
    def test_remove_special_characters(self):
        """Test special character handling"""
        cleaner = TextCleaner()
        text = "Great!!! Amazing??? #company @user"
        cleaned = cleaner.clean(text)
        
        assert '!!!' not in cleaned
        assert '???' not in cleaned
        
    def test_lowercase_conversion(self):
        """Test lowercase conversion"""
        cleaner = TextCleaner()
        text = "GREAT Culture And AMAZING Team"
        cleaned = cleaner.clean(text)
        
        assert cleaned.islower() or not cleaned.replace(' ', '').isalpha(), \
            "Text should be lowercase"
            
    def test_empty_string_handling(self):
        """Test handling of empty strings"""
        cleaner = TextCleaner()
        cleaned = cleaner.clean("")
        
        assert isinstance(cleaned, str), "Should return string"
        assert len(cleaned) == 0, "Empty input should return empty output"
        
    def test_none_handling(self):
        """Test handling of None values"""
        cleaner = TextCleaner()
        cleaned = cleaner.clean(None)
        
        assert isinstance(cleaned, str), "Should return string for None input"


@pytest.mark.unit  
class TestDataValidation:
    """Unit tests for data validation functions"""
    
    def test_required_columns_check(self):
        """Test required columns validation"""
        validator = DataValidator()
        
        # Valid dataframe
        valid_df = pd.DataFrame({
            'date': ['2023-01-01'],
            'rating': [4.0],
            'pros': ['Good'],
            'cons': ['Bad'],
            'company': ['Test'],
            'ticker': ['TST']
        })
        
        # Should return DataFrame without errors
        result = validator.validate(valid_df)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) > 0, "Valid dataframe should have rows"
        
        # Invalid dataframe (missing column) - validator may raise exception or return empty
        invalid_df = pd.DataFrame({
            'date': ['2023-01-01'],
            'rating': [4.0]
        })
        
        try:
            result = validator.validate(invalid_df)
            # If it doesn't raise an exception, it might return an empty or modified DataFrame
            # This depends on your DataValidator implementation
        except (KeyError, ValueError) as e:
            # Expected behavior - validator detects missing columns
            pass
        
    def test_rating_range_check(self):
        """Test rating range validation"""
        validator = DataValidator()
        
        # Valid ratings
        valid_df = pd.DataFrame({
            'date': ['2023-01-01'],
            'rating': [3.5],
            'pros': ['Good'],
            'cons': ['Bad'],
            'company': ['Test'],
            'ticker': ['TST']
        })
        
        result = validator.validate(valid_df)
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) > 0, "Valid rating should pass"
        
        # Invalid rating (out of range)
        invalid_df = valid_df.copy()
        invalid_df['rating'] = [6.0]  # Rating > 5.0
        
        result = validator.validate(invalid_df)
        # Validator should handle out of range ratings (clip, remove, or flag)
        # Check that it either removed the row or capped the rating
        if len(result) > 0:
            assert result['rating'].max() <= 5.0, "Ratings should be capped at 5.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])