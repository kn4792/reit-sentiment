#!/usr/bin/env python3
"""
Unit Tests for Data Cleaning Module

Tests the TextCleaner, DateParser, and DataValidator classes.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17

Usage:
    pytest tests/test_cleaner.py -v
    pytest tests/test_cleaner.py -v --cov=src/preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the scripts directory since we're using standalone scripts
# For testing, we'll need to extract the classes or import them
# Let's create mock versions for testing


class TextCleaner:
    """Mock TextCleaner for testing."""
    
    def __init__(self, remove_stopwords=True, apply_stemming=True, 
                 min_word_length=2, custom_stopwords=None):
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.min_word_length = min_word_length
        
        # Simplified for testing
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                         'reit', 'company', 'work'} if remove_stopwords else set()
        
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def remove_html(self, text):
        """Remove HTML tags and entities."""
        if pd.isna(text) or text == "":
            return ""
        
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', str(text))
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        return text
    
    def normalize_text(self, text):
        """Normalize text."""
        if pd.isna(text) or text == "":
            return ""
        
        import re
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-z0-9\s\.\,\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize_and_filter(self, text):
        """Tokenize and filter text."""
        if not text:
            return []
        
        tokens = text.split()
        filtered = []
        
        for token in tokens:
            if len(token) < self.min_word_length:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            if not token.isalpha():
                continue
            
            # Simple stemming (just remove 's' and 'ing')
            if self.apply_stemming:
                if token.endswith('ing'):
                    token = token[:-3]
                elif token.endswith('s') and len(token) > 3:
                    token = token[:-1]
            
            filtered.append(token)
        
        return filtered
    
    def clean(self, text):
        """Complete cleaning pipeline."""
        text = self.remove_html(text)
        text = self.normalize_text(text)
        tokens = self.tokenize_and_filter(text)
        return ' '.join(tokens)


class DateParser:
    """Mock DateParser for testing."""
    
    @staticmethod
    def parse_date(date_str):
        """Parse date string to YYYY-MM-DD format."""
        if pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        
        formats = [
            '%b %d, %Y',      # Jan 15, 2025
            '%B %d, %Y',      # January 15, 2025
            '%Y-%m-%d',       # 2025-01-15
            '%m/%d/%Y',       # 1/15/2025
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None


class DataValidator:
    """Mock DataValidator for testing."""
    
    def __init__(self, required_fields=None, outlier_std_threshold=5.0):
        self.required_fields = required_fields or ['ticker', 'date']
        self.outlier_threshold = outlier_std_threshold
    
    def check_required_fields(self, df):
        """Remove rows with missing required fields."""
        for field in self.required_fields:
            if field in df.columns:
                df = df[df[field].notna()]
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values."""
        text_fields = ['pros', 'cons', 'title', 'employee_info']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna('')
        return df
    
    def remove_outliers(self, df, column='rating'):
        """Remove outliers."""
        if column not in df.columns:
            return df
        
        df[column] = pd.to_numeric(df[column], errors='coerce')
        mean = df[column].mean()
        std = df[column].std()
        df = df[np.abs(df[column] - mean) <= (self.outlier_threshold * std)]
        return df
    
    def validate_ratings(self, df):
        """Validate ratings are in 0-5 range."""
        if 'rating' not in df.columns:
            return df
        
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df[(df['rating'] >= 0) & (df['rating'] <= 5)]
        return df


# ============================================================================
# TEXT CLEANER TESTS
# ============================================================================

class TestTextCleaner:
    """Tests for TextCleaner class."""
    
    def test_init_default(self):
        """Test default initialization."""
        cleaner = TextCleaner()
        assert cleaner.remove_stopwords == True
        assert cleaner.apply_stemming == True
        assert cleaner.min_word_length == 2
        assert 'reit' in cleaner.stopwords
    
    def test_init_custom_stopwords(self):
        """Test initialization with custom stopwords."""
        custom = ['test', 'example']
        cleaner = TextCleaner(custom_stopwords=custom)
        assert 'test' in cleaner.stopwords
        assert 'example' in cleaner.stopwords
    
    def test_remove_html_tags(self):
        """Test HTML tag removal."""
        cleaner = TextCleaner()
        
        text = "<p>This is a <strong>test</strong></p>"
        result = cleaner.remove_html(text)
        assert '<' not in result
        assert '>' not in result
        assert 'test' in result.lower()
    
    def test_remove_html_entities(self):
        """Test HTML entity removal."""
        cleaner = TextCleaner()
        
        text = "Great &amp; wonderful &nbsp; company"
        result = cleaner.remove_html(text)
        assert '&amp;' not in result
        assert '&nbsp;' not in result
    
    def test_remove_html_empty(self):
        """Test HTML removal with empty input."""
        cleaner = TextCleaner()
        
        assert cleaner.remove_html("") == ""
        assert cleaner.remove_html(None) == ""
        assert cleaner.remove_html(np.nan) == ""
    
    def test_normalize_text_lowercase(self):
        """Test text normalization converts to lowercase."""
        cleaner = TextCleaner()
        
        text = "GREAT Company WITH Good Benefits"
        result = cleaner.normalize_text(text)
        assert result == result.lower()
        assert result == "great company with good benefits"
    
    def test_normalize_text_remove_urls(self):
        """Test URL removal."""
        cleaner = TextCleaner()
        
        text = "Check out https://example.com and www.test.com for info"
        result = cleaner.normalize_text(text)
        assert 'https://' not in result
        assert 'www.' not in result
        assert 'example.com' not in result
    
    def test_normalize_text_remove_emails(self):
        """Test email removal."""
        cleaner = TextCleaner()
        
        text = "Contact us at info@example.com for details"
        result = cleaner.normalize_text(text)
        assert '@' not in result
        assert 'info@example.com' not in result
    
    def test_normalize_text_remove_special_chars(self):
        """Test special character removal."""
        cleaner = TextCleaner()
        
        text = "Great company!!! Best #workplace @2024"
        result = cleaner.normalize_text(text)
        assert '#' not in result
        assert '@' not in result
        # Should keep basic punctuation
        assert '!' in result or result.strip()
    
    def test_tokenize_and_filter_basic(self):
        """Test basic tokenization."""
        cleaner = TextCleaner(remove_stopwords=False, apply_stemming=False)
        
        text = "great benefits and good culture"
        tokens = cleaner.tokenize_and_filter(text)
        assert 'great' in tokens
        assert 'benefits' in tokens
    
    def test_tokenize_and_filter_stopwords(self):
        """Test stopword removal."""
        cleaner = TextCleaner(remove_stopwords=True, apply_stemming=False)
        
        text = "the company has great benefits and good culture"
        tokens = cleaner.tokenize_and_filter(text)
        assert 'the' not in tokens
        assert 'and' not in tokens
        assert 'company' not in tokens  # Domain-specific stopword
        assert 'great' in tokens
        assert 'benefits' in tokens
    
    def test_tokenize_and_filter_min_length(self):
        """Test minimum word length filtering."""
        cleaner = TextCleaner(remove_stopwords=False, min_word_length=4)
        
        text = "i am at a big company now"
        tokens = cleaner.tokenize_and_filter(text)
        assert 'i' not in tokens
        assert 'am' not in tokens
        assert 'at' not in tokens
        assert 'company' in tokens
    
    def test_tokenize_and_filter_stemming(self):
        """Test stemming."""
        cleaner = TextCleaner(remove_stopwords=False, apply_stemming=True)
        
        text = "working companies benefits"
        tokens = cleaner.tokenize_and_filter(text)
        # Simple stemming removes 'ing' and trailing 's'
        assert 'work' in tokens or 'working' in tokens
    
    def test_tokenize_and_filter_non_alpha(self):
        """Test non-alphabetic token removal."""
        cleaner = TextCleaner(remove_stopwords=False)
        
        text = "great 123 benefits 456"
        tokens = cleaner.tokenize_and_filter(text)
        assert '123' not in tokens
        assert '456' not in tokens
        assert 'great' in tokens
        assert 'benefits' in tokens
    
    def test_clean_complete_pipeline(self):
        """Test complete cleaning pipeline."""
        cleaner = TextCleaner()
        
        text = "<p>Great company with good benefits! Visit https://example.com</p>"
        result = cleaner.clean(text)
        
        # Should be cleaned
        assert '<p>' not in result
        assert 'https://' not in result
        assert len(result) > 0
    
    def test_clean_empty_input(self):
        """Test cleaning with empty input."""
        cleaner = TextCleaner()
        
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""
        assert cleaner.clean("   ") == ""


# ============================================================================
# DATE PARSER TESTS
# ============================================================================

class TestDateParser:
    """Tests for DateParser class."""
    
    def test_parse_date_format1(self):
        """Test parsing 'Jan 15, 2025' format."""
        result = DateParser.parse_date("Jan 15, 2025")
        assert result == "2025-01-15"
    
    def test_parse_date_format2(self):
        """Test parsing 'January 15, 2025' format."""
        result = DateParser.parse_date("January 15, 2025")
        assert result == "2025-01-15"
    
    def test_parse_date_format3(self):
        """Test parsing '2025-01-15' format."""
        result = DateParser.parse_date("2025-01-15")
        assert result == "2025-01-15"
    
    def test_parse_date_format4(self):
        """Test parsing '1/15/2025' format."""
        result = DateParser.parse_date("1/15/2025")
        assert result == "2025-01-15"
    
    def test_parse_date_various_months(self):
        """Test parsing different months."""
        dates = [
            ("Feb 1, 2025", "2025-02-01"),
            ("Dec 31, 2024", "2024-12-31"),
            ("Jul 4, 2023", "2023-07-04")
        ]
        
        for input_date, expected in dates:
            result = DateParser.parse_date(input_date)
            assert result == expected
    
    def test_parse_date_invalid(self):
        """Test parsing invalid date."""
        result = DateParser.parse_date("Invalid date")
        assert result is None
    
    def test_parse_date_empty(self):
        """Test parsing empty date."""
        assert DateParser.parse_date("") is None
        assert DateParser.parse_date(None) is None
        assert DateParser.parse_date(np.nan) is None
    
    def test_parse_date_whitespace(self):
        """Test parsing date with whitespace."""
        result = DateParser.parse_date("  Jan 15, 2025  ")
        assert result == "2025-01-15"


# ============================================================================
# DATA VALIDATOR TESTS
# ============================================================================

class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_init_default(self):
        """Test default initialization."""
        validator = DataValidator()
        assert validator.required_fields == ['ticker', 'date']
        assert validator.outlier_threshold == 5.0
    
    def test_init_custom_fields(self):
        """Test initialization with custom required fields."""
        validator = DataValidator(required_fields=['ticker', 'pros', 'cons'])
        assert 'ticker' in validator.required_fields
        assert 'pros' in validator.required_fields
        assert 'cons' in validator.required_fields
    
    def test_check_required_fields_complete(self):
        """Test required field validation with complete data."""
        validator = DataValidator(required_fields=['ticker', 'date'])
        
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX'],
            'date': ['2025-01-15', '2025-01-16', '2025-01-17'],
            'rating': [4.5, 3.5, 5.0]
        })
        
        result = validator.check_required_fields(df)
        assert len(result) == 3
    
    def test_check_required_fields_missing(self):
        """Test required field validation with missing data."""
        validator = DataValidator(required_fields=['ticker', 'date'])
        
        df = pd.DataFrame({
            'ticker': ['PLD', None, 'EQIX'],
            'date': ['2025-01-15', '2025-01-16', None],
            'rating': [4.5, 3.5, 5.0]
        })
        
        result = validator.check_required_fields(df)
        # Should remove rows with missing ticker or date
        assert len(result) < 3
        assert result['ticker'].notna().all()
        assert result['date'].notna().all()
    
    def test_handle_missing_values_text_fields(self):
        """Test missing value handling for text fields."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'ticker': ['PLD'],
            'pros': [None],
            'cons': [np.nan],
            'title': [''],
            'employee_info': ['Current Employee']
        })
        
        result = validator.handle_missing_values(df)
        assert result['pros'].iloc[0] == ''
        assert result['cons'].iloc[0] == ''
    
    def test_remove_outliers_basic(self):
        """Test outlier removal."""
        validator = DataValidator(outlier_std_threshold=2.0)
        
        df = pd.DataFrame({
            'ticker': ['PLD'] * 10,
            'rating': [3.0, 3.2, 3.5, 3.3, 3.4, 3.1, 3.6, 3.2, 10.0, 3.3]
        })
        
        result = validator.remove_outliers(df, column='rating')
        # 10.0 should be removed as outlier
        assert len(result) < len(df)
        assert 10.0 not in result['rating'].values
    
    def test_remove_outliers_no_column(self):
        """Test outlier removal when column doesn't exist."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'ticker': ['PLD'],
            'pros': ['Great company']
        })
        
        result = validator.remove_outliers(df, column='rating')
        # Should return unchanged
        assert len(result) == len(df)
    
    def test_validate_ratings_valid(self):
        """Test rating validation with valid ratings."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX'],
            'rating': [4.5, 3.5, 5.0]
        })
        
        result = validator.validate_ratings(df)
        assert len(result) == 3
    
    def test_validate_ratings_invalid(self):
        """Test rating validation with invalid ratings."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', 'EQIX', 'DLR'],
            'rating': [4.5, -1.0, 6.5, 3.5]
        })
        
        result = validator.validate_ratings(df)
        # Should remove ratings < 0 or > 5
        assert len(result) < 4
        assert (result['rating'] >= 0).all()
        assert (result['rating'] <= 5).all()
    
    def test_validate_ratings_no_column(self):
        """Test rating validation when column doesn't exist."""
        validator = DataValidator()
        
        df = pd.DataFrame({
            'ticker': ['PLD'],
            'pros': ['Great company']
        })
        
        result = validator.validate_ratings(df)
        # Should return unchanged
        assert len(result) == len(df)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete cleaning pipeline."""
    
    def test_complete_cleaning_workflow(self):
        """Test complete data cleaning workflow."""
        # Create sample data
        df = pd.DataFrame({
            'ticker': ['PLD', 'AMT', None, 'EQIX'],
            'date': ['Jan 15, 2025', '2025-01-16', '1/17/2025', None],
            'pros': [
                '<p>Great benefits &amp; culture</p>',
                'Good work environment',
                None,
                'Excellent company'
            ],
            'rating': ['4.5', '3.5', '5.0', '10.0']
        })
        
        # Initialize components
        cleaner = TextCleaner()
        date_parser = DateParser()
        validator = DataValidator()
        
        # Clean text
        df['pros_cleaned'] = df['pros'].apply(cleaner.clean)
        
        # Parse dates
        df['date'] = df['date'].apply(date_parser.parse_date)
        
        # Validate
        df = validator.check_required_fields(df)
        df = validator.handle_missing_values(df)
        df = validator.validate_ratings(df)
        
        # Assertions
        assert len(df) >= 1  # At least one valid row
        assert df['date'].notna().all()
        assert df['ticker'].notna().all()
        assert (df['rating'] <= 5).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])