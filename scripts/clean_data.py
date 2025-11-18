#!/usr/bin/env python3
"""
Data Cleaning Script for REIT Glassdoor Reviews

This script implements a comprehensive data cleaning pipeline for scraped
Glassdoor reviews. It handles text normalization, tokenization, stemming,
date parsing, and data quality validation.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
Version: 1.0

Usage:
    python scripts/clean_data.py --input data/raw/all_reit_reviews_*.csv
    python scripts/clean_data.py --input data/raw/PLD_reviews.csv --output data/processed/
    python scripts/clean_data.py --input data/raw/*.csv --batch
"""

import re
import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("⚠️  Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✓ NLTK data downloaded")


class TextCleaner:
    """
    Comprehensive text cleaning and preprocessing pipeline.
    
    Implements multi-stage text normalization:
    1. HTML/special character removal
    2. Tokenization
    3. Stopword removal
    4. Stemming
    5. Custom domain-specific filtering
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 apply_stemming: bool = True,
                 min_word_length: int = 2,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize text cleaner with preprocessing options.
        
        Args:
            remove_stopwords: Whether to remove English stopwords
            apply_stemming: Whether to apply Porter stemming
            min_word_length: Minimum word length to keep
            custom_stopwords: Additional domain-specific stopwords
        """
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.min_word_length = min_word_length
        
        # Initialize NLTK tools
        self.stemmer = PorterStemmer() if apply_stemming else None
        
        # Load stopwords
        if remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
            
            # Add custom domain-specific stopwords
            domain_stopwords = {
                'reit', 'company', 'work', 'working', 'employee', 
                'employer', 'job', 'people', 'time', 'would', 
                'could', 'get', 'one', 'also', 'like', 'make',
                'glassdoor', 'review', 'rating'
            }
            self.stopwords.update(domain_stopwords)
            
            # Add user-provided custom stopwords
            if custom_stopwords:
                self.stopwords.update(custom_stopwords)
        else:
            self.stopwords = set()
    
    def remove_html(self, text: str) -> str:
        """
        Remove HTML tags and entities from text.
        
        Args:
            text: Raw text potentially containing HTML
            
        Returns:
            Cleaned text with HTML removed
        """
        if pd.isna(text):
            return ""
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(str(text), 'html.parser')
        text = soup.get_text(separator=' ')
        
        # Remove HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text: lowercase, remove special chars, extra spaces.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\!\?]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_filter(self, text: str) -> List[str]:
        """
        Tokenize text and apply filtering (stopwords, length, stemming).
        
        Args:
            text: Normalized text string
            
        Returns:
            List of processed tokens
        """
        if not text:
            return []
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback to simple split
            tokens = text.split()
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < self.min_word_length:
                continue
            
            # Skip if stopword
            if self.remove_stopwords and token in self.stopwords:
                continue
            
            # Skip if not alphabetic
            if not token.isalpha():
                continue
            
            # Apply stemming if enabled
            if self.apply_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            
            filtered_tokens.append(token)
        
        return filtered_tokens
    
    def clean(self, text: str) -> str:
        """
        Complete cleaning pipeline: HTML removal -> normalization -> tokenization.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string (space-separated tokens)
        """
        # Step 1: Remove HTML
        text = self.remove_html(text)
        
        # Step 2: Normalize
        text = self.normalize_text(text)
        
        # Step 3: Tokenize and filter
        tokens = self.tokenize_and_filter(text)
        
        # Return as string
        return ' '.join(tokens)


class DateParser:
    """
    Parse and normalize Glassdoor date formats to YYYY-MM-DD.
    
    Handles various formats:
    - "Jan 15, 2025"
    - "January 15, 2025"
    - "2025-01-15"
    - "1/15/2025"
    """
    
    @staticmethod
    def parse_date(date_str: str) -> Optional[str]:
        """
        Parse date string to standardized YYYY-MM-DD format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Standardized date string or None if parsing fails
        """
        if pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        
        # Try multiple date formats
        formats = [
            '%b %d, %Y',      # Jan 15, 2025
            '%B %d, %Y',      # January 15, 2025
            '%Y-%m-%d',       # 2025-01-15
            '%m/%d/%Y',       # 1/15/2025
            '%d/%m/%Y',       # 15/1/2025
            '%Y/%m/%d',       # 2025/1/15
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If all formats fail, return None
        return None


class DataValidator:
    """
    Validate data quality and handle missing values.
    
    Implements:
    - Missing value detection and handling
    - Outlier detection (>5 standard deviations)
    - Data type validation
    - Required field checks
    """
    
    def __init__(self, 
                 required_fields: List[str] = None,
                 outlier_std_threshold: float = 5.0):
        """
        Initialize data validator.
        
        Args:
            required_fields: List of column names that must be non-null
            outlier_std_threshold: Number of std devs for outlier detection
        """
        self.required_fields = required_fields or ['ticker', 'date']
        self.outlier_threshold = outlier_std_threshold
    
    def check_required_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing required fields.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with complete required fields
        """
        initial_len = len(df)
        
        for field in self.required_fields:
            if field in df.columns:
                df = df[df[field].notna()]
        
        removed = initial_len - len(df)
        if removed > 0:
            print(f"  ⚠️  Removed {removed} rows with missing required fields")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with forward-fill (up to 5 days for time series).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # For text fields, fill with empty string
        text_fields = ['pros', 'cons', 'title', 'employee_info']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna('')
        
        # For rating, forward-fill within company (limit 5)
        if 'rating' in df.columns and 'ticker' in df.columns:
            df['rating'] = df.groupby('ticker')['rating'].fillna(method='ffill', limit=5)
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, column: str = 'rating') -> pd.DataFrame:
        """
        Remove outliers using standard deviation threshold.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers
            
        Returns:
            DataFrame with outliers removed
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
        if removed > 0:
            print(f"  ⚠️  Removed {removed} outlier rows in {column}")
        
        return df
    
    def validate_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate rating values are in reasonable range (0-5).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with valid ratings
        """
        if 'rating' not in df.columns:
            return df
        
        # Convert to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Filter to 0-5 range
        initial_len = len(df)
        df = df[(df['rating'] >= 0) & (df['rating'] <= 5)]
        
        removed = initial_len - len(df)
        if removed > 0:
            print(f"  ⚠️  Removed {removed} rows with invalid ratings")
        
        return df


def clean_reviews(input_path: Path, 
                  output_dir: Path,
                  text_cleaner: TextCleaner,
                  date_parser: DateParser,
                  validator: DataValidator) -> pd.DataFrame:
    """
    Execute complete data cleaning pipeline on review data.
    
    Steps:
    1. Load raw data
    2. Parse and normalize dates
    3. Clean text fields (pros, cons, title)
    4. Validate data quality
    5. Remove duplicates
    6. Save cleaned data
    
    Args:
        input_path: Path to raw CSV file
        output_dir: Directory for cleaned output
        text_cleaner: TextCleaner instance
        date_parser: DateParser instance
        validator: DataValidator instance
        
    Returns:
        Cleaned DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Cleaning: {input_path.name}")
    print('='*60)
    
    # Load data
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df):,} reviews")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return pd.DataFrame()
    
    if len(df) == 0:
        print("✗ Empty file, skipping")
        return df
    
    # Parse dates
    print("📅 Parsing dates...")
    if 'date' in df.columns:
        df['date'] = df['date'].apply(date_parser.parse_date)
        df = df[df['date'].notna()]  # Remove rows with unparseable dates
        print(f"  ✓ Parsed {len(df):,} valid dates")
    
    # Clean text fields
    print("🧹 Cleaning text...")
    text_fields = ['pros', 'cons', 'title']
    
    for field in text_fields:
        if field in df.columns:
            print(f"  • {field}...", end=' ')
            df[f'{field}_raw'] = df[field]  # Keep original
            df[f'{field}_cleaned'] = df[field].apply(text_cleaner.clean)
            print("✓")
    
    # Validate data
    print("✅ Validating data...")
    df = validator.check_required_fields(df)
    df = validator.handle_missing_values(df)
    df = validator.validate_ratings(df)
    df = validator.remove_outliers(df)
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['ticker', 'date', 'title', 'pros'], keep='first')
    removed = initial_len - len(df)
    if removed > 0:
        print(f"  ⚠️  Removed {removed} duplicate reviews")
    
    # Sort by company and date
    if 'ticker' in df.columns and 'date' in df.columns:
        df = df.sort_values(['ticker', 'date'])
    
    # Save cleaned data
    output_file = output_dir / f"cleaned_{input_path.stem}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved {len(df):,} cleaned reviews → {output_file}")
    
    return df


def main():
    print("Data Cleaning Started")
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Clean and preprocess REIT Glassdoor reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean single file
  python scripts/clean_data.py --input data/raw/PLD_reviews.csv
  
  # Clean all files matching pattern
  python scripts/clean_data.py --input "data/raw/*_reviews.csv" --batch
  
  # Custom output directory
  python scripts/clean_data.py --input data/raw/all_reit_reviews.csv --output data/processed/
  
  # Disable stemming
  python scripts/clean_data.py --input data/raw/all.csv --no-stemming
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file(s) - use wildcards for multiple files'
    )
    parser.add_argument(
        '--output',
        default='data/processed',
        help='Output directory (default: data/processed)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple files in batch mode'
    )
    parser.add_argument(
        '--no-stopwords',
        action='store_true',
        help='Disable stopword removal'
    )
    parser.add_argument(
        '--no-stemming',
        action='store_true',
        help='Disable Porter stemming'
    )
    parser.add_argument(
        '--min-word-length',
        type=int,
        default=2,
        help='Minimum word length to keep (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize cleaning components
    print("\n🔧 Initializing cleaning pipeline...")
    text_cleaner = TextCleaner(
        remove_stopwords=not args.no_stopwords,
        apply_stemming=not args.no_stemming,
        min_word_length=args.min_word_length
    )
    date_parser = DateParser()
    validator = DataValidator()
    
    # Find input files
    input_pattern = Path(args.input)
    
    if args.batch or '*' in str(input_pattern):
        input_files = list(input_pattern.parent.glob(input_pattern.name))
        if not input_files:
            print(f"✗ No files found matching: {args.input}")
            return
        print(f"✓ Found {len(input_files)} files to process")
    else:
        if not input_pattern.exists():
            print(f"✗ File not found: {args.input}")
            return
        input_files = [input_pattern]
    
    # Process files
    all_cleaned = []
    successful = 0
    failed = 0
    
    for input_file in input_files:
        try:
            cleaned_df = clean_reviews(
                input_file,
                output_dir,
                text_cleaner,
                date_parser,
                validator
            )
            
            if len(cleaned_df) > 0:
                all_cleaned.append(cleaned_df)
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"✗ Error processing {input_file.name}: {e}")
            failed += 1
            continue
    
    # Save combined file if batch mode
    if len(all_cleaned) > 1:
        print(f"\n{'='*60}")
        print("📊 COMBINING FILES")
        print('='*60)
        
        combined = pd.concat(all_cleaned, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_file = output_dir / f'all_cleaned_reviews_{timestamp}.csv'
        combined.to_csv(combined_file, index=False)
        
        print(f"✓ Combined {len(combined):,} reviews from {len(all_cleaned)} files")
        print(f"💾 Saved → {combined_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 CLEANING SUMMARY")
    print('='*60)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    if all_cleaned:
        total_reviews = sum(len(df) for df in all_cleaned)
        print(f"📝 Total cleaned reviews: {total_reviews:,}")
        
        # Sample statistics
        sample_df = all_cleaned[0]
        if 'pros_cleaned' in sample_df.columns:
            avg_pros_words = sample_df['pros_cleaned'].str.split().str.len().mean()
            print(f"📊 Avg words per review (pros): {avg_pros_words:.1f}")
    
    print('='*60)
    print("✅ Cleaning complete!")


if __name__ == '__main__':
    main()