#!/usr/bin/env python3
"""
MNIR Stage 1: Text Preprocessing for GenAI Productivity Measurement

Prepares REIT Glassdoor review data for Multinomial Inverse Regression (MNIR)
following Campbell & Shang (2021) "Tone at the Bottom" methodology.

- Creates continuous GenAI Intensity measure (not binary treatment)
- Measures GenAI-related language for each firm-year
- No pre/post analysis - just intensity variation across firms and time

This script:
1. Cleans review text (remove punctuation, numbers, stop words, Porter stemming)
2. Builds filtered vocabulary (words appearing in 5-50% of reviews)
3. Aggregates reviews to firm-year level
4. Calculates GenAI Intensity (outcome variable for MNIR)
5. Creates word count matrices for each review section (Pros, Cons)
6. Saves all outputs for MNIR Stage 2 regression

Output Files:
- vocabulary.json: Filtered word list
- word_counts_pros.csv: Word counts from Pros section
- word_counts_cons.csv: Word counts from Cons section
- firm_year_data.csv: Aggregated data with GenAI intensity and controls
- preprocessing_stats.json: Summary statistics

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-11-28


Usage:
    python scripts/mnir_preprocessing_keywords.py
    python scripts/mnir_preprocessing_keywords.py --input data/raw/all_reit_reviews_merged.csv
    python scripts/mnir_preprocessing_keywords.py --min-reviews 10 --max-pct 0.4
"""

import sys
import argparse
import warnings
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple
from collections import Counter

import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt', quiet=True)


class MNIRPreprocessor:
    """
    Preprocessor for MNIR text analysis.
    
    Implements Campbell et al. (2021) text cleaning and vocabulary building.
    """
    
    def __init__(self, min_reviews: int = 5, max_pct: float = 0.5):
        """
        Initialize preprocessor.
        
        Args:
            min_reviews: Minimum reviews a word must appear in (default: 5)
            max_pct: Maximum percentage of reviews a word can appear in (default: 0.5)
        """
        self.min_reviews = min_reviews
        self.max_pct = max_pct
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vocabulary = []
        
        # GenAI-related keywords for intensity calculation
        # These are stemmed versions of GenAI-related terms
        self.genai_keywords = {
            'ai', 'artifici', 'intellig',
            'chatgpt', 'gpt', 'openai',
            'generativ', 'llm', 'languag', 'model',
            'copilot', 'autom', 'automat',
            'machin', 'learn', 'neural', 'deep',
            'algorithm', 'predict', 'analyt',
            'digit', 'digital', 'technolog', 'tech',
            'platform', 'softwar', 'data', 'comput'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean text following Campbell methodology.
        
        Steps:
        1. Remove non-English characters, numbers, punctuation
        2. Convert to lowercase
        3. Tokenize and remove stop words
        4. Remove words < 3 characters
        5. Apply Porter stemming
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned, stemmed text
        """
        if pd.isna(text) or not text:
            return ""
        
        # Remove numbers and punctuation, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Lowercase
        text = text.lower()
        
        # Tokenize
        words = text.split()
        
        # Remove stop words and short words
        words = [w for w in words if w not in self.stop_words and len(w) >= 3]
        
        # Porter stemming
        words = [self.stemmer.stem(w) for w in words]
        
        return ' '.join(words)
    
    def build_vocabulary(self, df: pd.DataFrame, 
                        text_columns: List[str]) -> List[str]:
        """
        Build filtered vocabulary from corpus.
        
        Filtering rules (Campbell et al.):
        - Remove words appearing in < min_reviews reviews (too rare)
        - Remove words appearing in > max_pct of reviews (too common)
        
        Args:
            df: DataFrame with cleaned text
            text_columns: Columns containing cleaned text
            
        Returns:
            Sorted list of vocabulary words
        """
        print(f"\nBuilding vocabulary...")
        print(f"  Filtering: {self.min_reviews} ≤ reviews ≤ {int(self.max_pct * len(df))}")
        
        word_doc_count = Counter()
        total_reviews = len(df)
        
        # Count document frequency for each word
        for col in text_columns:
            for text in df[col].fillna(''):
                if text:
                    # Use set to count documents, not raw frequency
                    words_in_review = set(text.split())
                    word_doc_count.update(words_in_review)
        
        print(f"Found {len(word_doc_count):,} unique words before filtering")
        
        # Filter vocabulary
        vocabulary = []
        filtered_counts = {'too_rare': 0, 'too_common': 0}
        
        for word, doc_count in word_doc_count.items():
            # Skip if too rare
            if doc_count < self.min_reviews:
                filtered_counts['too_rare'] += 1
                continue
            
            # Skip if too common
            if doc_count / total_reviews > self.max_pct:
                filtered_counts['too_common'] += 1
                continue
            
            vocabulary.append(word)
        
        self.vocabulary = sorted(vocabulary)
        
        print(f"Vocabulary size: {len(self.vocabulary):,} words")
        print(f"Removed {filtered_counts['too_rare']:,} too rare (< {self.min_reviews} reviews)")
        print(f"Removed {filtered_counts['too_common']:,} too common (> {self.max_pct:.0%} of reviews)")
        
        return self.vocabulary
    
    def calculate_genai_intensity(self, text: str) -> float:
        """
        Calculate GenAI intensity for a text.
        
        GenAI Intensity = (# GenAI-related words) / (# total words)
        
        This becomes the response variable for MNIR.
        
        Args:
            text: Cleaned, stemmed text
            
        Returns:
            GenAI intensity score (0.0 to 1.0)
        """
        if not text or pd.isna(text):
            return 0.0
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        genai_word_count = sum(1 for w in words if w in self.genai_keywords)
        
        return genai_word_count / len(words)
    
    def aggregate_to_firm_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate reviews to firm-year level.
        
        For each ticker-year:
        - Combine all review text (pros and cons separately)
        - Calculate average rating
        - Count number of reviews
        - Calculate GenAI intensity
        
        Args:
            df: Review-level DataFrame
            
        Returns:
            Firm-year DataFrame
        """
        print(f"\nAggregating to firm-year level...")
        
        # Extract year from date
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
        
        # Drop rows with invalid dates
        n_invalid = df['year'].isna().sum()
        if n_invalid > 0:
            print(f"Dropping {n_invalid} reviews with invalid dates")
            df = df.dropna(subset=['year'])
        
        # Aggregation dictionary
        agg_dict = {
            'pros_mnir': lambda x: ' '.join(x.fillna('')),
            'cons_mnir': lambda x: ' '.join(x.fillna('')),
            'rating': 'mean',
            'title': 'count'  # number of reviews
        }
        
        # Group by ticker and year
        firm_year = df.groupby(['ticker', 'year']).agg(agg_dict).reset_index()
        firm_year.rename(columns={'title': 'review_count'}, inplace=True)
        
        # Calculate GenAI intensity for each section
        firm_year['genai_intensity_pros'] = firm_year['pros_mnir'].apply(
            self.calculate_genai_intensity
        )
        firm_year['genai_intensity_cons'] = firm_year['cons_mnir'].apply(
            self.calculate_genai_intensity
        )
        
        # Combined GenAI intensity (average of pros and cons)
        firm_year['genai_intensity'] = (
            firm_year['genai_intensity_pros'] + firm_year['genai_intensity_cons']
        ) / 2
        
        # Calculate total word counts
        firm_year['word_count_pros'] = firm_year['pros_mnir'].apply(
            lambda x: len(x.split()) if x else 0
        )
        firm_year['word_count_cons'] = firm_year['cons_mnir'].apply(
            lambda x: len(x.split()) if x else 0
        )
        
        print(f"Firm-years: {len(firm_year):,}")
        print(f"Unique firms: {firm_year['ticker'].nunique()}")
        print(f"Year range: {firm_year['year'].min():.0f} - {firm_year['year'].max():.0f}")
        
        return firm_year
    
    def create_word_count_matrix(self, 
                                 firm_year_df: pd.DataFrame,
                                 text_column: str) -> pd.DataFrame:
        """
        Create word count matrix for MNIR.
        
        For each firm-year, count occurrences of each vocabulary word.
        Uses 'wc_' prefix for all word columns to avoid collisions.
        
        Args:
            firm_year_df: Firm-year aggregated data
            text_column: Column containing text
            
        Returns:
            DataFrame with columns: ticker, year, wc_word1, wc_word2, ..., wc_wordN
        """
        print(f"Creating count matrix for {text_column}...")
        
        word_counts = []
        
        for idx, row in firm_year_df.iterrows():
            text = row[text_column]
            
            # Count words in this firm-year
            if pd.isna(text) or not text:
                counts = {f'wc_{word}': 0 for word in self.vocabulary}
            else:
                word_list = text.split()
                word_counter = Counter(word_list)
                
                # Only count words in vocabulary, prefix with 'wc_'
                counts = {
                    f'wc_{word}': word_counter.get(word, 0) 
                    for word in self.vocabulary
                }
            
            # Create row with identifiers + counts
            row_data = {
                'ticker': row['ticker'],
                'year': row['year'],
                **counts
            }
            word_counts.append(row_data)
        
        df = pd.DataFrame(word_counts)
        print(f"Matrix shape: {df.shape}")
        
        return df


def calculate_statistics(df: pd.DataFrame, 
                        firm_year_df: pd.DataFrame,
                        vocabulary: List[str]) -> Dict:
    """
    Calculate summary statistics for preprocessing.
    
    Args:
        df: Review-level data
        firm_year_df: Firm-year data
        vocabulary: Vocabulary list
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'preprocessing_date': datetime.now().isoformat(),
        'review_level': {
            'total_reviews': len(df),
            'unique_firms': int(df['ticker'].nunique()),
            'date_range': {
                'min': str(df['date'].min()),
                'max': str(df['date'].max())
            }
        },
        'firm_year_level': {
            'total_observations': len(firm_year_df),
            'unique_firms': int(firm_year_df['ticker'].nunique()),
            'year_range': {
                'min': int(firm_year_df['year'].min()),
                'max': int(firm_year_df['year'].max())
            },
            'avg_reviews_per_firm_year': float(firm_year_df['review_count'].mean()),
            'median_reviews_per_firm_year': float(firm_year_df['review_count'].median())
        },
        'vocabulary': {
            'size': len(vocabulary),
            'top_50_words': vocabulary[:50]
        },
        'genai_intensity': {
            'mean_pros': float(firm_year_df['genai_intensity_pros'].mean()),
            'mean_cons': float(firm_year_df['genai_intensity_cons'].mean()),
            'mean_combined': float(firm_year_df['genai_intensity'].mean()),
            'std_combined': float(firm_year_df['genai_intensity'].std()),
            'min_combined': float(firm_year_df['genai_intensity'].min()),
            'max_combined': float(firm_year_df['genai_intensity'].max()),
            'pct_with_genai_mentions': float(
                (firm_year_df['genai_intensity'] > 0).sum() / len(firm_year_df) * 100
            )
        }
    }
    
    return stats


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='MNIR preprocessing for REIT reviews (v3.0 - continuous GenAI measure)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default settings)
  python scripts/mnir_preprocessing_v3.py
  
  # Custom input file
  python scripts/mnir_preprocessing_v3.py --input data/raw/my_reviews.csv
  
  # Stricter vocabulary filtering
  python scripts/mnir_preprocessing_v3.py --min-reviews 10 --max-pct 0.4
  
  # Save to custom output directory
  python scripts/mnir_preprocessing_v3.py --output data/processed/mnir_v3
        """
    )
    
    parser.add_argument(
        '--input',
        default='data/raw/all_reit_reviews_merged.csv',
        help='Input CSV file with reviews (default: data/raw/all_reit_reviews_merged.csv)'
    )
    parser.add_argument(
        '--output',
        default='data/processed/mnir',
        help='Output directory (default: data/processed/mnir)'
    )
    parser.add_argument(
        '--min-reviews',
        type=int,
        default=5,
        help='Minimum reviews a word must appear in (default: 5, following Campbell et al.)'
    )
    parser.add_argument(
        '--max-pct',
        type=float,
        default=0.5,
        help='Maximum percentage of reviews a word can appear in (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MNIR STAGE 1: TEXT PREPROCESSING (v3.0)")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Min reviews: {args.min_reviews}")
    print(f"Max percentage: {args.max_pct}")
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print('='*70)
    
    if not input_path.exists():
        print(f"✗ File not found: {input_path}")
        return 1
    
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} reviews")
        print(f"Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1
    
    # Check required columns
    required_cols = ['pros', 'cons', 'ticker', 'date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {', '.join(missing_cols)}")
        return 1
    
    # Initialize preprocessor
    preprocessor = MNIRPreprocessor(
        min_reviews=args.min_reviews,
        max_pct=args.max_pct
    )
    
    # Clean text
    print(f"\n{'='*70}")
    print("CLEANING TEXT")
    print('='*70)
    
    print("Processing Pros...")
    df['pros_mnir'] = df['pros'].apply(preprocessor.clean_text)
    
    print("Processing Cons...")
    df['cons_mnir'] = df['cons'].apply(preprocessor.clean_text)
    
    # Optional: handle advice column if it exists
    if 'advice' in df.columns:
        print("Processing Advice...")
        df['advice_mnir'] = df['advice'].apply(preprocessor.clean_text)
        text_columns = ['pros_mnir', 'cons_mnir', 'advice_mnir']
    else:
        text_columns = ['pros_mnir', 'cons_mnir']
    
    print("Text cleaning complete")
    
    # Build vocabulary
    print(f"\n{'='*70}")
    print("BUILDING VOCABULARY")
    print('='*70)
    
    vocabulary = preprocessor.build_vocabulary(df, text_columns)
    
    # Save vocabulary
    vocab_file = output_dir / 'vocabulary.json'
    with open(vocab_file, 'w') as f:
        json.dump(vocabulary, f, indent=2)
    print(f"Saved: {vocab_file}")
    
    # Aggregate to firm-year
    print(f"\n{'='*70}")
    print("AGGREGATING TO FIRM-YEAR")
    print('='*70)
    
    firm_year_df = preprocessor.aggregate_to_firm_year(df)
    
    # Create word count matrices
    print(f"\n{'='*70}")
    print("CREATING WORD COUNT MATRICES")
    print('='*70)
    
    pros_counts = preprocessor.create_word_count_matrix(
        firm_year_df, 'pros_mnir'
    )
    pros_file = output_dir / 'word_counts_pros.csv'
    pros_counts.to_csv(pros_file, index=False)
    print(f"Saved: {pros_file}")
    
    cons_counts = preprocessor.create_word_count_matrix(
        firm_year_df, 'cons_mnir'
    )
    cons_file = output_dir / 'word_counts_cons.csv'
    cons_counts.to_csv(cons_file, index=False)
    print(f"Saved: {cons_file}")
    
    # Save firm-year data with controls
    print(f"\n{'='*70}")
    print("SAVING FIRM-YEAR DATA")
    print('='*70)
    
    firm_year_output = firm_year_df[[
        'ticker', 'year', 'review_count', 'rating',
        'genai_intensity', 'genai_intensity_pros', 'genai_intensity_cons',
        'word_count_pros', 'word_count_cons'
    ]].copy()
    
    firm_year_file = output_dir / 'firm_year_data.csv'
    firm_year_output.to_csv(firm_year_file, index=False)
    print(f"Saved: {firm_year_file}")
    
    # Calculate and save statistics
    print(f"\n{'='*70}")
    print("CALCULATING STATISTICS")
    print('='*70)
    
    stats = calculate_statistics(df, firm_year_df, vocabulary)
    
    stats_file = output_dir / 'preprocessing_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print('='*70)
    print(f"\nSummary Statistics:")
    print(f"Reviews processed: {len(df):,}")
    print(f"Firm-years: {len(firm_year_df):,}")
    print(f"Unique firms: {firm_year_df['ticker'].nunique()}")
    print(f"Year range: {firm_year_df['year'].min():.0f}-{firm_year_df['year'].max():.0f}")
    print(f"Vocabulary size: {len(vocabulary):,} words")
    print(f"\nGenAI Intensity Statistics:")
    print(f"Mean intensity: {stats['genai_intensity']['mean_combined']:.4f}")
    print(f"Std intensity: {stats['genai_intensity']['std_combined']:.4f}")
    print(f"Min intensity: {stats['genai_intensity']['min_combined']:.4f}")
    print(f"Max intensity: {stats['genai_intensity']['max_combined']:.4f}")
    print(f"% with GenAI mentions: {stats['genai_intensity']['pct_with_genai_mentions']:.1f}%")
    print(f"\nOutput Files:")
    print(f"{vocab_file.name}")
    print(f"{pros_file.name}")
    print(f"{cons_file.name}")
    print(f"{firm_year_file.name}")
    print(f"{stats_file.name}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Review preprocessing_stats.json for data quality")
    print(f"  2. Run MNIR Stage 2 regressions:")
    print(f"     python scripts/mnir_regression_keywords.py")
    print(f"  3. Model: Poisson(word_count ~ genai_intensity + year_FE + controls)")
    print(f"  4. Extract word weights (φ_j) from regressions")
    print(f"  5. Create GenAI Productivity Index for each firm-year")
    print('='*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())