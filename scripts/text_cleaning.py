#!/usr/bin/env python3
"""
Text Cleaning and Vocabulary Extraction for REIT Employee Reviews

This script processes raw Glassdoor reviews, performs text cleaning,
removes stopwords, and generates a comprehensive vocabulary list.

Author: Konain Abbas
Date: December 2025
"""

import pandas as pd
import re
import string
from collections import Counter
from typing import List, Set

# Built-in English stopwords (avoiding NLTK network dependency)
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def simple_word_tokenize(text: str) -> List[str]:
    """Simple word tokenizer that splits on whitespace and punctuation."""
    # Split on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


class TextCleaner:
    """Handles text cleaning and preprocessing for employee reviews."""
    
    def __init__(self, custom_stopwords: List[str] = None):
        """
        Initialize the TextCleaner.
        
        Parameters
        ----------
        custom_stopwords : List[str], optional
            Additional domain-specific stopwords to remove
        """
        # Get standard English stopwords only
        # No custom domain-specific stopwords - keep all potentially meaningful words
        # for manual review and context-based weighting
        self.stopwords = ENGLISH_STOPWORDS.copy()
        
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string.
        
        Parameters
        ----------
        text : str
            Raw text to clean
            
        Returns
        -------
        str
            Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces and basic punctuation
        # This preserves sentence structure for n-gram extraction
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        
        # Remove numbers (optional - comment out if you want to keep them)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_filter(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text and optionally remove stopwords.
        
        Parameters
        ----------
        text : str
            Cleaned text to tokenize
        remove_stopwords : bool, default=True
            Whether to remove stopwords
            
        Returns
        -------
        List[str]
            List of filtered tokens
        """
        if not text:
            return []
        
        # Tokenize  
        tokens = simple_word_tokenize(text)
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip if it's just punctuation
            if token in string.punctuation:
                continue
            
            # Skip if too short (likely not meaningful)
            if len(token) < 2:
                continue
            
            # Skip stopwords if requested
            if remove_stopwords and token in self.stopwords:
                continue
            
            filtered_tokens.append(token)
        
        return filtered_tokens


def process_reviews(input_file: str, output_dir: str = '.'):
    """
    Process all reviews and generate cleaned vocabulary.
    
    Parameters
    ----------
    input_file : str
        Path to the raw CSV file containing reviews
    output_dir : str, default='.'
        Directory to save output files
    """
    print("Loading reviews data...")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    print(f"Loaded {len(df):,} reviews from {df['company'].nunique()} companies")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Initialize cleaner
    cleaner = TextCleaner()
    
    # Process pros and cons separately (as you mentioned this is better)
    print("\nCleaning text...")
    df['pros_cleaned'] = df['pros'].apply(cleaner.clean_text)
    df['cons_cleaned'] = df['cons'].apply(cleaner.clean_text)
    
    # Tokenize and filter
    print("Tokenizing and removing stopwords...")
    df['pros_tokens'] = df['pros_cleaned'].apply(
        lambda x: cleaner.tokenize_and_filter(x, remove_stopwords=True)
    )
    df['cons_tokens'] = df['cons_cleaned'].apply(
        lambda x: cleaner.tokenize_and_filter(x, remove_stopwords=True)
    )
    
    # Count token lengths
    df['pros_token_count'] = df['pros_tokens'].apply(len)
    df['cons_token_count'] = df['cons_tokens'].apply(len)
    df['total_token_count'] = df['pros_token_count'] + df['cons_token_count']
    
    print(f"\nToken statistics:")
    print(f"  Mean tokens per review: {df['total_token_count'].mean():.1f}")
    print(f"  Median tokens per review: {df['total_token_count'].median():.1f}")
    print(f"  Max tokens in a review: {df['total_token_count'].max()}")
    
    # Build complete vocabulary
    print("\nBuilding vocabulary...")
    all_tokens = []
    for tokens in df['pros_tokens']:
        all_tokens.extend(tokens)
    for tokens in df['cons_tokens']:
        all_tokens.extend(tokens)
    
    # Count word frequencies
    word_freq = Counter(all_tokens)
    
    print(f"\nVocabulary statistics:")
    print(f"  Total tokens (with repetition): {len(all_tokens):,}")
    print(f"  Unique words (vocabulary size): {len(word_freq):,}")
    
    # Save vocabulary with frequencies
    vocab_df = pd.DataFrame([
        {'word': word, 'frequency': freq}
        for word, freq in word_freq.most_common()
    ])
    
    vocab_file = "data/processed/vocabulary.csv"
    vocab_df.to_csv(vocab_file, index=False)
    print(f"\nSaved vocabulary to: {vocab_file}")
    
    # Save cleaned reviews with tokens
    cleaned_file = "data/processed/reviews_cleaned.csv"
    
    # Prepare output dataframe
    output_df = df[[
        'title', 'rating', 'date', 'job_title', 'company', 'ticker', 
        'property_type', 'scrape_date',
        'pros_cleaned', 'cons_cleaned',
        'pros_token_count', 'cons_token_count', 'total_token_count'
    ]].copy()
    
    # Add tokens as space-separated strings for easier reading
    output_df['pros_tokens'] = df['pros_tokens'].apply(lambda x: ' '.join(x))
    output_df['cons_tokens'] = df['cons_tokens'].apply(lambda x: ' '.join(x))
    
    output_df.to_csv(cleaned_file, index=False)
    print(f"Saved cleaned reviews to: {cleaned_file}")
    
    # Display top 50 most common words
    print("\nTop 50 most frequent words:")
    print("-" * 60)
    for i, (word, freq) in enumerate(word_freq.most_common(50), 1):
        print(f"{i:2d}. {word:20s} {freq:6,d} occurrences")
    
    # Save statistics
    stats = {
        'total_reviews': len(df),
        'total_companies': df['company'].nunique(),
        'total_tokens': len(all_tokens),
        'vocabulary_size': len(word_freq),
        'avg_tokens_per_review': df['total_token_count'].mean(),
        'median_tokens_per_review': df['total_token_count'].median(),
        'date_range_start': df['date'].min(),
        'date_range_end': df['date'].max(),
    }
    
    stats_df = pd.DataFrame([stats])
    stats_file = "data/processed/cleaning_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\nSaved statistics to: {stats_file}")
    
    return df, vocab_df, word_freq


if __name__ == "__main__":
    import sys
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "data/raw/all_reviews.csv"
    
    # Get output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "data/processed"
    
    # Process reviews
    df, vocab_df, word_freq = process_reviews(input_file, output_dir)
    
    print("\n" + "="*60)
    print("Text cleaning complete!")
    print("="*60)