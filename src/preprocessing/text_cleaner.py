"""
Text Cleaning Module for REIT Sentiment Analysis

Provides comprehensive text preprocessing pipeline including HTML removal,
tokenization, stopword filtering, and stemming.

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
"""

import re
import pandas as pd
from typing import List, Optional, Set
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class TextCleaner:
    """
    Comprehensive text cleaning and preprocessing pipeline.
    
    Implements multi-stage text normalization:
    1. HTML/special character removal
    2. Tokenization
    3. Stopword removal
    4. Stemming
    5. Custom domain-specific filtering
    
    Attributes:
        remove_stopwords: Whether to remove English stopwords
        apply_stemming: Whether to apply Porter stemming
        min_word_length: Minimum word length to keep
        stopwords: Set of stopwords to remove
        stemmer: Porter stemmer instance
    
    Example:
        >>> cleaner = TextCleaner()
        >>> text = "<p>Great company with excellent benefits!</p>"
        >>> cleaned = cleaner.clean(text)
        >>> print(cleaned)
        'great compani excel benefit'
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
                'glassdoor', 'review', 'rating', 'thing', 'way',
                'really', 'lot', 'much', 'many', 'year', 'go'
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
            
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.remove_html("<p>Great &amp; wonderful</p>")
            'Great wonderful'
        """
        if pd.isna(text) or text == "":
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
            
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.normalize_text("GREAT Company!!!")
            'great company'
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
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
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
            
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.tokenize_and_filter("the company has great benefits")
            ['great', 'benefit']
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
            
        Example:
            >>> cleaner = TextCleaner()
            >>> text = "<p>Great benefits &amp; culture at https://example.com</p>"
            >>> cleaner.clean(text)
            'great benefit cultur'
        """
        # Step 1: Remove HTML
        text = self.remove_html(text)
        
        # Step 2: Normalize
        text = self.normalize_text(text)
        
        # Step 3: Tokenize and filter
        tokens = self.tokenize_and_filter(text)
        
        # Return as string
        return ' '.join(tokens)
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of text strings to clean
            
        Returns:
            List of cleaned text strings
            
        Example:
            >>> cleaner = TextCleaner()
            >>> texts = ["Great company", "Bad management"]
            >>> cleaner.clean_batch(texts)
            ['great compani', 'bad manag']
        """
        return [self.clean(text) for text in texts]


def clean_dataframe(df: pd.DataFrame, 
                    text_columns: List[str],
                    cleaner: Optional[TextCleaner] = None) -> pd.DataFrame:
    """
    Clean text columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_columns: List of column names containing text to clean
        cleaner: TextCleaner instance (creates default if None)
        
    Returns:
        DataFrame with cleaned text columns (adds '_cleaned' suffix)
        
    Example:
        >>> df = pd.DataFrame({'pros': ['Great company!', 'Bad management']})
        >>> df = clean_dataframe(df, ['pros'])
        >>> print(df['pros_cleaned'])
        0    great compani
        1    bad manag
    """
    if cleaner is None:
        cleaner = TextCleaner()
    
    df = df.copy()
    
    for col in text_columns:
        if col not in df.columns:
            continue
        
        # Keep original
        df[f'{col}_raw'] = df[col]
        
        # Create cleaned version
        df[f'{col}_cleaned'] = df[col].apply(cleaner.clean)
    
    return df


if __name__ == '__main__':
    # Example usage
    cleaner = TextCleaner()
    
    sample_texts = [
        "<p>Great company with excellent benefits &amp; culture!</p>",
        "Bad management, long hours, and poor communication.",
        "Visit https://example.com for more info about the job.",
    ]
    
    print("Text Cleaning Examples:")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        cleaned = cleaner.clean(text)
        print(f"\n{i}. Original: {text}")
        print(f"   Cleaned:  {cleaned}")