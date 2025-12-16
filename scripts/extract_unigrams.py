#!/usr/bin/env python3
"""
Unigram Extraction for REIT Employee Reviews

This script extracts unigrams (single words) from cleaned reviews,
calculates frequencies, and prepares data for subsequent weighting.

Author: Konain Abbas
Date: December 2025
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import json


class UnigramExtractor:
    """Extracts and analyzes unigrams from employee reviews."""
    
    def __init__(self):
        """Initialize the UnigramExtractor."""
        self.unigram_freq = Counter()
        self.unigram_by_company = defaultdict(Counter)
        self.unigram_by_property_type = defaultdict(Counter)
        self.unigram_by_job_title = defaultdict(Counter)
        self.unigram_by_sentiment = defaultdict(Counter)  # pros vs cons
        self.unigram_document_freq = Counter()  # How many reviews contain each word
    
    def extract_unigrams(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract unigrams from cleaned reviews.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cleaned and tokenized reviews
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing unigram statistics
        """
        print("Extracting unigrams from reviews...")
        
        total_reviews = len(df)
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"  Processing review {idx:,}/{total_reviews:,}", end='\r')
            
            # Get tokens
            pros_tokens = row['pros_tokens'].split() if isinstance(row['pros_tokens'], str) else []
            cons_tokens = row['cons_tokens'].split() if isinstance(row['cons_tokens'], str) else []
            
            # Get metadata
            company = row['company']
            property_type = row['property_type']
            job_title = row['job_title'] if pd.notna(row['job_title']) else 'Unknown'
            
            # Track unique words in this review for document frequency
            unique_words_in_review = set()
            
            # Process pros (positive sentiment)
            for word in pros_tokens:
                self.unigram_freq[word] += 1
                self.unigram_by_company[company][word] += 1
                self.unigram_by_property_type[property_type][word] += 1
                self.unigram_by_job_title[job_title][word] += 1
                self.unigram_by_sentiment['pros'][word] += 1
                unique_words_in_review.add(word)
            
            # Process cons (negative sentiment)
            for word in cons_tokens:
                self.unigram_freq[word] += 1
                self.unigram_by_company[company][word] += 1
                self.unigram_by_property_type[property_type][word] += 1
                self.unigram_by_job_title[job_title][word] += 1
                self.unigram_by_sentiment['cons'][word] += 1
                unique_words_in_review.add(word)
            
            # Update document frequency
            for word in unique_words_in_review:
                self.unigram_document_freq[word] += 1
        
        print(f"\n  Extracted {len(self.unigram_freq):,} unique unigrams")
        
        return self._build_unigram_dataframe(total_reviews)
    
    def _build_unigram_dataframe(self, total_reviews: int) -> pd.DataFrame:
        """
        Build a comprehensive dataframe of unigram statistics.
        
        Parameters
        ----------
        total_reviews : int
            Total number of reviews in the corpus
            
        Returns
        -------
        pd.DataFrame
            Unigram statistics
        """
        print("\nBuilding unigram statistics dataframe...")
        
        unigram_data = []
        
        for word, total_freq in self.unigram_freq.items():
            # Calculate basic statistics
            doc_freq = self.unigram_document_freq[word]
            doc_proportion = doc_freq / total_reviews
            
            # Calculate TF-IDF (will be useful for weighting)
            # IDF = log(N / df) where N is total docs and df is document frequency
            idf = np.log(total_reviews / doc_freq) if doc_freq > 0 else 0
            
            # Get sentiment distribution
            pros_freq = self.unigram_by_sentiment['pros'][word]
            cons_freq = self.unigram_by_sentiment['cons'][word]
            
            # Calculate sentiment ratio (positive to negative)
            if cons_freq > 0:
                sentiment_ratio = pros_freq / cons_freq
            else:
                sentiment_ratio = float('inf') if pros_freq > 0 else 1.0
            
            unigram_data.append({
                'word': word,
                'total_frequency': total_freq,
                'document_frequency': doc_freq,
                'document_proportion': doc_proportion,
                'idf': idf,
                'pros_frequency': pros_freq,
                'cons_frequency': cons_freq,
                'sentiment_ratio': sentiment_ratio,
                'pros_proportion': pros_freq / total_freq if total_freq > 0 else 0,
            })
        
        # Create dataframe and sort by frequency
        df_unigrams = pd.DataFrame(unigram_data)
        df_unigrams = df_unigrams.sort_values('total_frequency', ascending=False)
        df_unigrams = df_unigrams.reset_index(drop=True)
        
        return df_unigrams
    
    def save_results(self, df_unigrams: pd.DataFrame, output_dir: str = '.'):
        """
        Save unigram analysis results.
        
        Parameters
        ----------
        df_unigrams : pd.DataFrame
            Unigram statistics dataframe
        output_dir : str
            Directory to save output files
        """
        # Save main unigram file
        unigram_file = "data/processed/unigrams.csv"
        df_unigrams.to_csv(unigram_file, index=False)
        print(f"\nSaved unigrams to: {unigram_file}")
        
        # Save company-specific unigrams
        company_data = []
        for company, counter in self.unigram_by_company.items():
            for word, freq in counter.items():
                company_data.append({
                    'company': company,
                    'word': word,
                    'frequency': freq
                })
        
        df_company_unigrams = pd.DataFrame(company_data)
        company_file = "data/processed/unigrams_by_company.csv"
        df_company_unigrams.to_csv(company_file, index=False)
        print(f"Saved company-specific unigrams to: {company_file}")
        
        # Save property type-specific unigrams
        property_data = []
        for prop_type, counter in self.unigram_by_property_type.items():
            for word, freq in counter.items():
                property_data.append({
                    'property_type': prop_type,
                    'word': word,
                    'frequency': freq
                })
        
        df_property_unigrams = pd.DataFrame(property_data)
        property_file = "data/processed/unigrams_by_property_type.csv"
        df_property_unigrams.to_csv(property_file, index=False)
        print(f"Saved property type-specific unigrams to: {property_file}")
        
        # Save job title-specific unigrams (top titles only to keep file size manageable)
        job_data = []
        for job_title, counter in self.unigram_by_job_title.items():
            for word, freq in counter.items():
                job_data.append({
                    'job_title': job_title,
                    'word': word,
                    'frequency': freq
                })
        
        df_job_unigrams = pd.DataFrame(job_data)
        job_file = "data/processed/unigrams_by_job_title.csv"
        df_job_unigrams.to_csv(job_file, index=False)
        print(f"Saved job title-specific unigrams to: {job_file}")
        
        # Generate summary statistics
        print("\n" + "="*60)
        print("UNIGRAM SUMMARY STATISTICS")
        print("="*60)
        print(f"Total unique unigrams: {len(df_unigrams):,}")
        print(f"Total unigram occurrences: {df_unigrams['total_frequency'].sum():,}")
        print(f"\nTop 30 most frequent unigrams:")
        print("-"*60)
        for idx, row in df_unigrams.head(30).iterrows():
            print(f"{idx+1:3d}. {row['word']:20s} "
                  f"freq={row['total_frequency']:6,d} "
                  f"docs={row['document_frequency']:5,d} "
                  f"({row['document_proportion']*100:5.1f}%) "
                  f"pros/cons={row['sentiment_ratio']:6.2f}")
        
        # Words strongly associated with positive reviews
        print(f"\nTop 20 words most associated with PROS (high sentiment ratio):")
        print("-"*60)
        # Filter for words that appear at least 50 times
        frequent_unigrams = df_unigrams[df_unigrams['total_frequency'] >= 50].copy()
        frequent_unigrams = frequent_unigrams[frequent_unigrams['sentiment_ratio'] != float('inf')]
        frequent_unigrams = frequent_unigrams.sort_values('sentiment_ratio', ascending=False)
        
        for idx, row in frequent_unigrams.head(20).iterrows():
            print(f"{row['word']:20s} ratio={row['sentiment_ratio']:6.2f} "
                  f"(pros={row['pros_frequency']:4d}, cons={row['cons_frequency']:4d})")
        
        # Words strongly associated with negative reviews
        print(f"\nTop 20 words most associated with CONS (low sentiment ratio):")
        print("-"*60)
        bottom_sentiment = frequent_unigrams.sort_values('sentiment_ratio', ascending=True)
        
        for idx, row in bottom_sentiment.head(20).iterrows():
            print(f"{row['word']:20s} ratio={row['sentiment_ratio']:6.2f} "
                  f"(pros={row['pros_frequency']:4d}, cons={row['cons_frequency']:4d})")


def process_unigrams(cleaned_file: str, output_dir: str = '.'):
    """
    Process cleaned reviews and extract unigrams.
    
    Parameters
    ----------
    cleaned_file : str
        Path to the cleaned reviews CSV file
    output_dir : str
        Directory to save output files
    """
    print("Loading cleaned reviews...")
    df = pd.read_csv(cleaned_file, encoding='utf-8')
    print(f"Loaded {len(df):,} reviews")
    
    # Extract unigrams
    extractor = UnigramExtractor()
    df_unigrams = extractor.extract_unigrams(df)
    
    # Save results
    extractor.save_results(df_unigrams, output_dir)
    
    return df_unigrams, extractor


if __name__ == "__main__":
    import sys
    
    # Get input file
    if len(sys.argv) > 1:
        cleaned_file = sys.argv[1]
    else:
        cleaned_file = "data/processed/reviews_cleaned.csv"
    
    # Get output directory
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "data/processed"
    
    # Process unigrams
    df_unigrams, extractor = process_unigrams(cleaned_file, output_dir)
    
    print("\n" + "="*60)
    print("Unigram extraction complete!")
    print("="*60)