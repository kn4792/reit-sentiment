#!/usr/bin/env python3
"""
Trigram Extraction for REIT Employee Reviews

This script extracts trigrams (three-word phrases) from cleaned reviews,
identifies meaningful three-word combinations that co-occur, and calculates
statistics for identifying AI productivity measures.

Author: Konain Abbas
Date: December 2025
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from itertools import combinations


class TrigramExtractor:
    """Extracts and analyzes trigrams from employee reviews."""
    
    def __init__(self, min_freq: int = 3):
        """
        Initialize the TrigramExtractor.
        
        Parameters
        ----------
        min_freq : int, default=3
            Minimum frequency threshold for including trigrams
        """
        self.min_freq = min_freq
        self.trigram_freq = Counter()
        self.trigram_by_company = defaultdict(Counter)
        self.trigram_by_property_type = defaultdict(Counter)
        self.trigram_by_sentiment = defaultdict(Counter)
        self.trigram_document_freq = Counter()
        
        # Track co-occurrence patterns
        self.trigram_contexts = defaultdict(list)
    
    def extract_trigrams(self, df: pd.DataFrame, consecutive_only: bool = True,
                        max_window: int = 5) -> pd.DataFrame:
        """
        Extract trigrams from cleaned reviews.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cleaned and tokenized reviews
        consecutive_only : bool, default=True
            If True, only extract consecutive word triplets.
            If False, extract word triplets that co-occur within max_window.
        max_window : int, default=5
            Maximum distance between words in a trigram (only used if consecutive_only=False)
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing trigram statistics
        """
        print(f"Extracting trigrams from reviews (consecutive_only={consecutive_only})...")
        
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
            
            # Process pros and cons separately
            self._process_tokens_for_trigrams(
                pros_tokens, company, property_type, 'pros', 
                consecutive_only, max_window, row
            )
            self._process_tokens_for_trigrams(
                cons_tokens, company, property_type, 'cons', 
                consecutive_only, max_window, row
            )
        
        print(f"\n  Extracted {len(self.trigram_freq):,} unique trigrams")
        
        # Filter by minimum frequency
        filtered_trigrams = {
            trigram: freq for trigram, freq in self.trigram_freq.items()
            if freq >= self.min_freq
        }
        print(f"  After filtering (min_freq={self.min_freq}): {len(filtered_trigrams):,} trigrams")
        
        return self._build_trigram_dataframe(total_reviews, filtered_trigrams)
    
    def _process_tokens_for_trigrams(self, tokens: List[str], company: str,
                                    property_type: str, sentiment: str,
                                    consecutive_only: bool, max_window: int,
                                    row: pd.Series):
        """
        Process a list of tokens to extract trigrams.
        
        Parameters
        ----------
        tokens : List[str]
            List of tokens from a review segment
        company : str
            Company name
        property_type : str
            Property type
        sentiment : str
            'pros' or 'cons'
        consecutive_only : bool
            Whether to only extract consecutive trigrams
        max_window : int
            Maximum distance between words
        row : pd.Series
            Full row data for context storage
        """
        if len(tokens) < 3:
            return
        
        # Track unique trigrams in this review segment
        unique_trigrams = set()
        
        if consecutive_only:
            # Extract only consecutive trigrams (traditional n-gram approach)
            for i in range(len(tokens) - 2):
                word1, word2, word3 = tokens[i], tokens[i + 1], tokens[i + 2]
                trigram = (word1, word2, word3)
                
                self._record_trigram(trigram, company, property_type, sentiment)
                unique_trigrams.add(trigram)
                
                # Store context
                context = ' '.join(tokens[max(0, i-2):min(len(tokens), i+5)])
                self.trigram_contexts[trigram].append({
                    'company': company,
                    'property_type': property_type,
                    'sentiment': sentiment,
                    'context': context,
                    'date': row['date']
                })
        else:
            # Extract all co-occurring word triplets within the window
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + max_window, len(tokens))):
                    for k in range(j + 1, min(i + max_window, len(tokens))):
                        # Create ordered trigram (alphabetically for consistency)
                        trigram = tuple(sorted([tokens[i], tokens[j], tokens[k]]))
                        
                        self._record_trigram(trigram, company, property_type, sentiment)
                        unique_trigrams.add(trigram)
                        
                        # Store context
                        context = ' '.join(tokens[i:k+1])
                        self.trigram_contexts[trigram].append({
                            'company': company,
                            'property_type': property_type,
                            'sentiment': sentiment,
                            'context': context,
                            'date': row['date']
                        })
        
        # Update document frequency
        for trigram in unique_trigrams:
            self.trigram_document_freq[trigram] += 1
    
    def _record_trigram(self, trigram: Tuple[str, str, str], company: str,
                       property_type: str, sentiment: str):
        """Record a trigram occurrence with all metadata."""
        self.trigram_freq[trigram] += 1
        self.trigram_by_company[company][trigram] += 1
        self.trigram_by_property_type[property_type][trigram] += 1
        self.trigram_by_sentiment[sentiment][trigram] += 1
    
    def _build_trigram_dataframe(self, total_reviews: int,
                                filtered_trigrams: Dict) -> pd.DataFrame:
        """
        Build a comprehensive dataframe of trigram statistics.
        
        Parameters
        ----------
        total_reviews : int
            Total number of reviews
        filtered_trigrams : Dict
            Dictionary of trigrams that passed the frequency filter
            
        Returns
        -------
        pd.DataFrame
            Trigram statistics
        """
        print("\nBuilding trigram statistics dataframe...")
        
        trigram_data = []
        
        for trigram, total_freq in filtered_trigrams.items():
            word1, word2, word3 = trigram
            
            # Calculate statistics
            doc_freq = self.trigram_document_freq[trigram]
            doc_proportion = doc_freq / total_reviews
            
            idf = np.log(total_reviews / doc_freq) if doc_freq > 0 else 0
            
            # Get sentiment distribution
            pros_freq = self.trigram_by_sentiment['pros'][trigram]
            cons_freq = self.trigram_by_sentiment['cons'][trigram]
            
            # Calculate sentiment ratio
            if cons_freq > 0:
                sentiment_ratio = pros_freq / cons_freq
            else:
                sentiment_ratio = float('inf') if pros_freq > 0 else 1.0
            
            # Get sample contexts
            sample_contexts = [
                ctx['context'] for ctx in self.trigram_contexts[trigram][:5]
            ]
            
            trigram_data.append({
                'word1': word1,
                'word2': word2,
                'word3': word3,
                'trigram': f"{word1} {word2} {word3}",
                'total_frequency': total_freq,
                'document_frequency': doc_freq,
                'document_proportion': doc_proportion,
                'idf': idf,
                'pros_frequency': pros_freq,
                'cons_frequency': cons_freq,
                'sentiment_ratio': sentiment_ratio,
                'pros_proportion': pros_freq / total_freq if total_freq > 0 else 0,
                'sample_contexts': '; '.join(sample_contexts[:3])
            })
        
        # Create dataframe and sort by frequency
        df_trigrams = pd.DataFrame(trigram_data)
        df_trigrams = df_trigrams.sort_values('total_frequency', ascending=False)
        df_trigrams = df_trigrams.reset_index(drop=True)
        
        return df_trigrams
    
    def save_results(self, df_trigrams: pd.DataFrame, output_dir: str = '.'):
        """
        Save trigram analysis results.
        
        Parameters
        ----------
        df_trigrams : pd.DataFrame
            Trigram statistics dataframe
        output_dir : str
            Directory to save output files
        """
        # Save main trigram file
        trigram_file = f"{output_dir}/trigrams.csv"
        df_trigrams.to_csv(trigram_file, index=False)
        print(f"\nSaved trigrams to: {trigram_file}")
        
        # Save property type-specific trigrams
        property_data = []
        for prop_type, counter in self.trigram_by_property_type.items():
            for trigram, freq in counter.items():
                if freq >= self.min_freq:
                    property_data.append({
                        'property_type': prop_type,
                        'word1': trigram[0],
                        'word2': trigram[1],
                        'word3': trigram[2],
                        'trigram': f"{trigram[0]} {trigram[1]} {trigram[2]}",
                        'frequency': freq
                    })
        
        if property_data:
            df_property_trigrams = pd.DataFrame(property_data)
            property_file = f"{output_dir}/trigrams_by_property_type.csv"
            df_property_trigrams.to_csv(property_file, index=False)
            print(f"Saved property type-specific trigrams to: {property_file}")
        
        # Generate summary statistics
        print("\n" + "="*60)
        print("TRIGRAM SUMMARY STATISTICS")
        print("="*60)
        print(f"Total unique trigrams (after filtering): {len(df_trigrams):,}")
        print(f"Total trigram occurrences: {df_trigrams['total_frequency'].sum():,}")
        print(f"\nTop 30 most frequent trigrams:")
        print("-"*60)
        for idx, row in df_trigrams.head(30).iterrows():
            print(f"{idx+1:3d}. {row['trigram']:35s} "
                  f"freq={row['total_frequency']:4,d} "
                  f"docs={row['document_frequency']:3,d}")
        
        
        
        
        
        


def process_trigrams(cleaned_file: str, output_dir: str = '.',
                    consecutive_only: bool = True, min_freq: int = 3,
                    max_window: int = 5):
    """
    Process cleaned reviews and extract trigrams.
    
    Parameters
    ----------
    cleaned_file : str
        Path to the cleaned reviews CSV file
    output_dir : str
        Directory to save output files
    consecutive_only : bool
        Whether to only extract consecutive word triplets
    min_freq : int
        Minimum frequency threshold
    max_window : int
        Maximum window size for non-consecutive trigrams
    """
    print("Loading cleaned reviews...")
    df = pd.read_csv(cleaned_file, encoding='utf-8')
    print(f"Loaded {len(df):,} reviews")
    
    # Extract trigrams
    extractor = TrigramExtractor(min_freq=min_freq)
    df_trigrams = extractor.extract_trigrams(
        df, consecutive_only=consecutive_only, max_window=max_window
    )
    
    # Save results
    extractor.save_results(df_trigrams, output_dir)
    
    return df_trigrams, extractor


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
    
    # Get consecutive_only flag
    consecutive_only = '--consecutive' in sys.argv or True  # Default to consecutive
    
    # Get min frequency
    min_freq = 3
    for i, arg in enumerate(sys.argv):
        if arg == '--min-freq' and i + 1 < len(sys.argv):
            min_freq = int(sys.argv[i + 1])
    
    # Get max window
    max_window = 5
    for i, arg in enumerate(sys.argv):
        if arg == '--max-window' and i + 1 < len(sys.argv):
            max_window = int(sys.argv[i + 1])
    
    print(f"Configuration:")
    print(f"  Consecutive only: {consecutive_only}")
    print(f"  Minimum frequency: {min_freq}")
    print(f"  Maximum window: {max_window}")
    print()
    
    # Process trigrams
    df_trigrams, extractor = process_trigrams(
        cleaned_file, output_dir, consecutive_only, min_freq, max_window
    )
    
    print("\n" + "="*60)
    print("Trigram extraction complete!")
    print("="*60)