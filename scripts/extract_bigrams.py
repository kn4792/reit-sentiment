#!/usr/bin/env python3
"""
Bigram Extraction for REIT Employee Reviews

This script extracts bigrams (two-word phrases) from cleaned reviews,
identifies meaningful word pairs that co-occur, and calculates statistics
for subsequent weighting and AI productivity measure identification.

Author: Konain Abbas
Date: December 2025
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from itertools import combinations


class BigramExtractor:
    """Extracts and analyzes bigrams from employee reviews."""
    
    def __init__(self, min_freq: int = 5):
        """
        Initialize the BigramExtractor.
        
        Parameters
        ----------
        min_freq : int, default=5
            Minimum frequency threshold for including bigrams
        """
        self.min_freq = min_freq
        self.bigram_freq = Counter()
        self.bigram_by_company = defaultdict(Counter)
        self.bigram_by_property_type = defaultdict(Counter)
        self.bigram_by_sentiment = defaultdict(Counter)
        self.bigram_document_freq = Counter()
        
        # Track co-occurrence patterns
        self.word_pair_contexts = defaultdict(list)  # Store context for each bigram
    
    def extract_bigrams(self, df: pd.DataFrame, consecutive_only: bool = False) -> pd.DataFrame:
        """
        Extract bigrams from cleaned reviews.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cleaned and tokenized reviews
        consecutive_only : bool, default=False
            If True, only extract consecutive word pairs.
            If False, extract all word pairs that co-occur in same review.
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing bigram statistics
        """
        print(f"Extracting bigrams from reviews (consecutive_only={consecutive_only})...")
        
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
            self._process_tokens_for_bigrams(
                pros_tokens, company, property_type, 'pros', 
                consecutive_only, row
            )
            self._process_tokens_for_bigrams(
                cons_tokens, company, property_type, 'cons', 
                consecutive_only, row
            )
        
        print(f"\n  Extracted {len(self.bigram_freq):,} unique bigrams")
        
        # Filter by minimum frequency
        filtered_bigrams = {
            bigram: freq for bigram, freq in self.bigram_freq.items()
            if freq >= self.min_freq
        }
        print(f"  After filtering (min_freq={self.min_freq}): {len(filtered_bigrams):,} bigrams")
        
        return self._build_bigram_dataframe(total_reviews, filtered_bigrams)
    
    def _process_tokens_for_bigrams(self, tokens: List[str], company: str, 
                                   property_type: str, sentiment: str,
                                   consecutive_only: bool, row: pd.Series):
        """
        Process a list of tokens to extract bigrams.
        
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
            Whether to only extract consecutive bigrams
        row : pd.Series
            Full row data for context storage
        """
        if len(tokens) < 2:
            return
        
        # Track unique bigrams in this review segment for document frequency
        unique_bigrams = set()
        
        if consecutive_only:
            # Extract only consecutive bigrams (traditional n-gram approach)
            for i in range(len(tokens) - 1):
                word1, word2 = tokens[i], tokens[i + 1]
                bigram = (word1, word2)
                
                self._record_bigram(bigram, company, property_type, sentiment)
                unique_bigrams.add(bigram)
                
                # Store context
                context = ' '.join(tokens[max(0, i-2):min(len(tokens), i+3)])
                self.word_pair_contexts[bigram].append({
                    'company': company,
                    'property_type': property_type,
                    'sentiment': sentiment,
                    'context': context,
                    'date': row['date']
                })
        else:
            # Extract all co-occurring word pairs within the review
            # This captures semantic relationships even when words aren't adjacent
            for i, word1 in enumerate(tokens):
                for j, word2 in enumerate(tokens[i+1:], start=i+1):
                    # Only consider pairs within a reasonable window (e.g., 10 words)
                    if j - i > 10:
                        continue
                    
                    # Create ordered bigram (alphabetically for consistency)
                    bigram = tuple(sorted([word1, word2]))
                    
                    self._record_bigram(bigram, company, property_type, sentiment)
                    unique_bigrams.add(bigram)
                    
                    # Store context (words between the pair)
                    context = ' '.join(tokens[i:j+1])
                    self.word_pair_contexts[bigram].append({
                        'company': company,
                        'property_type': property_type,
                        'sentiment': sentiment,
                        'context': context,
                        'date': row['date']
                    })
        
        # Update document frequency
        for bigram in unique_bigrams:
            self.bigram_document_freq[bigram] += 1
    
    def _record_bigram(self, bigram: Tuple[str, str], company: str, 
                      property_type: str, sentiment: str):
        """Record a bigram occurrence with all metadata."""
        self.bigram_freq[bigram] += 1
        self.bigram_by_company[company][bigram] += 1
        self.bigram_by_property_type[property_type][bigram] += 1
        self.bigram_by_sentiment[sentiment][bigram] += 1
    
    def _build_bigram_dataframe(self, total_reviews: int, 
                               filtered_bigrams: Dict) -> pd.DataFrame:
        """
        Build a comprehensive dataframe of bigram statistics.
        
        Parameters
        ----------
        total_reviews : int
            Total number of reviews
        filtered_bigrams : Dict
            Dictionary of bigrams that passed the frequency filter
            
        Returns
        -------
        pd.DataFrame
            Bigram statistics
        """
        print("\nBuilding bigram statistics dataframe...")
        
        bigram_data = []
        
        for bigram, total_freq in filtered_bigrams.items():
            word1, word2 = bigram
            
            # Calculate statistics
            doc_freq = self.bigram_document_freq[bigram]
            doc_proportion = doc_freq / total_reviews
            
            # Calculate PMI (Pointwise Mutual Information)
            # PMI measures how much more likely two words are to co-occur than expected by chance
            # We'll need unigram frequencies for this - approximate from bigram data
            idf = np.log(total_reviews / doc_freq) if doc_freq > 0 else 0
            
            # Get sentiment distribution
            pros_freq = self.bigram_by_sentiment['pros'][bigram]
            cons_freq = self.bigram_by_sentiment['cons'][bigram]
            
            # Calculate sentiment ratio
            if cons_freq > 0:
                sentiment_ratio = pros_freq / cons_freq
            else:
                sentiment_ratio = float('inf') if pros_freq > 0 else 1.0
            
            # Get sample contexts (up to 5)
            sample_contexts = [
                ctx['context'] for ctx in self.word_pair_contexts[bigram][:5]
            ]
            
            bigram_data.append({
                'word1': word1,
                'word2': word2,
                'bigram': f"{word1} {word2}",
                'total_frequency': total_freq,
                'document_frequency': doc_freq,
                'document_proportion': doc_proportion,
                'idf': idf,
                'pros_frequency': pros_freq,
                'cons_frequency': cons_freq,
                'sentiment_ratio': sentiment_ratio,
                'pros_proportion': pros_freq / total_freq if total_freq > 0 else 0,
                'sample_contexts': '; '.join(sample_contexts[:3])  # Store 3 examples
            })
        
        # Create dataframe and sort by frequency
        df_bigrams = pd.DataFrame(bigram_data)
        df_bigrams = df_bigrams.sort_values('total_frequency', ascending=False)
        df_bigrams = df_bigrams.reset_index(drop=True)
        
        return df_bigrams
    
    def save_results(self, df_bigrams: pd.DataFrame, output_dir: str = '.'):
        """
        Save bigram analysis results.
        
        Parameters
        ----------
        df_bigrams : pd.DataFrame
            Bigram statistics dataframe
        output_dir : str
            Directory to save output files
        """
        # Save main bigram file
        bigram_file = f"{output_dir}/bigrams.csv"
        df_bigrams.to_csv(bigram_file, index=False)
        print(f"\nSaved bigrams to: {bigram_file}")
        
        # Save property type-specific bigrams
        property_data = []
        for prop_type, counter in self.bigram_by_property_type.items():
            for bigram, freq in counter.items():
                if freq >= self.min_freq:
                    property_data.append({
                        'property_type': prop_type,
                        'word1': bigram[0],
                        'word2': bigram[1],
                        'bigram': f"{bigram[0]} {bigram[1]}",
                        'frequency': freq
                    })
        
        if property_data:
            df_property_bigrams = pd.DataFrame(property_data)
            property_file = f"{output_dir}/bigrams_by_property_type.csv"
            df_property_bigrams.to_csv(property_file, index=False)
            print(f"Saved property type-specific bigrams to: {property_file}")
        
        # Generate summary statistics
        print("\n" + "="*60)
        print("BIGRAM SUMMARY STATISTICS")
        print("="*60)
        print(f"Total unique bigrams (after filtering): {len(df_bigrams):,}")
        print(f"Total bigram occurrences: {df_bigrams['total_frequency'].sum():,}")
        print(f"\nTop 30 most frequent bigrams:")
        print("-"*60)
        for idx, row in df_bigrams.head(30).iterrows():
            print(f"{idx+1:3d}. {row['bigram']:30s} "
                  f"freq={row['total_frequency']:5,d} "
                  f"docs={row['document_frequency']:4,d} "
                  f"ratio={row['sentiment_ratio']:6.2f}")
            
        # AI/Technology-related bigrams
        ai_keywords = ['ai', 'generati' 'artificial', 'intelligence', 'technology',
                       'digital', 'software', 'system', 'tool', 'platform',
                       'chatgpt', 'gpt', 'machine', 'learni', 'algo', 'bing',
                       'bard', 'robot', 'robotic', 'openai', 'llm', 'model', 'gemini',
                       'copilot', 'neural', 'network', 'automat', 'predict', 'data']

        #'ai', 'artifici', 'intellig',
            # 'chatgpt', 'gpt', 'openai',
            # 'generativ', 'llm', 'languag', 'model',
            # 'copilot', 'autom', 'automat',
            # 'machin', 'learn', 'neural', 'deep',
            # 'algorithm', 'predict', 'analyt',
            # 'digit', 'digital', 'technolog', 'tech',
            # 'platform', 'softwar', 'data', 'comput'

        
        
        ai_bigrams = df_bigrams[
            df_bigrams['word1'].str.contains('|'.join(ai_keywords), case=False, na=False) |
            df_bigrams['word2'].str.contains('|'.join(ai_keywords), case=False, na=False)
        ]
        
        if len(ai_bigrams) > 0:
            print(f"\nAI/Technology-related bigrams ({len(ai_bigrams)} found):")
            print("-"*60)
            for idx, row in ai_bigrams.head(20).iterrows():
                print(f"{row['bigram']:30s} freq={row['total_frequency']:4,d} "
                      f"ratio={row['sentiment_ratio']:6.2f}")
                

        



        
       
    


def process_bigrams(cleaned_file: str, output_dir: str = '.', 
                   consecutive_only: bool = False, min_freq: int = 5):
    """
    Process cleaned reviews and extract bigrams.
    
    Parameters
    ----------
    cleaned_file : str
        Path to the cleaned reviews CSV file
    output_dir : str
        Directory to save output files
    consecutive_only : bool
        Whether to only extract consecutive word pairs
    min_freq : int
        Minimum frequency threshold
    """
    print("Loading cleaned reviews...")
    df = pd.read_csv(cleaned_file, encoding='utf-8')
    print(f"Loaded {len(df):,} reviews")
    
    # Extract bigrams
    extractor = BigramExtractor(min_freq=min_freq)
    df_bigrams = extractor.extract_bigrams(df, consecutive_only=consecutive_only)
    
    # Save results
    extractor.save_results(df_bigrams, output_dir)
    
    return df_bigrams, extractor


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
    consecutive_only = '--consecutive' in sys.argv
    
    # Get min frequency
    min_freq = 5
    for i, arg in enumerate(sys.argv):
        if arg == '--min-freq' and i + 1 < len(sys.argv):
            min_freq = int(sys.argv[i + 1])
    
    print(f"Configuration:")
    print(f"  Consecutive only: {consecutive_only}")
    print(f"  Minimum frequency: {min_freq}")
    print()
    
    # Process bigrams
    df_bigrams, extractor = process_bigrams(
        cleaned_file, output_dir, consecutive_only, min_freq
    )
    
    print("\n" + "="*60)
    print("Bigram extraction complete!")
    print("="*60)