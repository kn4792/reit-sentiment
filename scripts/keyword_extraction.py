#!/usr/bin/env python3
"""
Keyword Extraction Script for REIT Glassdoor Reviews

Implements TF-IDF based keyword extraction with:
- Company-level keyword importance
- AI/technology keyword detection
- Category-based keyword classification
- N-gram support (unigrams, bigrams, trigrams)

Author: Konain Niaz (kn4792@rit.edu)
Date: 2025-01-17
Version: 1.0

Usage:
    python scripts/keyword_extraction.py --input data/processed/cleaned_reviews.csv
    python scripts/keyword_extraction.py --input data/processed/*.csv --categories ai,management
    python scripts/keyword_extraction.py --input data/processed/cleaned.csv --ngrams 2 --top-k 50
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')


class KeywordExtractor:
    """
    TF-IDF based keyword extraction for review analysis.
    
    Features:
    - Company-specific keyword importance
    - Multi-word phrases (n-grams)
    - Custom category detection (AI, management, culture, etc.)
    """
    
    def __init__(self,
                 max_features: int = 100,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        Initialize keyword extractor.
        
        Args:
            max_features: Maximum number of keywords to extract
            ngram_range: (min_n, max_n) for n-gram extraction
            min_df: Minimum document frequency for keywords
            max_df: Maximum document frequency (as fraction)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = None
        self.feature_names = None
    
    def fit(self, texts: List[str]):
        """
        Fit TF-IDF vectorizer on corpus.
        
        Args:
            texts: List of text documents
        """
        print(f"🔧 Fitting TF-IDF vectorizer...")
        print(f"  • N-gram range: {self.ngram_range}")
        print(f"  • Max features: {self.max_features}")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=None  # Already removed in cleaning
        )
        
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✓ Extracted {len(self.feature_names)} features")
    
    def extract_top_keywords(self, 
                            texts: List[str],
                            top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract top-k keywords from texts.
        
        Args:
            texts: List of text documents
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        # Transform texts
        tfidf_matrix = self.vectorizer.transform(texts)
        
        # Sum TF-IDF scores across documents
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Get keywords and scores
        keywords = [(self.feature_names[i], scores[i]) for i in top_indices]
        
        return keywords
    
    def extract_by_company(self,
                          df: pd.DataFrame,
                          text_column: str = 'pros_cleaned',
                          top_k: int = 20) -> pd.DataFrame:
        """
        Extract top keywords for each company.
        
        Args:
            df: DataFrame with company reviews
            text_column: Column containing cleaned text
            top_k: Number of keywords per company
            
        Returns:
            DataFrame with columns: ticker, keyword, tfidf_score, rank
        """
        print(f"\n🔍 Extracting keywords by company...")
        
        results = []
        
        for ticker in df['ticker'].unique():
            company_df = df[df['ticker'] == ticker]
            texts = company_df[text_column].fillna('').tolist()
            
            # Skip if no text
            if not texts or all(t == '' for t in texts):
                continue
            
            # Extract keywords
            keywords = self.extract_top_keywords(texts, top_k=top_k)
            
            # Store results
            for rank, (keyword, score) in enumerate(keywords, 1):
                results.append({
                    'ticker': ticker,
                    'keyword': keyword,
                    'tfidf_score': score,
                    'rank': rank
                })
            
            print(f"  ✓ {ticker}: {len(keywords)} keywords")
        
        results_df = pd.DataFrame(results)
        print(f"\n✓ Extracted keywords for {results_df['ticker'].nunique()} companies")
        
        return results_df


class CategoryClassifier:
    """
    Classify reviews into predefined categories based on keywords.
    
    Categories:
    - AI/Technology
    - Management
    - Culture
    - Compensation
    - Work-life balance
    """
    
    def __init__(self):
        """Initialize with predefined category keywords."""
        
        self.categories = {
            'ai_technology': {
                'ai', 'artificial', 'intelligence', 'machine', 'learning', 
                'algorithm', 'data', 'analytics', 'automation', 'digital',
                'technology', 'software', 'platform', 'system', 'tool',
                'chatgpt', 'gpt', 'neural', 'model', 'computer', 'vision',
                'nlp', 'deep', 'proptech', 'tech', 'technological'
            },
            
            'management': {
                'management', 'manager', 'leadership', 'executive', 'ceo',
                'director', 'supervisor', 'boss', 'lead', 'senior',
                'decision', 'strategy', 'vision', 'communication',
                'micromanage', 'transparent', 'support', 'feedback'
            },
            
            'culture': {
                'culture', 'team', 'colleague', 'collaborative', 'friendly',
                'environment', 'atmosphere', 'vibe', 'people', 'staff',
                'inclusive', 'diverse', 'diversity', 'respect', 'trust',
                'toxic', 'political', 'clique'
            },
            
            'compensation': {
                'salary', 'pay', 'compensation', 'wage', 'bonus', 'incentive',
                'benefit', 'insurance', 'healthcare', 'retirement', '401k',
                'stock', 'equity', 'raise', 'promotion', 'competitive'
            },
            
            'work_life_balance': {
                'balance', 'flexible', 'flexibility', 'remote', 'wfh',
                'hybrid', 'hour', 'overtime', 'vacation', 'pto', 'leave',
                'weekend', 'schedule', 'burnout', 'stress'
            }
        }
    
    def classify_text(self, text: str) -> Dict[str, bool]:
        """
        Classify single text into categories.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with category names as keys, boolean values
        """
        if not text or pd.isna(text):
            return {cat: False for cat in self.categories.keys()}
        
        words = set(text.lower().split())
        
        classifications = {}
        for category, keywords in self.categories.items():
            classifications[category] = bool(words & keywords)
        
        return classifications
    
    def classify_dataframe(self,
                          df: pd.DataFrame,
                          text_column: str = 'pros_cleaned') -> pd.DataFrame:
        """
        Classify all reviews in DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            
        Returns:
            DataFrame with category columns added
        """
        print(f"\n🏷️  Classifying reviews into categories...")
        
        # Classify each review
        classifications = df[text_column].apply(self.classify_text)
        
        # Add category columns
        for category in self.categories.keys():
            df[f'category_{category}'] = [c[category] for c in classifications]
        
        # Print statistics
        print("\n📊 Category distribution:")
        for category in self.categories.keys():
            count = df[f'category_{category}'].sum()
            pct = (count / len(df)) * 100
            print(f"  • {category}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def get_category_keywords(self, category: str) -> Set[str]:
        """
        Get keywords for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Set of keywords
        """
        return self.categories.get(category, set())


class AIAdoptionAnalyzer:
    """
    Analyze AI adoption mentions in reviews over time.
    
    Tracks mentions of AI/technology keywords and calculates
    adoption metrics at company-month level.
    """
    
    def __init__(self):
        """Initialize with AI-related keywords."""
        
        self.ai_keywords = {
            'ai', 'artificial intelligence', 'machine learning', 'ml',
            'automation', 'chatgpt', 'gpt', 'algorithm', 'data science',
            'analytics', 'predictive', 'neural network', 'deep learning',
            'nlp', 'computer vision', 'proptech'
        }
    
    def detect_ai_mentions(self, text: str) -> Dict[str, any]:
        """
        Detect AI mentions in text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with ai_mentioned (bool) and mention_count (int)
        """
        if not text or pd.isna(text):
            return {'ai_mentioned': False, 'mention_count': 0}
        
        text_lower = text.lower()
        
        # Count mentions
        mention_count = 0
        for keyword in self.ai_keywords:
            mention_count += text_lower.count(keyword)
        
        return {
            'ai_mentioned': mention_count > 0,
            'mention_count': mention_count
        }
    
    def analyze_dataframe(self,
                         df: pd.DataFrame,
                         text_column: str = 'pros_cleaned') -> pd.DataFrame:
        """
        Analyze AI mentions for all reviews.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            
        Returns:
            DataFrame with AI mention columns
        """
        print(f"\n🤖 Analyzing AI adoption mentions...")
        
        # Detect AI mentions
        ai_data = df[text_column].apply(self.detect_ai_mentions)
        
        df['ai_mentioned'] = [d['ai_mentioned'] for d in ai_data]
        df['ai_mention_count'] = [d['mention_count'] for d in ai_data]
        
        # Statistics
        total_mentions = df['ai_mentioned'].sum()
        pct = (total_mentions / len(df)) * 100
        
        print(f"  ✓ Found {total_mentions:,} reviews with AI mentions ({pct:.1f}%)")
        print(f"  ✓ Total AI keyword occurrences: {df['ai_mention_count'].sum():,}")
        
        return df
    
    def aggregate_by_time(self,
                         df: pd.DataFrame,
                         output_path: Path) -> pd.DataFrame:
        """
        Aggregate AI mentions to company-month level.
        
        Args:
            df: Review-level DataFrame
            output_path: Path to save aggregated data
            
        Returns:
            Aggregated DataFrame
        """
        print(f"\n📅 Aggregating AI mentions by company-month...")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Aggregate
        agg_dict = {
            'ai_mentioned': 'sum',
            'ai_mention_count': 'sum'
        }
        # Add a temp column that's always 1 for counting
        df['review_count_tmp'] = 1

        monthly = (
            df.groupby(['ticker', 'year_month'])
            .agg({**agg_dict, 'review_count_tmp': 'sum'})
            .reset_index()
        )
        monthly.rename(columns={'review_count_tmp': 'review_count'}, inplace=True)
        
        # Calculate percentage
        monthly['ai_mention_pct'] = (monthly['ai_mentioned'] / monthly['review_count']) * 100
        
        # Convert period to string
        monthly['year_month'] = monthly['year_month'].astype(str)
        
        # Save
        monthly.to_csv(output_path, index=False)
        print(f"✓ Saved {len(monthly):,} company-month observations → {output_path}")
        
        return monthly


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract keywords and analyze categories in REIT reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic keyword extraction
  python scripts/04_keyword_extraction.py --input data/processed/cleaned_reviews.csv
  
  # Extract bigrams and trigrams
  python scripts/04_keyword_extraction.py --input data/processed/cleaned.csv --ngrams 3
  
  # Focus on specific categories
  python scripts/04_keyword_extraction.py --input data/processed/cleaned.csv --categories ai,management
  
  # Custom number of keywords
  python scripts/04_keyword_extraction.py --input data/processed/cleaned.csv --top-k 50
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file with cleaned reviews'
    )
    parser.add_argument(
        '--output',
        default='data/results',
        help='Output directory (default: data/results)'
    )
    parser.add_argument(
        '--text-column',
        default='pros_cleaned',
        help='Column containing cleaned text (default: pros_cleaned)'
    )
    parser.add_argument(
        '--ngrams',
        type=int,
        default=2,
        help='Maximum n-gram size (default: 2 for bigrams)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of top keywords per company (default: 20)'
    )
    parser.add_argument(
        '--categories',
        help='Comma-separated list of categories to analyze (e.g., ai,management)'
    )
    parser.add_argument(
        '--skip-ai-analysis',
        action='store_true',
        help='Skip AI adoption analysis'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*60}")
    print("📂 LOADING DATA")
    print('='*60)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ File not found: {args.input}")
        return
    
    try:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df):,} reviews from {input_path.name}")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # Check for required columns
    if args.text_column not in df.columns:
        print(f"✗ Column '{args.text_column}' not found in data")
        return
    
    # Initialize keyword extractor
    print(f"\n{'='*60}")
    print("🔧 KEYWORD EXTRACTION")
    print('='*60)
    
    extractor = KeywordExtractor(
        max_features=100,
        ngram_range=(1, args.ngrams),
        min_df=2,
        max_df=0.8
    )
    
    # Fit on all texts
    all_texts = df[args.text_column].fillna('').tolist()
    extractor.fit(all_texts)
    
    # Extract keywords by company
    keywords_df = extractor.extract_by_company(
        df,
        text_column=args.text_column,
        top_k=args.top_k
    )
    
    # Save keywords
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    keywords_file = output_dir / f'keywords_tfidf_{timestamp}.csv'
    keywords_df.to_csv(keywords_file, index=False)
    print(f"\n💾 Saved keywords → {keywords_file}")
    
    # Category classification
    print(f"\n{'='*60}")
    print("🏷️  CATEGORY CLASSIFICATION")
    print('='*60)
    
    classifier = CategoryClassifier()
    df = classifier.classify_dataframe(df, text_column=args.text_column)
    
    # Save categorized reviews
    categorized_file = output_dir / f'reviews_categorized_{timestamp}.csv'
    df.to_csv(categorized_file, index=False)
    print(f"\n💾 Saved categorized reviews → {categorized_file}")
    
    # AI adoption analysis
    if not args.skip_ai_analysis:
        print(f"\n{'='*60}")
        print("🤖 AI ADOPTION ANALYSIS")
        print('='*60)
        
        ai_analyzer = AIAdoptionAnalyzer()
        df = ai_analyzer.analyze_dataframe(df, text_column=args.text_column)
        
        # Aggregate by time
        if 'ticker' in df.columns and 'date' in df.columns:
            ai_monthly_file = output_dir / f'ai_adoption_monthly_{timestamp}.csv'
            ai_monthly = ai_analyzer.aggregate_by_time(df, ai_monthly_file)
            
            print(f"\n📊 AI Adoption Statistics:")
            print(f"  • Date range: {ai_monthly['year_month'].min()} to {ai_monthly['year_month'].max()}")
            print(f"  • Companies: {ai_monthly['ticker'].nunique()}")
            print(f"  • Avg AI mention %: {ai_monthly['ai_mention_pct'].mean():.2f}%")
    
    # Category-specific keyword analysis
    if args.categories:
        print(f"\n{'='*60}")
        print("📊 CATEGORY-SPECIFIC KEYWORDS")
        print('='*60)
        
        categories = [c.strip() for c in args.categories.split(',')]
        
        for category in categories:
            col_name = f'category_{category}'
            if col_name not in df.columns:
                print(f"⚠️  Category '{category}' not found")
                continue
            
            # Filter to category
            cat_df = df[df[col_name] == True]
            
            if len(cat_df) == 0:
                print(f"⚠️  No reviews in category '{category}'")
                continue
            
            print(f"\n🔍 {category.upper()} ({len(cat_df):,} reviews)")
            
            # Extract keywords
            cat_texts = cat_df[args.text_column].fillna('').tolist()
            cat_keywords = extractor.extract_top_keywords(cat_texts, top_k=15)
            
            print("  Top keywords:")
            for i, (kw, score) in enumerate(cat_keywords[:10], 1):
                print(f"    {i:2d}. {kw:20s} ({score:.3f})")
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ KEYWORD EXTRACTION COMPLETE")
    print('='*60)
    print(f"📝 Reviews processed: {len(df):,}")
    print(f"🔑 Keywords extracted: {len(keywords_df):,}")
    print(f"💾 Output files saved to: {output_dir}")
    print('='*60)


if __name__ == '__main__':
    main()